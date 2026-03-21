/**
 * SpawnLab Cloudflare Worker
 *
 * Public API:
 *   POST /generate               { image: <base64>, resolution?, seed?, decimation_target?, texture_size?, remove_bg? }
 *   GET  /status/:id             → { id, status, created_at, updated_at, error? }
 *   GET  /result/:id             → GLB binary (when status=completed)
 *
 * Internal API (Bearer WORKER_SECRET):
 *   POST /internal/claim         { worker_id } → job params or 204
 *   GET  /internal/image/:id     → image binary
 *   POST /internal/result/:id    raw GLB body
 *   POST /internal/fail/:id      { error }
 *   POST /internal/heartbeat/:id { worker_id }
 *
 * Cron (every minute):
 *   Vast.ai autoscaler + stalled-job requeue
 */

// ── Constants ────────────────────────────────────────────────────────────────

const STALL_TIMEOUT_MS = 120_000; // requeue if no heartbeat for 2 min
const JOB_TTL_SECONDS  = 86400;   // KV TTL: 24 h

// ── Helpers ──────────────────────────────────────────────────────────────────

function uuid() {
  return crypto.randomUUID();
}

function jsonResp(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function errResp(msg, status = 400) {
  return jsonResp({ error: msg }, status);
}

async function requireSecret(request, env) {
  const auth = request.headers.get("Authorization") || "";
  if (auth !== `Bearer ${env.WORKER_SECRET}`) {
    return new Response("Unauthorized", { status: 401 });
  }
  return null; // OK
}

// ── KV helpers ───────────────────────────────────────────────────────────────

async function getJob(env, id) {
  const raw = await env.JOBS.get(`job:${id}`);
  return raw ? JSON.parse(raw) : null;
}

async function putJob(env, id, job) {
  await env.JOBS.put(`job:${id}`, JSON.stringify(job), {
    expirationTtl: JOB_TTL_SECONDS,
  });
}

async function getQueue(env) {
  const raw = await env.JOBS.get("pending_queue");
  return raw ? JSON.parse(raw) : [];
}

async function putQueue(env, queue) {
  await env.JOBS.put("pending_queue", JSON.stringify(queue));
}

async function getProcessingIds(env) {
  const raw = await env.JOBS.get("processing_ids");
  return raw ? JSON.parse(raw) : {};
}

async function putProcessingIds(env, ids) {
  await env.JOBS.put("processing_ids", JSON.stringify(ids));
}

// ── Public: POST /generate ───────────────────────────────────────────────────

async function handleGenerate(request, env) {
  let body;
  try {
    body = await request.json();
  } catch {
    return errResp("Invalid JSON body");
  }

  const { image, resolution = 1024, seed = 42, decimation_target = 300000,
          texture_size = 2048, remove_bg = true } = body;

  if (!image || typeof image !== "string") {
    return errResp("Missing required field: image (base64 string)");
  }
  if (![512, 1024, 1536].includes(Number(resolution))) {
    return errResp("resolution must be 512, 1024, or 1536");
  }

  // Decode and store image in R2
  let imageBytes;
  try {
    const bin = atob(image);
    imageBytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) imageBytes[i] = bin.charCodeAt(i);
  } catch {
    return errResp("Failed to decode image: invalid base64");
  }

  const jobId = uuid();
  const now = Date.now();

  await env.ASSETS.put(`inputs/${jobId}`, imageBytes, {
    httpMetadata: { contentType: "image/png" },
    customMetadata: { job_id: jobId },
  });

  const job = {
    id: jobId,
    status: "pending",
    params: { resolution: Number(resolution), seed, decimation_target, texture_size, remove_bg },
    created_at: now,
    updated_at: now,
  };
  await putJob(env, jobId, job);

  // Enqueue
  const queue = await getQueue(env);
  queue.push(jobId);
  await putQueue(env, queue);

  // Touch last_active_at so autoscaler knows there's demand
  await env.JOBS.put("last_active_at", String(now));

  return jsonResp({ id: jobId, status: "pending" }, 202);
}

// ── Public: GET /status/:id ──────────────────────────────────────────────────

async function handleStatus(id, env) {
  const job = await getJob(env, id);
  if (!job) return errResp("Job not found", 404);
  const { params: _, ...safe } = job; // don't expose internal params
  return jsonResp(safe);
}

// ── Public: GET /result/:id ──────────────────────────────────────────────────

async function handleResult(id, env) {
  const job = await getJob(env, id);
  if (!job) return errResp("Job not found", 404);
  if (job.status !== "completed") {
    return jsonResp({ id, status: job.status }, job.status === "failed" ? 422 : 202);
  }

  const obj = await env.ASSETS.get(`results/${id}.glb`);
  if (!obj) return errResp("Result not found", 404);

  return new Response(obj.body, {
    headers: {
      "Content-Type": "model/gltf-binary",
      "Content-Disposition": `attachment; filename="${id}.glb"`,
    },
  });
}

// ── Internal: POST /internal/claim ──────────────────────────────────────────

async function handleClaim(request, env) {
  const { worker_id } = await request.json().catch(() => ({}));
  if (!worker_id) return errResp("Missing worker_id");

  const queue = await getQueue(env);
  if (queue.length === 0) {
    return new Response(null, { status: 204 });
  }

  const jobId = queue.shift();
  await putQueue(env, queue);

  const job = await getJob(env, jobId);
  if (!job) {
    // Job disappeared (TTL expired?) — try next
    return new Response(null, { status: 204 });
  }

  const now = Date.now();
  job.status = "processing";
  job.worker_id = worker_id;
  job.updated_at = now;
  job.heartbeat_at = now;
  await putJob(env, jobId, job);

  // Track in processing_ids for stall recovery
  const pids = await getProcessingIds(env);
  pids[jobId] = { worker_id, heartbeat_at: now };
  await putProcessingIds(env, pids);

  return jsonResp({ job_id: jobId, params: job.params });
}

// ── Internal: GET /internal/image/:id ────────────────────────────────────────

async function handleGetImage(id, env) {
  const obj = await env.ASSETS.get(`inputs/${id}`);
  if (!obj) return errResp("Image not found", 404);
  return new Response(obj.body, {
    headers: { "Content-Type": "image/png" },
  });
}

// ── Internal: POST /internal/result/:id ──────────────────────────────────────

async function handlePostResult(request, id, env) {
  const glbBytes = await request.arrayBuffer();
  if (!glbBytes.byteLength) return errResp("Empty result body");

  await env.ASSETS.put(`results/${id}.glb`, glbBytes, {
    httpMetadata: { contentType: "model/gltf-binary" },
  });

  const job = await getJob(env, id);
  if (job) {
    job.status = "completed";
    job.updated_at = Date.now();
    await putJob(env, id, job);
  }

  // Remove from processing_ids
  const pids = await getProcessingIds(env);
  delete pids[id];
  await putProcessingIds(env, pids);

  return jsonResp({ ok: true });
}

// ── Internal: POST /internal/fail/:id ────────────────────────────────────────

async function handleFail(request, id, env) {
  const { error = "unknown error" } = await request.json().catch(() => ({}));

  const job = await getJob(env, id);
  if (job) {
    job.status = "failed";
    job.error = error;
    job.updated_at = Date.now();
    await putJob(env, id, job);
  }

  const pids = await getProcessingIds(env);
  delete pids[id];
  await putProcessingIds(env, pids);

  return jsonResp({ ok: true });
}

// ── Internal: POST /internal/heartbeat/:id ───────────────────────────────────

async function handleHeartbeat(request, id, env) {
  const now = Date.now();
  const job = await getJob(env, id);
  if (job && job.status === "processing") {
    job.heartbeat_at = now;
    job.updated_at = now;
    await putJob(env, id, job);
  }

  const pids = await getProcessingIds(env);
  if (pids[id]) {
    pids[id].heartbeat_at = now;
    await putProcessingIds(env, pids);
  }

  return jsonResp({ ok: true });
}

// ── Cron: stall recovery ──────────────────────────────────────────────────────

async function requeueStalledJobs(env) {
  const pids = await getProcessingIds(env);
  const now = Date.now();
  const stalled = [];

  for (const [jobId, info] of Object.entries(pids)) {
    if (now - info.heartbeat_at > STALL_TIMEOUT_MS) {
      stalled.push(jobId);
    }
  }

  if (stalled.length === 0) return;

  const queue = await getQueue(env);
  for (const jobId of stalled) {
    const job = await getJob(env, jobId);
    if (job && job.status === "processing") {
      job.status = "pending";
      job.updated_at = now;
      delete job.worker_id;
      await putJob(env, jobId, job);
      if (!queue.includes(jobId)) queue.push(jobId);
    }
    delete pids[jobId];
    console.log(`[stall] requeued job ${jobId}`);
  }

  await putQueue(env, queue);
  await putProcessingIds(env, pids);
}

// ── Cron: stall recovery only (instances managed manually via Vast.ai) ────────

async function runManager(env) {
  await requeueStalledJobs(env);
}

// ── Router ────────────────────────────────────────────────────────────────────

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const { method } = request;
    const path = url.pathname.replace(/\/$/, "") || "/";

    // Internal routes (require WORKER_SECRET)
    if (path.startsWith("/internal/")) {
      const authErr = await requireSecret(request, env);
      if (authErr) return authErr;

      if (method === "POST" && path === "/internal/claim") {
        return handleClaim(request, env);
      }
      const imgMatch = path.match(/^\/internal\/image\/([^/]+)$/);
      if (method === "GET" && imgMatch) {
        return handleGetImage(imgMatch[1], env);
      }
      const resultMatch = path.match(/^\/internal\/result\/([^/]+)$/);
      if (method === "POST" && resultMatch) {
        return handlePostResult(request, resultMatch[1], env);
      }
      const failMatch = path.match(/^\/internal\/fail\/([^/]+)$/);
      if (method === "POST" && failMatch) {
        return handleFail(request, failMatch[1], env);
      }
      const hbMatch = path.match(/^\/internal\/heartbeat\/([^/]+)$/);
      if (method === "POST" && hbMatch) {
        return handleHeartbeat(request, hbMatch[1], env);
      }
      return errResp("Not found", 404);
    }

    // Provisioning script + handler (served from R2 public bucket)
    if (method === "GET" && (path === "/provision.sh" || path === "/handler.py")) {
      return Response.redirect(
        `https://pub-d4542cd5f9bc434dbb7da007761dec7b.r2.dev${path}`,
        302
      );
    }

    // Public routes
    if (method === "POST" && path === "/generate") {
      return handleGenerate(request, env);
    }
    const statusMatch = path.match(/^\/status\/([^/]+)$/);
    if (method === "GET" && statusMatch) {
      return handleStatus(statusMatch[1], env);
    }
    const resultMatch = path.match(/^\/result\/([^/]+)$/);
    if (method === "GET" && resultMatch) {
      return handleResult(resultMatch[1], env);
    }

    return errResp("Not found", 404);
  },

  async scheduled(event, env, ctx) {
    ctx.waitUntil(runManager(env));
  },
};
