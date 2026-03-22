const { onRequest } = require("firebase-functions/v2/https");
const admin = require("firebase-admin");

admin.initializeApp();
const db = admin.firestore();
const bucket = admin.storage().bucket();

// Helper: verify worker auth
function checkWorkerAuth(req) {
  const auth = req.headers.authorization || "";
  return auth === `Bearer ${process.env.WORKER_SECRET}`;
}

// Helper: generate a signed URL (1 hour expiry)
async function signedUrl(filePath, action = "read", contentType) {
  const opts = { version: "v4", action, expires: Date.now() + 60 * 60 * 1000 };
  if (contentType) opts.contentType = contentType;
  const [url] = await bucket.file(filePath).getSignedUrl(opts);
  return url;
}

// ── claimJob ────────────────────────────────────────────────────────────────
// POST { worker_id }
// Returns 200: { job_id, params, image_url } or 204: no jobs

exports.claimJob = onRequest({ secrets: ["WORKER_SECRET"], invoker: "public" }, async (req, res) => {
  if (req.method !== "POST") return res.status(405).send("Method Not Allowed");
  if (!checkWorkerAuth(req)) return res.status(401).send("Unauthorized");

  const { worker_id } = req.body || {};
  if (!worker_id) return res.status(400).json({ error: "Missing worker_id" });

  const now = admin.firestore.FieldValue.serverTimestamp();

  try {
    const result = await db.runTransaction(async (tx) => {
      const snapshot = await tx.get(
        db.collection("jobs")
          .where("status", "==", "pending")
          .orderBy("created_at", "asc")
          .limit(1)
      );

      if (snapshot.empty) return null;

      const doc = snapshot.docs[0];
      tx.update(doc.ref, {
        status: "processing",
        worker_id,
        updated_at: now,
        heartbeat_at: now,
      });

      return { job_id: doc.id, params: doc.data().params };
    });

    if (!result) return res.status(204).send();

    // Signed URLs: download input image + upload output GLB
    const [files] = await bucket.getFiles({ prefix: `inputs/${result.job_id}.` });
    const imageUrl = files.length > 0
      ? await signedUrl(files[0].name, "read")
      : null;
    const uploadUrl = await signedUrl(
      `outputs/${result.job_id}.glb`, "write", "model/gltf-binary"
    );

    return res.status(200).json({ ...result, image_url: imageUrl, upload_url: uploadUrl });

  } catch (e) {
    console.error("claimJob error:", e);
    return res.status(500).json({ error: e.message });
  }
});

// ── heartbeat ───────────────────────────────────────────────────────────────
// POST { job_id }

exports.heartbeat = onRequest({ secrets: ["WORKER_SECRET"], invoker: "public" }, async (req, res) => {
  if (req.method !== "POST") return res.status(405).send("Method Not Allowed");
  if (!checkWorkerAuth(req)) return res.status(401).send("Unauthorized");

  const { job_id } = req.body || {};
  if (!job_id) return res.status(400).json({ error: "Missing job_id" });

  try {
    await db.collection("jobs").doc(job_id).update({
      heartbeat_at: admin.firestore.FieldValue.serverTimestamp(),
    });
    return res.status(200).json({ ok: true });
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
});

// ── completeJob ─────────────────────────────────────────────────────────────
// POST { job_id } — called after worker has uploaded GLB to the signed URL

exports.completeJob = onRequest({ secrets: ["WORKER_SECRET"], invoker: "public" }, async (req, res) => {
  if (req.method !== "POST") return res.status(405).send("Method Not Allowed");
  if (!checkWorkerAuth(req)) return res.status(401).send("Unauthorized");

  const { job_id } = req.body || {};
  if (!job_id) return res.status(400).json({ error: "Missing job_id" });

  try {
    await db.collection("jobs").doc(job_id).update({
      status: "done",
      updated_at: admin.firestore.FieldValue.serverTimestamp(),
    });
    return res.status(200).json({ ok: true });
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
});

// ── failJob ─────────────────────────────────────────────────────────────────
// POST { job_id, error }

exports.failJob = onRequest({ secrets: ["WORKER_SECRET"], invoker: "public" }, async (req, res) => {
  if (req.method !== "POST") return res.status(405).send("Method Not Allowed");
  if (!checkWorkerAuth(req)) return res.status(401).send("Unauthorized");

  const { job_id, error: errMsg } = req.body || {};
  if (!job_id) return res.status(400).json({ error: "Missing job_id" });

  try {
    await db.collection("jobs").doc(job_id).update({
      status: "failed",
      error: errMsg || "Unknown error",
      updated_at: admin.firestore.FieldValue.serverTimestamp(),
    });
    return res.status(200).json({ ok: true });
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
});
