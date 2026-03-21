# SpawnLab

Image-to-3D generation API powered by [TRELLIS.2-4B](https://huggingface.co/camenduru/TRELLIS.2-4B), running on Vast.ai GPU instances managed by a Cloudflare Worker.

## Architecture

```
Client
  │
  ▼
Cloudflare Worker (spawnlab-worker)
  ├── KV  — job queue + status
  ├── R2  — input images + output GLBs
  └── Cron (every 1 min) — Vast.ai autoscaler + stall recovery
        │
        ▼
  Vast.ai GPU instance(s)
    └── handler.py — polls /internal/claim, runs TRELLIS.2, uploads result
```

**Flow:**
1. Client POSTs image to `/generate` → gets `job_id`
2. Cloudflare stores image in R2 and enqueues job in KV
3. Cron fires every minute — spins up a Vast.ai GPU if queue is non-empty
4. GPU worker polls `/internal/claim` every 5s, processes job, POSTs GLB back
5. Client polls `/status/:id` then fetches binary GLB from `/result/:id`

---

## API

### `POST /generate`

Submit an image for 3D generation.

**Body (JSON):**
```json
{
  "image": "<base64-encoded PNG or JPG>",
  "resolution": 1024,
  "seed": 42,
  "decimation_target": 300000,
  "texture_size": 2048,
  "remove_bg": true
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `image` | string | required | Base64-encoded PNG or JPG |
| `resolution` | int | `1024` | `512`, `1024`, or `1536` |
| `seed` | int | `42` | |
| `decimation_target` | int | `300000` | Target face count |
| `texture_size` | int | `2048` | `512`, `1024`, or `2048` |
| `remove_bg` | bool | `true` | Run BiRefNet background removal |

**Response `202`:**
```json
{ "id": "uuid", "status": "pending" }
```

---

### `GET /status/:id`

```json
{ "id": "uuid", "status": "pending|processing|completed|failed", "created_at": 1234567890, "updated_at": 1234567890 }
```

---

### `GET /result/:id`

Returns the GLB binary (`model/gltf-binary`) when `status` is `completed`.

---

## Setup

### 1. Prerequisites

- [Node.js](https://nodejs.org) 18+
- [Wrangler CLI](https://developers.cloudflare.com/workers/wrangler/) (`npm install` in this repo)
- Cloudflare account (free tier is fine)
- Vast.ai account + API key
- Docker Hub account

---

### 2. Cloudflare resources

Install dependencies and create the KV namespace and R2 bucket:

```bash
npm install

npx wrangler kv namespace create JOBS
# → note the id value

npx wrangler kv namespace create JOBS --preview
# → note the preview_id value

npx wrangler r2 bucket create spawnlab-assets
```

Edit `wrangler.toml` and fill in the two IDs:

```toml
[[kv_namespaces]]
binding = "JOBS"
id = "<paste id here>"
preview_id = "<paste preview_id here>"
```

---

### 3. Secrets

```bash
npx wrangler secret put WORKER_SECRET
# → enter any random string (shared between Worker and GPU instances)

npx wrangler secret put VAST_API_KEY
# → paste your Vast.ai API key (console.vast.ai → Account → API Key)
```

---

### 4. Worker URL

After your first deploy the worker will be at:
```
https://spawnlab-worker.<your-subdomain>.workers.dev
```

Add it to `wrangler.toml` under `[vars]`:
```toml
[vars]
WORKER_URL = "https://spawnlab-worker.<your-subdomain>.workers.dev"
```

---

### 5. Docker Hub image

Update `wrangler.toml` with your Docker Hub username:
```toml
DOCKER_IMAGE = "yourusername/trellis2-worker:latest"
```

---

### 6. Deploy

```bash
npx wrangler deploy
```

Or just push to `main` — GitHub Actions will build the Docker image and deploy the Worker automatically.

---

### 7. GitHub Actions secrets

In your repo go to **Settings → Secrets and variables → Actions** and add:

| Secret | Value |
|--------|-------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `CLOUDFLARE_API_TOKEN` | CF token with Workers + KV + R2 edit permissions |
| `CLOUDFLARE_ACCOUNT_ID` | Your Cloudflare account ID |

The workflow triggers on pushes to `main` that touch `handler.py`, `Dockerfile`, `src/**`, or `wrangler.toml`.

---

## Testing

```bash
cp .env.template .env
# fill in WORKER_URL
```

```bash
# synthetic test image, fastest resolution
python test_endpoint.py --resolution 512

# your own image
python test_endpoint.py my_object.png --resolution 1024

# save to custom path
python test_endpoint.py my_object.png --output my_object.glb
```

Open the result in Blender or drag it into [gltf.report](https://gltf.report).

---

## Autoscaler behaviour

The cron job runs every minute and:

- **Scales up** one Vast.ai instance per pending job (capped at 4 instances). Selects the cheapest verified offer with ≥24 GB VRAM and ≥1 GPU.
- **Scales down** all instances if there has been no activity for 10 minutes (configurable via `IDLE_SCALE_DOWN_MINUTES` in `wrangler.toml`).
- **Requeues stalled jobs** if a processing job has had no heartbeat for 2 minutes (e.g. instance crashed).

To adjust GPU requirements edit `wrangler.toml`:
```toml
VAST_GPU_MIN_VRAM_GB = "24"   # minimum VRAM in GB
VAST_DISK_GB = "80"           # disk space per instance
IDLE_SCALE_DOWN_MINUTES = "10"
```
