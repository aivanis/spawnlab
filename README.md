# SpawnLab

Image-to-3D generation API powered by [TRELLIS.2-4B](https://huggingface.co/camenduru/TRELLIS.2-4B). You spawn GPU instances manually on Vast.ai; SpawnLab handles the job queue, result storage, and stall recovery automatically.

## Architecture

```
Client
  │
  ▼
Cloudflare Worker  (spawnlab-worker)
  ├── KV  — job queue + status
  ├── R2  — input images + output GLBs
  └── Cron (every 1 min) — stall/crash recovery
        ▲
        │  polls /internal/claim every 5s
Vast.ai GPU instance(s)  (spawned manually)
  └── handler.py — TRELLIS.2 inference loop
```

**Flow:**
1. Client POSTs image to `/generate` → gets `job_id`
2. Worker stores image in R2, enqueues job in KV
3. You start a Vast.ai instance with the provisioning script
4. Instance polls `/internal/claim`, processes the job, POSTs the GLB back
5. Client polls `/status/:id` → downloads GLB from `/result/:id`

---

## API

### `POST /generate`

**Body (JSON):**
```json
{
  "image": "<base64 PNG or JPG>",
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
| `remove_bg` | bool | `true` | BiRefNet background removal |

**Response `202`:**
```json
{ "id": "uuid", "status": "pending" }
```

### `GET /status/:id`

```json
{ "id": "uuid", "status": "pending|processing|completed|failed", "created_at": 0, "updated_at": 0 }
```

### `GET /result/:id`

Returns GLB binary (`model/gltf-binary`) when status is `completed`.

### `GET /provision.sh`

Returns the Vast.ai provisioning script (proxied from this repo's `provision.sh`).

---

## Deployment

### 1. Cloudflare resources

```bash
npm install

npx wrangler kv namespace create JOBS
# → copy the id

npx wrangler kv namespace create JOBS --preview
# → copy the preview_id

npx wrangler r2 bucket create spawnlab-assets
```

Paste the IDs into `wrangler.toml`:
```toml
[[kv_namespaces]]
binding = "JOBS"
id = "<paste id>"
preview_id = "<paste preview_id>"
```

### 2. Secrets

```bash
npx wrangler secret put WORKER_SECRET
# enter any random string — you'll also paste this into the Vast.ai template
```

### 3. Deploy the Worker

```bash
npx wrangler deploy
```

Your worker URL will be:
```
https://spawnlab-worker.<your-subdomain>.workers.dev
```

Update `wrangler.toml` with it:
```toml
[vars]
WORKER_URL = "https://spawnlab-worker.<your-subdomain>.workers.dev"
```

Then deploy once more:
```bash
npx wrangler deploy
```

### 4. GitHub Actions (auto-deploy on push)

Add these secrets in **Settings → Secrets and variables → Actions**:

| Secret | Value |
|--------|-------|
| `CLOUDFLARE_API_TOKEN` | CF token with Workers + KV + R2 edit permissions |
| `CLOUDFLARE_ACCOUNT_ID` | Your Cloudflare account ID |

The workflow triggers on pushes to `main` that touch `handler.py`, `Dockerfile`, `src/**`, or `wrangler.toml`.

---

## Spawning a GPU instance on Vast.ai

Use image:
```
vastai/pytorch:2.9.1-cuda-12.8.1-py312-24.04
```

Extra docker options:
```
-p 1111:1111 -p 8188:8188 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 72299:72299 -p 3001:3001
-e OPEN_BUTTON_PORT=1111
-e OPEN_BUTTON_TOKEN=1
-e JUPYTER_DIR=/
-e DATA_DIRECTORY=/workspace/
-e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard"
-e PROVISIONING_SCRIPT="https://spawnlab-worker.<your-subdomain>.workers.dev/provision.sh"
-e WORKER_URL="https://spawnlab-worker.<your-subdomain>.workers.dev"
-e WORKER_SECRET="your_secret_here"
```

The provisioning script will:
1. Install all Python and GPU dependencies
2. Clone the TRELLIS.2 repo
3. Download all model weights (~30 GB)
4. Start `handler.py` which polls the Worker for jobs

**Recommended specs:** ≥24 GB VRAM (RTX 3090 / 4090 / A100), ≥80 GB disk.

---

## Testing

```bash
cp .env.template .env
# fill in WORKER_URL

# synthetic test image, fastest resolution
python test_endpoint.py --resolution 512

# your own image
python test_endpoint.py my_object.png --resolution 1024 --output my_object.glb
```

Open the GLB in Blender or drag it into [gltf.report](https://gltf.report).
