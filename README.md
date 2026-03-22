# SpawnLab

Image-to-3D mesh generation powered by [TRELLIS.2-4B](https://huggingface.co/camenduru/TRELLIS.2-4B). GPU workers run on Vast.ai, job queue and storage are managed by Firebase.

## Architecture

```
Client (Web / Unity / Blender plugin)
  |
  v
Firebase Auth + Firestore + Storage
  |
  v
Cloud Functions (claimJob, completeJob, failJob, heartbeat)
  |
  v
Vast.ai GPU instance(s) (handler.py - polling loop)
```

**Flow:**
1. Client uploads image to Firebase Storage, creates a job doc in Firestore with `status: "pending"`
2. GPU worker calls `claimJob` Cloud Function, which atomically claims the oldest pending job
3. `claimJob` returns `job_id`, `params`, `image_url` (signed download URL), and `upload_url` (signed upload URL for GLB)
4. Worker downloads image, runs TRELLIS.2 inference, uploads GLB to the signed URL
5. Worker calls `completeJob` to mark the job as done
6. Client sees status change in real-time via Firestore listener, downloads GLB

## Web Dashboard

Live at: https://spawnlab-53283.web.app

- Sign in with Google
- View all jobs in real-time
- Submit new jobs (upload an image)
- Download completed GLB meshes

## Cloud Functions

| Function | Method | Description |
|----------|--------|-------------|
| `claimJob` | POST | Atomically claim next pending job. Returns `{ job_id, params, image_url, upload_url }` or 204 |
| `completeJob` | POST | Mark job as done (after GLB upload) |
| `failJob` | POST | Mark job as failed with error message |
| `heartbeat` | POST | Update heartbeat timestamp |

All worker endpoints require `Authorization: Bearer <WORKER_SECRET>`.

Base URL: `https://us-central1-spawnlab-53283.cloudfunctions.net`

## Job Parameters

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `resolution` | int | `1024` | `512`, `1024`, or `1536` |
| `seed` | int | `42` | |
| `decimation_target` | int | `100000` | Target face count |
| `texture_size` | int | `1024` | |
| `remove_bg` | bool | `true` | BiRefNet background removal |

## Vast.ai Template Setup

**Docker image:**
```
vastai/pytorch:2.9.1-cuda-12.8.1-py312-24.04
```

**Docker options:**
```
-p 1111:1111 -p 8188:8188 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 72299:72299 -p 3001:3001 -e OPEN_BUTTON_PORT=1111 -e OPEN_BUTTON_TOKEN=1 -e JUPYTER_DIR=/ -e DATA_DIRECTORY=/workspace/ -e PORTAL_CONFIG="localhost:1111:11111:/:Instance   Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter   Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard" -e PROVISIONING_SCRIPT="https://storage.googleapis.com/public-spawnlab/provision.sh" -e WORKER_SECRET="<your_worker_secret>" -e HF_TOKEN="<your_huggingface_token>"
```

**Environment variables:**

| Variable | Required | Description |
|----------|----------|-------------|
| `WORKER_SECRET` | Yes | Shared secret for Cloud Function auth |
| `HF_TOKEN` | Recommended | HuggingFace token for faster model downloads |
| `CLOUD_FUNCTIONS_URL` | No | Defaults to `https://us-central1-spawnlab-53283.cloudfunctions.net` |
| `PROVISIONING_SCRIPT` | Yes | URL to provision.sh |

**Recommended specs:** 24+ GB VRAM (RTX 3090 / 4090 / A100), 80+ GB disk.

The provisioning script will:
1. Install system and Python dependencies
2. Clone the TRELLIS.2 repo and install GPU extension wheels
3. Download model weights (~30 GB) from HuggingFace
4. Start `handler.py` which polls for jobs

## Firebase Setup

**Project:** `spawnlab-53283`

```bash
# Install Firebase CLI
npm install -g firebase-tools

# Login and deploy
firebase login
firebase deploy --project spawnlab-53283
```

**Secrets:**
```bash
firebase functions:secrets:set WORKER_SECRET --project spawnlab-53283
```

## Viewing Results

Open GLB files in Blender or drag them into [gltf.report](https://gltf.report).
