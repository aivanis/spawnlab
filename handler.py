import os, gc, io, time, threading, uuid
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HOME'] = '/app/cache'

import numpy as np
import torch
import requests as _requests
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image, ImageOps
from torch.amp import autocast
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

# ── Fix: redirect meta-device tensor ops to CPU during BiRefNet load ──────────
# BiRefNet's SwinTransformer calls torch.linspace(...).item() in __init__.
# When from_pretrained uses init_empty_weights() (meta-device context), linspace
# returns a meta tensor, and .item() raises RuntimeError.
# Root cause: DeviceContext.__torch_function__ adds device='meta' to all tensor
# creation ops via PyTorch's TorchFunctionMode dispatch (bypasses Python-level
# torch.linspace patches). Fix: temporarily redirect meta→CPU in DeviceContext,
# load BiRefNet (SwinTransformer __init__ now runs on CPU), then restore.
import contextlib
from torch.utils._device import DeviceContext as _DevCtx

_TENSOR_CREATION_OPS = None  # populated lazily after torch is imported

_TENSOR_CREATION_OPS = None  # populated lazily

@contextlib.contextmanager
def _meta_to_cpu_ctx():
    """Temporarily redirect tensor creation ops from meta device to CPU.
    Also patches mark_tied_weights_as_initialized to auto-call post_init() when
    all_tied_weights_keys is missing (BiRefNet's __init__ skips post_init())."""
    global _TENSOR_CREATION_OPS
    if _TENSOR_CREATION_OPS is None:
        _TENSOR_CREATION_OPS = {
            getattr(torch, n) for n in
            ('empty', 'zeros', 'ones', 'full', 'rand', 'randn',
             'arange', 'linspace', 'eye', 'zeros_like', 'ones_like', 'empty_like')
            if hasattr(torch, n)
        }

    # Patch 1: redirect meta tensor creation to CPU
    _orig_dcf = _DevCtx.__torch_function__
    def _patched_dcf(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in _TENSOR_CREATION_OPS and str(getattr(self, 'device', '')) == 'meta':
            kwargs.setdefault('device', 'cpu')
            return func(*args, **kwargs)
        return _orig_dcf(self, func, types, args, kwargs)
    _DevCtx.__torch_function__ = _patched_dcf

    # Patch 2: ensure all_tied_weights_keys exists (BiRefNet skips post_init)
    from transformers import PreTrainedModel as _PTM
    _orig_mark_tied = _PTM.mark_tied_weights_as_initialized
    def _safe_mark_tied(self, loading_info):
        if not hasattr(self, 'all_tied_weights_keys'):
            self.post_init()  # BiRefNet forgot to call this
        return _orig_mark_tied(self, loading_info)
    _PTM.mark_tied_weights_as_initialized = _safe_mark_tied

    try:
        yield
    finally:
        _DevCtx.__torch_function__ = _orig_dcf
        _PTM.mark_tied_weights_as_initialized = _orig_mark_tied

print("Loading BiRefNet background removal model...")
with _meta_to_cpu_ctx():
    _birefnet_model = AutoModelForImageSegmentation.from_pretrained(
        "camenduru/RMBG-2.0", trust_remote_code=True
    )
_birefnet_model.eval()
print("BiRefNet ready.")

import trellis2.pipelines.rembg as _rembg_pkg

class _PreloadedBiRefNet(_rembg_pkg.BiRefNet):
    """Wraps the already-loaded BiRefNet model — no tensor creation in __init__,
    safe to instantiate inside accelerate's meta-device context."""
    def __init__(self, model_name=None):
        self.model = _birefnet_model
        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

_rembg_pkg.BiRefNet = _PreloadedBiRefNet

# ── Load TRELLIS.2 pipeline (BiRefNet already loaded, no meta-tensor issue) ───
print("Loading TRELLIS.2 pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("camenduru/TRELLIS.2-4B")
pipeline.low_vram = False
pipeline.cuda()
print("Pipeline ready.")

# ── Move pre-loaded BiRefNet to GPU for inference ─────────────────────────────
birefnet = _birefnet_model
birefnet.to("cuda")
birefnet.eval()
pipeline.rembg_model = None  # we handle background removal ourselves

transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Background removal ────────────────────────────────────────────────────────

@torch.inference_mode()
def remove_background(image_pil: Image.Image) -> Image.Image:
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    original_size = image_pil.size
    w, h = image_pil.size
    new_w = ((w + 31) // 32) * 32
    new_h = ((h + 31) // 32) * 32
    padded = ImageOps.expand(image_pil, (0, 0, new_w - w, new_h - h), fill=0)
    tensor = transform_image(padded).unsqueeze(0).to("cuda")
    with autocast("cuda", dtype=torch.float16):
        pred = birefnet(tensor)[-1].sigmoid().cpu()[0].squeeze()
    mask = transforms.ToPILImage()(pred).crop((0, 0, original_size[0], original_size[1]))
    image_pil.putalpha(mask)
    return image_pil

# ── Image preprocessing ───────────────────────────────────────────────────────

@torch.inference_mode()
def preprocess_image(image: Image.Image, remove_bg: bool = True) -> Image.Image:
    max_size = max(image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS
        )
    if remove_bg:
        image = remove_background(image)

    arr = np.array(image)
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else np.full(arr.shape[:2], 255, dtype=np.uint8)
    if arr.shape[2] != 4:
        image = image.convert("RGBA")
        arr = np.array(image)

    coords = np.argwhere(alpha > 200)
    if len(coords) > 0:
        y0, x0 = coords.min(axis=0)[:2]
        y1, x1 = coords.max(axis=0)[:2]
        image = image.crop((x0, y0, x1 + 1, y1 + 1))

    w, h = image.size
    max_dim = max(w, h)
    squared = Image.new("RGBA", (max_dim, max_dim), (0, 0, 0, 0))
    squared.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))

    arr = np.array(squared).astype(np.float32) / 255.0
    arr[:, :, :3] *= arr[:, :, 3:4]
    return Image.fromarray((arr * 255).astype(np.uint8))

# ── Firebase init ─────────────────────────────────────────────────────────────

CLAIM_JOB_URL = os.environ.get("CLAIM_JOB_URL", "").rstrip("/")
WORKER_SECRET = os.environ.get("WORKER_SECRET", "")
INSTANCE_ID   = os.environ.get("VAST_INSTANCE_ID", f"vast-{uuid.uuid4().hex[:8]}")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SECONDS", "5"))
STORAGE_BUCKET = os.environ.get("FIREBASE_STORAGE_BUCKET", "")

_cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(_cred, {"storageBucket": STORAGE_BUCKET})
_db     = firestore.client()
_bucket = storage.bucket()

_CLAIM_HEADERS = {
    "Authorization": f"Bearer {WORKER_SECRET}",
    "Content-Type": "application/json",
}

# ── Core job processor ────────────────────────────────────────────────────────

def process_job(job_id: str, params: dict) -> None:
    resolution   = str(params.get("resolution", 1024))
    seed         = params.get("seed", 42)
    decimation   = params.get("decimation_target", 300000)
    texture_size = params.get("texture_size", 2048)
    remove_bg    = params.get("remove_bg", True)

    job_ref = _db.collection("jobs").document(job_id)

    def _fail(msg):
        print(f"[worker] job {job_id} failed: {msg}")
        job_ref.update({
            "status": "failed",
            "error": msg,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

    # --- Fetch image from Firebase Storage ---
    try:
        blob = _bucket.blob(f"inputs/{job_id}.png")
        image = Image.open(io.BytesIO(blob.download_as_bytes())).convert("RGBA")
    except Exception as e:
        _fail(f"Failed to fetch image: {e}")
        return

    # --- Validate resolution ---
    if resolution not in ("512", "1024", "1536"):
        _fail("resolution must be 512, 1024, or 1536")
        return

    pipeline_type = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}[resolution]

    # --- Preprocess ---
    try:
        image = preprocess_image(image, remove_bg=remove_bg)
    except Exception as e:
        _fail(f"Preprocessing failed: {e}")
        return

    # --- Inference ---
    gc.collect()
    torch.cuda.empty_cache()

    try:
        outputs, latents = pipeline.run(
            image,
            seed=seed,
            preprocess_image=False,
            pipeline_type=pipeline_type,
            return_latent=True,
        )
    except Exception as e:
        _fail(f"Inference failed: {e}")
        return

    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    # --- Decode latent to mesh ---
    try:
        shape_slat, tex_slat, res = latents
        mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
        mesh.simplify(16777216)
    except Exception as e:
        _fail(f"Mesh decode failed: {e}")
        del latents
        return

    del latents, shape_slat, tex_slat
    gc.collect()
    torch.cuda.empty_cache()

    # --- Export GLB ---
    def _to_glb(remesh):
        return o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation,
            texture_size=texture_size,
            remesh=remesh,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=False,
        )

    try:
        try:
            glb = _to_glb(remesh=True)
        except Exception:
            glb = _to_glb(remesh=False)

        tmp_path = f"/tmp/trellis_{uuid.uuid4().hex[:8]}.glb"
        glb.export(tmp_path, extension_webp=True)
        with open(tmp_path, "rb") as f:
            glb_bytes = f.read()
        os.unlink(tmp_path)
    except Exception as e:
        _fail(f"GLB export failed: {e}")
        del mesh
        return

    del mesh, glb
    gc.collect()
    torch.cuda.empty_cache()

    # --- Upload GLB to Firebase Storage ---
    try:
        blob = _bucket.blob(f"results/{job_id}.glb")
        blob.upload_from_string(glb_bytes, content_type="model/gltf-binary")
        print(f"[worker] job {job_id} done — {len(glb_bytes) // 1024} KB")
    except Exception as e:
        _fail(f"Failed to upload result: {e}")
        return

    # --- Mark completed ---
    job_ref.update({
        "status": "completed",
        "updated_at": firestore.SERVER_TIMESTAMP,
    })


# ── Polling loop ──────────────────────────────────────────────────────────────

def run_polling_loop():
    if not CLAIM_JOB_URL or not WORKER_SECRET:
        raise RuntimeError("CLAIM_JOB_URL and WORKER_SECRET must be set")

    print(f"[worker] starting — id={INSTANCE_ID}")

    while True:
        try:
            resp = _requests.post(
                CLAIM_JOB_URL,
                headers=_CLAIM_HEADERS,
                json={"worker_id": INSTANCE_ID},
                timeout=30,
            )

            if resp.status_code == 204:
                time.sleep(POLL_INTERVAL)
                continue

            resp.raise_for_status()
            job = resp.json()
            job_id = job["job_id"]
            params = job["params"]
            print(f"[worker] claimed job {job_id}")

            # Heartbeat thread
            job_ref = _db.collection("jobs").document(job_id)
            stop_hb = threading.Event()
            def _heartbeat(ref=job_ref):
                while not stop_hb.wait(30):
                    try:
                        ref.update({"heartbeat_at": firestore.SERVER_TIMESTAMP})
                    except Exception:
                        pass
            threading.Thread(target=_heartbeat, daemon=True).start()

            try:
                process_job(job_id, params)
            finally:
                stop_hb.set()

        except Exception as e:
            print(f"[worker] poll error: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_polling_loop()
