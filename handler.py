import os, gc, base64, io, tempfile, uuid
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HOME'] = '/app/cache'

import numpy as np
import torch
import runpod
from PIL import Image, ImageOps
from torch.amp import autocast
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

# ── Load RMBG background removal model ───────────────────────────────────────
print("Loading RMBG-2.0...")
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "camenduru/RMBG-2.0", trust_remote_code=True
)
birefnet.to("cuda")
birefnet.eval()

transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Load TRELLIS.2 pipeline ───────────────────────────────────────────────────
print("Loading TRELLIS.2 pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("camenduru/TRELLIS.2-4B")
pipeline.rembg_model = None  # we handle background removal ourselves
pipeline.low_vram = True
pipeline.cuda()
print("Pipeline ready.")

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

# ── RunPod handler ────────────────────────────────────────────────────────────

def handler(job):
    job_input = job["input"]

    # --- Decode image ---
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "Missing required field: image (base64 encoded PNG/JPG)"}

    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGBA")
    except Exception as e:
        return {"error": f"Failed to decode image: {e}"}

    # --- Parameters ---
    resolution = str(job_input.get("resolution", 1024))
    if resolution not in ("512", "1024", "1536"):
        return {"error": "resolution must be 512, 1024, or 1536"}

    pipeline_type = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}[resolution]
    seed            = job_input.get("seed", 42)
    decimation      = job_input.get("decimation_target", 300000)
    texture_size    = job_input.get("texture_size", 2048)
    remove_bg       = job_input.get("remove_bg", True)

    # --- Preprocess ---
    try:
        image = preprocess_image(image, remove_bg=remove_bg)
    except Exception as e:
        return {"error": f"Preprocessing failed: {e}"}

    # --- Inference (two-step: latent → mesh) ---
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
        return {"error": f"Inference failed: {e}"}

    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    # --- Decode latent to mesh ---
    try:
        shape_slat, tex_slat, res = latents
        mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
        mesh.simplify(16777216)
    except Exception as e:
        return {"error": f"Mesh decode failed: {e}"}

    del latents, shape_slat, tex_slat
    gc.collect()
    torch.cuda.empty_cache()

    # --- Export GLB ---
    try:
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=False,
        )

        tmp_path = f"/tmp/trellis_{uuid.uuid4().hex[:8]}.glb"
        glb.export(tmp_path, extension_webp=True)

        with open(tmp_path, "rb") as f:
            glb_b64 = base64.b64encode(f.read()).decode("utf-8")
        os.unlink(tmp_path)
    except Exception as e:
        return {"error": f"GLB export failed: {e}"}

    del mesh, glb
    gc.collect()
    torch.cuda.empty_cache()

    return {"glb": glb_b64}


runpod.serverless.start({"handler": handler})
