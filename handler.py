import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import base64
import io
import tempfile
import runpod
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

# Load model once at startup (before serverless.start)
print("Loading TRELLIS.2 pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("/app/models/TRELLIS.2-4B")
pipeline.cuda()
print("Pipeline ready.")


def handler(job):
    job_input = job["input"]

    # --- Input ---
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "Missing required field: image (base64 encoded)"}

    resolution = job_input.get("resolution", 1024)
    if resolution not in (512, 1024, 1536):
        return {"error": "resolution must be 512, 1024, or 1536"}

    decimation_target = job_input.get("decimation_target", 1000000)
    texture_size = job_input.get("texture_size", 2048)

    # --- Decode image ---
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to decode image: {e}"}

    # --- Run inference ---
    try:
        mesh = pipeline.run(image, resolution=resolution)[0]
        mesh.simplify(16777216)  # nvdiffrast limit
    except Exception as e:
        return {"error": f"Inference failed: {e}"}

    # --- Export GLB ---
    try:
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            glb_path = f.name

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        glb.export(glb_path, extension_webp=True)

        with open(glb_path, "rb") as f:
            glb_b64 = base64.b64encode(f.read()).decode("utf-8")

        os.unlink(glb_path)
    except Exception as e:
        return {"error": f"GLB export failed: {e}"}

    return {"glb": glb_b64}


runpod.serverless.start({"handler": handler})
