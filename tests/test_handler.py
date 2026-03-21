"""
Comprehensive tests for the TRELLIS.2 RunPod handler.
Run inside the Docker container:
    python tests/test_handler.py
"""

import sys
import os
import base64
import io
import struct
import time

# ── helpers ──────────────────────────────────────────────────────────────────

def ok(msg):   print(f"  [PASS] {msg}")
def fail(msg): print(f"  [FAIL] {msg}"); sys.exit(1)
def section(msg): print(f"\n{'='*60}\n {msg}\n{'='*60}")

def make_test_image_b64(width=512, height=512):
    """Create a simple RGB test image (solid color gradient) as base64."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    # Draw a rough cube-like shape so TRELLIS has something to work with
    draw.rectangle([100, 100, 400, 400], fill=(200, 100, 50))
    draw.polygon([(400, 100), (500, 50), (500, 350), (400, 400)], fill=(160, 80, 40))
    draw.polygon([(100, 100), (500, 50), (400, 100)], fill=(220, 140, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def is_valid_glb(data: bytes) -> bool:
    """Check GLB magic bytes and basic header."""
    if len(data) < 12:
        return False
    magic = struct.unpack("<I", data[:4])[0]
    version = struct.unpack("<I", data[4:8])[0]
    return magic == 0x46546C67 and version == 2  # 'glTF' magic, version 2

# ── 1. Dependency imports ─────────────────────────────────────────────────────

section("1. Dependency imports")

try:
    import torch
    ok(f"torch {torch.__version__}")
except ImportError as e:
    fail(f"torch: {e}")

try:
    assert torch.cuda.is_available(), "CUDA not available"
    ok(f"CUDA available — device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    fail(f"CUDA: {e}")

try:
    import flash_attn
    ok(f"flash_attn {flash_attn.__version__}")
except ImportError as e:
    fail(f"flash_attn: {e}")

try:
    import nvdiffrast
    ok("nvdiffrast")
except ImportError as e:
    fail(f"nvdiffrast: {e}")

try:
    import nvdiffrec_render
    ok("nvdiffrec_render")
except ImportError as e:
    fail(f"nvdiffrec_render: {e}")

try:
    import cumesh
    ok("cumesh")
except ImportError as e:
    fail(f"cumesh: {e}")

try:
    import flex_gemm
    ok("flex_gemm")
except ImportError as e:
    fail(f"flex_gemm: {e}")

try:
    import o_voxel
    ok("o_voxel")
except ImportError as e:
    fail(f"o_voxel: {e}")

try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    ok("trellis2.pipelines")
except ImportError as e:
    fail(f"trellis2: {e}")

try:
    import runpod
    ok(f"runpod {runpod.__version__}")
except ImportError as e:
    fail(f"runpod: {e}")

# ── 2. Model weights present ──────────────────────────────────────────────────

section("2. Model weights present")

MODEL_DIR = "/app/models/TRELLIS.2-4B"
expected_ckpts = [
    "pipeline.json",
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
    "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
    "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
    "ckpts/shape_enc_next_dc_f16c32_fp16.safetensors",
    "ckpts/tex_enc_next_dc_f16c32_fp16.safetensors",
]
for f in expected_ckpts:
    path = os.path.join(MODEL_DIR, f)
    if os.path.exists(path):
        ok(f"{f} ({os.path.getsize(path) / 1e6:.0f} MB)")
    else:
        fail(f"Missing: {path}")

# Check HF cache for external models
import glob
hf_cache = os.path.expanduser("~/.cache/huggingface/hub")

def hf_model_cached(name):
    safe_name = name.replace("/", "--")
    matches = glob.glob(os.path.join(hf_cache, f"models--{safe_name}*"))
    return len(matches) > 0

for model in ["microsoft/TRELLIS-image-large", "facebook/dinov3-vitl16-pretrain-lvd1689m", "briaai/RMBG-2.0"]:
    if hf_model_cached(model):
        ok(f"HF cache: {model}")
    else:
        fail(f"HF cache missing: {model}")

# ── 3. Pipeline loads ─────────────────────────────────────────────────────────

section("3. Pipeline loads into GPU memory")

t0 = time.time()
try:
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_DIR)
    pipeline.cuda()
    elapsed = time.time() - t0
    vram = torch.cuda.memory_allocated() / 1e9
    ok(f"Pipeline loaded in {elapsed:.1f}s — VRAM used: {vram:.1f} GB")
except Exception as e:
    fail(f"Pipeline load failed: {e}")

# ── 4. Handler input validation ───────────────────────────────────────────────

section("4. Handler input validation")

# Import handler module directly (bypasses runpod.serverless.start)
sys.path.insert(0, "/app")
# Monkey-patch runpod.serverless.start so importing handler doesn't block
import runpod
runpod.serverless.start = lambda config: None

# Re-use already-loaded pipeline by patching the module-level variable
import importlib.util, types
spec = importlib.util.spec_from_file_location("handler_module", "/app/handler.py")
handler_mod = types.ModuleType("handler_module")
# Inject the already-loaded pipeline so it doesn't reload
handler_mod.pipeline = pipeline
exec(open("/app/handler.py").read().replace(
    "pipeline = Trellis2ImageTo3DPipeline.from_pretrained",
    "pipeline = pipeline or Trellis2ImageTo3DPipeline.from_pretrained"
), handler_mod.__dict__)

handler = handler_mod.handler

# Missing image
result = handler({"input": {}})
assert "error" in result, "Expected error for missing image"
ok("Returns error when image missing")

# Invalid resolution
img_b64 = make_test_image_b64()
result = handler({"input": {"image": img_b64, "resolution": 999}})
assert "error" in result, "Expected error for invalid resolution"
ok("Returns error for invalid resolution")

# Bad base64
result = handler({"input": {"image": "not_valid_base64!!!"}})
assert "error" in result, "Expected error for bad base64"
ok("Returns error for bad base64")

# ── 5. Full inference (512 resolution — fastest) ──────────────────────────────

section("5. Full inference at 512 resolution")

img_b64 = make_test_image_b64()
t0 = time.time()
result = handler({
    "input": {
        "image": img_b64,
        "resolution": 512,
        "decimation_target": 100000,
        "texture_size": 1024,
    }
})
elapsed = time.time() - t0

if "error" in result:
    fail(f"Inference failed: {result['error']}")

assert "glb" in result, "No 'glb' key in result"
glb_bytes = base64.b64decode(result["glb"])
assert is_valid_glb(glb_bytes), "Output is not a valid GLB file"
ok(f"Inference completed in {elapsed:.1f}s — GLB size: {len(glb_bytes)/1e6:.1f} MB")

# Save for manual inspection
out_path = "/app/test_output.glb"
with open(out_path, "wb") as f:
    f.write(glb_bytes)
ok(f"GLB saved to {out_path}")

# ── 6. VRAM cleanup check ─────────────────────────────────────────────────────

section("6. VRAM after inference")
vram_after = torch.cuda.memory_allocated() / 1e9
ok(f"VRAM allocated: {vram_after:.1f} GB")

# ── Done ──────────────────────────────────────────────────────────────────────

section("ALL TESTS PASSED")
print(f"  RTX {torch.cuda.get_device_name(0)}")
print(f"  torch {torch.__version__} | CUDA {torch.version.cuda}")
print()
