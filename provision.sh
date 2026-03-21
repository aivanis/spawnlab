#!/bin/bash
# SpawnLab - TRELLIS.2 provisioning script for vastai/pytorch:2.9.1-cuda-12.8.1-py312-24.04
# Usage: set PROVISIONING_SCRIPT to the URL of this file in your Vast.ai template
#
# Required env vars (set in Vast.ai template):
#   WORKER_URL     - https://spawnlab-worker.anivanis.workers.dev
#   WORKER_SECRET  - shared secret

set -euo pipefail

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate main

export HF_HOME=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- System deps ---
apt-get update -qq
apt-get install -y --no-install-recommends \
    git git-lfs wget aria2 \
    libgl1 libglib2.0-0 libegl1

# --- Python: core packages ---
pip install --no-cache-dir \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    trimesh transformers zstandard kornia timm \
    plyfile requests hf_transfer huggingface_hub

pip install --no-cache-dir \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

pip install --no-cache-dir pillow-simd

# --- Clone TRELLIS.2 ---
if [ ! -d /workspace/TRELLIS.2 ]; then
    echo "=== Cloning TRELLIS.2 ==="
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch dev \
        https://github.com/camenduru/TRELLIS.2-hf /workspace/TRELLIS.2
fi

# --- GPU extension wheels (torch2.9+cu128, cp312) ---
echo "=== Installing GPU wheels ==="
CWHEELS="https://github.com/camenduru/wheels/releases/download/trellis2"
pip install --no-cache-dir "${CWHEELS}/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
pip install --no-cache-dir "${CWHEELS}/nvdiffrast-0.4.0-cp312-cp312-linux_x86_64.whl"
pip install --no-cache-dir "${CWHEELS}/nvdiffrec_render-0.0.0-cp312-cp312-linux_x86_64.whl"
pip install --no-cache-dir "${CWHEELS}/flex_gemm-0.0.1-cp312-cp312-linux_x86_64.whl"
pip install --no-cache-dir --no-deps "${CWHEELS}/cumesh-0.0.1-cp312-cp312-linux_x86_64.whl"
pip install --no-cache-dir --no-deps "${CWHEELS}/o_voxel-0.0.1-cp312-cp312-linux_x86_64.whl"

# --- Download models ---
echo "=== Downloading models ==="
python - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download("camenduru/TRELLIS.2-4B")
snapshot_download("camenduru/dinov3-vitl16-pretrain-lvd1689m")
snapshot_download("camenduru/RMBG-2.0")
PYEOF

# --- Download handler ---
echo "=== Downloading handler.py ==="
curl -fsSL "https://pub-d4542cd5f9bc434dbb7da007761dec7b.r2.dev/handler.py" -o /workspace/handler.py

# --- Create run script ---
cat > /workspace/run_spawnlab.sh << 'RUNSCRIPT'
#!/bin/bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate main
export HF_HOME=/workspace/hf_cache
export PYTHONPATH="/workspace/TRELLIS.2:${PYTHONPATH}"
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace
python -u handler.py
RUNSCRIPT
chmod +x /workspace/run_spawnlab.sh

# --- Launch ---
echo "=== Provisioning complete - starting SpawnLab worker ==="
/workspace/run_spawnlab.sh &
