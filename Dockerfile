FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HOME=/app/cache

# Camenduru wheel base URL (torch2.8+cu128, cp312, linux)
ARG CWHEELS=https://github.com/camenduru/wheels/releases/download/trellis2

# System dependencies
RUN apt-get update && apt-get install -y \
    git git-lfs wget curl aria2 \
    libjpeg-dev libgl1 libglib2.0-0 libegl1 \
    && rm -rf /var/lib/apt/lists/*

# ML dependencies
RUN pip install --no-cache-dir \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    trimesh transformers zstandard kornia timm \
    plyfile runpod requests hf_transfer huggingface_hub

RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

RUN pip install pillow-simd

# Pre-built GPU extension wheels (camenduru mirrors, torch2.8+cu128, cp312)
RUN pip install --no-cache-dir \
    "${CWHEELS}/cumesh-0.0.1-cp312-cp312-linux_x86_64.whl" \
    "${CWHEELS}/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl" \
    "${CWHEELS}/flex_gemm-0.0.1-cp312-cp312-linux_x86_64.whl" \
    "${CWHEELS}/nvdiffrast-0.4.0-cp312-cp312-linux_x86_64.whl" \
    "${CWHEELS}/nvdiffrec_render-0.0.0-cp312-cp312-linux_x86_64.whl" \
    "${CWHEELS}/o_voxel-0.0.1-cp312-cp312-linux_x86_64.whl"

# Clone camenduru's TRELLIS.2 fork (HF-compatible, correct pipeline.json)
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch dev \
    https://github.com/camenduru/TRELLIS.2-hf /app/TRELLIS.2

ENV PYTHONPATH="/app/TRELLIS.2:${PYTHONPATH}"

# Pre-download all models to HF cache (camenduru public mirrors, no token needed)
RUN HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download camenduru/TRELLIS.2-4B
RUN HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download camenduru/dinov3-vitl16-pretrain-lvd1689m
RUN HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download camenduru/RMBG-2.0

WORKDIR /app
COPY handler.py .

CMD ["python", "-u", "handler.py"]
