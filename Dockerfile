FROM runpod/pytorch:1.0.3-cu1281-torch260-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# cu126+torch2.6+cp312 pre-built wheel base URL
ARG WHEELS=https://github.com/PozzettiAndrea/cuda-wheels/releases/download

# System dependencies (PyTorch already in base image)
RUN apt-get update && apt-get install -y \
    git git-lfs \
    libjpeg-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Basic ML dependencies (from TRELLIS.2 setup.sh --basic)
RUN pip install \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
    trimesh transformers tensorboard pandas lpips zstandard kornia timm \
    runpod requests huggingface_hub

RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

RUN pip install pillow-simd

# Pre-built GPU extension wheels (cu126, torch2.6, cp312, linux)
RUN pip install \
    "${WHEELS}/flash_attn-latest/flash_attn-2.8.3%2Bcu126torch2.6-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl" \
    "${WHEELS}/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch2.6-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl" \
    "${WHEELS}/nvdiffrec_render-latest/nvdiffrec_render-0.0.1%2Bcu126torch2.6-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl" \
    "${WHEELS}/cumesh-latest/cumesh-0.0.1%2Bcu126torch2.6-cp312-cp312-manylinux_2_35_x86_64.whl" \
    "${WHEELS}/flex_gemm-latest/flex_gemm-1.0.0%2Bcu126torch2.6-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl" \
    "${WHEELS}/o_voxel-latest/o_voxel-0.0.1%2Bcu126torch2.6-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"

# Clone TRELLIS.2 repo and install (o-voxel already installed via wheel above)
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git /app/TRELLIS.2
RUN pip install -e /app/TRELLIS.2

# Download all model weights into image

# Main TRELLIS.2-4B weights
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/TRELLIS.2-4B', local_dir='/app/models/TRELLIS.2-4B')"

# Sparse structure decoder from original TRELLIS repo (referenced in pipeline.json)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/TRELLIS-image-large', \
    allow_patterns=['ckpts/ss_dec_conv3d_16l8_fp16*'])"

# DINOv3 image encoder (image_cond_model in pipeline.json)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('facebook/dinov3-vitl16-pretrain-lvd1689m')"

# Background removal model (rembg_model in pipeline.json)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('briaai/RMBG-2.0')"

WORKDIR /app
COPY handler.py .

CMD ["python", "-u", "handler.py"]
