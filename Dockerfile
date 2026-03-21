FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git git-lfs \
    libjpeg-dev libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3

# PyTorch 2.6.0 + CUDA 12.4
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Basic ML dependencies (from TRELLIS.2 setup.sh --basic)
RUN pip install \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
    trimesh transformers tensorboard pandas lpips zstandard kornia timm \
    runpod requests huggingface_hub

RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

RUN pip install pillow-simd

# flash-attn (pre-built wheel for cu124 + torch 2.6.0)
RUN pip install flash-attn==2.7.3

# Build GPU extensions (from TRELLIS.2 setup.sh)
RUN mkdir -p /tmp/extensions

RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast --no-build-isolation

RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec && \
    pip install /tmp/extensions/nvdiffrec --no-build-isolation

RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive && \
    pip install /tmp/extensions/CuMesh --no-build-isolation

RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive && \
    pip install /tmp/extensions/FlexGEMM --no-build-isolation

# Clone TRELLIS.2 repo (includes o-voxel submodule)
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git /app/TRELLIS.2

RUN cp -r /app/TRELLIS.2/o-voxel /tmp/extensions/o-voxel && \
    pip install /tmp/extensions/o-voxel --no-build-isolation

RUN pip install -e /app/TRELLIS.2

# Download model weights into image
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/TRELLIS.2-4B', local_dir='/app/models/TRELLIS.2-4B')"

WORKDIR /app
COPY handler.py .

CMD ["python", "-u", "handler.py"]
