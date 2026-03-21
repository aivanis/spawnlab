#!/bin/bash
# Downloads pre-built CUDA wheels for TRELLIS.2 dependencies
# cu126 + torch2.6 + cp312 + linux
# After running, upload the wheels/ folder contents to a GitHub release on your repo

set -e

WHEELS_DIR="wheels"
BASE_URL="https://github.com/PozzettiAndrea/cuda-wheels/releases/download"

mkdir -p "$WHEELS_DIR"

echo "Downloading wheels..."

curl -L -o "$WHEELS_DIR/flash_attn-2.8.3.whl" \
    "${BASE_URL}/flash_attn-latest/flash_attn-2.8.3%2Bcu126torch2.6-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"

curl -L -o "$WHEELS_DIR/nvdiffrast-0.4.0.whl" \
    "${BASE_URL}/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch2.6-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"

curl -L -o "$WHEELS_DIR/nvdiffrec_render-0.0.1.whl" \
    "${BASE_URL}/nvdiffrec_render-latest/nvdiffrec_render-0.0.1%2Bcu126torch2.6-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"

curl -L -o "$WHEELS_DIR/cumesh-0.0.1.whl" \
    "${BASE_URL}/cumesh-latest/cumesh-0.0.1%2Bcu126torch2.6-cp312-cp312-manylinux_2_35_x86_64.whl"

curl -L -o "$WHEELS_DIR/flex_gemm-1.0.0.whl" \
    "${BASE_URL}/flex_gemm-latest/flex_gemm-1.0.0%2Bcu126torch2.6-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"

curl -L -o "$WHEELS_DIR/o_voxel-0.0.1.whl" \
    "${BASE_URL}/o_voxel-latest/o_voxel-0.0.1%2Bcu126torch2.6-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"

echo ""
echo "Done. Files in ./$WHEELS_DIR:"
ls -lh "$WHEELS_DIR"
echo ""
echo "Next: upload these files to a GitHub release on your repo:"
echo "  gh release create wheels-cu126-torch260-cp312 wheels/*.whl --repo aivanis/trellis2-worker --title 'Wheels cu126+torch2.6+cp312'"
