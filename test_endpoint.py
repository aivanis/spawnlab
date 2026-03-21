"""
Test script for the trellis2-worker RunPod endpoint.

Usage:
    python test_endpoint.py [image_path] [--resolution 512|1024|1536]

Examples:
    python test_endpoint.py                        # generates a synthetic test image
    python test_endpoint.py my_object.png          # use your own image
    python test_endpoint.py my_object.png --resolution 1024
"""

import argparse
import base64
import io
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

API_KEY     = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

if not API_KEY or not ENDPOINT_ID:
    print("ERROR: Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in a .env file.")
    print("       Copy .env.template to .env and fill in your values.")
    sys.exit(1)

BASE_URL = f"https://api.runpod.io/v2/{ENDPOINT_ID}"
HEADERS  = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def image_to_b64(path: str) -> str:
    with Image.open(path) as img:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

def synthetic_image_b64() -> str:
    img = Image.new("RGB", (512, 512), (180, 180, 180))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 80, 430, 430], fill=(210, 110, 50))
    draw.polygon([(430, 80), (512, 30), (512, 380), (430, 430)], fill=(170, 90, 40))
    draw.polygon([(80, 80), (512, 30), (430, 80)], fill=(230, 150, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def poll_job(job_id: str, timeout: int = 600) -> dict:
    deadline = time.time() + timeout
    print(f"  Job ID: {job_id}  (polling every 5s, timeout {timeout}s)")
    while time.time() < deadline:
        resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "UNKNOWN")
        print(f"  Status: {status}", end="\r")
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            print()
            return data
        time.sleep(5)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test the trellis2-worker RunPod endpoint")
    parser.add_argument("image", nargs="?", help="Path to input image (PNG/JPG). Omit to use a synthetic test image.")
    parser.add_argument("--resolution", choices=["512", "1024", "1536"], default="512",
                        help="Output resolution (default: 512, fastest for testing)")
    parser.add_argument("--decimation", type=int, default=100_000,
                        help="Target face count after decimation (default: 100000)")
    parser.add_argument("--texture-size", type=int, default=1024, choices=[512, 1024, 2048],
                        help="Texture atlas size in px (default: 1024)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rembg", action="store_true", help="Skip background removal")
    parser.add_argument("--async-mode", action="store_true",
                        help="Submit async job and poll (use for long runs)")
    parser.add_argument("--output", default="output.glb", help="Path to save the output GLB")
    args = parser.parse_args()

    # Build image payload
    if args.image:
        print(f"Loading image: {args.image}")
        img_b64 = image_to_b64(args.image)
    else:
        print("No image provided — using synthetic test image.")
        img_b64 = synthetic_image_b64()

    payload = {
        "input": {
            "image":            img_b64,
            "resolution":       int(args.resolution),
            "decimation_target": args.decimation,
            "texture_size":     args.texture_size,
            "seed":             args.seed,
            "remove_bg":        not args.no_rembg,
        }
    }

    print(f"\nEndpoint: {ENDPOINT_ID}")
    print(f"Resolution: {args.resolution}  |  Decimation: {args.decimation}  |  Texture: {args.texture_size}px\n")

    t0 = time.time()

    if args.async_mode:
        # Submit and poll
        print("Submitting async job...")
        resp = requests.post(f"{BASE_URL}/run", headers=HEADERS, json=payload, timeout=30)
        resp.raise_for_status()
        job_id = resp.json()["id"]
        result = poll_job(job_id)
        output = result.get("output", {})
    else:
        # Synchronous call (waits up to 5 min)
        print("Submitting synchronous job (waiting for result)...")
        print("  Note: first call may take 2-3 min due to cold start.")
        resp = requests.post(f"{BASE_URL}/runsync", headers=HEADERS, json=payload, timeout=600)
        resp.raise_for_status()
        result = resp.json()
        output = result.get("output", {})

    elapsed = time.time() - t0

    # Handle result
    if "error" in output:
        print(f"\nWORKER ERROR: {output['error']}")
        sys.exit(1)

    if result.get("status") in ("FAILED", "CANCELLED", "TIMED_OUT"):
        print(f"\nJOB {result['status']}")
        print(json.dumps(result, indent=2))
        sys.exit(1)

    if "glb" not in output:
        print("\nUnexpected response:")
        print(json.dumps(result, indent=2))
        sys.exit(1)

    # Save GLB
    glb_bytes = base64.b64decode(output["glb"])
    with open(args.output, "wb") as f:
        f.write(glb_bytes)

    print(f"SUCCESS in {elapsed:.1f}s")
    print(f"GLB saved to: {args.output}  ({len(glb_bytes) / 1024:.0f} KB)")
    print(f"\nOpen in Blender or drag into https://gltf.report to inspect.")


if __name__ == "__main__":
    main()
