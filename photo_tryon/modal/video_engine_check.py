"""Puts one video through the deployed video engine exactly as the app does, and times it.

usage:
    python -m modal run video_engine_check.py --video <clip.mp4> --out <folder>
        [--top polo_pink] [--bottom khaki_chinos] [--start 1.8 --end 9.4]

--start/--end cut the clip first, on Modal (the engine takes at most 10 seconds). --top "" or --bottom ""
leaves that garment out.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import modal

app = modal.App("photo-tryon-video-engine-check")
# The engine's Leffa image, copied exactly so Modal reuses it: it has OpenCV for the cut.
cut_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "accelerate", "diffusers", "transformers", "einops", "fvcore",
                 "omegaconf", "onnxruntime", "opencv-python-headless", "pycocotools", "av", "config", "cloudpickle",
                 "peft", "safetensors", "scikit-image", "scipy", "timm", "pyyaml", "psutil", "huggingface_hub", "pillow",
                 "matplotlib", "pandas", "imageio", "tqdm", "regex", "torchmetrics")
    .run_commands("git clone --depth 1 https://github.com/franciszzj/Leffa /leffa")
)


@app.function(image=cut_image, cpu=2.0, timeout=300)
def cut(video: bytes, start: float, end: float) -> bytes:
    import os
    import tempfile
    import cv2
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video)
    cap = cv2.VideoCapture(f.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path = f.name + "_cut.mp4"
    writer = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if start * fps <= i <= end * fps:
            if writer is None:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    data = open(out_path, "rb").read()
    os.unlink(f.name)
    os.unlink(out_path)
    return data


@app.local_entrypoint()
def main(video: str, out: str, top: str = "polo_pink", bottom: str = "khaki_chinos", start: float = 0.0, end: float = 0.0):
    here = Path(__file__).resolve().parent
    base = subprocess.run([sys.executable, str(here / "engine_key.py"), "video"], capture_output=True, text=True,
                          cwd=here).stdout.strip().splitlines()[-1]
    assets = here.parents[1] / "app/src/main/assets/garments/real"
    names = {g["id"]: g["name"] for g in json.loads((assets / "catalog.json").read_text())["garments"]}
    dst = Path(out)
    dst.mkdir(parents=True, exist_ok=True)

    def call(method, path, data=None, kind="application/octet-stream"):
        req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": kind})
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read()
            return json.loads(raw) if r.headers.get("Content-Type", "").startswith("application/json") else raw

    data = Path(video).read_bytes()
    if end > 0:
        data = cut.remote(data, start, end)
        print(f"cut to {start}-{end} s: {len(data) / 1e6:.1f} MB")
    t0 = time.time()
    print("health:", call("GET", "/health"), f"({time.time() - t0:.0f} s, includes waking the web server)")
    session = f"check_{int(time.time())}"     # its own name each run, as the app gives each video its own
    call("PUT", f"/video/{session}", data)
    for g in (top, bottom):
        if g:
            call("PUT", f"/garment/{g}/front", (assets / f"{g}_front.png").read_bytes())
    t1 = time.time()
    body = {"session": session, "top": top or None, "bottom": bottom or None,
            "top_name": names.get(top), "bottom_name": names.get(bottom)}
    job = call("POST", "/job", json.dumps(body).encode(), "application/json")["job"]
    last = None
    while True:
        s = call("GET", f"/job/{job}")
        if s["stage"] != last:
            print(f"{time.time() - t1:5.0f} s  {s['stage']}")
            last = s["stage"]
        if s["state"] in ("done", "error"):
            break
        time.sleep(3)
    if s["state"] == "error":
        print("error:", s["error"])
        return
    for name in ("dressed.mp4", "original.mp4"):
        (dst / name).write_bytes(call("GET", f"/job/{job}/file/{name}"))
    print(f"done in {time.time() - t1:.0f} s; engine timings {s['seconds']}; saved in {dst}")
