"""Test: sharper dressed photos.

FASHN dresses each view exactly as the engine does (photo -> crop to 768 x 1024 -> FASHN, which draws at
576 x 768). Then:
  1. only the person comes from FASHN; the face, hair and the room come from the full-size photo;
  2. the view is made 1080 x 1440 (the phone's screen width) instead of 768 x 1024.
The engine's own result (FASHN's picture stretched to 768 x 1024) is saved beside it to compare.

A separate Modal app: the deployed engine is not touched.
    python -m modal run sharp_test.py --photos <folder with view_N.jpg> --out <folder> [--top navy_polo] [--bottom khaki_chinos]
"""
import io
import time
from pathlib import Path

import modal

app = modal.App("photo-tryon-sharp-test")
leffa_weights = modal.Volume.from_name("photo-tryon-leffa-weights")

VIEWS = 8
BACK = {3, 4, 5}                 # these show the back: they get the garments' back pictures
OUT_W, OUT_H = 1080, 1440
HEAD = [1, 2, 3, 11]             # body-part labels of hat, hair, sunglasses, face
BAG = 16

# The photo engine's two images, copied exactly so Modal reuses the ones already built.
leffa_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "accelerate", "diffusers", "transformers", "einops", "fvcore",
                 "omegaconf", "onnxruntime", "opencv-python-headless", "pycocotools", "av", "config", "cloudpickle",
                 "peft", "safetensors", "scikit-image", "scipy", "timm", "pyyaml", "psutil", "huggingface_hub", "pillow",
                 "matplotlib", "pandas", "imageio", "tqdm", "regex", "torchmetrics")
    .run_commands("git clone --depth 1 https://github.com/franciszzj/Leffa /leffa")
)
fashn_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "onnxruntime-gpu==1.22.0", "pillow")
    .run_commands(
        "git clone --depth 1 https://github.com/fashn-AI/fashn-vton-1.5 /fashn",
        "pip install -e /fashn",
        "python /fashn/scripts/download_weights.py --weights-dir /weights",
    )
)


def jpeg(picture, quality=92) -> bytes:
    out = io.BytesIO()
    picture.save(out, "JPEG", quality=quality)
    return out.getvalue()


@app.cls(image=leffa_image, volumes={"/ckpts": leffa_weights}, cpu=2.0, memory=4096, timeout=600,
         scaledown_window=120, max_containers=VIEWS)
class Sharp:
    @modal.enter()
    def load(self):
        import os
        import sys
        os.chdir("/leffa")
        sys.path.insert(0, "/leffa")
        from preprocess.humanparsing.run_parsing import Parsing
        self.parsing = Parsing(atr_path="/ckpts/humanparsing/parsing_atr.onnx", lip_path="/ckpts/humanparsing/parsing_lip.onnx")

    def labels(self, picture, size):
        import numpy as np
        from PIL import Image
        parse, _ = self.parsing(picture.convert("RGB").resize((384, 512)))
        return np.array(parse.resize(size, Image.NEAREST))

    @modal.method()
    def frame(self, person: bytes) -> dict:
        """The engine's crop (768 x 1024, what FASHN is given) and the same box from the full-size photo at 1080 x 1440."""
        import numpy as np
        from PIL import Image, ImageOps
        from leffa_utils.utils import resize_and_center
        photo = ImageOps.exif_transpose(Image.open(io.BytesIO(person))).convert("RGB")
        w, h = photo.size
        scale = min(384 / w, 512 / h)
        ox, oy = (384 - w * scale) / 2, (512 - h * scale) / 2
        parse, _ = self.parsing(resize_and_center(photo, 768, 1024).resize((384, 512)))
        ys, xs = np.where(np.array(parse.resize((384, 512), Image.NEAREST)) != 0)
        x0, x1 = (xs.min() - ox) / scale, (xs.max() - ox) / scale     # the engine's Framer, step for step
        y0, y1 = (ys.min() - oy) / scale, (ys.max() - oy) / scale
        height = (y1 - y0) * 1.12
        width = max(height * 0.75, (x1 - x0) * 1.1)
        height = width / 0.75
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = (int(cx - width / 2), int(cy - height / 2), int(cx + width / 2), int(cy + height / 2))
        arr = np.asarray(photo)
        crop = Image.fromarray(arr[np.clip(np.arange(box[1], box[3]), 0, h - 1)][:, np.clip(np.arange(box[0], box[2]), 0, w - 1)])
        return {"engine_input": jpeg(crop.resize((768, 1024), Image.LANCZOS), 95),
                "full": jpeg(crop.resize((OUT_W, OUT_H), Image.LANCZOS), 95), "crop_px": crop.size}

    @modal.method()
    def compose(self, dressed: bytes, full: bytes) -> bytes:
        """The person from FASHN's picture (stretched to 1080 x 1440); the face, hair and room from the full-size photo."""
        import cv2
        import numpy as np
        from PIL import Image
        up = Image.open(io.BytesIO(dressed)).convert("RGB").resize((OUT_W, OUT_H), Image.LANCZOS)
        orig = Image.open(io.BytesIO(full)).convert("RGB")
        old = self.labels(orig, (OUT_W, OUT_H))
        new = self.labels(up, (OUT_W, OUT_H))
        person = ((old != 0) & (old != BAG)) | ((new != 0) & (new != BAG))
        # A little room around the person comes from FASHN too: where the old clothes were wider than the new
        # ones, only FASHN's picture shows what is behind them.
        region = cv2.dilate(person.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
        head = cv2.erode(np.isin(old, HEAD).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        region &= ~head
        a = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 2.5)[..., None]
        out = np.asarray(orig, np.float32) * (1 - a) + np.asarray(up, np.float32) * a
        return jpeg(Image.fromarray(out.round().clip(0, 255).astype(np.uint8)), 92)


@app.cls(image=fashn_image, gpu=["L4", "A10G"], timeout=900, scaledown_window=60, max_containers=VIEWS)
class Fashn:
    @modal.enter()
    def load(self):
        from fashn_vton import TryOnPipeline
        self.pipe = TryOnPipeline(weights_dir="/weights", device="cuda")

    @modal.method()
    def dress(self, person: bytes, top: bytes | None, bottom: bytes | None) -> bytes:
        """The engine's FASHN step with its settings, but FASHN's own picture is returned unstretched."""
        from PIL import Image
        picture = Image.open(io.BytesIO(person)).convert("RGB")
        for garment, category in ((top, "tops"), (bottom, "bottoms")):
            if garment is None:
                continue
            picture = self.pipe(person_image=picture, garment_image=Image.open(io.BytesIO(garment)).convert("RGB"),
                                category=category, garment_photo_type="flat-lay", num_samples=1,
                                num_timesteps=30, guidance_scale=1.5, seed=42, segmentation_free=True).images[0]
        return jpeg(picture, 95)


def assets() -> Path:
    """The app's catalog pictures (worked out on this computer only)."""
    return Path(__file__).resolve().parents[2] / "app/src/main/assets/garments/real"


def garment(gid: str, face: str) -> bytes | None:
    """A catalog picture put on white, the back picture for the back views when there is one (as the engine does)."""
    from PIL import Image
    if not gid:
        return None
    path = assets() / f"{gid}_{face}.png"
    if not path.exists():
        path = assets() / f"{gid}_front.png"
    picture = Image.open(path).convert("RGBA")
    canvas = Image.new("RGB", picture.size, "white")
    canvas.paste(picture, (0, 0), picture)
    out = io.BytesIO()
    canvas.save(out, "PNG")
    return out.getvalue()


@app.local_entrypoint()
def main(photos: str, out: str, top: str = "navy_polo", bottom: str = "khaki_chinos"):
    from PIL import Image, ImageDraw
    src, dst = Path(photos), Path(out)
    (dst / "engine").mkdir(parents=True, exist_ok=True)
    (dst / "sharp").mkdir(parents=True, exist_ok=True)
    views = [i for i in range(VIEWS) if (src / f"view_{i}.jpg").exists()]
    t0 = time.time()
    framed = list(Sharp().frame.map([(src / f"view_{i}.jpg").read_bytes() for i in views]))
    for i, f in zip(views, framed):
        print(f"view {i}: the person's crop in the photo is {f['crop_px'][0]} x {f['crop_px'][1]} pixels")
    jobs = [(f["engine_input"], garment(top, "back" if i in BACK else "front"),
             garment(bottom, "back" if i in BACK else "front")) for i, f in zip(views, framed)]
    dressed = list(Fashn().dress.starmap(jobs))
    sharp = list(Sharp().compose.starmap([(d, f["full"]) for d, f in zip(dressed, framed)]))
    print(f"{len(views)} views in {time.time() - t0:.0f} s")

    zooms = []
    for i, d, s in zip(views, dressed, sharp):
        engine = Image.open(io.BytesIO(d)).resize((768, 1024), Image.LANCZOS)      # what the engine sends today
        engine.save(dst / "engine" / f"view_{i}.jpg", quality=92)
        (dst / "sharp" / f"view_{i}.jpg").write_bytes(s)
        # The same face-and-chest area of both, at the size the phone shows them (1080 wide).
        box = (int(OUT_W * 0.22), 0, int(OUT_W * 0.78), int(OUT_H * 0.36))
        a = engine.resize((OUT_W, OUT_H), Image.LANCZOS).crop(box)
        b = Image.open(io.BytesIO(s)).crop(box)
        pair = Image.new("RGB", (a.width * 2 + 10, a.height + 30), "white")
        pair.paste(a, (0, 30))
        pair.paste(b, (a.width + 10, 30))
        ImageDraw.Draw(pair).text((8, 8), f"view {i}: today (left) | sharper (right)", fill="black")
        zooms.append(pair)
    for k in range(0, len(zooms), 2):
        rows = zooms[k:k + 2]
        sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height
        sheet.save(dst / f"zoom_{k // 2}.jpg", quality=90)
    print("saved in", dst)
