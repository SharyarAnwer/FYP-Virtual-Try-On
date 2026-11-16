"""Dresses all eight views of one capture with FASHN VTON v1.5 alone, to compare with the engine.

A Modal app of its own: it does not touch the deployed engine (engine.py) or its results.

usage:
    python -m modal run fashn_all_views.py --photos <folder with view_0.jpg ... view_7.jpg> --out <folder>
        [--top maroon_polo] [--bottom khaki_chinos] [--compare <folder with the engine's view_N.jpg>]
        [--crop] [--white]

--crop crops each photo to the person first, with the deployed engine's own cropping (read-only use).
--white cuts the person out and puts them on a white background before dressing; the photo given to the
model is saved beside the result as input_view_N.jpg.

The garments are read from the app's catalog pictures: <id>_front.png and, for the three views of the
back (3, 4 and 5), <id>_back.png when the garment has one. With --compare, a sheet is written with the
photos, FASHN's views and the engine's views in three rows.
"""
import io
import time
from pathlib import Path

import modal

app = modal.App("photo-tryon-fashn-all-views")

fashn_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "onnxruntime-gpu==1.22.0", "pillow")
    .run_commands(
        "git clone --depth 1 https://github.com/fashn-AI/fashn-vton-1.5 /fashn",
        "pip install -e /fashn",
        "python /fashn/scripts/download_weights.py --weights-dir /weights",
    )
    .pip_install("rembg")
    .run_commands("python -c \"from rembg import new_session; new_session('isnet-general-use')\"")
)

VIEWS = 8
BACK = {3, 4, 5}


def assets() -> Path:
    """The app's catalog pictures. Only on this computer: the GPU containers never need them."""
    return Path(__file__).resolve().parents[2] / "app/src/main/assets/garments/real"


@app.cls(image=fashn_image, gpu=["L4", "A10G"], timeout=900, scaledown_window=60, max_containers=VIEWS)
class Fashn:
    @modal.enter()
    def load(self):
        from fashn_vton import TryOnPipeline
        self.pipe = TryOnPipeline(weights_dir="/weights", device="cuda")

    def on_white_background(self, picture):
        """The person cut out and put on pure white; only the largest piece is kept as the person."""
        import cv2
        import numpy as np
        from PIL import Image
        from rembg import new_session, remove
        if not hasattr(self, "cutter"):
            self.cutter = new_session("isnet-general-use")
        alpha = np.array(remove(picture, session=self.cutter, only_mask=True).convert("L"), np.float32) / 255
        pieces, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > 0.5).astype(np.uint8))
        if pieces > 2:
            biggest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
            alpha *= cv2.dilate((labels == biggest).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        a = alpha[..., None]
        rgb = np.asarray(picture, np.float32)
        return Image.fromarray((rgb * a + 255 * (1 - a)).clip(0, 255).astype(np.uint8))

    @modal.method()
    def dress(self, person: bytes, top: bytes | None, bottom: bytes | None, white: bool = False) -> dict:
        from PIL import Image, ImageOps
        t = time.time()
        picture = ImageOps.exif_transpose(Image.open(io.BytesIO(person))).convert("RGB")
        size = picture.size
        given = None
        if white:
            picture = self.on_white_background(picture)
            given = io.BytesIO()
            picture.save(given, "JPEG", quality=92)
            given = given.getvalue()
        # The shirt, then the trousers: the order that gave a clean result in the FASHN tests.
        for garment, category in ((top, "tops"), (bottom, "bottoms")):
            if garment is None:
                continue
            picture = self.pipe(person_image=picture, garment_image=Image.open(io.BytesIO(garment)).convert("RGB"),
                                category=category, garment_photo_type="flat-lay", num_samples=1,
                                num_timesteps=30, guidance_scale=1.5, seed=42, segmentation_free=True).images[0]
        out = io.BytesIO()
        picture.save(out, "JPEG", quality=92)
        return {"jpeg": out.getvalue(), "given": given, "seconds": round(time.time() - t, 1), "size_in": size, "size_out": picture.size}


def on_white(path: Path) -> bytes:
    """A catalog picture with a see-through background, put on white: FASHN expects a product photo."""
    from PIL import Image
    picture = Image.open(path).convert("RGBA")
    canvas = Image.new("RGB", picture.size, "white")
    canvas.paste(picture, (0, 0), picture)
    out = io.BytesIO()
    canvas.save(out, "PNG")
    return out.getvalue()


def garment(gid: str, face: str) -> bytes | None:
    if not gid:
        return None
    path = assets() / f"{gid}_{face}.png"
    if not path.exists():
        path = assets() / f"{gid}_front.png"
    return on_white(path)


@app.local_entrypoint()
def main(photos: str, out: str, top: str = "maroon_polo", bottom: str = "khaki_chinos", compare: str = "",
         crop: bool = False, white: bool = False):
    src, dst = Path(photos), Path(out)
    dst.mkdir(parents=True, exist_ok=True)
    views = [i for i in range(VIEWS) if (src / f"view_{i}.jpg").exists()]
    people = [(src / f"view_{i}.jpg").read_bytes() for i in views]
    if crop:
        framer = modal.Cls.from_name("photo-tryon-engine", "Framer")()
        people = list(framer.frame.map(people))
        print(f"cropped {len(people)} photos with the engine's cropping")
    jobs = [(people[k], garment(top, "back" if i in BACK else "front"), garment(bottom, "back" if i in BACK else "front"), white)
            for k, i in enumerate(views)]
    print(f"dressing {len(views)} views in {top} + {bottom} with FASHN VTON v1.5 ...")
    t0 = time.time()
    for i, r in zip(views, Fashn().dress.starmap(jobs, order_outputs=True)):
        (dst / f"view_{i}.jpg").write_bytes(r["jpeg"])
        if r.get("given"):
            (dst / f"input_view_{i}.jpg").write_bytes(r["given"])
        print(f"view {i}: {r['seconds']}s on its GPU, {r['size_in']} -> {r['size_out']}")
    print(f"all {len(views)} views in {time.time() - t0:.0f}s of waiting")

    if compare:
        from PIL import Image, ImageDraw, ImageOps
        rows = ([("given to FASHN", dst)] if white else [("photo", src)]) + [("FASHN", dst), ("engine", Path(compare))]
        sheet = Image.new("RGB", (384 * VIEWS, 540 * len(rows)), "white")
        draw = ImageDraw.Draw(sheet)
        for r, (name, folder) in enumerate(rows):
            for i in views:
                path = folder / (f"input_view_{i}.jpg" if r == 0 and white else f"view_{i}.jpg")
                if not path.exists():
                    continue
                pic = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
                pic.thumbnail((384, 512))
                sheet.paste(pic, (i * 384 + (384 - pic.width) // 2, r * 540 + 28))
                draw.text((i * 384 + 8, r * 540 + 8), f"{name} view {i}", fill="black")
        sheet.save(dst / "compare.jpg", quality=90)
        print("comparison sheet:", dst / "compare.jpg")
