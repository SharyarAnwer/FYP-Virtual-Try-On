"""The Photo Try-On engine on Modal: dresses a capture's eight views.

FASHN VTON v1.5 dresses all eight, each photo first cropped to the person so the model spends its pixels on
them and not the room. In the side-by-side test on the same eight photos, FASHN drew the right shirt in the
right colour on every view, where Leffa got the colour wrong on two and never managed the straight back.

Leffa is kept: set MODEL to "leffa" to go back to Leffa for seven views and FASHN for the straight back.

Once, to put Leffa's trained files into a Modal volume:
    python -m modal run engine.py::fetch_weights
Deploy (once, and again after any change):
    python -m modal deploy engine.py
The address for the app:
    python engine_key.py

The phone talks to it exactly as it talked to the Kaggle engine: PUT the eight photos and the garment
pictures, POST a job, GET its progress and the dressed views.
"""
import io
import json
import os
import secrets
import threading
import time
import traceback

import modal

app = modal.App("photo-tryon-engine")
weights = modal.Volume.from_name("photo-tryon-leffa-weights", create_if_missing=True)

VIEWS = [f"view_{i}" for i in range(8)]
ORDER = [0, 4, 2, 6, 1, 5, 3, 7]          # front, back, the sides, then the four in between
BACK = {3, 4, 5}                           # these show the back: they get the garments' back pictures
FASHN_VIEW = 4                             # the straight back
GPUS = ["L4", "A10G"]                      # whichever is free first
MODEL = "fashn"                            # "fashn": FASHN for all views. "leffa": Leffa, and FASHN for the straight back.

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
web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]", "pillow")


@app.function(image=leffa_image, volumes={"/ckpts": weights}, timeout=3600)
def fetch_weights():
    """Downloads Leffa's trained files into the volume: about 16 GB, once."""
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="franciszzj/Leffa", local_dir="/ckpts",
                      ignore_patterns=["pose_transfer.pth", "stable-diffusion-xl-1.0-inpainting-0.1/*", "examples/*", "assets/*"])
    weights.commit()
    print(os.listdir("/ckpts"))


@app.cls(image=leffa_image, gpu=GPUS, volumes={"/ckpts": weights}, timeout=600, scaledown_window=120, max_containers=4)
class LeffaDresser:
    STEPS = 12                             # 12 looked as good as 30 in testing
    HEAD = [1, 2, 3, 11]                   # hat, hair (a beard counts as hair), sunglasses, face

    @modal.enter()
    def load(self):
        import sys
        os.chdir("/leffa")
        sys.path.insert(0, "/leffa")
        import threading as _t
        from leffa.model import LeffaModel
        from leffa.inference import LeffaInference
        from leffa_utils.densepose_predictor import DensePosePredictor
        from preprocess.humanparsing.run_parsing import Parsing
        from preprocess.openpose.run_openpose import OpenPose
        ck = "/ckpts"
        self.densepose = DensePosePredictor(config_path=f"{ck}/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
                                            weights_path=f"{ck}/densepose/model_final_162be9.pkl")
        self.parsing = Parsing(atr_path=f"{ck}/humanparsing/parsing_atr.onnx", lip_path=f"{ck}/humanparsing/parsing_lip.onnx")
        self.openpose = OpenPose(body_model_path=f"{ck}/openpose/body_pose_model.pth")
        self.inference = {}
        for name, file in (("viton_hd", "virtual_tryon.pth"), ("dress_code", "virtual_tryon_dc.pth")):
            model = LeffaModel(pretrained_model_name_or_path=f"{ck}/stable-diffusion-inpainting",
                               pretrained_model=f"{ck}/{file}", dtype="float16")
            self.inference[name] = LeffaInference(model=model)
        self.lock = _t.Lock()

    def parse(self, picture):
        import numpy as np
        from PIL import Image
        parse, _ = self.parsing(picture.resize((384, 512)))
        return np.array(parse.resize((384, 512), Image.NEAREST))

    def prepare(self, photo):
        """A phone photo -> the person, upright, filling a 768 x 1024 picture: the same size in every view."""
        import numpy as np
        from PIL import Image, ImageOps
        from leffa_utils.utils import resize_and_center
        photo = ImageOps.exif_transpose(photo).convert("RGB")
        w, h = photo.size
        scale = min(384 / w, 512 / h)
        ox, oy = (384 - w * scale) / 2, (512 - h * scale) / 2
        body = self.parse(resize_and_center(photo, 768, 1024)) != 0
        ys, xs = np.where(body)
        if len(ys) < 500:
            return resize_and_center(photo, 768, 1024).convert("RGB")
        x0, x1 = (xs.min() - ox) / scale, (xs.max() - ox) / scale
        y0, y1 = (ys.min() - oy) / scale, (ys.max() - oy) / scale
        height = (y1 - y0) * 1.12
        width = max(height * 0.75, (x1 - x0) * 1.1)
        height = width / 0.75
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = [int(cx - width / 2), int(cy - height / 2), int(cx + width / 2), int(cy + height / 2)]
        canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), "white")
        canvas.paste(photo, (-box[0], -box[1]))
        out = canvas.resize((768, 1024), Image.LANCZOS)
        luma = np.asarray(out.convert("L"), np.float32)[100:920, 230:538].mean()
        if luma < 85:                      # a dim photo is lifted: the model repaints dark photos blotchily
            gamma = np.log(110 / 255) / np.log(max(luma, 10) / 255)
            out = Image.fromarray((255 * (np.asarray(out, np.float32) / 255) ** gamma).clip(0, 255).astype(np.uint8))
        return out

    @staticmethod
    def garment_picture(data: bytes):
        from PIL import Image
        picture = Image.open(io.BytesIO(data)).convert("RGBA")
        canvas = Image.new("RGB", (768, 1024), "white")
        picture.thumbnail((700, 950))
        canvas.paste(picture, ((768 - picture.width) // 2, (1024 - picture.height) // 2), picture)
        return canvas

    def put_on(self, person, garment, kind, model_type, seed=42):
        """Leffa's own steps, with the head kept and only the body repainted (see the Kaggle notebook)."""
        import cv2
        import numpy as np
        from PIL import Image, ImageFilter
        from leffa.transform import LeffaTransform
        from leffa_utils.utils import get_agnostic_mask_hd, get_agnostic_mask_dc
        src = person.convert("RGB")
        src_array = np.array(src)
        model_parse, _ = self.parsing(src.resize((384, 512)))
        keypoints = self.openpose(src.resize((384, 512)))
        if model_type == "viton_hd":
            mask = get_agnostic_mask_hd(model_parse, keypoints, kind)
            densepose = Image.fromarray(self.densepose.predict_seg(src_array)[:, :, ::-1])
        else:
            mask = get_agnostic_mask_dc(model_parse, keypoints, kind)
            seg = self.densepose.predict_iuv(src_array)[:, :, 0:1]
            densepose = Image.fromarray(np.concatenate([seg] * 3, axis=-1))
        parse = np.array(model_parse.resize((384, 512), Image.NEAREST))
        head = cv2.dilate(np.isin(parse, self.HEAD).astype(np.uint8), np.ones((5, 5), np.uint8))
        area = (np.array(mask.resize((384, 512), Image.NEAREST).convert("L")) > 127) & (head == 0)
        mask = Image.fromarray((area * 255).astype(np.uint8)).resize((768, 1024), Image.NEAREST)
        data = LeffaTransform()({"src_image": [src], "ref_image": [garment], "mask": [mask], "densepose": [densepose]})
        out = self.inference[model_type](data, ref_acceleration=False, num_inference_steps=self.STEPS,
                                         guidance_scale=2.5, seed=seed, repaint=False)
        generated = out["generated_image"][0].convert("RGB")
        body = parse != 0
        keep = area & (cv2.dilate(body.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
        rows = np.where(np.isin(parse, self.HEAD).any(axis=1))[0]
        if len(rows):
            keep[:rows.max()] &= body[:rows.max()]
        soft = Image.fromarray((keep * 255).astype(np.uint8)).resize((768, 1024), Image.NEAREST)
        soft = soft.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(3))
        soft = Image.composite(Image.new("L", soft.size, 0), soft, Image.fromarray(head * 255).resize((768, 1024), Image.NEAREST))
        return Image.composite(generated, src, soft)

    @modal.method()
    def frame(self, person: bytes) -> bytes:
        """The photo cropped and scaled as the dressed views are, for the view FASHN dresses."""
        from PIL import Image
        out = io.BytesIO()
        self.prepare(Image.open(io.BytesIO(person))).save(out, "JPEG", quality=95)
        return out.getvalue()

    @modal.method()
    def dress(self, person: bytes, top: bytes | None, bottom: bytes | None) -> bytes:
        from PIL import Image
        picture = self.prepare(Image.open(io.BytesIO(person)))
        if top is not None:
            picture = self.put_on(picture, self.garment_picture(top), "upper_body", "viton_hd")
        if bottom is not None:
            picture = self.put_on(picture, self.garment_picture(bottom), "lower_body", "dress_code")
        out = io.BytesIO()
        picture.save(out, "JPEG", quality=92)
        return out.getvalue()


@app.cls(image=leffa_image, volumes={"/ckpts": weights}, cpu=2.0, memory=4096, timeout=300, scaledown_window=300,
         max_containers=8)
class Framer:
    """Crops a photo to the person, on an ordinary CPU: Leffa's body-part model finds them, nothing else of Leffa runs."""

    @modal.enter()
    def load(self):
        import sys
        os.chdir("/leffa")
        sys.path.insert(0, "/leffa")
        from preprocess.humanparsing.run_parsing import Parsing
        self.parsing = Parsing(atr_path="/ckpts/humanparsing/parsing_atr.onnx", lip_path="/ckpts/humanparsing/parsing_lip.onnx")

    @modal.method()
    def frame(self, person: bytes) -> bytes:
        """The person filling a 768 x 1024 picture, with the margin the dressed views have."""
        import numpy as np
        from PIL import Image, ImageOps
        from leffa_utils.utils import resize_and_center
        photo = ImageOps.exif_transpose(Image.open(io.BytesIO(person))).convert("RGB")
        w, h = photo.size
        scale = min(384 / w, 512 / h)
        ox, oy = (384 - w * scale) / 2, (512 - h * scale) / 2
        parse, _ = self.parsing(resize_and_center(photo, 768, 1024).resize((384, 512)))
        body = np.array(parse.resize((384, 512), Image.NEAREST)) != 0
        ys, xs = np.where(body)
        if len(ys) < 500:
            out_picture = resize_and_center(photo, 768, 1024).convert("RGB")
        else:
            x0, x1 = (xs.min() - ox) / scale, (xs.max() - ox) / scale
            y0, y1 = (ys.min() - oy) / scale, (ys.max() - oy) / scale
            height = (y1 - y0) * 1.12
            width = max(height * 0.75, (x1 - x0) * 1.1)
            height = width / 0.75
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            box = (int(cx - width / 2), int(cy - height / 2), int(cx + width / 2), int(cy + height / 2))
            # Past the photo's edge the nearest row and column of the photo are repeated, not white: a white band
            # showed where feet were near the bottom edge.
            arr = np.asarray(photo)
            ys_i = np.clip(np.arange(box[1], box[3]), 0, h - 1)
            xs_i = np.clip(np.arange(box[0], box[2]), 0, w - 1)
            out_picture = Image.fromarray(arr[ys_i][:, xs_i]).resize((768, 1024), Image.LANCZOS)
        out = io.BytesIO()
        out_picture.save(out, "JPEG", quality=95)
        return out.getvalue()


@app.cls(image=fashn_image, gpu=GPUS, timeout=600, scaledown_window=120, max_containers=8)
class FashnDresser:
    @modal.enter()
    def load(self):
        from fashn_vton import TryOnPipeline
        self.pipe = TryOnPipeline(weights_dir="/weights", device="cuda")

    @modal.method()
    def dress(self, person: bytes, top: bytes | None, bottom: bytes | None) -> bytes:
        from PIL import Image
        picture = Image.open(io.BytesIO(person)).convert("RGB")
        # The shirt, then the trousers: the order that gave a clean straight back in the FASHN test.
        for garment, category in ((top, "tops"), (bottom, "bottoms")):
            if garment is None:
                continue
            picture = self.pipe(person_image=picture, garment_image=Image.open(io.BytesIO(garment)).convert("RGB"),
                                category=category, garment_photo_type="flat-lay", num_samples=1,
                                num_timesteps=30, guidance_scale=1.5, seed=42, segmentation_free=True).images[0]
        picture = picture.resize((768, 1024), Image.LANCZOS)          # the same size as Leffa's views
        out = io.BytesIO()
        picture.save(out, "JPEG", quality=92)
        return out.getvalue()


def on_white(data: bytes) -> bytes:
    """A garment picture with a see-through background, put on white: both models expect a product photo."""
    from PIL import Image
    picture = Image.open(io.BytesIO(data))
    if picture.mode in ("RGBA", "LA", "P"):
        picture = picture.convert("RGBA")
        canvas = Image.new("RGB", picture.size, "white")
        canvas.paste(picture, (0, 0), picture)
        picture = canvas
    out = io.BytesIO()
    picture.convert("RGB").save(out, "PNG")
    return out.getvalue()


@app.function(image=web_image, secrets=[modal.Secret.from_name("photo-tryon-key")],
              max_containers=1, scaledown_window=1200, timeout=1800)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response

    key = os.environ["TRYON_KEY"]
    api = FastAPI()
    sessions, garments, jobs = {}, {}, {}
    lock = threading.Lock()

    def error(code, message):
        return JSONResponse({"error": message}, status_code=code)

    def run_job(job):
        try:
            person = sessions[job["session"]]
            top = garments.get(job["top"]) if job["top"] else None
            bottom = garments.get(job["bottom"]) if job["bottom"] else None

            def one(index):
                view = VIEWS[index]
                face = "back" if index in BACK else "front"
                try:
                    photo = person[view]
                    by_fashn = MODEL == "fashn" or index == FASHN_VIEW
                    if by_fashn:
                        photo = Framer().frame.remote(photo)
                    dresser = FashnDresser() if by_fashn else LeffaDresser()
                    picture = dresser.dress.remote(photo,
                                                   (top.get(face) or top.get("front")) if top else None,
                                                   (bottom.get(face) or bottom.get("front")) if bottom else None)
                    with lock:
                        job["files"][view] = picture
                        job["parts"].append(view)
                        job["seconds"][view] = round(time.time() - job["t0"], 1)
                        job["stage"] = f"{len(job['parts'])} of {len(person)} views ready"
                except Exception:
                    with lock:
                        job["error"] = traceback.format_exc()[-900:]

            threads = [threading.Thread(target=one, args=(i,)) for i in ORDER if VIEWS[i] in person]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        except Exception:
            job["error"] = traceback.format_exc()[-900:]
        job["state"] = "error" if job["error"] else "done"
        job["stage"] = job["state"]
        job["seconds"]["total"] = round(time.time() - job["t0"], 1)

    @api.middleware("http")
    async def check_key(request: Request, call_next):
        parts = [p for p in request.url.path.split("/") if p]
        if len(parts) < 2 or parts[0] != "k" or parts[1] != key:
            return error(403, "wrong key")
        return await call_next(request)

    @api.get("/k/{k}/health")
    def health():
        return {"ok": True, "views": len(VIEWS), "model": "FASHN VTON v1.5" if MODEL == "fashn" else "Leffa, FASHN VTON v1.5 for the straight back",
                "busy": any(j["state"] == "running" for j in jobs.values()),
                "garments": {g: sorted(f) for g, f in garments.items()}}

    @api.put("/k/{k}/session/{session}/{view}")
    async def put_photo(session: str, view: str, request: Request):
        if view not in VIEWS:
            return error(404, "not found")
        sessions.setdefault(session, {})[view] = await request.body()
        return {"ok": True}

    @api.put("/k/{k}/garment/{garment}/{face}")
    async def put_garment(garment: str, face: str, request: Request):
        if face not in ("front", "back"):
            return error(404, "not found")
        garments.setdefault(garment, {})[face] = on_white(await request.body())
        return {"ok": True}

    @api.post("/k/{k}/job")
    async def post_job(request: Request):
        q = json.loads(await request.body() or b"{}")
        session, top, bottom = q.get("session"), q.get("top"), q.get("bottom")
        if session not in sessions or "view_0" not in sessions[session]:
            return error(400, "photos not uploaded")
        for g in (top, bottom):
            if g and g not in garments:
                return error(400, f"garment {g} not uploaded")
        if not top and not bottom:
            return error(400, "no garment chosen")
        name = f"{session}|{top}|{bottom}"
        with lock:
            for j in jobs.values():
                if j["name"] == name and j["state"] != "error":
                    return {"job": j["id"]}
            jid = secrets.token_hex(6)
            jobs[jid] = {"id": jid, "name": name, "session": session, "top": top, "bottom": bottom, "state": "running",
                         "stage": "dressing your views", "parts": [], "files": {}, "seconds": {}, "error": None, "t0": time.time()}
        threading.Thread(target=run_job, args=(jobs[jid],), daemon=True).start()
        return {"job": jid}

    @api.get("/k/{k}/job/{jid}")
    def get_job(jid: str):
        j = jobs.get(jid)
        if j is None:
            return error(404, "not found")
        return {k: j[k] for k in ("id", "state", "stage", "parts", "seconds", "error")}

    @api.get("/k/{k}/job/{jid}/file/{name}")
    def get_file(jid: str, name: str):
        j = jobs.get(jid)
        view = name[:-4] if name.endswith(".jpg") else name
        if j is None or view not in j["files"]:
            return error(404, "not found")
        return Response(j["files"][view], media_type="image/jpeg")

    return api
