"""The Video Try-On engine on Modal: dresses a short video (record, then dress; not live).

A Modal app of its own, with its own address, so the photo engine (engine.py) is never touched. Same key.
    Deploy (once, and again after any change):   python -m modal deploy video_engine.py
    The address for the app:                     python engine_key.py video

How a video is dressed (the settings that tested best in video_dry_run.py, 2026-11-17):
  1. 81 frames are taken evenly from the whole video (at most 10 s): MagicTryOn's clip length. The result is
     played back over the same time as the video, so about 8 frames a second for 10 s.
  2. The frames are cropped with one box around the person in all of them, to 624 x 832.
  3. Leffa's helpers (in the Leffa volume) find where the clothes go (masks), the outline and the face, and
     draw the body-part map (DensePose) on black.
  4. MagicTryOn-1.3B (normal model, 20 steps, TeaCache) dresses the top, then the bottom, inside a tight mask:
     the person plus 15 pixels, never the face, beard or hair, and never the room beside the head.

API, everything under /k/<key>:
    GET  /health
    PUT  /video/<session>                 the video file as recorded or chosen (mp4), at most 10 s
    PUT  /garment/<id>/front              a garment picture (a see-through background is put on white)
    POST /job   {"session", "top", "bottom", "top_name", "bottom_name"}   -> {"job"}
    GET  /job/<job>                       {"state": running | done | error, "stage", "error", "seconds"}
    GET  /job/<job>/file/dressed.mp4      the dressed video; original.mp4 is the same crop undressed

MagicTryOn (github.com/vivoCameraResearch/Magic-TryOn) is CC BY-NC-SA 4.0: non-commercial use only.
"""
import io
import json
import os
import secrets
import threading
import time
import traceback
from pathlib import Path

import modal

app = modal.App("photo-tryon-video-engine")
leffa_weights = modal.Volume.from_name("photo-tryon-leffa-weights")
mt_weights = modal.Volume.from_name("photo-tryon-magictryon-weights")
runs = modal.Volume.from_name("photo-tryon-video-runs", create_if_missing=True)
progress = modal.Dict.from_name("photo-tryon-video-progress", create_if_missing=True)

MT_COMMIT = "0ef4b17869d9cccc86c61f5b7ec656d365e0ae82"
FRAMES = 81
W, H = 624, 832
STEPS, GUIDANCE, SEED = 20, 3.0, 43
HEAD = [1, 2, 3, 11]             # body-part labels of hat, hair, sunglasses, face
BAG = 16
TIGHT = 15                       # the model may redraw the person plus this many pixels
MAX_SECONDS = 10.5
CAPTIONS = {                     # hand-written; any other garment is described by its catalog name
    "navy_polo": "a navy blue short-sleeve polo shirt with a classic collar, a two-button placket with white buttons and ribbed sleeve cuffs",
    "polo_pink": "a pink short-sleeve polo shirt with a ribbed collar, a two-button placket with white buttons and ribbed sleeve cuffs",
    "maroon_polo": "a maroon red short-sleeve polo shirt with a classic collar, a two-button placket with white buttons and ribbed sleeve cuffs",
    "khaki_chinos": "khaki beige straight-leg tailored trousers with belt loops, a pressed front crease and a small green and red tab at the waistband",
    "black_slim_trousers": "black straight-leg tailored trousers with belt loops, a waistband tab and pressed front creases",
}
NEGATIVE = ("衣服质量差的，纽扣不整齐的，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
            "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
            "杂乱的背景，三条腿，背景人很多，倒着走")            # MagicTryOn's own negative prompt

# The same images as video_dry_run.py, so Modal reuses the ones already built.
leffa_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "accelerate", "diffusers", "transformers", "einops", "fvcore",
                 "omegaconf", "onnxruntime", "opencv-python-headless", "pycocotools", "av", "config", "cloudpickle",
                 "peft", "safetensors", "scikit-image", "scipy", "timm", "pyyaml", "psutil", "huggingface_hub", "pillow",
                 "matplotlib", "pandas", "imageio", "tqdm", "regex", "torchmetrics")
    .run_commands("git clone --depth 1 https://github.com/franciszzj/Leffa /leffa")
)
mt_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "diffusers==0.32.2", "transformers==4.50.1", "accelerate==1.5.2",
                 "omegaconf==2.3.0", "einops", "opencv-python-headless==4.11.0.86", "imageio==2.37.0",
                 "imageio-ffmpeg==0.6.0", "safetensors", "sentencepiece", "ftfy", "regex", "timm", "tqdm", "numpy<2",
                 "pillow", "huggingface_hub==0.29.3", "func_timeout")
    .run_commands(f"git clone https://github.com/vivoCameraResearch/Magic-TryOn /mt && cd /mt && git checkout {MT_COMMIT}")
)
web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]", "pillow")


class VideoProblem(Exception):
    """A problem with the video itself, told to the user as it is."""


@app.function(image=leffa_image, gpu=["L4", "A10G"], memory=8192, timeout=1200,
              volumes={"/ckpts": leffa_weights, "/r": runs})
def prepare(video: bytes, run: str) -> dict:
    """81 frames from the whole video, cropped around the person, with masks, outline, face and body-part maps,
    saved in the runs volume under <run>/. Returns the frame rate that plays them over the video's own length."""
    import sys
    import tempfile
    os.chdir("/leffa")
    sys.path.insert(0, "/leffa")
    import cv2
    import numpy as np
    from PIL import Image
    from leffa_utils.densepose_predictor import DensePosePredictor
    from leffa_utils.utils import get_agnostic_mask_dc, get_agnostic_mask_hd, resize_and_center
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video)
    cap = cv2.VideoCapture(f.name)             # turns the frames upright from the video's rotation tag
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    os.unlink(f.name)
    if len(frames) < 10:
        raise VideoProblem("The video could not be read.")
    seconds = len(frames) / fps
    if seconds > MAX_SECONDS:
        raise VideoProblem(f"The video is {seconds:.0f} seconds long. Choose one of 10 seconds or less.")
    clip = [frames[i] for i in np.linspace(0, len(frames) - 1, FRAMES).round().astype(int)]

    ck = "/ckpts"
    parsing = Parsing(atr_path=f"{ck}/humanparsing/parsing_atr.onnx", lip_path=f"{ck}/humanparsing/parsing_lip.onnx")
    openpose = OpenPose(body_model_path=f"{ck}/openpose/body_pose_model.pth")
    densepose = DensePosePredictor(config_path=f"{ck}/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
                                   weights_path=f"{ck}/densepose/model_final_162be9.pkl")

    # One box around the person in every frame, so the person does not jump about.
    fh, fw = clip[0].shape[:2]
    scale = min(384 / fw, 512 / fh)
    ox, oy = (384 - fw * scale) / 2, (512 - fh * scale) / 2
    found = []
    for f in clip:
        photo = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        parse, _ = parsing(resize_and_center(photo, 768, 1024).resize((384, 512)))
        ys, xs = np.where(np.array(parse.resize((384, 512), Image.NEAREST)) != 0)
        if len(ys) >= 500:
            found.append(((xs.min() - ox) / scale, (ys.min() - oy) / scale, (xs.max() - ox) / scale, (ys.max() - oy) / scale))
    if len(found) < FRAMES // 2:
        raise VideoProblem("I could not find a person in most of the video. Stand where the camera sees all of you.")
    x0, y0 = min(b[0] for b in found), min(b[1] for b in found)
    x1, y1 = max(b[2] for b in found), max(b[3] for b in found)
    height = (y1 - y0) * 1.12
    width = max(height * 0.75, (x1 - x0) * 1.1)
    height = width / 0.75
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (int(cx - width / 2), int(cy - height / 2), int(cx + width / 2), int(cy + height / 2))
    ys_i = np.clip(np.arange(box[1], box[3]), 0, fh - 1)   # past the edge, the edge row/column repeats
    xs_i = np.clip(np.arange(box[0], box[2]), 0, fw - 1)
    cropped = [cv2.resize(f[ys_i][:, xs_i], (W, H), interpolation=cv2.INTER_AREA) for f in clip]

    folder = Path(f"/r/{run}")
    folder.mkdir(parents=True, exist_ok=True)
    last_good = {}
    for i, f in enumerate(cropped):
        small = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize((384, 512))
        try:
            parse, _ = parsing(small)
            keypoints = openpose(small)
            top = np.array(get_agnostic_mask_hd(parse, keypoints, "upper_body").convert("L").resize((W, H), Image.NEAREST)) > 127
            bottom = np.array(get_agnostic_mask_dc(parse, keypoints, "lower_body").convert("L").resize((W, H), Image.NEAREST)) > 127
            # The parts on black (Leffa's way). MagicTryOn's own way, over a grey copy of the frame, tested worse.
            body_map = densepose.visualizer.visualize(np.tile(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[:, :, None], [1, 1, 3]),
                                                      densepose.predict(f))
            labels = np.array(parse.resize((W, H), Image.NEAREST))
            body = (labels != 0) & (labels != BAG)
            # The face, beard and hair, a pixel smaller rather than bigger (a wider band kept the old collar).
            head = cv2.erode(np.isin(labels, HEAD).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            last_good = {"top": top, "bottom": bottom, "map": body_map, "body": body, "head": head}
        except Exception as e:                 # a frame the helpers cannot read borrows the one before it
            print(f"frame {i}: {e!r}")
            if not last_good:
                raise VideoProblem("I could not read your body in the first frames. Stand where the camera sees all of you.")
            top, bottom, body_map, body, head = (last_good[k] for k in ("top", "bottom", "map", "body", "head"))
        cv2.imwrite(str(folder / f"person_{i:03d}.png"), f)
        cv2.imwrite(str(folder / f"top_{i:03d}.png"), (top * 255).astype(np.uint8))
        cv2.imwrite(str(folder / f"bottom_{i:03d}.png"), (bottom * 255).astype(np.uint8))
        cv2.imwrite(str(folder / f"map_{i:03d}.png"), body_map)
        cv2.imwrite(str(folder / f"body_{i:03d}.png"), (body * 255).astype(np.uint8))
        cv2.imwrite(str(folder / f"head_{i:03d}.png"), (head * 255).astype(np.uint8))
    runs.commit()
    return {"fps": FRAMES / seconds, "seconds": round(seconds, 1)}


@app.cls(image=mt_image, gpu=["H100", "A100-80GB", "A100-40GB"], memory=32768, timeout=1800,
         volumes={"/w": mt_weights, "/r": runs}, scaledown_window=300, max_containers=2)
class VideoDresser:
    """MagicTryOn-1.3B, loaded once per container: a second video within 5 minutes skips the loading."""

    @modal.enter()
    def load(self):
        import sys
        os.chdir("/mt")
        sys.path.insert(0, "/mt")
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        from omegaconf import OmegaConf
        from magic_tryon.models import AutoencoderKLWan, AutoTokenizer, CLIPModel, WanT5EncoderModel
        from magic_tryon.models.cache_utils import get_teacache_coefficients
        from magic_tryon.models.wan_transformer3d_tryon import WanTransformer3DModel
        from magic_tryon.pipeline.pipeline_wan_tryon import WanFunInpaintPipeline
        from magic_tryon.utils.utils import filter_kwargs
        dtype = torch.bfloat16
        config = OmegaConf.load("config/wan2.1/wan_civitai.yaml")
        base = "/w/base"
        transformer = WanTransformer3DModel.from_pretrained(
            "/w/mt_1.3b", transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
            low_cpu_mem_usage=True, torch_dtype=dtype)
        vae = AutoencoderKLWan.from_pretrained(f"{base}/{config['vae_kwargs']['vae_subpath']}",
                                               additional_kwargs=OmegaConf.to_container(config["vae_kwargs"])).to(dtype)
        tokenizer = AutoTokenizer.from_pretrained(f"{base}/{config['text_encoder_kwargs']['tokenizer_subpath']}")
        text_encoder = WanT5EncoderModel.from_pretrained(
            f"{base}/{config['text_encoder_kwargs']['text_encoder_subpath']}",
            additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"])).to(dtype).eval()
        clip_encoder = CLIPModel.from_pretrained(f"{base}/{config['image_encoder_kwargs']['image_encoder_subpath']}").to(dtype).eval()
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"])))
        self.pipeline = WanFunInpaintPipeline(transformer=transformer, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
                                              scheduler=scheduler, clip_image_encoder=clip_encoder)
        if torch.cuda.get_device_properties(0).total_memory > 38e9:
            self.pipeline.to("cuda")
        else:
            self.pipeline.enable_model_cpu_offload(device="cuda")
        self.pipeline.transformer.enable_teacache(get_teacache_coefficients("wan2.1-fun-1.3b-inp"), STEPS, 0.10,
                                                  num_skip_start_steps=5, offload=False)
        # The garment line drawing, with AniLines' "detail" model, as MagicTryOn's tools make it.
        sys.path.insert(0, "/mt/inference/customize/AniLines")
        from network.line_extractor import LineExtractor
        self.lines = LineExtractor(2, 1, True).to("cuda")
        self.lines.load_state_dict(torch.load("/w/anilines/weights/detail.pth", map_location="cuda", weights_only=True))
        self.lines.eval()
        self.gpu = torch.cuda.get_device_name(0)

    def garment_files(self, data: bytes, name: str):
        """The garment on a white 768 x 1024 picture, and its line drawing."""
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from torch.amp import autocast
        picture = Image.open(io.BytesIO(data)).convert("RGBA")
        canvas = Image.new("RGB", (768, 1024), "white")
        picture.thumbnail((700, 950))
        canvas.paste(picture, ((768 - picture.width) // 2, (1024 - picture.height) // 2), picture)
        cloth = f"/tmp/{name}.png"
        canvas.save(cloth)
        g = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2GRAY)
        sobel = cv2.magnitude(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3))
        sobel = 255 - cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        x = torch.cat([torch.from_numpy(g)[None, None].float(), torch.from_numpy(sobel)[None, None].float()], 1).cuda() / 255.
        h, w = x.shape[-2:]
        x = torch.nn.functional.pad(x, (0, 8 - w % 8, 0, 8 - h % 8), mode="reflect")
        with torch.no_grad(), autocast(enabled=True, device_type="cuda"):
            pred = self.lines(x)[:, :, :h, :w]
        line = f"/tmp/{name}_line.png"
        Image.fromarray(np.clip(pred[0, 0].float().cpu().numpy() * 255 + 0.5, 0, 255).astype(np.uint8)).save(line)
        return cloth, line

    @modal.method()
    def dress(self, run: str, garments: list, fps: float, job: str) -> dict:
        """garments: [(area, picture bytes, caption)] with area "top" or "bottom", top first."""
        import cv2
        import imageio
        import numpy as np
        import torch
        from magic_tryon.utils.utils import get_video_to_video_latent_tryon_full
        runs.reload()
        folder = Path(f"/r/{run}")
        rgb = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        grey = lambda name: np.stack([cv2.imread(str(folder / f"{name}_{i:03d}.png"), 0) > 127 for i in range(FRAMES)])
        person = np.stack([rgb(folder / f"person_{i:03d}.png") for i in range(FRAMES)])
        body_map = [rgb(folder / f"map_{i:03d}.png") for i in range(FRAMES)]
        heads, old_body = grey("head"), grey("body")
        soften = lambda m: np.stack([cv2.GaussianBlur(x.astype(np.float32), (0, 0), 1.0) for x in m])[..., None]
        near = np.ones((41, 41), np.uint8)
        tight = np.ones((2 * TIGHT + 1, 2 * TIGHT + 1), np.uint8)

        def one_pass(frames, area: str, garment: bytes, caption: str):
            masks = grey(area) & ~heads                               # the face is never redrawn
            masks = masks & np.stack([cv2.dilate(b.astype(np.uint8), tight) > 0 for b in old_body])
            for i in range(FRAMES):
                rows = np.where(heads[i].any(axis=1))[0]
                if len(rows):
                    chin = rows.max()                                 # beside the head: the body only, never the room
                    masks[i, :chin + 1] &= old_body[i, :chin + 1]
                    masks[i] &= ~((cv2.dilate(heads[i].astype(np.uint8), near) > 0) & ~old_body[i])
            agnostic = frames.copy()
            agnostic[masks] = 127                                     # MagicTryOn's mid grey: "draw here"
            mask_video = np.repeat((masks * 255).astype(np.uint8)[..., None], 3, axis=-1)
            cloth, line = self.garment_files(garment, area)
            with torch.no_grad():
                video, masked, mask, pose, _, clip_image, cloth_image, line_image = get_video_to_video_latent_tryon_full(
                    list(frames), list(agnostic), list(mask_video), body_map, video_length=FRAMES, sample_size=[H, W],
                    fps=16, ref_image=cloth, line_image_path=line)
                out = self.pipeline("Model is wearing " + caption, num_frames=FRAMES, negative_prompt=NEGATIVE,
                                    height=H, width=W, generator=torch.Generator(device="cuda").manual_seed(SEED),
                                    guidance_scale=GUIDANCE, num_inference_steps=STEPS, video=video, masked_video=masked,
                                    mask_video=mask, pose_video=pose, cloth_image=cloth_image,
                                    cloth_line_image=line_image, clip_image=clip_image).videos
            drawn = (out[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255).round().clip(0, 255)
            a = soften(masks)
            return (frames * (1 - a) + drawn * a).round().clip(0, 255).astype(np.uint8)

        t0 = time.time()
        dressed = person
        names = {"top": "dressing the top", "bottom": "dressing the bottom"}
        for k, (area, garment, caption) in enumerate(garments):
            progress[job] = f"{names[area]} ({k + 1} of {len(garments)})"
            dressed = one_pass(dressed, area, garment, caption)
        progress[job] = "making the video"

        def mp4(frames) -> bytes:
            imageio.mimsave("/tmp/out.mp4", list(frames), fps=fps, codec="libx264", quality=8, macro_block_size=8)
            return Path("/tmp/out.mp4").read_bytes()
        return {"dressed": mp4(dressed), "original": mp4(person), "gpu": self.gpu, "seconds": round(time.time() - t0, 1)}


def on_white(data: bytes) -> bytes:
    """A garment picture with a see-through background, put on white (kept with its transparency removed)."""
    from PIL import Image
    picture = Image.open(io.BytesIO(data))
    out = io.BytesIO()
    if picture.mode in ("RGBA", "LA", "P"):
        picture = picture.convert("RGBA")
        canvas = Image.new("RGBA", picture.size, (255, 255, 255, 255))
        canvas.alpha_composite(picture)
        picture = canvas
    picture.convert("RGBA").save(out, "PNG")
    return out.getvalue()


@app.function(image=web_image, secrets=[modal.Secret.from_name("photo-tryon-key")],
              max_containers=1, scaledown_window=1200, timeout=3600)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response

    key = os.environ["TRYON_KEY"]
    api = FastAPI()
    videos, garments, jobs = {}, {}, {}
    lock = threading.Lock()

    def error(code, message):
        return JSONResponse({"error": message}, status_code=code)

    def caption(gid, name):
        return CAPTIONS.get(gid) or ("a " + (name or gid.replace("_", " ")).lower())

    def run_job(job):
        try:
            job["stage"] = "preparing your video"
            info = prepare.remote(videos[job["session"]], job["id"])
            job["seconds"]["prepare"] = round(time.time() - job["t0"], 1)
            job["stage"] = "dressing"
            chosen = [(area, garments[gid], caption(gid, job[f"{area}_name"]))
                      for area, gid in (("top", job["top"]), ("bottom", job["bottom"])) if gid]
            out = VideoDresser().dress.remote(job["id"], chosen, info["fps"], job["id"])
            job["files"] = {"dressed.mp4": out["dressed"], "original.mp4": out["original"]}
            job["gpu"] = out["gpu"]
            job["state"] = "done"
        except Exception as e:
            known = type(e).__name__ == "VideoProblem"
            job["error"] = str(e) if known else traceback.format_exc()[-900:]
            job["state"] = "error"
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
        return {"ok": True, "model": "MagicTryOn-1.3B", "max_seconds": 10,
                "busy": any(j["state"] == "running" for j in jobs.values())}

    @api.put("/k/{k}/video/{session}")
    async def put_video(session: str, request: Request):
        data = await request.body()
        if len(data) > 80_000_000:
            return error(413, "The video file is too big.")
        with lock:
            videos[session] = data
            for j in jobs.values():           # a new video: earlier jobs of this session are not reused for it
                if j["session"] == session:
                    j["name"] += "|replaced"
        return {"ok": True}

    @api.put("/k/{k}/garment/{garment}/{face}")
    async def put_garment(garment: str, face: str, request: Request):
        if face not in ("front", "back"):
            return error(404, "not found")
        garments[garment] = on_white(await request.body()) if face == "front" else garments.get(garment)
        return {"ok": True}

    @api.post("/k/{k}/job")
    async def post_job(request: Request):
        q = json.loads(await request.body() or b"{}")
        session, top, bottom = q.get("session"), q.get("top"), q.get("bottom")
        if session not in videos:
            return error(400, "video not uploaded")
        if not top and not bottom:
            return error(400, "no garment chosen")
        for g in (top, bottom):
            if g and g not in garments:
                return error(400, f"garment {g} not uploaded")
        name = f"{session}|{top}|{bottom}"
        with lock:
            for j in jobs.values():
                if j["name"] == name and j["state"] != "error":
                    return {"job": j["id"]}
            jid = secrets.token_hex(6)
            jobs[jid] = {"id": jid, "name": name, "session": session, "top": top, "bottom": bottom,
                         "top_name": q.get("top_name"), "bottom_name": q.get("bottom_name"), "state": "running",
                         "stage": "starting", "files": {}, "seconds": {}, "error": None, "t0": time.time()}
        threading.Thread(target=run_job, args=(jobs[jid],), daemon=True).start()
        return {"job": jid}

    @api.get("/k/{k}/job/{jid}")
    def get_job(jid: str):
        j = jobs.get(jid)
        if j is None:
            return error(404, "not found")
        stage = j["stage"]
        if j["state"] == "running" and stage == "dressing":
            stage = progress.get(jid, "loading the model")
        return {"id": j["id"], "state": j["state"], "stage": stage, "seconds": j["seconds"], "error": j["error"],
                "elapsed": round(time.time() - j["t0"], 1)}

    @api.get("/k/{k}/job/{jid}/file/{name}")
    def get_file(jid: str, name: str):
        j = jobs.get(jid)
        if j is None or name not in j["files"]:
            return error(404, "not found")
        return Response(j["files"][name], media_type="video/mp4")

    return api
