"""Dry run: a recorded turn, dressed as a video by MagicTryOn-1.3B (record, then dress; not live).

A separate Modal app: the deployed photo engine is not touched.

Once, to fetch MagicTryOn's files into their own volume (about 20 GB):
    python -m modal run video_dry_run.py::fetch_weights
Look at the clip first (one frame a second, to see where the turn starts and ends):
    python -m modal run video_dry_run.py --video clip.mp4 --out out --look
Then dress it (start and end in seconds; leave them out for the whole clip):
    python -m modal run video_dry_run.py --video clip.mp4 --out out [--start 1 --end 13] [--top navy_polo] [--bottom khaki_chinos]

How it works:
  1. 81 frames are taken evenly from the turn: MagicTryOn's clip length. Played at 16 a second they make a
     5-second spin.
  2. Every frame is cropped with one box, around the person in all of them, to 624 x 832 (3:4, the size the
     model was trained at). One box for all, so the person does not jump about.
  3. For every frame, Leffa's helpers (already in the Leffa volume) find where the clothes go (the mask) and
     draw the body-surface map (DensePose), the way MagicTryOn's own tools do.
  4. MagicTryOn dresses the shirt, then the trousers on the dressed video: the same order as the photos.

MagicTryOn (github.com/vivoCameraResearch/Magic-TryOn) is CC BY-NC-SA 4.0: non-commercial use only.
"""
import io
import os
import time
from pathlib import Path

import modal

app = modal.App("photo-tryon-video-dry-run")
leffa_weights = modal.Volume.from_name("photo-tryon-leffa-weights")
mt_weights = modal.Volume.from_name("photo-tryon-magictryon-weights", create_if_missing=True)

MT_COMMIT = "0ef4b17869d9cccc86c61f5b7ec656d365e0ae82"   # the MagicTryOn code this was written against
FRAMES = 81                  # the model's clip: the video encoder squeezes 81 frames into 21
FPS = 16
W, H = 624, 832              # 3:4, the size it was trained at
STEPS, GUIDANCE, SEED = 20, 3.0, 43                      # MagicTryOn's own settings
HEAD = [1, 2, 3, 11]         # body-part labels of hat, hair, sunglasses, face: never redrawn
BAG = 16                     # a bag is not part of the body
GREY_BODY_MAP = False        # True: body maps drawn as MagicTryOn's tool draws them (tested worse, see prepare)
CAPTIONS = {                 # written by hand; MagicTryOn uses a 7B captioning model for these
    "navy_polo": "a navy blue short-sleeve polo shirt with a classic collar, a two-button placket with white buttons and ribbed sleeve cuffs",
    "polo_pink": "a pink short-sleeve polo shirt with a ribbed collar, a two-button placket with white buttons and ribbed sleeve cuffs",
    "maroon_polo": "a maroon red short-sleeve polo shirt with a classic collar, a two-button placket with white buttons and ribbed sleeve cuffs",
    "khaki_chinos": "khaki beige straight-leg tailored trousers with belt loops, a pressed front crease and a small green and red tab at the waistband",
    "black_slim_trousers": "black straight-leg tailored trousers with belt loops, a waistband tab and pressed front creases",
}
NEGATIVE = ("衣服质量差的，纽扣不整齐的，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
            "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
            "杂乱的背景，三条腿，背景人很多，倒着走")            # MagicTryOn's own negative prompt

# The same image as the photo engine's Leffa image, so Modal reuses the one already built.
leffa_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "accelerate", "diffusers", "transformers", "einops", "fvcore",
                 "omegaconf", "onnxruntime", "opencv-python-headless", "pycocotools", "av", "config", "cloudpickle",
                 "peft", "safetensors", "scikit-image", "scipy", "timm", "pyyaml", "psutil", "huggingface_hub", "pillow",
                 "matplotlib", "pandas", "imageio", "tqdm", "regex", "torchmetrics")
    .run_commands("git clone --depth 1 https://github.com/franciszzj/Leffa /leffa")
)
# The photo engine's FASHN image, copied exactly so Modal reuses it: for the frame-by-frame comparison.
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
mt_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "diffusers==0.32.2", "transformers==4.50.1", "accelerate==1.5.2",
                 "omegaconf==2.3.0", "einops", "opencv-python-headless==4.11.0.86", "imageio==2.37.0",
                 "imageio-ffmpeg==0.6.0", "safetensors", "sentencepiece", "ftfy", "regex", "timm", "tqdm", "numpy<2",
                 "pillow", "huggingface_hub==0.29.3", "func_timeout")
    .run_commands(f"git clone https://github.com/vivoCameraResearch/Magic-TryOn /mt && cd /mt && git checkout {MT_COMMIT}")
)


@app.function(image=mt_image, volumes={"/w": mt_weights}, cpu=4, timeout=3600)
def fetch_weights():
    """MagicTryOn-1.3B, the Wan2.1 parts it needs (not the 32.9 GB big model), and the line-map model."""
    from huggingface_hub import hf_hub_download, snapshot_download
    snapshot_download("LuckyLiGY/MagicTryOn", local_dir="/w/base", allow_patterns=[
        "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "google/umt5-xxl/*", "xlm-roberta-large/*"])
    snapshot_download("LuckyLiGY/MagicTryOn-1.3B", local_dir="/w/mt_1.3b",
                      allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"])
    hf_hub_download("aidenpan/AniLines-Anime-Lineart-Extractor", "weights/detail.pth", repo_type="space",
                    local_dir="/w/anilines")
    mt_weights.commit()
    for root, _, files in os.walk("/w"):
        for f in files:
            p = os.path.join(root, f)
            print(f"{os.path.getsize(p) / 1e9:7.2f} GB  {p}")


@app.function(image=mt_image, volumes={"/w": mt_weights}, cpu=4, memory=16384, timeout=3600)
def fetch_turbo():
    """MagicTryOn-Turbo (6.31 GB): the small model trained to copy the big one in 4 steps. Its published code
    imports a folder missing from the repository, so it is run here through the main MagicTryOn code instead:
    the same model shape. This checks that its weights fit that model."""
    import sys
    import torch
    from huggingface_hub import hf_hub_download
    hf_hub_download("LuckyLiGY/MagicTryOn-Turbo", "model.pt", local_dir="/w/turbo")
    mt_weights.commit()
    state = torch.load("/w/turbo/model.pt", map_location="cpu")
    print("top-level keys:", list(state.keys()))
    gen = state["generator"]
    names = list(gen.keys())
    print(len(names), "weights; first:", names[:3], "dtype", gen[names[0]].dtype)
    os.chdir("/mt")
    sys.path.insert(0, "/mt")
    from omegaconf import OmegaConf
    from magic_tryon.models.wan_transformer3d_tryon import WanTransformer3DModel
    config = OmegaConf.load("config/wan2.1/wan_civitai.yaml")
    model = WanTransformer3DModel.from_pretrained(
        "/w/mt_1.3b", transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    stripped = {k[len("model."):] if k.startswith("model.") else k: v for k, v in gen.items()}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    print(f"fits the main model: {len(stripped) - len(unexpected)} used, {len(missing)} missing, {len(unexpected)} unexpected")
    print("missing:", missing[:5], "unexpected:", unexpected[:5])


def read_video(data: bytes):
    """Every frame (BGR) and the frame rate."""
    import tempfile
    import cv2
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(data)
    cap = cv2.VideoCapture(f.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    os.unlink(f.name)
    return frames, fps


def sheet(rows, height=260):
    """Rows of BGR pictures side by side, each scaled to one height -> one JPEG."""
    import cv2
    import numpy as np
    lines = []
    for row in rows:
        scaled = [cv2.resize(p, (int(p.shape[1] * height / p.shape[0]), height)) for p in row]
        lines.append(np.concatenate(scaled, axis=1))
    width = max(line.shape[1] for line in lines)
    lines = [np.pad(line, ((0, 0), (0, width - line.shape[1]), (0, 0))) for line in lines]
    return cv2.imencode(".jpg", np.concatenate(lines, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


@app.function(image=leffa_image, cpu=2.0, memory=4096, timeout=300)
def look(video: bytes, start: float = 0.0, end: float = 0.0, step: float = 1.0, height: int = 260) -> dict:
    """A frame every `step` seconds, labelled with its time, to see where the turn is."""
    import cv2
    import numpy as np
    frames, fps = read_video(video)
    end = end if end > 0 else len(frames) / fps
    picks = []
    for second in np.arange(start, end + 1e-6, step):
        i = min(len(frames) - 1, int(second * fps))
        f = frames[i].copy()
        cv2.putText(f, f"{second:g}s", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 255), 5)
        picks.append(f)
    rows = [picks[k:k + 8] for k in range(0, len(picks), 8)]
    h, w = frames[0].shape[:2]
    return {"sheet": sheet(rows, height), "info": f"{len(frames)} frames, {fps:.2f} fps, {w}x{h}, {len(frames) / fps:.1f} s"}


@app.function(image=leffa_image, gpu=["L4", "A10G"], memory=8192, timeout=1800,
              volumes={"/ckpts": leffa_weights, "/w": mt_weights})
def prepare(video: bytes, start: float, end: float, run: str, white: bool = False, room_picture: bool = False) -> dict:
    """The clip's 81 frames, cropped, with their masks and body maps, saved in the volume under runs/<run>.
    white: the person is cut out (by the body-part outline) and put on plain white before dressing."""
    import sys
    os.chdir("/leffa")
    sys.path.insert(0, "/leffa")
    import cv2
    import numpy as np
    from PIL import Image
    from leffa_utils.densepose_predictor import DensePosePredictor
    from leffa_utils.utils import get_agnostic_mask_dc, get_agnostic_mask_hd, resize_and_center
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose

    t0 = time.time()
    ck = "/ckpts"
    parsing = Parsing(atr_path=f"{ck}/humanparsing/parsing_atr.onnx", lip_path=f"{ck}/humanparsing/parsing_lip.onnx")
    openpose = OpenPose(body_model_path=f"{ck}/openpose/body_pose_model.pth")
    densepose = DensePosePredictor(config_path=f"{ck}/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
                                   weights_path=f"{ck}/densepose/model_final_162be9.pkl")

    def magic_body_map(f):
        """The body-part map drawn as MagicTryOn's own tool (DensePose apply_net "dp_segm") draws it: the parts
        in colour at 70 %, over a grey copy of the whole frame, room included. (Leffa's helper draws them on black.)"""
        grey = np.tile(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[:, :, None], [1, 1, 3]).astype(np.float32)
        results, boxes = densepose.predict(f)
        boxes = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes)
        fh_, fw_ = grey.shape[:2]
        for r, b in zip(results, boxes):
            x, y, w, h = [int(v) for v in b]
            if w <= 0 or h <= 0:
                continue
            labels = r.labels.cpu().numpy().astype(np.uint8)
            if labels.shape != (h, w):
                labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
            colour = cv2.applyColorMap((labels.astype(np.float32) * 255.0 / 24).clip(0, 255).astype(np.uint8),
                                       cv2.COLORMAP_PARULA).astype(np.float32)
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, fw_), min(y + h, fh_)
            labels, colour = labels[y0 - y:y1 - y, x0 - x:x1 - x], colour[y0 - y:y1 - y, x0 - x:x1 - x]
            region = grey[y0:y1, x0:x1]
            on = labels > 0
            region[on] = region[on] * 0.3 + colour[on] * 0.7
        return grey.round().clip(0, 255).astype(np.uint8)

    frames, fps = read_video(video)
    first = max(0, int(start * fps))
    last = len(frames) - 1 if end <= 0 else min(len(frames) - 1, int(end * fps))
    clip = [frames[i] for i in np.linspace(first, last, FRAMES).round().astype(int)]

    # One box around the person in every frame (as the photo crop does, but for all frames at once).
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
    if not found:
        raise RuntimeError("No person found in the clip")
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

    folder = Path(f"/w/runs/{run}")
    folder.mkdir(parents=True, exist_ok=True)
    last_good = {}
    misses = 0
    bodies = []
    for i, f in enumerate(cropped):
        small = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize((384, 512))
        try:
            parse, _ = parsing(small)
            keypoints = openpose(small)
            top = np.array(get_agnostic_mask_hd(parse, keypoints, "upper_body").convert("L").resize((W, H), Image.NEAREST)) > 127
            bottom = np.array(get_agnostic_mask_dc(parse, keypoints, "lower_body").convert("L").resize((W, H), Image.NEAREST)) > 127
            # Leffa's helper draws the parts on black. MagicTryOn's own way (over a grey copy of the frame, see
            # magic_body_map) was tried on 2026-11-17 and was worse: the model copied the old clothes' stripes and
            # painted shirt-coloured "wings" beside the shoulders.
            body_map = (magic_body_map(f) if GREY_BODY_MAP else
                        densepose.visualizer.visualize(np.tile(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[:, :, None], [1, 1, 3]),
                                                       densepose.predict(f)))
            labels = np.array(parse.resize((W, H), Image.NEAREST))
            body = (labels != 0) & (labels != BAG)          # the person's outline, to put the room back around it
            # The face, beard and hair, a pixel smaller rather than bigger: a wider band kept the old T-shirt's
            # collar showing between the beard and the new collar.
            head = cv2.erode(np.isin(labels, HEAD).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            last_good = {"top": top, "bottom": bottom, "map": body_map, "body": body, "head": head}
        except Exception as e:                # a frame the helpers cannot read borrows the one before it
            misses += 1
            print(f"frame {i}: {e!r}")
            if not last_good:
                raise
            top, bottom, body_map = last_good["top"], last_good["bottom"], last_good["map"]
            body, head = last_good["body"], last_good["head"]
        cv2.imwrite(str(folder / f"orig_{i:03d}.png"), f)
        if white:                             # the person on white, the outline's edge softened by a pixel
            soft = cv2.GaussianBlur(body.astype(np.float32), (0, 0), 1.2)[..., None]
            f = (f * soft + 255 * (1 - soft)).round().astype(np.uint8)
            cropped[i] = f
        cv2.imwrite(str(folder / f"person_{i:03d}.png"), f)
        cv2.imwrite(str(folder / f"top_{i:03d}.png"), (top * 255).astype(np.uint8))
        cv2.imwrite(str(folder / f"bottom_{i:03d}.png"), (bottom * 255).astype(np.uint8))
        cv2.imwrite(str(folder / f"map_{i:03d}.png"), body_map)
        cv2.imwrite(str(folder / f"body_{i:03d}.png"), (body * 255).astype(np.uint8))
        cv2.imwrite(str(folder / f"head_{i:03d}.png"), (head * 255).astype(np.uint8))
        bodies.append(body)

    # The empty room. The phone stands still while the person moves, so each spot of the room shows in some
    # frames: its colour is the middle value over those frames. A spot seen in fewer than 3 is "unknown".
    cover = np.stack([cv2.dilate(b.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0 for b in bodies])
    room = np.full((H, W, 3), 255, np.uint8)
    known = np.ones((H, W), bool)
    if room_picture and not white:            # on white, the room is known everywhere: it is white
        for c in range(3):
            stack = np.stack([f[:, :, c] for f in cropped]).astype(np.float32)
            stack[cover] = np.nan
            with np.errstate(all="ignore"):
                room[:, :, c] = np.nan_to_num(np.nanmedian(stack, axis=0)).clip(0, 255).astype(np.uint8)
        known = (~cover).sum(axis=0) >= 3
    cv2.imwrite(str(folder / "room.png"), room)
    cv2.imwrite(str(folder / "room_known.png"), (known * 255).astype(np.uint8))
    mt_weights.commit()

    def shade(f, m, colour):
        out = f.copy()
        out[m] = (0.5 * out[m] + 0.5 * np.array(colour)).astype(np.uint8)
        return out
    picks = np.linspace(0, FRAMES - 1, 8).round().astype(int)
    rows = [[cropped[i] for i in picks],
            [shade(cropped[i], cv2.imread(str(folder / f"top_{i:03d}.png"), 0) > 127, (0, 0, 255)) for i in picks],
            [shade(cropped[i], cv2.imread(str(folder / f"bottom_{i:03d}.png"), 0) > 127, (255, 0, 0)) for i in picks],
            [cv2.imread(str(folder / f"map_{i:03d}.png")) for i in picks],
            [room, cv2.cvtColor((known * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]]
    return {"sheet": sheet(rows, 300), "seconds": time.time() - t0, "misses": misses, "box": box,
            "frames": f"{first}..{last} of {len(frames)} at {fps:.2f} fps"}


@app.cls(image=leffa_image, volumes={"/ckpts": leffa_weights}, cpu=2.0, memory=4096, timeout=600,
         scaledown_window=120, max_containers=12)
class Parser:
    """The person's outline in a dressed frame, with the same body-part model as the preparing step."""

    @modal.enter()
    def load(self):
        import sys
        os.chdir("/leffa")
        sys.path.insert(0, "/leffa")
        from preprocess.humanparsing.run_parsing import Parsing
        self.parsing = Parsing(atr_path="/ckpts/humanparsing/parsing_atr.onnx", lip_path="/ckpts/humanparsing/parsing_lip.onnx")

    @modal.method()
    def body(self, jpeg: bytes) -> bytes:
        import cv2
        import numpy as np
        from PIL import Image
        parse, _ = self.parsing(Image.open(io.BytesIO(jpeg)).convert("RGB").resize((384, 512)))
        labels = np.array(parse.resize((W, H), Image.NEAREST))
        return cv2.imencode(".png", (((labels != 0) & (labels != BAG)) * 255).astype(np.uint8))[1].tobytes()


# Only the fast cards: on an A10G the same clip took three times as long.
@app.function(image=mt_image, gpu=["H100", "A100-80GB", "A100-40GB"], memory=32768, timeout=3600,
              volumes={"/w": mt_weights})
def dress(run: str, top: bytes, top_caption: str, bottom: bytes | None, bottom_caption: str,
          white: bool = False, blend: str = "mask", teacache: bool = True, turbo: bool = False,
          fill: str = "grey", tight: int = 0) -> dict:
    """MagicTryOn-1.3B: the shirt on the prepared clip, then the trousers on the dressed one.
    blend: "mask" takes the model's picture inside the whole mask (MagicTryOn's own repaint); "outline" takes it
    only on the person and fills the rest with the room picture (or white)."""
    import sys
    os.chdir("/mt")
    sys.path.insert(0, "/mt")
    import cv2
    import imageio
    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from PIL import Image
    from torch.amp import autocast
    from magic_tryon.models import AutoencoderKLWan, AutoTokenizer, CLIPModel, WanT5EncoderModel
    from magic_tryon.models.cache_utils import get_teacache_coefficients
    from magic_tryon.models.wan_transformer3d_tryon import WanTransformer3DModel
    from magic_tryon.pipeline.pipeline_wan_tryon import WanFunInpaintPipeline
    from magic_tryon.utils.utils import filter_kwargs, get_video_to_video_latent_tryon_full

    timings = {}
    t0 = time.time()
    gpu = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory
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
    pipeline = WanFunInpaintPipeline(transformer=transformer, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
                                     scheduler=scheduler, clip_image_encoder=clip_encoder)
    if memory > 38e9:
        pipeline.to("cuda")
    else:                                     # 24 GB cards: each part goes onto the card only while it works
        pipeline.enable_model_cpu_offload(device="cuda")
    if turbo:
        # MagicTryOn-Turbo: the same model shape, other weights (stored as "model.<name>").
        state = torch.load("/w/turbo/model.pt", map_location="cpu")["generator"]
        state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
        missing, unexpected = transformer.load_state_dict(state, strict=False)
        del state
        print(f"Turbo weights: {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            raise RuntimeError(f"Turbo weights do not fit: {missing[:5]}")
    elif teacache:                            # a speed-up that skips some steps; off = every step really drawn
        pipeline.transformer.enable_teacache(get_teacache_coefficients("wan2.1-fun-1.3b-inp"), STEPS, 0.10,
                                             num_skip_start_steps=5, offload=False)
    timings["load"] = time.time() - t0

    def turbo_draw(prompt, masked, mask, pose, cloth_image, line_image, clip_image):
        """MagicTryOn-Turbo's own 4-step drawing (turbo_inference + bidirectional_inference.py in its repository),
        rebuilt on the main code: inputs scaled to -1..1, latents clamped, no negative prompt."""
        import torch.nn.functional as F
        import torchvision.transforms.functional as TF
        sys.path.insert(0, "/mt/magictryon_turbo/models/wan")
        from flow_match import FlowMatchScheduler       # one plain file; its package imports a missing folder
        sched = FlowMatchScheduler(shift=8.0, sigma_min=0.0, extra_one_step=True)
        sched.set_timesteps(1000, training=True)
        scale = [s.to(device="cuda", dtype=dtype) for s in vae.scale]
        # The main code's encoder returns the mean and the spread (32 channels); Turbo's own returns the mean (16).
        enc = lambda v: torch.stack([vae.model.encode(u.unsqueeze(0), scale)[:, :16].float().clamp_(-1, 1).squeeze(0)
                                     for u in v.to(device="cuda", dtype=dtype)]).to(dtype)
        masked_lat = enc(masked * 2 - 1)
        pose_lat = enc(pose * 2 - 1)
        cloth_lat = enc(cloth_image * 2 - 1)
        line_lat = enc(line_image * 2 - 1)
        bs, _, frames_n, height, width = masked.shape
        m = mask[:, :1].to(device="cuda", dtype=dtype)
        m = torch.concat([torch.repeat_interleave(m[:, :, 0:1], repeats=4, dim=2), m[:, :, 1:]], dim=2)
        m = m.view(bs, m.shape[2] // 4, 4, height, width).transpose(1, 2)
        mask_lat = F.interpolate(m, size=list(masked_lat.shape[2:])).to(dtype)
        y = [masked_lat, pose_lat, mask_lat, cloth_lat, line_lat]
        clip_t = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device="cuda", dtype=dtype)
        image_clip = clip_encoder([clip_t[:, None, :, :]])
        tokens = tokenizer([prompt], padding="max_length", max_length=512, truncation=True,
                           add_special_tokens=True, return_tensors="pt")
        ids, att = tokens.input_ids.to("cuda"), tokens.attention_mask.to("cuda")
        lens = att.gt(0).sum(dim=1).long()
        context = text_encoder(ids, attention_mask=att)[0]
        context = [u[:v] for u, v in zip(context, lens)]
        lat_f, lat_h, lat_w = masked_lat.shape[2:]
        seq_len = lat_f * (lat_h // 2) * (lat_w // 2)
        noisy = torch.randn(1, lat_f, 16, lat_h, lat_w, generator=torch.Generator(device="cuda").manual_seed(42),
                            dtype=dtype, device="cuda")
        steps = [1000, 757, 522, 0]           # configs/wan_bidirectional_dmd_from_scratch.yaml, not warped
        sigmas, timesteps = sched.sigmas.double().cuda(), sched.timesteps.double().cuda()
        for k, t in enumerate(steps):
            tt = torch.full((1,), float(t), device="cuda")
            with torch.autocast("cuda", dtype=dtype):   # as the main pipeline runs the model
                flow = transformer(noisy.permute(0, 2, 1, 3, 4), t=tt, context=context, seq_len=seq_len,
                                   y=y, clip_fea=image_clip).permute(0, 2, 1, 3, 4)
            sigma = sigmas[torch.argmin((timesteps - float(t)).abs())]
            pred = (noisy.double() - sigma * flow.double()).to(dtype)        # the model's guess of the clean frames
            if k < len(steps) - 1:
                nxt = torch.full((pred.shape[0] * pred.shape[1],), float(steps[k + 1]), device="cuda")
                noisy = sched.add_noise(pred.flatten(0, 1), torch.randn_like(pred.flatten(0, 1)), nxt).unflatten(0, pred.shape[:2])
        video = torch.stack([vae.model.decode(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                             for u in pred.permute(0, 2, 1, 3, 4)])
        return (video * 0.5 + 0.5).clamp(0, 1)

    # The garment line drawing, with AniLines' "detail" model, as MagicTryOn's tools make it.
    sys.path.insert(0, "/mt/inference/customize/AniLines")
    from network.line_extractor import LineExtractor
    lines_model = LineExtractor(2, 1, True).to("cuda")
    lines_model.load_state_dict(torch.load("/w/anilines/weights/detail.pth", map_location="cuda", weights_only=True))
    lines_model.eval()

    def garment_files(data: bytes, name: str):
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
            pred = lines_model(x)[:, :, :h, :w]
        line = f"/tmp/{name}_line.png"
        Image.fromarray(np.clip(pred[0, 0].float().cpu().numpy() * 255 + 0.5, 0, 255).astype(np.uint8)).save(line)
        return cloth, line

    folder = Path(f"/w/runs/{run}")
    rgb = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    person = np.stack([rgb(folder / f"person_{i:03d}.png") for i in range(FRAMES)])
    body_map = [rgb(folder / f"map_{i:03d}.png") for i in range(FRAMES)]
    grey = lambda name: np.stack([cv2.imread(str(folder / f"{name}_{i:03d}.png"), 0) > 127 for i in range(FRAMES)])
    heads = grey("head")
    old_body = grey("body")                   # where the person was, in their own clothes
    room = rgb(folder / "room.png").astype(np.float32)
    room_known = cv2.imread(str(folder / "room_known.png"), 0) > 127

    def outline(frames):
        """Where the person is in each dressed frame: the body-part model, run on several machines at once."""
        jpegs = [cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                 for f in frames]
        return np.stack([cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE) > 127
                         for b in Parser().body.map(jpegs)])

    soften = lambda masks: np.stack([cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0) for m in masks])[..., None]
    # The model's own output, kept to try other ways of combining later; one folder per setting, so two runs
    # side by side do not write over each other.
    raw_dir = folder / f"raw_{'turbo' if turbo else 'base'}_{fill}_{tight}"
    raw_dir.mkdir(exist_ok=True)
    raw_frames = {}

    def one_pass(frames, area: str, garment: bytes, caption: str):
        masks = grey(area) & ~heads           # the face is never redrawn
        if tight > 0:                         # only the person and a margin may be redrawn: less room to invent
            k = np.ones((2 * tight + 1, 2 * tight + 1), np.uint8)
            masks = masks & np.stack([cv2.dilate(b.astype(np.uint8), k) > 0 for b in old_body])
        # Beside the head (from its top down to the chin) only the body may be redrawn, never the room: the
        # model drew thin dark lines there, like the arms of glasses.
        # Seen from the side, the same happened in front of the throat, just below the chin: so the room
        # within 20 pixels of the head is kept too.
        masks = masks.copy()
        near = np.ones((41, 41), np.uint8)
        for i in range(FRAMES):
            rows = np.where(heads[i].any(axis=1))[0]
            if len(rows):
                chin = rows.max()
                masks[i, :chin + 1] &= old_body[i, :chin + 1]
                masks[i] &= ~((cv2.dilate(heads[i].astype(np.uint8), near) > 0) & ~old_body[i])
        agnostic = frames.copy()
        if fill == "inpaint":
            # The area to redraw filled from the colours around it (the wall), instead of flat grey: what the
            # model copies where it draws no clothes is then wall, not grey. Done at half size, for speed.
            for i in range(FRAMES):
                small = cv2.resize(frames[i], (W // 2, H // 2), interpolation=cv2.INTER_AREA)
                hole = cv2.resize(masks[i].astype(np.uint8) * 255, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
                filled = cv2.resize(cv2.inpaint(small, hole, 3, cv2.INPAINT_TELEA), (W, H), interpolation=cv2.INTER_LINEAR)
                agnostic[i][masks[i]] = filled[masks[i]]
        else:
            agnostic[masks] = 127             # what MagicTryOn's get_masked_person leaves in the mask: mid grey
        mask_video = np.repeat((masks * 255).astype(np.uint8)[..., None], 3, axis=-1)
        cloth, line = garment_files(garment, area)
        with torch.no_grad():
            video, masked, mask, pose, _, clip_image, cloth_image, line_image = get_video_to_video_latent_tryon_full(
                list(frames), list(agnostic), list(mask_video), body_map, video_length=FRAMES, sample_size=[H, W],
                fps=FPS, ref_image=cloth, line_image_path=line)
            t_draw = time.time()
            if turbo:
                out = turbo_draw("Model is wearing " + caption, masked, mask, pose, cloth_image, line_image, clip_image)
            else:
                out = pipeline("Model is wearing " + caption, num_frames=FRAMES, negative_prompt=NEGATIVE, height=H, width=W,
                               generator=torch.Generator(device="cuda").manual_seed(SEED), guidance_scale=GUIDANCE,
                               num_inference_steps=STEPS, video=video, masked_video=masked, mask_video=mask, pose_video=pose,
                               cloth_image=cloth_image, cloth_line_image=line_image, clip_image=clip_image).videos
            timings[f"{area} drawing"] = time.time() - t_draw
        drawn = (out[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255).round().clip(0, 255)
        for i, f in enumerate(drawn.astype(np.uint8)):
            cv2.imwrite(str(raw_dir / f"{area}_{i:03d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        raw_frames[area] = drawn.astype(np.uint8)
        if blend == "mask":                   # the model's picture in the whole mask (the face is not in it)
            a = soften(masks)
            return (frames * (1 - a) + drawn * a).round().clip(0, 255).astype(np.uint8)
        t_outline = time.time()
        new_body = outline(drawn.astype(np.uint8))
        timings[f"{area} outline"] = time.time() - t_outline
        # Inside the mask, each pixel comes from one place:
        #   on the person in the dressed frame                  -> the model's picture (the new clothes)
        #   where the old, looser clothes were but no longer are -> the empty room, where the clip showed it
        #                                                          (on white: anywhere off the person -> white)
        #   anywhere else                                       -> the frame as given (the room, the face)
        # No widening of the outline: the grey the model leaves around the body stays out.
        use_room = masks & ~new_body if white else masks & old_body & ~new_body & room_known
        use_drawn = masks & new_body
        base = frames.astype(np.float32)
        a = soften(use_room)
        base = base * (1 - a) + room[None] * a
        a = soften(use_drawn)
        return (base * (1 - a) + drawn * a).round().clip(0, 255).astype(np.uint8)

    t = time.time()
    dressed = one_pass(person, "top", top, top_caption)
    timings["shirt"] = time.time() - t
    if bottom is not None:
        t = time.time()
        dressed = one_pass(dressed, "bottom", bottom, bottom_caption)
        timings["trousers"] = time.time() - t

    def mp4(frames) -> bytes:
        path = "/tmp/out.mp4"
        imageio.mimsave(path, list(frames), fps=FPS, codec="libx264", quality=8, macro_block_size=8)
        return Path(path).read_bytes()

    picks = np.linspace(0, FRAMES - 1, 8).round().astype(int)
    bgr = lambda f: cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
    original = (np.stack([rgb(folder / f"orig_{i:03d}.png") for i in range(FRAMES)])
                if (folder / "orig_000.png").exists() else person)     # runs prepared before orig_ was saved
    return {"dressed": mp4(dressed), "side_by_side": mp4(np.concatenate([original, dressed], axis=2)),
            "sheet": sheet([[bgr(original[i]) for i in picks], [bgr(person[i]) for i in picks],
                            [bgr(dressed[i]) for i in picks]], 360),
            "frames": [cv2.imencode(".jpg", bgr(dressed[i]), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes() for i in picks],
            "consecutive": sheet([[bgr(f) for f in dressed[24:32]], [bgr(f) for f in dressed[32:40]]], 360),
            "raw": sheet([[bgr(f[i]) for i in picks] for f in raw_frames.values()], 360),
            "timings": timings, "gpu": gpu}


@app.cls(image=fashn_image, gpu=["L4", "A10G"], volumes={"/w": mt_weights}, timeout=1800, scaledown_window=60,
         max_containers=10)
class FashnFrames:
    """For comparison: FASHN dressing each frame on its own, with the photo engine's settings and one seed."""

    @modal.enter()
    def load(self):
        from fashn_vton import TryOnPipeline
        self.pipe = TryOnPipeline(weights_dir="/weights", device="cuda")

    @modal.method()
    def dress(self, run: str, i: int, top: bytes, bottom: bytes | None) -> bytes:
        from PIL import Image
        picture = Image.open(f"/w/runs/{run}/person_{i:03d}.png").convert("RGB")
        for garment, category in ((top, "tops"), (bottom, "bottoms")):
            if garment is None:
                continue
            picture = self.pipe(person_image=picture, garment_image=Image.open(io.BytesIO(garment)).convert("RGB"),
                                category=category, garment_photo_type="flat-lay", num_samples=1,
                                num_timesteps=30, guidance_scale=1.5, seed=42, segmentation_free=True).images[0]
        out = io.BytesIO()
        picture.resize((W, H), Image.LANCZOS).save(out, "JPEG", quality=92)
        return out.getvalue()


@app.function(image=mt_image, cpu=4.0, memory=8192, timeout=600, volumes={"/w": mt_weights})
def compose(run: str, fashn: list, magic: bytes) -> dict:
    """Original | FASHN frame by frame | MagicTryOn, side by side; and 16 FASHN frames in a row, to see flicker."""
    import cv2
    import imageio
    import numpy as np
    person = [cv2.cvtColor(cv2.imread(f"/w/runs/{run}/person_{i:03d}.png"), cv2.COLOR_BGR2RGB) for i in range(FRAMES)]
    fash = [cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for b in fashn]
    Path("/tmp/magic.mp4").write_bytes(magic)
    mag = [np.asarray(f) for f in imageio.get_reader("/tmp/magic.mp4")][:FRAMES]

    def label(f, text):
        f = f.copy()
        cv2.putText(f, text, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        return f

    def mp4(frames) -> bytes:
        imageio.mimsave("/tmp/out.mp4", list(frames), fps=FPS, codec="libx264", quality=8, macro_block_size=8)
        return Path("/tmp/out.mp4").read_bytes()

    three = [np.concatenate([label(p, "Original"), label(f, "FASHN, frame by frame"), label(m, "MagicTryOn")], axis=1)
             for p, f, m in zip(person, fash, mag)]
    bgr = lambda f: cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
    picks = np.linspace(0, FRAMES - 1, 8).round().astype(int)
    return {"fashn": mp4(fash), "three_way": mp4(three),
            "sheet": sheet([[bgr(person[i]) for i in picks], [bgr(fash[i]) for i in picks], [bgr(mag[i]) for i in picks]], 360),
            "consecutive": sheet([[bgr(f) for f in fash[24:32]], [bgr(f) for f in fash[32:40]]], 360)}


def assets() -> Path:
    """The app's catalog pictures (worked out on this computer only)."""
    return Path(__file__).resolve().parents[2] / "app/src/main/assets/garments/real"


def on_white(path: Path) -> bytes:
    """A catalog picture with a see-through background, put on white, as the photo engine does."""
    from PIL import Image
    picture = Image.open(path).convert("RGBA")
    canvas = Image.new("RGB", picture.size, "white")
    canvas.paste(picture, (0, 0), picture)
    out = io.BytesIO()
    canvas.save(out, "PNG")
    return out.getvalue()


@app.local_entrypoint()
def main(video: str, out: str, look_only: bool = False, start: float = 0.0, end: float = 0.0, step: float = 1.0,
         height: int = 260,
         top: str = "navy_polo", bottom: str = "khaki_chinos", prepare_only: bool = False, run: str = "",
         fashn_demo: bool = False, magic: str = "", white: bool = False, blend: str = "mask", teacache: bool = True,
         turbo: bool = False, fill: str = "grey", tight: int = 0):
    """--prepare-only stops after the masks (it prints the run name); --run <name> dresses a clip prepared before.
    --fashn-demo --run <name> --magic <MagicTryOn's dressed.mp4>: FASHN on every frame, compared side by side."""
    dst = Path(out)
    dst.mkdir(parents=True, exist_ok=True)
    data = Path(video).read_bytes()
    if look_only:
        r = look.remote(data, start, end, step, height)
        (dst / "look.jpg").write_bytes(r["sheet"])
        print(r["info"], "->", dst / "look.jpg")
        return
    if fashn_demo:
        t = time.time()
        top_b = on_white(assets() / f"{top}_front.png")
        bottom_b = on_white(assets() / f"{bottom}_front.png") if bottom else None
        frames = list(FashnFrames().dress.starmap([(run, i, top_b, bottom_b) for i in range(FRAMES)]))
        print(f"FASHN dressed {FRAMES} frames in {time.time() - t:.0f} s")
        r = compose.remote(run, frames, Path(magic).read_bytes())
        for name, ext in (("fashn", "mp4"), ("three_way", "mp4"), ("sheet", "jpg"), ("consecutive", "jpg")):
            (dst / f"{name}.{ext}").write_bytes(r[name])
        print("saved in", dst)
        return
    t = time.time()
    if not run:
        run = time.strftime("%Y%m%d-%H%M%S")
        blend = "outline" if white else blend
        p = prepare.remote(data, start, end, run, white, blend == "outline")
        (dst / "prepared.jpg").write_bytes(p["sheet"])
        print(f"run {run} prepared in {p['seconds']:.0f} s: frames {p['frames']}, crop {p['box']}, "
              f"frames borrowed {p['misses']}")
        if prepare_only:
            return
    garment = lambda gid: (assets() / f"{gid}_front.png").read_bytes()
    r = dress.remote(run, garment(top), CAPTIONS[top], garment(bottom) if bottom else None, CAPTIONS.get(bottom, ""),
                     white, "outline" if white else blend, teacache, turbo, fill, tight)
    (dst / "raw.jpg").write_bytes(r["raw"])
    (dst / "dressed.mp4").write_bytes(r["dressed"])
    (dst / "side_by_side.mp4").write_bytes(r["side_by_side"])
    (dst / "sheet.jpg").write_bytes(r["sheet"])
    (dst / "consecutive.jpg").write_bytes(r["consecutive"])
    for k, f in enumerate(r["frames"]):
        (dst / f"frame_{k}.jpg").write_bytes(f)
    print(f"GPU {r['gpu']}; " + ", ".join(f"{k} {v:.0f} s" for k, v in r["timings"].items())
          + f"; whole run {time.time() - t:.0f} s")
    print("saved in", dst)
