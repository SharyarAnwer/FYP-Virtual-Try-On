"""Writes tryon_server.ipynb: the Photo Try-On engine for the eight-view turntable.

usage: python make_notebook_engine.py <output path>

Run on Kaggle as a saved version (a "batch" run) the notebook tests itself on the eight photos in its
input dataset and stops. Run from the editor with Run All, it serves the app until stopped.
"""
import json
import sys


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md(r"""
# Photo Try-On engine

The phone app's back end, on Kaggle's free GPUs. It takes eight photos of a person - one every 45
degrees round them - and an outfit, and dresses each photo with the open-source **Leffa** try-on
model. Shown one after another on the phone, the dressed photos are a turntable of real pictures.

**To use it with the app:** Accelerator **GPU T4 x2**, Internet **On**, then **Run All**. After about
five minutes the last cell prints `ADDRESS FOR THE APP`. Paste it into the app. Keep this tab open while
using the app, and stop the session when finished: the weekly GPU hours count while it runs.

Saved as a version instead (a batch run), it dresses the eight photos in its input dataset as a test
and stops.
"""),
    code(r"""
import time, os, glob, sys, io, json, re, threading, queue, subprocess, traceback, secrets, urllib.request, torch
MODE = "selftest" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Interactive") == "Batch" else "serve"
STEPS = 12                  # refinement steps per dressed picture: 12 looked as good as 30 in testing
T0 = time.time()
def stamp(what): print(f"[{time.time() - T0:6.0f}s] {what}", flush=True)
N_GPU = torch.cuda.device_count()
print("mode:", MODE, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPU)] or "NONE")
assert N_GPU >= 1, "No GPU in this session: set Accelerator to GPU T4 x2 in the panel on the right."
def run(cmd, tail=3):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if tail: print("\n".join((p.stdout + p.stderr).strip().splitlines()[-tail:]))
    return p.returncode
try:
    urllib.request.urlopen("https://huggingface.co", timeout=15)
except Exception as e:
    raise SystemExit("Internet is off: switch it on in the panel on the right, then Run All again.")

if not os.path.exists("/tmp/Leffa"): run("git clone -q --depth 1 https://github.com/franciszzj/Leffa /tmp/Leffa")
# Keep every package Kaggle already has at its version: replacing numpy or pillow in a running session breaks them.
run("pip freeze | grep '==' > /tmp/constraints.txt", tail=0)
run("pip install -q -c /tmp/constraints.txt fvcore omegaconf onnxruntime pycocotools av config cloudpickle einops peft")
from huggingface_hub import snapshot_download
CK = "/tmp/ckpts"
snapshot_download(repo_id="franciszzj/Leffa", local_dir=CK,
                  ignore_patterns=["pose_transfer.pth", "stable-diffusion-xl-1.0-inpainting-0.1/*", "examples/*", "assets/*"])
stamp("code, packages and model files ready")
"""),
    md("## The dressing models: both on each GPU, so two photos are dressed at once"),
    code(r"""
os.chdir("/tmp/Leffa"); sys.path.insert(0, "/tmp/Leffa")
import numpy as np, cv2
from PIL import Image, ImageFilter, ImageOps
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, get_agnostic_mask_dc
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

densepose_predictor = DensePosePredictor(config_path=f"{CK}/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
                                         weights_path=f"{CK}/densepose/model_final_162be9.pkl")
parsing = Parsing(atr_path=f"{CK}/humanparsing/parsing_atr.onnx", lip_path=f"{CK}/humanparsing/parsing_lip.onnx")
openpose = OpenPose(body_model_path=f"{CK}/openpose/body_pose_model.pth")

WEIGHTS = {"viton_hd": "virtual_tryon.pth", "dress_code": "virtual_tryon_dc.pth"}
inference = {}
for gpu in range(N_GPU):
    for name, weights in WEIGHTS.items():
        model = LeffaModel(pretrained_model_name_or_path=f"{CK}/stable-diffusion-inpainting",
                           pretrained_model=f"{CK}/{weights}", dtype="float16")
        inf = LeffaInference(model=model)
        if gpu: inf.device = f"cuda:{gpu}"; inf.model.to(inf.device); inf.pipe.device = inf.device
        inference[(gpu, name)] = inf
        stamp(f"loaded {name} on GPU {gpu}")
"""),
    md("""
## Preparing photos and dressing them

Three things here differ from Leffa's own demo, each from a fault seen in testing:

- **The head is kept.** Leffa's mask for tops protects the face but not hair, and a beard counts as
  hair, so beards were repainted. Face, hair and beard are taken out of the repainted area.
- **Only the body is repainted.** What the model painted is pasted back only on and just around the
  body: it drew scribbles by the ears and wiped out the door behind.
- **A wrong shirt is caught.** On views of the back the model sometimes paints a panel of another colour
  across the shirt. The share of the back, between the shoulders and the hips, that is far from the
  garment's own colour is measured; a back view over [WRONG_SHIRT] is tried again with another model or
  seed, and if every try is wrong it is left out of the turntable rather than shown.
"""),
    code(r'''
HEAD = [1, 2, 3, 11]                 # hat, hair (a beard counts as hair), sunglasses, face
WRONG_SHIRT = 0.08                   # share of the back far from the garment's colour above which a shirt is wrong
prep_lock = threading.Lock()         # the body-part and pose models are shared: one photo at a time

def parse_small(image768):
    with prep_lock:
        parse, _ = parsing(image768.resize((384, 512)))
    return np.array(parse.resize((384, 512), Image.NEAREST))

def prepare(photo):
    # A phone photo -> the person, upright, filling a 768 x 1024 picture: the same size in every view.
    photo = ImageOps.exif_transpose(photo).convert("RGB")
    w, h = photo.size
    scale = min(384 / w, 512 / h)
    ox, oy = (384 - w * scale) / 2, (512 - h * scale) / 2
    body = parse_small(resize_and_center(photo, 768, 1024)) != 0
    ys, xs = np.where(body)
    if len(ys) < 500: return resize_and_center(photo, 768, 1024).convert("RGB")
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
    if luma < 85:                    # a dim photo is lifted: the model repaints dark photos blotchily
        gamma = np.log(110 / 255) / np.log(max(luma, 10) / 255)
        out = Image.fromarray((255 * (np.asarray(out, np.float32) / 255) ** gamma).clip(0, 255).astype(np.uint8))
    return out

def prepare_garment(picture):
    picture = picture.convert("RGBA")
    canvas = Image.new("RGB", (768, 1024), "white")
    picture.thumbnail((700, 950))
    canvas.paste(picture, ((768 - picture.width) // 2, (1024 - picture.height) // 2), picture)
    return canvas

def lab(image_array):
    # 8-bit RGB -> CIE Lab with L 0..100 and a, b around 0.
    l = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB).astype(np.float32)
    l[..., 0] *= 100 / 255; l[..., 1:] -= 128
    return l

def garment_colour(garment):
    g = np.array(garment)
    cloth = ~((g[..., 0] > 235) & (g[..., 1] > 235) & (g[..., 2] > 235))
    return np.median(lab(g)[cloth], axis=0) if cloth.any() else np.zeros(3, np.float32)

def wrong_share(dressed, area, keypoints, colour):
    # The share of the back, between shoulders and hips, far from the garment's colour. Lightness counts
    # half, so a fold's shadow is not taken for another colour.
    try:
        k = np.array(keypoints["pose_keypoints_2d"], float).reshape(-1, 2)
        xs = sorted([k[2, 0], k[5, 0]]); top = (k[2, 1] + k[5, 1]) / 2; bottom = (k[8, 1] + k[11, 1]) / 2
        if min(xs) <= 0 or bottom <= top + 20 or xs[1] - xs[0] < 15: raise ValueError
        w, h = xs[1] - xs[0], bottom - top
        box = (int(xs[0] + 0.15 * w), int(xs[1] - 0.15 * w), int(top + 0.12 * h), int(bottom - 0.08 * h))
    except Exception:
        ys, xs_ = np.where(area)
        if len(ys) == 0: return 1.0
        x0, x1, y0, y1 = xs_.min(), xs_.max(), ys.min(), ys.max()
        box = (int(x0 + 0.3 * (x1 - x0)), int(x1 - 0.3 * (x1 - x0)), int(y0 + 0.2 * (y1 - y0)), int(y0 + 0.6 * (y1 - y0)))
    inside = np.zeros_like(area); inside[box[2]:box[3], box[0]:box[1]] = True
    inside &= area
    if inside.sum() < 200: return 0.0
    small = np.array(dressed.resize((384, 512)))
    d = lab(small)[inside] - colour
    far = np.sqrt((0.5 * d[:, 0]) ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2) > 25
    return round(float(far.mean()), 3)

def dress(person, garment, kind, model_type=None, gpu=0, steps=STEPS, seed=42, colour=None):
    # Returns (dressed picture, wrong-shirt share or None).
    model_type = model_type or ("viton_hd" if kind == "upper_body" else "dress_code")
    src = person.convert("RGB"); src_array = np.array(src)
    with prep_lock:
        model_parse, _ = parsing(src.resize((384, 512)))
        keypoints = openpose(src.resize((384, 512)))
        if model_type == "viton_hd":
            mask = get_agnostic_mask_hd(model_parse, keypoints, kind)
            densepose = Image.fromarray(densepose_predictor.predict_seg(src_array)[:, :, ::-1])
        else:
            mask = get_agnostic_mask_dc(model_parse, keypoints, kind)
            seg = densepose_predictor.predict_iuv(src_array)[:, :, 0:1]
            densepose = Image.fromarray(np.concatenate([seg] * 3, axis=-1))
    parse = np.array(model_parse.resize((384, 512), Image.NEAREST))
    head = cv2.dilate(np.isin(parse, HEAD).astype(np.uint8), np.ones((5, 5), np.uint8))
    area = (np.array(mask.resize((384, 512), Image.NEAREST).convert("L")) > 127) & (head == 0)
    mask = Image.fromarray((area * 255).astype(np.uint8)).resize((768, 1024), Image.NEAREST)
    data = LeffaTransform()({"src_image": [src], "ref_image": [garment], "mask": [mask], "densepose": [densepose]})
    out = inference[(gpu, model_type)](data, ref_acceleration=False, num_inference_steps=steps,
                                       guidance_scale=2.5, seed=seed, repaint=False)
    generated = out["generated_image"][0].convert("RGB")
    body = parse != 0
    keep = area & (cv2.dilate(body.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
    rows = np.where(np.isin(parse, HEAD).any(axis=1))[0]
    if len(rows): keep[:rows.max()] &= body[:rows.max()]
    soft = Image.fromarray((keep * 255).astype(np.uint8)).resize((768, 1024), Image.NEAREST)
    soft = soft.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(3))
    soft = Image.composite(Image.new("L", soft.size, 0), soft, Image.fromarray(head * 255).resize((768, 1024), Image.NEAREST))
    result = Image.composite(generated, src, soft)
    share = wrong_share(result, area, keypoints, colour) if colour is not None else None
    return result, share
'''),
    md("""
## Jobs

One outfit on one person is a job. The eight views are dressed in the order the wearer wants them -
front, back, the two sides, then the four in between - two at a time, and each is saved the moment it
is ready, so the phone can show it while the rest are made.

Views 3, 4 and 5 show the back and get the garments' back pictures. Testing settled which model draws
the shirt best on each: the first model on the back three-quarter views, the second on the straight back.
"""),
    code(r"""
VIEWS = [f"view_{i}" for i in range(8)]
ORDER = [0, 4, 2, 6, 1, 5, 3, 7]
BACK = {3, 4, 5}
# Tries for the shirt on each back view, in order: (model, seed). Other views get one try.
BACK_TRIES = {3: [("viton_hd", 42), ("viton_hd", 123), ("dress_code", 42)],
              5: [("viton_hd", 42), ("dress_code", 42), ("viton_hd", 123)],
              4: [("dress_code", 123), ("viton_hd", 42), ("dress_code", 42)]}
sessions, garments, jobs, job_order = {}, {}, {}, []
state_lock = threading.Lock()
work = threading.Semaphore(0)

def new_job(session, top, bottom):
    key = f"{session}|{top}|{bottom}"
    with state_lock:
        for j in jobs.values():
            if j["key"] == key and j["state"] != "error": return j["id"]
        jid = secrets.token_hex(6)
        os.makedirs(f"/tmp/jobs/{jid}", exist_ok=True)
        jobs[jid] = {"id": jid, "key": key, "session": session, "top": top, "bottom": bottom, "state": "queued",
                     "stage": "waiting", "parts": [], "skipped": [], "seconds": {}, "tries": {}, "error": None, "t0": time.time()}
        job_order.append(jid)
    work.release()
    return jid

def dress_view(job, index, gpu):
    view = VIEWS[index]
    picture = sessions[job["session"]][view]
    face = "back" if index in BACK else "front"
    if job["top"]:
        g = garments[job["top"]]
        shirt = g.get(face) or g["front"]
        colour = garment_colour(shirt)
        tries = BACK_TRIES.get(index, [("viton_hd", 42)])
        best = None
        for model, seed in tries:
            dressed, share = dress(picture, shirt, "upper_body", model, gpu=gpu, seed=seed, colour=colour)
            job["tries"].setdefault(view, []).append([model, seed, share])
            if best is None or share < best[1]: best = (dressed, share)
            if index not in BACK or share <= WRONG_SHIRT: break
        if index in BACK and best[1] > WRONG_SHIRT:
            job["skipped"].append(view)
            stamp(f"job {job['id']}: {view} left out - every try drew the wrong shirt {job['tries'][view]}")
            return
        picture = best[0]
    if job["bottom"]:
        g = garments[job["bottom"]]
        picture, _ = dress(picture, g.get(face) or g["front"], "lower_body", gpu=gpu)
    d = f"/tmp/jobs/{job['id']}"
    picture.save(f"{d}/{view}_tmp.jpg", quality=92); os.replace(f"{d}/{view}_tmp.jpg", f"{d}/{view}.jpg")
    with state_lock:
        job["parts"].append(view); job["seconds"][view] = round(time.time() - job["t0"], 1)
        job["stage"] = f"{len(job['parts'])} of {8 - len(job['skipped'])} views ready"
    stamp(f"job {job['id']}: {view} ready at {job['seconds'][view]}s")

def run_job(job):
    job["state"] = "running"; job["t0"] = time.time(); job["stage"] = "dressing your front view"
    have = [i for i in ORDER if VIEWS[i] in sessions[job["session"]]]
    todo = queue.Queue()
    for i in have: todo.put(i)
    def worker(gpu):
        while not job["error"]:
            try: i = todo.get_nowait()
            except queue.Empty: return
            try: dress_view(job, i, gpu)
            except Exception: job["error"] = traceback.format_exc()[-900:]; print(job["error"])
    threads = [threading.Thread(target=worker, args=(g,)) for g in range(N_GPU)]
    for t in threads: t.start()
    for t in threads: t.join()

def job_loop():
    while True:
        work.acquire()
        with state_lock: jid = next((j for j in job_order if jobs[j]["state"] == "queued"), None)
        if jid is None: continue
        job = jobs[jid]
        try: run_job(job)
        except Exception: job["error"] = traceback.format_exc()[-900:]; print(job["error"])
        job["state"] = "error" if job["error"] else "done"; job["stage"] = job["state"]
        job["seconds"]["total"] = round(time.time() - job["t0"], 1)
threading.Thread(target=job_loop, daemon=True).start()
"""),
    md("""
## The web server and the tunnel

Plain requests, so the phone needs no extra library: `PUT` a photo or a garment picture, `POST` a job,
`GET` its progress and its files. Every address starts with a key made fresh each session.
"""),
    code(r"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
KEY = secrets.token_hex(8)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body, kind="application/json"):
        if not isinstance(body, bytes): body = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", kind); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def route(self):
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        if len(parts) < 2 or parts[0] != "k" or parts[1] != KEY: self.send(403, {"error": "wrong key"}); return None
        return parts[2:]
    def body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))
    def do_GET(self):
        p = self.route()
        if p is None: return
        try:
            if p == ["health"]:
                self.send(200, {"ok": True, "views": len(VIEWS), "busy": any(j["state"] in ("queued", "running") for j in jobs.values()),
                                "garments": {g: sorted(f) for g, f in garments.items()}})
            elif len(p) == 2 and p[0] == "job" and p[1] in jobs:
                j = jobs[p[1]]
                self.send(200, {k: j[k] for k in ("id", "state", "stage", "parts", "skipped", "seconds", "error")})
            elif len(p) == 4 and p[0] == "job" and p[2] == "file" and p[1] in jobs and p[3] in {v + ".jpg" for v in VIEWS} \
                    and os.path.exists(f"/tmp/jobs/{p[1]}/{p[3]}"):
                self.send(200, open(f"/tmp/jobs/{p[1]}/{p[3]}", "rb").read(), "image/jpeg")
            else: self.send(404, {"error": "not found"})
        except Exception as e: self.send(500, {"error": str(e)})
    def do_PUT(self):
        p = self.route()
        if p is None: return
        try:
            data = self.body()
            if len(p) == 3 and p[0] == "session" and p[2] in VIEWS:
                sessions.setdefault(p[1], {})[p[2]] = prepare(Image.open(io.BytesIO(data)))
                self.send(200, {"ok": True})
            elif len(p) == 3 and p[0] == "garment" and p[2] in ("front", "back"):
                garments.setdefault(p[1], {})[p[2]] = prepare_garment(Image.open(io.BytesIO(data)))
                self.send(200, {"ok": True})
            else: self.send(404, {"error": "not found"})
        except Exception as e: self.send(500, {"error": str(e)})
    def do_POST(self):
        p = self.route()
        if p is None: return
        try:
            q = json.loads(self.body() or b"{}")
            if p != ["job"]: self.send(404, {"error": "not found"}); return
            if q.get("session") not in sessions or "view_0" not in sessions[q["session"]]: self.send(400, {"error": "photos not uploaded"}); return
            for g in (q.get("top"), q.get("bottom")):
                if g and g not in garments: self.send(400, {"error": f"garment {g} not uploaded"}); return
            if not q.get("top") and not q.get("bottom"): self.send(400, {"error": "no garment chosen"}); return
            self.send(200, {"job": new_job(q["session"], q.get("top"), q.get("bottom"))})
        except Exception as e: self.send(500, {"error": str(e)})

server = ThreadingHTTPServer(("0.0.0.0", 8000), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
LOCAL = f"http://127.0.0.1:8000/k/{KEY}"
stamp("web server up")

PUBLIC = None
if MODE == "serve":
    if not os.path.exists("/tmp/cloudflared"):
        urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "/tmp/cloudflared")
        os.chmod("/tmp/cloudflared", 0o755)
    tunnel = subprocess.Popen(["/tmp/cloudflared", "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"],
                              stdout=open("/tmp/tunnel.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(60):
        found = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("/tmp/tunnel.log").read())
        if found: PUBLIC = f"{found.group(0)}/k/{KEY}"; break
        time.sleep(1)
    assert PUBLIC, "The tunnel did not open:\n" + open("/tmp/tunnel.log").read()[-2000:]
    stamp("tunnel open")
"""),
    md("## Serve the app, or test once and stop"),
    code(r"""
def call(method, path, data=None, kind="application/octet-stream"):
    req = urllib.request.Request(LOCAL + path, data=data, method=method, headers={"Content-Type": kind})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
        return json.loads(raw) if r.headers.get("Content-Type", "").startswith("application/json") else raw

if MODE == "selftest":
    OUT = "/kaggle/working/selftest"; os.makedirs(OUT, exist_ok=True)
    found = glob.glob("/kaggle/input/**/view_0.jpg", recursive=True)
    assert found, "Add the dataset with view_0.jpg ... view_7.jpg under Input."
    IN = os.path.dirname(found[0])
    for v in VIEWS: call("PUT", f"/session/test/{v}", open(f"{IN}/{v}.jpg", "rb").read())
    for gid in ("navy_polo", "khaki_chinos"):
        for face in ("front", "back"): call("PUT", f"/garment/{gid}/{face}", open(f"{IN}/garment_{gid}_{face}.jpg", "rb").read())
    job = call("POST", "/job", json.dumps({"session": "test", "top": "navy_polo", "bottom": "khaki_chinos"}).encode(), "application/json")["job"]
    while True:
        s = call("GET", f"/job/{job}")
        if s["state"] in ("done", "error"): break
        time.sleep(2)
    print(json.dumps(s, indent=1))
    print("tries on back views:", json.dumps(jobs[job]["tries"], indent=1))
    sheet = Image.new("RGB", (384 * 8, 512 * 2), "white")
    for i, v in enumerate(VIEWS):
        sheet.paste(sessions["test"][v].resize((384, 512)), (i * 384, 0))
        if os.path.exists(f"/tmp/jobs/{job}/{v}.jpg"):
            pic = Image.open(f"/tmp/jobs/{job}/{v}.jpg"); pic.save(f"{OUT}/{v}.jpg", quality=90)
            sheet.paste(pic.resize((384, 512)), (i * 384, 512))
    sheet.save(f"{OUT}/sheet.jpg", quality=88)
    print("GPU memory peak (GB):", [round(torch.cuda.max_memory_allocated(i) / 1e9, 1) for i in range(N_GPU)])
    stamp("self-test finished")
else:
    print("\n" + "=" * 78 + f"\nADDRESS FOR THE APP:\n\n    {PUBLIC}\n\n" + "=" * 78 + "\n", flush=True)
    while True:
        time.sleep(60)
        busy = sum(j["state"] in ("queued", "running") for j in jobs.values())
        print(f"[{time.strftime('%H:%M')}] serving | outfits so far {len(jobs)}, in progress {busy}", flush=True)
"""),
]

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("notebook written,", len(cells), "cells")
