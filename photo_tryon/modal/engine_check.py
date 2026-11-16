"""Puts one outfit through the deployed engine exactly as the app does, and times each view.
usage: python engine_check.py <folder with view_0.jpg ... view_7.jpg> <output folder> [top id] [bottom id]"""
import json, sys, time, urllib.request, subprocess
from pathlib import Path
base = subprocess.run([sys.executable, "engine_key.py"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
top, bottom = (sys.argv[3:5] + ["navy_polo", "khaki_chinos"][len(sys.argv[3:5]):])
assets = Path(__file__).resolve().parents[2] / "app/src/main/assets/garments/real"
def call(method, path, data=None, kind="application/octet-stream"):
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": kind})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
        return json.loads(raw) if r.headers.get("Content-Type", "").startswith("application/json") else raw
t0 = time.time()
print("health:", call("GET", "/health")["model"], f"({time.time() - t0:.0f}s, includes waking the web server)")
for i in range(8): call("PUT", f"/session/check2/view_{i}", (src / f"view_{i}.jpg").read_bytes())
for g in (top, bottom):
    for face in ("front", "back"): call("PUT", f"/garment/{g}/{face}", (assets / f"{g}_{face}.png" if (assets / f"{g}_{face}.png").exists() else assets / f"{g}_front.png").read_bytes())
t1 = time.time()
job = call("POST", "/job", json.dumps({"session": "check2", "top": top, "bottom": bottom}).encode(), "application/json")["job"]
seen = set()
while True:
    s = call("GET", f"/job/{job}")
    for part in s["parts"]:
        if part not in seen:
            seen.add(part); (out / f"{part}.jpg").write_bytes(call("GET", f"/job/{job}/file/{part}.jpg"))
            print(f"{part} ready after {time.time() - t1:.0f}s")
    if s["state"] in ("done", "error"): break
    time.sleep(2)
print("state:", s["state"], "| error:", (s["error"] or "")[-400:], "| total", f"{time.time() - t1:.0f}s")
