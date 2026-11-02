"""
Offline check of the mesh warp that draws trousers (M4).

    python tools/preview_trousers.py [--id khaki_chinos] [--top navy_polo]

Bends each pair of trousers through a set of leg poses with the same maths as OverlayView.kt's
drawTrousers - grid, weights, pelvis and leg bones, blend - and writes
tools/trouser_preview_<id>.png. The last tile puts a shirt on over them, drawn as
preview_mesh.py draws it, to check where the waistband sits under the shirt's hem. Like
preview_mesh.py for shirts, it exists so a warp problem shows up here rather than on a phone.
The two implementations must change together.

How the warp works
------------------
Three bones. The pelvis carries the waistband: it maps the garment's hip anchors onto the
wearer's hips, and a point one leg length below their midpoint onto the point one of the wearer's
leg lengths below theirs, so the hips' spacing sets its scale across and the wearer's legs set it
down. Each leg swings from its hip to its ankle, as long as the wearer's leg and as wide as the
pelvis is across it (preview_garments.segment_matrix, as for a shirt's sleeves).

Every grid vertex follows the pelvis above the hip anchors and its leg from a little below the
crotch down, blending in between; left of the crotch it is the left leg's, right of it the
right's, blended across a narrow band where the legs meet.

The app also handles a leg whose ankle is out of sight (it points the leg at the knee instead)
and one with neither in sight (it hangs from the pelvis). The preview's stand-in body always has
both, so only the main path is mirrored here.
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

import preview_mesh as shirt
from preview_garments import affine_from_points, segment_matrix, checkerboard

REAL = shirt.REAL

# TrouserMesh. Change together.
LEGS_BELOW_CROTCH = 0.1  # legs take over fully this far below the crotch, x the rise
SPLIT = 0.05             # half-width of the blend between the legs, x the hip anchors' spacing

# Stand-in body: preview_mesh's, with legs. Hip to ankle per unit of shoulder-joint width: 0.49
# of standing height against 0.2 for the shoulder joints' spacing (anthropometric; the 0.2 is
# what preview_mesh's HIP_DROP of 1.45 implies for a 0.29 shoulder-to-hip drop).
LEG_PER_SHOULDER_WIDTH = 2.45
CANVAS = (1080, 1700)

# Each leg's angle out from straight down, in degrees.
POSES = [("standing", 2, 2), ("feet apart", 10, 10), ("left leg out 25°", 25, 2),
         ("legs crossing in", -4, -4)]


def pt(p):
    return np.array([p["x"], p["y"]], float)


def leg_weights(entry, points):
    """(n, 3) weights for pelvis, left leg, right leg (TrouserMesh)."""
    a = entry["anchors"]
    hip_y = (a["left_hip"]["y"] + a["right_hip"]["y"]) / 2
    waist_y = (entry["waist"]["left"]["y"] + entry["waist"]["right"]["y"]) / 2
    crotch = entry["crotch"]
    legs = shirt.smoothstep(hip_y, crotch["y"] + LEGS_BELOW_CROTCH * (crotch["y"] - waist_y), points[:, 1])
    band = SPLIT * (a["right_hip"]["x"] - a["left_hip"]["x"])
    left = shirt.smoothstep(crotch["x"] + band, crotch["x"] - band, points[:, 0])
    return np.stack([1 - legs, legs * left, legs * (1 - left)], axis=1)


def body(left_deg, right_deg):
    """preview_mesh's stand-in body, arms at rest, with straight legs at the given angles."""
    view = shirt.body(8, 8)
    length = LEG_PER_SHOULDER_WIDTH * shirt.SHOULDER_WIDTH
    for side, sign, deg in (("left", -1, left_deg), ("right", 1, right_deg)):
        r = math.radians(deg)
        hx, hy = view[f"{side}_hip"]
        ax, ay = hx + sign * length * math.sin(r), hy + length * math.cos(r)
        view[f"{side}_ankle"] = (ax, ay)
        view[f"{side}_knee"] = ((hx + ax) / 2, (hy + ay) / 2)
    for name in list(view):
        view[name] = (view[name][0], view[name][1] - 120)   # make room below for the feet
    return view


def down(a, b):
    """Unit vector square to the line a -> b, on the image-down side (OverlayView.computePelvis)."""
    d = np.array(b, float) - np.array(a, float)
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    return n if n[1] >= 0 else -n


def pelvis(entry, view):
    """The pelvis bone and its handedness (OverlayView.computePelvis)."""
    a = entry["anchors"]
    g_l, g_r = pt(a["left_hip"]), pt(a["right_hip"])
    g_len = (np.linalg.norm(pt(a["left_ankle"]) - g_l) + np.linalg.norm(pt(a["right_ankle"]) - g_r)) / 2
    v_l, v_r = np.array(view["left_hip"], float), np.array(view["right_hip"], float)
    v_len = (np.linalg.norm(np.array(view["left_ankle"]) - v_l) +
             np.linalg.norm(np.array(view["right_ankle"]) - v_r)) / 2
    src = [tuple(g_l), tuple(g_r), tuple((g_l + g_r) / 2 + down(g_l, g_r) * g_len)]
    dst = [tuple(v_l), tuple(v_r), tuple((v_l + v_r) / 2 + down(v_l, v_r) * v_len)]
    m = affine_from_points(src, dst)
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    return m, (1.0 if det >= 0 else -1.0)


def place(entry, src, view):
    """Every grid vertex on the body: each bone's placement blended by weight (drawTrousers)."""
    a = entry["anchors"]
    m, handedness = pelvis(entry, view)
    bones = [m]
    for side in ("left", "right"):
        leg = segment_matrix(tuple(pt(a[f"{side}_hip"])), tuple(pt(a[f"{side}_ankle"])),
                             view[f"{side}_hip"], view[f"{side}_ankle"], m, handedness)
        bones.append(leg if leg is not None else m)
    return shirt.skin(src, leg_weights(entry, src), bones)


def render(entry, texture, left_deg, right_deg, top=None):
    w, h = texture.size
    cols, rows, src = shirt.grid(w, h)
    view = body(left_deg, right_deg)
    canvas = checkerboard(*CANVAS).convert("RGB")
    canvas = shirt.draw_mesh(canvas, texture, cols, rows, src, place(entry, src, view))
    if top is not None:
        t_entry, t_texture, t_sleeves = top
        t_cols, t_rows, t_src = shirt.grid(*t_texture.size)
        t_dst = shirt.place(t_entry, t_texture, t_src, shirt.vertex_weights(t_entry, t_src), view)
        canvas = shirt.draw_mesh(canvas, t_texture, t_cols, t_rows, t_src, t_dst)
        canvas = shirt.draw_mesh(canvas, t_sleeves, t_cols, t_rows, t_src, t_dst)
    d = ImageDraw.Draw(canvas)
    bone = (40, 170, 180)
    for side in ("left", "right"):
        chain = [view[f"{side}_hip"], view[f"{side}_knee"], view[f"{side}_ankle"]]
        d.line(chain, fill=bone, width=4)
        for x, y in chain:
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(255, 255, 255), outline=bone, width=3)
    d.line([view["left_hip"], view["right_hip"]], fill=bone, width=4)
    return canvas


def weights_image(entry, texture):
    """The photo tinted by weight: red = left leg, blue = right leg, grey = pelvis."""
    w, h = texture.size
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
    wt = leg_weights(entry, pts).reshape(h, w, 3)
    base = np.asarray(checkerboard(w, h).convert("RGB"), np.float32)
    tex = np.asarray(texture, np.float32)
    alpha = tex[..., 3:4] / 255
    grey = tex[..., :3].mean(axis=2, keepdims=True).repeat(3, axis=2)
    tint = (wt[..., 0:1] * grey + wt[..., 1:2] * np.array([235, 70, 60]) + wt[..., 2:3] * np.array([60, 110, 235]))
    out = tint * alpha + base * (1 - alpha)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=None)
    ap.add_argument("--top", default="navy_polo", help="shirt worn over the trousers in the last tile")
    args = ap.parse_args()
    with open(os.path.join(REAL, "catalog.json"), encoding="utf-8") as handle:
        catalog = json.load(handle)
    top = None
    for entry in catalog["garments"]:
        if entry["id"] == args.top:
            texture = shirt.clear_neck(Image.open(os.path.join(REAL, entry["front"])).convert("RGBA"), entry)
            top = (entry, texture, shirt.sleeve_only(texture, entry))
    for entry in catalog["garments"]:
        if entry["category"] != "lower_body" or (args.id and entry["id"] != args.id):
            continue
        texture = Image.open(os.path.join(REAL, entry["front"])).convert("RGBA")
        tiles = [("weights (red/blue = legs)", weights_image(entry, texture))]
        for label, l, r in POSES:
            tiles.append((label, render(entry, texture, l, r)))
        if top is not None:
            tiles.append((f"with {top[0]['name']}", render(entry, texture, 2, 2, top)))
        scale = 0.36
        cells = [(label, im.resize((int(im.width * scale * (CANVAS[1] / im.height)) if i == 0 else int(im.width * scale),
                                    int(CANVAS[1] * scale)), Image.LANCZOS)) for i, (label, im) in enumerate(tiles)]
        width = sum(c[1].width for c in cells) + 20 * (len(cells) + 1)
        sheet = Image.new("RGB", (width, cells[0][1].height + 50), (247, 249, 250))
        d = ImageDraw.Draw(sheet)
        x = 20
        for label, im in cells:
            d.text((x, 12), label, fill=(20, 24, 28))
            sheet.paste(im, (x, 36))
            x += im.width + 20
        out = os.path.join("tools", f"trouser_preview_{entry['id']}.png")
        sheet.save(out)
        print(f"wrote {out}  ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
