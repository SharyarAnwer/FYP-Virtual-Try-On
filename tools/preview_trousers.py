"""
Offline check of the mesh warp that draws trousers (M4).

    python tools/preview_trousers.py [--id khaki_chinos] [--top navy_polo]

Bends each pair of trousers through a set of leg poses with the same maths as OverlayView.kt's
drawTrousers - grid, weights, widening, pelvis and leg bones, blend - and writes
tools/trouser_preview_<id>.png. The last tile puts a shirt on over them, drawn as
preview_mesh.py draws it, to check where the waistband sits under the shirt's hem. Like
preview_mesh.py for shirts, it exists so a warp problem shows up here rather than on a phone.
The two implementations must change together.

How the warp works
------------------
Five bones. The pelvis carries the waistband: it maps the garment's hip anchors onto the
wearer's hips, and a point one leg length below their midpoint onto the point one of the wearer's
leg lengths below theirs, so the hips' spacing sets its scale across and the wearer's legs set it
down. Leg lengths run hip to knee to ankle, so bending the knees does not squash the seat. Each
leg has a thigh, hip to knee, and a shin, knee to ankle, each as long as the wearer's and as wide
as the pelvis is across it (preview_garments.segment_matrix, as for a shirt's sleeves).

Every grid vertex follows the pelvis above the hip anchors and its leg from a little below the
crotch down, blending in between; left of the crotch it is the left leg's, right of it the
right's, blended across a narrow band where the legs meet; and down each leg, the thigh above the
knee and the shin below, blended across a band at the knee.

How the fit works
-----------------
Before skinning, the grid is widened to the wearer (TrouserMesh.widen), in one of three modes
(OverlayView.FitMode): "hips" fits each side of the garment at its hip anchors to the wearer's
hip, one widening for every row; "rows" fits the seat row by row to the wearer's width there and
each leg to its measured thickness, outer and inner sides apart; "off" leaves the photo's width.
The seat widens about the garment's centre line, the legs below the knee about their own.

The app also handles a leg whose ankle is out of sight (it points the leg at the knee instead)
and one with neither in sight (it hangs from the pelvis), and fits a seat row by row to the
wearer's width at that height. The preview's stand-in body always has both legs and is the same
width at every height, so only the main path is mirrored here.
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
KNEE_BLEND = 0.08        # half-width of the thigh/shin blend at the knee, x that leg's hip-to-ankle

# OverlayView.fitTrousers and BodyWidthMeter. Change together.
FIT_ALLOWANCE = shirt.FIT_ALLOWANCE
MIN_EASE, MAX_EASE = shirt.MIN_EASE, shirt.MAX_EASE
LEG_SAMPLES = [0.4, 0.7, 1.0, 1.3, 1.6, 1.85]   # 0 hip, 1 knee, 2 ankle

# The app measures the wearer live: each side's half-width at the hips, and each leg's outer and
# inner half-thickness down its length. The preview has no wearer, so it stands one in, in
# hip-joint spacings: hips about what the tester measured in loose trousers in M4's shots 1 and 2
# (2.0 and 2.3 spacings across), legs tapering from thigh to ankle (0.84 across at the knee in
# shot 2).
STANDIN_HIP_HALF = 1.05
STANDIN_LEG_HALF = ([0.4, 1.0, 1.85], [0.5, 0.42, 0.33])

# Stand-in body: preview_mesh's, with legs. Hip to ankle per unit of shoulder-joint width: 0.49
# of standing height against 0.2 for the shoulder joints' spacing (anthropometric; the 0.2 is
# what preview_mesh's HIP_DROP of 1.45 implies for a 0.29 shoulder-to-hip drop).
LEG_PER_SHOULDER_WIDTH = 2.45
CANVAS = (1080, 1700)

# (label, fit mode, left leg, right leg); each leg is (thigh angle, shin angle, thigh length):
# angles in degrees out from straight down, and the length shortens a thigh turned towards the
# camera, as in a squat.
STRAIGHT = (2, 2, 1.0)
POSES = [("rows, standing", "rows", STRAIGHT, STRAIGHT),
         ("rows, feet apart", "rows", (10, 10, 1.0), (10, 10, 1.0)),
         ("rows, left knee out", "rows", (35, -8, 1.0), STRAIGHT),
         ("rows, slight squat", "rows", (30, -12, 0.8), (30, -12, 0.8)),
         ("rows, step", "rows", (18, 4, 1.0), (-4, 6, 1.0)),
         ("hips, left knee out", "hips", (35, -8, 1.0), STRAIGHT)]


def pt(p):
    return np.array([p["x"], p["y"]], float)


def leg_weights(entry, points):
    """(n, 5) weights for pelvis, left thigh, left shin, right thigh, right shin (TrouserMesh)."""
    a = entry["anchors"]
    hip_y = (a["left_hip"]["y"] + a["right_hip"]["y"]) / 2
    waist_y = (entry["waist"]["left"]["y"] + entry["waist"]["right"]["y"]) / 2
    crotch = entry["crotch"]
    legs = shirt.smoothstep(hip_y, crotch["y"] + LEGS_BELOW_CROTCH * (crotch["y"] - waist_y), points[:, 1])
    band = SPLIT * (a["right_hip"]["x"] - a["left_hip"]["x"])
    left = shirt.smoothstep(crotch["x"] + band, crotch["x"] - band, points[:, 0])
    columns = [1 - legs]
    for s, name in enumerate(("left", "right")):
        side = legs * (left if s == 0 else 1 - left)
        knee_band = KNEE_BLEND * np.linalg.norm(pt(a[f"{name}_ankle"]) - pt(a[f"{name}_hip"]))
        knee_y = a[f"{name}_knee"]["y"]
        shin = shirt.smoothstep(knee_y - knee_band, knee_y + knee_band, points[:, 1])
        columns += [side * (1 - shin), side * shin]
    return np.stack(columns, axis=1)


def trouser_mesh(entry, texture):
    """TrouserMesh's per-row data, and measure()'s widths from the photo."""
    a = entry["anchors"]
    w, h = texture.size
    cols, rows, src = shirt.grid(w, h)
    ys = np.arange(rows + 1) * h / rows
    sides = ("left", "right")
    hips = [pt(a[f"{s}_hip"]) for s in sides]
    knees = [pt(a[f"{s}_knee"]) for s in sides]
    ankles = [pt(a[f"{s}_ankle"]) for s in sides]
    centre = (hips[0][0] + hips[1][0]) / 2
    axis = np.array([np.interp(ys, [hips[s][1], knees[s][1], ankles[s][1]],
                               [hips[s][0], knees[s][0], ankles[s][0]]) for s in (0, 1)])
    leg_t = np.array([np.where(ys < knees[s][1], (ys - hips[s][1]) / (knees[s][1] - hips[s][1]),
                               1 + (ys - knees[s][1]) / (ankles[s][1] - knees[s][1])) for s in (0, 1)])
    blend = np.array([shirt.smoothstep(hips[s][1], knees[s][1], ys) for s in (0, 1)])
    hip_row = min(max(int(math.floor((hips[0][1] + hips[1][1]) / 2 / h * rows + 0.5)), 0), rows)

    alpha = np.asarray(texture.getchannel("A")) >= 128
    crotch_y = entry["crotch"]["y"]
    half = np.zeros((2, rows + 1))
    outer = np.zeros((2, rows + 1))
    inner = np.zeros((2, rows + 1))
    first = [-1, -1]
    for r in range(rows + 1):
        y = min(r * h // rows, h - 1)
        xs = np.nonzero(alpha[y])[0]
        if not len(xs):
            continue
        lo, hi = int(xs[0]), int(xs[-1])
        half[0][r] = centre - lo
        half[1][r] = hi + 1 - centre
        if y <= crotch_y:
            continue
        x = lo
        while x + 1 < w and alpha[y, x + 1] and x + 1 < centre:
            x += 1
        outer[0][r] = max(0.0, axis[0][r] - lo)
        inner[0][r] = max(0.0, x + 1 - axis[0][r])
        x = hi
        while x - 1 >= 0 and alpha[y, x - 1] and x - 1 > centre:
            x -= 1
        outer[1][r] = max(0.0, hi + 1 - axis[1][r])
        inner[1][r] = max(0.0, axis[1][r] - x)
        for s in (0, 1):
            if first[s] < 0 and outer[s][r] > 0:
                first[s] = r
    first = [f if f >= 0 else rows for f in first]
    # Going down, a row more than HEM_DROP narrower than the one above takes its width: a hem's
    # rounded corner or the rows past it, not the cut (TrouserMesh.measure).
    for widths in (half[0], half[1], outer[0], outer[1], inner[0], inner[1]):
        for r in range(1, rows + 1):
            if widths[r] < (1 - shirt.HEM_DROP) * widths[r - 1]:
                widths[r] = widths[r - 1]
    return {"cols": cols, "rows": rows, "src": src, "centre": centre, "axis": axis, "t": leg_t,
            "blend": blend, "hip_row": hip_row, "half": half, "outer": outer, "inner": inner,
            "first": first}


def widen(m, outer, inner):
    """TrouserMesh.widen: the grid widened, side by side, about each row's pivot."""
    src, cols, centre = m["src"], m["cols"], m["centre"]
    r = np.repeat(np.arange(m["rows"] + 1), cols + 1)
    x = src[:, 0]
    side = (x >= centre).astype(int)
    pivot = centre + m["blend"][side, r] * (m["axis"][side, r] - centre)
    d = x - pivot
    outward = np.where(side == 0, d <= 0, d >= 0)
    moved = x + np.where(outward, outer[side, r], inner[side, r]) * d
    out = src.copy()
    out[:, 0] = np.where(side == 0, np.minimum(moved, centre), np.maximum(moved, centre))
    return out


def body(left, right):
    """preview_mesh's stand-in body, arms at rest, with each leg posed as POSES describes."""
    view = shirt.body(8, 8)
    half = LEG_PER_SHOULDER_WIDTH * shirt.SHOULDER_WIDTH / 2
    for side, sign, (thigh_deg, shin_deg, thigh_len) in (("left", -1, left), ("right", 1, right)):
        t, s = math.radians(thigh_deg), math.radians(shin_deg)
        hx, hy = view[f"{side}_hip"]
        kx, ky = hx + sign * half * thigh_len * math.sin(t), hy + half * thigh_len * math.cos(t)
        view[f"{side}_knee"] = (kx, ky)
        view[f"{side}_ankle"] = (kx + sign * half * math.sin(s), ky + half * math.cos(s))
    for name in list(view):
        view[name] = (view[name][0], view[name][1] - 120)   # make room below for the feet
    return view


def down(a, b):
    """Unit vector square to the line a -> b, on the image-down side (OverlayView.computePelvis)."""
    d = np.array(b, float) - np.array(a, float)
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    return n if n[1] >= 0 else -n


def leg_path(hip, knee, ankle):
    """Hip to knee to ankle, along the leg."""
    hip, knee, ankle = (np.array(p, float) for p in (hip, knee, ankle))
    return np.linalg.norm(knee - hip) + np.linalg.norm(ankle - knee)


def pelvis(entry, view):
    """The pelvis bone and its handedness (OverlayView.computePelvis)."""
    a = entry["anchors"]
    g_l, g_r = pt(a["left_hip"]), pt(a["right_hip"])
    g_len = sum(leg_path(pt(a[f"{s}_hip"]), pt(a[f"{s}_knee"]), pt(a[f"{s}_ankle"]))
                for s in ("left", "right")) / 2
    v_l, v_r = np.array(view["left_hip"], float), np.array(view["right_hip"], float)
    v_len = sum(leg_path(view[f"{s}_hip"], view[f"{s}_knee"], view[f"{s}_ankle"])
                for s in ("left", "right")) / 2
    src = [tuple(g_l), tuple(g_r), tuple((g_l + g_r) / 2 + down(g_l, g_r) * g_len)]
    dst = [tuple(v_l), tuple(v_r), tuple((v_l + v_r) / 2 + down(v_l, v_r) * v_len)]
    m = affine_from_points(src, dst)
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    return m, (1.0 if det >= 0 else -1.0)


def bones(entry, view):
    """Pelvis, left thigh, left shin, right thigh, right shin (OverlayView.legBones)."""
    a = entry["anchors"]
    m, handedness = pelvis(entry, view)
    out = [m]
    for side in ("left", "right"):
        hip, knee, ankle = (tuple(pt(a[f"{side}_{j}"])) for j in ("hip", "knee", "ankle"))
        thigh = segment_matrix(hip, knee, view[f"{side}_hip"], view[f"{side}_knee"], m, handedness)
        thigh = thigh if thigh is not None else m
        shin = segment_matrix(knee, ankle, view[f"{side}_knee"], view[f"{side}_ankle"], m, handedness)
        out += [thigh, shin if shin is not None else thigh]
    return out


def ease(body_half, allowance, garment_half, scale):
    return float(np.clip(body_half * allowance / (garment_half * scale) - 1, MIN_EASE, MAX_EASE))


def fit(entry, m, view, mats, mode):
    """OverlayView.fitTrousers against the stand-in wearer: (outer, inner), each (2, rows + 1)."""
    rows = m["rows"]
    outer = np.zeros((2, rows + 1))
    inner = np.zeros((2, rows + 1))
    if mode == "off":
        return outer, inner
    allowance = FIT_ALLOWANCE.get(entry["fitStyle"], FIT_ALLOWANCE["regular"])
    p = mats[0]
    row_scale = math.hypot(p[0][0], p[1][0])
    spacing = math.dist(view["left_hip"], view["right_hip"])
    hip_half = STANDIN_HIP_HALF * spacing
    hip_ease = [ease(hip_half, allowance, m["half"][s][m["hip_row"]], row_scale) for s in (0, 1)]
    if mode == "hips":
        for s in (0, 1):
            outer[s].fill(hip_ease[s])
            inner[s].fill(hip_ease[s])
        return outer, inner

    a = entry["anchors"]
    for s, name in enumerate(("left", "right")):
        scales = []
        for k, (j0, j1) in enumerate((("hip", "knee"), ("knee", "ankle"))):
            d = pt(a[f"{name}_{j1}"]) - pt(a[f"{name}_{j0}"])
            ux, uy = -d[1] / np.linalg.norm(d), d[0] / np.linalg.norm(d)
            bone = mats[1 + 2 * s + k]
            scales.append(math.hypot(bone[0][0] * ux + bone[0][1] * uy, bone[1][0] * ux + bone[1][1] * uy))
        for r in range(rows + 1):
            g = m["half"][s][r]
            seat = hip_ease[s] if g < 1 else ease(hip_half, allowance, g, row_scale)
            lr = max(r, m["first"][s])
            body_leg = float(np.interp(m["t"][s][lr], *STANDIN_LEG_HALF)) * spacing
            leg_scale = scales[0 if m["t"][s][lr] < 1 else 1]
            o = m["outer"][s][lr]
            i = m["inner"][s][lr]
            leg_out = hip_ease[s] if o < 1 else ease(body_leg, allowance, o, leg_scale)
            leg_in = hip_ease[s] if i < 1 else ease(body_leg, allowance, i, leg_scale)
            f = m["blend"][s][r]
            outer[s][r] = (1 - f) * seat + f * leg_out
            inner[s][r] = (1 - f) * seat + f * leg_in
    return outer, inner


def render(entry, texture, m, mode, left, right, top=None):
    view = body(left, right)
    mats = bones(entry, view)
    outer, inner = fit(entry, m, view, mats, mode)
    dst = shirt.skin(widen(m, outer, inner), leg_weights(entry, m["src"]), mats)
    canvas = checkerboard(*CANVAS).convert("RGB")
    canvas = shirt.draw_mesh(canvas, texture, m["cols"], m["rows"], m["src"], dst)
    if top is not None:
        t_entry, t_texture, t_sleeves = top
        t_cols, t_rows, t_src = shirt.grid(*t_texture.size)
        t_dst = shirt.place(t_entry, t_texture, t_src, shirt.vertex_weights(t_entry, t_src), view)
        canvas = shirt.draw_mesh(canvas, t_texture, t_cols, t_rows, t_src, t_dst)
        canvas = shirt.draw_mesh(canvas, t_sleeves, t_cols, t_rows, t_src, t_dst)
    d = ImageDraw.Draw(canvas)
    bone = (40, 170, 180)
    if mode != "off":
        # The stand-in wearer's hips, as wide as the fit was told they are.
        spacing = math.dist(view["left_hip"], view["right_hip"])
        cx = (view["left_hip"][0] + view["right_hip"][0]) / 2
        hy = view["left_hip"][1]
        d.line([(cx - STANDIN_HIP_HALF * spacing, hy), (cx + STANDIN_HIP_HALF * spacing, hy)],
               fill=(230, 80, 60), width=3)
    for side in ("left", "right"):
        chain = [view[f"{side}_hip"], view[f"{side}_knee"], view[f"{side}_ankle"]]
        d.line(chain, fill=bone, width=4)
        for x, y in chain:
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(255, 255, 255), outline=bone, width=3)
    d.line([view["left_hip"], view["right_hip"]], fill=bone, width=4)
    return canvas


def weights_image(entry, texture):
    """The photo tinted by weight: red/orange = left thigh/shin, blue/cyan = right, grey = pelvis."""
    w, h = texture.size
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
    wt = leg_weights(entry, pts).reshape(h, w, 5)
    base = np.asarray(checkerboard(w, h).convert("RGB"), np.float32)
    tex = np.asarray(texture, np.float32)
    alpha = tex[..., 3:4] / 255
    grey = tex[..., :3].mean(axis=2, keepdims=True).repeat(3, axis=2)
    tint = (wt[..., 0:1] * grey + wt[..., 1:2] * np.array([235, 70, 60]) + wt[..., 2:3] * np.array([245, 170, 50])
            + wt[..., 3:4] * np.array([60, 110, 235]) + wt[..., 4:5] * np.array([60, 200, 225]))
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
        m = trouser_mesh(entry, texture)
        tiles = [("weights (thighs red/blue, shins orange/cyan)", weights_image(entry, texture))]
        for label, mode, l, r in POSES:
            tiles.append((label, render(entry, texture, m, mode, l, r)))
        if top is not None:
            tiles.append((f"rows, with {top[0]['name']}", render(entry, texture, m, "rows", STRAIGHT, STRAIGHT, top)))
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
