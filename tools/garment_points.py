"""
Finds a garment photo's key points and writes its catalog entry.

    python tools/garment_points.py --id polo_pink --name "Pink Polo" [--gender unisex]
                                   [--fit regular] [--dir app/src/main/assets/garments/real]

Reads <id>_front.png (and <id>_back.png if present) produced by prepare_garment.py, finds the
shoulder tips, armpits, cuffs and hem from the silhouette, derives the body anchors from them,
and adds or replaces the garment in <dir>/catalog.json. It also writes
tools/garment_points_<id>.png showing every point, which is the thing to check: detection is
automatic, and a point in the wrong place looks exactly like a broken warp on the phone.

Garment points versus body anchors
----------------------------------
Shoulder tips, armpits and cuffs are *garment* features, found in the photo. The anchors are
where the *body's* landmarks should land on the garment, and a body joint is never on the
garment's outline: the shoulder joint sits inboard of the seam and below the top of the
shoulder; the hip joint sits well above the hem. The anchors are therefore derived from the
garment points using body proportions, and those proportions were measured on the test device
rather than taken from anthropometric tables (see the constants below).
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

# --- body proportions, measured on device ------------------------------------------------
# Shoulder joint relative to the garment's shoulder tip, as fractions of the tip-to-tip span.
# Tuned in M3 on the placeholder shirts: the joint sits ~11% of the span inboard of the seam
# and ~10% of it below the top of the shoulder.
SHOULDER_INSET = 0.11
SHOULDER_DROP = 0.10
# Hip landmarks below the shoulder landmarks, per unit of joint-to-joint shoulder width.
# M3 on device: 515 screen px shoulder-to-hip against 353 px shoulder width = 1.46.
HIP_DROP_PER_SHOULDER_WIDTH = 1.45
# Elbow below the shoulder joint along the upper arm, per unit of shoulder width. M3.5 pose-3
# shots (arm at shoulder height) gave 0.75 and 0.91; 0.83 is their mean and matches anatomy.
UPPER_ARM_PER_SHOULDER_WIDTH = 0.83

SIZE_BANDS = [
    {"label": "S", "shoulderToTorsoRatio": {"min": 0.58, "max": 0.70}, "heightRangeCm": {"min": 150, "max": 168}},
    {"label": "M", "shoulderToTorsoRatio": {"min": 0.68, "max": 0.80}, "heightRangeCm": {"min": 165, "max": 180}},
    {"label": "L", "shoulderToTorsoRatio": {"min": 0.78, "max": 0.92}, "heightRangeCm": {"min": 177, "max": 196}},
]


def silhouette(path):
    return np.asarray(Image.open(path).getchannel("A")) > 128


def profiles(mask):
    """Leftmost and rightmost garment column per row (-1 where the row is empty)."""
    h, w = mask.shape
    any_row = mask.any(axis=1)
    left = np.where(any_row, mask.argmax(axis=1), -1)
    right = np.where(any_row, w - 1 - mask[:, ::-1].argmax(axis=1), -1)
    return left, right


def knee(xs, ys):
    """Point on a curve farthest from the chord joining its ends: where its bend is sharpest."""
    x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]
    dx, dy = x1 - x0, y1 - y0
    dist = np.abs(dy * (xs - x0) - dx * (ys - y0)) / math.hypot(dx, dy)
    i = int(np.argmax(dist))
    return float(xs[i]), float(ys[i])


def side_points(mask, outline, side):
    """
    Shoulder tip, armpit, cuff centre and hem corner for one side of the garment.

    `outline` is the left or right profile; `side` is -1 for image-left, +1 for image-right, so
    "outward" is always side * x.
    """
    h, w = mask.shape
    rows = np.where(outline >= 0)[0]
    top, bottom = rows[0], rows[-1]
    out = side * outline.astype(float)                   # larger = further outward

    # Cuff: the sleeve's outermost row. The body below it is much narrower.
    cuff_row = int(rows[np.argmax(out[rows])])

    # Body side: the outline in the lower torso, well clear of sleeves and hem curve.
    body_rows = rows[(rows > cuff_row + 0.06 * h) & (rows < bottom - 0.08 * h)]
    body_x = float(np.median(outline[body_rows]))

    # Shoulder tip: sharpest bend of the outline between the neck and the cuff. The upper part
    # of that curve is the shoulder slope, the lower part the sleeve's outer edge.
    neck_row = int(top + 0.06 * h)
    seg = np.arange(neck_row, cuff_row + 1)
    sx, sy = knee(outline[seg].astype(float), seg.astype(float))

    # Armpit: the apex of the notch between sleeve and body, found by walking up a column just
    # outside the body side until the sleeve closes over it.
    probe_x = int(round(body_x + side * 3))             # 3 px outward of the body side
    y = int(body_rows[0])
    while y > cuff_row - 0.12 * h and not mask[y, probe_x]:
        y -= 1
    armpit = (body_x, float(y))

    # Cuff centre: midpoint of the cuff hem's two ends. The outer end is the cuff row's outline
    # point; the inner end is the lowest point of the sleeve's bottom edge, searched only in
    # columns outside the body and rows above it so neither the body nor the armpit apex -
    # which sits on the same probe column - can be mistaken for it.
    outer = (float(outline[cuff_row]), float(cuff_row))
    lo, hi = sorted((probe_x, int(outline[cuff_row])))
    above_body = mask[:int(body_rows[0]), :]
    lowest = [(x, int(np.where(above_body[:, x])[0].max()))
              for x in range(lo, hi + 1) if above_body[:, x].any() and x != probe_x]
    ix, iy = max(lowest, key=lambda p: p[1])
    inner = (float(ix), float(iy))
    sleeve_cols = [x for x, _ in lowest]
    cuff = ((outer[0] + inner[0]) / 2, (outer[1] + inner[1]) / 2)

    hem_row = int(bottom - 0.015 * h)
    hem = (float(outline[hem_row]), float(hem_row))

    # Below the armpit the sleeve/torso boundary must pass between the cuff, hanging outward,
    # and the body side, running down to the hem. Its direction bisects the two: armpit-to-
    # cuff-inner-corner and armpit-to-hem-corner. A boundary that simply continued the shoulder-
    # to-armpit line ran nearly straight down and cut into any body that flares towards the hem.
    def unit(p, q):
        v = np.array(q, float) - np.array(p, float)
        return v / np.linalg.norm(v)
    split = unit(armpit, inner) + unit(armpit, hem)
    split /= np.linalg.norm(split)
    below = (armpit[0] + 100 * split[0], armpit[1] + 100 * split[1])

    return {"shoulder_tip": (sx, sy), "armpit": armpit, "below": below, "cuff": cuff,
            "cuff_outer": outer, "cuff_inner": inner, "hem": hem, "sleeve_cols": len(sleeve_cols)}


def derive_anchors(left, right):
    span = right["shoulder_tip"][0] - left["shoulder_tip"][0]
    anchors = {}
    for name, pts, side in (("left", left, -1), ("right", right, 1)):
        tx, ty = pts["shoulder_tip"]
        anchors[f"{name}_shoulder"] = (tx - side * SHOULDER_INSET * span, ty + SHOULDER_DROP * span)
    joint_width = anchors["right_shoulder"][0] - anchors["left_shoulder"][0]

    for name, pts in (("left", left), ("right", right)):
        jx, jy = anchors[f"{name}_shoulder"]
        cx, cy = pts["cuff"]
        length = math.hypot(cx - jx, cy - jy)
        reach = UPPER_ARM_PER_SHOULDER_WIDTH * joint_width
        anchors[f"{name}_elbow"] = (jx + (cx - jx) / length * reach, jy + (cy - jy) / length * reach)
        # Hip x only matters through the hip midpoint; the hem corners keep it on the centreline.
        hip_y = jy + HIP_DROP_PER_SHOULDER_WIDTH * joint_width
        anchors[f"{name}_hip"] = (pts["hem"][0], hip_y)
    return anchors, joint_width


def dominant_colour(path):
    rgba = np.asarray(Image.open(path).convert("RGBA")).reshape(-1, 4)
    solid = rgba[rgba[:, 3] == 255][:, :3]
    r, g, b = (int(v) for v in np.median(solid, axis=0))
    return f"#{r:02X}{g:02X}{b:02X}"


def preview(path, left, right, anchors, out_path):
    img = Image.open(path).convert("RGBA")
    board = Image.new("RGBA", img.size, (238, 241, 243, 255))
    board.alpha_composite(img)
    d = ImageDraw.Draw(board)

    def mark(p, colour, label, r=7):
        x, y = p
        d.ellipse([x - r, y - r, x + r, y + r], outline=colour, width=3)
        d.line([x - r - 5, y, x + r + 5, y], fill=colour, width=2)
        d.line([x, y - r - 5, x, y + r + 5], fill=colour, width=2)
        d.text((x + r + 6, y - 7), label, fill=(20, 24, 28, 255))

    garment = (255, 170, 0, 255)
    body = (0, 150, 190, 255)
    for pts in (left, right):
        d.line([pts["shoulder_tip"], pts["armpit"], pts["below"]], fill=garment, width=2)
        for key in ("shoulder_tip", "armpit", "cuff", "hem"):
            mark(pts[key], garment, key.replace("_", " "))
    for name, p in anchors.items():
        mark(p, body, name, r=9)
    d.text((10, 8), "orange = garment points found in the photo;  blue = body anchors derived from them",
           fill=(20, 24, 28, 255))
    board.convert("RGB").save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gender", default="unisex")
    ap.add_argument("--fit", default="regular")
    ap.add_argument("--body-types", default="slim,average,broad")
    ap.add_argument("--neck", default=None,
                    help='neck opening polygon on the front photo, "x,y x,y ..." in canvas px. '
                         "Kept from the existing entry when omitted.")
    ap.add_argument("--dir", default=os.path.join("app", "src", "main", "assets", "garments", "real"))
    args = ap.parse_args()

    front = os.path.join(args.dir, f"{args.id}_front.png")
    back = os.path.join(args.dir, f"{args.id}_back.png")
    mask = silhouette(front)
    h, w = mask.shape
    left_outline, right_outline = profiles(mask)
    left = side_points(mask, left_outline, -1)
    right = side_points(mask, right_outline, 1)
    anchors, joint_width = derive_anchors(left, right)

    r2 = lambda p: {"x": round(p[0], 1), "y": round(p[1], 1)}
    entry = {
        "id": args.id,
        "name": args.name,
        "category": "upper_body",
        "render": "mesh",
        "imageWidth": w,
        "imageHeight": h,
        "front": os.path.basename(front),
        "back": os.path.basename(back) if os.path.exists(back) else None,
        "anchors": {k: r2(v) for k, v in anchors.items()},
        "armholes": {
            side: {"shoulder": r2(pts["shoulder_tip"]), "armpit": r2(pts["armpit"]),
                   "below": r2(pts["below"])}
            for side, pts in (("left", left), ("right", right))
        },
        "colour": {"name": args.name.split()[0].lower(), "hex": dominant_colour(front)},
        "gender": args.gender,
        "fitStyle": args.fit,
        "suitableBodyTypes": args.body_types.split(","),
        "sizes": SIZE_BANDS,
        "placeholder": False,
    }

    catalog_path = os.path.join(args.dir, "catalog.json")
    catalog = {"schemaVersion": 3, "garments": []}
    if os.path.exists(catalog_path):
        with open(catalog_path, encoding="utf-8") as handle:
            catalog = json.load(handle)
    catalog["schemaVersion"] = 3
    catalog["note"] = ("Real garments, drawn by mesh warping. Written by tools/garment_points.py; "
                       "anchors are derived from detected garment points, so re-run it after "
                       "replacing a photo rather than editing anchors by hand.")
    # The neck opening is marked by hand, not detected, so a re-run must not lose it.
    previous = next((g for g in catalog["garments"] if g["id"] == args.id), {})
    if args.neck:
        entry["neckOpening"] = [{"x": float(x), "y": float(y)}
                                for x, y in (pair.split(",") for pair in args.neck.split())]
    elif previous.get("neckOpening"):
        entry["neckOpening"] = previous["neckOpening"]
    catalog["garments"] = [g for g in catalog["garments"] if g["id"] != args.id] + [entry]
    with open(catalog_path, "w", encoding="utf-8") as handle:
        json.dump(catalog, handle, indent=2)
        handle.write("\n")

    out = os.path.join("tools", f"garment_points_{args.id}.png")
    preview(front, left, right, anchors, out)
    print(f"{args.id}: {w}x{h}, shoulder-joint width {joint_width:.0f} px")
    for side, pts in (("left", left), ("right", right)):
        print(f"  {side:5s} tip {pts['shoulder_tip']}  armpit {pts['armpit']}  cuff {tuple(round(v) for v in pts['cuff'])}  hem {pts['hem']}")
    for k, v in anchors.items():
        print(f"  anchor {k:15s} ({v[0]:.0f}, {v[1]:.0f})")
    print(f"wrote {catalog_path} and {out}")


if __name__ == "__main__":
    main()
