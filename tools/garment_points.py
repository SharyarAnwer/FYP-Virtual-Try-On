"""
Finds a garment photo's key points and writes its catalog entry.

    python tools/garment_points.py --id polo_pink --name "Pink Polo" [--gender unisex]
                                   [--fit regular] [--dir app/src/main/assets/garments/real]
    python tools/garment_points.py --id khaki_chinos --name "Khaki Chinos" --category lower_body

Reads <id>_front.png (and <id>_back.png if present) produced by prepare_garment.py, finds the
garment's key points from the silhouette - for a shirt the shoulder tips, armpits, cuffs and hem;
for trousers the waistband corners, crotch and hem corners - derives the body anchors from them,
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

Placed by the neckline, not the shoulder seams
----------------------------------------------
Where the shoulder seams sit is a matter of style: on the shoulder for a fitted polo, halfway down
the upper arm for a dropped-shoulder shirt, nowhere at all on a raglan. Every shirt hangs from the
base of the neck, though, so the shoulder joints are placed from the neck points - where the
collar's outer edge meets the shoulder line, a sharp step in the photo's top edge - at a fixed
multiple of the neck's width. Placed by the seams, a dropped-shoulder shirt was drawn with its
shoulders 23% too wide: the sleeves stuck out like wings and the whole shirt sat low.

Everything outward of the shoulder joint then belongs to the sleeve: the armhole boundary starts
just outside the joint rather than at the seam, so a dropped shoulder's extra fabric bends down
with the arm, as it drapes on a person. If no neck step is found - a collarless neckline, say -
the tool falls back to placing the joints from the seams, and says so.
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

# --- body proportions, measured on device ------------------------------------------------
# Shoulder joints from the neck points. All three were measured on the pink polo, whose shoulder
# anchors were checked on the tester in M3 and whose seams sit on the shoulder, so seams and
# neckline agree there:
#   joint-to-joint width       = 2.18 x the neck width (collar edge to collar edge)
#   joint below the top edge   = 0.195 x joint width, measured at the joint's own x
#   armhole starts outward of the joint by 0.141 x joint width, on the top edge
NECK_TO_SHOULDER = 2.18
JOINT_BELOW_TOP = 0.195
ARMHOLE_OUTSIDE_JOINT = 0.141
# The collar step must be at least this tall, as a fraction of image height, to count.
MIN_NECK_STEP = 0.015

# Fallback, placing the joints from the seams: the joint relative to the garment's shoulder tip,
# as fractions of the tip-to-tip span. Tuned in M3 on the placeholder shirts: the joint sits ~11%
# of the span inboard of the seam and ~10% of it below the top of the shoulder.
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

# --- trousers (M4): anthropometric estimates, not yet measured on device -------------------
# Heights as fractions of standing height for an adult man: a mid-rise waistband at about 0.57,
# the hip joint 0.53, the crotch 0.465, the knee 0.285, the ankle bone 0.04, and a full-length
# trouser hem 0.025.
# Hip landmarks below the waistband's top edge, as a fraction of the rise (waistband to crotch):
# (0.57 - 0.53) / (0.57 - 0.465).
HIP_BELOW_WAIST = 0.38
# Knee along the leg, as a fraction of hip to ankle: (0.53 - 0.285) / (0.53 - 0.04).
KNEE_ALONG_LEG = 0.5
# Ankle above the hem, as a fraction of hip to hem: (0.04 - 0.025) / (0.53 - 0.025).
ANKLE_ABOVE_HEM = 0.03
# The hip's x is each leg's centre this far below the crotch, as a fraction of the rise: where
# the legs have parted but the thigh is still at its widest.
THIGH_BELOW_CROTCH = 0.1


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


def gap_at(row, centre, side):
    """
    In one row, walking outward from the centre: the body's outer edge and the next garment pixel
    beyond the background after it, or None if the row has no such gap.
    """
    w = len(row)
    x = centre
    if not row[x]:
        return None
    while 0 <= x + side < w and row[x + side]:
        x += side
    edge = x
    while 0 <= x + side < w and not row[x + side]:
        x += side
    if not (0 <= x + side < w):
        return None
    return edge, x + side


def armpit_apex(mask, side, start, stop, centre, persist=4):
    """Highest row in [start, stop) with a sleeve/body gap that persists `persist` rows; see side_points."""
    for y in range(max(start, 0), stop):
        gaps = [gap_at(mask[y + k], centre, side) for k in range(persist)]
        if all(g is not None for g in gaps):
            edge, sleeve = gaps[0]
            return (float(edge + sleeve) / 2, float(y))
    return None


def side_points(mask, outline, side):
    """
    Shoulder tip, armpit, cuff centre and hem corner for one side of the garment.

    `outline` is the left or right profile; `side` is -1 for image-left, +1 for image-right, so
    "outward" is always side * x.
    """
    h, w = mask.shape
    rows = np.where(outline >= 0)[0]
    top, bottom = rows[0], rows[-1]
    # Centre column, for walking outward: midway between the garment's leftmost and rightmost.
    cols = np.where(mask.any(axis=0))[0]
    centre = int((cols[0] + cols[-1]) / 2)
    out = side * outline.astype(float)                   # larger = further outward

    # Cuff: the sleeve's outermost row. The body below it is much narrower.
    cuff_row = int(rows[np.argmax(out[rows])])

    # Body side: the outline in the lower torso, well clear of sleeves and hem curve. Taken from
    # the garment's lower part rather than just below the cuff row: a sleeve laid out to the side
    # reaches well below its own outermost point, and those rows are still sleeve.
    span = bottom - top
    body_rows = rows[(rows > bottom - 0.45 * span) & (rows < bottom - 0.08 * h)]
    body_x = float(np.median(outline[body_rows]))

    # Shoulder tip: sharpest bend of the outline between the neck and the cuff. The upper part
    # of that curve is the shoulder slope, the lower part the sleeve's outer edge.
    neck_row = int(top + 0.06 * h)
    seg = np.arange(neck_row, cuff_row + 1)
    sx, sy = knee(outline[seg].astype(float), seg.astype(float))

    # Armpit: the apex of the notch between sleeve and body - the highest row, below the shoulder
    # tip, in which walking outward from the centre leaves the body, crosses background, and
    # meets the sleeve again. The gap must persist for a few rows so a stray pixel is not taken
    # for it. Climbing a single column just outside the lower body side found the armpit too low
    # whenever the body was wider at the chest than at the waist, as a worn shape usually is: the
    # column ran into the body's side below the real armpit.
    armpit = armpit_apex(mask, side, int(sy), int(bottom - 0.1 * h), centre)
    if armpit is None:
        raise SystemExit(f"no armpit found on the {'left' if side < 0 else 'right'}: is there a gap "
                         "between sleeve and body in the photo?")
    probe_x = int(round(armpit[0] + side * 3))

    # Cuff centre: midpoint of the cuff hem's two ends. The outer end is the cuff row's outline
    # point; the inner end is the lowest point of the sleeve's bottom edge. In each column
    # outward of the probe, the sleeve's bottom edge is where the column's first run of garment
    # - shoulder, then sleeve - ends: below it is the notch. Taking the end of that first run,
    # not the column's lowest garment pixel, keeps a body that flares out towards the hem from
    # being mistaken for sleeve.
    outer = (float(outline[cuff_row]), float(cuff_row))
    lo, hi = sorted((probe_x, int(outline[cuff_row])))

    def first_run_end(x):
        col = mask[:, x]
        start = int(np.argmax(col))
        return start + int(np.argmax(~col[start:])) - 1 if not col[start:].all() else h - 1

    lowest = [(x, first_run_end(x)) for x in range(lo, hi + 1)
              if side * (x - probe_x) > 0 and mask[:, x].any()]
    ix, iy = max(lowest, key=lambda p: p[1])
    inner = (float(ix), float(iy))
    sleeve_cols = [x for x, _ in lowest]
    cuff = ((outer[0] + inner[0]) / 2, (outer[1] + inner[1]) / 2)

    # Hem corner: the body side just above the hem, at the hem's height. Reading the outline on
    # the hem row itself put the corner partway along the hem whenever the garment sat slightly
    # tilted or its hem curved up at the sides, since that row then missed the side altogether.
    hem_row = int(bottom - 0.015 * h)
    side_rows = rows[(rows > bottom - 0.10 * h) & (rows < bottom - 0.03 * h)]
    hem = (float(np.median(outline[side_rows])), float(hem_row))

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


def top_edge(mask):
    """The garment's topmost row in each column (image height where the column is empty)."""
    h, _ = mask.shape
    return np.where(mask.any(axis=0), mask.argmax(axis=0), h).astype(float)


def neck_points(mask, left, right):
    """
    The two neck points, where the collar's outer edge meets the shoulder line: on each side of
    the centre, the steepest drop in the top edge. None if either side has no clear step.
    """
    h, w = mask.shape
    top = top_edge(mask)
    centre = (left["shoulder_tip"][0] + right["shoulder_tip"][0]) / 2
    k = max(3, round(0.01 * w))
    found = []
    for side in (-1, 1):
        best, at = 0.0, None
        x = int(round(centre))
        while 0 <= x + side * k < w and abs(x - centre) < 0.35 * w:
            step = top[x + side * k] - top[x]
            if step > best:
                best, at = step, x
            x += side
        if at is None or best < MIN_NECK_STEP * h:
            return None
        found.append((at + side * k / 2, float(top[at + side * k])))
    return found[0], found[1]


def derive_anchors(mask, left, right):
    """
    Body anchors, and where each armhole meets the shoulder line. Returns
    (anchors, joint width, {"left"/"right": armhole shoulder point}, neck points or None).
    """
    anchors = {}
    armhole_top = {}
    neck = neck_points(mask, left, right)
    if neck is not None:
        top = top_edge(mask)
        (lx, _), (rx, _) = neck
        centre = (lx + rx) / 2
        joint_width = NECK_TO_SHOULDER * (rx - lx)
        span = right["shoulder_tip"][0] - left["shoulder_tip"][0]
        if not (0 < joint_width < span):
            neck = None              # implausible: joints outside the shoulders; use the seams
    if neck is not None:
        for name, side in (("left", -1), ("right", 1)):
            jx = centre + side * joint_width / 2
            anchors[f"{name}_shoulder"] = (jx, top[int(round(jx))] + JOINT_BELOW_TOP * joint_width)
            ax = jx + side * ARMHOLE_OUTSIDE_JOINT * joint_width
            armhole_top[name] = (ax, float(top[int(round(ax))]))
    else:
        span = right["shoulder_tip"][0] - left["shoulder_tip"][0]
        for name, pts, side in (("left", left, -1), ("right", right, 1)):
            tx, ty = pts["shoulder_tip"]
            anchors[f"{name}_shoulder"] = (tx - side * SHOULDER_INSET * span, ty + SHOULDER_DROP * span)
            armhole_top[name] = pts["shoulder_tip"]
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
    return anchors, joint_width, armhole_top, neck


def extreme(mask, dx, dy, where=None):
    """The garment pixel farthest in the direction (dx, dy), among the pixels in `where` if given."""
    ys, xs = np.nonzero(mask if where is None else mask & where)
    i = int(np.argmax(dx * xs + dy * ys))
    return float(xs[i]), float(ys[i])


def background_runs(row):
    """(first, last) x of each run of background lying between the row's outermost garment pixels."""
    cols = np.where(row)[0]
    if len(cols) == 0:
        return []
    runs, start = [], None
    for x in range(int(cols[0]), int(cols[-1]) + 1):
        if not row[x] and start is None:
            start = x
        elif row[x] and start is not None:
            runs.append((start, x - 1))
            start = None
    return runs


def crotch_apex(mask, centre):
    """
    The top of the gap between the legs: the gap is followed up from just above the hems, row by
    row, to the last row in which it is still open. Followed rather than searched for, so a gap
    elsewhere - between a belt loop and the waistband, say - is never taken for it.
    """
    h, _ = mask.shape
    rows = np.where(mask.any(axis=1))[0]
    gap, apex = None, None
    for y in range(int(rows[-1] - 0.02 * h), int(rows[0]), -1):
        runs = background_runs(mask[y])
        if gap is not None:
            runs = [r for r in runs if r[0] <= gap[1] and r[1] >= gap[0]]
            if not runs:
                break
        if not runs:
            continue
        gap = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - centre))
        apex = ((gap[0] + gap[1]) / 2, float(y))
    if apex is None:
        raise SystemExit("no gap between the legs found: do the legs touch in the photo?")
    return apex


def leg_centre(mask, y, crotch_x, side):
    """
    The x midway across one leg in row `y`, below the crotch: from the leg's outer edge to its
    inner edge, the first background pixel walking inward from the outside.
    """
    row = mask[int(round(y))]
    cols = np.where(row)[0]
    cols = cols[cols < crotch_x] if side < 0 else cols[cols > crotch_x]
    outer = int(cols.min() if side < 0 else cols.max())
    x = outer
    while row[x - side] and side * (x - side - crotch_x) > 0:
        x -= side
    return (outer + x) / 2


def trouser_points(mask):
    """
    Waistband corners, crotch apex and each leg's hem corners, keyed by image side.

    Corners are the garment's extreme pixels on a 45-degree diagonal - top-outer for the
    waistband, bottom-outer and bottom-inner for each hem - which finds a corner whether the
    waistband sits level or slightly tilted, as it does in the khaki chinos' back view.
    """
    h, w = mask.shape
    xs = np.arange(w)[None, :]
    waist = {"left": extreme(mask, -1, -1), "right": extreme(mask, 1, -1)}
    crotch = crotch_apex(mask, (waist["left"][0] + waist["right"][0]) / 2)
    hems = {}
    for name, side in (("left", -1), ("right", 1)):
        leg = np.broadcast_to(side * (xs - crotch[0]) > 0, mask.shape)
        hems[name] = {"outer": extreme(mask, side, 1, leg), "inner": extreme(mask, -side, 1, leg)}
    return {"waist": waist, "crotch": crotch, "hems": hems}


def derive_leg_anchors(mask, points):
    """
    Hip, knee and ankle anchors for each leg, from the garment points and body proportions.

    Every anchor sits on its leg's centre line. Heights: the hip a fixed fraction of the rise
    below the waistband, the ankle just above the hem - trousers are bought to the wearer's leg
    length, so the hem is taken to fall at the ankle - and the knee a fixed fraction of the way
    between them. The hip's x is the leg's centre just below the crotch, the top of the thigh.
    """
    crotch_x, crotch_y = points["crotch"]
    waist_y = (points["waist"]["left"][1] + points["waist"]["right"][1]) / 2
    rise = crotch_y - waist_y
    hip_y = waist_y + HIP_BELOW_WAIST * rise
    thigh_y = crotch_y + THIGH_BELOW_CROTCH * rise
    anchors = {}
    for name, side in (("left", -1), ("right", 1)):
        hem = points["hems"][name]
        hem_y = (hem["outer"][1] + hem["inner"][1]) / 2
        ankle_y = hem_y - ANKLE_ABOVE_HEM * (hem_y - hip_y)
        knee_y = hip_y + KNEE_ALONG_LEG * (ankle_y - hip_y)
        anchors[f"{name}_hip"] = (leg_centre(mask, thigh_y, crotch_x, side), hip_y)
        anchors[f"{name}_knee"] = (leg_centre(mask, knee_y, crotch_x, side), knee_y)
        anchors[f"{name}_ankle"] = (leg_centre(mask, ankle_y, crotch_x, side), ankle_y)
    return anchors


def dominant_colour(path):
    rgba = np.asarray(Image.open(path).convert("RGBA")).reshape(-1, 4)
    solid = rgba[rgba[:, 3] == 255][:, :3]
    r, g, b = (int(v) for v in np.median(solid, axis=0))
    return f"#{r:02X}{g:02X}{b:02X}"


def preview(path, left, right, anchors, armhole_top, neck, out_path):
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
    for name, pts in (("left", left), ("right", right)):
        d.line([armhole_top[name], pts["armpit"], pts["below"]], fill=garment, width=2)
        for key in ("shoulder_tip", "armpit", "cuff", "hem"):
            mark(pts[key], garment, key.replace("_", " "))
    for p in neck or ():
        mark(p, garment, "neck")
    for name, p in anchors.items():
        mark(p, body, name, r=9)
    method = "neck points" if neck else "shoulder seams (no neck step found)"
    d.text((10, 8), "orange = garment points found in the photo;  blue = body anchors derived from "
           f"them;  shoulders placed from the {method}", fill=(20, 24, 28, 255))
    board.convert("RGB").save(out_path)


def preview_legs(path, points, anchors, out_path):
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
    for name in ("left", "right"):
        d.line([anchors[f"{name}_hip"], anchors[f"{name}_knee"], anchors[f"{name}_ankle"]],
               fill=body, width=2)
        mark(points["waist"][name], garment, "waist")
        mark(points["hems"][name]["outer"], garment, "hem")
        mark(points["hems"][name]["inner"], garment, "hem")
    mark(points["crotch"], garment, "crotch")
    for name, p in anchors.items():
        mark(p, body, name.split("_")[1], r=9)
    d.text((10, 8), "orange = garment points;  blue = body anchors", fill=(20, 24, 28, 255))
    board.convert("RGB").save(out_path)


def write_catalog(directory, entry):
    """Adds or replaces `entry` in <directory>/catalog.json. Returns the catalog's path."""
    catalog_path = os.path.join(directory, "catalog.json")
    catalog = {"schemaVersion": 3, "garments": []}
    if os.path.exists(catalog_path):
        with open(catalog_path, encoding="utf-8") as handle:
            catalog = json.load(handle)
    catalog["schemaVersion"] = 3
    catalog["note"] = ("Real garments, drawn by mesh warping. Written by tools/garment_points.py; "
                       "anchors are derived from detected garment points, so re-run it after "
                       "replacing a photo rather than editing anchors by hand.")
    catalog["garments"] = [g for g in catalog["garments"] if g["id"] != entry["id"]] + [entry]
    with open(catalog_path, "w", encoding="utf-8") as handle:
        json.dump(catalog, handle, indent=2)
        handle.write("\n")
    return catalog_path


def previous_entry(directory, garment_id):
    """The garment's current catalog entry, or {} if it has none yet."""
    catalog_path = os.path.join(directory, "catalog.json")
    if not os.path.exists(catalog_path):
        return {}
    with open(catalog_path, encoding="utf-8") as handle:
        return next((g for g in json.load(handle)["garments"] if g["id"] == garment_id), {})


def common_fields(args, front, back, mask, category):
    """The catalog fields every garment has, whatever its category."""
    h, w = mask.shape
    return {
        "id": args.id,
        "name": args.name,
        "category": category,
        "render": "mesh",
        "imageWidth": w,
        "imageHeight": h,
        "front": os.path.basename(front),
        "back": os.path.basename(back) if os.path.exists(back) else None,
    }


# What M7 reads about a garment, and its value when neither the command line nor the garment's
# existing entry gives one.
DESCRIPTIVE_DEFAULTS = {
    "gender": "unisex",
    "fitStyle": "regular",
    "suitableBodyTypes": ["slim", "average", "broad"],
    "occasion": None,
    "legShape": None,
}


def descriptive_fields(args, front, previous):
    """
    What M6 and M7 read: colour, gender, fit, body types, occasion, leg shape and sizes. Each comes
    from the command line if given there, else from the garment's existing entry - these are
    judged by hand, so a re-run to fix the anchors must not reset them - else a default. The
    colour's hex is measured from the photo every time; its family is judged by hand like the rest.
    """
    given = {
        "gender": args.gender,
        "fitStyle": args.fit,
        "suitableBodyTypes": args.body_types.split(",") if args.body_types else None,
        "occasion": args.occasion,
        "legShape": args.leg_shape,
    }
    fields = {}
    for key, default in DESCRIPTIVE_DEFAULTS.items():
        value = given[key] if given[key] is not None else previous.get(key, default)
        if value is not None:
            fields[key] = value
    colour = {"name": args.name.split()[0].lower(), "hex": dominant_colour(front)}
    family = args.colour_family or previous.get("colour", {}).get("family")
    if family:
        colour["family"] = family
    fields.update({"colour": colour, "sizes": SIZE_BANDS, "placeholder": False})
    return fields


def r2(p):
    return {"x": round(p[0], 1), "y": round(p[1], 1)}


def upper_body(args, front, back, mask):
    left_outline, right_outline = profiles(mask)
    left = side_points(mask, left_outline, -1)
    right = side_points(mask, right_outline, 1)
    anchors, joint_width, armhole_top, neck = derive_anchors(mask, left, right)

    entry = common_fields(args, front, back, mask, "upper_body")
    entry.update({
        "anchors": {k: r2(v) for k, v in anchors.items()},
        "armholes": {
            side: {"shoulder": r2(armhole_top[side]), "armpit": r2(pts["armpit"]),
                   "below": r2(pts["below"]),
                   # The cuff hem's two ends: the sleeve's own centre line and width come from them.
                   "cuffOuter": r2(pts["cuff_outer"]), "cuffInner": r2(pts["cuff_inner"])}
            for side, pts in (("left", left), ("right", right))
        },
    })
    previous = previous_entry(args.dir, args.id)
    entry.update(descriptive_fields(args, front, previous))
    # The neck opening is marked by hand, not detected, so a re-run must not lose it.
    if args.neck:
        entry["neckOpening"] = [{"x": float(x), "y": float(y)}
                                for x, y in (pair.split(",") for pair in args.neck.split())]
    elif previous.get("neckOpening"):
        entry["neckOpening"] = previous["neckOpening"]
    catalog_path = write_catalog(args.dir, entry)

    out = os.path.join("tools", f"garment_points_{args.id}.png")
    preview(front, left, right, anchors, armhole_top, neck, out)
    h, w = mask.shape
    print(f"{args.id}: {w}x{h}, shoulder-joint width {joint_width:.0f} px")
    if neck:
        print(f"  shoulders placed from the neck points {tuple(round(v) for v in neck[0])} "
              f"and {tuple(round(v) for v in neck[1])}")
    else:
        print("  no usable neck step - shoulders placed from the seams; check the preview")
    for side, pts in (("left", left), ("right", right)):
        print(f"  {side:5s} tip {pts['shoulder_tip']}  armpit {pts['armpit']}  cuff {tuple(round(v) for v in pts['cuff'])}  hem {pts['hem']}")
    for k, v in anchors.items():
        print(f"  anchor {k:15s} ({v[0]:.0f}, {v[1]:.0f})")
    print(f"wrote {catalog_path} and {out}")


def lower_body(args, front, back, mask):
    points = trouser_points(mask)
    anchors = derive_leg_anchors(mask, points)

    entry = common_fields(args, front, back, mask, "lower_body")
    entry.update({
        "anchors": {k: r2(v) for k, v in anchors.items()},
        "waist": {side: r2(points["waist"][side]) for side in ("left", "right")},
        "crotch": r2(points["crotch"]),
        # Each leg's hem: its outer and inner corners. The leg's own centre line and width at the
        # hem come from them, as a sleeve's do from its cuff.
        "legs": {side: {"hemOuter": r2(points["hems"][side]["outer"]),
                        "hemInner": r2(points["hems"][side]["inner"])}
                 for side in ("left", "right")},
    })
    entry.update(descriptive_fields(args, front, previous_entry(args.dir, args.id)))
    catalog_path = write_catalog(args.dir, entry)

    out = os.path.join("tools", f"garment_points_{args.id}.png")
    preview_legs(front, points, anchors, out)
    h, w = mask.shape
    rise = points["crotch"][1] - (points["waist"]["left"][1] + points["waist"]["right"][1]) / 2
    print(f"{args.id}: {w}x{h}, rise (waistband to crotch) {rise:.0f} px")
    for side in ("left", "right"):
        print(f"  {side:5s} waist {points['waist'][side]}  hem outer {points['hems'][side]['outer']}"
              f"  hem inner {points['hems'][side]['inner']}")
    print(f"  crotch {points['crotch']}")
    for k, v in anchors.items():
        print(f"  anchor {k:15s} ({v[0]:.0f}, {v[1]:.0f})")
    print(f"wrote {catalog_path} and {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--category", default="upper_body", choices=("upper_body", "lower_body"),
                    help="upper_body for shirts, lower_body for trousers")
    # Judged by hand, so kept from the garment's existing entry when omitted (see descriptive_fields).
    ap.add_argument("--gender", default=None, help="unisex, male or female")
    ap.add_argument("--fit", default=None, help="slim, regular or relaxed")
    ap.add_argument("--body-types", default=None, help='e.g. "slim,average"')
    ap.add_argument("--occasion", default=None, help="casual, smart-casual or formal")
    ap.add_argument("--leg-shape", default=None, help="trousers: tapered, straight or wide")
    ap.add_argument("--colour-family", default=None, help="neutral, warm or cool")
    ap.add_argument("--neck", default=None,
                    help='neck opening polygon on the front photo, "x,y x,y ..." in canvas px. '
                         "Kept from the existing entry when omitted. Shirts only.")
    ap.add_argument("--dir", default=os.path.join("app", "src", "main", "assets", "garments", "real"))
    args = ap.parse_args()

    front = os.path.join(args.dir, f"{args.id}_front.png")
    back = os.path.join(args.dir, f"{args.id}_back.png")
    mask = silhouette(front)
    if args.category == "lower_body":
        lower_body(args, front, back, mask)
    else:
        upper_body(args, front, back, mask)


if __name__ == "__main__":
    main()
