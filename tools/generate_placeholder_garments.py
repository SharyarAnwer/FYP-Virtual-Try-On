"""
Generates placeholder garment assets and the catalog that describes them.

These are stand-ins, not real clothing. DeepFashion supplies photographs of people
wearing clothes, not cut-out garments on transparent backgrounds, so real assets are
manual work. Synthetic shapes are also the better thing to develop the overlay against:
a clean geometric silhouette makes a misaligned transform obvious, whereas a photograph
hides a 5% error in the folds of the fabric.

The important property here is that the PNGs and their anchor points are generated from the
same geometry constants. Anchors recorded by hand drift away from the artwork the moment
either is edited, and a drifted anchor looks exactly like a broken transform.

Each garment is written as layers (schema v2, M3.5): a torso, a sleeve per arm, and for long
sleeves a forearm per arm. Every layer is a full-size canvas with the part drawn where it
sits when the arms hang at rest, so any layer can always fall back to its parent's transform
and still land in the right place. Each limb also gets an outline layer; see outline_layer for
why outlines are separate.

Run:  python tools/generate_placeholder_garments.py
"""

import json
import math
import os

from PIL import Image, ImageDraw, ImageFilter

CANVAS_W, CANVAS_H = 512, 640
CENTRE_X = CANVAS_W // 2

# --- shared body geometry -------------------------------------------------------------
# Every anchor below is read from these names, never re-typed as a literal.
SHOULDER_Y = 110
SHOULDER_HALF = 108           # left shoulder = CENTRE_X - SHOULDER_HALF
NECK_HALF = 42
NECK_DIP_Y = 146
HEM_Y = 580
HEM_HALF = 104
WAIST_Y = 300
WAIST_HALF = 98

# Where the garment is *drawn*.
LEFT_SHOULDER = (CENTRE_X - SHOULDER_HALF, SHOULDER_Y)
RIGHT_SHOULDER = (CENTRE_X + SHOULDER_HALF, SHOULDER_Y)
LEFT_HEM = (CENTRE_X - HEM_HALF, HEM_Y)
RIGHT_HEM = (CENTRE_X + HEM_HALF, HEM_Y)

# Where the garment is *anchored*, which is not the same thing.
#
# An anchor answers "which body landmark should this point of the artwork sit on", and
# MediaPipe's shoulder landmark is the joint centre — measurably inboard of where a shirt's
# shoulder seam falls on a real torso. Anchoring the seam to the joint therefore squeezes the
# whole garment inward, and the shirt renders narrower than the body wearing it.
SHOULDER_ANCHOR_INSET = 24

# The same joint-versus-surface gap applies vertically: the joint centre sits a few centimetres
# below the top of the shoulder, so anchoring the drawn shoulder line to the joint left the
# shirt hanging low. Measured on device, the top of the shoulder sat ~43 screen px above the
# joint.
#
# Converting that to artwork pixels needs the *vertical* scale of the transform, not the
# horizontal one. The affine stretches the two axes independently — on the test subject about
# 2.1 screen px per artwork px across the shoulders but only ~1.3 down the torso. A first attempt
# used the horizontal factor, chose 20, and closed only ~60% of the gap. Solving with the vertical
# scale, which the drop itself raises slightly by shortening the shoulder-to-hip span, gives 34.
#
# Because the two scales depend on the wearer's shoulder-to-torso proportions, this value is
# tuned to one tester. Broader or longer-bodied users will see the shoulder line sit a little
# differently.
SHOULDER_ANCHOR_DROP = 34
LEFT_SHOULDER_ANCHOR = (CENTRE_X - SHOULDER_HALF + SHOULDER_ANCHOR_INSET, SHOULDER_Y + SHOULDER_ANCHOR_DROP)
RIGHT_SHOULDER_ANCHOR = (CENTRE_X + SHOULDER_HALF - SHOULDER_ANCHOR_INSET, SHOULDER_Y + SHOULDER_ANCHOR_DROP)

# Likewise the hem is not the hip. A shirt falls past the hip joint, so the anchor sits above
# the drawn hem and the remaining fabric hangs below the landmark, as it would on a body.
HIP_ANCHOR_Y = 540
LEFT_HIP_ANCHOR = (CENTRE_X - HEM_HALF, HIP_ANCHOR_Y)
RIGHT_HIP_ANCHOR = (CENTRE_X + HEM_HALF, HIP_ANCHOR_Y)

# Arms at rest, in the torso's own artwork units. These only define where each sleeve sits when
# nothing is moving and which way its axis runs; once elbows and wrists are tracked, the segment
# transform stretches each part to the wearer's real limb length. Values come from the first M3
# test shot, mapped back into artwork space: the upper arm hangs ~10 degrees out from vertical.
UPPER_ARM_LEN = 200
UPPER_ARM_ANGLE = 10
FOREARM_LEN = 168
FOREARM_ANGLE = 12


def _limb_end(start, length, degrees_out, side):
    """End of a limb hanging `degrees_out` from vertical, away from the body on `side` (-1 left)."""
    rad = math.radians(degrees_out)
    return (round(start[0] + side * length * math.sin(rad)),
            round(start[1] + length * math.cos(rad)))


LEFT_ELBOW_ANCHOR = _limb_end(LEFT_SHOULDER_ANCHOR, UPPER_ARM_LEN, UPPER_ARM_ANGLE, -1)
LEFT_WRIST_ANCHOR = _limb_end(LEFT_ELBOW_ANCHOR, FOREARM_LEN, FOREARM_ANGLE, -1)

# Sleeve widths in artwork px. The sleeve's round cap at the shoulder has radius half the top
# width, centred on the joint, so the joint-to-shoulder-line distance (34) must not be much less
# than it or the cap bulges above the shoulder line. 70 wide leaves a ~3 px rise, invisible.
SLEEVE_TOP_W = 70
SHORT_CUFF_W = 66
SHORT_SLEEVE_FRACTION = 0.55      # a tee sleeve reaches about halfway down the upper arm
ELBOW_W = 60
FOREARM_CUFF_W = 50
FOREARM_FRACTION = 0.92           # a long sleeve stops just short of the wrist

# Where the arm meets the body, as a fraction of the way from shoulder joint to elbow. A long
# sleeve's underarm seam is drawn from here down to the cuff (see seam_layer) and never above it,
# because above the armpit the sleeve's inner edge lies across the chest.
ARMPIT_FRACTION = 0.35

# The torso's square shoulder corner sits 41.6 px from the shoulder joint, outside the sleeve's
# 35 px round cap, so it poked out as a small step whenever an arm was raised. The corner is cut
# off with a chamfer between two points that both lie inside the cap: one on the shoulder line
# directly above the joint, one on the side seam just below it. The visible shoulder outline then
# comes from the sleeve cap, which is round like a real shoulder.
def _on_shoulder_line(x):
    """y of the torso's top edge at x, on the straight run from neck to shoulder corner."""
    (x0, y0), (x1, y1) = (CENTRE_X - NECK_HALF, SHOULDER_Y + 6), LEFT_SHOULDER
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


CHAMFER_TOP = (LEFT_SHOULDER_ANCHOR[0], round(_on_shoulder_line(LEFT_SHOULDER_ANCHOR[0]), 1))
CHAMFER_SIDE = (LEFT_SHOULDER[0], LEFT_SHOULDER_ANCHOR[1] + 16)
for _pt in (CHAMFER_TOP, CHAMFER_SIDE):
    assert math.dist(_pt, LEFT_SHOULDER_ANCHOR) < SLEEVE_TOP_W / 2, "chamfer must stay inside the cap"

OUT_DIR = os.path.join("app", "src", "main", "assets", "garments")


def mirror(point):
    x, y = point
    return (2 * CENTRE_X - x, y)


def body_polygon():
    """Torso outline, clockwise from the left of the neck, with chamfered shoulders."""
    return [
        (CENTRE_X - NECK_HALF, SHOULDER_Y + 6),          # 0 neck, left
        CHAMFER_TOP,                                     # 1 chamfer, left
        CHAMFER_SIDE,                                    # 2
        (CENTRE_X - WAIST_HALF - 10, WAIST_Y),           # 3
        LEFT_HEM,                                        # 4
        RIGHT_HEM,                                       # 5
        (CENTRE_X + WAIST_HALF + 10, WAIST_Y),           # 6
        mirror(CHAMFER_SIDE),                            # 7 chamfer, right
        mirror(CHAMFER_TOP),                             # 8
        (CENTRE_X + NECK_HALF, SHOULDER_Y + 6),          # 9 neck, right
        (CENTRE_X + NECK_HALF - 14, NECK_DIP_Y - 14),    # 10 neckline scoop
        (CENTRE_X, NECK_DIP_Y),                          # 11
        (CENTRE_X - NECK_HALF + 14, NECK_DIP_Y - 14),    # 12
    ]


# Outline runs, deliberately skipping the two chamfer edges (1-2 and 7-8). Those edges are
# always under a sleeve cap, so they are internal joins rather than garment edges; if a pose ever
# exposes one by a pixel, a same-coloured fill disappears where a dark stroke would not.
BODY_OUTLINE_RUNS = [[8, 9, 10, 11, 12, 0, 1], [2, 3, 4, 5, 6, 7]]


def _band(start, end, w_start, w_end, fraction=1.0):
    """Quad around the axis start->end, `fraction` of the way along, tapering in width."""
    ux, uy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(ux, uy)
    ux, uy = ux / length, uy / length
    nx, ny = -uy, ux
    cx, cy = start[0] + ux * length * fraction, start[1] + uy * length * fraction
    a0 = (start[0] + nx * w_start / 2, start[1] + ny * w_start / 2)
    b0 = (start[0] - nx * w_start / 2, start[1] - ny * w_start / 2)
    a1 = (cx + nx * w_end / 2, cy + ny * w_end / 2)
    b1 = (cx - nx * w_end / 2, cy - ny * w_end / 2)
    return a0, b0, b1, a1


def _circle(draw, centre, radius, fill):
    x, y = centre
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill)


def draw_limb(fill_rgb, start, end, w_start, w_end, fraction, cap_start, cap_end):
    """
    One limb layer: a tapered band along start->end plus optional round joint caps, fill only.

    The caps exist so that rotating the part about its joint cannot expose a corner or open a
    gap. The limb's outline is a separate layer, see outline_layer.
    """
    image = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    fill = fill_rgb + (255,)
    draw.polygon(list(_band(start, end, w_start, w_end, fraction)), fill=fill)
    if cap_start:
        _circle(draw, start, w_start / 2, fill)
    if cap_end:
        _circle(draw, end, w_end / 2, fill)
    return image


# How far a limb's outline extends beyond its fill, in artwork px, matching the torso's stroke.
OUTLINE_PX = 3


def outline_layer(fill_layer, edge_rgb, thickness=OUTLINE_PX):
    """
    The limb's silhouette grown by `thickness` px, in the edge colour.

    The app draws every limb outline first, then the torso, then every limb fill. So wherever a
    limb overlaps the torso or another limb, its outline ends up covered, and wherever the limb
    forms the edge of the garment, a band `thickness` px wide survives around it. Drawing edges
    directly on the fill layer instead put a line across the chest wherever a sleeve overlapped
    the torso, which at rest is most of the sleeve. Growing the whole silhouette also outlines
    the round shoulder cap where it forms the shoulder's edge, and nowhere else.
    """
    alpha = fill_layer.getchannel("A").filter(ImageFilter.MaxFilter(2 * thickness + 1))
    outline = Image.new("RGBA", fill_layer.size, edge_rgb + (0,))
    outline.putalpha(alpha)
    return outline


def seam_layer(start, end, w_start, w_end, fraction, t_from, t_to, edge_rgb):
    """
    The limb's inner edge between two fractions of its length, as a line layer.

    The app draws seams last, over every fill. That is the point: at rest a long sleeve's inner
    edge lies over the shirt's body, so the outline pass hides it, and without a seam the sleeve
    fuses into the body from armpit to elbow like a batwing. A seam restores the arm-body boundary
    from the armpit down, which is where a real long sleeve shows one, without bringing back a
    line across the chest above the armpit.
    """
    ux, uy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(ux, uy)
    ux, uy = ux / length, uy / length
    nx, ny = -uy, ux

    def inner(t):
        # Same side as _band's b0/b1, which for a left-side limb faces the body.
        w = w_start + (w_end - w_start) * (t / fraction)
        return (start[0] + ux * length * t - nx * w / 2, start[1] + uy * length * t - ny * w / 2)

    image = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    ImageDraw.Draw(image).line([inner(t_from), inner(t_to)], fill=edge_rgb + (255,),
                               width=OUTLINE_PX)
    return image


def mirror_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def collar_polygon():
    """Folded collar for the polo, sitting over the neckline."""
    return [
        (CENTRE_X - NECK_HALF - 10, SHOULDER_Y),
        (CENTRE_X - 8, NECK_DIP_Y + 26),
        (CENTRE_X + 8, NECK_DIP_Y + 26),
        (CENTRE_X + NECK_HALF + 10, SHOULDER_Y),
        (CENTRE_X + NECK_HALF - 12, SHOULDER_Y - 14),
        (CENTRE_X, NECK_DIP_Y - 4),
        (CENTRE_X - NECK_HALF + 12, SHOULDER_Y - 14),
    ]


def shade(rgb, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in rgb)


def draw_torso(fill_rgb, collar=False):
    image = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    fill = fill_rgb + (255,)
    # A darker edge keeps the garment readable against skin of any tone, which matters
    # when the only way to judge a transform is looking at where the edge lands.
    edge = shade(fill_rgb, 0.65) + (255,)
    body = body_polygon()
    draw.polygon(body, fill=fill)
    for run in BODY_OUTLINE_RUNS:
        draw.line([body[i] for i in run], fill=edge, width=3, joint="curve")
    if collar:
        draw.polygon(collar_polygon(), fill=shade(fill_rgb, 0.88) + (255,), outline=edge, width=3)
    return image


def draw_parts(spec):
    """
    Returns [(part_name, fill, outline, seam, from, to, parent)] in back-to-front fill order.

    Right-side layers are mirror images of the left ones. Mirroring the whole canvas is only
    correct because every left anchor has its right counterpart at the mirrored x, which is
    how the anchors are defined above.
    """
    fill = spec["fill"]
    edge = shade(fill, 0.65)
    flip = lambda layer: None if layer is None else mirror_image(layer)

    def limb(start_pt, end_pt, w_start, w_end, fraction, cap_start, cap_end, seam=None):
        layer = draw_limb(fill, start_pt, end_pt, w_start, w_end, fraction, cap_start, cap_end)
        line = None
        if seam is not None:
            line = seam_layer(start_pt, end_pt, w_start, w_end, fraction, seam[0], seam[1], edge)
        return layer, outline_layer(layer, edge), line

    parts = [("torso", draw_torso(fill, spec["collar"]), None, None, None, None, None)]

    if spec["sleeve"] == "short":
        # No seam: a tee sleeve ends close to armpit level, so there is almost no underarm edge
        # to show, and at rest it looked right without one.
        sleeve = limb(LEFT_SHOULDER_ANCHOR, LEFT_ELBOW_ANCHOR, SLEEVE_TOP_W, SHORT_CUFF_W,
                      SHORT_SLEEVE_FRACTION, cap_start=True, cap_end=False)
        parts += [
            ("left_sleeve", *sleeve, "left_shoulder", "left_elbow", "torso"),
            ("right_sleeve", *map(flip, sleeve), "right_shoulder", "right_elbow", "torso"),
        ]
    else:
        upper = limb(LEFT_SHOULDER_ANCHOR, LEFT_ELBOW_ANCHOR, SLEEVE_TOP_W, ELBOW_W, 1.0,
                     cap_start=True, cap_end=True, seam=(ARMPIT_FRACTION, 1.0))
        fore = limb(LEFT_ELBOW_ANCHOR, LEFT_WRIST_ANCHOR, ELBOW_W, FOREARM_CUFF_W, FOREARM_FRACTION,
                    cap_start=True, cap_end=False, seam=(0.0, FOREARM_FRACTION))
        parts += [
            ("left_sleeve", *upper, "left_shoulder", "left_elbow", "torso"),
            ("right_sleeve", *map(flip, upper), "right_shoulder", "right_elbow", "torso"),
            ("left_forearm", *fore, "left_elbow", "left_wrist", "left_sleeve"),
            ("right_forearm", *map(flip, fore), "right_elbow", "right_wrist", "right_sleeve"),
        ]
    return parts


# --- catalog --------------------------------------------------------------------------
# Anchor keys are MediaPipe pose landmark names so the app can map them without a lookup table.
# The schema deliberately carries more than the overlay needs: M6 reads `sizes` and M7
# reads colour, gender and category. Defining it once here avoids rewriting every asset
# file twice later.

GARMENTS = [
    {
        "id": "tee_white",
        "name": "Basic White Tee",
        "fill": (238, 238, 236),
        "sleeve": "short",
        "collar": False,
        "colour": {"name": "white", "hex": "#EEEEEC"},
        "gender": "unisex",
        "fitStyle": "regular",
        "suitableBodyTypes": ["slim", "average", "broad"],
    },
    {
        "id": "shirt_navy_long",
        "name": "Navy Long Sleeve",
        "fill": (38, 58, 98),
        "sleeve": "long",
        "collar": False,
        "colour": {"name": "navy", "hex": "#263A62"},
        "gender": "male",
        "fitStyle": "slim",
        "suitableBodyTypes": ["slim", "average"],
    },
    {
        "id": "polo_red",
        "name": "Red Polo",
        "fill": (166, 54, 48),
        "sleeve": "short",
        "collar": True,
        "colour": {"name": "red", "hex": "#A63630"},
        "gender": "unisex",
        "fitStyle": "loose",
        "suitableBodyTypes": ["average", "broad"],
    },
]

# Body-proportion bands, not centimetres. A single camera with no depth sensor cannot
# recover real measurements, so M6 compares the ratio of shoulder width to torso height
# against these bands instead. Stated as a declared simplification, not a hidden one.
SIZE_BANDS = [
    {"label": "S", "shoulderToTorsoRatio": {"min": 0.58, "max": 0.70}, "heightRangeCm": {"min": 150, "max": 168}},
    {"label": "M", "shoulderToTorsoRatio": {"min": 0.68, "max": 0.80}, "heightRangeCm": {"min": 165, "max": 180}},
    {"label": "L", "shoulderToTorsoRatio": {"min": 0.78, "max": 0.92}, "heightRangeCm": {"min": 177, "max": 196}},
]


def anchor_json(point):
    return {"x": point[0], "y": point[1]}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Clear artwork from earlier runs, including schema-v1 single-image files, so a renamed
    # or removed layer can never linger in the APK.
    for stale in os.listdir(OUT_DIR):
        if stale.endswith(".png"):
            os.remove(os.path.join(OUT_DIR, stale))

    entries = []
    for spec in GARMENTS:
        anchors = {
            "left_shoulder": anchor_json(LEFT_SHOULDER_ANCHOR),
            "right_shoulder": anchor_json(RIGHT_SHOULDER_ANCHOR),
            "left_hip": anchor_json(LEFT_HIP_ANCHOR),
            "right_hip": anchor_json(RIGHT_HIP_ANCHOR),
            "left_elbow": anchor_json(LEFT_ELBOW_ANCHOR),
            "right_elbow": anchor_json(mirror(LEFT_ELBOW_ANCHOR)),
        }
        if spec["sleeve"] == "long":
            anchors["left_wrist"] = anchor_json(LEFT_WRIST_ANCHOR)
            anchors["right_wrist"] = anchor_json(mirror(LEFT_WRIST_ANCHOR))

        parts = []
        for name, image, outline, seam, start_lm, end_lm, parent in draw_parts(spec):
            filename = f"{spec['id']}_{name}.png"
            image.save(os.path.join(OUT_DIR, filename), "PNG")
            part = {"name": name, "kind": "torso" if parent is None else "segment",
                    "asset": filename}
            if outline is not None:
                outline_name = f"{spec['id']}_{name}_outline.png"
                outline.save(os.path.join(OUT_DIR, outline_name), "PNG")
                part["outline"] = outline_name
            if seam is not None:
                seam_name = f"{spec['id']}_{name}_seam.png"
                seam.save(os.path.join(OUT_DIR, seam_name), "PNG")
                part["seam"] = seam_name
            if parent is not None:
                part.update({"from": start_lm, "to": end_lm, "parent": parent})
            parts.append(part)

        entries.append({
            "id": spec["id"],
            "name": spec["name"],
            "category": "upper_body",
            "imageWidth": CANVAS_W,
            "imageHeight": CANVAS_H,
            "anchors": anchors,
            "parts": parts,
            "colour": spec["colour"],
            "gender": spec["gender"],
            "fitStyle": spec["fitStyle"],
            "suitableBodyTypes": spec["suitableBodyTypes"],
            "sizes": SIZE_BANDS,
            "placeholder": True,
        })

    catalog = {
        "schemaVersion": 2,
        "note": (
            "Placeholder artwork generated by tools/generate_placeholder_garments.py. "
            "Each garment is a set of full-canvas layers listed in back-to-front draw order; "
            "a torso layer follows shoulders and hips, and each segment layer follows the two "
            "landmarks named in 'from' and 'to', falling back to its parent's transform when "
            "those are not visible. A limb's 'outline' layer is drawn beneath the torso and all "
            "fills, so it only shows where the limb is the garment's outer edge; a 'seam' layer is "
            "drawn over everything. Replace the PNGs "
            "with real cut-outs and update anchors to match; every other field stays as-is."
        ),
        "garments": entries,
    }

    with open(os.path.join(OUT_DIR, "catalog.json"), "w", encoding="utf-8") as handle:
        json.dump(catalog, handle, indent=2)
        handle.write(chr(10))

    print(f"wrote {len(entries)} garments + catalog.json to {OUT_DIR}")
    for entry in entries:
        names = ", ".join(part["name"] for part in entry["parts"])
        print(f"  {entry['id']:<18} {len(entry['parts'])} layers: {names}")


if __name__ == "__main__":
    main()
