"""
Renders garment previews from the catalog, checking artwork and anchors before any device test.

Writes two images:

  tools/garment_preview.png       every garment with its layers composited at rest, anchors marked
  tools/garment_articulation.png  every garment with arms moved through a set of test poses

The first catches anchors that have drifted off the artwork: four numbers in a JSON file cannot
be eyeballed, so they are drawn onto it. The second exercises the articulation maths offline.
It implements the same segment transform as OverlayView.kt, step for step, so a gap opening at a
shoulder or elbow shows up here instead of on a phone. If the two implementations ever diverge,
this preview stops being evidence — change them together.

Run:  python tools/preview_garments.py
"""

import json
import math
import os

from PIL import Image, ImageDraw

CATALOG = os.path.join("app", "src", "main", "assets", "garments", "catalog.json")
OUT_REST = os.path.join("tools", "garment_preview.png")
OUT_POSES = os.path.join("tools", "garment_articulation.png")

CHECKER = 16
PAD = 28
LABEL_H = 34

ANCHOR_COLOURS = {
    "left_shoulder": (255, 92, 92),
    "right_shoulder": (92, 168, 255),
    "left_hip": (255, 196, 64),
    "right_hip": (120, 220, 140),
    "left_elbow": (214, 92, 214),
    "right_elbow": (92, 214, 214),
    "left_wrist": (160, 110, 60),
    "right_wrist": (60, 110, 160),
}

# Test poses: (upper-arm angle, forearm angle), degrees out from hanging straight down.
# 180 on the forearm means pointing straight up, i.e. the elbow bent a right angle.
POSES = [
    ("rest", 10, 12),
    ("arms 45°", 45, 45),
    ("arms horizontal", 90, 90),
    ("elbows bent up", 90, 180),
]
UPPER_ARM_LEN = 200     # matches the generator's rest pose, so length scale is 1 and only
FOREARM_LEN = 168       # rotation is under test
# The torso is stretched the way the tester's body stretches it: in M3 the transform measured
# ~2.1 screen px per artwork px across the shoulders but ~1.3 down the torso, a 1.6:1 ratio.
# Previewing at 1:1 hides every width problem that only appears on a real body.
POSE_TORSO_SCALE = (1.6, 1.0)
POSE_OFFSET = (120, 70)
POSE_CANVAS = (1080, 720)
POSE_SCALE = 0.46


def checkerboard(width, height):
    """Transparency needs a backdrop, or a white garment on white is invisible."""
    board = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(board)
    for y in range(0, height, CHECKER):
        for x in range(0, width, CHECKER):
            if (x // CHECKER + y // CHECKER) % 2:
                draw.rectangle([x, y, x + CHECKER - 1, y + CHECKER - 1], fill=(222, 226, 230, 255))
    return board


LAYER = {"fill": 0, "outline": 1, "seam": 2}


def load_layers(entry, base_dir):
    """[(fill, outline or None, seam or None)] per part, in catalog order."""
    def img(name):
        return Image.open(os.path.join(base_dir, name)).convert("RGBA") if name else None
    return [(img(part["asset"]), img(part.get("outline")), img(part.get("seam")))
            for part in entry["parts"]]


def draw_order(entry):
    """
    The app's four passes, as (part index, which layer): every limb outline, then the torso,
    then every limb fill, then every seam. Outlines go underneath so that anything overlapping
    them covers them, leaving an outline only where a limb is the garment's outer edge; seams go
    on top so the underarm edge of a long sleeve shows even where it lies over the body.
    """
    parts = entry["parts"]
    limbs = [i for i, part in enumerate(parts) if part["kind"] != "torso"]
    torso = [i for i, part in enumerate(parts) if part["kind"] == "torso"]
    return ([(i, "outline") for i in limbs] + [(i, "fill") for i in torso] +
            [(i, "fill") for i in limbs] + [(i, "seam") for i in limbs])


# --- affine maths, mirroring OverlayView.kt ------------------------------------------------

def affine_from_points(src, dst):
    """2x3 matrix mapping three src points onto three dst points (what setPolyToPoly does)."""
    (x0, y0), (x1, y1), (x2, y2) = src
    det = x0 * (y1 - y2) - y0 * (x1 - x2) + (x1 * y2 - x2 * y1)
    if abs(det) < 1e-9:
        return None
    inv = [
        [(y1 - y2) / det, (y2 - y0) / det, (y0 - y1) / det],
        [(x2 - x1) / det, (x0 - x2) / det, (x1 - x0) / det],
        [(x1 * y2 - x2 * y1) / det, (x2 * y0 - x0 * y2) / det, (x0 * y1 - x1 * y0) / det],
    ]
    rows = []
    for axis in (0, 1):
        d = [p[axis] for p in dst]
        rows.append([
            inv[0][0] * d[0] + inv[0][1] * d[1] + inv[0][2] * d[2],
            inv[1][0] * d[0] + inv[1][1] * d[1] + inv[1][2] * d[2],
            inv[2][0] * d[0] + inv[2][1] * d[1] + inv[2][2] * d[2],
        ])
    return rows  # [[a, b, c], [d, e, f]]  ->  x' = a x + b y + c


def torso_width_scale(torso, direction):
    """
    How many view px one artwork px of the torso spans along a view direction.

    The torso transform is anisotropic (wider than it is tall on most people), so there is no
    single "body scale". A sleeve's width is measured perpendicular to the arm, and that
    direction swings from horizontal (arm hanging) to vertical (arm raised), so the scale has to
    be read along it: k = 1 / |T^-1 . d|.
    """
    (a, b, _), (d, e, _) = torso
    det = a * e - b * d
    ix = (e * direction[0] - b * direction[1]) / det
    iy = (-d * direction[0] + a * direction[1]) / det
    return 1.0 / math.hypot(ix, iy)


def segment_matrix(art_a, art_b, view_a, view_b, torso, handedness):
    """
    Length follows the limb, width follows the body.

    Along the limb axis the part stretches to the wearer's actual segment length. Across it,
    the part scales with the torso's stretch in that same direction, so a sleeve stays as thick
    as the shirt it belongs to whether the arm hangs or is raised, and keeps that thickness
    when the arm points toward the camera and looks short. Handedness copies the torso's: if the
    torso transform mirrors the artwork, the limbs must mirror with it or a sleeve's outer edge
    would end up facing the body.
    """
    ax, ay = art_b[0] - art_a[0], art_b[1] - art_a[1]
    a_len = math.hypot(ax, ay)
    vx, vy = view_b[0] - view_a[0], view_b[1] - view_a[1]
    v_len = math.hypot(vx, vy)
    if a_len < 1 or v_len < 1:
        return None
    reach = 64.0
    view_dir = (handedness * -vy / v_len, handedness * vx / v_len)
    width_scale = torso_width_scale(torso, view_dir)
    art_perp = (-ay / a_len * reach, ax / a_len * reach)
    view_perp = (view_dir[0] * reach * width_scale, view_dir[1] * reach * width_scale)
    src = [art_a, art_b, (art_a[0] + art_perp[0], art_a[1] + art_perp[1])]
    dst = [view_a, view_b, (view_a[0] + view_perp[0], view_a[1] + view_perp[1])]
    return affine_from_points(src, dst)


def apply(matrix, point):
    (a, b, c), (d, e, f) = matrix
    return (a * point[0] + b * point[1] + c, d * point[0] + e * point[1] + f)


def invert(matrix):
    (a, b, c), (d, e, f) = matrix
    det = a * e - b * d
    return ((e / det), (-b / det), (b * f - c * e) / det,
            (-d / det), (a / det), (c * d - a * f) / det)


def warp_layer(layer, matrix, size):
    return layer.transform(size, Image.AFFINE, invert(matrix), resample=Image.BILINEAR)


# --- sheets --------------------------------------------------------------------------------

def rest_sheet(catalog, base_dir):
    tiles = []
    for entry in catalog["garments"]:
        size = (entry["imageWidth"], entry["imageHeight"])
        tile = checkerboard(*size)
        layers = load_layers(entry, base_dir)
        for i, which in draw_order(entry):
            layer = layers[i][LAYER[which]]
            if layer is not None:
                tile.alpha_composite(layer)

        draw = ImageDraw.Draw(tile)
        for name, point in entry["anchors"].items():
            x, y = point["x"], point["y"]
            colour = ANCHOR_COLOURS.get(name, (255, 0, 255)) + (255,)
            draw.line([x - 12, y, x + 12, y], fill=colour, width=3)
            draw.line([x, y - 12, x, y + 12], fill=colour, width=3)
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=colour, width=3)
            draw.text((x + 15, y - 7), name, fill=(20, 24, 28, 255))
            if not (0 <= x < size[0] and 0 <= y < size[1]):
                print(f"  WARNING {entry['id']}: anchor {name} at ({x},{y}) is outside the image")

        label = f"{entry['id']}  ({len(entry['parts'])} layers)"
        draw.text((10, 8), label, fill=(20, 24, 28, 255))
        tiles.append(tile)

    width = sum(t.width for t in tiles) + PAD * (len(tiles) + 1)
    height = max(t.height for t in tiles) + PAD * 2 + LABEL_H
    sheet = Image.new("RGBA", (width, height), (247, 249, 250, 255))
    x = PAD
    for tile in tiles:
        sheet.alpha_composite(tile, (x, PAD + LABEL_H))
        x += tile.width + PAD
    ImageDraw.Draw(sheet).text(
        (PAD, PAD // 2),
        f"Garment catalog, {len(tiles)} items at rest with anchors marked (schema v{catalog['schemaVersion']})",
        fill=(20, 24, 28, 255),
    )
    sheet.save(OUT_REST, "PNG")
    print(f"wrote {OUT_REST}  ({sheet.width}x{sheet.height})")


def pose_view_points(entry, upper_deg, fore_deg):
    """Body landmarks for a test pose. The torso is stretched like the tester's, then held still."""
    ox, oy = POSE_OFFSET
    sx, sy = POSE_TORSO_SCALE
    a = {name: (p["x"] * sx + ox, p["y"] * sy + oy) for name, p in entry["anchors"].items()}
    view = dict(a)
    for side_name, side in (("left", -1), ("right", 1)):
        shoulder = a[f"{side_name}_shoulder"]
        r1, r2 = math.radians(upper_deg), math.radians(fore_deg)
        elbow = (shoulder[0] + side * UPPER_ARM_LEN * math.sin(r1),
                 shoulder[1] + UPPER_ARM_LEN * math.cos(r1))
        wrist = (elbow[0] + side * FOREARM_LEN * math.sin(r2), elbow[1] + FOREARM_LEN * math.cos(r2))
        view[f"{side_name}_elbow"] = elbow
        view[f"{side_name}_wrist"] = wrist
    return view


def render_pose(entry, layers, upper_deg, fore_deg):
    anchors = {name: (p["x"], p["y"]) for name, p in entry["anchors"].items()}
    view = pose_view_points(entry, upper_deg, fore_deg)

    hip_mid = lambda pts: ((pts["left_hip"][0] + pts["right_hip"][0]) / 2,
                           (pts["left_hip"][1] + pts["right_hip"][1]) / 2)
    torso = affine_from_points(
        [anchors["left_shoulder"], anchors["right_shoulder"], hip_mid(anchors)],
        [view["left_shoulder"], view["right_shoulder"], hip_mid(view)],
    )
    (a, b, _), (d, e, _) = torso
    handedness = 1.0 if a * e - b * d >= 0 else -1.0

    canvas = checkerboard(*POSE_CANVAS)
    matrices = {}
    for part in entry["parts"]:
        if part["kind"] == "torso":
            matrix = torso
        else:
            matrix = segment_matrix(anchors[part["from"]], anchors[part["to"]],
                                    view[part["from"]], view[part["to"]], torso, handedness)
            if matrix is None:
                matrix = matrices[part["parent"]]
        matrices[part["name"]] = matrix

    for i, which in draw_order(entry):
        layer = layers[i][LAYER[which]]
        if layer is not None:
            name = entry["parts"][i]["name"]
            canvas.alpha_composite(warp_layer(layer, matrices[name], POSE_CANVAS))

    # The skeleton the garment is meant to follow, drawn on top.
    draw = ImageDraw.Draw(canvas)
    bone = (40, 170, 180, 255)
    for side in ("left", "right"):
        chain = [view[f"{side}_shoulder"], view[f"{side}_elbow"], view[f"{side}_wrist"]]
        draw.line(chain, fill=bone, width=3)
        for x, y in chain:
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 255, 255, 255), outline=bone, width=2)
    return canvas


def pose_sheet(catalog, base_dir):
    cell_w = int(POSE_CANVAS[0] * POSE_SCALE)
    cell_h = int(POSE_CANVAS[1] * POSE_SCALE)
    garments = catalog["garments"]
    width = PAD + len(POSES) * (cell_w + PAD)
    height = PAD + LABEL_H + len(garments) * (cell_h + LABEL_H + PAD)
    sheet = Image.new("RGBA", (width, height), (247, 249, 250, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, PAD // 2), "Articulation check: garment layers driven by the same segment "
              "transform as the app. Teal = body skeleton.", fill=(20, 24, 28, 255))

    y = PAD + LABEL_H
    for entry in garments:
        layers = load_layers(entry, base_dir)
        x = PAD
        for label, upper, fore in POSES:
            cell = render_pose(entry, layers, upper, fore).resize((cell_w, cell_h), Image.LANCZOS)
            draw.text((x, y), f"{entry['id']}: {label}", fill=(20, 24, 28, 255))
            sheet.alpha_composite(cell, (x, y + LABEL_H - 14))
            x += cell_w + PAD
        y += cell_h + LABEL_H + PAD

    sheet.save(OUT_POSES, "PNG")
    print(f"wrote {OUT_POSES}  ({sheet.width}x{sheet.height})")


def main():
    with open(CATALOG, encoding="utf-8") as handle:
        catalog = json.load(handle)
    base_dir = os.path.dirname(CATALOG)
    rest_sheet(catalog, base_dir)
    pose_sheet(catalog, base_dir)


if __name__ == "__main__":
    main()
