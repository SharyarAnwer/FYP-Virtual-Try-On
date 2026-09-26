"""
Generates placeholder garment assets and the catalog that describes them.

These are stand-ins, not real clothing. DeepFashion supplies photographs of people
wearing clothes, not cut-out garments on transparent backgrounds, so real assets are
manual work. Synthetic shapes are also the better thing to develop the overlay against:
a clean geometric silhouette makes a misaligned transform obvious, whereas a photograph
hides a 5% error in the folds of the fabric.

The important property here is that the PNG and its anchor points are generated from the
same geometry constants. Anchors recorded by hand drift away from the artwork the moment
either is edited, and a drifted anchor looks exactly like a broken transform.

Run:  python tools/generate_placeholder_garments.py
"""

import json
import os

from PIL import Image, ImageDraw

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

LEFT_SHOULDER = (CENTRE_X - SHOULDER_HALF, SHOULDER_Y)
RIGHT_SHOULDER = (CENTRE_X + SHOULDER_HALF, SHOULDER_Y)
LEFT_HEM = (CENTRE_X - HEM_HALF, HEM_Y)
RIGHT_HEM = (CENTRE_X + HEM_HALF, HEM_Y)

OUT_DIR = os.path.join("app", "src", "main", "assets", "garments")


def mirror(point):
    x, y = point
    return (2 * CENTRE_X - x, y)


def body_polygon():
    """Torso outline, clockwise from the left of the neck."""
    return [
        (CENTRE_X - NECK_HALF, SHOULDER_Y + 6),
        LEFT_SHOULDER,
        (CENTRE_X - WAIST_HALF - 10, WAIST_Y),
        LEFT_HEM,
        RIGHT_HEM,
        (CENTRE_X + WAIST_HALF + 10, WAIST_Y),
        RIGHT_SHOULDER,
        (CENTRE_X + NECK_HALF, SHOULDER_Y + 6),
        # neckline scoop
        (CENTRE_X + NECK_HALF - 14, NECK_DIP_Y - 14),
        (CENTRE_X, NECK_DIP_Y),
        (CENTRE_X - NECK_HALF + 14, NECK_DIP_Y - 14),
    ]


def short_sleeve():
    return [
        LEFT_SHOULDER,
        (62, 178),
        (94, 262),
        (168, 228),
        (CENTRE_X - SHOULDER_HALF + 12, SHOULDER_Y + 44),
    ]


def long_sleeve():
    return [
        LEFT_SHOULDER,
        (62, 178),
        (52, 306),
        (84, 438),
        (136, 430),
        (140, 302),
        (168, 214),
        (CENTRE_X - SHOULDER_HALF + 12, SHOULDER_Y + 44),
    ]


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


def draw_garment(fill_rgb, sleeve="short", collar=False):
    image = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    fill = fill_rgb + (255,)
    # A darker edge keeps the garment readable against skin of any tone, which matters
    # when the only way to judge M3's transform is looking at where the edge lands.
    edge = shade(fill_rgb, 0.65) + (255,)

    sleeve_points = short_sleeve() if sleeve == "short" else long_sleeve()
    for shape in (sleeve_points, [mirror(p) for p in sleeve_points], body_polygon()):
        draw.polygon(shape, fill=fill, outline=edge, width=3)

    if collar:
        draw.polygon(collar_polygon(), fill=shade(fill_rgb, 0.88) + (255,), outline=edge, width=3)

    return image


# --- catalog --------------------------------------------------------------------------
# Anchor keys are MediaPipe pose landmark names so M3 can map them without a lookup table.
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    entries = []

    for spec in GARMENTS:
        image = draw_garment(spec["fill"], spec["sleeve"], spec["collar"])
        filename = f"{spec['id']}.png"
        image.save(os.path.join(OUT_DIR, filename), "PNG")

        entries.append({
            "id": spec["id"],
            "name": spec["name"],
            "category": "upper_body",
            "asset": filename,
            "imageWidth": CANVAS_W,
            "imageHeight": CANVAS_H,
            "anchors": {
                "left_shoulder": {"x": LEFT_SHOULDER[0], "y": LEFT_SHOULDER[1]},
                "right_shoulder": {"x": RIGHT_SHOULDER[0], "y": RIGHT_SHOULDER[1]},
                "left_hip": {"x": LEFT_HEM[0], "y": LEFT_HEM[1]},
                "right_hip": {"x": RIGHT_HEM[0], "y": RIGHT_HEM[1]},
            },
            "colour": spec["colour"],
            "gender": spec["gender"],
            "fitStyle": spec["fitStyle"],
            "suitableBodyTypes": spec["suitableBodyTypes"],
            "sizes": SIZE_BANDS,
            "placeholder": True,
        })

    catalog = {
        "schemaVersion": 1,
        "note": (
            "Placeholder artwork generated by tools/generate_placeholder_garments.py. "
            "Replace the PNGs with real cut-outs and update anchors to match; every other "
            "field stays as-is."
        ),
        "garments": entries,
    }

    with open(os.path.join(OUT_DIR, "catalog.json"), "w", encoding="utf-8") as handle:
        json.dump(catalog, handle, indent=2)
        handle.write("\n")

    print(f"wrote {len(entries)} garments + catalog.json to {OUT_DIR}")
    for entry in entries:
        print(f"  {entry['id']:<18} {entry['colour']['name']:<6} {entry['gender']}")


if __name__ == "__main__":
    main()
