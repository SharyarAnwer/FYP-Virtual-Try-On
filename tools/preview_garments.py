"""
Renders a contact sheet of every garment in the catalog with its anchor points marked.

This is the check that M2 actually works: an anchor is just four numbers in a JSON file,
and four numbers cannot be eyeballed. Drawing them onto the artwork is the only way to
see that "left_shoulder" really sits on the garment's left shoulder.

Worth re-running whenever artwork or anchors change — especially when placeholder shapes
are replaced with real cut-outs, since that is exactly when anchors silently stop matching.

Run:  python tools/preview_garments.py
"""

import json
import os

from PIL import Image, ImageDraw

CATALOG = os.path.join("app", "src", "main", "assets", "garments", "catalog.json")
OUT = os.path.join("tools", "garment_preview.png")

CHECKER = 16
PAD = 28
LABEL_H = 34

ANCHOR_COLOURS = {
    "left_shoulder": (255, 92, 92),
    "right_shoulder": (92, 168, 255),
    "left_hip": (255, 196, 64),
    "right_hip": (120, 220, 140),
}


def checkerboard(width, height):
    """Transparency needs a backdrop, or a white garment on white is invisible."""
    board = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(board)
    for y in range(0, height, CHECKER):
        for x in range(0, width, CHECKER):
            if (x // CHECKER + y // CHECKER) % 2:
                draw.rectangle([x, y, x + CHECKER - 1, y + CHECKER - 1], fill=(222, 226, 230, 255))
    return board


def main():
    with open(CATALOG, encoding="utf-8") as handle:
        catalog = json.load(handle)

    garments = catalog["garments"]
    base_dir = os.path.dirname(CATALOG)

    tiles = []
    for entry in garments:
        art = Image.open(os.path.join(base_dir, entry["asset"])).convert("RGBA")
        tile = checkerboard(art.width, art.height)
        tile.alpha_composite(art)

        draw = ImageDraw.Draw(tile)
        for name, point in entry["anchors"].items():
            x, y = point["x"], point["y"]
            colour = ANCHOR_COLOURS.get(name, (255, 0, 255)) + (255,)
            draw.line([x - 14, y, x + 14, y], fill=colour, width=3)
            draw.line([x, y - 14, x, y + 14], fill=colour, width=3)
            draw.ellipse([x - 7, y - 7, x + 7, y + 7], outline=colour, width=3)
            draw.text((x + 18, y - 7), name, fill=(20, 24, 28, 255))

            in_bounds = 0 <= x < entry["imageWidth"] and 0 <= y < entry["imageHeight"]
            if not in_bounds:
                print(f"  WARNING {entry['id']}: anchor {name} at ({x},{y}) is outside the image")

        draw.text((10, 8), f"{entry['id']}  ({entry['colour']['name']}, {entry['gender']})",
                  fill=(20, 24, 28, 255))
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
        f"Garment catalog — {len(tiles)} items, anchors marked (schema v{catalog['schemaVersion']})",
        fill=(20, 24, 28, 255),
    )

    sheet.save(OUT, "PNG")
    print(f"wrote {OUT}  ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
