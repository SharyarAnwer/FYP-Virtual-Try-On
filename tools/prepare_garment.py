"""
Turns a garment photo on a plain background into app-ready cut-outs.

    python tools/prepare_garment.py SOURCE --id polo_pink [--views 2] [--height 1024]
                                    [--out app/src/main/assets/garments/real]

--views 2 means the source holds two views side by side, front on the left and back on the
right, as many product images do. Each view is written as <id>_front.png / <id>_back.png, on one
shared canvas size with every view top-aligned and centred, so M5.5 can switch between them
without the garment jumping.

Background removal assumes a plain, roughly uniform background, estimated from the image
border. It keeps only background that is *connected to the border*: white buttons, a white
logo or a light collar lining inside the garment stay opaque, which a plain colour key would
have punched holes through.

Edges are matted rather than thresholded. An edge pixel is a blend of garment and background,
so its alpha is how far it lies along the line from background colour to garment colour, and
its colour is recovered by undoing that blend. Without this, every garment gets a light halo
that shows clearly against dark clothing underneath.
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BG_TOLERANCE = 24          # RGB distance from the background colour still counted as background
# Garment pixels within this many px of the background get a matted alpha rather than a solid
# one. 2 was too narrow: where a cuff touches the body the gap between them tapers to a point,
# and pixels at the tip - a blend of shadowed crease and background - sat more than 2 px from
# any pure background, so they stayed solid and showed as a pale speck at the armpit.
EDGE_BAND = 4
OUTER_RING = 2             # background px this close to the garment may still carry its colour
MARGIN = 0.04              # canvas margin around the garment, as a fraction of its height


def background_colour(rgb):
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    return np.median(border, axis=0)


def connected_background(near_bg):
    """Background pixels reachable from the image border, as a boolean array."""
    h, w = near_bg.shape
    padded = np.full((h + 2, w + 2), 255, np.uint8)       # a ring of background all round, so
    padded[1:-1, 1:-1] = np.where(near_bg, 255, 0)       # one fill reaches every border pixel
    # .copy(): an image made by fromarray shares the array's memory read-only, and floodfill's
    # writes to it are silently discarded.
    image = Image.fromarray(padded, "L").copy()
    ImageDraw.floodfill(image, (0, 0), 128)
    return (np.asarray(image) == 128)[1:-1, 1:-1]


def matte(rgb, bg_colour):
    near_bg = np.linalg.norm(rgb - bg_colour, axis=2) < BG_TOLERANCE
    background = connected_background(near_bg)
    garment = ~background

    def dilate(mask, radius):
        img = Image.fromarray((mask * 255).astype(np.uint8), "L")
        return np.asarray(img.filter(ImageFilter.MaxFilter(2 * radius + 1))) > 127

    near_edge = dilate(background, EDGE_BAND)
    interior = garment & ~near_edge
    band = (garment & near_edge) | (dilate(garment, OUTER_RING) & background)

    # Garment colour near each edge pixel, pushed outward from the solid interior.
    weight = Image.fromarray((interior * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(4))
    w = np.asarray(weight, np.float32)[..., None] / 255.0
    premult = np.zeros_like(rgb)
    for c in range(3):
        channel = Image.fromarray((rgb[..., c] * interior).astype(np.uint8), "L")
        premult[..., c] = np.asarray(channel.filter(ImageFilter.GaussianBlur(4)), np.float32)
    fg_est = np.where(w > 1e-3, premult / np.maximum(w * 255.0, 1e-3) * 255.0, rgb)

    # Alpha along the background-to-garment line; colour recovered by undoing the blend.
    d_fg = fg_est - bg_colour
    alpha_band = np.sum((rgb - bg_colour) * d_fg, axis=2) / np.maximum(np.sum(d_fg * d_fg, axis=2), 1.0)
    alpha_band = np.clip(alpha_band, 0.0, 1.0)

    alpha = np.where(interior, 1.0, 0.0)
    alpha = np.where(band, alpha_band, alpha)
    alpha = np.where(garment | band, alpha, 0.0)

    a = alpha[..., None]
    unblended = (rgb - (1.0 - a) * bg_colour) / np.maximum(a, 1e-3)
    colour = np.where(a >= 0.3, unblended, fg_est)
    colour = np.where(interior[..., None], rgb, colour)
    colour = np.clip(colour, 0, 255)

    out = np.dstack([colour, alpha * 255.0]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def split_views(rgba, count):
    """Split side-by-side views at the widest fully transparent column gaps."""
    if count == 1:
        return [rgba]
    alpha = np.asarray(rgba.getchannel("A"))
    empty = alpha.max(axis=0) == 0
    gaps, start = [], None
    for x, e in enumerate(empty):
        if e and start is None:
            start = x
        elif not e and start is not None:
            gaps.append((start, x))
            start = None
    occupied = np.where(~empty)[0]
    inner = [(a, b) for a, b in gaps if a > occupied[0] and b < occupied[-1]]
    if len(inner) < count - 1:
        raise SystemExit(f"expected {count} views but found {len(inner) + 1}; are they touching?")
    cuts = sorted(sorted(inner, key=lambda g: g[1] - g[0], reverse=True)[:count - 1])
    edges = [0] + [(a + b) // 2 for a, b in cuts] + [rgba.width]
    return [rgba.crop((edges[i], 0, edges[i + 1], rgba.height)) for i in range(count)]


def compose(views, target_height):
    boxes = [v.getchannel("A").getbbox() for v in views]
    width = max(b[2] - b[0] for b in boxes)
    height = max(b[3] - b[1] for b in boxes)
    margin = int(height * MARGIN)
    canvas = (width + 2 * margin, height + 2 * margin)

    out = []
    for view, box in zip(views, boxes):
        piece = view.crop(box)
        sheet = Image.new("RGBA", canvas, (0, 0, 0, 0))
        sheet.paste(piece, ((canvas[0] - piece.width) // 2, margin))
        scale = target_height / canvas[1]
        size = (round(canvas[0] * scale), target_height)
        # Resample premultiplied, or transparent pixels bleed their colour into the edges.
        out.append(sheet.convert("RGBa").resize(size, Image.LANCZOS).convert("RGBA"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source")
    ap.add_argument("--id", required=True)
    ap.add_argument("--views", type=int, default=2, choices=(1, 2))
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--out", default=os.path.join("app", "src", "main", "assets", "garments", "real"))
    args = ap.parse_args()

    rgb = np.asarray(Image.open(args.source).convert("RGB"), np.float32)
    bg = background_colour(rgb)
    cutout = matte(rgb, bg)
    views = compose(split_views(cutout, args.views), args.height)

    os.makedirs(args.out, exist_ok=True)
    names = ["front", "back"][:args.views]
    for name, view in zip(names, views):
        path = os.path.join(args.out, f"{args.id}_{name}.png")
        view.save(path, "PNG")
        coverage = (np.asarray(view.getchannel("A")) > 0).mean()
        print(f"wrote {path}  {view.width}x{view.height}  garment covers {coverage:.0%} of canvas")
    print(f"background colour estimated as RGB({bg[0]:.0f}, {bg[1]:.0f}, {bg[2]:.0f})")


if __name__ == "__main__":
    main()
