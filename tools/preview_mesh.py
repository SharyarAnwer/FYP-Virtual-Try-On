"""
Offline check of the mesh warp that draws real garments (M3.6).

    python tools/preview_mesh.py [--id polo_pink]

Bends each real garment's photo through a set of arm poses using the same maths as
OverlayView.kt - grid, weights, per-bone transforms, blend, two-pass draw - and writes
tools/mesh_preview_<id>.png. Like preview_garments.py for the layered placeholders, it exists so
a warp problem shows up here rather than on a phone. The two implementations must change
together.

How the warp works
------------------
A regular grid is laid over the photo. Every grid vertex gets a weight per bone - torso, left
sleeve, right sleeve - from which side of the armhole it lies on: the line from shoulder tip to
armpit, continued below the armpit through the gap between the hanging cuff and the body side.
Weights blend smoothly across a band either side of that line, so the sleeve bends away from the
body without a hard join. Each frame, every vertex moves to the weighted blend of where each bone's
transform would put it (linear blend skinning), and the photo is drawn bent over the grid.

It is then drawn twice: the whole garment, then only its sleeves again on top. A single bent
mesh draws its triangles in grid order, so where a warped sleeve overlaps the body the body
could land on top; the second pass guarantees sleeves are in front.
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

from preview_garments import affine_from_points, segment_matrix, checkerboard

REAL = os.path.join("app", "src", "main", "assets", "garments", "real")

# Grid cell size in artwork px, and blend half-width as a fraction of shoulder-joint width.
# Both mirrored in OverlayView.kt.
MESH_CELL = 16
BLEND = 0.06
# Fixed torso widening below the armpits (FIXED_TORSO_EASE in OverlayView.kt).
TORSO_EASE = 0.18

# Test body, built from the proportions measured on device (see garment_points.py).
SHOULDER_WIDTH = 330
HIP_DROP = 1.45
UPPER_ARM = 0.83
CANVAS = (1080, 1180)
SHOULDER_Y = 250

POSES = [("arms at rest", 8, 8), ("arms 45°", 45, 45), ("arms horizontal", 90, 90),
         ("left 20°, right 70°", 20, 70)]


# --- weights -------------------------------------------------------------------------------

def side_boundary(entry, side):
    """
    One side's sleeve/torso boundary as pieces (origin, direction, length or None, flow).

    The boundary runs shoulder tip -> armpit -> the 'below' point, extended as rays past both
    ends. `flow` is the piece's top-to-bottom direction, which is what the outward normal is
    computed from. The upward ray points *up*, so taking its normal from its own direction gave
    it the wrong side: the neck counted as sleeve on both sides at once.
    """
    arm = entry["armholes"][side]
    p = lambda k: np.array([arm[k]["x"], arm[k]["y"]], float)
    s, a, b = p("shoulder"), p("armpit"), p("below")
    seg = (a - s) / np.linalg.norm(a - s)
    low = (b - a) / np.linalg.norm(b - a)
    return [(s, -seg, None, seg), (s, seg, float(np.linalg.norm(a - s)), seg), (a, low, None, low)]


def signed_distance(points, pieces, outward):
    """Signed distance of each point to the boundary, positive on the sleeve side."""
    best = np.full(len(points), np.inf)
    signed = np.zeros(len(points))
    for origin, direction, length, flow in pieces:
        rel = points - origin
        t = np.clip(rel @ direction, 0.0, length if length is not None else np.inf)
        closest = origin + t[:, None] * direction
        dist = np.linalg.norm(points - closest, axis=1)
        normal = outward * np.array([flow[1], -flow[0]])
        take = dist < best
        best = np.where(take, dist, best)
        signed = np.where(take, np.sign(rel @ normal) * dist, signed)
    return signed


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def joint_width(entry):
    a = entry["anchors"]
    return a["right_shoulder"]["x"] - a["left_shoulder"]["x"]


def vertex_weights(entry, points):
    """(n, 3) weights for torso, left sleeve, right sleeve."""
    band = BLEND * joint_width(entry)
    w_left = smoothstep(-band, band, signed_distance(points, side_boundary(entry, "left"), -1))
    w_right = smoothstep(-band, band, signed_distance(points, side_boundary(entry, "right"), 1))
    w_torso = np.clip(1.0 - w_left - w_right, 0.0, 1.0)
    # Normalise: a vertex's position is a weighted *average* of where each bone puts it, and
    # weights summing past 1 throw the vertex beyond all of them.
    weights = np.stack([w_torso, w_left, w_right], axis=1)
    return weights / weights.sum(axis=1, keepdims=True)


def grid(w, h):
    cols = max(1, round(w / MESH_CELL))
    rows = max(1, round(h / MESH_CELL))
    xs = np.linspace(0, w, cols + 1)
    ys = np.linspace(0, h, rows + 1)
    gx, gy = np.meshgrid(xs, ys)
    return cols, rows, np.stack([gx.ravel(), gy.ravel()], axis=1)


# --- pose ----------------------------------------------------------------------------------

def body(left_deg, right_deg):
    cx = CANVAS[0] / 2
    half = SHOULDER_WIDTH / 2
    pts = {"left_shoulder": (cx - half, SHOULDER_Y), "right_shoulder": (cx + half, SHOULDER_Y)}
    hip_y = SHOULDER_Y + HIP_DROP * SHOULDER_WIDTH
    pts["left_hip"] = (cx - half * 0.55, hip_y)
    pts["right_hip"] = (cx + half * 0.55, hip_y)
    for side, sign, deg in (("left", -1, left_deg), ("right", 1, right_deg)):
        r = math.radians(deg)
        sx, sy = pts[f"{side}_shoulder"]
        length = UPPER_ARM * SHOULDER_WIDTH
        pts[f"{side}_elbow"] = (sx + sign * length * math.sin(r), sy + length * math.cos(r))
    return pts


def bone_matrices(entry, view):
    a = {k: (v["x"], v["y"]) for k, v in entry["anchors"].items()}
    mid = lambda p: ((p["left_hip"][0] + p["right_hip"][0]) / 2, (p["left_hip"][1] + p["right_hip"][1]) / 2)
    torso = affine_from_points([a["left_shoulder"], a["right_shoulder"], mid(a)],
                               [view["left_shoulder"], view["right_shoulder"], mid(view)])
    (ta, tb, _), (td, te, _) = torso
    handedness = 1.0 if ta * te - tb * td >= 0 else -1.0
    sleeves = []
    for side in ("left", "right"):
        m = segment_matrix(a[f"{side}_shoulder"], a[f"{side}_elbow"],
                           view[f"{side}_shoulder"], view[f"{side}_elbow"], torso, handedness)
        sleeves.append(m if m is not None else torso)
    return [torso] + sleeves


def apply_all(matrix, pts):
    (a, b, c), (d, e, f) = matrix
    return np.stack([a * pts[:, 0] + b * pts[:, 1] + c, d * pts[:, 0] + e * pts[:, 1] + f], axis=1)


EASE_FADE = 3.5      # GarmentMesh.EASE_FADE, in blend half-widths


def ease_ramp(entry, ys):
    """0 at shoulder-joint height rising to 1 at armpit height (GarmentMesh.easeRamp)."""
    a, holes = entry["anchors"], entry["armholes"]
    shoulder_y = (a["left_shoulder"]["y"] + a["right_shoulder"]["y"]) / 2
    armpit_y = (holes["left"]["armpit"]["y"] + holes["right"]["armpit"]["y"]) / 2
    return smoothstep(shoulder_y, armpit_y, ys)


def ease_weights(entry, points):
    """1 over the torso and up to the armhole, fading to 0 inside each sleeve (GarmentMesh.easeWeights)."""
    fade = EASE_FADE * BLEND * joint_width(entry)
    fl = smoothstep(0.0, fade, signed_distance(points, side_boundary(entry, "left"), -1))
    fr = smoothstep(0.0, fade, signed_distance(points, side_boundary(entry, "right"), 1))
    return 1 - np.maximum(fl, fr)


def widen(entry, points, amount):
    """Torso, and the inner part of each sleeve, pushed away from the centre line (GarmentMesh.widen)."""
    a = entry["anchors"]
    centre_x = (a["left_shoulder"]["x"] + a["right_shoulder"]["x"]) / 2
    e = amount * ease_ramp(entry, points[:, 1])
    out = points.copy()
    out[:, 0] = points[:, 0] + ease_weights(entry, points) * (points[:, 0] - centre_x) * e
    return out


def skin(points, weights, matrices):
    out = np.zeros_like(points)
    for k, m in enumerate(matrices):
        out += weights[:, k:k + 1] * apply_all(m, points)
    return out


# --- rasteriser (stands in for Canvas.drawBitmapMesh) --------------------------------------

def draw_mesh(target, texture, cols, rows, src, dst):
    """Draw `texture` bent over the grid, triangle by triangle in grid order, like drawBitmapMesh."""
    tex = np.asarray(texture, np.float32) / 255.0
    th, tw = tex.shape[:2]
    out = np.asarray(target, np.float32).copy() / 255.0
    H, W = out.shape[:2]
    stride = cols + 1
    for r in range(rows):
        for c in range(cols):
            i00 = r * stride + c
            for tri in ((i00, i00 + 1, i00 + stride), (i00 + 1, i00 + stride + 1, i00 + stride)):
                d = dst[list(tri)]
                s = src[list(tri)]
                x0, y0 = np.floor(d.min(axis=0)).astype(int)
                x1, y1 = np.ceil(d.max(axis=0)).astype(int)
                x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
                if x1 < x0 or y1 < y0:
                    continue
                gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
                (ax, ay), (bx, by), (cx, cy) = d
                det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
                if abs(det) < 1e-9:
                    continue
                l1 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / det
                l2 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / det
                l3 = 1 - l1 - l2
                inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
                if not inside.any():
                    continue
                u = l1 * s[0, 0] + l2 * s[1, 0] + l3 * s[2, 0]
                v = l1 * s[0, 1] + l2 * s[1, 1] + l3 * s[2, 1]
                ui = np.clip(u.astype(int), 0, tw - 1)
                vi = np.clip(v.astype(int), 0, th - 1)
                px = tex[vi, ui]
                region = out[y0:y1 + 1, x0:x1 + 1]
                a = (px[..., 3:4] * inside[..., None])
                region[..., :3] = px[..., :3] * a + region[..., :3] * (1 - a)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def sleeve_only(texture, entry):
    """The photo with every pixel on the torso side cleared: the second pass draws only sleeves."""
    w, h = texture.size
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
    keep = np.zeros(len(pts), bool)
    for side, outward in (("left", -1), ("right", 1)):
        keep |= signed_distance(pts, side_boundary(entry, side), outward) >= 0
    rgba = np.asarray(texture).copy()
    rgba[..., 3] = np.where(keep.reshape(h, w), rgba[..., 3], 0)
    return Image.fromarray(rgba, "RGBA")


def render(entry, texture, sleeves, left_deg, right_deg):
    w, h = texture.size
    cols, rows, src = grid(w, h)
    weights = vertex_weights(entry, src)
    view = body(left_deg, right_deg)
    dst = skin(widen(entry, src, TORSO_EASE), weights, bone_matrices(entry, view))
    canvas = checkerboard(*CANVAS).convert("RGB")
    canvas = draw_mesh(canvas, texture, cols, rows, src, dst)
    canvas = draw_mesh(canvas, sleeves, cols, rows, src, dst)
    d = ImageDraw.Draw(canvas)
    bone = (40, 170, 180)
    for side in ("left", "right"):
        chain = [view[f"{side}_hip"], view[f"{side}_shoulder"], view[f"{side}_elbow"]]
        d.line(chain, fill=bone, width=4)
        for x, y in chain:
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(255, 255, 255), outline=bone, width=3)
    d.line([view["left_shoulder"], view["right_shoulder"]], fill=bone, width=4)
    return canvas


def weights_image(entry, texture):
    """The photo tinted by weight: red = left sleeve, blue = right sleeve, grey = torso."""
    w, h = texture.size
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
    wt = vertex_weights(entry, pts).reshape(h, w, 3)
    base = np.asarray(checkerboard(w, h).convert("RGB"), np.float32)
    tex = np.asarray(texture, np.float32)
    alpha = tex[..., 3:4] / 255
    grey = tex[..., :3].mean(axis=2, keepdims=True).repeat(3, axis=2)
    tint = (wt[..., 0:1] * grey + wt[..., 1:2] * np.array([235, 70, 60]) + wt[..., 2:3] * np.array([60, 110, 235]))
    out = tint * alpha + base * (1 - alpha)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def clear_neck(texture, entry):
    """Makes the collar's inside back transparent, as GarmentCatalog.loadLayers does."""
    polygon = [(p["x"], p["y"]) for p in entry.get("neckOpening", [])]
    if len(polygon) < 3:
        return texture
    scale = 4                                      # supersampled mask for an anti-aliased edge
    mask = Image.new("L", (texture.width * scale, texture.height * scale), 0)
    ImageDraw.Draw(mask).polygon([(x * scale, y * scale) for x, y in polygon], fill=255)
    mask = mask.resize(texture.size, Image.LANCZOS)
    alpha = np.asarray(texture.getchannel("A"), np.float32) * (1 - np.asarray(mask, np.float32) / 255)
    out = texture.copy()
    out.putalpha(Image.fromarray(alpha.astype(np.uint8), "L"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=None)
    args = ap.parse_args()
    with open(os.path.join(REAL, "catalog.json"), encoding="utf-8") as handle:
        catalog = json.load(handle)
    for entry in catalog["garments"]:
        if args.id and entry["id"] != args.id:
            continue
        texture = clear_neck(Image.open(os.path.join(REAL, entry["front"])).convert("RGBA"), entry)
        sleeves = sleeve_only(texture, entry)
        tiles = [("weights (red/blue = sleeves)", weights_image(entry, texture))]
        for label, l, r in POSES:
            tiles.append((label, render(entry, texture, sleeves, l, r)))
        scale = 0.42
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
        out = os.path.join("tools", f"mesh_preview_{entry['id']}.png")
        sheet.save(out)
        print(f"wrote {out}  ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
