"""
Turns a downloaded 3D glasses model (.glb) into an app-ready pair (M5).

    python tools/prepare_glasses.py SOURCE --id wood_frames --name "Wood frames"
        [--width-mm W] [--frame-budget 6000] [--lens-budget 1500] [--arm-budget 2500]
        [--lenses NAME ...] [--arms NAME ...] [--skip NAME ...]
        [--out app/src/main/assets/glasses]

Writes <out>/<id>.glb, adds the pair to <out>/catalog.json (replacing one with the same id), and
saves tools/glasses_preview_<id>.png - the pair facing the camera, turned 35 and 70 degrees - to
check before trying it on.

Models found online are made for viewers on a desktop, not for a phone drawing them sixty times a
second beside two vision models: one pair came with 120,640 triangles, ten times what the app
can draw in time, in units ten times too large, with its parts named Object_0 to Object_2. So
each pair goes through the same steps:

1. Read: every mesh in the default scene, placed by its nodes' transforms. A turn on the
   scene's root a little off whole quarter turns is how the model was shown in its viewer, not
   how it is made, and is straightened (--keep-tilt keeps it).
2. Find the parts. The lenses are the see-through parts nearly as wide as the frame; the model
   is then centred across on them. The arms are the parts reaching far back behind the front, or
   named like arms (temple, earhook, arm, handle), split into left and right; the frame is the
   rest. A frame made in one piece with its arms is cut off them by shape: its front crosses
   the middle of the face, the arms never do. --lenses, --arms and --skip name parts by hand
   when the guess is wrong. A lens's inner surface (named interior, inner or inside) is left
   out: drawn as a flat tint it only darkens the lens a second time.
3. Bake the colours: each corner takes its material's colour times its texture there, so the
   finished file needs no pictures. Fine detail in a texture blurs: corners are millimetres
   apart. How solid each corner is drawn is baked too, from the material's alpha, the texture's
   transparency and how see-through it is - the material's transmission, times its see-through
   map if it has one. Some models mark their lenses only so: one shared material, the frame
   solid and the glass clear by its pictures alone. Triangles drawn not at all are dropped.
   Whether a part is metal, and how polished, comes from its material's factors times its
   metal-and-roughness picture, averaged over the part.
4. Simplify each part on its own, to its own budget, so the lenses cannot use up the frame's
   detail and no part merges into another. fast_simplification merges, again and again, the
   pair of neighbouring corners whose merge moves the surface least; flat areas lose most of
   their triangles, curves and corners keep more. Corners shared by triangles are welded first,
   or the seams of the old texture layout would split the part into patches. A lens keeps its
   outline, so it still fits its rim.
5. Carry the colours over: each corner left takes the mean colour of the corners merged into it.
6. Recalculate the normals - which way the surface faces at each corner - of a simplified part:
   averaged over the triangles round a corner that bend by less than 45 degrees from each other,
   so curves shade smoothly and real edges stay sharp. A part not simplified keeps its own.
7. Scale to real size, in metres. Downloaded models come in any unit, so the frame's width
   decides: a width that is a pair of glasses in metres, decimetres, centimetres or millimetres
   (110-165 mm) is taken in that unit; anything else is scaled to 140 mm, a typical adult
   frame. --width-mm sets the real width when the product's size is known.
8. Measure where the arms bend down behind the ears, from the lenses' front: along each arm its
   lower edge is followed back, a straight line fitted to its front part, and the bend is where
   the edge falls clearly below that line. The app hides the bent ends behind the ears.
9. Write: metres, +Y up, the front facing +Z, as glTF has it; meshes named Frame, Lenses,
   ArmLeft and ArmRight; each material's colour per corner (COLOR_0), its alpha, see-through
   and metal settings kept. The source's title, author, licence and link are copied into the
   file.
"""

import argparse
import hashlib
import io
import json
import math
import os
import re
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw

import fast_simplification

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NORMALISE = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}
NUM = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

LENS_INNER = re.compile(r"lens.*(interior|inner|inside)", re.I)
ARM_NAME = re.compile(r"temple|earhook|arm|handle", re.I)

# A see-through part - drawn less solid than SEE_THROUGH_PART on average - at least LENS_SHARE
# of the model's width is a lens; one narrower belongs to the frame. Both lenses as one part span
# nine tenths of the width, a single lens about a third, a clear nose pad about a tenth.
SEE_THROUGH_PART = 0.9
LENS_SHARE = 0.2

# Alpha of a fully see-through corner, dark tints then clear ones (see see_through_alpha); a
# triangle no corner of which is drawn more solid than INVISIBLE is dropped.
SEE_THROUGH_DARK = 0.7
SEE_THROUGH_CLEAR = 0.25
INVISIBLE = 0.02
# A part reaching further back than this share of the model's width behind its front is an arm.
ARM_REACH = 0.3

# A frame made in one piece with its arms (see front_section): slices of 1/FRONT_SLICES of the
# model's width; a slice crosses the middle with a triangle within CENTRE_SHARE of the width of
# the centre line, and stands tall at FRONT_HEIGHT_SHARE of the part's tallest slice.
FRONT_SLICES = 40
CENTRE_SHARE = 0.15
FRONT_HEIGHT_SHARE = 0.5

# Real frame widths a pair can have, m; the width used when the file's unit is unknown. Adult
# frames run about 125-150 mm. Wider than this, a guess in decimetres turned one pair into
# 192 mm with 73 mm lenses.
WIDTH_RANGE = (0.110, 0.165)
DEFAULT_WIDTH = 0.140

# A scene root turned less than this from a whole number of quarter turns is straightened: the
# turn is only how the model was shown in its viewer (see straighten).
STRAIGHTEN_MAX_DEG = 45.0

# Normals: triangles round a corner bending less than this from each other are smoothed together.
CREASE_DEG = 45.0

# Arm bend: lower edge sampled in slices of SLICE_MM, the line fitted to the front FIT_SHARE of
# the arm, the bend where the edge falls BEND_DROP_MM below it. An arm with no bend is taken to
# bend at NO_BEND_SHARE of its reach.
SLICE_MM = 5.0
FIT_SHARE = 0.6
BEND_DROP_MM = 4.0
NO_BEND_SHARE = 0.85


# ------------------------------------------------------------------------------------------ read

class Glb:
    def __init__(self, path):
        data = open(path, "rb").read()
        self.sha256 = hashlib.sha256(data).hexdigest()
        magic, version, _ = struct.unpack_from("<4sII", data, 0)
        if magic != b"glTF" or version != 2:
            raise SystemExit(f"{path}: not a glTF 2.0 binary file")
        off, self.json, self.bin = 12, None, b""
        while off + 8 <= len(data):
            length, kind = struct.unpack_from("<I4s", data, off)
            if kind == b"JSON":
                self.json = json.loads(data[off + 8: off + 8 + length])
            elif kind == b"BIN\x00":
                self.bin = data[off + 8: off + 8 + length]
            off += 8 + length
        if self.json is None:
            raise SystemExit(f"{path}: no JSON chunk")
        self.images = {}

    def accessor(self, i):
        a = self.json["accessors"][i]
        if "sparse" in a:
            raise SystemExit("sparse accessors are not supported")
        v = self.json["bufferViews"][a["bufferView"]]
        dt = np.dtype(COMPONENT[a["componentType"]])
        n = NUM[a["type"]]
        start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = v.get("byteStride", dt.itemsize * n)
        count = a["count"]
        raw = np.frombuffer(self.bin, dtype=np.uint8, count=stride * (count - 1) + dt.itemsize * n, offset=start)
        rows = np.lib.stride_tricks.as_strided(raw, shape=(count, dt.itemsize * n), strides=(stride, 1))
        out = np.ascontiguousarray(rows).view(dt).reshape(count, n).astype(np.float64)
        if a.get("normalized") and a["componentType"] in NORMALISE:
            out = np.maximum(out / NORMALISE[a["componentType"]], -1.0)
        return out

    def sample(self, texture_ref, attrs):
        """
        The texture [texture_ref] names, read at each corner through its texture coordinates: RGBA
        as stored, 0..1. Pictures are kept as bytes and only the corners' pixels are converted -
        models come with several 4096 x 4096 pictures, and as floats each would take 400 MB.
        """
        key = "TEXCOORD_%d" % texture_ref.get("texCoord", 0)
        if key not in attrs:
            return None
        i = self.json["textures"][texture_ref["index"]]["source"]
        if i not in self.images:
            img = self.json["images"][i]
            if "bufferView" not in img:
                raise SystemExit("only images inside the .glb are supported")
            v = self.json["bufferViews"][img["bufferView"]]
            start = v.get("byteOffset", 0)
            self.images[i] = np.asarray(Image.open(io.BytesIO(self.bin[start: start + v["byteLength"]])).convert("RGBA"))
        pixels = self.images[i]
        uv = self.accessor(attrs[key]) % 1.0
        rows = np.round(uv[:, 1] * (pixels.shape[0] - 1)).astype(int)
        cols = np.round(uv[:, 0] * (pixels.shape[1] - 1)).astype(int)
        return pixels[rows, cols].astype(np.float64) / 255.0


def linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def local_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    if "scale" in node:
        m = np.diag(list(node["scale"]) + [1.0]) @ m
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        r = np.eye(4)
        r[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        m = r @ m
    if "translation" in node:
        t = np.eye(4)
        t[:3, 3] = node["translation"]
        m = t @ m
    return m


class Part:
    """One primitive: its corners, triangles and material, placed in the model's frame."""

    def __init__(self, label, positions, normals, colours, triangles, material, names="", alphas=None):
        self.label = label
        # How solid each corner is drawn, 0..1: its own transparency and how see-through it is,
        # baked together (see read_primitive).
        self.alphas = np.ones(len(positions)) if alphas is None else alphas
        # Node and mesh names only: material names are shared too freely to say what a part is
        # (the Khronos sunglasses' whole front uses the material "temples").
        self.names = names
        self.positions = positions
        self.normals = normals
        self.colours = colours  # linear RGB per corner
        self.triangles = triangles
        self.material = material
        self.role = None
        self.simplified_from = None


def straighten(m):
    """
    [m] with its turn snapped to the nearest whole quarter turns, scale and position kept, and
    how far it was off, in degrees. Sketchfab stores the turn a model is shown at in its viewer
    on the scene's root: one pair came tipped back 3 degrees, another turned 16.
    """
    a = m[:3, :3]
    scale = abs(np.linalg.det(a)) ** (1.0 / 3.0)
    if scale == 0.0:
        return m, 0.0
    r = a / scale
    snapped = np.zeros((3, 3))
    taken = set()
    for col in np.argsort(-np.abs(r).max(axis=0)):
        for row in np.argsort(-np.abs(r[:, col])):
            if row not in taken:
                snapped[row, col] = 1.0 if r[row, col] >= 0 else -1.0
                taken.add(row)
                break
    off = snapped.T @ r
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(off) - 1.0) / 2.0))))
    out = m.copy()
    out[:3, :3] = snapped * scale
    return out, angle


def read_parts(glb, keep_tilt=False):
    g = glb.json
    materials = g.get("materials", [])
    parts = []

    def walk(index, parent, names):
        node = g["nodes"][index]
        m = local_matrix(node)
        if not names and not keep_tilt:
            straight, angle = straighten(m)
            if 0.5 < angle < STRAIGHTEN_MAX_DEG:
                print("straightened %r: shown turned %.1f degrees in its viewer" % (node.get("name", ""), angle))
                m = straight
        world = parent @ m
        names = names + [node.get("name", "")]
        if "mesh" in node:
            mesh = g["meshes"][node["mesh"]]
            for p in mesh["primitives"]:
                if p.get("mode", 4) != 4:
                    continue
                parts.append(read_primitive(glb, p, world, names, mesh.get("name", ""), materials))
        for child in node.get("children", []):
            walk(child, world, names)

    scene = g["scenes"][g.get("scene", 0)] if g.get("scenes") else {"nodes": list(range(len(g["nodes"])))}
    for root in scene["nodes"]:
        walk(root, np.eye(4), [])
    return parts


def read_primitive(glb, p, world, names, mesh_name, materials):
    attrs = p["attributes"]
    pos = glb.accessor(attrs["POSITION"])
    pos = (np.c_[pos, np.ones(len(pos))] @ world.T)[:, :3]
    normals = None
    if "NORMAL" in attrs:
        n = glb.accessor(attrs["NORMAL"]) @ world[:3, :3].T
        normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    tris = glb.accessor(p["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in p \
        else np.arange(len(pos)).reshape(-1, 3)
    m = materials[p["material"]] if "material" in p else {}
    pbr = m.get("pbrMetallicRoughness", {})
    factor = np.array(pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]), dtype=np.float64)
    colours = np.tile(factor[:3], (len(pos), 1))
    alphas = np.full(len(pos), factor[3])
    if "baseColorTexture" in pbr:
        px = glb.sample(pbr["baseColorTexture"], attrs)
        if px is not None:
            colours = colours * linear(px[:, :3])
            alphas = alphas * px[:, 3]
    if "COLOR_0" in attrs:
        corner = glb.accessor(attrs["COLOR_0"])
        colours = colours * corner[:, :3]
        if corner.shape[1] == 4:
            alphas = alphas * corner[:, 3]
    mode = m.get("alphaMode", "OPAQUE")
    if mode == "MASK":
        alphas = (alphas >= m.get("alphaCutoff", 0.5)).astype(np.float64)
    elif mode != "BLEND":
        alphas = np.ones(len(pos))
    # See-through: how much light passes, per corner, from the material and its see-through map.
    ext = m.get("extensions", {}).get("KHR_materials_transmission", {})
    transmission = np.full(len(pos), float(ext.get("transmissionFactor", 0.0)))
    if "transmissionTexture" in ext:
        px = glb.sample(ext["transmissionTexture"], attrs)
        if px is not None:
            transmission = transmission * px[:, 0]
    alphas = alphas * (1.0 - transmission * (1.0 - see_through_alpha(colours)))
    # Corners nothing of which is drawn: their triangles are dropped.
    visible = alphas[tris].max(axis=1) > INVISIBLE
    tris = tris[visible]
    # Metal and polish: the factors, times the metal-and-roughness picture where there is one
    # (blue metal, green roughness), averaged over the part - the app shades a part as one.
    metallic = float(pbr.get("metallicFactor", 1.0))
    roughness = float(pbr.get("roughnessFactor", 1.0))
    if "metallicRoughnessTexture" in pbr and len(tris):
        px = glb.sample(pbr["metallicRoughnessTexture"], attrs)
        if px is not None:
            used = np.unique(tris)
            metallic *= float(px[used, 2].mean())
            roughness *= float(px[used, 1].mean())
    material = {
        "name": m.get("name", ""),
        "metallic": round(metallic, 3),
        "roughness": round(roughness, 3),
        "doubleSided": bool(m.get("doubleSided", False)),
    }
    shape_names = "/".join(n for n in names if n) + "|" + mesh_name
    return Part(shape_names + "|" + material["name"], pos, normals, colours, tris, material, shape_names, alphas)


def see_through_alpha(colours):
    """
    How solid a fully see-through corner is drawn. The app draws flat colours and cannot bend
    light through glass, so glass is drawn partly transparent instead: a dark tint, like a
    sunglass lens, mostly hides what is behind it, a clear part, like a nose pad, barely does.
    The same rule as GlbModel's in the app, which still applies it to models not prepared here.
    """
    lightness = srgb(colours) @ np.array([0.299, 0.587, 0.114])
    return SEE_THROUGH_DARK + (SEE_THROUGH_CLEAR - SEE_THROUGH_DARK) * lightness


# ------------------------------------------------------------------------------------ the parts

def matches(label, patterns):
    return any(p.lower() in label.lower() for p in patterns)


def used_points(group):
    return np.concatenate([p.positions[np.unique(p.triangles)] for p in group if len(p.triangles)])


def find_roles(parts, lenses, arms, skip):
    allp = used_points(parts)
    width = np.ptp(allp[:, 0])
    front = allp[:, 2].max()
    # Lenses, and parts to leave out.
    for p in parts:
        used = np.unique(p.triangles)
        see_through = len(used) > 0 and p.alphas[used].mean() < SEE_THROUGH_PART
        p.reaches_back = len(used) > 0 and p.positions[used, 2].min() < front - ARM_REACH * width
        if len(used) == 0 or matches(p.label, skip) or LENS_INNER.search(p.label):
            p.role = "skip"
        elif lenses:
            p.role = "Lenses" if matches(p.label, lenses) else None
        elif see_through and not p.reaches_back and np.ptp(p.positions[used, 0]) >= LENS_SHARE * width:
            p.role = "Lenses"
    # Centred across on the lenses - or the whole model, without them - before anything is told
    # apart by side: one pair sat 13.9 units off its file's centre line, both arms on one side.
    kept = [p for p in parts if p.role != "skip"]
    centre_on = [p for p in kept if p.role == "Lenses"] or kept
    xs = used_points(centre_on)[:, 0]
    shift = np.array([(xs.min() + xs.max()) / 2.0, 0.0, 0.0])
    for p in kept:
        p.positions = p.positions - shift
    # Arms - and the front of a frame made in one piece with its arms, cut off them.
    out = []
    for p in kept:
        if p.role is None:
            is_arm = matches(p.label, arms) if arms else (ARM_NAME.search(p.names) or p.reaches_back)
            p.role = "Arms" if is_arm else "Frame"
        if p.role != "Arms":
            out.append(p)
            continue
        zone = front_section(p, width)
        if zone.any():
            out.append(subset(p, zone, "Frame"))
            if zone.all():
                continue
            p = subset(p, ~zone, "Arms")
        # Left and right: glTF's +X is the model's own left.
        centre_x = p.positions[p.triangles].mean(axis=1)[:, 0]
        for side, keep in (("ArmLeft", centre_x > 0), ("ArmRight", centre_x <= 0)):
            if keep.any():
                out.append(subset(p, keep, side))
    return out


def front_section(p, width):
    """
    Which triangles of an arm part are really a frame's front: some models make the frame and
    both arms one piece. Walking back from the part's front in slices, the front is the slices
    up to the last that crosses the middle of the face - a front always holds the bridge, arms
    never come near it - or still stands as tall as half the part's tallest slice, as the rims
    and end pieces do behind the bridge. A part whose front slice does not cross the middle is
    all arm.
    """
    centres = p.positions[p.triangles].mean(axis=1)
    depth = centres[:, 2].max() - centres[:, 2]
    slices = (depth // (width / FRONT_SLICES)).astype(int)
    count = slices.max() + 1
    heights = np.zeros(count)
    crosses = np.zeros(count, dtype=bool)
    for k in range(count):
        sel = slices == k
        if sel.any():
            heights[k] = np.ptp(centres[sel, 1])
            crosses[k] = (np.abs(centres[sel, 0]) < CENTRE_SHARE * width).any()
    end = 0
    if crosses[0]:
        for k in range(count):
            if not (crosses[k] or heights[k] >= FRONT_HEIGHT_SHARE * heights.max()):
                break
            end = k + 1
    return slices < end


def subset(p, keep, role):
    tris = p.triangles[keep]
    used = np.unique(tris)
    remap = np.full(len(p.positions), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    q = Part(p.label, p.positions[used], None if p.normals is None else p.normals[used], p.colours[used],
             remap[tris], p.material, p.names, p.alphas[used])
    q.role = role
    return q


def face_front(parts):
    """Turns the model round if it faces -Z: its arms must run back from the frame, towards -Z."""
    arms = [p for p in parts if p.role.startswith("Arm")]
    frame = [p for p in parts if not p.role.startswith("Arm")]
    if not arms or not frame:
        return False
    arm_z = np.concatenate([p.positions[:, 2] for p in arms]).mean()
    frame_z = np.concatenate([p.positions[:, 2] for p in frame]).mean()
    if arm_z < frame_z:
        return False
    for p in parts:
        p.positions = p.positions * [-1.0, 1.0, -1.0]
        if p.normals is not None:
            p.normals = p.normals * [-1.0, 1.0, -1.0]
    return True


def real_scale(parts, width_mm):
    front = np.concatenate([p.positions for p in parts if not p.role.startswith("Arm")])
    width = np.ptp(front[:, 0])
    if width_mm:
        return width_mm / 1000.0 / width, "set to %.1f mm" % width_mm
    for unit, name in ((1.0, "metres"), (0.1, "decimetres"), (0.01, "centimetres"), (0.001, "millimetres")):
        if WIDTH_RANGE[0] <= width * unit <= WIDTH_RANGE[1]:
            return unit, "file is in %s" % name
    return DEFAULT_WIDTH / width, "unit unknown, scaled to %.0f mm" % (DEFAULT_WIDTH * 1000)


# ------------------------------------------------------------------------------------- simplify

def weld(p, tolerance):
    keys = np.round(p.positions / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse).astype(np.float64)
    colours = np.stack([np.bincount(inverse, weights=p.colours[:, k]) for k in range(3)], axis=1) / counts[:, None]
    alphas = np.bincount(inverse, weights=p.alphas) / counts
    tris = inverse[p.triangles]
    ok = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    return p.positions[first], colours, alphas, tris[ok]


def simplify(p, budget, tolerance):
    if len(p.triangles) <= budget:
        p = subset(p, np.ones(len(p.triangles), dtype=bool), p.role)  # drops corners left unused
        if p.normals is None:
            p.normals = corner_normals_unwelded(p)
        return p
    points, colours, alphas, tris = weld(p, tolerance)
    lens = p.role == "Lenses"
    dec_points, dec_tris, collapses = fast_simplification.simplify(points, tris.astype(np.int32), target_count=budget,
                                                                   return_collapses=True, preserve_border=lens)
    # The replay says which simplified corner each original corner went into, for carrying the
    # colours over. Its own corner positions are not used: on the Black & violet frame it threw
    # 96 of 3,033 corners up to half a metre out, though its triangles matched the simplifier's.
    _, replay_tris, mapping = fast_simplification.replay_simplification(points, tris.astype(np.int32), collapses)
    if not np.array_equal(np.asarray(replay_tris), np.asarray(dec_tris)):
        mapping = nearest(points, np.asarray(dec_points))
        print("  %-9s replay did not match; colours carried to the nearest corners" % p.role)
    keep = mapping >= 0
    counts = np.maximum(np.bincount(mapping[keep], minlength=len(dec_points)).astype(np.float64), 1.0)
    dec_colours = np.stack([np.bincount(mapping[keep], weights=colours[keep, k], minlength=len(dec_points))
                            for k in range(3)], axis=1) / counts[:, None]
    dec_alphas = np.bincount(mapping[keep], weights=alphas[keep], minlength=len(dec_points)) / counts
    positions, normals, out_colours, out_alphas, out_tris = crease_normals(
        np.asarray(dec_points, dtype=np.float64), np.asarray(dec_tris, dtype=np.int64), dec_colours, dec_alphas)
    q = Part(p.label, positions, normals, out_colours, out_tris, p.material, p.names, out_alphas)
    q.role = p.role
    q.simplified_from = len(p.triangles)
    return q


def nearest(points, targets, chunk=2048):
    """For each of [points], the index of the nearest of [targets]."""
    out = np.empty(len(points), dtype=np.int64)
    t2 = (targets ** 2).sum(axis=1)
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        d = t2[None, :] - 2.0 * p @ targets.T
        out[start:start + chunk] = d.argmin(axis=1)
    return out


def corner_normals_unwelded(p):
    """Flat normals for a part that came with none."""
    a, b, c = (p.positions[p.triangles[:, k]] for k in range(3))
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    out = np.zeros_like(p.positions)
    for k in range(3):
        out[p.triangles[:, k]] = n
    return out


def crease_normals(points, tris, colours, alphas):
    """Per-corner normals smoothed across bends under CREASE_DEG; corners split where they differ."""
    a, b, c = (points[tris[:, k]] for k in range(3))
    area_n = np.cross(b - a, c - a)
    unit = area_n / np.maximum(np.linalg.norm(area_n, axis=1, keepdims=True), 1e-12)
    faces_at = [[] for _ in range(len(points))]
    for f, t in enumerate(tris):
        for v in t:
            faces_at[v].append(f)
    cos_crease = math.cos(math.radians(CREASE_DEG))
    index = {}
    positions, normals, out_colours, out_alphas, out_tris = [], [], [], [], []
    for f, t in enumerate(tris):
        corners = []
        for v in t:
            near = [g for g in faces_at[v] if unit[g] @ unit[f] >= cos_crease]
            n = area_n[near].sum(axis=0)
            n /= max(np.linalg.norm(n), 1e-12)
            key = (int(v), int(round(n[0] * 64)), int(round(n[1] * 64)), int(round(n[2] * 64)))
            if key not in index:
                index[key] = len(positions)
                positions.append(points[v])
                normals.append(n)
                out_colours.append(colours[v])
                out_alphas.append(alphas[v])
            corners.append(index[key])
        out_tris.append(corners)
    return (np.array(positions), np.array(normals), np.array(out_colours), np.array(out_alphas),
            np.array(out_tris, dtype=np.int64))


# ------------------------------------------------------------------------------------ the bend

def ear_from_mm(parts):
    lenses = [p for p in parts if p.role == "Lenses"] or [p for p in parts if p.role == "Frame"]
    front = np.concatenate([p.positions[:, 2] for p in lenses]).max()
    bends = []
    for side in ("ArmLeft", "ArmRight"):
        arm = [p for p in parts if p.role == side]
        if not arm:
            continue
        pts = np.concatenate([p.positions for p in arm]) * 1000.0
        depth = front * 1000.0 - pts[:, 2]
        height = pts[:, 1]
        start, end = depth.min(), depth.max()
        edges = np.arange(start, end + SLICE_MM, SLICE_MM)
        mids, bottoms = [], []
        for lo in edges[:-1]:
            sel = (depth >= lo) & (depth < lo + SLICE_MM)
            if sel.any():
                mids.append(lo)
                bottoms.append(height[sel].min())
        mids, bottoms = np.array(mids), np.array(bottoms)
        fit = mids < start + FIT_SHARE * (end - start)
        bend = start + NO_BEND_SHARE * (end - start)
        if fit.sum() >= 2:
            slope, offset = np.polyfit(mids[fit], bottoms[fit], 1)
            below = np.nonzero(bottoms < slope * mids + offset - BEND_DROP_MM)[0]
            if len(below):
                bend = mids[below[0]]
        bends.append(bend)
    return float(np.mean(bends)) if bends else None


# --------------------------------------------------------------------------------------- write

def write_glb(path, parts, source_asset, source_sha):
    roles = ["Frame", "Lenses", "ArmLeft", "ArmRight"]
    materials, material_index = [], {}
    meshes, nodes = [], []
    accessors, views = [], []
    blob = bytearray()

    def add(array, kind, component, target, with_bounds=False):
        data = np.ascontiguousarray(array).tobytes()
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data)
        acc = {"bufferView": len(views) - 1, "componentType": component, "count": len(array), "type": kind}
        if with_bounds:
            acc["min"] = [float(x) for x in array.min(axis=0)]
            acc["max"] = [float(x) for x in array.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    for role in roles:
        prims = []
        for p in (q for q in parts if q.role == role):
            m = p.material
            # See-through is baked into each corner's alpha, so a part with any corner less than
            # solid is simply blended; no transmission is left for the app to apply again.
            blend = bool(p.alphas.min() < 1.0)
            key = json.dumps(dict(m, blend=blend), sort_keys=True)
            if key not in material_index:
                mat = {"name": m["name"] or role,
                       "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                                                "metallicFactor": m["metallic"], "roughnessFactor": m["roughness"]}}
                if blend:
                    mat["alphaMode"] = "BLEND"
                if m["doubleSided"]:
                    mat["doubleSided"] = True
                material_index[key] = len(materials)
                materials.append(mat)
            big = len(p.positions) > 65535
            prims.append({
                "attributes": {
                    "POSITION": add(p.positions.astype(np.float32), "VEC3", 5126, 34962, with_bounds=True),
                    "NORMAL": add(p.normals.astype(np.float32), "VEC3", 5126, 34962),
                    "COLOR_0": add(np.c_[p.colours, p.alphas].astype(np.float32), "VEC4", 5126, 34962),
                },
                "indices": add(p.triangles.reshape(-1).astype(np.uint32 if big else np.uint16), "SCALAR",
                               5125 if big else 5123, 34963),
                "material": material_index[key],
                "mode": 4,
            })
        if prims:
            meshes.append({"name": role, "primitives": prims})
            nodes.append({"name": role, "mesh": len(meshes) - 1})

    asset = {"version": "2.0", "generator": "VirtualTryOn tools/prepare_glasses.py"}
    if "copyright" in source_asset:
        asset["copyright"] = source_asset["copyright"]
    extras = dict(source_asset.get("extras", {}))
    extras["preparedFromSha256"] = source_sha
    asset["extras"] = extras
    gltf = {"asset": asset, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
            "meshes": meshes, "materials": materials, "accessors": accessors, "bufferViews": views,
            "buffers": [{"byteLength": len(blob)}]}
    while len(blob) % 4:
        blob.append(0)
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON"))
        f.write(js)
        f.write(struct.pack("<I4s", len(blob), b"BIN\x00"))
        f.write(blob)


def update_catalog(path, entry):
    catalog = {"note": "Glasses pairs the app offers, in the order the Glasses button steps through them. "
                       "Each file is written by tools/prepare_glasses.py: metres, +Y up, the front facing +Z, "
                       "real size, parts named Frame, Lenses, ArmLeft and ArmRight. earFromMm is how far "
                       "behind the front of the lenses the arms bend down behind the ears.",
               "glasses": []}
    if os.path.exists(path):
        catalog = json.load(open(path, encoding="utf-8"))
    pairs = [g for g in catalog["glasses"] if g["id"] != entry["id"]]
    for i, g in enumerate(catalog["glasses"]):
        if g["id"] == entry["id"]:
            # Fields set by hand - frame shape, frame colour, lenses, gender, for M7 - are kept;
            # the ones this tool measures are replaced.
            pairs.insert(i, dict(g, **entry))
            break
    else:
        pairs.append(entry)
    catalog["glasses"] = pairs
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(catalog, f, indent=2)
        f.write("\n")


# ------------------------------------------------------------------------------------- preview

def preview(parts, path):
    tris, cols, alphas = [], [], []
    for p in parts:
        t = p.positions[p.triangles]
        tris.append(t)
        cols.append(srgb(p.colours[p.triangles].mean(axis=1)))
        alphas.append(p.alphas[p.triangles].mean(axis=1))
    tris = np.concatenate(tris) * 1000.0
    cols = np.concatenate(cols)
    alphas = np.concatenate(alphas)
    allp = tris.reshape(-1, 3)
    cy = (allp[:, 1].min() + allp[:, 1].max()) / 2
    size, scale = 520, 2.6
    views = []
    for yaw in (0, 35, 70):
        a = math.radians(yaw)
        rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        pts = tris @ rot.T
        image = Image.new("RGB", (size, size), (190, 160, 140))
        draw = ImageDraw.Draw(image, "RGBA")
        for i in np.argsort(pts[:, :, 2].mean(axis=1)):
            t = pts[i]
            n = np.cross(t[1] - t[0], t[2] - t[0])
            nl = np.linalg.norm(n)
            light = 0.45 + 0.55 * abs(n[2]) / nl if nl > 0 else 0.45
            rgb = tuple(int(min(255, ch * light * 255)) for ch in cols[i])
            draw.polygon([(size / 2 + q[0] * scale, size / 2 - (q[1] - cy) * scale) for q in t],
                         fill=rgb + (int(alphas[i] * 255),))
        draw.text((8, 8), "turn %d deg" % yaw, fill=(0, 0, 0))
        views.append(image)
    sheet = Image.new("RGB", (size * len(views), size))
    for i, v in enumerate(views):
        sheet.paste(v, (i * size, 0))
    sheet.save(path)


# ---------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source")
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True, help="the name the Glasses button shows")
    ap.add_argument("--width-mm", type=float, default=None, help="the frame's real width, if known")
    ap.add_argument("--frame-budget", type=int, default=6000)
    ap.add_argument("--lens-budget", type=int, default=1500)
    ap.add_argument("--arm-budget", type=int, default=2500, help="per arm")
    ap.add_argument("--lenses", nargs="*", default=[], help="name parts that are the lenses")
    ap.add_argument("--arms", nargs="*", default=[], help="name parts that are the arms")
    ap.add_argument("--skip", nargs="*", default=[], help="name parts to leave out")
    ap.add_argument("--keep-tilt", action="store_true", help="do not straighten the scene's root")
    ap.add_argument("--out", default=os.path.join("app", "src", "main", "assets", "glasses"))
    args = ap.parse_args()

    glb = Glb(args.source)
    parts = find_roles(read_parts(glb, args.keep_tilt), args.lenses, args.arms, args.skip)
    for p in parts:
        print("  %-9s %6d triangles  metal %.2f rough %.2f  %s" % (p.role, len(p.triangles), p.material["metallic"],
                                                                p.material["roughness"], p.label))
    if not any(p.role == "Lenses" for p in parts):
        print("warning: no lenses found; name them with --lenses")
    if not any(p.role.startswith("Arm") for p in parts):
        print("warning: no arms found; name them with --arms")
    if face_front(parts):
        print("turned round: the model faced -Z")

    scale, why = real_scale(parts, args.width_mm)
    for p in parts:
        p.positions = p.positions * scale
    front = np.concatenate([p.positions for p in parts if not p.role.startswith("Arm")])
    width_mm = np.ptp(front[:, 0]) * 1000.0
    print("scale x%g (%s): frame %.1f mm wide" % (scale, why, width_mm))

    budgets = {"Frame": args.frame_budget, "Lenses": args.lens_budget,
               "ArmLeft": args.arm_budget, "ArmRight": args.arm_budget}
    tolerance = width_mm / 1000.0 * 1e-5
    out_parts = []
    for role, budget in budgets.items():
        group = [p for p in parts if p.role == role]
        total = sum(len(p.triangles) for p in group)
        for p in group:
            share = max(4, int(round(budget * len(p.triangles) / total))) if total > budget else len(p.triangles)
            out_parts.append(simplify(p, share, tolerance))
    for p in out_parts:
        before = p.simplified_from or len(p.triangles)
        print("  %-9s %6d -> %6d triangles" % (p.role, before, len(p.triangles)))
    triangles = sum(len(p.triangles) for p in out_parts)

    bend = ear_from_mm(out_parts)
    print("triangles %d; arms bend %.0f mm behind the lenses' front" % (triangles, bend or float("nan")))

    os.makedirs(args.out, exist_ok=True)
    file = args.id + ".glb"
    write_glb(os.path.join(args.out, file), out_parts, glb.json.get("asset", {}), glb.sha256)
    entry = {"id": args.id, "name": args.name, "file": file, "widthMm": round(width_mm, 1),
             "earFromMm": round(bend, 1) if bend else None, "triangles": triangles}
    update_catalog(os.path.join(args.out, "catalog.json"), entry)
    preview_path = os.path.join("tools", "glasses_preview_%s.png" % args.id)
    preview(out_parts, preview_path)
    print("wrote %s (%d KB), catalog entry, %s" % (os.path.join(args.out, file),
                                                   os.path.getsize(os.path.join(args.out, file)) // 1024, preview_path))


if __name__ == "__main__":
    sys.exit(main())
