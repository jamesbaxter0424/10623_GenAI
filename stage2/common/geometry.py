"""Geometry helpers. Local coordinates: metres, x to the right, y DOWN, origin at the top-left corner
of the record's window, so a point (x, y) lies in pixel row int(y / px), column int(x / px)."""
import numpy as np
import shapely
from scipy import ndimage as ndi
from shapely.geometry import Polygon, box
from skimage.draw import polygon as _fill

from . import config as C


def clean(g):
    """Largest valid polygon of a geometry (None if empty)."""
    if g is None or g.is_empty:
        return None
    if not g.is_valid:
        g = shapely.make_valid(g)
    parts = [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and p.area > 0]
    return max(parts, key=lambda p: p.area) if parts else None


def rasterize(polys, shape, px=C.PX):
    """Instance labels (1..n, 0 = none); the first polygon wins where polygons overlap."""
    lab = np.zeros(shape, np.int32)
    for i, g in enumerate(polys, 1):
        for p in getattr(g, "geoms", [g]):
            if p.is_empty or p.geom_type != "Polygon":
                continue
            x0, y0, x1, y1 = p.bounds
            r0, c0 = max(int(y0 / px) - 1, 0), max(int(x0 / px) - 1, 0)
            r1, c1 = min(int(y1 / px) + 2, shape[0]), min(int(x1 / px) + 2, shape[1])
            if r1 <= r0 or c1 <= c0:
                continue
            m = np.zeros((r1 - r0, c1 - c0), bool)
            for k, ring in enumerate([p.exterior, *p.interiors]):
                xy = np.asarray(ring.coords)
                rr, cc = _fill(xy[:, 1] / px - 0.5 - r0, xy[:, 0] / px - 0.5 - c0, shape=m.shape)
                m[rr, cc] = k == 0
            sub = lab[r0:r1, c0:c1]
            sub[m & (sub == 0)] = i
    return lab


def own_distance(lab, px=C.PX):
    """For every building pixel: distance (m) from the pixel centre to the edge of its own instance."""
    d = np.zeros(lab.shape, np.float32)
    for q, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        m = np.pad(lab[sl] == q, 1)
        e = ndi.distance_transform_edt(m)[1:-1, 1:-1] - 0.5
        d[sl] = np.maximum(d[sl], np.where(m[1:-1, 1:-1], e, 0))
    return d * px


def up(a, f=C.F):
    """Nearest-neighbour upsampling of the last two axes (4 m cells -> pixels)."""
    return np.repeat(np.repeat(a, f, -2), f, -1)


def to_cells(lab, value, f=C.F):
    """1 m labels + one integer per instance -> (building cell mask, majority value per cell) at 4 m.
    A cell is a building if at least half of its pixels are (same rule as the data pipeline)."""
    h, w = lab.shape[0] // f, lab.shape[1] // f
    sub = lab[:h * f, :w * f].reshape(h, f, w, f).transpose(0, 2, 1, 3).reshape(h, w, f * f)
    built = (sub > 0).mean(-1) >= 0.5
    vals = np.concatenate([[-1], np.asarray(value, np.int64).reshape(-1)])[sub]
    n = max(int(vals.max()) + 1, 1)
    cnt = np.stack([(vals == v).sum(-1) for v in range(n)], -1)
    return built, np.where(built, cnt.argmax(-1), -1)


def cells_polygon(mask, cell=C.CELL):
    """Union of the True cells as one geometry."""
    rr, cc = np.nonzero(mask)
    return shapely.union_all(shapely.box(cc * cell, rr * cell, (cc + 1) * cell, (rr + 1) * cell))


def obb(g):
    """Minimum rotated rectangle: (cx, cy, long, short, angle of the long side in radians)."""
    m = shapely.minimum_rotated_rectangle(g)
    if m.geom_type != "Polygon":
        x0, y0, x1, y1 = g.bounds
        return (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 0.1), max(y1 - y0, 0.1), 0.0
    p = np.asarray(m.exterior.coords)[:4]
    e1, e2 = p[1] - p[0], p[2] - p[1]
    a, b = np.hypot(*e1), np.hypot(*e2)
    v = e1 if a >= b else e2
    c = p.mean(0)
    return float(c[0]), float(c[1]), float(max(a, b)), float(min(a, b)), float(np.arctan2(v[1], v[0]))


def rect(cx, cy, a, b, t):
    """Rectangle polygon: side a along angle t, side b across it."""
    c, s = np.cos(t), np.sin(t)
    p = np.array([[-a, -b], [a, -b], [a, b], [-a, b]]) / 2
    return Polygon(np.stack([cx + p[:, 0] * c - p[:, 1] * s, cy + p[:, 0] * s + p[:, 1] * c], 1))


def block_frame(inside):
    """Long axis of the block: (cx, cy, long, short, theta) with theta in [-90, 90) degrees (radians)."""
    if not inside.any():
        h, w = inside.shape
        return w * C.CELL / 2, h * C.CELL / 2, w * C.CELL, h * C.CELL, 0.0
    r = np.nonzero(inside.any(1))[0]                         # outermost cell corners of every row are enough
    lo = inside[r].argmax(1)
    hi = inside.shape[1] - inside[r][:, ::-1].argmax(1)
    pts = np.concatenate([np.stack([x, y], 1) for x in (lo, hi) for y in (r, r + 1)]) * C.CELL
    cx, cy, a, b, t = obb(shapely.multipoints(pts.astype(float)))
    return cx, cy, a, b, (t + np.pi / 2) % np.pi - np.pi / 2


def polygonize(lab, px=C.PX, tol=1.0):
    """Instance labels -> {label: polygon}. Shared walls stay shared: no gaps, no overlaps."""
    ids, polys = [], []
    for q, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        m = lab[sl] == q
        boxes = []
        for r in range(m.shape[0]):                          # one box per horizontal run of pixels
            d = np.diff(np.concatenate([[0], m[r].astype(np.int8), [0]]))
            for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
                boxes.append(box((a + sl[1].start) * px, (r + sl[0].start) * px,
                                 (b + sl[1].start) * px, (r + sl[0].start + 1) * px))
        ids.append(q)
        polys.append(shapely.union_all(boxes))
    if not polys:
        return {}
    simp = shapely.coverage_simplify(np.array(polys, dtype=object), tol) if tol > 0 else polys
    out = {}
    for q, g, raw in zip(ids, simp, polys):
        g = clean(g) or clean(raw)
        if g is not None:
            out[q] = g
    return out


def trim_overlaps(polys):
    """Remove overlaps: larger polygons keep their area. Returns a list (None where nothing is left)."""
    order = sorted(range(len(polys)), key=lambda i: -polys[i].area)
    out, kept = [None] * len(polys), []
    for i in order:
        g = polys[i]
        near = [k for k in kept if k.intersects(g)]
        if near:
            g = g.difference(shapely.union_all(near))
        g = clean(g)
        if g is not None and g.area >= C.MIN_BLDG_M2:
            out[i] = g
            kept.append(g)
    return out
