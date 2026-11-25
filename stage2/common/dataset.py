"""Model inputs built from records.

RasterFrame    condition image at 1 m, rotated to the block's long axis (experiment raster_diffusion)
canonical      condition image rotated to the block's long axis, 64 x 64, plus the frame that maps
               local metres <-> canonical coordinates (experiments set_diffusion, globalmapper_map)
"""
import numpy as np
import shapely
from scipy import ndimage as ndi

from . import config as C
from .geometry import block_frame, rasterize

COND_CHANNELS = C.N_CLASSES + 2          # one-hot map, target mask, distance to street
G = 64                                   # canonical grid size


def street_distance(cls4):
    """Distance (m) of every cell to the nearest street cell, clipped and scaled to 0..1."""
    street = cls4 == 0
    d = ndi.distance_transform_edt(~street) * C.CELL if street.any() else np.full(cls4.shape, C.STREET_CLIP_M)
    return np.minimum(d, C.STREET_CLIP_M) / C.STREET_CLIP_M


def cond_cells(cls4, inside):
    """[12, h, w] float32 condition on the 4 m grid."""
    one = (cls4[None] == np.arange(C.N_CLASSES)[:, None, None]).astype(np.float32)
    return np.concatenate([one, inside[None].astype(np.float32), street_distance(cls4)[None].astype(np.float32)])


class RasterFrame:
    """1 m pixel grid aligned with the block's long axis (rows across, columns along). Aligning matters:
    on the city grid, walls at an angle become staircases and widths come out about 1 m too large."""

    def __init__(self, inside, margin=12.0):
        cx, cy, a, b, self.theta = block_frame(inside)
        self.W, self.H = int(np.ceil(a + 2 * margin)), int(np.ceil(b + 2 * margin))
        c, s = np.cos(self.theta), np.sin(self.theta)
        self.R = np.array([[c, s], [-s, c]])                 # local metres -> frame metres
        self.o = np.array([self.W / 2, self.H / 2]) - self.R @ np.array([cx, cy])

    def poly_to(self, g):
        return shapely.transform(g, lambda q: q @ self.R.T + self.o)

    def poly_back(self, g):
        return shapely.transform(g, lambda q: (q - self.o) @ self.R)

    def _cells(self, shape):
        x, y = np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5)
        p = (np.stack([x, y], -1) - self.o) @ self.R
        c, r = np.floor(p[..., 0] / C.CELL).astype(int), np.floor(p[..., 1] / C.CELL).astype(int)
        return r, c, (r >= 0) & (c >= 0) & (r < shape[0]) & (c < shape[1])

    def cond(self, cls4, inside):
        """[12, H, W] float32 condition (nearest-neighbour)."""
        r, c, ok = self._cells(cls4.shape)
        out = np.zeros((COND_CHANNELS, self.H, self.W), np.float32)
        out[0] = 1.0
        out[:, ok] = cond_cells(cls4, inside)[:, r[ok], c[ok]]
        return out

    def labels(self, polys):
        return rasterize([self.poly_to(g) for g in polys], (self.H, self.W), 1.0)


class Frame:
    """Block frame. Canonical coordinates u (long axis), v (across) in about [-1, 1]; 1 = `half` metres.
    Blocks longer than about 230 m get larger canonical cells, so every block fits."""

    def __init__(self, inside):
        self.cx, self.cy, self.long, self.short, self.theta = block_frame(inside)
        self.cell = max(C.CELL, 1.1 * self.long / G)
        self.half = self.cell * G / 2

    def to_uv(self, x, y):
        c, s = np.cos(self.theta), np.sin(self.theta)
        dx, dy = np.asarray(x) - self.cx, np.asarray(y) - self.cy
        return (dx * c + dy * s) / self.half, (-dx * s + dy * c) / self.half

    def to_xy(self, u, v):
        c, s = np.cos(self.theta), np.sin(self.theta)
        a, b = np.asarray(u) * self.half, np.asarray(v) * self.half
        return self.cx + a * c - b * s, self.cy + a * s + b * c

    def scalars(self):
        return np.array([np.log(self.cell / C.CELL), self.long / 200.0, self.short / 100.0], np.float32)


def canonical(cls4, inside, frame=None):
    """([12, 64, 64] float32, Frame): the map resampled (nearest) into the block frame."""
    f = frame or Frame(inside)
    g = (np.arange(G) + 0.5) / G * 2 - 1
    u, v = np.meshgrid(g, g)                                 # rows = v, columns = u
    x, y = f.to_xy(u, v)
    r, c = np.floor(y / C.CELL).astype(int), np.floor(x / C.CELL).astype(int)
    ok = (r >= 0) & (c >= 0) & (r < cls4.shape[0]) & (c < cls4.shape[1])
    cond = cond_cells(cls4, inside)
    out = np.zeros((COND_CHANNELS, G, G), np.float32)
    out[0] = 1.0                                             # outside the window: street
    out[:, ok] = cond[:, r[ok], c[ok]]
    return out, f
