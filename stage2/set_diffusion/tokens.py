"""Buildings <-> tokens, and the training dataset.

A token is one building as a rotated rectangle in the BLOCK FRAME (common/dataset.Frame):
  0 exist    +1 building, -1 empty
  1 u        position along the block's long axis   (metres / SCALE)
  2 v        position across the block              (metres / SCALE)
  3 log a    log size along u, standardised
  4 log b    log size along v, standardised
  5 angle    rotation relative to the block, in [-45, 45) degrees / 45
  6 log h    log height, standardised
Positions are in metres (not divided by the block size), so a rowhouse has the same size in every block.
The rectangle keeps the centre, the orientation and the side ratio of the minimum rotated rectangle and is
scaled to the building's true area.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from common import config as C
from common.dataset import canonical
from common.geometry import obb, rect, trim_overlaps
from common.noise import corrupt
from common.records import polys

DIM = 7
SCALE = 16.0                                                 # metres per unit of position: values reach +-8, so
                                                             # 0.5 m is 0.03 and survives the last noise levels
# mean and std of log side (m) and log height (m) over the training buildings
LOG_A, LOG_B, LOG_H = (2.05, 0.55), (2.40, 0.53), (1.99, 0.50)


def boxes(rec, frame):
    """[n, 6] u, v, a, b, angle (metres, radians, block frame), height."""
    out = []
    for p, h in zip(polys(rec), rec["height"]):
        cx, cy, lo, sh, t = obb(p)
        k = np.sqrt(p.area / max(lo * sh, 1e-6))
        d = (t - frame.theta + np.pi / 4) % np.pi - np.pi / 4        # [-45, 135)
        a, b = (lo, sh) if d < np.pi / 4 else (sh, lo)               # side along u, side along v
        d = d if d < np.pi / 4 else d - np.pi / 2
        u, v = frame.to_uv(cx, cy)
        out.append([u * frame.half, v * frame.half, a * k, b * k, d, h])
    return np.array(out, np.float32).reshape(-1, 6)


def encode(b, n):
    """[m, 6] boxes -> [n, 7] tokens (largest buildings first if there are more than n)."""
    x = np.zeros((n, DIM), np.float32)
    x[:, 0] = -1
    if len(b) > n:
        b = b[np.argsort(-b[:, 2] * b[:, 3])[:n]]
    m = len(b)
    if m:
        s = lambda v, k: (np.log(np.maximum(v, 0.5)) - k[0]) / k[1]
        x[:m] = np.stack([np.ones(m), b[:, 0] / SCALE, b[:, 1] / SCALE, s(b[:, 2], LOG_A), s(b[:, 3], LOG_B),
                          b[:, 4] / (np.pi / 4), s(b[:, 5], LOG_H)], 1)
    return x


def order(x):
    """Real buildings first: those on the negative side of the long axis, then the others, along the axis."""
    return x[np.lexsort((x[:, 1], x[:, 2] >= 0, x[:, 0] <= 0))]


def decode(x, thresh=0.0):
    """[n, 7] tokens -> [m, 6] boxes."""
    x = np.asarray(x, np.float64)
    x = x[x[:, 0] > thresh]
    e = lambda v, k: np.exp(np.clip(v, -4, 6) * k[1] + k[0])
    return np.stack([x[:, 1] * SCALE, x[:, 2] * SCALE, e(x[:, 3], LOG_A), e(x[:, 4], LOG_B),
                     x[:, 5] * np.pi / 4, e(x[:, 6], LOG_H)], 1)


def to_polygons(b, frame, trim=True):
    """boxes -> (polygons, heights) in local coordinates."""
    g = []
    for u, v, a, c, d, _ in b:
        x, y = frame.to_xy(u / frame.half, v / frame.half)
        g.append(rect(float(x), float(y), a, c, d + frame.theta))
    h = np.clip(b[:, 5], C.H_MIN, C.H_MAX) if len(b) else np.zeros(0)
    if trim:
        keep = [(p, z) for p, z in zip(trim_overlaps(g), h) if p is not None]
        return [p for p, _ in keep], np.array([z for _, z in keep])
    ok = [i for i, p in enumerate(g) if p.area >= C.MIN_BLDG_M2]
    return [g[i] for i in ok], h[ok]


class Blocks(Dataset):
    def __init__(self, recs, cfg):
        self.recs, self.cfg = recs, cfg

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r, cfg = self.recs[i], self.cfg
        rng = np.random.default_rng(np.random.randint(2 ** 31))
        m = corrupt(r["cls4"], r["inside"], rng) if cfg["noise"] else r["cls4"]
        cond, f = canonical(m, r["inside"])
        x = encode(boxes(r, f), cfg["tokens"])
        if cfg["augment"]:
            if rng.random() < 0.5:                           # flip along the block
                cond, x[:, 1], x[:, 5] = cond[:, :, ::-1], -x[:, 1], -x[:, 5]
            if rng.random() < 0.5:                           # flip across the block
                cond, x[:, 2], x[:, 5] = cond[:, ::-1], -x[:, 2], -x[:, 5]
        x = order(x) if cfg["order"] == "sorted" else x[rng.permutation(len(x))]
        return {"cond": torch.from_numpy(cond.copy()), "scal": torch.from_numpy(f.scalars()),
                "half": torch.tensor(f.half, dtype=torch.float32), "x0": torch.from_numpy(x)}
