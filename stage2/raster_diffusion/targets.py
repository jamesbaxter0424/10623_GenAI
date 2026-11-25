"""Target maps at 1 m in the block-aligned frame, scaled to [-1, 1], and the training dataset.

channel 0  building (+1) or not (-1)
channel 1  distance to the wall of the pixel's OWN building, clipped at dist_clip
channel 2  building height, log scale
"""
import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import Dataset

from common.dataset import COND_CHANNELS, RasterFrame
from common.geometry import own_distance
from common.noise import corrupt
from common.records import polys

C_INSIDE = COND_CHANNELS - 2                               # channel of the target-block mask


def encode(lab, heights, cfg):
    d = own_distance(lab)
    h = np.concatenate([[0.0], np.asarray(heights, np.float32)])[lab]
    return np.stack([np.where(lab > 0, 1.0, -1.0),
                     np.minimum(d, cfg["dist_clip"]) / cfg["dist_clip"] * 2 - 1,
                     np.log1p(h) / np.log1p(cfg["height_max"]) * 2 - 1]).astype(np.float32)


def decode(x, cfg):
    """(mask, distance in m, height in m) from one [3, H, W] array."""
    d = (np.clip(x[1], -1, 1) + 1) / 2 * cfg["dist_clip"]
    h = np.expm1((np.clip(x[2], -1, 1) + 1) / 2 * np.log1p(cfg["height_max"]))
    return x[0] > 0, d, h


def weights(target, inside_px, cfg):
    """Per-pixel, per-channel loss weights [3, H, W]."""
    near = ndi.binary_dilation(inside_px, iterations=3)
    w = np.where(near, 1.0, cfg["w_outside"]).astype(np.float32)
    w = np.stack([w] * 3)
    wall = (target[0] > 0) & (target[1] < 2 * 1.5 / cfg["dist_clip"] - 1)        # within 1.5 m of a wall
    w[1] *= np.where(wall, 1 + cfg["w_wall"], 1.0).astype(np.float32)
    return w


def pad16(a, value=0.0):
    """Pad the last two axes to multiples of 16."""
    h, w = a.shape[-2:]
    p = [(0, 0)] * (a.ndim - 2) + [(0, -h % 16), (0, -w % 16)]
    return np.pad(a, p, constant_values=value)


def pad_cond(cond):
    c = pad16(cond)
    c[0, cond.shape[1]:] = 1.0                               # padding is street
    c[0, :, cond.shape[2]:] = 1.0
    return c


class Blocks(Dataset):
    def __init__(self, recs, cfg):
        self.recs, self.cfg = recs, cfg

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r, cfg = self.recs[i], self.cfg
        rng = np.random.default_rng(np.random.randint(2 ** 31))
        cls4 = corrupt(r["cls4"], r["inside"], rng) if cfg["noise"] else r["cls4"]
        f = RasterFrame(r["inside"])
        cond = f.cond(cls4, r["inside"])
        target = encode(f.labels(polys(r)), r["height"], cfg)
        w = weights(target, cond[C_INSIDE] > 0, cfg)
        C3 = np.zeros((COND_CHANNELS, *cfg["crop"]), np.float32)
        C3[0] = 1.0
        T3 = np.full((3, *cfg["crop"]), -1.0, np.float32)
        W3 = np.full((3, *cfg["crop"]), cfg["w_outside"], np.float32)
        src, dst = [], []
        for size, n in zip(cond.shape[1:], cfg["crop"]):      # random position; large blocks are cropped
            if size <= n:
                k = int(rng.integers(0, n - size + 1))
                src.append(slice(0, size))
                dst.append(slice(k, k + size))
            else:
                k = int(rng.integers(0, size - n + 1))
                src.append(slice(k, k + n))
                dst.append(slice(0, n))
        C3[:, dst[0], dst[1]] = cond[:, src[0], src[1]]
        T3[:, dst[0], dst[1]] = target[:, src[0], src[1]]
        W3[:, dst[0], dst[1]] = w[:, src[0], src[1]]
        if cfg["augment"]:
            for ax in (1, 2):
                if rng.random() < 0.5:
                    C3, T3, W3 = (np.flip(a, ax) for a in (C3, T3, W3))
        return {k: torch.from_numpy(a.copy()) for k, a in (("cond", C3), ("x0", T3), ("w", W3))}
