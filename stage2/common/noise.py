"""Training-time corruption of the 4 m map, so that stage 2 does not only see perfect inputs.

Stage-1 samples have no matching real buildings, so they cannot be training pairs; instead the real
map of the TARGET block is corrupted (the surroundings stay real, as at test time):
  - boundary: building cells next to open space become open and the reverse (p_edge per cell)
  - class:    whole regions change height class by one step (p_class) or use (p_use)
  - holes:    small blobs of 1-4 cells are removed or added (n_blob on average)
A block is left clean with probability p_clean.
"""
import numpy as np
from scipy import ndimage as ndi

DEFAULT = {"p_clean": 0.3, "p_edge": 0.10, "p_class": 0.10, "p_use": 0.05, "n_blob": 1.0}


def corrupt(cls4, inside, rng, p=None):
    p = {**DEFAULT, **(p or {})}
    if rng.random() < p["p_clean"]:
        return cls4
    c = cls4.copy()
    b = inside & (c >= 2)
    open_ = inside & (c == 1)
    if not b.any():
        return c
    # 1. boundary
    inner = b & ndi.binary_dilation(open_)
    outer = open_ & ndi.binary_dilation(b)
    near = ndi.distance_transform_edt(~b, return_distances=False, return_indices=True)
    flip_in = inner & (rng.random(c.shape) < p["p_edge"])
    flip_out = outer & (rng.random(c.shape) < p["p_edge"])
    grown = c[near[0], near[1]]
    c[flip_in] = 1
    c[flip_out] = grown[flip_out]
    # 2. class of whole regions (connected cells of one class)
    for v in np.unique(c[inside & (c >= 2)]):
        lab, n = ndi.label(inside & (c == v))
        for k in range(1, n + 1):
            use, h = (int(v) - 2) // 4, (int(v) - 2) % 4
            if rng.random() < p["p_class"]:
                h = int(np.clip(h + rng.choice([-1, 1]), 0, 3))
            if rng.random() < p["p_use"]:
                use = 1 - use
            c[lab == k] = 2 + 4 * use + h
    # 3. small blobs
    rr, cc = np.nonzero(inside)
    for _ in range(rng.poisson(p["n_blob"])):
        i = rng.integers(len(rr))
        sl = (slice(rr[i], rr[i] + rng.integers(1, 3)), slice(cc[i], cc[i] + rng.integers(1, 3)))
        sub = c[sl]
        ok = inside[sl]
        sub[ok] = 1 if rng.random() < 0.5 else grown[rr[i], cc[i]]
    return c
