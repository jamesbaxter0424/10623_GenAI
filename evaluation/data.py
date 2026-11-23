"""Reading the NYC data: map mosaics, block table, facing-block pairs, test windows.

Block ids everywhere are row indices of blocks.parquet (the values stored in block.npy,
edges.parquet and samples.parquet).
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import ndimage

import config as C


class City:
    def __init__(self, root=C.DATA_ROOT):
        self.root = root
        self.cls = np.load(os.path.join(root, "cls.npy"), mmap_mode="r")
        self.block = np.load(os.path.join(root, "block.npy"), mmap_mode="r")
        self.water = np.load(os.path.join(root, "water.npy"), mmap_mode="r")
        self.blocks = pd.read_parquet(
            os.path.join(root, "blocks.parquet"),
            columns=["coverage", "far_fp", "median_floors", "res_share", "hclass", "is_res", "valid", "split"])
        self.samples = pd.read_parquet(os.path.join(root, "samples.parquet")).set_index("sample_id")
        edges = pd.read_parquet(os.path.join(root, "edges.parquet"))
        self.nb = len(self.blocks)
        nbrs = [[] for _ in range(self.nb)]
        for i, j in zip(edges["i"].to_numpy(), edges["j"].to_numpy()):
            nbrs[i].append(j)
            nbrs[j].append(i)
        self.neighbors = nbrs
        self._load_cache()
        self.floors = fit_floors(self)
        self._sw_cache = {}

    # ------------------------------------------------------------------ cache
    def _load_cache(self):
        """Per-block class counts and bounding boxes over the whole city (computed once)."""
        os.makedirs(C.CACHE_DIR, exist_ok=True)
        f = os.path.join(C.CACHE_DIR, "block_stats.npz")
        if os.path.exists(f):
            z = np.load(f)
            self.counts, self.bbox = z["counts"], z["bbox"]
            return
        counts = np.zeros((self.nb, C.N_CLASSES), np.int64)
        for r in range(0, self.cls.shape[0], 500):
            b = np.asarray(self.block[r:r + 500]).ravel()
            c = np.asarray(self.cls[r:r + 500]).ravel().astype(np.int64)
            k = b >= 0
            counts += np.bincount(b[k].astype(np.int64) * C.N_CLASSES + c[k],
                                  minlength=self.nb * C.N_CLASSES).reshape(self.nb, C.N_CLASSES)
        objs = ndimage.find_objects(np.asarray(self.block) + 1, max_label=self.nb)
        bbox = np.full((self.nb, 4), -1, np.int64)            # r0, r1, c0, c1 (exclusive end)
        for i, o in enumerate(objs):
            if o is not None:
                bbox[i] = (o[0].start, o[0].stop, o[1].start, o[1].stop)
        np.savez(f, counts=counts, bbox=bbox)
        self.counts, self.bbox = counts, bbox

    # ------------------------------------------------------------------ windows
    def window(self, sid):
        """(cls, block, water) arrays of one test window and its target block ids."""
        s = self.samples.loc[sid]
        r0, c0 = int(s.row0), int(s.col0)
        sl = (slice(r0, r0 + C.N), slice(c0, c0 + C.N))
        return (np.asarray(self.cls[sl]).astype(np.int64), np.asarray(self.block[sl]).astype(np.int64),
                np.asarray(self.water[sl]).astype(bool), [int(t) for t in s.targets])

    # ------------------------------------------------------------------ real block values
    def mean_floors_real(self, b):
        """Mean floors over the building cells of block b in the real map (0 if none)."""
        cnt = self.counts[b]
        n = cnt[2:].sum()
        return float((cnt * self.floors).sum() / n) if n else 0.0

    def street_wall_real(self, b):
        """Built share of the street-edge cells of block b in the real map."""
        if b not in self._sw_cache:
            r0, r1, c0, c1 = self.bbox[b]
            if r0 < 0:
                self._sw_cache[b] = np.nan
            else:
                r0, c0 = max(r0 - 1, 0), max(c0 - 1, 0)
                blk = np.asarray(self.block[r0:r1 + 1, c0:c1 + 1])
                cls = np.asarray(self.cls[r0:r1 + 1, c0:c1 + 1])
                self._sw_cache[b] = street_wall(cls, blk == b, blk < 0)
        return self._sw_cache[b]


def street_edge(inside, street):
    """Cells of `inside` with a 4-neighbour in `street` (street or water cells)."""
    s = np.pad(street, 1, constant_values=False)
    touch = s[:-2, 1:-1] | s[2:, 1:-1] | s[1:-1, :-2] | s[1:-1, 2:]
    return inside & touch


def street_wall(cls, inside, street):
    edge = street_edge(inside, street)
    n = edge.sum()
    return float((cls[edge] >= 2).sum() / n) if n else np.nan


def fit_floors(city):
    """Floors per height class, least squares of block FAR on per-class building share
    over valid training blocks. Returns f[class] (0 for street and open)."""
    f = os.path.join(C.CACHE_DIR, "floors.json")
    if os.path.exists(f):
        return np.array(json.load(open(f))["f_by_class"])
    b = city.blocks
    use = (b["valid"] & (b["split"] == "train")).to_numpy()
    cnt = city.counts[use]
    area = cnt.sum(1)
    keep = area > 0
    h = cnt[keep][:, 2:6] + cnt[keep][:, 6:10]                       # cells per height class
    X = h / area[keep][:, None]
    y = b["far_fp"].to_numpy()[use][keep]
    floors, *_ = np.linalg.lstsq(X, y, rcond=None)
    f_by_class = np.zeros(C.N_CLASSES)
    f_by_class[2:6] = floors
    f_by_class[6:10] = floors
    json.dump({"floors_per_height_class": floors.round(3).tolist(), "f_by_class": f_by_class.tolist(),
               "n_blocks": int(keep.sum())}, open(f, "w"), indent=1)
    return f_by_class


def test_order(city, seed=C.TEST_SEED, split="test"):
    """Fixed random ordering of the clean windows of a split (for "test", identical to the
    nyc_tools.py subset; "train" is used only for the train-vs-test overfitting check)."""
    s = city.samples.reset_index()
    s = s[(s["split"] == split) & (~s["ctx_other_split"])]
    return [int(i) for i in np.random.default_rng(seed).permutation(s["sample_id"].to_numpy())]
