"""NYC block-massing data in the format of Liu et al.'s scene-graph dataset (baseline).

Reads data/processed/city/ from the nyc-massing-data pipeline. Each item is one
512 m window (128 x 128 cells, 4 m) and its block graph, given to Liu's 2D stage
*as is* (the "original Liu model" baseline of the project):

  node feature x (27) = one-hot label (10) + Liu's coarse 4x4 position patch (17)
      label: dominant use x height class of the block, in map-class ids
             res 1-2/3-5/6-12/13+ -> 2..5, non-res -> 6..9, block without buildings -> 1 (open)
      Liu's road-type one-hot is dropped (no counterpart for city blocks).
  heatmap  = one-hot cell at the centroid of the block's cells inside the window
             (ground-truth position used for joint training, as in Liu)
  nodes    = blocks with >= min_visible_cells cells inside the window (20 cells = 320 m2
             in the configs); smaller slivers cut by the window edge are dropped.
  edges    = k nearest blocks by centroid (symmetric). The street-facing graph of
             the dataset is change 1 and is NOT used here.
  NOT given: numeric targets (coverage, FAR, ...), block polygons, map context.

Change 1 (numeric=True, graph="street"; configs/nyc_train_c1.yaml):
  node feature x (33) = the 27 above + 6 numbers (see numeric_features):
      coverage, log(1+FAR), log(1+median floors)/2, residential share, log10(area m2)/5,
      unreliable flag. True values for context blocks, the target table for targets
      (true table in training). Blocks failing the dataset checks get zeros and flag 1.
  edges    = blocks facing each other across a street (edges.parquet of the dataset),
             edge_attr (2) = facing length / 100 m (capped at 5), street width / 30 m.

  bev  = class map of the whole window (target of the 2D diffusion)
  mask = cells of the 1-4 target blocks (only used for evaluation)
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

N = 128            # window size in cells
PATCH_GRID = 4     # Liu's coarse position grid
NUM_CLASSES = 10


def block_labels(blk):
    """Liu-style node label per block in map-class ids (see module docstring)."""
    h = blk["hclass"].to_numpy()
    res = blk["is_res"].to_numpy().astype(bool)
    lab = np.where(res, 2 + h, 6 + h)
    return np.where(h < 0, 1, lab).astype(np.int64)


def position_patch(r, c):
    """Liu's coarse position feature (datasets/scene_graph_dataset.py::_position_patch),
    kept as is, including the special slot 0 for patch (0, 0)."""
    ri = min(PATCH_GRID - 1, int(r / N * PATCH_GRID))
    ci = min(PATCH_GRID - 1, int(c / N * PATCH_GRID))
    feat = np.zeros(1 + PATCH_GRID * PATCH_GRID, dtype=np.float32)
    if ri == 0 and ci == 0:
        feat[0] = 1.0
    else:
        feat[1 + ri * PATCH_GRID + ci] = 1.0
    return feat


NUM_NUMERIC = 6


def numeric_features(blk):
    """Change 1: six numbers per block, scaled to about 0-1 (see module docstring)."""
    valid = blk["valid"].to_numpy().astype(bool)
    f = np.stack([
        blk["coverage"].to_numpy(),
        np.log1p(blk["far_fp"].to_numpy()),
        np.log1p(blk["median_floors"].to_numpy()) / 2,
        blk["res_share"].to_numpy(),
        np.log10(np.maximum(blk["area_m2"].to_numpy(), 1.0)) / 5,
        np.zeros(len(blk)),
    ], 1).astype(np.float32)
    f = np.nan_to_num(f)
    f[~valid] = 0.0
    f[~valid, -1] = 1.0
    return f


def street_edges(e_i, e_j, e_attr, nodes):
    """Change 1: street-facing edges among the window's nodes, both directions."""
    sel = np.isin(e_i, nodes) & np.isin(e_j, nodes)
    i = np.searchsorted(nodes, e_i[sel])
    j = np.searchsorted(nodes, e_j[sel])
    ei = np.concatenate([np.stack([i, j]), np.stack([j, i])], 1).astype(np.int64)
    ea = np.concatenate([e_attr[sel], e_attr[sel]], 0).astype(np.float32)
    return ei, ea


def knn_edges(cent, k):
    """Symmetric k-nearest-neighbour edges between node centroids, both directions."""
    n = len(cent)
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64)
    d = np.linalg.norm(cent[:, None] - cent[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    kk = min(k, n - 1)
    nn = np.argsort(d, axis=1)[:, :kk]
    pairs = {(min(i, j), max(i, j)) for i in range(n) for j in nn[i]}
    e = np.array(sorted(pairs), dtype=np.int64).T
    return np.concatenate([e, e[::-1]], axis=1)


class NYCSceneGraphDataset(Dataset):
    def __init__(self, root, split="train", knn=4, feature_mask_ratio=0.0,
                 exclude_ctx_leak=True, sample_ids=None, max_samples=None, repeat=1,
                 min_visible_cells=0, numeric=False, graph="knn"):
        self.root = root
        self.knn = knn
        self.numeric = numeric
        self.graph = graph
        self.min_visible_cells = min_visible_cells
        self.feature_mask_ratio = feature_mask_ratio
        self.repeat = repeat
        self.cls = np.load(os.path.join(root, "cls.npy"), mmap_mode="r")
        self.block = np.load(os.path.join(root, "block.npy"), mmap_mode="r")
        blk = pd.read_parquet(os.path.join(root, "blocks.parquet"),
                              columns=["block_id", "hclass", "is_res"])
        self.label = block_labels(blk)
        if numeric:
            num = pd.read_parquet(os.path.join(root, "blocks.parquet"),
                                  columns=["coverage", "far_fp", "median_floors", "res_share",
                                           "area_m2", "valid"])
            self.num = numeric_features(num)
        if graph == "street":
            e = pd.read_parquet(os.path.join(root, "edges.parquet"))
            self.e_i, self.e_j = e["i"].to_numpy(), e["j"].to_numpy()
            self.e_attr = np.stack([np.minimum(e["shared_m"].to_numpy(), 500.0) / 100.0,
                                    e["street_m"].to_numpy() / 30.0], 1).astype(np.float32)
        elif graph != "knn":
            raise ValueError(f"unknown graph {graph}")

        s = pd.read_parquet(os.path.join(root, "samples.parquet"))
        s = s[s["split"] == split]
        if exclude_ctx_leak:
            s = s[~s["ctx_other_split"]]
        if sample_ids is not None:
            s = s.set_index("sample_id").loc[list(sample_ids)].reset_index()
        if max_samples:
            s = s.iloc[:max_samples]
        self.samples = s.reset_index(drop=True)
        print(f"NYCSceneGraphDataset[{split}]: {len(self.samples)} windows"
              + (f" x{repeat}" if repeat > 1 else ""))

    def __len__(self):
        return len(self.samples) * self.repeat

    def window(self, idx):
        r = self.samples.iloc[idx % len(self.samples)]
        r0, c0 = int(r["row0"]), int(r["col0"])
        cls = np.asarray(self.cls[r0:r0 + N, c0:c0 + N], dtype=np.int64)
        blk = np.asarray(self.block[r0:r0 + N, c0:c0 + N], dtype=np.int64)
        return r, cls, blk

    def build_graph(self, blk, targets=()):
        """Nodes = blocks with >= min_visible_cells cells inside the window (slivers of
        blocks cut by the window edge are dropped; target blocks are always kept)."""
        nodes, cnt = np.unique(blk[blk >= 0], return_counts=True)
        keep = (cnt >= self.min_visible_cells) | np.isin(nodes, targets)
        nodes = nodes[keep]
        inside = np.isin(blk, nodes)
        rr, cc = np.nonzero(inside)
        local = np.searchsorted(nodes, blk[inside])
        cnt = np.bincount(local, minlength=len(nodes))
        cent = np.stack([np.bincount(local, rr, len(nodes)) / cnt,
                         np.bincount(local, cc, len(nodes)) / cnt], 1)
        labels = self.label[nodes]

        feats, heat = [], np.zeros((len(nodes), N, N), dtype=np.float32)
        for i, (r, c) in enumerate(cent):
            pos = position_patch(r, c)
            if self.feature_mask_ratio > 0 and np.random.rand() < self.feature_mask_ratio:
                pos = np.zeros_like(pos)
            parts = [np.eye(NUM_CLASSES, dtype=np.float32)[labels[i]], pos]
            if self.numeric:
                parts.append(self.num[nodes[i]])
            feats.append(np.concatenate(parts))
            heat[i, min(N - 1, int(r)), min(N - 1, int(c))] = 1.0
        return nodes, cent, labels, np.stack(feats), heat

    def __getitem__(self, idx):
        r, cls, blk = self.window(idx)
        targets = np.asarray(r["targets"], dtype=np.int64)
        nodes, cent, labels, feats, heat = self.build_graph(blk, targets)
        extra = {}
        if self.graph == "street":
            ei, ea = street_edges(self.e_i, self.e_j, self.e_attr, nodes)
            extra["edge_attr"] = torch.from_numpy(ea)
        else:
            ei = knn_edges(cent, self.knn)
        sg = Data(x=torch.from_numpy(feats),
                  edge_index=torch.from_numpy(ei),
                  y=torch.from_numpy(labels),
                  heatmaps=torch.from_numpy(heat), **extra)
        return {"file_path": str(int(r["sample_id"])),
                "scene_graph": sg,
                "bev": torch.from_numpy(cls),
                "mask": torch.from_numpy(np.isin(blk, targets))}


def load_subset(path, n=None):
    ids = json.load(open(path))["sample_ids"]
    return ids[:n] if n else ids
