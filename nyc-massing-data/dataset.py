"""Loader for the NYC block-massing dataset (data/processed/city/).

    from dataset import NYCMassing
    ds = NYCMassing("train")          # or "val", "test"
    x = ds[0]

Each item is one 512 m window (128 x 128 cells @ 4 m) with 1-4 masked target blocks:

  cls        int64 [128,128]  class 0..9 (0 street, 1 open, 2-5 res 1-2/3-5/6-12/13+ floors, 6-9 non-res)
  mask       bool  [128,128]  cells of the target blocks (what the generator fills)
  node_map   int64 [128,128]  local node index of the block in each cell, -1 = street/water
  water      bool  [128,128]
  node_attr  float32 [n, 5]   coverage, FAR, height class, residential share, log10 area (m2)
  node_target bool [n]        True for the 1-4 target blocks
  edge_index int64 [2, E]     blocks facing each other across a street (both directions)
  edge_attr  float32 [E, 2]   facing length (m), street width (m)
  block_ids  list[str]        "<boro>-<block>-<piece>" per node
  sample_id, split

For target nodes node_attr holds the TRUE table (training). At test time replace
those rows with the table parsed by the language model.
Returns numpy arrays; wrap with torch.from_numpy as needed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "data" / "processed" / "city"
N = 128
ATTR = ["coverage", "far_fp", "hclass", "res_share", "log_area"]


class NYCMassing:
    def __init__(self, split="train", root=ROOT, exclude_ctx_leak=True, k=None):
        root = Path(root)
        self.cls = np.load(root / "cls.npy", mmap_mode="r")
        self.block = np.load(root / "block.npy", mmap_mode="r")
        self.water = np.load(root / "water.npy", mmap_mode="r")
        blk = pd.read_parquet(root / "blocks.parquet",
                              columns=["block_id", "coverage", "far_fp", "hclass", "res_share", "area_m2"])
        blk["log_area"] = np.log10(blk["area_m2"].clip(lower=1))
        blk["hclass"] = blk["hclass"].clip(lower=0)          # -1 (empty block) -> 0
        self.attr = blk[ATTR].to_numpy("float32")
        self.attr = np.nan_to_num(self.attr)
        self.block_id = blk["block_id"].to_numpy()
        e = pd.read_parquet(root / "edges.parquet")
        self.e_i, self.e_j = e["i"].to_numpy(), e["j"].to_numpy()
        self.e_attr = e[["shared_m", "street_m"]].to_numpy("float32")
        s = pd.read_parquet(root / "samples.parquet")
        s = s[s["split"] == split]
        if exclude_ctx_leak:
            s = s[~s["ctx_other_split"]]
        if k is not None:
            s = s[s["k"] == k]
        self.samples = s.reset_index(drop=True)
        self.grid = json.loads((root / "grid.json").read_text())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r = self.samples.iloc[idx]
        r0, c0 = int(r["row0"]), int(r["col0"])
        cls = np.asarray(self.cls[r0:r0 + N, c0:c0 + N], dtype=np.int64)
        blk = np.asarray(self.block[r0:r0 + N, c0:c0 + N])
        water = np.asarray(self.water[r0:r0 + N, c0:c0 + N])
        targets = np.asarray(r["targets"], dtype=np.int64)

        nodes = np.unique(blk[blk >= 0])
        node_map = np.where(blk >= 0, np.searchsorted(nodes, np.maximum(blk, 0)), -1).astype(np.int64)
        mask = np.isin(blk, targets)

        sel = np.isin(self.e_i, nodes) & np.isin(self.e_j, nodes)
        ei = np.searchsorted(nodes, self.e_i[sel])
        ej = np.searchsorted(nodes, self.e_j[sel])
        edge_index = np.concatenate([np.stack([ei, ej]), np.stack([ej, ei])], 1).astype(np.int64)
        edge_attr = np.concatenate([self.e_attr[sel], self.e_attr[sel]], 0)

        return {
            "cls": cls, "mask": mask, "node_map": node_map, "water": water,
            "node_attr": self.attr[nodes], "node_target": np.isin(nodes, targets),
            "edge_index": edge_index, "edge_attr": edge_attr,
            "block_ids": self.block_id[nodes].tolist(),
            "sample_id": int(r["sample_id"]), "split": r["split"],
            "origin_m": (self.grid["x0"] + c0 * self.grid["cell_m"], self.grid["y_top"] - r0 * self.grid["cell_m"]),
        }


try:  # optional torch wrapper
    import torch

    class NYCMassingTorch(torch.utils.data.Dataset):
        def __init__(self, *a, **kw):
            self.ds = NYCMassing(*a, **kw)

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            x = self.ds[i]
            return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in x.items()}
except ImportError:
    pass
