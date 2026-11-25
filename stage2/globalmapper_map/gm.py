"""The trained GlobalMapper (official code, our NYC weights), wrapped for stage 2.

Needs three things that are NOT in git (ask James):
  --gm    the official repository (github.com/Arking1995/GlobalMapper)
  --ckpt  val_least_loss_all.pth, our NYC weights
  --gmdata  gm_nyc/: {train,val,test}/{processed/*.gpickle, index.csv, geom.pkl} and norm.json,
            made by baselines/globalmapper_nyc/prep_nyc.py
Nothing in GlobalMapper is retrained: the encoder gives latents, the decoder turns latents into buildings.
"""
import json
import math
import os
import pickle
import sys
import types

import numpy as np
import pandas as pd
import shapely
from shapely import wkb

from common import config as C
from common.geometry import clean, trim_overlaps

SLOTS = 120


def add_args(ap):
    ap.add_argument("--gm", default=os.environ.get("GM_REPO"), help="official GlobalMapper repository")
    ap.add_argument("--ckpt_gm", default=os.environ.get("GM_CKPT"), help="val_least_loss_all.pth")
    ap.add_argument("--gmdata", default=os.environ.get("GM_DATA"), help="gm_nyc folder")


def check_args(a):
    for k, v in (("--gm / GM_REPO", a.gm), ("--ckpt_gm / GM_CKPT", a.ckpt_gm), ("--gmdata / GM_DATA", a.gmdata)):
        if not v or not os.path.exists(v):
            sys.exit(f"{k} is not set or does not exist: {v!r}. See globalmapper_map/README.md, step 1.")


class Geometry:
    """Per-block canonical axis data (no torch needed)."""

    def __init__(self, gmdata, split):
        d = os.path.join(gmdata, split)
        idx = pd.read_csv(os.path.join(d, "index.csv"))
        self.k = dict(zip(idx.block_row, idx.k))
        with open(os.path.join(d, "geom.pkl"), "rb") as f:
            self.geom = pickle.load(f)

    def __contains__(self, row):
        return row in self.k


class GlobalMapper:
    def __init__(self, gm, ckpt, gmdata, dev):
        import torch
        np.float = float                                     # the release uses np.float
        tb = types.ModuleType("tensorboard_logger")
        tb.configure = tb.log_value = lambda *a, **k: None
        sys.modules.setdefault("tensorboard_logger", tb)
        for p in (gm, os.path.join(C.REPO, "baselines", "globalmapper_nyc")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import networkx as nx
        nx.read_gpickle = lambda p: pickle.load(open(p, "rb"))
        from graph_util import read_train_yaml
        from model import AttentionBlockGenerator_independent_cnn
        from urban_dataset import get_transform, graph2vector_processed, test_graph_transform
        opt = read_train_yaml(gm, filename="train_gnn.yaml")
        opt["N"], opt["device"] = SLOTS, str(dev)
        self.model = AttentionBlockGenerator_independent_cnn(opt, N=SLOTS)
        self.model.load_state_dict(torch.load(ckpt, map_location=dev))
        self.model.to(dev).eval()
        self.dev, self.gm, self.gmdata, self.torch = dev, gm, gmdata, torch
        self.cnn_t = get_transform(noise_range=0.0, noise_type=None, isaug=False, rescale_size=64)
        self.vec, self.tr = graph2vector_processed, test_graph_transform
        self.norm = json.load(open(os.path.join(gmdata, "norm.json")))["norm"]
        self.geo, self._data = {}, {}

    def geometry(self, split):
        if split not in self.geo:
            self.geo[split] = Geometry(self.gmdata, split)
        return self.geo[split]

    def data(self, split, row):
        from PIL import Image
        from torch_geometric.data import Data
        key = (split, row)
        if key not in self._data:
            torch = self.torch
            with open(os.path.join(self.gmdata, split, "processed", f"{self.geometry(split).k[row]}.gpickle"), "rb") as f:
                g = pickle.load(f)
            img = self.cnn_t(Image.fromarray(g.graph["binary_mask"]))
            scale = g.graph["block_scale"]
            cond = torch.cat((img, torch.tensor([math.log(scale, 20) / 2.0]).expand(img.shape)), 0)
            size, pos, feat, edge, idx, asp, long_side, shp, iou = self.vec(g)
            d = Data(x=feat, node_pos=pos, edge_index=edge, node_size=size, node_idx=idx, asp_rto=asp,
                     long_side=long_side, b_shape=shp, b_iou=iou, blockshape_latent=np.zeros(40),
                     block_scale=scale, block_condition=cond, org_binary_mask=img)
            self._data[key] = self.tr(d)
        return self._data[key]

    def batch(self, split, rows):
        from torch_geometric.data import Batch
        return Batch.from_data_list([self.data(split, r) for r in rows]).to(self.dev)

    def encode(self, split, rows):
        """Posterior mean and log-variance, numpy [n, 512]."""
        with self.torch.no_grad():
            mu, lv = self.model.encode(self.batch(split, rows))
        return mu.cpu().numpy(), lv.cpu().numpy()

    def decode(self, split, rows, z):
        """z [n, 512] -> list of dicts of slot arrays (standardised, as the release outputs them)."""
        torch = self.torch
        with torch.no_grad():
            b = self.batch(split, rows)
            n = len(rows)
            cond = self.model.cnn_encode(b.block_condition.view(n, 2, 64, 64).float())
            out = self.model.decode(torch.as_tensor(z, dtype=torch.float32, device=self.dev), cond, b.edge_index)
        ex, px, py, sx, sy, _, iou = [o.view(n, SLOTS, -1).cpu().numpy() for o in out]
        return [{"exist": ex[i, :, 0] >= 0.5, "px": px[i, :, 0], "py": py[i, :, 0], "sx": sx[i, :, 0],
                 "sy": sy[i, :, 0], "iou": iou[i, :, 0]} for i in range(n)]

    def polygons(self, split, rec, slots, trim=True):
        """Decoded slots -> building polygons in the record's local coordinates."""
        from gm_geom import graph_to_polygons
        geom = self.geometry(split).geom[rec["row"]]
        g = graph_to_polygons(slots["exist"], slots["px"], slots["py"], slots["sx"], slots["sy"], slots["iou"],
                              geom, self.norm, self.gm)
        ox, oy = origin(rec)
        g = [clean(shapely.transform(p, lambda q: np.stack([q[:, 0] - ox, oy - q[:, 1]], 1))) for p in g]
        g = [p for p in g if p is not None and p.area >= C.MIN_BLDG_M2]
        return [p for p in trim_overlaps(g) if p is not None] if trim else g


_grid = {}


def origin(rec, data=None):
    """Map coordinates (m) of the record's local origin (top-left corner)."""
    if not _grid:
        _grid.update(json.load(open(os.path.join(data or C.DATA, "info.json")))["grid"])
    return _grid["x0"] + rec["c0"] * C.CELL, _grid["y_top"] - rec["r0"] * C.CELL


AXIS_W, AXIS_H, AXIS_V = 128, 32, 1.25                       # grid along / across, range of pos_y


def axis_points(rec, geom):
    """Local coordinates of a grid in GlobalMapper's canonical space: columns follow the block's main
    axis (pos_x from -1 to 1), rows are pos_y from -AXIS_V to AXIS_V. Same warp as the release's
    inverse_warp_bldg_by_midaxis, so cell (row, col) is where a slot with that position is drawn."""
    p = np.asarray(wkb.loads(geom["midaxis"]).coords, float)[:, :2]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = (np.arange(AXIS_W) + 0.5) / AXIS_W * cum[-1]
    i = np.clip(np.searchsorted(cum, s, "right") - 1, 0, len(seg) - 1)
    tan = np.diff(p, axis=0)[i] / np.maximum(seg[i], 1e-9)[:, None]
    base = p[i] + tan * (s - cum[i])[:, None]
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
    v = ((np.arange(AXIS_H) + 0.5) / AXIS_H * 2 - 1) * AXIS_V / 2 * geom["asp"] * cum[-1]
    xy = base[None] + v[:, None, None] * nrm[None] + np.asarray(geom["center"])
    ox, oy = origin(rec)
    return xy[..., 0] - ox, oy - xy[..., 1], np.array([cum[-1] / 200.0, geom["asp"], geom["scale"] / 200.0], np.float32)


def axis_map(cond, pts):
    """cond [12, h, w] on 4 m cells -> [12, AXIS_H, AXIS_W]."""
    x, y = pts
    r, c = np.floor(y / C.CELL).astype(int), np.floor(x / C.CELL).astype(int)
    ok = (r >= 0) & (c >= 0) & (r < cond.shape[1]) & (c < cond.shape[2])
    out = np.zeros((cond.shape[0], AXIS_H, AXIS_W), np.float32)
    out[0] = 1.0
    out[:, ok] = cond[:, r[ok], c[ok]]
    return out
