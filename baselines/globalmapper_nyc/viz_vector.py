"""Vector drawing of what the NYC-trained GlobalMapper draws (no 4 m grid), for one target block of
six test windows: real footprints, recon, random x2, prior x2. Latent modes, seed and latent
statistics as in gen_nyc.py. Each rectangle is drawn shrunk by sqrt(occupancy) (filled) and unshrunk
(thin outline), as in gm_geom.graph_to_polygons.

  python viz_vector.py --data ~/genai-final/data/gm_nyc --gm ~/genai-final/third_party/GlobalMapper \
      --ckpt ~/genai-final/checkpoints/globalmapper_nyc/val_least_loss_all.pth \
      --city ~/genai-final/data/city --bldg ~/genai-final/data/nyc_bldg --out vector_examples.png
"""
import argparse
import json
import os
import pickle
import sys
import types

import numpy as np
import pandas as pd

np.float = float
_tb = types.ModuleType("tensorboard_logger")
_tb.configure = lambda *a, **k: None
_tb.log_value = lambda *a, **k: None
sys.modules["tensorboard_logger"] = _tb
import networkx as nx  # noqa: E402

nx.read_gpickle = lambda p: pickle.load(open(p, "rb"))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gm_geom import load_norm, unstandardise  # noqa: E402

WINDOWS = [84112, 16148, 27493, 9498, 43433, 43146]


def rects(exist, px, py, sx, sy, iou, geom, norm, gm_path):
    """As gm_geom.graph_to_polygons, but returns (shrunk, unshrunk) polygons."""
    import shapely.affinity as sa
    from shapely import wkb
    sys.path.insert(0, gm_path)
    from geo_utils import inverse_warp_bldg_by_midaxis
    keep = np.asarray(exist) > 0
    if not keep.any():
        return [], []
    x, y, a, b = unstandardise(np.asarray(px)[keep], np.asarray(py)[keep], np.asarray(sx)[keep],
                               np.asarray(sy)[keep], norm)
    pos = np.stack([np.clip(x, -1, 1), y], 1)
    size = np.stack([np.clip(a, 1e-3, None), np.clip(b, 1e-3, None)], 1)
    med = wkb.loads(geom["midaxis"])
    polys, _, _ = inverse_warp_bldg_by_midaxis(pos.copy(), size.copy(), med, geom["asp"])
    cx, cy = geom["center"]
    shr, full = [], []
    for p, q in zip(polys, np.asarray(iou)[keep]):
        s = float(np.sqrt(np.clip(q, 0.05, 1.0)))
        full.append(sa.translate(p, cx, cy))
        shr.append(sa.translate(sa.scale(p, s, s, origin="centroid"), cx, cy))
    return shr, full


def main():
    ap = argparse.ArgumentParser()
    for k in ["data", "gm", "ckpt", "city", "bldg", "out"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, a.gm)
    import torch
    from torch_geometric.data import Batch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely import wkb

    from graph_util import read_train_yaml
    from model import AttentionBlockGenerator_independent_cnn
    from urban_dataset import UrbanGraphDataset, get_transform, test_graph_transform

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    opt = read_train_yaml(a.gm, filename="train_gnn.yaml")
    N = opt["template_width"] * opt["template_height"]
    opt["N"] = N
    dev = torch.device("cuda:0")
    opt["device"] = str(dev)
    model = AttentionBlockGenerator_independent_cnn(opt, N=N)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.to(dev).eval()
    norm = load_norm(a.data)
    cnn_t = get_transform(noise_range=10.0, noise_type="gaussian", isaug=False, rescale_size=64)

    def dataset(split):
        class Fixed(UrbanGraphDataset):
            _n = None

            def len(self):
                if self._n is None:
                    self._n = UrbanGraphDataset.len(self)
                return self._n
        ds = Fixed(os.path.join(a.data, split), transform=test_graph_transform, cnn_transform=cnn_t)
        idx = pd.read_csv(os.path.join(a.data, split, "index.csv"))
        geom = pickle.load(open(os.path.join(a.data, split, "geom.pkl"), "rb"))
        return ds, dict(zip(idx.block_row, idx.k)), dict(zip(idx.block_row, idx.n_bldg)), geom

    @torch.no_grad()
    def encode(ds, ks):
        b = Batch.from_data_list([ds[int(k)] for k in ks]).to(dev)
        return model.encode(b)

    @torch.no_grad()
    def decode(ds, k, z):
        d = ds[int(k)]
        S = z.shape[0]
        b = Batch.from_data_list([d] * S).to(dev)
        cond = model.cnn_encode(b.block_condition.view(S, 2, 64, 64))
        exist, px, py, sx, sy, shp, iou = model.decode(z, cond, b.edge_index)
        ex = torch.ge(exist, 0.5).view(S, N).cpu().numpy()
        f = lambda t: t.view(S, N).cpu().numpy()
        return ex, f(px), f(py), f(sx), f(sy), f(iou)

    va, _, _, _ = dataset("val")
    zs = []
    for s in range(0, len(va), 200):
        mu, lv = encode(va, range(s, min(s + 200, len(va))))
        zs.append(model.reparameterize(mu, lv).cpu().numpy())
    zs = np.concatenate(zs)
    z_mean, z_std = zs.mean(0), zs.std(0)

    te, k_of, nb_of, geom = dataset("test")
    blk = np.load(os.path.join(a.city, "block.npy"), mmap_mode="r")
    samples = pd.read_parquet(os.path.join(a.city, "samples.parquet")).set_index("sample_id")
    blocks = pd.read_parquet(os.path.join(a.city, "blocks.parquet"), columns=["block_id", "geometry"])
    bld = pd.concat([pd.read_parquet(os.path.join(a.bldg, f), columns=["block_id", "geometry"])
                     for f in sorted(os.listdir(a.bldg)) if f.endswith(".parquet")])
    by_block = bld.groupby("block_id")["geometry"].apply(list).to_dict()

    rows = []
    for sid in WINDOWS:
        s = samples.loc[sid]
        r0, c0 = int(s.row0), int(s.col0)
        wb = np.asarray(blk[r0:r0 + 128, c0:c0 + 128]).astype(np.int64)
        tg = [int(t) for t in s.targets if int(t) in k_of]
        t = max(tg, key=lambda b: wkb.loads(blocks.geometry.iloc[b]).area)
        bgeom = wkb.loads(blocks.geometry.iloc[t])
        real = [wkb.loads(g) for g in by_block.get(blocks.block_id.iloc[t], [])]
        panels = [("real footprints", real, None)]
        mu, lv = encode(te, [k_of[t]])
        z = model.reparameterize(mu.repeat(2, 1), lv.repeat(2, 1))
        out = decode(te, k_of[t], z[:1])
        panels.append(("recon",) + rects(*(o[0] for o in out), geom[t], norm, a.gm))
        zr = torch.from_numpy(rng.normal(z_mean, z_std, (2, len(z_mean))).astype(np.float32)).to(dev)
        out = decode(te, k_of[t], zr)
        for j in range(2):
            panels.append((f"random {j + 1}",) + rects(*(o[j] for o in out), geom[t], norm, a.gm))
        ctx = [b for b in np.unique(wb) if b >= 0 and b not in s.targets and b in k_of and nb_of[b] > 0]
        cen = lambda b: np.array(wkb.loads(blocks.geometry.iloc[b]).centroid.coords[0])
        d = np.array([np.linalg.norm(cen(b) - cen(t)) for b in ctx])
        pick = np.argsort(d)[:a.k]
        w = 1.0 / np.maximum(d[pick], 1.0)
        w = w / w.sum()
        mu, _ = encode(te, [k_of[ctx[j]] for j in pick])
        zp = (torch.from_numpy(w.astype(np.float32)).to(dev)[:, None] * mu).sum(0)
        noise = rng.normal(0, 0.5 * z_std, (2, len(z_mean))).astype(np.float32)
        out = decode(te, k_of[t], zp[None] + torch.from_numpy(noise).to(dev))
        for j in range(2):
            panels.append((f"prior {j + 1}",) + rects(*(o[j] for o in out), geom[t], norm, a.gm))
        rows.append((sid, t, bgeom, panels))
        print(sid, t, [(p[0], len(p[1])) for p in panels], flush=True)

    fig, ax = plt.subplots(len(rows), 6, figsize=(6 * 3.2, len(rows) * 3.2))
    for i, (sid, t, bgeom, panels) in enumerate(rows):
        x0, y0, x1, y1 = bgeom.bounds
        m = 0.08 * max(x1 - x0, y1 - y0)
        for j, (name, polys, full) in enumerate(panels):
            A = ax[i, j]
            for g in (bgeom.geoms if bgeom.geom_type == "MultiPolygon" else [bgeom]):
                A.fill(*g.exterior.xy, color="#e6e6e6", ec="#888888", lw=0.8)
            for p in full or []:
                A.plot(*p.exterior.xy, color="#1f5fbf", lw=0.4, alpha=0.6)
            for p in polys:
                for g in (p.geoms if p.geom_type == "MultiPolygon" else [p]):
                    A.fill(*g.exterior.xy, color="#d9822b", ec="k", lw=0.3)
            A.set_xlim(x0 - m, x1 + m)
            A.set_ylim(y0 - m, y1 + m)
            A.set_aspect("equal")
            A.set_xticks([])
            A.set_yticks([])
            A.set_title(f"{name}: {len(polys)} bldgs", fontsize=9)
            if j == 0:
                A.set_ylabel(f"window #{sid}", fontsize=9)
    fig.suptitle("NYC-trained GlobalMapper, vector output (orange: rectangle shrunk to predicted occupancy; "
                 "blue outline: unshrunk rectangle; grey: block)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(a.out, dpi=90)


if __name__ == "__main__":
    main()
