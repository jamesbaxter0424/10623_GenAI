"""Generate the target blocks of the first 1,000 test windows with the NYC-trained GlobalMapper
and write them in our evaluation format (<sample_id>.npz with gen [S,128,128], gt).

  python gen_nyc.py --data ~/genai-final/data/gm_nyc --gm ~/genai-final/third_party/GlobalMapper \
      --ckpt ~/genai-final/checkpoints/globalmapper_nyc/val_least_loss_all.pth \
      --city ~/genai-final/data/city --mode random --out ~/genai-final/generated/globalmapper_nyc_random

Modes:
  random  the release's test.py with is_reconstruct = False: z ~ Normal(z_mean, z_std), the mean and
          std of encoder samples over the validation split (the release reads them from a
          "sample_stats" file written by a reconstruction run).
  prior   the paper's sparse-prior generation (Sec. 4.4; not in the release): z = distance-weighted
          sum of the encoder means of the k = 5 nearest non-target blocks of the window that have
          buildings, plus Normal(0, (0.5 z_std)^2) noise for diversity (the paper does not give the
          noise scale). Falls back to random if the window has no such block.
  recon   reference only: each target block's own encoding (the release's default test mode).
Each target block is decoded with its own block mask (the condition). Generated rectangles are
mapped back with the release's inverse transform, shrunk to the predicted occupancy, and drawn
inside the target block on the real map: target cells become open space (class 1), building cells
class 2. GlobalMapper predicts no height or use, so only class-free metrics are meaningful.
Buildings outside the target block are cut off (GlobalMapper's Out-Block).
"""
import argparse
import json
import os
import pickle
import sys
import time
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
from gm_geom import graph_to_polygons, load_norm, rasterise  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ["data", "gm", "ckpt", "city", "out"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--mode", choices=["random", "prior", "recon"], default="random")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, a.gm)
    import torch
    from torch_geometric.data import Batch

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
        class Fixed(UrbanGraphDataset):              # the release re-scans the folder on every len()
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
        mu, log_var = model.encode(b)
        return mu, log_var

    @torch.no_grad()
    def decode(ds, k, z):
        d = ds[int(k)]
        S = z.shape[0]
        b = Batch.from_data_list([d] * S).to(dev)
        cond = model.cnn_encode(b.block_condition.view(S, 2, 64, 64))
        exist, px, py, sx, sy, shp, iou = model.decode(z, cond, b.edge_index)
        ex = torch.ge(exist, 0.5).view(S, N).cpu().numpy()           # raw output >= 0.5, as test.py
        f = lambda t: t.view(S, N).cpu().numpy()
        return ex, f(px), f(py), f(sx), f(sy), f(iou)

    # latent statistics over the validation split (the release's sample_stats)
    va, _, _, _ = dataset("val")
    zs = []
    for s in range(0, len(va), 200):
        mu, lv = encode(va, range(s, min(s + 200, len(va))))
        zs.append(model.reparameterize(mu, lv).cpu().numpy())
    zs = np.concatenate(zs)
    z_mean, z_std = zs.mean(0), zs.std(0)
    print(f"latent stats from {len(zs)} validation blocks", flush=True)

    te, k_of, nb_of, geom = dataset("test")
    grid = json.load(open(os.path.join(a.city, "grid.json")))
    cls = np.load(os.path.join(a.city, "cls.npy"), mmap_mode="r")
    blk = np.load(os.path.join(a.city, "block.npy"), mmap_mode="r")
    samples = pd.read_parquet(os.path.join(a.city, "samples.parquet")).set_index("sample_id")
    order = json.load(open(os.path.join(a.city, "test_subset.json")))["sample_ids"][:a.n]
    blocks_c = pd.read_parquet(os.path.join(a.city, "blocks.parquet"), columns=["geometry"])
    from shapely import wkb
    cent = {}

    def centroid(row):
        if row not in cent:
            c = wkb.loads(blocks_c.geometry.iloc[row]).centroid
            cent[row] = np.array([c.x, c.y])
        return cent[row]

    out_dir = os.path.join(a.out, "Generated")
    os.makedirs(out_dir, exist_ok=True)
    stats = {"targets": 0, "missing_graph": 0, "prior_fallback": 0, "buildings": 0, "cut_cells": 0}
    t0 = time.time()
    for i, sid in enumerate(order):
        s = samples.loc[sid]
        r0, c0 = int(s.row0), int(s.col0)
        real = np.asarray(cls[r0:r0 + 128, c0:c0 + 128]).astype(np.uint8)
        wb = np.asarray(blk[r0:r0 + 128, c0:c0 + 128]).astype(np.int64)
        gens = np.repeat(real[None], a.samples, 0)
        tg = [int(t) for t in s.targets]
        for t in tg:
            stats["targets"] += 1
            inside = wb == t
            gens[:, inside] = 1
            if t not in k_of:
                stats["missing_graph"] += 1
                continue
            S = a.samples
            if a.mode == "recon":
                mu, lv = encode(te, [k_of[t]])
                z = model.reparameterize(mu.repeat(S, 1), lv.repeat(S, 1))
            else:
                z = torch.from_numpy(rng.normal(z_mean, z_std, (S, len(z_mean))).astype(np.float32)).to(dev)
                if a.mode == "prior":
                    ctx = [b for b in np.unique(wb) if b >= 0 and b not in tg and b in k_of and nb_of[b] > 0]
                    if ctx:
                        d = np.array([np.linalg.norm(centroid(b) - centroid(t)) for b in ctx])
                        pick = np.argsort(d)[:a.k]
                        w = 1.0 / np.maximum(d[pick], 1.0)
                        w = w / w.sum()
                        mu, _ = encode(te, [k_of[ctx[j]] for j in pick])
                        zp = (torch.from_numpy(w.astype(np.float32)).to(dev)[:, None] * mu).sum(0)
                        noise = rng.normal(0, 0.5 * z_std, (S, len(z_mean))).astype(np.float32)
                        z = zp[None] + torch.from_numpy(noise).to(dev)
                    else:
                        stats["prior_fallback"] += 1
            ex, px, py, sx, sy, iou = decode(te, k_of[t], z)
            for j in range(S):
                polys = graph_to_polygons(ex[j], px[j], py[j], sx[j], sy[j], iou[j], geom[t], norm, a.gm)
                stats["buildings"] += len(polys)
                m = rasterise(polys, r0, c0, grid["x0"], grid["y_top"], grid["cell_m"])
                stats["cut_cells"] += int((m & ~inside).sum())
                gens[j, m & inside] = 2
        np.savez_compressed(os.path.join(out_dir, f"{sid}.npz"), gen=gens, gt=real)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(order)} windows ({time.time() - t0:.0f}s)", flush=True)
    stats["seconds"] = round(time.time() - t0)
    json.dump(stats, open(os.path.join(a.out, "gen_stats.json"), "w"), indent=1)
    print(stats, flush=True)


if __name__ == "__main__":
    main()
