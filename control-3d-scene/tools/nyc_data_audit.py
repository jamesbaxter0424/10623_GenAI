"""Audit of the NYC dataset for the generation task (what can go wrong downstream).

  python tools/nyc_data_audit.py --root DATA
Sections:
  A  FAR recoverable from the 10-class map (error floor of any FAR metric on generated maps)
  B  how well one node label (use x height class) describes the block's cells
  C  row houses merged by the 4 m raster (building count / size metrics)
  D  water drawn as street
  E  overlap between windows (effective dataset size)
  F  invalid (flagged) blocks shown as context
  G  split composition (borough, rare classes in the test subset)
  H  footprint FAR vs PLUTO by borough
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import ndimage

N = 128
CELL_M2 = 16.0


def per_block_class_counts(cls, block, nb, rows=500):
    out = np.zeros((nb, 10), np.int64)
    for r in range(0, cls.shape[0], rows):
        b = np.asarray(block[r:r + rows]).ravel()
        c = np.asarray(cls[r:r + rows]).ravel().astype(np.int64)
        k = b >= 0
        out += np.bincount(b[k].astype(np.int64) * 10 + c[k], minlength=nb * 10).reshape(nb, 10)
    return out


def section_a(cnt, blk):
    v = blk["valid"].to_numpy()
    area = blk["area_m2"].to_numpy()
    bld = cnt[:, 2:10]
    h_cells = bld[:, 0:4] + bld[:, 4:8]                       # cells per height bin
    cov_map = h_cells.sum(1) * CELL_M2 / area
    print("\n## A. FAR recoverable from the class map (valid blocks)")
    print(f"coverage map vs target: corr {np.corrcoef(cov_map[v], blk.coverage[v])[0, 1]:.3f}, "
          f"MAE {np.abs(cov_map[v] - blk.coverage[v]).mean():.3f}")
    far = blk["far_fp"].to_numpy()
    for name, floors in [("bin midpoints 1.5/4/9/20", np.array([1.5, 4, 9, 20])),
                         ("best fit (least squares)", None)]:
        X = h_cells[v] * CELL_M2 / area[v, None]
        if floors is None:
            floors, *_ = np.linalg.lstsq(X, far[v], rcond=None)
        pred = X @ floors
        err = np.abs(pred - far[v])
        rel = err / np.maximum(far[v], 0.1)
        print(f"  {name:28s} floors {np.round(floors, 1)}: corr {np.corrcoef(pred, far[v])[0, 1]:.3f}, "
              f"MAE {err.mean():.3f}, median rel. err {np.median(rel):.1%}")
        by = pd.DataFrame({"h": blk.hclass[v].to_numpy(), "err": err, "rel": rel}).groupby("h")
        print("    by block height class (MAE / median rel.):",
              {int(h): (round(g.err.mean(), 2), f"{g.rel.median():.0%}") for h, g in by})


def section_b(cnt, blk, labels):
    v = (blk["valid"].to_numpy()) & (labels >= 2)
    bld = cnt[v, 2:10]
    lab = labels[v] - 2
    match = bld[np.arange(len(lab)), lab] / np.maximum(bld.sum(1), 1)
    h_cells = bld[:, 0:4] + bld[:, 4:8]
    lab_h = lab % 4
    taller = (h_cells * (np.arange(4)[None] > lab_h[:, None])).sum(1) / np.maximum(h_cells.sum(1), 1)
    w = bld.sum(1)
    print("\n## B. One node label vs the block's building cells (valid blocks with buildings)")
    print(f"share of building cells whose class equals the node label: median {np.median(match):.0%}, "
          f"cell-weighted {np.average(match, weights=w):.0%}")
    print(f"blocks where < 50% of building cells match the label: {np.mean(match < 0.5):.1%}")
    print(f"share of building cells TALLER than the node label: cell-weighted {np.average(taller, weights=w):.0%}; "
          f"blocks with any 13+ cells but label below 13+: {np.mean((h_cells[:, 3] > 0) & (lab_h < 3)):.1%}")


def section_c(ds_windows, cls, block, blk, n=400, seed=0):
    rng = np.random.default_rng(seed)
    nb = blk["n_bldg"].to_numpy()
    rows = []
    for _, r in ds_windows.iloc[rng.choice(len(ds_windows), n, replace=False)].iterrows():
        r0, c0 = int(r.row0), int(r.col0)
        c = np.asarray(cls[r0:r0 + N, c0:c0 + N]); b = np.asarray(block[r0:r0 + N, c0:c0 + N])
        for t in r.targets:
            m = (b == t) & (c >= 2)
            if nb[t] > 0:
                k = ndimage.label(m)[1]
                rows.append((nb[t], k, blk.hclass.iat[t]))
    d = pd.DataFrame(rows, columns=["n_bldg", "components", "h"])
    d["ratio"] = d.n_bldg / d.components.clip(lower=1)
    print("\n## C. Buildings merged by the 4 m raster (target blocks of 400 test windows)")
    print(f"real buildings per block median {d.n_bldg.median():.0f}; connected building patches per block "
          f"median {d.components.median():.0f}; buildings per patch median {d.ratio.median():.1f}")
    print("  buildings per patch by height class:",
          {int(h): round(g.ratio.median(), 1) for h, g in d.groupby("h")})
    print(f"  blocks where >= 5 real buildings become one patch on average: {np.mean(d.ratio >= 5):.0%}")


def section_d(samples, water, cls):
    print("\n## D. Water (drawn as class 0 = street)")
    res = {}
    for split, s in samples.groupby("split"):
        share = []
        for _, r in s.sample(min(1500, len(s)), random_state=0).iterrows():
            w = np.asarray(water[int(r.row0):int(r.row0) + N, int(r.col0):int(r.col0) + N])
            c = np.asarray(cls[int(r.row0):int(r.row0) + N, int(r.col0):int(r.col0) + N])
            share.append((w.mean(), (w & (c == 0)).sum() / max((c == 0).sum(), 1)))
        share = np.array(share)
        res[split] = (f"windows with >10% water {np.mean(share[:, 0] > 0.1):.0%}",
                      f"water share of 'street' cells {share[:, 1].mean():.0%}")
    for k, v in res.items():
        print(f"  {k:5s}: {v[0]}, {v[1]}")


def section_e(samples, shape):
    print("\n## E. Overlap between windows")
    for split, s in samples.groupby("split"):
        cover = np.zeros((shape[0] // 8 + 17, shape[1] // 8 + 17), np.int32)   # 32 m grid
        for r0, c0 in zip(s.row0 // 8, s.col0 // 8):
            cover[r0:r0 + 16, c0:c0 + 16] += 1
        covered = cover > 0
        print(f"  {split:5s}: {len(s)} windows, each 32 m cell is seen by {cover[covered].mean():.1f} windows on average "
              f"-> {covered.sum() * 1024 / 1e6:.0f} km2 of distinct area "
              f"(= {covered.sum() / 256:.0f} non-overlapping windows)")


def section_f(samples, block, blk, n=500):
    print("\n## F. Context blocks that failed the validity checks")
    flags = [c for c in blk.columns if c.startswith("flag_")]
    rng = np.random.default_rng(0)
    tot, inval, fl = 0, 0, pd.Series(0, index=flags)
    for _, r in samples.iloc[rng.choice(len(samples), n, replace=False)].iterrows():
        b = np.asarray(block[int(r.row0):int(r.row0) + N, int(r.col0):int(r.col0) + N])
        ids, cnt = np.unique(b[b >= 0], return_counts=True)
        ids = ids[cnt >= 20]
        tot += len(ids)
        bad = ~blk.valid.to_numpy()[ids]
        inval += bad.sum()
        fl += blk.iloc[ids[bad]][flags].sum()
    print(f"  {inval / tot:.1%} of context nodes are invalid blocks; flags among them: "
          + ", ".join(f"{k[5:]} {int(v)}" for k, v in fl.items() if v))


def section_g(samples, blk, subset_ids):
    print("\n## G. Split composition")
    s = samples.copy()
    s["boro"] = [blk.boro.iat[t[0]] for t in s.targets]
    print(pd.crosstab(s.split, s.boro, normalize="index").round(2).to_string())
    lab_h = blk.hclass.to_numpy()
    sub = s.set_index("sample_id").loc[subset_ids[:200]]
    tg = np.concatenate(sub.targets.to_numpy())
    allt = np.concatenate(s[s.split == "test"].targets.to_numpy())
    print("  target blocks per height class in test subset (first 200 windows):",
          dict(zip(*np.unique(lab_h[tg], return_counts=True))),
          "| full test:", dict(zip(*np.unique(lab_h[allt], return_counts=True))))


def section_h(blk):
    print("\n## H. Footprint FAR (training target) vs PLUTO, valid blocks")
    v = blk[blk.valid & (blk.far_pluto > 0.05)]
    r = v.far_fp / v.far_pluto
    print("  median far_fp / far_pluto by borough:", r.groupby(v.boro).median().round(2).to_dict())
    print("  share with |ratio - 1| > 30%:", (abs(r - 1) > 0.3).groupby(v.boro).mean().round(2).to_dict())


def main(root):
    cls = np.load(os.path.join(root, "cls.npy"), mmap_mode="r")
    block = np.load(os.path.join(root, "block.npy"), mmap_mode="r")
    water = np.load(os.path.join(root, "water.npy"), mmap_mode="r")
    blk = pd.read_parquet(os.path.join(root, "blocks.parquet"),
                          columns=["block_id", "valid", "boro", "hclass", "is_res", "coverage", "far_fp",
                                   "far_pluto", "area_m2", "n_bldg"] +
                                  [c for c in pd.read_parquet(os.path.join(root, "blocks.parquet")).columns
                                   if c.startswith("flag_")])
    blk["n_bldg"] = blk["n_bldg"].fillna(0).astype(int)
    samples = pd.read_parquet(os.path.join(root, "samples.parquet"))
    samples = samples[~samples.ctx_other_split]
    subset = json.load(open(os.path.join(root, "test_subset.json")))["sample_ids"]
    h = blk.hclass.to_numpy(); res = blk.is_res.to_numpy().astype(bool)
    labels = np.where(h < 0, 1, np.where(res, 2 + h, 6 + h))

    cnt = per_block_class_counts(cls, block, len(blk))
    section_a(cnt, blk)
    section_b(cnt, blk, labels)
    section_c(samples[samples.split == "test"], cls, block, blk)
    section_d(samples, water, cls)
    section_e(samples, cls.shape)
    section_f(samples[samples.split == "train"], block, blk)
    section_g(samples, blk, subset)
    section_h(blk)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    main(p.parse_args().root)
