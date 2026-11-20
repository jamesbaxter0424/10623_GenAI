"""Evaluate one model's generated maps (or the real map as the oracle) on the NYC test set.

  # a model: a folder of <sample_id>.npz with gen [S,128,128] uint8, gt, mask
  python evaluate.py --pred_dir PATH --name liu_baseline [--n 1000]
  # the oracle row: the real map scored with the same code
  python evaluate.py --oracle --name oracle
  # paired comparison of two evaluated models
  python evaluate.py --compare results/A results/B

Writes results/<name>/{per_block.csv, pairs.csv, footprint.csv, summary.json, summary.md}.
The CNN two-sample test is a separate script (c2st.py); its result is merged when present.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import config as C
from condition import block_values, hit, targets
from context import block_context, neighbours, window_street
from data import City, test_order
from diversity import block_diversity
from footprint import MEASURES, block_regions
from stats import ClusterBootstrap, paired_difference


# ---------------------------------------------------------------------- per-window records
def score_window(city, sid, gens):
    """gens: [S,128,128] classes. Returns lists of block rows, pair rows, footprint rows, diversity rows."""
    cls, blk, water, tgts = city.window(sid)
    street = window_street(blk, water)
    f = city.floors
    rows, pairs, fps, divs = [], [], [], []
    for t in tgts:
        inside = blk == t
        tv = targets(city, t)
        real = block_values(cls[inside], f)
        rctx = block_context(cls, inside, street, f)
        nbrs = [(n, city.mean_floors_real(n), city.street_wall_real(n)) for n in neighbours(city, t, tgts)]
        reg = block_regions(cls, inside)
        for m in MEASURES:
            fps += [(sid, t, -1, m, v) for v in reg[m]]
        divs.append({"sample_id": sid, "block": t, "diversity": block_diversity(gens[:, inside])})
        for s in range(gens.shape[0]):
            g = gens[s]
            gv = block_values(g[inside], f)
            gctx = block_context(g, inside, street, f)
            rows.append({"sample_id": sid, "block": t, "s": s, **tv,
                         "cov": gv["cov"], "far": gv["far"], "hcls": gv["hcls"], "res": gv["res"],
                         "sw": gctx["sw"], "H": gctx["H"],
                         "r_cov": real["cov"], "r_far": real["far"], "r_hcls": real["hcls"], "r_res": real["res"],
                         "r_sw": rctx["sw"], "r_H": rctx["H"]})
            for n, hn, swn in nbrs:
                pairs.append({"sample_id": sid, "block": t, "s": s, "nbr": n,
                              "d_gen": gctx["H"] - hn, "d_real": rctx["H"] - hn,
                              "swd_gen": abs(gctx["sw"] - swn), "swd_real": abs(rctx["sw"] - swn)})
            reg = block_regions(g, inside)
            for m in MEASURES:
                fps += [(sid, t, s, m, v) for v in reg[m]]
    return rows, pairs, fps, divs


# ---------------------------------------------------------------------- per-block table
def per_block(rows):
    d = pd.DataFrame(rows)
    d["cov_err"] = (d["cov"] - d["t_cov"]).abs()
    d["far_err"] = (d["far"] - d["t_far"]).abs()
    d["far_rel_err"] = d["far_err"] / d["t_far"].clip(lower=0.1)
    d["height_ok"] = np.where(d["t_hcls"] >= 0, (d["hcls"] == d["t_hcls"]).astype(float), np.nan)
    d["res_err"] = (d["res"] - d["t_res"]).abs()
    both = d["res"].notna() & d["t_res"].notna()
    d["use_ok"] = np.where(both, ((d["res"] >= 0.5) == (d["t_res"] >= 0.5)).astype(float), np.nan)
    d["hit"] = [float(hit(a, b) and hit(c, e)) for a, b, c, e in zip(d["far"], d["t_far"], d["cov"], d["t_cov"])]
    keep = ["cov_err", "far_err", "far_rel_err", "height_ok", "res_err", "use_ok", "hit",
            "cov", "far", "sw", "H", "t_cov", "t_far", "t_hcls", "t_res", "r_cov", "r_far", "r_sw", "r_H"]
    return d.groupby(["sample_id", "block"], as_index=False)[keep].mean()


# ---------------------------------------------------------------------- summary statistic
def wd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return wasserstein_distance(a, b) if len(a) and len(b) else np.nan


def summary_stat(t):
    pb, pr, fp, dv = t["blocks"], t["pairs"], t["fp"], t["div"]
    out = {
        "cov_err": pb["cov_err"].mean(),
        "far_err": pb["far_err"].mean(),
        "far_rel_err_median": pb["far_rel_err"].median(),
        "height_acc": pb["height_ok"].mean(),
        "res_err": pb["res_err"].mean(),
        "use_acc": pb["use_ok"].mean(),
        "hit_rate": pb["hit"].mean(),
        "boundary_wd": wd(pr["d_gen"], pr["d_real"]),
        "street_wall_wd": wd(pb["sw"], pb["r_sw"]),
        "street_wall_nbr_wd": wd(pr["swd_gen"], pr["swd_real"]),
        "diversity": dv["diversity"].mean(),
    }
    fps = []
    for m in MEASURES:
        g = fp.loc[(fp["measure"] == m) & (fp["s"] >= 0), "value"].to_numpy()
        r = fp.loc[(fp["measure"] == m) & (fp["s"] < 0), "value"].to_numpy()
        sd = np.std(r) if len(r) > 1 else np.nan
        out[f"fp_{m}"] = wd(g, r) / sd if sd and sd > 0 else np.nan
        fps.append(out[f"fp_{m}"])
    out["footprint"] = float(np.nanmean(fps))
    return out


ROWS = [  # (key, label, better)
    ("far_err", "FAR err.", "lower"), ("far_rel_err_median", "FAR rel. err. (median)", "lower"),
    ("cov_err", "Cov. err.", "lower"), ("height_acc", "Height acc.", "higher"),
    ("res_err", "Res. err.", "lower"), ("use_acc", "Use acc. (text only)", "higher"),
    ("hit_rate", "Hit rate (FAR and cov. within 10%)", "higher"),
    ("boundary_wd", "Boundary (WD of height difference)", "lower"),
    ("street_wall_wd", "Street wall (WD)", "lower"),
    ("street_wall_nbr_wd", "Street wall vs neighbour (WD)", "lower"),
    ("footprint", "Footprint (mean normalised WD)", "lower"),
    ("fp_area", "  region area", "lower"), ("fp_convexity", "  region convexity", "lower"),
    ("fp_nn_dist", "  nearest-region distance", "lower"), ("fp_density", "  regions per ha", "lower"),
    ("diversity", "Diversity", "-"),
]


def write_summary(out_dir, name, res, info):
    c2st_f = os.path.join(out_dir, "c2st.json")
    c2st = json.load(open(c2st_f)) if os.path.exists(c2st_f) else None
    json.dump({"name": name, **info, "metrics": res, "c2st": c2st}, open(os.path.join(out_dir, "summary.json"), "w"),
              indent=1)
    lines = [f"# {name}", "", f"{info['n_windows']} windows, {info['n_blocks']} target blocks, "
             f"{info['n_samples']} sample(s) per window, {info['n_pairs']} block-neighbour pairs. "
             f"95% block-cluster bootstrap intervals ({info['n_boot']} draws).", "",
             "| Metric | Value | 95% CI | Better |", "|---|---|---|---|"]
    for k, label, better in ROWS:
        p, lo, hi = res[k]
        lines.append(f"| {label} | {p:.3f} | [{lo:.3f}, {hi:.3f}] | {better} |")
    if c2st:
        lines.append(f"| CNN acc. (two-sample test) | {c2st['acc_mean']:.3f} | "
                     f"[{c2st['acc_min']:.3f}, {c2st['acc_max']:.3f}] over {c2st['n_splits']} splits | -> 0.5 |")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------- main
def load_gens(pred_dir):
    files = glob.glob(os.path.join(pred_dir, "**", "*.npz"), recursive=True)
    return {int(os.path.splitext(os.path.basename(f))[0]): f for f in files}


def evaluate(city, ids, gen_fn, name, n_boot=C.N_BOOT, check_gt=None):
    t0 = time.time()
    rows, pairs, fps, divs = [], [], [], []
    for i, sid in enumerate(ids):
        gens, gt = gen_fn(sid)
        if check_gt is not None and gt is not None:
            real = city.window(sid)[0]
            if not np.array_equal(gt.astype(np.int64), real):
                raise ValueError(f"window {sid}: gt in the prediction file does not match the dataset")
        r, p, f, d = score_window(city, sid, gens)
        rows += r; pairs += p; fps += f; divs += d
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(ids)} windows ({time.time() - t0:.0f}s)", flush=True)
    tables = {"blocks": per_block(rows), "pairs": pd.DataFrame(pairs),
              "fp": pd.DataFrame(fps, columns=["sample_id", "block", "s", "measure", "value"]),
              "div": pd.DataFrame(divs)}
    res = ClusterBootstrap(tables).run(summary_stat, n_boot=n_boot)
    out_dir = os.path.join(C.RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    tables["blocks"].to_csv(os.path.join(out_dir, "per_block.csv"), index=False)
    tables["pairs"].to_csv(os.path.join(out_dir, "pairs.csv"), index=False)
    tables["fp"].to_csv(os.path.join(out_dir, "footprint.csv"), index=False)
    info = {"n_windows": len(ids), "n_blocks": int(tables["blocks"]["block"].nunique()),
            "n_samples": int(pd.DataFrame(rows)["s"].max() + 1), "n_pairs": len(pairs), "n_boot": n_boot,
            "floors_per_height_class": city.floors[2:6].round(2).tolist(), "seconds": round(time.time() - t0)}
    print(write_summary(out_dir, name, res, info))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir")
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--name")
    p.add_argument("--n", type=int, default=C.N_TEST)
    p.add_argument("--n_boot", type=int, default=C.N_BOOT)
    p.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    a = p.parse_args()

    if a.compare:
        pa, pb = (pd.read_csv(os.path.join(d, "per_block.csv")) for d in a.compare)
        for m in ["far_err", "cov_err", "height_ok", "res_err", "hit"]:
            print(paired_difference(pa, pb, m, n_boot=a.n_boot))
        return

    city = City()
    order = test_order(city)[:a.n]
    if a.oracle:
        def gen_fn(sid):
            return city.window(sid)[0][None], None
        evaluate(city, order, gen_fn, a.name or "oracle", a.n_boot)
        return
    files = load_gens(a.pred_dir)
    ids = [s for s in order if s in files]
    missing = len(order) - len(ids)
    if missing:
        print(f"warning: {missing} of the first {len(order)} test windows have no prediction file", file=sys.stderr)

    def gen_fn(sid):
        z = np.load(files[sid])
        g = z["gen"]
        return (g[None] if g.ndim == 2 else g).astype(np.int64), z["gt"] if "gt" in z else None
    evaluate(city, ids, gen_fn, a.name or os.path.basename(os.path.normpath(a.pred_dir)), a.n_boot, check_gt=True)


if __name__ == "__main__":
    main()
