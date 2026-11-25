"""Score predictions and compare methods.

  python -m common.evaluate --pred out/rule/test1000_oracle.jsonl --name rule_oracle
  python -m common.evaluate --pred out/x/test1000_stage1.jsonl --name x_stage1 --input out/stage1/nyc_c3.pkl
  python -m common.evaluate --real --name real                  # the metrics on the real buildings
  python -m common.evaluate --compare rule_oracle diffusion_oracle globalmapper_oracle set_oracle

--input is the map the method was given: "oracle" (the real map) or a stage1_input.py file.
Blocks without a prediction are scored as empty (and counted in "missing").
Writes results/<name>/summary.json and summary.md; per_block.csv stays local (git-ignored).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from .records import input_maps, load, polys, read_pred, stage1_maps

NAMES = [("count_err", "Building count error (relative, per block)"), ("wd_count", "Building count (WD)"),
         ("wd_area", "Footprint area (WD)"), ("wd_width", "Width (WD)"), ("wd_length", "Length (WD)"),
         ("wd_height", "Height (WD)"), ("wd_dz", "Height step between touching buildings (WD)"),
         ("touch_err", "Share of touching buildings (abs. error)"), ("overlap", "Overlapping area"),
         ("outside", "Area outside the block"), ("coverage_err", "Coverage error (relative)"),
         ("volume_err", "Volume error (relative)"), ("map_iou", "Agreement with the input map, IoU (higher better)"),
         ("hclass_acc", "Height class agrees with the input map (higher better)"),
         ("score", "Mean score used to pick checkpoints")]


def bootstrap(real, preds, n, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(real), len(real))
        vals.append(M.summarise([real[k] for k in i], [preds[k] for k in i]))
    return {k: [float(np.nanpercentile([v[k] for v in vals], q)) for q in (2.5, 97.5)] for k, _ in NAMES}


def run(a):
    recs = load(a.subset, a.data)
    maps = stage1_maps(a.input)
    pred = None if a.real else read_pred(a.pred)
    real, preds, rows, missing, t0 = [], [], [], 0, time.time()
    for r in recs:
        real.append(M.block_stats(polys(r), r["height"], r))
        if pred is None:
            got = {0: (polys(r), r["height"])}
        else:
            got = pred.get(r["row"])
            if got is None:
                missing += 1
                got = {0: ([], np.zeros(0))}
        ms = input_maps(r, maps, max(got) + 1)
        ps = [M.block_stats(g, h, r, ms[s]) for s, (g, h) in sorted(got.items())]
        preds.append(ps)
        for s, p in zip(sorted(got), ps):
            rows.append({"row": r["row"], "boro": r["boro"], "sample": s, "n_real": real[-1]["n"], "n_pred": p["n"],
                         **M.per_block_scores(real[-1], p)})
    out = M.summarise(real, preds)
    out.update(name=a.name, subset=a.subset, input=a.input, missing=missing, pred=str(a.pred),
               ci=bootstrap(real, preds, a.boot) if a.boot else {})
    d = os.path.join(C.RESULTS, a.name)
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(d, "per_block.csv"), index=False)
    lines = [f"# {a.name}", "", f"{len(recs)} blocks ({a.subset}), input `{a.input}`, {missing} without prediction.",
             f"Buildings: {out['buildings_pred']:.0f} predicted, {out['buildings_real']} real.", "",
             "| Metric | Value | 95% CI (blocks) |", "|---|---|---|"]
    for k, label in NAMES:
        ci = out["ci"].get(k)
        lines.append(f"| {label} | {out[k]:.3f} | {f'{ci[0]:.3f} - {ci[1]:.3f}' if ci else ''} |")
    open(os.path.join(d, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), f"\n({time.time() - t0:.0f} s)")


def compare(names):
    s = {n: json.load(open(os.path.join(C.RESULTS, n, "summary.json"))) for n in names}
    head = "| Metric | " + " | ".join(names) + " |"
    lines = [head, "|---" * (len(names) + 1) + "|"]
    for k, label in NAMES:
        lines.append(f"| {label} | " + " | ".join(f"{s[n][k]:.3f}" for n in names) + " |")
    lines.append("| Blocks without prediction | " + " | ".join(str(s[n]["missing"]) for n in names) + " |")
    txt = "\n".join(lines)
    os.makedirs(C.RESULTS, exist_ok=True)
    open(os.path.join(C.RESULTS, "compare.md"), "w").write(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--name")
    ap.add_argument("--subset", default="test1000")
    ap.add_argument("--input", default="oracle")
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--real", action="store_true", help="score the real buildings (sanity row)")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--compare", nargs="+")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare)
    else:
        if not a.name or (not a.real and not a.pred):
            ap.error("--name and --pred (or --real) are required")
        run(a)


if __name__ == "__main__":
    main()
