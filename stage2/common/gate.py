"""Gates between the training stages. Exits with code 1 if the gate is not passed, so run.sh stops.

  python -m common.gate A --run out/rd_overfit8 [--max_score 0.10]
  python -m common.gate B --run out/rd_pilot --rule results/rule_val200/summary.json

A: the model can memorise 8 blocks (validation on the same blocks, score below --max_score).
B: trained on 2,000 blocks, it beats the rule baseline on val200 in at least one of the distribution
   metrics (count, area, width, height, height step), reported with all the others.
"""
import argparse
import json
import os
import sys

import pandas as pd

DISTS = ("count_err", "wd_area", "wd_width", "wd_height", "wd_dz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["A", "B"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--max_score", type=float, default=0.10)
    ap.add_argument("--rule", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "results", "rule_val200", "summary.json"))
    a = ap.parse_args()
    v = pd.read_csv(os.path.join(a.run, "val.csv"))
    b = v.loc[v["score"].idxmin()]
    if a.gate == "A":
        ok = b["score"] <= a.max_score
        print(f"GATE A: best score {b['score']:.3f} at step {int(b['step'])} (need <= {a.max_score}) ->", "PASS" if ok else "FAIL")
    else:
        r = json.load(open(a.rule))
        wins = [k for k in DISTS if b[k] < r[k]]
        for k in DISTS + ("overlap", "map_iou", "score"):
            print(f"  {k:10s} model {b[k]:.3f}  rule {r[k]:.3f}", "<- better" if k in wins else "")
        ok = len(wins) >= 1
        print(f"GATE B: better than the rule baseline in {len(wins)} of {len(DISTS)} ->", "PASS" if ok else "FAIL")
        if ok and b["score"] > r["score"]:
            print("  WARNING: the mean score is still worse than the rule baseline. Passing on one metric is a weak"
                  " result: look at figures and at map_iou, and discuss before starting gate C.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
