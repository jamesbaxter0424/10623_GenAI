"""Sanity checks of the metrics on the real map and on deliberately damaged copies of it.

  python tests/test_metrics.py [--n 60]

1. oracle (real map)           : distribution metrics are 0, height accuracy is 1
2. heights one class up        : FAR error and Boundary rise, coverage error unchanged
3. buildings removed           : coverage error = target coverage, street wall rises
4. cells shuffled inside block : coverage/FAR/height unchanged, footprint rises
5. four identical samples      : diversity is 0
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import City, test_order  # noqa: E402
from evaluate import per_block, score_window, summary_stat  # noqa: E402
import pandas as pd  # noqa: E402

from footprint import MEASURES  # noqa: E402,F401


def run(city, ids, damage, n_samples=1):
    rows, pairs, fps, divs = [], [], [], []
    for sid in ids:
        cls, blk, _, tgts = city.window(sid)
        gens = np.stack([damage(cls.copy(), blk, tgts, k) for k in range(n_samples)])
        r, p, f, d = score_window(city, sid, gens)
        rows += r; pairs += p; fps += f; divs += d
    t = {"blocks": per_block(rows), "pairs": pd.DataFrame(pairs),
         "fp": pd.DataFrame(fps, columns=["sample_id", "block", "s", "measure", "value"]), "div": pd.DataFrame(divs)}
    return summary_stat(t)


def same(c, b, t, k):
    return c


def up_one(c, b, t, k):
    m = np.isin(b, t) & (c >= 2) & ((c - 2) % 4 < 3)
    c[m] += 1
    return c


def no_buildings(c, b, t, k):
    c[np.isin(b, t) & (c >= 2)] = 1
    return c


def shuffled(c, b, t, k):
    rng = np.random.default_rng(k)
    for x in t:
        m = b == x
        c[m] = rng.permutation(c[m])
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    n = p.parse_args().n
    city = City()
    ids = test_order(city)[:n]
    ok = []

    def check(name, cond):
        ok.append(cond)
        print(("PASS " if cond else "FAIL ") + name)

    o = run(city, ids, same)
    check("oracle: boundary = 0", o["boundary_wd"] < 1e-9)
    check("oracle: street wall = 0", o["street_wall_wd"] < 1e-9 and o["street_wall_nbr_wd"] < 1e-9)
    check("oracle: footprint = 0", o["footprint"] < 1e-9)
    check("oracle: height accuracy = 1", o["height_acc"] > 0.999)
    check(f"oracle: coverage error small ({o['cov_err']:.3f})", o["cov_err"] < 0.03)
    print(f"     oracle FAR err {o['far_err']:.3f}, res err {o['res_err']:.3f}, hit rate {o['hit_rate']:.2f}")

    u = run(city, ids, up_one)
    check(f"heights up: FAR error rises ({o['far_err']:.3f} -> {u['far_err']:.3f})", u["far_err"] > o["far_err"] * 1.5)
    check(f"heights up: boundary rises ({u['boundary_wd']:.2f})", u["boundary_wd"] > 0.5)
    check("heights up: coverage error unchanged", abs(u["cov_err"] - o["cov_err"]) < 1e-9)
    check(f"heights up: height accuracy drops ({u['height_acc']:.2f})", u["height_acc"] < 0.3)

    e = run(city, ids, no_buildings)
    check(f"no buildings: coverage error rises ({e['cov_err']:.3f})", e["cov_err"] > 0.2)
    check(f"no buildings: street wall distance rises ({e['street_wall_wd']:.3f})", e["street_wall_wd"] > 0.2)
    check(f"no buildings: hit rate drops ({e['hit_rate']:.2f})", e["hit_rate"] < o["hit_rate"])

    s = run(city, ids, shuffled, n_samples=4)
    check("shuffle: coverage error unchanged", abs(s["cov_err"] - o["cov_err"]) < 1e-9)
    check(f"shuffle: footprint rises ({s['footprint']:.3f})", s["footprint"] > 0.3)
    check(f"shuffle: diversity > 0 ({s['diversity']:.2f})", s["diversity"] > 0.1)

    d = run(city, ids, same, n_samples=4)
    check("identical samples: diversity = 0", d["diversity"] < 1e-9)

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
