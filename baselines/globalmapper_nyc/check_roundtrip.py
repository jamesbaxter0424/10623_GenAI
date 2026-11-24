"""Round trip of the conversion: real buildings -> graph -> polygons, drawn over the real ones,
plus the coverage the 4 m raster of the round trip gives vs the real coverage.

  python check_roundtrip.py --out /tmp/gm_nyc_test --split val --gm .../GlobalMapper --city .../city --bldg .../nyc_bldg
"""
import argparse
import os
import pickle

import matplotlib
import numpy as np
import pandas as pd
from shapely import wkb

from gm_geom import graph_to_polygons, load_norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ["out", "split", "gm", "city", "bldg"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    norm = load_norm(a.out)
    idx = pd.read_csv(os.path.join(a.out, a.split, "index.csv"))
    geom = pickle.load(open(os.path.join(a.out, a.split, "geom.pkl"), "rb"))
    blocks = pd.read_parquet(os.path.join(a.city, "blocks.parquet"), columns=["block_id", "geometry", "coverage"])
    bld = pd.concat([pd.read_parquet(os.path.join(a.bldg, f), columns=["block_id", "geometry"])
                     for f in sorted(os.listdir(a.bldg))])
    pick = idx[idx.n_bldg > 3].sample(a.n, random_state=0) if (idx.n_bldg > 3).sum() >= a.n else idx.head(a.n)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    errs = []
    for ax, r in zip(axes.flat, pick.itertuples()):
        g = pickle.load(open(os.path.join(a.out, a.split, "processed", f"{r.k}.gpickle"), "rb"))
        at = lambda key: np.array([g.nodes[k][key] for k in range(120)])
        polys = graph_to_polygons(at("exist"), at("posx"), at("posy"), at("size_x"), at("size_y"), at("iou"),
                                  geom[r.block_row], norm, a.gm)
        blk = wkb.loads(blocks.geometry.iloc[r.block_row])
        real = [wkb.loads(x) for x in bld[bld.block_id == blocks.block_id.iloc[r.block_row]].geometry]
        for p in (blk.geoms if blk.geom_type == "MultiPolygon" else [blk]):
            ax.plot(*p.exterior.xy, color="k", lw=1)
        for p in real:
            for q in (p.geoms if p.geom_type == "MultiPolygon" else [p]):
                ax.fill(*q.exterior.xy, color="#9ec5f0", lw=0)
        for p in polys:
            ax.plot(*p.exterior.xy, color="#e3692b", lw=1)
        med = wkb.loads(geom[r.block_row]["midaxis"])
        cx, cy = geom[r.block_row]["center"]
        mx, my = med.xy
        ax.plot(np.array(mx) + cx, np.array(my) + cy, "--", color="green", lw=1)
        cov_rt = sum(p.intersection(blk).area for p in polys) / blk.area
        cov_real = sum(p.intersection(blk).area for p in real) / blk.area
        errs.append(abs(cov_rt - cov_real))
        ax.set_title(f"row {r.block_row}: {len(real)} real / {len(polys)} kept; cov {cov_real:.2f} -> {cov_rt:.2f}",
                     fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("Round trip: blue = real buildings, orange = from the converted graph, green = main axis")
    fig.tight_layout()
    f = os.path.join(a.out, f"roundtrip_{a.split}.png")
    fig.savefig(f, dpi=90)
    print("saved", f, "| mean |coverage error| of the round trip:", round(float(np.mean(errs)), 3))


if __name__ == "__main__":
    main()
