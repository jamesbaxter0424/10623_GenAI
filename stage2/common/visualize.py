"""Side-by-side figure: input map, real buildings, predictions of several methods.

  python -m common.visualize --preds rule=out/rule/val200_oracle.jsonl diffusion=out/.../val200.jsonl \
      --subset val200 --n 6 --out out/figures/val.png
"""
import argparse
import os

import numpy as np

from . import config as C
from .records import input_maps, load, polys, read_pred, stage1_maps

# street, open, residential 1-2/3-5/6-12/13+, non-residential 1-2/3-5/6-12/13+
CLASS_COLORS = ["#fcfcfb", "#e8e6df", "#cde2fb", "#86b6ef", "#2a78d6", "#0d366b",
                "#f8d5c4", "#f0a27f", "#eb6834", "#8a3413"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="*", default=[], help="name=path.jsonl")
    ap.add_argument("--subset", default="val200")
    ap.add_argument("--input", default="oracle")
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--rows", nargs="*", type=int, help="block rows; default: n random blocks with 15-60 buildings")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    preds = {q.split("=", 1)[0]: read_pred(q.split("=", 1)[1]) for q in a.preds}
    recs = load(a.subset, a.data)
    if a.rows:
        recs = [r for r in recs if r["row"] in a.rows]
    else:
        ok = [r for r in recs if 15 <= len(r["polys"]) <= 60 and all(r["row"] in p for p in preds.values())] or recs
        pick = np.random.default_rng(1).choice(len(ok), min(a.n, len(ok)), replace=False)
        recs = [ok[i] for i in pick]
    maps = stage1_maps(a.input)
    cols = ["input map (4 m)", "real"] + list(preds)
    cmap = LinearSegmentedColormap.from_list("h", ["#86b6ef", "#2a78d6", "#0d366b"])
    norm = LogNorm(3, 60, clip=True)
    fig, ax = plt.subplots(len(recs), len(cols), figsize=(3.4 * len(cols), 3.2 * len(recs)), squeeze=False,
                           facecolor="#fcfcfb", constrained_layout=True)
    for i, r in enumerate(recs):
        h, w = r["cls4"].shape
        ext = (0, w * C.CELL, h * C.CELL, 0)
        m = input_maps(r, maps, 1)[0]
        ax[i, 0].imshow(m, cmap=ListedColormap(CLASS_COLORS), vmin=-0.5, vmax=9.5, extent=ext, interpolation="nearest")
        shown = [(polys(r), r["height"])] + [p.get(r["row"], {}).get(0, ([], [])) for p in preds.values()]
        for j, (g, hs) in enumerate(shown, 1):
            for p, z in zip(g, hs):
                ax[i, j].fill(*np.asarray(p.exterior.coords).T, fc=cmap(norm(z)), ec="#fcfcfb", lw=0.6)
            ax[i, j].text(0.02, 0.02, f"{len(g)} buildings", transform=ax[i, j].transAxes, fontsize=8, color="#52514e")
        for j in range(len(cols)):
            x = ax[i, j]
            x.set_xlim(0, w * C.CELL)
            x.set_ylim(h * C.CELL, 0)
            x.set_aspect("equal")
            x.set_xticks([])
            x.set_yticks([])
            for s in x.spines.values():
                s.set_color("#e1e0d9")
            if i == 0:
                x.set_title(cols[j], loc="left", fontsize=10)
        ax[i, 0].set_ylabel(r["block_id"], fontsize=8, color="#52514e")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.01, ticks=[3, 6, 12, 25, 60])
    cb.ax.set_yticklabels(["3", "6", "12", "25", "60+"])
    cb.set_label("building height (m)", fontsize=8)
    cb.ax.minorticks_off()
    cb.outline.set_visible(False)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=130, facecolor="#fcfcfb")
    print("written", a.out)


if __name__ == "__main__":
    main()
