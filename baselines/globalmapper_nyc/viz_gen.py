"""Figure: real map and the generated samples of a few test windows, for each mode.

  python viz_gen.py --gen GEN_ROOT --modes random prior recon --city .../city --out FIG.png
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

PAL = ["#c8c8c8", "#e8e0cc", "#ffe29a", "#fdb04f", "#e3692b", "#9b2318",
       "#c6dbf7", "#7aa9f0", "#3b6fd1", "#16306e"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--modes", nargs="+", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    order = json.load(open(os.path.join(a.city, "test_subset.json")))["sample_ids"][:a.n]
    blk = np.load(os.path.join(a.city, "block.npy"), mmap_mode="r")
    samples = pd.read_parquet(os.path.join(a.city, "samples.parquet")).set_index("sample_id")
    cols = 1 + 2 * len(a.modes)
    fig, axes = plt.subplots(a.n, cols, figsize=(2.3 * cols, 2.3 * a.n))
    cmap = ListedColormap(PAL)
    for r, sid in enumerate(order):
        s = samples.loc[sid]
        wb = np.asarray(blk[int(s.row0):int(s.row0) + 128, int(s.col0):int(s.col0) + 128])
        mask = np.isin(wb, [int(t) for t in s.targets])
        ys, xs = np.nonzero(mask)
        pad = 12
        sl = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
        panels = []
        z0 = np.load(os.path.join(a.gen, a.modes[0], "Generated", f"{sid}.npz"))
        panels.append(("real", z0["gt"]))
        for m in a.modes:
            z = np.load(os.path.join(a.gen, m, "Generated", f"{sid}.npz"))
            panels += [(f"{m} 1", z["gen"][0]), (f"{m} 2", z["gen"][1])]
        for c, (title, im) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im[sl], cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
            ax.contour(mask[sl], levels=[0.5], colors="magenta", linewidths=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
        axes[r, 0].set_ylabel(f"#{sid}", fontsize=8)
    fig.suptitle("NYC-trained GlobalMapper: target blocks (magenta); buildings drawn as one class (no height)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
