"""NYC baseline helpers.

  python tools/nyc_tools.py subset --root DATA --out DATA/test_subset.json
      fixed random order of the clean test windows (seed 0); every model and the
      evaluation use the first n of this list.
  python tools/nyc_tools.py check --root DATA --out DIR [--n 8]
      figures of the adapter output: class map, target blocks, node centroids
      coloured by node label, kNN edges; plus a summary of graph sizes.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets.nyc_dataset import NYCSceneGraphDataset  # noqa: E402
from utils.nyc_vis import COLORS, legend, outline, to_rgb  # noqa: E402


def make_subset(root, out, split="test", seed=0):
    s = pd.read_parquet(os.path.join(root, "samples.parquet"))
    s = s[(s["split"] == split) & (~s["ctx_other_split"])]
    ids = np.random.default_rng(seed).permutation(s["sample_id"].to_numpy()).tolist()
    json.dump({"split": split, "seed": seed, "n": len(ids), "sample_ids": [int(i) for i in ids]},
              open(out, "w"))
    k = s.set_index("sample_id").loc[ids[:200], "k"].value_counts().sort_index().to_dict()
    print(f"{out}: {len(ids)} {split} windows; group sizes in first 200: {k}")


def check(root, out, n=8, seed=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    ds = NYCSceneGraphDataset(root, "train")
    idx = np.random.default_rng(seed).choice(len(ds), n, replace=False)

    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(4.2 * cols, 4.4 * np.ceil(n / cols)))
    for ax, i in zip(axes.ravel(), idx):
        item = ds[int(i)]
        sg = item["scene_graph"]
        cent = np.array([np.unravel_index(int(h.argmax()), h.shape) for h in sg.heatmaps])
        ax.imshow(to_rgb(item["bev"].numpy()), interpolation="nearest")
        outline(ax, item["mask"].numpy())
        for a, b in sg.edge_index.numpy().T:
            if a < b:
                ax.plot([cent[a, 1], cent[b, 1]], [cent[a, 0], cent[b, 0]], "k-", lw=0.5, alpha=0.6)
        lab = sg.y.numpy()
        ax.scatter(cent[:, 1], cent[:, 0], c=COLORS[lab] / 255, s=28, edgecolors="k", linewidths=0.7, zorder=3)
        ax.set_title(f"window {item['file_path']}: {len(lab)} nodes, {sg.edge_index.shape[1] // 2} edges",
                     fontsize=9)
        ax.set_xlim(-0.5, 127.5); ax.set_ylim(127.5, -0.5); ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Baseline adapter check: node dots = block centroid, colour = node label "
                 "(use x height class); lines = 4-NN edges; magenta = target blocks", fontsize=10)
    legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(os.path.join(out, "adapter_check.png"), dpi=100)

    # graph size statistics over 500 random windows
    nodes, edges, lab_hist = [], [], np.zeros(10, int)
    for i in np.random.default_rng(seed + 1).choice(len(ds), 500, replace=False):
        sg = ds[int(i)]["scene_graph"]
        nodes.append(sg.num_nodes); edges.append(sg.edge_index.shape[1] // 2)
        lab_hist += np.bincount(sg.y.numpy(), minlength=10)
    x0 = ds[0]["scene_graph"].x
    print(f"node feature dim {x0.shape[1]}; nodes/window median {np.median(nodes):.0f} "
          f"(min {min(nodes)}, max {max(nodes)}); edges/window median {np.median(edges):.0f}")
    print("node label share:", {i: round(v / lab_hist.sum(), 3) for i, v in enumerate(lab_hist)})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["subset", "check"])
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()
    make_subset(a.root, a.out) if a.cmd == "subset" else check(a.root, a.out, a.n)
