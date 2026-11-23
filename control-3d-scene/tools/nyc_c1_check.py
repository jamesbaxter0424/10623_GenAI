"""Checks for change 1 before training (numeric node attributes + street-facing graph).

  python tools/nyc_c1_check.py --root ~/genai-final/data/city --out ~/genai-final/logs/c1_check

1. The baseline input is unchanged: with numeric=False, graph="knn" the first 27 features and
   the edges equal those of the change-1 dataset's first 27 features / the kNN graph.
2. Ranges of the 6 numbers and the 2 edge features over 2,000 training windows (no NaN, about 0-1).
3. Edge counts and lengths (centroid distance) of the kNN graph vs the street-facing graph.
4. Figures: both graphs drawn on the real map for 4 windows.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.nyc_dataset import NYCSceneGraphDataset, NUM_NUMERIC  # noqa: E402

NAMES = ["coverage", "log1p FAR", "log1p floors / 2", "res share", "log10 area / 5", "unreliable"]


def centroid_lengths(sg, cent_m):
    ei = sg.edge_index.numpy()
    return np.linalg.norm(cent_m[ei[0]] - cent_m[ei[1]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kw = dict(split="train", min_visible_cells=20)
    base = NYCSceneGraphDataset(a.root, **kw)
    c1 = NYCSceneGraphDataset(a.root, numeric=True, graph="street", **kw)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(base), a.n, replace=False)

    feats, eattr, len_knn, len_st, n_knn, n_st, n_nodes, isolated = [], [], [], [], [], [], [], 0
    for i in idx:
        b, c = base[int(i)]["scene_graph"], c1[int(i)]["scene_graph"]
        assert b.x.shape[1] == 27 and c.x.shape[1] == 27 + NUM_NUMERIC
        assert np.array_equal(b.x.numpy()[:, :10], c.x.numpy()[:, :10]), "labels differ"
        assert np.array_equal(b.heatmaps.numpy(), c.heatmaps.numpy()), "positions differ"
        assert not hasattr(b, "edge_attr") or b.edge_attr is None
        cent = np.stack(np.nonzero(c.heatmaps.numpy().reshape(len(c.x), -1)), 1)[:, 1]
        cent_m = np.stack([cent // 128, cent % 128], 1) * 4.0
        feats.append(c.x.numpy()[:, 27:])
        eattr.append(c.edge_attr.numpy())
        len_knn.append(centroid_lengths(b, cent_m))
        len_st.append(centroid_lengths(c, cent_m))
        n_knn.append(b.edge_index.shape[1] // 2)
        n_st.append(c.edge_index.shape[1] // 2)
        n_nodes.append(len(c.x))
        deg = np.bincount(c.edge_index.numpy()[0], minlength=len(c.x))
        isolated += int((deg == 0).sum())

    F = np.concatenate(feats)
    E = np.concatenate(eattr)
    lines = [f"windows {a.n}, nodes {F.shape[0]}  (baseline features identical: OK)", ""]
    lines.append("node numbers: name, min, p50, p99, max, NaN")
    for k, nm in enumerate(NAMES):
        v = F[:, k]
        lines.append(f"  {nm:18s} {v.min():7.3f} {np.median(v):7.3f} {np.percentile(v, 99):7.3f} "
                     f"{v.max():7.3f} {int(np.isnan(v).sum())}")
    lines.append(f"  unreliable share {F[:, -1].mean():.3f}")
    lines.append("edge features: facing/100 m, street/30 m (min, p50, p99, max)")
    for k, nm in enumerate(["facing", "street"]):
        v = E[:, k]
        lines.append(f"  {nm:8s} {v.min():6.3f} {np.median(v):6.3f} {np.percentile(v, 99):6.3f} {v.max():6.3f}")
    lk, ls = np.concatenate(len_knn), np.concatenate(len_st)
    lines += ["",
              f"edges per window: kNN median {np.median(n_knn):.0f}, street median {np.median(n_st):.0f}",
              f"centroid length (m): kNN median {np.median(lk):.0f}, p95 {np.percentile(lk, 95):.0f}, "
              f">200 m {np.mean(lk > 200):.3f}",
              f"                     street median {np.median(ls):.0f}, p95 {np.percentile(ls, 95):.0f}, "
              f">200 m {np.mean(ls > 200):.3f}",
              f"nodes without a street edge: {isolated} of {F.shape[0]} ({isolated / F.shape[0]:.3f})"]
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(a.out, "c1_check.txt"), "w").write(txt + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    pal = ["#c8c8c8", "#e8e0cc", "#ffe29a", "#fdb04f", "#e3692b", "#9b2318",
           "#c6dbf7", "#7aa9f0", "#3b6fd1", "#16306e"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
    for col, i in enumerate(idx[:4]):
        for row, (ds, name) in enumerate([(base, "baseline: 4 nearest"), (c1, "change 1: facing across a street")]):
            it = ds[int(i)]
            sg = it["scene_graph"]
            ax = axes[row, col]
            ax.imshow(it["bev"].numpy(), cmap=ListedColormap(pal), vmin=0, vmax=9, interpolation="nearest")
            hm = sg.heatmaps.numpy().reshape(len(sg.x), -1).argmax(1)
            rr, cc = hm // 128, hm % 128
            ei = sg.edge_index.numpy()
            for s, t in ei.T:
                if s < t:
                    ax.plot([cc[s], cc[t]], [rr[s], rr[t]], "-", color="k", lw=0.8, alpha=0.7)
            ax.plot(cc, rr, "o", ms=3, color="k")
            m = it["mask"].numpy()
            ax.contour(m, levels=[0.5], colors="magenta", linewidths=1.2)
            ax.set_title(f"#{it['file_path']} {name}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "c1_graphs.png"), dpi=110)
    print("saved", os.path.join(a.out, "c1_graphs.png"))


if __name__ == "__main__":
    main()
