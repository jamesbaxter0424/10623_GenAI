"""Figure: what the original Liu model actually receives for one NYC sample.

  python tools/nyc_input_figure.py --root DATA --sample 43146 --out FILE.png

Numpy-only re-implementation of the graph construction in datasets/nyc_dataset.py
(label, Liu's 4x4 position code, kNN-4 edges, >= 20 visible cells), so it runs
without torch.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

N, G, K, MIN_CELLS = 128, 4, 4, 20
COLORS = np.array([[217, 217, 214], [212, 230, 205], [189, 215, 245], [107, 165, 232], [37, 106, 196],
                   [10, 46, 107], [248, 200, 178], [238, 146, 98], [196, 82, 28], [112, 40, 12]]) / 255
NAMES = ["street", "open", "res 1-2", "res 3-5", "res 6-12", "res 13+",
         "non-res 1-2", "non-res 3-5", "non-res 6-12", "non-res 13+"]


def labels_of(blk):
    h = blk["hclass"].to_numpy()
    res = blk["is_res"].to_numpy().astype(bool)
    return np.where(h < 0, 1, np.where(res, 2 + h, 6 + h)).astype(int)


def patch_slot(r, c):
    ri, ci = min(G - 1, int(r / N * G)), min(G - 1, int(c / N * G))
    return 0 if ri == ci == 0 else 1 + ri * G + ci


def knn(cent, k):
    d = np.linalg.norm(cent[:, None] - cent[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :min(k, len(cent) - 1)]
    return sorted({(min(i, j), max(i, j)) for i in range(len(cent)) for j in nn[i]})


def main(root, sid, out):
    cls = np.load(os.path.join(root, "cls.npy"), mmap_mode="r")
    block = np.load(os.path.join(root, "block.npy"), mmap_mode="r")
    blk = pd.read_parquet(os.path.join(root, "blocks.parquet"), columns=["hclass", "is_res"])
    s = pd.read_parquet(os.path.join(root, "samples.parquet")).set_index("sample_id").loc[sid]
    r0, c0, targets = int(s.row0), int(s.col0), [int(t) for t in s.targets]
    m = np.asarray(cls[r0:r0 + N, c0:c0 + N]).astype(int)
    b = np.asarray(block[r0:r0 + N, c0:c0 + N]).astype(int)
    mask = np.isin(b, targets)

    ids, cnt = np.unique(b[b >= 0], return_counts=True)
    ids = ids[(cnt >= MIN_CELLS) | np.isin(ids, targets)]
    cent = np.array([np.argwhere(b == i).mean(0) for i in ids])
    lab = labels_of(blk)[ids]
    is_t = np.isin(ids, targets)
    slots = np.array([patch_slot(r, c) for r, c in cent])
    edges = knn(cent, K)
    feat = np.zeros((len(ids), 27))
    feat[np.arange(len(ids)), lab] = 1
    feat[np.arange(len(ids)), 10 + slots] = 1
    order = np.argsort(~is_t, kind="stable")          # target nodes first

    fig = plt.figure(figsize=(15, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.32, wspace=0.18)

    # (a) real map: known to us, not to the model
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(COLORS[m], interpolation="nearest")
    ax.contour(mask.astype(float), levels=[.5], colors="magenta", linewidths=1.6)
    ax.set_title("(a) Real tile = training TARGET\nnot an input; magenta = target blocks (scoring only)",
                 fontsize=10, color="#a00")
    ax.axis("off")

    # (b) the graph as the model gets it: nodes only know their 4x4 cell
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, N); ax.set_ylim(N, 0); ax.set_aspect("equal")
    for k in range(1, G):
        ax.axhline(k * N / G, color="#bbb", lw=.8, ls="--"); ax.axvline(k * N / G, color="#bbb", lw=.8, ls="--")
    # spread nodes inside their 4x4 cell (their exact position is NOT known to the model)
    pos = np.zeros_like(cent)
    for sl in np.unique(slots):
        idx = np.nonzero(slots == sl)[0]
        ri, ci = (0, 0) if sl == 0 else divmod(sl - 1, G)
        n = len(idx); cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols))
        for j, i in enumerate(idx):
            pos[i] = (ri * 32 + (j // cols + .5) * 32 / rows, ci * 32 + (j % cols + .5) * 32 / cols)
    for i, j in edges:
        ax.plot([pos[i, 1], pos[j, 1]], [pos[i, 0], pos[j, 0]], "k-", lw=.5, alpha=.45)
    ax.scatter(pos[:, 1], pos[:, 0], c=COLORS[lab], s=70, edgecolors="k", linewidths=.7, zorder=3)
    ax.scatter(pos[is_t, 1], pos[is_t, 0], s=230, facecolors="none", edgecolors="magenta", linewidths=1.8, zorder=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"(b) INPUT: block graph ({len(ids)} nodes, {len(edges)} edges)\n"
                 "each node knows only its class and which 4x4 cell it is in", fontsize=10, color="#060")

    # (c) training-only exact centers
    ax = fig.add_subplot(gs[0, 2])
    canvas = np.ones((N, N, 3))
    ax.imshow(canvas)
    ax.scatter(cent[:, 1], cent[:, 0], c=COLORS[lab], s=45, marker="s", edgecolors="k", linewidths=.6)
    ax.scatter(cent[is_t, 1], cent[is_t, 0], s=200, facecolors="none", edgecolors="magenta", linewidths=1.8)
    for k in range(1, G):
        ax.axhline(k * N / G, color="#ddd", lw=.6, ls="--"); ax.axvline(k * N / G, color="#ddd", lw=.6, ls="--")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(c) Exact block centers: TRAINING ONLY\ntest time: predicted by the localization head",
                 fontsize=10, color="#a60")

    # (d) node feature matrix
    ax = fig.add_subplot(gs[1, :2])
    img = np.ones((len(ids), 27, 3))
    for row, i in enumerate(order):
        img[row, lab[i]] = COLORS[lab[i]]
        img[row, 10 + slots[i]] = (.15, .15, .15)
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(27))
    ax.set_xticklabels(NAMES + ["slot 0\n(cell 0,0)", "unused"] +
                       [f"cell\n{(s - 1) // G},{(s - 1) % G}" for s in range(2, 17)], fontsize=6.5, rotation=90)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([("TARGET " if is_t[i] else "") + f"node {r}" for r, i in enumerate(order)], fontsize=6)
    for t in ax.get_yticklabels():
        if t.get_text().startswith("TARGET"):
            t.set_color("magenta")
    ax.axvline(9.5, color="k", lw=1.2)
    for x in np.arange(-.5, 27, 1):
        ax.axvline(x, color="#eee", lw=.4)
    for y in np.arange(.5, len(ids) - 1, 1):
        ax.axhline(y, color="#ddd", lw=.4)
    ax.axhline(is_t.sum() - .5, color="magenta", lw=1.2)
    ax.add_patch(Rectangle((-.5, -.5), 10, len(ids), fill=False, lw=1.5, ec="#060"))
    ax.add_patch(Rectangle((9.5, -.5), 17, len(ids), fill=False, lw=1.5, ec="#06a"))
    ax.set_title("(d) INPUT: node features, one row per block = 27 numbers: "
                 "class one-hot (10, left)  +  Liu's 4x4 position code (17, right)", fontsize=10, color="#060")

    # (e) what is not given
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    txt = ("NOT given to the model\n\n"
           "x  block outline, size or shape\n"
           "x  coverage, FAR, floors, residential share\n"
           "    (only one class label per block)\n"
           "x  exact position (only the 4x4 cell)\n"
           "x  the surrounding map:\n"
           "    the whole 512 m tile is generated\n"
           "x  the target mask (used only for scoring)\n\n"
           "Added by our changes\n"
           "(1) numeric attributes + street-facing edges\n"
           "(2) block polygons instead of guessed centers\n"
           "(3) real surrounding map + mask as input")
    ax.text(0, .98, txt, va="top", fontsize=10, family="monospace")

    fig.legend([Patch(color=c) for c in COLORS], NAMES, loc="lower center", ncol=10, frameon=False, fontsize=8.5)
    fig.suptitle(f"What the original Liu model receives for one sample (#{sid}, {len(targets)} target blocks)",
                 fontsize=13, y=.995)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"{out}: {len(ids)} nodes ({is_t.sum()} targets), {len(edges)} edges, "
          f"slots used {np.unique(slots).tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--sample", type=int, default=43146)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    main(a.root, a.sample, a.out)
