"""Figures for the NYC baseline: class maps, graphs, and generated samples."""
import numpy as np

# Same palette as nyc-massing-data/qa (street, open, res 1-2..13+, non-res 1-2..13+)
COLORS = np.array([
    [217, 217, 214], [212, 230, 205],
    [189, 215, 245], [107, 165, 232], [37, 106, 196], [10, 46, 107],
    [248, 200, 178], [238, 146, 98], [196, 82, 28], [112, 40, 12],
], dtype=np.uint8)
NAMES = ["street", "open", "res_1-2", "res_3-5", "res_6-12", "res_13+",
         "non_1-2", "non_3-5", "non_6-12", "non_13+"]


def to_rgb(cls):
    return COLORS[np.asarray(cls, dtype=np.int64)]


def outline(ax, mask, color="magenta"):
    ax.contour(np.asarray(mask, dtype=float), levels=[0.5], colors=color, linewidths=1.2)


def legend(fig):
    from matplotlib.patches import Patch
    fig.legend([Patch(color=COLORS[i] / 255) for i in range(10)], NAMES,
               loc="lower center", ncol=10, frameon=False, fontsize=8)


def save_sample_figure(path, gt, mask, gens, title=""):
    """gt | gt with target blocks masked | generated samples (target outlined)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = 2 + len(gens)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.0))
    holed = to_rgb(gt).copy()
    holed[np.asarray(mask, bool)] = 255
    panels = [("real", to_rgb(gt)), ("input (masked)", holed)] + \
             [(f"sample {i + 1}", to_rgb(g)) for i, g in enumerate(gens)]
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img, interpolation="nearest")
        outline(ax, mask)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)
