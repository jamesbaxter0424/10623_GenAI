"""Training curves of the NYC-trained CityDreamer and a convergence summary.

  python plot_training.py --vqgan vqgan_log.jsonl --sampler sampler_log.jsonl --out DIR
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def read(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def smooth(y, k):
    k = max(1, min(k, len(y)))
    c = np.cumsum(np.insert(np.asarray(y, float), 0, 0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, np.nan), out])


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel("hours", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.tick_params(colors=INK2, labelsize=8.5)


def per_hour_change(h, v):
    """Change of a validation curve per hour, over consecutive 1-hour windows."""
    h, v = np.asarray(h), np.asarray(v)
    edges = np.arange(0, np.floor(h.max()) + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        ia, ib = np.argmin(abs(h - a)), np.argmin(abs(h - b))
        out.append((b, v[ib] - v[ia]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqgan", required=True)
    ap.add_argument("--sampler", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    vq, sp = read(a.vqgan), read(a.sampler)

    vh = np.array([r["hours"] for r in vq]); vseg = np.array([r["seg"] for r in vq])
    vv = [r for r in vq if "val_acc" in r]
    vvh = np.array([r["hours"] for r in vv])
    vacc = np.array([r["val_acc"] for r in vv]); vbacc = np.array([r["val_building_acc"] for r in vv])
    sh = np.array([r["hours"] for r in sp]); srw = np.array([r["rw_elbo"] for r in sp])
    sv = [r for r in sp if "val_rw_elbo" in r]
    svh = np.array([r["hours"] for r in sv]); sval = np.array([r["val_rw_elbo"] for r in sv])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor="#fcfcfb")
    for ax in axes.flat:
        ax.set_facecolor("#fcfcfb")

    ax = axes[0, 0]
    ax.plot(vh, vseg, color=BLUE, lw=1, alpha=0.25)
    ax.plot(vh, smooth(vseg, 10), color=BLUE, lw=2)
    ax.set_ylim(0, min(1.2, np.nanmax(vseg) * 1.05))
    style(ax, "(a) VQGAN training loss (class cross-entropy)", "loss")
    ax.annotate(f"{np.nanmean(vseg[-10:]):.3f}", (vh[-1], np.nanmean(vseg[-10:])),
                xytext=(4, 6), textcoords="offset points", fontsize=8.5, color=INK)

    ax = axes[0, 1]
    ax.plot(vvh, vacc, "-o", color=BLUE, lw=2, ms=4, label="all cells")
    ax.plot(vvh, vbacc, "-o", color=ORANGE, lw=2, ms=4, label="building cells")
    ax.axhline(0.85, color=INK2, lw=1, ls="--")
    ax.text(0.05, 0.853, "gate 0.85", color=INK2, fontsize=8.5, va="bottom")
    for y, c in ((vacc[-1], INK), (vbacc[-1], INK)):
        ax.annotate(f"{y:.3f}", (vvh[-1], y), xytext=(4, -3), textcoords="offset points", fontsize=8.5, color=c)
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    style(ax, "(b) VQGAN validation reconstruction (512 windows)", "accuracy")

    ax = axes[1, 0]
    ax.plot(sh, srw, color=BLUE, lw=1, alpha=0.2)
    ax.plot(sh, smooth(srw, 25), color=BLUE, lw=2, label="training (moving average)")
    ax.plot(svh, sval, "-o", color=ORANGE, lw=2, ms=4, label="validation (200 windows)")
    ax.set_ylim(0.55, 1.0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.annotate(f"{sval[-1]:.3f}", (svh[-1], sval[-1]), xytext=(4, -10), textcoords="offset points",
                fontsize=8.5, color=INK)
    style(ax, "(c) Sampler loss (reweighted ELBO)", "loss")

    ax = axes[1, 1]
    ch = per_hour_change(svh, sval)
    xs = [c[0] for c in ch]; ys = [c[1] for c in ch]
    ax.bar(xs, ys, width=0.7, color=ORANGE, edgecolor="#fcfcfb", linewidth=2)
    ax.axhline(0, color="#b5b4ae", lw=0.8)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:+.3f}", (x, y), xytext=(0, -11 if y < 0 else 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels([f"{int(x - 1)}-{int(x)}" for x in xs])
    style(ax, "(d) Sampler validation loss change per hour", "change per hour")
    fig.suptitle("NYC-trained CityDreamer: training curves", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "citydreamer_training.png"), dpi=150, facecolor=fig.get_facecolor())

    def val_at(h, v, t):
        return v[np.argmin(abs(h - t))]
    lines = [
        f"VQGAN: {len(vq)} log rows, last step {vq[-1]['step']}, {vh[-1]:.2f} h",
        f"  val acc {vacc[0]:.3f} (step {vv[0]['step']}) -> {vacc[-1]:.3f}; building {vbacc[0]:.3f} -> {vbacc[-1]:.3f}",
        f"  val acc gain, last hour: {vacc[-1] - val_at(vvh, vacc, vvh[-1] - 1):+.4f}; "
        f"first hour: {val_at(vvh, vacc, 1) - vacc[0]:+.4f}",
        f"  train CE last 10 rows {np.mean(vseg[-10:]):.4f}; 10 rows one hour earlier "
        f"{np.mean(vseg[(vh > vh[-1] - 1.1) & (vh < vh[-1] - 0.9)]):.4f}",
        f"Sampler: last step {sp[-1]['step']}, {sh[-1]:.2f} h",
        f"  val rw_elbo {sval[0]:.4f} -> {sval[-1]:.4f}",
        "  change per hour: " + ", ".join(f"{int(x - 1)}-{int(x)} h {y:+.4f}" for x, y in ch),
    ]
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(a.out, "citydreamer_training.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
