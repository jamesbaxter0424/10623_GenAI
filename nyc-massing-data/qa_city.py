"""QA for the full-city build: qa/city/summary.md + figures."""
import json

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from config import CITY, CLASSES, INTERIM, OUT, QA
from dataset import NYCMassing

C = OUT / "city"
Q = QA / "city"
CLASS_COLORS = ["#d9d8d2", "#cfe3c4", "#b7d3f6", "#6da7ec", "#256abf", "#0d366b",
                "#f7c6ad", "#ef9466", "#c9521f", "#7a2c0c"]
SPLIT_COLORS = {"train": "#c3c2b7", "val": "#1baf7a", "test": "#eb6834"}
plt.rcParams.update({"font.size": 9, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def md(df, floatfmt=".3f"):
    return df.to_markdown(floatfmt=floatfmt)


def main():
    Q.mkdir(parents=True, exist_ok=True)
    blk = gpd.read_parquet(C / "blocks.parquet")
    edges = pd.read_parquet(C / "edges.parquet")
    s = pd.read_parquet(C / "samples.parquet")
    chk = pd.read_csv(C / "raster_check.csv")
    fun = pd.read_csv(INTERIM / "funnel.csv")
    fun = fun[fun["pilot"].isin(CITY)]
    out = ["# Full-city dataset QA\n"]

    # --- per borough headline numbers
    rows = {}
    for b in CITY:
        bl = pd.read_parquet(INTERIM / b / "buildings_final.parquet", columns=["link", "floors_src"])
        fh = json.loads((INTERIM / b / "floor_height.json").read_text())
        k = blk[blk["boro"] == b]
        v = k[k["valid"]]
        f = fun[(fun["pilot"] == b) & (fun["step"] == "feature_code")]
        rows[b] = {
            "footprints_raw": int(f["n_in"].iloc[0]) if len(f) else np.nan,
            "buildings_kept": len(bl),
            "link_rate": (bl["link"] != "none").mean(),
            "floor_h_m": fh["floor_height_m"],
            "blocks": len(k), "valid_blocks": len(v), "valid_%": 100 * len(v) / len(k),
            "corr_FAR_valid": v["far_pluto"].corr(v["far_fp"]),
            "median_far_fp/far_pluto": (v["far_fp"] / v["far_pluto"].where(v["far_pluto"] > 0.1)).median(),
            "corr_cov_vec_ras": chk[(chk["boro"] == b) & chk["valid"]][["coverage_vec", "coverage_ras"]].corr().iloc[0, 1],
        }
    t = pd.DataFrame(rows)
    t["NYC"] = [t.loc["footprints_raw"].sum(), t.loc["buildings_kept"].sum(), np.nan, np.nan,
                len(blk), int(blk["valid"].sum()), 100 * blk["valid"].mean(),
                blk[blk["valid"]]["far_pluto"].corr(blk[blk["valid"]]["far_fp"]), np.nan,
                chk[chk["valid"]][["coverage_vec", "coverage_ras"]].corr().iloc[0, 1]]
    out += ["## Headline numbers per borough\n", md(t), "\n"]

    # --- why blocks are invalid
    steps = fun[fun["stage"] == "3_block"].pivot_table(index="step", columns="pilot", values="dropped", aggfunc="sum")
    out += ["## Blocks removed by each check (core blocks)\n", md(steps.fillna(0).astype(int)), "\n"]

    # --- split
    sp = json.loads((C / "splits.json").read_text())
    out += ["## Split by community district\n",
            f"seed {sp['seed']}; val CDs: {sp.get('val')}; test CDs: {sp.get('test')}\n"]
    v = blk[blk["valid"]]
    tab = pd.DataFrame({"valid_blocks": v.groupby("split").size(),
                        "samples": s.groupby("split").size(),
                        "samples_clean_ctx": s[~s["ctx_other_split"]].groupby("split").size()})
    out += [md(tab), "\n", "Samples by group size:\n", md(pd.crosstab(s["k"], s["split"], margins=True)), "\n"]
    cov = pd.Series(np.concatenate(s["targets"].to_numpy())).value_counts()
    out += [f"Valid blocks used as a target: {cov.size} / {len(v)}; "
            f"appearances per block median {cov.median():.0f}, max {cov.max()}\n"]

    # --- target distributions per split
    desc = v.groupby("split")[["coverage", "far_fp", "median_floors", "res_share", "area_ha"]].median()
    out += ["## Median block targets per split\n", md(desc), "\n"]
    hc = pd.crosstab(v["split"], v["hclass"], normalize="index")
    out += ["Height class share per split (0: 1-2, 1: 3-5, 2: 6-12, 3: 13+ floors):\n", md(hc), "\n"]

    # --- graph
    deg = blk["degree"]
    out += ["## Block graph\n",
            f"edges {len(edges)}, mean degree {deg.mean():.2f}, isolated blocks {int((deg == 0).sum())}, "
            f"touching (no street between) {int(edges['touching'].sum())}; "
            f"street width median {edges['street_m'].median():.1f} m, facing length median "
            f"{edges['shared_m'].median():.0f} m\n"]

    # --- class shares
    cls = np.load(C / "cls.npy", mmap_mode="r")
    water = np.load(C / "water.npy", mmap_mode="r")
    cc = np.bincount(np.asarray(cls)[~np.asarray(water)], minlength=10)
    out += ["## Class shares on land\n",
            md(pd.DataFrame({"class": [CLASSES[i] for i in range(10)], "share": cc / cc.sum()}).set_index("class"),
               ".4f"), "\n"]
    (Q / "summary.md").write_text("\n".join(out), encoding="utf-8")

    # --- figure: split map
    fig, ax = plt.subplots(figsize=(9, 9))
    pts = blk.representative_point()
    for k, col in SPLIT_COLORS.items():
        m = (blk["split"] == k).to_numpy()
        ax.scatter(pts.x[m], pts.y[m], s=0.4, c=col, label=k)
    ax.set_aspect("equal"); ax.set_axis_off(); ax.legend(markerscale=15, loc="upper left")
    ax.set_title("Blocks by split (community districts)")
    fig.savefig(Q / "01_split_map.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # --- figure: random samples per split with mask + graph
    rng = np.random.default_rng(1)
    cmap = ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    for r, split in enumerate(["train", "val", "test"]):
        ds = NYCMassing(split)
        for c, i in enumerate(rng.choice(len(ds), 4, replace=False)):
            x = ds[int(i)]
            ax = axes[r, c]
            ax.imshow(x["cls"], cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
            ax.contour(x["mask"], levels=[0.5], colors="#e0007a", linewidths=1.6)
            yy, xx = np.indices(x["node_map"].shape)
            nm = x["node_map"]
            n = len(x["node_attr"])
            cy = np.bincount(nm[nm >= 0], weights=yy[nm >= 0], minlength=n) / np.maximum(np.bincount(nm[nm >= 0], minlength=n), 1)
            cx = np.bincount(nm[nm >= 0], weights=xx[nm >= 0], minlength=n) / np.maximum(np.bincount(nm[nm >= 0], minlength=n), 1)
            for a, b in x["edge_index"].T[: x["edge_index"].shape[1] // 2]:
                ax.plot([cx[a], cx[b]], [cy[a], cy[b]], c="#0b0b0b", lw=0.4, alpha=0.5)
            ax.scatter(cx, cy, s=4, c=np.where(x["node_target"], "#e0007a", "#0b0b0b"), zorder=3)
            t = x["node_attr"][x["node_target"]]
            ax.set_title(f"{split} #{x['sample_id']}  k={len(t)}\n" +
                         " | ".join(f"cov {a:.2f} FAR {b:.1f} h{int(h)} res {d:.1f}" for a, b, h, d, _ in t[:2]),
                         fontsize=7)
            ax.set_axis_off()
    fig.legend(handles=[Patch(color=CLASS_COLORS[i], label=CLASSES[i]) for i in range(10)],
               loc="lower center", ncol=10)
    fig.suptitle("Random samples: class map, target mask (magenta), block graph")
    fig.savefig(Q / "02_samples.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # --- figure: target distributions by split
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for ax, col, rng_ in zip(axes, ["coverage", "far_fp", "median_floors", "res_share"],
                             [(0, 1), (0, 15), (0, 40), (0, 1)]):
        for k, c in SPLIT_COLORS.items():
            ax.hist(v[v["split"] == k][col].clip(*rng_), bins=40, range=rng_, density=True,
                    histtype="step", color=c if k != "train" else "#555", label=k)
        ax.set_title(col)
    axes[0].legend()
    fig.savefig(Q / "03_target_dists.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print((Q / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
