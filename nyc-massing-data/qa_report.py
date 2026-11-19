"""QA report for the pilot run: figures in qa/*.png and numbers in qa/summary.md."""
import json

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from config import CLASSES, INTERIM, OUT, PILOTS, QA

PILOT_COLOR = dict(zip(PILOTS, ["#2a78d6", "#eb6834", "#1baf7a"]))
CLASS_COLORS = ["#d9d8d2", "#cfe3c4",                       # street, open
                "#b7d3f6", "#6da7ec", "#256abf", "#0d366b",  # residential 1-2 .. 13+
                "#f7c6ad", "#ef9466", "#c9521f", "#7a2c0c"]  # non-residential 1-2 .. 13+
WATER = "#a9d6e5"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "legend.frameon": False})


def load(name):
    d = INTERIM / name
    return (gpd.read_parquet(d / "blocks.parquet"), gpd.read_parquet(d / "buildings_final.parquet"),
            pd.read_csv(OUT / name / "raster_check.csv"), pd.read_csv(OUT / name / "vanish.csv"),
            pd.read_csv(OUT / name / "tiles.csv"), json.loads((d / "floor_height.json").read_text()))


def cand(blk):
    """Core blocks that pass every check except the FAR-mismatch one."""
    bad = ["flag_incomplete", "flag_no_pluto", "flag_too_big", "flag_too_small", "flag_thin",
           "flag_water", "flag_coverage_bad", "flag_construction"]
    return blk[blk["in_core"] & ~blk[bad].any(axis=1)]


def scatter_panel(ax, data, x, y, title, lim=None, hollow=None):
    for name, df in data.items():
        h = df[hollow] if hollow else pd.Series(False, index=df.index)
        ax.scatter(df.loc[~h, x], df.loc[~h, y], s=14, alpha=0.75, color=PILOT_COLOR[name], label=name,
                   edgecolor="#fcfcfb", linewidth=0.5)
        if h.any():
            ax.scatter(df.loc[h, x], df.loc[h, y], s=16, facecolor="none", edgecolor=PILOT_COLOR[name], linewidth=1)
    hi = lim or np.nanmax([np.nanmax(df[[x, y]].to_numpy()) for df in data.values() if len(df)])
    ax.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--", zorder=0)
    ax.set_xlim(0, hi), ax.set_ylim(0, hi)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel(x), ax.set_ylabel(y)


def main():
    QA.mkdir(exist_ok=True)
    R = {n: load(n) for n in PILOTS}
    md = ["# Pilot QA summary\n"]

    # ---- funnel
    f = pd.read_csv(INTERIM / "funnel.csv")
    f["drop_%"] = (100 * f["dropped"] / f["n_in"].where(f["n_in"] > 0)).round(1)
    md += ["## Funnel (row counts per cleaning step)\n", f.to_markdown(index=False), "\n"]
    big = f[(f["drop_%"] > 10) & ~f["step"].isin(["core_blocks", "feature_code"])]
    md += ["**Steps dropping >10% (check):**\n", big[["pilot", "stage", "step", "drop_%"]].to_markdown(index=False)
           if len(big) else "none", "\n"]

    # ---- headline numbers per pilot
    rows = []
    for n, (blk, b, rc, van, tiles, fh) in R.items():
        v = blk[blk["valid"]]
        sb = b[(b["floors_src"] == "height") & b["floors_pluto"].gt(0)]
        rows.append({
            "pilot": n,
            "buildings": len(b),
            "link_rate": (b["link"] != "none").mean(),
            "link_mappluto/base/spatial": "/".join(f"{(b['link'] == k).mean():.2f}" for k in ["mappluto", "base", "spatial"]),
            "height_missing": b["flag_height_missing"].mean(),
            "floor_h_m": round(fh["floor_height_m"], 2),
            "floor_disagree(single-bldg lots)": b["flag_floor_disagree"].sum() / max(1, (b["floors_src"] == "pluto_single").sum()),
            "valid_blocks": len(v),
            "corr_FAR_before_mismatch_drop": cand(blk)["far_pluto"].corr(cand(blk)["far_fp"]),
            "corr_FAR_valid": v["far_pluto"].corr(v["far_fp"]),
            "median_lotarea/poly": v["lotarea_vs_poly"].median(),
            "corr_cov_vec_vs_ras": rc[rc["valid"]]["coverage_vec"].corr(rc[rc["valid"]]["coverage_ras"]),
            "mean|cov_vec-ras|": (rc["coverage_vec"] - rc["coverage_ras"]).abs().mean(),
            "bldg_vanished": 1 - van["survived"].mean() if len(van) else np.nan,
            "tiles": len(tiles), "tiles_water>10%": int(tiles["flag_water"].sum()),
        })
    head = pd.DataFrame(rows).set_index("pilot").T
    md += ["## Headline checks\n", head.to_markdown(floatfmt=".3f"), "\n",
           "Targets: link_rate ≥ 0.90; corr FAR ≈ 0.88 (Park Slope); lotarea/poly ≈ 1; "
           "|cov_vec − ras| small; floor height 2.5–5 m.\n"]

    # ---- fig 1: consistency scatters
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    scatter_panel(ax[0], {n: cand(r[0]) for n, r in R.items()}, "far_pluto", "far_fp",
                  "Block FAR: PLUTO vs footprint×floors (hollow = mismatch, excluded)",
                  hollow="flag_far_mismatch")
    scatter_panel(ax[1], {n: r[2][r[2]["valid"]] for n, r in R.items()}, "coverage_vec", "coverage_ras",
                  "Block coverage: vector vs 4 m raster", lim=1)
    sb = {n: r[1][(r[1]["floors_src"] == "pluto_single")] for n, r in R.items()}
    scatter_panel(ax[2], sb, "floors_pluto", "floors_h", "Floors: PLUTO vs roof height / floor h (1-bldg lots)")
    ax[0].legend(loc="upper left")
    fig.tight_layout(), fig.savefig(QA / "01_consistency.png", dpi=150), plt.close(fig)

    # ---- fig 2: block distributions
    cols = [("coverage", (0, 1)), ("far_pluto", None), ("median_floors", None), ("res_share", (0, 1))]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    for a, (c, rng) in zip(ax, cols):
        allv = pd.concat([r[0][r[0]["valid"]][c] for r in R.values()]).dropna()
        bins = np.linspace(*(rng or (0, allv.quantile(.99))), 21)
        for n, r in R.items():
            a.hist(r[0][r[0]["valid"]][c].dropna().clip(bins[0], bins[-1]), bins=bins, histtype="step",
                   lw=2, color=PILOT_COLOR[n], label=n)
        a.set_title(c, loc="left", color=INK)
        a.set_ylabel("blocks")
    ax[0].legend()
    fig.tight_layout(), fig.savefig(QA / "02_block_distributions.png", dpi=150), plt.close(fig)

    # ---- fig 3: class frequency + vanish by size
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    w = 0.27
    for k, (n, r) in enumerate(R.items()):
        t = r[4]
        cnt = np.array([t[f"n_{CLASSES[c]}"].sum() for c in range(10)], float)
        ax[0].bar(np.arange(10) + (k - 1) * w, cnt / cnt.sum(), w - 0.03, color=PILOT_COLOR[n], label=n)
        van = r[3]
        if len(van):
            bins = [0, 16, 30, 60, 120, 1e9]
            g = van.groupby(pd.cut(van["area_m2"], bins), observed=False)["survived"].agg(["mean", "size"])
            ax[1].plot(range(len(g)), 1 - g["mean"], marker="o", ms=6, lw=2, color=PILOT_COLOR[n], label=n)
    ax[0].set_xticks(range(10), [CLASSES[c] for c in range(10)], rotation=30)
    ax[0].set_title("Share of cells per class", loc="left"), ax[0].legend()
    ax[1].set_xticks(range(5), ["<16", "16–30", "30–60", "60–120", ">120"])
    ax[1].set_xlabel("footprint m²"), ax[1].set_title("Share of buildings lost at 4 m", loc="left")
    fig.tight_layout(), fig.savefig(QA / "03_classes_vanish.png", dpi=150), plt.close(fig)

    # ---- fig 4: tile mosaics per pilot
    cmap = ListedColormap(CLASS_COLORS)
    for n in PILOTS:
        tiles = R[n][4]
        nx = tiles["tile_id"].str.split("_").str[-2].astype(int).max() + 1
        ny = tiles["tile_id"].str.split("_").str[-1].astype(int).max() + 1
        fig, axs = plt.subplots(ny, nx, figsize=(3 * nx, 3 * ny + 0.8), squeeze=False)
        for _, t in tiles.iterrows():
            i, j = map(int, t["tile_id"].split("_")[-2:])
            z = np.load(OUT / n / "tiles" / f"{t['tile_id']}.npz")
            a = axs[ny - 1 - j, i]
            a.imshow(z["cls"], cmap=cmap, vmin=-0.5, vmax=9.5, interpolation="nearest")
            if z["water"].any():
                a.imshow(np.ma.masked_where(~z["water"], z["water"]), cmap=ListedColormap([WATER]), interpolation="nearest")
            a.set_xticks([]), a.set_yticks([]), a.grid(False)
            a.set_title(t["tile_id"], fontsize=8, color=MUTED)
        handles = [Patch(color=CLASS_COLORS[c], label=CLASSES[c]) for c in range(10)] + [Patch(color=WATER, label="water")]
        fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8)
        fig.suptitle(f"{n}: class maps (512 m tiles, 4 m cells)", x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0.07, 1, 0.97))
        fig.savefig(QA / f"04_tiles_{n}.png", dpi=130), plt.close(fig)

    # ---- fig 5: block map with flags per pilot (vector, for eyeballing against a real map)
    for n, (blk, b, *_rest) in R.items():
        fig, ax = plt.subplots(figsize=(8, 8))
        reason = np.select(
            [blk["valid"], ~blk["in_core"], blk["flag_incomplete"], blk["flag_too_big"] | blk["flag_too_small"] | blk["flag_thin"],
             blk["flag_construction"], blk["flag_far_mismatch"]],
            ["valid", "outside core", "incomplete", "size/shape", "under construction", "FAR mismatch"], "water/no PLUTO")
        palette = {"valid": "#cfe3c4", "outside core": "#efeee9", "incomplete": "#e1e0d9",
                   "size/shape": "#eda100", "under construction": "#6250d6", "FAR mismatch": "#e34948",
                   "water/no PLUTO": "#1baf7a"}
        for r_, c in palette.items():
            sub = blk[reason == r_]
            if len(sub):
                sub.plot(ax=ax, color=c, edgecolor="#fcfcfb", linewidth=0.8)
        b.plot(ax=ax, color=np.where(b["is_res"].fillna(False).astype(bool), "#256abf", "#c9521f"), linewidth=0)
        ax.legend(handles=[Patch(color=c, label=k) for k, c in palette.items()] +
                  [Patch(color="#256abf", label="res building"), Patch(color="#c9521f", label="non-res building")],
                  loc="upper left", fontsize=7)
        ax.set_axis_off(), ax.set_title(f"{n}: blocks by status", loc="left")
        fig.tight_layout(), fig.savefig(QA / f"05_blocks_{n}.png", dpi=130), plt.close(fig)

    # worst disagreements to inspect by hand
    md.append("## Blocks to inspect by hand (largest FAR disagreement)\n")
    for n, r in R.items():
        v = r[0][r[0]["valid"]].copy()
        v["far_diff"] = v["far_fp"] - v["far_pluto"]
        md += [f"### {n}\n", v.reindex(v["far_diff"].abs().sort_values(ascending=False).index)
               [["block_id", "area_ha", "coverage", "far_pluto", "far_fp", "median_floors", "res_share", "n_bldg"]]
               .head(8).to_markdown(index=False, floatfmt=".2f"), "\n"]
    (QA / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(head.to_string())
    print(f"\nwrote {QA}")


if __name__ == "__main__":
    main()
