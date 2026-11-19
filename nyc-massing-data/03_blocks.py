"""Stage 3: block polygons + per-block targets (coverage, FAR, median floors, residential share).

A block is one connected piece of the union of a tax block's lots, so a tax block
cut by a street (Broadway in Midtown) gives two blocks: "1-00864-a", "1-00864-b".

Out per pilot (data/interim/<pilot>/):
  blocks.parquet           one row per block, attributes + flags + `valid`
  buildings_final.parquet  buildings with block = piece, unknown use filled from block
"""
import json
import string

import geopandas as gpd
import numpy as np
import pandas as pd

import funnel
from common import bbox_m, pilot_dir
from config import (EXCLUDE_CONSTRUCTION, EXCLUDE_MISMATCH, MAX_BLOCK_HA, MIN_BLOCK_HA,
                    MIN_COMPACTNESS, MISMATCH_ABS, MISMATCH_RATIO, AREAS,
                    RES_SHARE_THRESHOLD, SQFT, height_class)


def weighted_median(x, w):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    c = np.cumsum(w)
    return x[np.searchsorted(c, c[-1] / 2)]


def assign_to_piece(gdf, pieces):
    """Map each row (by representative point, same tax block) to a piece block_id."""
    pts = gpd.GeoDataFrame({"tax_block": gdf["block_id"].to_numpy()}, index=gdf.index,
                           geometry=gdf.geometry.representative_point(), crs=gdf.crs)
    pc = pieces[["block_id", "tax_block", "geometry"]].rename(columns={"block_id": "piece"})
    hit = gpd.sjoin(pts, pc, predicate="within")
    hit = hit[hit["tax_block_left"] == hit["tax_block_right"]]
    out = hit[~hit.index.duplicated()]["piece"].reindex(gdf.index)
    miss = out.isna()
    if miss.any():      # point fell in a sliver: nearest piece of the same tax block
        near = gpd.sjoin_nearest(pts[miss], pc, max_distance=5)
        near = near[near["tax_block_left"] == near["tax_block_right"]]
        out.loc[near.index.unique()] = near[~near.index.duplicated()]["piece"]
    return out


def _suffix(i):
    """0 -> a, 25 -> z, 26 -> aa, 27 -> ab ..."""
    L = string.ascii_lowercase
    return L[i] if i < 26 else L[i // 26 - 1] + L[i % 26]


def build_blocks(name, lots):
    tb = lots.dissolve("block_id", as_index=False)[["block_id", "geometry"]]
    tb["geometry"] = tb.geometry.buffer(0.05).buffer(-0.05)      # close slivers between lots
    parts = tb.explode(index_parts=False).reset_index(drop=True)
    parts = parts[parts.area > 1.0].copy()
    parts["a"] = parts.area
    parts = parts.sort_values(["block_id", "a"], ascending=[True, False])
    parts["tax_block"] = parts["block_id"]
    parts["n_parts"] = parts.groupby("tax_block")["a"].transform("size")
    k = parts.groupby("tax_block").cumcount()
    parts["block_id"] = parts["tax_block"] + "-" + k.map(_suffix)
    blk = gpd.GeoDataFrame(parts.drop(columns="a"), geometry="geometry", crs=lots.crs).reset_index(drop=True)

    blk["area_ha"] = blk.area / 1e4
    blk["compactness"] = 4 * np.pi * blk.area / blk.length ** 2
    blk["flag_too_big"] = blk["area_ha"] > MAX_BLOCK_HA
    blk["flag_too_small"] = blk["area_ha"] < MIN_BLOCK_HA
    blk["flag_thin"] = blk["compactness"] < MIN_COMPACTNESS

    dl = bbox_m(name, "download_bbox")
    blk["flag_incomplete"] = ~blk.within(dl.buffer(-2))          # cut by the download box
    blk["in_core"] = blk.representative_point().within(bbox_m(name, "core_bbox"))

    land = gpd.read_parquet(pilot_dir(name) / "land.parquet").geometry.iloc[0]
    import shapely
    shapely.prepare(land)
    on_land = blk.within(land)
    blk["water_share"] = 0.0
    off = ~on_land
    blk.loc[off, "water_share"] = 1 - blk[off].intersection(land).area / blk[off].area
    blk["flag_water"] = blk["water_share"] > 0.10
    return blk


def main():
    funnel.reset("3_block", AREAS)
    for name in AREAS:
        print(f"== {name}")
        d = pilot_dir(name)
        lots = gpd.read_parquet(d / "lots.parquet")
        pluto = pd.read_parquet(d / "pluto.parquet")
        b = gpd.read_parquet(d / "buildings.parquet")
        uc_bbl = set(json.loads((d / "construction_bbls.json").read_text()))
        blk = build_blocks(name, lots)
        split = blk[blk["n_parts"] > 1]["tax_block"].nunique()
        print(f"    tax blocks split by streets into pieces: {split}")

        lots["piece"] = assign_to_piece(lots, blk)
        n0 = len(b)
        b["piece"] = assign_to_piece(b, blk)
        b = b[b["piece"].notna()].copy()
        funnel.log(name, "3_block", "bldg_in_piece", n0, len(b))
        b["tax_block"], b["block_id"] = b["block_id"], b["piece"]

        # footprint part inside its block
        ov = gpd.overlay(b.reset_index(names="bidx")[["bidx", "floors", "is_res", "block_id", "geometry"]],
                         blk[["block_id", "geometry"]].rename(columns={"block_id": "blk"}),
                         how="intersection", keep_geom_type=True)
        ov = ov[ov["block_id"] == ov["blk"]]
        ov["a"] = ov.area
        ov["gfa"] = ov["a"] * ov["floors"]
        agg = ov.groupby("blk").agg(fp_area=("a", "sum"), gfa_fp=("gfa", "sum"), n_bldg=("bidx", "nunique"))
        agg["median_floors"] = ov.groupby("blk").apply(lambda t: weighted_median(t["floors"], t["a"]),
                                                       include_groups=False)
        known = ov[ov["is_res"].notna()]
        agg["res_share_fp"] = known.groupby("blk").apply(
            lambda t: (t["gfa"] * t["is_res"].astype(float)).sum() / t["gfa"].sum() if t["gfa"].sum() else np.nan,
            include_groups=False)
        blk = blk.merge(agg, left_on="block_id", right_index=True, how="left")
        blk[["fp_area", "gfa_fp", "n_bldg"]] = blk[["fp_area", "gfa_fp", "n_bldg"]].fillna(0)

        # PLUTO sums per piece, through the lot polygons that carry a PLUTO record
        lp = lots[lots["pluto_bbl"].notna() & lots["piece"].notna()][["pluto_bbl", "bbl", "piece"]]
        lp = lp.merge(pluto, left_on="pluto_bbl", right_on="bbl", suffixes=("_lot", ""))
        lp["uc"] = lp["bbl"].isin(uc_bbl) | lp["bbl_lot"].isin(uc_bbl)
        pa = lp.groupby("piece").agg(lotarea=("lotarea", "sum"), bldgarea=("bldgarea", "sum"),
                                     resarea=("resarea", "sum"), n_lots=("bbl", "size"),
                                     n_far_impossible=("flag_far_impossible", "sum"),
                                     flag_construction=("uc", "any"))
        print(f"    PLUTO records reached through lot polygons: {lp['bbl'].nunique()} / {len(pluto)}")
        blk = blk.merge(pa, left_on="block_id", right_index=True, how="left")
        blk["flag_no_pluto"] = blk["n_lots"].isna()
        blk["flag_construction"] = blk["flag_construction"].fillna(False).astype(bool)
        blk["flag_pluto_suspect"] = blk["n_far_impossible"].fillna(0) > 0

        a = blk.area
        blk["coverage"] = blk["fp_area"] / a
        blk["far_fp"] = blk["gfa_fp"] / a                          # training target (matches the map)
        blk["far_pluto"] = (blk["bldgarea"] / blk["lotarea"].where(blk["lotarea"] > 0)).where(~blk["flag_pluto_suspect"])
        blk["lotarea_vs_poly"] = blk["lotarea"] * SQFT / a        # should be ~1
        blk["res_share"] = (blk["resarea"] / blk["bldgarea"].where(blk["bldgarea"] > 0)).clip(0, 1)
        blk["res_share"] = blk["res_share"].fillna(blk["res_share_fp"])
        blk["is_res"] = blk["res_share"] >= RES_SHARE_THRESHOLD
        blk["hclass"] = np.where(blk["median_floors"].notna(), height_class(blk["median_floors"].fillna(1)), -1)
        blk["flag_coverage_bad"] = blk["coverage"] > 1.001
        diff = (blk["far_fp"] - blk["far_pluto"]).abs()
        ratio = np.maximum(blk["far_fp"], blk["far_pluto"]) / np.minimum(blk["far_fp"], blk["far_pluto"]).clip(lower=0.05)
        blk["flag_far_mismatch"] = (diff > MISMATCH_ABS) & (ratio > MISMATCH_RATIO)

        # validity, logged on core blocks only
        core = blk[blk["in_core"]]
        funnel.log(name, "3_block", "core_blocks", len(blk), len(core), "blocks whose point is in core bbox")
        checks = ["flag_incomplete", "flag_no_pluto", "flag_too_big", "flag_too_small", "flag_thin",
                  "flag_water", "flag_coverage_bad"]
        checks += ["flag_construction"] if EXCLUDE_CONSTRUCTION else []
        checks += ["flag_far_mismatch"] if EXCLUDE_MISMATCH else []
        valid = pd.Series(True, index=core.index)
        for f in checks:
            before = int(valid.sum())
            valid &= ~core[f]
            funnel.log(name, "3_block", f, before, int(valid.sum()))
        print(f"    (info) pieces of split tax blocks in core: {int((core['n_parts'] > 1).sum())}, "
              f"PLUTO-suspect blocks: {int(core['flag_pluto_suspect'].sum())}")
        blk["valid"] = False
        blk.loc[valid.index, "valid"] = valid

        # fill unknown building use from block
        b["use_src"] = np.where(b["is_res"].notna(), "lot", "block")
        fill = b["is_res"].isna()
        b.loc[fill, "is_res"] = b.loc[fill, "block_id"].map(blk.set_index("block_id")["is_res"])
        print(f"    building use filled from block: {int(fill.sum())}")

        blk.to_parquet(d / "blocks.parquet")
        b.drop(columns="piece").to_parquet(d / "buildings_final.parquet")
        v = blk[blk["valid"]]
        print(f"    valid blocks {len(v)}; corr(far_pluto, far_fp) = {v['far_pluto'].corr(v['far_fp']):.3f}; "
              f"median lotarea/poly = {v['lotarea_vs_poly'].median():.3f}")


if __name__ == "__main__":
    main()
