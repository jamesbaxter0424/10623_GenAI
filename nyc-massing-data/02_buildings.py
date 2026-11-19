"""Stage 2: clean building footprints, link them to lots/PLUTO, derive floors and use.

Out per pilot: data/interim/<pilot>/buildings.parquet
  bin, feature_code, area_m2, height_m, floors, floors_src, hclass, is_res,
  pluto_bbl, link (mappluto/base/spatial/none), block_id, flags
Also data/interim/<pilot>/floor_height.json (estimated metres per floor).
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd

import funnel
from common import norm_bbl, num, pilot_dir, read_raw_geo
from config import (FLOOR_DISAGREE, FLOOR_HEIGHT_FALLBACK, FLOOR_HEIGHT_M, FT,
                    KEEP_FEATURE_CODES, MAX_HEIGHT_M, MIN_FOOTPRINT_M2, AREAS, RAW,
                    height_class)


def main():
    funnel.reset("2_bldg", AREAS)
    for name in AREAS:
        print(f"== {name}")
        d = pilot_dir(name)
        pluto = pd.read_parquet(d / "pluto.parquet").set_index("bbl")
        lots = gpd.read_parquet(d / "lots.parquet")
        b = read_raw_geo(name, "footprints")
        b = num(b, ["feature_code", "height_roof", "bin", "construction_year"])

        n0 = len(b)
        codes = b["feature_code"].value_counts().to_dict()
        # remember lots with buildings under construction: their blocks' map is incomplete
        uc = b[b["feature_code"] == 5100]
        uc_bbl = set(norm_bbl(uc["mappluto_bbl"]).dropna()) | set(norm_bbl(uc["base_bbl"]).dropna())
        (d / "construction_bbls.json").write_text(json.dumps(sorted(uc_bbl)))
        b = b[b["feature_code"].isin(KEEP_FEATURE_CODES)].copy()
        funnel.log(name, "2_bldg", "feature_code", n0, len(b), f"all codes {codes}")

        n1 = len(b)
        b["geometry"] = b.geometry.make_valid()
        b = b[~b.geometry.is_empty & b.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        funnel.log(name, "2_bldg", "valid_geom", n1, len(b))

        b["area_m2"] = b.geometry.area
        n2 = len(b)
        b = b[b["area_m2"] >= MIN_FOOTPRINT_M2].copy()
        funnel.log(name, "2_bldg", f"area>={MIN_FOOTPRINT_M2}m2", n2, len(b))

        # height
        b["height_m"] = b["height_roof"] * FT
        b["flag_height_missing"] = ~(b["height_m"] > 0)
        b["flag_height_absurd"] = b["height_m"] > MAX_HEIGHT_M
        b.loc[b["flag_height_missing"] | b["flag_height_absurd"], "height_m"] = np.nan

        # link to PLUTO: mappluto_bbl -> base_bbl -> spatial (point in lot)
        b["mappluto_bbl"] = norm_bbl(b["mappluto_bbl"]).to_numpy()
        b["base_bbl"] = norm_bbl(b["base_bbl"]).to_numpy()
        pts = b.geometry.representative_point()
        sj = gpd.sjoin(gpd.GeoDataFrame(geometry=pts, crs=b.crs),
                       lots[["pluto_bbl", "block_id", "geometry"]], predicate="within")
        sj = sj[~sj.index.duplicated()]
        b["lot_pluto_bbl"] = sj["pluto_bbl"].reindex(b.index)
        b["block_id"] = sj["block_id"].reindex(b.index)        # geometry is authoritative for block

        b["pluto_bbl"] = pd.NA
        b["link"] = "none"
        for col, tag in [("mappluto_bbl", "mappluto"), ("base_bbl", "base"), ("lot_pluto_bbl", "spatial")]:
            ok = b["pluto_bbl"].isna() & b[col].isin(pluto.index)
            b.loc[ok, "pluto_bbl"] = b.loc[ok, col]
            b.loc[ok, "link"] = tag
        vc = b["link"].value_counts().to_dict()
        funnel.log(name, "2_bldg", "link_pluto", len(b), int((b["link"] != "none").sum()), str(vc))
        b["flag_block_mismatch"] = b["pluto_bbl"].notna() & b["block_id"].notna() & (
            b["pluto_bbl"].map(pluto["block_id"]) != b["block_id"])

        n3 = len(b)
        b = b[b["block_id"].notna()].copy()
        funnel.log(name, "2_bldg", "inside_a_lot", n3, len(b), "footprint point not in any lot -> drop")

        # floors
        pl_floors = b["pluto_bbl"].map(pluto["numfloors"])
        pl_nbldg = b["pluto_bbl"].map(pluto["numbldgs"])
        fh = FLOOR_HEIGHT_M
        if fh is None:
            per_lot = b[b["feature_code"] == 2100].groupby("pluto_bbl").size()
            single = b[(b["feature_code"] == 2100) & (pl_nbldg == 1)
                       & (b["pluto_bbl"].map(per_lot) == 1) & (pl_floors > 0) & b["height_m"].notna()]
            ratio = single["height_m"] / pl_floors[single.index]
            fh = float(ratio.median()) if len(ratio) >= 30 else FLOOR_HEIGHT_FALLBACK
            info = {"floor_height_m": fh, "n_single_building_lots": int(len(ratio)),
                    "q25": float(ratio.quantile(.25)), "q75": float(ratio.quantile(.75)),
                    "share_2.5_to_5m": float(ratio.between(2.5, 5).mean())}
        else:
            info = {"floor_height_m": fh, "source": "config"}
        (d / "floor_height.json").write_text(json.dumps(info, indent=2))
        print(f"    floor height: {info}")

        # Floors: official PLUTO count when the lot holds exactly this one building
        # (roof height / floor height miscounts tall 1-storey warehouses and towers
        # with mechanical floors); roof height otherwise.
        fl_h = np.maximum(1, np.round(b["height_m"] / fh))
        n_main = b[b["feature_code"] == 2100].groupby("pluto_bbl").size()
        single = (b["feature_code"] == 2100) & (pl_nbldg == 1) & (b["pluto_bbl"].map(n_main) == 1) & (pl_floors > 0)
        b["floors"] = np.where(single, pl_floors, fl_h)
        b["floors_src"] = np.select([single, fl_h.notna()], ["pluto_single", "height"], None)
        use_pl = b["floors"].isna() & (pl_floors > 0) & (pl_nbldg <= 1)
        b.loc[use_pl, "floors"] = pl_floors[use_pl]
        b.loc[use_pl, "floors_src"] = "pluto"
        gar = b["floors"].isna() & (b["feature_code"] == 5110)
        b.loc[gar, "floors"], b.loc[gar, "floors_src"] = 1, "garage_default"
        rest = b["floors"].isna() & (pl_floors > 0)
        b.loc[rest, "floors"], b.loc[rest, "floors_src"] = pl_floors[rest], "pluto_multi_bldg"
        b["floors_src"] = b["floors_src"].fillna("none")
        print(f"    floors_src: {b['floors_src'].value_counts().to_dict()}")
        b["floors_pluto"] = pl_floors
        b["floors_h"] = fl_h
        b["flag_floor_disagree"] = single & ((fl_h - pl_floors).abs() > FLOOR_DISAGREE)

        n4 = len(b)
        b = b[b["floors"].notna()].copy()
        funnel.log(name, "2_bldg", "has_floors", n4, len(b))
        b["hclass"] = height_class(b["floors"])

        # use from linked lot; unknown filled later from block majority (stage 3)
        b["is_res"] = b["pluto_bbl"].map(pluto["is_res"]).astype("boolean")
        for c in [c for c in b if c.startswith("flag_")]:
            print(f"    {c}: {int(b[c].sum())} / {len(b)}")

        keep = ["bin", "feature_code", "area_m2", "height_m", "floors", "floors_src", "floors_pluto", "floors_h",
                "hclass", "is_res", "pluto_bbl", "link", "block_id", "construction_year"] + \
               [c for c in b if c.startswith("flag_")] + ["geometry"]
        b[keep].to_parquet(d / "buildings.parquet")


if __name__ == "__main__":
    main()
