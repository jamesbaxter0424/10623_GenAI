"""Stage 1: clean tax-lot polygons and PLUTO, join them, build the land mask.

Out per pilot (data/interim/<pilot>/):
  pluto.parquet  cleaned PLUTO table, one row per BBL, with use + QA flags
  lots.parquet   tax-lot polygons (base lots only) with matched PLUTO BBL
  land.parquet   land polygon (water excluded) clipped to the download bbox
"""
import geopandas as gpd
import numpy as np
import pandas as pd

import funnel
from common import block_key, is_flag, norm_bbl, num, pilot_dir, read_geojson, read_raw_geo, read_raw_pluto, bbox_m
from config import (BUILTFAR_OUTLIER, CRS_M, DROP_LOT_FLAGS, AREAS,
                    PLUTO_FAR_PER_FLOOR_MAX, RAW, RES_SHARE_THRESHOLD)

BORO_CODE = {"MN": "1", "BX": "2", "BK": "3", "QN": "4", "SI": "5"}
PLUTO_NUM = ["block", "lot", "cd", "landuse", "lotarea", "bldgarea", "resarea", "comarea",
             "numbldgs", "numfloors", "unitsres", "builtfar", "yearbuilt", "latitude", "longitude"]


def clean_pluto(name):
    p = read_raw_pluto(name)
    n0 = len(p)
    p = num(p, PLUTO_NUM)
    p["bbl"] = norm_bbl(p["bbl"])
    p = p[p["bbl"].notna()]
    funnel.log(name, "1_lots", "pluto_valid_bbl", n0, len(p))
    n1 = len(p)
    p = p.drop_duplicates("bbl")
    funnel.log(name, "1_lots", "pluto_dedupe_bbl", n1, len(p))

    p["boro"] = p["borough"].map(BORO_CODE)
    p["block_id"] = block_key(p["boro"], p["block"])
    p["is_condo_bill"] = p["lot"].between(7501, 7599)

    # missing, not zero
    p["flag_lotarea_missing"] = ~(p["lotarea"] > 0)
    p["flag_bldgarea_missing"] = p["bldgarea"].isna() | ((p["bldgarea"] == 0) & (p["numbldgs"] > 0))
    p["flag_landuse_missing"] = p["landuse"].isna()
    p["flag_far_outlier"] = (p["builtfar"] > BUILTFAR_OUTLIER) & (p["borough"] != "MN")
    # physically impossible: more floor area than floors x lot area (e.g. 445 5th Ave, FAR 131 on 6 floors)
    p["flag_far_impossible"] = (p["numfloors"] > 0) & (
        p["builtfar"] > p["numfloors"] * PLUTO_FAR_PER_FLOOR_MAX + 0.5)

    # use: share of residential floor area, fallback to land-use code
    share = p["resarea"] / p["bldgarea"].where(p["bldgarea"] > 0)
    p["res_share"] = share.clip(0, 1)
    p["is_res"] = np.where(share.notna(), share >= RES_SHARE_THRESHOLD,
                           p["landuse"].isin([1, 2, 3]))
    for c in [c for c in p if c.startswith("flag_")]:
        print(f"    {c}: {int(p[c].sum())} / {len(p)}")
    return p


def clean_lots(name, pluto):
    g = read_raw_geo(name, "taxlots")
    n0 = len(g)
    g["geometry"] = g.geometry.make_valid()
    g = g[~g.geometry.is_empty & g.geometry.geom_type.isin(["Polygon", "MultiPolygon", "GeometryCollection"])]
    g["geometry"] = g.geometry.apply(_polys_only)
    g = g[~g.geometry.is_empty]
    funnel.log(name, "1_lots", "taxlot_valid_geom", n0, len(g))

    n1 = len(g)
    drop = np.zeros(len(g), bool)
    for c in DROP_LOT_FLAGS:
        if c in g:
            drop |= is_flag(g[c]).to_numpy()
    # Air/sub lots overlap a base lot -> drop to avoid double counting. But over rail
    # yards (Penn Station, Grand Central) there is no base lot polygon, so keep an
    # air/sub lot unless it is mostly (>50%) covered by ordinary lots.
    # area of each flagged lot covered by base lots (base lots do not overlap each other,
    # so summing pairwise intersections is exact; avoids a borough-wide union)
    flagged = g[drop]
    base = g[~drop][["geometry"]]
    pairs = gpd.sjoin(flagged[["geometry"]], base, predicate="intersects")
    other = gpd.GeoSeries(base.geometry.loc[pairs["index_right"]].to_numpy(), index=pairs.index, crs=g.crs)
    inter = pairs.geometry.intersection(other).area
    covered = inter.groupby(level=0).sum().reindex(flagged.index, fill_value=0) / flagged.geometry.area
    redundant = pd.Series(False, index=g.index)
    redundant.loc[flagged.index] = covered > 0.5
    g = g[~redundant.to_numpy()]
    funnel.log(name, "1_lots", "taxlot_drop_air_sub", n1, len(g),
               f"{int(drop.sum())} flagged, {int(drop.sum() - redundant.sum())} kept (no base lot below)")

    g["bbl"] = norm_bbl(g["bbl"])
    n2 = len(g)
    g = g[g["bbl"].notna()].dissolve("bbl", as_index=False, aggfunc="first")
    funnel.log(name, "1_lots", "taxlot_dedupe_bbl", n2, len(g), "dissolve duplicate BBL")
    g["block_id"] = block_key(g["boro"], pd.to_numeric(g["block"]))

    # join to PLUTO: (1) same BBL, (2) PLUTO lot point inside polygon (condos: billing BBL 75xx)
    g["pluto_bbl"] = g["bbl"].where(g["bbl"].isin(pluto["bbl"]))
    g["pluto_match"] = np.where(g["pluto_bbl"].notna(), "bbl", None)
    pts = pluto[~pluto["bbl"].isin(g["pluto_bbl"].dropna())].dropna(subset=["longitude"])
    pts = gpd.GeoDataFrame(pts[["bbl"]], geometry=gpd.points_from_xy(pts.longitude, pts.latitude),
                           crs="EPSG:4326").to_crs(CRS_M)
    hit = gpd.sjoin(pts, g[g["pluto_bbl"].isna()][["bbl", "geometry"]].rename(columns={"bbl": "lot_bbl"}),
                    predicate="within")
    hit = hit.drop_duplicates("lot_bbl")  # one PLUTO record per lot polygon
    m = g["bbl"].map(hit.set_index("lot_bbl")["bbl"])
    g.loc[m.notna(), "pluto_bbl"] = m[m.notna()]
    g.loc[m.notna(), "pluto_match"] = "point_in_lot"
    vc = g["pluto_match"].fillna("none").value_counts().to_dict()
    funnel.log(name, "1_lots", "taxlot_pluto_join", len(g), int(g["pluto_bbl"].notna().sum()), str(vc))
    keep = ["bbl", "boro", "block", "lot", "block_id", "pluto_bbl", "pluto_match",
            "condo_flag", "easement_flag", "geometry"]
    return g[[c for c in keep if c in g]]


def _polys_only(geom):
    from shapely.geometry import MultiPolygon
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [p for p in getattr(geom, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
    flat = []
    for p in parts:
        flat += list(p.geoms) if p.geom_type == "MultiPolygon" else [p]
    return MultiPolygon(flat) if flat else MultiPolygon()


def land_mask(name):
    b = read_geojson(RAW / "boroughs.geojson")
    dl = bbox_m(name, "download_bbox")
    land = b.geometry.make_valid().union_all().intersection(dl)
    return gpd.GeoDataFrame(geometry=[land], crs=CRS_M)


def main():
    funnel.reset("1_lots", AREAS)
    for name in AREAS:
        print(f"== {name}")
        d = pilot_dir(name)
        pluto = clean_pluto(name)
        lots = clean_lots(name, pluto)
        pluto.to_parquet(d / "pluto.parquet")
        lots.to_parquet(d / "lots.parquet")
        land_mask(name).to_parquet(d / "land.parquet")


if __name__ == "__main__":
    main()
