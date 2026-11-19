import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from config import AREAS, CRS_LL, CRS_M, INTERIM, RAW


def norm_bbl(s):
    """'3010200035.00000000' / 3010200035 / '3010200035' -> '3010200035' (or <NA>)."""
    s = pd.Series(s, dtype="string").str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    return s.where(s.str.fullmatch(r"\d{10}"))


def is_flag(s):
    """Tax-lot flags are letters ('A' air lot, 'S' sub lot, ...) or null; treat any
    non-empty value other than N/0/false as set."""
    s = pd.Series(s, dtype="string").str.strip().str.upper()
    return s.notna() & (s != "") & ~s.isin(["N", "NO", "0", "FALSE", "F"])


def num(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_geojson(path):
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs(CRS_LL)
    return g.to_crs(CRS_M)


def pilot_dir(name, kind=INTERIM):
    d = kind / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def bbox_m(name, which="core_bbox"):
    """Core or download bbox of a pilot as a polygon in CRS_M."""
    meta = json.loads((RAW / name / "meta.json").read_text())
    b = meta[which] if which in meta else AREAS[name]["bbox"]
    return gpd.GeoSeries([box(*b)], crs=CRS_LL).to_crs(CRS_M).iloc[0]


def block_key(boro, block):
    return boro.astype(str) + "-" + block.astype(int).astype(str).str.zfill(5)


def read_raw_geo(name, kind):
    """Raw tax lots / footprints of an area: parquet (city download) or geojson (pilot)."""
    pq = RAW / name / f"{kind}.parquet"
    if pq.exists():
        g = gpd.read_parquet(pq)
        g = g.set_crs(CRS_LL) if g.crs is None else g
        for c in [c for c in g.columns if c != "geometry"]:     # same dtypes as geojson -> str
            g[c] = g[c].astype("string").astype(object).where(g[c].notna(), None)
        return g.to_crs(CRS_M)
    return read_geojson(RAW / name / f"{kind}.geojson")


def read_raw_pluto(name):
    pq = RAW / name / "pluto.parquet"
    if pq.exists():
        return pd.read_parquet(pq).astype("string").astype(object)
    return pd.read_csv(RAW / name / "pluto.csv", dtype=str)
