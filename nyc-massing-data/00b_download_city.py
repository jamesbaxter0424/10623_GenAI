"""Stage 0 (city): download PLUTO, tax lots and building footprints for all of NYC.

Paged through the Socrata API, one parquet per page under data/raw/nyc/<ds>/ so an
interrupted run resumes where it stopped. Then split per borough into
data/raw/<BORO>/{pluto.parquet, taxlots.parquet, footprints.parquet, meta.json}
so stages 1-3 run on each borough exactly as on a pilot area.

    python 00b_download_city.py            # download (resumes) + split
    python 00b_download_city.py --split    # only re-split
"""
import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import geopandas as gpd
import pandas as pd
import requests

from config import CITY, CRS_LL, DS_FOOTPRINTS, DS_PLUTO, DS_TAXLOTS, RAW

BASE = "https://data.cityofnewyork.us/resource"
PAGE = 50_000
SELECT = {
    DS_PLUTO: ["borough", "block", "lot", "bbl", "cd", "landuse", "bldgclass", "lotarea", "bldgarea",
               "resarea", "comarea", "numbldgs", "numfloors", "unitsres", "builtfar", "residfar",
               "commfar", "facilfar", "zonedist1", "yearbuilt", "latitude", "longitude", "version"],
    DS_TAXLOTS: ["the_geom", "bbl", "boro", "block", "lot", "air_lot_flag", "sub_lot_flag",
                 "condo_flag", "easement_flag"],
    DS_FOOTPRINTS: ["the_geom", "bin", "base_bbl", "mappluto_bbl", "feature_code", "height_roof",
                    "construction_year"],
}
NAMES = {DS_PLUTO: "pluto", DS_TAXLOTS: "taxlots", DS_FOOTPRINTS: "footprints"}
DIGIT = {"1": "MN", "2": "BX", "3": "BK", "4": "QN", "5": "SI"}


def get(url, params):
    for attempt in range(6):
        try:
            r = requests.get(url, params=params, timeout=600)
            if r.ok:
                return r
            print(f"    HTTP {r.status_code}, retry {attempt + 1}: {r.text[:200]}")
        except requests.RequestException as e:
            print(f"    {type(e).__name__}, retry {attempt + 1}")
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {url} {params}")


def count(ds):
    r = get(f"{BASE}/{ds}.json", {"$select": "count(*)"})
    return int(r.json()[0]["count"])


def fetch_page(ds, k, out):
    f = out / f"{k:04d}.parquet"
    if f.exists():
        return
    params = {"$select": ",".join(SELECT[ds]), "$limit": PAGE, "$offset": k * PAGE, "$order": ":id"}
    t = time.time()
    if ds == DS_PLUTO:
        df = pd.read_csv(io.StringIO(get(f"{BASE}/{ds}.csv", params).text), dtype=str)
    else:
        feats = get(f"{BASE}/{ds}.geojson", params).json()["features"]
        df = gpd.GeoDataFrame.from_features(feats, crs=CRS_LL)
    tmp = f.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(f)
    print(f"    {NAMES[ds]} page {k}: {len(df)} rows, {time.time() - t:.0f}s")


def download():
    for ds in [DS_PLUTO, DS_TAXLOTS, DS_FOOTPRINTS]:
        out = RAW / "nyc" / NAMES[ds]
        out.mkdir(parents=True, exist_ok=True)
        n = count(ds)
        pages = (n + PAGE - 1) // PAGE
        print(f"== {NAMES[ds]}: {n} rows, {pages} pages")
        with ThreadPoolExecutor(3) as ex:
            list(ex.map(lambda k: fetch_page(ds, k, out), range(pages)))
        got = sum(len(pd.read_parquet(f, columns=[SELECT[ds][1]])) for f in out.glob("*.parquet"))
        assert got == n, f"{NAMES[ds]}: {got} rows on disk, {n} on server"


def read_all(name):
    files = sorted((RAW / "nyc" / name).glob("*.parquet"))
    if name == "pluto":
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return gpd.GeoDataFrame(pd.concat([gpd.read_parquet(f) for f in files], ignore_index=True), crs=CRS_LL)


def first_digit(s):
    return pd.Series(s, dtype="string").str.strip().str[0]


def split():
    pluto, lots, fp = read_all("pluto"), read_all("taxlots"), read_all("footprints")
    lots["_b"] = first_digit(lots["boro"].astype(str))
    fp["_b"] = first_digit(fp["base_bbl"]).fillna(first_digit(fp["mappluto_bbl"]))
    print(f"footprints without a BBL borough: {int(fp['_b'].isna().sum())} / {len(fp)} (dropped)")
    version = sorted(pluto["version"].dropna().unique().tolist())
    for digit, boro in DIGIT.items():
        d = RAW / boro
        d.mkdir(exist_ok=True)
        lb = lots[lots["_b"] == digit].drop(columns="_b")
        lb.to_parquet(d / "taxlots.parquet")
        fp[fp["_b"] == digit].drop(columns="_b").to_parquet(d / "footprints.parquet")
        pluto[pluto["borough"] == boro].to_parquet(d / "pluto.parquet")
        w, s, e, n = lb.total_bounds
        bbox = [float(w) - 0.01, float(s) - 0.01, float(e) + 0.01, float(n) + 0.01]
        meta = {"downloaded": str(date.today()), "core_bbox": bbox, "download_bbox": bbox,
                "margin_m": 0, "pluto_version": version, "mode": "borough",
                "datasets": {"pluto": DS_PLUTO, "footprints": DS_FOOTPRINTS, "taxlots": DS_TAXLOTS}}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{boro}: lots {len(lb)}, footprints {int((fp['_b'] == digit).sum())}, "
              f"pluto {int((pluto['borough'] == boro).sum())}")
    assert set(CITY) == set(DIGIT.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true", help="skip download, only split")
    a = ap.parse_args()
    if not a.split:
        download()
    split()
