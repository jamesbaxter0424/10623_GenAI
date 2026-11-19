"""Stage 0: download only the pilot areas from NYC Open Data (Socrata API).

Output per pilot: data/raw/<pilot>/{taxlots,footprints}.geojson, pluto.csv, meta.json
plus data/raw/boroughs.geojson (land mask). Re-running skips files that exist
unless --force is given.
"""
import argparse
import io
import json
import math
import time
from datetime import date

import pandas as pd
import requests

from config import (DS_BOROUGHS, DS_FOOTPRINTS, DS_PLUTO, DS_TAXLOTS, MARGIN_M,
                    PILOTS, RAW)

BASE = "https://data.cityofnewyork.us/resource"
PAGE = 50_000


def get(url, params):
    for attempt in range(4):
        r = requests.get(url, params=params, timeout=300)
        if r.ok:
            return r
        print(f"    HTTP {r.status_code}, retry {attempt + 1}: {r.text[:200]}")
        time.sleep(3 * (attempt + 1))
    r.raise_for_status()


def fetch_geojson(ds, where):
    feats = []
    offset = 0
    while True:
        r = get(f"{BASE}/{ds}.geojson",
                {"$where": where, "$limit": PAGE, "$offset": offset, "$order": ":id"})
        page = r.json()["features"]
        feats += page
        print(f"    {ds}: {len(feats)} features")
        if len(page) < PAGE:
            break
        offset += PAGE
    return {"type": "FeatureCollection", "features": feats}


def fetch_csv(ds, where):
    frames, offset = [], 0
    while True:
        r = get(f"{BASE}/{ds}.csv",
                {"$where": where, "$limit": PAGE, "$offset": offset, "$order": ":id"})
        df = pd.read_csv(io.StringIO(r.text), dtype=str)
        frames.append(df)
        if len(df) < PAGE:
            break
        offset += PAGE
    return pd.concat(frames, ignore_index=True)


def expand(bbox, m):
    w, s, e, n = bbox
    dlat = m / 111_320
    dlon = m / (111_320 * math.cos(math.radians((s + n) / 2)))
    return (w - dlon, s - dlat, e + dlon, n + dlat)


def within_box(bbox):
    """Features that INTERSECT the box. (Socrata's within_box() only returns features
    fully inside it, which silently drops large lots such as Prospect Park.)"""
    w, s, e, n = bbox
    return f"intersects(the_geom, 'POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))')"


def main(force=False):
    RAW.mkdir(parents=True, exist_ok=True)
    bpath = RAW / "boroughs.geojson"
    if force or not bpath.exists():
        print("boroughs (land mask)")
        bpath.write_text(json.dumps(fetch_geojson(DS_BOROUGHS, "1=1")))

    for name, p in PILOTS.items():
        d = RAW / name
        d.mkdir(exist_ok=True)
        dl = expand(p["bbox"], MARGIN_M)
        print(f"== {name}  download bbox {tuple(round(v, 5) for v in dl)}")

        if force or not (d / "taxlots.geojson").exists():
            (d / "taxlots.geojson").write_text(json.dumps(fetch_geojson(DS_TAXLOTS, within_box(dl))))
        if force or not (d / "footprints.geojson").exists():
            (d / "footprints.geojson").write_text(json.dumps(fetch_geojson(DS_FOOTPRINTS, within_box(dl))))

        # PLUTO: exactly the (borough, block)s present in the tax lots, so block sums are complete.
        if force or not (d / "pluto.csv").exists():
            lots = json.loads((d / "taxlots.geojson").read_text())["features"]
            boro_code = {"1": "MN", "2": "BX", "3": "BK", "4": "QN", "5": "SI"}
            keys = sorted({(boro_code[f["properties"]["boro"]], int(float(f["properties"]["block"])))
                           for f in lots})
            frames = []
            for boro in sorted({k[0] for k in keys}):
                blocks = [b for bb, b in keys if bb == boro]
                for i in range(0, len(blocks), 150):
                    chunk = ",".join(str(b) for b in blocks[i:i + 150])
                    frames.append(fetch_csv(DS_PLUTO, f"borough='{boro}' AND block in({chunk})"))
            pluto = pd.concat(frames, ignore_index=True)
            pluto.to_csv(d / "pluto.csv", index=False)
            print(f"    pluto: {len(pluto)} lots in {len(keys)} blocks")

        meta = {
            "downloaded": str(date.today()),
            "core_bbox": p["bbox"],
            "download_bbox": dl,
            "margin_m": MARGIN_M,
            "datasets": {"pluto": DS_PLUTO, "footprints": DS_FOOTPRINTS,
                         "taxlots": DS_TAXLOTS, "boroughs": DS_BOROUGHS},
        }
        pl = pd.read_csv(d / "pluto.csv", dtype=str, usecols=lambda c: c == "version")
        meta["pluto_version"] = sorted(pl["version"].dropna().unique().tolist())
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        sizes = {f.name: f"{f.stat().st_size / 1e6:.1f} MB" for f in d.iterdir()}
        print(f"    files: {sizes}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
