"""Shared settings for the NYC block pilot pipeline.

Every cleaning rule that the team still has to decide on is a constant here,
so changing a decision means editing one line and re-running the stages.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "data" / "processed"
QA = ROOT / "qa"

# Metric CRS for all geometry work (UTM 18N, metres).
CRS_M = "EPSG:32618"
CRS_LL = "EPSG:4326"

# Pilot areas: core bbox in lon/lat (west, south, east, north).
# We download core + MARGIN_M so that blocks at the edge of the core are complete;
# blocks touching the download boundary are flagged `incomplete` and excluded.
PILOTS = {
    "park_slope": {"boro": "BK", "bbox": (-73.990, 40.664, -73.966, 40.682)},  # row houses
    "midtown":    {"boro": "MN", "bbox": (-73.995, 40.748, -73.975, 40.762)},  # towers
    "lic":        {"boro": "QN", "bbox": (-73.950, 40.736, -73.926, 40.754)},  # industrial + new towers + rail yard
}
MARGIN_M = 300

# Full city: one area per borough (blocks never cross a borough line).
# Raw data comes from 00b_download_city.py; core = download = whole borough.
CITY = {b: {"boro": b} for b in ["MN", "BX", "BK", "QN", "SI"]}

# Which areas stages 1-4 run on: NYC_SET=pilot (default) or NYC_SET=city,
# or one area, e.g. NYC_SET=BK.
import os as _os
_set = _os.environ.get("NYC_SET", "pilot")
AREAS = {"pilot": PILOTS, "city": CITY}.get(_set) or {_set: {**PILOTS, **CITY}[_set]}

# Socrata dataset ids (NYC Open Data)
DS_PLUTO = "64uk-42ks"
DS_FOOTPRINTS = "5zhs-2jue"
DS_TAXLOTS = "i38t-6if2"
DS_BOROUGHS = "gthc-hcne"   # borough boundaries, water excluded -> land mask

FT = 0.3048
SQFT = FT * FT

# ---- Stage 1: lots ---------------------------------------------------------
# Tax-lot polygons flagged as these overlap a base lot and would double count.
DROP_LOT_FLAGS = ["air_lot_flag", "sub_lot_flag"]
# Use rule: residential if ResArea/BldgArea >= this; LandUse 01-03 fall back to res
# when BldgArea is missing. [TEAM DECISION: mixed use 04]
RES_SHARE_THRESHOLD = 0.5
BUILTFAR_OUTLIER = 15.0        # flag (not drop) lots above this outside Manhattan

# ---- Stage 2: buildings ----------------------------------------------------
# 2100 Building, 5110 Garage. [TEAM DECISION: garages 5110, under construction 5100]
KEEP_FEATURE_CODES = {2100, 5110}
MIN_FOOTPRINT_M2 = 10.0
MAX_HEIGHT_M = 450.0
# Floor height (m) used to turn roof height into floors. None = estimate it from
# lots that contain exactly one building (data-driven), falling back to 3.2.
FLOOR_HEIGHT_M = None
FLOOR_HEIGHT_FALLBACK = 3.2
FLOOR_DISAGREE = 2             # flag if |floors_from_height - PLUTO NumFloors| > this

# ---- Stage 3: blocks -------------------------------------------------------
MAX_BLOCK_HA = 20.0
MIN_BLOCK_HA = 0.1
MIN_COMPACTNESS = 0.15         # Polsby-Popper 4*pi*A/P^2
# A tax block can be cut by a street (e.g. Broadway in Midtown). Each connected
# piece of the lot union becomes its own block "<boro>-<block>-<a|b|..>".
# Lot-level PLUTO sanity: FAR cannot exceed floors (coverage <= 1).
PLUTO_FAR_PER_FLOOR_MAX = 1.2   # flag lot if builtfar > numfloors * this + 0.5
# Footprint map vs PLUTO disagree (stale footprints, new construction):
# flag if |far_fp - far_pluto| > MISMATCH_ABS and ratio > MISMATCH_RATIO.
# Training targets use far_fp (consistent with the map); PLUTO is only a check.
MISMATCH_ABS = 1.0
MISMATCH_RATIO = 1.5
EXCLUDE_MISMATCH = True        # [TEAM DECISION]
EXCLUDE_CONSTRUCTION = True    # blocks with a 5100 "under construction" footprint

# ---- Stage 4: raster -------------------------------------------------------
TILE_M = 512
CELL_M = 4
SUPERSAMPLE = 4                # rasterize at 1 m, then aggregate 4x4
BUILDING_COVER_MIN = 0.5       # cell is building if >= 50% covered
WATER_TILE_FLAG = 0.10         # flag tiles with > 10% water cells

HEIGHT_BINS = [(1, 2), (3, 5), (6, 12), (13, 10_000)]   # floors
CLASSES = {
    0: "street", 1: "open",
    2: "res_1-2", 3: "res_3-5", 4: "res_6-12", 5: "res_13+",
    6: "non_1-2", 7: "non_3-5", 8: "non_6-12", 9: "non_13+",
}


def height_class(floors):
    """floors -> 0..3"""
    import numpy as np
    f = np.asarray(floors, dtype=float)
    return np.select([f <= 2, f <= 5, f <= 12], [0, 1, 2], 3)
