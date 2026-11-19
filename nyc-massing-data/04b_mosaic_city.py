"""Stage 4 (city): one 4 m class map of all of NYC, on a grid aligned to 512 m.

Same cell rules as the pilot tiles (04_raster.py): a cell is a building if >= 50%
of its 1 m sub-pixels are covered, class = most common building class among them;
other cells inside a block are `open`, the rest `street`; water cells are `street`
and marked in `water`. Samples (stage 5) are 128x128 crop windows of this mosaic.

Out (data/processed/city/):
  blocks.parquet   all blocks of the five boroughs, row number = block index used in the maps
  cls.npy          uint8 [H, W]   class 0..9
  block.npy        int32 [H, W]   block index, -1 = street / water
  water.npy        bool  [H, W]
  grid.json        origin (x0, y_top) in EPSG:32618, cell size, H, W
  raster_check.csv per block: vector vs raster coverage
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin

from config import BUILDING_COVER_MIN, CELL_M, CITY, CRS_M, INTERIM, OUT, RAW, SUPERSAMPLE, TILE_M
from common import read_geojson

S = SUPERSAMPLE
CITY_OUT = OUT / "city"


def coarse(a, h, w):
    """[h*S, w*S] -> [h, w, S*S] sub-pixels of each cell."""
    return a.reshape(h, S, w, S).transpose(0, 2, 1, 3).reshape(h, w, S * S)


def raster_strip(args):
    """Rasterize one horizontal strip (TILE_M tall, full width)."""
    r0, h, W, x0, ytop, bgeom, bcls, bidx, kgeom, kidx, land = args
    fine = (h * S, W * S)
    tf = from_origin(x0, ytop, CELL_M / S, CELL_M / S)
    kw = dict(out_shape=fine, transform=tf, fill=0)
    f_cls = rasterize(zip(bgeom, bcls), dtype="uint8", **kw) if len(bgeom) else np.zeros(fine, "uint8")
    f_blk = rasterize(zip(kgeom, kidx + 1), dtype="int32", **kw) if len(kgeom) else np.zeros(fine, "int32")
    f_land = rasterize([(g, 1) for g in land], dtype="uint8", **kw) if len(land) else np.zeros(fine, "uint8")

    c = coarse(f_cls, h, W)
    is_b = (c > 0).mean(-1) >= BUILDING_COVER_MIN
    counts = np.stack([(c == v).sum(-1) for v in range(2, 10)], -1)
    bcls_cell = counts.argmax(-1) + 2

    k = coarse(f_blk, h, W)
    in_blk = (k > 0).mean(-1) >= 0.5
    blk_cell = k.max(-1)                                  # exact when a cell holds one block
    kmin = np.where(k > 0, k, np.iinfo(np.int32).max).min(-1)
    mixed = (blk_cell > 0) & (kmin != blk_cell)
    for i, j in zip(*np.nonzero(mixed)):                  # few cells on block-block seams
        v = k[i, j]
        blk_cell[i, j] = np.bincount(v[v > 0]).argmax()
    water = coarse(f_land, h, W).mean(-1) < 0.5

    cls = np.where(is_b, bcls_cell, np.where(in_blk, 1, 0)).astype("uint8")
    cls[water & ~is_b] = 0
    block = np.where(in_blk | is_b, blk_cell - 1, -1).astype("int32")
    return r0, cls, block, water


def main():
    CITY_OUT.mkdir(parents=True, exist_ok=True)
    blks, blds = [], []
    for boro in CITY:
        d = INTERIM / boro
        k = gpd.read_parquet(d / "blocks.parquet")
        k["boro"] = boro
        blks.append(k)
        b = gpd.read_parquet(d / "buildings_final.parquet")
        blds.append(b[["block_id", "is_res", "hclass", "geometry"]])
    blk = gpd.GeoDataFrame(pd.concat(blks, ignore_index=True), crs=CRS_M)
    bld = gpd.GeoDataFrame(pd.concat(blds, ignore_index=True), crs=CRS_M)
    assert blk["block_id"].is_unique
    bld["cls"] = np.where(bld["is_res"].fillna(False).astype(bool), 2, 6) + bld["hclass"].astype(int)
    blk.to_parquet(CITY_OUT / "blocks.parquet")
    print(f"blocks {len(blk)}, buildings {len(bld)}")

    land = read_geojson(RAW / "boroughs.geojson").geometry.make_valid()
    land = gpd.GeoSeries(land.explode(index_parts=False).to_numpy(), crs=CRS_M)

    minx, miny, maxx, maxy = blk.total_bounds
    x0 = np.floor(minx / TILE_M) * TILE_M - TILE_M        # one tile of margin on every side
    ytop = np.ceil(maxy / TILE_M) * TILE_M + TILE_M
    W = int((np.ceil(maxx / TILE_M) * TILE_M + TILE_M - x0) / CELL_M)
    H = int((ytop - (np.floor(miny / TILE_M) * TILE_M - TILE_M)) / CELL_M)
    print(f"grid {H} x {W} cells ({H * CELL_M / 1000:.1f} x {W * CELL_M / 1000:.1f} km)")
    grid = {"crs": CRS_M, "x0": float(x0), "y_top": float(ytop), "cell_m": CELL_M, "H": H, "W": W}
    (CITY_OUT / "grid.json").write_text(json.dumps(grid, indent=2))

    h = TILE_M // CELL_M
    bsi, ksi, lsi = bld.sindex, blk.sindex, land.sindex
    jobs = []
    for r0 in range(0, H, h):
        top = ytop - r0 * CELL_M
        from shapely.geometry import box
        strip = box(x0, top - h * CELL_M, x0 + W * CELL_M, top)
        bi = bld.iloc[bsi.query(strip, predicate="intersects")]
        ki = blk.iloc[ksi.query(strip, predicate="intersects")]
        li = land.iloc[lsi.query(strip, predicate="intersects")].intersection(strip)
        jobs.append((r0, h, W, x0, top, bi.geometry.to_numpy(), bi["cls"].to_numpy(), None,
                     ki.geometry.to_numpy(), ki.index.to_numpy(), li[~li.is_empty].to_numpy()))

    cls = np.zeros((H, W), "uint8")
    block = np.full((H, W), -1, "int32")
    water = np.ones((H, W), bool)
    t = time.time()
    with ProcessPoolExecutor(12) as ex:
        for n, (r0, c, b, w) in enumerate(ex.map(raster_strip, jobs)):
            cls[r0:r0 + h], block[r0:r0 + h], water[r0:r0 + h] = c, b, w
            if n % 10 == 0:
                print(f"  strip {n + 1}/{len(jobs)}  {time.time() - t:.0f}s")
    np.save(CITY_OUT / "cls.npy", cls)
    np.save(CITY_OUT / "block.npy", block)
    np.save(CITY_OUT / "water.npy", water)

    # vector vs raster coverage per block
    cells = np.bincount(block[block >= 0], minlength=len(blk))
    bcells = np.bincount(block[(block >= 0) & (cls >= 2)], minlength=len(blk))
    chk = pd.DataFrame({"block_id": blk["block_id"], "boro": blk["boro"], "valid": blk["valid"],
                        "cells": cells, "coverage_vec": blk["coverage"],
                        "coverage_ras": np.where(cells > 0, bcells / np.maximum(cells, 1), np.nan),
                        "area_ha": blk["area_ha"]})
    chk.to_csv(CITY_OUT / "raster_check.csv", index=False)
    v = chk[chk["valid"] & (chk["cells"] > 0)]
    print(f"valid blocks rasterized {len(v)} / {int(chk['valid'].sum())}; "
          f"corr cov vec/ras {v['coverage_vec'].corr(v['coverage_ras']):.4f}, "
          f"mean |diff| {(v['coverage_vec'] - v['coverage_ras']).abs().mean():.4f}")
    print("class shares on land:", np.round(np.bincount(cls[~water], minlength=10) / (~water).sum(), 4))


if __name__ == "__main__":
    main()
