"""Stage 4: rasterize pilot cores into 512 m tiles (128x128 @ 4 m, 10 classes).

A cell is a building if >= BUILDING_COVER_MIN of its 1 m sub-pixels are covered;
its class is the most common building class among those sub-pixels.
Non-building cells inside a block are `open`, the rest `street`. Water cells
(outside the land mask) are `street` in the class map and marked in `water`.

Out per pilot (data/processed/<pilot>/):
  tiles/<tile_id>.npz   cls uint8[128,128], block int32[128,128] (-1 = none), water bool, block_ids
  tiles.csv             tile bounds, water share, class counts
  raster_check.csv      per block fully inside a tile: vector vs raster coverage
  vanish.csv            per building: survived rasterization or not
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin

from common import bbox_m, pilot_dir
from config import (BUILDING_COVER_MIN, CELL_M, CLASSES, OUT, PILOTS, SUPERSAMPLE, TILE_M,
                    WATER_TILE_FLAG)

N = TILE_M // CELL_M          # 128
S = SUPERSAMPLE
FINE = N * S                  # 512 sub-pixels per side


def coarse(a):
    """[FINE,FINE] -> [N,N,S*S] view of each cell's sub-pixels."""
    return a.reshape(N, S, N, S).transpose(0, 2, 1, 3).reshape(N, N, S * S)


def mode_last(x, n_values):
    """Most common value along the last axis (values in 0..n_values-1)."""
    counts = np.stack([(x == v).sum(-1) for v in range(n_values)], -1)
    return counts.argmax(-1), counts


def main():
    for name in PILOTS:
        print(f"== {name}")
        d = pilot_dir(name)
        out = pilot_dir(name, OUT)
        (out / "tiles").mkdir(exist_ok=True)
        blk = gpd.read_parquet(d / "blocks.parquet").reset_index(drop=True)
        b = gpd.read_parquet(d / "buildings_final.parquet").reset_index(drop=True)
        land = gpd.read_parquet(d / "land.parquet").geometry.iloc[0]
        b["cls"] = np.where(b["is_res"].fillna(False).astype(bool), 2, 6) + b["hclass"].astype(int)

        x0, y0, x1, y1 = bbox_m(name, "core_bbox").bounds
        nx, ny = int((x1 - x0) // TILE_M), int((y1 - y0) // TILE_M)
        ox, oy = x0 + ((x1 - x0) - nx * TILE_M) / 2, y0 + ((y1 - y0) - ny * TILE_M) / 2
        tiles, checks, vanish = [], [], []
        for i in range(nx):
            for j in range(ny):
                tx, ty = ox + i * TILE_M, oy + j * TILE_M
                tid = f"{name}_{i}_{j}"
                tf = from_origin(tx, ty + TILE_M, 1.0 * CELL_M / S, 1.0 * CELL_M / S)
                kw = dict(out_shape=(FINE, FINE), transform=tf, fill=0)

                bi = b.cx[tx:tx + TILE_M, ty:ty + TILE_M]
                bl = blk.cx[tx:tx + TILE_M, ty:ty + TILE_M]
                f_cls = rasterize(zip(bi.geometry, bi["cls"]), dtype="uint8", **kw) if len(bi) else np.zeros((FINE, FINE), "uint8")
                f_bid = rasterize(zip(bi.geometry, bi.index + 1), dtype="int32", **kw) if len(bi) else np.zeros((FINE, FINE), "int32")
                f_blk = rasterize(zip(bl.geometry, bl.index + 1), dtype="int32", **kw) if len(bl) else np.zeros((FINE, FINE), "int32")
                f_land = rasterize([(land, 1)], dtype="uint8", **kw)

                c_cls = coarse(f_cls)
                is_b = (c_cls > 0).mean(-1) >= BUILDING_COVER_MIN
                # mode among building sub-pixels only
                counts = np.stack([(c_cls == v).sum(-1) for v in range(2, 10)], -1)
                bcls = counts.argmax(-1) + 2

                c_blk = coarse(f_blk)
                in_blk = (c_blk > 0).mean(-1) >= 0.5
                # block id per cell = most common nonzero id
                flat = c_blk.reshape(-1, S * S)
                blk_cell = np.array([np.bincount(r[r > 0]).argmax() if (r > 0).any() else 0 for r in flat]).reshape(N, N)
                water = coarse(f_land).mean(-1) < 0.5

                cls = np.where(is_b, bcls, np.where(in_blk, 1, 0)).astype("uint8")
                cls[water & ~is_b] = 0
                block = np.where(in_blk | is_b, blk_cell - 1, -1).astype("int32")
                ids = blk["block_id"].to_numpy()
                np.savez_compressed(out / "tiles" / f"{tid}.npz", cls=cls, block=block, water=water,
                                    block_ids=ids[np.unique(block[block >= 0])] if (block >= 0).any() else np.array([]),
                                    block_index=np.unique(block[block >= 0]), origin=np.array([tx, ty]))

                cc = np.bincount(cls.ravel(), minlength=10)
                tiles.append({"tile_id": tid, "x0": tx, "y0": ty, "water_share": water.mean(),
                              "flag_water": water.mean() > WATER_TILE_FLAG,
                              **{f"n_{CLASSES[k]}": int(cc[k]) for k in range(10)}})

                # vector vs raster coverage for valid blocks fully inside the tile
                from shapely.geometry import box
                tb = box(tx, ty, tx + TILE_M, ty + TILE_M)
                for k, r in bl[bl.within(tb)].iterrows():
                    m = block == k
                    if m.sum() == 0:
                        continue
                    checks.append({"tile_id": tid, "block_id": r["block_id"], "valid": r["valid"],
                                   "coverage_vec": r["coverage"], "coverage_ras": (cls[m] >= 2).mean(),
                                   "area_ha": r["area_ha"], "hclass_vec": r["hclass"],
                                   "cells": int(m.sum())})

                # which buildings survive (any of their sub-pixels lie in a building cell)
                up = np.repeat(np.repeat(is_b, S, 0), S, 1)
                alive = set(np.unique(f_bid[up & (f_bid > 0)]) - 1)
                inside = bi[bi.within(tb)]
                for k, r in inside.iterrows():
                    vanish.append({"tile_id": tid, "area_m2": r["area_m2"], "feature_code": r["feature_code"],
                                   "survived": k in alive})
                print(f"    {tid}: water {water.mean():.2f}, building cells {(cls >= 2).mean():.2f}")

        pd.DataFrame(tiles).to_csv(out / "tiles.csv", index=False)
        pd.DataFrame(checks).to_csv(out / "raster_check.csv", index=False)
        pd.DataFrame(vanish).to_csv(out / "vanish.csv", index=False)


if __name__ == "__main__":
    main()
