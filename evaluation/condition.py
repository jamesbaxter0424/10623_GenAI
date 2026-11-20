"""Condition metrics: coverage, FAR, height class and residential share of one block.

All values are computed from the class map cells g of one target block, with f the floors
per class (config/data.fit_floors). Definitions follow the target table of the dataset:
  coverage   = building cells / block cells                 (Text-to-City density error, L1)
  FAR        = sum of floors over building cells / block cells
  height     = area-weighted median floors class of the building cells, as median_floors
               in nyc-massing-data/03_blocks.py (one cell = equal area)
  res share  = residential floor area / total floor area, as ResArea / BldgArea in PLUTO
"""
import numpy as np

import config as C


def block_values(g, f):
    """g: 1-D array of classes of the block's cells. Returns dict of generated values."""
    n = g.size
    bld = g >= 2
    nb = int(bld.sum())
    out = {"cov": nb / n, "far": float(f[g].sum() / n), "n_bld": nb}
    if nb:
        h = np.sort((g[bld] - 2) % 4)
        out["hcls"] = int(h[int(np.ceil(nb / 2)) - 1])          # lower weighted median, as the dataset
        w = f[g[bld]]
        out["res"] = float((w * (g[bld] <= 5)).sum() / w.sum())
    else:
        out["hcls"] = -1
        out["res"] = np.nan
    return out


def targets(city, b):
    r = city.blocks.iloc[b]
    return {"t_cov": float(r.coverage), "t_far": float(r.far_fp), "t_hcls": int(r.hclass),
            "t_res": float(r.res_share) if r.res_share == r.res_share else np.nan}


def hit(gen, tgt):
    tol = max(C.HIT_REL_TOL * abs(tgt), C.HIT_ABS_TOL)
    return abs(gen - tgt) <= tol
