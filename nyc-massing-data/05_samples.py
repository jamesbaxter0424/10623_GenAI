"""Stage 5 (city): split, block graph, target groups and sample windows.

1. Community district per block (mode of PLUTO `cd` over the tax block) and a
   train / val / test split by community district, stratified by borough.
2. Block graph. Every street cell of the mosaic is given to its nearest block
   (a street-space Voronoi). Two blocks are connected when their street regions
   share >= MIN_FACING_M of boundary, i.e. they face each other across a street.
   Diagonal blocks at an intersection share only a point and are not connected.
3. Target groups: connected sets of 1-4 valid blocks of one split, grown at
   random from every valid block (one group per size per seed, deduplicated).
4. One 512 m window per group, placed at random so that the group lies inside
   with >= MARGIN_M of context on every side.

Out (data/processed/city/):
  blocks.parquet   + cd, split, node attributes (overwrites stage-4 file, same rows)
  edges.parquet    i < j, shared_m (facing length), street_m (estimated street width), touching
  samples.parquet  sample_id, split, k, targets (block indices), row0, col0, ctx_other_split
  splits.json      community districts per split
  quantiles.json   NYC quantiles of the block targets (for words like "dense")
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage

from config import CELL_M, CITY, INTERIM, OUT, TILE_M

CITY_OUT = OUT / "city"
SEED = 0
TEST_FRAC, VAL_FRAC = 0.17, 0.10     # of community districts per borough  [TEAM DECISION]
MIN_FACING_M = 16.0                  # shared street-region boundary for an edge
MAX_HALF_STREET_M = 40.0             # street cells further than this from any block are not assigned
MARGIN_M = 48.0                      # min context around a target group inside the window
K_MAX = 4
N = TILE_M // CELL_M


# ---------------------------------------------------------------- 1. split
def community_districts(blk):
    cds = []
    for boro in CITY:
        p = pd.read_parquet(INTERIM / boro / "pluto.parquet", columns=["block_id", "cd"])
        cds.append(p.dropna())
    p = pd.concat(cds)
    mode = p.groupby("block_id")["cd"].agg(lambda s: int(s.value_counts().index[0]))
    return blk["tax_block"].map(mode)


def make_split(blk):
    """Regular CDs are xNN with NN <= 18; 1xx..5xx with NN >= 26 are joint interest
    areas (parks, airports) -> always train."""
    rng = np.random.default_rng(SEED)
    split = {}
    for boro, digit in zip(["MN", "BX", "BK", "QN", "SI"], range(1, 6)):
        cds = sorted({int(c) for c in blk["cd"].dropna() if int(c) // 100 == digit})
        regular = [c for c in cds if c % 100 <= 18]
        order = list(rng.permutation(regular))
        n_test = max(1, round(TEST_FRAC * len(regular)))
        n_val = round(VAL_FRAC * len(regular))
        for i, c in enumerate(order):
            split[int(c)] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        for c in set(cds) - set(regular):
            split[int(c)] = "train"
    return split


# ---------------------------------------------------------------- 2. graph
def block_graph(block):
    street = block < 0
    dist, (ri, ci) = ndimage.distance_transform_edt(street, return_indices=True)
    lab = block[ri, ci]
    lab[street & (dist * CELL_M > MAX_HALF_STREET_M)] = -1
    d_m = (dist * CELL_M).astype("float32")
    del ri, ci, dist

    rows = []
    for a, b, da, db in [(lab[:, :-1], lab[:, 1:], d_m[:, :-1], d_m[:, 1:]),
                         (lab[:-1, :], lab[1:, :], d_m[:-1, :], d_m[1:, :])]:
        m = (a != b) & (a >= 0) & (b >= 0)
        i, j = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
        rows.append(pd.DataFrame({"i": i, "j": j, "w": da[m] + db[m],
                                  "touch": (da[m] == 0) & (db[m] == 0)}))
    e = pd.concat(rows)
    g = e.groupby(["i", "j"]).agg(n=("w", "size"), w=("w", "median"), touch=("touch", "sum")).reset_index()
    g["shared_m"] = g["n"] * CELL_M
    g["street_m"] = g["w"] + CELL_M            # both half widths + the seam cell
    g["touching"] = g["touch"] * CELL_M >= MIN_FACING_M
    g = g[g["shared_m"] >= MIN_FACING_M]
    return g[["i", "j", "shared_m", "street_m", "touching"]].reset_index(drop=True)


# ---------------------------------------------------------------- 3-4. samples
def cell_bounds(block, n):
    """Row/col bounding box of every block index in the mosaic."""
    r, c = np.nonzero(block >= 0)
    v = block[r, c]
    out = np.full((n, 4), -1, int)
    df = pd.DataFrame({"v": v, "r": r, "c": c}).groupby("v").agg(r0=("r", "min"), r1=("r", "max"),
                                                                 c0=("c", "min"), c1=("c", "max"))
    out[df.index] = df.to_numpy()
    return out


def grow_groups(blk, adj, rng):
    ok = blk["valid"].to_numpy()
    split = blk["split"].to_numpy()
    groups = set()
    for s in np.nonzero(ok)[0]:
        for k in range(1, K_MAX + 1):
            g = [s]
            for _ in range(20 * k):
                if len(g) == k:
                    break
                frontier = sorted({n for m in g for n in adj.get(m, ()) if ok[n] and split[n] == split[s]} - set(g))
                if not frontier:
                    break
                g.append(int(rng.choice(frontier)))
            if len(g) == k:
                groups.add(tuple(sorted(int(x) for x in g)))
    return sorted(groups, key=lambda t: (len(t), t))


def place(groups, bounds, H, W, rng):
    m = int(MARGIN_M // CELL_M)
    out = []
    for g in groups:
        b = bounds[list(g)]
        r0, r1, c0, c1 = b[:, 0].min(), b[:, 1].max(), b[:, 2].min(), b[:, 3].max()
        # window rows [row0, row0+N) must contain [r0-m, r1+m]
        lo_r, hi_r = r1 + m - N + 1, r0 - m
        lo_c, hi_c = c1 + m - N + 1, c0 - m
        lo_r, lo_c = max(lo_r, 0), max(lo_c, 0)
        hi_r, hi_c = min(hi_r, H - N), min(hi_c, W - N)
        if lo_r > hi_r or lo_c > hi_c:
            out.append((-1, -1))          # group too large for one window
            continue
        out.append((int(rng.integers(lo_r, hi_r + 1)), int(rng.integers(lo_c, hi_c + 1))))
    return out


def main():
    blk = gpd.read_parquet(CITY_OUT / "blocks.parquet")
    block = np.load(CITY_OUT / "block.npy")
    H, W = block.shape
    rng = np.random.default_rng(SEED)

    blk["cd"] = community_districts(blk).astype("Int64")
    splits = make_split(blk)
    blk["split"] = blk["cd"].map(lambda c: splits.get(int(c), "train") if pd.notna(c) else "train")
    print("blocks without cd:", int(blk["cd"].isna().sum()))

    # node attributes (all blocks, also context-only ones)
    blk["area_m2"] = blk.area
    blk["res_share"] = blk["res_share"].fillna(blk["res_share_fp"]).fillna(0.0)
    blk["median_floors"] = blk["median_floors"].fillna(0)
    blk["hclass"] = blk["hclass"].astype(int)          # -1 = no buildings
    cells = np.bincount(block[block >= 0], minlength=len(blk))
    blk["cells"] = cells
    blk.loc[blk["cells"] < 4, "valid"] = False          # too small to show at 4 m

    edges = block_graph(block)
    print(f"edges {len(edges)}; mean degree {2 * len(edges) / len(blk):.2f}; "
          f"touching {int(edges['touching'].sum())}")
    adj = {}
    for i, j in zip(edges["i"], edges["j"]):
        adj.setdefault(int(i), []).append(int(j))
        adj.setdefault(int(j), []).append(int(i))
    blk["degree"] = [len(adj.get(i, [])) for i in range(len(blk))]

    groups = grow_groups(blk, adj, rng)
    pos = place(groups, cell_bounds(block, len(blk)), H, W, rng)
    s = pd.DataFrame({"targets": [list(g) for g in groups], "k": [len(g) for g in groups],
                      "row0": [p[0] for p in pos], "col0": [p[1] for p in pos]})
    n0 = len(s)
    s = s[s["row0"] >= 0].reset_index(drop=True)
    print(f"groups {n0}, too large for a window {n0 - len(s)}")
    split_arr = blk["split"].to_numpy()
    s["split"] = [split_arr[t[0]] for t in s["targets"]]

    # context of a training window that shows val/test blocks -> flag (default: excluded)
    other = []
    for r0, c0, sp in zip(s["row0"], s["col0"], s["split"]):
        ids = np.unique(block[r0:r0 + N, c0:c0 + N])
        ids = ids[ids >= 0]
        other.append(bool((split_arr[ids] != sp).any()) if sp == "train" else
                     bool((split_arr[ids] == "test").any()) if sp == "val" else False)
    s["ctx_other_split"] = other
    s.insert(0, "sample_id", np.arange(len(s)))

    q = {}
    v = blk[blk["valid"]]
    for c in ["coverage", "far_fp", "median_floors", "res_share", "area_m2"]:
        q[c] = {"city": {f"q{int(p * 100)}": float(v[c].quantile(p)) for p in
                         [0.1, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 0.9]}}
        for b in CITY:
            vb = v[v["boro"] == b][c]
            q[c][b] = {f"q{int(p * 100)}": float(vb.quantile(p)) for p in [1 / 3, 0.5, 2 / 3]}
    (CITY_OUT / "quantiles.json").write_text(json.dumps(q, indent=2))
    inv = {}
    for cd, sp in sorted(splits.items()):
        inv.setdefault(sp, []).append(cd)
    (CITY_OUT / "splits.json").write_text(json.dumps({"seed": SEED, **inv}, indent=2))

    blk.to_parquet(CITY_OUT / "blocks.parquet")
    edges.to_parquet(CITY_OUT / "edges.parquet")
    s.to_parquet(CITY_OUT / "samples.parquet")

    print("\nvalid target blocks per split:")
    print(blk[blk["valid"]].groupby("split").size().to_string())
    print("\nsamples (k x split):")
    print(pd.crosstab(s["k"], s["split"], margins=True).to_string())
    print("\ntrain samples with val/test blocks in context:",
          int(s[(s["split"] == "train")]["ctx_other_split"].sum()))
    print("val samples with test blocks in context:", int(s[(s["split"] == "val")]["ctx_other_split"].sum()))


if __name__ == "__main__":
    main()
