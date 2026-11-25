"""Per-building metrics. The stage-1 metrics work on 4 m cells, where touching houses merge, so they
cannot see whether a method separates buildings; these work on polygons.

block_stats(polys, heights, rec, cls4) -> values of one block (one sample)
summarise(real, pred)                  -> table of scores over blocks (lower is better, except *_iou)
"""
import numpy as np
import shapely

from . import config as C
from .geometry import obb, rasterize, to_cells, up

DIST = ("area", "width", "length", "height")     # per building
PAIR = ("dz",)                                   # per pair of touching buildings


def wd(a, b):
    """1-D Wasserstein distance between two samples (exact, any sizes)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if len(a) == 0 or len(b) == 0:
        return np.nan
    v = np.concatenate([a, b])
    v.sort()
    d = np.diff(v)
    fa = np.searchsorted(a, v[:-1], "right") / len(a)
    fb = np.searchsorted(b, v[:-1], "right") / len(b)
    return float((np.abs(fa - fb) * d).sum())


def touching(polys, gap=0.3, min_len=1.0):
    """Index pairs of buildings that share a wall (closer than `gap` along at least `min_len` m)."""
    if len(polys) < 2:
        return np.zeros((0, 2), int)
    g = np.array(polys, dtype=object)
    i, j = shapely.STRtree(g).query(g, predicate="dwithin", distance=gap)
    k = i < j
    i, j = i[k], j[k]
    if len(i) == 0:
        return np.zeros((0, 2), int)
    near = shapely.intersection(shapely.boundary(g[i]), shapely.buffer(g[j], gap))
    k = shapely.length(near) >= min_len
    return np.stack([i[k], j[k]], 1)


def block_stats(polys, heights, rec, cls4=None):
    """cls4: the input map the method saw (default: the real map). Values of one block."""
    cls4 = rec["cls4"] if cls4 is None else cls4
    inside = rec["inside"]
    h = np.asarray(heights, float)
    out = {"n": len(polys), "n_cells": int((inside & (cls4 >= 2)).sum())}
    if len(polys):
        g = np.array(polys, dtype=object)
        out["area"] = shapely.area(g)
        o = np.array([obb(p)[2:4] for p in polys])
        out["length"], out["width"] = o[:, 0], o[:, 1]
        out["height"] = h
        total, union = float(out["area"].sum()), float(shapely.area(shapely.union_all(g)))
        t = touching(polys)
        out["dz"] = np.abs(h[t[:, 0]] - h[t[:, 1]])
        out["touch"] = float(len(np.unique(t)) / len(polys))
    else:
        for k in DIST + PAIR:
            out[k] = np.zeros(0)
        total = union = out["touch"] = 0.0
    out["overlap"] = (total - union) / total if total else 0.0
    out["footprint"], out["volume"] = union, float((out["area"] * out["height"]).sum())
    # consistency with the input map, on 4 m cells inside the block
    H, W = inside.shape
    lab = rasterize(polys, (H * C.F, W * C.F))
    built, hc = to_cells(lab, np.digitize(h, HEIGHT_EDGES) if len(polys) else [])
    want = inside & (cls4 >= 2)
    got = built & inside
    u = (want | got).sum()
    out["map_iou"] = float((want & got).sum() / u) if u else 1.0
    both = want & got
    out["hclass_acc"] = float((hc[both] == C.hclass_of(cls4[both])).mean()) if both.any() else np.nan
    px = (lab > 0).sum()
    out["outside"] = float(((lab > 0) & ~up(inside)).sum() / px) if px else 0.0
    return out


# height (m) boundaries between the four height classes: output of fit_height_edges() on all training
# buildings. Classes come from floors, so even the real buildings agree only 87% of the time.
HEIGHT_EDGES = (9.1, 18.1, 40.5)


def fit_height_edges(recs):
    """Boundaries that misclassify the fewest training buildings."""
    h = np.concatenate([r["height"] for r in recs])
    k = C.hclass_of(np.concatenate([r["cls"] for r in recs]))
    edges = []
    for q in range(3):
        lo, hi = np.sort(h[k == q]), np.sort(h[k == q + 1])
        c = np.arange(3, 80, 0.1)
        bad = (1 - np.searchsorted(lo, c) / len(lo)) + np.searchsorted(hi, c) / len(hi)
        edges.append(float(round(c[bad.argmin()], 1)))
    return edges


def per_block_scores(real, pred):
    """Scores that exist for a single block."""
    n = max(real["n"], 1)
    f, v = max(real["footprint"], 1.0), max(real["volume"], 1.0)
    return {"count_err": abs(pred["n"] - real["n"]) / n, "coverage_err": abs(pred["footprint"] - real["footprint"]) / f,
            "volume_err": abs(pred["volume"] - real["volume"]) / v, "touch_err": abs(pred["touch"] - real["touch"]),
            "overlap": pred["overlap"], "outside": pred["outside"], "map_iou": pred["map_iou"],
            "hclass_acc": pred["hclass_acc"]}


LOWER = ("count_err", "wd_count", "wd_area", "wd_width", "wd_length", "wd_height", "wd_dz", "touch_err",
         "overlap", "outside", "coverage_err", "volume_err")
HIGHER = ("map_iou", "hclass_acc")
# mean of these picks checkpoints on val200 (all lower-is-better, each roughly 0..1)
SELECT = ("count_err", "wd_area", "wd_width", "wd_height", "wd_dz", "touch_err", "overlap", "1-map_iou")


def summarise(real, preds):
    """real: list of block_stats; preds: list (same order) of lists of block_stats (samples).
    Wasserstein distances pool all buildings of all blocks, on log values (so that a few towers and
    warehouses do not dominate), and are divided by the real std."""
    out = {}
    rows = [np.mean([list(per_block_scores(r, p).values()) for p in ps], 0) for r, ps in zip(real, preds)]
    keys = list(per_block_scores(real[0], preds[0][0]))
    tab = np.array(rows)
    for k, v in zip(keys, np.nanmean(tab, 0)):
        out[k] = float(v)
    for k in DIST + PAIR:
        f = np.log1p if k == "dz" else (lambda x: np.log(np.maximum(x, 0.1)))
        a = f(np.concatenate([r[k] for r in real]))
        b = f(np.concatenate([p[k] for ps in preds for p in ps]))
        out[f"wd_{k}"] = wd(a, b) / max(float(a.std()), 1e-6) if len(a) else np.nan
    a = np.array([r["n"] for r in real], float)
    out["wd_count"] = wd(a, [p["n"] for ps in preds for p in ps]) / max(float(a.std()), 1e-6)
    out["score"] = float(np.nanmean([1 - out["map_iou"] if k == "1-map_iou" else out[k] for k in SELECT]))
    if not np.isfinite([out[k] for k in ("wd_area", "wd_width", "wd_height")]).all():
        out["score"] = 9.0                                   # no buildings at all
    out["blocks"], out["buildings_real"] = len(real), int(a.sum())
    out["buildings_pred"] = float(np.mean([np.mean([p["n"] for p in ps]) for ps in preds]) * len(real))
    return out


def score_predictions(recs, predict, samples=1, maps=None):
    """predict(rec, cls4, sample) -> (polys, heights). Used for validation during training."""
    from .records import input_maps, polys as real_polys
    real, preds = [], []
    for r in recs:
        real.append(block_stats(real_polys(r), r["height"], r))
        ps = []
        for s, m in enumerate(input_maps(r, maps, samples)):
            g, h = predict(r, m, s)
            ps.append(block_stats(g, h, r, m))
        preds.append(ps)
    return summarise(real, preds)
