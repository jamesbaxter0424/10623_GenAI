"""Reading records, and writing / reading predictions (one format for every method)."""
import json
import os
import pickle
import zlib

import numpy as np
import shapely

from . import config as C
from .geometry import clean

_cache = {}


def _split(data, name):
    key = (data, name)
    if key not in _cache:
        with open(os.path.join(data, f"{name}.pkl"), "rb") as f:
            _cache[key] = {q["row"]: q for q in pickle.load(f)}
    return _cache[key]


def names(data=C.DATA):
    with open(os.path.join(data, "splits.json")) as f:
        return json.load(f)


def load(subset, data=C.DATA):
    """Records of a named list: overfit8, pilot2000, train, val200, test1000."""
    src = _split(data, "val" if subset.startswith("val") else "test" if subset.startswith("test") else "train")
    return [src[r] for r in names(data)[subset]]


def polys(rec):
    return [shapely.from_wkb(p) for p in rec["polys"]]


def labels(rec):
    h, w = rec["cls4"].shape
    return np.frombuffer(zlib.decompress(rec["lab"]), np.uint16).reshape(h * C.F, w * C.F).astype(np.int32)


class Writer:
    """Predictions as JSON lines: {"row", "sample", "buildings": [{"wkt", "h"}]}, local coordinates."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = open(path, "w")

    def add(self, row, sample, geoms, heights):
        b = [{"wkt": shapely.to_wkt(g, rounding_precision=3), "h": round(float(h), 2)}
             for g, h in zip(geoms, heights)]
        self.f.write(json.dumps({"row": int(row), "sample": int(sample), "buildings": b}) + "\n")

    def close(self):
        self.f.close()


def read_pred(path):
    """{row: {sample: (polygons, heights)}}"""
    out = {}
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            pairs = [(clean(shapely.from_wkt(b["wkt"])), b["h"]) for b in q["buildings"]]   # rounding can
            pairs = [(g, h) for g, h in pairs if g is not None]                             # break validity
            out.setdefault(q["row"], {})[q["sample"]] = ([g for g, _ in pairs], np.array([h for _, h in pairs], float))
    return out


def stage1_maps(path):
    """{row: uint8 [S, h, w]} written by stage1_input.py; None for the real map ("oracle")."""
    if path in (None, "", "oracle"):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def input_maps(rec, maps, samples):
    """cls4 for each of `samples` samples: the real map, or the stage-1 samples (repeated if fewer)."""
    if maps is None:
        return [rec["cls4"]] * samples
    m = maps[rec["row"]]
    return [m[s % len(m)] for s in range(samples)]
