"""Predict latents from maps and decode them with GlobalMapper."""
import numpy as np
import torch

from common.dataset import cond_cells

from .gm import axis_map, axis_points
from .quantize import decode


def map_inputs(rec, cls4, geom, pts=None):
    x, y, scal = pts if pts is not None else axis_points(rec, geom)
    return axis_map(cond_cells(cls4, rec["inside"]), (x, y)), scal


@torch.no_grad()
def predict_batch(model, gm, hts, centre, split, recs, maps, cfg, seed, dev):
    """[(polygons, heights)] for equally long lists of records and maps. Blocks without a GlobalMapper
    graph (more than 120 buildings, or a failed conversion) come back empty."""
    geo = gm.geometry(split)
    ok = [i for i, r in enumerate(recs) if r["row"] in geo]
    out = [([], np.zeros(0)) for _ in recs]
    if not ok:
        return out
    inp = [map_inputs(recs[i], maps[i], geo.geom[recs[i]["row"]]) for i in ok]
    b = {"cond": torch.from_numpy(np.stack([c for c, _ in inp])).to(dev),
         "scal": torch.from_numpy(np.stack([s for _, s in inp])).to(dev)}
    g = torch.Generator(dev).manual_seed(seed)
    z = model.sample(b, rounds=cfg["rounds"], temperature=cfg["temperature"], generator=g).cpu().numpy()
    if cfg["head"] == "codes":
        z = decode(z, centre)
    slots = gm.decode(split, [recs[i]["row"] for i in ok], z)
    for i, s in zip(ok, slots):
        p = gm.polygons(split, recs[i], s, cfg["trim"])
        out[i] = (p, hts.sample(p, maps[i], recs[i]["inside"], np.random.default_rng([seed, recs[i]["row"]])))
    return out
