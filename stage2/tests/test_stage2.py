"""Quick checks on synthetic blocks (CPU, under a minute, no data needed):  cd stage2 && python -m pytest -q"""
import os
import sys
import zlib

import numpy as np
import pytest
import shapely
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common import metrics as M                                    # noqa: E402
from common import records as R                                    # noqa: E402
from common.dataset import Frame, RasterFrame, canonical           # noqa: E402
from common.diffusion import Diffusion                             # noqa: E402
from common.geometry import own_distance, polygonize, rasterize, rect, to_cells, trim_overlaps  # noqa: E402
from common.noise import corrupt                                   # noqa: E402
from common.rule_split import fit, load_table, predict             # noqa: E402


def make(angle=0.0, n=8, w=6.0, d=15.0, seed=0):
    """A block with one row of n attached houses, rotated by `angle` radians."""
    S = 40
    c = S * 4 / 2
    houses = [rect((i - (n - 1) / 2) * w, 0, w, d, 0) for i in range(n)]
    move = lambda g: shapely.affinity.translate(shapely.affinity.rotate(g, angle, origin=(0, 0), use_radians=True), c, c)
    houses = [move(g) for g in houses]
    block = move(rect(0, 0, n * w + 20, 50, 0))
    lab = rasterize(houses, (S * 4, S * 4))
    cls = np.full(n, 3, np.uint8)
    built, k = to_cells(lab, cls)
    ins, _ = to_cells(rasterize([block], (S * 4, S * 4)), [1])
    cls4 = np.where(built, k, np.where(ins, 1, 0)).astype(np.uint8)
    return {"row": seed, "block_id": "t", "boro": "BK", "r0": 0, "c0": 0, "cls4": cls4, "inside": ins | built,
            "polys": [shapely.to_wkb(g) for g in houses], "height": np.full(n, 11.0, np.float32), "cls": cls,
            "lab": zlib.compress(lab.astype(np.uint16).tobytes())}


def cfg(name):
    with open(os.path.join(ROOT, name, "config.yaml")) as f:
        return yaml.safe_load(f)


def test_real_buildings_score_zero():
    r = make(0.3)
    s = M.summarise([M.block_stats(R.polys(r), r["height"], r)], [[M.block_stats(R.polys(r), r["height"], r)]])
    for k in ("count_err", "wd_area", "wd_width", "wd_height", "overlap", "coverage_err"):
        assert abs(s[k]) < 1e-6
    assert s["map_iou"] > 0.99


def test_merged_row_is_penalised():
    r = make()
    u = shapely.union_all(np.array(R.polys(r), dtype=object).tolist())
    s = M.summarise([M.block_stats(R.polys(r), r["height"], r)], [[M.block_stats([u.envelope], [11.0], r)]])
    assert s["count_err"] > 0.8 and s["map_iou"] > 0.9


def test_touching_and_overlap():
    a, b, c = rect(0, 0, 6, 10, 0), rect(6, 0, 6, 10, 0), rect(30, 0, 6, 10, 0)
    assert M.touching([a, b, c]).tolist() == [[0, 1]]
    t = [g for g in trim_overlaps([a, rect(4, 0, 6, 10, 0)]) if g is not None]
    assert shapely.area(shapely.intersection(t[0], t[1])) < 1e-6


@pytest.mark.parametrize("angle", [0.0, 0.5, 1.2])
def test_frames_round_trip(angle):
    r = make(angle)
    f = Frame(r["inside"])
    u, v = f.to_uv(50.0, 70.0)
    assert np.allclose(f.to_xy(u, v), (50.0, 70.0))
    g = RasterFrame(r["inside"])
    p = R.polys(r)[0]
    assert shapely.hausdorff_distance(g.poly_back(g.poly_to(p)), p) < 1e-6
    c, _ = canonical(r["cls4"], r["inside"])
    assert c.shape == (12, 64, 64) and np.allclose(c[:10].sum(0), 1)


@pytest.mark.parametrize("angle", [0.0, 0.5])
def test_raster_targets_give_back_the_houses(angle):
    from raster_diffusion.sampling import to_buildings
    from raster_diffusion.targets import encode
    c, r = cfg("raster_diffusion"), make(angle)
    f = RasterFrame(r["inside"])
    x = encode(f.labels(R.polys(r)), r["height"], c)
    g, h = to_buildings(x, f.cond(r["cls4"], r["inside"]), f, c)
    assert len(g) == 8 and np.allclose(h, 11.0, atol=0.2)
    inter = shapely.area(shapely.union_all(g))
    assert abs(sum(p.area for p in g) - inter) < 1e-3                 # no overlaps


def test_raster_training_item_and_model():
    from raster_diffusion.model import build
    from raster_diffusion.targets import Blocks
    from raster_diffusion.train import make_loss
    c = {**cfg("raster_diffusion"), "base": 32, "mults": [1, 2], "blocks": 1, "crop": [64, 96]}
    b = Blocks([make(0.4)], c)[0]
    b = {k: v[None] for k, v in b.items()}
    for mode in ("diffusion", "regression"):
        loss = make_loss({**c, "mode": mode})(build(c), b)["total"]
        loss.backward()
        assert torch.isfinite(loss)


@pytest.mark.parametrize("angle", [0.0, 0.5, 1.4])
def test_set_tokens_round_trip(angle):
    from set_diffusion.tokens import boxes, decode, encode, to_polygons
    r = make(angle)
    f = Frame(r["inside"])
    g, h = to_polygons(decode(encode(boxes(r, f), 16)), f)
    s = M.block_stats(g, h, r)
    assert len(g) == 8 and s["overlap"] < 1e-3 and np.allclose(np.sort(s["width"]), 6.0, atol=0.2)
    real = shapely.union_all(R.polys(r))
    assert shapely.area(shapely.intersection(real, shapely.union_all(g))) / real.area > 0.97


def test_set_model_losses_and_guidance():
    from set_diffusion.losses import make_loss, overlap
    from set_diffusion.model import build
    from set_diffusion.sampling import guided
    from set_diffusion.tokens import Blocks
    c = {**cfg("set_diffusion"), "dim": 64, "layers": 1, "heads": 2, "tokens": 16}
    b = {k: v[None] for k, v in Blocks([make(0.4)], c)[0].items()}
    loss = make_loss(c)(build(c), b)["total"]
    loss.backward()
    assert torch.isfinite(loss)
    real = (b["x0"][..., 0] > 0).float()
    assert overlap(b["x0"], real).item() < 0.05
    z = b["x0"].clone()
    z[..., 1:3] = 0.0                                              # all houses in the same place
    assert overlap(z, real).item() > 0.9
    z[..., 1] += torch.linspace(-0.05, 0.05, 16)
    assert overlap(guided(z, 0.2), real).item() <= overlap(z, real).item()


def test_codes_model_fills_every_token():
    from globalmapper_map.model import build
    c = {**cfg("globalmapper_map"), "dim": 32, "layers": 1, "heads": 2, "levels": 5}
    m = build(c)
    b = {"cond": torch.zeros(2, 12, 32, 128), "scal": torch.zeros(2, 3), "code": torch.randint(0, 5, (2, 512))}
    out = m.loss(b)
    out["total"].backward()
    code = m.eval().sample(b, rounds=4)
    assert code.shape == (2, 512) and code.max() < 5 and torch.isfinite(out["total"])


def test_quantisation_is_close_with_many_levels():
    from globalmapper_map.quantize import decode, encode, fit_bins
    mu = np.random.default_rng(0).normal(size=(2000, 4)).astype(np.float32)
    err = [np.abs(decode(encode(mu, e), c) - mu).mean() for e, c in (fit_bins(mu, L) for L in (5, 20, 50))]
    assert err[0] > err[1] > err[2] and err[1] < 0.1


def test_diffusion_recovers_a_known_point():
    d = Diffusion()
    target = torch.tensor([[0.5, -0.25]])

    def model(x, t):                                               # the exact v for a one-point data set
        a, s = (d._ex(q, x) for q in d.alpha_sigma(t))
        return a * (x - a * target) / s.clamp(min=1e-4) - s * target
    assert torch.allclose(d.sample(model, (1, 2), "cpu", 20), target, atol=1e-3)


def test_noise_keeps_the_surroundings():
    r, rng = make(0.2), np.random.default_rng(0)
    for _ in range(50):
        c = corrupt(r["cls4"], r["inside"], rng)
        assert (c[~r["inside"]] == r["cls4"][~r["inside"]]).all() and c.max() <= 9


def test_rule_baseline(tmp_path):
    recs = [make(0.3, 8, seed=i) for i in range(6)]                # at least 5 regions per size bin
    fit(recs, str(tmp_path / "t.pkl"))
    g, h = predict(load_table(str(tmp_path / "t.pkl")), recs[1], recs[1]["cls4"], 0)
    assert 4 <= len(g) <= 12 and len(g) == len(h)


def test_predictions_round_trip(tmp_path):
    r, p = make(0.3), str(tmp_path / "p.jsonl")
    w = R.Writer(p)
    w.add(r["row"], 0, R.polys(r), r["height"])
    w.close()
    g, h = R.read_pred(p)[r["row"]][0]
    assert len(g) == 8 and np.allclose(h, 11.0)


def test_polygonize_has_no_gaps():
    lab = np.zeros((20, 30), int)
    lab[3:15, 2:8], lab[3:15, 8:15], lab[3:12, 15:21] = 1, 2, 3
    q = polygonize(lab, 1.0, 1.0)
    assert abs(sum(g.area for g in q.values()) - shapely.area(shapely.union_all(list(q.values())))) < 1e-6
    assert own_distance(lab).max() > 2


def test_set_model_with_and_without_qk_norm():
    from set_diffusion.model import build
    c = {**cfg("set_diffusion"), "dim": 64, "layers": 2, "heads": 2, "tokens": 16}
    assert c["qk_norm"] is True
    b = {"cond": torch.randn(2, 12, 64, 64), "scal": torch.randn(2, 3), "half": torch.full((2,), 60.0)}
    old = {k: v for k, v in c.items() if k != "qk_norm"}           # checkpoints from before Nov 25
    for q in (c, old):
        m = build(q)
        torch.nn.init.normal_(m.out.weight)
        out = m(torch.randn(2, 16, 7), torch.rand(2), m.encode_map(b))
        out.sum().backward()
        assert out.shape == (2, 16, 7) and torch.isfinite(out).all()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert "blocks.0.sa.in_proj_weight" in build(old).state_dict()


def _fit(tmp_path, name, spike, **extra):
    """Train a linear layer for 1,200 steps; the data of step `spike` are 1e4 times too large."""
    from common import train_utils as T
    d = tmp_path / name
    d.mkdir()
    c = {"dir": str(d), "name": name, "subset": "t", "lr": 1e-3, "weight_decay": 0.0, "warmup": 1, "clip": 1.0, "ema": 0.9,
         "amp": False, "max_steps": 1200, "max_hours": 1.0, "log_every": 100, "val_every": 1200, "patience": 9,
         "min_delta": 0.0, **extra}
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 1)
    g = torch.Generator().manual_seed(1)
    data = [{"x": torch.randn(8, 4, generator=g) * (1e4 if i == spike else 1.0)} for i in range(1200)]
    T.fit(c, m, data, lambda q, b: {"total": ((q(b["x"]) - b["x"].sum(1, keepdim=True)) ** 2).mean()},
          lambda q: {"score": 1.0})
    log = __import__("pandas").read_csv(d / "log.csv")
    return m.cpu(), log


def test_gradient_guard_is_off_by_default_and_skips_spikes(tmp_path):
    a, la = _fit(tmp_path, "off", -1)
    b, lb = _fit(tmp_path, "zero", -1, skip_grad=0)
    assert torch.equal(a.weight, b.weight) and "skipped" not in la and "skipped" not in lb
    _, lc = _fit(tmp_path, "spike", 1100, skip_grad=5.0)
    assert round(lc["skipped"].iloc[11] * 100) == 1                # only the spike, in steps 1101-1200
    assert lc["skipped"].iloc[:11].sum() == 0
