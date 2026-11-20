"""CityDreamer's layout generator as a context-aware baseline (released weights, no training).

For each test window:
  1. our 4 m class map -> CityDreamer layout: height field (metres / 640) + one-hot of its six
     classes, at 9 px per cell (0.444 m/px; CityDreamer uses OSM zoom 18, ~0.45 m/px in NYC);
  2. the VQGAN encodes it into 16-px tokens (a 512-px window = 32 x 32 tokens);
  3. tokens touching a target block are masked; the absorbing-diffusion sampler fills them while
     the other tokens (the real surroundings) stay fixed - the same inpainting CityDreamer uses
     to extend a city;
  4. the result is decoded and converted back to our classes inside the target blocks only;
     everything outside stays the real map. Files match evaluation/README.md (<sample_id>.npz).

CityDreamer has no land use, so generated buildings are written as residential (classes 2-5);
report Res. err. as n/a and run c2st.py with --collapse_use for this model.

  python run_citydreamer.py --selftest                       # conversion check, no model, CPU
  python run_citydreamer.py --check_model --sids 43146       # load weights, encode/decode only
  python run_citydreamer.py --sids 43146 --viz 1             # one window, with a figure
  python run_citydreamer.py --n 1000                         # the test set (resumable)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "evaluation"))
import config as C                       # noqa: E402
from data import City, test_order        # noqa: E402

PX = 9                     # pixels per 4 m cell
FLOOR_M = 3.9              # metres per floor (dataset: 3.7-4.1 m fitted per borough)
MAX_H = 640.0              # CityDreamer height normalisation
FLOOR_BINS = (2.5, 5.5, 12.5)        # 1-2 / 3-5 / 6-12 / 13+ floors -> height class 0..3
# CityDreamer classes (scripts/dataset_generator.py); 0 = null, ignored in the one-hot
CD_NULL, CD_ROAD, CD_BLD, CD_GREEN, CD_CONSTR, CD_COAST, CD_OTHER = range(7)
H_ROAD, H_GREEN, H_COAST = 4.0, 16.0, 0.0     # heights CityDreamer gives non-building pixels
OPEN_AS = {"null": CD_NULL, "green": CD_GREEN, "other": CD_OTHER}


# ---------------------------------------------------------------------- conversion
def to_layout(cls, water, floors, open_as="null"):
    """Our class map [n,n] -> CityDreamer layout [7, n*PX, n*PX] (float32)."""
    seg = np.full(cls.shape, CD_ROAD, np.int64)
    hf = np.full(cls.shape, H_ROAD, np.float32)
    op = cls == C.OPEN
    seg[op] = OPEN_AS[open_as]
    if open_as == "green":
        hf[op] = H_GREEN
    b = cls >= 2
    seg[b] = CD_BLD
    hf[b] = floors[cls[b]] * FLOOR_M
    seg[water] = CD_COAST
    hf[water] = H_COAST
    seg = np.repeat(np.repeat(seg, PX, 0), PX, 1)
    hf = np.repeat(np.repeat(hf, PX, 0), PX, 1)
    onehot = np.stack([seg == k for k in range(1, 7)]).astype(np.float32)
    return np.concatenate([hf[None] / MAX_H, onehot], 0)


def to_classes(layout, cls, inside):
    """CityDreamer output [S,7,H,H] -> our class maps [S,n,n]; only `inside` cells change.
    A cell is a building if >= half of its pixels are; its height class comes from the median
    building height; every other cell inside a block is open ground. CityDreamer cannot tell
    open ground from road (its training treats untagged 'null' pixels as road), so non-building
    cells are not written as street."""
    S, n = layout.shape[0], cls.shape[0]
    hf = layout[:, 0] * MAX_H
    seg = layout[:, 1:].argmax(1) + 1
    rr, cc = np.nonzero(inside)

    def cells(a):       # [S,H,H] -> [S,K,PX*PX] for the inside cells
        a = a.reshape(S, n, PX, n, PX).transpose(0, 1, 3, 2, 4)
        return a[:, rr, cc].reshape(S, len(rr), PX * PX)

    bld, h = cells(seg == CD_BLD), cells(hf)
    frac_b = bld.mean(-1)
    hb = np.where(bld, h, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(hb, -1)
    hc = np.digitize(np.nan_to_num(med) / FLOOR_M, FLOOR_BINS)
    new = np.where(frac_b >= 0.5, 2 + hc, C.OPEN)
    g = np.repeat(cls[None], S, 0)
    g[:, rr, cc] = new
    return g


def collapse_use(c):
    c = c.copy()
    c[c >= 6] -= 4
    return c


def comparable(c):
    """What CityDreamer can express inside a block: building height class, or not built."""
    c = collapse_use(c)
    c[c < 2] = C.OPEN
    return c


# ---------------------------------------------------------------------- model
def load_models(cd_root, ckpt_path, device):
    import torch
    sys.path.insert(0, cd_root)
    import models.vqgan
    import models.sampler
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    vqae = models.vqgan.VQAutoEncoder(cfg)
    sampler = models.sampler.AbsorbingDiffusionSampler(cfg)
    for m, k in ((vqae, "vqae"), (sampler, "sampler")):
        sd = {(kk[7:] if kk.startswith("module.") else kk): v for kk, v in ck[k].items()}
        m.load_state_dict(sd, strict=True)          # fail loudly instead of skipping weights
    return vqae.to(device).eval(), sampler.to(device).eval(), cfg


def window_starts(lo, hi, F, G, stride):
    """Token-window starts (size F) covering [lo, hi) inside a grid of size G."""
    if hi - lo <= F:
        s = [(lo + hi) // 2 - F // 2]
    else:
        k = int(np.ceil((hi - lo - F) / stride)) + 1
        s = np.linspace(lo, hi - F, k).round().astype(int).tolist()
    return sorted({min(max(v, 0), G - F) for v in s})


def inpaint(vqae, sampler, cfg, layout, target_px, S, steps, temp, device):
    """Fill the tokens touching `target_px` with S samples. Returns [S,7,H,H] and #windows."""
    import torch
    F = cfg.NETWORK.VQGAN.ATTN_RESOLUTION
    SC = cfg.NETWORK.VQGAN.RESOLUTION // F
    MASK = cfg.NETWORK.VQGAN.N_EMBED
    H = layout.shape[1]
    G = H // SC
    tok = target_px.reshape(G, SC, G, SC).any((1, 3))
    r_, c_ = np.nonzero(tok)
    stride = int(F * 0.75)                                    # CityDreamer's own overlap
    wins = [(r, c) for r in window_starts(r_.min(), r_.max() + 1, F, G, stride)
            for c in window_starts(c_.min(), c_.max() + 1, F, G, stride) if tok[r:r + F, c:c + F].any()]
    x = torch.from_numpy(layout)[None].to(device)
    grid = torch.full((G, G), -1, dtype=torch.long)
    with torch.no_grad():
        for r, c in wins:                                     # context tokens, encoded per window
            _, _, info = vqae.encode(x[..., r * SC:(r + F) * SC, c * SC:(c + F) * SC])
            idx = info["min_encoding_indices"].view(F, F).cpu()
            sub = grid[r:r + F, c:c + F]
            sub[sub < 0] = idx[sub < 0]
        grid = grid.unsqueeze(0).repeat(S, 1, 1)
        grid[:, torch.from_numpy(tok)] = MASK
        for r, c in wins:                                     # fill masked tokens window by window
            xt = grid[:, r:r + F, c:c + F].reshape(S, F * F).to(device)
            if (xt == MASK).any():
                xt = sampler.sample(S, steps, x_t=xt.clone(), temperature=temp, device=device)
            grid[:, r:r + F, c:c + F] = xt.view(S, F, F).cpu()
        assert (grid != MASK).all() and (grid >= 0).all()
        out = np.zeros((S,) + layout.shape, np.float32)
        best = np.full((G, G), -1)
        ii, jj = np.mgrid[0:F, 0:F]
        cent = np.minimum(np.minimum(ii, F - 1 - ii), np.minimum(jj, F - 1 - jj))
        book = vqae.quantize.get_codebook()
        for r, c in wins:                                     # each masked token from its most central window
            quant = book[grid[:, r:r + F, c:c + F].to(device)].permute(0, 3, 1, 2).contiguous()
            dec = vqae.decode(quant).float().cpu().numpy()
            sel = tok[r:r + F, c:c + F] & (cent > best[r:r + F, c:c + F])
            best[r:r + F, c:c + F][sel] = cent[sel]
            pix = np.repeat(np.repeat(sel, SC, 0), SC, 1)
            out[:, :, r * SC:(r + F) * SC, c * SC:(c + F) * SC][:, :, pix] = dec[:, :, pix]
    return out, len(wins)


def reconstruct(vqae, cfg, layout, target_px, device):
    """Encode/decode without masking (checks weights and conversion, no sampling)."""
    import torch
    F = cfg.NETWORK.VQGAN.ATTN_RESOLUTION
    SC = cfg.NETWORK.VQGAN.RESOLUTION // F
    H = layout.shape[1]
    G = H // SC
    rr, cc = np.nonzero(target_px.reshape(G, SC, G, SC).any((1, 3)))
    r = min(max((rr.min() + rr.max()) // 2 - F // 2, 0), G - F)
    c = min(max((cc.min() + cc.max()) // 2 - F // 2, 0), G - F)
    out = np.zeros((1,) + layout.shape, np.float32)
    x = torch.from_numpy(layout)[None].to(device)[..., r * SC:(r + F) * SC, c * SC:(c + F) * SC]
    with torch.no_grad():
        dec, _ = vqae(x)
    out[0, :, r * SC:(r + F) * SC, c * SC:(c + F) * SC] = dec[0].float().cpu().numpy()
    return out


# ---------------------------------------------------------------------- figure
PALETTE = np.array([[200, 200, 200], [235, 225, 200], [255, 230, 150], [250, 180, 80], [225, 110, 50],
                    [160, 40, 30], [190, 220, 255], [120, 170, 240], [60, 110, 210], [20, 50, 140]], np.uint8)


def save_figure(path, sid, cls, gens, inside):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rr, cc = np.nonzero(inside)
    r0, r1 = max(rr.min() - 16, 0), min(rr.max() + 17, cls.shape[0])
    c0, c1 = max(cc.min() - 16, 0), min(cc.max() + 17, cls.shape[1])
    panels = [("real", cls)] + [(f"sample {i + 1}", g) for i, g in enumerate(gens)]
    fig, ax = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.2))
    edge = inside & ~np.pad(inside, 1)[2:, 1:-1] | inside & ~np.pad(inside, 1)[:-2, 1:-1] \
        | inside & ~np.pad(inside, 1)[1:-1, 2:] | inside & ~np.pad(inside, 1)[1:-1, :-2]
    for a, (t, m) in zip(ax, panels):
        img = PALETTE[m].copy()
        img[edge] = (255, 0, 255)
        a.imshow(img[r0:r1, c0:c1], interpolation="nearest")
        a.set_title(t, fontsize=9)
        a.axis("off")
    fig.suptitle(f"CityDreamer (zero-shot), window {sid}; magenta: target blocks", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def selftest(city, ids):
    """No model: our map -> layout -> our map must give back the real map inside the targets
    (up to what CityDreamer can express) and leave everything outside untouched."""
    for sid in ids:
        cls, blk, water, tgts = city.window(sid)
        inside = np.isin(blk, tgts)
        lay = to_layout(cls, water, city.floors)
        back = to_classes(lay[None], cls, inside)[0]
        ok = np.array_equal(comparable(back)[inside], comparable(cls)[inside]) \
            and np.array_equal(back[~inside], cls[~inside])
        print(f"window {sid}: layout {lay.shape}, round trip exact: {ok}")
        if not ok:
            d = (comparable(back) != comparable(cls)) & inside
            print("  differing cells:", int(d.sum()), "real", np.unique(cls[d]), "back", np.unique(back[d]))
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cd_root", default=os.path.expanduser("~/genai-final/third_party/CityDreamer"))
    ap.add_argument("--ckpt", default=os.path.expanduser("~/genai-final/checkpoints/citydreamer/LayoutGen.pth"))
    ap.add_argument("--out", default=os.path.expanduser("~/genai-final/generated/citydreamer_test1000"))
    ap.add_argument("--n", type=int, default=C.N_TEST)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--sids", default="", help="comma-separated window ids instead of the test ordering")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--steps", type=int, default=32, help="sampler steps (CityDreamer's inference uses 32)")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--open_as", default="null", choices=sorted(OPEN_AS))
    ap.add_argument("--viz", type=int, default=0, help="save a figure for the first VIZ windows")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check_model", action="store_true", help="encode/decode only, no sampling")
    a = ap.parse_args()

    city = City()
    ids = [int(s) for s in a.sids.split(",") if s] or test_order(city)[a.start:a.start + a.n]
    if a.selftest:
        sys.exit(0 if selftest(city, ids[:20]) else 1)

    import torch
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available; pass --device cpu for a (slow) CPU run")
    vqae, sampler, cfg = load_models(a.cd_root, a.ckpt, a.device)
    print("weights loaded (strict); VQGAN", cfg.NETWORK.VQGAN.RESOLUTION, "px,",
          cfg.NETWORK.VQGAN.ATTN_RESOLUTION, "tokens; device", a.device, flush=True)

    if a.check_model:
        for sid in ids:
            cls, blk, water, tgts = city.window(sid)
            inside = np.isin(blk, tgts)
            lay = to_layout(cls, water, city.floors, a.open_as)
            inside_px = np.repeat(np.repeat(inside, PX, 0), PX, 1)
            rec = reconstruct(vqae, cfg, lay, inside_px, a.device)
            back = to_classes(rec, cls, inside)[0]
            agree = (comparable(back) == comparable(cls))[inside].mean()
            b_real = cls[inside] >= 2
            b_rec = back[inside] >= 2
            print(f"window {sid}: encode/decode agreement inside targets {agree:.3f}; "
                  f"building share real {b_real.mean():.3f} rec {b_rec.mean():.3f}")
            os.makedirs(os.path.join(a.out, "viz"), exist_ok=True)
            save_figure(os.path.join(a.out, "viz", f"{sid}_reconstruct.png"), sid, cls, [back], inside)
        return

    os.makedirs(a.out, exist_ok=True)
    if a.viz:
        os.makedirs(os.path.join(a.out, "viz"), exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    json.dump({"args": vars(a), "floors": city.floors.tolist(), "px": PX, "floor_m": FLOOR_M}, log)
    log.write("\n")
    t_all = time.time()
    for k, sid in enumerate(ids):
        f = os.path.join(a.out, f"{sid}.npz")
        if os.path.exists(f):
            continue
        t0 = time.time()
        cls, blk, water, tgts = city.window(sid)
        inside = np.isin(blk, tgts)
        lay = to_layout(cls, water, city.floors, a.open_as)
        inside_px = np.repeat(np.repeat(inside, PX, 0), PX, 1)
        out, nw = inpaint(vqae, sampler, cfg, lay, inside_px, a.samples, a.steps, a.temp, a.device)
        gens = to_classes(out, cls, inside)
        np.savez_compressed(f, gen=gens.astype(np.uint8), gt=cls.astype(np.uint8), mask=inside)
        dt = time.time() - t0
        json.dump({"sid": sid, "windows": nw, "sec": round(dt, 2)}, log)
        log.write("\n")
        log.flush()
        if k < a.viz:
            save_figure(os.path.join(a.out, "viz", f"{sid}.png"), sid, cls, gens, inside)
        if k < 5 or k % 50 == 0:
            el = time.time() - t_all
            print(f"[{k + 1}/{len(ids)}] window {sid}: {nw} token windows, {dt:.1f}s "
                  f"(elapsed {el / 60:.1f} min)", flush=True)
    print("CITYDREAMER_DONE", flush=True)


if __name__ == "__main__":
    main()
