"""Stage 3: fill the target blocks of test windows with the NYC-trained CityDreamer.

For each window: encode the real map into 32 x 32 tokens (one token = 4 x 4 cells), mask the tokens
that touch a target block, let the sampler fill them while all other tokens stay fixed (CityDreamer's
own inpainting), decode, and keep the generated classes inside the target blocks only. Output files
follow evaluation/README.md (<sample_id>.npz with gen [S,128,128], gt, mask).

  python generate.py --ckpt SAMPLER/latest.pth --split val --n 2 --viz 2 --out DIR    # look first
  python generate.py --ckpt SAMPLER/latest.pth --uncond 8 --out DIR                    # unconditional
  python generate.py --ckpt SAMPLER/latest.pth --n 1000 --viz 10 --out DIR             # test set
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import common as K

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation"))
from data import City, test_order   # noqa: E402  (evaluation/data.py: same windows and ordering as Liu)


def target_tokens(inside, F):
    s = K.N // F
    return inside.reshape(F, s, F, s).any((1, 3)).reshape(-1)


@torch.no_grad()
def inpaint(cfg, vqae, sampler, cls, inside, S, steps, temp, device, recon=False):
    F = cfg.NETWORK.VQGAN.ATTN_RESOLUTION
    MASK = cfg.NETWORK.VQGAN.N_EMBED
    x = torch.from_numpy(cls).to(device).long()[None]
    oh, _ = K.to_input(x)
    idx = K.encode_idx(vqae, oh)                                   # [1, 1024]
    if recon:   # reference row: real tokens decoded back, no sampling (CityDreamer's ceiling)
        out = idx.repeat(S, 1)
    else:
        tok = torch.from_numpy(target_tokens(inside, F)).to(device)
        xt = idx.repeat(S, 1)
        xt[:, tok] = MASK
        out = sampler.sample(S, steps, x_t=xt, temperature=temp, device=device)
    assert (out != MASK).all()
    gen = K.decode_idx(vqae, cfg, out).argmax(1).cpu().numpy()     # [S, N, N]
    g = np.repeat(cls[None], S, 0)
    g[:, inside] = gen[:, inside]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--sids", default="")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--steps", type=int, default=32, help="sampling steps (CityDreamer's inference uses 32)")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--viz", type=int, default=0)
    ap.add_argument("--uncond", type=int, default=0, help="draw this many unconditional samples and exit")
    ap.add_argument("--recon", action="store_true",
                    help="reference row: encode and decode the real map, no sampling")
    a = ap.parse_args()
    device = "cuda"
    os.makedirs(a.out, exist_ok=True)
    cfg, vqae, sampler, ck = K.load_ckpt(a.ckpt, device)
    print(f"checkpoint step {ck.get('step')}", flush=True)

    if a.uncond:
        with torch.no_grad():
            torch.manual_seed(0)
            idx = sampler.sample(a.uncond, a.steps, temperature=a.temp, device=device)
            g = K.decode_idx(vqae, cfg, idx).argmax(1).cpu().numpy()
        real, _ = K.load_windows("val", max_n=a.uncond, seed=0)
        K.save_grid(os.path.join(a.out, "uncond.png"),
                    [("real val", r) for r in real] + [("generated", x) for x in g],
                    "NYC-trained CityDreamer: real validation windows (top) vs unconditional samples (bottom)",
                    ncol=a.uncond)
        print("saved", os.path.join(a.out, "uncond.png"))
        return

    city = City()
    if a.sids:
        ids = [int(s) for s in a.sids.split(",")]
    elif a.split == "test":
        ids = test_order(city)[:a.n]
    else:
        ids = [int(s) for s in K.sample_table("val").sample(n=a.n, random_state=0)["sample_id"]]
    if a.viz:
        os.makedirs(os.path.join(a.out, "viz"), exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    json.dump({"args": vars(a), "ckpt_step": ck.get("step")}, log); log.write("\n")
    t_all = time.time()
    for k, sid in enumerate(ids):
        f = os.path.join(a.out, f"{sid}.npz")
        if os.path.exists(f):
            continue
        t0 = time.time()
        cls, blk, _, tgts = city.window(sid)
        inside = np.isin(blk, tgts)
        torch.manual_seed(sid)
        g = inpaint(cfg, vqae, sampler, cls, inside, a.samples, a.steps, a.temp, device, a.recon)
        assert np.array_equal(g[:, ~inside], np.repeat(cls[None], a.samples, 0)[:, ~inside])
        np.savez_compressed(f, gen=g.astype(np.uint8), gt=cls.astype(np.uint8), mask=inside)
        dt = time.time() - t0
        json.dump({"sid": sid, "sec": round(dt, 2)}, log); log.write("\n"); log.flush()
        if k < a.viz:
            K.save_grid(os.path.join(a.out, "viz", f"{sid}.png"),
                        [("real", cls)] + [(f"sample {i + 1}", x) for i, x in enumerate(g)],
                        f"NYC-trained CityDreamer, window {sid} ({a.split}); magenta: target blocks", inside=inside)
        if k < 3 or (k + 1) % 100 == 0:
            print(f"[{k + 1}/{len(ids)}] {sid}: {dt:.1f}s (elapsed {(time.time() - t_all) / 60:.1f} min)", flush=True)
    print("GENERATE_DONE", flush=True)


if __name__ == "__main__":
    main()
