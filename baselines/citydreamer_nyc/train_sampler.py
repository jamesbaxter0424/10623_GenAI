"""Stage 2: train CityDreamer's absorbing-diffusion sampler on the tokens of the frozen VQGAN.

As in CityDreamer (core/sampler/train.py): tokens of each (randomly flipped) training window are
encoded by the frozen VQGAN; a random fraction t/T of them is masked; the transformer predicts the
masked tokens; loss = reweighted ELBO; Adam lr 2e-4, betas (0.9, 0.999), linear warm-up over 7,500
iterations, batch 10. Training stops at a wall-clock budget.

  python train_sampler.py --vqgan VQ/latest.pth --overfit 8 --steps 2000 --out OUT
  python train_sampler.py --vqgan VQ/latest.pth --hours 8 --out OUT    (resumes from OUT/latest.pth)
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import common as K


def rw_elbo_loss(sampler, cfg, x0):
    T = cfg.NETWORK.SAMPLER.TOTAL_STEPS
    t, pt, logits, x0_ignore = sampler(x0)
    cil = torch.nn.functional.cross_entropy(logits, x0_ignore, ignore_index=-1, reduction="none").sum(1)
    rw = ((1 - t / T) * cil / (math.log(2) * x0.size(1))).mean()
    elbo = (cil / t / pt / (math.log(2) * x0.size(1))).mean()
    return rw, elbo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqgan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0, help="0 = CityDreamer's TRAIN.SAMPLER.BATCH_SIZE")
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", type=int, default=1, help="bf16 autocast (codebook lookup stays float32)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda"
    os.makedirs(a.out, exist_ok=True)
    cfg, vqae, _, _ = K.load_ckpt(a.vqgan, device, sampler=False)
    for p in vqae.parameters():
        p.requires_grad_(False)
    _, s_mod = K.cd_models()
    sampler = s_mod.AbsorbingDiffusionSampler(cfg).to(device)
    S = cfg.TRAIN.SAMPLER
    bs = a.batch or S.BATCH_SIZE
    opt = torch.optim.Adam(sampler.parameters(), lr=S.LR, weight_decay=S.WEIGHT_DECAY, betas=S.BETAS)
    step = 0
    latest = os.path.join(a.out, "latest.pth")
    if os.path.exists(latest) and not a.overfit:
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        sampler.load_state_dict(ck["sampler"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print("resumed at step", step, flush=True)

    if a.overfit:
        train, _ = K.load_windows("train", max_n=a.overfit, seed=a.seed)
        val = train
    else:
        train, _ = K.load_windows("train")
        val, _ = K.load_windows("val", max_n=200, seed=a.seed)
    print(f"train {len(train)}, val {len(val)}; sampler {sum(p.numel() for p in sampler.parameters()) / 1e6:.1f}M "
          f"params; batch {bs}; lr {S.LR} (warm-up {S.N_WARMUP_ITERS})", flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    sampler.train()
    while True:
        if step <= S.N_WARMUP_ITERS:
            for g in opt.param_groups:
                g["lr"] = S.LR * step / S.N_WARMUP_ITERS
        idx = np.random.randint(0, len(train), bs)
        x = torch.from_numpy(train[idx]).to(device).long()
        oh, _ = K.to_input(x, flip=not a.overfit)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.bf16)):
            x0 = K.encode_idx(vqae, oh)
            loss, elbo = rw_elbo_loss(sampler, cfg, x0)
        if torch.isnan(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        el = (time.time() - t0) / 3600
        done = (a.steps and step >= a.steps) or el >= a.hours
        if step % 100 == 0 or done:
            rec = {"step": step, "hours": round(el, 3), "rw_elbo": round(loss.item(), 4), "elbo": round(elbo.item(), 4)}
            if step % a.val_every == 0 or done:
                sampler.eval()
                with torch.no_grad():
                    torch.manual_seed(1)
                    vl = []
                    for i in range(0, len(val), 20):
                        oh, _ = K.to_input(torch.from_numpy(val[i:i + 20]).to(device).long())
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.bf16)):
                            vl.append(rw_elbo_loss(sampler, cfg, K.encode_idx(vqae, oh))[0].item())
                    rec["val_rw_elbo"] = round(float(np.mean(vl)), 4)
                sampler.train()
            if step % a.save_every == 0 or done:
                torch.save({"cfg": cfg, "vqae": vqae.state_dict(), "sampler": sampler.state_dict(),
                            "opt": opt.state_dict(), "step": step}, latest)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
        if done:
            break
    print("SAMPLER_DONE", flush=True)


if __name__ == "__main__":
    main()
