"""Stage 1: train CityDreamer's VQGAN on our NYC training windows.

As in CityDreamer (core/vqgan/train.py): Adam, betas (0.5, 0.9), lr = 4.5e-6 x batch, loss =
segmentation cross-entropy + quantisation loss. Its height L1 and smoothness terms are dropped
because our maps have no height channel (height is part of the class). Batch 32 x 128^2 px equals
the original 2 x 512^2 px per step. Training stops at a wall-clock budget.

  python train_vqgan.py --overfit 8 --steps 1500 --out OUT     # pipeline check: must reach ~1.0
  python train_vqgan.py --hours 4 --out OUT                      # full run (resumes from OUT/latest.pth)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import common as K


@torch.no_grad()
def recon_acc(vqae, windows, device, bs=64):
    vqae.eval()
    tot = ok = btot = bok = 0
    for i in range(0, len(windows), bs):
        x = torch.from_numpy(windows[i:i + bs]).to(device).long()
        oh, _ = K.to_input(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _ = vqae(oh)
        p = pred.argmax(1)
        tot += x.numel(); ok += (p == x).sum().item()
        b = x >= 2
        btot += b.sum().item(); bok += (p[b] == x[b]).sum().item()
    vqae.train()
    return ok / tot, bok / max(btot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=0, help="stop after this many steps (0 = time budget only)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--overfit", type=int, default=0, help="train and score on this many training windows")
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", type=int, default=1, help="bf16 autocast (codebook lookup stays float32)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda"
    os.makedirs(a.out, exist_ok=True)
    cfg = K.make_cfg()
    vqae, _ = K.build(cfg, device, sampler=False)
    lr = cfg.TRAIN.VQGAN.BASE_LR * a.batch
    opt = torch.optim.Adam(vqae.parameters(), lr=lr, weight_decay=cfg.TRAIN.VQGAN.WEIGHT_DECAY,
                           betas=cfg.TRAIN.VQGAN.BETAS)
    step = 0
    latest = os.path.join(a.out, "latest.pth")
    if os.path.exists(latest) and not a.overfit:
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        vqae.load_state_dict(ck["vqae"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print("resumed at step", step, flush=True)

    if a.overfit:
        train, _ = K.load_windows("train", max_n=a.overfit, seed=a.seed)
        val = train
    else:
        train, _ = K.load_windows("train")
        val, _ = K.load_windows("val", max_n=a.n_val, seed=a.seed)
    print(f"train {len(train)} windows, val {len(val)}; params "
          f"{sum(p.numel() for p in vqae.parameters()) / 1e6:.1f}M; lr {lr:.2e}", flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    flip = not a.overfit
    vqae.train()
    while True:
        idx = np.random.randint(0, len(train), a.batch)
        x = torch.from_numpy(train[idx]).to(device).long()
        oh, x = K.to_input(x, flip=flip)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.bf16)):
            pred, q_loss = vqae(oh)
        seg = F.cross_entropy(pred.float(), x) * cfg.TRAIN.VQGAN.SEG_LOSS_FACTOR
        loss = seg + q_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        el = (time.time() - t0) / 3600
        done = (a.steps and step >= a.steps) or el >= a.hours
        if step % 100 == 0 or done:
            rec = {"step": step, "hours": round(el, 3), "seg": round(seg.item(), 4), "quant": round(q_loss.item(), 5)}
            if step % a.val_every == 0 or done:
                acc, bacc = recon_acc(vqae, val, device)
                rec.update(val_acc=round(acc, 4), val_building_acc=round(bacc, 4))
                torch.save({"cfg": cfg, "vqae": vqae.state_dict(), "opt": opt.state_dict(), "step": step,
                            "val_acc": acc, "val_building_acc": bacc}, latest)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
        if done:
            break
    print("VQGAN_DONE", flush=True)


if __name__ == "__main__":
    main()
