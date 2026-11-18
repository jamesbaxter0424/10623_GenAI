"""Smoke test + timing for the NYC baseline on the real data loader.

  python tools/nyc_bench.py -c configs/nyc_train.yaml --iters 60 --batch_sizes 16 24
Reports per batch size: seconds per training step, share of time waiting for data,
peak GPU memory, and (once) seconds per generated window (100 diffusion steps).
"""
import argparse
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets.data import build_train_loader  # noqa: E402
from models import build_model  # noqa: E402
from utils.config import Config  # noqa: E402


def bench(cfg, iters):
    torch.cuda.reset_peak_memory_stats()
    model = build_model(cfg).cuda().train()
    model.freeze_localization_head()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    loader, _ = build_train_loader(cfg)
    it = iter(loader)
    t_data = t_all = 0.0
    losses = []
    for i in range(iters):
        t0 = time.time()
        batch = next(it)
        sg, bev = batch["scene_graph"].cuda(), batch["bev"].cuda()
        torch.cuda.synchronize(); t1 = time.time()
        out = model(sg, bev, mode="joint")
        opt.zero_grad(); out["total"].backward(); opt.step()
        torch.cuda.synchronize(); t2 = time.time()
        if i >= 10:  # skip warm-up
            t_data += t1 - t0; t_all += t2 - t0
        losses.append(float(out["total"]))
    n = iters - 10
    return model, dict(s_per_step=t_all / n, data_wait=t_data / t_all,
                       peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                       loss_first=sum(losses[:5]) / 5, loss_last=sum(losses[-5:]) / 5,
                       finite=all(l == l for l in losses))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[16])
    a = p.parse_args()
    base = yaml.safe_load(open(a.config))
    n_train = None
    for bs in a.batch_sizes:
        cfg = Config(**{**base, "batch_size": bs})
        model, r = bench(cfg, a.iters)
        if n_train is None:
            from datasets.nyc_dataset import NYCSceneGraphDataset
            n_train = len(NYCSceneGraphDataset(cfg.nyc_root, cfg.train_split))
        epoch_min = n_train / bs * r["s_per_step"] / 60
        print(f"[batch {bs}] {r['s_per_step']:.3f} s/step | data wait {100 * r['data_wait']:.0f}% | "
              f"peak GPU {r['peak_gb']:.1f} GB | epoch {epoch_min:.1f} min | "
              f"loss {r['loss_first']:.3f} -> {r['loss_last']:.3f} | finite {r['finite']}", flush=True)
        del model; torch.cuda.empty_cache()

    # generation speed: one batch of 8 windows, 100 diffusion steps
    cfg = Config(**{**base, "batch_size": 8, "feature_mask_ratio": 0.0})
    model = build_model(cfg).cuda().eval()
    loader, _ = build_train_loader(cfg)
    sg = next(iter(loader))["scene_graph"].cuda()
    torch.cuda.synchronize(); t0 = time.time()
    out = model.sample(sg)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[generate] batch 8 x 100 steps: {dt:.1f} s -> {dt / 8:.2f} s per window, "
          f"output {tuple(out.shape)}", flush=True)


if __name__ == "__main__":
    main()
