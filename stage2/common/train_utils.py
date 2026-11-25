"""Training loop shared by the three experiments: steps (not epochs), EMA, checkpoints, resume,
validation by generation, early stopping, time limit.

Files in <out>/<name>/: config.yaml, log.csv (training loss), val.csv (validation scores),
latest.pt (resume), best.pt (best validation score, EMA weights), DONE.txt (why training stopped).
"""
import argparse
import copy
import csv
import json
import os
import random
import time

import numpy as np
import torch
import yaml

from . import config as C


def parse(default_config, extra=()):
    """config.yaml + command-line overrides: --set key=value (value parsed as YAML)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--name", required=True)
    ap.add_argument("--subset", default=None, help="overfit8 / pilot2000 / train")
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--out", default=C.OUT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    for args, kw in extra:
        ap.add_argument(*args, **kw)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    for q in a.set:
        k, v = q.split("=", 1)
        if k not in cfg:
            ap.error(f"unknown setting {k!r}; known: {sorted(cfg)}")
        cfg[k] = yaml.safe_load(v)
    if a.subset:
        cfg["subset"] = a.subset
    cfg.update(name=a.name, data=a.data, dir=os.path.join(a.out, a.name))
    os.makedirs(cfg["dir"], exist_ok=True)
    with open(os.path.join(cfg["dir"], "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return a, cfg


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay, self.model, self.n = decay, copy.deepcopy(model).eval(), 0
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for a, b in zip(self.model.state_dict().values(), model.state_dict().values()):
            if a.dtype.is_floating_point:
                a.lerp_(b, 1 - d)
            else:
                a.copy_(b)


def worker_seed(_):
    np.random.seed(torch.initial_seed() % 2 ** 32)


def infinite(loader):
    while True:
        yield from loader


def fit(cfg, model, loader, loss_fn, validate, resume=False):
    """loss_fn(model, batch) -> dict of tensors with "total"; validate(ema_model) -> dict with "score"
    (lower is better). Stops at max_steps, after `patience` validations without improvement, or at
    max_hours (checked between steps, hours count over all resumed runs)."""
    dev = device()
    model.to(dev)
    # optional guard (off unless the config has skip_grad > 0): skip a step whose gradient norm is more than
    # skip_grad times its running average. The average also moves on skipped steps, so it can adapt.
    guard, gavg, gn_seen = float(cfg.get("skip_grad") or 0), 0.0, 0
    ema = EMA(model, cfg["ema"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    amp = dev.type == "cuda" and cfg.get("amp", True)
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
    step, best, bad, spent = 0, float("inf"), 0, 0.0
    d = cfg["dir"]
    if resume and os.path.exists(os.path.join(d, "latest.pt")):
        k = torch.load(os.path.join(d, "latest.pt"), map_location=dev, weights_only=False)
        model.load_state_dict(k["model"])
        ema.model.load_state_dict(k["ema"])
        ema.n = k["ema_n"]
        opt.load_state_dict(k["opt"])
        step, best, bad, spent = k["step"], k["best"], k["bad"], k["hours"]
        print(f"resumed at step {step}, best score {best:.4f}, {spent:.2f} h used", flush=True)
    else:
        for f in ("log.csv", "val.csv", "DONE.txt"):
            if os.path.exists(os.path.join(d, f)):
                os.remove(os.path.join(d, f))
    n = sum(p.numel() for p in model.parameters())
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    print(f"{cfg['name']}: {n / 1e6:.2f} M parameters, {gpu}, subset {cfg['subset']}", flush=True)

    def save(name, **kw):
        tmp = os.path.join(d, name + ".tmp")
        torch.save({"cfg": cfg, "step": step, **kw}, tmp)
        os.replace(tmp, os.path.join(d, name))

    def log(file, row):
        path = os.path.join(d, file)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    t0, acc, why = time.time(), {}, "max_steps"
    hours = lambda: spent + (time.time() - t0) / 3600
    for batch in infinite(loader):
        if step >= cfg["max_steps"]:
            break
        model.train()
        lr = cfg["lr"] * min(1.0, (step + 1) / max(cfg["warmup"], 1))
        for g in opt.param_groups:
            g["lr"] = lr
        batch = {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast(dev.type, dtype=dtype, enabled=amp):
            losses = loss_fn(model, batch)
        if not torch.isfinite(losses["total"]):
            raise RuntimeError(f"loss is not finite at step {step}: {losses}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        skip = False
        if guard > 0:
            g, limit = float(norm), guard * gavg
            skip = gn_seen >= 1000 and not g <= limit        # "not <=" also catches nan and inf
            g = min(g, limit if gn_seen >= 1000 else 1e3) if g == g else limit
            gavg = g if gn_seen == 0 else 0.99 * gavg + 0.01 * g
            gn_seen += 1
            acc["grad"] = acc.get("grad", 0.0) + g
            acc["skipped"] = acc.get("skipped", 0.0) + skip
        if not skip:
            scaler.step(opt)
            scaler.update()
            ema.update(model)
        else:
            scaler.update()
        step += 1
        for k, v in losses.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach())
        if step % cfg["log_every"] == 0:
            row = {"step": step, "hours": round(hours(), 4), **{k: v / cfg["log_every"] for k, v in acc.items()}}
            log("log.csv", row)
            print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
            acc = {}
        out_of_time = hours() >= cfg["max_hours"]
        if step % cfg["val_every"] == 0 or step == cfg["max_steps"] or out_of_time:
            s = validate(ema.model)
            log("val.csv", {"step": step, "hours": round(hours(), 4), **s})
            better = s["score"] < best - cfg["min_delta"]
            if s["score"] < best:
                best = s["score"]
                save("best.pt", model=ema.model.state_dict(), val=s)
            bad = 0 if better else bad + 1
            print(f"== step {step} val score {s['score']:.4f} (best {best:.4f}, {bad} without improvement) "
                  f"{hours():.2f} h", flush=True)
            save("latest.pt", model=model.state_dict(), ema=ema.model.state_dict(), ema_n=ema.n,
                 opt=opt.state_dict(), best=best, bad=bad, hours=hours())
            if bad >= cfg["patience"]:
                why = "no improvement"
                break
            if out_of_time:
                why = "max_hours"
                break
    res = {"stopped": why, "step": step, "hours": round(hours(), 3), "best_score": best, "gpu": gpu, "parameters": n}
    with open(os.path.join(d, "DONE.txt"), "w") as f:
        json.dump(res, f, indent=1)
    print("TRAIN_DONE", res, flush=True)


def load_best(path, build, dev=None):
    """(model in eval mode, cfg) from best.pt or latest.pt."""
    dev = dev or device()
    k = torch.load(path, map_location=dev, weights_only=False)
    m = build(k["cfg"]).to(dev)
    m.load_state_dict(k.get("ema", k["model"]))
    return m.eval(), k["cfg"]
