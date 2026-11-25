"""Losses for the set diffusion model."""
import torch

from common.diffusion import Diffusion

from .tokens import LOG_A, LOG_B, SCALE


def sides(x):
    a = torch.exp((x[..., 3] * LOG_A[1] + LOG_A[0]).clamp(0, 6))
    b = torch.exp((x[..., 4] * LOG_B[1] + LOG_B[0]).clamp(0, 6))
    return a, b


def overlap(x, weight):
    """Mean share of each building covered by others. Boxes are treated as aligned with the block
    (true for most buildings). x [B, N, 7] clean tokens, weight [B, N] in 0..1 (1 = real building)."""
    u, v = x[..., 1] * SCALE, x[..., 2] * SCALE
    a, b = sides(x)
    du = (a[:, :, None] + a[:, None]) / 2 - (u[:, :, None] - u[:, None]).abs()
    dv = (b[:, :, None] + b[:, None]) / 2 - (v[:, :, None] - v[:, None]).abs()
    inter = du.clamp(min=0) * dv.clamp(min=0)
    inter = inter * weight[:, :, None] * weight[:, None] * (1 - torch.eye(x.shape[1], device=x.device))
    share = inter.sum(-1) / (a * b).clamp(min=1.0)
    return (share.clamp(max=1.0) * weight).sum(1) / weight.sum(1).clamp(min=1.0)


def make_loss(cfg):
    diff = Diffusion()

    def loss_fn(model, b):
        x0 = b["x0"]
        real = (x0[..., 0] > 0).to(x0.dtype)
        xt, t, v = diff.training_pair(x0)
        p = model(xt, t, model.encode_map(b)).float()
        err = (p - v) ** 2
        exist = err[..., 0].mean()
        # geometry of empty tokens carries no information; a small weight keeps it bounded
        w = real + 0.05 * (1 - real)
        geom = (err[..., 1:] * w[..., None]).sum() / (w.sum() * err[..., 1:].shape[-1])
        out = {"exist": exist, "geom": geom}
        total = cfg["w_exist"] * exist + cfg["w_geom"] * geom
        if cfg["w_overlap"] > 0:
            with torch.autocast(x0.device.type, enabled=False):
                alpha, _ = diff.alpha_sigma(t)               # only where the clean set is visible
                ov = (overlap(diff.x0_from_v(xt, t, p), real) * alpha ** 4).mean()
            out["overlap"] = ov
            total = total + cfg["w_overlap"] * ov
        return {"total": total, **out}
    return loss_fn
