"""Continuous Gaussian diffusion (cosine schedule, v-prediction, DDIM), for images and for sets.

The model is any callable model(x_t, t, *cond) -> v with t in [0, 1] (float tensor of shape [B]).
"""
import math

import torch


class Diffusion:
    def __init__(self, steps=1000):
        self.steps = steps

    @staticmethod
    def alpha_sigma(t):
        """alpha_t, sigma_t for t in [0, 1] (cosine schedule); t = 0 is clean, t = 1 is noise."""
        a = torch.cos(t * math.pi / 2).clamp(min=1e-4)
        return a, (1 - a ** 2).clamp(min=0).sqrt()

    @staticmethod
    def _ex(v, x):
        return v.view(-1, *([1] * (x.ndim - 1)))

    def training_pair(self, x0, t=None, noise=None):
        """Random t and noise: returns x_t, t, target v."""
        t = torch.rand(x0.shape[0], device=x0.device) if t is None else t
        noise = torch.randn_like(x0) if noise is None else noise
        a, s = (self._ex(q, x0) for q in self.alpha_sigma(t))
        return a * x0 + s * noise, t, a * noise - s * x0

    def x0_from_v(self, xt, t, v):
        a, s = (self._ex(q, xt) for q in self.alpha_sigma(t))
        return a * xt - s * v

    @torch.no_grad()
    def sample(self, model, shape, device, steps=50, eta=0.0, clip=None, generator=None, fix=None):
        """DDIM. clip: clamp the predicted x0 to [-clip, clip]. fix(x0) -> x0 can impose known values."""
        x = torch.randn(shape, device=device, generator=generator)
        ts = torch.linspace(1, 0, steps + 1, device=device)
        for i in range(steps):
            t = torch.full((shape[0],), float(ts[i]), device=device)
            v = model(x, t)
            a, s = (self._ex(q, x) for q in self.alpha_sigma(t))
            x0 = a * x - s * v
            if clip is not None:
                x0 = x0.clamp(-clip, clip)
            if fix is not None:
                x0 = fix(x0)
            eps = (x - a * x0) / s.clamp(min=1e-4)
            t2 = torch.full((shape[0],), float(ts[i + 1]), device=device)
            a2, s2 = (self._ex(q, x) for q in self.alpha_sigma(t2))
            if i == steps - 1:
                return x0
            sd = eta * s2 / s.clamp(min=1e-4) * (1 - (a / a2) ** 2).clamp(min=0).sqrt()
            x = a2 * x0 + (s2 ** 2 - sd ** 2).clamp(min=0).sqrt() * eps
            if eta > 0:
                x = x + sd * torch.randn(shape, device=device, generator=generator)
        return x
