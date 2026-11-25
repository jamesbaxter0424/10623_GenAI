"""Map -> GlobalMapper latent.

codes     the latent is 512 integers (20 levels). 64 tokens of 8 dimensions each; a Transformer that
          looks at the map fills in masked tokens (MaskGIT / COHO style), so the dimensions stay
          consistent with each other instead of being drawn independently.
gaussian  fallback: mean and variance of the latent given the map.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.dataset import COND_CHANNELS

from .gm import AXIS_H, AXIS_W


class MapEncoder(nn.Module):
    """[12, 32, 128] axis-aligned map -> 8 x 32 tokens."""

    def __init__(self, dim):
        super().__init__()
        def block(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1), nn.GroupNorm(16, o), nn.SiLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(16, o), nn.SiLU())
        self.net = nn.Sequential(block(COND_CHANNELS, 64, 1), block(64, 128, 2), block(128, dim, 2))
        self.pos = nn.Parameter(torch.randn(1, (AXIS_H // 4) * (AXIS_W // 4), dim) * 0.02)
        self.scal = nn.Linear(3, dim)

    def forward(self, cond, scal):
        return self.net(cond).flatten(2).transpose(1, 2) + self.pos + self.scal(scal)[:, None]


class Codes(nn.Module):
    def __init__(self, cfg, latent=512):
        super().__init__()
        d, self.L, self.G = cfg["dim"], cfg["levels"], cfg["group"]
        self.n = latent // self.G
        self.map = MapEncoder(d)
        self.emb = nn.Parameter(torch.randn(latent, self.L, 32) * 0.02)      # one table per latent dimension
        self.inp = nn.Linear(32 * self.G, d)
        self.mask = nn.Parameter(torch.zeros(d))
        self.pos = nn.Parameter(torch.randn(1, self.n, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, cfg["heads"], 4 * d, cfg["dropout"], "gelu", batch_first=True, norm_first=True)
        self.body = nn.TransformerDecoder(layer, cfg["layers"], nn.LayerNorm(d))
        self.out = nn.Linear(d, self.G * self.L)

    def logits(self, code, masked, ctx):
        """code [B, 512] (ignored where masked), masked [B, n] bool -> [B, n, G, L]."""
        B = code.shape[0]
        e = self.emb[torch.arange(code.shape[1], device=code.device), code]          # [B, 512, 32]
        h = self.inp(e.reshape(B, self.n, -1))
        h = torch.where(masked[..., None], self.mask.to(h.dtype), h) + self.pos
        return self.out(self.body(h, ctx)).view(B, self.n, self.G, self.L)

    def loss(self, b):
        ctx, code = self.map(b["cond"], b["scal"]), b["code"]
        ratio = torch.cos(torch.rand(len(code), 1, device=code.device) * math.pi / 2)
        score = torch.rand(len(code), self.n, device=code.device)
        masked = score <= torch.maximum(ratio, score.min(1, keepdim=True).values)
        lg = self.logits(code, masked, ctx).float()
        tgt = code.view(len(code), self.n, self.G)
        ce = F.cross_entropy(lg.reshape(-1, self.L), tgt.reshape(-1), reduction="none").view(tgt.shape)
        m = masked[..., None].float()
        total = (ce * m).sum() / (m.sum() * self.G)
        acc = ((lg.argmax(-1) == tgt).float() * m).sum() / (m.sum() * self.G)
        return {"total": total, "acc": acc}

    @torch.no_grad()
    def sample(self, b, rounds=12, temperature=1.0, generator=None):
        ctx = self.map(b["cond"], b["scal"])
        B, dev = len(ctx), ctx.device
        code = torch.zeros(B, self.n * self.G, dtype=torch.long, device=dev)
        masked = torch.ones(B, self.n, dtype=torch.bool, device=dev)
        for r in range(rounds):
            p = (self.logits(code, masked, ctx).float() / max(temperature, 1e-3)).softmax(-1)
            new = torch.multinomial(p.reshape(-1, self.L), 1, generator=generator).view(B, self.n, self.G)
            conf = p.gather(-1, new[..., None])[..., 0].clamp(min=1e-9).log().mean(-1)
            conf = conf.masked_fill(~masked, float("inf"))                           # accepted tokens stay
            old = code.view(B, self.n, self.G)
            code = torch.where(masked[..., None], new, old).view(B, -1)
            left = int(math.floor(self.n * math.cos((r + 1) / rounds * math.pi / 2))) if r < rounds - 1 else 0
            masked = torch.zeros_like(masked)
            if left > 0:                                                             # least confident again
                masked.scatter_(1, conf.topk(left, 1, largest=False).indices, True)
        return code


class Gaussian(nn.Module):
    def __init__(self, cfg, latent=512):
        super().__init__()
        d = cfg["dim"]
        self.map = MapEncoder(d)
        layer = nn.TransformerEncoderLayer(d, cfg["heads"], 4 * d, cfg["dropout"], "gelu", batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, cfg["layers"], nn.LayerNorm(d))
        self.out = nn.Linear(d, 2 * latent)

    def stats(self, b):
        m, s = self.out(self.body(self.map(b["cond"], b["scal"])).mean(1)).float().chunk(2, -1)
        return m, s.clamp(-5, 2)

    def loss(self, b):
        m, s = self.stats(b)
        return {"total": (s + 0.5 * ((b["mu"] - m) / s.exp()) ** 2).mean(), "mse": ((b["mu"] - m) ** 2).mean()}

    @torch.no_grad()
    def sample(self, b, temperature=1.0, generator=None, **_):
        m, s = self.stats(b)
        return m + s.exp() * temperature * torch.randn(m.shape, device=m.device, generator=generator)


def build(cfg):
    return Codes(cfg) if cfg["head"] == "codes" else Gaussian(cfg)
