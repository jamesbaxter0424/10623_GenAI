"""CNN over the map + Transformer over the building tokens.

Each token looks at (1) all other tokens (self-attention: do not overlap, line up), (2) the whole map
(cross-attention: how many buildings and where), (3) the map feature under its current position.
With order: sorted every token also knows its slot number, so the model only has to decide how many
slots are used; with order: random the set is unordered and only the noise tells the tokens apart.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.dataset import COND_CHANNELS, G

from .tokens import DIM, SCALE


def fourier(x, n, max_freq):
    f = torch.exp(torch.linspace(0, math.log(max_freq), n, device=x.device)) * math.pi
    a = x[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class MapEncoder(nn.Module):
    """64 x 64 map -> fine features (32 x 32, for look-ups) and 256 coarse tokens (16 x 16)."""

    def __init__(self, dim):
        super().__init__()
        def block(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1), nn.GroupNorm(16, o), nn.SiLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(16, o), nn.SiLU())
        self.a = block(COND_CHANNELS, 64, 1)
        self.b = block(64, 128, 2)
        self.c = block(128, dim, 2)
        self.fine = nn.Conv2d(64 + 128, dim, 1)
        self.pos = nn.Parameter(torch.randn(1, (G // 4) ** 2, dim) * 0.02)
        self.scal = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, cond, scal):
        a = self.a(cond)
        b = self.b(a)
        c = self.c(b)
        fine = self.fine(torch.cat([F.avg_pool2d(a, 2), b], 1))
        return fine, c.flatten(2).transpose(1, 2) + self.pos, self.scal(scal)


class NormedAttention(nn.Module):
    """Multi-head attention with normalised queries and keys (QK-norm): the attention logits cannot grow
    without limit, a usual cause of training that diverges late. Called like nn.MultiheadAttention."""

    def __init__(self, dim, heads, drop):
        super().__init__()
        self.heads, self.drop = heads, drop
        self.q, self.k, self.v, self.out = (nn.Linear(dim, dim) for _ in range(4))
        self.qn, self.kn = nn.LayerNorm(dim // heads), nn.LayerNorm(dim // heads)
        for m in (self.q, self.k, self.v, self.out):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, q, k, v, need_weights=False):
        s = lambda x: x.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        q, k, v = s(self.q(q)), s(self.k(k)), s(self.v(v))
        q, k = self.qn(q).to(v.dtype), self.kn(k).to(v.dtype)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return self.out(x.transpose(1, 2).flatten(2)), None


class Block(nn.Module):
    def __init__(self, dim, heads, drop, qk_norm=False):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(dim, elementwise_affine=False) for _ in range(3))
        att = (lambda: NormedAttention(dim, heads, drop)) if qk_norm else (
            lambda: nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True))
        self.sa, self.ca = att(), att()
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * dim, dim))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, x, ctx, e):
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.mod(e)[:, None].chunk(9, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + (1 + g1) * self.sa(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + s2) + b2
        x = x + (1 + g2) * self.ca(h, ctx, ctx, need_weights=False)[0]
        h = self.n3(x) * (1 + s3) + b3
        return x + (1 + g3) * self.ff(h)


class SetDenoiser(nn.Module):
    def __init__(self, dim=256, layers=8, heads=8, dropout=0.0, slots=0, qk_norm=False):
        super().__init__()
        self.slot = nn.Parameter(torch.randn(1, slots, dim) * 0.02) if slots else None
        self.map = MapEncoder(dim)
        self.inp = nn.Linear(DIM + 2 * 2 * 8, dim)
        self.look = nn.Linear(dim, dim)
        self.time = nn.Sequential(nn.Linear(32, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(Block(dim, heads, dropout, qk_norm) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, DIM)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def encode_map(self, b):
        return (*self.map(b["cond"], b["scal"]), b["half"])

    def forward(self, x, t, enc):
        fine, ctx, scal, half = enc
        e = self.time(fourier(t[:, None], 16, 100.0)) + scal
        uv = x[..., 1:3] * SCALE / half[:, None, None]       # canonical coordinates in [-1, 1]
        here = F.grid_sample(fine, uv[:, :, None].to(fine.dtype), align_corners=False)[..., 0].transpose(1, 2)
        h = self.inp(torch.cat([x, fourier(x[..., 1:3] * SCALE / 64, 8, 64.0)], -1)) + self.look(here)
        if self.slot is not None:
            h = h + self.slot
        for blk in self.blocks:
            h = blk(h, ctx, e)
        return self.out(self.norm(h))


def build(cfg):
    return SetDenoiser(cfg["dim"], cfg["layers"], cfg["heads"], cfg["dropout"],
                       cfg["tokens"] if cfg["order"] == "sorted" else 0,
                       cfg.get("qk_norm", False))            # older checkpoints have no such setting
