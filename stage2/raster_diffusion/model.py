"""U-Net with time conditioning. Fully convolutional except for one self-attention at 1/8 resolution,
so it is trained on 256 x 256 crops and run on whole blocks of any size (padded to multiples of 16)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.dataset import COND_CHANNELS

OUT = 3                                                      # building mask, distance, height


def time_embedding(t, dim):
    f = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    a = t[:, None] * 1000 * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Res(nn.Module):
    def __init__(self, i, o, emb, drop=0.0):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, i), nn.Conv2d(i, o, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(32, o), nn.Conv2d(o, o, 3, padding=1)
        self.e = nn.Linear(emb, 2 * o)
        self.drop = nn.Dropout(drop)
        self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        a, b = self.e(F.silu(e))[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + a) + b
        return self.skip(x) + self.c2(self.drop(F.silu(h)))


class Attention(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n, self.heads = nn.GroupNorm(32, c), heads
        self.qkv, self.o = nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, self.heads, c // self.heads, h * w).transpose(-1, -2).unbind(1)
        y = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(b, c, h, w)
        return x + self.o(y)


class UNet(nn.Module):
    def __init__(self, base=64, mults=(1, 2, 3, 4), blocks=2, dropout=0.0):
        super().__init__()
        emb = base * 4
        self.base = base
        self.time = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Conv2d(OUT + COND_CHANNELS, base, 3, padding=1)
        ch, skips = base, [base]
        self.down = nn.ModuleList()
        for i, m in enumerate(mults):
            for _ in range(blocks):
                self.down.append(Res(ch, base * m, emb, dropout))
                ch = base * m
                skips.append(ch)
            if i < len(mults) - 1:
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skips.append(ch)
        self.mid1, self.att, self.mid2 = Res(ch, ch, emb, dropout), Attention(ch), Res(ch, ch, emb, dropout)
        self.upb = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(blocks + 1):
                self.upb.append(Res(ch + skips.pop(), base * m, emb, dropout))
                ch = base * m
            if i > 0:
                self.upb.append(nn.Conv2d(ch, ch, 3, padding=1))
        self.out = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv2d(ch, OUT, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, cond):
        e = self.time(time_embedding(t, self.base))
        h = self.inp(torch.cat([x, cond], 1))
        hs = [h]
        for m in self.down:
            h = m(h, e) if isinstance(m, Res) else m(h)
            hs.append(h)
        h = self.mid2(self.att(self.mid1(h, e)), e)
        for m in self.upb:
            if isinstance(m, Res):
                h = m(torch.cat([h, hs.pop()], 1), e)
            else:
                h = m(F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(h)


def build(cfg):
    return UNet(cfg["base"], tuple(cfg["mults"]), cfg["blocks"], cfg["dropout"])
