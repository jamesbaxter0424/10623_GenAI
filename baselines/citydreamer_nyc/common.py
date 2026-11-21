"""Shared pieces for CityDreamer's layout generator retrained on our NYC data.

The model code is CityDreamer's own (third_party/CityDreamer/models/{vqgan,sampler}.py, unmodified).
Only the configuration changes, so that it reads our 128 x 128 maps of 10 classes:

  NETWORK.VQGAN.N_IN_CHANNELS / N_OUT_CHANNELS  7 -> 10   (our 10 classes as one-hot; no height channel)
  NETWORK.VQGAN.RESOLUTION                    512 -> 128  (our window)
  NETWORK.VQGAN.N_CHANNEL_FACTORS   [1,1,2,2,4] -> [2,2,4] (the last three levels of the original:
                                                           downsampling x4, so 128 px -> 32 x 32 tokens,
                                                           the same token grid and attention level)
Everything else (codebook 512, sampler 24 x 512, 256 diffusion steps, optimisers) is the original.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.expanduser("~/genai-final")
CD_ROOT = os.environ.get("CD_ROOT", os.path.join(ROOT, "third_party", "CityDreamer"))
DATA = os.environ.get("NYC_DATA", os.path.join(ROOT, "data", "city"))
N, NC = 128, 10


def cd_models():
    if CD_ROOT not in sys.path:
        sys.path.insert(0, CD_ROOT)
    import models.sampler
    import models.vqgan
    return models.vqgan, models.sampler


def make_cfg():
    spec = importlib.util.spec_from_file_location("cd_config", os.path.join(CD_ROOT, "config.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = m.cfg
    v = cfg.NETWORK.VQGAN
    v.N_IN_CHANNELS = NC
    v.N_OUT_CHANNELS = NC
    v.RESOLUTION = N
    v.N_CHANNEL_FACTORS = [2, 2, 4]
    assert v.ATTN_RESOLUTION == 32 and cfg.NETWORK.SAMPLER.BLOCK_SIZE == 32 * 32
    cfg.DATASETS.OSM_LAYOUT.N_CLASSES = NC
    cfg.DATASETS.OSM_LAYOUT.IGNORED_CLASSES = []
    return cfg


def fp32_quantizer(vqae):
    """Keep the codebook lookup in float32 when the rest runs under bf16 autocast, so token
    assignment is not affected by low precision."""
    q = vqae.quantize
    fwd = q.forward

    def forward(z, *args, **kw):
        with torch.autocast("cuda", enabled=False):
            return fwd(z.float(), *args, **kw)
    q.forward = forward
    return vqae


def build(cfg, device, sampler=True):
    vq_mod, s_mod = cd_models()
    vqae = fp32_quantizer(vq_mod.VQAutoEncoder(cfg).to(device))
    smp = s_mod.AbsorbingDiffusionSampler(cfg).to(device) if sampler else None
    return vqae, smp


def load_ckpt(path, device, sampler=True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vqae, smp = build(ck["cfg"], device, sampler and "sampler" in ck)
    vqae.load_state_dict(ck["vqae"], strict=True)
    if smp is not None:
        smp.load_state_dict(ck["sampler"], strict=True)
    return ck["cfg"], vqae.eval(), (smp.eval() if smp is not None else None), ck


# ---------------------------------------------------------------------- data
def sample_table(split):
    """Same windows as the Liu baseline (NYCSceneGraphDataset): split, minus windows whose
    context shows blocks of another split."""
    s = pd.read_parquet(os.path.join(DATA, "samples.parquet"))
    return s[(s["split"] == split) & (~s["ctx_other_split"])].reset_index(drop=True)


def load_windows(split, max_n=None, seed=0):
    s = sample_table(split)
    if max_n:
        s = s.sample(n=min(max_n, len(s)), random_state=seed).reset_index(drop=True)
    cls = np.load(os.path.join(DATA, "cls.npy"))                 # whole city in RAM (~150 MB)
    out = np.empty((len(s), N, N), np.uint8)
    for i, (r, c) in enumerate(zip(s["row0"].to_numpy(), s["col0"].to_numpy())):
        out[i] = cls[r:r + N, c:c + N]
    assert out.max() < NC
    return out, s["sample_id"].to_numpy()


def to_input(x_int, flip=False):
    """[B,N,N] int classes on device -> one-hot [B,NC,N,N] float (random flips as in CityDreamer)."""
    if flip:
        b = x_int.shape[0]
        h = torch.rand(b, device=x_int.device) < 0.5
        v = torch.rand(b, device=x_int.device) < 0.5
        x_int = torch.where(h[:, None, None], x_int.flip(-1), x_int)
        x_int = torch.where(v[:, None, None], x_int.flip(-2), x_int)
    return F.one_hot(x_int, NC).permute(0, 3, 1, 2).float(), x_int


@torch.no_grad()
def encode_idx(vqae, onehot):
    _, _, info = vqae.encode(onehot)
    return info["min_encoding_indices"].view(onehot.shape[0], -1)


@torch.no_grad()
def decode_idx(vqae, cfg, idx):
    """[B, 1024] token ids -> class logits [B, NC, N, N]."""
    F_ = cfg.NETWORK.VQGAN.ATTN_RESOLUTION
    quant = vqae.quantize.get_codebook()[idx].view(idx.shape[0], F_, F_, -1).permute(0, 3, 1, 2).contiguous()
    return vqae.decode(quant)


PALETTE = np.array([[200, 200, 200], [235, 225, 200], [255, 230, 150], [250, 180, 80], [225, 110, 50],
                    [160, 40, 30], [190, 220, 255], [120, 170, 240], [60, 110, 210], [20, 50, 140]], np.uint8)


def save_grid(path, panels, title, ncol=None, inside=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = ncol or len(panels)
    nrow = int(np.ceil(len(panels) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.8 * nrow), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for a, (t, m) in zip(ax.ravel(), panels):
        img = PALETTE[m].copy()
        if inside is not None:
            p = np.pad(inside, 1)
            edge = inside & ~(p[2:, 1:-1] & p[:-2, 1:-1] & p[1:-1, 2:] & p[1:-1, :-2])
            img[edge] = (255, 0, 255)
        a.imshow(img, interpolation="nearest")
        a.set_title(t, fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
