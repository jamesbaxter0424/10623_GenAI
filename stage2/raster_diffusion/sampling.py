"""Run the trained U-Net on whole blocks."""
import numpy as np
import torch

from common.dataset import RasterFrame
from common.diffusion import Diffusion

from .instances import buildings
from .targets import C_INSIDE, decode, pad_cond


@torch.no_grad()
def generate_maps(model, cond, cfg, steps, seed, dev):
    """cond [12, H, W] numpy -> [3, H, W] numpy in [-1, 1]."""
    h, w = cond.shape[1:]
    c = torch.from_numpy(pad_cond(cond)).to(dev)[None]
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    shape = (1, 3, *c.shape[-2:])
    if cfg["mode"] == "regression":
        with amp:
            x = model(torch.zeros(shape, device=dev), torch.zeros(1, device=dev), c).float()
    else:
        g = torch.Generator(dev).manual_seed(seed)

        def f(x, t):
            with amp:
                return model(x, t, c).float()
        x = Diffusion().sample(f, shape, dev, steps, clip=1.0, generator=g)
    return x[0, :, :h, :w].cpu().numpy()


def predict(model, rec, cls4, cfg, steps, seed, dev):
    """(polygons, heights) for one block and one sample; also works for the regression mode."""
    f = RasterFrame(rec["inside"])
    cond = f.cond(cls4, rec["inside"])
    return to_buildings(generate_maps(model, cond, cfg, steps, seed, dev), cond, f, cfg)


def to_buildings(x, cond, frame, cfg):
    """Maps in the block frame -> (polygons, heights) in local coordinates."""
    mask, dist, height = decode(x, cfg)
    g, h = buildings(mask & (cond[C_INSIDE] > 0), dist, height, cfg)
    return [frame.poly_back(p) for p in g], h


def block_seed(row, sample):
    return int(np.random.SeedSequence([row, sample]).generate_state(1)[0])
