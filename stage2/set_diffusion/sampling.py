"""Sample building sets for whole blocks (several samples of one block in one batch)."""
import torch

from common.dataset import canonical
from common.diffusion import Diffusion

from .losses import overlap
from .tokens import DIM, decode, to_polygons


def guided(x0, strength):
    """One gradient step that reduces overlaps between the boxes that currently exist."""
    with torch.enable_grad():
        z = x0.detach().requires_grad_(True)
        w = torch.sigmoid(4 * z.detach()[..., 0])
        g = torch.autograd.grad(overlap(z, w).sum() * z.shape[1], z)[0]
    out = x0.clone()
    out[..., 1:5] = x0[..., 1:5] - strength * g[..., 1:5].clamp(-0.2, 0.2)
    return out


@torch.no_grad()
def predict_batch(model, recs, maps, cfg, seed, dev, steps=None):
    """recs and maps: equally long lists. Returns [(polygons, heights)]."""
    pairs = [canonical(m, r["inside"]) for r, m in zip(recs, maps)]
    b = {"cond": torch.stack([torch.from_numpy(c) for c, _ in pairs]).to(dev),
         "scal": torch.stack([torch.from_numpy(f.scalars()) for _, f in pairs]).to(dev),
         "half": torch.tensor([f.half for _, f in pairs], dtype=torch.float32, device=dev)}
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    with amp:
        enc = model.encode_map(b)

    def f(x, t):
        with amp:
            return model(x, t, enc).float()
    fix = (lambda x0: guided(x0, cfg["guide_overlap"])) if cfg["guide_overlap"] > 0 else None
    g = torch.Generator(dev).manual_seed(seed)
    x = Diffusion().sample(f, (len(recs), cfg["tokens"], DIM), dev, steps or cfg["sample_steps"],
                           clip=16.0, generator=g, fix=fix).cpu().numpy()
    return [to_polygons(decode(q, cfg["exist_thresh"]), fr, cfg["trim"]) for q, (_, fr) in zip(x, pairs)]
