"""Pre-training checks for change 3 (context input). Exits non-zero if any check fails.

  python tools/nyc_c3_check.py --root DATA

1. No leak: the condition does not depend on the real map inside the target mask.
2. Unmasked behaviour unchanged: with an all-True mask the diffusion loss equals the loss with
   mask=None (same random draws), so change 1's code path is untouched.
3. Masked loss: changing the model's prediction outside the mask does not change the loss.
4. Sampling keeps every cell outside the mask equal to the real map.
5. Condition dropout replaces only the graph part (BEM); context channels survive.
6. One training step forward/backward on a real batch, with the condition size printed.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.nyc_dataset import NYCSceneGraphDataset          # noqa: E402
from datasets.scene_graph_dataset import collate_scene_graphs  # noqa: E402
from models.sg_to_bev import SGtoBEV                            # noqa: E402

FAILED = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    if not ok:
        FAILED.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = NYCSceneGraphDataset(a.root, split="train", max_samples=4, min_visible_cells=20,
                              numeric=True, graph="street")
    batch = collate_scene_graphs([ds[i] for i in range(2)])
    sg, bev, mask = batch["scene_graph"].to(dev), batch["bev"].to(dev), batch["mask"].to(dev)
    print(f"device {dev}; masked cells per window: {mask.flatten(1).sum(1).tolist()}")

    torch.manual_seed(0)
    model = SGtoBEV(num_classes=10, in_channels=33, hidden_channels=32, heads=4, bev_size=128,
                    num_timesteps=100, auxiliary_loss_weight=5e-4, uncond_prob=0.1,
                    edge_dim=2, context=True).to(dev)
    bem = torch.randn(2, 32, 128, 128, 1, device=dev)

    # 1. no leak
    bev2 = bev.clone()
    bev2[mask] = (bev2[mask] + 3) % 10
    c1 = model._with_context(bem, bev, mask)
    c2 = model._with_context(bem, bev2, mask)
    check("1 condition independent of the real map inside the mask", torch.equal(c1, c2))
    check("1 condition has 32 + 10 + 1 channels", c1.shape[1] == 43)

    # 2. all-True mask == no mask (same random draws)
    diff = model.diffusion
    diff.train(False)
    x = bev.unsqueeze(-1)
    cond = c1
    torch.manual_seed(1)
    l_none = diff(x, cond)
    torch.manual_seed(1)
    l_all = diff(x, cond, mask=torch.ones_like(mask).unsqueeze(-1))
    check(f"2 all-True mask loss {l_all.item():.6f} == no-mask loss {l_none.item():.6f}",
          torch.allclose(l_none, l_all, rtol=1e-5, atol=1e-6))

    # 3. predictions outside the mask do not affect the loss
    real_fn = diff.denoise_fn
    m5 = mask.unsqueeze(1).unsqueeze(-1)

    class Perturbed(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, xt, c, t):
            out = self.inner(xt, c, t)
            return torch.where(m5, out, out + 5.0 * torch.randn_like(out))
    torch.manual_seed(2)
    l_a = diff(x, cond, mask=mask.unsqueeze(-1))
    diff.denoise_fn = Perturbed(real_fn)
    torch.manual_seed(2)
    l_b = diff(x, cond, mask=mask.unsqueeze(-1))
    diff.denoise_fn = real_fn
    check(f"3 masked loss unchanged by outside predictions ({l_a.item():.6f} vs {l_b.item():.6f})",
          torch.allclose(l_a, l_b, rtol=1e-5, atol=1e-6))

    # 4. sampling keeps the outside real (few steps for speed)
    model.eval()
    out = model.sample(sg, num_steps=3, use_gt_positions=True, bev=bev, mask=mask)
    check("4 sampled map equals the real map outside the mask", torch.equal(out[~mask], bev[~mask]))
    check("4 sampled map has the right shape", tuple(out.shape) == tuple(bev.shape))

    # 5. dropout replaces only the BEM
    model.train()
    model.uncond_prob = 1.0
    gnn_out = model.gnn(sg)
    b_drop = model._bem(gnn_out, sg, sg.heatmaps)
    cd = model._with_context(b_drop, bev, mask)
    check("5 dropped condition: BEM part all ones", torch.all(cd[:, :32] == 1).item())
    check("5 dropped condition: context part intact", torch.equal(cd[:, 32:], c1[:, 32:]))
    model.uncond_prob = 0.1

    # 6. one training step
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = model(sg, bev, mode="joint", mask=mask)
    losses["total"].backward()
    opt.step()
    ok = all(torch.isfinite(v).item() for v in losses.values())
    check("6 forward/backward finite: " + ", ".join(f"{k} {v.item():.4f}" for k, v in losses.items()), ok)
    n = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n / 1e6:.2f} M")

    print("ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} CHECK(S) FAILED")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
