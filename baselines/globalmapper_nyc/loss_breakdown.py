"""Per-term loss of the NYC-trained GlobalMapper on the validation split, computed exactly as the
release's graph_trainer.validation (weights from train_gnn.yaml, batch 200), plus latent statistics
and building counts (real vs recon vs random vs N(0, I)).

  python loss_breakdown.py --data ~/genai-final/data/gm_nyc --gm ~/genai-final/third_party/GlobalMapper \
      --ckpt ~/genai-final/checkpoints/globalmapper_nyc/val_least_loss_all.pth --out loss_breakdown.json
"""
import argparse
import json
import os
import pickle
import sys
import types

import numpy as np

np.float = float
_tb = types.ModuleType("tensorboard_logger")
_tb.configure = lambda *a, **k: None
_tb.log_value = lambda *a, **k: None
sys.modules["tensorboard_logger"] = _tb
import networkx as nx  # noqa: E402

nx.read_gpickle = lambda p: pickle.load(open(p, "rb"))


def main():
    ap = argparse.ArgumentParser()
    for k in ["data", "gm", "ckpt", "out"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--transform", choices=["test", "train"], default="test")
    a = ap.parse_args()
    sys.path.insert(0, a.gm)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    from graph_util import read_train_yaml
    from model import AttentionBlockGenerator_independent_cnn
    from urban_dataset import UrbanGraphDataset, get_transform, graph_transform, test_graph_transform

    torch.manual_seed(0)
    opt = read_train_yaml(a.gm, filename="train_gnn.yaml")
    N = opt["template_width"] * opt["template_height"]
    opt["N"] = N
    dev = torch.device("cuda:0")
    opt["device"] = str(dev)
    model = AttentionBlockGenerator_independent_cnn(opt, N=N)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.to(dev).eval()

    class Fixed(UrbanGraphDataset):
        _n = None

        def len(self):
            if self._n is None:
                self._n = UrbanGraphDataset.len(self)
            return self._n
    cnn_t = get_transform(noise_range=10.0, noise_type="gaussian", isaug=False, rescale_size=64)
    ds = Fixed(os.path.join(a.data, "val"), transform=test_graph_transform if a.transform == "test" else graph_transform,
               cnn_transform=cnn_t)
    loader = DataLoader(ds, batch_size=opt["batch_size"], shuffle=False, num_workers=4)

    L = {"Posloss": nn.MSELoss(reduction="sum"), "ShapeCEloss": nn.CrossEntropyLoss(reduction="sum"),
         "Iouloss": nn.MSELoss(reduction="sum"), "ExistBCEloss": nn.BCEWithLogitsLoss(reduction="sum"),
         "Sizeloss": nn.MSELoss(reduction="sum"), "ExtSumloss": nn.MSELoss(reduction="sum")}
    wt = {"exist": opt["exist_weight"], "pos": opt["pos_weight"], "size": opt["size_weight"],
          "iou": opt["iou_weight"], "shape": opt["shape_weight"], "extsum": opt["extsum_weight"],
          "kld": opt["kld_weight"]}
    raw = {k: [] for k in wt}
    nblk = []
    mus, lvs = [], []
    cnt = {"real": [], "recon": [], "random_valstats": [], "standard_normal": []}
    batches = []
    with torch.no_grad():
        for data in loader:
            data = data.to(dev)
            B = data.ptr.shape[0] - 1
            exist, pos, size, mu, log_var, b_shape, b_iou = model(data)
            exist_gt = data.x[:, 0].unsqueeze(1)
            raw["exist"].append(L["ExistBCEloss"](exist, exist_gt).item())
            raw["pos"].append(L["Posloss"](pos, data.org_node_pos).item())
            raw["size"].append(L["Sizeloss"](size, data.org_node_size).item())
            raw["kld"].append(torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0).item())
            exist_out = torch.ge(exist, 0.5).type(torch.uint8)
            raw["extsum"].append(L["ExtSumloss"](torch.sum(exist_out), torch.sum(exist_gt)).item())
            raw["shape"].append(L["ShapeCEloss"](b_shape, data.b_shape_gt).item())
            raw["iou"].append(L["Iouloss"](b_iou, data.b_iou).item())
            nblk.append(B)
            mus.append(mu.cpu())
            lvs.append(log_var.cpu())
            cnt["real"].append(exist_gt.view(B, N).sum(1).cpu())
            cnt["recon"].append(torch.ge(exist, 0.5).view(B, N).sum(1).cpu())
            batches.append(data.cpu())
    mu = torch.cat(mus)
    lv = torch.cat(lvs)
    # latent statistics over validation (as gen_nyc.py: z = reparameterize(mu, log_var))
    z = model.reparameterize(mu, lv)
    z_mean, z_std = z.mean(0), z.std(0)
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        for data in batches:
            data = data.to(dev)
            B = data.ptr.shape[0] - 1
            cond = model.cnn_encode(data.block_condition.view(B, 2, 64, 64))
            zr = (z_mean + z_std * torch.randn(B, mu.shape[1], generator=g)).to(dev)
            ex = model.decode(zr, cond, data.edge_index)[0]
            cnt["random_valstats"].append(torch.ge(ex, 0.5).view(B, N).sum(1).cpu())
            z0 = torch.randn(B, mu.shape[1], generator=g).to(dev)
            ex = model.decode(z0, cond, data.edge_index)[0]
            cnt["standard_normal"].append(torch.ge(ex, 0.5).view(B, N).sum(1).cpu())

    # extsum gradient check on one real batch with grad enabled
    model.train(False)
    data = batches[0].to(dev)
    exist = model(data)[0]
    exist_out = torch.ge(exist, 0.5).type(torch.uint8)
    ext = L["ExtSumloss"](torch.sum(exist_out), torch.sum(data.x[:, 0].unsqueeze(1)))
    ext_has_grad = bool(ext.requires_grad)
    bce = L["ExistBCEloss"](exist, data.x[:, 0].unsqueeze(1))
    bce_has_grad = bool(bce.requires_grad)

    nb = np.array(nblk, float)
    res = {"transform": a.transform, "n_blocks": int(nb.sum()), "n_batches": len(nb), "weights": wt,
           "raw_per_batch": {k: float(np.mean(v)) for k, v in raw.items()},
           "weighted_per_batch": {k: float(np.mean(v)) * wt[k] for k, v in raw.items()}}
    tot = sum(res["weighted_per_batch"].values())
    res["total_per_batch"] = tot
    res["share"] = {k: v / tot for k, v in res["weighted_per_batch"].items()}
    # per block: summed terms divided by blocks in the batch; KL is already a mean over blocks
    per_blk = {k: float(np.sum(np.array(v)) / nb.sum()) * wt[k] for k, v in raw.items() if k != "kld"}
    per_blk["kld"] = float(np.average(raw["kld"], weights=nb)) * wt["kld"]
    res["weighted_per_block"] = per_blk
    geo = per_blk["pos"] + per_blk["size"]
    res["geometry_per_block_weighted"] = geo
    res["geometry_over_kld_per_block"] = geo / per_blk["kld"]
    res["kld_effective_weight_vs_summed_terms"] = wt["kld"] / float(nb.mean())
    res["extsum_has_grad"] = ext_has_grad
    res["exist_bce_has_grad"] = bce_has_grad
    res["latent"] = {"dim": int(mu.shape[1]),
                     "mean_abs_mu": float(mu.abs().mean()),
                     "mean_over_dims_of_std_of_mu": float(mu.std(0).mean()),
                     "mean_log_var": float(lv.mean()),
                     "mean_posterior_std": float(torch.exp(0.5 * lv).mean()),
                     "mean_over_dims_of_std_of_z": float(z_std.mean()),
                     "mean_abs_z_mean": float(z_mean.abs().mean()),
                     "kld_per_dim": float(np.average(raw["kld"], weights=nb)) / int(mu.shape[1])}
    res["buildings_per_block"] = {k: float(torch.cat(v).float().mean()) for k, v in cnt.items()}
    res["buildings_per_block_median"] = {k: float(torch.cat(v).float().median()) for k, v in cnt.items()}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
