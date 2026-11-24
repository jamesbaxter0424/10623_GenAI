"""Train the official GlobalMapper on NYC blocks.

  python train_nyc.py --data ~/genai-final/data/gm_nyc --gm ~/genai-final/third_party/GlobalMapper \
      --out ~/genai-final/checkpoints/globalmapper_nyc --max_hours 12 --max_epochs 250

Same as the release's train.py, except:
  - train / validation sets are our NYC train and val splits (the release samples 20% at random);
  - stops after --max_epochs or --max_hours. total_epochs stays 1000 as in train_gnn.yaml, so the
    learning-rate milestones are the release's 600 and 800 (scheduler stepped per batch, as in
    graph_trainer.py);
  - compatibility for current libraries only: networkx.read_gpickle -> pickle, np.float -> float,
    tensorboard_logger replaced by a CSV log;
  - num_workers 4 (the machine has 4 vCPUs); --max_samples for the overfit test.
Model, losses, loss weights, optimiser, batch size and data transforms are the release's.
"""
import argparse
import csv
import os
import pickle
import sys
import time
import types

import numpy as np

np.float = float                                     # graph_trainer.py uses np.float
_tb = types.ModuleType("tensorboard_logger")
_tb.configure = lambda *a, **k: None
_tb.log_value = lambda *a, **k: None
sys.modules["tensorboard_logger"] = _tb

import networkx as nx  # noqa: E402

nx.read_gpickle = lambda p: pickle.load(open(p, "rb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_hours", type=float, default=12.0)
    ap.add_argument("--max_epochs", type=int, default=250)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--val_dir", default="val")
    a = ap.parse_args()
    sys.path.insert(0, a.gm)
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import MultiStepLR
    from torch_geometric.loader import DataLoader

    from graph_trainer import train, validation
    from graph_util import read_train_yaml
    from model import AttentionBlockGenerator_independent_cnn
    import math

    import torch
    from PIL import Image
    from torch_geometric.data import Data

    import urban_dataset
    from urban_dataset import UrbanGraphDataset, get_transform, graph_transform

    class CachedUrbanGraphDataset(UrbanGraphDataset):
        """The release's UrbanGraphDataset.get, with the parsed graph cached in memory. The parts
        that are random per call (gaussian noise on the block mask, the random flip in
        graph_transform) are still applied on every call, exactly as in the release."""
        cache = {}
        _n = None

        def len(self):                               # the release re-scans the folder on every call
            if self._n is None:
                self._n = UrbanGraphDataset.len(self)
            return self._n

        def get(self, idx):
            if idx not in self.cache:
                g = pickle.load(open(os.path.join(self.processed_dir, f"{idx}.gpickle"), "rb"))
                self.cache[idx] = (g.graph["binary_mask"], g.graph["block_scale"],
                                   urban_dataset.graph2vector_processed(g))
            mask, block_scale, vec = self.cache[idx]
            block_mask = Image.fromarray(mask)
            trans_image = self.cnn_transforms(block_mask)
            scale_channel = torch.tensor([math.log(block_scale, 20) / 2.0]).expand(trans_image.shape)
            block_condition = torch.cat((trans_image, scale_channel), 0)
            node_size, node_pos, node_feature, edge_index, node_idx, asp_rto, long_side, b_shape, b_iou = vec
            return Data(x=node_feature, node_pos=node_pos, edge_index=edge_index, node_size=node_size,
                        node_idx=node_idx, asp_rto=asp_rto, long_side=long_side, b_shape=b_shape, b_iou=b_iou,
                        blockshape_latent=np.zeros(40), block_scale=block_scale, block_condition=block_condition,
                        org_binary_mask=self.base_transform(block_mask))

        def preload(self):
            for i in range(self.len()):
                self.get(i)

    opt = read_train_yaml(a.gm, filename="train_gnn.yaml")
    N = opt["template_width"] * opt["template_height"]
    opt["N"] = int(N)
    opt["save_record"] = False          # logging below; checkpoints saved here
    device = torch.device("cuda:0")
    opt["device"] = str(device)
    random_seed = 42
    torch.manual_seed(random_seed)

    loss_dict = {
        "Posloss": nn.MSELoss(reduction="sum"), "ShapeCEloss": nn.CrossEntropyLoss(reduction="sum"),
        "Iouloss": nn.MSELoss(reduction="sum"), "ExistBCEloss": nn.BCEWithLogitsLoss(reduction="sum"),
        "CELoss": nn.CrossEntropyLoss(reduction="none"), "Sizeloss": nn.MSELoss(reduction="sum"),
        "ExtSumloss": nn.MSELoss(reduction="sum")}

    cnn_transform = get_transform(noise_range=10.0, noise_type="gaussian", isaug=False, rescale_size=64)
    tr = CachedUrbanGraphDataset(os.path.join(a.data, "train"), transform=graph_transform, cnn_transform=cnn_transform)
    CachedUrbanGraphDataset.cache = {}
    tr.preload()
    tr_cache = CachedUrbanGraphDataset.cache
    va = CachedUrbanGraphDataset(os.path.join(a.data, a.val_dir), transform=graph_transform, cnn_transform=cnn_transform)
    CachedUrbanGraphDataset.cache = {}
    va.preload()
    va_cache = CachedUrbanGraphDataset.cache

    def bind(ds, c):                                 # each dataset keeps its own cache
        ds.cache = c
        return ds
    tr, va = bind(tr, tr_cache), bind(va, va_cache)
    print("cached", len(tr_cache), "training and", len(va_cache), "validation graphs", flush=True)
    if a.max_samples:
        tr = tr[np.arange(a.max_samples)]
        va = tr
    opt["num_data"] = len(tr)
    print(f"train {len(tr)} graphs, validation {len(va)} graphs", flush=True)
    bs = min(opt["batch_size"], len(tr))
    train_loader = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=4, drop_last=False,
                              persistent_workers=True)
    val_loader = DataLoader(va, batch_size=bs, shuffle=False, num_workers=4, persistent_workers=True)

    model = AttentionBlockGenerator_independent_cnn(opt, N=N).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(opt["lr"]), weight_decay=1e-6)
    start = opt["start_epoch"]
    scheduler = MultiStepLR(optimizer, milestones=[(opt["total_epochs"] - start) * 0.6,
                                                   (opt["total_epochs"] - start) * 0.8], gamma=0.3)
    os.makedirs(a.out, exist_ok=True)
    log_f = open(os.path.join(a.out, "training_history.csv"), "a", newline="")
    log = csv.writer(log_f)
    log.writerow(["epoch", "hours", "train_loss", "train_acc", "val_loss", "val_acc", "val_geo_loss", "lr"])
    best = {"val_loss": None, "val_geo": None}
    t0 = time.time()
    for epoch in range(start, start + a.max_epochs):
        t_acc, t_loss = train(model, epoch, train_loader, device, opt, loss_dict, optimizer, scheduler)
        v_acc, v_loss, v_geo = validation(model, epoch, val_loader, device, opt, loss_dict, scheduler)
        hours = (time.time() - t0) / 3600
        lr = optimizer.param_groups[0]["lr"]
        log.writerow([epoch, round(hours, 4), float(t_loss), float(t_acc), float(v_loss), float(v_acc), float(v_geo), lr])
        log_f.flush()
        print(f"Epoch {epoch:03d} {hours:.2f} h | train loss {t_loss:.1f} acc {float(t_acc):.4f} | "
              f"val loss {v_loss:.1f} acc {float(v_acc):.4f} geo {v_geo:.1f} | lr {lr:.2e}", flush=True)
        if best["val_loss"] is None or v_loss <= best["val_loss"]:
            best["val_loss"] = v_loss
            torch.save(model.state_dict(), os.path.join(a.out, "val_least_loss_all.pth"))
        if best["val_geo"] is None or v_geo <= best["val_geo"]:
            best["val_geo"] = v_geo
            torch.save(model.state_dict(), os.path.join(a.out, "val_least_loss_geo.pth"))
        torch.save(model.state_dict(), os.path.join(a.out, "latest.pth"))
        if hours >= a.max_hours:
            print("time budget reached", flush=True)
            break
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
