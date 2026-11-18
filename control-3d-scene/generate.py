import os

import numpy as np
import torch
from tqdm import tqdm

from utils.visualize import load_color_map, save_ply

class Generator:
    def __init__(self, cfg, model, loader):
        self.cfg = cfg
        self.module = model.module if hasattr(model, "module") else model
        self.loader = loader
        self.device = torch.device("cuda", cfg.get("local_rank", 0))
        self.num_steps = cfg.get("num_steps", None)
        self.out_dir = os.path.join(cfg.log_path, "Generated")
        self.ply_dir = os.path.join(cfg.log_path, "GeneratedPly")
        os.makedirs(self.out_dir, exist_ok=True)
        if cfg.get("save_ply", True):
            os.makedirs(self.ply_dir, exist_ok=True)
        self.colors = load_color_map() if cfg.get("dataset", "carla") != "nyc" else None

    @torch.no_grad()
    def run_nyc(self):
        """Per input window: `samples_per_input` generated maps + ground truth + target mask
        -> Generated/<sample_id>.npz (gen [S,128,128], gt, mask), and PNGs for the first few."""
        from utils.nyc_vis import save_sample_figure
        S = self.cfg.get("samples_per_input", 4)
        use_gt = self.cfg.get("use_gt_positions", False)
        n_png = self.cfg.get("png_n", 16)
        png_dir = os.path.join(self.cfg.log_path, "png")
        os.makedirs(png_dir, exist_ok=True)
        done = 0
        for batch in tqdm(self.loader, desc="generate[nyc]"):
            sg = batch["scene_graph"].to(self.device)
            gens = [self.module.sample(sg, num_steps=self.num_steps, use_gt_positions=use_gt)
                    .cpu().numpy().astype(np.uint8) for _ in range(S)]
            gens = np.stack(gens, 1)                       # [B, S, H, W]
            gt = batch["bev"].numpy().astype(np.uint8)
            mask = batch["mask"].numpy()
            for i, sid in enumerate(batch["file_path"]):
                np.savez_compressed(os.path.join(self.out_dir, f"{sid}.npz"),
                                    gen=gens[i], gt=gt[i], mask=mask[i])
                if done < n_png:
                    acc = (gens[i][:, mask[i]] == gt[i][mask[i]]).mean()
                    save_sample_figure(os.path.join(png_dir, f"{sid}.png"), gt[i], mask[i],
                                       gens[i], title=f"sample {sid}  |  cell acc. in mask {acc:.2f}")
                done += 1
        print(f"Generated {done} inputs x {S} samples -> {self.out_dir}")

    @torch.no_grad()
    def run(self):
        if self.cfg.get("dataset", "carla") == "nyc":
            self.module.eval()
            return self.run_nyc()
        self.module.eval()
        produced = 0
        limit = self.cfg.get("generation_num", 0)
        for batch in tqdm(self.loader, desc=f"generate[{self.cfg.stage}]"):
            if self.cfg.stage == "sg_to_bev":
                sg = batch["scene_graph"].to(self.device)
                out = self.module.sample(sg, num_steps=self.num_steps)
            else:
                bev = batch["bev"].to(self.device)
                out = self.module.sample(bev, num_steps=self.num_steps)
            out = out.cpu().numpy().astype(np.uint8)

            for i in range(out.shape[0]):
                tag = f"{produced:05d}"
                np.save(os.path.join(self.out_dir, f"{tag}.npy"), out[i])
                if self.cfg.get("save_ply", True):
                    save_ply(out[i], os.path.join(self.ply_dir, f"{tag}.ply"), self.colors)
                produced += 1
                if limit and produced >= limit:
                    print(f"Generated {produced} samples -> {self.out_dir}")
                    return
        print(f"Generated {produced} samples -> {self.out_dir}")
