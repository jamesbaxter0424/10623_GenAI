import glob
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.distributed import is_main_process

class Trainer:
    def __init__(self, cfg, model, optimizer, scheduler_iter, scheduler_epoch,
                 loader, sampler):
        self.cfg = cfg
        self.model = model
        self.module = model.module if hasattr(model, "module") else model
        self.optimizer = optimizer
        self.scheduler_iter = scheduler_iter
        self.scheduler_epoch = scheduler_epoch
        self.loader = loader
        self.sampler = sampler
        self.device = torch.device("cuda", cfg.get("local_rank", 0))
        self.start_epoch = 0

        self.log_path = cfg.log_path
        self.writer = None
        if is_main_process():
            os.makedirs(self.log_path, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.log_path, "tb"))

        if cfg.get("resume_optimizer", False) and cfg.get("resume_path", ""):
            ckpt = torch.load(cfg.resume_path, map_location="cpu")
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_epoch = ckpt.get("epoch", 0)
            if is_main_process():
                print(f"Resumed optimizer/epoch from {cfg.resume_path} "
                      f"(start epoch {self.start_epoch})")

    def _step(self, batch):
        if self.cfg.stage == "sg_to_bev":
            sg = batch["scene_graph"].to(self.device)
            if self.cfg.mode == "loc_posttrain":
                return self.model(sg, mode="loc")
            bev = batch["bev"].to(self.device)
            if self.cfg.get("context", False):   # change 3: target-block mask
                return self.model(sg, bev, mode="joint", mask=batch["mask"].to(self.device))
            return self.model(sg, bev, mode="joint")
        bev = batch["bev"].to(self.device)
        scene = batch["scene_3d"].to(self.device)
        return self.model(bev, scene)

    def _out_of_time(self):
        hours = self.cfg.get("max_hours", None)
        return bool(hours) and (time.time() - self.t0) / 3600 >= float(hours)

    def train(self):
        self.t0 = time.time()
        print_every = self.cfg.get("print_every", 20)
        line_end = "\n" if self.cfg.get("log_newline", False) else "\r"
        epoch = self.start_epoch
        # whole_epochs (change 3, Slurm segments): never cut an epoch; stop before an epoch that
        # would not finish within max_hours, and save latest.tar after every epoch.
        whole = self.cfg.get("whole_epochs", False)
        last_epoch_h = 0.0
        for epoch in range(self.start_epoch, self.cfg.epochs):
            el_h = (time.time() - self.t0) / 3600
            if self._out_of_time() or (whole and self.cfg.get("max_hours", None)
                                       and el_h + 1.1 * last_epoch_h >= float(self.cfg.max_hours)):
                print(f"\nTime budget of {self.cfg.max_hours} h reached before epoch {epoch + 1}.")
                epoch -= 1  # latest.tar records the last finished epoch
                break
            t_epoch = time.time()
            self.model.train()
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            running, window, count = {}, {}, 0
            max_iters = self.cfg.get("max_iters", 0)
            for batch in self.loader:
                if max_iters and count >= max_iters:
                    break
                self.optimizer.zero_grad()
                losses = self._step(batch)
                losses["total"].backward()
                if self.cfg.get("clip_norm", None):
                    # norm before clipping, logged to spot instabilities
                    losses["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_norm)
                if self.cfg.get("clip_value", None):
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.cfg.clip_value)
                self.optimizer.step()
                if self.scheduler_iter:
                    self.scheduler_iter.step()

                count += 1
                for k, v in losses.items():
                    running[k] = running.get(k, 0.0) + float(v.detach())
                    window[k] = window.get(k, 0.0) + float(v.detach())
                if is_main_process() and count % print_every == 0:
                    # mean over the last `print_every` steps (not the epoch so far)
                    msg = ", ".join(f"{k}: {window[k] / print_every:.4f}" for k in window)
                    window = {}
                    el = (time.time() - self.t0) / 3600
                    print(f"[{self.cfg.stage}/{self.cfg.mode}] epoch {epoch + 1}/{self.cfg.epochs} "
                          f"iter {count}/{len(self.loader)} | {msg} | {el:.2f} h", end=line_end,
                          flush=True)
                if not whole and self._out_of_time():
                    break

            if self.scheduler_epoch:
                self.scheduler_epoch.step()
            if is_main_process():
                avg = {k: running[k] / max(count, 1) for k in running}
                print(f"\n[{self.cfg.stage}/{self.cfg.mode}] epoch {epoch + 1} done | "
                      + ", ".join(f"{k}: {v:.4f}" for k, v in avg.items()))
                for k, v in avg.items():
                    self.writer.add_scalar(f"train/{k}", v, epoch + 1)
                if (epoch + 1) % self.cfg.check_every == 0:
                    self._save_checkpoint(epoch + 1)
                elif whole:
                    self._save_checkpoint(epoch + 1, name="latest.tar")
            last_epoch_h = (time.time() - t_epoch) / 3600
        if is_main_process():
            self._save_checkpoint(epoch + 1, name="latest.tar")

    def _save_checkpoint(self, epoch, name=None):
        ckpt = {"epoch": epoch, "model": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict()}
        path = os.path.join(self.log_path, name or f"epoch{epoch}.tar")
        torch.save(ckpt, path)
        if name != "latest.tar":
            torch.save(ckpt, os.path.join(self.log_path, "latest.tar"))
        print(f"Saved checkpoint: {path}", flush=True)
        keep = self.cfg.get("keep_ckpts", 0)
        if keep:  # each checkpoint is ~0.7 GB: keep only the newest `keep` epoch files
            olds = sorted(glob.glob(os.path.join(self.log_path, "epoch*.tar")),
                          key=os.path.getmtime)
            for p in olds[:-keep]:
                os.remove(p)
