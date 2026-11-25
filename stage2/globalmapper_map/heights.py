"""Step 2: building heights. GlobalMapper predicts none, so a small network predicts the height of each
polygon from the map and the polygon. It is trained on REAL buildings and needs no GlobalMapper files.

  python -m globalmapper_map.heights --fit              # a few minutes on CPU: out/gm_heights.pt

Output is a normal distribution of log height. Buildings of the same class in one block share most of
the random part (correlation rho, measured on touching pairs of training buildings), because attached
houses usually have the same height.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from common import config as C
from common.geometry import obb, rasterize
from common.metrics import touching
from common.noise import corrupt
from common.records import load, polys

PATH = os.path.join(C.OUT, "gm_heights.pt")
NF = 3 + C.N_CLASSES + C.N_CLASSES


def features(g, cls4, inside):
    """[n, NF]: log area / width / length, class under the building, class shares of the block."""
    H, W = cls4.shape
    lab = rasterize(g, (H * C.F, W * C.F))
    px = np.repeat(np.repeat(cls4, C.F, 0), C.F, 1)
    cnt = np.zeros((len(g) + 1, C.N_CLASSES))
    np.add.at(cnt, (lab.ravel(), px.ravel()), 1)
    cnt = cnt[1:]
    cnt[:, :2] *= 1e-3                                       # street / open only if nothing else is below
    cls = cnt.argmax(1)
    o = np.array([obb(p)[2:4] for p in g]).reshape(-1, 2)
    area = np.array([p.area for p in g])
    share = np.bincount(cls4[inside], minlength=C.N_CLASSES) / max(inside.sum(), 1)
    f = np.concatenate([np.log(np.maximum(np.stack([area, o[:, 1], o[:, 0]], 1), 0.5)) - [4.5, 2.0, 2.7],
                        np.eye(C.N_CLASSES)[cls], np.repeat(share[None], len(g), 0)], 1)
    return f.astype(np.float32), cls


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(NF, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))

    def forward(self, f):
        m, s = self.n(f).unbind(-1)
        return m + 2.0, s.clamp(-4, 1)                       # mean and log std of log height


class Heights:
    def __init__(self, path=PATH):
        k = torch.load(path, map_location="cpu", weights_only=False)
        self.net, self.rho = Net(), k["rho"]
        self.net.load_state_dict(k["net"])
        self.net.eval()

    @torch.no_grad()
    def sample(self, g, cls4, inside, rng):
        if not g:
            return np.zeros(0)
        f, cls = features(g, cls4, inside)
        m, s = (q.numpy() for q in self.net(torch.from_numpy(f)))
        shared = rng.standard_normal(C.N_CLASSES)[cls]
        eps = np.sqrt(self.rho) * shared + np.sqrt(1 - self.rho) * rng.standard_normal(len(g))
        return np.clip(np.exp(m + np.exp(s) * eps), C.H_MIN, C.H_MAX)


def fit(data=C.DATA, path=PATH, blocks=6000, epochs=8, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    recs = load("train", data)
    recs = [recs[i] for i in rng.permutation(len(recs))[:blocks]]
    F, Y, pairs, n = [], [], [], 0
    for k, r in enumerate(recs):
        g = polys(r)
        if not g:
            continue
        f, cls = features(g, corrupt(r["cls4"], r["inside"], rng), r["inside"])
        F.append(f)
        Y.append(np.log(r["height"]))
        if k % 5 == 0:
            t = touching(g)
            t = t[cls[t[:, 0]] == cls[t[:, 1]]]
            pairs.append(t + n)
        n += len(g)
    F, Y = torch.from_numpy(np.concatenate(F)), torch.from_numpy(np.concatenate(Y))
    pairs = np.concatenate(pairs)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for e in range(epochs):
        tot = 0.0
        for i in torch.randperm(len(F)).split(1024):
            m, s = net(F[i])
            loss = (s + 0.5 * ((Y[i] - m) / s.exp()) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(i)
        print(f"epoch {e + 1}: nll {tot / len(F):.4f}", flush=True)
    with torch.no_grad():
        m, s = net(F)
        z = ((Y - m) / s.exp()).clamp(-3, 3).numpy()
    rho = float(np.clip(np.corrcoef(z[pairs[:, 0]], z[pairs[:, 1]])[0, 1], 0, 0.98))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"net": net.state_dict(), "rho": rho}, path)
    print(f"{len(F)} buildings, median abs error {float((m.exp() - Y.exp()).abs().median()):.2f} m, rho {rho:.2f} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", required=True)
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--out", default=PATH)
    a = ap.parse_args()
    fit(a.data, a.out)


if __name__ == "__main__":
    main()
