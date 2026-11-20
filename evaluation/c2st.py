"""CNN two-sample test of block realism (Lopez-Paz & Oquab, ICLR 2017).

For every test target block we crop its bounding box from the real map and from one generated
sample, set everything outside the block to empty, and resize to 64x64 (nearest). A small CNN
is trained to tell real from generated on half of the blocks and tested on the other half; the
real and generated crops of a block are always in the same half. Accuracy near 0.5 means the
generated blocks cannot be told apart from real ones. Repeated over several random splits.

  python c2st.py --pred_dir PATH --name liu_baseline     (needs torch: conda env genai)
  python c2st.py --oracle --name oracle                  (fake = real copy, expect ~0.5)
  python c2st.py --shuffle --name shuffle_check          (cells shuffled, expect >> 0.5)
  python c2st.py --pred_dir PATH --name citydreamer --collapse_use
      (models without land use: residential and non-residential merged, and street and open
       merged inside the block, in both real and generated crops)
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

import config as C
from data import City, test_order
from evaluate import load_gens


def collapse_use(c):
    """Merge non-residential into residential height classes and street into open."""
    c = c.copy()
    c[c >= 6] -= 4
    c[c < 2] = C.OPEN
    return c


def crop(cls_window, inside, size=C.C2ST_SIZE):
    rr, cc = np.nonzero(inside)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    side = max(r1 - r0, c1 - c0)
    r0 -= (side - (r1 - r0)) // 2
    c0 -= (side - (c1 - c0)) // 2
    idx = r0 + (np.arange(size) * side / size).astype(int)
    jdx = c0 + (np.arange(size) * side / size).astype(int)
    valid_r = (idx >= 0) & (idx < cls_window.shape[0])
    valid_c = (jdx >= 0) & (jdx < cls_window.shape[1])
    ii, jj = np.clip(idx, 0, cls_window.shape[0] - 1), np.clip(jdx, 0, cls_window.shape[1] - 1)
    m = inside[np.ix_(ii, jj)] & valid_r[:, None] & valid_c[None, :]
    c = cls_window[np.ix_(ii, jj)]
    x = np.zeros((C.N_CLASSES + 1, size, size), np.float32)
    for k in range(C.N_CLASSES):
        x[k] = (c == k) & m
    x[C.N_CLASSES] = m
    return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [C.N_CLASSES + 1, 32, 64, 64, 128]
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
        self.f = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))

    def forward(self, x):
        return self.f(x).squeeze(1)


def one_split(X, y, unit, seed, device, epochs=C.C2ST_EPOCHS):
    rng = np.random.default_rng(seed)
    units = np.unique(unit)
    train_units = set(rng.permutation(units)[: len(units) // 2])
    tr = np.array([u in train_units for u in unit])
    torch.manual_seed(seed)
    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt, yt = torch.tensor(X[tr]), torch.tensor(y[tr], dtype=torch.float32)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xt[b].to(device)), yt[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.cat([net(torch.tensor(X[~tr][i:i + 512]).to(device)).cpu() for i in range(0, (~tr).sum(), 512)])
    return float(((p > 0).numpy() == (y[~tr] > 0.5)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=C.N_TEST)
    ap.add_argument("--splits", type=int, default=C.C2ST_SPLITS)
    ap.add_argument("--collapse_use", action="store_true",
                    help="merge land uses and street/open in real and generated crops")
    a = ap.parse_args()

    city = City()
    order = test_order(city)[:a.n]
    files = load_gens(a.pred_dir) if a.pred_dir else None
    rng = np.random.default_rng(C.SEED)
    X, y, unit = [], [], []
    for sid in order:
        if files is not None and sid not in files:
            continue
        cls, blk, _, tgts = city.window(sid)
        if files is not None:
            g = np.load(files[sid])["gen"]
            g = (g[None] if g.ndim == 2 else g).astype(np.int64)
            fake = g[rng.integers(g.shape[0])]
        elif a.shuffle:
            fake = cls.copy()
            for t in tgts:
                m = blk == t
                fake[m] = rng.permutation(fake[m])
        else:
            fake = cls
        if a.collapse_use:
            cls, fake = collapse_use(cls), collapse_use(fake)
        for t in tgts:
            inside = blk == t
            X += [crop(cls, inside), crop(fake, inside)]
            y += [1, 0]
            unit += [t, t]
    X, y, unit = np.stack(X), np.array(y), np.array(unit)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accs = [one_split(X, y, unit, s, device) for s in range(a.splits)]
    res = {"acc_mean": float(np.mean(accs)), "acc_min": float(np.min(accs)), "acc_max": float(np.max(accs)),
           "accs": accs, "n_blocks": int(len(np.unique(unit))), "n_crops": int(len(y)), "n_splits": a.splits,
           "device": device, "collapse_use": a.collapse_use}
    out = os.path.join(C.RESULTS_DIR, a.name)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "c2st.json"), "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
