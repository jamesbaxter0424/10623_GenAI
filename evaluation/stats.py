"""Block-cluster bootstrap. Test windows overlap and share blocks (dataset audit D2), so we
resample target blocks with replacement and keep every row of a sampled block together
(Efron & Tibshirani 1993)."""
import numpy as np

import config as C


class ClusterBootstrap:
    def __init__(self, tables, key="block"):
        """tables: dict name -> DataFrame with a `key` column."""
        self.tables = tables
        self.units = np.unique(np.concatenate([t[key].to_numpy() for t in tables.values()]))
        pos = {u: i for i, u in enumerate(self.units)}
        self.groups = {}
        for name, t in tables.items():
            g = [[] for _ in self.units]
            for row, u in enumerate(t[key].to_numpy()):
                g[pos[u]].append(row)
            self.groups[name] = [np.array(x, np.int64) for x in g]

    def run(self, stat, n_boot=C.N_BOOT, seed=C.SEED, ci=C.CI):
        """stat(tables) -> dict of scalars. Returns {metric: (point, lo, hi)}."""
        point = stat(self.tables)
        rng = np.random.default_rng(seed)
        draws = {k: [] for k in point}
        for _ in range(n_boot):
            pick = rng.integers(0, len(self.units), len(self.units))
            res = {}
            for name, t in self.tables.items():
                idx = np.concatenate([self.groups[name][i] for i in pick])
                res[name] = t.iloc[idx]
            v = stat(res)
            for k in point:
                draws[k].append(v[k])
        a = (1 - ci) / 2
        out = {}
        for k, p in point.items():
            d = np.array(draws[k], float)
            d = d[np.isfinite(d)]
            lo, hi = (np.quantile(d, a), np.quantile(d, 1 - a)) if len(d) else (np.nan, np.nan)
            out[k] = (float(p), float(lo), float(hi))
        return out


def paired_difference(a, b, metric, n_boot=C.N_BOOT, seed=C.SEED, ci=C.CI):
    """Mean of (a - b) for a per-block metric over the blocks both models were scored on,
    with a block-cluster bootstrap interval. a, b: per_block DataFrames."""
    m = a[["sample_id", "block", metric]].merge(b[["sample_id", "block", metric]],
                                              on=["sample_id", "block"], suffixes=("_a", "_b")).dropna()
    m["diff"] = m[metric + "_a"] - m[metric + "_b"]
    boot = ClusterBootstrap({"m": m})
    r = boot.run(lambda t: {"diff": t["m"]["diff"].mean()}, n_boot, seed, ci)["diff"]
    return {"metric": metric, "n_blocks": int(m["block"].nunique()), "mean_diff": r[0], "lo": r[1], "hi": r[2],
            "significant": not (r[1] <= 0 <= r[2])}
