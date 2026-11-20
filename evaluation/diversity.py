"""Diversity: share of block cells that differ between two samples of the same input,
averaged over all pairs of samples (multi-sample pairwise distance as in BicycleGAN,
Zhu et al. 2017, with cell disagreement instead of LPIPS for class maps)."""
from itertools import combinations

import numpy as np


def block_diversity(gens_in_block):
    """gens_in_block: array [S, n_cells] of classes. NaN with fewer than two samples."""
    s = gens_in_block.shape[0]
    if s < 2:
        return np.nan
    return float(np.mean([(gens_in_block[a] != gens_in_block[b]).mean() for a, b in combinations(range(s), 2)]))
