"""Surroundings metrics over pairs (target block, real block facing it across a street).

Boundary:    d = mean floors of target - mean floors of neighbour; the Wasserstein distance
             between generated and real distributions of d (GlobalMapper, COHO use WD to
             compare generated and real distributions; NYC ZR 23-443 regulates height
             transitions between adjacent districts).
Street wall: built share of the target's street-edge cells (Built Front, 4 m from the street
             edge); WD between generated and real distributions, and WD of the difference to
             the neighbour's street wall (NYC ZR 23-431 aligns new street walls with neighbours).

Note: a paired error |d_gen - d_real| is not reported because the neighbour term cancels;
it equals the height error of the target itself.
"""
import numpy as np

from data import street_wall


def mean_floors(g, f):
    bld = g >= 2
    return float(f[g[bld]].mean()) if bld.any() else 0.0


def neighbours(city, b, targets):
    """Real blocks facing block b across a street, excluding the window's other targets."""
    return [n for n in city.neighbors[b] if n not in targets]


def window_street(blk, water):
    """Street cells of a window (block id < 0 and not water)."""
    return (blk < 0) & ~water


def block_context(gen_window, inside, street, f):
    g = gen_window[inside]
    return {"H": mean_floors(g, f), "sw": street_wall(gen_window, inside, street)}
