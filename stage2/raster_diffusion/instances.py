"""Generated maps -> buildings: seeds (far from their own wall), watershed, merge fragments, polygons."""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from common import config as C
from common.geometry import polygonize


def split(mask, dist, core, min_area):
    """Instance labels from the building mask and the distance-to-own-wall map."""
    seeds, n = ndi.label(mask & (dist >= core))
    lab = watershed(-dist, seeds, mask=mask) if n else np.zeros(mask.shape, np.int32)
    rest, _ = ndi.label(mask & (lab == 0))                   # parts without a seed (thin buildings)
    lab = np.where(rest > 0, rest + lab.max(), lab)
    area = np.bincount(lab.ravel()) * C.PX ** 2
    small = np.nonzero((area > 0) & (area < min_area))[0]
    small = small[small > 0]
    if len(small):
        keep = np.where(np.isin(lab, small), 0, lab)
        lab = watershed(np.zeros(mask.shape), keep, mask=mask & (lab > 0)) if keep.any() else keep
    return lab.astype(np.int32)                              # fragments with no neighbour disappear


def buildings(mask, dist, height, cfg):
    """(polygons, heights) of one block; maps must already be restricted to the block."""
    lab = split(mask, dist, cfg["core"], cfg["min_area"])
    g = polygonize(lab, C.PX, cfg["simplify"])
    ids = [q for q in g if g[q].area >= cfg["min_area"]]
    hs = ndi.median(height, lab, ids) if ids else []
    return [g[q] for q in ids], np.clip(np.asarray(hs, float), C.H_MIN, C.H_MAX)
