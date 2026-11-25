"""Settings shared by every stage-2 experiment. Change values here, nowhere else."""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))        # stage2/
REPO = os.path.dirname(ROOT)
CITY = os.environ.get("NYC_DATA", os.path.join(REPO, "nyc-massing-data", "data", "processed", "city"))
BLDG = os.environ.get("NYC_BLDG", os.path.join(REPO, "nyc-massing-data", "data", "interim"))
DATA = os.environ.get("STAGE2_DATA", os.path.join(ROOT, "data"))          # built by make_data.py
OUT = os.environ.get("STAGE2_OUT", os.path.join(ROOT, "out"))             # checkpoints, predictions
RESULTS = os.path.join(ROOT, "results")                                   # score tables (in git)

CELL = 4.0            # stage-1 map cell (m)
PX = 1.0              # stage-2 pixel (m)
F = int(CELL / PX)    # pixels per cell
MARGIN = 4            # cells of context around the block
N_CLASSES = 10        # 0 street, 1 open, 2-5 residential 1-2/3-5/6-12/13+ floors, 6-9 non-residential
MIN_BLDG_M2 = 10.0    # smaller footprints are dropped
H_MIN, H_MAX = 2.0, 450.0
STREET_CLIP_M = 40.0  # distance-to-street channel is clipped here

N_PILOT, N_VAL, N_TEST = 2000, 200, 1000
SEED = 0


def hclass_of(cls):
    """Class 2..9 -> height class 0..3 (also works on arrays); -1 for street / open."""
    c = np.asarray(cls).astype(np.int64)
    return np.where(c >= 2, (c - 2) % 4, -1)
