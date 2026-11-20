"""Settings shared by every metric. Change values here, not in the metric files."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get(
    "NYC_DATA", os.path.join(HERE, "..", "nyc-massing-data", "data", "processed", "city"))
CACHE_DIR = os.path.join(HERE, "cache")
RESULTS_DIR = os.path.join(HERE, "results")

N = 128                    # window size in cells
CELL_M = 4.0               # cell size in metres
CELL_M2 = CELL_M * CELL_M

# map classes: 0 street, 1 open, 2-5 residential 1-2/3-5/6-12/13+ floors, 6-9 non-residential
STREET, OPEN = 0, 1
RES_CLASSES = (2, 3, 4, 5)
NONRES_CLASSES = (6, 7, 8, 9)
N_CLASSES = 10

# test set
TEST_SEED = 0              # same fixed ordering as control-3d-scene/tools/nyc_tools.py subset
N_TEST = 1000              # first N windows of the ordering

# condition metrics
HIT_REL_TOL = 0.10         # hit rate: within +-10% of the target ...
HIT_ABS_TOL = 0.05         # ... or within +-0.05 for small targets

# statistics
N_BOOT = 2000
CI = 0.95
SEED = 0

# CNN two-sample test
C2ST_SIZE = 64
C2ST_SPLITS = 5
C2ST_EPOCHS = 15
