"""Row-count funnel log shared by all stages: data/interim/funnel.csv"""
import csv
from config import INTERIM

PATH = INTERIM / "funnel.csv"
FIELDS = ["pilot", "stage", "step", "n_in", "n_out", "dropped", "note"]


def reset(stage, areas=None):
    """Remove earlier rows of this stage so re-runs don't duplicate."""
    if not PATH.exists():
        return
    with open(PATH, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if not (r["stage"] == stage and (areas is None or r["pilot"] in areas))]
    _write(rows)


def log(pilot, stage, step, n_in, n_out, note=""):
    rows = []
    if PATH.exists():
        with open(PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows.append(dict(pilot=pilot, stage=stage, step=step, n_in=n_in, n_out=n_out,
                     dropped=n_in - n_out, note=note))
    _write(rows)
    pct = 100 * (n_in - n_out) / n_in if n_in else 0
    print(f"  [{pilot}] {stage}/{step}: {n_in} -> {n_out} (-{pct:.1f}%) {note}")


def _write(rows):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
