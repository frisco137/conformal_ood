"""Every path this experiment reads or writes, in one place.

Layout:
    phase_2/src/       scripts and this module
    phase_2/results/   JSON measurement records
    phase_2/arrays/    .npz array stores (data splits, Jacobians, predictions)
    phase_2/logs/      run logs
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent
PHASE2 = SRC.parent
EXPERIMENTS = PHASE2.parent
REPO = EXPERIMENTS.parent

RESULTS = PHASE2 / "results"
ARRAYS = PHASE2 / "arrays"
LOGS = PHASE2 / "logs"

#: imported controls live here (wrap, TargetedImitator); added to sys.path by
#: the scripts that need them rather than at import time of this module.
TIER0 = EXPERIMENTS / "tier0_instrument"

for _d in (RESULTS, ARRAYS, LOGS):
    _d.mkdir(exist_ok=True)
