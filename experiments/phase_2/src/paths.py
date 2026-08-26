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

#: Phase 1's instrument directory. Phase 2 reads two of its stored arrays
#: (exp3_reduced_jacobians.npz, exp_negeigvec.json) to reproduce the audit's
#: localisation in exp 2.2. The control MAPS now come from
#: experiments.phase_1.core.controls, imported normally.
TIER0 = EXPERIMENTS / "phase_1" / "tier0_instrument"

for _d in (RESULTS, ARRAYS, LOGS):
    _d.mkdir(exist_ok=True)
