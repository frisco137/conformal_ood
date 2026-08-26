# What does a frozen tabular foundation model compute?

A discriminative mechanistic battery over TabPFN v2, TabICL v2, and TabSwift.
Research objective and question set: [`phase_2/plan.md`](phase_2/plan.md).

## Layout

```
models/         Model definitions, checkpoints, and the instrumented access layer.
                Start here: models/README.md
phase_2/        The experiments, their results, and the plan.
intermediate/   Everything else, archived. Gitignored. See intermediate/README.md
```

Nothing else lives at the root. `models/` and `phase_2/` are the project;
`intermediate/` is the history.

## Getting started

```bash
python -m models.interp.verify                       # 17 checks x 4 models
python -m models.interp.verify --task classification
```

```python
from models import load, summary_table
print(summary_table())

m = load("tabpfn_v2", task="regression", device="cuda")
trace = m.run(X_train, y_train, X_test)
trace.pred        # original y units, equals m.estimator.predict()
trace.resid[6]    # (n_tokens, d_model) residual stream at block 6's output
```

Read [`models/README.md`](models/README.md) before writing an experiment. It
states the conventions — units, token layout, layer indexing, residual-stream
definition, attention records — that every experiment is entitled to assume, and
lists the check that enforces each one.

## Status

| | |
|---|---|
| Model layer | Complete and verified: 4 models × 2 tasks, 17/17 checks each. |
| Experiments | Being rewritten against the new model layer. The code under `phase_2/` predates it. |

The experiment code in `phase_2/src` and `phase_2/experiments` still imports the
old wrappers and is **known broken** until it is ported. `phase_2/plan.md` and
`phase_2/results/` are the parts that carry forward.

## Environment

Use the system interpreter (`/usr/bin/python3`), not `.venv` — `tabpfn` and
`tabicl` are installed there. Dependencies are pinned in `pyproject.toml`.

Three checkpoint sources are gated upstream (HTTP 401) and cannot be
re-downloaded: TabSwift's weights (committed here as the only copy), and the
TabPFN v2.5/v2.6 classifiers (unavailable — see `models.registry.UNAVAILABLE`).
