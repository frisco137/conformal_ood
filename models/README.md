# `models/` — model definitions, checkpoints, and instrumented access

Everything the project needs to load a frozen tabular foundation model and look
inside it. Two entry points and nothing else:

```python
from models import load, SPECS

m = load("tabpfn_v2", task="regression", device="cuda")
trace = m.run(X_train, y_train, X_test)

trace.pred          # (n_test,) in the ORIGINAL units of y — equals m.estimator.predict()
trace.resid[6]      # (n_tokens, d_model) residual stream at the output of block 6
trace.layout        # where the support rows and query rows are in that sequence
```

`SPECS` is the single source of truth for architecture facts and capabilities.
**Nothing anywhere else may hardcode a layer count, a width, a token offset, or
a checkpoint name** — import it from `models.registry`.

---

## 1. The roster

Every number below was read off a loaded checkpoint, and
`models.interp.verify` re-checks it at runtime.

| id | blocks | d_model | prefix tokens | state | classifier | predictive distribution |
|---|---|---|---|---|---|---|
| `tabpfn_v2` | 12 | 192 | 0 | per-**cell** | yes | yes (bar distribution) |
| `tabicl_v2` | 12 | 512 | 0 | per-row | yes | yes (999 quantiles) |
| `tabswift` | 24 | 192 | **64** (registers) | per-row | yes | **no** (point estimate) |
| `tabpfn_v2_5` | 18 | 192 | **64** (thinking tokens) | per-**cell** | no (gated) | yes |

`ROSTER` is the headline three. `tabpfn_v2_5` exists only as the *within-family
scale bar*: it answers "how different do two checkpoints of the same
architecture look?", which is the calibration any cross-model distance claim
needs.

### Where the weights come from

| id | code | weights |
|---|---|---|
| `tabpfn_v2`, `tabpfn_v2_5` | pip `tabpfn>=7.1.0`, unmodified | HF `Prior-Labs/*`, local cache |
| `tabicl_v2` | pip `tabicl==2.1.1`, unmodified | HF `jingang/TabICL` |
| `tabswift` | `models/vendor/tabswift` @ upstream `8edf8f0b` (5 documented patches) | `models/checkpoints/tabswift/swift.ckpt`, **committed**, byte-identical to HF `LAMDA-Tabular/TabSwift` |

Nothing is forked. TabPFN and TabICL are installed packages accessed purely
through PyTorch hooks, so there is no fork to drift. TabSwift is vendored
because it has no distributable package, and its checkpoint is committed for
offline reproducibility. Upstream (`github.com/LAMDA-Tabular/TabSwift`, HF
`LAMDA-Tabular/TabSwift`) is public Apache-2.0 and ships **one** `swift.ckpt`
serving both tasks; our copy is byte-identical to it. An earlier revision of
this file claimed the upstream repo was gated behind HTTP 401 — that was an
artefact of an undocumented vendor edit pointing at a repo that does not exist.
See `vendor/PATCHES.md`.

### Deliberately unavailable

Recorded in `registry.UNAVAILABLE` so their absence is a documented fact rather
than a silent omission: **TabPFN v1** (no checkpoint on disk, codebase predates
this sklearn API), **TabPFN v2.5/v2.6 classifiers** (HF repos gated, HTTP 401),
**Mitra**, **nanoTabPFN**.

---

## 2. Conventions you can rely on

Stated in full in `interp/conventions.py`; each one is enforced by a test.

**Units.** `trace.pred` is in the original units of the `y` you passed, and
equals the estimator's own `.predict()` to within `1e-4`. Adapters never
standardise `y` internally — each model's own preprocessing owns that. If you
want standardised targets, standardise the task before calling.

**Token layout.** Every model consumes `[prefix][support][query]`. `n_prefix`
is read from the loaded model's config, never hardcoded (TabSwift's 64
registers, TabPFN 2.5's 64 thinking tokens). Use
`trace.layout.support_slice` / `.query_slice` instead of arithmetic.

**Layer index.** `l` runs over the ICL stack, `0 <= l < n_layers`, and
`resid[l]` is that block's **output**. Adapters never truncate — TabSwift
exposes all 24 blocks. Stages *before* the ICL stack (TabICL's column embedder
and row interaction, TabSwift's column/row stages) are outside this index.

**Residual stream.** `trace.resid[l]` is `(n_tokens, d_model)` for every model.
TabPFN is per-cell — its native state is `(items, feature_groups, d_model)` — so
the canonical row vector is the **last feature slot**, because that is the only
slot the decoder reads (`encoder_out[:, single_eval_pos:, -1]`). The unreduced
tensor is always available as `resid_full[l]`; nothing is discarded.

**Attention.** Recorded per forward call, never collapsed. A TabPFN block runs
item attention *twice* (`query_to_context`, then `context_to_context`) and both
survive as separate records. `AttentionRecord.probs` keeps
`(batch, heads, q, k)` — pooling over heads or feature groups is your explicit
choice, via `.mean_over_heads()` / `.mean_over_heads_and_batch()`.

**Capabilities.** `SPECS[...].has_predictive_distribution` is `False` for
TabSwift, and `predict_quantiles` raises rather than returning a made-up number.
Gate uncertainty experiments on the flag.

---

## 3. Reading and writing the residual stream

```python
from models import load
from models.interp import CaptureSpec

m = load("tabicl_v2", task="regression")

# read
trace = m.run(X_tr, y_tr, X_te, capture=CaptureSpec(
    resid=True, attention=True, layer_pred=True, layers=[0, 5, 11]))
trace.support(5)                      # (n_train, 512)
trace.attn_of_kind(5, "all_to_support").mean_over_heads_and_batch()
trace.layer_pred[5]                   # logit-lens read, original y units

# write — this is a causal intervention, not an observation
donor = m.run(X_other, y_other, X_te, capture=CaptureSpec(resid_full=True, layers=[3]))
out = m.run(X_tr, y_tr, X_te, patches=[m.make_patch(3, donor.resid_full[3])])
```

`make_patch(layer, values, positions=...)` overwrites block `layer`'s output, so
the patched state genuinely flows into block `layer+1`. `positions` selects
along the sequence axis (use `trace.layout` to build it).

`layer_pred` is *not* a hand-trained per-layer decoder. It overwrites the last
block's output with block `l`'s state and re-runs the model's real predict path,
so it decodes through the model's own norm + decoder + output transform. At
`l = n_layers - 1` this is the identity, which is why
`layer_pred[-1] == pred` exactly.

### Negative control

```python
m = load("tabpfn_v2", random_weights_seed=7)
```
A randomly initialised clone of the same architecture. It is a **seed**, so the
control is one fixed network — reproducible across calls, across instances, and
across runs. Normalisation gains are left at their trained values by default
(randomising a LayerNorm gain to N(0,1) gives a differently-conditioned network,
not "the same architecture with random weights").

---

## 4. Verifying it

```bash
python -m models.interp.verify                      # roster, regression
python -m models.interp.verify --task classification
python -m models.interp.verify --models tabpfn_v2 -v
```

17 checks per model. **Run this after any dependency upgrade** — it is the
thing that turns a silent behaviour change into a failed test. Current status:
all four models, both tasks, 17/17.

Shape and NaN checks pass on a broken hook, so every check here targets a
specific failure mode:

| check | what it catches |
|---|---|
| `prediction_roundtrip` | predictions left in the wrong units |
| `logit_lens_identity` | a decode path that isn't the model's own tail |
| `capture_call_count` | silent chunking, or a hook on a stale module |
| `layout_matches_data` | hardcoded token offsets |
| `query_permutation` | query rows silently reordered vs `X_test` |
| `identity_patch` | wrong slicing; a write that lands elsewhere than the read |
| `zero_patch_moves_output` | a "patch" that is secretly a no-op |
| `cross_task_patch` | that the exposed stream is causally sufficient |
| `attention_module_is_live` | hooks installed on a shadowed second copy of a package |
| `attention_selfcheck` | recovered probabilities that don't reproduce real attention |
| `attention_calls_complete` | keeping one of several attention calls per block |
| `observation_is_free` | an "observer" that changes what the model computes |
| `no_global_mutation` | a monkeypatch left installed |
| `determinism` | unseeded state |
| `random_control_reproducible` | a control that is a different network each call |
| `registry_matches_reality` | architecture facts drifting from the checkpoint |

Two of these found real bugs during development and are worth knowing about:

* **`attention_module_is_live`.** TabICL was originally vendored. The checkpoint
  unpickles classes under the absolute path `tabicl._model.*`, so the model was
  built from the *pip* copy while hooks sat on the *vendored* one. Every capture
  came back empty and every shape check still passed. This is why TabICL must
  not be vendored.
* **`attention_selfcheck`.** These models run under `torch.autocast`, and
  `einsum`/`matmul` are on the autocast cast-to-fp16 list — calling `.float()`
  on the inputs is not enough, autocast casts them straight back. The recovered
  probabilities were wrong by up to 18% while still looking like a plausible
  attention matrix. The observers now disable autocast explicitly.

### Checkpoint integrity

```python
from models.registry import verify_manifest, write_manifest
verify_manifest()   # SHA-256 of every in-repo checkpoint
```

---

## 5. How attention is recovered without changing the model

All four models use fused SDPA/FlashAttention, which never materialises the
probability matrix. Rather than replace the attention implementation (the
previous approach — a permanent global monkeypatch that shifted predictions by
~0.75% of their standard deviation), a `FunctionObserver` wraps the attention
entry point *for the duration of one `with` block*:

1. it calls the **original** function for the returned value, so the model's
   numerics are untouched — asserted by `observation_is_free`;
2. it separately recomputes the probabilities in fp32 from the same q/k;
3. it checks that `probs @ v` reproduces the original's output, with a tolerance
   set by the dtype the model actually computed in;
4. it restores the original in a `finally` — asserted by `no_global_mutation`.

There is exactly one such entry point per model family:
`MultiHeadAttention.compute_attention_heads` (TabPFN) and
`sdpa_with_flattened_batch` (TabICL, TabSwift).

---

## 6. Layout

```
models/
├── README.md            you are here
├── registry.py          SPECS: the single source of truth + checkpoint manifest
├── loaders.py           load()
├── checkpoints/
│   ├── MANIFEST.json    sha256 + size of every in-repo checkpoint
│   └── tabswift/swift.ckpt
├── vendor/
│   ├── PATCHES.md       every change made to vendored upstream code
│   └── tabswift/        upstream TabSwift (no distributable package exists)
└── interp/
    ├── conventions.py   the frozen contracts, with rationale
    ├── hooks.py         ForwardCapture / ForwardPatch / FunctionObserver
    ├── trace.py         Trace, AttentionRecord
    ├── base.py          InterpModel: the uniform API
    ├── adapters/        one per model family
    └── verify.py        the battery
```
