# TabPFN Circuit-Level Mechanistic Interpretability

This directory ports the circuit-level analyses in [`circuits/`](../circuits/) (which
target the Spectral-Mixture GP PFN) to **TabPFN**. The goal is the same: find and
causally verify the internal components that (a) perform in-context classification on
prior-conforming data and (b) react when the data violates the prior (OOD).

It mirrors the structure of `circuits/`: one shared hooks module plus one file per
experiment.

```
circuits_tabpfn/
├── tabpfn_hooks.py                  # Shared: model loading, data, CE loss, ManualMHA, divergence collector
├── attention_divergence_ablation.py # Experiment 1: attention JSD + head/MLP zero-ablation
├── activation_patching.py           # Experiment 2: activation patching (3 subexperiments)
├── results/                         # All figures + JSON outputs
└── README.md
```

---

## How TabPFN differs from the GP-PFN (and why this code looks different)

| | GP-PFN (`circuits/`) | TabPFN (here) |
|---|---|---|
| Layers × heads | 6 × 4 = 24 | **12 × 4 = 48** |
| Hidden dim / head dim | 256 / 64 | **512 / 128** |
| Task | regression | **10-class classification** |
| Loss / metric | MSE | **cross-entropy (NLL)** on query logits |
| Attention per layer | 1 call | **2 calls** (context→context, query→context) |
| Input shape | `(N, seq, 1)` (transposed in code) | `(seq, batch, features)` directly |

Two consequences drive the implementation:

1. **Cross-entropy, not MSE.** TabPFN's decoder outputs class logits of shape
   `(query_len, batch, 10)`. "Prediction quality" is measured by the cross-entropy of
   those logits against the true query labels (`tabpfn_hooks.compute_ce_loss`), ignoring
   the padding label `-100`.

2. **Two attention calls per layer.** With `efficient_eval_masking=True`
   ([`tabpfn/layer.py`](../tabpfn_repo/tabpfn/layer.py#L106-L111)) every encoder layer
   calls `self_attn` twice:
   - **ctoc** — context attends to context: `self_attn(src[:p], src[:p], src[:p])`
   - **qtoc** — query attends to context:   `self_attn(src[p:], src[:p], src[:p])`

   where `p = single_eval_pos`. All hooking, ablation and patching code handles the two
   calls separately, distinguishing them by call order (ctoc first, qtoc second).

Also note: TabPFN feeds **only the context labels** `y[:p]` into the model; query labels
`y[p:]` are never consumed, they are only used as evaluation targets. This fact is used
in the label-shuffle patching subexperiment (see below).

### `ManualMHA` — the core intervention tool

To ablate or patch *individual heads*, `tabpfn_hooks.ManualMHA` replaces a layer's
`self_attn.forward` with an explicit re-implementation of multi-head attention (QKV
projection → scaled dot-product softmax → per-head output → out_proj). This exposes the
per-head, pre-`out_proj` outputs so we can zero them (ablation), record them, or
substitute them from a cache (patching). A self-test (`python3 tabpfn_hooks.py`) verifies
that in passthrough mode it reproduces the stock `nn.MultiheadAttention` output to within
`1e-4`, so any measured effect comes from the intervention, not from a numerical mismatch.

---

## Data

Generated on-the-fly via `tabpfn_ood/generate_data.py` (level-2 categories):

- **In-prior** (`get_inprior_data`) — prior-conforming data the model handles well. ID.
- **In-prior with noise** (`without_noise=False`) — used as a corruption in patching.
- **Out-prior** (`get_outprior_data`) — anti-prior data. OOD.

The checkpoint loaded is
`tabpfn_repo/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt`
(12 layers, 4 heads, emsize 512, 10 classes, 100 features), via
`tabpfn.scripts.model_builder.load_model_only_inference`.

Default config: `seq_len=500`, `single_eval_pos=450` (450 context / 50 query),
`N≈500` (experiment 1) / `N=256` (experiment 2).

---

## Experiment 1 — Attention divergence + ablation

```bash
python3 attention_divergence_ablation.py            # defaults: seq=500, ctx=450, N=500
python3 attention_divergence_ablation.py --n_samples 300 --batch_size 50
```

**Part A — Attention divergence.** For each of the 48 heads, the Jensen-Shannon
Divergence between the mean ID and mean OOD attention maps is computed *separately* for
the ctoc and qtoc attention. Mean maps are accumulated as running sums so memory stays
bounded regardless of `N`.

**Part B — Zero-ablation.** Each head (via `ManualMHA`, zeroing the head before
`out_proj`) and each MLP (zeroing the `linear2` output) is ablated in turn, and the
increase in cross-entropy on **in-prior** query points (`ΔCE`) is recorded.

**Outputs** (`results/`):
- `attn_jsd_heatmap_ctoc.png`, `attn_jsd_heatmap_qtoc.png` — 12×4 JSD heatmaps
- `ablation_delta_ce_bar.png` — ranked ΔCE bar chart for all 48 heads + 12 MLPs
- `ablation_ce_heatmap.png` — 12×4 per-head ΔCE heatmap
- `cross_reference_jsd_vs_ablation.png` — JSD vs ΔCE scatter (ctoc & qtoc, with Pearson r)
- `experiment1_results.json`

---

## Experiment 2 — Activation patching

```bash
python3 activation_patching.py                      # runs all 4 subexperiments
python3 activation_patching.py --subexp shuffle_y   # a single subexperiment
python3 activation_patching.py --subexp oop_same_features
python3 activation_patching.py --n_samples 128 --batch_size 64
```

The **clean** run is always in-prior data. The four subexperiments differ
only in how the **corrupted** run is built:

| subexp | corrupted run | pairing | evaluation target |
|---|---|---|---|
| `shuffle_y`  | same clean batch, **labels shuffled** | matched x | **true** query labels (both runs) |
| `noise`      | same clean batch **+ Gaussian feature noise** | matched x | each run's own labels |
| `out_prior`  | **out-prior** data (independent draw) | by index | each run's own labels |
| `oop_same_features` | **same X**, labels re-derived from a random causal DAG over the feature columns | matched x (`x_corr == x_clean`) | each run's own labels |

For `shuffle_y`, since the model only consumes context labels and clean/corrupt share
the same `x`, both runs are scored against the *true* query labels — otherwise patching
against randomly-shuffled targets would be meaningless.

### Subexperiment 4 — `oop_same_features` (out-of-prior, same features)

The earlier corruptions trade off two things you actually want at the same time. A
pure resample (`out_prior`) lands far from the clean run, so there is little shared
structure for patching to *recover* toward. `noise`/`shuffle_y` stay close but are not
really a different *function* of the data. `oop_same_features` is designed to be both
**close** (it shares the exact feature matrix `X`) and **genuinely out-of-prior** (the
labels come from a labeling mechanism outside TabPFN's MLP prior), while still being a
**deterministic, learnable function of `X`** — so in-context recovery is possible.

**Construction** (`generate_data.oop_same_features`, exposed as
`tabpfn_hooks.make_oop_same_features`):

1. Draw a clean in-prior table `(X, y_clean)`.
2. Sample a random **causal DAG** whose only roots are the feature columns:
   - Nodes `0 … F-1` are the **feature roots** (`F = num_features`); they are
     assigned the clean feature columns. Nodes `F … n_nodes-1` are internal hidden
     nodes (one of which becomes the target). `n_nodes ≫ F` (default `3·F`).
   - Edges only point from a **lower index to a higher index**, which makes the graph
     acyclic by construction. Each candidate edge is included independently with
     probability `edge_prob` (default `0.10`).
   - **Every internal node is forced to have ≥ 1 parent** among the strictly-earlier
     nodes (if none were sampled, one earlier node is drawn at random). This guarantees
     every internal node — and therefore the target — is causally downstream of the
     feature roots, i.e. the labels are a function of `X` and nothing else. The feature
     columns are the *only* roots.
   - Each edge carries a Gaussian weight and a random **per-edge activation** in
     `{Linear, ReLU, Tanh, Sigmoid, ELU}` — the same activation set the anti-prior
     (`AntiLinearLayer`) uses.
3. The **target node** is chosen at random from the internal nodes whose in-degree is
   in the top quartile (≥ 1 guaranteed) — i.e. a node with a "good number" of incoming
   edges.
4. **Propagate** the clean features through the DAG, **noise-free**, in index order.
   Each node value is the in-degree-normalised sum of its per-edge activated, weighted
   parent values (`sum_e act_e(w_e · v_parent) / √(in-degree)`, the `AntiLinearLayer`
   scaling). A fresh DAG is drawn per call.
5. **Bin** the target-node values into class labels with the *exact same* pipeline the
   real generators use (`process_batch_data`: normalize → `MulticlassRank` →
   consecutive-class compression → label rotation), giving `y_corr`.

The corrupted table is `(X, y_corr)`: identical features, a new anti-prior-but-learnable
labeling function. Because `x_corr == x_clean`, forward/reverse patching isolate exactly
the components that implement the *labeling rule* the table follows, with the input
representation held fixed.

**Sanity check** — `sanity_oop_same_features.py` draws `--n_tables` tables (a fresh
clean table + a new random DAG corruption each), runs TabPFN on the clean and corrupted
version of every table, and plots the two **query-accuracy** distributions as a quantile
candlestick (`Low=10th, Open=25th, mid=50th, Close=75th, High=90th`, the repo
convention). A valid corruption should sit **below** the clean accuracy (it is
out-of-prior) yet **above** chance `1/num_classes` (it is still learnable in-context).

```bash
python3 sanity_oop_same_features.py                 # 100 tables, seq=500, ctx=450
python3 sanity_oop_same_features.py --n_tables 200 --edge_prob 0.05
```

Outputs: `results/oop_same_features_sanity_candlestick.png`, `results/oop_same_features_sanity.json`.

**Workflow per subexperiment:**
1. Baseline clean CE and corrupt CE.
2. **Forward patching** (clean → corrupt): cache clean activations, patch them into the
   corrupted run. `recovery = (CE_corrupt − CE_patched) / (CE_corrupt − CE_clean)`.
3. **Reverse patching** (corrupt → clean): patch corrupt activations into the clean run.
   `degradation = (CE_patched − CE_clean) / (CE_corrupt − CE_clean)`.

**Patch granularities:** full layer output (`layer`), whole attention block / all heads
(`attn`, patches both ctoc and qtoc calls), FFN output (`mlp`), and — new vs the GP-PFN
reference — **per individual head** (`head`, via `ManualMHA` record/patch).

**Outputs** (per subexp, under `results/patching_<subexp>/`):
- `forward_recovery_curves.png` — recovery vs layer for layer/attn/mlp + mean-over-heads
- `reverse_degradation_curves.png` — degradation vs layer
- `component_heatmaps.png` — 12×[attn,mlp,layer] forward recovery & reverse degradation
- `head_recovery_heatmap.png` — 12×4 per-head forward recovery
- `head_degradation_heatmap.png` — 12×4 per-head reverse degradation
- `patching_<subexp>_results.json`

---

## Notes / knobs

- All scripts auto-select `cuda:0` if available, else CPU.
- `single_eval_pos` is the context/query split; attention cost scales with the context
  length squared, so lower it (or `seq_len`) if you hit memory limits.
- The `attn` and `head` patch modes recompute attention without the fused kernel; a run
  is heavier than plain inference but well within an A6000's memory at the defaults.
- JSON files contain the raw numbers behind every figure for downstream analysis.
