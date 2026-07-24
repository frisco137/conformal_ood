# Mechanistic Interpretability Suite

A toolkit for probing, visualising, and understanding the internal representations
of tabular foundation models. This suite provides **hook-based activation
extraction** scripts and comprehensive **architecture documentation** for every
model in the `models/` repository.

---

## Supported Models

| Model | Architecture Class | Embed Dim | ICL Dim | ICL Blocks | Normalization | Positional Enc | Decoder |
|---|---|---|---|---|---|---|---|
| **TabPFN v3** | `TabPFNV3` | 128 | 512 | 24 | LayerNorm | SSMax + RoPE | ManyClassDecoder (attn) |
| **TabPFN v2** | `PerFeatureTransformer` | 192 | 192 | 12 groups × (row+col+MLP) | LayerNorm | Column embeddings | MLP (Linear→GELU→Linear) |
| **TabICL v2** | `TabICL` | 128 | 512 | 12 | LayerNorm (bias-free option) | SSMax + RoPE | MLP (Linear→GELU→Linear) |
| **Google TabFM** | `TabFM` | 256 | 2048 (8×256) | 24 | RMSNorm | RoPE (row), PerDimScale | MLP |

---

## Architecture Documentation

### TabPFN v3

A 4-stage pipeline: cell embedding → per-column distribution learning → row
aggregation → in-context learning.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TabPFN v3 Pipeline                          │
│                                                                     │
│  Input: X [n_train+n_test, n_features]  +  y_train [n_train]       │
│                                                                     │
│  ┌─── Stage 0: Cell Embedding ──────────────────────────────┐       │
│  │  Feature Grouping (circular shift, group_size=3)          │       │
│  │  Linear(group_size → embed_dim=128)                       │       │
│  │  → x_embed: [B, n_rows, n_col_groups, 128]               │       │
│  └───────────────────────────────────────────────────────────┘       │
│                            ↓                                        │
│  ┌─── Stage 1: Feature Distribution Embedder ───────────────┐       │
│  │  Per-column: reshape → [B×n_cols, n_rows, 128]            │       │
│  │  3× InducedSelfAttentionBlock:                            │       │
│  │    ┌─ cross_attn1: ind_pts[128,128] ← rows  (SSMax)      │       │
│  │    │  cross_attn2: rows ← ind_pts[128,128]               │       │
│  │    └─ (repeat 3×)                                         │       │
│  │  LN_w + Linear_w → [B, n_rows, n_cols, 128]              │       │
│  └───────────────────────────────────────────────────────────┘       │
│                            ↓                                        │
│  ┌─── Stage 2: Column Aggregator ───────────────────────────┐       │
│  │  4 CLS tokens prepended per row → [B, n_rows, C+4, 128]  │       │
│  │  3× TransformerBlock (RoPE, GELU FFN):                    │       │
│  │    Pre-LN → Self-Attention → Residual → Pre-LN → FFN     │       │
│  │  CLS readout: first 4 tokens → concat → [B, n_rows, 512] │       │
│  └───────────────────────────────────────────────────────────┘       │
│                            ↓                                        │
│  ┌─── Stage 3: ICL Transformer ─────────────────────────────┐       │
│  │  y_encoder: OneHot+Linear (clf) or Linear (reg)           │       │
│  │  y-embedding added to train row representations           │       │
│  │  24× ICLTransformerBlock:                                 │       │
│  │    Pre-LN → ICLAttention(q=all, kv=train only, SSMax)     │       │
│  │    → Residual → Pre-LN → FFN → Residual                  │       │
│  │  Output norm: LayerNorm                                   │       │
│  └───────────────────────────────────────────────────────────┘       │
│                            ↓                                        │
│  ┌─── Decoder ──────────────────────────────────────────────┐       │
│  │  Classification: ManyClassDecoder (attention, 6 heads)    │       │
│  │    q=test_repr, kv=class_prototypes → logits              │       │
│  │  Regression: Linear → GELU → Linear → scalar             │       │
│  └───────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

**Key submodules** (for hooking):
- `model.x_embed` — Stage 0 cell embedding linear
- `model.feature_distribution_embedder.layers[i].cross_attn_block1/2` — Stage 1 ISABs
- `model.column_aggregator.blocks[i].attention/.mlp` — Stage 2 transformer blocks
- `model.icl_blocks[i].icl_attention/.mlp` — Stage 3 ICL transformer blocks
- `model.output_norm` — Final layer norm
- `model.many_class_decoder` — Classification decoder

---

### TabPFN v2

An interleaved row/column attention architecture that treats each cell as a
separate token. Each layer group applies row-wise attention (features within a
row attend to each other), column-wise attention (cells within a column attend
to each other), and a per-cell MLP.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TabPFN v2 Pipeline                          │
│                                                                     │
│  Input: X [n_rows, B, n_features]  +  y [n_rows, B]                │
│                                                                     │
│  ┌─── Encoder Pipeline ─────────────────────────────────────┐       │
│  │  RemoveEmptyFeatures → NanHandling → NormalizeGroups      │       │
│  │  → FeatureTransform → NormalizeGroups                     │       │
│  │  → LinearInput(grouped_features → embed_dim=192)          │       │
│  │  + Column positional embeddings (random, fixed per col)   │       │
│  │  → x_enc: [B, n_rows, n_col_groups, 192]                 │       │
│  └───────────────────────────────────────────────────────────┘       │
│                            ↓                                        │
│  ┌─── y Encoder → target as extra column ───────────────────┐       │
│  │  NanHandling → MulticlassTargetEncoder                    │       │
│  │  → LinearInput → [B, n_rows, 1, 192]                     │       │
│  │  Concatenated as final column: [B, n_rows, C+1, 192]     │       │
│  └───────────────────────────────────────────────────────────┘       │
│                            ↓                                        │
│  ┌─── 12× Layer Groups ────────────────────────────────────┐        │
│  │  Each group contains 4 sub-modules:                      │        │
│  │                                                          │        │
│  │  [0] Row Attention (MultiHeadAttention):                 │        │
│  │      All features within each row attend to each other   │        │
│  │      → reshape [B*R, C, E] for attention over C          │        │
│  │                                                          │        │
│  │  [1] Column Attention (MultiHeadAttention):              │        │
│  │      All cells within each column attend to each other   │        │
│  │      → reshape [B*C, R, E] for attention over R          │        │
│  │      Test rows only attend to train rows (implicit mask) │        │
│  │                                                          │        │
│  │  [2] MLP: Linear(192→384) → GELU → Linear(384→192)      │        │
│  │                                                          │        │
│  │  [3] LayerNorms: 3× LayerNorm (one after each of above)  │       │
│  │                                                          │        │
│  │  Post-norm architecture: x → sublayer → x + out → LN     │       │
│  └──────────────────────────────────────────────────────────┘        │
│                            ↓                                        │
│  ┌─── Decoder ──────────────────────────────────────────────┐       │
│  │  Extract test rows from last column (target column)       │       │
│  │  → Linear(192→384) → GELU → Linear(384→n_classes)        │       │
│  └───────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

**Key submodules** (for hooking):
- `model.encoder[-1]` — Final encoder step (LinearInputEncoderStep)
- `model.y_encoder[-1]` — Final target encoder step
- `model.feature_positional_embedding_embeddings` — Column positional embeddings
- `model.transformer_encoder.layers[i][0]` — Row-wise MultiHeadAttention (group i)
- `model.transformer_encoder.layers[i][1]` — Column-wise MultiHeadAttention
- `model.transformer_encoder.layers[i][2]` — MLP
- `model.transformer_encoder.layers[i][3][0/1/2]` — LayerNorms
- `model.decoder_dict["standard"]` — Decoder MLP

---

### TabICL v2

A 3-stage pipeline similar in spirit to TabPFN v3 but with different internals:
scalable softmax (SSMax), RoPE, and a `QuantileToDistribution` regression head.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TabICL v2 Pipeline                          │
│                                                                     │
│  Input: X [B, n_rows, n_features]  +  y_train [B, n_train]         │
│                                                                     │
│  ┌─── Stage 1: ColEmbedding (col_embedder) ─────────────────┐      │
│  │  Feature Grouping (circular shift, group_size=3)          │      │
│  │  Per-column: [B×n_cols, n_rows, embed_dim=128]            │      │
│  │  3× InducedSelfAttentionBlock:                            │      │
│  │    ind_pts[128] → cross-attn with SSMax (qassmax-mlp-ew)  │      │
│  │    Two MAB sub-blocks per ISAB                            │      │
│  │  Target-aware: y-encoding added to train embeddings       │      │
│  │  → [B, n_rows, n_cols, 128]                               │      │
│  └───────────────────────────────────────────────────────────┘      │
│                            ↓                                        │
│  ┌─── Stage 2: RowInteraction (row_interactor) ─────────────┐      │
│  │  4 CLS tokens prepended per row                           │      │
│  │  Encoder: 3× MultiheadAttentionBlock:                     │      │
│  │    Pre-LN → Self-Attention (RoPE, GELU FFN) → Residual   │      │
│  │  CLS readout: 4 tokens → concat → [B, n_rows, 512]       │      │
│  │  Output RMSNorm                                           │      │
│  └───────────────────────────────────────────────────────────┘      │
│                            ↓                                        │
│  ┌─── Stage 3: ICLearning (icl_predictor) ──────────────────┐      │
│  │  y_encoder: OneHot+Linear (clf) or Linear (reg)           │      │
│  │  y-embedding added to train representations               │      │
│  │  12× Encoder blocks (MultiheadAttentionBlock):            │      │
│  │    Pre-LN → Self-Attention (train_size KV mask, SSMax)    │      │
│  │    → Residual → Pre-LN → FFN → Residual                  │      │
│  │  LayerNorm → decoder                                      │      │
│  └───────────────────────────────────────────────────────────┘      │
│                            ↓                                        │
│  ┌─── Decoder ──────────────────────────────────────────────┐       │
│  │  Classification: Linear(512→1024) → GELU → Linear(1024→C)│       │
│  │  Regression: same, then QuantileToDistribution             │      │
│  └───────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key submodules** (for hooking):
- `model.col_embedder` — Stage 1 output (full ColEmbedding module)
- `model.col_embedder.blocks[i].multihead_attn1/2` — ISABs in column embedding
- `model.row_interactor` — Stage 2 output
- `model.row_interactor.encoder.blocks[i].attn` — Row interaction attention
- `model.icl_predictor.tf_icl.blocks[i].attn` — ICL transformer attention
- `model.icl_predictor.ln` — Output layer norm
- `model.icl_predictor.decoder` — Decoder MLP

---

### Google TabFM

A 6-stage pipeline with Fourier-based cell embedding, dual col/row interaction
rounds, and a 24-layer ICL transformer. Notable for SwiGLU FFN, PerDimScale,
q/k RMSNorm, and interleaved RoPE.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Google TabFM Pipeline                         │
│                                                                     │
│  Input: X [B, n_rows, n_features] + y [B, n_rows] + train_size [B] │
│                                                                     │
│  ┌─── CellEmbedder ────────────────────────────────────────┐        │
│  │  Feature grouping (circular shift, group_size=3)         │        │
│  │  Fourier expansion: x → [sin(x·freq), cos(x·freq)]      │        │
│  │    32 frequencies, separate num/cat linear projections    │        │
│  │  Sum over group → [B, T, n_cols, 256]                    │        │
│  │  y_embedder: Embedding(10, 256) [clf] / MLP(1→256) [reg] │       │
│  │  Target-aware: y-embedding added to train cell embeds     │       │
│  └──────────────────────────────────────────────────────────┘        │
│                            ↓                                        │
│  ┌─── ColEmbedding 1 (SetTransformer) ──────────────────────┐       │
│  │  Per-column: [B×n_cols, T, 256]                           │       │
│  │  3× InducedSelfAttentionBlock:                            │       │
│  │    ind_pts[256] → MAB1 (SwiGLU FFN, q/k RMSNorm, PDS)    │       │
│  │    rows ← MAB2 (SwiGLU FFN)                              │       │
│  │  RMSNorm + Linear → [B, T, n_cols, 256]                  │       │
│  └───────────────────────────────────────────────────────────┘       │
│                            ↓                                        │
│  ┌─── RowInteraction 1 (Encoder + CLS, output_full=True) ──┐       │
│  │  8 CLS tokens prepended per row                           │       │
│  │  3× MultiheadAttentionBlock:                              │       │
│  │    RoPE + PerDimScale + q/k RMSNorm + SwiGLU FFN         │       │
│  │  RMSNorm → full output: [B, T, n_cols+8, 256]            │       │
│  └───────────────────────────────────────────────────────────┘       │
│                            ↓                                        │
│  ┌─── ColEmbedding 2 (second round, same architecture) ─────┐      │
│  │  → [B, T, n_cols+8, 256]                                  │      │
│  └───────────────────────────────────────────────────────────┘      │
│                            ↓                                        │
│  ┌─── RowInteraction 2 (Encoder + CLS, output_full=False) ──┐      │
│  │  3× MAB blocks (same)                                     │      │
│  │  CLS readout: 8 tokens → concat → [B, T, 2048]           │      │
│  └───────────────────────────────────────────────────────────┘      │
│                            ↓                                        │
│  ┌─── ICLearning ───────────────────────────────────────────┐       │
│  │  y_encoder: OneHot+Linear(10→2048) [clf] / MLP(1→2048)   │       │
│  │  y-encoding added to train representations (masked)       │       │
│  │  24× MAB Encoder blocks:                                  │       │
│  │    NO RoPE, PerDimScale, q/k RMSNorm, SwiGLU FFN         │       │
│  │    Train-only attention mask                               │       │
│  │  RMSNorm → decoder                                        │       │
│  └───────────────────────────────────────────────────────────┘       │
│                            ↓                                        │
│  ┌─── Decoder ──────────────────────────────────────────────┐       │
│  │  MLP(2048→4096→10) [clf] / MLP(2048→4096→1) [reg]        │       │
│  └───────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

**Key submodules** (for hooking):
- `model.cell_embedder` — Fourier cell embedding
- `model.col_embedder.tf_col.blocks[i].mab1/.mab2` — 1st SetTransformer ISABs
- `model.row_interactor.tf_row.blocks[i].attn` — 1st Encoder attention
- `model.col_embedder_2.tf_col.blocks[i].mab1/.mab2` — 2nd SetTransformer ISABs
- `model.row_interactor_2.tf_row.blocks[i].attn` — 2nd Encoder attention
- `model.icl_predictor.tf_icl.blocks[i].attn` — ICL attention
- `model.icl_predictor.ln` — Output RMSNorm
- `model.icl_predictor.decoder` — Decoder MLP

---

## Architectural Differences (Deep Dive)

### Attention Mechanisms

| Feature | TabPFN v3 | TabPFN v2 | TabICL v2 | Google TabFM |
|---|---|---|---|---|
| **Column attention** | InducedSelfAttention (SetTransformer) | Full self-attention (AlongRowAttention) | InducedSelfAttention | InducedSelfAttention (SetTransformer) |
| **Row attention** | Transformer blocks + CLS readout | Full self-attention (AlongColumnAttention) with multi-query for test | Transformer blocks + CLS readout | Encoder blocks + CLS readout |
| **ICL attention** | Full self-attention, train-only KV | N/A (interleaved row/col IS the ICL) | Full self-attention, train-only KV | Full self-attention, train-only mask |
| **Positional encoding** | SSMax (qassmax-mlp-elementwise) + RoPE | Column embeddings (random) | SSMax (qassmax-mlp-elementwise) + RoPE | RoPE (interleaved) + PerDimScale |
| **q/k normalization** | None | None | None | RMSNorm on q and k |
| **Attention scaling** | Standard √d | Standard √d | SSMax scaling | PerDimScale (softplus) at scale=1.0 |

### Feed-Forward Networks

| Feature | TabPFN v3 | TabPFN v2 | TabICL v2 | Google TabFM |
|---|---|---|---|---|
| **Activation** | GELU | GELU | GELU | SwiGLU |
| **Architecture** | Linear→GELU→Linear | Linear→GELU→Linear | Linear→GELU→Linear | Linear_gate + Linear→SiLU(gate)×proj→Linear |
| **Normalization** | Pre-norm (LayerNorm) | Post-norm (LayerNorm) | Pre-norm (LayerNorm) | Pre-norm + Post-norm (RMSNorm) |

### Target Handling

| Feature | TabPFN v3 | TabPFN v2 | TabICL v2 | Google TabFM |
|---|---|---|---|---|
| **Classification** | OneHot+Linear | Preprocessing pipeline | OneHot+Linear | OneHot+Linear (Embedding) |
| **Regression** | Linear(1→d) | Preprocessing pipeline | Linear(1→d) | MLP(1→d) |
| **Target injection** | Added to train rows before ICL | Appended as extra column | Added to train rows before ICL | Added to train cells at embedding |

---

## Usage

### Quick Start

```python
# Run individual model hook demo
python models/mech_interp_suite/hooks_tabpfn_v3.py
python models/mech_interp_suite/hooks_tabpfn_v2.py
python models/mech_interp_suite/hooks_tabicl_v2.py
python models/mech_interp_suite/hooks_google_tabfm.py

# Run all sanity checks
python models/mech_interp_suite/run_all_sanity_checks.py
```

### Using HookManager in Your Own Code

```python
from models.mech_interp_suite.utils import HookManager, generate_gaussian_data
from collections import OrderedDict

# 1. Load and fit your model
clf = ...  # any of the four classifiers
clf.fit(X_train, y_train)

# 2. Access the inner PyTorch model
inner_model = clf.models_[0]  # TabPFN v3/v2
# inner_model = clf.model_     # TabICL v2
# inner_model = clf.model      # Google TabFM

# 3. Define which layers to hook
targets = OrderedDict({
    "layer_0_attn": inner_model.some.path.to.attention,
    "layer_0_mlp":  inner_model.some.path.to.mlp,
})

# 4. Run with hooks
with HookManager(targets) as hm:
    preds = clf.predict_proba(X_test)

# 5. Access activations
for name, tensor in hm.activations.items():
    print(f"{name}: shape={tensor.shape}, mean={tensor.float().mean():.4f}")
```

### Accessing Model Internals

```python
# TabPFN v3 — access via clf.models_[0] after fit()
clf = get_tabpfn_v3_classifier(device="cpu")
clf.fit(X_train, y_train)
model = clf.models_[0]  # TabPFNV3 nn.Module

# TabPFN v2 — access via clf.models_[0] after fit()
clf = get_tabpfn_v2_classifier(device="cpu")
clf.fit(X_train, y_train)
model = clf.models_[0]  # PerFeatureTransformer nn.Module

# TabICL v2 — access via clf.model_ after fit()
clf = get_tabicl_v2_classifier(device="cpu")
clf.fit(X_train, y_train)
model = clf.model_  # TabICL nn.Module

# Google TabFM — access via clf.model (always available)
clf = get_google_tabfm_classifier(device="cpu")
model = clf.model  # TabFM nn.Module (available before fit)
```

---

## File Overview

| File | Purpose |
|---|---|
| `utils.py` | `HookManager` context-manager, data generators, activation printer/validator |
| `hooks_tabpfn_v3.py` | Hook demo for TabPFN v3 (63 hooks across all 4 stages) |
| `hooks_tabpfn_v2.py` | Hook demo for TabPFN v2 (76 hooks across 12 layer groups) |
| `hooks_tabicl_v2.py` | Hook demo for TabICL v2 (30 hooks across 3 stages) |
| `hooks_google_tabfm.py` | Hook demo for Google TabFM (80+ hooks across 6 stages) |
| `run_all_sanity_checks.py` | Runs all 4 hook scripts, validates, prints summary |
| `README.md` | This file — architecture docs and usage guide |
