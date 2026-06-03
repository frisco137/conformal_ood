"""
activation_patching.py — Experiment 2 for TabPFN
=================================================

Causal tracing via activation patching, adapted from
`circuits/activation_patching.py` to TabPFN (12 layers x 4 heads, classifier,
two-call attention). Adds **per-head** patching, which the reference did not do.

Four subexperiments, differing only in how the *corrupted* run is built. In all
of them the **clean** run is in-prior data.

  subexp1 (shuffle_y) : corrupt = the SAME clean batch, with all output labels
                        shuffled. Because TabPFN consumes only the *context*
                        labels y[:p] (query labels are never fed to the model),
                        the shuffle misleads the model via the context; both
                        clean and corrupt runs are evaluated against the TRUE
                        query labels (clean & corrupt share identical x).
  subexp2 (noise)     : corrupt = in-prior data generated WITH noise
                        (independent draw, aligned by index). Each run evaluated
                        against its own true labels.
  subexp3 (out_prior) : corrupt = out-prior (anti-prior) data (independent
                        draw). Each run evaluated against its own true labels.
  subexp4 (oop_same_features) : corrupt = SAME features X, labels re-derived
                        from a random causal DAG rooted at the feature columns
                        (out-of-prior labeling, deterministic learnable function
                        of X). x_corr == x_clean; each run vs its own labels.
                        See README "Subexperiment 4" for the construction.

Workflow (per subexp), mirroring the GP-PFN pipeline:
  - clean run, corrupt run -> baseline CEs.
  - FORWARD patching (clean -> corrupt): patch clean activations into the
    corrupted run; recovery = (CE_corrupt - CE_patched) / (CE_corrupt - CE_clean).
  - REVERSE patching (corrupt -> clean): patch corrupt activations into the clean
    run; degradation = (CE_patched - CE_clean) / (CE_corrupt - CE_clean).

Patch granularities:
  - "layer" : full encoder-layer output
  - "attn"  : self-attention sub-block output (ALL heads in the layer; both the
              ctoc and qtoc calls are patched)
  - "mlp"   : FFN (linear2) output
  - "head"  : a single attention head's pre-out_proj output (per-head; via ManualMHA)

Outputs (per subexp, under results/patching_<subexp>/):
  - forward_recovery_curves.png   (layer / attn / mlp / mean-head, vs layer)
  - reverse_degradation_curves.png
  - component_heatmaps.png        (12 x [attn,mlp,layer] forward & reverse)
  - head_recovery_heatmap.png     (12 x 4 forward recovery)
  - head_degradation_heatmap.png  (12 x 4 reverse degradation)
  - patching_<subexp>_results.json
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from tabpfn_hooks import (
    load_tabpfn, make_inprior, make_outprior, make_oop_same_features, shuffle_labels,
    ManualMHA, NLAYERS, NHEAD, N_CLASSES,
)

plt.rcParams.update({
    'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
MODES = ["layer", "attn", "mlp"]


# ==============================================================================
# Per-batch CE (sum + count) — lets us aggregate exact CE across batches
# ==============================================================================
@torch.no_grad()
def batch_ce(model, bx, by_input, by_target, p):
    logits = model((bx, by_input.float()), single_eval_pos=p)   # (q, b, C)
    tgt = by_target[p:].long()
    fl = logits.reshape(-1, logits.shape[-1])
    ft = tgt.reshape(-1)
    valid = ft != -100
    if int(valid.sum()) == 0:
        return 0.0, 0
    ce = F.cross_entropy(fl[valid], ft[valid], reduction='sum')
    return float(ce.item()), int(valid.sum().item())


@torch.no_grad()
def dataset_ce(model, x, y_input, y_target, p, device, bs):
    s, c = 0.0, 0
    for i in range(0, x.shape[1], bs):
        bx = x[:, i:i + bs].to(device)
        byi = y_input[:, i:i + bs].to(device)
        byt = y_target[:, i:i + bs].to(device)
        cs, cc = batch_ce(model, bx, byi, byt, p)
        s += cs; c += cc
    return s / max(c, 1)


# ==============================================================================
# Source-activation caching for layer / attn / mlp (all layers, one run)
# ==============================================================================
class SourceCache:
    def __init__(self):
        self.layer_out = {}
        self.attn_out = {}    # layer -> list of the two self_attn outputs (ctoc, qtoc)
        self.mlp_out = {}
        self.hooks = []

    def install(self, model):
        for idx, layer in enumerate(model.transformer_encoder.layers):
            self.attn_out[idx] = []

            def mk_layer(l):
                def hook(m, i, o):
                    self.layer_out[l] = o.detach()
                return hook

            def mk_attn(l):
                def hook(m, i, o):
                    out = o[0] if isinstance(o, tuple) else o
                    self.attn_out[l].append(out.detach())
                return hook

            def mk_mlp(l):
                def hook(m, i, o):
                    self.mlp_out[l] = o.detach()
                return hook

            self.hooks.append(layer.register_forward_hook(mk_layer(idx)))
            self.hooks.append(layer.self_attn.register_forward_hook(mk_attn(idx)))
            self.hooks.append(layer.linear2.register_forward_hook(mk_mlp(idx)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def install_patch(model, layer_idx, mode, cache: SourceCache):
    """Install a patch hook on one layer for a given mode. Returns a remove() fn."""
    layer = model.transformer_encoder.layers[layer_idx]

    if mode == "layer":
        cached = cache.layer_out[layer_idx]
        def hook(m, i, o):
            B = o.shape[1]
            return cached[:, :B, :]
        h = layer.register_forward_hook(hook)
        return lambda: h.remove()

    if mode == "mlp":
        cached = cache.mlp_out[layer_idx]
        def hook(m, i, o):
            B = o.shape[1]
            return cached[:, :B, :]
        h = layer.linear2.register_forward_hook(hook)
        return lambda: h.remove()

    if mode == "attn":
        cached_list = cache.attn_out[layer_idx]   # [ctoc_out, qtoc_out]
        state = {"call": 0}
        def hook(m, i, o):
            c = cached_list[state["call"] % len(cached_list)]
            state["call"] += 1
            out = o[0] if isinstance(o, tuple) else o
            B = out.shape[1]
            patched = c[:, :B, :]
            return (patched, o[1]) if isinstance(o, tuple) else patched
        h = layer.self_attn.register_forward_hook(hook)
        return lambda: (h.remove(), state.update(call=0))

    raise ValueError(mode)


# ==============================================================================
# Layer / attn / mlp patching (one direction)
# ==============================================================================
def run_layerwise_patching(model, x_src, ysrc_in, x_tgt, ytgt_in, ytgt_eval,
                           p, device, bs):
    """
    Patch source activations into the target run. Evaluate target run vs ytgt_eval.
    Returns dict: mode -> {layer -> patched_ce}.
    """
    acc = {m: {l: [0.0, 0] for l in range(NLAYERS)} for m in MODES}  # [ce_sum, count]

    for i in range(0, x_tgt.shape[1], bs):
        bxs = x_src[:, i:i + bs].to(device)
        bys = ysrc_in[:, i:i + bs].to(device)
        bxt = x_tgt[:, i:i + bs].to(device)
        byt = ytgt_in[:, i:i + bs].to(device)
        byte = ytgt_eval[:, i:i + bs].to(device)

        # 1. cache source activations (all layers) in one run
        cache = SourceCache()
        cache.install(model)
        with torch.no_grad():
            _ = model((bxs, bys.float()), single_eval_pos=p)
        cache.remove()

        # 2. for each mode & layer, patch into target run
        for mode in MODES:
            for l in range(NLAYERS):
                remove = install_patch(model, l, mode, cache)
                cs, cc = batch_ce(model, bxt, byt, byte, p)
                remove()
                acc[mode][l][0] += cs
                acc[mode][l][1] += cc

    return {m: {l: acc[m][l][0] / max(acc[m][l][1], 1) for l in range(NLAYERS)} for m in MODES}


# ==============================================================================
# Per-head patching (one direction), via ManualMHA record/patch
# ==============================================================================
def run_headwise_patching(model, x_src, ysrc_in, x_tgt, ytgt_in, ytgt_eval,
                          p, device, bs):
    """Returns matrix (NLAYERS, NHEAD) of patched CE."""
    acc = np.zeros((NLAYERS, NHEAD))
    cnt = np.zeros((NLAYERS, NHEAD))

    for i in range(0, x_tgt.shape[1], bs):
        bxs = x_src[:, i:i + bs].to(device)
        bys = ysrc_in[:, i:i + bs].to(device)
        bxt = x_tgt[:, i:i + bs].to(device)
        byt = ytgt_in[:, i:i + bs].to(device)
        byte = ytgt_eval[:, i:i + bs].to(device)

        for l in range(NLAYERS):
            mha = model.transformer_encoder.layers[l].self_attn
            # 1. record source per-head outputs for this layer
            rec = ManualMHA(mha)
            rec.record = True
            rec.install(); rec.reset()
            with torch.no_grad():
                _ = model((bxs, bys.float()), single_eval_pos=p)
            recorded = rec.recorded            # [ctoc (B,nhead,Lq,hd), qtoc (...)]
            rec.uninstall()

            # 2. patch each head into the target run
            for h in range(NHEAD):
                pm = ManualMHA(mha)
                pm.patch_cache = recorded
                pm.patch_heads = {h}
                pm.install(); pm.reset()
                cs, cc = batch_ce(model, bxt, byt, byte, p)
                pm.uninstall()
                acc[l, h] += cs
                cnt[l, h] += cc

    return acc / np.maximum(cnt, 1)


# ==============================================================================
# Build clean/corrupt data per subexperiment
# ==============================================================================
NOISE_SIGMA = 1.0   # std of feature noise added for the "noise" subexp

def build_subexp_data(subexp, n, seq_len):
    """
    Returns a dict with:
      x_clean, yclean_in, yclean_eval, x_corr, ycorr_in, ycorr_eval

    Design principles:
      shuffle_y : clean base is in-prior WITH noise (so labels are non-degenerate
                  and shuffling actually breaks the x->y mapping, creating a real gap).
                  corrupt = same (x, y) but context labels shuffled. Both runs share
                  identical x; evaluated against the TRUE query labels.
      noise     : corrupt = same datasets with Gaussian noise added to x features
                  (x_corr = x_clean + sigma * randn, y unchanged). Matched pairs, so
                  patching is example-to-example coherent and corrupt_ce > clean_ce.
      out_prior : corrupt = out-prior data, independent draw (no meaningful match).
      oop_same_features : corrupt = SAME features, with labels re-derived from a
                  random causal DAG rooted at the feature columns (out-of-prior
                  labeling, but a deterministic learnable function of X). Matched x;
                  each run evaluated against its own labels.
    """
    # shuffle_y needs a noisy clean base so labels are diverse and shuffling matters
    if subexp == "shuffle_y":
        x_clean, y_clean = make_inprior(n, seq_len, without_noise=False)
        x_corr = x_clean.clone()
        ycorr_in = shuffle_labels(y_clean)          # shuffled context labels mislead model
        ycorr_eval = y_clean.clone()                # evaluate BOTH runs vs true labels
        yclean_eval = y_clean.clone()

    elif subexp == "noise":
        x_clean, y_clean = make_inprior(n, seq_len, without_noise=True)
        # Corrupt = same datasets + feature noise; y is shared/unchanged.
        # This keeps the pair matched: patching example i's activations into the
        # corrupt run for example i is meaningful because they share the same
        # underlying dataset and labels.
        x_corr = x_clean + NOISE_SIGMA * torch.randn_like(x_clean)
        ycorr_in = y_clean.clone()
        ycorr_eval = y_clean.clone()
        yclean_eval = y_clean.clone()
    elif subexp == "out_prior":
        x_clean, y_clean = make_inprior(n, seq_len, without_noise=True)
        x_corr, y_corr = make_outprior(n, seq_len)
        ycorr_in = y_corr
        ycorr_eval = y_corr.clone()
        yclean_eval = y_clean.clone()
    elif subexp == "oop_same_features":
        # Same features X, but labels re-derived from a random causal DAG rooted at
        # the feature columns: an out-of-prior labeling that is still a deterministic,
        # learnable function of X. x_corr == x_clean (the matched-pair ideal).
        x_clean, y_clean, y_corr = make_oop_same_features(n, seq_len)
        x_corr = x_clean.clone()
        ycorr_in = y_corr
        ycorr_eval = y_corr.clone()
        yclean_eval = y_clean.clone()
    else:
        raise ValueError(subexp)

    return {
        "x_clean": x_clean, "yclean_in": y_clean, "yclean_eval": yclean_eval,
        "x_corr": x_corr, "ycorr_in": ycorr_in, "ycorr_eval": ycorr_eval,
    }


# ==============================================================================
# Plotting
# ==============================================================================
def plot_curves(per_mode, head_mat, ce_clean, ce_corr, title, ylabel, path, full_recovery_color):
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    colors = {"layer": "#4f46e5", "attn": "#10b981", "mlp": "#ef4444"}
    layers = list(range(NLAYERS))
    gap = ce_corr - ce_clean
    for mode in MODES:
        frac = [_to_fraction(per_mode[mode][l], ce_clean, ce_corr, gap, title) for l in layers]
        ax.plot(layers, frac, marker='o', linewidth=2.3, color=colors[mode], label=f"{mode.upper()}")
    # mean head
    head_frac = [_to_fraction(head_mat[l].mean(), ce_clean, ce_corr, gap, title) for l in layers]
    ax.plot(layers, head_frac, marker='s', linewidth=2.0, color="#f59e0b",
            linestyle='--', label="HEAD (mean over heads)")
    ax.set_xlabel("Layer Patched", fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=12, pad=10)
    ax.set_xticks(layers)
    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.axhline(1, color=full_recovery_color, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def _to_fraction(patched_ce, ce_clean, ce_corr, gap, title):
    if abs(gap) < 1e-10:
        return 0.0
    if "Forward" in title:
        return (ce_corr - patched_ce) / gap      # recovery
    else:
        return (patched_ce - ce_clean) / gap      # degradation


def plot_component_heatmaps(fwd_modes, rev_modes, ce_clean, ce_corr, path):
    gap = ce_corr - ce_clean
    modes = ["attn", "mlp", "layer"]
    fwd = np.zeros((NLAYERS, len(modes)))
    rev = np.zeros((NLAYERS, len(modes)))
    for j, m in enumerate(modes):
        for l in range(NLAYERS):
            fwd[l, j] = (ce_corr - fwd_modes[m][l]) / gap if abs(gap) > 1e-10 else 0.0
            rev[l, j] = (rev_modes[m][l] - ce_clean) / gap if abs(gap) > 1e-10 else 0.0

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 8), dpi=200)
    for ax, mat, ttl, cmap in [(a1, fwd, "Forward: Recovery", "Blues"),
                               (a2, rev, "Reverse: Degradation", "Reds")]:
        vmin, vmax = mat.min(), mat.max()
        # give the colorbar a little headroom when all values are the same
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-3
        im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(ttl, fontweight='bold', fontsize=12)
        ax.set_xlabel("Component", fontweight='bold'); ax.set_ylabel("Layer", fontweight='bold')
        ax.set_xticks(range(len(modes))); ax.set_xticklabels([m.upper() for m in modes])
        ax.set_yticks(range(NLAYERS))
        mid = (vmin + vmax) / 2
        for l in range(NLAYERS):
            for j in range(len(modes)):
                ax.text(j, l, f"{mat[l, j]:.2f}", ha='center', va='center', fontsize=8,
                        color='white' if mat[l, j] > mid else 'black')
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_head_heatmap(mat_ce, ce_clean, ce_corr, forward, title, path):
    gap = ce_corr - ce_clean
    frac = np.zeros_like(mat_ce)
    for l in range(NLAYERS):
        for h in range(NHEAD):
            if abs(gap) < 1e-10:
                frac[l, h] = 0.0
            elif forward:
                frac[l, h] = (ce_corr - mat_ce[l, h]) / gap
            else:
                frac[l, h] = (mat_ce[l, h] - ce_clean) / gap
    fig, ax = plt.subplots(figsize=(6, 9), dpi=200)
    vmin, vmax = frac.min(), frac.max()
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-3
    mid = (vmin + vmax) / 2
    im = ax.imshow(frac, cmap='Blues' if forward else 'Reds', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xlabel("Head Index", fontweight='bold'); ax.set_ylabel("Layer Index", fontweight='bold')
    ax.set_xticks(range(NHEAD)); ax.set_yticks(range(NLAYERS))
    ax.set_title(title, fontweight='bold', fontsize=12)
    for l in range(NLAYERS):
        for h in range(NHEAD):
            ax.text(h, l, f"{frac[l, h]:.2f}", ha='center', va='center', fontsize=7,
                    color='white' if frac[l, h] > mid else 'black')
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


# ==============================================================================
def run_subexp(model, subexp, args, device):
    print(f"\n{'='*70}\nSUBEXPERIMENT: {subexp}\n{'='*70}")
    out_dir = os.path.join(RESULTS_DIR, f"patching_{subexp}")
    os.makedirs(out_dir, exist_ok=True)

    d = build_subexp_data(subexp, args.n_samples, args.seq_len)
    p, bs = args.single_eval_pos, args.batch_size

    # Baselines
    ce_clean = dataset_ce(model, d["x_clean"], d["yclean_in"], d["yclean_eval"], p, device, bs)
    ce_corr = dataset_ce(model, d["x_corr"], d["ycorr_in"], d["ycorr_eval"], p, device, bs)
    print(f"Baseline clean CE: {ce_clean:.6f} | corrupt CE: {ce_corr:.6f}")

    # ----- FORWARD: clean -> corrupt (source=clean acts, target run=corrupt) -----
    print("Forward patching (clean -> corrupt): layer/attn/mlp ...")
    fwd_modes = run_layerwise_patching(
        model, d["x_clean"], d["yclean_in"],
        d["x_corr"], d["ycorr_in"], d["ycorr_eval"], p, device, bs)
    print("Forward patching: per-head ...")
    fwd_head = run_headwise_patching(
        model, d["x_clean"], d["yclean_in"],
        d["x_corr"], d["ycorr_in"], d["ycorr_eval"], p, device, bs)

    # ----- REVERSE: corrupt -> clean (source=corrupt acts, target run=clean) -----
    print("Reverse patching (corrupt -> clean): layer/attn/mlp ...")
    rev_modes = run_layerwise_patching(
        model, d["x_corr"], d["ycorr_in"],
        d["x_clean"], d["yclean_in"], d["yclean_eval"], p, device, bs)
    print("Reverse patching: per-head ...")
    rev_head = run_headwise_patching(
        model, d["x_corr"], d["ycorr_in"],
        d["x_clean"], d["yclean_in"], d["yclean_eval"], p, device, bs)

    # ----- Plots -----
    plot_curves(fwd_modes, fwd_head, ce_clean, ce_corr,
                "Forward Activation Patching: Restoring Clean Performance",
                "Recovery Fraction (1 = full recovery)",
                os.path.join(out_dir, "forward_recovery_curves.png"), "green")
    plot_curves(rev_modes, rev_head, ce_clean, ce_corr,
                "Reverse Activation Patching: Degrading Clean Performance",
                "Degradation Fraction (1 = full degradation)",
                os.path.join(out_dir, "reverse_degradation_curves.png"), "red")
    plot_component_heatmaps(fwd_modes, rev_modes, ce_clean, ce_corr,
                            os.path.join(out_dir, "component_heatmaps.png"))
    plot_head_heatmap(fwd_head, ce_clean, ce_corr, True,
                      "Per-Head Forward Patching: Recovery",
                      os.path.join(out_dir, "head_recovery_heatmap.png"))
    plot_head_heatmap(rev_head, ce_clean, ce_corr, False,
                      "Per-Head Reverse Patching: Degradation",
                      os.path.join(out_dir, "head_degradation_heatmap.png"))

    # ----- JSON -----
    summary = {
        "subexp": subexp,
        "config": vars(args),
        "baselines": {"clean_ce": ce_clean, "corrupt_ce": ce_corr},
        "forward_patched_ce": {m: fwd_modes[m] for m in MODES},
        "reverse_patched_ce": {m: rev_modes[m] for m in MODES},
        "forward_head_ce": fwd_head.tolist(),
        "reverse_head_ce": rev_head.tolist(),
    }
    with open(os.path.join(out_dir, f"patching_{subexp}_results.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Saved subexperiment '{subexp}' results to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=500)
    ap.add_argument("--single_eval_pos", type=int, default=450)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--subexp", type=str, default="all",
                    choices=["all", "shuffle_y", "noise", "out_prior", "oop_same_features"])
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = load_tabpfn(device)

    subexps = (["shuffle_y", "noise", "out_prior", "oop_same_features"]
               if args.subexp == "all" else [args.subexp])
    for se in subexps:
        run_subexp(model, se, args, device)

    print("\nExperiment 2 complete!")


if __name__ == "__main__":
    main()
