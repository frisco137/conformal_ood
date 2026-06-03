"""
sanity_oop_same_features.py — sanity check for the `oop_same_features` corruption
=================================================================================

Verifies that the "out-of-prior, same features" corruption (Experiment 2,
subexperiment 4) behaves as intended:

  * it is genuinely HARDER for TabPFN than the clean in-prior table it is paired
    with (lower query accuracy) -> evidence the labeling is out-of-prior, and
  * it is still LEARNABLE in-context (accuracy well above chance = 1/num_classes)
    -> evidence the DAG labeling is a predictable function of the features.

Procedure: draw `--n_tables` tables (each a fresh clean in-prior table + its
matched DAG-corrupted labeling, a NEW random DAG per table). Run TabPFN on both
the clean and corrupted version of every table and record the query accuracy.
The two accuracy distributions are drawn as a quantile candlestick (same OHLC
convention used elsewhere in the repo: Low=10th, Open=25th, median=50th,
Close=75th, High=90th).

Output (results/):
  - oop_same_features_sanity_candlestick.png
  - oop_same_features_sanity.json
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from tabpfn_hooks import (
    load_tabpfn, make_oop_same_features, compute_ce_loss, N_CLASSES,
)

plt.rcParams.update({
    'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def plot_candlestick(ax, x_pos, data, color, label):
    """Quantile candlestick (repo convention): Low=10th, Open=25th, median=50th,
    Close=75th, High=90th."""
    q10, q25, q50, q75, q90 = np.percentile(data, [10, 25, 50, 75, 90])
    ax.plot([x_pos, x_pos], [q10, q90], color='black', linewidth=1.5, zorder=2)
    rect = plt.Rectangle((x_pos - 0.2, q25), 0.4, q75 - q25, facecolor=color,
                         edgecolor='black', label=label, zorder=3)
    ax.add_patch(rect)
    ax.plot([x_pos - 0.2, x_pos + 0.2], [q50, q50], color='black', linewidth=1.5, zorder=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tables", type=int, default=100)
    ap.add_argument("--seq_len", type=int, default=500)
    ap.add_argument("--single_eval_pos", type=int, default=450)
    ap.add_argument("--n_nodes", type=int, default=None,
                    help="DAG node count (default 3*num_features inside the generator)")
    ap.add_argument("--edge_prob", type=float, default=0.1)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = load_tabpfn(device)
    p = args.single_eval_pos

    clean_acc, corr_acc = [], []
    print(f"Generating + evaluating {args.n_tables} tables "
          f"(seq_len={args.seq_len}, ctx={p}, query={args.seq_len - p})...")
    for t in tqdm(range(args.n_tables), desc="tables"):
        # Fresh clean table + a NEW random DAG corruption (seed varies per table).
        x_clean, y_clean, y_corr = make_oop_same_features(
            1, args.seq_len, n_nodes=args.n_nodes, edge_prob=args.edge_prob,
            device=device, seed=t)
        # Clean: in-prior labels.  Corrupt: same X, DAG labels (fed AND evaluated).
        _, acc_c = compute_ce_loss(model, x_clean, y_clean, device, p, return_acc=True)
        _, acc_x = compute_ce_loss(model, x_clean, y_corr, device, p, return_acc=True)
        clean_acc.append(100.0 * acc_c)
        corr_acc.append(100.0 * acc_x)

    clean_acc = np.array(clean_acc)
    corr_acc = np.array(corr_acc)
    chance = 100.0 / N_CLASSES

    print(f"\nClean  accuracy: mean {clean_acc.mean():.2f}%  median {np.median(clean_acc):.2f}%")
    print(f"Corrupt accuracy: mean {corr_acc.mean():.2f}%  median {np.median(corr_acc):.2f}%")
    print(f"Chance level: {chance:.2f}%")

    # ----- Candlestick -----
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    plot_candlestick(ax, 1, clean_acc, color='#3b82f6', label='Clean (in-prior)')
    plot_candlestick(ax, 2, corr_acc, color='#ef4444', label='Corrupt (DAG, same X)')
    ax.axhline(chance, color='grey', linestyle='--', alpha=0.7,
               label=f'Chance ({chance:.0f}%)')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Clean\n(in-prior)', 'Corrupt\n(oop_same_features)'])
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(max(0, min(clean_acc.min(), corr_acc.min()) - 5), 105)
    ax.set_ylabel("TabPFN Query Accuracy (%)", fontweight='bold')
    ax.set_title("oop_same_features sanity check\n"
                 "(Low=10th, Open=25th, mid=50th, Close=75th, High=90th)",
                 fontweight='bold', fontsize=12)
    ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "oop_same_features_sanity_candlestick.png")
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

    # ----- JSON -----
    def stats(a):
        q10, q25, q50, q75, q90 = [float(np.percentile(a, q)) for q in (10, 25, 50, 75, 90)]
        return {"mean": float(a.mean()), "std": float(a.std()),
                "q10": q10, "q25": q25, "median": q50, "q75": q75, "q90": q90,
                "min": float(a.min()), "max": float(a.max())}

    summary = {
        "config": vars(args),
        "chance_pct": chance,
        "clean_accuracy_pct": stats(clean_acc),
        "corrupt_accuracy_pct": stats(corr_acc),
        "clean_accuracy_all": clean_acc.tolist(),
        "corrupt_accuracy_all": corr_acc.tolist(),
    }
    with open(os.path.join(RESULTS_DIR, "oop_same_features_sanity.json"), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
