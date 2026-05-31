import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Ensure the parent directory is in path to import from the repository
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gp_pfn_sm.test_suites import (
    generate_id_low_noise,
    generate_id_high_noise,
    generate_near_ood,
    generate_far_ood,
    generate_real_world_ood
)
from gp_pfn_ood.pfn_hooks import PFNHookManager

def compute_fpr_at_95_tpr(labels, scores):
    """
    Compute False Positive Rate at 95% True Positive Rate.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Find the index where TPR is closest to or just above 95%
    idx = np.where(tpr >= 0.95)[0][0]
    return fpr[idx]

def extract_gradient_metrics(model, xs, ys, num_splits=15, single_eval_pos=80, device="cpu"):
    """
    Extract gradient-based metrics for a batch of context datasets.
    xs: (B, N, 1)
    ys: (B, N)
    """
    B, N, _ = xs.shape
    
    # We will accumulate metrics for each batch item
    metrics = {
        "grad_norm_mean": [],
        "grad_norm_std": [],
        "grad_cosine_mean": [],
        "grad_cosine_std": [],
        "grad_var_mean": []
    }
    
    # Ensure gradients are enabled for all model parameters
    for p in model.parameters():
        p.requires_grad = True
        
    # We only compute gradients with respect to the decoder parameters
    # to be extremely fast and avoid huge memory consumption
    decoder_params = list(model.decoder.parameters())
    
    for i in range(B):
        # Extract single batch item and repeat it num_splits times
        xs_i = xs[i:i+1].repeat(num_splits, 1, 1).to(device) # (num_splits, N, 1)
        ys_i = ys[i:i+1].repeat(num_splits, 1).to(device)    # (num_splits, N)
        
        # Apply random permutations to create different train/query splits
        perms = torch.stack([torch.randperm(N, device=device) for _ in range(num_splits)], dim=0)
        xs_perm = torch.gather(xs_i, 1, perms.unsqueeze(-1))
        ys_perm = torch.gather(ys_i, 1, perms)
        
        # Forward pass
        src_x = xs_perm.transpose(0, 1)
        src_y = ys_perm.transpose(0, 1)
        
        # Model forward
        logits = model((src_x, src_y), single_eval_pos=single_eval_pos) # (N - single_eval_pos, num_splits, 1)
        targets = src_y[single_eval_pos:].unsqueeze(-1)
        
        # Loss per split
        loss = torch.mean((logits - targets) ** 2, dim=0).squeeze(-1) # (num_splits,)
        
        # Calculate gradients for each split
        split_grads = []
        for j in range(num_splits):
            model.zero_grad()
            # Backward pass (retain graph until the last split)
            loss[j].backward(retain_graph=(j < num_splits - 1))
            
            # Extract flat gradient vector
            grad_vec = torch.cat([p.grad.flatten() for p in decoder_params if p.grad is not None])
            split_grads.append(grad_vec.detach().cpu())
            
        # Convert list of gradients to a tensor: shape (num_splits, D)
        split_grads = torch.stack(split_grads) # (num_splits, D)
        
        # Compute gradient norms
        norms = torch.norm(split_grads, p=2, dim=1) # (num_splits,)
        metrics["grad_norm_mean"].append(norms.mean().item())
        metrics["grad_norm_std"].append(norms.std().item())
        
        # Compute cosine similarities between all pairs of splits
        # Normalize gradients to unit vectors
        normed_grads = split_grads / (norms.unsqueeze(1) + 1e-8)
        # Compute cosine similarity matrix
        cos_sim_matrix = torch.mm(normed_grads, normed_grads.t()) # (num_splits, num_splits)
        # Extract upper triangular values (excluding diagonal)
        triu_indices = torch.triu_indices(num_splits, num_splits, offset=1)
        cos_sims = cos_sim_matrix[triu_indices[0], triu_indices[1]]
        
        metrics["grad_cosine_mean"].append(cos_sims.mean().item())
        metrics["grad_cosine_std"].append(cos_sims.std().item())
        
        # Compute coordinate-wise variance
        var_coords = torch.var(split_grads, dim=0) # (D,)
        metrics["grad_var_mean"].append(var_coords.mean().item())
        
    # Convert metrics to numpy arrays
    for k in metrics:
        metrics[k] = np.array(metrics[k])
        
    return metrics

def run_experiment():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "gp_pfn_sm/checkpoints/spectral_mixture.pt"
    
    # Initialize PFN Hook Manager and model
    hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, device=device)
    model = hook_manager.model
    
    # Define test suites to generate data
    # We use a batch size of B=100 for each suite
    B = 100
    test_suites = {
        "Suite 1 (ID Low Noise)": generate_id_low_noise,
        "Suite 2 (ID High Noise)": generate_id_high_noise,
        "Suite 3 (Near-OOD)": generate_near_ood,
        "Suite 4 (Far-OOD)": generate_far_ood,
        "Suite 5 (Real-World OOD)": generate_real_world_ood
    }
    
    suite_data = {}
    print("\n--- Extracting Gradient Updates across Test Suites ---")
    for name, generator_fn in test_suites.items():
        print(f"Generating data and running backprop for {name}...")
        xs, ys = generator_fn(batch_size=B, device="cpu")
        
        # Get gradient metrics
        grad_metrics = extract_gradient_metrics(model, xs, ys, num_splits=15, single_eval_pos=80, device=device)
        
        # Also compute prediction MSE baseline on query points (80 to 100)
        with torch.no_grad():
            src_x = xs.transpose(0, 1).to(device)
            src_y = ys.transpose(0, 1).to(device)
            logits = model((src_x, src_y), single_eval_pos=80)
            targets = src_y[80:].unsqueeze(-1)
            prediction_mse = torch.mean((logits - targets) ** 2, dim=[0, 2]).cpu().numpy()
            
        # Compute combined magnitude-coherence product metric
        # OOD Score = grad_norm_mean * grad_cosine_mean
        magnitude_coherence = grad_metrics["grad_norm_mean"] * grad_metrics["grad_cosine_mean"]
            
        suite_data[name] = {
            "prediction_mse": prediction_mse,
            "grad_magnitude_coherence": magnitude_coherence,
            **grad_metrics
        }
        
    # Save the raw results
    results_dir = "backprop_gp_pfn"
    os.makedirs(results_dir, exist_ok=True)
    
    # ------------------ Plotting distributions of gradient metrics ------------------
    print("\nGenerating boxplots of metrics...")
    metric_keys = ["prediction_mse", "grad_norm_mean", "grad_cosine_mean", "grad_magnitude_coherence"]
    metric_titles = [
        "Prediction MSE (Baseline)",
        "Gradient Norm Mean (Magnitude)",
        "Gradient Cosine Similarity Mean (Coherence)",
        "Magnitude-Coherence Product (Ours)"
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ["#475569", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
    suite_names = list(test_suites.keys())
    
    for idx, (key, title) in enumerate(zip(metric_keys, metric_titles)):
        ax = axes[idx]
        plot_data = [suite_data[suite][key] for suite in suite_names]
        
        box = ax.boxplot(plot_data, patch_artist=True, tick_labels=[s.replace("Suite ", "") for s in suite_names])
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gradient_metrics_distributions.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # ------------------ Evaluate Pairwise OOD Detection (ID Low Noise as Negative) ------------------
    print("\nEvaluating pairwise OOD detection metrics (Negative: ID Low Noise)...")
    
    neg_suite_pure = "Suite 1 (ID Low Noise)"
    roc_tasks = [
        ("Suite 2 (ID High Noise)", "Robustness to Noise (Target: Chance)"),
        ("Suite 3 (Near-OOD)", "Near-OOD Detection"),
        ("Suite 4 (Far-OOD)", "Far-OOD Detection"),
        ("Suite 5 (Real-World OOD)", "Real-World OOD Detection")
    ]
    
    evaluation_results_pure = []
    
    # We will plot ROC curves for pure evaluation
    fig_roc, axes_roc = plt.subplots(2, 2, figsize=(12, 11), dpi=300)
    axes_roc = axes_roc.flatten()
    
    indicators = {
        "Prediction MSE (Baseline)": ("prediction_mse", '#cbd5e1', '-'),
        "Gradient Norm Mean": ("grad_norm_mean", '#3b82f6', '-'),
        "Gradient Cosine Similarity": ("grad_cosine_mean", '#f59e0b', '-'),
        "Magnitude-Coherence Product (Ours)": ("grad_magnitude_coherence", '#ef4444', '-')
    }
    
    for idx, (pos_suite, task_name) in enumerate(roc_tasks):
        ax = axes_roc[idx]
        
        for name_ind, (key_ind, color_ind, style_ind) in indicators.items():
            neg_scores = suite_data[neg_suite_pure][key_ind]
            pos_scores = suite_data[pos_suite][key_ind]
            
            # Since high cosine similarity implies ID and low similarity implies OOD,
            # we negate the scores for OOD evaluation so that higher scores mean OOD.
            # However, for the product metric, higher is OOD, so no negation is needed.
            if key_ind == "grad_cosine_mean":
                neg_scores = -neg_scores
                pos_scores = -pos_scores
                
            labels = np.concatenate([np.zeros_like(neg_scores), np.ones_like(pos_scores)])
            scores = np.concatenate([neg_scores, pos_scores])
            
            fpr, tpr, _ = roc_curve(labels, scores)
            auc_roc = auc(fpr, tpr)
            
            prec, rec, _ = precision_recall_curve(labels, scores)
            auc_pr = auc(rec, prec)
            
            fpr95 = compute_fpr_at_95_tpr(labels, scores)
            
            evaluation_results_pure.append({
                "Task": pos_suite.replace("Suite ", ""),
                "Indicator": name_ind,
                "AUROC": auc_roc,
                "AUPR": auc_pr,
                "FPR@95": fpr95
            })
            
            ax.plot(fpr, tpr, color=color_ind, linestyle=style_ind, linewidth=2, label=f"{name_ind} (AUC = {auc_roc:.3f})")
            
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
        ax.set_title(task_name, fontweight='bold', fontsize=12)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=8.5)
        ax.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gradient_metrics_roc_curves_pure.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # ------------------ Evaluate Robust Mixed OOD Detection (Negative: ID Low + ID High Noise) ------------------
    print("\nEvaluating robust mixed OOD detection metrics (Negative: ID Low + ID High Noise)...")
    
    evaluation_results_mixed = []
    
    # We will plot ROC curves for mixed evaluation
    fig_roc_m, axes_roc_m = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    mixed_roc_tasks = [
        ("Suite 3 (Near-OOD)", "Near-OOD vs Mixed ID", axes_roc_m[0]),
        ("Suite 4 (Far-OOD)", "Far-OOD vs Mixed ID", axes_roc_m[1]),
        ("Suite 5 (Real-World OOD)", "Real-World OOD vs Mixed ID", axes_roc_m[2])
    ]
    
    for pos_suite, task_name, ax in mixed_roc_tasks:
        for name_ind, (key_ind, color_ind, style_ind) in indicators.items():
            # Union of ID Low and ID High Noise
            neg_scores = np.concatenate([
                suite_data["Suite 1 (ID Low Noise)"][key_ind],
                suite_data["Suite 2 (ID High Noise)"][key_ind]
            ])
            pos_scores = suite_data[pos_suite][key_ind]
            
            if key_ind == "grad_cosine_mean":
                neg_scores = -neg_scores
                pos_scores = -pos_scores
                
            labels = np.concatenate([np.zeros_like(neg_scores), np.ones_like(pos_scores)])
            scores = np.concatenate([neg_scores, pos_scores])
            
            fpr, tpr, _ = roc_curve(labels, scores)
            auc_roc = auc(fpr, tpr)
            
            prec, rec, _ = precision_recall_curve(labels, scores)
            auc_pr = auc(rec, prec)
            
            fpr95 = compute_fpr_at_95_tpr(labels, scores)
            
            evaluation_results_mixed.append({
                "Task": pos_suite.replace("Suite ", ""),
                "Indicator": name_ind,
                "AUROC": auc_roc,
                "AUPR": auc_pr,
                "FPR@95": fpr95
            })
            
            ax.plot(fpr, tpr, color=color_ind, linestyle=style_ind, linewidth=2.2, label=f"{name_ind} (AUC = {auc_roc:.3f})")
            
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
        ax.set_title(task_name, fontweight='bold', fontsize=12)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gradient_metrics_roc_curves_mixed.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save the numerical results as CSV tables
    df_pure = pd.DataFrame(evaluation_results_pure)
    df_pure.to_csv(os.path.join(results_dir, "backprop_experiment_metrics_pure.csv"), index=False)
    
    df_mixed = pd.DataFrame(evaluation_results_mixed)
    df_mixed.to_csv(os.path.join(results_dir, "backprop_experiment_metrics_mixed.csv"), index=False)
    
    # Save raw suite data summary
    raw_data_summary = []
    for name in suite_names:
        for k in ["prediction_mse", "grad_norm_mean", "grad_cosine_mean", "grad_magnitude_coherence"]:
            vals = suite_data[name][k]
            raw_data_summary.append({
                "Suite": name.replace("Suite ", ""),
                "Metric": k,
                "Mean": np.mean(vals),
                "Std": np.std(vals),
                "Min": np.min(vals),
                "Max": np.max(vals)
            })
    df_raw = pd.DataFrame(raw_data_summary)
    df_raw.to_csv(os.path.join(results_dir, "raw_metrics_summary.csv"), index=False)
    
    # Print tables to console
    print("\n" + "="*80)
    print("                      RAW METRIC STATISTICS BY TEST SUITE")
    print("="*80)
    print(df_raw.to_string(index=False))
    print("="*80)

    print("\n" + "="*80)
    print("           PAIRWISE EVALUATION METRICS (NEGATIVE: ID LOW NOISE ONLY)")
    print("="*80)
    print(df_pure.to_string(index=False))
    print("="*80)

    print("\n" + "="*80)
    print("         MIXED ID EVALUATION METRICS (NEGATIVE: ID LOW + ID HIGH NOISE)")
    print("="*80)
    print(df_mixed.to_string(index=False))
    print("="*80)
    print(f"\nAll plots and metrics saved in '{results_dir}/' folder.")

if __name__ == "__main__":
    run_experiment()
