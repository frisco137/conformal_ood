import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import umap

# Ensure the parent directory is in path to import from the repository
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransformersCanDoBayesianInference')))

from gp_pfn_sm.test_suites import (
    generate_id_low_noise,
    generate_id_high_noise,
    generate_near_ood,
    generate_far_ood,
    generate_real_world_ood
)
from gp_pfn_sm.ood_metric_study import compute_mahalanobis
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
    """
    B, N, _ = xs.shape
    
    metrics = {
        "grad_norm_mean": [],
        "grad_cosine_mean": [],
    }
    
    for p in model.parameters():
        p.requires_grad = True
        
    decoder_params = list(model.decoder.parameters())
    
    for i in range(B):
        # Extract single batch item and repeat it num_splits times
        xs_i = xs[i:i+1].repeat(num_splits, 1, 1).to(device)
        ys_i = ys[i:i+1].repeat(num_splits, 1).to(device)
        
        perms = torch.stack([torch.randperm(N, device=device) for _ in range(num_splits)], dim=0)
        xs_perm = torch.gather(xs_i, 1, perms.unsqueeze(-1))
        ys_perm = torch.gather(ys_i, 1, perms)
        
        src_x = xs_perm.transpose(0, 1)
        src_y = ys_perm.transpose(0, 1)
        
        logits = model((src_x, src_y), single_eval_pos=single_eval_pos)
        targets = src_y[single_eval_pos:].unsqueeze(-1)
        
        loss = torch.mean((logits - targets) ** 2, dim=0).squeeze(-1)
        
        split_grads = []
        for j in range(num_splits):
            model.zero_grad()
            loss[j].backward(retain_graph=(j < num_splits - 1))
            
            grad_vec = torch.cat([p.grad.flatten() for p in decoder_params if p.grad is not None])
            split_grads.append(grad_vec.detach().cpu())
            
        split_grads = torch.stack(split_grads)
        
        norms = torch.norm(split_grads, p=2, dim=1)
        metrics["grad_norm_mean"].append(norms.mean().item())
        
        normed_grads = split_grads / (norms.unsqueeze(1) + 1e-8)
        cos_sim_matrix = torch.mm(normed_grads, normed_grads.t())
        triu_indices = torch.triu_indices(num_splits, num_splits, offset=1)
        cos_sims = cos_sim_matrix[triu_indices[0], triu_indices[1]]
        metrics["grad_cosine_mean"].append(cos_sims.mean().item())
        
    for k in metrics:
        metrics[k] = np.array(metrics[k])
        
    return metrics

def run_comparison():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "gp_pfn_sm/checkpoints/spectral_mixture.pt"
    
    # Initialize PFN Hook Manager and model
    hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, device=device)
    model = hook_manager.model
    
    print("\n[Step 1] Fitting PCA & UMAP on activations...")
    # Generate Calibration and Reference fitting sets
    x_cal, y_cal = generate_id_low_noise(batch_size=1000, device="cpu")
    x_near_ref, y_near_ref = generate_near_ood(batch_size=500, device="cpu")
    x_far_ref, y_far_ref = generate_far_ood(batch_size=500, device="cpu")
    
    with torch.no_grad():
        src_x = x_cal.transpose(0, 1).to(device)
        src_y = y_cal.transpose(0, 1).to(device)
        _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
    layer_activations, _ = hook_manager.get_aggregated_features()
    X_cal = layer_activations[5].numpy()
    hook_manager.clear_cache()
    
    with torch.no_grad():
        src_x = x_near_ref.transpose(0, 1).to(device)
        src_y = y_near_ref.transpose(0, 1).to(device)
        _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
    layer_activations, _ = hook_manager.get_aggregated_features()
    X_near_ref = layer_activations[5].numpy()
    hook_manager.clear_cache()
    
    with torch.no_grad():
        src_x = x_far_ref.transpose(0, 1).to(device)
        src_y = y_far_ref.transpose(0, 1).to(device)
        _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
    layer_activations, _ = hook_manager.get_aggregated_features()
    X_far_ref = layer_activations[5].numpy()
    hook_manager.clear_cache()
    
    X_fit = np.concatenate([X_cal, X_near_ref, X_far_ref], axis=0)
    
    pca_10d = PCA(n_components=10, random_state=42)
    pca_10d.fit(X_fit)
    X_cal_pca_10d = pca_10d.transform(X_cal)
    pca_10d_mean = np.mean(X_cal_pca_10d, axis=0)
    pca_10d_cov = np.cov(X_cal_pca_10d, rowvar=False)
    
    umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_2d.fit(X_fit)
    X_cal_umap = umap_2d.transform(X_cal)
    umap_mean = np.mean(X_cal_umap, axis=0)
    umap_cov = np.cov(X_cal_umap, rowvar=False)
    
    print("\n[Step 2] Processing test suites and extracting metrics...")
    B = 200  # Batch size of 200 for all suites to get very stable evaluation
    test_suites = {
        "Suite 1 (ID Low Noise)": generate_id_low_noise,
        "Suite 2 (ID High Noise)": generate_id_high_noise,
        "Suite 3 (Near-OOD)": generate_near_ood,
        "Suite 4 (Far-OOD)": generate_far_ood,
        "Suite 5 (Real-World OOD)": generate_real_world_ood
    }
    
    suite_data = {}
    for name, generator_fn in test_suites.items():
        print(f"Generating data and computing all baselines for {name}...")
        xs, ys = generator_fn(batch_size=B, device="cpu")
        
        # 1. Representational features
        with torch.no_grad():
            src_x = xs.transpose(0, 1).to(device)
            src_y = ys.transpose(0, 1).to(device)
            _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
            
        layer_activations, _ = hook_manager.get_aggregated_features()
        hook_manager.clear_cache()
        
        X_test_layer = layer_activations[5].numpy()
        X_test_umap = umap_2d.transform(X_test_layer)
        X_test_pca_10d = pca_10d.transform(X_test_layer)
        
        d_m_umap = compute_mahalanobis(X_test_umap, umap_mean, umap_cov)
        d_m_pca = compute_mahalanobis(X_test_pca_10d, pca_10d_mean, pca_10d_cov)
        
        # 2. Prediction MSE (eval pos 50 and 80)
        with torch.no_grad():
            logits_50 = hook_manager.model((src_x, src_y), single_eval_pos=50)
            targets_50 = src_y[50:].unsqueeze(-1)
            mse_50 = torch.mean((logits_50 - targets_50) ** 2, dim=[0, 2]).cpu().numpy()
            
            logits_80 = hook_manager.model((src_x, src_y), single_eval_pos=80)
            targets_80 = src_y[80:].unsqueeze(-1)
            mse_80 = torch.mean((logits_80 - targets_80) ** 2, dim=[0, 2]).cpu().numpy()
        hook_manager.clear_cache()
        
        # 3. Backprop gradients
        grad_metrics = extract_gradient_metrics(model, xs, ys, num_splits=15, single_eval_pos=80, device=device)
        magnitude_coherence = grad_metrics["grad_norm_mean"] * grad_metrics["grad_cosine_mean"]
        
        suite_data[name] = {
            "umap_mahalanobis": d_m_umap,
            "pca_mahalanobis": d_m_pca,
            "prediction_mse_50": mse_50,
            "prediction_mse_80": mse_80,
            "grad_norm_mean": grad_metrics["grad_norm_mean"],
            "grad_cosine_mean": grad_metrics["grad_cosine_mean"],
            "grad_magnitude_coherence": magnitude_coherence
        }
        
    hook_manager.remove_hooks()
    
    # ------------------ Evaluation Setup ------------------
    results_dir = "backprop_gp_pfn"
    os.makedirs(results_dir, exist_ok=True)
    
    indicators = {
        "UMAP-2D Mahalanobis": "umap_mahalanobis",
        "PCA-10D Mahalanobis": "pca_mahalanobis",
        "Prediction MSE (50 Qs)": "prediction_mse_50",
        "Prediction MSE (20 Qs)": "prediction_mse_80",
        "Gradient Norm Mean": "grad_norm_mean",
        "Gradient Cosine Similarity": "grad_cosine_mean",
        "Magnitude-Coherence Product": "grad_magnitude_coherence"
    }
    
    # --- TABLE A: Pairwise Evaluation (Negative: ID Low Noise) ---
    print("\nComputing Table A: Pairwise Evaluation (Negative: ID Low Noise)...")
    pairwise_tasks = [
        ("Suite 2 (ID High Noise)", "Robustness to Noise"),
        ("Suite 3 (Near-OOD)", "Near-OOD Detection"),
        ("Suite 4 (Far-OOD)", "Far-OOD Detection"),
        ("Suite 5 (Real-World OOD)", "Real-World OOD Detection")
    ]
    
    pairwise_rows = []
    neg_suite = "Suite 1 (ID Low Noise)"
    
    for task_suite, task_name in pairwise_tasks:
        for ind_name, key in indicators.items():
            neg_scores = suite_data[neg_suite][key]
            pos_scores = suite_data[task_suite][key]
            
            # Negate cosine similarity because higher means ID
            if key == "grad_cosine_mean":
                neg_scores = -neg_scores
                pos_scores = -pos_scores
                
            labels = np.concatenate([np.zeros_like(neg_scores), np.ones_like(pos_scores)])
            scores = np.concatenate([neg_scores, pos_scores])
            
            fpr, tpr, _ = roc_curve(labels, scores)
            auc_roc = auc(fpr, tpr)
            
            prec, rec, _ = precision_recall_curve(labels, scores)
            auc_pr = auc(rec, prec)
            
            fpr95 = compute_fpr_at_95_tpr(labels, scores)
            
            pairwise_rows.append({
                "Evaluation Task": task_name,
                "Indicator": ind_name,
                "AUROC": auc_roc,
                "AUPR": auc_pr,
                "FPR@95": fpr95
            })
            
    df_pairwise = pd.DataFrame(pairwise_rows)
    df_pairwise.to_csv(os.path.join(results_dir, "comparison_table_pairwise.csv"), index=False)
    
    # --- TABLE B: Robust Mixed ID Evaluation (Negative: ID Low + ID High Noise) ---
    print("\nComputing Table B: Mixed ID Evaluation (Negative: ID Low + ID High Noise)...")
    mixed_tasks = [
        ("Suite 3 (Near-OOD)", "Near-OOD Detection"),
        ("Suite 4 (Far-OOD)", "Far-OOD Detection"),
        ("Suite 5 (Real-World OOD)", "Real-World OOD Detection")
    ]
    
    mixed_rows = []
    
    for task_suite, task_name in mixed_tasks:
        for ind_name, key in indicators.items():
            # Combine Low and High Noise ID
            neg_scores = np.concatenate([
                suite_data["Suite 1 (ID Low Noise)"][key],
                suite_data["Suite 2 (ID High Noise)"][key]
            ])
            pos_scores = suite_data[task_suite][key]
            
            if key == "grad_cosine_mean":
                neg_scores = -neg_scores
                pos_scores = -pos_scores
                
            labels = np.concatenate([np.zeros_like(neg_scores), np.ones_like(pos_scores)])
            scores = np.concatenate([neg_scores, pos_scores])
            
            fpr, tpr, _ = roc_curve(labels, scores)
            auc_roc = auc(fpr, tpr)
            
            prec, rec, _ = precision_recall_curve(labels, scores)
            auc_pr = auc(rec, prec)
            
            fpr95 = compute_fpr_at_95_tpr(labels, scores)
            
            mixed_rows.append({
                "Evaluation Task": task_name,
                "Indicator": ind_name,
                "AUROC": auc_roc,
                "AUPR": auc_pr,
                "FPR@95": fpr95
            })
            
    df_mixed = pd.DataFrame(mixed_rows)
    df_mixed.to_csv(os.path.join(results_dir, "comparison_table_mixed.csv"), index=False)
    
    # ------------------ Print Formatted Results ------------------
    print("\n" + "="*90)
    print("           TABLE A: PAIRWISE OOD DETECTION (NEGATIVE: ID LOW NOISE ONLY)")
    print("="*90)
    # Pivot for clean comparison
    pivoted_pairwise = df_pairwise.pivot(index="Evaluation Task", columns="Indicator", values=["AUROC", "FPR@95"])
    print(pivoted_pairwise.to_string())
    print("="*90)
    
    print("\n" + "="*90)
    print("         TABLE B: ROBUST MIXED ID DETECTION (NEGATIVE: ID LOW + ID HIGH NOISE)")
    print("="*90)
    pivoted_mixed = df_mixed.pivot(index="Evaluation Task", columns="Indicator", values=["AUROC", "FPR@95"])
    print(pivoted_mixed.to_string())
    print("="*90)
    
    # Save a comparison markdown report
    with open(os.path.join(results_dir, "comparison_report.md"), "w") as f:
        f.write("# baseline vs backprop comparison\n\n")
        f.write("## Table A: Pairwise Evaluation (Negative: ID Low Noise)\n\n")
        f.write(df_pairwise.to_markdown(index=False))
        f.write("\n\n## Table B: Robust Mixed ID Evaluation (Negative: ID Low + ID High Noise)\n\n")
        f.write(df_mixed.to_markdown(index=False))
        
    print(f"\nAll comparison files saved in '{results_dir}/' folder.")

if __name__ == "__main__":
    run_comparison()
