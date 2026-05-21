import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add PFN repo to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransformersCanDoBayesianInference')))

import transformer
import bar_distribution

# Exact GP implementation in PyTorch for comparison
def gp_posterior(X_train, y_train, X_test, lengthscale=0.1, outputscale=1.0, noise=1e-4):
    """
    Computes the exact GP posterior mean and standard deviation at X_test.
    """
    # Helper to calculate squared pairwise distance
    def sq_dist(A, B):
        return torch.cdist(A, B) ** 2

    # Covariance matrices
    K_XX = outputscale * torch.exp(-0.5 * sq_dist(X_train, X_train) / (lengthscale ** 2)) + noise * torch.eye(len(X_train), device=X_train.device)
    K_XXs = outputscale * torch.exp(-0.5 * sq_dist(X_train, X_test) / (lengthscale ** 2))
    K_XsXs = outputscale * torch.exp(-0.5 * sq_dist(X_test, X_test) / (lengthscale ** 2))
    
    # Solve for mean
    L = torch.linalg.cholesky(K_XX)
    y_train = y_train.view(-1, 1)
    beta = torch.linalg.solve_triangular(L, y_train, upper=False)
    alpha = torch.linalg.solve_triangular(L.T, beta, upper=True)
    post_mean = (K_XXs.T @ alpha).squeeze(-1)
    
    # Solve for variance
    v = torch.linalg.solve_triangular(L, K_XXs, upper=False)
    post_var = torch.diagonal(K_XsXs) - torch.sum(v ** 2, dim=0)
    post_var = torch.clamp(post_var, min=1e-8) + noise
    post_std = post_var.sqrt()
    
    return post_mean, post_std

def generate_gp_data(num_points, lengthscale=0.1, outputscale=1.0, noise=1e-4, device='cpu'):
    """
    Generates synthetic training and test data from a GP prior.
    """
    # Sample points in [0, 1]
    X = torch.rand(num_points, 1, device=device)
    # Compute covariance
    dist_mat = torch.cdist(X, X) ** 2
    K = outputscale * torch.exp(-0.5 * dist_mat / (lengthscale ** 2)) + noise * torch.eye(num_points, device=device)
    L = torch.linalg.cholesky(K)
    eps = torch.randn(num_points, 1, device=device)
    y = (L @ eps).squeeze(-1)
    return X, y

@torch.no_grad()
def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Checkpoint choices
    choices = ['160K', '800K', '4M']
    models = {}
    
    # Load all models
    for choice in choices:
        ckpt_path = f'parameter_free_ood/checkpoints/onefeature_gp_ls.1_pnf_{choice}.pt'
        print(f"Loading PFN model: {ckpt_path}...")
        model = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Fix PyTorch version compatibility
        for m in model.modules():
            if m.__class__.__name__ == 'TransformerEncoderLayer':
                if not hasattr(m, 'norm_first'):
                    m.norm_first = False
            elif m.__class__.__name__ == 'GELU':
                if not hasattr(m, 'approximate'):
                    m.approximate = 'none'
        model.eval()
        model.to(device)
        models[choice] = model
    
    # GP Hyperparameters (matching training settings)
    gp_hps = {'noise': 1e-4, 'outputscale': 1.0, 'lengthscale': 0.1}
    
    # Let's generate a single GP dataset for visualization
    # We will use N = 4 training points (just like in the interactive space)
    torch.manual_seed(42) # Set seed for reproducibility of the plot
    X_train, y_train = generate_gp_data(4, lengthscale=gp_hps['lengthscale'], outputscale=gp_hps['outputscale'], noise=gp_hps['noise'], device=device)
    
    # Grid of test points in [0, 1]
    X_test = torch.linspace(0, 1, 100, device=device).unsqueeze(1)
    
    # Get exact GP posterior
    gp_mean, gp_std = gp_posterior(X_train, y_train, X_test, **gp_hps)
    gp_mean_np = gp_mean.cpu().numpy()
    gp_std_np = gp_std.cpu().numpy()
    gp_lb = gp_mean_np - gp_std_np
    gp_ub = gp_mean_np + gp_std_np
    
    X_test_np = X_test.cpu().squeeze(-1).numpy()
    X_train_np = X_train.cpu().squeeze(-1).numpy()
    y_train_np = y_train.cpu().numpy()
    
    # Create beautiful comparison plots
    plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    pfn_results_viz = {}
    
    for idx, choice in enumerate(choices):
        ax = axes[idx]
        model = models[choice]
        
        # PFN inputs format: cat(X_train, X_test) unsqueezed batch, single_eval_pos=len(X_train)
        src_x = torch.cat([X_train, X_test], 0).unsqueeze(1)
        src_y = y_train.unsqueeze(1)
        
        # Run inference
        logits = model((src_x, src_y), single_eval_pos=len(X_train))
        
        # Get mean and bounds (68.2% quantile, i.e., 1-std equivalent)
        pfn_mean = model.criterion.mean(logits).squeeze(1).cpu().numpy()
        bounds = model.criterion.quantile(logits, center_prob=0.682).squeeze(1).cpu().numpy()
        pfn_lb = bounds[:, 0]
        pfn_ub = bounds[:, 1]
        
        # Plot exact GP posterior (green)
        ax.plot(X_test_np, gp_mean_np, color='#2ca02c', linestyle='--', linewidth=1.5, label='Exact GP Posterior Mean')
        ax.fill_between(X_test_np, gp_lb, gp_ub, color='#2ca02c', alpha=0.1, label='Exact GP 68.2% CI')
        
        # Plot PFN posterior (blue)
        ax.plot(X_test_np, pfn_mean, color='#1f77b4', linewidth=2.0, label=f'PFN-{choice} Predicted Mean')
        ax.fill_between(X_test_np, pfn_lb, pfn_ub, color='#1f77b4', alpha=0.15, label=f'PFN-{choice} 68.2% CI')
        
        # Plot training data points
        ax.scatter(X_train_np, y_train_np, color='black', s=60, zorder=5, label='Training Data')
        
        ax.set_title(f"GP Posterior vs. PFN-{choice} (4 training points)", fontsize=12, fontweight='bold')
        ax.set_ylabel("y")
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        ax.set_ylim(-3, 3)
        
    axes[-1].set_xlabel("x")
    plt.tight_layout()
    
    os.makedirs('parameter_free_ood/plots', exist_ok=True)
    plot_path = 'parameter_free_ood/plots/gp_pfn_inference_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved inference comparison plot to {plot_path}")
    
    # ------------------ Quantitative Evaluation (MSE) ------------------
    print("\nStarting quantitative MSE evaluation...")
    num_datasets = 100
    training_sizes = [2, 4, 8, 16]
    
    # We will compute MSE between PFN predicted mean and exact GP posterior mean
    # for different numbers of training points.
    mse_results = {choice: {size: [] for size in training_sizes} for choice in choices}
    
    # Set seed for evaluation data generation
    torch.manual_seed(12345)
    
    for d_idx in range(num_datasets):
        if (d_idx + 1) % 20 == 0:
            print(f"  Processed {d_idx + 1}/{num_datasets} datasets...")
            
        for size in training_sizes:
            # Generate dataset from GP prior
            # We generate (size + 100) points, first 'size' are training, next 100 are test points
            X_all, y_all = generate_gp_data(size + 100, lengthscale=gp_hps['lengthscale'], outputscale=gp_hps['outputscale'], noise=gp_hps['noise'], device=device)
            X_train_d = X_all[:size]
            y_train_d = y_all[:size]
            X_test_d = X_all[size:]
            y_test_d = y_all[size:]
            
            # Compute exact GP posterior mean at test points
            gp_mean_d, _ = gp_posterior(X_train_d, y_train_d, X_test_d, **gp_hps)
            
            for choice in choices:
                model = models[choice]
                
                # Format inputs
                src_x = torch.cat([X_train_d, X_test_d], 0).unsqueeze(1)
                src_y = y_train_d.unsqueeze(1)
                
                # Run inference
                logits = model((src_x, src_y), single_eval_pos=size)
                
                # PFN mean prediction
                pfn_mean_d = model.criterion.mean(logits).squeeze(1)
                
                # Compute MSE with exact GP mean
                mse_gp = torch.mean((pfn_mean_d - gp_mean_d) ** 2).item()
                mse_results[choice][size].append(mse_gp)
                
    # Compile table
    rows = []
    for choice in choices:
        row = {'Model': f"PFN-{choice}"}
        for size in training_sizes:
            avg_mse = np.mean(mse_results[choice][size])
            row[f"N={size} MSE"] = f"{avg_mse:.5f}"
        rows.append(row)
        
    df = pd.DataFrame(rows)
    print("\nMean Squared Error (MSE) compared to Exact GP Posterior Mean:")
    print(df.to_string(index=False))
    
    # Save the table to a CSV file
    csv_path = 'parameter_free_ood/plots/mse_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nMSE table saved to {csv_path}")

if __name__ == '__main__':
    main()
