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
    def sq_dist(A, B):
        return torch.cdist(A, B) ** 2

    K_XX = outputscale * torch.exp(-0.5 * sq_dist(X_train, X_train) / (lengthscale ** 2)) + noise * torch.eye(len(X_train), device=X_train.device)
    K_XXs = outputscale * torch.exp(-0.5 * sq_dist(X_train, X_test) / (lengthscale ** 2))
    K_XsXs = outputscale * torch.exp(-0.5 * sq_dist(X_test, X_test) / (lengthscale ** 2))
    
    L = torch.linalg.cholesky(K_XX)
    y_train = y_train.view(-1, 1)
    beta = torch.linalg.solve_triangular(L, y_train, upper=False)
    alpha = torch.linalg.solve_triangular(L.T, beta, upper=True)
    post_mean = (K_XXs.T @ alpha).squeeze(-1)
    
    v = torch.linalg.solve_triangular(L, K_XXs, upper=False)
    post_var = torch.diagonal(K_XsXs) - torch.sum(v ** 2, dim=0)
    post_var = torch.clamp(post_var, min=1e-8) + noise
    post_std = post_var.sqrt()
    
    return post_mean, post_std

def sample_gp_function(X_grid, lengthscale=0.1, outputscale=1.0, noise=1e-4, device='cpu'):
    """
    Samples a single continuous function over the specified grid from the GP prior.
    """
    dist_mat = torch.cdist(X_grid, X_grid) ** 2
    K = outputscale * torch.exp(-0.5 * dist_mat / (lengthscale ** 2)) + noise * torch.eye(len(X_grid), device=device)
    L = torch.linalg.cholesky(K)
    eps = torch.randn(len(X_grid), 1, device=device)
    y = (L @ eps).squeeze(-1)
    return y

@torch.no_grad()
def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the PFN-4M model
    ckpt_path = 'parameter_free_ood/checkpoints/onefeature_gp_ls.1_pnf_4M.pt'
    print(f"Loading model: {ckpt_path}...")
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    for m in model.modules():
        if m.__class__.__name__ == 'TransformerEncoderLayer':
            if not hasattr(m, 'norm_first'):
                m.norm_first = False
        elif m.__class__.__name__ == 'GELU':
            if not hasattr(m, 'approximate'):
                m.approximate = 'none'
    model.eval()
    model.to(device)
    
    # GP Hyperparameters
    gp_hps = {'noise': 1e-4, 'outputscale': 1.0, 'lengthscale': 0.1}
    
    # Create a regular grid of 100 points in [0, 1]
    X_grid = torch.linspace(0, 1, 100, device=device).unsqueeze(1)
    
    # Sample a function
    torch.manual_seed(42)
    y_grid = sample_gp_function(X_grid, **gp_hps, device=device)
    
    # Let's perform interpolation with varying N (number of observed training points)
    # We select N points from the grid to observe, and mask the remaining (100 - N) points
    N_values = [2, 4, 6, 8, 10, 12, 15, 20]
    
    # Prepare plotting
    fig, axes = plt.subplots(len(N_values), 1, figsize=(10, 3 * len(N_values)), sharex=True)
    
    results = []
    
    for idx, N in enumerate(N_values):
        ax = axes[idx]
        
        # Select N training indices regularly spaced along the grid
        train_indices = np.linspace(0, 99, N, dtype=int)
        test_indices = np.array([i for i in range(100) if i not in train_indices])
        
        X_train = X_grid[train_indices]
        y_train = y_grid[train_indices]
        X_test = X_grid[test_indices]
        y_test = y_grid[test_indices]
        
        # 1. Exact GP prediction
        gp_mean, gp_std = gp_posterior(X_train, y_train, X_test, **gp_hps)
        
        # 2. PFN-4M prediction
        # PFN format: cat(X_train, X_test) unsqueezed batch, single_eval_pos=len(X_train)
        src_x = torch.cat([X_train, X_test], 0).unsqueeze(1)
        src_y = y_train.unsqueeze(1)
        
        logits = model((src_x, src_y), single_eval_pos=len(X_train))
        pfn_mean = model.criterion.mean(logits).squeeze(1)
        pfn_bounds = model.criterion.quantile(logits, center_prob=0.682).squeeze(1)
        
        pfn_lb = pfn_bounds[:, 0]
        pfn_ub = pfn_bounds[:, 1]
        
        # Compute MSEs
        # MSE compared to the exact analytical GP posterior mean
        mse_vs_gp = torch.mean((pfn_mean - gp_mean) ** 2).item()
        # MSE compared to the actual true function values (y_test)
        mse_vs_true = torch.mean((pfn_mean - y_test) ** 2).item()
        
        # GP posterior MSE vs true values for comparison
        gp_mse_vs_true = torch.mean((gp_mean - y_test) ** 2).item()
        
        results.append({
            'N': N,
            'MSE_vs_GP': mse_vs_gp,
            'PFN_MSE_vs_True': mse_vs_true,
            'GP_MSE_vs_True': gp_mse_vs_true
        })
        
        # Convert to numpy for plotting
        X_grid_np = X_grid.cpu().squeeze(-1).numpy()
        y_grid_np = y_grid.cpu().numpy()
        X_train_np = X_train.cpu().squeeze(-1).numpy()
        y_train_np = y_train.cpu().numpy()
        X_test_np = X_test.cpu().squeeze(-1).numpy()
        pfn_mean_np = pfn_mean.cpu().numpy()
        pfn_lb_np = pfn_lb.cpu().numpy()
        pfn_ub_np = pfn_ub.cpu().numpy()
        gp_mean_np = gp_mean.cpu().numpy()
        
        # Plot true function
        ax.plot(X_grid_np, y_grid_np, color='gray', linestyle=':', label='True Function f(x)')
        # Plot observed points
        ax.scatter(X_train_np, y_train_np, color='black', s=50, zorder=5, label=f'Observed Points (N={N})')
        # Plot Exact GP Mean
        ax.plot(X_test_np, gp_mean_np, color='#2ca02c', linestyle='--', label='Exact GP Mean')
        # Plot PFN Mean & Bounds
        ax.plot(X_test_np, pfn_mean_np, color='#1f77b4', label='PFN Mean')
        ax.fill_between(X_test_np, pfn_lb_np, pfn_ub_np, color='#1f77b4', alpha=0.15, label='PFN 68.2% CI')
        
        ax.set_title(f"Interpolation with N = {N} (MSE vs Exact GP: {mse_vs_gp:.6f})", fontsize=11, fontweight='bold')
        ax.set_ylabel("y")
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3, 3)
        
    axes[-1].set_xlabel("x")
    plt.tight_layout()
    plot_path = 'parameter_free_ood/plots/gp_pfn_interpolation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved interpolation plot to {plot_path}")
    
    df = pd.DataFrame(results)
    print("\nInterpolation MSE Results:")
    print(df.to_string(index=False))
    df.to_csv('parameter_free_ood/plots/interpolation_results.csv', index=False)

if __name__ == '__main__':
    main()
