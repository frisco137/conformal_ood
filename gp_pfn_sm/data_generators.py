import math
import torch
from typing import Tuple

def spectral_mixture_kernel(
    x1: torch.Tensor, 
    x2: torch.Tensor, 
    weights: torch.Tensor, 
    means: torch.Tensor, 
    scales: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Spectral Mixture kernel covariance matrix.
    x1: shape (B, N1, 1)
    x2: shape (B, N2, 1)
    weights: shape (B, Q)
    means: shape (B, Q) (frequencies / spectral centers)
    scales: shape (B, Q) (spectral variances)
    
    Returns:
    - Covariance matrix K: shape (B, N1, N2)
    """
    # pairwise differences tau = x1 - x2^T (B, N1, N2)
    tau = x1 - x2.transpose(1, 2)
    tau_sq = tau.unsqueeze(-1) ** 2 # (B, N1, N2, 1)
    
    w = weights.view(weights.shape[0], 1, 1, -1)
    mu = means.view(means.shape[0], 1, 1, -1)
    sig = scales.view(scales.shape[0], 1, 1, -1)
    
    # K(tau) = sum_q w_q * exp(-2 * pi^2 * tau^2 * sig_q^2) * cos(2 * pi * tau * mu_q)
    exp_term = torch.exp(-2.0 * (math.pi ** 2) * tau_sq * (sig ** 2))
    cos_term = torch.cos(2.0 * math.pi * tau.unsqueeze(-1) * mu)
    return (w * exp_term * cos_term).sum(dim=-1)

def generate_spectral_mixture_gp_batch(
    batch_size: int, 
    num_points: int = 100, 
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of GP datasets from a Spectral Mixture (SM) GP prior.
    Follows a hierarchical prior for the mixture components:
    - Max components = 5
    - Active components sampled uniformly between 1 and 5
    - Frequencies (means) and lengthscales (scales) sampled from custom distributions
    
    Returns:
    - X: shape (batch_size, num_points, 1)
    - y: shape (batch_size, num_points)
    """
    # 1. Stratified sampling of X in [0, 1] with jitter
    bin_width = 1.0 / num_points
    grid_starts = torch.linspace(0.0, 1.0 - bin_width, num_points, device=device).view(1, num_points, 1)
    grid_starts = grid_starts.expand(batch_size, -1, -1)
    jitter = torch.rand(batch_size, num_points, 1, device=device) * bin_width * 0.6
    xs = grid_starts + jitter
    # Sort xs along seq_len dimension to maintain ordering for possible plotting/inference alignment
    xs, _ = xs.sort(dim=1)
    
    # 2. Set up mixture components
    max_components = 5
    # Always have exactly 5 active components
    n_active = torch.ones((batch_size, 1), device=device, dtype=torch.long) * max_components
    idx = torch.arange(max_components, device=device).expand(batch_size, -1)
    mask = (idx < n_active).float()

    means = torch.rand(batch_size, max_components, device=device)
    means[:, 0] = means[:, 0] * 0.2 
    means[:, 1:] = means[:, 1:] * 2.5
    
    scales = torch.rand(batch_size, max_components, device=device)
    scales[:, 0] = scales[:, 0] * 0.1 + 0.01 
    scales[:, 1:] = scales[:, 1:] * 0.7 + 0.05
    
    weights = torch.rand(batch_size, max_components, device=device)
    weights = weights * mask 
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

    # 3. Compute Covariance Matrix K
    K = spectral_mixture_kernel(xs, xs, weights, means, scales)
    
    # Add noise & jitter for numerical stability
    noise_level = torch.rand(batch_size, 1, device=device) * 0.001
    K = K + (noise_level.unsqueeze(-1) ** 2 + 1e-5) * torch.eye(num_points, device=device).unsqueeze(0)
    
    # 4. Sample y ~ N(0, K) using robust Eigendecomposition to prevent Cholesky failures
    S, U = torch.linalg.eigh(K)
    S = torch.clamp(S, min=0.0)
    eps = torch.randn(batch_size, num_points, 1, device=device)
    ys = (U @ (S.sqrt().unsqueeze(-1) * eps)).squeeze(-1)
    
    # 5. Normalize targets to mean zero and unit variance per sample
    ys = ys - ys.mean(dim=-1, keepdim=True)
    ys = ys / (ys.std(dim=-1, keepdim=True) + 1e-8)
    
    # 6. Apply random permutation to break sequence ordering (standard PFN requirement)
    perm = torch.randperm(num_points, device=device)
    xs = xs[:, perm, :]
    ys = ys[:, perm]
    
    return xs, ys
