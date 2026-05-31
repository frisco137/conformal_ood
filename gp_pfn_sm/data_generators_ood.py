import math
import torch
from typing import Tuple
from gp_pfn_sm.data_generators import spectral_mixture_kernel

def generate_id_spectral_mixture_gp_batch(
    batch_size: int, 
    num_points: int = 100, 
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of In-Distribution (ID) datasets from the trained
    Spectral Mixture GP prior with exactly 5 active components and frequencies in [0, 2.5].
    """
    # 1. Stratified sampling of X in [0, 1] with jitter
    bin_width = 1.0 / num_points
    grid_starts = torch.linspace(0.0, 1.0 - bin_width, num_points, device=device).view(1, num_points, 1)
    grid_starts = grid_starts.expand(batch_size, -1, -1)
    jitter = torch.rand(batch_size, num_points, 1, device=device) * bin_width * 0.6
    xs = grid_starts + jitter
    xs, _ = xs.sort(dim=1)
    
    # 2. Set up mixture components - 5 active components
    max_components = 5
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
    
    # 4. Sample y ~ N(0, K)
    S, U = torch.linalg.eigh(K)
    S = torch.clamp(S, min=0.0)
    eps = torch.randn(batch_size, num_points, 1, device=device)
    ys = (U @ (S.sqrt().unsqueeze(-1) * eps)).squeeze(-1)
    
    # 5. Normalize targets
    ys = ys - ys.mean(dim=-1, keepdim=True)
    ys = ys / (ys.std(dim=-1, keepdim=True) + 1e-8)
    
    # 6. Apply random permutation
    perm = torch.randperm(num_points, device=device)
    xs = xs[:, perm, :]
    ys = ys[:, perm]
    
    return xs, ys

def generate_ood_spectral_mixture_gp_batch(
    batch_size: int, 
    num_points: int = 100, 
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of Out-of-Distribution (OOD) datasets from the
    Spectral Mixture GP prior with exactly 5 active components but high, disjoint frequencies in [5.0, 10.0].
    """
    # 1. Stratified sampling of X in [0, 1] with jitter
    bin_width = 1.0 / num_points
    grid_starts = torch.linspace(0.0, 1.0 - bin_width, num_points, device=device).view(1, num_points, 1)
    grid_starts = grid_starts.expand(batch_size, -1, -1)
    jitter = torch.rand(batch_size, num_points, 1, device=device) * bin_width * 0.6
    xs = grid_starts + jitter
    xs, _ = xs.sort(dim=1)
    
    # 2. Set up mixture components - 5 active components
    max_components = 5
    n_active = torch.ones((batch_size, 1), device=device, dtype=torch.long) * max_components
    idx = torch.arange(max_components, device=device).expand(batch_size, -1)
    mask = (idx < n_active).float()
    
    # Generate disjoint frequencies in [5.0, 10.0]
    means = torch.rand(batch_size, max_components, device=device)
    means = means * 5.0 + 5.0
    
    # Keep scales and weights identical to ID prior distribution setup
    scales = torch.rand(batch_size, max_components, device=device)
    scales = scales * 0.7 + 0.05
    
    weights = torch.rand(batch_size, max_components, device=device)
    weights = weights * mask 
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
    
    # 3. Compute Covariance Matrix K
    K = spectral_mixture_kernel(xs, xs, weights, means, scales)
    
    # Add noise & jitter for numerical stability
    noise_level = torch.rand(batch_size, 1, device=device) * 0.001
    K = K + (noise_level.unsqueeze(-1) ** 2 + 1e-5) * torch.eye(num_points, device=device).unsqueeze(0)
    
    # 4. Sample y ~ N(0, K)
    S, U = torch.linalg.eigh(K)
    S = torch.clamp(S, min=0.0)
    eps = torch.randn(batch_size, num_points, 1, device=device)
    ys = (U @ (S.sqrt().unsqueeze(-1) * eps)).squeeze(-1)
    
    # 5. Normalize targets
    ys = ys - ys.mean(dim=-1, keepdim=True)
    ys = ys / (ys.std(dim=-1, keepdim=True) + 1e-8)
    
    # 6. Apply random permutation
    perm = torch.randperm(num_points, device=device)
    xs = xs[:, perm, :]
    ys = ys[:, perm]
    
    return xs, ys

def get_id_and_ood_data(
    num_samples_per_class: int = 3000, 
    num_points: int = 100, 
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns ID and OOD Spectral Mixture datasets.
    """
    print(f"Generating ID Spectral Mixture GP data (N={num_samples_per_class})...")
    x_id, y_id = generate_id_spectral_mixture_gp_batch(num_samples_per_class, num_points, device)
    
    print(f"Generating OOD High-Frequency Spectral Mixture GP data (N={num_samples_per_class})...")
    x_ood, y_ood = generate_ood_spectral_mixture_gp_batch(num_samples_per_class, num_points, device)
    
    return x_id, y_id, x_ood, y_ood
