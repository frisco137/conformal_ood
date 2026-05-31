import math
import numpy as np
import torch
from typing import Tuple

def generate_matern_gp_batch(
    batch_size: int, 
    num_points: int = 100, 
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of In-Distribution (ID) datasets from a GP prior with a Matérn 5/2 kernel
    and hyperpriors as described in Müller's paper.
    
    Hyperpriors:
    - Lengthscale l ~ Gamma(3.0, 6.0)
    - Output scale sf2 ~ Gamma(2.0, 0.15)
    - Noise sn2 ~ Gamma(0.0001, 1.0)
    
    Returns:
    - X: shape (batch_size, num_points, 1)
    - y: shape (batch_size, num_points)
    """
    # Sample hyper-parameters per batch element
    # torch.distributions.Gamma takes (concentration/shape, rate/inverse_scale)
    l_dist = torch.distributions.Gamma(3.0, 6.0)
    sf2_dist = torch.distributions.Gamma(2.0, 0.15)
    sn2_dist = torch.distributions.Gamma(0.0001, 1.0)
    
    l = l_dist.sample((batch_size, 1, 1)).to(device)    # (B, 1, 1)
    sf2 = sf2_dist.sample((batch_size, 1, 1)).to(device) # (B, 1, 1)
    sn2 = sn2_dist.sample((batch_size, 1, 1)).to(device) # (B, 1, 1)
    
    # x uniformly sampled from [0, 1]
    x = torch.rand(batch_size, num_points, 1, device=device)
    
    # Compute pairwise distances
    D = torch.cdist(x, x) # (B, N, N)
    
    # Matérn 5/2 Covariance:
    # K(d) = sf2 * (1 + sqrt(5)*d/l + 5*d^2 / 3*l^2) * exp(-sqrt(5)*d/l)
    val = math.sqrt(5.0) * D / l
    K = sf2 * (1.0 + val + (5.0 * D**2) / (3.0 * l**2)) * torch.exp(-val)
    
    # Add noise & jitter for numerical stability
    K = K + sn2 * torch.eye(num_points, device=device).unsqueeze(0)
    jitter = 1e-6 * torch.eye(num_points, device=device).unsqueeze(0)
    K = K + jitter
    
    # Sample y ~ N(0, K) using stable eigenvalue decomposition
    S, U = torch.linalg.eigh(K)
    S = torch.clamp(S, min=0.0)
    eps = torch.randn(batch_size, num_points, 1, device=device)
    y = (U @ (S.sqrt().unsqueeze(-1) * eps)).squeeze(-1)
    
    # Normalize y to zero mean and unit variance per sample
    y = y - y.mean(dim=-1, keepdim=True)
    y = y / (y.std(dim=-1, keepdim=True) + 1e-8)
    
    return x, y

def generate_discontinuous_step_batch(
    batch_size: int, 
    num_points: int = 100, 
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of discontinuous step function datasets.
    Functions are piecewise flat with 1 to 3 sudden vertical jumps
    at random x thresholds, with added N(0, 0.01) noise.
    
    Returns:
    - X: shape (batch_size, num_points, 1)
    - y: shape (batch_size, num_points)
    """
    x = torch.rand(batch_size, num_points, 1, device=device)
    y = torch.zeros(batch_size, num_points, device=device)
    
    for b in range(batch_size):
        xb = x[b, :, 0]
        num_jumps = np.random.randint(1, 4)
        thresholds = np.random.uniform(0.0, 1.0, num_jumps)
        thresholds.sort()
        heights = np.random.uniform(-2.0, 2.0, num_jumps + 1)
        
        idx = torch.zeros(num_points, dtype=torch.long, device=device)
        for t in thresholds:
            idx += (xb > t).long()
            
        heights_tensor = torch.tensor(heights, dtype=torch.float32, device=device)
        yb = heights_tensor[idx]
        
        # Add N(0, 0.01) noise (std = 0.1)
        yb = yb + torch.randn(num_points, device=device) * 0.1
        y[b] = yb
        
    # Normalize y to zero mean and unit variance per sample
    y = y - y.mean(dim=-1, keepdim=True)
    y = y / (y.std(dim=-1, keepdim=True) + 1e-8)
    
    return x, y

def generate_periodic_batch(
    batch_size: int, 
    num_points: int = 100, 
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of high-frequency periodic function datasets.
    y = A * sin(w * x + phi) + N(0, 0.01) with w in [50, 100]
    
    Returns:
    - X: shape (batch_size, num_points, 1)
    - y: shape (batch_size, num_points)
    """
    x = torch.rand(batch_size, num_points, 1, device=device)
    
    # Amplitude A ~ Uniform(0.5, 2.0)
    A = 0.5 + 1.5 * torch.rand(batch_size, 1, 1, device=device)
    # Frequency w ~ Uniform(50, 100)
    w = 50.0 + 50.0 * torch.rand(batch_size, 1, 1, device=device)
    # Phase phi ~ Uniform(0, 2*pi)
    phi = 2.0 * math.pi * torch.rand(batch_size, 1, 1, device=device)
    
    y = (A * torch.sin(w * x + phi)).squeeze(-1)
    
    # Add N(0, 0.01) noise (std = 0.1)
    y = y + torch.randn(batch_size, num_points, device=device) * 0.01
    
    # Normalize y to zero mean and unit variance per sample
    y = y - y.mean(dim=-1, keepdim=True)
    y = y / (y.std(dim=-1, keepdim=True) + 1e-8)
    
    return x, y

def generate_chirp_batch(
    batch_size: int, 
    num_points: int = 100, 
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of non-stationary chirp signal datasets.
    y = sin(exp(alpha * x)) + N(0, 0.01) where alpha ~ Uniform(3, 6)
    
    Returns:
    - X: shape (batch_size, num_points, 1)
    - y: shape (batch_size, num_points)
    """
    x = torch.rand(batch_size, num_points, 1, device=device)
    
    # alpha ~ Uniform(3, 6)
    alpha = 3.0 + 3.0 * torch.rand(batch_size, 1, 1, device=device)
    
    y = torch.sin(torch.exp(alpha * x)).squeeze(-1)
    
    # Add N(0, 0.01) noise (std = 0.1)
    y = y + torch.randn(batch_size, num_points, device=device) * 0.1
    
    # Normalize y to zero mean and unit variance per sample
    y = y - y.mean(dim=-1, keepdim=True)
    y = y / (y.std(dim=-1, keepdim=True) + 1e-8)
    
    return x, y

def get_id_and_ood_data(
    num_samples_per_class: int = 3000, 
    num_points: int = 100, 
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates the full set of ID and OOD datasets.
    Returns:
    - x_id: shape (num_samples, num_points, 1)
    - y_id: shape (num_samples, num_points)
    - x_ood: shape (num_samples, num_points, 1)
    - y_ood: shape (num_samples, num_points)
    """
    print("Generating In-Distribution (Matérn 5/2 GP) data...")
    x_id, y_id = generate_matern_gp_batch(num_samples_per_class, num_points, device)
    
    print("Generating Out-of-Distribution (Periodic, Chirp) data...")
    num_sub = num_samples_per_class // 2
    x_periodic, y_periodic = generate_periodic_batch(num_sub, num_points, device)
    x_chirp, y_chirp = generate_chirp_batch(num_sub, num_points, device)
    
    x_ood = torch.cat([x_periodic, x_chirp], dim=0)
    y_ood = torch.cat([y_periodic, y_chirp], dim=0)
    
    # Shuffle OOD pool to mix periodic and chirp functions
    perm = torch.randperm(x_ood.shape[0])
    x_ood = x_ood[perm]
    y_ood = y_ood[perm]
    
    return x_id, y_id, x_ood, y_ood
