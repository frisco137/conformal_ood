import math
import numpy as np
import torch
from typing import Tuple

def generate_rbf_gp_batch(
    batch_size: int, 
    num_points: int = 100, 
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of GP datasets from a GP prior with an RBF (Squared Exponential) kernel.
    Hyperpriors:
    - Lengthscale l ~ Gamma(3.0, 6.0)  (mean = 0.5)
    - Output scale sf2 ~ Gamma(2.0, 0.15) (mean = 13.3)
    - Noise sn2 ~ Gamma(0.0001, 1.0)
    
    Returns:
    - X: shape (batch_size, num_points, 1)
    - y: shape (batch_size, num_points)
    """
    l_dist = torch.distributions.Gamma(3.0, 6.0)
    sf2_dist = torch.distributions.Gamma(2.0, 0.15)
    sn2_dist = torch.distributions.Gamma(0.0001, 1.0)
    
    l = l_dist.sample((batch_size, 1, 1)).to(device)    # (B, 1, 1)
    sf2 = sf2_dist.sample((batch_size, 1, 1)).to(device) # (B, 1, 1)
    sn2 = sn2_dist.sample((batch_size, 1, 1)).to(device) # (B, 1, 1)
    
    x = torch.rand(batch_size, num_points, 1, device=device)
    D = torch.cdist(x, x) # (B, N, N)
    
    # RBF Covariance: K(d) = sf2 * exp(-d^2 / (2 * l^2))
    K = sf2 * torch.exp(- (D**2) / (2.0 * l**2))
    
    # Add noise & jitter
    K = K + sn2 * torch.eye(num_points, device=device).unsqueeze(0)
    jitter = 1e-6 * torch.eye(num_points, device=device).unsqueeze(0)
    K = K + jitter
    
    # Sample y ~ N(0, K)
    S, U = torch.linalg.eigh(K)
    S = torch.clamp(S, min=0.0)
    eps = torch.randn(batch_size, num_points, 1, device=device)
    y = (U @ (S.sqrt().unsqueeze(-1) * eps)).squeeze(-1)
    
    # Normalize y to zero mean and unit variance per sample
    y = y - y.mean(dim=-1, keepdim=True)
    y = y / (y.std(dim=-1, keepdim=True) + 1e-8)
    
    return x, y

def generate_matern_gp_batch(
    batch_size: int, 
    num_points: int = 100, 
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of GP datasets from a GP prior with a Matérn 5/2 kernel.
    Hyperpriors:
    - Lengthscale l ~ Gamma(3.0, 6.0)  (mean = 0.5)
    - Output scale sf2 ~ Gamma(2.0, 0.15) (mean = 13.3)
    - Noise sn2 ~ Gamma(0.0001, 1.0)
    
    Returns:
    - X: shape (batch_size, num_points, 1)
    - y: shape (batch_size, num_points)
    """
    l_dist = torch.distributions.Gamma(3.0, 6.0)
    sf2_dist = torch.distributions.Gamma(2.0, 0.15)
    sn2_dist = torch.distributions.Gamma(0.0001, 1.0)
    
    l = l_dist.sample((batch_size, 1, 1)).to(device)    # (B, 1, 1)
    sf2 = sf2_dist.sample((batch_size, 1, 1)).to(device) # (B, 1, 1)
    sn2 = sn2_dist.sample((batch_size, 1, 1)).to(device) # (B, 1, 1)
    
    x = torch.rand(batch_size, num_points, 1, device=device)
    D = torch.cdist(x, x) # (B, N, N)
    
    # Matérn 5/2 Covariance
    val = math.sqrt(5.0) * D / l
    K = sf2 * (1.0 + val + (5.0 * D**2) / (3.0 * l**2)) * torch.exp(-val)
    
    # Add noise & jitter
    K = K + sn2 * torch.eye(num_points, device=device).unsqueeze(0)
    jitter = 1e-6 * torch.eye(num_points, device=device).unsqueeze(0)
    K = K + jitter
    
    # Sample y ~ N(0, K)
    S, U = torch.linalg.eigh(K)
    S = torch.clamp(S, min=0.0)
    eps = torch.randn(batch_size, num_points, 1, device=device)
    y = (U @ (S.sqrt().unsqueeze(-1) * eps)).squeeze(-1)
    
    # Normalize y
    y = y - y.mean(dim=-1, keepdim=True)
    y = y / (y.std(dim=-1, keepdim=True) + 1e-8)
    
    return x, y

def generate_periodic_gp_batch(
    batch_size: int, 
    num_points: int = 100, 
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of GP datasets from a GP prior with a standard Periodic (Exp-Sine-Squared) kernel.
    Hyperpriors:
    - Frequency f ~ Uniform(0.2, 1.0)  (reduced frequency)
    - Lengthscale l ~ Uniform(1.5, 3.0) (increased lengthscale for smooth waves)
    - Output scale sf2 ~ Gamma(2.0, 0.15)
    - Noise sn2 ~ Gamma(0.0001, 1.0)
    
    Returns:
    - X: shape (batch_size, num_points, 1)
    - y: shape (batch_size, num_points)
    """
    sf2_dist = torch.distributions.Gamma(2.0, 0.15)
    sn2_dist = torch.distributions.Gamma(0.0001, 1.0)
    
    sf2 = sf2_dist.sample((batch_size, 1, 1)).to(device) # (B, 1, 1)
    sn2 = sn2_dist.sample((batch_size, 1, 1)).to(device) # (B, 1, 1)
    f = (torch.rand(batch_size, 1, 1, device=device) * 1.5 + 0.5) # [0.5, 2.0]
    l = (torch.rand(batch_size, 1, 1, device=device) * 1.5 + 1.5) # [1.5, 3.0]

    
    x = torch.rand(batch_size, num_points, 1, device=device)
    
    # Pairwise differences
    diff = x.unsqueeze(2) - x.unsqueeze(1) # (B, N, N, 1)
    diff = diff.squeeze(-1) # (B, N, N)
    
    # Exp-Sine-Squared periodic kernel: K = sf2 * exp(-2 * sin^2(pi * |x - x'| * f) / l^2)
    sin_term = torch.sin(math.pi * diff * f)
    K = sf2 * torch.exp(- 2.0 * (sin_term**2) / (l**2))
    
    # Add noise & jitter
    K = K + sn2 * torch.eye(num_points, device=device).unsqueeze(0)
    jitter = 1e-6 * torch.eye(num_points, device=device).unsqueeze(0)
    K = K + jitter
    
    # Sample y ~ N(0, K)
    S, U = torch.linalg.eigh(K)
    S = torch.clamp(S, min=0.0)
    eps = torch.randn(batch_size, num_points, 1, device=device)
    y = (U @ (S.sqrt().unsqueeze(-1) * eps)).squeeze(-1)
    
    # Normalize y
    y = y - y.mean(dim=-1, keepdim=True)
    y = y / (y.std(dim=-1, keepdim=True) + 1e-8)
    
    return x, y
