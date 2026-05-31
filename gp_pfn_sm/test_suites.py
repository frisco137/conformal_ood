import os
import torch
import numpy as np
import pandas as pd
from gp_pfn_sm.ood_metric_study import generate_swept_sm_batch

def generate_id_low_noise(batch_size, num_points=100, device="cpu"):
    """
    Suite 1: In-Prior, Low Noise (Clean Negatives).
    """
    return generate_swept_sm_batch(batch_size, freq_max=2.5, num_points=num_points, device=device)

def generate_id_high_noise(batch_size, num_points=100, device="cpu"):
    """
    Suite 2: In-Prior, High Observation Noise.
    We generate standard SM GP but overwrite the noise level to be around 0.4.
    """
    # Stratified sampling of X in [0, 1] with jitter
    bin_width = 1.0 / num_points
    grid_starts = torch.linspace(0.0, 1.0 - bin_width, num_points, device=device).view(1, num_points, 1)
    grid_starts = grid_starts.expand(batch_size, -1, -1)
    jitter = torch.rand(batch_size, num_points, 1, device=device) * bin_width * 0.6
    xs = grid_starts + jitter
    xs, _ = xs.sort(dim=1)
    
    # 5 active components
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
    
    from gp_pfn_sm.data_generators import spectral_mixture_kernel
    K = spectral_mixture_kernel(xs, xs, weights, means, scales)
    
    # High observation noise: 0.1 + small jitter
    noise_level = torch.ones(batch_size, 1, device=device) * 0.1
    K = K + (noise_level.unsqueeze(-1) ** 2 + 1e-5) * torch.eye(num_points, device=device).unsqueeze(0)
    
    # Sample y ~ N(0, K)
    S, U = torch.linalg.eigh(K)
    S = torch.clamp(S, min=0.0)
    eps = torch.randn(batch_size, num_points, 1, device=device)
    ys = (U @ (S.sqrt().unsqueeze(-1) * eps)).squeeze(-1)
    
    # Normalize targets
    ys = ys - ys.mean(dim=-1, keepdim=True)
    ys = ys / (ys.std(dim=-1, keepdim=True) + 1e-8)
    
    # Random permutation
    perm = torch.randperm(num_points, device=device)
    xs = xs[:, perm, :]
    ys = ys[:, perm]
    
    return xs, ys

def generate_near_ood(batch_size, num_points=100, device="cpu"):
    """
    Suite 3: Near-OOD (Parametric shift - higher frequencies).
    """
    return generate_swept_sm_batch(batch_size, freq_max=7.5, num_points=num_points, device=device)

def generate_far_ood(batch_size, num_points=100, device="cpu"):
    """
    Suite 4: Far-OOD (Parametric shift - extremely high frequencies).
    """
    return generate_swept_sm_batch(batch_size, freq_max=15.0, num_points=num_points, device=device)

def generate_real_world_ood(batch_size, num_points=100, device="cpu"):
    """
    Suite 5: Real-World OOD.
    Loads Melbourne daily temperatures and airline passengers, slices them, normalizes them,
    and returns batches of size batch_size.
    """
    data_dir = "gp_pfn_sm/data"
    temp_path = os.path.join(data_dir, "temperatures.csv")
    airline_path = os.path.join(data_dir, "airline.csv")
    
    # Load and clean temperature data
    temp_df = pd.read_csv(temp_path)
    # The temperature column is named "Temp" or "Daily minimum temperatures"
    temp_col = temp_df.columns[1]
    temp_vals = pd.to_numeric(temp_df[temp_col], errors='coerce').dropna().values
    
    # Load airline passenger data
    airline_df = pd.read_csv(airline_path)
    airline_col = airline_df.columns[1]
    airline_vals = pd.to_numeric(airline_df[airline_col], errors='coerce').dropna().values
    
    # Prepare all possible slices of length 100
    slices = []
    
    # Temperature slices (sliding window, step 2 to keep slices relatively diverse)
    for i in range(0, len(temp_vals) - num_points, 2):
        slices.append(temp_vals[i : i + num_points])
        
    # Airline slices (sliding window, step 1)
    for i in range(0, len(airline_vals) - num_points, 1):
        slices.append(airline_vals[i : i + num_points])
        
    # Randomly shuffle slices to sample from
    np.random.seed(42)
    np.random.shuffle(slices)
    
    xs_list = []
    ys_list = []
    
    for i in range(batch_size):
        # Retrieve a slice (looping if we run out of slices)
        slice_idx = i % len(slices)
        y_raw = slices[slice_idx]
        
        # Construct sorted x-values in [0, 1] with stratified jitter
        bin_width = 1.0 / num_points
        grid = np.linspace(0.0, 1.0 - bin_width, num_points)
        jitter = np.random.rand(num_points) * bin_width * 0.6
        x = grid + jitter
        x = np.sort(x)
        
        # Standardize y
        y = (y_raw - np.mean(y_raw)) / (np.std(y_raw) + 1e-8)
        
        # Apply random permutation
        perm = np.random.permutation(num_points)
        x_perm = x[perm].reshape(num_points, 1)
        y_perm = y[perm]
        
        xs_list.append(x_perm)
        ys_list.append(y_perm)
        
    xs = torch.tensor(np.array(xs_list), dtype=torch.float32, device=device)
    ys = torch.tensor(np.array(ys_list), dtype=torch.float32, device=device)
    
    return xs, ys
