import sys
import os
import torch
import numpy as np

# Add parent directory and PFN repo to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransformersCanDoBayesianInference')))

import encoders
import positional_encodings
from transformer import TransformerModel
from gp_pfn_sm.data_generators import spectral_mixture_kernel

def generate_max_complexity_sm_batch(batch_size, num_points=100, device="cpu"):
    """
    Generates a batch of Spectral Mixture GP datasets at maximum complexity
    (i.e. exactly 5 active components for every sample).
    """
    # 1. Stratified sampling of X in [0, 1] with jitter
    bin_width = 1.0 / num_points
    grid_starts = torch.linspace(0.0, 1.0 - bin_width, num_points, device=device).view(1, num_points, 1)
    grid_starts = grid_starts.expand(batch_size, -1, -1)
    jitter = torch.rand(batch_size, num_points, 1, device=device) * bin_width * 0.6
    xs = grid_starts + jitter
    xs, _ = xs.sort(dim=1)
    
    # 2. Set up mixture components - Max complexity: exactly 5 active components
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
    
    noise_level = torch.rand(batch_size, 1, device=device) * 0.001
    K = K + (noise_level.unsqueeze(-1) ** 2 + 1e-5) * torch.eye(num_points, device=device).unsqueeze(0)
    
    # 4. Sample y ~ N(0, K)
    S, U = torch.linalg.eigh(K)
    S = torch.clamp(S, min=0.0)
    eps = torch.randn(batch_size, num_points, 1, device=device)
    ys = (U @ (S.sqrt().unsqueeze(-1) * eps)).squeeze(-1)
    
    # 5. Normalize targets to mean zero and unit variance
    ys = ys - ys.mean(dim=-1, keepdim=True)
    ys = ys / (ys.std(dim=-1, keepdim=True) + 1e-8)
    
    # 6. Apply random permutation
    perm = torch.randperm(num_points, device=device)
    xs = xs[:, perm, :]
    ys = ys[:, perm]
    
    return xs, ys

def nadaraya_watson_predict(x_context, y_context, x_query, bandwidth):
    """
    Nadaraya-Watson kernel regression (Gaussian kernel smoother).
    """
    dist_sq = (x_query[:, None] - x_context[None, :]) ** 2
    weights = np.exp(-dist_sq / (2.0 * bandwidth ** 2))
    sum_weights = np.sum(weights, axis=1, keepdims=True)
    sum_weights = np.where(sum_weights == 0, 1e-12, sum_weights)
    pred = np.sum(weights * y_context[None, :], axis=1) / sum_weights.squeeze(1)
    return pred

def tune_bandwidth_loocv(x_context, y_context):
    """
    Finds the optimal bandwidth h for Nadaraya-Watson via Leave-One-Out CV on the context points.
    """
    bandwidths = np.logspace(-2, -0.3, 40) # 0.01 to ~0.5
    best_h = 0.05
    best_err = float('inf')
    
    n = len(x_context)
    if n <= 1:
        return 0.05
        
    for h in bandwidths:
        errs = []
        for i in range(n):
            x_train = np.delete(x_context, i)
            y_train = np.delete(y_context, i)
            x_val = x_context[i]
            y_val = y_context[i]
            
            y_pred = nadaraya_watson_predict(x_train, y_train, np.array([x_val]), h)
            errs.append((y_pred[0] - y_val) ** 2)
            
        mean_err = np.mean(errs)
        if mean_err < best_err:
            best_err = mean_err
            best_h = h
            
    return best_h

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 1. Load PFN model matching architecture
    emsize = 256
    nhead = 4
    nhid = 512
    nlayers = 6
    seq_len = 100
    eval_pos = 50
    
    encoder = encoders.Linear(1, emsize)
    y_encoder = encoders.Linear(1, emsize)
    pos_encoder = positional_encodings.PositionalEncoding(emsize, seq_len * 2)
    
    model = TransformerModel(
        encoder=encoder,
        n_out=1,
        ninp=emsize,
        nhead=nhead,
        nhid=nhid,
        nlayers=nlayers,
        dropout=0.0,
        y_encoder=y_encoder,
        pos_encoder=pos_encoder
    )
    
    checkpoint_path = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/checkpoints/spectral_mixture.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Set seed for reproducible controlled study
    torch.manual_seed(42)
    np.random.seed(42)
    
    num_samples = 100
    print(f"Generating {num_samples} MAX COMPLEXITY Spectral Mixture GP samples (n_active = 5)...")
    # Generate test batch with n_active = 5
    xs, ys = generate_max_complexity_sm_batch(batch_size=num_samples, num_points=seq_len, device="cpu")
    
    # PFN predictions
    x_input = xs.transpose(0, 1).to(device)
    y_input = ys.transpose(0, 1).to(device)
    
    with torch.no_grad():
        output = model((x_input, y_input), single_eval_pos=eval_pos)
        pfn_preds = output.squeeze(-1).cpu().numpy()
        
    xs_np = xs.squeeze(-1).numpy()
    ys_np = ys.numpy()
    
    pfn_mses = []
    smoother_mses = []
    
    for i in range(num_samples):
        xi = xs_np[i]
        yi = ys_np[i]
        
        # Split into context (50) and query (50)
        x_context = xi[:eval_pos]
        y_context = yi[:eval_pos]
        
        x_query = xi[eval_pos:]
        y_query = yi[eval_pos:]
        
        # PFN prediction
        pfn_pred_i = pfn_preds[:, i]
        pfn_mse = np.mean((pfn_pred_i - y_query) ** 2)
        pfn_mses.append(pfn_mse)
        
        # Tune bandwidth
        h_opt = tune_bandwidth_loocv(x_context, y_context)
        
        # Predict using local smoother
        smoother_pred_i = nadaraya_watson_predict(x_context, y_context, x_query, h_opt)
        smoother_mse = np.mean((smoother_pred_i - y_query) ** 2)
        smoother_mses.append(smoother_mse)
        
    pfn_mses = np.array(pfn_mses)
    smoother_mses = np.array(smoother_mses)
    
    mean_pfn = np.mean(pfn_mses)
    std_pfn = np.std(pfn_mses)
    
    mean_smoother = np.mean(smoother_mses)
    std_smoother = np.std(smoother_mses)
    
    print(f"\nControlled Study Results (Max Complexity, N={num_samples} samples, Context Size={eval_pos}):")
    print(f"------------------------------------------------------------------")
    print(f"PFN Model (Spectral Mixture):")
    print(f"  Mean MSE:           {mean_pfn:.6f}")
    print(f"  Std Deviation MSE:  {std_pfn:.6f}")
    print(f"")
    print(f"Local Smoother (Nadaraya-Watson, LOOCV-tuned):")
    print(f"  Mean MSE:           {mean_smoother:.6f}")
    print(f"  Std Deviation MSE:  {std_smoother:.6f}")
    print(f"------------------------------------------------------------------")
    
    # Save results to txt file
    with open("/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/logs/controlled_study_max_complexity.txt", "w") as f:
        f.write(f"Max Complexity PFN Mean MSE: {mean_pfn:.6f}\n")
        f.write(f"Max Complexity PFN Std MSE: {std_pfn:.6f}\n")
        f.write(f"Max Complexity Local Smoother Mean MSE: {mean_smoother:.6f}\n")
        f.write(f"Max Complexity Local Smoother Std MSE: {std_smoother:.6f}\n")

if __name__ == "__main__":
    main()
