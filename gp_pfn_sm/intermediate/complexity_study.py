import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory and PFN repo to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransformersCanDoBayesianInference')))

import encoders
import positional_encodings
from transformer import TransformerModel
from gp_pfn_sm.data_generators import spectral_mixture_kernel

def generate_complexity_sm_batch(batch_size, q_active, num_points=100, device="cpu"):
    """
    Generates a batch of Spectral Mixture GP datasets with exactly q_active active components.
    """
    # 1. Stratified sampling of X in [0, 1] with jitter
    bin_width = 1.0 / num_points
    grid_starts = torch.linspace(0.0, 1.0 - bin_width, num_points, device=device).view(1, num_points, 1)
    grid_starts = grid_starts.expand(batch_size, -1, -1)
    jitter = torch.rand(batch_size, num_points, 1, device=device) * bin_width * 0.6
    xs = grid_starts + jitter
    xs, _ = xs.sort(dim=1)
    
    # 2. Set up mixture components
    max_components = 5
    n_active = torch.ones((batch_size, 1), device=device, dtype=torch.long) * q_active
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
    
    # Keep track of signal variance before jitter/noise for analysis
    # Diagonal of K represents the prior variance of the clean signal
    clean_variance = torch.diagonal(K, dim1=1, dim2=2).mean(dim=-1)
    
    noise_level = torch.rand(batch_size, 1, device=device) * 0.001
    jitter_val = 1e-5
    K = K + (noise_level.unsqueeze(-1) ** 2 + jitter_val) * torch.eye(num_points, device=device).unsqueeze(0)
    
    # 4. Sample y ~ N(0, K)
    S, U = torch.linalg.eigh(K)
    S = torch.clamp(S, min=0.0)
    eps = torch.randn(batch_size, num_points, 1, device=device)
    ys = (U @ (S.sqrt().unsqueeze(-1) * eps)).squeeze(-1)
    
    # Save standard deviation of signal vs noise
    raw_std = ys.std(dim=-1)
    
    # 5. Normalize targets
    ys = ys - ys.mean(dim=-1, keepdim=True)
    ys = ys / (ys.std(dim=-1, keepdim=True) + 1e-8)
    
    # 6. Apply random permutation
    perm = torch.randperm(num_points, device=device)
    xs = xs[:, perm, :]
    ys = ys[:, perm]
    
    return xs, ys, clean_variance.cpu().numpy(), raw_std.cpu().numpy()

def nadaraya_watson_predict(x_context, y_context, x_query, bandwidth):
    dist_sq = (x_query[:, None] - x_context[None, :]) ** 2
    weights = np.exp(-dist_sq / (2.0 * bandwidth ** 2))
    sum_weights = np.sum(weights, axis=1, keepdims=True)
    sum_weights = np.where(sum_weights == 0, 1e-12, sum_weights)
    pred = np.sum(weights * y_context[None, :], axis=1) / sum_weights.squeeze(1)
    return pred

def tune_bandwidth_loocv(x_context, y_context):
    bandwidths = np.logspace(-2, -0.3, 40)
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
    
    # Load model
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
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    num_samples = 100
    
    q_values = [1, 2, 3, 4, 5]
    results = {}
    
    for q in q_values:
        print(f"Running evaluation for Q={q} active components...")
        xs, ys, clean_vars, raw_stds = generate_complexity_sm_batch(batch_size=num_samples, q_active=q, num_points=seq_len, device="cpu")
        
        # PFN Predictions
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
            
            x_context = xi[:eval_pos]
            y_context = yi[:eval_pos]
            
            x_query = xi[eval_pos:]
            y_query = yi[eval_pos:]
            
            # PFN Prediction
            pfn_pred_i = pfn_preds[:, i]
            pfn_mse = np.mean((pfn_pred_i - y_query) ** 2)
            pfn_mses.append(pfn_mse)
            
            # Local Smoother
            h_opt = tune_bandwidth_loocv(x_context, y_context)
            smoother_pred_i = nadaraya_watson_predict(x_context, y_context, x_query, h_opt)
            smoother_mse = np.mean((smoother_pred_i - y_query) ** 2)
            smoother_mses.append(smoother_mse)
            
        results[q] = {
            'pfn_mean_mse': np.mean(pfn_mses),
            'pfn_std_mse': np.std(pfn_mses),
            'smoother_mean_mse': np.mean(smoother_mses),
            'smoother_std_mse': np.std(smoother_mses),
            'clean_var_mean': np.mean(clean_vars),
            'raw_std_mean': np.mean(raw_stds)
        }
        
    # Print results
    print("\n" + "="*80)
    print(f"{'Q':<5} | {'PFN Mean MSE':<15} | {'PFN Std MSE':<12} | {'NW Mean MSE':<15} | {'NW Std MSE':<12} | {'Raw Std':<10}")
    print("-"*80)
    for q in q_values:
        res = results[q]
        print(f"{q:<5} | {res['pfn_mean_mse']:<15.6f} | {res['pfn_std_mse']:<12.6f} | {res['smoother_mean_mse']:<15.6f} | {res['smoother_std_mse']:<12.6f} | {res['raw_std_mean']:<10.4f}")
    print("="*80)
    
    # Save text results
    with open("/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/logs/complexity_study_results.txt", "w") as f:
        f.write("Q,pfn_mean_mse,pfn_std_mse,smoother_mean_mse,smoother_std_mse,clean_var_mean,raw_std_mean\n")
        for q in q_values:
            res = results[q]
            f.write(f"{q},{res['pfn_mean_mse']},{res['pfn_std_mse']},{res['smoother_mean_mse']},{res['smoother_std_mse']},{res['clean_var_mean']},{res['raw_std_mean']}\n")
            
    # Plot results
    plt.rcParams.update({
        'font.size': 11,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    pfn_means = [results[q]['pfn_mean_mse'] for q in q_values]
    pfn_stds = [results[q]['pfn_std_mse'] for q in q_values]
    smoother_means = [results[q]['smoother_mean_mse'] for q in q_values]
    smoother_stds = [results[q]['smoother_std_mse'] for q in q_values]
    
    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    ax1.errorbar(q_values, pfn_means, yerr=pfn_stds, fmt='o-', color='#4f46e5', linewidth=2, capsize=4, label='PFN (SM-trained)')
    ax1.errorbar(q_values, smoother_means, yerr=smoother_stds, fmt='s-', color='#ef4444', linewidth=2, capsize=4, label='Local Smoother (NW)')
    
    ax1.set_xlabel('Number of Active Mixture Components (Q)', fontweight='bold')
    ax1.set_ylabel('Mean Squared Error (MSE)', fontweight='bold')
    ax1.set_title('Inference MSE vs Prior Complexity (Q)', fontweight='bold', fontsize=13, pad=12)
    ax1.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/complexity_vs_mse.png", bbox_inches='tight')
    plt.close()
    print("Complexity study plot saved to gp_pfn_sm/complexity_vs_mse.png")

if __name__ == "__main__":
    main()
