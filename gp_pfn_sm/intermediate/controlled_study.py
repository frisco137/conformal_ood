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
from gp_pfn_sm.data_generators import generate_spectral_mixture_gp_batch

def nadaraya_watson_predict(x_context, y_context, x_query, bandwidth):
    """
    Nadaraya-Watson kernel regression (Gaussian kernel smoother).
    """
    # Compute pairwise distance squared (N_query, N_context)
    dist_sq = (x_query[:, None] - x_context[None, :]) ** 2
    weights = np.exp(-dist_sq / (2.0 * bandwidth ** 2))
    
    # Normalize weights
    sum_weights = np.sum(weights, axis=1, keepdims=True)
    # Avoid division by zero
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
    print(f"Generating {num_samples} Spectral Mixture GP samples...")
    # Generate test batch
    xs, ys = generate_spectral_mixture_gp_batch(batch_size=num_samples, num_points=seq_len, device="cpu")
    
    # PFN predictions
    # Transpose to sequence-first for model
    x_input = xs.transpose(0, 1).to(device)
    y_input = ys.transpose(0, 1).to(device)
    
    with torch.no_grad():
        output = model((x_input, y_input), single_eval_pos=eval_pos) # (seq_len - eval_pos, batch_size, 1)
        pfn_preds = output.squeeze(-1).cpu().numpy() # (50, 100)
        
    xs_np = xs.squeeze(-1).numpy() # (100, 100)
    ys_np = ys.numpy() # (100, 100)
    
    pfn_mses = []
    smoother_mses = []
    
    for i in range(num_samples):
        # Extract features and targets
        xi = xs_np[i]
        yi = ys_np[i]
        
        # Split into context (first 50) and query (remaining 50)
        x_context = xi[:eval_pos]
        y_context = yi[:eval_pos]
        
        x_query = xi[eval_pos:]
        y_query = yi[eval_pos:]
        
        # PFN prediction for this sample
        pfn_pred_i = pfn_preds[:, i]
        pfn_mse = np.mean((pfn_pred_i - y_query) ** 2)
        pfn_mses.append(pfn_mse)
        
        # Tune bandwidth for local smoother via LOOCV
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
    
    print(f"\nControlled Study Results (N={num_samples} samples, Context Size={eval_pos}):")
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
    with open("/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/logs/controlled_study_results.txt", "w") as f:
        f.write(f"PFN Mean MSE: {mean_pfn:.6f}\n")
        f.write(f"PFN Std MSE: {std_pfn:.6f}\n")
        f.write(f"Local Smoother Mean MSE: {mean_smoother:.6f}\n")
        f.write(f"Local Smoother Std MSE: {std_smoother:.6f}\n")

if __name__ == "__main__":
    main()
