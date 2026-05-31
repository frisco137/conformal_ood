import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np

# Add PFN repo to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransformersCanDoBayesianInference')))

import encoders
import positional_encodings
from transformer import TransformerModel
import data_generators

@torch.no_grad()
def evaluate_on_prior(model, device, prior_fn, seq_len=100, batch_size=256):
    """
    Evaluates the model on data generated from a specific prior function.
    Returns the average MSE and NLL (assuming noise variance = 0.01).
    """
    model.eval()
    x_test, y_test = prior_fn(batch_size=batch_size, num_points=seq_len, device=device)
    
    # We evaluate at multiple context lengths (20, 40, 60, 80, 99)
    eval_positions = [20, 40, 60, 80, 99]
    total_mse = 0.0
    total_nll = 0.0
    
    # Needs to match train/eval format
    # data: (x, y) where x is (seq_len, batch_size, 1), y is (seq_len, batch_size)
    x_input = x_test.transpose(0, 1)
    y_input = y_test.transpose(0, 1)
    
    for eval_pos in eval_positions:
        logits = model((x_input, y_input), single_eval_pos=eval_pos) # (seq_len - eval_pos, batch_size, 1)
        targets = y_input[eval_pos:].unsqueeze(-1)
        
        mse = torch.mean((logits - targets) ** 2).item()
        nll = 0.5 * math.log(2.0 * math.pi * 0.01) + mse / 0.02
        
        total_mse += mse
        total_nll += nll
        
    return total_mse / len(eval_positions), total_nll / len(eval_positions)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on: {device}")
    
    # Architecture params
    emsize = 256
    nhead = 4
    nhid = 512
    nlayers = 6
    seq_len = 100
    
    checkpoint_dir = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfns_multi_prior/checkpoints"
    model_types = ["rbf_matern", "rbf_periodic", "matern_periodic", "all_three"]
    
    # Priors to test on
    test_priors = {
        "RBF GP": data_generators.generate_rbf_gp_batch,
        "Matérn 5/2 GP": data_generators.generate_matern_gp_batch,
        "Periodic GP": data_generators.generate_periodic_gp_batch
    }
    
    results = {}
    
    for m_type in model_types:
        checkpoint_path = os.path.join(checkpoint_dir, f"{m_type}.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found for {m_type} at {checkpoint_path}")
            continue
            
        print(f"Loading model {m_type}...")
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
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        
        results[m_type] = {}
        for p_name, p_fn in test_priors.items():
            mse, nll = evaluate_on_prior(model, device, p_fn, seq_len=seq_len)
            results[m_type][p_name] = {"MSE": mse, "NLL": nll}
            print(f"  Evaluated on {p_name} | MSE: {mse:.4f} | NLL: {nll:.4f}")
            
    # Generate Markdown Table
    md = "# Multi-Prior PFN Evaluation Results\n\n"
    md += "This table shows the performance (Mean Squared Error and Negative Log-Likelihood) of the four trained PFN models when evaluated across standard RBF GP, standard Matérn GP, and standard Periodic GP test datasets.\n\n"
    md += "| Model Mixture | RBF GP (MSE / NLL) | Matérn 5/2 GP (MSE / NLL) | Periodic GP (MSE / NLL) |\n"
    md += "| --- | --- | --- | --- |\n"
    
    for m_type in model_types:
        row = f"| **{m_type.upper()}** "
        for p_name in ["RBF GP", "Matérn 5/2 GP", "Periodic GP"]:
            if m_type in results and p_name in results[m_type]:
                res = results[m_type][p_name]
                row += f"| {res['MSE']:.4f} / {res['NLL']:.2f} "
            else:
                row += "| N/A "
        row += "|\n"
        md += row
        
    output_path = "/home/psquare_a6000/.gemini/antigravity/brain/3744a7c6-ba28-4e40-86b9-acc13fb62a92/artifacts/multi_prior_pfn_results.md"
    with open(output_path, "w") as f:
        f.write(md)
        
    print(f"\nMarkdown results table written to {output_path}")

if __name__ == "__main__":
    main()
