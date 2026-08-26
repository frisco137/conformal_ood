import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context
from experiments.tier2_audit.instrument_c_d import get_Q, get_metrics, compute_reduced_jacobian

def compute_standard_jacobian_projected(model_id, X, y, Q, h):
    n = len(y)
    model = load(model_id, task="regression", device="cuda")
    def predict(y_eval):
        model.estimator.fit(X, y_eval)
        return model.estimator.predict(X).flatten()
        
    J_std = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        m_plus = predict(y + h * e_j)
        m_minus = predict(y - h * e_j)
        J_std[:, j] = (m_plus - m_minus) / (2 * h)
        
    return Q.T @ J_std @ Q

def main():
    seed = 42
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
    Q = get_Q(y)
    
    t = 1e-3
    print("Computing reduced basis...")
    J_red = compute_reduced_jacobian("tabicl_v2", X, y, Q, t)
    
    print("Computing standard basis...")
    J_std_proj = compute_standard_jacobian_projected("tabicl_v2", X, y, Q, t)
    
    a_red, n_red, norm_red = get_metrics(J_red)
    a_std, n_std, norm_std = get_metrics(J_std_proj)
    
    print(f"\nSeed {seed}")
    print(f"Reduced basis: asym={a_red:.4f}, negeig={n_red:.4f}, norm={norm_red:.4f}")
    print(f"Standard basis: asym={a_std:.4f}, negeig={n_std:.4f}, norm={norm_std:.4f}")
    
    diff = np.linalg.norm(J_red - J_std_proj, 'fro')
    print(f"Diff ||J_red - J_std_proj||_F = {diff:.4e}")

if __name__ == "__main__":
    main()
