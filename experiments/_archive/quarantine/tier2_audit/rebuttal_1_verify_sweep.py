import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context
from experiments.tier2_audit.instrument_c_d import get_Q, get_metrics

def compute_reduced_jacobian_verbose(model_id, X, y, Q, t, seed):
    from experiments.tier2_audit.instrument_c_d import _MODEL_CACHE
    n, m_dim = Q.shape
    if model_id not in _MODEL_CACHE:
        _MODEL_CACHE[model_id] = load(model_id, task="regression", device="cuda")
    model = _MODEL_CACHE[model_id]
    
    def predict(y_eval):
        model.estimator.fit(X, y_eval)
        return model.estimator.predict(X)
        
    J_hat = np.zeros((m_dim, m_dim))
    
    first_diff = None
    last_diff = None
    
    for j in range(m_dim):
        q_j = Q[:, j]
        # Print ACTUAL t used
        if j == 0:
            print(f"    [Seed {seed}] Inner loop: using t = {t}")
            y_plus = y + t * q_j
            y_minus = y - t * q_j
            if t == 1e-3 or t == 1e-1:
                print(f"      y+t*q_0 (first 3): {y_plus[:3]}")
                print(f"      y-t*q_0 (first 3): {y_minus[:3]}")

        m_plus = predict(y + t * q_j).flatten()
        m_minus = predict(y - t * q_j).flatten()
        v_j = (m_plus - m_minus) / (2 * t)
        J_hat[:, j] = (Q.T @ v_j).flatten()
        
    return J_hat

def main():
    print("--- Section 1a & 1b Verification ---")
    seeds = [42, 100, 200, 300, 400]
    ts = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    
    # Check 1b: Print first three y values per seed to confirm they differ
    print("1b Check: First 3 y values per seed:")
    y_cache = {}
    for seed in seeds:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
        y_cache[seed] = (X, y)
        print(f"  Seed {seed}: {y[:3]}")
        
    # Check 1a: Amplitude Sweep with verbosity
    print("\n1a Check: Amplitude Sweep (t in 1e-3 to 1e-1)")
    
    results = {t: {"asym": [], "negeig": [], "norm_QJQ": []} for t in ts}
    negeigs_per_seed = {seed: [] for seed in seeds}
    
    for seed in seeds:
        print(f"\n--- Context Seed {seed} ---")
        X, y = y_cache[seed]
        Q = get_Q(y)
        for t in ts:
            J = compute_reduced_jacobian_verbose("tabicl_v2", X, y, Q, t, seed)
            
            # Print ||Q^T J Q||_F
            norm_QJQ = np.linalg.norm(J, 'fro')
            print(f"  t={t:.1e} -> ||Q^T J Q||_F = {norm_QJQ:.8f}")
            
            # Get metrics
            a, n_eig, _ = get_metrics(J)
            results[t]["asym"].append(a)
            results[t]["negeig"].append(n_eig)
            results[t]["norm_QJQ"].append(norm_QJQ)
            if t == 1e-3:
                negeigs_per_seed[seed] = n_eig
                
    print("\n--- Summary of Sweep (1a) ---")
    for t in ts:
        a_mean, a_std = np.mean(results[t]["asym"]), np.std(results[t]["asym"])
        # print full array
        print(f"t={t:.1e}:")
        print(f"  ||J||_F per context: {np.array(results[t]['norm_QJQ'])}")
        print(f"  asym per context: {np.array(results[t]['asym'])}")
        print(f"  Aggregated: asym={a_mean:.6f} +/- {a_std:.6f}")
        
    print("\n--- Summary of NegEig (1b) ---")
    print("Unrounded negeig per seed at t=1e-3:")
    for seed in seeds:
        print(f"  Seed {seed}: {negeigs_per_seed[seed]}")
        
if __name__ == "__main__":
    main()
