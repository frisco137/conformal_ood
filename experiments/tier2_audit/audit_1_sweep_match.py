import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.core.context import generate_audit_context
from experiments.tier2_audit.instrument_c_d import get_Q, get_metrics, compute_reduced_jacobian

def compute_standard_jacobian_projected(model_id, X, y, Q, h):
    from experiments.tier2_audit.instrument_c_d import _MODEL_CACHE
    n = len(y)
    if model_id not in _MODEL_CACHE:
        _MODEL_CACHE[model_id] = load(model_id, task="regression", device="cuda")
    model = _MODEL_CACHE[model_id]
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
    import json
    seeds = [42, 100, 200, 300, 400]
    ts = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    
    # 1a. Sweep reduced basis t
    print("Running 1a: Amplitude sweep (reduced-basis) on TabICL v2")
    results_1a = {t: {"asym": [], "negeig": []} for t in ts}
    
    for seed in seeds:
        print(f" Seed {seed}")
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
        Q = get_Q(y)
        for t in ts:
            J_red = compute_reduced_jacobian("tabicl_v2", X, y, Q, t)
            a_red, n_red, _ = get_metrics(J_red)
            results_1a[t]["asym"].append(a_red)
            results_1a[t]["negeig"].append(n_red)
            
    print("\nResults 1a:")
    for t in ts:
        a_mean = np.mean(results_1a[t]["asym"])
        a_std = np.std(results_1a[t]["asym"])
        n_mean = np.mean(results_1a[t]["negeig"])
        print(f" t={t:.1e}: asym={a_mean:.3f} +/- {a_std:.3f}, negeig={n_mean:.3f}")

    # 1b & 1c. Instrument comparison at matched amplitude (t = 1e-3)
    print("\nRunning 1b & 1c: Instrument comparison at t=1e-3 (without Richardson extrapolation)")
    t_match = 1e-3
    diffs = []
    
    for seed in seeds:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
        Q = get_Q(y)
        
        J_red = compute_reduced_jacobian("tabicl_v2", X, y, Q, t_match)
        J_std_proj = compute_standard_jacobian_projected("tabicl_v2", X, y, Q, t_match)
        
        a_red, _, _ = get_metrics(J_red)
        a_std, _, _ = get_metrics(J_std_proj)
        
        diff = a_red - a_std
        diffs.append(diff)
        
        print(f" Seed {seed}: Asym Red={a_red:.4f}, Asym StdProj={a_std:.4f} -> Diff={diff:.4f}")
        
    d_mean = np.mean(diffs)
    d_std = np.std(diffs)
    print(f"\nPaired diff (Red - StdProj): {d_mean:.4f} +/- {d_std:.4f}")
    
    if np.abs(d_mean) > 0.05: # > 5% difference is massive
        print("\nHALT CONDITION MET: Instruments fundamentally disagree beyond context spread.")
    else:
        print("\nInstruments match at matched amplitude!")

if __name__ == "__main__":
    main()
