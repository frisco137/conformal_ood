import sys
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context
from experiments.phase_1.core.surrogates import ExactGP, HierarchicalGP

def get_Q(y):
    n = len(y)
    v1 = np.ones(n)
    v2 = y - np.mean(y)
    A = np.random.randn(n, n)
    A[:, 0] = v1
    A[:, 1] = v2
    Q_full, _ = np.linalg.qr(A)
    return Q_full[:, 2:]

_MODEL_CACHE = {}

def get_model_predict_fn(model_id, X, y):
    if model_id == "exact_gp_wrapped":
        egp = ExactGP(X, sigma=0.5)
        def predict(y_eval):
            mean_y, std_y = np.mean(y_eval), np.std(y_eval) + 1e-8
            return mean_y + std_y * egp.predict((y_eval - mean_y) / std_y)
        return predict
    elif model_id == "hierarchical_gp_wrapped":
        hgp = HierarchicalGP(X, sigma=0.5)
        def predict(y_eval):
            mean_y, std_y = np.mean(y_eval), np.std(y_eval) + 1e-8
            return mean_y + std_y * hgp.predict((y_eval - mean_y) / std_y)
        return predict
    else:
        if model_id not in _MODEL_CACHE:
            _MODEL_CACHE[model_id] = load(model_id, task="regression", device="cuda")
        model = _MODEL_CACHE[model_id]
        def predict(y_eval):
            model.estimator.fit(X, y_eval)
            return model.estimator.predict(X, output_type="mean") if model_id == "tabpfn_v2" else model.estimator.predict(X)
        return predict

def compute_curvature(predict_fn, y, Q, t, dither=False, N=10, delta=6.87e-4, seed=42):
    n, m_dim = Q.shape
    curvatures = []
    
    # Base prediction
    if dither:
        m_base = np.zeros(n)
        np.random.seed(seed)
        noise = np.random.uniform(-delta/2, delta/2, (N, n))
        for k in range(N):
            m_base += predict_fn(y + noise[k])
        m_base /= N
    else:
        m_base = predict_fn(y).flatten()
        
    norm_m_base = np.linalg.norm(m_base)
    if norm_m_base < 1e-12:
        norm_m_base = 1.0 # fallback

    for j in range(m_dim):
        q_j = Q[:, j]
        if dither:
            m_plus = np.zeros(n)
            m_minus = np.zeros(n)
            np.random.seed(seed + j + 100)
            noise = np.random.uniform(-delta/2, delta/2, (N, n))
            for k in range(N):
                e = noise[k]
                m_plus += predict_fn(y + e + t * q_j)
                m_minus += predict_fn(y + e - t * q_j)
            m_plus /= N
            m_minus /= N
        else:
            m_plus = predict_fn(y + t * q_j).flatten()
            m_minus = predict_fn(y - t * q_j).flatten()
            
        # ||m(y+tq) - 2m(y) + m(y-tq)|| / (t^2 ||m(y)||)
        diff = m_plus - 2 * m_base + m_minus
        c = np.linalg.norm(diff) / ((t**2) * norm_m_base)
        curvatures.append(c)
        
    return float(np.mean(curvatures))

def run_curvature_sweep(model_id, dither=False, ts=None):
    if ts is None:
        ts = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 1e1]
    
    # Fast sweep on 1 seed
    seed = 42
    out_sweep = {t: [] for t in ts}
    
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
    Q = get_Q(y)
    predict_fn = get_model_predict_fn(model_id, X, y)
    
    for t in ts:
        c = compute_curvature(predict_fn, y, Q, t, dither=dither, seed=seed)
        out_sweep[t].append(c)
        
    # Full 5-seed measurement at plateau t=0.1
    t_plateau = 0.1
    out_plateau = []
    seeds = [42, 100, 200, 300, 400]
    for s in seeds:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        Q = get_Q(y)
        predict_fn = get_model_predict_fn(model_id, X, y)
        c = compute_curvature(predict_fn, y, Q, t_plateau, dither=dither, seed=s)
        out_plateau.append(c)
        
    return out_sweep, out_plateau

def main():
    models = [
        ("exact_gp_wrapped", False, [1e-2, 1e-1, 1.0]),
        ("hierarchical_gp_wrapped", False, [1e-2, 1e-1, 1.0]),
        ("tabicl_v2", False, [1e-2, 1e-1, 1.0]),
        ("tabpfn_v2", True, [1e-2, 1e-1, 1.0]),
        ("tabswift", False, [1e-2, 1e-1, 1.0])
    ]
    
    results = {}
    for model_id, dither, ts in models:
        print(f"Running sweep for {model_id}...")
        res_sweep, res_plateau = run_curvature_sweep(model_id, dither=dither, ts=ts)
        results[model_id] = {"sweep": res_sweep, "plateau": res_plateau}
        print("  Sweep:")
        for t, vals in res_sweep.items():
            print(f"    t={t}: curve = {np.mean(vals):.3e}")
        print(f"  Plateau (t=0.1, 5 seeds): {np.mean(res_plateau):.3e} +/- {np.std(res_plateau):.3e}")
            
    with open("experiments/tier2_audit/results_phase2_curvature.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
