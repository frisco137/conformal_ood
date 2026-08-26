import sys
class MockAnalytics:
    def __getattr__(self, name): return lambda *args, **kwargs: None
sys.modules['analytics'] = MockAnalytics()

import os
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["ANALYTICS_DISABLED"] = "1"
import logging
logging.getLogger("analytics").setLevel(logging.CRITICAL)
try:
    import posthog
    posthog.disabled = True
except: pass

import numpy as np
import torch
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.core.surrogates import ExactGP, HierarchicalGP, Imitator
from experiments.core.context import generate_audit_context
sys.modules['analytics'] = MockAnalytics()

import os
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["ANALYTICS_DISABLED"] = "1"
import logging
logging.getLogger("analytics").setLevel(logging.CRITICAL)
try:
    import posthog
    posthog.disabled = True
except: pass

def get_Q(y):
    n = len(y)
    v1 = np.ones(n)
    v2 = y - np.mean(y)
    A = np.random.randn(n, n)
    A[:, 0] = v1
    A[:, 1] = v2
    Q_full, _ = np.linalg.qr(A)
    return Q_full[:, 2:]

def get_model(model_id, seed, X):
    # Dummy object that mocks the expected wrapper struct
    class MockWrapper:
        def __init__(self, est):
            self.estimator = est
    
    # We create the surrogate wrappers that respond to .predict(y)
    if model_id == "exactgp":
        return MockWrapper(ExactGP(X, sigma=0.5))
    elif model_id == "hierarchicalgp":
        return MockWrapper(HierarchicalGP(X))
    elif model_id == "imitator":
        est = Imitator(X, sigma=0.5)
        return MockWrapper(est)
    else:
        return load(model_id, task="regression", device="cuda")

def compute_metrics(J):
    norm_J = np.linalg.norm(J)
    J_anti = (J - J.T) / 2
    norm_anti = np.linalg.norm(J_anti)
    A1 = norm_anti / norm_J if norm_J > 0 else 0
    
    J_sym = (J + J.T) / 2
    eigs = np.linalg.eigvalsh(J_sym)
    neg_eigs = eigs[eigs < -1e-10]
    
    negeig = abs(min(0.0, np.min(eigs))) / np.linalg.norm(J, 2) if norm_J > 0 else 0.0
    negfrac = len(neg_eigs) / len(eigs)
    
    return A1, negeig, negfrac, J, J_anti

def get_jacobian_and_curvature(model_id, X, y, Q, t, dither=False):
    model = get_model(model_id, 42, X)
    def predict(y_eval):
        if hasattr(model.estimator, 'fit'):
            model.estimator.fit(X, y_eval)
            if model_id == "tabpfn_v2":
                return model.estimator.predict(X, output_type="mean")
            return model.estimator.predict(X).flatten()
        else:
            # Analytical controls directly take y_eval in predict
            return model.estimator.predict(y_eval).flatten()

    n, m_dim = Q.shape
    J_cols = []
    C_cols = []
    
    m_base = predict(y)
    drift = 0.0
    
    # Store intermediate raw vectors for provenance (A.1)
    raw_prov = {}
    
    for j in range(m_dim):
        q_j = Q[:, j]
        if dither:
            m_plus = np.zeros(n)
            m_minus = np.zeros(n)
            np.random.seed(42 + j)
            noise = np.random.uniform(-6.87e-4, 6.87e-4, (10, n))
            for k in range(10):
                e = noise[k]
                m_plus += predict(y + e + t * q_j)
                m_minus += predict(y + e - t * q_j)
            m_plus /= 10
            m_minus /= 10
        else:
            m_plus = predict(y + t * q_j).flatten()
            m_minus = predict(y - t * q_j).flatten()
            
        if j == 0:
            raw_prov['m_base'] = m_base.tolist()[:5]
            raw_prov['m_plus'] = m_plus.tolist()[:5]
            raw_prov['m_minus'] = m_minus.tolist()[:5]
            raw_prov['second_diff'] = (m_plus - 2*m_base + m_minus).tolist()[:5]
            
        J_cols.append((m_plus - m_minus) / (2 * t))
        C_cols.append(np.linalg.norm(m_plus - 2 * m_base + m_minus) / (t**2 * np.linalg.norm(m_base)))
        drift = max(drift, np.linalg.norm(m_plus - m_base) / np.linalg.norm(m_base))
        
    J_Q = np.column_stack(J_cols)
    J = Q.T @ J_Q
    
    mean_curv = np.mean(C_cols)
    
    raw_prov['J_first_row'] = J[0, :5].tolist()
    
    return J, mean_curv, drift, raw_prov

def profile_jacobian(J, mode="column"):
    n = J.shape[0]
    A = []
    b = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(J[i, j]) > 1e-10 and abs(J[j, i]) > 1e-10:
                row = np.zeros(n)
                row[i] = 1
                row[j] = -1
                if mode == "column":
                    val = np.log(abs(J[i, j])) - np.log(abs(J[j, i]))
                else:
                    val = np.log(abs(J[j, i])) - np.log(abs(J[i, j]))
                A.append(row)
                b.append(val)
    A = np.array(A)
    b = np.array(b)
    log_s, res, _, _ = np.linalg.lstsq(A, b, rcond=None)
    log_s -= np.mean(log_s)
    s = np.exp(log_s)
    
    if mode == "column":
        J_scaled = J * s[np.newaxis, :]
    else:
        J_scaled = J * s[:, np.newaxis]
        
    asym = np.linalg.norm(J_scaled - J_scaled.T) / (2 * np.linalg.norm(J_scaled))
    return asym, s

def run_5seed(model_id, t, do_profiling=True):
    seeds = [42, 100, 200, 300, 400]
    res = {'A1': [], 'negeig': [], 'negfrac': [], 'curvature': [], 'col_asym': [], 'row_asym': [], 'prov': {}}
    
    print(f"\n--- Running 5 seeds for {model_id} ---")
    for s in seeds:
        print(f"Seed {s}: generating context")
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        Q = get_Q(y)
        
        print(f"Seed {s}: getting jacobian")
        J, curv, drift, prov = get_jacobian_and_curvature(model_id, X, y, Q, t, dither=False)
        print(f"Seed {s}: computing metrics")
        A1, negeig, negfrac, _, J_anti = compute_metrics(J)
        
        res['A1'].append(A1)
        res['negeig'].append(negeig)
        res['negfrac'].append(negfrac)
        res['curvature'].append(curv)
        
        if s == 42:
            res['prov'] = prov
            res['prov']['J_anti_first_row'] = J_anti[0, :5].tolist()
            res['prov']['norm_J'] = float(np.linalg.norm(J))
            res['prov']['norm_J_anti'] = float(np.linalg.norm(J_anti))
            
        if do_profiling:
            print(f"Seed {s}: profiling col")
            col_asym, s_col = profile_jacobian(J, "column")
            print(f"Seed {s}: profiling row")
            row_asym, s_row = profile_jacobian(J, "row")
            res['col_asym'].append(col_asym)
            res['row_asym'].append(row_asym)
            if s == 42:
                res['prov']['s_col'] = s_col[:5].tolist()
                res['prov']['s_row'] = s_row[:5].tolist()
        print(f"Seed {s}: done")
                
    return res

def main():
    out = {}
    
    # 1. ExactGP (Control) 5-seed
    out['exactgp'] = run_5seed('exactgp', 1e-3, do_profiling=False)
    
    # 2. HierarchicalGP 5-seed
    out['hierarchicalgp'] = run_5seed('hierarchicalgp', 1e-1, do_profiling=False)
    
    # 3. TabICL v2 5-seed
    out['tabicl_v2'] = run_5seed('tabicl_v2', 1e-3, do_profiling=True)
    
    # 4. TabICL Curvature vs Drift Amplitude Sweep (Section D)
    print("\n--- TabICL Curvature vs Drift Sweep ---")
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
    Q = get_Q(y)
    out['tabicl_sweep'] = {}
    for t_amp in [1e-3, 1e-2, 1e-1]:
        _, curv, drift, _ = get_jacobian_and_curvature('tabicl_v2', X, y, Q, t_amp, dither=False)
        out['tabicl_sweep'][str(t_amp)] = {'curvature': float(curv), 's_y': float(drift)}
        print(f"t={t_amp} | Curv: {curv:.4e} | s_y: {drift:.4e}")
        
    # 5. Targeted Imitator
    out['imitator'] = run_5seed('imitator', 1e-3, do_profiling=False)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open('experiments/tier2_audit/results_batch1.json', 'w') as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
        
    print("Batch 1 Complete!")

if __name__ == "__main__":
    main()
