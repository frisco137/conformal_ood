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
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context
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

def get_jacobian(model, X, y, Q, t):
    def predict(y_eval):
        model.estimator.fit(X, y_eval)
        return model.estimator.predict(X)

    n, m_dim = Q.shape
    J_cols = []
    C_cols = []
    
    m_base = predict(y)
    raw_prov = {}
    
    for j in range(m_dim):
        q_j = Q[:, j]
        m_plus = predict(y + t * q_j)
        m_minus = predict(y - t * q_j)
            
        if j == 0:
            raw_prov['m_base'] = m_base.tolist()[:5]
            raw_prov['m_plus'] = m_plus.tolist()[:5]
            raw_prov['m_minus'] = m_minus.tolist()[:5]
            
        J_cols.append((m_plus - m_minus) / (2 * t))
        C_cols.append(np.linalg.norm(m_plus - 2 * m_base + m_minus) / (t**2 * np.linalg.norm(m_base)))
        
    J_Q = np.column_stack(J_cols)
    J = Q.T @ J_Q
    
    mean_curv = np.mean(C_cols)
    
    raw_prov['J_first_row'] = J[0, :5].tolist()
    
    return J, mean_curv, raw_prov

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



def main():
    seeds = [42, 100, 200, 300, 400]
    t = 1e-1
    model_id = "tabswift"
    
    res = {'A1': [], 'negeig': [], 'negfrac': [], 'col_asym': [], 'row_asym': [], 'curvature': [], 'prov': {}}
    
    print(f"Loading {model_id}...")
    model = load(model_id, task="regression", device="cuda")
    
    for s in seeds:
        print(f"Running seed {s}")
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        Q = get_Q(y)
        
        J, curv, prov = get_jacobian(model, X, y, Q, t)
        A1, negeig, negfrac, _, J_anti = compute_metrics(J)
        
        res['A1'].append(float(A1))
        res['negeig'].append(float(negeig))
        res['negfrac'].append(float(negfrac))
        res['curvature'].append(float(curv))
        
        col_asym, s_col = profile_jacobian(J, "column")
        row_asym, s_row = profile_jacobian(J, "row")
        res['col_asym'].append(float(col_asym))
        res['row_asym'].append(float(row_asym))
        
        if s == 42:
            res['prov'] = prov
            res['prov']['J_anti_first_row'] = J_anti[0, :5].tolist()
            res['prov']['norm_J'] = float(np.linalg.norm(J))
            res['prov']['norm_J_anti'] = float(np.linalg.norm(J_anti))
            res['prov']['s_col'] = s_col[:5].tolist()
            res['prov']['s_row'] = s_row[:5].tolist()
            
    with open('experiments/tier2_audit/results_tabswift.json', 'w') as f:
        json.dump(res, f, indent=2)
        
if __name__ == "__main__":
    main()
