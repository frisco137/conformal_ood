import sys
class MockAnalytics:
    def __getattr__(self, name): return lambda *args, **kwargs: None
sys.modules['analytics'] = MockAnalytics()
import numpy as np
import time
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context
from experiments.tier2_audit.audit_phase1_metrics import get_metrics_canonical, get_Q

def compute_jacobian_fast(model_id, X, y, Q, t):
    n, m_dim = Q.shape
    J_hat = np.zeros((m_dim, m_dim))
    
    if model_id == "tabpfn_v2":
        model = load(model_id, task="regression", device="cuda")
        for j in range(m_dim):
            q_j = Q[:, j]
            m_plus = np.zeros(n)
            m_minus = np.zeros(n)
            np.random.seed(42 + j)
            noise = np.random.uniform(-6.87e-4/2, 6.87e-4/2, (10, n))
            for k in range(10):
                model.estimator.fit(X, y + noise[k] + t * q_j)
                m_plus += model.estimator.predict(X, output_type="mean")
                model.estimator.fit(X, y + noise[k] - t * q_j)
                m_minus += model.estimator.predict(X, output_type="mean")
            v_j = (m_plus - m_minus) / (2 * t * 10)
            J_hat[:, j] = (Q.T @ v_j).flatten()
    else:
        model = load(model_id, task="regression", device="cuda")
        for j in range(m_dim):
            q_j = Q[:, j]
            model.estimator.fit(X, y + t * q_j)
            m_plus = model.estimator.predict(X)
            model.estimator.fit(X, y - t * q_j)
            m_minus = model.estimator.predict(X)
            v_j = (m_plus - m_minus) / (2 * t)
            J_hat[:, j] = (Q.T @ v_j).flatten()
    return J_hat

def phase3():
    if len(sys.argv) != 2:
        print("Usage: python3 audit_phase3.py <model_id>")
        return
    model_id = sys.argv[1]
    
    print(f"--- {model_id} ---")
    # Latency (E2.7)
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
    model = load(model_id, task="regression", device="cuda")
    start = time.time()
    for _ in range(10):
        model.estimator.fit(X, y)
        if model_id == "tabpfn_v2":
            _ = model.estimator.predict(X, output_type="mean")
        else:
            _ = model.estimator.predict(X)
    end = time.time()
    latency = (end - start) / 10
    print(f"  Latency (E2.7): {latency*1000:.1f} ms/query")
    
    # Cross-channel variance (E2.5) & Non-degeneracy (E2.6)
    Q = get_Q(y)
    t = 0.1 if model_id != "tabicl_v2" else 1e-3
    J = compute_jacobian_fast(model_id, X, y, Q, t)
    
    diags = np.diag(J)
    cross_var = np.var(diags)
    print(f"  Cross-channel var (E2.5): {cross_var:.4e}")
    
    sym_J = (J + J.T) / 2
    eigs = np.sort(np.linalg.eigvalsh(sym_J))[::-1]
    top2_ratio = eigs[0] / (eigs[1] + 1e-8)
    print(f"  Top-2 eig ratio (E2.6): {top2_ratio:.4f}")

if __name__ == "__main__":
    phase3()
