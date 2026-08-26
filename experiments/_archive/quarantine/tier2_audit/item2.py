import sys
import numpy as np
import torch
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context
from experiments.phase_1.core.surrogates import ExactGP, HierarchicalGP

# Mock analytics to prevent telemetry hangs
class MockAnalytics:
    def __getattr__(self, name): return lambda *args, **kwargs: None
sys.modules['analytics'] = MockAnalytics()

def get_Q(y):
    n = len(y)
    v1 = np.ones(n)
    v2 = y - np.mean(y)
    A = np.random.randn(n, n)
    A[:, 0] = v1
    A[:, 1] = v2
    Q_full, _ = np.linalg.qr(A)
    return Q_full[:, 2:]

def compute_curvature_and_drift(model_id, X, y, Q, t, dither=False):
    if model_id == "exact_gp_wrapped":
        egp = ExactGP(X, sigma=0.5)
        def predict(y_eval):
            mean_y, std_y = np.mean(y_eval), np.std(y_eval) + 1e-8
            return mean_y + std_y * egp.predict((y_eval - mean_y) / std_y)
    elif model_id == "hierarchical_gp_wrapped":
        hgp = HierarchicalGP(X, sigma=0.5)
        def predict(y_eval):
            mean_y, std_y = np.mean(y_eval), np.std(y_eval) + 1e-8
            return mean_y + std_y * hgp.predict((y_eval - mean_y) / std_y)
    else:
        model = load(model_id, task="regression", device="cuda")
        def predict(y_eval):
            model.estimator.fit(X, y_eval)
            if model_id == "tabpfn_v2":
                return model.estimator.predict(X, output_type="mean")
            return model.estimator.predict(X).flatten()

    n, m_dim = Q.shape
    curvatures = []
    
    # Base prediction
    if dither:
        m_base = np.zeros(n)
        np.random.seed(42)
        noise = np.random.uniform(-6.87e-4, 6.87e-4, (10, n))
        for k in range(10):
            m_base += predict(y + noise[k])
        m_base /= 10
    else:
        m_base = predict(y).flatten()
        
    norm_m_base = np.linalg.norm(m_base)

    drift_ratios = []
    std_base = np.std(y)

    print(f"\n--- {model_id} ---")
    for j in range(m_dim):
        q_j = Q[:, j]
        if dither:
            m_plus = np.zeros(n)
            m_minus = np.zeros(n)
            np.random.seed(42 + j + 100)
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
            
        diff = m_plus - 2 * m_base + m_minus
        c = np.linalg.norm(diff) / ((t**2) * norm_m_base)
        curvatures.append(c)

        if j < 3 and model_id != "exact_gp_wrapped" and model_id != "hierarchical_gp_wrapped":
            print(f"Dir {j} raw diff[:5]: {diff[:5]}")
            
        # Drift bound (Item 2.4)
        std_plus = np.std(y + t * q_j)
        std_minus = np.std(y - t * q_j)
        drift = (std_plus - 2 * std_base + std_minus) / (t**2 * std_base)
        drift_ratios.append(np.abs(drift))
        
    avg_c = float(np.mean(curvatures))
    avg_drift = float(np.mean(drift_ratios))
    print(f"Curvature: {avg_c:.3e}")
    print(f"Drift ratio (s_y): {avg_drift:.3e}")
    return avg_c, avg_drift

def main():
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
    Q = get_Q(y)
    
    tasks = [
        ("exact_gp_wrapped", 1e-1, False),
        ("hierarchical_gp_wrapped", 1e-1, False),
        ("tabicl_v2", 1e-1, False),
        ("tabpfn_v2", 1e-2, True),
        ("tabswift", 1e-1, False)
    ]
    
    for mid, t, dither in tasks:
        compute_curvature_and_drift(mid, X, y, Q, t, dither)

if __name__ == "__main__":
    main()
