import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context

def get_Q(y):
    n = len(y)
    v1 = np.ones(n)
    v2 = y - np.mean(y)
    A = np.random.randn(n, n)
    A[:, 0] = v1
    A[:, 1] = v2
    Q_full, _ = np.linalg.qr(A)
    return Q_full[:, 2:]

def test_curvature_vectors():
    print("=== ITEM 1: TabICL Curvature Vectors ===")
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
    Q = get_Q(y)
    
    model = load("tabicl_v2", task="regression", device="cuda")
    
    # 1.3: Confirm the curvature code path invokes the same model handle
    # We will print inside the predict function to prove it executes
    call_counter = [0]
    def predict(y_eval):
        call_counter[0] += 1
        model.estimator.fit(X, y_eval)
        # Using output_type is only for tabpfn, tabicl doesn't have it
        return model.estimator.predict(X).flatten()

    t = 0.1
    m_base = predict(y)
    print(f"[1.3] m_base call counter: {call_counter[0]}")
    
    # Check 3 directions
    for j in range(3):
        q_j = Q[:, j]
        m_plus = predict(y + t * q_j)
        m_minus = predict(y - t * q_j)
        
        diff = m_plus - 2 * m_base + m_minus
        
        print(f"\n--- Direction {j} ---")
        print(f"[1.1] Raw second difference vector (first 5 elements): {diff[:5]}")
        print(f"[1.1] Norm of diff: {np.linalg.norm(diff):.3e}")
        
        print(f"[1.2] m_base (first 5):  {m_base[:5]}")
        print(f"[1.2] m_plus (first 5):  {m_plus[:5]}")
        print(f"[1.2] m_minus (first 5): {m_minus[:5]}")
        
        # Check if they are identically equal
        if np.allclose(m_plus, m_base) and np.allclose(m_minus, m_base):
            print(">>> WARNING: m_plus, m_base, m_minus are IDENTICAL! Function is constant in this direction! <<<")
        elif np.allclose(diff, np.zeros_like(diff)):
            print(">>> WARNING: diff is IDENTICALLY ZERO! Function is perfectly linear! <<<")
            
    # 1.4 Sanity check on known nonzero curvature
    print("\n=== ITEM 1.4: Sanity Check (m(y) = y + 0.1*y^2) ===")
    def predict_known(y_eval):
        return y_eval + 0.1 * (y_eval ** 2)
        
    m_base_known = predict_known(y)
    norm_m_base_known = np.linalg.norm(m_base_known)
    c_knowns = []
    for j in range(Q.shape[1]):
        q_j = Q[:, j]
        m_plus_known = predict_known(y + t * q_j)
        m_minus_known = predict_known(y - t * q_j)
        diff_known = m_plus_known - 2 * m_base_known + m_minus_known
        c_knowns.append(np.linalg.norm(diff_known) / ((t**2) * norm_m_base_known))
        
    print(f"Known map measured curvature (mean over Q): {np.mean(c_knowns):.3e}")
    # Analytic: diff = (y + tq) + 0.1(y + tq)^2 - 2(y + 0.1y^2) + (y - tq) + 0.1(y - tq)^2
    # diff = 0.2 * t^2 * q^2
    # c = ||0.2 q^2|| / ||y + 0.1y^2||
    q_j = Q[:, 0]
    expected_diff = 0.2 * (t**2) * (q_j**2)
    expected_c = np.linalg.norm(0.2 * (q_j**2)) / norm_m_base_known
    print(f"Known map analytic curvature for q_0: {expected_c:.3e}")

if __name__ == "__main__":
    test_curvature_vectors()
