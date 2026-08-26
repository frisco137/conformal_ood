import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context

# Mock analytics
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

def get_jacobian(model_id, X, y, Q, t, dither=False):
    model = load(model_id, task="regression", device="cuda")
    def predict(y_eval):
        model.estimator.fit(X, y_eval)
        if model_id == "tabpfn_v2":
            return model.estimator.predict(X, output_type="mean")
        return model.estimator.predict(X).flatten()

    n, m_dim = Q.shape
    J_cols = []
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
        J_cols.append((m_plus - m_minus) / (2 * t))
    J_Q = np.column_stack(J_cols)
    J = Q.T @ J_Q
    return J

def profile_jacobian(J, mode="column"):
    # J_ij * s_j = J_ji * s_i => log s_i - log s_j = log|J_ij| - log|J_ji|
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
                    # s = sigma^2
                    # log s_i - log s_j = log|J_ij| - log|J_ji|
                    val = np.log(abs(J[i, j])) - np.log(abs(J[j, i]))
                else:
                    # mode == "row"
                    # d_i * J_ij = d_j * J_ji => log d_i - log d_j = log|J_ji| - log|J_ij|
                    val = np.log(abs(J[j, i])) - np.log(abs(J[i, j]))
                A.append(row)
                b.append(val)
    A = np.array(A)
    b = np.array(b)
    # Solve least squares for log_s
    log_s, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
    # Center log_s so mean(s) = 1
    log_s -= np.mean(log_s)
    s = np.exp(log_s)
    
    # Rescale J
    if mode == "column":
        # J_scaled = J * diag(s)
        J_scaled = J * s[np.newaxis, :]
    else:
        # J_scaled = diag(s) * J
        J_scaled = J * s[:, np.newaxis]
        
    asym = np.linalg.norm(J_scaled - J_scaled.T) / (2 * np.linalg.norm(J_scaled))
    return J_scaled, asym, s, res[0] if len(res) > 0 else 0.0

def main():
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
    Q = get_Q(y)
    
    tasks = [
        ("tabicl_v2", 1e-1, False),
        ("tabpfn_v2", 1e-2, True),
        ("tabswift", 1e-1, False)
    ]
    
    print("=== ITEM 3: Profiling ===")
    for mid, t, dither in tasks:
        print(f"\nComputing Jacobian for {mid}...")
        J = get_jacobian(mid, X, y, Q, t, dither)
        raw_asym = np.linalg.norm(J - J.T) / (2 * np.linalg.norm(J))
        print(f"Raw asym: {raw_asym:.4f}")
        
        # Column profile
        _, asym_col, _, res_col = profile_jacobian(J, mode="column")
        print(f"[{mid}] Column-system post-profile asym: {asym_col:.4f} (residual: {res_col:.2e})")
        
        # Row profile
        _, asym_row, _, res_row = profile_jacobian(J, mode="row")
        print(f"[{mid}] Row-system post-profile asym: {asym_row:.4f} (residual: {res_row:.2e})")

if __name__ == "__main__":
    main()
