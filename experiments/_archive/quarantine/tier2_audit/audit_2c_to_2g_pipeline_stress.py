import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.phase_1.core.context import generate_audit_context
from experiments.tier2_audit.instrument_c_d import get_metrics

# 2c to 2g pipeline stress tests

def get_Q_svd(y):
    n = len(y)
    v1 = np.ones(n)
    v2 = y - np.mean(y)
    A = np.random.randn(n, n)
    A[:, 0] = v1
    A[:, 1] = v2
    U, _, _ = np.linalg.svd(A, full_matrices=True)
    return U[:, 2:]

def get_Q_qr(y):
    n = len(y)
    v1 = np.ones(n)
    v2 = y - np.mean(y)
    A = np.random.randn(n, n)
    A[:, 0] = v1
    A[:, 1] = v2
    Q_full, _ = np.linalg.qr(A)
    return Q_full[:, 2:]

def generate_random_Q(y):
    n = len(y)
    v1 = np.ones(n) / np.linalg.norm(np.ones(n))
    v2 = y - np.mean(y)
    v2 = v2 / np.linalg.norm(v2)
    
    Q = []
    for _ in range(n - 2):
        while True:
            v = np.random.randn(n)
            # orthogonalize against v1, v2 and previous
            v -= np.dot(v, v1) * v1
            v -= np.dot(v, v2) * v2
            for q in Q:
                v -= np.dot(v, q) * q
            if np.linalg.norm(v) > 1e-5:
                v = v / np.linalg.norm(v)
                Q.append(v)
                break
    return np.column_stack(Q)

def compute_reduced_jacobian_analytic(predict_fn, y, Q, t=1e-3):
    n, m_dim = Q.shape
    J_hat = np.zeros((m_dim, m_dim))
    for j in range(m_dim):
        q_j = Q[:, j]
        m_plus = predict_fn(y + t * q_j)
        m_minus = predict_fn(y - t * q_j)
        v_j = (m_plus - m_minus) / (2 * t)
        J_hat[:, j] = Q.T @ v_j
    return J_hat

def main():
    np.random.seed(42)
    y = np.random.randn(100)
    n = len(y)
    
    # 2c. Symmetry of the estimator itself
    # Construct analytic asymmetric map: m(y) = A y
    A_asym = np.random.randn(n, n)
    # Ensure it's asymmetric
    A_asym = A_asym - A_asym.T + np.diag(np.random.randn(n))
    def predict_asym(y_eval):
        return A_asym @ y_eval
        
    Q = get_Q_qr(y)
    J_red = compute_reduced_jacobian_analytic(predict_asym, y, Q)
    a, n_eig, norm_J = get_metrics(J_red)
    print(f"2c. Asymmetric Map: asym={a:.4f} (should be > 0)")
    assert a > 0.1, "Failed to preserve asymmetry!"
    
    # 2d. Basis invariance
    Q_svd = get_Q_svd(y)
    Q_gs = generate_random_Q(y)
    
    J_svd = compute_reduced_jacobian_analytic(predict_asym, y, Q_svd)
    J_gs = compute_reduced_jacobian_analytic(predict_asym, y, Q_gs)
    
    a_svd, _, _ = get_metrics(J_svd)
    a_gs, _, _ = get_metrics(J_gs)
    print(f"2d. Basis Invariance: QR asym={a:.4f}, SVD asym={a_svd:.4f}, GS asym={a_gs:.4f}")
    assert np.abs(a - a_svd) < 1e-4 and np.abs(a - a_gs) < 1e-4, "Basis invariance failed!"
    
    # 2e. Null perturbation
    J_null = compute_reduced_jacobian_analytic(predict_asym, y, Q, t=0.0)
    # wait, t=0 causes div by zero, let's implement the null test conceptually by passing a constant function
    def predict_constant(y_eval):
        return np.ones(n)
    J_const = compute_reduced_jacobian_analytic(predict_constant, y, Q, t=1e-3)
    norm_const = np.linalg.norm(J_const, 'fro')
    print(f"2e. Null Perturbation: norm={norm_const:.4e} (should be ~0)")
    assert norm_const < 1e-10, "Null perturbation failed!"
    
    # 2f. Permutation consistency
    P = np.zeros((n, n))
    perm = np.random.permutation(n)
    for i in range(n):
        P[i, perm[i]] = 1.0
        
    y_perm = P @ y
    # Q for permuted y
    Q_perm = P @ Q  # This is a valid Q matrix for y_perm because P is orthogonal
    
    def predict_asym_permuted(y_eval):
        # the model evaluates the permuted targets. 
        # m_perm(y_perm) = P * m(P^T y_perm)
        return P @ predict_asym(P.T @ y_eval)
        
    J_perm = compute_reduced_jacobian_analytic(predict_asym_permuted, y_perm, Q_perm)
    # J_perm should be exactly J_red mathematically
    diff = np.linalg.norm(J_red - J_perm, 'fro')
    print(f"2f. Permutation Consistency: diff={diff:.4e} (should be ~0)")
    assert diff < 1e-10, "Permutation consistency failed!"
    
    # 2g. Transpose check
    # m_i(y) = y_1 for all i
    # So J_{i1} = 1, all other J_{ij} = 0
    def predict_transpose(y_eval):
        return np.ones(n) * y_eval[1]
    
    J_trans = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n); e_j[j] = 1.0
        m_plus = predict_transpose(y + 1e-3 * e_j)
        m_minus = predict_transpose(y - 1e-3 * e_j)
        J_trans[:, j] = (m_plus - m_minus) / 2e-3
        
    # J_trans[:, 1] should be 1
    assert np.allclose(J_trans[:, 1], 1.0)
    assert np.allclose(J_trans[:, 0], 0.0)
    print(f"2g. Transpose Check: Passed. J[:, 1] is correctly the derivative w.r.t y_1")
    
    print("\nAll pipeline stress tests passed!")

if __name__ == "__main__":
    main()
