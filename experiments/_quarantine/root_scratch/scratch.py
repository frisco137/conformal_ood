import numpy as np
from models import load
from experiments.core.context import generate_audit_context
from experiments.tier2_audit.instrument_c_d import get_Q

X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
Q = get_Q(y)
q_j = Q[:, 0]
model = load("tabicl_v2", task="regression", device="cuda")

for t in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
    model.estimator.fit(X, y + t * q_j)
    m_plus = model.estimator.predict(X).flatten()
    model.estimator.fit(X, y - t * q_j)
    m_minus = model.estimator.predict(X).flatten()
    diff = (m_plus - m_minus) / (2 * t)
    print(f"t={t:.1e}, diff norm: {np.linalg.norm(diff):.6f}, first elem: {diff[0]:.6f}, Q^T J Q norm: {np.linalg.norm(diff):.6f}")
