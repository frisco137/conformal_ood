import numpy as np
from models import load
from experiments.core.context import generate_audit_context
from experiments.tier2_audit.instrument_c_d import get_Q, compute_reduced_jacobian, get_metrics

seeds = [42, 100, 200, 300, 400]
for seed in seeds:
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
    print(f"Seed {seed}, y[:3]: {y[:3]}")
    Q = get_Q(y)
    J = compute_reduced_jacobian("tabicl_v2", X, y, Q, 1e-3)
    a, n_eig, _ = get_metrics(J)
    print(f"  asym: {a:.6f}, negeig: {n_eig:.6f}")
