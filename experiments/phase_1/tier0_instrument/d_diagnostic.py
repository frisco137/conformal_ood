import sys
import numpy as np
from pathlib import Path
import warnings

# Suppress PyTorch/Scikit-learn warnings for cleaner output
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models import load
from experiments.phase_1.core.context import generate_audit_context
from experiments.phase_1.core.metrics import central_jacobian, asym, negeig

def run_diagnostic(model_name):
    print(f"\n{'='*80}")
    print(f"Diagnostic for: {model_name}")
    print(f"{'='*80}")
    
    X, y, f = generate_audit_context(n=100, d=5, seed=42)
    
    print(f"Loading {model_name}...")
    model = load(model_name, task="regression", device="cuda")
    
    def model_func(y_eval):
        trace = model.run(X, y_eval, X)
        return trace.pred
        
    print("\n--- D1: Establish the units ---")
    print(f"Raw Input (y) - Mean: {np.mean(y):.4f}, Std: {np.std(y):.4f}, Min: {np.min(y):.4f}, Max: {np.max(y):.4f}")
    
    m_y = model_func(y)
    print(f"Raw Output (m)- Mean: {np.mean(m_y):.4f}, Std: {np.std(m_y):.4f}, Min: {np.min(m_y):.4f}, Max: {np.max(m_y):.4f}")
    
    mae = np.mean(np.abs(m_y - y))
    print(f"Mean Absolute Error |m_i - y_i|: {mae:.4f}")
    
    print("\nWrapper introspection:")
    if hasattr(model.estimator, 'model_'):
        print(f"  Estimator model_ class: {model.estimator.model_.__class__.__name__}")
    if hasattr(model.estimator, 'preprocess_transforms'):
        print(f"  Preprocess transforms: {model.estimator.preprocess_transforms}")
    else:
        print("  preprocess_transforms attribute not found.")
        
    print("\n--- D4: Test Wrapper Form (Forward Pass Only) ---")
    c_shift = 10.0
    m_shift = model_func(y + c_shift)
    shift_err = np.mean(np.abs(m_shift - (m_y + c_shift)))
    print(f"Shift equivariance error m(y + c) - (m(y) + c): {shift_err:.4e}")
    
    c_scale = 2.0
    m_scale = model_func(y * c_scale)
    scale_err = np.mean(np.abs(m_scale - (m_y * c_scale)))
    print(f"Scale homogeneity error m(c*y) - c*m(y): {scale_err:.4e}")
    
    print("\n--- D2 & D3: Relative h Sweep & r1 Tuning ---")
    s_y = np.std(y) + 1e-8
    c_values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    
    print(f"{'c':<10} | {'h':<10} | {'r1':<10} | {'norm(J)':<10} | {'asym':<10} | {'negeig':<10}")
    print("-" * 80)
    
    for c in c_values:
        h = c * s_y
        J = central_jacobian(model_func, y, h=h)
        
        ones = np.ones(len(y))
        r1 = np.linalg.norm(J @ ones - ones) / np.linalg.norm(ones)
        norm_J = np.linalg.norm(J)
        a = asym(J)
        e = negeig(J)
        
        print(f"{c:<10.0e} | {h:<10.4e} | {r1:<10.4e} | {norm_J:<10.4f} | {a:<10.4f} | {e:<10.4f}")

def main():
    models = ["tabpfn_v2", "tabswift"]
    for m in models:
        try:
            run_diagnostic(m)
        except Exception as e:
            print(f"Error running {m}: {e}")

if __name__ == "__main__":
    main()
