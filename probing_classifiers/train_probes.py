import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

# Import local modules
sys.path.insert(0, "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_ood")
sys.path.append("/home/psquare_a6000/Desktop/conformal_ood")
from data_generators import get_id_and_ood_data
from pfn_hooks import PFNHookManager

# Premium styling
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
})

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device for PFN inference: {device}")

    # Paths
    checkpoint_path = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_test_run/checkpoints/onefeature_gp_ls.1_pnf_4M.pt"
    base_dir = "/home/psquare_a6000/Desktop/conformal_ood/probing_classifiers"
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Generate Datasets (3000 ID and 3000 OOD samples)
    print("Generating datasets for probe training...")
    num_samples_per_class = 3000
    x_id, y_id, x_ood, y_ood = get_id_and_ood_data(
        num_samples_per_class=num_samples_per_class,
        num_points=100,
        device="cpu"
    )

    # Combine ID and OOD
    x_all = torch.cat([x_id, x_ood], dim=0) # (6000, 100, 1)
    y_all = torch.cat([y_id, y_ood], dim=0) # (6000, 100)
    labels = np.array([0] * num_samples_per_class + [1] * num_samples_per_class) # 0 = ID, 1 = OOD

    # 2. Extract Activations from Trained PFN
    hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, device=device)
    hook_manager.model.eval()

    batch_size = 128
    num_samples = len(x_all)
    print("Extracting PFN hidden activations...")

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="PFN Inference"):
            bx = x_all[i : i + batch_size].to(device)
            by = y_all[i : i + batch_size].to(device)

            src_x = bx.transpose(0, 1)
            src_y = by.transpose(0, 1)

            _ = hook_manager.model((src_x, src_y), single_eval_pos=100)

    layer_activations, _ = hook_manager.get_aggregated_features()
    hook_manager.remove_hooks()

    num_layers = len(layer_activations)
    print(f"Extracted activations for {num_layers} layers. Activations shape: {layer_activations[0].shape}")

    # 3. Train Probing Classifiers per Layer
    probe_results = []
    
    # Train-test split of indices
    indices = np.arange(num_samples)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    print("\nTraining linear probing classifiers (Logistic Regression)...")
    for l in range(num_layers):
        X_layer = layer_activations[l].numpy()
        X_train = X_layer[train_idx]
        X_test = X_layer[test_idx]

        # Fit Logistic Regression probe
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary')
        auc = roc_auc_score(y_test, probs)

        print(f"Layer {l} Probe -> Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        # Save model
        model_path = os.path.join(models_dir, f"logistic_regression_layer_{l}.joblib")
        joblib.dump(clf, model_path)

        probe_results.append({
            "layer": l,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(auc)
        })

    # Save results to JSON and CSV
    with open(os.path.join(base_dir, "probe_results.json"), 'w') as f:
        json.dump(probe_results, f, indent=4)

    df_results = pd.DataFrame(probe_results)
    df_results.to_csv(os.path.join(base_dir, "probe_results.csv"), index=False)

    # 4. Plot Probe Performance vs. Layer Depth
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(df_results["layer"], df_results["accuracy"], marker='o', color='#1f77b4', linewidth=2.5, label='Test Accuracy')
    plt.plot(df_results["layer"], df_results["auc"], marker='s', color='#ff7f0e', linewidth=2.5, label='Test ROC-AUC')
    
    plt.title("ID vs. OOD Linear Probing Classifier Performance", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("PFN Layer Depth", fontsize=11)
    plt.ylabel("Classification Performance Score", fontsize=11)
    plt.xticks(df_results["layer"])
    plt.ylim(0.45, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(base_dir, "probe_performance_vs_depth.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(base_dir, "probe_performance_vs_depth.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nLinear probe training complete! All outputs saved in {base_dir}/")

if __name__ == "__main__":
    main()
