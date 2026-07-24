#!/usr/bin/env python3
import sys
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Ensure the parent directory of models is in sys.path
parent_dir = str(Path(__file__).parent.parent.resolve())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sanity_check")

def generate_classification_data(seed=42):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(100, 10))
    y_train = rng.choice([0, 1], size=(100,))
    X_test = rng.normal(size=(50, 10))
    y_test = rng.choice([0, 1], size=(50,))
    return X_train, y_train, X_test, y_test

def generate_regression_data(seed=42):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(100, 10))
    y_train = rng.normal(size=(100,))
    X_test = rng.normal(size=(50, 10))
    y_test = rng.normal(size=(50,))
    return X_train, y_train, X_test, y_test

def check_classification(model_name, get_model_fn, X_train, y_train, X_test, y_test):
    logger.info(f"Checking Classification for {model_name}...")
    try:
        clf = get_model_fn(device="cpu")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info(f"{model_name} Classification Accuracy: {acc:.4f} (Chance is ~0.5)")
        return acc, None
    except Exception as e:
        logger.error(f"{model_name} Classification Failed: {e}", exc_info=True)
        return None, str(e)

def check_regression(model_name, get_model_fn, X_train, y_train, X_test, y_test):
    logger.info(f"Checking Regression for {model_name}...")
    try:
        reg = get_model_fn(device="cpu")
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        logger.info(f"{model_name} Regression MSE: {mse:.4f}, R2: {r2:.4f}")
        return mse, r2, None
    except Exception as e:
        logger.error(f"{model_name} Regression Failed: {e}", exc_info=True)
        return None, None, str(e)

def main():
    print("=" * 80)
    print("RUNNING GAUSSIAN NOISE SANITY CHECKS FOR ALL MODELS")
    print("=" * 80)

    # 1. Generate datasets
    X_train_c, y_train_c, X_test_c, y_test_c = generate_classification_data()
    X_train_r, y_train_r, X_test_r, y_test_r = generate_regression_data()

    # 2. Define classifiers and regressors
    classifiers = {
        "TabPFN v3": models.get_tabpfn_v3_classifier,
        "TabPFN v2": models.get_tabpfn_v2_classifier,
        "TabICL v2": models.get_tabicl_v2_classifier,
        "Google TabFM": models.get_google_tabfm_classifier,
    }

    regressors = {
        "TabPFN v3": models.get_tabpfn_v3_regressor,
        "TabPFN v2": models.get_tabpfn_v2_regressor,
        "TabICL v2": models.get_tabicl_v2_regressor,
        "Google TabFM": models.get_google_tabfm_regressor,
    }

    results = []

    print("\n--- CLASSIFICATION SANITY CHECKS ---")
    for name, getter in classifiers.items():
        acc, err = check_classification(name, getter, X_train_c, y_train_c, X_test_c, y_test_c)
        results.append({
            "model": name,
            "task": "classification",
            "metric": f"Accuracy: {acc:.4f}" if acc is not None else "ERROR",
            "status": "PASSED" if err is None else f"FAILED: {err}"
        })

    print("\n--- REGRESSION SANITY CHECKS ---")
    for name, getter in regressors.items():
        mse, r2, err = check_regression(name, getter, X_train_r, y_train_r, X_test_r, y_test_r)
        results.append({
            "model": name,
            "task": "regression",
            "metric": f"MSE: {mse:.4f}, R2: {r2:.4f}" if mse is not None else "ERROR",
            "status": "PASSED" if err is None else f"FAILED: {err}"
        })

    print("\n" + "=" * 80)
    print("SUMMARY OF SANITY CHECK RESULTS")
    print("=" * 80)
    print(f"{'Model':<15} | {'Task':<15} | {'Metric':<25} | {'Status':<15}")
    print("-" * 80)
    failed = False
    for res in results:
        print(f"{res['model']:<15} | {res['task']:<15} | {res['metric']:<25} | {res['status']:<15}")
        if "FAILED" in res["status"]:
            failed = True
    print("=" * 80)

    if failed:
        sys.exit(1)
    else:
        print("ALL SANITY CHECKS COMPLETED SUCCESSFULLY!")
        sys.exit(0)

if __name__ == "__main__":
    main()
