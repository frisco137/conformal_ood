# Standalone Models Package

This directory provides a self-contained interface to load and query **TabPFN v3**, **TabPFN v2**, **TabICL v2**, and **Google TabFM** models (both classifiers and regressors). 

The package is designed to be completely standalone. Even if other folders inside `conformal_ood` are moved or deleted, this package remains fully functional. To achieve this, local copies of the `tabicl` and `tabfm` codebases are embedded within this folder.

## Directory Structure

```
models/
├── README.md             # This document
├── __init__.py           # Package entrypoint & sys.path setup
├── tabpfn_v3.py          # TabPFN v3 classifier and regressor wrappers
├── tabpfn_v2.py          # TabPFN v2 classifier and regressor wrappers
├── tabicl_v2.py          # TabICL v2 classifier and regressor wrappers
├── google_tabfm.py       # Google TabFM classifier and regressor wrappers
├── tabicl/               # Standalone embedded copy of the TabICL codebase
├── tabfm/                # Standalone embedded copy of the Google TabFM codebase
└── run_sanity_check.py   # Script to verify all models on Gaussian noise
```

---

## Getting Started

To use the models, ensure you are running python within the project's virtual environment. Import the desired model loader from the `models` package:

```python
import models

# 1. TabPFN v3
clf_v3 = models.get_tabpfn_v3_classifier(device="cpu")
reg_v3 = models.get_tabpfn_v3_regressor(device="cpu")

# 2. TabPFN v2
clf_v2 = models.get_tabpfn_v2_classifier(device="cpu")
reg_v2 = models.get_tabpfn_v2_regressor(device="cpu")

# 3. TabICL v2
clf_icl = models.get_tabicl_v2_classifier(device="cpu")
reg_icl = models.get_tabicl_v2_regressor(device="cpu")

# 4. Google TabFM
clf_tfm = models.get_google_tabfm_classifier(device="cpu")
reg_tfm = models.get_google_tabfm_regressor(device="cpu")
```

All models conform to the standard `scikit-learn` interface (`fit`, `predict`, `predict_proba`).

---

## API Reference

### `models.get_tabpfn_v3_classifier(device="cpu", **kwargs)`
Instantiates and returns a `TabPFNClassifier` initialized with the default TabPFN v3 checkpoint (`tabpfn-v3-classifier-v3_default.ckpt`).

### `models.get_tabpfn_v3_regressor(device="cpu", **kwargs)`
Instantiates and returns a `TabPFNRegressor` initialized with the default TabPFN v3 checkpoint (`tabpfn-v3-regressor-v3_default.ckpt`).

### `models.get_tabpfn_v2_classifier(device="cpu", **kwargs)`
Instantiates and returns a `TabPFNClassifier` initialized with the default TabPFN v2 checkpoint (`tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`).

### `models.get_tabpfn_v2_regressor(device="cpu", **kwargs)`
Instantiates and returns a `TabPFNRegressor` initialized with the default TabPFN v2 checkpoint (`tabpfn-v2-regressor.ckpt`).

### `models.get_tabicl_v2_classifier(device="cpu", **kwargs)`
Instantiates and returns a local `TabICLClassifier` loaded with the default TabICL v2 classifier checkpoint (`tabicl-classifier-v2-20260212.ckpt`).

### `models.get_tabicl_v2_regressor(device="cpu", **kwargs)`
Instantiates and returns a local `TabICLRegressor` loaded with the default TabICL v2 regressor checkpoint (`tabicl-regressor-v2-20260212.ckpt`).

### `models.get_google_tabfm_classifier(device="cpu", **kwargs)`
Instantiates and returns a local `TabFMClassifier` loaded with pre-trained PyTorch weights (`google/tabfm-1.0.0-pytorch`).

### `models.get_google_tabfm_regressor(device="cpu", **kwargs)`
Instantiates and returns a local `TabFMRegressor` loaded with pre-trained PyTorch weights (`google/tabfm-1.0.0-pytorch`).

---

## Running Sanity Checks

To run the verification test, execute the sanity check script:

```bash
.venv/bin/python models/run_sanity_check.py
```

This runs both classification and regression for each model variant on synthetic Gaussian noise data and prints their performance (Accuracy/MSE).
