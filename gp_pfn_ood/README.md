# Unsupervised Mechanistic Interpretability of PFNs for GP Regression

This directory contains a modular pipeline for performing an unsupervised mechanistic interpretability analysis on the pre-trained Prior-Data Fitted Network (PFN) for 1D Gaussian Process regression.

The analysis is designed to investigate how the Transformer encoder’s internal representations differentiate between **In-Distribution (ID)** functional structures (Gaussian Processes with hyperpriors) and **Out-of-Distribution (OOD)** functional signals (discontinuous steps, high-frequency periodic waves, and non-stationary chirps) without using supervised training, classification labels, or linear probes.

---

## 1. Directory Structure

```directory
gp_pfn_ood/
├── data_generators.py    # Synthetic data generation for ID and OOD functions
├── pfn_hooks.py          # PyTorch MultiheadAttention patching and layer forward hooks
├── analyze_mechanisms.py # Main orchestration script running analyses and saving plots/JSON
├── README.md             # Technical documentation of the experiment
└── results/
    ├── mechanistic_summary.json  # Exported metrics (Silhouette scores, entropies, intrinsic dims)
    └── mechanistic_plots/        # High-resolution PNG visualizations of the analysis
```

---

## 2. Requirements & Installation

To run this pipeline, install the following dependencies in your Python environment:

```bash
pip install numpy torch matplotlib tqdm scikit-learn umap-learn
```

---

## 3. Analysis Pipeline Details

The pipeline executes the following 5 steps:

### Step 1: Data Generation (`data_generators.py`)
Generates 6,000 datasets (sequence length $N = 100$, feature dimension $d = 1$) split equally between:
1. **In-Distribution (ID) (3,000 datasets)**: Samples from a Gaussian Process prior with a Matérn kernel ($\nu=2.5$). The hyper-parameters for each dataset are drawn from distributions matching Samuel Müller's PFN paper:
   - Lengthscale $\ell \sim \Gamma(3.0, 6.0)$
   - Output scale $\sigma_f^2 \sim \Gamma(2.0, 0.15)$
   - Noise variance $\sigma_n^2 \sim \Gamma(0.0001, 1.0)$
2. **Out-of-Distribution (OOD) (3,000 datasets pooled)**:
   - **Discontinuous Step (1,000)**: Piecewise flat functions with 1 to 3 vertical jumps at random thresholds plus $\mathcal{N}(0, 0.01)$ noise.
   - **High-Frequency Periodic (1,000)**: Sine waves $y = A \sin(\omega x + \phi) + \mathcal{N}(0, 0.01)$ with $\omega \in [50, 100]$.
   - **Non-Stationary Chirp (1,000)**: Signals $y = \sin(e^{\alpha x}) + \mathcal{N}(0, 0.01)$ where $\alpha \sim \text{Uniform}(3, 6)$.

### Step 2: PyTorch Residual Stream Hooking (`pfn_hooks.py`)
- Automatically patches `MultiheadAttention.forward` call to set `need_weights=True`, forcing the layers to output attention probability distributions.
- Attaches forward hooks on all 6 `TransformerEncoderLayer` modules in the network to capture:
  - The residual stream activations $z_l$ post-FFN and LayerNorm (shape `[seq_len, batch_size, hidden_dim]`).
  - The attention maps from `self_attn` (shape `[batch_size, seq_len, seq_len]`).
- Tensors are detached and immediately offloaded to the CPU to prevent GPU memory overflow.

### Step 3: Inference and Feature Extraction
- The 6,000 datasets are evaluated in batches of 128 under `torch.no_grad()`.
- Sequence-mean representations are computed by averaging the residual stream $z_l$ along the sequence dimension, yielding a single vector of size `[hidden_dim]` per dataset at each layer.

### Step 4 & 5: Unsupervised Analysis & Outputs (`analyze_mechanisms.py`)
The pipeline runs the following five unsupervised analyses across the layer depth (0 to 5):

* **Analysis A: Dimensionality Reduction (PCA / t-SNE / UMAP)**: Fits 2D representations of sequence-mean activations at each layer and plots the scatters colored strictly by binary label (ID vs. OOD). Saves to `results/mechanistic_plots/layer_{l}_dim_reduction.png`.
* **Analysis B: Cluster Separation (Silhouette Score)**: Fits a $K=2$ K-Means clustering algorithm on sequence-mean activations without using class labels. Evaluates the Silhouette Score to measure natural cluster separation. Saves plot to `results/mechanistic_plots/silhouette_scores.png`.
* **Analysis C: Intrinsic Dimensionality (PCA Explained Variance)**: Fits PCA separately on the ID and OOD activations. Measures how many principal components are required to explain 95% of the variance. Saves plot to `results/mechanistic_plots/intrinsic_dimensionality.png`.
* **Analysis D: Shannon Attention Entropy**: Computes the Shannon entropy of the attention distribution averaged across heads and tokens:
  $$H = - \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^N A_{ij} \log (A_{ij} + \epsilon)$$
  Plots the mean entropy with shaded variance across layers for ID and OOD. Saves plot to `results/mechanistic_plots/attention_entropy.png`.
* **Analysis E: Residual Update Magnitude ($\Delta z_l$)**: Measures the $L_2$ norm of consecutive layer representation updates $\|z_l - z_{l-1}\|_2$. Plots the update magnitude for transitions. Saves plot to `results/mechanistic_plots/residual_updates.png`.

---

## 4. Execution

To run the pipeline manually, execute the following command from the workspace root:

```bash
python3 gp_pfn_ood/analyze_mechanisms.py
```

This will run the entire extraction and analysis pipeline, saving all plot figures to `gp_pfn_ood/results/mechanistic_plots/` and the numerical summary to `gp_pfn_ood/results/mechanistic_summary.json`.
