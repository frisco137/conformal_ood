# baseline vs backprop comparison

## Table A: Pairwise Evaluation (Negative: ID Low Noise)

| Evaluation Task          | Indicator                   |    AUROC |     AUPR |   FPR@95 |
|:-------------------------|:----------------------------|---------:|---------:|---------:|
| Robustness to Noise      | UMAP-2D Mahalanobis         | 0.495223 | 0.940485 |    0.94  |
| Robustness to Noise      | PCA-10D Mahalanobis         | 0.651038 | 0.966055 |    0.935 |
| Robustness to Noise      | Prediction MSE (50 Qs)      | 1        | 1        |    0     |
| Robustness to Noise      | Prediction MSE (20 Qs)      | 1        | 1        |    0     |
| Robustness to Noise      | Gradient Norm Mean          | 0.999875 | 0.999874 |    0     |
| Robustness to Noise      | Gradient Cosine Similarity  | 0.9958   | 0.994967 |    0.015 |
| Robustness to Noise      | Magnitude-Coherence Product | 0.2605   | 0.451891 |    1     |
| Near-OOD Detection       | UMAP-2D Mahalanobis         | 0.480878 | 0.94321  |    0.93  |
| Near-OOD Detection       | PCA-10D Mahalanobis         | 0.683792 | 0.972954 |    0.935 |
| Near-OOD Detection       | Prediction MSE (50 Qs)      | 1        | 1        |    0     |
| Near-OOD Detection       | Prediction MSE (20 Qs)      | 1        | 1        |    0     |
| Near-OOD Detection       | Gradient Norm Mean          | 1        | 1        |    0     |
| Near-OOD Detection       | Gradient Cosine Similarity  | 0.9793   | 0.953301 |    0.04  |
| Near-OOD Detection       | Magnitude-Coherence Product | 1        | 1        |    0     |
| Far-OOD Detection        | UMAP-2D Mahalanobis         | 1        | 1        |    0     |
| Far-OOD Detection        | PCA-10D Mahalanobis         | 1        | 1        |    0     |
| Far-OOD Detection        | Prediction MSE (50 Qs)      | 1        | 1        |    0     |
| Far-OOD Detection        | Prediction MSE (20 Qs)      | 1        | 1        |    0     |
| Far-OOD Detection        | Gradient Norm Mean          | 1        | 1        |    0     |
| Far-OOD Detection        | Gradient Cosine Similarity  | 0.973275 | 0.930195 |    0.065 |
| Far-OOD Detection        | Magnitude-Coherence Product | 1        | 1        |    0     |
| Real-World OOD Detection | UMAP-2D Mahalanobis         | 0.99285  | 0.999569 |    0     |
| Real-World OOD Detection | PCA-10D Mahalanobis         | 0.999992 | 1        |    0     |
| Real-World OOD Detection | Prediction MSE (50 Qs)      | 1        | 1        |    0     |
| Real-World OOD Detection | Prediction MSE (20 Qs)      | 1        | 1        |    0     |
| Real-World OOD Detection | Gradient Norm Mean          | 1        | 1        |    0     |
| Real-World OOD Detection | Gradient Cosine Similarity  | 0.98955  | 0.985675 |    0.025 |
| Real-World OOD Detection | Magnitude-Coherence Product | 0.893875 | 0.941386 |    0.97  |

## Table B: Robust Mixed ID Evaluation (Negative: ID Low + ID High Noise)

| Evaluation Task          | Indicator                   |    AUROC |     AUPR |      FPR@95 |
|:-------------------------|:----------------------------|---------:|---------:|------------:|
| Near-OOD Detection       | UMAP-2D Mahalanobis         | 0.484144 | 0.517031 | 0.944118    |
| Near-OOD Detection       | PCA-10D Mahalanobis         | 0.561811 | 0.598278 | 0.950294    |
| Near-OOD Detection       | Prediction MSE (50 Qs)      | 0.999788 | 0.999565 | 0.0025      |
| Near-OOD Detection       | Prediction MSE (20 Qs)      | 0.999688 | 0.999362 | 0.0025      |
| Near-OOD Detection       | Gradient Norm Mean          | 0.999912 | 0.999825 | 0           |
| Near-OOD Detection       | Gradient Cosine Similarity  | 0.525125 | 0.31028  | 0.52        |
| Near-OOD Detection       | Magnitude-Coherence Product | 0.999788 | 0.999578 | 0           |
| Far-OOD Detection        | UMAP-2D Mahalanobis         | 0.999902 | 0.999873 | 0.000294118 |
| Far-OOD Detection        | PCA-10D Mahalanobis         | 1        | 1        | 0           |
| Far-OOD Detection        | Prediction MSE (50 Qs)      | 1        | 1        | 0           |
| Far-OOD Detection        | Prediction MSE (20 Qs)      | 1        | 1        | 0           |
| Far-OOD Detection        | Gradient Norm Mean          | 1        | 1        | 0           |
| Far-OOD Detection        | Gradient Cosine Similarity  | 0.50525  | 0.304587 | 0.5325      |
| Far-OOD Detection        | Magnitude-Coherence Product | 1        | 1        | 0           |
| Real-World OOD Detection | UMAP-2D Mahalanobis         | 0.99264  | 0.994726 | 0.000294118 |
| Real-World OOD Detection | PCA-10D Mahalanobis         | 0.999959 | 0.999958 | 0           |
| Real-World OOD Detection | Prediction MSE (50 Qs)      | 0.999663 | 0.999307 | 0.0025      |
| Real-World OOD Detection | Prediction MSE (20 Qs)      | 0.99915  | 0.998288 | 0.005       |
| Real-World OOD Detection | Gradient Norm Mean          | 0.999387 | 0.998732 | 0.005       |
| Real-World OOD Detection | Gradient Cosine Similarity  | 0.65665  | 0.385858 | 0.5125      |
| Real-World OOD Detection | Magnitude-Coherence Product | 0.908412 | 0.917393 | 0.76        |