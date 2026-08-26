Camp directory: b
Number of files: 11
Files:
- A Statistical Perspective on In-Context Learning Softmax Attention as Nadaraya-Watson Estimator.md
- Emergence of Causal Structure Representations in In-Context Learning.md
- In-Context Learning as Kernel Regression.md
- Is One Layer Enough Layer Duplication and Redundancy in Tabular Foundation Models.md
- KernelICL Non-Parametric Kernel Regression Surrogates for Tabular In-Context Learners.md
- Layerwise Probing of In-Context Feature Selection in Tabular Transformers.md
- Spectral Analysis of In-Context Learning in Tabular Transformers.md
- TabPFN Behaves like 1-Nearest Neighbor in 2D Feature Spaces.md
- What Has TabPFN Learned Comparing In-Context Tabular Predictions to Gaussian Process Regressors.md
- Where Computation Lives Functional Head Specialization in Tabular Foundation Models.md
Date of extraction: 2026-08-08

## A Statistical Perspective on In-Context Learning Softmax Attention as Nadaraya-Watson Estimator

### Metadata
- **Title as printed in the document:** Supervised learning pays attention
- **Authors:** Erin Craig, Robert Tibshirani
- **Venue / year:** December 11, 2025
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** Title in document ("Supervised learning pays attention") is completely different from the filename ("A Statistical Perspective on In-Context Learning Softmax Attention as Nadaraya-Watson Estimator.md"). No venue or arXiv ID.

### One-paragraph summary
The paper adapts the concept of attention from in-context learning to standard tabular supervised learning methods like lasso regression and gradient boosting. It proposes "attention lasso", which fits a personalized local model for each test point by weighting the training data based on supervised similarity. The similarity is derived using random forest proximity or ridge regression coefficients, and the final prediction is a convex combination of a global baseline model and this personalized local model.

### Object of study
A mathematical construction (attention-weighted lasso regression and gradient boosting, using random forest proximity or ridge coefficients for weights) and tabular supervised learning. No neural network or LLM is studied; the focus is on standard ML methods applied to tabular data.

### Central claims
1. `[PARA]` Applying attention-based observation weighting to standard tabular supervised models improves predictive performance compared to unweighted baseline models. (Section 1, Section 3.1)
2. `[PARA]` Attention weighting allows the model to adapt to heterogeneous data and overlapping subgroups dynamically, without requiring pre-specified clusters. (Section 1, Section 3.2)
3. `[PARA]` Attention-weighted linear models attain lower mean squared error than standard linear models under a mixture-of-models data generating process. (Section 1, Appendix A Theorem A.10)
4. `[INFER]` Random forest proximity is empirically a much better similarity measure for attention weights than ridge-weighted inner products because it captures non-linear feature interactions. (Section 2.1)

### Key equations
Equation (1):
$$w_i^* = 	ext{softmax}({x^*}^T \hat{W} x_i)$$
- $w_i^*$: attention weight for the $i$-th training observation for test point $x^*$
- $	ext{softmax}$: row-wise softmax function
- $x^*$: test observation feature vector
- $\hat{W}$: diagonal matrix whose entries are absolute values of coefficients from ridge regression
- $x_i$: $i$-th training observation feature vector

Equation (2):
$$\hat{y}^* = (1-m)\hat{y}^*_{	ext{base}} + m \hat{y}^*_{	ext{attn}}$$
- $\hat{y}^*$: final prediction for test point $x^*$
- $m$: mixing hyperparameter selected via cross-validation ($m \in [0, 1]$)
- $\hat{y}^*_{	ext{base}}$: prediction from baseline model fit without weights
- $\hat{y}^*_{	ext{attn}}$: prediction from attention-weighted model

### Notation notes
`[NOT FOUND]`

### Method / instrument
The authors developed a novel algorithm (attention lasso/attention weighting) that first computes a supervised similarity score between a test point and all training points using a random forest. These similarity scores are passed through a softmax to create attention weights. A separate, weighted lasso (or boosting) model is then fit for the test point using these weights, and its prediction is blended with a global baseline model using a cross-validated mixing parameter. They benchmarked this on 12 UCI datasets and 4 synthetic datasets against standard models (lasso, LightGBM, XGBoost, RF, KNN). They also provided theoretical analysis comparing attention lasso to standard lasso under a Gaussian mixture data generating process.

### Scope conditions
The empirical benchmarking is scoped to tabular regression tasks with fewer than $n=5000$ observations. The theoretical guarantee of reduced prediction error (Theorem A.10) is proved specifically for linear models on data generated from a mixture of two linear models with isotropic Gaussian covariates (equal covariance $\Sigma_1 = \Sigma_2 = \Sigma$) where the mixture subgroups are separable with respect to the ridge-weighted metric (Assumption A.4).

### Epistemic status of each central claim
1. MEASURED (toy)
2. MEASURED (toy)
3. PROVED
4. ASSERTED

### Measured vs assumed
The paper assumes that "one-size-fits-all" models are sub-optimal for heterogeneous data where latent subgroups exist. The paper measures the predictive improvement of their personalized attention-weighted models on real and synthetic datasets. The theoretical reduction in MSE is proved under assumed Gaussian mixture data-generating conditions.

### Interventions on inputs
`[NOT FOUND]`

### Sensitivity and derivative analysis
`[NOT FOUND]`

### Relationship between prediction and observed targets
`[NOT FOUND]`

### Datasets, models, scale
12 regression datasets from the UCI Machine Learning Repository with $n < 5000$. Four simulated settings with $n=300$ and $p=30$ or $p=100$. Models evaluated include lasso, XGBoost, LightGBM, Random Forest, and KNN. Computations used standard R packages.

### Limitations stated by the authors
`[NOT FOUND]`

### Open questions named by the authors
`[NOT FOUND]`

### Notes outside the schema
The paper contains a detailed mathematical mapping (Appendix B) between Gaussian kernel regression weights and softmax attention weights, concluding they are identical only when the data rows are $\ell_2$-normalized and the softmax temperature $	au$ equals the kernel bandwidth squared $\sigma^2$. The authors note that the computational cost of their method is similar to leave-one-out cross validation, as they fit a model for every single test point. This is extremely slow compared to standard inference.

### Importance rating
2. It is a statistics paper adapting the *concept* of attention into classical tabular models (lasso, boosting), rather than analyzing frozen in-context predictors. While it helps draw a mathematical link between attention and Nadaraya-Watson estimators, it does not actually study the object of interest (LLMs/transformers performing ICL).

**DEEP PASS RECOMMENDED: no**


## Emergence of Causal Structure Representations in In-Context Learning

### Metadata
- **Title as printed in the document:** Does TabPFN Understand Causal Structures?
- **Authors:** Omar Swelam, Lennart Purucker, Jake Robertson, Hanne Raum, Joschka Boedecker, Frank Hutter
- **Venue / year:** EurIPS 2025 Workshop: AI for Tabular Data.
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** Title in document ("Does TabPFN Understand Causal Structures?") differs completely from filename ("Emergence of Causal Structure Representations in In-Context Learning.md"). The venue says "EurIPS 2025" which may be a typo for NeurIPS.

### One-paragraph summary
The paper investigates whether TabPFN, a transformer-based tabular foundation model pre-trained on synthetic data, encodes causal information in its internal representations. The authors freeze TabPFN's encoder and train a dual-attention decoder with causal tokens to extract and decode these representations into causal adjacency matrices. Their evaluations show that TabPFN's mid-range layers contain causal information that outperforms several traditional causal discovery algorithms, though performance degrades on larger, denser graphs.

### Object of study
A frozen production checkpoint (TabPFNv2 classification backbone encoder).

### Central claims
1. `[PARA]` TabPFN's embeddings contain causal information that can be extracted using an adapter framework for causal discovery. (Abstract, Section 4.2)
2. `[PARA]` The causal information in TabPFN is concentrated in its mid-range layers (specifically layers 4-6) rather than its initial or final layers. (Abstract, Section 4.2)
3. `[PARA]` The proposed extraction framework relying on frozen TabPFN embeddings outperforms traditional statistical baseline methods (GIES, IGSP, DCDI) in causal discovery. (Section 4.2)
4. `[PARA]` Using random weights or a fine-tuned encoder that degrades predictive performance also diminishes causal accuracy, suggesting TabPFN's causal encoding is a result of its pre-training for predictive tasks. (Section 4.2)
5. `[INFER]` The method struggles to scale to larger feature sizes, particularly on dense graph structures, likely because TabPFN was pre-trained primarily on small, sparse graphs for predictive objectives that focus only on a single target variable. (Section 4.2, Appendix D.2)

### Key equations
Equation (1):
$$\mathcal{L} = rac{1}{f^2} \sum_{i,j=1}^{f} A_{ij} \log(\hat{A}_{ij}) + (1 - A_{ij}) \log(1 - \hat{A}_{ij})$$
- $\mathcal{L}$: Binary cross-entropy loss
- $f$: number of features
- $A_{ij}$: ground-truth adjacency matrix entry
- $\hat{A}_{ij}$: predicted edge probability

Equation (2):
$$x_j = f_j(\mathbf{x}_{	ext{pa}(j)}) + arepsilon_j$$
- $x_j$: causal variable
- $f_j$: data generating function (linear or random Fourier feature)
- $\mathbf{x}_{	ext{pa}(j)}$: parents of $x_j$
- $arepsilon_j$: additive noise term

### Notation notes
`[NOT FOUND]`

### Method / instrument
The authors developed a framework using a frozen TabPFNv2 encoder. They feed it a dataset of observational and interventional samples where each feature value is paired with a binary interventional label. The representations from an intermediate layer (layer 4) are extracted. They then train a dual-attention decoder with $t=30$ learnable causal tokens that cross-attend to these frozen representations. The causal tokens are aggregated into 4 per-feature tokens via min, max, mean, and std, projected linearly, and scored via dot product to predict edge probabilities of the causal DAG. The decoder is trained on 100,000 synthetic datasets (sampled from 5 graph topologies) using cross-entropy and an acyclicity constraint, and evaluated on 500 new datasets against AVICI and statistical baselines (GIES, IGSP, DCDI).

### Scope conditions
The claims are evaluated on synthetic tabular datasets with 4 to 20 features, featuring linear or random Fourier feature mechanisms and additive Gaussian, Laplace, or Cauchy noise. The graphs include Erdős-Rényi, Scale-Free, Watts-Strogatz, Stochastic Block, and Geometric Random Graphs. The interventions are limited to single-variable interventions with known targets.

### Epistemic status of each central claim
1. MEASURED (production)
2. MEASURED (production)
3. MEASURED (production)
4. MEASURED (production)
5. ASSERTED

### Measured vs assumed
The authors assume that TabPFN's pre-training on synthetic data generated from SCMs might implicitly endow it with causal knowledge. They assume causal sufficiency in their data generation. They measure the ROC AUC and AP scores of their adjacency matrix predictions compared to statistical baselines and AVICI, measuring how layer choice and encoder weights impact causal extraction performance.

### Interventions on inputs
Yes. The paper modifies the input dataset to TabPFN: to differentiate interventional from observational samples, they append a binary label to each feature indicating if it is interventional. "This makes our data input shape different from the original TabPFN input shape... to be (n, f, 2)". They generate interventional data by single-variable interventions with uniform random values. However, they do not ablate or intervene on TabPFN's weights (other than testing a random-initialized or worsened baseline), instead they freeze them.

### Sensitivity and derivative analysis
`[NOT FOUND]`

### Relationship between prediction and observed targets
`[NOT FOUND]`

### Datasets, models, scale
Datasets are synthetically generated DAGs (Erdős-Rényi, Scale-Free, Watts-Strogatz, SBM, Geometric Random Graphs) with 4-20 features, 300 observational and 300 interventional samples (or sometimes just observational). The model is TabPFNv2 classification backbone (frozen), supplemented by a 6M parameter decoder (3.6M learnable). Training was 100,000 steps with batch size 32. Evaluated against AVICI, GIES, IGSP, DCDI.

### Limitations stated by the authors
The method faces an increasing degradation in Average Precision (AP) scores at scale (when the number of features increases). The authors note this scaling issue is more pronounced in dense graph structures (Erdős-Rényi, SBM, Watts-Strogatz).

### Open questions named by the authors
`[NOT FOUND]`

### Notes outside the schema
The model architecture aggregates 30 causal tokens down to 4 tokens using a simple concatenation of standard statistical operations (max, min, mean, std) over the token dimension before performing the linear projections for DAG prediction (Appendix A.3). This is an interesting manual inductive bias for summarizing transformer representations. The authors also show that removing their learned decoder entirely and using TabPFN's encoder to process the causal tokens as if they were test points results in the worst performance, proving that the decoder itself is critical to extracting the causal signal (Appendix D.1).

### Importance rating
4. This is a very interesting paper directly interrogating the internal representations of a frozen TFM (TabPFNv2) to see if it implicitly learns causal structures from its predictive pre-training. It cleanly isolates the frozen features and demonstrates they contain usable causal signal.

**DEEP PASS RECOMMENDED: no**


## Is One Layer Enough Layer Duplication and Redundancy in Tabular Foundation Models

### Metadata
- **Title as printed in the document:** Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models
- **Authors:** Amir Rezaei Balef, Mykhailo Koshil, Katharina Eggensperger
- **Venue / year:** Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026
- **arXiv or other ID, transcribed exactly as it appears:** `[NOT FOUND]`
- **Metadata anomalies:** The title in the document ("Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models") differs slightly from the filename ("Is One Layer Enough Layer Duplication and Redundancy in Tabular Foundation Models.md").

### One-paragraph summary
This paper provides the first large-scale mechanistic study of layer-wise dynamics in tabular foundation models (TFMs), comparing six state-of-the-art models (TabPFN v1/v2/2.5, TabICL, LimiX-2M/16M). The authors investigate how predictions emerge across depth using methods inspired by mechanistic interpretability of large language models (LLMs), such as embedding similarity (CKA), separation gap analysis, probing classifiers, a "tabular logit lens" (individual layer decoders), and layer ablations (skipping, repeating, swapping). They discover that TFMs iteratively refine their representations through highly redundant and overlapping computations in middle and later layers, and are capable of significant self-repair when these layers are removed, unlike early layers which are critical. Informed by this depth-wise redundancy, they design a proof-of-concept "looped" single-layer TFM (nanoTabPFNlooped) that achieves comparable performance to a six-layer model using only 20% of the parameters. 

### Object of study
Tabular Foundation Models (TFMs): specifically TabPFN(v1), TabPFN(v2), TabPFN(2.5), TabICL, LimiX-2M, and LimiX-16M.

### Central claims
1. `[MEASURED (production)]` Inference in TFMs unfolds iteratively, with early layers performing critical latent mapping and middle/later layers performing highly redundant, overlapping refinement of representations. (Section 3, Experiments 1-6)
2. `[MEASURED (production)]` TFMs exhibit substantial self-repair; skipping middle or later layers causes an immediate performance drop that is recovered by subsequent layers, indicating shared computational roles. (Section 3, Experiment 6)
3. `[MEASURED (production)]` TFMs form depth-wise "blocks" of layers where the embedding representation changes very little, unlike the sharper transitions observed between blocks or in early layers. (Section 3, Experiment 1)
4. `[MEASURED (production)]` Unlike LLMs, which rely heavily on their final layer for residual calibration ("sharpening"), TFMs are much less dependent on their final layers and can form highly accurate predictions early in the forward pass. (Section 4)
5. `[MEASURED (toy)]` A proof-of-concept looped single-layer model (`nanoTabPFNlooped`) can match the performance of a standard six-layer model by repeatedly applying the same transformer block, proving that depth primarily facilitates iterative computation rather than distinct transformations. (Section 5)

### Key equations
`[NOT FOUND]`

### Notation notes
- $h_\ell(x_i)$: representation of sample $x_i$ at layer $\ell$.
- $\Delta_\ell$: Separation gap at layer $\ell$ (difference between average inter-class and intra-class distances).

### Method / instrument
The authors apply several mechanistic interpretability techniques to pre-trained frozen TFMs. These include: 1) Measuring embedding similarity (cosine and linear CKA) across layers. 2) Calculating the "separation gap" (distance between classes vs within classes). 3) Training probing classifiers (logistic regression, KNN, LDA) on intermediate embeddings. 4) A "Tabular Logit Lens," where individual decoders are trained for each intermediate layer to decode early representations. 5) Structural interventions involving skipping, repeating, and swapping layers. 6) Measuring "self-repair" by tracking intermediate performance recovery after a layer ablation. They test on the PMLBmini (34 datasets) and TabArena (15 datasets) benchmark suites for binary classification, multiclass classification, and regression. Finally, they train a small toy model (`nanoTabPFN`) to validate the "looped" single-layer hypothesis.

### Scope conditions
The empirical studies were conducted on specific small-to-medium scale tabular classification and regression tasks (datasets with $\le 10,000$ samples and $\le 100$ features). The models studied are encoder-only transformer TFMs. The "looped" model experiment was a small-scale proof-of-concept (`nanoTabPFN`) and was not tested on the largest architectures like TabPFN(2.5). All models were evaluated without ensembling.

### Epistemic status of each central claim
1. MEASURED (production)
2. MEASURED (production)
3. MEASURED (production)
4. MEASURED (production)
5. MEASURED (toy)

### Measured vs assumed
The authors assume that standard techniques for measuring LLM layer dynamics (CKA, logit lens, ablations) can be meaningfully adapted to encoder-only TFMs. They measure the internal distances, similarity matrices, and predictive accuracies of the models under these various analytical lenses and ablations. The conclusion that "one layer is enough" is inferred from the robustness to ablation and directly measured in the toy `nanoTabPFNlooped` experiment.

### Interventions on inputs
No. The interventions were structural (ablating, repeating, and swapping layers inside the frozen model), but the data inputs (TabArena, PMLBmini) were standard benchmark datasets without synthetic perturbation of the input space.

### Sensitivity and derivative analysis
The authors evaluate structural sensitivity extensively by skipping, swapping, and repeating layers (Section 3, Experiment 5). They find models are highly sensitive to swapping adjacent layers and highly sensitive to skipping the very first layer, but highly robust to skipping middle and later layers.

### Relationship between prediction and observed targets
`[NOT FOUND]`

### Datasets, models, scale
- **Models:** TabPFN(v1), TabPFN(v2), TabPFN(2.5), TabICL, LimiX-2M, LimiX-16M, and custom `nanoTabPFN`.
- **Datasets:** 15 binary, 6 multiclass, and 6 regression tasks from TabArena ($\le 10,000$ samples, $\le 100$ features); 34 binary tasks from PMLBmini ($\le 500$ samples); 20 multiclass tasks from OpenML CC18. 

### Limitations stated by the authors
The authors state they did not systematically investigate at what point task complexity necessitates greater effective depth. The Tabular Logit Lens relies on open-source TabICL priors, which may be suboptimal for other models. The looped transformer experiment was conducted only on the small `nanoTabPFN` architecture and may not directly extrapolate to larger models. All models were evaluated without ensembling.

### Open questions named by the authors
Future work includes studying TFMs at the neuronal and circuit level, studying the impact of prior design and pre-training setups using these methods, investigating if LLM-based tabular predictors show similar behavior, and scaling up the recurrent/looped transformer approach to larger models.

### Notes outside the schema
The paper maps four stages of TFM inference: 1) Latent Mapping (encoder extension), 2) Feature Engineering and Labeling (where labels form iteratively and classes separate), 3) Prediction Ensembling (aligning features with the decoder), and 4) Prediction Calibration (where ROC-AUC is stable but balanced accuracy and entropy adjust). This provides a nice contrast to the known LLM stages (Detokenization, Feature Engineering, Prediction Ensembling, Residual Calibration/Sharpening).

### Importance rating
4. This is an excellent paper that peeks into the black box of tabular foundation models, which are completely distinct from standard LLMs despite both using transformers. It provides highly detailed layer-wise behavior analysis across the major current models, establishing that the bulk of their computation is just iterative redundant refinement.

**DEEP PASS RECOMMENDED: yes**


## In-Context Learning as Kernel Regression

### Metadata
- **Title as printed in the document:** Understanding Emergent In-Context Learning from a Kernel Regression Perspective
- **Authors:** Chi Han, Ziqi Wang, Han Zhao, Heng Ji
- **Venue / year:** Transactions on Machine Learning Research (09/2025)
- **arXiv or other ID, transcribed exactly as it appears:** Reviewed on OpenReview: `https: // openreview. net/ forum? id= 6rD50Q6yYz`
- **Metadata anomalies:** The title in the document ("Understanding Emergent In-Context Learning from a Kernel Regression Perspective") differs from the filename ("In-Context Learning as Kernel Regression.md").

### One-paragraph summary
The paper investigates the mechanisms enabling pre-trained transformer-based language models to perform in-context learning. It establishes a theoretical framework showing that Bayesian inference on in-context prompts converges asymptotically to a non-parametric kernel regression as the number of demonstrative samples increases. The authors empirically evaluate this hypothesis on GPT-J-6B by examining attention weights and intermediate features, demonstrating that LLMs distribute attention in a manner consistent with kernel regression—particularly by attending to the labels of prediction-similar demonstration samples—and that replacing the model's actual predictions with a manual kernel regression over its attention weights reconstructs the model's outputs with high accuracy. 

### Object of study
An LLM (specifically GPT-J 6B, a decoder-only Transformer architecture) and theoretical mathematical formulations modeling language as a Hidden Markov Model (HMM). 

### Central claims
1. `[PROVED]` Bayesian inference on in-context prompts asymptotically converges to a form of kernel regression over the demonstration samples as the number of samples grows. (Theorem 1)
2. `[MEASURED (production)]` During in-context learning, attention within LLMs is concentrated on the labels of demonstration samples, mirroring the weighting mechanism of kernel regression. (Section 5.1)
3. `[MEASURED (production)]` Using the attention values manually to compute a kernel regression over the demonstration labels can reconstruct the LLM's actual predictions with up to 89.2% accuracy. (Section 5.2)
4. `[MEASURED (production)]` The attention heads responsible for ICL primarily attend to demonstration samples that exhibit high "prediction similarity" to the test sample. (Section 5.3)
5. `[PARA]` The kernel regression framework provides explanations for several empirical ICL phenomena, including why retrieving similar samples helps, why ICL is sensitive to label formatting, and why pre-training bias persists. (Section 4.2)

### Key equations
Equation (6):
$$y_{	ext{reg}} = \sum_i \mathbf{e}(y_i) rac{\exp( 	ext{vec}(T_{\mathbf{x}})^T \Sigma_{p_{	ext{pre-train}}}^{-1} 	ext{vec}(T_{\mathbf{x}'}) )}{\sum_{i'} \exp( 	ext{vec}(T_{\mathbf{x}})^T \Sigma_{p_{	ext{pre-train}}}^{-1} 	ext{vec}(T_{\mathbf{x}'}) )}$$
- $y_{	ext{reg}}$: prediction vector in the form of kernel regression
- $\mathbf{e}(y_i)$: one-hot vector for index $y_i$ (label information)
- $	ext{vec}(T_{\mathbf{x}})$: flattened vector of the transition matrix after sequence $\mathbf{x}$, denoting the semantic "belief state"
- $\Sigma_{p_{	ext{pre-train}}}^{-1}$: inverse covariance matrix of the pre-training transition elements
- $T_{\mathbf{x}'}$: transition matrix for sample $\mathbf{x}'$

Equation (7):
$$\left\| y_{	ext{reg}} - P(Y | \mathbf{o}_{ICL}, p_{	ext{pre-train}}) 
ight\|_1 \le O\left( \sqrt{rac{\eta^2 \epsilon_	heta \ln(1/\delta)}{n}} + \epsilon_r 
ight)$$
- $y_{	ext{reg}}$: kernel regression logit vector
- $P(Y | \mathbf{o}_{ICL}, p_{	ext{pre-train}})$: conditional likelihood of $Y$ conditioned on ICL prompt
- $\eta$: upper bound of $T_{\mathbf{o}[0:l]}$ Frobenius-norm
- $\epsilon_	heta$: difference between sequences generated by task initial state $s_	heta$ and pre-training distribution
- $n$: number of in-context samples
- $\delta$: probability bound
- $\epsilon_r$: lower bound of delimiter token emission rate

### Notation notes
`[NOT FOUND]`

### Method / instrument
The authors developed a theoretical proof under the assumption that the pre-training corpus is a mixture of Hidden Markov Models (HMMs) to show Bayesian inference converges to kernel regression. Empirically, they probed a frozen production checkpoint (GPT-J 6B). They extracted and averaged attention maps across layers to verify attention is concentrated on demonstration labels. They then performed an ablation/reconstruction experiment where they used the extracted attention values as weights in a manual kernel regression to reconstruct the model's next-token predictions, measuring how closely this synthetic output matched both the LLM's true logits and the ground truth labels. They also trained a Ridge regression on intermediate key and value vectors to prove they linearly encode prediction states and label information.

### Scope conditions
Theoretical convergence (Theorem 1) holds asymptotically as the number of demonstrative samples $n 	o \infty$, under the assumption that language can be modeled as a mixture of HMMs and assuming delimiter tokens indicate the start of independent samples (Assumption 1) and tasks are distinguishable (Assumption 2).

### Epistemic status of each central claim
1. PROVED
2. MEASURED (production)
3. MEASURED (production)
4. MEASURED (production)
5. ASSERTED

### Measured vs assumed
The paper assumes that language pre-training corpora can be modeled as a mixture of Hidden Markov Models, and that LLMs perform Bayesian inference over this mixture during ICL. It proves mathematically that this Bayesian inference approaches kernel regression. It then measures the actual attention maps and internal hidden states of GPT-J 6B on the GLUE-sst2 dataset and others to verify if these theoretical constructs map onto physical Transformer mechanisms.

### Interventions on inputs
Yes. In Section 4.2 (Table 1), the authors measure the effect of Out-Of-Distribution (OOD) inputs by using GPT-3.5-turbo to perturb the text inputs into "rare word" synonyms, "complex" structures, or adding "typos", observing that typos drastically drop ICL accuracy.

### Sensitivity and derivative analysis
`[NOT FOUND]`

### Relationship between prediction and observed targets
`[NOT FOUND]`

### Datasets, models, scale
Datasets used for empirical analysis include SST2, MNLI, Rotten Tomatoes, and Tweet Eval. The model studied is GPT-J 6B (a decoder-only Transformer). Empirical experiments were run on a single Tesla V100 GPU. The theoretical baseline assumes a sequence of length $l$ and $n$ demonstration samples.

### Limitations stated by the authors
The theoretical HMM mixture framework does not fully capture the complexity and generality of natural language with unbounded lengths and might be highly inefficient. Some ICL phenomena (like sensitivity to sample order or robustness to perturbed labels) cannot be explained by this kernel regression framework. The equivalence assumes LLMs always conduct Bayesian inference, which some recent literature disputes.

### Open questions named by the authors
Understanding the effect of sample orderings and robustness to perturbed labels in ICL remains an open question, requiring insights into the implicit reasoning capabilities of LLMs beyond simple kernel regression.

### Notes outside the schema
The paper provides an interesting mapping between theoretical concepts and Transformer components (Section 4.1 & 5.4): The label information $\mathbf{e}(y_i)$ maps to the `value` vectors. The semantic information vector $	ext{vec}(T_{\mathbf{x}})$ maps to the `key` and `query` vectors. The authors verified this mapping by performing ridge regression directly on the intermediate `key` and `value` vectors (Section 5.4) and found that `value` vectors perfectly encode the label and `key` vectors encode the LLM's context prediction. Also, the actual attention mechanism in GPT-J only strongly matched the theoretical kernel regression in a few middle-late layers (layers 16-21).

### Importance rating
4. This paper provides a very solid theoretical link showing that Bayesian inference (the standard theoretical model for ICL) reduces to kernel regression. Crucially, it backs this up with strong empirical evidence showing that LLMs actually compute this kernel regression using their attention heads on the label tokens. It is highly relevant to understanding what frozen in-context predictors compute.

**DEEP PASS RECOMMENDED: no**


## KernelICL Non-Parametric Kernel Regression Surrogates for Tabular In-Context Learners

### Metadata
- **Title as printed in the document:** Interpretable Tabular Foundation Models via In-Context Kernel Regression
- **Authors:** Ratmir Miftachov, Bruno Charron, Simon Valentin
- **Venue / year:** Preprint. February 3, 2026.
- **arXiv or other ID, transcribed exactly as it appears:** `[NOT FOUND]`
- **Metadata anomalies:** The title in the document ("Interpretable Tabular Foundation Models via In-Context Kernel Regression") differs significantly from the filename ("KernelICL Non-Parametric Kernel Regression Surrogates for Tabular In-Context Learners.md").

### One-paragraph summary
The paper addresses the black-box nature of tabular foundation models by introducing KernelICL, a framework that replaces the opaque MLP prediction head with explicit kernel functions (Gaussian, dot-product, kNN). By leveraging symmetric in-context embeddings—where training and test samples share identical projection mechanisms—the model produces predictions that are transparent weighted averages of training labels. The authors also present a two-dimensional taxonomy to classify kernel regression methods based on interpretability and dataset dependence, and they introduce perplexity as a metric to quantify the inspectability of the resulting weight distributions. Empirical evaluation on 55 TALENT benchmark datasets demonstrates that KernelICL variants achieve performance on par with existing models like TabICL while offering quantifiable, sample-based interpretability.

### Object of study
Tabular Foundation Models (specifically fine-tuned variations of TabICL) and kernel regression mechanisms.

### Central claims
1. `[ASSERTED]` KernelICL provides quantifiable, sample-based interpretability by replacing the opaque prediction head with transparent kernel functions. (Section 1)
2. `[ASSERTED]` A two-dimensional taxonomy unifies standard kernel methods, modern neighbor-based approaches, and attention mechanisms by classifying them based on interpretability and dataset dependence. (Section 3.1)
3. `[PROVED]` Symmetric in-context embeddings enable the use of distance-based kernels (like Gaussian and kNN) in ICL by ensuring query and key spaces are unified. (Section 3.2)
4. `[MEASURED (production)]` KernelICL variants achieve performance comparable to TabICL (e.g., KernelICL-Dot achieves 82.88% accuracy vs TabICL at 83.33%) across 55 TALENT datasets. (Section 4.2)
5. `[MEASURED (production)]` The accuracy-inspectability trade-off can be explicitly controlled by tuning the kernel scale, allowing practitioners to optimize for sparse, inspectable weights (measured via perplexity) with minimal accuracy loss. (Section 4.4)

### Key equations
Equation (4):
$$q_D = k_D = h_D, h_D : \mathcal{X} 	o \mathcal{H}$$
- $q_D$: query embedding function
- $k_D$: key embedding function
- $h_D$: shared geometric embedding function

Equation (8):
$$E = 	ext{TF}_{	ext{icl}}( 	ext{TF}_{	ext{row}}( 	ext{TF}_{	ext{col}}(X_{	ext{train}} \cup X_{	ext{test}}) ), 	ext{TF}_{	ext{row}}( 	ext{TF}_{	ext{col}}(X_{	ext{train}}) ), y_{	ext{train}} )$$
- $E$: in-context embedding
- $	ext{TF}_{	ext{col}}$: column-wise Set Transformer
- $	ext{TF}_{	ext{row}}$: row-wise self-attention
- $	ext{TF}_{	ext{icl}}$: label-conditioned in-context learning transformer

Equation (11):
$$K_\gamma(q, k) = \exp \left( - rac{\|q - k\|^2}{2\gamma^2} 
ight)$$
- $K_\gamma(q, k)$: Gaussian kernel function
- $q, k$: projected query and key embeddings
- $\gamma$: scale parameter controlling locality

### Notation notes
- $h_D$: the shared embedding function mapping inputs to a common space $\mathcal{H}$.
- $	ext{PPL}(w)$: perplexity of the weight vector, quantifying inspectability/sparsity.

### Method / instrument
The authors developed a theoretical taxonomy for kernel regression methods. They then modified the TabICL architecture by retaining its column and row processing stages but altering the final ICL stage to process training samples as queries, ensuring symmetric embeddings. They replaced the final MLP with explicit kernel functions (Dot-product, Gaussian, kNN). The embedding module was fine-tuned end-to-end on synthetic data using cross-entropy loss. They empirically evaluated the resulting KernelICL models on 55 TALENT benchmark datasets and on synthetic datasets (Moons, Circles, Linear with noise), measuring both predictive accuracy and weight inspectability (via perplexity).

### Scope conditions
The method targets tabular foundation models evaluated on small-to-medium tabular classification datasets (up to ~109,000 samples and ~1000 features in TALENT). The symmetric embedding modification incurs a computational overhead (up to 2x for large datasets) because training samples must be passed as both context and queries.

### Epistemic status of each central claim
1. ASSERTED
2. ASSERTED
3. PROVED
4. MEASURED (production)
5. MEASURED (production)

### Measured vs assumed
The authors assume that sample-based case retrieval (weighted averaging of training labels) is a highly desirable form of interpretability for domain experts. They mathematically prove that their symmetric embedding design unifies the query and key spaces. They directly measure the predictive accuracy (ROC-AUC/accuracy) and inspectability (perplexity) of these models on synthetic and real-world benchmark datasets.

### Interventions on inputs
Yes. In Section 4.1 (Synthetic Validation), the authors evaluate the robustness of learned embeddings by appending 18 Gaussian noise features to 2 signal features on synthetic Scikit-learn datasets (Moons, Circles, Linear), observing that standard kernels degrade while KernelICL filters the noise effectively.

### Sensitivity and derivative analysis
The authors perform extensive ablations on kernel scale calibration, the effect of symmetric vs non-symmetric embeddings, and the projection dimension $d_k$ (Section 4.3). They map the trade-off curve between perplexity (sparsity) and accuracy across different hyperparameters (Section 4.4).

### Relationship between prediction and observed targets
`[NOT FOUND]`

### Datasets, models, scale
- **Models:** KernelICL (built on TabICL), compared against TabICL, TabPFN, ModernNCA, and traditional tree-based models (CatBoost, XGBoost, etc.).
- **Datasets:** 55 binary classification datasets from the TALENT benchmark (ranging from 645 to 109,099 samples and 3 to 970 features); synthetic Scikit-learn datasets (Moons, Circles, Linear). 

### Limitations stated by the authors
The symmetric embedding procedure incurs an overhead, approaching a 2x factor in embedding time for large training sets compared to asymmetric embeddings. The kNN kernel is non-differentiable due to the sorting operation, so it requires embeddings trained with the Gaussian kernel as a proxy during fine-tuning.

### Open questions named by the authors
The authors hope this work encourages further research on interpretable prediction mechanisms for tabular foundation models. They mention that whether the model's clinically aligned neighbor selection on the Pima Indians Diabetes dataset reflects true learned causal structure requires further clinical validation.

### Notes outside the schema
The paper makes a compelling argument for switching from exponential dot-product attention (alignment-based) to isotropic distance-based kernels (like Gaussian and kNN) to achieve "spatial" interpretability, and implements the architectural changes needed to make this possible in an ICL setup.

### Importance rating
4. This paper introduces an elegant architectural tweak to tabular foundation models that forces their internal attention to act as explicit kernel regression, providing strong out-of-the-box interpretability. 

**DEEP PASS RECOMMENDED: no**


## Layerwise Probing of In-Context Feature Selection in Tabular Transformers

### Metadata
- **Title as printed in the document:** TabPFN Through The Looking Glass: An interpretability study of TabPFN and its internal representations
- **Authors:** Aviral Gupta, Armaan Sethi, Dhruv Kumar
- **Venue / year:** Not specified
- **arXiv or other ID, transcribed exactly as it appears:** `[NOT FOUND]`
- **Metadata anomalies:** The title in the document ("TabPFN Through The Looking Glass: An interpretability study of TabPFN and its internal representations") differs significantly from the filename ("Layerwise Probing of In-Context Feature Selection in Tabular Transformers.md"). The venue/year is absent.

### One-paragraph summary
This paper investigates the internal mechanistic representations of the TabPFN v2 tabular foundation model. The authors run a series of linear probing and logit lens experiments on synthetic datasets with known linear and arithmetic functional relationships ($z = \alpha x + \beta$, $z = a \cdot b + c$). They demonstrate that TabPFN encodes linear regression coefficients, intermediate arithmetic computation results (e.g., $a \cdot b$), and copied input values directly in its hidden residual stream, particularly in the middle layers (e.g., Layer 5-6). While the correct predicted answer forms computationally early (around Layer 5) as detected by probes, the model requires subsequent layers (up to Layer 7-8) to project this answer into the valid output space geometry required by the final unembedding head, demonstrating structured, multi-step hierarchical computation within tabular foundational models.

### Object of study
TabPFN v2 (a transformer-based tabular foundation model) and its internal residual stream activations.

### Central claims
1. `[MEASURED (production)]` Linear regression coefficients modeled by TabPFN are linearly decodable from the model's internal activations within a single context fit, showing the model encodes the functional relationship it is computing. (Section 3.2)
2. `[MEASURED (production)]` When approximating arithmetic relationships ($z = a \cdot b + c$), TabPFN encodes necessary intermediate computational quantities (like $a \cdot b$) in its middle layers before formulating the final answer, mirroring a mathematical order of operations. (Section 3.3)
3. `[MEASURED (production)]` The correct final answer for simple functional relationships can be extracted via linear probes as early as Layer 5. (Section 3.4)
4. `[MEASURED (production)]` While the answer is semantically computed early (by Layer 5), the model uses subsequent layers (especially Layer 7) to geometrically align this representation with the final output unembedding manifold, as shown by logit lens experiments. (Section 3.4)
5. `[MEASURED (production)]` Input values and their combinations are copied and encoded directly within the activations of the final answer token by the middle layers (Layer 5), acting similarly to copy/induction heads in LLMs. (Section 3.5)

### Key equations
Equation in Section 3.2:
$$y = \alpha x + \beta$$
- $y$: dependent variable
- $x$: independent variable
- $\alpha, \beta$: linear coefficients to be recovered via probing

Equation in Section 3.3:
$$z = a \cdot b + c$$
- $z$: final output variable
- $a, b, c$: input variables
- $a \cdot b$: intermediate product probed from hidden activations

### Notation notes
- $R^2$: Coefficient of determination used to score the linear probes' success.
- $k$: embedding dimension of TabPFN tokens.

### Method / instrument
The authors generated synthetic tabular datasets with known mathematical relationships (linear and compound arithmetic). They ran inference using a frozen pre-trained TabPFN v2 model and extracted the hidden layer activations at each layer. They trained lightweight probes (linear mappings or shallow MLPs) to map these flattened activation vectors to specific mathematical properties: functional coefficients ($\alpha, \beta$), intermediate computations ($a \cdot b$), and the final output ($z$). They evaluated probe performance using $R^2$ and MSE across layers and across probe complexities. They also utilized the logit lens technique (applying the final unembedding matrix to intermediate layers) to contrast the raw emergence of the answer with the probe-extracted answer.

### Scope conditions
The experiments are strictly limited to simple, synthetic arithmetic (additive/multiplicative) toy expressions and linear regression relationships, and do not test highly complex or noisy functions found in real-world tabular datasets. The coefficient probing results (Section 3.2) only generalize within a single combined context fit, and not across separate models fitted to isolated datasets.

### Epistemic status of each central claim
1. MEASURED (production)
2. MEASURED (production)
3. MEASURED (production)
4. MEASURED (production)
5. MEASURED (production)

### Measured vs assumed
The authors assume that standard LLM mechanistic interpretability techniques (linear probing, logit lens) are valid tools for analyzing continuous-value tabular transformers. They directly measure the $R^2$ accuracy of these linear probes on internal hidden states to conclude that the information (coefficients, intermediates, answers, inputs) is linearly encoded in the residual stream. 

### Interventions on inputs
No. The authors control the synthetic data generation to create specific arithmetic relationships, but they do not perform causal interventions (like activation patching or structural dataset perturbation) during the forward pass itself.

### Sensitivity and derivative analysis
The authors test the sensitivity of the probe results to the complexity of the probe itself (by adding hidden layers to an MLP). They show that increasing probe complexity actually decreases the $R^2$ score and increases MSE (Figure 2 and Figure 4), providing strong evidence that the target quantities are explicitly encoded *linearly* in the representations, rather than being computed by a complex non-linear probe.

### Relationship between prediction and observed targets
`[NOT FOUND]`

### Datasets, models, scale
- **Models:** TabPFN v2.
- **Datasets:** Synthetic toy datasets generated specifically to exhibit simple mathematical relationships ($y = \alpha x + \beta$, and $z = a \cdot b + c$).
- **Scale:** Small scale, single machine inference probing.

### Limitations stated by the authors
The analysis is limited to simple arithmetic (additive/multiplicative) toy expressions, which do not reflect the highly complex, noisy functions found in real-world tabular datasets. The paper acknowledges that coefficients are not encoded as a universal quantity across different model fits, meaning the representation geometry shifts depending on the context provided.

### Open questions named by the authors
Future work could entail finding the specific mechanisms/circuits through which the answer is computed using causal mechanistic interpretability methods like activation patching, or manipulating the function through steering vectors or vector ablation.

### Notes outside the schema
The distinction between the "probe-accessible" solution (Layer 5) and the "native output alignment" solution (Layer 8) provides an interesting insight into transformer computation: the bulk of late-stage processing in this task is simply rotating/projecting an already-solved answer into the correct basis for the output head.

### Importance rating
3. The paper is somewhat basic (probing simple math relationships), but it successfully confirms that the residual stream of TabPFN behaves mechanically similarly to an LLM doing math, storing intermediate computation steps linearly. 

**DEEP PASS RECOMMENDED: no**


## Spectral Analysis of In-Context Learning in Tabular Transformers

### Metadata
- **Title as printed in the document:** Mechanistic Evidence for Spectral Structures in Prior-Data Fitted Networks
- **Authors:** Kaustubh Sharma, Srijan Tiwari, Ojasva Nema, Parikshit Pareek
- **Venue / year:** Preprint (Year unstated, likely 2025/2026 based on citations).
- **arXiv or other ID, transcribed exactly as it appears:** `[NOT FOUND]`
- **Metadata anomalies:** The title in the document ("Mechanistic Evidence for Spectral Structures in Prior-Data Fitted Networks") differs from the filename ("Spectral Analysis of In-Context Learning in Tabular Transformers.md"). 

### One-paragraph summary
This paper investigates whether Prior-Data Fitted Networks (PFNs)—including TabPFN, VA-PFN, and DVA-PFN—implicitly learn and utilize structured Bayesian priors (specifically spectral representations of stationary kernels) during inference. Through mechanistic interpretability techniques (linear probing, activation patching, and targeted subspace patching), the authors demonstrate that the mean-pooled latent attention score ($ar{H}$) explicitly organizes spectral information along a low-dimensional causal subspace. This property emerges after a single cross-attention step and transfers even to real-world time-series data. Building on these findings, the authors introduce a "Filter Bank Decoder" that extracts explicit, portable spectral densities (and by extension, kernel matrices) from frozen PFN latents without any test-time optimization, achieving Gaussian Process regression performance competitive with iterative kernel discovery methods.

### Object of study
Prior-Data Fitted Networks (specifically TabPFN, VA-PFN, and DVA-PFN) and their internal latent attention score representations.

### Central claims
1. `[MEASURED (production)]` Spectral information (frequency and weight) is linearly decodable from the mean-pooled latent attention score ($ar{H}$) in PFNs, establishing that Bayesian structure is represented in an explicitly organized form. (Section 3.2)
2. `[MEASURED (production)]` The latent attention score ($ar{H}$) acts as a causal carrier of spectral identity, as shown by activation patching experiments where swapping $ar{H}$ fully transfers the spectral identity of the prediction. (Section 4.1)
3. `[MEASURED (production)]` Causally active spectral information is concentrated in a low-dimensional subspace within the latent representation; targeted patching of just the top principal components drives the causal effect more efficiently than random directions. (Section 4.2)
4. `[MEASURED (production)]` The causal spectral subspace encoding is an emergent property that holds not just on synthetic sinusoids but also on real-world time series data (Airline Passengers, Milk Production) evaluated via lag-embedded TabPFN. (Section 4.3)
5. `[MEASURED (toy)]` A diagnostic "Filter Bank Decoder" can successfully extract explicit spectral densities and kernel matrices from frozen PFN latents, achieving GP regression MSE competitive with DKL and RFF baselines at significantly lower latency. (Section 5)

### Key equations
Equation (1):
$$k(\tau) = \sum_{q=1}^{Q} w_q \exp \left( -2\pi^2 \tau^2 \sigma_q^2 \right) \cos (2\pi \tau \mu_q)$$
- $k(\tau)$: Spectral Mixture (SM) stationary covariance kernel
- $w_q$: mixture weight for component $q$
- $\sigma_q^2$: bandwidth for component $q$
- $\mu_q$: center frequency for component $q$

### Notation notes
- $ar{H}$: mean-pooled latent attention score across positions.
- CE: Causal Effect, measuring the fractional shift in prediction when patching activations.
- $f$: generating frequency of the input sinusoid.
- PC0: the first principal component of the latent embeddings.

### Method / instrument
The authors apply mechanistic interpretability techniques to continuous-variable foundation models. They use parameter-free geometric alignment checks (Pearson correlation of pairwise distances), linear and MLP probing classifiers to measure the accessibility of frequency information in $ar{H}$, and full-tensor as well as targeted subspace activation patching to establish causal links between $ar{H}$ and model predictions. They validate this on TabPFN (trained on tabular SCMs) and custom VA-PFN/DVA-PFN models (trained on spectral priors) using synthetic 1D sinusoids, 5D RBF/Matérn Gaussian Processes, and real-world time-series. Finally, they design a "Filter Bank Decoder"—a lightweight read-out head utilizing Multi-Query Attention (MQA) pooling—to explicitly decode spectral mixture parameters from the frozen representations.

### Scope conditions
The probing and causal patching experiments are largely evaluated on synthetic stationary processes (sinusoids, RBF/Matérn GPs) and two real-world time-series datasets converted to tabular regression via lag embeddings. The decoder extracts specifically stationary kernels via Bochner’s theorem (spectral mixtures), and is evaluated on standard 1D and low-dimensional (5D, 10D) kernel-cookbook benchmarks.

### Epistemic status of each central claim
1. MEASURED (production)
2. MEASURED (production)
3. MEASURED (production)
4. MEASURED (production)
5. MEASURED (toy)

### Measured vs assumed
The authors measure the correlation between the input frequency and the latent representations ($\rho_\Delta$, $|r|_{\text{PC0}}$), measure the $R^2$ of linear probes extracting these frequencies, and measure the Causal Effect (CE) when intervening on the latent state. The performance of the extracted kernel is measured via MSE against oracle GPs and baselines. They assume that recovering a spectral mixture kernel represents successful extraction of the PFN's internal Bayesian prior.

### Interventions on inputs
Yes. The core of the paper relies on internal structural interventions (activation patching) where the latent state $ar{H}$ for an input signal $A$ is swapped with the latent state of input signal $B$, observing the resulting causal effect on the output.

### Sensitivity and derivative analysis
The authors perform targeted subspace patching (a dose-response curve) across dimensions $k$ ranging from 1 to the full latent dimension $d$. They show that swapping only the top $k$ spectral principal components is significantly more causally effective than swapping non-spectral or random components (Figure 1). They also test how the spectral representation evolves across transformer layers, observing a characteristic "rise-plateau-decline" trajectory (Section 3.3).

### Relationship between prediction and observed targets
`[NOT FOUND]`

### Datasets, models, scale
- **Models:** TabPFN (Version 2.5), VA-PFN, DVA-PFN (Decoupled-Value Attention).
- **Datasets:** Synthetic 1D sinusoids, 5D RBF and Matérn GPs, standard kernel-cookbook datasets, and real-world time-series (Airline Passengers, Milk Production).

### Limitations stated by the authors
The probing experiments are restricted to stationary kernels representable as spectral mixtures. The causal subspace analysis has not yet been extended to settings involving simultaneous noise, distributional shifts, and active dimensionality. The proposed decoder is a diagnostic read-out, and the consistent MSE gap indicates that some predictive information escapes the spectral-mixture parameterization. They have not verified exactly *how* the attention constructs the spectral representation (path patching remains future work).

### Open questions named by the authors
Verifying the hypothesis that the MLP sub-layers apply a learned rectifying nonlinearity (analogous to a periodogram's square-and-average step) via path patching remains future work.

### Notes outside the schema
The paper strongly contrasts with the prevailing view that PFNs act as opaque end-to-end function approximators. By successfully extracting a functional, portable kernel matrix from the frozen activations, it provides compelling evidence that the network truly learns to perform amortized Bayesian inference with structured internal priors. 

### Importance rating
4. This is a very strong paper that successfully translates LLM mechanistic interpretability techniques (probing, activation patching, PCA dose-response) to continuous tabular foundation models, proving that these models literally learn internal kernel matrices.

**DEEP PASS RECOMMENDED: yes**


## What Has TabPFN Learned Comparing In-Context Tabular Predictions to Gaussian Process Regressors

### Metadata
- **Title as printed in the document:** On the Uncertainty Quantification Ability of Tabular Foundation Models
- **Authors:** Tyler R. Johnson, Kian Ben-Jacob, Nima Negarandeh, Oriol Vendrell-Gallart, Ramin Bostanabad
- **Venue / year:** `[NOT FOUND]` (Only author affiliations given: University of California, Irvine. Pre-print style).
- **arXiv or other ID, transcribed exactly as it appears:** `[NOT FOUND]`
- **Metadata anomalies:** The title in the document ("On the Uncertainty Quantification Ability of Tabular Foundation Models") differs from the filename ("What Has TabPFN Learned Comparing In-Context Tabular Predictions to Gaussian Process Regressors.md").

### One-paragraph summary
This paper empirically compares the Uncertainty Quantification (UQ) and predictive accuracy of the Tabular Prior-Data Fitted Network (TabPFN v2.5) against classical Gaussian Processes (GPs) across six analytical regression benchmarks and two real-world datasets. The authors fix the GP settings to the defaults of a standard package (GP+) and evaluate both methods without per-task fine-tuning across different dataset sizes ($10D_x$ vs $40D_x$), noise levels, and input dimensionalities ($D_x \in [4, 40]$). They find that while TabPFN provides a highly competitive, scalable, and fast alternative (especially when datasets are large or complex), GPs generally yield superior predictive accuracy and UQ (measured via Relative RMSE and Negative Interval Score) in data-scarce settings or when the underlying GP kernel is a good prior for the task (e.g., Griewank function). TabPFN often exhibits higher run-to-run variability in its predictive mean at very low sample sizes but manages to maintain well-calibrated predictive intervals despite this. 

### Object of study
Uncertainty Quantification (UQ) capabilities of Tabular Foundation Models (specifically TabPFN v2.5) compared to Gaussian Processes in regression tasks.

### Central claims
1. `[MEASURED (toy)]` GPs built with reasonable defaults generally provide superior predictive accuracy and UQ in data-scarce settings (e.g., $N=10D_x$) compared to TabPFN v2.5. (Section 4.1)
2. `[MEASURED (toy)]` TabPFN v2.5 exhibits considerably higher variability in RRMSE than GPs when context data is very small, but paradoxically maintains well-calibrated predictive intervals (NIS) even when its mean predictions are poor. (Section 4.1)
3. `[MEASURED (toy)]` When the GP kernel is a good prior for the underlying function (e.g., the Griewank benchmark), GPs substantially outperform TabPFN in both sample efficiency and UQ. (Section 4.1)
4. `[MEASURED (toy)]` Modifying the GP kernel (e.g., using a Power Exponential kernel instead of a Gaussian one) can dramatically improve GP performance on specific tasks (Dixon-Price, Rosenbrock) where TabPFN otherwise remains competitive or superior against default GPs. (Section 4.2)
5. `[MEASURED (toy)]` The end-to-end inference cost of TabPFN on a GPU is roughly constant ($pprox 1.4$ s) across tested dataset sizes, whereas exact GP training time on a CPU scales poorly as $N$ and $D_x$ grow (taking over 800 seconds in the largest scenarios). (Section 4.3)
6. `[MEASURED (production)]` On real-world datasets (Elevators, Pumadyn), the qualitative trend holds: TabPFN outperforms default GPs on Elevators, but GPs outperform TabPFN on Pumadyn, confirming that performance strongly depends on how well the implicit vs explicit prior matches the specific function class. (Section 4.4)

### Key equations
Equation (1):
$$ O(N_q N_c^2) $$
- End-to-end computational cost scaling of TabPFN in-context inference without caching.
- $N_c$: number of labeled context (training) rows
- $N_q$: number of query rows

Equation (4):
$$ RRMSE = \frac{1}{c} \sqrt{\frac{1}{N_{test}} \sum_{i=1}^{N_{test}} (y_i - \hat{y}_i)^2} $$
- Relative Root Mean Squared Error
- $c$: problem-dependent scalar (response standard deviation in test data)
- $y_i, \hat{y}_i$: true and predicted values

### Notation notes
- $D_x$: Input dimensionality.
- $N$: Training dataset size (scaled as multiples of $D_x$).
- NIS: Negative Interval Score, used to evaluate the quality of the predictive distribution (UQ).

### Method / instrument
The authors perform a controlled empirical comparison. They simulate datasets from six known analytical functions (Wing Weight, Buckling, Ackley, Dixon-Price, Griewank, Rosenbrock) and sample two real-world datasets (Elevators, Pumadyn). They systematically vary training data size ($N=10D_x$ or $40D_x$) and noise variance. They evaluate TabPFN v2.5 (running on GPU) and standard GPs (from the GP+ package running on CPU with default settings) using Relative Root Mean Squared Error (RRMSE) for accuracy, Negative Interval Score (NIS) for UQ calibration, and wall-clock time for computational cost. 

### Scope conditions
The study focuses on regression tasks with input dimensionalities up to $D_x=40$ and relatively small training datasets ($N \le 40D_x = 1600$ max). The GPs are evaluated specifically using a set of fixed default configurations (usually constant mean and Gaussian kernel) without per-task fine-tuning or scalable GP approximations (like sparse GPs), which limits the GP ceiling but ensures a fair "out-of-the-box" comparison against TabPFN.

### Epistemic status of each central claim
1. MEASURED (toy)
2. MEASURED (toy)
3. MEASURED (toy)
4. MEASURED (toy)
5. MEASURED (toy)
6. MEASURED (production)

### Measured vs assumed
The authors measure point prediction accuracy (RRMSE), UQ quality (NIS), and wall-clock execution time for both methods across 20 independent data splits. They assume that using a default GP package configuration provides a fair baseline representing "out-of-the-box" usage, matching the user experience of TabPFN.

### Interventions on inputs
Yes. The authors intervene on the datasets by varying the sample size ($10D_x$ vs $40D_x$) and the injected noise levels (low vs high noise multipliers) to observe how both models respond to data scarcity and noise.

### Sensitivity and derivative analysis
The authors test the sensitivity of the GP baseline to kernel choice (Gaussian vs Power Exponential) on specific tasks (Dixon-Price and Rosenbrock), demonstrating that the GP's performance can drastically improve with a better-specified prior.

### Relationship between prediction and observed targets
`[NOT FOUND]`

### Datasets, models, scale
- **Models:** TabPFN v2.5, Gaussian Processes (via the GP+ python library).
- **Datasets:** Analytical functions (Wing Weight, Buckling, Ackley, Dixon-Price, Griewank, Rosenbrock) with $D_x \in [4, 40]$. Real-world datasets: Elevators ($D_x=18$), Pumadyn ($D_x=32$).

### Limitations stated by the authors
The study intentionally excludes recent advances in scalable GPs (e.g., sparse approximations) and complex kernel constructions, meaning the GP baselines represent "vanilla" exact GPs. The comparison of timing is hardware-dependent (CPU for GP vs GPU for TabPFN). The analytic benchmarks are mostly evaluated with Gaussian or Student's t-distribution noise.

### Open questions named by the authors
The authors state that future research should pursue application-oriented comparisons, such as evaluating these models in the context of Bayesian optimization, multi-fidelity modeling, or multi-task emulation.

### Notes outside the schema
The finding that TabPFN produces "highly variable RRMSE but tight NIS" on very small datasets is an interesting characteristic of its UQ: the model seems to "know" when its mean prediction is unstable and outputs appropriately wide, calibrated intervals.

### Importance rating
3. A solid empirical benchmarking paper. It provides useful, rigorous quantitative comparisons of TabPFN's UQ capabilities against classical GPs across varying dimensionalities and noise levels, though it lacks the mechanistic depth of the earlier papers.

**DEEP PASS RECOMMENDED: no**


## TabPFN Behaves like 1-Nearest Neighbor in 2D Feature Spaces

### Metadata
- **Title as printed in the document:** What exactly has TabPFN learned to do?
- **Authors:** Calvin McCarter
- **Venue / year:** Blogposts Track at ICLR 2024 (and updated in 2025).
- **arXiv or other ID, transcribed exactly as it appears:** `[NOT FOUND]`
- **Metadata anomalies:** The title in the document ("What exactly has TabPFN learned to do?") differs from the filename ("TabPFN Behaves like 1-Nearest Neighbor in 2D Feature Spaces.md"). The document is a blog post submitted to a conference track, not a traditional paper.

### One-paragraph summary
This blog post explores the inductive biases of the TabPFN (v1 and v2) tabular foundation model by treating it as a black-box function approximator and visualizing its behavior on simple, out-of-distribution toy datasets. On 1D binary classification tasks, TabPFN generates sigmoid-like probability curves but fails to extrapolate periodic patterns and exhibits bizarre asymmetric responses to duplicated samples. On 2D multiclass tasks (with one sample per class), the base model partitions space nonsensically, but with ensembling, its predictions closely match a Voronoi diagram (1-nearest-neighbor behavior). The author also evaluates TabPFN on high-dimensional out-of-domain tasks (gene expression cancer classification and computer vision via flattened MNIST/CIFAR-10), where it performs surprisingly well against XGBoost and SVMs, but sometimes lags behind simple logistic regression. The post argues that while TabPFN is not a magic bullet, it has learned a robust "world model" of small-n statistical learning that generalizes surprisingly well, but users should interactively test these models on toy problems to understand their specific inductive biases.

### Object of study
The learned inductive biases of the TabPFN (v1 and v2) models.

### Central claims
1. `[MEASURED (toy)]` On simple 1D binary classification, TabPFN generates smooth sigmoid-like boundaries resembling inverse-square-root Euclidean distance attention, but it fails to capture or extrapolate periodic patterns. (Section 2)
2. `[MEASURED (toy)]` TabPFN responds asymmetrically to sample duplication (repeating one class shifts the boundary strangely in the opposite direction), suggesting suboptimal artifacts in its training or architecture. (Section 2)
3. `[MEASURED (toy)]` In 2D multiclass classification with single samples per class, TabPFN without ensembling produces nonsensical decision boundaries, but with 32 ensembles, it closely approximates a 1-Nearest-Neighbor classifier (Voronoi partitioning). (Section 3)
4. `[MEASURED (production)]` On high-dimensional, out-of-domain tasks like gene expression classification (BladderBatch) and flattened image classification (MNIST/CIFAR-10), TabPFN performs competitively against XGBoost and SVMs without hyperparameter tuning, though it sometimes underperforms Logistic Regression. (Sections 4 & 5)
5. `[MEASURED (toy)]` The newer TabPFN-v2 model exhibits "bumpy" decision boundaries on 1D tasks due to its hash-based fingerprint features, but impressively approximates the parity function with very few samples (e.g., 99% accuracy on 10D parity using only 12% of the truth table). (Appendix A)

### Key equations
`[NOT FOUND]`

### Notation notes
- $f_{D, \theta}$: The black-box function approximation generated by TabPFN with parameters $\theta$ and context data $D$.

### Method / instrument
The author probes the trained TabPFN (v1 and v2) models as black-box function generators. For 1D and 2D toy problems, the author plots the model's predicted probability surface (decision boundaries) across the input space under various manipulations (e.g., repeating features, duplicating samples, periodic arrangements). For empirical evaluation, the author compares TabPFN against standard baselines (Logistic Regression, SVC, XGBoost) using default hyperparameters on high-dimensional out-of-domain datasets (BladderBatch, MNIST, CIFAR-10), as well as on exhaustive truth tables for the Boolean parity function (Appendix A).

### Scope conditions
The toy experiments are limited to very low dimensions (1D and 2D) and tiny sample sizes to visualize decision boundaries. The empirical evaluations deliberately avoid hyperparameter search to simulate extreme small-n regimes where cross-validation is risky, which might disadvantage baselines like XGBoost.

### Epistemic status of each central claim
1. MEASURED (toy)
2. MEASURED (toy)
3. MEASURED (toy)
4. MEASURED (production)
5. MEASURED (toy)

### Measured vs assumed
The author assumes that examining a model's behavior on simple 1D/2D toy tasks provides valid intuition about its general inductive biases. The author directly measures the predicted probability surfaces of TabPFN and visually compares them to geometric ideals (Voronoi diagrams, periodic functions). Predictive accuracy and F1-scores are directly measured on the high-dimensional datasets.

### Interventions on inputs
Yes. The author systematically intervenes on the input context data by duplicating specific features, duplicating specific samples, and arranging samples in grid-spaced versus randomly-spaced layouts to observe how the model's decision boundary shifts in response.

### Sensitivity and derivative analysis
The author tests the sensitivity of the model to the number of ensembles (1 vs 32), showing that ensembling smooths out 1D boundaries and is absolutely critical for sensible 2D behavior. The author also tests sensitivity to the number of duplicated samples (1, 4, 16, 64) and the number of training samples provided for the CV/parity tasks.

### Relationship between prediction and observed targets
`[NOT FOUND]`

### Datasets, models, scale
- **Models:** TabPFN-v1, TabPFN-v2, Logistic Regression, SVC, XGBoost.
- **Datasets:** Synthetic 1D/2D toy points, BladderBatch (57 samples, 22k features), MNIST, CIFAR-10, Boolean Parity truth tables.

### Limitations stated by the authors
The author notes that their analysis is inherently specific to the exact checkpoint of TabPFN released by the authors (`prior_diff_real_checkpoint_n_0_epoch_42.cpkt`), rather than an evaluation of the general PFN meta-learning methodology. TabPFN-v2 could not be run on the BladderBatch and CIFAR-10 tasks due to out-of-memory errors caused by its feature embeddings.

### Open questions named by the authors
The author questions whether the bizarre behavior observed (e.g., the asymmetric response to duplicating the "red" class) was actually optimal for the synthetic pretraining data, or if it is an artifact of the architecture. The author also suggests that future work could meta-learn PFNs with robust hinge-type losses for domains where linear models currently excel.

### Notes outside the schema
The finding that TabPFN-v2 can learn the 10D parity function with 99% accuracy using only 127 examples (12% of the truth table) is remarkable, as parity is notoriously difficult for standard neural networks and tree ensembles to learn without exhaustive data.

### Importance rating
4. This blog post provides excellent visual and intuitive insights into how TabPFN actually interpolates space, exposing both its surprisingly elegant 1-NN behavior and its bizarre failure modes. 

**DEEP PASS RECOMMENDED: no**


## Where Computation Lives Functional Head Specialization in Tabular Foundation Models

### Metadata
- **Title as printed in the document:** Where Computation Lives Inside TabPFN: Causal Localisation of Attention Head Function
- **Authors:** Atharva Gupta, Dhruv Kumar, Murari Mandal, Saurabh Deshpande
- **Venue / year:** Proceedings of the 2nd ICML Workshop on Foundation Models for Structured Data, Seoul, South Korea. 2026.
- **arXiv or other ID, transcribed exactly as it appears:** `[NOT FOUND]`
- **Metadata anomalies:** The title in the document ("Where Computation Lives Inside TabPFN: Causal Localisation of Attention Head Function") differs slightly from the filename ("Where Computation Lives Functional Head Specialization in Tabular Foundation Models.md").

### One-paragraph summary
This paper presents the first causal mechanistic analysis of the TabPFN-2.5 tabular foundation model, aiming to localize specific computations to individual attention heads and layers. Through activation patching, ablation, and attention entropy analysis on two synthetic regression datasets (Multiplication and Pairwise-50), the authors identify clear functional specialization within the feature-wise self-attention module. They find that one specific head (Head 2) exerts a dominant causal effect on predictions (up to 5x larger than other heads), though the depth at which this computation peaks shifts depending on task complexity (Layer 0 for Multiplication, Layer 16 for Pairwise-50). The remaining two heads (Heads 0 and 1) act symmetrically at late layers on simpler tasks. The authors also demonstrate that TabPFN is highly resistant to inference-time contrastive activation steering because, unlike LLMs, its relational in-context learning (ICL) mechanism does not encode static, extractable task vectors, but rather relies entirely on context-dependent attention.

### Object of study
The internal mechanisms of TabPFN-2.5, specifically the causal role of individual attention heads and layers within the feature-wise self-attention module.

### Central claims
1. `[MEASURED (toy)]` TabPFN-2.5's feature-wise self-attention exhibits strong functional specialization across heads: Head 2 acts as a dominant computational head (its ablation effect is 2-5x larger than others), while Heads 0 and 1 generally operate symmetrically in late layers. (Section 2.4)
2. `[MEASURED (toy)]` The depth (layer) at which the dominant head (Head 2) performs its critical computation shifts based on task complexity (e.g., Layer 0 for a 3-feature Multiplication task, Layer 16 for a 50-feature Pairwise task), while its peak ablation magnitude remains consistent. (Section 2.4)
3. `[MEASURED (toy)]` Attentional selectivity (low entropy) is necessary but not sufficient for causal necessity; for example, Head 0 exhibits the same low entropy as Head 2 at Layer 0 but has near-zero causal effect when ablated. (Section 2.4 / Appendix E)
4. `[MEASURED (toy)]` At the token level, when evaluating a single sample, causal information funnels entirely through the test-label token position, but when evaluating large batches, the causal signal is highly distributed across samples. (Appendix F)
5. `[MEASURED (toy)]` Contrastive activation steering fails to generalize across samples in TabPFN-2.5; injecting a target direction vector produces near-zero MSE improvement on held-out test splits. (Section 3 / Appendix G)
6. `[ASSERTED]` TabPFN’s resistance to activation steering stems from its purely relational ICL architecture, which encodes task relationships entirely through context-dependent attention rather than forming stable, extractable "function vectors" as LLMs do. (Section 3 / Appendix G)

### Key equations
Equation (1) (MHA formulation):
$$ \hat{V}^{(h)}_\ell = \text{Softmax}\left( \frac{X W_Q^{(h)} (X W_K^{(h)})^T}{\sqrt{d_h}} \right) X W_V^{(h)} $$
- $\hat{V}^{(h)}_\ell$: Pre-projection output of attention head $h$ at layer $\ell$
- $X$: Input representation
- $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}$: Per-head projection matrices

Equation (2) (Patching setup):
$$ x^{\text{patched}} = x^{\text{corr}} [S \leftarrow x^{\text{clean}}] $$
- $x^{\text{patched}}$: Patched forward pass state
- $S$: Intervention site
- $x^{\text{clean}}$: Clean activation value

### Notation notes
- $h^*$: Target attention head for intervention.
- $\ell$: Transformer layer index.
- $\sigma$: Standard deviation, used as a unit for measuring ablation and patching effects.
- $	ilde{H}$: Normalized attention entropy.

### Method / instrument
The authors apply mechanistic interpretability techniques to TabPFN-2.5. They use causal activation patching (replacing corrupted activations with clean ones) and ablation (zeroing out activations) at multiple granularities: full residual stream, feature-block level, and attention-head level. They measure the resulting causal effect on the model's regression output. They also compute normalized Shannon entropy over the attention weights to measure selectivity. Finally, they attempt contrastive activation steering by injecting a mean-difference vector (multiplicative vs additive context) into the residual stream to test steerability.

### Scope conditions
The experiments are strictly limited to two synthetic regression datasets (a 3D Multiplication task and a 50D Pairwise multiplication task). The analysis primarily targets the feature-wise self-attention module (self-attn-between-features) of TabPFN-2.5 Regressor.

### Epistemic status of each central claim
1. MEASURED (toy)
2. MEASURED (toy)
3. MEASURED (toy)
4. MEASURED (toy)
5. MEASURED (toy)
6. ASSERTED

### Measured vs assumed
The authors measure patching recovery ratios, ablation magnitudes in standard deviations ($\sigma$), attention entropy, and MSE improvement for steering. They assume that the failure of contrastive steering on the Multiplication task reflects a general architectural property of pure relational ICL (the absence of stable function vectors), though they acknowledge this is a preliminary finding on a single task.

### Interventions on inputs
Yes. The entire methodology relies on internal interventions (patching/ablating activations) during the forward pass. They also intervene on the input datasets by applying specific corruptions (e.g., mean shift or Gaussian replace) to separate the clean and corrupted states.

### Sensitivity and derivative analysis
The authors test sensitivity to the type of data corruption (mean shift vs Gaussian replace) and note that Gaussian replace fails for the Pairwise-50 dataset due to sign cancellation in the quadratic target, necessitating the use of mean shift to obtain a clean directional signal. They also analyze how the causal effect changes across all 18 layers.

### Relationship between prediction and observed targets
`[NOT FOUND]`

### Datasets, models, scale
- **Models:** TabPFN-2.5 Regressor (18 transformer layers, 3 attention heads, model dimension 192).
- **Datasets:** Synthetic Multiplication Dataset ($d=3$, $y = a\cdot b + c$) and Synthetic Pairwise-50 Dataset ($d=50$, $y = (\sum x_i)^2$). Context size varies ($n=64$ to $512$).

### Limitations stated by the authors
The scope is limited to two synthetic datasets, which is insufficient to establish the observed head classes (or the shifting depth of Head 2's peak) as general properties of TabPFN-2.5. The steering failure is also a preliminary finding on a single task using a single direction estimator (mean-diff). Direct attention weight visualization was not performed.

### Open questions named by the authors
Does the shifting depth of the dominant head's computation reflect a genuine task-complexity-dependent depth shift, or is it an artifact of these particular tasks? Does the null result for activation steering hold for other tasks and more expressive direction estimators (e.g., linear probes or PCA on contrastive pairs)?

### Notes outside the schema
The finding that TabPFN lacks the steerable "function vectors" found in LLMs is theoretically interesting, highlighting a divergence in how continuous tabular FMs and discrete language models implement in-context learning. LLMs seem to compress the task into a static vector in the context window, while TabPFN computes the relationship fresh at every layer via attention.

### Importance rating
4. This is an excellent, rigorous mechanistic interpretability paper that localizes specific computations within TabPFN's feature-attention heads and provides a compelling architectural explanation for its resistance to activation steering. 

**DEEP PASS RECOMMENDED: yes**



## A Mechanistic Study of Tabular Foundation Models

### Metadata
- **Title as printed in the document:** A Mechanistic Study of Tabular Foundation Models
- **Authors:** Marin Biloš, James T. Wilson, Anderson Schneider, Yuriy Nevmyvaka
- **Venue / year:** Morgan Stanley, 2024/2025
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** [NOT FOUND]

### One-paragraph summary
The paper provides a mechanistic analysis of contemporary tabular foundation models (TabPFNv2, TabICLv2, Mitra), focusing on how they process in-context examples. The authors find that these models learn distinct algorithms and form predictions using different readout mechanisms, such as an attention-weighted label vote for TabPFNv2 and a nearest-class prototype for TabICLv2. They also investigate permutation invariances, discovering that redundant positional components can be safely removed, and design mechanism-grounded adversarial perturbations that expose model-specific vulnerabilities (e.g., hub and rank attacks).

### Object of study
The mechanisms, invariances, and adversarial robustness of tabular foundation models (TabPFNv2, TabICLv2, Mitra).

### Central claims
1. [PARA] Different tabular foundation models employ qualitatively distinct similarity-based readouts to form their predictions, such as TabPFNv2 using attention-weighted votes and TabICLv2 using a nearest class prototype.
2. [PARA] Model-specific invariances (e.g., column-permutation invariances) can be restored exactly by surgically editing positional components without cost to performance.
3. [PARA] Representation collapse is not a significant practical concern for these models because of architectural mitigations that sidestep the failure mode.
4. [PARA] Mechanism-grounded data perturbations (e.g., hub-poisoning, rank attacks, SVD hiding) can be engineered against these models, exposing readout-specific weaknesses while non-linear tree baselines remain robust.

### Key equations
[NOT FOUND]

### Notation notes
L$k$ denotes the $k$-th layer of a model.

### Method / instrument
Mechanistic interpretability techniques including layerwise linear probes, representation geometry analysis, activation patching, permutation and ablation tests, and targeted adversarial attacks.

### Scope conditions
The empirical analysis is based on standard tabular classification and regression benchmark suites (e.g., OpenML grids).

### Epistemic status of each central claim
1. SUPPORTED
2. SUPPORTED
3. SUPPORTED
4. SUPPORTED

### Measured vs assumed
Measurements on representation layer outputs and adversarial attack effects; assumes that linear probes on features isolate where useful representation is formed.

### Interventions on inputs
SVD hiding, rank scaling, hub poisoning, label permutation, boundary poisoning, and padding attacks.

### Sensitivity and derivative analysis
Activation patching and per-block knockout analysis to study resilience of internal representations.

### Relationship between prediction and observed targets
[INFER] The models map tabular data onto target classes/values via attention-weighted voting or prototype distance reading.

### Datasets, models, scale
OpenML benchmarks (49 classification, 10 regression); TabPFNv2, TabICLv2, Mitra.

### Limitations stated by the authors
[NOT FOUND]

### Open questions named by the authors
[NOT FOUND]

### Notes outside the schema
[NONE]

### Importance rating
4

DEEP PASS RECOMMENDED: no
### Uncertainty flags
[NOT FOUND]
