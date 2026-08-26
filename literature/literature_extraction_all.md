# Camp directory name
a

# Number of .md files found in it, and the full list of filenames
12 files:
- How Transformers Learn In-Context Beyond Gradient Descent.md
- Linear Transformers Are Versatile In-Context Learners.md
- Transformers Can Learn Posterior Predictive Distributions In-Context.md
- Transformers Implement Functional Gradient Descent.md
- Transformers Learn Higher-Order Optimization Algorithms for In-Context Learning.md
- Transformers as Statisticians Provable In-Context Learning with Hidden Features.md
- Transformers learn in-context learning by gradient descent.md
- Transformers learn to implement preconditioned gradient descent for in-context learning.md
- What learning algorithm is in-context learning Investigations with linear models.md
- Why Can GPT Learn In-Context Language Models Implicitly Perform Gradient Descent.md

# Date of extraction
2026-08-08


## How Transformers Learn In-Context Beyond Gradient Descent

### Metadata
- **Title as printed in the document:** How Well Can Transformers Emulate In-context Newton’s Method?
- **Authors:** Angeliki Giannou, Liu Yang, Tianhao Wang, Dimitris Papailiopoulos, Jason D. Lee
- **Venue / year:** March 6, 2024
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** Title in the document ("How Well Can Transformers Emulate In-context Newton’s Method?") differs from the filename ("How Transformers Learn In-Context Beyond Gradient Descent.md").

### One-paragraph summary
The paper theoretically demonstrates that linear attention Transformers with ReLU layers can approximate higher-order optimization algorithms, specifically Newton's method, for in-context learning. By constructively designing the weights, the authors show that Transformers can perform matrix inversion via Newton's iteration with logarithmic depth relative to the target error, enabling them to solve linear regression and optimize the regularized logistic loss for logistic regression.

### Object of study
A mathematical construction (weights hand-specified, never trained) of linear attention Transformers with ReLU layers. Additionally, a small model trained from scratch on a synthetic ICL task (linear and logistic regression) to compare against the construction.

### Central claims
1. [PARA] Transformers can efficiently perform matrix inversion via Newton's iteration using constructed weights. (Theorem 1.1)
2. [PARA] Transformers can compute the least-square solution for linear regression by inverting the covariance matrix using Newton's iteration. (Theorem 1.1, Theorem 4.2)
3. [PARA] Transformers can perform Newton's method to efficiently optimize the regularized logistic loss for logistic regression. (Theorem 1.1, Theorem 5.1)
4. [PARA] Approximating Newton's method on the regularized logistic loss requires only $\log \log(1/\epsilon)$ many layers and $1/\epsilon^8$ width to achieve $\epsilon$ error. (Theorem 1.1)

### Key equations
(3.2) $\hat{\mathbf{w}} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{y}$
Here $\hat{\mathbf{w}}$ is the least-square solution, $\mathbf{A}$ is the input data matrix (covariates), and $\mathbf{y}$ is the target vector.

(3.3) $\mathbf{X}_{t+1} = \mathbf{X}_t(2\mathbf{I}_d - \mathbf{A}\mathbf{X}_t)$
Here $\mathbf{A}$ is the matrix to invert, and $\mathbf{X}_t$ is the approximation of the inverse at step $t$.

(3.5) $f(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-y_i \mathbf{w}^\top \mathbf{a}_i)) + \frac{\mu}{2} \|\mathbf{w}\|_2^2$
Here $f(\mathbf{w})$ is the regularized logistic loss, $\mathbf{a}_i$ is the covariate vector, $y_i \in \{-1, 1\}$ is the label, $\mathbf{w}$ is the parameter vector, and $\mu$ is the regularization parameter.

(3.7) $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta(\mathbf{x}_t)(\nabla^2f(\mathbf{x}_t))^{-1}\nabla f(\mathbf{x}_t)$
Here $\mathbf{x}_t$ is the approximation of the parameter vector at step $t$, $f$ is the objective function, $\nabla f$ is the gradient, $\nabla^2 f$ is the Hessian, and $\eta(\mathbf{x}_t)$ is the step-size.

### Notation notes
[NOT FOUND]

### Method / instrument
Mathematical proofs providing concrete constructions of Transformers (hand-specifying weights for linear attention and feed-forward ReLU layers) to emulate steps of Newton's method. They also perform training runs of small linear self-attention models (with and without LayerNorm) on synthetic linear and logistic regression data, comparing the models' in-distribution and out-of-distribution loss and outputs against different orders of Newton's iteration.

### Scope conditions
The theoretical constructions apply to linear attention Transformers (softmax removed) with position-wise feed-forward layers using ReLU activations. The logistic regression results assume the data features have bounded norm ($\|\mathbf{a}_i\|_2 \le 1$) and rely on the strong convexity provided by the $L_2$ regularizer ($\mu > 0$) ensuring the loss is self-concordant.

### Epistemic status of each central claim
1. PROVED
2. PROVED
3. PROVED
4. PROVED

### Measured vs assumed
The theoretical capabilities of the Transformer architecture to implement Newton's method are proved via construction. It is assumed as a framework that trained models might implement similar algorithms. The empirical similarity between trained model outputs and Newton's iteration steps is measured (toy).

### Interventions on inputs
The paper perturbs inputs during behavioral benchmarking by keeping the context batch fixed, selecting a specific test sample $\mathbf{x}_{\text{test}}$, and varying its first coordinate across $[-a, a]$ for different values of $a$ (both in-distribution and out-of-distribution) to evaluate the model's output compared to Newton's iteration.

### Sensitivity and derivative analysis
The paper explicitly bounds the error induced by each approximation step of the Hessian and gradient in their construction. They compute the derivative of the approximation function for the step size $g(z) = 1/(1+\sqrt{z})$ as $g'(z) = -1/(2(1+\sqrt{z})^2\sqrt{z})$ to show that an $\epsilon$ change in $z$ yields a $\sqrt{\epsilon}$ change in $g(z)$, establishing the width requirements.

### Relationship between prediction and observed targets
[INFER] The output prediction for linear regression is $\hat{y} = \mathbf{a}_{\text{test}}^\top\mathbf{X}_T\mathbf{A}^\top\mathbf{y}$.

### Datasets, models, scale
Synthetic Gaussian data. Linear self-attention (LSA) models with 1 to 6 layers, embedding dimension 64, 4 attention heads for linear regression (data dimension 10, 50 context samples). For logistic regression, embedding dimension 32, 4 heads, data dimension 5, 26 context samples.

### Limitations stated by the authors
"the optimal training method for linear Transformers beyond 5 - 6 layers remains unidentified."

### Open questions named by the authors
[NOT FOUND]

### Notes outside the schema
The authors note that trained Transformers seem to outperform Newton's method for the initial layers/steps. They attribute this to Transformers having slightly higher order capacity than standard Newton's iteration (e.g. producing up to the 9th power of the matrix in two layers vs the 7th power for second-order Newton's). They also observe that models trained with LayerNorm perform better, but in out-of-distribution regimes, models with 5 layers seem to perform better than those with 7, 10, or 12 layers.

### Importance rating
4. The paper provides a highly detailed theoretical construction of how a transformer can implement a complex higher-order optimization algorithm (Newton's method) for a non-linear task (logistic regression), which pushes the boundary of what is known about in-context learning mechanisms.

DEEP PASS RECOMMENDED: no
### Uncertainty flags
[NOT FOUND]

## Linear Transformers Are Versatile In-Context Learners

### Metadata
- **Title as printed in the document:** Linear Transformers are Versatile In-Context Learners
- **Authors:** Max Vladymyrov, Johannes von Oswald, Mark Sandler, Rong Ge
- **Venue / year:** 38th Conference on Neural Information Processing Systems (NeurIPS 2024).
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** The printed title case differs slightly from the filename ("Linear Transformers are Versatile In-Context Learners" vs "Linear Transformers Are Versatile In-Context Learners.md").

### One-paragraph summary
The paper proves that each layer of a linear transformer maintains a weight vector for an implicit linear regression problem and performs a complex variant of preconditioned gradient descent. By training linear transformers on a mixed noise variance linear regression problem, the authors show that these models discover intricate optimization algorithms involving momentum and adaptive step-size rescaling based on noise levels, outperforming or matching exact ridge regression baselines.

### Object of study
A single-head, decoder-only linear transformer (excluding MLPs and LayerNorm) trained from scratch on synthetic noisy linear regression tasks, both with fixed noise and mixed noise variance. Also, mathematical constructions of linear transformer parameter updates.

### Central claims
1. [PARA] Each layer of every linear transformer maintains a weight vector for an underlying linear regression problem. (Theorem 4.1)
2. [PARA] The algorithm executed by linear transformers can be interpreted as a variant of preconditioned gradient descent with momentum-like behavior. (Lemma 4.4)
3. [PARA] GD++ (a model trained on fixed noise) performs a second-order optimization algorithm for the least squares problem. (Theorem 5.1)
4. [PARA] Linear transformers trained on mixed noise variance problems discover optimization algorithms that dynamically adjust step sizes and scale based on noise levels. (Section 5.2, 5.3)

### Key equations
(6) $\mathcal{L}_{fixed}(P, Q) = \mathbb{E}_	au \mathbb{E}_{e_1, \dots, e_n, e_{n+1}} [ (y_t - y_	heta(\{e_1, \dots, e_n\}, e_{n+1}))^2 ]$
Here $\mathcal{L}_{fixed}$ is the in-context loss for fixed noise variance, $	au$ is the input sequence, $e_i$ are the data points, $y_t$ is the test label, and $y_	heta$ is the transformer's prediction.

(7) $\mathcal{L}_{mixed}(P, Q) = \mathbb{E}_{\sigma_	au \sim p(\sigma_	au)} \mathbb{E}_	au \mathbb{E}_{e_1, \dots, e_n, e_{n+1}} [ (y_t - y_	heta(\{e_1, \dots, e_n\}, e_{n+1}))^2 ]$
Here $\mathcal{L}_{mixed}$ is the loss for mixed noise variance, $\sigma_	au$ is the noise level sampled from $p(\sigma_	au)$.

(10) $P = 	ext{diag}(\omega_{xx}, \dots, \omega_{xx}, \omega_{yx}), Q = 	ext{diag}(\omega_{xx}, \dots, \omega_{xx}, \omega_{xy})$
Here $P$ and $Q$ are reparameterized diagonal matrices of the linear attention head.

(12) $y^l_i = \langle w^l, x_i 
angle$
Here $y^l_i$ is the output of the linear transformer at layer $l$ for the $i$-th token, $w^l$ is the implicit weight vector, and $x_i$ is the input feature.

### Notation notes
$GD^{++}$: refers to a specific variant of a diagonal linear transformer trained on a fixed noise variance problem.

### Method / instrument
Mathematical proofs reverse-engineering the parameter updates of linear transformers (Theorem 4.1, Lemma 4.3). Training single-head linear transformers (with full or diagonal matrices) on synthetic noisy linear regression datasets with uniform and categorical noise variances. Evaluating them by comparing their predictions against exact closed-form Ridge Regression baselines using an "adjusted evaluation loss".

### Scope conditions
Decoder-only linear transformers without MLPs and LayerNorm. The theoretical results (like Theorem 4.2) assume sequences where data points $x_i \sim \mathcal{N}(0, I)$ and sequence length $n \gg d$. Theorem 5.2 assumes $n 	o \infty$.

### Epistemic status of each central claim
1. PROVED
2. ASSERTED
3. PROVED
4. MEASURED (toy)

### Measured vs assumed
The fact that each layer maintains an implicit linear model is theoretically proven and assumed as the framework. The behavior of transformers learning complex optimization strategies (like adaptive step sizes and early stopping) is empirically measured on models they train.

### Interventions on inputs
[NOT FOUND]

### Sensitivity and derivative analysis
[NOT FOUND]

### Relationship between prediction and observed targets
[PARA] The model's prediction is read from the negative of the last coordinate of the final query token: $y_	heta = -y_{n+1}^L$.

### Datasets, models, scale
Synthetic mixed linear regression data. Single-head, decoder-only linear transformers with 1 to 7 layers. Context length $N=20$, dimension $D=10$. Trained with Adam optimizer for 200,000 iterations, batch size 2,048. Tested with 100,000 novel sequences.

### Limitations stated by the authors
"focus on simplified linear models, analysis of primarily diagonal attention matrices, and the need for further exploration into the optimality of discovered algorithms, generalization to complex function classes, scalability with larger datasets, and applicability to more complex transformer architectures."

### Open questions named by the authors
[NOT FOUND]

### Notes outside the schema
The paper finds that even restricted models with only diagonal parameter matrices (DIAG) match the performance of full-matrix models (FULL) on mixed noise variance tasks. However, under categorical noise, DIAG extrapolated better for unseen variances, whereas FULL performed worse out-of-distribution despite lower in-distribution error.

### Importance rating
4. The paper makes a solid mathematical and empirical contribution by proving linear transformers maintain implicit regression weights and showing they discover sophisticated, noise-adaptive optimization algorithms.

DEEP PASS RECOMMENDED: no
### Uncertainty flags
[NOT FOUND]

## Transformers Can Learn Posterior Predictive Distributions In-Context

### Metadata
- **Title as printed in the document:** Transformers Can Learn Posterior Predictive Distributions In-Context
- **Authors:** Gyeonghun Kang, Changwoo J. Lee, Xiang Cheng
- **Venue / year:** 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026.
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** [NOT FOUND]

### One-paragraph summary
The paper theoretically and empirically investigates the algorithmic capability of transformers to learn posterior predictive distributions (PPDs) in context, specifically focusing on Gaussian process regression. The authors show by construction that a transformer can implement a gradient descent algorithm (a Richardson iteration) targeting the predictive mean and variance, followed by a shallow MLP that maps these moments to binned probabilities. They bound the approximation error as a function of attention depth and bin resolution, demonstrating that attention normalization is crucial for generalizing beyond the pretraining sample size range, while deeper attention mitigates the difficulty of larger sample sizes.

### Object of study
Prior-data fitted networks (PFNs) built with a transformer architecture, consisting of self-attention blocks and an MLP head, applied to Gaussian process regression and Bayesian linear regression for in-context learning of posterior predictive distributions.

### Central claims
1. [PARA] Transformers can implement an iterative solver (Richardson iteration) via attention blocks to compute the predictive mean and variance of a GP. (Theorem 3.1)
2. [PARA] A shallow MLP can map these computed moments to a binned distribution, effectively approximating the PPD up to errors controlled by depth and bin resolution. (Theorem 4.1)
3. [PARA] Attention normalization (Jacobi preconditioning) is essential for generalizing to sample sizes beyond the pretraining range, ensuring the solver remains stable. (Theorem 5.1)
4. [PARA] Even with preconditioning, generalizing to substantially larger sample sizes is intrinsically more difficult due to worsening condition numbers, requiring deeper attention layers to supply sufficient iteration budget. (Theorem 5.3)

### Key equations
(7) $u^{(l+1)}_j(x) = u^{(l)}_j(x) - \eta^{(l)} \sigma^2 u^{(l)}_j(x) + \eta^{(l)} \sum_{i=1}^n \kappa(x_i, x_j) (v_i - u^{(l)}_i(x))$ (Richardson iteration for kernel ridge regression)
(14) $u^{(l+1)}_j(x) = u^{(l)}_j(x) - \eta^{(l)} \frac{\sigma^2}{s_j} u^{(l)}_j(x) + \eta^{(l)} \sum_{i=1}^n \frac{\kappa(x_i, x_j)}{s_j} (v_i - u^{(l)}_i(x))$ (Jacobi preconditioned Richardson iteration)

### Notation notes
$L$: number of self-attention blocks (depth)
$C$: number of bins for discretizing the PPD
$G$: Gram matrix
$s_x$: kernel aggregate for $x$, used in preconditioning

### Method / instrument
Mathematical proofs by construction (Theorem 3.1, 4.1, 5.1, 5.3). Pretraining a transformer model (with and without normalized attention) on synthetic Gaussian process data (RBF and linear kernels) over varying context sizes. Inference on synthetic datasets evaluating mean squared error, Total Variation distance, Continuous Ranked Probability Score (CRPS), and interval coverage/width.

### Scope conditions
Gaussian process regression (with RBF and linear kernels). Transformers with a specific construction mapping self-attention to iterative solvers and an MLP head to a piecewise-constant density output.

### Epistemic status of each central claim
1. PROVED
2. PROVED
3. MEASURED
4. PROVED/MEASURED

### Measured vs assumed
The mathematical theorems (e.g. attention implementing Richardson iteration) are proved by construction, acting as a theoretical bound. The generalization capabilities (the necessity of normalized attention and increased depth for larger $n$) are both proven and then measured via simulations.

### Interventions on inputs
[PARA] Covariate shift where the evaluation inputs are drawn from a uniform distribution instead of a Gaussian distribution.

### Sensitivity and derivative analysis
Sensitivity of the findings to the RBF lengthscale (Table 2), and application to hierarchical GPs with random hyperparameters (Table 3), showing that the CRPS improves with depth.

### Relationship between prediction and observed targets
[PARA] The network outputs a set of logits which are transformed via softmax to represent probabilities over $C$ bins, inducing a piecewise-constant density that directly approximates the PPD.

### Datasets, models, scale
Synthetic datasets generated from Bayesian linear regression and RBF GP models. Transformers with $L \in \{2, 4, 8, 16, 32\}$ layers, $C \in \{16, 32, 64, 128, 256\}$ bins. $n \in [64, 512]$ context sizes. Moderate input dimensions $d \in \{2, 4, 5, 8, 16\}$. Small-scale real datasets (Sacramento home prices, Walker Lake).

### Limitations stated by the authors
"we view Theorem 5.3 as clarifying the role of depth in a simplified setting rather than claiming that its scaling law governs all practical PFNs." And focus is on moderate dimensions common in GP practice (up to 16 dimensions).

### Open questions named by the authors
Extension to fully Bayesian GP regression with multi-head attention where mixture weights are represented by an additional readout module. (Remark 3.2)

### Notes outside the schema
[NOT FOUND]

### Importance rating
4. Highly relevant theoretical work demonstrating exactly how transformers can perform in-context learning of predictive distributions for GPs.

DEEP PASS RECOMMENDED: no
### Uncertainty flags
[NOT FOUND]

## Transformers Implement Functional Gradient Descent to Learn Non-Linear Functions In Context

### Metadata
- **Title as printed in the document:** Transformers Implement Functional Gradient Descent to Learn Non-Linear Functions In Context
- **Authors:** Xiang Cheng, Yuxin Chen, Suvrit Sra
- **Venue / year:** [NOT FOUND]
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** Title in document includes "to Learn Non-Linear Functions In Context", filename is just "Transformers Implement Functional Gradient Descent.md"

### One-paragraph summary
The paper theoretically and empirically shows that Transformers with non-linear activations can learn non-linear functions in-context by implementing functional gradient descent in a Reproducing Kernel Hilbert Space (RKHS). They demonstrate that if the non-linearity of the attention module matches the underlying data distribution (modeled as a Kernel Gaussian Process), the functional gradient descent converges to the Bayes optimal predictor. They also show that multi-head Transformers with different activations can implement optimal predictors for composite kernels, and analyze the optimization landscape to show functional gradient descent is a stationary point of the in-context loss.

### Object of study
Non-linear Transformers (with ReLU, softmax, or other activations) performing in-context learning of non-linear functions (generated by Kernel Gaussian Processes).

### Central claims
1. [PARA] Transformers can implement functional gradient descent with respect to an RKHS metric if the attention non-linearity matches the kernel. (Proposition 1)
2. [PARA] When the non-linear module matches the generating kernel of the data, the functional gradient descent construction converges to the Bayes optimal predictor as the number of layers increases. (Proposition 2)
3. [PARA] Multi-head Transformers with different per-head activations can implement Bayes-optimal functional gradient descent for RKHS obtained by composing the individual kernels. (Proposition 3)
4. [PARA] The functional gradient descent construction is a stationary point of the in-context loss under a sparsity constraint on the value matrix. (Theorem 1)

### Key equations
(3.1) $f_{l+1} = f_l - r_l \nabla L(f_l)$
(3.2) $y^{(i)} - f_l(x^{(i)}) = y_l^{(i)}$

### Notation notes
$\tilde{h}$: generalized attention non-linear activation function
$K$: Kernel function
$\mathbf{H}$: Reproducing Kernel Hilbert space

### Method / instrument
Mathematical proofs of parameter constructions for functional gradient descent and stationary points of in-context loss. Empirical training of linear, ReLU, and softmax Transformers on synthetic data generated from Kernel Gaussian Processes (linear, ReLU, exponential) to verify Bayes-optimal predictions and stationary points.

### Scope conditions
Assumes specific sparsity constraints on query and key matrices for the construction to exactly match standard attention modules (Assumption 1).

### Epistemic status of each central claim
1. PROVED
2. PROVED
3. PROVED
4. PROVED

### Measured vs assumed
Theoretical bounds and stationary points are proved, while the actual convergence to these stationary points during training is measured empirically on synthetic data.

### Interventions on inputs
[NOT FOUND]

### Sensitivity and derivative analysis
Multi-head combinations (e.g. 1 head linear + 1 head exp) measured against composite data generating processes.

### Relationship between prediction and observed targets
The Transformer's output predictions at each layer are shown to match the iterations of functional gradient descent minimizing the empirical loss.

### Datasets, models, scale
Synthetic datasets generated from $K$ Gaussian Processes (linear, relu, exp). Covariates $x^{(i)} \in \mathbb{R}^5$ sampled from unit sphere. Transformers with 3 layers (up to 8 in some plots), varying context lengths up to 14.

### Limitations stated by the authors
[NOT FOUND]

### Open questions named by the authors
[NOT FOUND]

### Notes outside the schema
[NOT FOUND]

### Importance rating
4. Highly relevant theoretical work generalizing in-context gradient descent algorithms to non-linear settings.

DEEP PASS RECOMMENDED: no
### Uncertainty flags
[NOT FOUND]

## Transformers Learn Higher-Order Optimization Algorithms for In-Context Learning

### Metadata
- **Title as printed in the document:** Transformers Learn to Achieve Second-Order Convergence Rates for In-Context Linear Regression
- **Authors:** Deqing Fu, Tian-Qi Chen, Robin Jia, Vatsal Sharan
- **Venue / year:** [NOT FOUND]
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** Title in document includes "to Achieve Second-Order Convergence Rates", filename uses "Learn Higher-Order Optimization Algorithms".

### One-paragraph summary
The authors demonstrate that Transformers trained for in-context linear regression learn an algorithm exhibiting second-order convergence rates, closely matching Iterative Newton's Method, rather than first-order Gradient Descent (GD). Empirically, they show that Transformer layers match Iterative Newton iterations linearly (converging in a few layers) and GD exponentially, and the Transformer maintains high performance even on ill-conditioned data where GD struggles. Theoretically, they prove that a Transformer can implement $k$ iterations of Newton's method using $k+O(1)$ layers and $O(d)$ hidden dimension.

### Object of study
Transformer models (specifically GPT-2 architecture variants) performing in-context learning for linear regression.

### Central claims
1. [PARA] Successive Transformer layers progressively improve predictions, with intermediate layers linearly matching the iterations of Iterative Newton's Method (a second-order method) and exponentially matching GD.
2. [PARA] Transformers share the superlinear convergence rate of Iterative Newton's Method, which is exponentially faster than Gradient Descent.
3. [PARA] Transformers perform well on ill-conditioned data, maintaining their convergence rate similar to Newton's Method, whereas GD's convergence is polynomially slowed down by the condition number.
4. [PARA] Transformers can theoretically implement $k$ iterations of Newton's method with $k + O(1)$ layers and $O(d)$ hidden dimension. (Theorem 5.1)

### Key equations
(4) $\hat{\mathbf{w}}^{Newton}_{k} = \mathbf{M}_k \mathbf{X}^\top \mathbf{y}$
(5) $\mathbf{M}_k = 2\mathbf{M}_{k-1} - \mathbf{M}_{k-1} \mathbf{S} \mathbf{M}_{k-1}$ (Iterative Newton's Method updates)

### Notation notes
$\mathbf{S} = \mathbf{X}^\top \mathbf{X}$
$\mathbf{M}_k$: approximation of $\mathbf{S}^\dagger$

### Method / instrument
Empirical comparison of trained Transformers (GPT-2, 12 layers, 8 heads) against iterative optimization algorithms (Iterative Newton, GD, OGD, CG, BFGS, L-BFGS) by computing "Similarity of Errors" (cosine similarity of residuals) and "Similarity of Induced Weights" across matching steps. Theoretical proof constructing Transformer weights to implement Iterative Newton.

### Scope conditions
In-context linear regression task with $n=40$ and $d=20$.

### Epistemic status of each central claim
1. MEASURED
2. MEASURED
3. MEASURED
4. PROVED

### Measured vs assumed
The similarity of the Transformer's behavior to Iterative Newton's Method is measured across layers using synthetic data. The theoretical capability to implement Iterative Newton is proved by construction.

### Interventions on inputs
[PARA] Tested on ill-conditioned data (condition number $\kappa=100$). Also tested on noisy linear regression ($\sigma=0.1$) and 2-layer MLP (ReLU and Tanh) non-linear function classes.

### Sensitivity and derivative analysis
Ablations on hidden dimensions (shows $O(d)$ hidden size is required), varying number of attention heads (1 head vs 8 heads), and varying depth up to 24 layers.

### Relationship between prediction and observed targets
The Transformer's predictions are compared with the outputs of algorithmic iterations applied to the observed contexts to find best-matching steps.

### Datasets, models, scale
Synthetic linear regression data, $d=20$, up to 40 context examples. Transformer models based on GPT-2 with 12 layers (up to 24) and hidden dimensions around 256 (ablated down to 8).

### Limitations stated by the authors
[NOT FOUND]

### Open questions named by the authors
Understanding fully how Transformers solve the 2-layer MLP regression problem in-context and whether it achieves a different optimum compared to SGD.

### Notes outside the schema
The authors contrast Transformers with LSTMs, showing LSTMs behave more like Online Gradient Descent (OGD) and do not learn second-order methods.

### Importance rating
5. Excellent paper showing second-order dynamics inside Transformers.

DEEP PASS RECOMMENDED: no
### Uncertainty flags
[NOT FOUND]

## Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection

### Metadata
- **Title as printed in the document:** Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection
- **Authors:** Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, Song Mei
- **Venue / year:** [NOT FOUND]
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** [NOT FOUND]

### One-paragraph summary
The paper provides a comprehensive statistical theory for how transformers can perform in-context learning by implementing standard machine learning algorithms like ridge regression, Lasso, and generalized linear models. The key theoretical innovation is showing how a single transformer can adaptively perform "in-context algorithm selection"—such as choosing between different regularization strengths for ridge regression, or between regression and classification—without explicit prompting, using mechanisms like Pre-ICL testing and Post-ICL validation. They demonstrate both theoretically and empirically that transformers can simultaneously approach the Bayes-optimal performance across mixed tasks by simulating this algorithm selection process.

### Object of study
Transformers (with ReLU activation instead of softmax) performing in-context learning on various regression and classification tasks, specifically their ability to implement algorithm selection.

### Central claims
1. [PARA] Transformers can implement standard machine learning algorithms in context (least squares, ridge regression, GLMs, Lasso, GD on 2-layer NNs) with near-optimal statistical prediction power.
2. [PARA] A single transformer can perform in-context algorithm selection to adaptively choose the best base algorithm for a given sequence without explicit prompting, using either Pre-ICL testing or Post-ICL validation mechanisms.
3. [PARA] Transformers can achieve nearly Bayes-optimal in-context learning on noisy linear models with mixed noise levels by using the Post-ICL validation mechanism.
4. [PARA] Transformers can be pretrained with polynomially many training sequences to perform these in-context learning tasks.

### Key equations
(1) $\mathbf{h}'_i = \mathbf{h}_i + \frac{1}{N} \sum_{j=1}^N \sigma(\langle \mathbf{Q}_m \mathbf{h}_i, \mathbf{K}_m \mathbf{h}_j 
angle) \mathbf{V}_m \mathbf{h}_j$
(ICRidge) $\mathbf{w}^{\mathrm{ridge}}_{\lambda} := rg\min_{\mathbf{w}} \frac{1}{N} \sum_{i=1}^N (\mathbf{w}^\top \mathbf{x}_i - y_i)^2 + \lambda \|\mathbf{w}\|_2^2$

### Notation notes
$\sigma(t) = 	ext{ReLU}(t)$
$N$: number of in-context examples

### Method / instrument
Theoretical constructions of transformer weights to implement optimization steps and algorithm selection (using attention and MLP layers), combined with statistical generalization bounds. Empirical validation by training transformers on synthetic datasets (like noisy linear regression with mixed noise levels) and measuring their alignment with analytical Bayes predictors.

### Scope conditions
Theoretical constructions assume bounded features and labels, and use normalized ReLU attention instead of standard softmax attention. The algorithm selection mechanisms are demonstrated for specific mixtures (e.g. regression + classification, mixed noise levels).

### Epistemic status of each central claim
1. PROVED
2. PROVED
3. PROVED
4. PROVED

### Measured vs assumed
Theoretical capabilities and sample complexities are proved, while the actual emergence of in-context algorithm selection during training is measured empirically.

### Interventions on inputs
[NOT FOUND]

### Sensitivity and derivative analysis
Measured performance of trained transformers on separate mixed tasks (e.g. noise levels $\sigma_1$ and $\sigma_2$), showing that a single model approaches the Bayes-optimal performance of both individual task predictors without being explicitly told which task it is solving.

### Relationship between prediction and observed targets
The transformer's output is evaluated against the true test labels and compared analytically to the theoretical Bayes predictor for the task.

### Datasets, models, scale
Synthetic datasets representing linear models, GLMs, and sparse linear models. Theoretical bounds are given for arbitrary dimensions $d$ and sample sizes $N$, typically requiring $O(\log N)$ layers.

### Limitations stated by the authors
[NOT FOUND]

### Open questions named by the authors
[NOT FOUND]

### Notes outside the schema
The paper frames transformers as "statisticians" capable of running validation sets and testing statistics implicitly within their forward pass.

### Importance rating
5. A comprehensive and novel theoretical framework for ICL algorithm selection.

DEEP PASS RECOMMENDED: no
### Uncertainty flags
[NOT FOUND]

## Transformers Learn In-Context by Gradient Descent

### Metadata
- **Title as printed in the document:** Transformers Learn In-Context by Gradient Descent
- **Authors:** Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, Jo˜ao Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, Max Vladymyrov
- **Venue / year:** Proceedings of the 40th International Conference on Machine Learning, Honolulu, Hawaii, USA. PMLR 202, 2023.
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** [NOT FOUND]

### One-paragraph summary
This paper proposes that in-context learning in transformers is implemented by gradient-based optimization within the forward pass. The authors provide an explicit weight construction demonstrating that a single linear self-attention layer can implement a step of gradient descent on a regression loss. They show empirically that transformers trained on simple regression tasks converge to weights that match this construction or behave identically to gradient descent. Furthermore, they extend this to show that multi-layer transformers learn an iterative curvature correction (outperforming standard GD) and that transformers with MLPs solve non-linear regression by performing gradient descent on deep representations.

### Object of study
Transformers (specifically using linear self-attention, and sometimes MLPs) performing in-context learning of linear and non-linear regression tasks.

### Central claims
1. [PARA] A single linear self-attention layer can be explicitly constructed to perform an update identical to a single step of gradient descent on a mean squared error loss. (Proposition 1)
2. [PARA] Transformers optimized on linear regression tasks converge to the explicit gradient descent construction, effectively becoming mesa-optimizers.
3. [PARA] Multi-layer self-attention transformers surpass plain gradient descent by learning an iterative curvature correction algorithm (termed GD++).
4. [PARA] By incorporating MLPs, transformers can solve non-linear regression tasks by performing gradient descent on deep data representations. (Proposition 2)
5. [PARA] Standard attention with softmax can act as a copying mechanism that prepares tokens for gradient descent learning in subsequent layers. (Proposition 3)

### Key equations
(2) $L(W) = \frac{1}{2} \sum_{i=1}^N (W x_i - y_i)^2$
(4) $\Delta y_i = \Delta W x_i$
(5) $\min_{	heta} \mathbb{E}_{\tau} [\| \hat{y}_	heta(\{e_{\tau,1}, ..., e_{\tau,N}\}, e_{\tau, N+1}) - y_{\tau, 	ext{test}} \|^2]$

### Notation notes
$\Delta W$: weight change from one step of GD
GD++: gradient descent variant with iterative data transformation
LSA: linear self-attention

### Method / instrument
Explicit construction of attention weights to match GD mathematically. Empirical comparison of trained transformer outputs, loss trajectories, and partial derivatives against those of explicit GD algorithms (including interpolating between trained weights and the explicit construction).

### Scope conditions
Training largely focuses on noiseless linear regression tasks where the input and output data are merged into a single token initially, though they show how a copying mechanism relaxes this.

### Epistemic status of each central claim
1. PROVED
2. MEASURED
3. MEASURED
4. MEASURED
5. PROVED

### Measured vs assumed
Theoretical constructions are proved. The alignment between the learned transformer weights/predictions and the GD constructs are measured using cosine similarity and L2 distance.

### Interventions on inputs
Testing trained models on out-of-distribution tasks (e.g., larger input scales, scaled teacher weights) to show they maintain alignment with GD and fail in the same ways as GD.

### Sensitivity and derivative analysis
Comparison of the partial derivatives of the transformer predictions against the analytical sensitivities of gradient descent steps to confirm algorithmic alignment.

### Relationship between prediction and observed targets
The transformer's predicted output is shown to match the prediction that would be produced by taking one or more steps of gradient descent on the in-context targets.

### Datasets, models, scale
Synthetic linear regression tasks, $N=10$, $Nx=10$, $Ny=1$. Also non-linear sine-wave regression tasks. Transformer models include single LSA layers, 2-layer recurrent LSA, 5-layer LSA, and models with MLPs.

### Limitations stated by the authors
They note that regression with noisy data and weight regularization were not analyzed, and expect their linear SA findings to explain only part of a complex process in real large language models.

### Open questions named by the authors
[NOT FOUND]

### Notes outside the schema
The authors link their findings to the "mesa-optimization" concept and draw parallels to the "induction head" copying mechanism.

### Importance rating
5. A seminal paper providing empirical evidence for the gradient descent hypothesis in transformers.

DEEP PASS RECOMMENDED: no
### Uncertainty flags
[NOT FOUND]

## Transformers learn to implement preconditioned gradient descent for in-context learning

### Metadata
- **Title as printed in the document:** Transformers learn to implement preconditioned gradient descent for in-context learning
- **Authors:** Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, Suvrit Sra
- **Venue / year:** 37th Conference on Neural Information Processing Systems (NeurIPS 2023)
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** [NOT FOUND]

### One-paragraph summary
This paper investigates whether transformers can learn to implement gradient-based algorithms when trained on random linear regression instances. By analyzing the loss landscape of linear transformers (attention without softmax), the authors prove that the global minimum of a single-layer transformer implements one step of preconditioned gradient descent, where the preconditioner adapts to both data distribution and sample inadequacy. For multi-layer transformers, they characterize critical points of the training objective, demonstrating that under certain parameter structures, transformers learn to implement iterative adaptive gradient methods (like Newton's method or GD++).

### Object of study
Linear transformers (self-attention without softmax) performing in-context learning on random linear regression tasks.

### Central claims
1. [PARA] The global minimum of the training objective for a single-layer linear transformer implements a single step of preconditioned gradient descent.
2. [PARA] For a two-layer transformer with symmetric weights trained on isotropic data, the global minimizer corresponds to gradient descent with adaptive coordinate-wise stepsizes.
3. [PARA] For a multi-layer transformer with restricted parameters, a stationary point of the training objective corresponds to preconditioned gradient descent using the inverse data covariance as the preconditioner.
4. [PARA] For a multi-layer transformer with relaxed parameters, a stationary point corresponds to a novel preconditioned gradient method (resembling GD++) that iteratively transforms covariates and takes preconditioned steps.

### Key equations
(5) $\min_{	heta} \mathbb{E}_{Z_0, w^\star} [ (y^{(n+1)} - 	ext{TF}_L(Z_0; 	heta))^2 ]$
(6) $Q_0 = \frac{1}{n} \left( \Sigma + \frac{1}{n} I \right)^{-1}$

### Notation notes
$Z_0$: input prompt matrix containing examples and query
$P_i, Q_i$: value and key-query parameter matrices for attention

### Method / instrument
Theoretical loss landscape analysis identifying global optima and critical points of the in-context training objective. Empirical validation by training linear transformers using ADAM and comparing the converged weights/loss against the theoretically derived critical points (using normalized Frobenius norm distance).

### Scope conditions
Theoretical analysis relies on linear attention (no softmax), though an extension to single-layer ReLU attention is provided. Data is assumed to be Gaussian.

### Epistemic status of each central claim
1. PROVED
2. PROVED
3. PROVED
4. PROVED

### Measured vs assumed
Theoretical optimum structures are proved, and their alignment with empirically trained transformer weights is measured.

### Interventions on inputs
[NOT FOUND]

### Sensitivity and derivative analysis
Gradients of the training objective with respect to the parameter matrices are analyzed to establish critical points.

### Relationship between prediction and observed targets
The learned weights are shown to match analytical constructions that map input covariates and labels to the true target via gradient descent updates.

### Datasets, models, scale
Synthetic random linear regression instances ($d=5$, $n=20$). Single, two-layer, and three-layer linear transformers.

### Limitations stated by the authors
The analysis primarily focuses on linear attention. The proof for multi-layer transformers only guarantees the existence of critical points, not necessarily global optima.

### Open questions named by the authors
The effect of nonlinear activations (like softmax) on the main results. A refined landscape analysis to understand all critical points and their (sub)optimality.

### Notes outside the schema
[NOT FOUND]

### Importance rating
5. Highly theoretical paper grounding empirical observations of gradient descent in ICL into formal landscape analyses.

DEEP PASS RECOMMENDED: no
### Uncertainty flags
[NOT FOUND]

## WHAT LEARNING ALGORITHM IS IN-CONTEXT LEARNING? INVESTIGATIONS WITH LINEAR MODELS

### Metadata
- **Title as printed in the document:** WHAT LEARNING ALGORITHM IS IN-CONTEXT LEARNING? INVESTIGATIONS WITH LINEAR MODELS
- **Authors:** Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, Denny Zhou
- **Venue / year:** ICLR 2023
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** [NOT FOUND]

### One-paragraph summary
This paper investigates the hypothesis that transformer-based in-context learners implicitly implement standard learning algorithms by analyzing their behavior on linear regression tasks. The authors first prove constructively that transformers can implement steps of gradient descent and closed-form ridge regression using a modest number of layers and hidden units. Empirically, they demonstrate that trained transformers transition between implementing different learning algorithms (like gradient descent, ridge regression, and ordinary least squares) depending on their depth, width, and the noise level in the data, ultimately matching Bayesian minimum-risk predictors under uncertainty. Finally, they use probing to show that learners' late layers non-linearly encode interpretable intermediate quantities, such as the moment matrix and the least-squares weight vector.

### Object of study
Autoregressive transformers performing in-context learning on linear regression tasks.

### Central claims
1. [PARA] Transformers can compute a single step of gradient descent for linear regression with a constant number of layers and $O(d)$ hidden space.
2. [PARA] Transformers can compute a single Sherman-Morrison update for closed-form ridge regression with a constant number of layers and $O(d^2)$ hidden space.
3. [PARA] Trained in-context learners behaviorally match the predictions of standard algorithms (OLS, Ridge, GD), transitioning between them based on model capacity (depth/width).
4. [PARA] In noisy settings, in-context learners closely match the predictions of the optimal Bayesian minimum-risk estimator.
5. [PARA] The moment matrix $X^\top Y$ and the OLS weight vector $\mathbf{w}_{\text{OLS}}$ can be successfully decoded from the hidden representations of trained in-context learners using non-linear probes.

### Key equations
(8) $L(\theta) = \mathbb{E}_{f, x_i} [ (f(x_n) - T_\theta(x_1, f(x_1), \dots, x_n))^2 ]$
(14) $\mathbf{w}' = \mathbf{w} + \frac{y_n - \mathbf{x}_n^\top \mathbf{w}}{1 + \mathbf{x}_n^\top (\dots) \mathbf{x}_n} (\dots) \mathbf{x}_n$

### Notation notes
$d$: input dimension
SPD: Squared prediction difference
ILWD: Implicit linear weight difference

### Method / instrument
Theoretical construction of transformer weights to implement RAW (Read-Arithmetic-Write) primitives. Empirical measurement of the squared prediction difference and implicit linear weight difference between trained transformers and baseline algorithms. Probing experiments (using MLPs and attention masks) on hidden states to decode algorithmic intermediate variables.

### Scope conditions
Analysis is restricted to linear regression problems with Gaussian priors and noise.

### Epistemic status of each central claim
1. PROVED
2. PROVED
3. MEASURED
4. MEASURED
5. MEASURED

### Measured vs assumed
Implementability bounds are proved, while the behavioral matching and internal encoding of algorithms in practically trained models are measured.

### Interventions on inputs
Varying the number of training examples, adding label noise, and restricting model depth and width to observe behavioral phase transitions.

### Sensitivity and derivative analysis
[NOT FOUND]

### Relationship between prediction and observed targets
The transformer's output is compared both to the true labels (loss) and to the outputs/weights generated by reference algorithms (OLS, Ridge, GD).

### Datasets, models, scale
Synthetic linear regression datasets ($d=8, 16$). Autoregressive transformers trained with depths up to 16 and hidden sizes up to 1024.

### Limitations stated by the authors
The probing indicates nonlinear encoding, which differs from the linear theoretical constructions. The analysis is limited to linear functions.

### Open questions named by the authors
Extending this methodology to larger-scale examples of ICL, such as language models, to determine if they also use interpretable algorithms.

### Notes outside the schema
[NOT FOUND]

### Importance rating
5. A rigorous mix of theory (constructive proofs), behavioral evaluation, and internal probing that confirms the gradient descent/regression algorithmic hypothesis.

DEEP PASS RECOMMENDED: no
### Uncertainty flags
[NOT FOUND]

## Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers

### Metadata
- **Title as printed in the document:** Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers
- **Authors:** Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, Furu Wei
- **Venue / year:** [NOT FOUND]
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** [NOT FOUND]

### One-paragraph summary
This paper proposes that in-context learning in GPT models can be understood as implicit finetuning, where the language model acts as a meta-optimizer that produces meta-gradients through forward computation. By analyzing the dual form between linear attention and gradient descent, the authors show that attention applied to demonstration tokens is equivalent to a parameter update on the model. They provide empirical evidence on six NLP classification tasks, showing that ICL behaves similarly to explicit finetuning in terms of prediction coverage, attention output updates, and attention weight distributions. Inspired by this dual view, they introduce a momentum-based attention mechanism that averages attention values like gradient momentum, which improves performance on both language modeling and ICL tasks.

### Object of study
Off-the-shelf pretrained GPT models (1.3B and 2.7B) performing in-context learning and explicit finetuning on NLP classification tasks.

### Central claims
1. [PARA] Transformer attention can be understood via a dual form of gradient descent, where attention to demonstration tokens acts as a parameter update (meta-gradients).
2. [PARA] In-context learning covers most of the correct predictions made by explicit finetuning (measured by Rec2FTP).
3. [PARA] ICL updates attention output representations in a direction highly similar to explicit finetuning (measured by SimAOU).
4. [PARA] ICL generates attention weights to query and training tokens that are highly correlated with those produced by finetuning.
5. [PARA] A momentum-based attention mechanism, designed by analogy to gradient descent with momentum, outperforms vanilla attention in language modeling and ICL.

### Key equations
(11) $\text{Attn}(X, X', \mathbf{q}) pprox W_{\text{ZSL}} \mathbf{q} + \Delta W_{\text{ICL}} \mathbf{q}$
(18) $\text{MoAttn}(V, K, \mathbf{q})_i = (1-\eta) \text{Attn}_i + \eta \text{MoAttn}_{i-1}$

### Notation notes
$\Delta W_{\text{ICL}}$: Implicit parameter updates generated by in-context learning
$W_{\text{ZSL}}$: Zero-shot learning parameters

### Method / instrument
Theoretical derivation of the dual form. Empirical comparison of ICL and a constrained 1-step explicit finetuning setup using metrics like Rec2FTP (prediction recall), SimAOU (cosine similarity of representation updates), and Kendall rank correlation (for attention weights). Experimental evaluation of momentum-based attention by training 350M parameter models from scratch.

### Scope conditions
Qualitative theoretical analysis relies on a relaxed linear attention form (removing softmax and scaling). Empirical analysis is restricted to classification tasks.

### Epistemic status of each central claim
1. PROVED
2. MEASURED
3. MEASURED
4. MEASURED
5. MEASURED

### Measured vs assumed
The similarities between ICL and finetuning behaviors are explicitly measured. The performance gain of momentum attention is measured.

### Interventions on inputs
[NOT FOUND]

### Sensitivity and derivative analysis
[NOT FOUND]

### Relationship between prediction and observed targets
Evaluated by classification accuracy on downstream validation sets for both ICL and finetuning.

### Datasets, models, scale
6 classification datasets (SST2, SST5, MR, Subj, AGNews, CB). Pretrained GPT 1.3B and 2.7B models for analysis. 350M parameter models for the momentum experiments.

### Limitations stated by the authors
The theoretical derivation considers a relaxed linear attention form, whereas standard attention may be more complex. Analysis is limited to classification tasks due to computational costs.

### Open questions named by the authors
Figuring out how ICL works in other architectures (like LSTMs). Investigating the mechanism of standard Transformer attention without the linear approximation. Exploring ICL mechanisms in multiple choice and open-ended generation tasks.

### Notes outside the schema
[NOT FOUND]

### Importance rating
5. Explores the connection between ICL and gradient descent in real-world large language models and tasks, moving beyond synthetic regression tasks.

DEEP PASS RECOMMENDED: no
### Uncertainty flags
[NOT FOUND]


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




# Camp directory name: c
Number of `.md` files found in it: 11

List of filenames:
- Amortized Multi-Task Bayesian In-Context Learning.md
- An Explanation of In-Context Learning as Implicit Bayesian Inference.md
- Are Large Language Models Really Not Bayesian Addressing the Mode-Gap and Position Invariance in In-Context Martingale Tests.md
- Full Bayesian Inference in Context via Prior-Fitted Networks with Normalizing Flows.md
- Martingale Posteriors for Prior-Fitted Networks.md
- Multi-Task Amortized Bayesian In-Context Learning.md
- Position The Future of Bayesian Prediction is Prior-Fitted.md
- Prior-Adaptive In-Context Bayesian Learning.md
- Statistical Foundations of Prior-Fitted Networks.md
- Transformers Can Do Bayesian Inference.md
- What and How Does In-Context Learning Learn Bayesian Model Averaging, Generalized Random Feature Model, and Pattern Matching.md

Date of extraction: 2026-08-08

## Amortized Multi-Task Bayesian In-Context Learning

### Metadata
- **Title as printed in the document:** Amortized In-Context Bayesian Posterior Estimation
- **Authors:** Sarthak Mittal, Niels Leif Bracher, Guillaume Lajoie, Priyank Jaini, Marcus Brubaker
- **Venue / year:** Preprint.
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** The filename "Amortized Multi-Task Bayesian In-Context Learning.md" differs from the printed title "Amortized In-Context Bayesian Posterior Estimation".

### One-paragraph summary
The paper provides a comparative analysis of amortized in-context Bayesian posterior estimation methods (forward KL vs reverse KL objectives, and different architectures like Transformers, DeepSets, GRU) for finding the posterior distribution over parameters. It evaluates on synthetic out-of-distribution tasks, cases of model misspecification, and transfer from simulated to real tabular problems, finding that reverse KL with Transformers and Normalizing Flows performs best for predictive tasks.

### Object of study
Amortized in-context Bayesian posterior estimators (using DeepSets, Transformers, or GRUs to model a distribution parameterized by a diagonal Gaussian or normalizing flows) trained from scratch on synthetic datasets. The likelihood models are Gaussian mean, GMM, linear regression, and nonlinear binary classification.

### Central claims
1. [PARA] Forward KL is better at capturing multi-modal posteriors in low dimensions, but reverse KL performs better in high dimensions. (Section 5)
2. [PARA] Reverse KL provides the flexibility to choose arbitrary datasets for training, improving out-of-distribution and sim-to-real transfer since it doesn't require access to true parameter samples during training. (Section 3, Section 4.3)
3. [PARA] Non-permutation invariant architectures like GRUs can outperform DeepSets, though Transformers perform the best overall. (Section 5)
4. [PARA] Normalizing flows substantially help forward KL models but only marginally help reverse KL models, as reverse KL tends to be mode-seeking. (Section 5)

### Key equations
(5) $p(	heta|D) = rac{p(D|	heta)p(	heta)}{\int p(D|	heta)p(	heta)d	heta}$
Here $p(	heta|D)$ is the posterior, $p(D|	heta)$ is the likelihood, and $p(	heta)$ is the prior.

(10) $\mathcal{L}_{	ext{Fwd-KL}}(\phi) = \mathbb{E}_{D \sim \chi} [ D_{KL}( p(\cdot|D) || q_\phi(\cdot|D) ) ]$
Here $\chi$ denotes the distribution over observations $D$, $p$ is the true posterior, and $q_\phi$ is the amortized estimator.

(12) $\mathcal{L}_{	ext{Fwd-KL}}(\phi) = - \mathbb{E}_{	heta \sim p} \mathbb{E}_{D \sim p(\cdot|	heta)} [ \log q_\phi(	heta|D) ] + C$
This is the simplified forward KL objective when $D$ is sampled from the assumed underlying model $p$.

(14) $\mathcal{L}_{	ext{Rev-KL}}(\phi) = \mathbb{E}_{D \sim \chi} [ D_{KL}( q_\phi(\cdot|D) || p(\cdot|D) ) ]$
Here reverse KL is minimized over an arbitrary distribution $\chi$ over datasets.

### Notation notes
$\chi$: denotes a distribution over observations $D$.

### Method / instrument
Training amortized posterior estimators (DeepSets, Transformers, GRUs) using forward KL and reverse KL objectives on synthetic tasks (Gaussian, GMM, LR, nonlinear classification) and evaluating them via predictive metrics (L2 loss, accuracy) and sample-based metrics (2-Wasserstein to MCMC). They also test sim-to-real generalization by evaluating on OpenML tabular datasets zero-shot and finetuning.

### Scope conditions
Synthetic parameter distributions versus real-world OpenML tabular datasets. The conclusions regarding reverse KL superiority in high dimensions and forward KL in low dimensions apply explicitly within the tested modalities (up to 100D for linear models). Results assume the models are trained directly on sets of observations using either simulated or out-of-distribution target data.

### Epistemic status of each central claim
1. MEASURED (toy)
2. MEASURED (toy)
3. MEASURED (toy)
4. MEASURED (toy)

### Measured vs assumed
The general mathematical formulations of KL divergences (forward vs reverse) and their simplified forms for simulation-based inference are assumed as the theoretical framework. The relative performance of the objectives and architectures on specific Bayesian tasks, as well as the effects of normalizing flows, are empirically measured on models they train.

### Interventions on inputs
[PARA] To generalize to variable feature dimensions, the paper embeds low-dimensional problems in a higher-dimensional space (100D) by masking unnecessary dimensions (setting extra features and parameters to 0).

### Sensitivity and derivative analysis
[NOT FOUND]

### Relationship between prediction and observed targets
[PARA] The model outputs a distribution $q_\phi(	heta|D)$ over parameters. The prediction for a target $x^*$ is based on the predictive distribution approximated as $\int p(x^*|	heta) q_\phi(	heta|D) d	heta$.

### Datasets, models, scale
Synthetic datasets (Gaussian mean, GMM, LR, nonlinear classification) and OpenML tabular datasets. Models include GRU, DeepSets, and Transformers, outputting parameters for a diagonal Gaussian or normalizing flows. Evaluated up to 100 dimensions.

### Limitations stated by the authors
"scaling our approach to more complex probabilistic models, leveraging better modeling choices for high-dimensional problems, and training a single model for multiple probabilistic models are important directions of future work."

### Open questions named by the authors
[NOT FOUND]

### Notes outside the schema
A surprising side result is that GRUs (which are non-permutation invariant) often outperform DeepSets (which are permutation invariant) for posterior estimation. The authors hypothesize this is due to DeepSets' reliance on a fixed pooling operator limiting expressivity.

### Importance rating
3. The paper is a strong benchmarking study comparing forward vs reverse KL and architectures for amortized posterior estimation, though it is more focused on the estimators themselves than what large pre-trained LLMs compute.
DEEP PASS RECOMMENDED: no

### Uncertainty flags
[NOT FOUND]

## An Explanation of In-Context Learning as Implicit Bayesian Inference

### Metadata
- **Title as printed in the document:** An Explanation of In-context Learning as Implicit Bayesian Inference
- **Authors:** Sang Michael Xie, Aditi Raghunathan, Percy Liang, Tengyu Ma
- **Venue / year:** [NOT FOUND]
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** The printed title case differs slightly from the filename ("In-context" vs "In-Context").

### One-paragraph summary
The paper provides a theoretical framework where in-context learning in large language models emerges from pretraining on documents with long-range coherence (modeled as a mixture of Hidden Markov Models with latent document-level concepts). The authors prove that during prompting, despite a distribution mismatch between concatenated independent examples and coherent pretraining documents, the LM can implicitly infer the shared latent concept to perform in-context learning through Bayesian marginalization. They introduce a synthetic dataset (GINC) where Transformers and LSTMs exhibit in-context learning, showing empirically that in-context accuracy improves with scale and depends heavily on example ordering.

### Object of study
A theoretical mixture of HMMs representing document generation with latent concepts. Also, empirical study of Transformer and LSTM models trained from scratch on a small-scale synthetic dataset (GINC) built according to this HMM mixture structure.

### Central claims
1. [PARA] In-context learning emerges when pretraining documents have long-range coherence, where the LM must implicitly infer a latent concept. (Section 1)
2. [PARA] As the number of in-context examples goes to infinity, the in-context predictor asymptotically achieves the optimal expected error under a distinguishability condition, despite distribution mismatch. (Theorem 1)
3. [PARA] In settings where the prompt concept is not perfectly distinguishable, the expected error of the in-context predictor decreases as the length of each example increases. (Theorem 2)
4. [PARA] Transformers and LSTMs trained on the synthetic GINC dataset exhibit in-context learning that improves with scale, model size, and number of examples. (Section 4)

### Key equations
(1) $p(	ext{output}|	ext{prompt}) = \int p(	ext{output}|	ext{prompt, concept}) p(	ext{concept}|	ext{prompt}) d	ext{concept}$
This is the posterior predictive distribution for a completion given the prompt, marginalizing out the latent concepts.

(6) $p(y|S_n, x_{	ext{test}}) = rac{\int p(S_n, x_{	ext{test}}|	heta) p(y|x_{	ext{test}}, 	heta) p(	heta) d	heta}{\int p(S_n, x_{	ext{test}}|	heta) p(	heta) d	heta}$
This expands the probability of the target label given the prompt $S_n$ and test input $x_{	ext{test}}$ under the pretraining distribution $p$.

(14) $E_{O \sim p_{	ext{prompt}}} [D_{KL}(p_{	ext{prompt}}^j || p_	heta^j) - D_{KL}(p_{	ext{prompt}}^j || p_{	heta^*}^j)] > \epsilon_{	ext{start}}^	heta + \epsilon_{	ext{delim}}^	heta$
The distinguishability condition, which requires the KL divergence signal between output distributions under the prompt concept $	heta^*$ and another concept $	heta$ to be larger than the distribution mismatch error terms at the start and delimiter.

### Notation notes
$	heta$: a latent concept that parameterizes the transitions of an HMM for a pretraining document.
$S_n$: a prompt consisting of a sequence of $n$ training examples.
$O_i^{	ext{ex}}$: the $i$-th input-output pair combined with the previous delimiter.

### Method / instrument
Mathematical proofs characterizing the limit of the posterior predictive distribution $p(	ext{output}|	ext{prompt})$ under a mixture of HMMs. Training GPT-2-based Transformers (4, 12, 16 layers) and LSTMs from scratch on a synthetic GINC dataset generated from a mixture of 5 HMM concepts. Measuring zero-shot and in-context classification accuracy as the number of examples, sequence lengths, and model scales increase. Ablations on pretraining structure and extrapolation to novel concepts.

### Scope conditions
Theoretical results apply when the pretraining distribution is exactly a mixture of HMMs with a shared document-level latent concept, and when the prompt delimiter and start states satisfy specific regularity and transition bounds (Assumptions 1-5). The model is assumed to perfectly fit the pretraining distribution in theory.

### Epistemic status of each central claim
1. ASSUMED (conceptually) / MEASURED (toy)
2. PROVED
3. PROVED
4. MEASURED (toy)

### Measured vs assumed
The mathematical model of pretraining data as a mixture of HMMs and the assumption that the LM perfectly learns the pretraining distribution are assumed. The ability of small Transformers and LSTMs to exhibit in-context learning in the constructed GINC dataset is empirically measured on models they train.

### Interventions on inputs
[PARA] Ablation of the pretraining data structure by training on data from only one concept (no mixture) or random transitions, showing in-context learning fails without the latent structure.
[PARA] Testing extrapolation to prompts generated from random concepts not seen during pretraining.

### Sensitivity and derivative analysis
[NOT FOUND]

### Relationship between prediction and observed targets
[PARA] The in-context predictor output $f_n(x_{	ext{test}})$ is derived by marginalizing out the latent concepts over the pretraining distribution conditioned on the prompt examples.

### Datasets, models, scale
Synthetic dataset (GINC) with vocab sizes 50, 100, 150, totaling ~10 million tokens. GPT-2 Transformers (29M to 115M parameters) and LSTMs (28M parameters). Tested with up to 64 examples in context.

### Limitations stated by the authors
"Our theory ... only studies the pretraining distribution."
They "leave extending our analysis to variable [example length] $k$ as future work."
They "leave bridging the mismatch between training examples, which are consistently length $k$, and test examples, which have random length, to future work."

### Open questions named by the authors
"We leave analysis of the effect of model scaling and model architecture as open questions."
"Future directions include more precise asymptotic results about the posterior distribution and results under misspecification/extrapolation."

### Notes outside the schema
The paper observes that scaling up the number of Transformer parameters steadily improves in-context accuracy even when the pretraining validation loss remains exactly the same, suggesting larger models improve in-context learning beyond just better perplexity. They also note LSTMs outperformed Transformers on the GINC dataset despite having fewer parameters.

### Importance rating
4. A fundamental theoretical paper linking in-context learning to Bayesian inference of latent concepts from a mixture of HMMs, although restricted to synthetic environments.
DEEP PASS RECOMMENDED: no

### Uncertainty flags
[NOT FOUND]

## Are Large Language Models Really Not Bayesian Addressing the Mode-Gap and Position Invariance in In-Context Martingale Tests

### Metadata
- **Title as printed in the document:** LLMs are Bayesian in Expectation, Not Realization
- **Authors:** Leon Chlon, Zein Khamis Fatima Sheaib, Maggie Chlon, Mahdi El Zein, MarcAntonio M. Awada
- **Venue / year:** Preprint.
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** The filename "Are Large Language Models Really Not Bayesian Addressing the Mode-Gap and Position Invariance in In-Context Martingale Tests.md" differs drastically from the printed title "LLMs are Bayesian in Expectation, Not Realization".

### One-paragraph summary
The paper addresses the objection that exact Bayesian posterior predictives for exchangeable data must be invariant to task-preserving order, whereas transformers are order-sensitive. The authors argue that exchangeability violations do not binary-refute Bayesian in-context learning but are instead priced by prequential log loss, where the expected regret of a single ordering decomposes into the regret of an order-averaged predictor plus an order-averaging gain. Through experiments on Qwen2.5 and Llama models involving closed-form Bayesian tasks, continuous linear regression, positional interventions, and permutation mixtures, they show that order sensitivity has a bounded code-length cost and that LLMs' predictive distributions sit much closer to the Bayesian posterior predictive than to a frequentist maximum-likelihood plug-in baseline.

### Object of study
Pretrained language models (Qwen2.5-7B/14B, Llama-3.1-8B) and small transformers trained from scratch. The evaluated predictive distributions are compared against closed-form exact Bayesian posterior predictive distributions (KT/Dirichlet finite-alphabet prediction, coarsened continuous Bayesian linear regression).

### Central claims
1. [PARA] Structural Bayesianity (invariance to task-preserving reordering) is not the correct evaluation metric; operational Bayes-competitiveness, measured by excess prequential code length against a Bayesian reference (cumulative predictive KL), is the appropriate measure. (Section 2)
2. [PARA] The expected prequential regret of a single admissible ordering decomposes exactly into the regret of the order-averaged predictor plus an order-averaging gain. (Theorem 2)
3. [PARA] Transformers evaluated on finite discrete prediction and Bayesian linear regression display predictive distributions closer to the Bayesian posterior predictive than to a frequentist plug-in baseline, especially in the small-data regime. (Section 4)
4. [PARA] Exchangeability failure (order sensitivity) is a property of the positional encoding a transformer is given rather than the transformer architecture itself. (Section 4.4)

### Key equations
(2) $E_P [\sum_{t=1}^n \ln rac{b_t(Y_t | H_t)}{q_t(Y_t | H_t)}] = E_P [\sum_{t=1}^n D_{KL}(b_t(\cdot|H_t) || q_t(\cdot|H_t))]$
Excess prequential code length under a Bayesian reference equals cumulative predictive KL.

(7) $E_P[D_{KL}(b_t || q_t^g)] = E_P[D_{KL}(b_t || ar{q}_t)] + J_t$
The regret of a single serialization equals the regret of the order-averaged predictor $ar{q}_t$ plus an order-averaging gain $J_t$.

(11) $q_{i}^{\lambda} = (1-\lambda) ar{q}_i + \lambda rac{1}{m}$
A safe-code floor that bounds the worst-case code length by mixing the normalized candidate distribution with a uniform distribution.

### Notation notes
$Z_t$: task-relevant information state at prefix $t$, which must be serialized.
$J_t$: order-averaging gain (the log-loss cost of using one admissible ordering instead of the averaged predictive distribution).

### Method / instrument
Theoretical formulation of prequential comparison and order-averaging decompositions. Empirical audits on Qwen2.5 and Llama-3.1 comparing their next-token probability distributions to exact Bayesian predictives (KT/Dirichlet for discrete, OLS vs Bayesian continuous BLR). Position interventions (normal, demo-local, random-offset) and from-scratch positional-encoding ablations to isolate task-preserving versus semantic order. Activation patching of learned subspaces to test whether decoded sufficient statistics are causally used to shape the next-token distribution. Downstream order-averaging log-loss evaluations on MMLU/GSM8K.

### Scope conditions
Order averaging is justified only for serializations that preserve the information state $Z_t$. If reordering changes the label, semantics, or trajectory (e.g. chronological tasks), it is not task-preserving. Continuous tasks are evaluated by scoring the induced distribution over discretized digit events, not the full continuous density.

### Epistemic status of each central claim
1. PROVED
2. PROVED
3. MEASURED (production)
4. MEASURED (toy)

### Measured vs assumed
The theoretical identities connecting log loss to predictive KL and the order-averaging decomposition are proved as mathematical properties of prequential predictors. The actual code-length cost of order sensitivity and the proximity of LLM predictive distributions to exact Bayesian versus frequentist baselines are empirically measured on frozen released checkpoints and from-scratch models.

### Interventions on inputs
[PARA] Resets of demonstration-local positions to preserve within-demonstration token order but remove absolute position, comparing performance against semantic-first controls.
[PARA] Clean/corrupt prompt pairs are formed with different sufficient statistics; activation patching of a learned low-rank subspace restores candidate logit differences, showing that the statistic is causally used.
[PARA] Formed banded evidence permutations (varying order of exchangeable evidence chunks) in an evidence-grounded QA task to measure dispersion and order-averaging gain.

### Sensitivity and derivative analysis
[NOT FOUND]

### Relationship between prediction and observed targets
[PARA] The model's sequence predictions $q_t(\cdot | y_{<t})$ are evaluated via prequential log loss against the true observations and compared against the Bayesian reference $b_t(\cdot | y_{<t})$ via predictive KL divergence.

### Datasets, models, scale
Qwen2.5-7B/14B (Instruct and base), Llama-3.1-8B. MMLU, GSM8K, BBH, FEVER, HotpotQA, NQ-Open, PopQA. Small transformers trained from scratch on i.i.d. Bernoulli sequences. Evaluated at context lengths supporting up to 256 examples.

### Limitations stated by the authors
"The results do not show that transformers instantiate literal posterior distributions, nor that all order effects are harmless."
"Order averaging is appropriate only when order is task-preserving."

### Open questions named by the authors
[NOT FOUND]

### Notes outside the schema
The paper emphasizes that a frequentist maximum-likelihood plug-in is actually a degenerate sequential predictor at small data sizes (assigning infinite code length to unseen symbols), precisely where the LM is admissible and matches the Bayesian posterior predictive's reservation of mass for unseen symbols.

### Importance rating
5. A rigorous and persuasive theoretical defense of the Bayesian interpretation against structural exchangeability critiques, backed by highly precise empirical measurements of operational Bayes-competitiveness in production models.
DEEP PASS RECOMMENDED: yes
A deeper pass should target the theoretical derivations of the safe-code floors and the precise mathematical connection between the order-averaging gain and empirical permutation sensitivities.

### Uncertainty flags
[NOT FOUND]

## Full Bayesian Inference in Context via Prior-Fitted Networks with Normalizing Flows

### Metadata
- **Title as printed in the document:** Can Transformers Learn Full Bayesian Inference In Context?
- **Authors:** Arik Reuter, Tim G. J. Rudner, Vincent Fortuin, David Rugamer
- **Venue / year:** Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025.
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** The filename "Full Bayesian Inference in Context via Prior-Fitted Networks with Normalizing Flows.md" differs from the printed title "Can Transformers Learn Full Bayesian Inference In Context?".

### One-paragraph summary
The paper introduces a framework for performing full Bayesian inference in context for commonly used statistical models (like GLMs and latent factor models) without parameter updates. By combining a TabPFN encoder with a diffusion transformer-decoder trained via continuous normalizing flows (flow matching), the model is trained entirely on synthetic samples from the joint prior-likelihood distribution. Extensive experiments on synthetic and real-world tabular datasets show that this in-context learner can produce complex, high-dimensional posterior samples comparable in quality to state-of-the-art MCMC methods and often better than varied Variational Inference baselines, avoiding issues with model misspecification or restricted variational families.

### Object of study
An in-context learning architecture (TabPFN encoder + diffusion transformer decoder with flow matching) trained on synthetic joint distributions to output full posterior samples for Generalized Linear Models (GLMs), Factor Analysis (FA), and Gaussian Mixture Models (GMMs). It is evaluated on tabular regression/classification datasets.

### Central claims
1. [PARA] Transformers can perform full Bayesian inference in context for complex models like GLMs and latent factor models by mapping datasets to posterior samples via continuous normalizing flows. (Abstract, Section 3)
2. [PARA] Learning the posterior can be framed as minimizing flow-matching objectives using samples directly from the joint prior-likelihood distribution, avoiding explicit likelihood evaluations or variational approximations. (Section 3.1)
3. [PARA] The proposed ICL approach yields posterior samples that closely match gold-standard HMC and often outperform structured and unstructured Variational Inference methods on both synthetic and real-world data, especially for non-Gaussian or multi-modal posteriors. (Section 4)
4. [PARA] Flow matching is essential for this performance; ablations substituting diffusion objectives or simple Gaussian approximations perform significantly worse. (Section 4.4)

### Key equations
(1) $R_	heta := \mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})} [d(f_	heta(\mathbf{x}), f_0(\mathbf{x}))]$
The ideal objective measuring divergence between the model's posterior $f_	heta(\mathbf{x})$ and the true posterior $f_0(\mathbf{x})$.

(2) $	ilde{R}_	heta := \mathbb{E}_{(\mathbf{x}, \mathbf{z}) \sim P^{\mathbf{x}, \mathbf{z}}} [L_d(\mathbf{x}, \mathbf{z}, 	heta)]$
The tractable surrogate objective that uses samples from the joint distribution to implicitly learn the posterior.

(7) $L_{CFM}(\mathbf{x}, \mathbf{z}^{(1)}, 	heta) := \mathbb{E}_{t \sim U([0,1]), \mathbf{z}^{(0)} \sim P_B} [ ||v_{t, 	heta}^{\mathbf{x}}(\gamma_t(\mathbf{z}^{(1)} | \mathbf{z}^{(0)})) - (\mathbf{z}^{(1)} - (1 - \omega)\mathbf{z}^{(0)})||_2^2 ]$
The conditional flow matching objective used to train the diffusion decoder.

### Notation notes
$\mathbf{z}$: latent variables or parameters over which the posterior is inferred.
$\mathbf{x}$: a dataset containing $K$ samples.
$P_B$: the base distribution for normalizing flows (standard normal).

### Method / instrument
Training a transformer (TabPFN-like encoder and diffusion decoder) using conditional flow matching on synthetic datasets generated from the priors and likelihoods of specific statistical models (GLMs, FA, GMMs). The trained models are evaluated zero-shot on 50 synthetic and 17 real-world tabular datasets from the Grinsztajn benchmark. Posterior samples are compared to HMC and VI baselines using C2ST (Classifier 2-Sample Test), MMD, and Wasserstein-2 distances.

### Scope conditions
Tested up to 50 dimensions, finding that beyond 20-50 dimensions, ICL performs comparably but does not outperform VI, possibly due to metric degradation in high dimensions. Tested against mild distribution shifts, showing performance degrades for larger gaps between training and testing distributions. Does not assume a single parameter per data point (can infer global parameters).

### Epistemic status of each central claim
1. MEASURED (production)
2. PROVED
3. MEASURED (production)
4. MEASURED (toy)

### Measured vs assumed
The mathematical formulation linking joint distribution sampling to conditional flow matching is assumed/derived mathematically. The actual ability of transformers to match or exceed MCMC/VI quality on real-world datasets is empirically measured, as is the necessity of flow matching versus diffusion.

### Interventions on inputs
[PARA] Ablated the generative architecture by replacing continuous normalizing flows (flow matching) with standard diffusion objectives and Gaussian parameterizations, finding flow matching crucial for success.
[PARA] Intervened on the encoder architecture, showing transformer encoders significantly outperform MLP encoders of the same size.

### Sensitivity and derivative analysis
[NOT FOUND]

### Relationship between prediction and observed targets
[PARA] The model is not primarily scored on point predictions of targets, but on the statistical discrepancy (C2ST, MMD, $W_2$) between the empirical distribution of its generated posterior samples and those generated by HMC (the target).

### Datasets, models, scale
50 synthetic datasets and 17 real-world datasets from Grinsztajn tabular benchmark. Architecture combines TabPFN encoder with a diffusion transformer decoder. Baselines include HMC (NUTS), Laplace Approximation, and ADVI (diagonal, full, structured, IAF).

### Limitations stated by the authors
"We hypothesize that a key reason for the failure to detect meaningful differences between the methods in high dimensions is due to the curse of dimensionality affecting our metrics."
"the relatively high number of latent variables in comparison to the limited number of data-points can yield overly flexible assumptions on the variational posterior causing the VI methods to overfit."

### Open questions named by the authors
[NOT FOUND]

### Notes outside the schema
The approach represents amortized inference at the dataset level, essentially compiling inference into a single forward pass by leveraging synthetic prior-predictive samples, sidestepping the need for explicitly defined or constrained variational families.

### Importance rating
4. A highly capable applied framework bridging prior-fitted networks and continuous normalizing flows to scale Bayesian in-context learning to full, continuous, multivariate posteriors on real tabular tasks.
DEEP PASS RECOMMENDED: no

### Uncertainty flags
[NOT FOUND]

## Martingale Posteriors for Prior-Fitted Networks

### Metadata
- **Title as printed in the document:** Uncertainty Quantification for Prior-Data Fitted Networks using Martingale Posteriors
- **Authors:** Thomas Nagler, David Rugamer
- **Venue / year:** Preprint.
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** The printed title is "Uncertainty Quantification for Prior-Data Fitted Networks using Martingale Posteriors", but the filename is "Martingale Posteriors for Prior-Fitted Networks.md". Also, there is a spelling difference (Prior-Data Fitted vs Prior-Fitted).

### One-paragraph summary
The paper proposes an efficient method to quantify the uncertainty of point estimates (such as conditional means or quantiles) produced by Prior-Data Fitted Networks (PFNs). Because PFNs only approximate the posterior predictive distribution (PPD) and cannot separate epistemic from aleatoric uncertainty, the authors use the Martingale Posterior (MP) framework. They introduce an approximate martingale posterior (AMP) algorithm that initializes with the PFN's PPD estimate and iteratively updates it via a computationally efficient nonparametric resampling scheme based on Gaussian copulas. This method achieves principled, tuning-free Bayesian credible intervals in seconds without requiring repeated calls to the underlying PFN, and theoretical analysis confirms its convergence properties.

### Object of study
Uncertainty quantification methods for PFNs (specifically TabPFN and TabICL). The target mathematical object is the Bayesian posterior distribution over predictive summaries (like conditional quantiles or means).

### Central claims
1. [PARA] PFNs approximate the posterior predictive distribution but do not inherently provide uncertainty quantification for predictive summaries (conditional means/quantiles) because they mix epistemic and aleatoric uncertainty. (Section 2.1)
2. [PARA] The Martingale Posterior framework can be adapted for conditional inference to quantify the uncertainty of PFN point estimates by using the PFN's PPD as a starting point. (Section 3)
3. [PARA] A surrogate update based on Gaussian copulas avoids computationally prohibitive iterative evaluations of the PFN (which would be $O(N^3)$), achieving $O(BN)$ complexity while retaining martingale properties. (Section 3.2, 3.6)
4. [PARA] The proposed Approximate Martingale Posterior (AMP) algorithm theoretically converges to a valid limit posterior distribution and empirically achieves correct coverage and well-calibrated credible intervals across synthetic and real-world tabular datasets. (Section 3.3, 4)

### Key equations
(2) $P_k(y) = (1-lpha_{n+k-1}) P_{k-1}(y) + lpha_{n+k-1} H_
ho(P_{k-1}(y), P_{k-1}(y_{n+k}))$
The computationally efficient surrogate update for the conditional CDF using Gaussian copulas.

(3) $lpha_i = 2^eta (i + 1)^{-eta}, eta > 1/2$
The learning rate schedule parameterized by $eta$ to accommodate the slower contraction rates typical of nonparametric regression.

### Notation notes
PPD: Posterior predictive distribution, $p(y|x, D_n)$.
$lpha_i$: Learning rate determining the contraction rate of the posterior.
$
ho$: A bandwidth hyperparameter for the Gaussian copula update that smooths the updates (set to 0.99).
$N$: Number of forward samples in one 'chain' of the AMP algorithm.
$B$: Number of independent 'chains' (replications).

### Method / instrument
Mathematical derivation of conditional martingale posteriors and efficient Gaussian copula updates. Theoretical proofs of convergence using time-uniform Azuma-Hoeffding inequalities. Simulated experiments using Bayesian additive models (B-splines) to compare AMP credible intervals against analytically optimal intervals. Benchmarking on 8 UCI tabular datasets (airfoil, boston, concrete, etc.) for 90% quantile and median regression, comparing AMP coverage, width, and runtime against a frequentist bootstrap baseline on TabPFN and TabICL.

### Scope conditions
Focuses on i.i.d. tabular prediction problems. Currently evaluates posterior credible intervals pointwise for each query point independently (does not provide uniform credible sets for joint posterior inference across multiple $x$).

### Epistemic status of each central claim
1. PROVED
2. PROVED
3. PROVED
4. PROVED / MEASURED (production)

### Measured vs assumed
The convergence of the algorithm, the martingale property of the copula updates, and the contraction rates are proved mathematically. The empirical coverage, interval width, and runtime efficiency of the AMP algorithm on TabPFN and TabICL are measured on synthetic and real tabular datasets.

### Interventions on inputs
[PARA] Ablated the initial PPD estimate by artificially compressing (scale by 0.8) and widening (scale by 1.25) it, showing that this directly proportionally affects the final coverage.
[PARA] Altered the learning rate schedule by setting $eta=1$ and removing the finite-sample blow-up factor, both of which led to severe undercoverage as predicted by theory.

### Sensitivity and derivative analysis
[NOT FOUND]

### Relationship between prediction and observed targets
[PARA] The method generates an approximate posterior distribution for predictive summaries (e.g., conditional median). Performance is evaluated by checking if a surrogate "true" conditional quantile (derived from an oracle model trained on a larger dataset) falls within the estimated credible intervals (Coverage), as well as evaluating the interval width.

### Datasets, models, scale
Models: TabPFN, TabICL. Datasets: Synthetic Bayesian additive models, 8 UCI datasets (airfoil, boston, concrete, diabetes, energy, fish, forest_fire, real). Training sets range from ~50 to 1200 instances. Hyperparameters: $N=50$ forward samples, $B=50$ chains.

### Limitations stated by the authors
"Our theory and experiments currently focus on iid tabular prediction problems."
"the current implementation computes posterior credible intervals for each query point separately. Thus, the computational cost scales with the size of the test set"
"does not provide uniform credible sets for joint posterior inference."

### Open questions named by the authors
"A possible direction is to combine the marginal posteriors ... through a dependence model ... Developing such joint posterior extensions is a promising direction for future work."

### Notes outside the schema
The authors emphasize that a naive frequentist bootstrap requires $O(n^2)$ refits of the PFN and severely undercovers because it ignores the unavoidable estimation bias in nonparametric regression problems, whereas AMP runs in seconds with a single PFN evaluation and achieves correct coverage.

### Importance rating
4. A highly practical and mathematically elegant solution to the critical problem of providing principled epistemic uncertainty quantification for point estimates generated by PFNs.
DEEP PASS RECOMMENDED: no

### Uncertainty flags
[NOT FOUND]

## Multi-Task Amortized Bayesian In-Context Learning

### Metadata
- **Title as printed in the document:** Multi-Task Bayesian In-Context Learning
- **Authors:** Qingyang Zhu, Eric Karl Oermann, Kyunghyun Cho
- **Venue / year:** Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026.
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** The filename "Multi-Task Amortized Bayesian In-Context Learning.md" differs from the printed title "Multi-Task Bayesian In-Context Learning".

### One-paragraph summary
The paper proposes Multi-Task Bayesian In-Context Learning, a framework for amortized hierarchical Bayesian predictive inference where prior information is explicitly represented as a prefix of in-context datasets. Unlike standard Prior-Data Fitted Networks where a single training prior is fixed in the model weights, this approach provides a direct test-time interface for adaptation to new priors by simply appending different prior-generating datasets to the prompt. Evaluated on linear and logistic regression under in-distribution and out-of-distribution heavy-tailed priors, as well as high-dimensional flow-based priors, the method matches oracle Bayesian predictors (like MCMC) while achieving orders of magnitude faster inference. It also demonstrates practical utility on real-world spatiotemporal climate data.

### Object of study
An amortized hierarchical Bayesian predictive inference engine instantiated as a small GPT-2 decoder-only transformer. The transformer is conditioned on sequences of datasets, where the first $K$ datasets act as "prior tasks" (drawn from a shared prior) and the last dataset is the "target task". Evaluated on synthetic regression/classification tasks and ERA5 climate data.

### Central claims
1. [PARA] Existing amortized inference approaches (like PFNs) bake a single implicit prior into model weights, making them inflexible and brittle under prior shifts. (Section 1)
2. [PARA] Representing prior information explicitly as a prefix of in-context datasets (multi-task Bayesian ICL) allows for controllable test-time adaptation to new priors without parameter updates. (Section 1, 4)
3. [PARA] The multi-task ICL matches the oracle hierarchical Bayesian predictive inference quantitatively and mechanistically, correctly interpreting the prefix as prior information rather than naive evidence pooling. (Section 5.2)
4. [PARA] The approach generalizes robustly to out-of-meta-distribution (OoMD) heavy-tailed priors provided the training mixture is sufficiently broad, matching the generalization bounds of exact hierarchical MCMC, while being orders of magnitude faster. (Section 5.3, 5.4)

### Key equations
(1) $p(y_t | x_t, C_{t-1}) pprox F_	heta(x_t, C_{t-1})$
Standard definition of in-context learning approximating the posterior predictive distribution (PPD).

(4) $C_{t-1} = D_{prior} \oplus D_{tgt} = (D^1, \dots, D^K, D^{K+1})$
The proposed input representation, appending $K$ prior datasets before the target evidence dataset to control the induced prior.

(7) $L(	heta) = \mathbb{E}[ - \sum_{(x_t, y_t) \in T} \log p_	heta(y_t | x_t, y_{<t}) ]$
The cross-entropy training objective which marginalizes over the hierarchical meta-distribution (simplified form).

### Notation notes
$D_{prior}$: the prefix sequence of $K$ datasets that share the same latent prior distribution as the target task.
$D_{tgt}$: the target dataset providing evidence.
$\lambda$: episode-level hyperparameters specifying the prior distribution.

### Method / instrument
Training a small GPT-2 model on multi-task episodes containing $K=20$ prior tasks and 1 target task. Evaluating the predictive KL divergence of the neural model against exact Bayesian reference models (MCMC, SVI, hierarchical MCMC/SVI) across varying target context lengths. Checking mechanistic alignment by systematically altering prior prefixes and comparing predictions against pooled and oracle MCMC baselines. Evaluating robustness across heavy-tailed (Student's t) priors and structured high-dimensional (spiral flow) priors. Benchmarking real-world performance on ERA5 spatiotemporal temperature prediction under IID and severe seasonal OOD splits.

### Scope conditions
Tested on relatively low-dimensional tasks (e.g. $d=8$ for regression) and structured spiral flow priors up to $d=8$. Acknowledges computational limits due to quadratic attention costs for long multi-task sequences. Notes that the sequential architecture lacks explicit permutation invariance, which can be suboptimal under severe OOD distribution shifts compared to Set-MT aggregation variants.

### Epistemic status of each central claim
1. ASSUMED (conceptually)
2. PROVED / MEASURED
3. MEASURED (toy)
4. MEASURED (toy/production)

### Measured vs assumed
The generative structure and properties of hierarchical Bayesian inference are assumed/derived mathematically. The actual performance of multi-task ICL in matching exact MCMC baselines, its test-time adaptability, its scaling behavior, and its performance on real-world ERA5 data are empirically measured.

### Interventions on inputs
[PARA] Altered the prior datasets $D_{prior}$ while keeping the target evidence dataset $D_{tgt}$ fixed; observed systematic changes in the variance of predicted logits that quantitatively matched oracle Bayesian prior-updating behavior rather than naive evidence pooling.

### Sensitivity and derivative analysis
[NOT FOUND]

### Relationship between prediction and observed targets
[PARA] The neural model predicts the next-token probability $p_	heta(y_t | x_t, y_{<t})$ in the target dataset conditioned on prior datasets. This predicted distribution is evaluated via KL divergence against the exact Bayesian posterior predictive distribution derived from MCMC.

### Datasets, models, scale
Model: small GPT-2 decoder-only transformer trained from scratch.
Datasets: Synthetic linear/logistic regression, hierarchical Student's t distributions, hierarchical spiral flow distributions, ERA5 real-world climate dataset. $K=20$ tasks per episode, $M=50$ examples per task, $d=8$ dimensions.

### Limitations stated by the authors
"Multi-task ICL might suffer from the attention cost that scales quadratically with sequence length."
"This architecture also does not explicitly enforce permutation invariance, either within a dataset or across datasets"

### Open questions named by the authors
[NOT FOUND]

### Notes outside the schema
The observation in ERA5 experiments that Set-MT (a variant enforcing set permutation invariance) outperforms the sequential MT under severe seasonal temporal shifts suggests that standard causal transformers overfit to spurious sequential orderings when generalized to non-IID temporal domains.

### Importance rating
4. A highly novel architecture mapping hierarchical Bayesian meta-learning into standard in-context prompting, enabling test-time prior manipulation previously absent in PFNs.
DEEP PASS RECOMMENDED: no

### Uncertainty flags
[NOT FOUND]

## Position: The Future of Bayesian Prediction Is Prior-Fitted

### Metadata
- **Title as printed in the document:** Position: The Future of Bayesian Prediction Is Prior-Fitted
- **Authors:** Samuel Müller, Arik Reuter, Noah Hollmann, David Rügamer, Frank Hutter
- **Venue / year:** Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025.
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** The filename is "Position The Future of Bayesian Prediction is Prior-Fitted.md" and the title is "Position: The Future of Bayesian Prediction Is Prior-Fitted".

### One-paragraph summary
This position paper argues that Prior-Data Fitted Networks (PFNs) represent the future of Bayesian prediction, especially in domains with limited real-world data but abundant computational resources for pre-training. PFNs amortize the cost of Bayesian inference by training on synthetically generated datasets sampled from a prior, reducing test-time inference to a single fast forward pass. The authors highlight the advantages of this declarative approach—such as the ability to encode complex priors (e.g., TabPFN's structural causal models) without explicitly defining likelihoods or latent variables—while acknowledging current limitations like restricted interpretability and quadratic scaling with dataset size. They propose future research directions, including incorporating extra contextual inputs (e.g., PPL interpreters), integrating reinforcement learning for Bayesian optimization, and improving architectural scaling and interpretability mechanisms, to fully realize the potential of probabilistic foundation models.

### Object of study
Prior-Data Fitted Networks (PFNs), a class of neural models that perform amortized Bayesian inference by pre-training on synthetic datasets drawn from a predefined prior over datasets.

### Central claims
1. [INFER] PFNs represent a paradigm shift in Bayesian prediction, moving from imperative per-dataset inference (like MCMC or VI) to declarative prior specification where algorithms are defined simply by the data generation process. (Section 1, 3)
2. [INFER] PFNs are uniquely positioned to exploit exponentially growing pre-training compute to solve data-scarce problems, where real-world data collection has stagnated. (Section 1, 3)
3. [INFER] PFNs possess significant advantages over traditional Bayesian methods: they are easier to implement, can handle highly complex latent structures (like SCMs) without explicit modeling, avoid data leakage by using only synthetic training data, and are orders of magnitude faster at inference time. (Section 2.3, 3)
4. [INFER] Current limitations of PFNs (e.g., poor interpretability, scaling to large datasets, struggles with identical examples or heterogeneous feature scales) are not fundamental but present actionable research opportunities, such as latent prediction, architectural modifications (linear attention, zero attention), and adaptive compute allocation. (Section 3, 5, 6)

### Key equations
(1) $L(\theta) = \mathbb{E}_{D \sim p(D), (x_{test}, y_{test}) \sim D} [ -\log q_\theta(y_{test} | x_{test}, D_{train}) ]$
The pre-training cross-entropy objective which allows the neural network to approximate the true posterior predictive distribution.

(2) $L(\theta) = \mathbb{E}_{D \sim p(D)} [ KL(p(y | x_{test}, D_{train}) || q_\theta(y | x_{test}, D_{train})) ]$
Shows that minimizing the cross-entropy loss is equivalent to minimizing the expected KL-divergence to the true Bayesian posterior predictive distribution.

(3) $q_\theta(y|x, D) \approx \mathbb{E}_{y_1 \sim q_\theta(\cdot|x_1, D)} \dots \mathbb{E}_{y_n \sim q_\theta(\cdot|x_n, (x_{1:n-1}, y_{1:n-1}) \cup D)} [ q_\theta(y|x, (x_{1:n}, y_{1:n}) \cup D) ]$
The Martingale property, a theoretical condition that true Bayesian predictions (and optimally trained PFNs) should satisfy, indicating consistency when conditioning on their own sequential predictions.

### Notation notes
$D$: A dataset sampled from the prior $p(D)$.
$q_\theta$: The Prior-Data Fitted Network (PFN) approximating the posterior predictive distribution.
$\xi$: Latent variables used to define the dataset generative process $p(D|\xi)$.

### Method / instrument
A position paper synthesizing literature on PFNs. It provides conceptual analysis, comparative evaluation against traditional methods (MCMC, VI, GPs), and theoretical framing (e.g., the Martingale property). It also presents a small empirical demonstration (Figure 2) evaluating whether PFNs satisfy the Martingale property over sequential rollouts.

### Scope conditions
Focuses on supervised learning tasks (tabular data, learning curves, time series) and data-scarce domains. Acknowledges that PFNs currently struggle on very large datasets compared to specialized tree-based or large-scale neural network models, and may not fully replace non-Bayesian treatments for tasks like large-scale language modeling.

### Epistemic status of each central claim
1. ASSUMED (conceptual argument)
2. ASSUMED (conceptual argument)
3. PROVED (by reference to existing literature)
4. ASSUMED (hypothesizing future research)

### Measured vs assumed
The claims about PFN advantages and speed are based on prior published empirical measurements (e.g., TabPFN). The claims about the "future of Bayesian prediction", compute trends, and the viability of proposed future research directions (like RL for BO or PPL interpreters) are assumed or hypothesized as part of the position piece. The small experiment on the Martingale property (Figure 2) is a measured demonstration.

### Interventions on inputs
[NOT FOUND]

### Sensitivity and derivative analysis
[NOT FOUND]

### Relationship between prediction and observed targets
[PARA] Emphasizes that PFNs directly model the predictive distribution $p(y|x, D)$ rather than learning a posterior over the latents $p(\xi|D)$, distinguishing them from Simulation-Based Inference (SBI) methods.

### Datasets, models, scale
Models: TabPFN, GPs, BNNs, MCMC, VI. Discusses scaling up to 10,000 examples (TabPFN v2). Points out pre-training efficiency (2,700 GPU hours for TabPFN v2).

### Limitations stated by the authors
"PFNs can be less interpretable compared to traditional methods, as they hide the latent from the user."
"The support set of datasets and their generating distributions is typically limited and less well defined ... it is less clear what data they work well on."
"PFNs work best for smaller datasets and are commonly outperformed on large datasets."
"PFNs face slow inference times due to their architecture combining dataset fitting with prediction."
"encoder-only transformers without positional embeddings struggle to count identical examples"
"PFNs struggle with heterogeneous data distributions ... requiring prior knowledge of feature distributions for preprocessing."

### Open questions named by the authors
"Understanding when and why these approximation failures occur remains an open challenge."
"understanding how the prior and observed data interact [for interpretability]"
"Finding commonalities and differences in the way PFNs handle data across different priors, seeds, architectures and training steps."

### Notes outside the schema
The paper serves as a roadmap for the PFN community, advocating for extensions like "in-context interpreters" where users provide probabilistic programs as prompts, and RL-based adaptive computation to improve inference efficiency.

### Importance rating
5. A definitive position paper establishing the broader vision, current taxonomy, and future research agenda for Prior-Data Fitted Networks.
DEEP PASS RECOMMENDED: no

### Uncertainty flags
[NOT FOUND]

## Prior-Adaptive In-Context Bayesian Learning

### Metadata
- **Title as printed in the document:** Distribution Transformers: Fast Approximate Bayesian Inference With On-The-Fly Prior Adaptation
- **Authors:** George Whittle, Juliusz Ziomek, Jacob Rawling, Maike A. Osborne
- **Venue / year:** Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026.
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** The filename is "Prior-Adaptive In-Context Bayesian Learning.md", but the printed title is "Distribution Transformers: Fast Approximate Bayesian Inference With On-The-Fly Prior Adaptation".

### One-paragraph summary
The paper introduces Distribution Transformers (DTs), a novel architecture for amortized Bayesian inference that explicitly supports on-the-fly prior adaptation without retraining. To achieve this and maintain approximate conjugacy for sequential inference tasks, DTs represent both the prior and posterior distributions as Gaussian Mixture Models (GMMs). A transformer decoder maps the prior GMM components (embedded as an unordered token sequence) to the posterior GMM components, conditioned on observed data via cross-attention. Extensive experiments on static and sequential tasks (e.g., sensor fusion, stochastic volatility) demonstrate that DTs achieve faster inference times and lower expected negative log-likelihoods compared to fixed-prior amortized models like PFNs, as well as classical methods like SVI and Extended Kalman Filters, matching the performance of computationally intensive Particle Filters at a fraction of the cost.

### Object of study
Distribution Transformers (DTs), a neural architecture that maps arbitrary prior probability distributions (represented as Gaussian Mixture Models) and datasets to approximate posterior distributions (also represented as GMMs) for use in both static and sequential Bayesian inference.

### Central claims
1. [PARA] Existing amortized Bayesian inference methods (like PFNs) typically fix the prior during training; changing it at test time requires retraining, and they do not preserve family structure between prior and posterior, limiting sequential composition. (Section 1)
2. [PARA] Distribution Transformers overcome these limitations by tokenizing distributions as GMMs and performing posterior updates within the same parametric family, bridging amortized inference and sequential filtering. (Section 1)
3. [PARA] DTs can approximate true posteriors more accurately than PFNs and SVI (especially when the prior is shifted at test time) because of the expressive GMM representation, rather than the Riemannian distributions used by PFNs. (Section 4.1, 4.2)
4. [PARA] Due to their approximate conjugacy, DTs excel in sequential inference settings, tracking true states much more accurately than Extended Kalman Filters and matching Particle Filters while being nearly 50x faster. (Section 4.3)

### Key equations
(1) $q_\theta(x) = \sum_i w_i \mathcal{N}(x; \mu_i, \Sigma_i)$
The definition of the Gaussian Mixture Model used to represent all probability distributions (priors and posteriors) as an unordered sequence of component parameters.

(2) $\ell'_\theta = \mathbb{E}_{p(\phi, x, z)} [ - \log q_\theta(f(x) | \phi) - \log q_\theta(f(x) | z, \phi) ]$
The combined training objective (loss function) that minimizes the expected KL-divergence between the true and approximate posterior, alongside a prior regularization term to ensure a shared latent space.

### Notation notes
$\phi$: Parameters of the prior distribution.
$z$: Observations or dataset.
$f$: A sample-space transform mapping the sample space to $\mathbb{R}^n$ for the GMM approximation.
$q_\theta$: The Distribution Transformer's GMM approximation.

### Method / instrument
A novel transformer decoder architecture equipped with custom embedding/unembedding layers for GMM parameters. Trained on synthetic data generated from meta-priors. Evaluated on an analytical verification task (inverse-gamma prior), an intractable posterior task (Gaussian Process with hyperpriors), a quantum system parameter inference task, and two sequential inference problems (Bayesian sensor fusion and a 10-dimensional factor-structure stochastic volatility model). Baselines include SVI, MCMC, PFNs, TabPFNv2, ACE, Extended Kalman Filter (EKF), and Particle Filter (PF).

### Scope conditions
Focuses on distributions that can be approximated by GMMs. Acknowledges that the computational complexity is quadratic in the number of GMM components and quadratic in the underlying latent variable dimension due to full-covariance decoding. The sequential inference experiments are evaluated up to moderate depths to check for accumulated errors.

### Epistemic status of each central claim
1. PROVED
2. PROVED
3. MEASURED (toy)
4. MEASURED (toy/production)

### Measured vs assumed
The universality of GMMs and permutation equivariance of the architecture are mathematically grounded. The equivalence of the loss function to expected KL-divergence is proved (Proposition 3.1). The inference speed, posterior NLL, and tracking accuracy are empirically measured against baselines.

### Interventions on inputs
[PARA] Tested the models under "narrow" and "wide" meta-prior regimes to evaluate how the models handle shifts in the test-time prior. PFNs failed completely under wide meta-priors because their implicit prior did not cover the target space, whereas DTs adapted successfully by conditioning on the explicitly provided prior parameters.

### Sensitivity and derivative analysis
[NOT FOUND]

### Relationship between prediction and observed targets
[PARA] The model predicts a full GMM posterior density over the latent variables $x$. This is evaluated using Expected KL-Divergence against analytical ground truth (when available) or Expected Negative Log-Likelihood against test data or oracle MCMC/Particle Filter outputs.

### Datasets, models, scale
Models: 2-component and 5-component DTs, compared with 15-bucket and 5000-bucket PFNs, SVI, EKF, and PF.
Datasets: Synthetic 1D inverse-gamma problem, 5D GP hyperposterior, simulated 2-level quantum system, 4D state sensor fusion, and 10D stochastic volatility model. Batch size 1000 problems for timing benchmarks.

### Limitations stated by the authors
"training must cover a higher-dimensional space, increasing offline training cost relative to fixed-prior baselines"
"computational complexity is quadratic in the number of GMM components (via self-attention) and quadratic in the underlying latent variable dimension (via full-covariance decoding per token)"
"use in sequential settings may accumulate error over recursions"

### Open questions named by the authors
[NOT FOUND]

### Notes outside the schema
The key distinction of this approach compared to other recent prior-adaptive ICL methods (like ACE or Multi-Task Bayesian ICL) is the focus on approximate conjugacy—outputting a posterior in the same GMM format as the prior—which is critical for sequential filtering (time-series) applications.

### Importance rating
4. An innovative architecture bridging amortized inference and sequential filtering by using GMMs as a universal, tokenizable probabilistic interface for transformers.
DEEP PASS RECOMMENDED: no

### Uncertainty flags
[NOT FOUND]

## Statistical Foundations of Prior-Fitted Networks

### Metadata
- **Title as printed in the document:** Statistical Foundations of Prior-Data Fitted Networks
- **Authors:** Thomas Nagler
- **Venue / year:** Proceedings of the 40th International Conference on Machine Learning, Honolulu, Hawaii, USA. PMLR 202, 2023.
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** Filename has "Prior-Fitted Networks", title printed is "Prior-Data Fitted Networks".

### One-paragraph summary
This paper provides a theoretical and statistical foundation for Prior-Data Fitted Networks (PFNs). While PFNs are originally motivated by Bayesian nonparametrics (approximating a posterior predictive distribution on simulated data), this work argues that their empirical in-context learning behavior on test sets larger than their pre-training sets can be better understood through a frequentist lens: as pre-tuned, but untrained predictors. The author demonstrates that the variance of such a fixed predictor vanishes if its sensitivity to individual training samples diminishes as sample size grows, a property naturally satisfied by the attention mechanisms in transformer architectures. However, for the bias to vanish asymptotically, the predictor must be appropriately localized around the test feature. Standard transformers lack this localization property, leading the author to propose a simple, post-hoc "Localized PFN" variant that uses only k-nearest neighbors during inference to reduce asymptotic bias.

### Object of study
Prior-Data Fitted Networks (PFNs) and their in-context learning mechanics from a statistical generalization perspective.

### Central claims
1. [PARA] The posterior predictive distributions (PPDs) that PFNs aim to approximate can learn from data if the prior has large enough support and does not concentrate too much away from the true hypothesis. (Section 3)
2. [PARA] PFNs pre-trained on a distribution of small sample sizes implicitly regularize the expected complexity of the network. (Section 4)
3. [INFER] The in-context learning behavior of fixed, pre-trained PFNs on novel, large datasets can be explained by treating them as untrained frequentist predictors whose hyperparameters have been meta-learned. (Section 5.1)
4. [PROVED] For a fixed PFN predictor, the variance vanishes as sample size grows if its sensitivity to individual training samples diminishes, a condition mathematically guaranteed by the standard transformer's attention mechanism. (Theorem 5.2, Theorem 6.2)
5. [PROVED] A PFN's bias can only vanish if the predictor is appropriately "local" (sensitive primarily to training samples near the test feature). Since transformers are not inherently local in this sense, their bias decreases initially but plateaus; a post-hoc localization (e.g., k-nearest neighbors) can force the bias to continue decreasing. (Theorem 5.4, Section 6.4, Figure 1)

### Key equations
(1) $q_\theta^*(y|x, D_n) \approx \pi(y|x, D_n) = \int p(y|x) d\Pi(p | D_n)$
The PFN $q_\theta$ is trained to approximate the true posterior predictive distribution $\pi$.

(2) $\sup_{n, D_n, D_n', y, x} | q_\theta(y|x, D_n) - q_\theta(y|x, D_n') | \le L n^{-\alpha}$
The diminishing sensitivity condition required for a predictor's variance to vanish as sample size $n$ increases.

(3) $\operatorname{Var}_{P_0}[q_\theta(y|x, D_n)] \le \frac{1}{2} L^2 n^{1-2\alpha}$
McDiarmid's inequality bound showing that if sensitivity diminishes with $\alpha > 1/2$, the variance of the predictor vanishes.

### Notation notes
$\Pi$: The prior distribution over conditional probability models $p$.
$\pi(y|x, D_n)$: The posterior predictive distribution (PPD).
$q_\theta$: The Prior-Data Fitted Network (PFN) parameterized by $\theta$.
$D_n$: A dataset of size $n$.
$P_0$: The true data generating distribution.

### Method / instrument
Theoretical statistical analysis of in-context learning in PFNs, drawing on Bayesian nonparametrics, frequentist bias-variance decomposition, and concentration inequalities (McDiarmid's inequality). The theoretical claims regarding transformer variance and bias are numerically validated using a small simulated dataset ($n$ up to 4000) passed to the pre-trained TabPFN model.

### Scope conditions
The theoretical results apply primarily to the $iid$ setting for classification/regression. The bias and variance analyses assume standard transformer architectures (e.g., softmax attention) and evaluate limits as inference sample size $n \to \infty$.

### Epistemic status of each central claim
1. PROVED (by reference to Bayesian nonparametrics)
2. ASSUMED (conceptual interpretation)
3. ASSUMED (conceptual interpretation)
4. PROVED (Theorem 6.2)
5. PROVED / MEASURED (Theorem 5.4, numerical validation)

### Measured vs assumed
The mathematical bounds (diminishing sensitivity, vanishing variance, necessary conditions for vanishing bias) are formally proved. The practical manifestation of these bounds (TabPFN's variance dropping at $1/n$ and its bias plateauing around $n \approx 1000$ unless localized) is measured via simulation in Section 6.5.

### Interventions on inputs
[PARA] The numerical validation (Section 6.5) varies the inference sample size $n$ up to 4000 to observe the behavior of bias and variance on a TabPFN model that was pre-trained only on $n \le 1000$.

### Sensitivity and derivative analysis
[NOT FOUND]

### Relationship between prediction and observed targets
[PARA] The analysis decomposes the expected prediction error of the PFN (relative to the true conditional probabilities) into a variance component and a squared bias component.

### Datasets, models, scale
Models: TabPFN (version 0.1.8).
Datasets (Simulation): Synthetic data generated from $p_0(1|X) = 1/2 + \sin(\mathbf{1}^\top X)/2$, with $X \sim N(0, I_5)$. Sample sizes up to 4000.

### Limitations stated by the authors
"Owing to the standard transformer architecture, the maximal feature size is fixed, and the algorithm scales quadratically in the number of samples."
Acknowledges that current PFN implementations are constrained to small sample sizes and features.

### Open questions named by the authors
"The rate at which corresponding [structural causal model] posteriors contract is a complex issue and poses an interesting open question."
Suggests future research could adapt the transformer architecture to incorporate localization into pre-training, or augment it with Bayesian averaging mechanisms.

### Notes outside the schema
The paper explicitly bridges the gap between the Bayesian motivation of PFNs and their frequentist evaluation, providing a mechanistic explanation for why and how transformers perform in-context learning on previously unseen sample sizes.

### Importance rating
5. A rigorous and essential theoretical contribution that demystifies the statistical mechanics of Prior-Data Fitted Networks.
DEEP PASS RECOMMENDED: no

### Uncertainty flags
[NOT FOUND]

## Transformers Can Do Bayesian Inference

### Metadata
- **Title as printed in the document:** TRANSFORMERS CAN DO BAYESIAN INFERENCE
- **Authors:** Samuel Müller, Noah Hollmann, Sebastian Pineda, Josif Grabocka, Frank Hutter
- **Venue / year:** Published as a conference paper at ICLR 2022
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** Title printed is all-caps "TRANSFORMERS CAN DO BAYESIAN INFERENCE".

### One-paragraph summary
This seminal paper introduces Prior-Data Fitted Networks (PFNs), demonstrating that Transformers can perform amortized Bayesian inference for arbitrary priors in a single forward pass. By restating posterior approximation as a supervised learning problem on set-valued inputs, a PFN is trained on synthetic datasets sampled from a given prior over supervised learning tasks. During inference, the trained PFN takes an actual dataset and a test point as input and outputs a predictive distribution that closely approximates the true posterior predictive distribution (PPD). To handle continuous distributions flexibly, the authors introduce a novel discretized regression head called the Riemann Distribution. Experimental evaluations show that PFNs can near-perfectly mimic the tractable PPDs of Gaussian Processes and provide faster, better-calibrated inference for intractable priors (e.g., GPs with hyperpriors, Bayesian Neural Networks) compared to SVI and MCMC methods, culminating in strong performance on small tabular classification and few-shot image classification tasks.

### Object of study
Prior-Data Fitted Networks (PFNs) and their capacity to approximate Bayesian posterior predictive distributions using standard Transformer architectures equipped with a Riemann Distribution regression head.

### Central claims
1. [INFER] Transformers trained via supervised learning on synthetic datasets generated from a prior can directly approximate Bayesian posterior predictive distributions for that prior, without needing explicit posterior inference during test time. (Section 1, 3)
2. [PROVED] The proposed training objective (Prior-Data Negative Log-Likelihood) is mathematically equivalent (up to a constant) to minimizing the expected Kullback-Leibler divergence between the exact posterior predictive distribution and the PFN's approximation. (Section 3, Insight 1, Corollary 1.1)
3. [MEASURED] PFNs can mimic exact Gaussian Process regression near-perfectly, and when applied to intractable models (like GPs with hyperpriors or BNNs), they provide approximations superior to MCMC (NUTS) and Variational Inference (SVI) while running orders of magnitude (200x to 10,000x) faster at inference time. (Section 5)
4. [MEASURED] PFNs equipped with a BNN prior over architectures enable tuning-free tabular classification in a single forward pass, outperforming heavily tuned baselines (e.g., XGBoost, CatBoost) in both predictive accuracy and uncertainty calibration (Expected Calibration Error). (Section 6)

### Key equations
(1) $\ell_\theta = \mathbb{E}_{D \cup \{(x,y)\} \sim p(D)} [ -\log q_\theta(y|x, D) ]$
The Prior-Data Negative Log-Likelihood (Prior-Data NLL) loss function used to train PFNs on synthetic datasets.

(2) $\ell_\theta = \mathbb{E}_{x,D \sim p(D)} [ H(p(\cdot|x, D), q_\theta(\cdot|x, D)) ]$
Shows that minimizing the cross-entropy over the prior is equivalent to matching the posterior predictive distribution.

### Notation notes
$D$: A supervised dataset $\{(x_i, y_i)\}_{i=1}^n$.
$p(t)$: The prior over the latent task (or function) $t$.
$p(D)$: The generative process of drawing datasets from the prior.
$q_\theta$: The PFN model parameterized by $\theta$.
$\mathbf{B}$: The buckets forming the Riemann Distribution regression head.

### Method / instrument
Introduction of a novel learning algorithm (PFN) using a permutation-invariant Transformer encoder and a discretized "Riemann distribution" for regression tasks. Evaluated empirically across multiple settings: comparing against exact GP posteriors for validation; benchmarking inference speed and negative log-likelihood against NUTS, MLE-II, and SVI for GPs with hyperpriors and BNNs; assessing ROC AUC and Expected Calibration Error on 20 OpenML tabular datasets against standard ML baselines; and evaluating accuracy on the Omniglot few-shot image classification task.

### Scope conditions
Focuses on small-scale datasets during inference due to the quadratic scaling of standard Transformers with respect to context size (number of training examples). The empirical validation on tabular datasets is restricted to datasets with $<100$ features and evaluated using 30 training samples.

### Epistemic status of each central claim
1. ASSUMED / PROVED (Conceptually framed, theoretically justified)
2. PROVED
3. MEASURED
4. MEASURED

### Measured vs assumed
The theoretical equivalence between the loss function and KL divergence to the exact PPD is formally proved. The inference speeds, calibration errors (ECE), log-likelihoods, and ROC AUCs are empirically measured in comparative benchmarks.

### Interventions on inputs
[PARA] The method inherently operates by sampling synthetic datasets from different priors (e.g., GP, GP with hyperpriors, BNNs over architectures, synthetic stroke models) to intervene on the inductive bias embedded into the PFN during training.

### Sensitivity and derivative analysis
[NOT FOUND]

### Relationship between prediction and observed targets
[PARA] The model directly outputs a probabilistic prediction for the target variable $y$ given the query $x$ and the context $D$. This is represented as a Riemann Distribution (discretized probability mass) for regression or standard softmax for classification. The model is trained to maximize the log-likelihood of the true target under this predicted distribution.

### Datasets, models, scale
Models: PFN (Transformer encoder), XGBoost, CatBoost, Logistic Regression, GP, BNN, NUTS, SVI (Pyro).
Datasets: 20 curated tabular datasets from OpenML AutoML Benchmark (simplified to balanced binary classification, 30 training samples), Omniglot (5-way 5-shot). Training datasets for PFN are purely synthetic, simulating up to 2000 examples (for GP toy tasks) or small contexts for tabular/Omniglot.

### Limitations stated by the authors
"Currently, it is hard to reap the benefits of deep learning for Bayesian methods" (Motivating limitation of prior work).
Does not explicitly discuss scaling limits to large $n$ in the conclusion, though the experimental setup is constrained to small $n$.

### Open questions named by the authors
"Work on finding novel priors that are now feasible to approximate using PFNs."
"Work on architectures that are well-fit for this task, as we simply used a slight adaption of current Transformer models."
"Work on scaling PFNs to even larger real-world problems."
"Work on using our model for the amortized simulation-based inference setting."

### Notes outside the schema
This is the original foundational paper for PFNs, establishing the method, the term "Prior-Data Fitted Networks", and proving its equivalence to Bayesian inference.

### Importance rating
5. Foundational paper that introduced Prior-Data Fitted Networks.
DEEP PASS RECOMMENDED: no

### Uncertainty flags
[NOT FOUND]

## What and How does In-Context Learning Learn? Bayesian Model Averaging, Parameterization, and Generalization

### Metadata
- **Title as printed in the document:** What and How does In-Context Learning Learn? Bayesian Model Averaging, Parameterization, and Generalization
- **Authors:** Yufeng Zhang, Fengzhuo Zhang, Zhuoran Yang, Zhaoran Wang
- **Venue / year:** [NOT FOUND]
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** The filename ends with "Bayesian Model Averaging, Generalized Random Feature Model, and Pattern Matching.md", but the printed title ends with "Bayesian Model Averaging, Parameterization, and Generalization". The authors have equal contribution markers but no venue/year is explicitly stated on the first page block.

### One-paragraph summary
This paper provides a comprehensive theoretical framework for understanding In-Context Learning (ICL) in Large Language Models (LLMs) from a Bayesian perspective. The authors formulate ICL as a process where a perfectly pretrained transformer implicitly implements Bayesian Model Averaging (BMA) over a latent variable model governing the token sequences. By adopting an online learning framework, they establish that the ICL regret (the average prediction error over prompt examples) decays at a rate of $O(1/T)$, meaning that prompting with more examples allows the model to approach the performance of an oracle that knows the true underlying concept. Furthermore, they demonstrate that the transformer architecture—specifically its attention mechanism—naturally encodes this BMA algorithm. Crucially, they use the PAC-Bayes framework to provide a fine-grained statistical analysis of the pretraining stage, showing that the pretraining error in total variation distance is bounded by the sum of an approximation error (which decays exponentially with network depth) and a generalization error (which decays sublinearly with the pretraining dataset size).

### Object of study
The mechanism of In-Context Learning (ICL) in Large Language Models (LLMs), particularly how transformers implicitly execute Bayesian Model Averaging, and the statistical properties of their pretraining and prompting errors.

### Central claims
1. [PROVED] Perfectly pretrained LLMs implicitly perform Bayesian Model Averaging (BMA) during in-context learning without updating their parameters, computing a posterior distribution over a hidden concept from the prompt examples and predicting the response by aggregating over this posterior. (Proposition 4.1)
2. [PROVED] From an online learning perspective, the ICL regret (prediction error across $T$ prompt examples compared to the true concept) of a perfectly pretrained model decays at a rate of $O(1/T)$. (Corollary 4.2)
3. [PROVED] A specific variant of the attention mechanism mathematically encodes BMA, and standard softmax attention converges to this BMA-encoding attention as the prompt length approaches infinity. (Proposition 4.3)
4. [PROVED] The total variation error of a pretrained language model is bounded by the sum of an approximation error (which decays exponentially as transformer depth increases) and a generalization error (which decays sublinearly with the number of tokens in the pretraining dataset). (Theorem 5.3, Proposition 5.4)
5. [PROVED] The ICL regret of an imperfectly pretrained model is bounded by the sum of the ideal ICL regret ($O(1/T)$) and the pretraining error. (Theorem 6.2)

### Key equations
(1) $\mathbb{P}(r_{t+1} = \cdot \mid \text{prompt}_t) = \int \mathbb{P}(r_{t+1} = \cdot \mid c_{t+1}, S_t, z) \mathbb{P}(z \mid S_t) dz$
Shows that the conditional probability predicted by the LLM is equivalent to Bayesian Model Averaging over the latent concept $z$.

(2) $\text{regret}_t \le \frac{\log(1/P_Z(z^*))}{t}$
The ICL regret bound for a perfectly pretrained model, showing $O(1/t)$ decay based on the prior probability of the true concept $z^*$.

(3) $\Delta_{\text{pre}}(N_p, T_p, \delta) \le \text{Approximation Error} + O \left( \sqrt{\frac{D \cdot \bar{D} + \log(N_p T_p/\delta)}{N_p T_p / t_{mix}}} \right)$
The PAC-Bayes bound on the pretraining error in total variation distance (Theorem 5.3).

### Notation notes
$S_t$: Sequence of the first $t$ ICL examples.
$z$: The hidden concept parameterizing the latent variable model.
$P_Z(z)$: The prior distribution of the hidden concept.
$\Delta_{\text{pre}}$: The pretraining error.
$N_p, T_p$: The number of independent trajectories and tokens per trajectory in the pretraining dataset.

### Method / instrument
Theoretical analysis employing Bayesian statistics, online learning (regret bounds), and statistical learning theory (PAC-Bayes generalization bounds). The authors build a latent variable model for text generation and rigorously analyze both the pretraining phase (via cross-entropy minimization) and the prompting phase (via conditional probability evaluation).

### Scope conditions
Assumes tokens are generated from a latent variable model parameterized by a hidden concept $z \in Z$. The generalization bounds apply to standard transformer architectures (with layer normalization, multi-head attention, and feed-forward networks) under the assumption that the prompt distribution is covered by the pretraining distribution (Assumption 6.1).

### Epistemic status of each central claim
1. PROVED
2. PROVED
3. PROVED
4. PROVED
5. PROVED

### Measured vs assumed
All central claims are mathematically proved as theorems or propositions based on specified structural assumptions (e.g., Assumptions 5.1, 5.2, 6.1) regarding the data generating process and network architecture. There are no empirical measurements (experiments) in the truncated text provided.

### Interventions on inputs
[PARA] The authors analyze what happens when ICL is prompted with "wrong input-output mappings" (perturbed responses). They provide theoretical conditions (distinguishability of concepts) under which ICL remains robust to such perturbations (Proposition 6.3).

### Sensitivity and derivative analysis
[NOT FOUND]

### Relationship between prediction and observed targets
[PARA] Evaluates performance using "ICL regret," measuring the difference in log-likelihood (prediction error) between the model's iterative predictions and the predictions of an oracle possessing the true hidden concept $z^*$.

### Datasets, models, scale
Models: Theoretical abstractions of Transformer architectures (Autoregressive LLMs). No specific datasets are used; the analysis relies on general probability spaces for tokens and latent concepts.

### Limitations stated by the authors
"existing works fail to explain why LLMs [have] the ability for ICL, how the attention mechanism is related to the ICL ability, and how pretraining influences ICL." (Motivating limitations of prior literature, addressed by this paper).

### Open questions named by the authors
[NOT FOUND]

### Notes outside the schema
This paper connects three major aspects of ICL—Bayesian model averaging, architectural induction via attention, and PAC-Bayes generalization—into a single cohesive theoretical framework.

### Importance rating
5. A highly rigorous and comprehensive theoretical unification of pretraining, architecture, and prompting for In-Context Learning.
DEEP PASS RECOMMENDED: no

### Uncertainty flags
[NOT FOUND]



## What Can Transformers Learn In-Context? A Case Study of Simple Function Classes

### Metadata
- **Title as printed in the document:** What Can Transformers Learn In-Context? A Case Study of Simple Function Classes
- **Authors:** Shivam Garg, Dimitris Tsipras, Percy Liang, Gregory Valiant
- **Venue / year:** 2022
- **arXiv or other ID, transcribed exactly as it appears:** [NOT FOUND]
- **Metadata anomalies:** The filename ("What Can Transformers Learn In-Context Case Studies in Learning Function Classes.md") differs slightly from the title ("What Can Transformers Learn In-Context? A Case Study of Simple Function Classes").

### One-paragraph summary
The paper demonstrates that Transformers can be trained from scratch to in-context learn simple function classes, including linear functions, sparse linear functions, two-layer ReLU neural networks, and decision trees. The trained models achieve performance comparable to optimal or task-specific algorithms (e.g., least squares, Lasso, gradient descent on NNs, greedy tree learning) and exhibit robustness to various distribution shifts between training and inference prompts.

### Object of study
Standard decoder-only Transformers (GPT-2 family, 12 layers, 8 heads, 256 embedding dimension) trained from scratch to in-context learn specific function classes via squared error loss.

### Central claims
1. [PARA] Transformers can be trained to in-context learn linear functions, performing comparably to the optimal least squares estimator.
2. [PARA] The in-context learning ability of the trained model is robust to out-of-distribution prompts, such as skewed covariance, noisy linear regression, and domain shifts between in-context examples and query inputs.
3. [PARA] Transformers can in-context learn more complex function classes, including sparse linear functions, two-layer ReLU neural networks, and decision trees, matching or exceeding task-specific algorithms.
4. [PARA] Increasing model capacity improves in-context learning performance and enables learning of higher-dimensional functions and robustness to distribution shifts.

### Key equations
(1) $\epsilon = \mathbb{E}_{f \sim \mathcal{D}_\mathcal{F}, x_i, x_{	ext{query}} \sim \mathcal{D}_\mathcal{X}} [\ell(M((x_1, f(x_1), \dots, x_k, f(x_k), x_{	ext{query}})), f(x_{	ext{query}}))]$
(2) $\min_	heta \mathbb{E}_{f, x_i} [rac{1}{k} \sum_{i=1}^k \ell(M_	heta(P^i), f(x_{i+1}))]$

### Notation notes
$P$: prompt sequence containing in-context examples and query input.
$\mathcal{D}_\mathcal{F}$: distribution over functions.
$\mathcal{D}_\mathcal{X}$: distribution over inputs.

### Method / instrument
Training Transformers from scratch using a curriculum on synthetic prompts generated from specific function classes (linear, sparse linear, neural networks, decision trees). Evaluating the normalized squared error of predictions on new, unseen functions and out-of-distribution prompts compared to baseline algorithms (least squares, Lasso, nearest neighbors, etc.).

### Scope conditions
Function classes are well-defined and low-dimensional (e.g., $d=20$). Inputs are typically drawn from isotropic Gaussian distributions unless shifted during testing.

### Epistemic status of each central claim
1. PROVED
2. SUPPORTED
3. SUPPORTED
4. SUPPORTED

### Measured vs assumed
Measures squared error of model predictions against ground truth and baselines. Assumes the prompt format is sufficient to induce learning algorithms.

### Interventions on inputs
Extensive out-of-distribution testing for linear functions: skewed covariance, low-dimensional subspace, label noise, scaling prompt inputs/weights, restricting in-context examples to one orthant, query input orthogonal to in-context inputs, query input matching an in-context example.

### Sensitivity and derivative analysis
Computes the gradient of the model prediction with respect to the query input, showing it aligns perfectly with the true weight vector $w$ (or its projection) for linear functions.

### Relationship between prediction and observed targets
[INFER] The output prediction maps to the target output of the chosen latent function given the query inputs.

### Datasets, models, scale
Synthetic datasets generated on the fly. Decoder-only Transformer (GPT-2 family, 9.5M parameters, 12 layers, 8 heads, 256 dim). Training for 500k steps with batch size 64.

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
