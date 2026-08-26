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
