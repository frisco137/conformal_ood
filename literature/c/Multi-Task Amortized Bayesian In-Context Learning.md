# **Multi-Task Bayesian In-Context Learning** 

**Qingyang Zhu**<sup>1</sup> **Eric Karl Oermann**<sup>1 2</sup> **Kyunghyun Cho**<sup>1</sup> 

## **Abstract** 

Bayesian predictive inference provides a principled framework for uncertainty quantification, data efficiency, and robust generalization. However, exact inference is often intractable, and scalable approximations may remain computationally expensive or require restrictive modeling assumptions that degrade predictive performance. PriorData Fitted and in-context models have recently emerged as an amortized alternative by learning to map datasets directly to predictive distributions, but existing approaches are tightly coupled to the support of the training prior and lack explicit mechanisms for adapting to new priors at test time, resulting in limited robustness under distribution shift. We introduce a multi-task in-context learning framework for amortized hierarchical Bayesian predictive inference that explicitly represents prior information as a prefix of in-context datasets. A transformer trained on sequences of prior and target tasks learns to adapt its predictions across families of priors. On a suite of evaluations with increasing difficulty, including outof-meta-distribution priors and priors with highdimensional latent structures, our method matches oracle Bayesian predictors while being orders of magnitude faster. We further demonstrate its practical relevance on a real-world spatiotemporal temperature prediction benchmark. Code is available at https://github.com/martianmart ina/multi-task-bayesian-icl/. 

## **1. Introduction** 

Bayesian predictive inference provides a principled framework for data-efficient learning, calibrated uncertainty, and robust decision-making by combining prior knowledge with observed evidence. 

1New York University 2NYU Langone Health. Correspondence to: Kyunghyun Cho _<_ kyunghyun.cho@nyu.edu _>_ . 

_Proceedings of the 43_<sup>_rd_</sup> _International Conference on Machine Learning_ , Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s). 

However, computing the posterior predictive distribution (PPD) requires integrating over latent variables and is typically intractable (Blei et al., 2017; MacKay, 1992). Markov chain Monte Carlo (MCMC) methods (Neal, 1996; Andrieu et al., 2003) offer asymptotically exact inference but are often prohibitively slow at test time, especially in high dimensions or with complex likelihoods. Faster approximation alternatives such as (stochastic) variational inference (Blei et al., 2017) optimize an evidence lower bound but introduce bias when the variational family is misspecified. Moreover, both approaches depend on carefully specified generative models which we often lack knowledge of, making performance sensitive to modeling choices and brittle under distribution shift. 

A recent line of work explores amortized posterior and predictive inference by training neural models on simulated data (Edwards & Storkey, 2017; Garnelo et al., 2018; Cranmer et al., 2020; Lueckmann et al., 2021). These models directly map observations or datasets to posterior or predictive distributions, avoiding per-instance optimization or sampling at test time. In-context learning (ICL) (Brown et al., 2020), as an instance of meta-learning (Hochreiter et al., 2001; Finn et al., 2017; Kirsch et al., 2022), extends this paradigm to a substantially larger scale and greater flexibility by representing datasets as sequences and conditioning directly on long contexts of examples (Agarwal et al., 2024). Several works further interpret ICL as performing implicit Bayesian inference (Xie et al., 2022; Akyurek et al.¨ , 2023; Mittal et al., 2025). As instantiations along the direction, Prior-Data Fitted Networks (PFNs) and TabPFNs (Muller¨ et al., 2022; Hollmann et al., 2023) demonstrate that ICL can closely match Bayesian oracles for several task families, suggesting that transformers can implement Bayesian inference in their activations. 

However, a fundamental limitation is shared across these amortized inference approaches. Although the posterior predictive distribution depends on both the prior and the likelihood, existing methods typically predict conditioning only on the evidence dataset. By meta training the model over many datasets sampled from the training prior, that latent distribution is baked into model weights and cannot be modified at test time without retraining or fine-tuning. In many real settings, however, the prior is not fixed. It may vary across users, domains, or environments, and one 

1 

**Multi-Task Bayesian In-Context Learning** 

may wish to explicitly adapt it to encode different beliefs or preferences. Under such prior shifts, predictors trained with a fixed implicit prior lack an explicit mechanism for adaptation, making their out-of-distribution behavior unclear. 

In this work, we introduce a simple mechanism to make amortized in-context prediction Bayesian with controllable priors. We propose _Multi-Task Bayesian In-Context Learning_ , a framework for amortized hierarchical Bayesian predictive inference in which prior information is represented as a prefix of in-context datasets, as illustrated in Figure 1. In contrast to PFNs, where a single prior is implicitly fixed in the model parameters, our construction exposes a direct testtime interface for adaptation to many priors: changing the prefix datasets modifies the induced prior and correspondingly steers the posterior predictive distribution, without any parameter updates. 

We evaluate multi-task Bayesian ICL across increasingly challenging regimes, including both in-meta-distribution and out-of-meta-distribution priors, priors with varying tail-heaviness, high-dimensional structured priors, and realworld environmental data. 

To summarize, our contributions are fourfold: 

- **Flexible Test-Time Adaptation Framework for ICL:** We introduce a framework that represents priors explicitly as prefixes of in-context datasets. This provides a direct interface for controllable adaptation to new priors at test time without any parameter updates. 

- **Hierarchical Bayesian Inference Engine:** We show that our approach serves as a hierarchical Bayesian predictive inference engine. It quantitatively matches oracle Bayesian predictors across a diverse range of task families. 

- **Robust Generalization:** We demonstrate robust generalization under systematically controlled out-ofmeta-distribution (OoMD) prior shifts. We also find the OoMD generalization pattern of our approach is aligned with that of hierarchical Bayesian inference. 

- **Inference Efficiency:** We demonstrate that our method achieves orders-of-magnitude faster inference than classical Bayesian baselines such as MCMC and SVI. 

## **2. In-Context Learning: Preliminaries** 

We refer to _In-Context Learning_ (ICL) as a setting in which a blackbox function approximates the following posterior predictive distribution (PPD): 



For instance, if _y ∈_ R, we are solving a regression problem, 



where 



_F_ is a blackbox function that can handle a variable-length sequence/set, such as a recurrent network or an attentionbased neural network. We use _θ_ = ( _wµ, wσ_ 2 _, ϕ_ ) to refer to all the parameters of this in-context learner. 

We can train the in-context learner, that is, estimating the parameters, with a large number of tasks and associated example sequences: 



where _l_<sup>_k_</sup> is the maximum number of examples we have collected for the _k_ -th task. The loss is then a usual crossentropy loss: 



## **3. How can In-Context Learning be Bayesian?** 

In Bayesian learning, a PPD _p_ ( _yt|xt, Ct−_ 1) can be written down as: 



where _Z_ is a latent variable that needs to be marginalized out. This reflects a generative story in which each input _x_ is given to us, a sample of the latent variable _Z_ is drawn from the prior distribution, and given the pair ( _x, Z_ ), we draw the associated output _y_ from the likelihood distribution. Once we observe a series of _t−_ 1 pairs _Ct−_ 1, we can narrow down the potential values _Z_ can take by inferring the posterior distribution. 

From this perspective, we must specify two ingredients in Bayesian learning. They are the prior _p_ ( _Z_ ) and likelihood distributions _p_ ( _y|x, Z_ ), where we assume the input is independent of _Z_ and is fixed. This is clear from how the posterior distribution _q_ ( _z|Ci_ ) depends on both of these distributions, and if we change either of these, the posterior 

2 

**Multi-Task Bayesian In-Context Learning** 

distribution changes accordingly, and so does the predictive distribution. We express the balance between our own prior belief and observation of data, by appropriately choosing these two distributions. 

A typical definition of in-context learning from the previous section however works directly with the PPD. This implies that this balance between the prior and likelihood is also made arbitrarily through the process of learning from many tasks, and that there is no way for us to directly control this trade-off. This prevents us from fully utilizing the power of in-context learning. In other words, we must be able to introduce an extra knob to in-context learning, in order for us to truly use it as a Bayesian learning algorithm. 

## **4. Multi-Task Bayesian In-Context Learning** 

A major challenge in making in-context learning more Bayesian is the lack of our knowledge of and control over the implicit latent variable _Z_ . This latent variable is implicitly defined and marginalized out as part of the process of in-context learning. Even if we know what this latent variable was, it is unclear what is the best way to represent this latent variable and the prior distribution over it and present it to the in-context learner. 

**Input Representation.** We instead assume that we are provided with an extra set of datasets, where each dataset corresponds to a set of observations drawn from a distinct generative process but that shares the prior distribution. That is, by carefully considering these extra set of datasets, we can infer the prior distribution over the latent variable. More specifically, we assume the availability of _K_ extra datasets: 



where 





By appending and marking these extra datasets, _D_ prior = � _D_<sup>1</sup> _, . . . , D_<sup>_K_�</sup> , at the beginning of _Ct−_ 1 from Eq. (2) as the extra datasets informative of the prior _p_ ( _Z_ ), we can introduce the prior distribution knob into in-context learning. That is, as a desideratum, the Bayesian in-context learner can now be trained to capture 



Changing _D_ prior has the same impact as correspondingly altering the prior _p_ ( _Z_ ) on the right hand side. 



_Figure 1._ We propose a framework for amortized hierarchical Bayesian predictive inference using in-context learning. Tasks are drawn from a meta-distribution over priors rather than a single fixed prior. A transformer adapts to different priors by conditioning on prior datasets within a single context. 

In practice, we would create a long sequence of ( _x, y_ ) tuples, where each consecutive subsequence corresponds to a separate dataset and is separated by neighboring ones with a special token. For instance, a typical prefix for the model to condition on would take the form of 



This construction naturally defines a _multi-task_ episode, in which each dataset of size _M_ corresponds to a distinct _task_ defined by a latent variable _Z_ drawn from a shared prior _p_ ( _Z_ ), with the first _K_ tasks serving as prior tasks and the final task serving as the target task. 

**Model Instantiation.** In this work, the in-context learner _F_ ( _·_ ) is instantiated as a decoder-only transformer. Each token corresponds to an input pair ( _xt, yt−_ 1), represented by concatenating their raw values and projecting them into the model embedding space; special tokens are encoded using special values of _x_ . At each position, the transformer produces a hidden state _ht_ , which is mapped through lightweight linear projection heads to the parameters of the predictive distribution defining _pθ_ ( _yt | ·_ ). 

**Training Objective.** The model is trained to maximize the log-likelihood of target-task observations conditioned on both the target context and the prior datasets. 

Each training episode is generated hierarchically. We first sample episode-level hyperparameters _λ ∼ p_ ( _λ_ ) from a meta-distribution. Conditioned on _λ_ , we sample _K_ + 1 task parameters _{wk}_<sup>_K_</sup> _k_ =1<sup>+1and generate corresponding datasets</sup> _{Dk}_<sup>_K_</sup> _k_ =1<sup>+1.The first</sup><sup>_K_datasets are provided as prior con-</sup> text, and the ( _K_ + 1)-th dataset defines the target task. We 

3 

**Multi-Task Bayesian In-Context Learning** 

provide the fully expanded hierarchical Bayesian predictive expression according to the data generation procedure in Appendix A. 

Let _T_ denote the set of target positions in the input sequence. The training objective minimizes the expected negative loglikelihood _L_ ( _θ_ ) in the following form: 



**Likelihood Instantiation.** We consider two likelihoods with latent task **w** _∈_ R<sup>_d_</sup> : linear regression _y |_ **x** _,_ **w** _∼ N_ ( **w**<sup>_⊤_</sup> **x** _, σ_<sup>2</sup> **I** ) and logistic regression _y |_ **x** _,_ **w** _∼_ Bernoulli( _σ_ ( **w**<sup>_⊤_</sup> **x** )) _,_ where _σ_ ( _·_ ) is the sigmoid function. The linear model has a closed-form PPD when _p_ ( **w** ) is Gaussian, while the logistic model is typically non-conjugate and PPD is intractable. This necessitates an approximate inference algorithm. 

## **5. Experiments** 

We first evaluate whether a multi-task in-context learner can act as an amortized hierarchical Bayesian predictor: (i) when the prior family matches training, (ii) under controlled distribution shift in the prior, and (iii) when the latent generative process becomes high-dimensional and structured. 

|**Sec.**|**Prior family**|**Episode latents**_λ_|**Task latents**|
|---|---|---|---|
|5.2|_N_(_µ_**1**_,_**I**)|_µ_|_{_**w**_k}_<sup>_K_+1</sup><br>_k_=1|
|5.3|StudentT_ν_(_µ_**1**_,_**I**)|_µ, ν_|_{_**w**_k}_<sup>_K_+1</sup><br>_k_=1|
|5.4|[_fA_]#_N_(_µ_**1**_,_**I**)|_µ, A ∈_R<sup>_d×d_</sup>|_{_**z**_k}_<sup>_K_+1</sup><br>_k_=1|



_Table 1._ Summary of per-section experimental setup variation. Each episode contains _K_ prior tasks and one target task; episodelevel latents are shared across tasks. 

### **5.1. Shared Setup** 

**Data.** We define a _prior family_ as a parametric class _{p_ ( **w** _| λ_ ) : _λ ∈_ Λ _}_ , such as a Gaussian distribution with varying mean (section 5.2) or a Student’s _t_ -distribution with both varying mean and varying degrees of freedom (section 5.3). Meta-training induces a _meta-distribution p_ ( _λ_ ) over prior parameters. A test episode is _In-Meta-Distribution (IMD)_ if its _λ_ lies within the support of training _p_ ( _λ_ ), and _Out-of-Meta-Distribution (OoMD)_ otherwise. 

In all experiments, the input dimension is _d_ = 8 and is sampled as **x** _∼N_ ( **0** _,_ **I** ). the output _y_ is scalar, the number of prior tasks is _K_ = 20, and each task contains _M_ = 50 context points. The choice of prior family _p_ ( **w** _| λ_ ) and meta-distribution _p_ ( _λ_ ) varies across experiments (Table 1). For linear regression, we fix the observation noise to _σ_ = 

0 _._ 5. For logistic regression, due to the existence of the sigmoid function, we evaluate models with _µ_ = 0. See detailed explanation in Appendix G.3. 

**Model.** We train a small GPT-2 model<sup>1</sup> with full-causal mask from scratch with Rotary Position Embeddings (Su et al., 2024). We document the hyperparameters and model training details in Appendix C. 

**Predictive Metrics** Since all models are trained by minimizing cross-entropy loss, which in expectation corresponds to minimizing the Kullback-Leibler (KL) divergence to the target predictive distribution, KL is the most directly aligned evaluation metric. We evaluate all models using KL from their predicted PPD to the oracle PPD. When the oracle is intractable, we use an MCMC model given the correct prior to approximate it. Additionally, we report results using Total Variation (TV) divergence in Appendix F, which exhibit consistent qualitative trends with results using KL. 

**Bayesian Oracle and Baseline Models.** We compare neural predictors against four Bayesian reference models that differ along two axes: (i) the inference algorithm, and (ii) the information available at test time, which leads to either standard Bayesian Inference or Hierarchical Bayesian Inference. We summarize the major differences between the reference models in Table 2. We include the full model configurations in the Appendix D. 

**(i) MCMC/SVI** is a privileged reference that is conditioned on the ground truth prior hyperparameters _λ_ used to generate each test episode. It therefore does not need to infer the prior from data, and performs posterior inference only over task-level latents. It models PPD via _p_ ( _y_<sup>_∗_</sup> _|_ **x**<sup>_∗_</sup> _, D_ tgt _, λ_ ) _,_ with the oracle and fixed prior _p_ ( **w** _| λ_ ). 

Such MCMC model when given sufficiently long chains serves as the oracle reference for computing predictive divergences when PPD is intractable as in logistic regression. 

In contrast, SVI relies on a restricted variational family and is not guaranteed to recover the true posterior predictive, even with unlimited optimization. 

**(ii) MCMC-HIER/SVI-HIER** is a more realistic hierarchical Bayesian reference that receives the same inputs as multi-task ICL: _K_ prior datasets and one target dataset. It is explicitly specified with the correct prior family but must infer the episode-level parameter _λ_ from the prior datasets with the same meta-prior _p_ ( _λ_ ) as the distribution used to generate the training data for ICL. It models PPD via _p_ HIER( _y_<sup>_∗_</sup> _|_ **x**<sup>_∗_</sup> _, D_ tgt _, {D_ prior _}_ ) _._ 

**Performance Expectations.** Across all experiments, all four Bayesian exact inference models are instantiated with 

1https://github.com/karpathy/nanoGPT 

4 

**Multi-Task Bayesian In-Context Learning** 

|**Model**|**Observed Data**|**Correct Generative Model Specifcation**|**Access to True****_λ_**|**Asymptotically Exact?**|
|---|---|---|---|---|
|MCMC (Oracle)|_D_tgt|✓|✓|✓|
|SVI|_D_tgt|✓|✓|_×_|
|MCMC-HIER|_{D_prior_}_<sup>_K_</sup><br>_k_=1<sup>_, D_tgt</sup>|✓|_×_|✓|
|SVI-HIER|_{D_prior_}_<sup>_K_</sup><br>_k_=1<sup>_, D_tgt</sup>|✓|_×_|_×_|
|ICL w/ prefx|_{D_prior_}_<sup>_K_</sup><br>_k_=1<sup>_, D_tgt</sup>|_×_|_×_|_×_|
|ICL no prefx|_D_tgt|_×_|_×_|_×_|



_Table 2._ Comparison of Bayesian reference models and ICL models in terms of modeling knowledge and inference properties. All Bayesian references are explicitly specified with the correct probabilistic generative model class. Oracle models are conditioned on the true episode-level prior hyperparameters _λ_ , whereas hierarchical models must infer _λ_ from prior datasets given the same inputs as neural model. MCMC-based methods are asymptotically unbiased given sufficient compute, while variational methods introduce approximation bias since the variational posterior family may be restricted. ICL models have no knowlegde of the true data generation process. Both only observes the evidence of **x** and _y_ . 

the correct generative model for each task, including the functional form of the prior and likelihood. Therefore, the Bayesian inference models are privileged with knowledge of the true data-generating process. They only need to infer the specific latent variables and parameters in each episode from the observed datasets. In contrast, the neural in-context learner observes only input–output pairs ( **x** _, y_ ) and must implicitly learn the structure from data. 

Consequently, when inference is evaluated with sufficiently many posterior samples, the Bayesian models constitute oracle upper bounds on achievable predictive performance under the same observational inputs. In particular, we expect the predictive divergence to satisfy 



when the Bayesian models are correctly specified and models are evaluated in expectation over infinite samples. 

### **5.2. In-Meta-Distribution Hierarchical Bayesian Predictive Inference** 

We first investigate whether the amortized in-context learner can implement _hierarchical Bayesian predictive inference_ when the test prior is IMD, i.e., when the meta-prior over prior parameters is correctly specified. We evaluate this hypothesis along two complementary axes: (i) whether the neural predictor quantitatively matches the oracle posterior predictive distribution, and (ii) whether the model correctly interprets the prefix datasets as _prior information_ . 

For both linear regression and logistic regression, we use the same prior family _p_ ( **w** _| µ_ ) = _N_ ( **w** ; _µ_ **1** _,_ **I** ) and meta-prior _p_ ( _µ_ ) = _U_ ( _−_ 8 _,_ 8). 

### 5.2.1. QUANTITATIVE MATCH: DIVERGENCE FROM BAYESIAN ORACLE 

In the linear setting, shown in Fig. 2, the x-axis varies the number of target-task observations while the prior prefix 



_Figure 2._ KL divergence between the PPDs produced by different inference methods and the oracle PPD for linear regression, evaluated across multiple target context lengths and test priors. 

remains fixed. Multi-task ICL (with prefix) closely matches MCMC-HIER and achieves negligible KL across all context lengths under IMD priors. This indicates the neural model learns a predictive mapping from ( _D_ prior _, D_ tgt) to the posterior predictive distribution quantitatively consistent with that of hierarchical Bayesian inference. Under OoMD priors, multi-task ICL exhibits superior robustness to MCMC-HIER in low-data regimes. We conjecture that this is due to the neural model’s ability to extrapolate in the space of priors. On the other hand, MCMC-HIER is strictly 

5 

#### **Multi-Task Bayesian In-Context Learning** 



<!-- Start of picture text -->
0.5 MCMC 0.6 MCMC<br>MCMC-HIER MCMC-HIER<br>0.4 SVISVI-HIER 0.5 SVISVI-HIER<br>with prefix (mean=0.0027) with prefix (mean=0.0051)<br>0.3 no prefix (mean=0.12) 0.4 no prefix (mean=0.013)<br>0.3<br>0.2<br>0.2<br>0.1 0.1<br>0.0 0.0<br>200 400 600 800 1000 200 400 600 800 1000<br># samples (MCMC) / steps (SVI) # samples (MCMC) / steps (SVI)<br>(a)  Target context length 5. (b)  Target context length 20.<br>KL KL<br><!-- End of picture text -->

_Figure 3._ KL divergence between the PPDs produced by different inference methods and the oracle PPD (approximated from converged MCMC) for logistic regression. The x-axis is the number of posterior samples for MCMC models and the number of optimization steps for SVI models. MCMC models are given 1000 warmup steps prior to collecting any posterior samples. 

constrained to work with the mis-specified prior. In addition, ICL without prefix performs consistently worse than the multi-task ICL, especially under OoMD priors. Without access to the prior prefix, ICL relies on a fixed prior distribution that is baked into its weights and lacks a mechanism for test-time prior adaptation. 

In the case of logistic regression, we also need to examine two regimes separated by the number of observations in the target task, i.e., the evidence. When the number of evidence samples is low, the influence of the choice of the prior is greater. We see this effect in Figure 3 (a) where the ICL without prior cannot match the oracle PPD. The proposed approach of multi-task ICL however can recover the oracle PPD as well as the other baseline approaches. As the number of evidence samples increases, the likelihood increasingly dominates and the effect of the prior diminishes. This allows ICL without prior to eventually match the performance of multi-task ICL, as evident from Figure 3 (b). This behavior is consistent with Bayesian posterior concentration, and highlights that accurate prior inference is most critical in the low-data regime. 

### 5.2.2. MECHANISM MATCH: PRIOR ADAPTABILITY CHECK 

Beyond quantitative agreement, we examine whether the neural model _mechanistically_ interprets the prefix datasets as prior information, rather than ignoring them or treating them as additional target evidence. 

To isolate the effect of the prior prefix, we hold the target prediction problem fixed and vary only the auxiliary prior data. We first sample a target task **w** tgt _∼N_ ( **0** _,_ **I** ), fix its target context _D_ tgt and query inputs **X** query, and then append different prior prefixes sampled from different shifted prior distributions. 

**Qualitative prior adaptability.** Figure 4a illustrates how the distribution of model’s predicted logits _p_ ( **w**<sup>_T_</sup> **x** _| D_ prior) change as the preceding prior prefix is varied. It reveals 



_(a)_ Histograms with different colors show the model’s predicted logit distributions under different prior prefixes. 



_(b)_ KL divergence between multi-task ICL and Bayesian references. Pooled MCMC corresponds to the MCMC that incorrectly treat prior data as evidence for target task. 

_Figure 4._ Prior adaptability check for logistic regression with a fixed target context of length = 5 and different prior prefixes. 

systematic changes in the variance of the distribution of predicted logits as a function of the prior prefix. This demonstrates that the prefix exerts a coherent and controllable influence on the predictive distribution. 

**Ruling out evidence pooling.** A potential alternative explanation is that the model simply pools prior and target data points as if they were generated by a single latent task, rather than interpreting the prefix as prior information. To test this hypothesis, we compare neural predictions against two Bayesian baselines: (i) MCMC-HIER POOL, which assumes that a single latent generates both prior and target datapoints and jointly infers the latent and prior mean, and (ii) MCMC-ORACLE, which is conditioned on the true prior parameters and represents the optimal Bayesian predictor. 

Figure 4b reports KL divergence between neural predictions and these references under prior adaptations. The neural model is significantly closer to MCMC-ORACLE than to MCMC-HIER POOL, indicating that its behavior is not naive evidence pooling and instead aligns with correct Bayesian conditioning on prior information. 

### **5.3. Robustness to OoMD and Increasingly Heavy-Tailed Priors** 

Having established that multi-task ICL recovers hierarchical Bayesian prediction under matched meta-training condi- 

6 

**Multi-Task Bayesian In-Context Learning** 

tions, we next investigate robustness when the test prior becomes OoMD and systematically harder. We focus on controlled shifts in tail-heaviness, where posterior inference becomes statistically unstable and computationally expensive for classical Bayesian methods. We use Student’s _t_ priors over latent tasks, i.e., **w** _∼_ StudentT _ν_ ( _µ_ **1** _,_ **I** ), where smaller degrees of freedom _ν_ induce heavier tails. 

Each result is shown as a heatmap. For all models, the location parameter is sampled per episode as _µ ∼U_ ( _−_ 8 _,_ 8). The tail-heaviness is controlled by a discrete grid of log _ν ∈ {_ 3 _,_ 2 _,_ 1 _,_ 0 _, −_ 1 _, −_ 2 _, −_ 3 _}_ . Each row corresponds to a neural model meta-trained on a finite mixture over degrees of freedom 



where _r_ is the row label and denotes the minimum log _ν_ included in the training mixture. Therefore, lower rows correspond to a broader mixture with increasingly heavytailed training priors. Each column evaluates generalization to a test prior with _µ_ = 0 and a specific log _ν_ (left to right), sweeping both IMD and OoMD regimes. 

For a Student’s _t_ -distribution, the variance is undefined for _ν ≤_ 2 and the mean is undefined for _ν ≤_ 1. Accordingly, the transition from log _ν_ = 1 to log _ν_ = 0 marks a qualitative shift in statistical regularity, while log _ν ≤−_ 1 corresponds to extremely heavy-tailed regimes. This transition is clearly reflected in the heatmap results from Figure 5. 

When the training meta-prior remains relatively narrow and does not include extreme heavy-tailed components ( _ν ≤_ 2), multi-task ICL already exhibits non-trivial OoMD generalization, closely mirroring the behavior of MCMC-HIER. In particular, a model trained only on log _ν_ = 3 generalizes reliably up to test log _ν_ = 1. However, generalization degrades as the test prior enters regimes with undefined variance or mean, indicating a sharp increase in inherent difficulty for inference. 

Importantly, this degradation is not arbitrary, but follows a systematic pattern as the test prior deviates from the training support. In particular, the generalization behavior exhibits a clear _threshold pattern_ : accurate generalization to a given test tail-heaviness requires the training mixture to include sufficiently heavy-tailed components. Empirically, light and moderately heavy tails (log _ν ≥_ 1) are already well covered by training on log _ν_ = 3 alone, whereas entering the regime with undefined moments (log _ν ≤_ 0) requires progressively heavier-tailed training mixtures. This threshold structure is aligned with that of MCMC-HIER: both models display nearly identical heatmap patterns. This indicates that extrapolation beyond the training support is possible but fundamentally bounded. The fact that this alignment persists even in OoMD regimes also provides strong evidence that multi-task ICL has truly learned a mechanism consistent with hierarchical Bayesian predictive inference. 

Finally, when the training mixture includes sufficiently heavy-tailed components (down to log _ν_ = _−_ 2 or _−_ 3), the multi-task ICL achieves near-oracle performance across the full sweep of test priors, from light-tailed Gaussian-like regimes to extremely heavy-tailed distributions. Therefore, the model has learned a generalizable amortized inference mechanism rather than overfitting to a narrow prior family, and that exposure to sufficient prior diversity during metatraining enables robust extrapolation across distributional regimes. 

SVI-HIER exhibits qualitatively different behavior: its performance is largely invariant across rows, indicating that expanding the support of the _ν_ mixture in the meta-prior is not sufficient to improve performance under heavy-tailed priors, even when those priors are IMD. This highlights the intrinsic difficulty of inference in heavy-tailed regimes and underscores that the strong OoMD generalization observed for multi-task ICL is non-trivial and not merely a consequence of broader prior coverage. 

### **5.4. Scaling to Flow-Based Priors with High-Dimensional Parameters** 

While section 5.3 probes robustness under increasing tailheaviness within a low-dimensional parametric family, realworld priors often exhibit substantially richer structure that cannot be captured by a small number of scalar parameters. 

To stress-test whether multi-task ICL can scale beyond textbook prior families, we construct priors by pushing forward a base Gaussian distribution through a normalizing flow (Chen et al., 2018), spiral flow specifically. This induces complex and highly non-Gaussian geometries in the task distribution, parameterized by a dense R<sup>_d×d_</sup> transformation matrix. We train and evaluate ICL across different spiral flow instantiations, each defined by a distinct hidden parameter matrix. This tests generalization across highdimensional prior parameterizations. 



During meta-training, _µ ∼U_ ( _−_ 8 _,_ 8) is sampled per episode. Due to the rotational symmetry of the spiral flow, nonzero means yield qualitatively distinct transformed distributions. Therefore, we fix _µ_ = 4 to induce nontrivial geometric warping at test time. The definition of the spiral flow and visualizations of the induced priors for different _µ_ are provided in Appendix E. We also report evaluations on OoMD _µ_ = 12 in Appendix G.4. 

We report predictive divergence in Figure 6 as a function of wall-clock time, since MCMC exhibits a large per-sample time cost in this setting. To ensure a fair comparison, we configure MCMC-HIER with the minimum number of warmup steps required to reliably reach the oracle performance. Although MCMC-HIER converges quickly after 

7 

**Multi-Task Bayesian In-Context Learning** 







_Figure 5._ Heatmaps of KL divergence to the oracle PPD from ( **left** ) multi-task ICL, ( **middle** ) hierarchical MCMC, and ( **right** ) hierarchical SVI. Rows index the minimum log _ν_ included in the meta-training mixture (increasing tail-heaviness downward), and columns sweep the test prior log _ν_ , covering IMD and OoMD regimes. Lower (darker) values indicate better agreement with the oracle. 

warmup, its runtime remains dominated by warmup and persample costs. In contrast, multi-task Bayesian ICL achieves comparable quality while being orders-of-magnitude faster, demonstrating its scalability to complex flow-based priors. 



_Figure 6._ Predictive KL divergence versus wall-clock inference time for flow-based priors with high-dimensional latents. 

### **5.5. Real-World Evaluation on ERA5** 

We next evaluate whether this prior-prefix mechanism still provides practical benefits in a real-world setting where data is noisy and latent structure is complex and unclear. We use the spatiotemporal temperature prediction task from ERA5 climate data (Store et al., 2024; Hersbach et al., 2023). The model needs to predict surface air temperature from latitude, longitude, time, and elevation, while auxiliary datasets are sampled from the same spatial region but from different non-overlapping time windows. 

We consider two 2019 evaluation protocols. In the _IID split_ , train, validation, and test examples are distinct random samples from the full year of 2019, so validation and test are distributionally matched to training. In the _OOD split_ , training examples are sampled from the first six months of 2019 excluding the final 14 days, validation examples from those final 14 days, and test examples from the last six 

months of 2019, creating a severe seasonal temporal shift. The _2020 Test_ column evaluates the checkpoints selected on the 2019 IID validation set on the full year of 2020. This tests future-year temporal generalization while preserving full seasonal coverage. 

Table 3 demonstrates that prior prefixes are useful for realworld spatiotemporal prediction when the training data cover the relevant seasonal variation. Under the 2019 IID split, MT with _K_ = 2 consistently improves over _K_ = 0 on validation, test, and 2020 Test, showing that auxiliary datasets provide useful local context beyond the target dataset alone. MT also outperforms Set-MT in this regime, suggesting that the sequential prior-prefix model can better exploit structured correlations across datasets when those correlations remain valid at test time. The 2019 OOD split is qualitatively different: training on early-year data and testing on late-year data induces a severe seasonal shift, so correlations that are useful for validation may fail on the test distribution. Indeed, the MT hyperparameter rankings by validation NLL and OOD test NLL are nearly reversed, consistent with domain-generalization observations that in-distribution and out-of-distribution performance can be negatively correlated when models rely on features or correlations that are predictive in training environments but fail under shift (Makino et al., 2025). In this severe regime, the stronger permutation-invariant inductive bias of Set-MT appears to improve robustness by limiting reliance on orderor prefix-specific correlations. Similarly for models with _K_ = 0. Overall, when the training data cover the relevant seasonal structure, prior prefixes consistently improve MTICL and generalize well to the future-year 2020 evaluation. 

Please see Appendix H for full details about this experiment and the implementation of Set-MT model. 

8 

**Multi-Task Bayesian In-Context Learning** 

_Table 3._ Results on ERA5 under IID/OOD splits. Entries are NLL/MSE. MT denotes multi-task ICL, and Set-MT denotes multitask ICL with set aggregation over prior datasets. All entries use the learning rate and checkpoint selected by best validation NLL. 

|Split|Confg|Val_↓_|Test_↓_|2020 Test_↓_|
|---|---|---|---|---|
||MT,_K_=0|-1.72 / .004|-2.02 / .004|-2.00 / .003|
|2019|MT,_K_=2|**-2.29 / .003**|**-2.33 / .003**|**-2.31 / .003**|
|IID|Set-MT,_K_=0|-2.04 / .004|-2.17 / .004|-2.15 / .004|
||Set-MT,_K_=2|-2.16 / .004|-2.18 / .004|-2.17 / .003|
||MT,_K_=0|-1.28 / .007|-0.39 /**.007**|–|
|2019|MT,_K_=2|**-2.06 / .006**|7.13 / 1.254|–|
|OOD|Set-MT,_K_=0|-0.94 / .008|**-0.58**/ .041|–|
||Set-MT,_K_=2|-1.64 / .007|-0.20 / .057|–|



## **6. Related Work** 

**Amortized Inference Through Meta-Learning.** There have been a series of studies in which deep neural networks were trained to imitate or closely approximate computationally intractable inference procedures in probabilistic models. This line of studies include Neural Processes and their variants, Neural Statisticians, (meta-)Simulation-Based Inferences and Prior-Data Fitted Networks (Gordon et al., 2019; Edwards & Storkey, 2017; Bruinsma et al., 2021; Gloeckler et al., 2024; Muller et al.¨ , 2022; Hollmann et al., 2023; Muller et al.¨ , 2025). These approaches all share the characteristics of meta-learning (Hochreiter et al., 2001; Santoro et al., 2016; Finn et al., 2017), in which a set of sample sets are drawn from a meta-distribution over distributions, and a neural network is trained to work with a set of observations and outputs a desirable target distribution, such as a posterior distribution. Until recently, these studies have however been constrained to a relatively small scale, working with only a few tens or hundreds of samples at a time. Furthermore, they have largely focused on handling likelihood via a set of samples and treated the prior implicitly. We identify this latter point as a significant omission in this line of investigation and study the impact of explicitly encoding the prior via a separate set of prior-shared datasets in the context of meta-inference. 

**Prior-Flexible Amortized Inference.** A complementary line of work studies amortized inference models with explicit interfaces for changing the prior at test time. Chang et al. (2025) introduce a general transformer-based conditioning engine that explicitly represents task-relevant latent variables and allows users to provide priors over those latents at inference time as histogram-like distributions. Similarly, Whittle et al. (2026) allow representing priors and posteriors as Gaussian mixture models. In contrast, instead of requiring users to specify priors directly in latent space, we specify prior information in data space through auxiliary in-context datasets. 

**In-Context Learning and Multi-Task Contexts.** It was recently found that large-scale language models, when pretrained on a large amount of the web corpus, exhibits an in-context learning capability (Brown et al., 2020). This observation has led to the realization that LLMs, or the transformers underlying them, are capable of performing sophisticated learning and inference even at a very large scale, such as handling thousands of training examples in a single forward pass (Agarwal et al., 2024). This has led to the recent surge of interest in meta-inference in the context of ICL (Garg et al., 2022; Kirsch et al., 2022; Xie et al., 2022; Akyurek¨ et al., 2023; Von Oswald et al., 2023; Raventos´ et al., 2023; Coda-Forno et al., 2023; Reuter et al., 2025). In these studies, prior distributions were treated implicitly, similarly to earlier meta-inference studies. 

Several recent works have also extended contextconditioned prediction to multi-task or multi-dataset settings. Shen et al. (2023) combine meta-learning and multitask learning in episodic multi-task settings. Ashman et al. (2024) extend Transformer Neural Processes to condition not only on a target context set but also on additional related datasets, using a pseudo-token set-based architecture, and prove the benefit of this related-dataset prefix against a target-only variant. Li et al. (2025) likewise use multiple in-context datasets, but focus on transfer for Bayesian optimization, where the model serves as a surrogate optimized for regret and robustness to negative transfer. In contrast, we evaluate against Bayesian reference posterior predictive distributions directly. 

## **7. Conclusion and Limitations** 

We introduced _Multi-Task Bayesian In-Context Learning_ , a simple framework that bridges the flexibility and scalability of in-context learning with the principled structure of hierarchical Bayesian inference by representing priors explicitly as in-context dataset prefixes, enabling test-time control over the prior. Empirically, our approach matches oracle Bayesian predictors across diverse prior families, generalizes robustly under out-of-meta-distribution shifts, and achieves orders-of-magnitude faster inference than classical Bayesian baselines. We further demonstrate its practical applicability on a real-world environmental benchmark. 

We also acknowledge the limitations of our approach. Multitask ICL might suffer from the attention cost that scales quadratically with sequence length. The in-context multitask setting further increases the computational cost. This architecture also does not explicitly enforce permutation invariance, either within a dataset or across datasets, although posterior predictive inference should not depend on arbitrary orderings of the same evidence. However, empirically, we find the model’s sensitivity to permutations can be diminutive as shown in Appendix G.1. 

9 

**Multi-Task Bayesian In-Context Learning** 

## **Acknowledgements** 

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) with a grant funded by the Ministry of Science and ICT (MSIT) of the Republic of Korea in connection with the Global AI Frontier Lab International Collaborative Research. (No. RS-2024-00469482 & RS-2024-00509257). 

## **Impact Statement** 

This work contributes a general framework for controllable amortized inference in neural sequence models, bridging ideas from in-context learning and hierarchical Bayesian modeling. By enabling explicit representation and adaptation of prior assumptions at test time, the approach supports more flexible, transparent, and robust predictive systems that is computationally efficient. In personalized medicine, for example, electronic health records could be used to provide similar patients as prior context, with our model enabling rapid adaptation of treatment response predictions for a new patient with limited observations in a setting where both speed (clinical time constraints) and calibrated uncertainty (informed consent, risk communication) are essential. Alternatively for drug discovery historical trials could serve as prior context, our method could enable faster identification of valid candidates with reliable confidence estimates—critical for costly downstream study planning. The ability to swap prior contexts without retraining also supports iterative trial designs where the target shifts as projects evolve. 

## **References** 

- Agarwal, R., Singh, A., Zhang, L., Bohnet, B., Rosias, L., Chan, S., Zhang, B., Anand, A., Abbas, Z., Nova, A., Co-Reyes, J. D., Chu, E., Behbahani, F., Faust, A., and Larochelle, H. Many-shot in-context learning. In Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., and Zhang, C. (eds.), _Advances in Neural Information Processing Systems_ , volume 37, pp. 76930– 76966. Curran Associates, Inc., 2024. doi: 10.52202/079 017-2447. 

- Akyurek,¨ E., Schuurmans, D., Andreas, J., Ma, T., and Zhou, D. What learning algorithm is in-context learning? investigations with linear models. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum ?id=0g0X4H8yN4I. 

- Andrieu, C., De Freitas, N., Doucet, A., and Jordan, M. I. An introduction to mcmc for machine learning. _Machine learning_ , 50(1):5–43, 2003. 

- Ashman, M., Diaconu, C., Weller, A., and Turner, R. E. 

In-context in-context learning with transformer neural processes. In Antoran, J. and Naesseth, C. A. (eds.),´ _Proceedings of the 6th Symposium on Advances in Approximate Bayesian Inference_ , volume 253 of _Proceedings of Machine Learning Research_ , pp. 1–29. PMLR, 21 Jul 2024. URL https://proceedings.mlr.pres s/v253/ashman24a.html. 

- Bingham, E., Chen, J. P., Jankowiak, M., Obermeyer, F., Pradhan, N., Karaletsos, T., Singh, R., Szerlip, P., Horsfall, P., and Goodman, N. D. Pyro: Deep universal probabilistic programming. _Journal of machine learning research_ , 20(28):1–6, 2019. 

- Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. Variational inference: A review for statisticians. _Journal of the American Statistical Association_ , 112(518):859–877, Apr 2017. ISSN 1537-274X. doi: 10.1080/01621459.2017.1285773. URL http://dx.doi.org/10.1080/0162145 9.2017.1285773. 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33: 1877–1901, 2020. 

- Bruinsma, W., Requeima, J., Foong, A. Y. K., Gordon, J., and Turner, R. E. The gaussian neural process. In _Third Symposium on Advances in Approximate Bayesian Inference_ , 2021. URL https://openreview.net /forum?id=rzsDn7Vzxf. 

- Chang, P. E., Loka, N. R. B. S., Huang, D., Remes, U., Kaski, S., and Acerbi, L. Amortized probabilistic conditioning for optimization, simulation and inference. In _The 28th International Conference on Artificial Intelligence and Statistics_ , 2025. URL https://openreview.n et/forum?id=3kRGSyp309. 

- Chen, R. T., Rubanova, Y., Bettencourt, J., and Duvenaud, D. K. Neural ordinary differential equations. _Advances in neural information processing systems_ , 31, 2018. 

- Coda-Forno, J., Binz, M., Akata, Z., Botvinick, M., Wang, J. X., and Schulz, E. Meta-in-context learning in large language models. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL https: //openreview.net/forum?id=sx0xpaO0za. 

- Cranmer, K., Brehmer, J., and Louppe, G. The frontier of simulation-based inference. _Proceedings of the National Academy of Sciences_ , 117(48):30055–30062, May 2020. ISSN 1091-6490. doi: 10.1073/pnas.1912789117. URL http://dx.doi.org/10.1073/pnas.191278 9117. 

10 

**Multi-Task Bayesian In-Context Learning** 

- Edwards, H. and Storkey, A. Towards a neural statistician. In _5th International Conference on Learning Representations_ , pp. 1–13, 2017. 

- Finn, C., Abbeel, P., and Levine, S. Model-agnostic metalearning for fast adaptation of deep networks. In _International conference on machine learning_ , pp. 1126–1135. PMLR, 2017. 

- Garg, S., Tsipras, D., Liang, P. S., and Valiant, G. What can transformers learn in-context? a case study of simple function classes. In Koyejo, S., Mohamed, S., Agarwal, 

- A., Belgrave, D., Cho, K., and Oh, A. (eds.), _Advances in Neural Information Processing Systems_ , volume 35, pp. 30583–30598. Curran Associates, Inc., 2022. 

- Garnelo, M., Schwarz, J., Rosenbaum, D., Viola, F., Rezende, D. J., Eslami, S. M. A., and Teh, Y. W. Neural processes. _ArXiv_ , abs/1807.01622, 2018. URL https: //api.semanticscholar.org/CorpusID: 49568863. 

- Gloeckler, M., Deistler, M., Weilbach, C. D., Wood, F., and Macke, J. H. All-in-one simulation-based inference. In _Forty-first International Conference on Machine Learning_ , 2024. URL https://openreview.net/for um?id=DL79HYCFFq. 

- Gordon, J., Bronskill, J., Bauer, M., Nowozin, S., and Turner, R. Meta-learning probabilistic inference for prediction. In _International Conference on Learning Representations_ , 2019. URL https://openreview.net /forum?id=HkxStoC5F7. 

- Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horanyi,´ A., Munoz Sabater, J., Nicolas, J., Peubey, C., Radu, R.,˜ Rozum, I., et al. Era5 hourly data on single levels from 1940 to present. _Copernicus climate change service (c3s) climate data store (cds)_ , 10(1.24381):24381, 2023. 

- Hochreiter, S., Younger, A. S., and Conwell, P. R. Learning to learn using gradient descent. In _International conference on artificial neural networks_ , pp. 87–94. Springer, 2001. 

- Hoffman, M. D., Gelman, A., et al. The no-u-turn sampler: adaptively setting path lengths in hamiltonian monte carlo. _J. Mach. Learn. Res._ , 15(1):1593–1623, 2014. 

- Hollmann, N., Muller, S., Eggensperger, K., and Hutter, F.¨ TabPFN: A transformer that solves small tabular classification problems in a second. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id= cp5PvcI6w8_. 

- Kirsch, L., Harrison, J., Sohl-Dickstein, J., and Metz, L. General-purpose in-context learning by meta-learning transformers. _arXiv preprint arXiv:2212.04458_ , 2022. 

- Li, Y. L., Daulton, S., Muller, S., Wilson, A. G., and Bakshy,¨ E. Robust multi-task modeling for bayesian optimization via in-context learning. In _NeurIPS 2025 Workshop: Reliable ML from Unreliable Data_ , 2025. URL https: //openreview.net/forum?id=iwqJLEPgvF. 

- Lueckmann, J.-M., Boelts, J., Greenberg, D., Goncalves, P., and Macke, J. Benchmarking simulation-based inference. In _International conference on artificial intelligence and statistics_ , pp. 343–351. PMLR, 2021. 

- MacKay, D. J. A practical bayesian framework for backpropagation networks. _Neural computation_ , 4(3):448–472, 1992. 

- Makino, T., Park, J. W., Tagasovska, N., Kudo, T., Coelho, P., Huetter, J.-C., Yao, H., Hoeckendorf, B., Leote, A. C., Ra, S., et al. Supervised contrastive block disentanglement. _arXiv preprint arXiv:2502.07281_ , 2025. 

- Mittal, S., Elmoznino, E., Gagnon, L., Bhardwaj, S., Lajoie, G., and Sridhar, D. Does learning the right latent variables necessarily improve in-context learning? In _Forty-second International Conference on Machine Learning_ , 2025. URL https://openreview.net/forum?id= tuYmhlu62K. 

- Muller, S., Hollmann, N., Arango, S. P., Grabocka, J., and¨ Hutter, F. Transformers can do bayesian inference. In _International Conference on Learning Representations_ , 2022. URL https://openreview.net/forum ?id=KSugKcbNf9. 

- Muller,¨ S., Reuter, A., Hollmann, N., Rugamer,¨ D., and Hutter, F. Position: The future of bayesian prediction is prior-fitted. In _Forty-second International Conference on Machine Learning Position Paper Track_ , 2025. URL https://openreview.net/forum?id=5Hpm 74b1Ga. 

- Neal, R. M. Bayesian leaning for neural networks, 1996. 

- Raventos, A., Paul, M., Chen, F., and Ganguli, S.´ Pretraining task diversity and the emergence of non-bayesian in-context learning for regression. _Advances in neural information processing systems_ , 36:14228–14246, 2023. 

- Reuter, A., Rudner, T. G. J., Fortuin, V., and Rugamer, D.¨ Can transformers learn full bayesian inference in context? In _Forty-second International Conference on Machine Learning_ , 2025. URL https://openreview.net /forum?id=9Ip6fihKbc. 

- Santoro, A., Bartunov, S., Botvinick, M., Wierstra, D., and Lillicrap, T. Meta-learning with memory-augmented neural networks. In _International conference on machine learning_ , pp. 1842–1850. PMLR, 2016. 

11 

**Multi-Task Bayesian In-Context Learning** 

- Shen, J., Zhen, X., Wang, Q., and Worring, M. Episodic multi-task learning with heterogeneous neural processes. _Advances in Neural Information Processing Systems_ , 36: 75214–75228, 2023. 

- Store, C. C. D. et al. Era5 hourly data on single levels from 1940 to present. _Datos recuperados entre noviembre_ , 2024. 

- Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. _Neurocomput._ , 568(C), February 2024. ISSN 0925-2312. doi: 10.1016/j.neucom.2023.127063. URL https://doi.org/10.1016/j.neucom.2023. 127063. 

- Von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., and Vladymyrov, M. Transformers learn in-context by gradient descent. In _International Conference on Machine Learning_ , pp. 35151–35174. PMLR, 2023. 

- Whittle, G., Ziomek, J., Rawling, J., and Osborne, M. A. Distribution transformers: Fast approximate bayesian inference with on-the-fly prior adaptation, 2026. URL https://arxiv.org/abs/2502.02463. 

- Xie, S. M., Raghunathan, A., Liang, P., and Ma, T. An explanation of in-context learning as implicit bayesian inference. In _International Conference on Learning Representations_ , 2022. URL https://openreview.n et/forum?id=RdJVFCHjUMI. 

12 

**Multi-Task Bayesian In-Context Learning** 

## **A. Hierarchical Bayesian Predictive Expression** 

When training the multi-task ICL using negative log-likelihood, we expect the model to learn the following hierarchical Bayesian model as an empirical desideratum reflecting the data generation procedure. 



Here, **w** tgt denotes the latent task parameter for the target task, while **w**<sup>(</sup><sup>_k_)</sup> denotes the latent task parameter for the _k_ -th prior dataset. The variable _λ_ for each input sequence parameterizes the prior distribution _p_ ( **w** _| λ_ ) for that sequence. We sample _λ_ from a fixed meta-prior _p_ ( _λ_ ) during training. The target context is denoted by _Ct−_ 1 = _{_ ( **x** _j, yj_ ) _}_<sup>_t_</sup> _j_<sup>_−_</sup> =1<sup>1,which</sup> contains the observed input-output pairs from the target task before predicting _y∗_ at query input **x** _∗_ . 

## **B. Linear Regression PPD Derivation** 

We consider the case of Bayesian linear regression with the following prior and likelihood distributions: 



where **y** = [ _y_ 1 _, . . . , yN_ ]<sup>_⊤_</sup> , **X** = [ **x**<sup>_⊤_</sup> 1<sup>;</sup><sup>_. . ._;</sup><sup>**x**</sup><sup>_⊤_</sup> _N_<sup>].</sup> 

Since Gaussian is the conjugate prior for the Gaussian likelihood, the posterior is also Gaussian with the following mean and covariance: 



The PPD is then another Gaussian with the following mean and covariance, and clearly a function of prior mean **_µ_** _w_ : 



It is clear that the predictive distribution is a function of the mean **_µ_** _w_ of the prior distribution. When **_µ_** _w_ = 0, we get an important special case of ridge regression: 



## **C. Hyperparameters and Training Details** 

For all the experiments on synthetic data, we use the following hyperparameters for GPT2: hidden dimension 128, feedforward dimension 512, 8 layers, and 8 attention heads, equipped with Rotary Position Embeddings (Su et al., 2024). We train on 10M sequences with 5K validation sequences, using a batch size of 4096 and sweeping the learning rate from 10<sup>_−_4</sup> to 5 _×_ 10<sup>_−_3</sup> . Validation data are sampled from the training distribution with distinct random seeds. Models are trained for up to 100 epochs, and we select the checkpoint with the lowest validation loss. 

13 

**Multi-Task Bayesian In-Context Learning** 

## **D. Bayesian Exact Inference Model Configurations** 

We use Pyro (Bingham et al., 2019) for probabilistic modeling for MCMC with NUTS (Hoffman et al., 2014) and SVI. We summarize the specific model configurations in Table 4 and Table 5. Since we use MCMC to approximate the ground truth PPD, we set a large number of posterior samples to ensure the MCMC model converges to the ground truth PPD. Additionally, in section 5.4, since we are comparing the inference time between inference algorithms, we use minimal warmup steps MCMC need to reflect the minimal inference cost for MCMC. 

|**Model**|**Warmup Steps**|**Num. Posterior Samples**|**Num. Thinning**<br>**Num. **|**Chains**|
|---|---|---|---|---|
|MCMC (oracle)|1000|10000|10|1|
|MCMC-hier|1000|1000|10|1|



_Table 4._ Summary of Bayesian model configurations for MCMC-based methods used in experiments. 

|**Model**|**Num. Posterior Samples**|**Num. Opt Steps**|**Variational Family / LR**|
|---|---|---|---|
|SVI|200|1000|Diagonal Normal /1_×_10<sup>_−_2</sup>|



_Table 5._ Summary of Bayesian model configurations for SVI-based methods used in experiments. 

## **E. Spiral-Flow-Based Priors** 

### **E.1. Definition** 

**Episode-level parameters.** For each episode, we sample a location parameter and a skew-symmetric matrix that defines the spiral flow: 



By construction, _A_ is skew-symmetric and has<sup>_d_</sup><sup><u>(</u></sup><sup>_d_</sup> 2<sup>_−_1)</sup> degrees of freedom. 



where exp( _·_ ) denotes the matrix exponential. 

**Task-level latents and pushforward prior.** Conditioned on the episode parameters ( _µ, A_ ), each task _k_ draws an independent base latent 

which induces the task prior 



14 

**Multi-Task Bayesian In-Context Learning** 

### **E.2. Visualizations** 

Below we show the t-SNE and UMAP (2D) visualizations of _N_ ( **_µ_** _,_ **I** ) samples after a Spiral Flow transformation for different _µ_ . 









_Figure 7._ t-SNE and UMAP (2D) visualizations of _N_ ( **_µ_** _,_ **I** ) samples after a randomly initialized Spiral Flow transformation. The Spiral Flow is rotationally symmetric, therefore a standard Gaussian distribution will stay standard Gaussian after transformation. 

15 

**Multi-Task Bayesian In-Context Learning** 

## **F. Results measured using TV divergence** 

Below we show the results using TV divergence to the ground truth PPD. 



<!-- Start of picture text -->
0.40 0.40<br>MCMC MCMC<br>0.35 MCMCSVI -HIER 0.35 MCMC-HIERSVI<br>SVI-HIER SVI-HIER<br>0.30 with prefix (mean=0.025) 0.30 with prefix (mean=0.032)<br>no prefix (mean=0.16) no prefix (mean=0.047)<br>0.25 0.25<br>0.20 0.20<br>0.15 0.15<br>0.10 0.10<br>0.05 0.05<br>0.00<br>200 400 600 800 1000 200 400 600 800 1000<br># samples (MCMC) / steps (SVI) # samples (MCMC) / steps (SVI)<br>(a)  Target context length 5. (b)  Target context length 20.<br>TV TV<br><!-- End of picture text -->

_Figure 8._ TV divergence between the PPDs produced by different inference methods and the oracle PPD (approximated from converged MCMC) for logistic regression. The x-axis is the number of posterior samples for MCMC models and the number of optimization steps for SVI models. MCMC models are given 1000 warmup steps prior to collecting any posterior samples. 







_Figure 9._ Heatmaps of TV divergence to the oracle PPD from ( **left** ) multi-task ICL, ( **middle** ) hierarchical MCMC, and ( **right** ) hierarchical SVI. Rows index the minimum log _ν_ included in the meta-training mixture (increasing tail-heaviness downward), and columns sweep the test prior log _ν_ , covering IMD and OoMD regimes. Lower (darker) values indicate better agreement with the oracle. 

## **G. Supplementary Analyses** 

### **G.1. Permutation Sensitivity Evaluations.** 

We evaluate permutation sensitivity for prior **w** _∼N_ ( **0** _, I_ ) and target context length 20 in the logistic regression setting (same setup as Figure 3(b)). For each evaluation example, we generate 10 permuted versions of the same prefix and compute: (i) the average pairwise symmetric KL between the resulting model PPDs, using 0 _._ 5�KL( _q_ 1 _∥, q_ 2) + KL( _q_ 2 _∥ q_ 1)�, and (ii) the mean and standard deviation of KL to the oracle across the 10 permutations. We then average these per-example statistics over 60 independently sampled evaluation examples and report mean _±_ SEM in Table 6. We can see that although the decoder-only causal prefix is not exchangeable by construction, our model’s empirical sensitivity to permutations is diminutive under this experiment. Model performance also remains close to the oracle. 

16 

**Multi-Task Bayesian In-Context Learning** 



_Figure 10._ TV divergence versus wall-clock inference time for spiral flow priors with high-dimensional latents. 

_Table 6._ Permutation sensitivity in the logistic regression setting with prior **w** _∼N_ ( **0** _, I_ ) and target context length 20, matching Figure 3(b). Entries are mean _±_ SEM over 60 evaluation examples. 

|Permutation|Oracle KL|Oracle KL Std.|Pairwise Sym. KL|Pairwise Sym. KL Std.|
|---|---|---|---|---|
|Prior dataset order|0_._005_±_0_._004|0_._0005_±_0_._0006|0_._0002_±_0_._0001|0_._0001_±_0_._0002|
|Within-dataset points|0_._005_±_0_._004|0_._0005_±_0_._0005|0_._0001_±_0_._0001|0_._0001_±_0_._0001|



### **G.2. Experiments on Varying the Number of Prior Datasets** _K_ 

### G.2.1. TRAINING ON FIXED _K_ 

To study how model performance depends on the amount of prior evidence, we trained separate models for number of prior tasks _K ∈{_ 1 _,_ 5 _,_ 20 _,_ 30 _}_ and evaluated each model in the corresponding regime. 

We report two quantities. First, to assess predictive quality, we measure KL between the PPD estimated by multi-task ICL and the oracle MCMC PPD. Second, to probe “shrinkage” of prior given more evidence of prior (larger _K_ ), we also measure the variability of the model’s PPD under repeated resampling of the prior prefix while holding the target context fixed. For each test target context and each _K_ , we sample 10 prior prefixes from the same prior, compute the model PPD for each prefix, and report the average pairwise KL divergence among these PPDs. If the model is performing hierarchical inference, this between-prefix variability should decrease with _K_ , since larger collections of prior tasks provide a more concentrated estimate of the shared prior. 

In terms of quality, KL to oracle is already low at _K_ = 1 and remains small across all tested _K_ . More importantly, in terms of hierarchical inference, the between-prefix variability decreases sharply and monotonically with _K_ . Thus, as more prior datasets are provided, the inferred predictive distribution becomes markedly less sensitive to the particular sampled prefix, which is exactly the qualitative behavior expected if the model is using additional prior-task evidence to form a more concentrated estimate of the shared prior. 

17 

**Multi-Task Bayesian In-Context Learning** 

_Table 7._ Sensitivity to the number of prior datasets _K_ . Entries are mean _±_ SEM. 

|_K_|KL to oracle|KL std. across prefxes|Pairwise sym. KL|Pairwise sym. KL std.|
|---|---|---|---|---|
|1|0_._0064_±_0_._00064|0_._0068_±_0_._00092|0_._0074_±_0_._00071|0_._0088_±_0_._00085|
|5|0_._0036_±_0_._00018|0_._0026_±_0_._00023|0_._0036_±_0_._00031|0_._0043_±_0_._00041|
|20|0_._0052_±_0_._00051|0_._0014_±_0_._00015|0_._0014_±_0_._00014|0_._0014_±_0_._00018|
|30|0_._0040_±_0_._00027|0_._00098_±_0_._00010|0_._00090_±_0_._000074|0_._00098_±_0_._000095|



### G.2.2. TRAINING ON VARYING _K_ AND LENGTH EXTRAPOLATION TEST 

In addition, we trained a single multi-task ICL model with the number of prior datasets _K_ sampled during training from _K ∈_ [0 _,_ 10], and evaluated this same model with varying _K_ . 

We test both In-Meta-Distribution (IMD) settings, where _K ∈{_ 1 _,_ 5 _,_ 10 _}_ , and Out-of-Meta-Distribution (OoMD) settings, where _K ∈{_ 15 _,_ 20 _}_ and _K_ is larger than max training _K_ . We focus on OoMD larger _K_ rather than smaller _K_ because smaller- _K_ cases can always be obtained during training by subsampling from examples with larger _K_ , so generalization to fewer prior datasets is naturally covered by the variable- _K_ training setup. In contrast, inference with substantially larger _K_ than seen in training is the more meaningful and difficult test also due to length extrapolation. The rest of the setup matches our previous rebuttal experiment. 

We use the same two metrics as in our previous response: KL to oracle for predictive accuracy and pairwise symmetric KL across prior-prefix resamplings for shrinkage/PPD stability. 

_Table 8._ Comparison across different numbers of prior datasets _K_ . Entries are mean _±_ SEM. 

|Method|_K_ = 1|_K_ = 5|_K_ = 10|_K_ = 15(OoMD)|_K_ = 20(OoMD)|
|---|---|---|---|---|---|
|MCMC-Hier|0_._0071_±_0_._0011|0_._0039_±_0_._00059|0_._0025_±_0_._00024|0_._0022_±_0_._00023|0_._0022_±_0_._00014|
|SVI-Hier|0_._0089_±_0_._0012|0_._0041_±_0_._00040|0_._0032_±_0_._00023|0_._0033_±_0_._00030|0_._0022_±_0_._00013|
|Multi-task ICL|**0**_._**0064**_±_**0**_._**00097**|**0**_._**0027**_±_**0**_._**00037**|**0**_._**0014**_±_**0**_._**00013**|0_._029_±_0_._0028|0_._032_±_0_._0027|



_Table 9._ Sensitivity of multi-task ICL across different numbers of prior datasets _K_ . Entries are mean _±_ SEM. 

|_K_|KL to oracle|Pairwise sym. KL|
|---|---|---|
|1|0_._0062_±_0_._00062|0_._0078_±_0_._00073|
|5|0_._0027_±_0_._00017|0_._0034_±_0_._00030|
|10|0_._0014_±_0_._000072|0_._0016_±_0_._00012|
|15(OoMD)|0_._024_±_0_._0010|0_._026_±_0_._00079|
|20(OoMD)|0_._031_±_0_._0013|0_._035_±_0_._00094|



In the IMD regime, multi-task ICL adapts very well to different amounts of prior evidence: KL to oracle remains low and is even slightly better than MCMC-hier on the test dataset, while pairwise symmetric KL decreases sharply with _K_ , indicating that the model’s PPD becomes less sensitive to the sampled prior prefix as more prior datasets are observed. This is the expected qualitative signature of prior shrinkage under hierarchical inference. 

In the OoMD regime ( _K_ = 15 _,_ 20), performance degrades in both metrics, but the model still yields a reasonably bounded KL. We believe this degradation is driven largely by sequence-length extrapolation: when _K_ at test time is much larger than the maximum seen during training, the total input length, which scales with both _K_ and per-dataset size _M_ , becomes far longer than anything encountered during training, and such length extrapolation remains challenging for standard transformers. 

### **G.3. Logistic Regression Results with Non-Zero Prior Mean.** 

In this section we explain why we fix the mean of prior distribution _µ_ = 0 when using logistic regression likelihood. 

In the logistic model _y |_ **x** _,_ **w** _∼_ Bernoulli( _σ_ ( **w**<sup>_⊤_</sup> **x** )) with **w** _∼N_ ( _µ_ **1** _, I_ ) and **x** _∼N_ ( **0** _, I_ ), for fixed **x** we have **w**<sup>_⊤_</sup> **x** _∼N_ ( _µ_ **1**<sup>_⊤_</sup> **x** _, ∥_ **x** _∥_<sup>2</sup> ). When _µ_ = 0, the logit distribution is centered around 0, so after marginalizing over **w** , the predictive probabilities remain concentrated around the high-uncertainty regime of the sigmoid rather than its saturated tails. 

18 

**Multi-Task Bayesian In-Context Learning** 

As _|µ|_ increases, the logit distribution shifts away from 0, so the sigmoid is more often in a saturated regime and the labels become increasingly close to deterministic. Thus, large _|µ|_ reduces the difficulty of the posterior predictive task in logistic regression. This saturation effect is specific to the logistic likelihood and is the reason we fixed _µ_ = 0 in that setting. 

In Table 10, we additionally evaluated logistic regression over the same range _µ ∈_ [ _−_ 8 _,_ 8] as in the linear regression experiments and the qualitative conclusions remain unchanged for multi-task ICL, although OoMD test prior does have a negative impact. 

The setting is the same as in Figure 3(b) (target context length 20), except that here we additionally vary _µ_ across columns. Table values are KL to oracle (mean ± sem). For the Bayesian baselines, we report the 1000-sample (MCMC) / 1000-step (SVI) results, which correspond to the maximal-compute setting in Figure 3(b) and to their best achieved performance. 

_Table 10._ Comparison of different values of _µ_ under logistic regression. Values are reported as mean _±_ standard error across test samples. All values are scaled by 10<sup>_−_3</sup> . 

|Method|_µ_= 0|_µ_= 4|_µ_= 8|_µ_= 10(OoMD)|
|---|---|---|---|---|
|MCMC|1_._41_±_0_._09|0_._71_±_0_._04|0_._39_±_0_._02|0_._33_±_0_._02|
|MCMC-hier|2_._17_±_0_._14|1_._30_±_0_._10|1_._59_±_0_._29|5_._59_±_0_._30|
|SVI|1_._66_±_0_._07|0_._70_±_0_._03|0_._39_±_0_._01|0_._30_±_0_._02|
|SVI-hier|2_._22_±_0_._13|126_._57_±_10_._92|428_._20_±_28_._67|580_._19_±_35_._70|
|ICL (with prefx)|5_._06_±_0_._55|1_._81_±_0_._30|1_._33_±_0_._23|3_._69_±_0_._27|
|ICL (no prefx)|12_._64_±_2_._01|14_._13_±_2_._96|37_._04_±_2_._91|62_._67_±_3_._42|



### **G.4. OoMD Results for FLow-Based Prior Experiments.** 

To directly evaluate OoMD generalization, we ran an additional experiment at _µ_ = 12, which lies outside the training support. The new result in Table 11 supports the same conclusion as in the in-distribution setting: multi-task ICL remains 

_Table 11._ Comparison of hierarchical Bayesian references and multi-task ICL when testing under OoMD spiral-flow prior with _µ_ = 12. 

|Method|_K_|KL to oracle|Wall-clock time (s)|
|---|---|---|---|
|MCMC-hier|50|0_._264_±_0_._024|1_._30_×_10<sup>3</sup>|
|MCMC-hier|1000|0_._246_±_0_._024|1_._65_×_10<sup>4</sup>|
|SVI-hier|1000|0_._788_±_0_._043|9_._49_×_10<sup>2</sup>|
|Multi-task ICL|N/A|0_._262_±_0_._024|6_._24_×_10<sup>_−_3</sup>|



close to hierarchical MCMC in predictive accuracy even under an OoMD shift in _µ_ , while being several orders of magnitude faster at inference time. 

## **H. ERA5 Details** 

### **H.1. Experiment Details** 

**Dataset and task.** We consider surface air temperature prediction over Central Europe using ERA5 data from 2019 and 2020. The spatial domain covers latitudes in [42<sup>_◦_</sup> _,_ 53<sup>_◦_</sup> ] and longitudes in [8<sup>_◦_</sup> _,_ 28<sup>_◦_</sup> ]. The data have a spatial resolution of 0 _._ 25<sup>_◦_</sup> in both latitude and longitude. We subsample the temporal dimension to retain 6-hourly timestamps, namely 00:00, 06:00, 12:00, and 18:00. Each input coordinate **x** is four-dimensional, consisting of latitude, longitude, time, and elevation. Elevation is computed from the geopotential variable in ERA5. The response _y_ is the standardized 2-meter temperature. Both inputs and outputs are standardized using statistics computed on the corresponding training distribution. 

**Input sequence construction.** Each dataset is constructed by sampling a 10 _×_ 10 latitude-longitude patch, corresponding to a 2 _._ 5<sup>_◦_</sup> _×_ 2 _._ 5<sup>_◦_</sup> spatial region, together with a 3-step temporal window spanning 18 hours. This yields 10 _×_ 10 _×_ 3 = 300 spatiotemporal points per dataset. The target dataset consists of one such 300-point patch. For _K ∈{_ 0 _,_ 2 _}_ , we optionally sample _K_ auxiliary datasets from the same spatial patch but from non-overlapping temporal windows. These auxiliary datasets serve as prior-prefix datasets. The loss is applied only to the 300 target-dataset points. 

19 

**Multi-Task Bayesian In-Context Learning** 



_Figure 11._ Ablation model of multi-task ICL that has the permutation invariance between prior datasets enforced. 

**Splits.** We use 16,000 training episodes, 2,000 validation episodes, and 2,000 test episodes. In the 2019 IID split, train, validation, and test episodes are sampled from the full year of 2019. In the 2019 OOD split, training episodes are sampled from the first six months of 2019 excluding the final 14 days, validation episodes are sampled from those final 14 days, and test episodes are sampled from the last six months of 2019. For the 2020 Test evaluation, we take checkpoints selected on the 2019 IID validation set and evaluate them on episodes sampled from the full year of 2020. 

**Training and model selection.** We train all models with batch size 64 using AdamW with weight decay 10<sup>_−_2</sup> . The learning-rate schedule uses 500 warmup steps followed by cosine decay. We apply gradient clipping with maximum norm 1.0 and enable RoPE. Models are trained for at most 1000 epochs, with early stopping patience 50. Early stopping and checkpoint selection is based on validation negative log-likelihood (NLL). For each model and split, we sweep the learning rate over 



### **H.2. Set-Aggregated Multi-Task ICL** 

We implement a variant of multi-task ICL, which we refer to as Set-MT in Table 3, that treats the collection of prior datasets as an unordered set while preserving the internal order of observations within each dataset. We do not enforce permutation invariance within each dataset because, for the ERA5 task, datapoints inside one dataset are sampled from the same local spatiotemporal patch and are therefore naturally correlated. 

The model architecture is illustrated in Figure 11. Each prior dataset is first processed independently by a shared causal-mask transformer, producing a sequence of hidden embeddings for that prior dataset. We use local positional embeddings within each dataset, resetting the positional indices for every prior dataset. This prevents the model from depending on an arbitrary ordering of the prior datasets. 

The resulting prior-dataset representations are then concatenated and passed through a bidirectional-mask transformer, allowing information to be exchanged across all prior datasets. Because the prior datasets are encoded with shared parameters and local positional embeddings, this block acts as a permutation-equivariant set aggregator over prior datasets. The target dataset is then processed together with the aggregated prior representation using a final hybrid block-structured-mask transformer. In this final block, prior tokens can attend bidirectionally to other prior tokens, while target tokens attend to all prior tokens and only to preceding target tokens. The model therefore conditions target predictions on aggregated prior information without allowing leakage from future target outputs. 

Finally, the model predicts the target responses autoregressively using the same Gaussian likelihood parameterization as MT-ICL. In our ERA5 experiments, we use a 4-layer transformer for each of the blocks described above. 

20 

