**Can Transformers Learn Full Bayesian Inference In Context?** 

**Arik Reuter**<sup>1</sup> **Tim G. J. Rudner**<sup>2</sup> **Vincent Fortuin**<sup>3 4 5</sup> **David R¨ugamer**<sup>1 5</sup> 

# **Abstract** 

Transformers have emerged as the dominant architecture in the field of deep learning, with a broad range of applications and remarkable in-context learning (ICL) capabilities. While not yet fully understood, ICL has already proved to be an intriguing phenomenon, allowing transformers to learn in context—without requiring further training. In this paper, we further advance the understanding of ICL by demonstrating that transformers can perform full Bayesian inference for commonly used statistical models in context. More specifically, we introduce a general framework that builds on ideas from prior fitted networks and continuous normalizing flows and enables us to infer complex posterior distributions for models such as generalized linear models and latent factor models. Extensive experiments on real-world datasets demonstrate that our ICL approach yields posterior samples that are similar in quality to state-of-the-art MCMC or variational inference methods that do not operate in context. The source code for this paper is available at https://github.com/ArikReuter/ ICL_for_Full_Bayesian_Inference 

tion answering or text summarization, using a fixed model without requiring any gradient-based fine-tuning, simply by referencing the context. Thereby, ICL enables the generation of real-time solutions through a localized understanding of data without explicit re-training (Dong et al., 2022; Garg et al., 2022). 

A fundamental benefit of ICL with LLMs is its versatility. Almost every NLP task involving small data can be solved in context using LLMs, while the performance often surpasses existing baselines (Touvron et al., 2023; OpenAI, 2023; Anil et al., 2023). Additionally, achieving this performance can be very straightforward, requiring only suitably formulated prompts in natural language. Excellent results across a broad variety of tasks, combined with fast inference times and ease of usability, have made in-context learning a machine learning tool employed by millions of people (Eloundou et al., 2023). 

Furthermore, ICL has recently shown remarkable promise for regression and classification tasks involving tabular data, with tabular prior-data fitted networks (TabPFNs) dominating benchmarks alongside minimal prediction time (Hollmann et al., 2022; 2025; Hoo et al., 2024; Robertson et al., 2024). While the internet serves as a suitable source for the massive data needed to train in-context learners on text, TabPFNs demonstrate that training on purely synthetic data facilitates the development of in-context learners for tabular data. 

# **1. Introduction** 

In-context learning (ICL) has become a fundamental principle in natural language processing (NLP) with large language models (LLMs) as ubiquitous in-context learners. The core principle of ICL is that a system adapts to a given task based on information provided in its context. This enables the system to address complex problems, such as ques- 

> 1Department of Statistics, LMU Munich, Munich, Germany<sup>2</sup> Center for Data Science, New York University, New York, USA<sup>3</sup> Department of Computer Science, Technical University of Munich, Munich, Germany 4Helmholtz AI, Munich, Germany. 5Munich Center for Machine Learning (MCML), Munich, Germany. Correspondence to: Arik Reuter _<_ arik.reuter@campus.lmu.de _>_ . 

_Proceedings of the 42_<sup>_nd_</sup> _International Conference on Machine Learning_ , Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). 

While PFNs perform Bayesian inference, they target a univariate, typically discrete, posterior predictive distribution. In numerous applications, however, high-dimensional and continuous posteriors _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> of (latent) variables **_z_** given data **_x_** play a key role.<sup>1</sup> This includes areas such as healthcare (Kyrimi et al., 2021; Abdullah et al., 2022; Etzioni & Kadane, 1995), physics (Gebhard et al., 2025; Brehmer & Cranmer, 2022; Dax et al., 2024), and neuroscience (Lueckmann et al., 2017; Sohn & Narain, 2021). We use the notion of _full Bayesian inference_ for methods yielding potentially complex and high-dimensional posterior distributions—in contrast to, for instance, methods that yield only the pos- 

1 We do not assume any specific form of **_z_** . That is, there can be a single **_z_** _j_ associated with each data point **_x_** _j_ in **_x_** , but the case where a single “global” **_z_** governs the behavior of each **_x_** _j_ in **_x_** is equally included in this notation. 

1 

**Can Transformers Learn Full Bayesian Inference In Context?** 



<!-- Start of picture text -->
“Summary: The<br>fairytale is about...”<br>s 1 s 2 . . .<br>... ... ...<br>... ... ...<br>t 1 t 2 . . . tK− 1 tK x 1 x 2 . . . x K<br>“Summarize this text: Once upon<br>Dataset x<br>a time there was a girl named...”<br><!-- End of picture text -->





<!-- Start of picture text -->
(b)  ICL for full Bayesian inference.<br><!-- End of picture text -->

**(a)** ICL for text summarization using LLMs. 

Figure 1: **(a)** An LLM generates a summary _s_ 1 _, s_ 2 _, . . ._ of a text _t_ 1 _, t_ 2 _, . . . , tK_ through autoregressive sampling while referring to the context using masked self-attention. **(b)** A dataset **_x_** is processed with a transformer encoder. Subsequently, cross attention allows generating samples from the posterior conditioned on **_x_** in context using a diffusion transformer (decoder). The samples are generated by solving a neural differential equation defining a continuous normalizing flow. 

terior predictive or point estimates of the posterior as, for example Hollmann et al. (2022). However, performing full Bayesian inference can be challenging, even for relatively simple models such as generalized linear models (GLMs; Nelder & Wedderburn, 1972). Two common issues when performing full Bayesian inference include (a) slow inference time, particularly when using sampling-based methods (Sommer et al., 2025; 2024), and (b) model misspecification. Although potentially restrictive modeling assumptions are often necessary to make Bayesian inference efficient or even feasible, they can lead to suboptimal predictive performance (Wang & Blei, 2019; Walker, 2013). 

In this paper, we address the following question: _Can we leverage in-context learning to effectively perform full Bayesian inference?_ In doing so, we aim to obtain an incontext learner that can perform the mapping **_x_** _�→ P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> for a specific probabilistic model, and, analogous to LLMs, (a) allows for the rapid generation of samples from a posterior of interest during deployment and (b) can flexibly adapt to a broad range of inputs, thereby overcoming issues arising from model misspecification. More specifically, our approach combines a TabPFN encoder (Hollmann et al., 2022) and a diffusion transformer-decoder (Peebles & Xie, 2023) that is trained via flow matching (Lipman et al., 2022). 

We present the results of our in-context learning approach on extensive real-world and synthetic datasets in Section 4 and discuss the challenges and the transformative potential of in-context learning for full Bayesian inference in Section 5. 

To summarize, our main contributions are as follows: 

1. We develop, train, and examine a model that yields samples from the posterior distribution _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> given data **_x_** as context without any (explicit) parameter updates or parametric assumptions about the posterior. 

2. To achieve this, we propose to use synthetic samples from the joint distribution _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> in order to train a large transformer model that performs ICL regarding the posterior _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> , and provide a general framework to analyze the circumstances that enable learning _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> purely through samples from _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> . 

3. We then analyze the efficacy of our approach for GLMs and latent factor models, namely Gaussian mixture models (GMMs) and factor analysis (FA). For these applications, we show that including the “prior” used for TabPFNs results in reliably inferring posterior distributions on real-world data. 

4. In a variety of experiments, we demonstrate that this approach yields posterior samples that are very similar to those from a Hamiltonian Monte Carlo sampler. Furthermore, we find that the quality of the samples from our ICL approach is preferable, when compared to various popular VI techniques that do not operate in context. 

5. Finally, we conduct ablation studies of our approach, examining, for instance, alternative diffusion objectives and Gaussian approximations in place of flow matching, the model’s performance on out-of-distribution data, and the impact of problem dimensionality. 

2 

**Can Transformers Learn Full Bayesian Inference In Context?** 

# **2. Related Work** 

Beyond the perspective of prior-data fitted networks, the contribution of this work can be summarized from the viewpoints of recent work on in-context learning, amortized Bayesian inference, and, simulation-based inference. 

**In-Context Learning.** ICL is a special case of metalearning (Hospedales et al., 2021) characterized by using a large pre-trained model in order to learn from a context dataset without explicitly updating task-specific parameters. Several recent lines of work investigate the in-context learning capabilities of transformers (Garg et al., 2022; Ahuja et al., 2023; Wang et al., 2024; Chan et al., 2022). 

Garg et al. (2022) show that a model similar to GPT-2 can implicitly implement various interesting function classes in context. More specifically, the model learns to reproduce the predictions of different statistical models such as (sparse) linear functions, decision trees, and even two-layer neural networks. This approach can be extended to multiple families of functions and even mixtures of tasks (Ahuja et al., 2023). Kirsch et al. (2022) investigate ICL as a general principle for meta-learning. However, the results by Garg et al. (2022) and Ahuja et al. (2023) are restricted to relatively small problem scales and scalar-valued predictions instead of multivariate posterior distributions. Additionally, the experiments are conducted exclusively on simulated data. In contrast, our results show that (large) transformer models can effectively learn multivariate posterior distributions over latent variables in context on real-world datasets. Furthermore, the focus on latent-variable models naturally steers our investigation toward unsupervised in-context learning, where the primary objective is to uncover the underlying structure of the data rather than to make predictions based on input-target paris presented in the context. 

Concurrently, Mittal et al. (2025a) conduct a comparative analysis of amortized in-context Bayesian posterior estimation methods, ablating over different optimization objectives and architectural choices, and also reporting results on outof-distribution performance and flow-matching methods. They focus on evaluating the posterior mean and downstream predictive performance, whereas we evaluate full posterior distributions. In an analogous setup, Mittal et al. (2025b) assess the effectiveness of learning point estimates versus learning entire distributions in context for the goal of predictive performance. 

**Amortized Inference.** Amortized inference is a central paradigm in the field of variational inference (Kingma, 2013; Zhai et al., 2018; Kim et al., 2018; Margossian & Blei, 2023). A commonly used idea here is to model the posterior distribution _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> of latent variables **_z_** given a dataset **_x_** via a factorized density _p_ ( **_z_** _|_ **_x_** ) _≈_<sup>�</sup><sup>_K_</sup> _j_ =1<sup>_qθ_(</sup><sup>**_z_**</sup><sup>_j|h_</sup><sup>**_ϕ_**(</sup><sup>**_x_**</sup><sup>_j_)).In</sup> 

contrast to our more general assumption, each datapoint **_x_** _j_ in **_x_** is assumed to have a corresponding latent variable **_z_** _j_ . While the parameter _θ_ determines global aspects of the variational distribution, the function _h_ **_ϕ_** is shared for all **_x_** _j_ and thus amortized across data **_x_** . Variational autoencoders (Kingma, 2013; Rezende et al., 2014) and neural processes (Garnelo et al., 2018a;b; Rudner et al., 2018) are important model classes based on amortized inference. 

In comparison, our ICL approach amortizes its parameters on the level of datasets, such that a single functional relationship is learned for a set _D ⊂_ ( _X × Z_ )<sup>_N_</sup> of datasets. From this point of view, _D_ = _{_ ( **_x_** _i,_ **_z_** _i_ ) _}_<sup>_N_</sup> _i_ =1<sup>comprising</sup><sup>_N_</sup> datasets **_x_** _i ∈X_ and the corresponding latent variables **_z_** _i ∈Z_ can be seen as a “meta-dataset” for which we perform amortized inference. This is similar in nature to the setup by Le et al. (2017), who use recurrent neural networks to “compile” inference based on execution traces of probabilistic programs by training on simulated data. 

Unlike in amortized variational inference, we do not use the notion of an evidence lower bound (Blei et al., 2017) or even the Kullback-Leibler divergence to learn the posterior, but utilize ideas that also appear in the context of simulationbased inference. 

**Simulation-Based Inference.** Analogously to latent variable models, some scientific simulations, for instance in neuroscience or astrophysics (Fan & Markram, 2019; Schmit & Pritchard, 2018), allow to draw samples from the joint distribution _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> of data and latent variable of interest. Amortized posterior inference in this context is referred to as simulation-based inference (SBI; Cranmer et al., 2020). Several recent approaches focus on using neural networks to directly infer aspects of the likelihood _p_ ( **_x_** _|_ **_z_** ), the posterior _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> or the joint distribution _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> . More specifically, techniques based on discrete normalizing flows (Dax et al., 2021) or flow-matching (Wildberger et al., 2024) are used to approximate the posterior _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> , while (Gloeckler et al., 2024) propose to use a transformer-based diffusion model in order to approximate the joint distribution _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> . In recent work, Vetter et al. (2025) directly a pre-trained TabPFN to auto-regressively sample _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> leveraging ICL. 

From a simulation-based inference viewpoint, we demonstrate that sample-based posterior estimation can be used for full Bayesian inference in complex scenarios arising in commonly used latent variable models, and demonstrate the effectiveness of this approach on real-world datasets. 

# **3. In-Context Learning for Full Bayesian Inference** 

Bayesian inference is a tool of central importance for countless applications. However, exact posterior inference can 

3 

**Can Transformers Learn Full Bayesian Inference In Context?** 

become computationally expensive when using samplingbased methods (Hastings, 1970; Hoffman et al., 2014; Betancourt, 2017) and even impossible when relying on fully factorized VI methods, which can incur substantial approximation errors (Bishop et al., 2002; Blei, 2012; Margossian & Blei, 2023). Amortized variational inference can alleviate those issues but typically requires the development of specialized and complex modeling frameworks (Kingma, 2013; Srivastava & Sutton, 2017; Garnelo et al., 2018b; Lin et al., 2021). Another issue with variational inference arises from having to choose a variational distribution. While insufficient flexibility in this respect can lead to overly simplistic posteriors, a too flexible variational distribution might overfit the given data (Cremer et al., 2018). 

We propose a simple and effective solution based on ideas from ICL, which can be seen as conducting amortized inference on a dataset level. Training a model on a potentially unlimited amount of synthetic datasets yields an in-context learner that can not only approximate a vast, almost arbitrarily large, class of distributions but is also highly efficient when used for sampling. Furthermore, this does not incur the same issues with overly or insufficiently flexible distribution assumptions that are present in VI. More specifically, empirical results show that a major strength of TabPFN, for instance, is its ability to adapt flexibly to the complexity of the problem at hand, thus removing the need for extensive hyperparameter tuning (Hollmann et al., 2022). 

In the following, we describe a sufficient general condition, as well as a specific framework that allows to train probabilistic in-context learners on simulated data. 

The idea underlying the proposed approach is founded on two observations relating to full Bayesian inference and the working principle of PFNs: First, many Bayesian models have a generative formulation that allows the simulation of arbitrarily large amounts of training samples from the joint distribution _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> . We assume that samples from _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> comprise a dataset **_x_** = _{_ **_x_** _j}_<sup>_K_</sup> _j_ =1<sup>containing</sup><sup>_K_samples</sup><sup>**_x_**</sup><sup>_j∈X_</sup> and a corresponding (latent) variable **_z_** _∈Z_ .<sup>2</sup> This joint distribution _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> corresponds to the “prior” in PFNs and allows the training of a large neural network that implicitly learns to perform Bayesian inference. Second, Bayesian inference is especially useful for smaller datasets **_x_** that can be processed in a single forward pass. This makes an entire dataset a viable context for Bayesian ICL. 

More specifically, the central goal is to develop a method allowing to infer the posterior distribution _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> of latent variables **_z_** _∈Z_ , given observations **_x_** _∈X_ using ICL. From a supervised-learning perspective, we thus aim to 

2 We do not assume any specific form of **_z_** . That is, there can be a single **_z_** _j_ associated with each data point **_x_** _j_ in **_x_** , but the case where a single “global” **_z_** governs the behavior of each **_x_** _j_ in **_x_** is equally included in this notation. 

directly learn the mapping _f_ 0 : _X →M_ ( _Z_ ) _,_ **_x_** _�→ P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> , where _M_ ( _Z_ ) is the space of all probability measures. Therefore, we want a model _fθ_ ( **_x_** ) = _Q_<sup>**_z_**</sup> _θ_<sup>_|_</sup><sup>**_x_**</sup> for the posterior to be as close as possible to the true posterior _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> = _f_ 0( **_x_** ). We measure “closeness” w.r.t. some divergence _d_ : _M_ ( _Z_ ) _×M_ ( _Z_ ) _→_ [0 _, ∞_ ). When considering the expected divergence over data samples **_x_** _∼ P_<sup>**_x_**</sup> , this gives rise to the following objective: _Rθ_ := E **_x_** _∼p_ ( **_x_** ) [ _d_ ( _fθ_ ( **_x_** ) _, f_ 0( **_x_** ))], which can also be directly expressed as 



Note that we use the notion of a divergence _d_ loosely to refer to any measure of similarity of two distributions. Although _Rθ_ itself is usually intractable, specific choices of _d_ and the use of the joint distribution _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> make Equation (1) accessible via 



where the loss function _Ld_ depends on _d_ and the structure of _Q_<sup>**_z_**</sup> _θ_<sup>_|_</sup><sup>**_x_**</sup> (discussed in detail later). Performing empirical risk minimization for _R∼θ_ with samples from the joint distribution _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> then corresponds to learning to approximate _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> . The model for the posterior _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> is thereby only implicitly defined by the joint distribution _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> . While this requires the ability to sample from _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> , drawing samples from the joint distribution is often a weak requirement in terms of model specification that immediately follows from specifying the generative process of a model. Furthermore, a simple sufficient condition that follows directly from the law of total expectation implies the equivalence of _Rθ_ and _∼ Rθ_ : 



For instance, choosing _d_ to be the forward Kullback-Leibler divergence _d_ KL( _Q_<sup>**_z_**</sup> _θ_<sup>_|_</sup><sup>**_x_**</sup> _, P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> ) = DKL [ _p_ ( _·|_ **_x_** ) _||qθ_ ( _·|_ **_x_** )] im¨ plies that _Ld_ KL( **_x_** _,_ **_z_** _, θ_ ) = _−_ log _qθ_ ( **_z_** _|_ **_x_** ) + _const._ (Muller et al., 2021). In this case, minimizing _R∼θ_ thus directly corresponds to performing maximum likelihood inference on samples from _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> . 

## **3.1. Defining the Form of the Posterior** 

To learn the posterior distribution _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> in context, we use the framework of flow matching (Lipman et al., 2022). More specifically, we utilize continuous normalizing flows (CNFs) to specify and ultimately sample from _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> . CNFs, currently excelling in the field of image synthesis (Esser et al., 2024), do not only allow to flexibly learn almost arbitrary distributions, but are also found to be more sample-efficient 

4 

**Can Transformers Learn Full Bayesian Inference In Context?** 

in training than for instance diffusion objectives (Lipman et al., 2022; Wildberger et al., 2024). Furthermore, unlike discrete normalizing flows (Papamakarios et al., 2021a), CNF objectives do not limit the architecture of the used neural network, allowing to incorporate complex conditioning on the data **_x_** in addition to flexibly modeling the posterior, which is a crucial aspect of our ICL framework. Refer to Appendix D for more information on CNFs. 

Assuming Gaussian conditional probability paths with an optimal-transport mean- and variance-function (Lipman et al., 2022), one obtains the following discrepancy measure _d_ CFM between _Q_<sup>**_z_**</sup> _θ_<sup>_|_</sup><sup>**_x_**</sup> := [ _ψθ,_ 1( _·|_ **_x_** )] _♯PB_ and _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> : 



## 3.1.1. NORMALIZING FLOWS 

The key idea of modeling a distribution _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> with normalizing flows (see, e.g., Papamakarios et al., 2021b), which are the basis of CNFs, is to assume that _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> is the result of “pushing forward” a simple base distribution _PB_ into _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> using a conditional flow _ψθ_ ( _·|_ **_x_** ): 



Therefore, one assumes that samples from _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> are generated by first drawing **_z_**<sup>(0)</sup> _∼ PB_ , and then applying _ψθ_ ( _·|_ **_x_** ), such that _ψθ_ ( **_z_**<sup>(0)</sup> _|_ **_x_** ) _∼ P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> . The base distribution _PB_ is commonly set to be a standard normal distribution, i.e., _PB_ = _N_ (0 _, I_ ). The conditional flow _ψθ_ ( _·|_ **_x_** ) is the object to be learned, such that our model of _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> is defined as _Qθ_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> := [ _ψθ_ ( _·|_ **_x_** )] _♯PB_ . 

## 3.1.2. CONTINUOUS NORMALIZING FLOWS 

In flow matching (Lipman et al., 2022), which we will use to obtain an in-context learner for full Bayesian inference, the normalizing flow _ψθ_ ( _·|_ **_x_** ) is implicitly defined via a (conditional) vector field _vt,_<sup>_θ_</sup> **_x_**<sup>ofanordinarydifferential</sup> equation (ODE): 



where 0 _≤ t ≤_ 1. The first condition _dtd_<sup>_ψθ,t_(</sup><sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**)=</sup> _vt,_<sup>_θ_</sup> **_x_**<sup>(</sup><sup>_ψθ,t_(</sup><sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**))meansthat</sup><sup>_v_</sup> _t,_<sup>_θ_</sup> **_x_**<sup>describesthechangein</sup> _ψθ,t_ ( **_z_** _|_ **_x_** ) at time _t_ , and the second condition _ψθ,_ 0( **_z_** _|_ **_x_** ) = **_z_** implies that initially the flow is just the identity. The family of vector fields _vt,_<sup>_θ_</sup> **_x_**<sup>is parameterized by a neural network</sup> whose parameters _θ_ will be learned. In order to ultimately compute the flow _v_ 1<sup>_θ_</sup> _,_ **_x_**<sup>, that yields</sup><sup>_Q_</sup><sup>**_z_**</sup> _θ_<sup>_|_</sup><sup>**_x_**</sup> = [ _ψθ,_ 1( _·|_ **_x_** )] _♯PB_ , a numerical ODE solver can be used to forward-solve the ODE, which ultimately corresponds to evaluating _ψ_ 1 _,_ **_x_** at a data point **_z_**<sup>(0)</sup> _∼ PB_ . This construction implies very generic assumptions regarding the structure of _Q_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> , which include the existence of a density of the target distribution wrt. the Lebesgue measure, and the assumption that _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> can be represented by a mixture distribution over the marginal probability paths at time point _t_ = 1 (Lipman et al., 2022). Please note that the mathematical understanding of Flow Matching, its properties and assumptions, are still actively researched (Wildberger et al., 2024). 

where the expectation is taken w.r.t. to three random variables: a uniform time-step _t ∼U_ ([0 _,_ 1]), samples from the base distribution **_z_**<sup>(0)</sup> _∼ PB_ , and samples from the ground-truth conditional distribution **_z_**<sup>(1)</sup> _∼ P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> . We define _γt_ ( **_z_**<sup>(1)</sup> _|_ **_z_**<sup>(0)</sup> ) := (1 _− ωt_ ) **_z_**<sup>(0)</sup> + _t_ **_z_**<sup>(1)</sup> . 

We refer to (Wildberger et al., 2024) for mathematical results on the relationship of _d_ CFM and the (forward) KullbackLeibler divergence. The hyperparameter _ω_ = 1 _− σ_ min, where _σ_ min is the variance at time _t_ = 1 in the Gaussian conditional probability paths, appears to have negligible influence when set to a value sufficiently close to one (Lipman et al., 2022).<sup>3</sup> 

In order to make optimizing 



tractable, and thus train our in-context learner, we make use of the sufficient condition in Proposition 1. Thus, the divergence _d_ CFM admits the re-formulation as an objective _R∼θ_ using samples from the joint distribution _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> . We can therefore optimize _R∼θ_ using _N_ independent and identically distributed (i.i.d.) samples _ti ∼U_ ([0 _,_ 1]) from the timedistribution, **_z_** _i_<sup>(0)</sup> _∼ PB_ from the base distribution, and ( **_z_** _i_<sup>(1)</sup> _,_ **_x_** _i_ ) _∼ P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> from the joint distribution. With this, we obtain the following objective function used for the training of the ICL models: 



## **3.2. Sampling from the Joint Distribution** 

In order to learn a model that can perform posterior inference according to Section 3.1, we require to sample ( **_x_** _,_ **_z_** ) _∼ P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> . Given _p_ ( **_x_** _,_ **_z_** ) = _p_ ( **_x_** _|_ **_z_** ) _p_ ( **_z_** ), this is always possible as long as one can draw samples from _P_<sup>**_z_**</sup> and then from _P_<sup>**_x_**</sup><sup>_|_</sup><sup>**_z_**</sup> . Hence, this is a relatively weak requirement allowing for a broad variety of priors and observation models. More specifically, for ICL, we generate a training dataset _D_ which comprises i.i.d. samples _{_ ( **_x_** _i,_ **_z_** _i_ ) _}_<sup>_N_</sup> _i_ =1<sup>re-</sup> sulting from sampling **_z_** _i ∼ P_<sup>**_z_**</sup> and then **_x_** _i ∼ P_<sup>**_x_**</sup><sup>_|_</sup><sup>**_z_**</sup><sup>_i_</sup> . We 

3In our experiments, we follow (Wildberger et al., 2024) and set _ω_ := 1 _−_ 10<sup>_−_4</sup> for all experiments. 

5 

**Can Transformers Learn Full Bayesian Inference In Context?** 

use this simple yet fundamental and very general template to generate samples from the joint _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> for GLMs, factor analysis (FA), and Gaussian mixture models (GMMs) in our later applications. Please refer to Appendix A for more details on the data generating processes. 

## **3.3. The Architecture** 

In order to implement the idea of learning full Bayesian inference in context, we extend ideas of diffusion transformers (Peebles & Xie, 2023), where the conditioning on the time _t_ is implemented via adaptive layer norm (adaLN) blocks initialized as the identity function. As we potentially require complex conditioning on the data **_x_** , an additional transformer encoder is added. The input to the decoder is a vector in the form (1 _− ωt_ ) **_z_**<sup>(0)</sup> + _t_ **_z_**<sup>(1)</sup> , which is treated as a sequence with length one and processed by a transformer decoder without self-attention, but the adaLN blocks. Therefore, the decoder has an equivalent interpretation as a multilayer perceptron with skip-connections, cross-attention, and adaptive layer normalization. For the final processing in the decoder, only conditional feedforward layers with adaptive layer normalization are used. This corresponds exactly to the architecture of the decoder before, albeit without cross attention. We call this part an “MLP with Conditioning”. Samples for the time _t ∈_ [0 _,_ 1] are mapped onto a conditioning vector using several fully connected layers, which yields a richer representation of _t_ that is well-suited as an input to the adaLN blocks. Figure 2 depicts of the resulting architecture. 

## **3.4. Implementing Flow Matching** 

During the training phase, a tuple ( **_z_**<sup>(1)</sup> _,_ **_x_** ) is drawn from the distribution _P_<sup>**_z_**</sup><sup>_,_</sup><sup>**_x_**</sup> . Additionally, a time step _t ∼U_ [0 _,_ 1] and a sample **_z_**<sup>(0)</sup> is drawn from the base distribution _PB_ , which is a standard Gaussian for all our applications. Subsequently, the ground-truth conditional flow _ψ_ ( **_z_**<sup>(0)</sup> _|_ **_x_** ) = (1 _− ωt_ ) **_z_**<sup>(0)</sup> + _t_ **_z_**<sup>(1)</sup> is computed, pushing forward _PB_ into _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> up to time-point _t_ . The transformer encoder processes **_x_** and the decoder takes the representation of the encoder into account in order to output _vt,_<sup>_θ_</sup> **_x_**<sup>(</sup><sup>_ψ_(</sup><sup>**_z_**(0)</sup><sup>_|_</sup><sup>**_x_**)).This output</sup> should match the vector field that describes how the groundtruth flow _ψ_ ( **_z_**<sup>(0)</sup> _|_ **_x_** ) continues at time _t_ . The discrepancy to the ground-truth vector field is measured with the MSE-loss in Equation (7). 

In the sampling phase, we are given **_x_** and the goal is to sample from _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> . To do so, first a vector **_z_**<sup>(0)</sup> _∼ PB_ is drawn. The data **_x_** is passed through the encoder. The decoder defines a function that maps a time-point _t_ and a vector **_ν_** onto a vector field: ( _t,_ **_ν_** ) _�→ vt,_<sup>_θ_</sup> **_x_**<sup>(</sup><sup>**_ν_**) taking</sup><sup>**_x_**into</sup> account. This function is given to an ODE-solver in order to forward-solve the corresponding ODE with boundary conditions 0 _≤ t ≤_ 1. 



<!-- Start of picture text -->
vt, θ x ((1  − ωt ) z (0) +  t z (1))<br>MLP with Conditioning<br>+<br>Scale<br>Feed Forward<br>Scale and Shift<br>Norm<br>Nlayers×<br>+<br>Scale<br>Cross Attention<br>Scale and Shift<br>Encoder Norm<br>MLP<br>x (1  − ωt ) z (0) +  t z (1) t<br><!-- End of picture text -->

Figure 2: Architecture to perform ICL for full Bayesian inference. A relatively large transformer encoder, similar to that in TabPFN (Hollmann et al., 2022) processes a dataset **_x_** and yields a representation used in the decoder. The decoder outputs a vector field defining a flow for a given input vector conditioned on the encoder output and the time. We condition on the time in each cross-attention and each feed-forward block. Please note that the size of the parts of the architecture does not correspond to the number of allocated parameters in this figure. 

# **4. Experiments** 

To show that the proposed methodology is not just an abstract concept, we derive exemplary use cases that demonstrate how well ICL is able to keep up with MCMC and VI approaches in practice. 

For this, we will use two prominent statistical modeling classes, namely generalized linear models (GLMs) and latent factor models. For the latent factor models, we consider factor analysis (FA) and Gaussian mixture models (GMMs). 

**Modeling Scenarios.** We use seven different scenarios for the GLMs, where we vary the prior distribution on the parameters, the conditional distribution of the response, and whether an intercept is included. For FA, we vary the form of the priors and dimensionalities of variables leading to four different scenarios. For the GMMs, we investigate different dimensionalities as well as prior configurations also in four different scenarios. We refer to Appendix A for details on the model structure and scenarios. 

6 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 1: Summarized results for GLMs. Average performance of VI methods and our ICL approach on 50 synthetic and 17 real-world datasets across 7 different GLM scenarios. Comparison to the analytical solution when available and HMC otherwise. Lower is better for all metrics. The best average result is marked in **bold** . 

|**Model**|**Synthetic**<br>|**Re**<br>|**al-Worl**<br>|**d**<br>|
|---|---|---|---|---|
||C2ST MMD _W_2|C2ST|MMD|_W_2|
|LA|1.000 2.770 2.049|1.000|2.091|0.849|
|VI: Diagonal|0.869 1.586 1.742|0.819|0.583|0.529|
|VI: Full|0.714 1.016 1.601|0.668|0.116|0.374|
|VI: Structured|0.711 0.929 1.580|0.664|0.109|**0.370**|
|VI: IAF|0.784 1.648 2.349|0.732|0.516|0.680|
|ICL (ours)|**0.657 0.183 0.556**|**0.648**|**0.090**|0.387|



For our experiments, we train a separate model from scratch for each GLM, GMM, and FA scenario using synthetic samples from the joint distribution _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_x_**</sup> ; i.e. we train seven separate models to cover the GLM scenarios, six separate models for the FA scenarios and four separate models for the GMM scenarios. Please refer to Appendix B for more details. 

**Datasets.** We evaluate the methods on 50 synthetic datasets and 17 real-world datasets from a benchmark suite for tabular regression problems proposed by Grinsztajn et al. (2022). We refer to Appendix C for more details on the preprocessing of the datasets. 

**Methods.** Apart from a comparison with a gold standard, we compare our ICL approach to a Laplace approximation (LA; Daxberger et al., 2021) and different established VI methods based on automatic differentiation VI (Kucukelbir et al., 2017). For the variational distribution, we use a normal distribution with 1) a diagonal and 2) a full covariance matrix, as well as 3) a structured normal distribution with linear dependencies between the latent variables, and 4) an approach based on inverse autoregressive flows (IAF; Kingma et al., 2016). Appendix F includes a discussion regarding the hyperparameters of all considered methods. 

**Evaluation Process.** For every synthetic and real-world dataset, 1000 posterior samples from each method are compared against samples from the analytical solution, if available, or from a Hamiltonian Monte Carlo (HMC) sampler with a NUTS kernel (Hoffman et al., 2014) as the gold standard. If posteriors are unimodal, we run a single chain. In the multimodal case, we use three times the number of modes as the number of Markov chains. 

**Evaluation Metrics.** Three metrics are employed to compare samples from different approximations of the poste- 

Table 2: Results for GLMs. Real-world Evaluation on 17 datasets: Linear regression with a gamma prior on the coefficients _β_ , and an inverse gamma prior on the variance _σ_<sup>2</sup> of the responses (scenario 5). Comparison to HMC samples. All results within two standard errors of the best average result are marked in **bold** . 

|**Model**|C2ST|MMD|_W_2|
|---|---|---|---|
|LA|1.000 (_±_0.000)|1.982(_±_0.126)|0.623 (_±_0.084)|
|VI: Diagonal|0.810 (_±_0.036)|0.441(_±_0.252)|0.384 (_±_0.089)|
|VI: Full|0.711 (_±_0.038)|0.148(_±_0.093)|**0.279**(_±_0.056)|
|VI: Structured|0.705 (_±_0.032)|0.140(_±_0.081)|**0.269**(_±_0.045)|
|VI: IAF|0.777 (_±_0.106)|0.684(_±_0.939)|0.625 (_±_0.525)|
|ICL (ours)|**0.610**(_±_0.045)|**0.046**(_±_0.020)|**0.242**(_±_0.038)|



rior distribution. The first metric is a classifier 2-sample test (C2ST; Lueckmann et al., 2021; Lopez-Paz & Oquab, 2016), where the ROC-AUC score of a random forest classifier, trained to distinguish between samples from the gold standard and the method in question, is utilized. For random forest, we use default hyperparameters, as defined in Scikitlearn (Pedregosa et al., 2011) and 10-fold cross-validation. We use a random forest with the given hyperparameters as a highly performative classifier in order to detect small deviations in distributions, even though this incurs the risk that the C2ST quickly saturates at a value of one, especially in high-dimensional cases. The second metric is the maximum mean discrepancy (MMD) between the two distributions (gold-standard and each tested method) with an exponential kernel (Gretton et al., 2012). The third metric is the empirical Wasserstein-2 distance ( _W_ 2; Givens & Shortt, 1984) of the two distributions, as implemented in the POT library (Flamary et al., 2021). 

## **4.1. Generalized Linear Models** 

Across seven different variants of GLMs, we find that ICL yields samples that have overall the highest agreement with the gold standard (see Table 1). Specifically on the synthetic datasets, the C2ST, MMD and _W_ 2 metrics indicate that the posterior distribution can be approximated more accurately with ICL than via variational inference. 

Particularly in cases where the posterior has a shape deviating from a normal distribution, ICL and HMC agree more closely than VI. For instance, in the case where a gamma prior, i.e. a skewed distribution, is used on the coefficients of a regression model, we find that ICL substantially outperforms VI both on synthetic and real-world data (see Table 2). On the real-world data, ICL still matches the performance of VI methods and has the best (or not significantly worse than the best) performance in terms of C2ST in four out of seven cases (see Table 2). Please refer to Appendix I for the detailed experimental results summarized in Table 1. 

7 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 3: Summarized results for FA. Average performance of VI methods and our ICL approach on 50 synthetic and 17 real-world datasets across 6 different FA scenarios. Comparison to HMC samples. Lower is better for all metrics. The best average result is marked in **bold** . 

|**Model**|**Synthetic**<br>|**Re**|**al-Worl**<br>|**d**<br>|
|---|---|---|---|---|
||C2ST MMD _W_2|C2ST|MMD|_W_2|
|LA|1.000 4.115 2.543|1.000|4.127|0.597|
|VI: Diagonal|0.999 3.321 1.998|0.960|1.220|0.288|
|VI: Full|0.993 3.222 1.955|0.950|1.173|0.281|
|VI: Structured|0.995 3.404 2.079|0.955|1.189|0.283|
|VI: IAF|0.987 3.226 1.973|0.902|0.969|**0.251**|
|ICL (ours)|**0.568 0.057 0.409**|**0.751**|**0.673**|0.583|



## **4.2. Factor analysis** 

On the factor analysis tasks, ICL has notably lower dissimilarity scores compared to the gold standard than all other considered methods in the synthetic evaluation (Table 3). Notably, an average C2ST score of 0.568 is remarkably close to the theoretical lower bound of 0.5. Regarding the real world datasets, C2ST and MMD indicate that our ICL approach yields samples most similar to the reference, while the average _W_ 2 score is substantially higher. We hypothesize that this discrepancy in the metrics might be caused by numerical issues when computing the empirical _W_ 2 distance. Furthermore, the relatively high number of latent variables in comparison to the limited number of data-points can yield overly flexible assumptions on the variational posterior causing the VI methods to overfit. See Appendix I for the detailed experimental results summarized in Table 3. 

## **4.3. Gaussian Mixture Models** 

Full Bayesian inference for GMMs is more challenging than for GLMs or FA. First, the generative process of GMMs involves discrete assignments to clusters, which poses a challenge not only for NUTS, but especially for VI methods. Second, the dimensionality of the posterior samples can be relatively large since for diagonal normal distributions, each component of the mixture has a mean and a variance parameter per dimension. Finally, the considered GMMs are not identifiable leading to multi-modal posterior distributions, which are impossible to perfectly approximate with commonly used VI methods based on Gaussian approximations. 

Due to this inherent difficulty of the GMM scenarios, we find the overall performances of all models to be worse than in the GLM and FA cases. In particular, the C2ST metric is almost saturated for the VI approaches and has a value of around 83 percent for ICL (Table 4). The MMD and _W_ 2 metrics also indicate that ICL yields samples with higher agreement with the reference than the other approaches on synthetic data. A plot of the marginals of the posterior 

Table 4: Summarized Results for GMMs. Average performance of VI methods and our ICL approach on 50 synthetic and 17 real-world datasets across 4 different GMM scenarios. Comparison to HMC samples in all cases. The best average result is marked in **bold** . 

|**Model**<br>**S**<br>|**ynthetic**<br>|**Re**<br>|**al-Wor**|**ld**<br>|
|---|---|---|---|---|
|C2ST|MMD _W_2|C2ST|MMD|_W_2|
|LA<br>1.000|3.916 8.324|1.000|3.385|12.740|
|VI: Diagonal<br>0.994|2.676 7.938|0.992|2.182|11.633|
|VI: Full<br>0.995|2.556 7.947|0.987|2.143|11.696|
|VI: Structured<br>0.994|2.595 7.929|0.988|2.129|11.521|
|VI: IAF<br>0.985|2.308 7.489|0.957|1.845|11.541|
|ICL (ours)<br>**0.825 **|**0.706 4.348**|**0.881**|**1.051 **|**10.691**|



shows high agreement between the posterior distributions of both HMC and ICL while VI is incapable of perfectly approximating a bimodal distribution and exhibits typical mode-seeking behavior (Figure 3). Note that also the VI approach based on inverse autoregressive flows, which in theory allows flexible modeling of a wide range of posterior shapes, fails to learn the bi-modality accurately from the limited number of 50 data points in this GMM scenario. This demonstrates the strength of our ICL approach in flexibly learning distributions agnostic of the provided sample size. Please refer to Appendix I for the detailed experimental results summarized in Table 4. 

## **4.4. Ablations and Further Experimental Results** 

In this subsection, we present various ablations concerning our ICL approach to full Bayesian inference. Due to limited space, most of the results are deferred to the appendix. 

**Alternatives to Flow Matching.** Appendix L contains results from an ablation study using diffusion objectives instead of flow matching, while Appendix K investigates the use of a multivariate Gaussian to parametrize _Q_<sup>**_z_**</sup> _θ_<sup>_|_</sup><sup>**_x_**</sup> . The empirical results from these ablations strongly indicate that flow matching is essential for achieving a close approximation of the gold-standard posterior in the scenarios we consider. 

**Dimensionality.** In addition, we investigate the effect of the dimensionality _K_ of the latent variable **_z_** _∈_ R<sup>_K_</sup> for all seven different GLM scenarios. The key takeaway from our results is that for _K_ = 20 and _K_ = 50, the ICL approach performs comparably to the other methods in terms of sample similarity to HMC, but does not outperform them. Please refer to Appendix O for more details. We hypothesize that a key reason for the failure to detect meaningful differences between the methods in high dimensions is due to the curse of dimensionality affecting our metrics. 

8 

**Can Transformers Learn Full Bayesian Inference In Context?** 



Figure 3: Density plots for the marginals of the posterior for GMM scenario 1. Comparison to HMC samples on a synthetic dataset whose density is depicted as a dotted line. Only the marginals of the first two components of the mean and the variance are shown. The density of the posterior obtained via HMC is depicted as a dotted line. While the ICL method aligns with the gold-standard HMC, the VI methods have a lower level of agreement and exhibit modeseeking behaviour. 

**Out-of-distribution Performance.** We further investigate the robustness of our method under mild distribution shifts in Appendix N. Our results indicate that the performance of our ICL method remains relatively stable for small distribution shifts, but increasingly degrades for a larger gap between the training and testing distribution. 

**Predictive Performance.** Additionally, we evaluate our ICL method and all variational approaches with respect to the predictive performance of the considered GLM setups in Appendix J. Results confirm the strong performance of variational inference methods in terms of point prediction, especially for high dimensionalities, while the ICL method is generally competitive. 

**Architecture.** Appendix M discusses results regarding the effect of using an MLP-based architecture. Our experimental findings confirm that the transformer-based encoder performs significantly better than an equally sized MLP encoder. 

**The Classifier in the C2ST Metric.** Finally, we validate the choice of a random forest classifier for the C2ST metric (Appendix Q). We find that employing a nonlinear neural network and utilizing a random forest yields an overall analogous picture in terms of the performance of all methods. 

# **5. Discussion** 

This paper explores in-context learning for the purpose of full Bayesian inference in latent variable models. We propose to use conditional flow matching as a generic and flexible framework to approximate posterior distributions and an architecture that utilizes a transformer encoder for potentially complex conditioning on the data. We find that our ICL approach yields a closer approximation of the posterior than several state-of-the-art variational inference methods across different datasets and model setups. This does not only hold for synthetic, but also real-world tabular datasets. 

**Limitations.** While our experiments indicate the effectiveness of ICL as a Bayesian inference method, it requires an extensive up-front training routine on modern GPU hardware. Despite ICL being consistently faster at inference time than the considered HMC methods, the overall computational burden to train our approach is much higher. 

Furthermore, the goal of this work is to show that ICL can effectively learn full Bayesian inference. Our experiments therefore focus on relatively simple posterior distributions where we can compare against established methods, such as HMC. Additionally, increased dimensionality of the problems considered poses a challenge to both the ICL method and the metrics we employ. Further, as with many other ICL approaches, large datasets as a context can become computationally very expensive. 

**Outlook and Future Work.** Despite its vast up-front computational cost, ICL has not only proven fundamentally transformative in the field of NLP (Brown et al., 2020; Touvron et al., 2023), but has recently started to transform the field of tabular machine learning (Hollmann et al., 2022). Exploring the frontiers of ICL in terms of full Bayesian inference, starting from the feasibility results of this work, might therefore lead to similarly fertile territories. 

Although ICL performs well even when trained on data that may differ from real-world distributions, its flexibility is limited by the structure of the training data. If the synthetic data is highly unrealistic, ICL may fail— much like any model with a misspecified hypothesis space that imposes an unsuitable inductive bias. 

While flexible state-of-the-art sampling-based methods, such as HMC, serve as an efficient and highly effective reference in terms of inference for standard and statistical methods discussed in this paper, the proposed ICL approach is fundamentally more general in nature. In particular, any probabilistic model for which a generative process is conceivable can be fitted using our ICL approach—the potential for fitting models beyond the horizon of standard Bayesian methods is therefore manifold. 

9 

**Can Transformers Learn Full Bayesian Inference In Context?** 

# **Impact Statement** 

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. 

# **Acknowledgements** 

We thank Beste Aydemir and Svea Reuter for their valuable support and insightful comments. VF was supported by the Branco Weiss Fellowship. DR’s research is funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 548823575. We also gratefully acknowledge funding provided to AR by the Munich Center for Machine Learning (MCML). 

# **References** 

- Abdullah, A. A., Hassan, M. M., and Mustafa, Y. T. A review on bayesian deep learning in healthcare: Applications and challenges. _IEEE Access_ , 10:36538–36562, 2022. 

- Ahuja, K., Panwar, M., and Goyal, N. In-context learning through the bayesian prism. _arXiv preprint arXiv:2306.04891_ , 2023. 

- Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican, K., et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

- Betancourt, M. A conceptual introduction to hamiltonian monte carlo. _arXiv preprint arXiv:1701.02434_ , 2017. 

- Bingham, E., Chen, J. P., Jankowiak, M., Obermeyer, F., Pradhan, N., Karaletsos, T., Singh, R., Szerlip, P., Horsfall, P., and Goodman, N. D. Pyro: Deep universal probabilistic programming. _Journal of machine learning research_ , 20(28):1–6, 2019. 

- Bishop, C., Spiegelhalter, D., and Winn, J. Vibes: A variational inference engine for bayesian networks. _Advances in neural information processing systems_ , 15, 2002. 

- Blei, D. M. Probabilistic topic models. _Communications of the ACM_ , 55(4):77–84, 2012. 

- Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. Variational inference: A review for statisticians. _Journal of the American statistical Association_ , 112(518):859–877, 2017. 

- Brehmer, J. and Cranmer, K. Simulation-based inference methods for particle physics. In _Artificial Intelligence for High Energy Physics_ , pp. 579–611. World Scientific, 2022. 

- Brosse, N., Durmus, A., and Moulines, E. The promises and pitfalls of stochastic gradient langevin dynamics. _Advances in Neural Information Processing Systems_ , 31, 2018. 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33: 1877–1901, 2020. 

- Chan, S. C., Dasgupta, I., Kim, J., Kumaran, D., Lampinen, A. K., and Hill, F. Transformers generalize differently from information stored in context vs in weights. _arXiv preprint arXiv:2210.05675_ , 2022. 

- Chen, R. T. Q. torchdiffeq, 2018. URL https:// github.com/rtqichen/torchdiffeq. 

- Chen, T., Fox, E., and Guestrin, C. Stochastic gradient hamiltonian monte carlo. In _International conference on machine learning_ , pp. 1683–1691. PMLR, 2014. 

- Cranmer, K., Brehmer, J., and Louppe, G. The frontier of simulation-based inference. _Proceedings of the National Academy of Sciences_ , 117(48):30055–30062, 2020. 

- Cremer, C., Li, X., and Duvenaud, D. Inference suboptimality in variational autoencoders. In _International conference on machine learning_ , pp. 1078–1086. PMLR, 2018. 

- Dao, Q., Phung, H., Nguyen, B., and Tran, A. Flow matching in latent space. _arXiv preprint arXiv:2307.08698_ , 2023. 

- Dax, M., Green, S. R., Gair, J., Macke, J. H., Buonanno, A., and Scholkopf, B.¨ Real-time gravitational wave science with neural posterior estimation. _Physical review letters_ , 127(24):241103, 2021. 

- Dax, M., Green, S. R., Gair, J., Gupte, N., Purrer, M., Ray-¨ mond, V., Wildberger, J., Macke, J. H., Buonanno, A., and Scholkopf, B. Real-time gravitational-wave inference¨ for binary neutron stars using machine learning. _arXiv preprint arXiv:2407.09602_ , 2024. 

- Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., and Hennig, P. Laplace redux-effortless bayesian deep learning. _Advances in Neural Information Processing Systems_ , 34:20089–20103, 2021. 

- Dong, Q., Li, L., Dai, D., Zheng, C., Wu, Z., Chang, B., Sun, X., Xu, J., and Sui, Z. A survey on in-context learning. _arXiv preprint arXiv:2301.00234_ , 2022. 

- Dormand, J. R. and Prince, P. J. A family of embedded runge-kutta formulae. _Journal of computational and applied mathematics_ , 6(1):19–26, 1980. 

10 

**Can Transformers Learn Full Bayesian Inference In Context?** 

- Eloundou, T., Manning, S., Mishkin, P., and Rock, D. Gpts are gpts: An early look at the labor market impact potential of large language models. _arXiv preprint arXiv:2303.10130_ , 2023. 

- Esser, P., Kulal, S., Blattmann, A., Entezari, R., Muller, J.,¨ Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In _Forty-first International Conference on Machine Learning_ , 2024. 

- Etzioni, R. D. and Kadane, J. B. Bayesian statistical methods in public health and medicine. _Annual review of public health_ , 16(1):23–41, 1995. 

- Fahrmeir, L., Kneib, T., Lang, S., Marx, B., Fahrmeir, L., Kneib, T., Lang, S., and Marx, B. _Regression models_ . Springer, 2013. 

- Fan, X. and Markram, H. A brief history of simulation neuroscience. _Frontiers in neuroinformatics_ , 13:32, 2019. 

- Flamary, R., Courty, N., Gramfort, A., Alaya, M. Z., Boisbunon, A., Chambon, S., Chapel, L., Corenflos, A., Fatras, K., Fournier, N., et al. Pot: Python optimal transport. _Journal of Machine Learning Research_ , 22(78):1–8, 2021. 

- Garg, S., Tsipras, D., Liang, P. S., and Valiant, G. What can transformers learn in-context? a case study of simple function classes. _Advances in Neural Information Processing Systems_ , 35:30583–30598, 2022. 

- Garnelo, M., Rosenbaum, D., Maddison, C., Ramalho, T., Saxton, D., Shanahan, M., Teh, Y. W., Rezende, D., and Eslami, S. A. Conditional neural processes. In _International conference on machine learning_ , pp. 1704–1713. PMLR, 2018a. 

- Garnelo, M., Schwarz, J., Rosenbaum, D., Viola, F., Rezende, D. J., Eslami, S., and Teh, Y. W. Neural processes. _arXiv preprint arXiv:1807.01622_ , 2018b. 

- Gebhard, T. D., Wildberger, J., Dax, M., Kofler, A., Angerhausen, D., Quanz, S. P., and Scholkopf, B.¨ Flow matching for atmospheric retrieval of exoplanets: Where reliability meets adaptive noise levels. _Astronomy & Astrophysics_ , 693:A42, 2025. 

- Givens, C. R. and Shortt, R. M. A class of wasserstein metrics for probability distributions. _Michigan Mathematical Journal_ , 31(2):231–240, 1984. 

- Gloeckler, M., Deistler, M., Weilbach, C., Wood, F., and Macke, J. H. All-in-one simulation-based inference. _arXiv preprint arXiv:2404.09636_ , 2024. 

- Gretton, A., Borgwardt, K. M., Rasch, M. J., Scholkopf, B.,¨ and Smola, A. A kernel two-sample test. _The Journal of Machine Learning Research_ , 13(1):723–773, 2012. 

- Grinsztajn, L., Oyallon, E., and Varoquaux, G. Why do treebased models still outperform deep learning on typical tabular data? _Advances in neural information processing systems_ , 35:507–520, 2022. 

- Grunwald, P. and van Ommen, T.¨ Inconsistency of bayesian inference for misspecified linear models, and a proposal for repairing it. _Bayesian Analysis_ , 12(4):1069–1103, 2017. 

- Hastings, W. Monte carlo sampling methods using markov chains and their applications. _Biometrika_ , 57(1):97–109, 1970. 

- Hoffman, M. D., Gelman, A., et al. The no-u-turn sampler: adaptively setting path lengths in hamiltonian monte carlo. _J. Mach. Learn. Res._ , 15(1):1593–1623, 2014. 

- Hollmann, N., Muller,¨ S., Eggensperger, K., and Hutter, F. Tabpfn: A transformer that solves small tabular classification problems in a second. _arXiv preprint arXiv:2207.01848_ , 2022. 

- Hollmann, N., Muller, S., Purucker, L., Krishnakumar, A.,¨ Korfer, M., Hoo, S. B., Schirrmeister, R. T., and Hutter,¨ F. Accurate predictions on small data with a tabular foundation model. _Nature_ , 637(8045):319–326, 2025. 

- Hoo, S. B., Muller, S., Salinas, D., and Hutter, F. The tabular¨ foundation model tabpfn outperforms specialized time series forecasting models based on simple features. In _NeurIPS Workshop on Time Series in the Age of Large Models_ , 2024. 

- Hospedales, T., Antoniou, A., Micaelli, P., and Storkey, A. Meta-learning in neural networks: A survey. _IEEE transactions on pattern analysis and machine intelligence_ , 44(9):5149–5169, 2021. 

- Ioffe, S. Batch normalization: Accelerating deep network training by reducing internal covariate shift. _arXiv preprint arXiv:1502.03167_ , 2015. 

- Izmailov, P., Vikram, S., Hoffman, M. D., and Wilson, A. G. G. What are bayesian neural network posteriors really like? In _International conference on machine learning_ , pp. 4629–4640. PMLR, 2021. 

- Kim, Y., Wiseman, S., Miller, A., Sontag, D., and Rush, A. Semi-amortized variational autoencoders. In _International Conference on Machine Learning_ , pp. 2678–2687. PMLR, 2018. 

- Kingma, D. P. Auto-encoding variational bayes. _arXiv preprint arXiv:1312.6114_ , 2013. 

11 

**Can Transformers Learn Full Bayesian Inference In Context?** 

- Kingma, D. P. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I., and Welling, M. Improved variational inference with inverse autoregressive flow. _Advances in neural information processing systems_ , 29, 2016. 

- Kirsch, L., Harrison, J., Sohl-Dickstein, J., and Metz, L. General-purpose in-context learning by meta-learning transformers. _arXiv preprint arXiv:2212.04458_ , 2022. 

- Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., and Blei, D. M. Automatic differentiation variational inference. _Journal of machine learning research_ , 18(14):1–45, 2017. 

- Kyrimi, E., McLachlan, S., Dube, K., Neves, M. R., Fahmi, A., and Fenton, N. A comprehensive scoping review of bayesian networks in healthcare: Past, present and future. _Artificial Intelligence in Medicine_ , 117:102108, 2021. 

- Lawley, D. N. and Maxwell, A. E. Factor analysis as a statistical method. _Journal of the Royal Statistical Society. Series D (The Statistician)_ , 12(3):209–229, 1962. 

- Le, T. A., Baydin, A. G., and Wood, F. Inference compilation and universal probabilistic programming. In _Artificial Intelligence and Statistics_ , pp. 1338–1348. PMLR, 2017. 

- Li, C., Chen, C., Carlson, D., and Carin, L. Preconditioned stochastic gradient langevin dynamics for deep neural networks. In _Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence_ , pp. 1788–1794, 2016. 

- Lienen, M., Kollovieh, M., and Gunnemann, S.¨ Generative modeling with bayesian sample inference. _arXiv preprint arXiv:2502.07580_ , 2025. 

- Lin, X., Wu, J., Zhou, C., Pan, S., Cao, Y., and Wang, B. Task-adaptive neural process for user cold-start recommendation. In _Proceedings of the Web Conference 2021_ , pp. 1306–1316, 2021. 

- Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. _arXiv preprint arXiv:2210.02747_ , 2022. 

- Lopes, H. F. and West, M. Bayesian model assessment in factor analysis. _Statistica Sinica_ , pp. 41–67, 2004. 

- Lopez-Paz, D. and Oquab, M. Revisiting classifier twosample tests. _arXiv preprint arXiv:1610.06545_ , 2016. 

- Loshchilov, I. and Hutter, F. Sgdr: Stochastic gradient descent with warm restarts. _arXiv preprint arXiv:1608.03983_ , 2016. 

- Lueckmann, J.-M., Goncalves, P. J., Bassetto, G., Ocal, K.,<sup>¨</sup> Nonnenmacher, M., and Macke, J. H. Flexible statistical inference for mechanistic models of neural dynamics. _Advances in neural information processing systems_ , 30, 2017. 

- Lueckmann, J.-M., Boelts, J., Greenberg, D., Goncalves, P., and Macke, J. Benchmarking simulation-based inference. In _International conference on artificial intelligence and statistics_ , pp. 343–351. PMLR, 2021. 

- Mangoubi, O. and Vishnoi, N. K. Nonconvex sampling with the metropolis-adjusted langevin algorithm. In _Conference on learning theory_ , pp. 2259–2293. PMLR, 2019. 

- Margossian, C. C. and Blei, D. M. Amortized variational inference: When and why? _arXiv preprint arXiv:2307.11018_ , 2023. 

- Mittal, S., Bengio, Y., Malkin, N., and Lajoie, G. In-context parametric inference: Point or distribution estimators? _arXiv preprint arXiv:2502.11617_ , 2025a. 

- Mittal, S., Bracher, N. L., Lajoie, G., Jaini, P., and Brubaker, M. Amortized in-context bayesian posterior estimation. _arXiv preprint arXiv:2502.06601_ , 2025b. 

- Muller, S., Hollmann, N., Arango, S. P., Grabocka, J., and¨ Hutter, F. Transformers can do bayesian-inference by meta-learning on prior-data. In _Fifth Workshop on MetaLearning at the Conference on Neural Information Processing Systems_ , 2021. 

- Murphy, K. P. _Probabilistic machine learning: Advanced topics_ . MIT press, 2023. 

- Nelder, J. A. and Wedderburn, R. W. Generalized linear models. _Journal of the Royal Statistical Society Series A: Statistics in Society_ , 135(3):370–384, 1972. 

- OpenAI. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., and Lakshminarayanan, B. Normalizing flows for probabilistic modeling and inference. _Journal of Machine Learning Research_ , 22(57):1–64, 2021a. 

- Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., and Lakshminarayanan, B. Normalizing flows for probabilistic modeling and inference. _Journal of Machine Learning Research_ , 22(57):1–64, 2021b. 

- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., et al. Scikit-learn: Machine learning in python. _the Journal of machine Learning research_ , 12:2825–2830, 2011. 

12 

**Can Transformers Learn Full Bayesian Inference In Context?** 

- Peebles, W. and Xie, S. Scalable diffusion models with transformers. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 4195–4205, 2023. 

- Phan, D., Pradhan, N., and Jankowiak, M. Composable effects for flexible and accelerated probabilistic programming in numpyro. _arXiv preprint arXiv:1912.11554_ , 2019. 

- Rezende, D. J., Mohamed, S., and Wierstra, D. Stochastic backpropagation and approximate inference in deep generative models. In _International conference on machine learning_ , pp. 1278–1286. PMLR, 2014. 

- Robertson, J., Hollmann, N., Awad, N., and Hutter, F. Fairpfn: Transformers can do counterfactual fairness. In _ICML 2024 Next Generation of AI Safety Workshop_ , 2024. 

- Rudner, T. G., Fortuin, V., Teh, Y. W., and Gal, Y. On the connection between neural processes and gaussian processes with deep kernels. In _Workshop on Bayesian Deep Learning, NeurIPS_ , pp. 14, 2018. 

- Rummel, R. J. _Applied factor analysis_ . Northwestern University Press, 1988. 

- Sahoo, S., Gokaslan, A., De Sa, C. M., and Kuleshov, V. Diffusion models with learned adaptive noise. _Advances in Neural Information Processing Systems_ , 37:105730– 105779, 2024. 

- Salazar, S. Vart: variational regression trees. _Advances in Neural Information Processing Systems_ , 36:45681– 45693, 2023. 

- Schmit, C. J. and Pritchard, J. R. Emulation of reionization simulations for bayesian inference of astrophysics parameters using neural networks. _Monthly Notices of the Royal Astronomical Society_ , 475(1):1213–1223, 2018. 

- Sohn, H. and Narain, D. Neural implementations of bayesian inference. _Current Opinion in Neurobiology_ , 70: 121–129, 2021. 

- Sommer, E., Wimmer, L., Papamarkou, T., Bothmann, L., Bischl, B., and Rugamer,¨ D. Connecting the dots: Is mode-connectedness the key to feasible sample-based inference in bayesian neural networks? In _Proceedings of the 41st International Conference on Machine Learning_ . PMLR, 2024. 

- Sommer, E., Robnik, J., Nozadze, G., Seljak, U., and Rugamer, D.¨ Microcanonical Langevin Ensembles: Advancing the Sampling of Bayesian Neural Networks. In _The Thirteenth International Conference on Learning Representations_ , 2025. 

- Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution. _Advances in neural information processing systems_ , 32, 2019. 

- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. _arXiv preprint arXiv:2011.13456_ , 2020. 

Srivastava, A. and Sutton, C. Autoencoding variational inference for topic models. _arXiv preprint arXiv:1703.01488_ , 2017. 

- Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

- Vetter, J., Gloeckler, M., Gedon, D., and Macke, J. H. Effortless, simulation-efficient bayesian inference using tabular foundation models. _arXiv preprint arXiv:2504.17660_ , 2025. 

- Walker, S. G. Bayesian inference with misspecified models. _Journal of statistical planning and inference_ , 143(10): 1621–1633, 2013. 

- Wang, X., Zhu, W., Saxon, M., Steyvers, M., and Wang, W. Y. Large language models are latent variable models: Explaining and finding good demonstrations for incontext learning. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Wang, Y. and Blei, D. Variational bayes under model misspecification. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Welling, M. and Teh, Y. W. Bayesian learning via stochastic gradient langevin dynamics. In _Proceedings of the 28th international conference on machine learning (ICML-11)_ , pp. 681–688. Citeseer, 2011. 

- Wildberger, J., Dax, M., Buchholz, S., Green, S., Macke, J. H., and Scholkopf,¨ B. Flow matching for scalable simulation-based inference. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Yeo, I.-K. and Johnson, R. A. A new family of power transformations to improve normality or symmetry. _Biometrika_ , 87(4):954–959, 2000. 

- Yim, J., Campbell, A., Foong, A. Y., Gastegger, M., Jimenez-Luna,´ J., Lewis, S., Satorras, V. G., Veeling, B. S., Barzilay, R., Jaakkola, T., et al. Fast protein backbone generation with se (3) flow matching. _arXiv preprint arXiv:2310.05297_ , 2023. 

13 

**Can Transformers Learn Full Bayesian Inference In Context?** 

- Yim, J., Campbell, A., Mathieu, E., Foong, A. Y., Gastegger, M., Jimenez-Luna, J., Lewis, S., Satorras, V. G., Veeling,´ B. S., Noe,´ F., et al. Improved motif-scaffolding with se (3) flow matching. _arXiv preprint arXiv:2401.04082_ , 2024. 

- Zhai, J., Zhang, S., Chen, J., and He, Q. Autoencoder and its various variants. In _2018 IEEE international conference on systems, man, and cybernetics (SMC)_ , pp. 415–419. IEEE, 2018. 

- Zhao, W., Shi, M., Yu, X., Zhou, J., and Lu, J. Flowturbo: Towards real-time flow-based image generation with velocity refiner. _arXiv preprint arXiv:2409.18128_ , 2024. 

- Zheng, K., Lu, C., Chen, J., and Zhu, J. Improved techniques for maximum likelihood estimation for diffusion odes. In _International Conference on Machine Learning_ , pp. 42363–42389. PMLR, 2023. 

14 

**Can Transformers Learn Full Bayesian Inference In Context?** 

# **Appendix** 

# **A. Data-generating Processses** 

This section contains more details on the data generating processes of the latent variable models we fit via ICL. 

## **A.1. Generalized Linear Models** 

In this section we expand the description and explanation regarding GLMs from Section 3.2. GLMs are among the most commonly used statistical models with myriads of applications (Nelder & Wedderburn, 1972; Fahrmeir et al., 2013). In the context of GLMs, we assume that the response _y_ follows a distribution _P_<sup>_y|_</sup><sup>**_u_**</sup> depending on the linear predictor _η_ := **_u_**<sup>_⊤_</sup> **_β_** and an additional parameter _σ_<sup>2</sup> . We denote the covariates as **_u_** , the regression coefficients as **_β_** , and use _σ_<sup>2</sup> for the variance of the response. The mean of _P_<sup>_y|_</sup><sup>**_u_**</sup> depends on the linear predictor via a link function _g_ , such that _g_ (E[ _y|_ **_u_** ]) = **_u_**<sup>_⊤_</sup> **_β_** . Ultimately, the density of distribution of the response _y_ depending on the linear predictor and the additional parameter is denoted by _p_ ( _y|g_ � **_u_**<sup>_⊤_</sup> **_β_** � _, σ_<sup>2</sup> ). To showcase the flexibility of our framework, we experiment with different priors _P_<sup>_β_</sup> on the regression coefficients, _P_<sup>_σ_2</sup> on the parameter _σ_<sup>2</sup> , and also different parametric distributions of the response. Additionally, to include covariates **_u_** that resemble practically relevant tabular data in the generative process, allowing for meaningful inference on real-world datasets, we utilize samples from the Tab-PFN “prior” for _P_<sup>**_u_**</sup> . 

GLMs belong to the framework of latent variable models defined by data **_x_** and (latent) variables **_z_** , where the data comprises covariates and response **_x_** := ( **_u_** _, y_ ). The variables of interest are the coefficients **_z_** := **_β_** . This yields the following generative process for a set of synthetic samples _D_ := _{_ ( **_x_** _i,_ **_z_** _i_ ) _}_<sup>_N_</sup> _i_ =1<sup>from</sup><sup>_P_</sup><sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**:</sup> 

We consider seven different GLM scenarios by varying the structure of the prior distributions and the conditional distribution of the response (Table 5). In particular, we consider a normal _N_ (0 _,_ 1) prior, a Laplace(0 _,_ 1) and a gamma Ga(1 _,_ 1) prior that factorizes over the coefficients _βj_ contained in **_β_** = ( _β_ 1 _, . . . , βp_ ). In two cases we include an intercept in the model using a normal prior _N_ (0 _,_ 9) with a relatively large variance. We consider regression cases with a normally distributed response _N_ ( **_u_**<sup>_⊤_</sup> **_β_** _, σ_<sup>2</sup> ), a Bernoulli distributed response Bin(1 _,_ sigmoid( **_u_**<sup>_⊤_</sup> **_β_** )), i.e. logistic regression, and a response following a gamma distribution Ga( _σ_<sup>_−_2</sup> exp ( **_u_**<sup>_⊤_</sup> **_β_** ) _, σ_<sup>_−_2</sup> exp(2 **_u_**<sup>_⊤_</sup> **_β_** )). In the last case, we set exp( **_u_**<sup>_⊤_</sup> **_β_** ) to be the mean and _σ_<sup>2</sup> to be the conditional variance of the response. An inverse gamma prior IG(5 _,_ 2) is used on the variance _σ_<sup>2</sup> for each scenario except the logistic regression. We fix the number of covariates and thus also the dimensionality of **_β_** at _p_ = 5 and set the number of data points per dataset to _K_ = 50. 

**Algorithm 2** Generation of synthetic data for GLMs 

**Require:** Number of datasets _N_ , number of samples per dataset _K_ , distributions _P_<sup>**_β_**</sup> _, P_<sup>**_σ_2**</sup> _, P_<sup>**_u_**</sup> , **Ensure:** A dataset _D_ of input-output pairs ( **_x_** _i,_ **_z_** _i_ ) for _i_ = 1 _, . . . , N_ . 

1: Initialize _D ←_ ∅ 

2: **for** _i_ = 1 _→ N_ **do** 3: Draw **_β_** _i ∼ P_<sup>**_β_**</sup> 4: Draw _σi_<sup>2</sup><sup>_∼P_</sup><sup>**_σ_2**</sup> 5: **for** _j_ = 1 _→ K_ **do** 6: Draw **_u_** _i,j ∼ P_<sup>**_u_**</sup> 7: Draw _yi,j ∼ p_ � _y_ �� _g−_ 1� **_u_**<sup>_⊤_</sup> _i,j_<sup>**_β_**</sup><sup>_i_</sup> � _, σi_<sup>2</sup> � 8: **end for** _K_ 9: Set **_x_** _i_ := � ( **_u_** _i,j, yi,j_ )� _j_ =1 10: Set **_z_** _i_ := **_β_** _i_ 11: Update _D ←D ∪{_ � **_x_** _i,_ **_z_** _i_ � _}_ 12: **end for** 

15 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 5: Distribution of variables for the considered GLM scenarios. 

|**Scenario**|_βi,j_|_βi,_0|_σ_<sup>2</sup><br>_i_|_yi,j|_(**_u_**_i,j,_ **_β_**_i, β_0_,i, σ_<sup>2</sup><br>_i_ <sup>)</sup>|
|---|---|---|---|---|
|Scenario 1|_N_(0_,_1)|-|IG(5_,_2)|_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup>|
|Scenario 2|_N_(0_,_1)|_N_(0_,_9)|IG(5_,_2)|_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup>|
|Scenario 3|Laplace(0_,_1)|-|IG(5_,_2)|_N_(**_u_**<sup>_⊤_</sup><br>_ij_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup>|
|Scenario 4|Laplace(0_,_1)|_N_(0_,_9)|IG(5_,_2)|_,_<br>_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup>|
|Scenario 5|Ga(1_,_1)|-|IG(5_,_2)|_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup>|
|Scenario 6|_N_(0_,_1)|-|-|Bin(1_,_sigmoid(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i_))</sup>|
|Scenario 7|_N_(0_,_1)|-|IG(5_,_2)|Ga(_σ_<sup>_−_2</sup><br>_i_<br>exp (**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i_)</sup><sup>_, σ−_2</sup><br>_i_<br>exp(2**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i_))</sup>|



## **A.2. Factor Analysis** 

The goal of factor analysis is to explain data **_x_** in terms of latent, typically lower-dimensional, factors **_z_** (Lawley & Maxwell, 1962; Rummel, 1988). In the Bayesian setting, one assumes a prior _P_<sup>**_z_**</sup> on the latent variable **_z_** , a prior _P_<sup>**_W_**</sup> on the factor loading matrix **_W_** and additional priors _P_<sup>**Ψ**</sup> and _P_<sup>**_µ_**</sup> on the covariance matrix and the mean vector. The conditional distribution _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> of the data given **_z_** has mean E[ **_z_** _|_ **_x_** ] = **_W z_** + **_µ_** and covariance matrix Cov[ **_z_** _|_ **_x_** ] = **Ψ** . In the case where _P_<sup>**_z_**</sup> and _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> are Gaussian, one can set _P_<sup>**_z_**</sup> = _N_ ( **0** _, I_ ) and assume a diagonal covariance matrix **Ψ** without loosing expressiveness of the model (Murphy, 2023). We make the assumption that **_W_** is lower triangular with positive entries on the diagonal in order to ensure identifiability of the model (Lopes & West, 2004). Additionally, we assume that the distributions **_µ_** , **Ψ** and _P_<sup>**_W_**</sup> fully factorize. In order to ensure that the diagonal of **_W_** is positive, we consider absolute values in the generative process. Algorithm 3 details the data generating process. 

Table 6 summarizes the different configurations for FA. We assume a Gaussian prior on the mean components, and an inverse gamma prior on the elements of the diagonal covariance matrix **Ψ** . For the factor loading matrix **_W_** , independent normal and Laplace priors are investigated. Furthermore, we use a normal prior on the latent factors **_z_** _i_ in five cases and a Laplace prior in one case. We vary the number of samples _K_ per dataset **_x_** , the dimensionality _P_ of each data point, as well as the dimensionality **_z_** _dim_ . 

Table 6: Distribution and dimensionalitites of variables for the considered FA scenarios. 

|**Scenario**|_K_|_P_|_µi,j_|Ψ_i,j,j_|_Wi,j,k_|_zi,j_|**_z_**_dim_|
|---|---|---|---|---|---|---|---|
|Scenario 1|50|3|_N_(0_,_1)|IG(5_,_1)|_N_(0_,_1)|_N_(0_,_1)|3|
|Scenario 2|50|3|_N_(0_,_0_._1)|IG(5_,_1)|Laplace(0_,_10)|_N_(0_,_1)|3|
|Scenario 3|25|5|_N_(0_,_0_._1)|IG(5_,_2)|_N_(0_,_3)|_N_(0_,_1)|3|
|Scenario 4|25|15|_N_(0_,_0_._1)|IG(5_,_2)|_N_(0_,_3)|_N_(0_,_1)|5|
|Scenario 5|25|5|_N_(0_,_0_._1)|IG(5_,_2)|Laplace(0_,_3)|_N_(0_,_1)|3|
|Scenario 6|25|5|_N_(0_,_0_._1)|IG(5_,_2)|_N_(0_,_3)|Laplace(0_,_1)|3|



**Algorithm 3** Generation of synthetic data for FA 

**Require:** Number of datasets _N_ , number of samples _K_ , and distributions _P_<sup>**_µ_**</sup> _, P_<sup>**Ψ**</sup> _, P_<sup>**_W_**</sup> _, P_<sup>**_z_**</sup> . **Ensure:** A dataset _D_ containing ( **_x_** _i,_ **_z_** _i_ ) for _i_ = 1 _, . . . , N_ . 

1: Initialize _D ←_ ∅ 2: **for** _i_ = 1 _→ N_ **do** 3: Draw **_µ_** _i ∼ P_<sup>**_µ_**</sup> 4: Draw **Ψ** _i ∼ P_<sup>**Ψ**</sup> 5: Draw **_W_** _i ∼ P_<sup>**_W_**</sup> 6: Draw **_z_** _i ∼ P_<sup>**_z_**</sup> 7: **for** _j_ = 1 _→ K_ **do** 8: Draw **_x_** _i,j ∼N_ � **_W_** _i_ **_z_** _i_ + **_µ_** _i,_ **Ψ** _i_ � 9: **end for** 10: Update _D ←D ∪{_ � **_x_** _i,_ **_z_** _i_ � _}_ 11: **end for** 

16 

**Can Transformers Learn Full Bayesian Inference In Context?** 

## **A.3. Gaussian Mixture Models** 

Table 7: Distribution and dimensionalitites of variables for the considered GMM scenarios. 

|**Scenario**|_K_|_M_|_L_|**_ϕ_**_i_|_σ_<sup>2</sup><br>_i,m,l_|_µi,m,l|σ_<sup>2</sup><br>_i,m,l_<br>|
|---|---|---|---|---|---|---|
|Scenario 1|50|5|1|Dir(1)|IG(5_,_2)|_N_(0_,_3_σ_<sup>2</sup><br>_i,m,l_<sup>)</sup>|
|Scenario 2|25|3|3|Dir(1)|IG(5_,_2)|_N_(0_,_3_σ_<sup>2</sup><br>_i,m,l_<sup>)</sup><br>|
|Scenario 3|50|3|5|Dir(0_._5)|IG(5_,_2)|_N_(0_,_5_σ_<sup>2</sup><br>_i,m,l_<sup>)</sup>|
|Scenario 4|50|3|3|Dir(1)|IG(5_,_2)|_N_(0_,_3_σ_<sup>2</sup><br>_i,m,l_<sup>)</sup>|



In GMMs one assumes that the data of interest is generated by a convex combination of _M_ (multivariate) normal distributions, such that _p_ ( **_x_** _|_ **_z_** ) =<sup>�</sup><sup>_M_</sup> _m_ =1<sup>**_ϕ_**</sup><sup>_mpm_(</sup><sup>**_x_**), where the probability vector</sup><sup>**_ϕ_**= (</sup><sup>_ϕ_1</sup><sup>_, . . . , ϕM_) comprises the mixture weights and</sup> _pm_ denotes the _m_ -th mixture component. We consider _pm_ to take the form of a diagonal Gaussian with mean vector **_µ_** _m_ and covariance matrix with diagonal elements **_σ_** _m_<sup>2.We assume a prior</sup><sup>_P_</sup><sup>**_ϕ_**on</sup><sup>**_ϕ_**,a prior</sup><sup>_P_</sup><sup>**_σ_**2on the variances of each</sup> component and a prior _P_<sup>**_µ_**</sup><sup>_|_</sup><sup>**_σ_**2</sup> for the means that depends on the variance of the respective component. More specifically, we assume a symmetric Dirichlet prior on **_ϕ_** such that _P_<sup>**_ϕ_**</sup> = Dir( _αDir_ ) and an independent inverse gamma distribution as prior on each component _σm_<sup>2of</sup><sup>**_σ_**</sup> _m_<sup>2.The prior on each component of</sup><sup>**_µ_**</sup><sup>_i,m∈_R</sup><sup>_L_is then given by an independent normal</sup> distribution _P_<sup>**_µ_**</sup><sup>_|_</sup><sup>**_σ_**</sup> _i,m,l_<sup>2</sup> = _N_ (0 _, λσi,m,l_<sup>2).We use</sup><sup>_ωi,j_to denote the assignment of datapoint</sup><sup>_j_a component.Algorithm 4</sup> details the data generating process and Table 7 summarizes the different setups regarding the prior distributions. 

**Algorithm 4** Generation of synthetic data for a GMM. 

**Require:** Number of datasets _N_ , mixture dimension parameters _M_ , _L_ , number of samples _K_ , and distributions _P_<sup>**_ϕ_**</sup> _, P_<sup>**_σ_**2</sup> _, P_<sup>**_µ_**</sup><sup>_|_</sup><sup>**_σ_**2</sup> . **Ensure:** A dataset _D_ containing ( **_x_** _i,_ **_z_** _i_ ) for _i_ = 1 _, . . . , N_ . 

1: Initialize _D ←_ ∅ 

2: **for** _i_ = 1 _→ N_ **do** 3: Draw **_ϕ_** _i ∼ P_<sup>**_ϕ_**</sup> 4: **for** _m_ = 1 _→ M_ **do** 5: **for** _l_ = 1 _→ L_ **do** 6: Draw _σi,m,l_<sup>2</sup><sup>_∼P_</sup><sup>**_σ_**2</sup> 7: Draw _µi,m,l ∼ P_<sup>**_µ_**</sup><sup>_|_</sup><sup>**_σ_**</sup> _i,m,l_<sup>2</sup> 8: **end for** 9: **end for** 10: **for** _j_ = 1 _→ K_ **do** 11: Draw _ωi,j ∼_ Cat� **_ϕ_** _i_ � 12: Draw **_x_** _i,j ∼N_ � **_µ_** _i,ωi,j ,_ **_σ_** _i,ω_<sup>2</sup> _i,j_ � 13: **end for** 14: Set **_z_** _i_ := �� _σi,m,l_<sup>2</sup><sup>_,µi,m,l_</sup> �� _m_ =1 _,...,M l_ =1 _,...,L_ 15: Update _D ←D ∪_ �( **_x_** _i,_ **_z_** _i_ )� 16: **end for** 

17 

**Can Transformers Learn Full Bayesian Inference In Context?** 

# **B. Generating Realistic Data** 

While we assume a data-generating process such as the one in Algorithm 2, this is not necessarily the data-generating process that produces the data in the model’s application as an in-context learner. Even when the generative process _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> underlying a statistical model is sophisticated and complex in nature, model misspecification is inevitable in almost every practical application. While mismatches between the real data-generating processes and model assumptions can lead to various problems in traditional Bayesian modeling (Grunwald & van Ommen¨ , 2017), the question of model misspecification plays a somewhat different and yet an especially central role for our ICL approach. 

More specifically, the ICL model learns the relationship between _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> and a datapoint **_x_** exclusively based on synthetic samples from the marginal _P_<sup>**_x_**</sup> implied by the statistical model with generative process _P_<sup>**_x_**</sup><sup>_,_</sup><sup>**_z_**</sup> . Given a real-world dataset **_x_**<sup>_∗_</sup> _∼ P_<sup>**_x_**</sup><sup>_∗_</sup> , model misspecification in terms of _P_<sup>**_x_**</sup><sup>_∗_</sup> implies that the in-context learner needs to infer the posterior based on out-of-distribution data, where the problem is aggravated the more unrealistic _P_<sup>**_x_**</sup> is. 

To be able to access a reference or ground truth distribution, the data generating processes in our experiments need to match the structure of the GLM, FA and GMM approaches. While the generative processes of FA and GMMs directly prescribe how all parts of the data are generated, this can potentially cause a discrepancy between synthetically generated and real-world datasets. However, our empirical results (Section 4.1) demonstrate that the in-context learner can generalize to real-world data despite the discrepancy to the simulated datasets. 

In the aforementioned GLM case, the distribution of the covariates _P_<sup>**_u_**</sup> does not affect the structure of _P_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> in the data generating process (cf. Algorithm 2). We can therefore use a flexible prior _P_<sup>**_u_**</sup> such as the TabPFN-“prior” (Hollmann et al., 2022) to generate covariates **_u_** and thereby effectively tackle the issue of model specification. 

# **C. Preprocessing of the Real-world Datasets** 

The real-world datasets considered for the evaluation of all methods are proposed in a benchmark study by Grinsztajn et al. (2022). We standardize all features, scale and shift the target such that it has the mean and variance implied by the prior structure of the respective generative model. Furthermore, for the GLM scenarios, we apply a Yeo-Johnson transform on the target variable (Yeo & Johnson, 2000) before applying the scaling. In cases where the number of features in the real-world dataset exceeds that of our scenario, we select those features with the most distinct values in the original dataset and randomly sub-sample the appropriate number of samples from the real-world datasets for our experiments. 

# **D. Background on Conditional Flow-matching** 

Flow matching, initially used in image synthesis leverages normalizing flows (Papamakarios et al., 2021b) to model arbitrary distributions. Continuous normalizing flows (Lipman et al., 2022) have emerged as a potent tool for modeling complex distributions. For example, recent advancements have shown its effectiveness in state-of-the-art image generation, outperforming diffusion-based methods in likelihood and sample quality on ImageNet (Lipman et al., 2022). Techniques like FlowTurbo have accelerated class-conditional and text-to-image generation, setting new benchmarks (Zhao et al., 2024). Additionally, applying flow matching in latent spaces of pretrained autoencoders has enhanced computational efficiency and scalability for high-resolution image synthesis (Dao et al., 2023). Similarly, flow-based models have been successfully applied to protein structure prediction, improving accuracy and efficiency in modeling complex protein conformations (Yim et al., 2024; 2023). 

In the area of simulation-based inference, Wildberger et al. (2024) introduce the idea of using continuous normalizing flows in order to efficiently approximate complex posterior distributions. In particular, they apply the framework to the field of gravitational-wave inference, substantially outperforming approaches based on discrete flows. Furthermore, they demonstrate good performance on the existing SBI-Benchmark (Lueckmann et al., 2021) using a simple MLP-based architecture. 

# **E. Relationship of our approach to density estimation methods** 

An alternative to flow matching for parameterizing our model _Q_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> would be explicit (conditional) density estimation. While knowledge of the explicit density of a distribution can be useful for downstream tasks, we would like to reemphasize that we consider the problem of full Bayesian inference via _sampling_ from the posterior in this paper. 

18 

### **Can Transformers Learn Full Bayesian Inference In Context?** 

Popular explicit density estimation methods include i-DODE (Zheng et al., 2023), which proposes several techniques for improving maximum likelihood estimation from diffusion ordinary differential equations (ODEs), including velocity parameterization and techniques for variance reduction, leading to faster convergence. Additionally,in (Sahoo et al., 2024) log-likelihood estimation is improved by casting the learned diffusion process as a variational posterior that yields a tighter evidence lower bound on the actual likelihood. (Lienen et al., 2025) propose a novel generative model using iterative Gaussian posterior inference and empirically demonstrate that it yields strong results in log-likelihood estimation. Furthermore, Salazar (2023) use variational inference to learn Bayesian regression trees, which could be used for multivariate density estimation. 

Note that it is, in principle, also possible to recover the explicit density of _Q_<sup>**_z_**</sup><sup>_|_</sup><sup>**_x_**</sup> , which is parametrized in the Flow Matching Framework that is used for our approach (Wildberger et al., 2024). 

# **F. Hyperparameters, Software and Computational Setup** 

In this section, we detail the hyperparameters, used software and computational setups for all our experiments. 

## **F.1. ICL** 

To ensure maximum comparability across different experiments, we fix the hyperparameters for all ICL experiments: For the architecture of the model introduced in Section 3.3, we use the following configuration: The dimensionality of encoder representations is set to 512 and is expanded to 1024 in the feed-forward blocks. We use 8 heads and 8 encoder layers with a dropout rate of 0.1. For the decoder part we also use 512 as the dimensionality of the representations and 1024 as the intermediate representation in the feed-forward layers and a dropout rate of 0.1. Furthermore, 3 simple fully connected layers with adaLN conditioning are used for final processing in the decoder. For the time conditioning, we use 3 simple fully connected layers to map the scalar-valued time _t_ onto a 512 dimensional conditioning vector that is used for the adaLN blocks in the decoder. This yields a model of around 43.1 million parameters. We use no tokenization for either the encoder or the decoder and simple embedding layers to map the encoder- and decoder-input onto the feed-forward dimensions. 

We use an Adam optimizer (Kingma, 2014) with a cosine learning rate schedule (Loshchilov & Hutter, 2016), where the maximum learning rate is 5 _·_ 10<sup>_−_4</sup> , the final division factor is 10<sup>4</sup> and 10 percent of the epochs are used for warm-up. We use a weight decay parameter of 10<sup>_−_5</sup> and a batch size of 1024 and gradient clipping with a maximum gradient norm of one. We use in total 75 million synthetic samples for all scenarios. Of the total number, half, i.e. 37.5 million, are used for training and 10 percent for validation and the remaining 40 percent for testing. Note that we observe convergence of the loss usually much earlier than after this training duration, but fix the number of samples for consistency across experiments. A single L4 GPU is used for the GLM scenarios and a single A100 GPU for the FA and GMM cases. 

To solve the ODE for the sample generation, dopri5 (Dormand & Prince, 1980) as implemented in Torchdiffeq (Chen, 2018) is used in the adjoint version. We set the relative and absolute tolerance to 10<sup>_−_7</sup> . The _σ_ min parameter in the CNF-loss is set to 10<sup>_−_4</sup> . 

## **F.2. HMC** 

We use HMC with a NUTS kernel (Hoffman et al., 2014) as a reference for all experiments where no analytical solution is available. We set the number of burn-in samples to 500 and use one chain for all uni-modal problems and three times the number of potential modes in all other cases. More specifically, we use _M ×_ 3 chains for all GMM scenarios. The Pyro implementation of NUTS is used for the GLM scenarios (Bingham et al., 2019) and the conceptually identical, albeit computationally faster implementation in Numpyro for the FA and GMM cases (Phan et al., 2019). 

## **F.3. VI** 

For the variational inference methods, we utilize automatic guide generation based on the ground-truth data-generating processes (Kucukelbir et al., 2017). Pyro is used for the implementation of the probabilistic programs, which we also use to sample the synthetic training data, for the automatic guide generation, and for the implementation of the actual VI methods (Bingham et al., 2019). Default hyperparameters, as well as an Adam optimizer (Kingma, 2014) with a learning rate of 10<sup>_−_2</sup> is used for all methods except for AutoIAF where a learning rate of 10<sup>_−_3</sup> is used. We perform 2000 full-batch gradient update steps for each method. 

19 

### **Can Transformers Learn Full Bayesian Inference In Context?** 



Figure 4: Learning curves for GLM scenario 1 with a Normal Prior on the coefficients **_β_** and an Inverse Gamma prior on _σ_ 2. 





Figure 5: Learning curves for GMM Figure 6: Learning curves for GMM scenario 1 with _M_ = 5 components, scenario 3 with _M_ = 3 components, _K_ = 50 datapoints and _L_ = 1 dimen- _K_ = 50 datapoints and _L_ = 5 dimensions. sions. 

# **G. Runtimes** 

We use a single L4 GPU for generating samples based on our ICL approach and HMC in the GLM scenarios, a single A100 for our ICL approach and HMC in the FA and GMM scenarios, and an Intel(R) Xeon(R) CPU @ 2.20GHz CPU with two virtual cores and 40 gigabytes of RAM for the VI methods. Across all considered GLM scenarios, pre-training takes on average 14 _._ 89 hours with a standard error of 18 _._ 01 minutes. For the FA scenarios, on average 3 _._ 95 hours with a standard error of 11 _._ 38 minutes is used for pretraining and for the GMM scenarios 10 _._ 63 with a standard error of 72 _._ 88 minutes. 

When applied in order to generate samples for a new dataset, the benchmarked VI methods have, as expected the lowest runtime. The Laplace approximation is the fastest of all methods, while our ICL appraoch has consistently a lower runtime compared to HMC. Overall, the ICL method takes around 2 minutes on the GLM tasks, around 30 seconds in the FA scenarios and less than 2 minutes for the inference regarding the GMM tasks. 

This difference is especially pronounced in the FA and GMM scenarios. Please note that the runtime of the ICL method also fundamentally depends on the used precision for solving the underlying differential equation where we use a relatively high relative and absolute precision of 10<sup>_−_7</sup> . Decreasing this value might lead to significantly faster inference time while maintaining sample quality. 

20 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 8: Runtime Metrics for all GLM, FA, and GMM Scenarios 

|**Scenario**|**Method**|**Mean Runtime(s)**|
|---|---|---|
||Laplace Approximation|10_._48 (_±_0_._25)|
||VI: DiagonalNormal|12_._02 (_±_0_._26)<br>|
||VI: MultivariateNormal|13_._70 (_±_0_._29)|
|GLM|VI: Structured Normal|19_._81 (_±_0_._98)|
||VI:IAF|15_._44 (_±_0_._30)|
||HMC|120_._24 (_±_13_._94)|
||**ICL(ours)**|107_._79(_±_17_._36)|
||Laplace Approximation|17_._85 (_±_0_._21)|
||VI: DiagonalNormal|20_._94 (_±_0_._66)|
||VI: MultivariateNormal|20_._84 (_±_0_._28)|
|FA|VI: Structured Normal|36_._17 (_±_0_._61)|
||VI:IAF|23_._75 (_±_0_._38)|
||HMC|248_._26 (_±_57_._88)|
||**ICL(ours)**|31_._49(_±_4_._97)|
||Laplace Approximation|27_._52 (_±_0_._40)|
||VI: DiagonalNormal|29_._74 (_±_0_._57)|
||VI: MultivariateNormal|30_._50 (_±_0_._41)|
|GMM|VI: Structured Normal|42_._44 (_±_0_._44)|
||VI:IAF|33_._39 (_±_0_._49)|
||HMC|239_._67 (_±_32_._71)|
||**ICL(ours)**|93_._88(_±_10_._47)|



# **H. Ablation: Different Learning Rates for VI** 

To investigate the role of the learning rate parameter for the benchmarked VI methods, we record the performance for learning-rate values of 10<sup>_−_2</sup> , 10<sup>_−_3</sup> and 10<sup>_−_4</sup> across a prototypical GLM, a FA and a GMM scenario, where we use 10 synthetic and 10 real-world datasets. In summary, while we find the VI methods to often be quite robust to the choice of the learning rate, those results also confirm our choice of setting the learning rate to 10<sup>_−_2</sup> for the Laplace approximation, variational inference with a diagonal normal distribution, a multivariate normal distribution and a structured normal distribution, and to a value of 10<sup>_−_3</sup> for the VI approach with inverse autoregressive flows. 

For the GLM-scenario, we find in terms of the C2ST metric that VI with an ordinary multivariate normal distribution and VI with a structured normal distribution and a learning rate of 10<sup>_−_2</sup> are the best models on the synthetic data. While MMD also indicates that this learning rate yields ideal results for those models, VI with inverse auoregressive flows has good values across the different learning rates with the minimum for 10<sup>_−_3</sup> . The _W_ 2 metric indicates a similar tendency. 

Regarding the learning rate for the FA scenario, one can first see that no single learning rate seems to dominate substantially given the variance of the results. However, on the synthetic data for the Laplace approximation, as well as VI with a diagonal normal distribution, a multivariate normal and a structured normal distribution, the lowest average result is obtained for a learning rate of 10<sup>_−_2</sup> , while for VI with inverse autoregressive flows the best performance is obtained when the learning rate equals 10<sup>_−_3</sup> . The real-world results are the best for VI with a structured normal distribution and a learning rate of 10<sup>_−_2</sup> . 

For the GMM scenario, we find that VI with a diagonal, structured and ordinary normal distribution obtain the best results, namely for learning rates of 10<sup>_−_2</sup> and 10<sup>_−_3</sup> , taking the variance into account. Just considering the averages leads to the conclusion that 10<sup>_−_2</sup> is the best choice here. The results on the real-world data confirm that 10<sup>_−_2</sup> is the optimal choice for VI with a diagonal normal and ordinary multivariate normal, while VI with inverse autoregressive flows has good results across all choices regarding the learning rate. 

21 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 9: Results of VI methods with different learning rates on 10 synthetic and 10 real-world datasets: Linear regression with a normal prior on the coefficients **_β_** and an inverse gamma prior on the variance _σ_<sup>2</sup> (scenario 1). Comparison to HMC samples. All results within two standard errors of the best average result are marked in **bold** . 

|**Model**|**LR**|**S**|**ynthetic Evaluatio**|**n**|**Re**|**al-World Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Laplace Approximation|1e-2|1.000 (_±_0.000)|2.342(_±_0.390)|2.121 (_±_0.100)|1.000 (_±_0.000)|2.134(_±_0.107)|2.095 (_±_0.062)|
|Laplace Approximation|1e-3|1.000 (_±_0.000)|2.341(_±_0.389)|2.121 (_±_0.100)|1.000 (_±_0.000)|2.133(_±_0.108)|2.095 (_±_0.062)|
|Laplace Approximation|1e-4|1.000 (_±_0.000)|2.341(_±_0.389)|2.121 (_±_0.100)|1.000 (_±_0.000)|2.133(_±_0.108)|2.095 (_±_0.062)|
|VI: DiagonalNormal|1e-2|0.892 (_±_0.074)|0.921(_±_0.374)|**1.411**(_±_0.174)|0.889 (_±_0.062)|0.819(_±_0.343)|1.339 (_±_0.190)|
|VI: DiagonalNormal|1e-3|0.966 (_±_0.024)|1.588(_±_0.540)|**1.672**(_±_0.203)|0.981 (_±_0.017)|1.685(_±_0.331)|1.739 (_±_0.139)|
|VI: DiagonalNormal|1e-4|0.971 (_±_0.010)|1.572(_±_0.300)|1.666 (_±_0.081)|0.849 (_±_0.030)|0.575(_±_0.127)|**1.221**(_±_0.098)|
|VI: MultivariateNormal|1e-2|**0.725**(_±_0.064)|**0.523**(_±_0.242)|**1.114**(_±_0.261)|**0.625**(_±_0.051)|**0.470**(_±_0.066)|**0.918**(_±_0.119)|
|VI: MultivariateNormal|1e-3|0.964 (_±_0.008)|1.455(_±_0.327)|1.617 (_±_0.100)|0.853 (_±_0.052)|0.634(_±_0.266)|**1.238**(_±_0.151)|
|VI: MultivariateNormal|1e-4|0.984 (_±_0.005)|1.848(_±_0.324)|1.773 (_±_0.079)|0.899 (_±_0.020)|0.807(_±_0.094)|**1.345**(_±_0.079)|
|VI: Structured Normal|1e-2|**0.734**(_±_0.063)|**0.541**(_±_0.254)|**1.119**(_±_0.264)|**0.670**(_±_0.047)|**0.467**(_±_0.086)|**1.060**(_±_0.130)|
|VI: Structured Normal|1e-3|0.882 (_±_0.042)|0.719(_±_0.315)|1.335 (_±_0.149)|0.776 (_±_0.045)|**0.473**(_±_0.081)|**1.064**(_±_0.131)|
|VI: Structured Normal|1e-4|0.890(_±_0.027)|0.710 (_±_0.290)|1.347(_±_0.138)|0.771(_±_0.049)|**0.468** (_±_0.078)|**1.062** (_±_0.128)|
|VI: IAF|1e-2|0.840 (_±_0.036)|**0.502**(_±_0.262)|**1.272**(_±_0.170)|**0.614**(_±_0.045)|**0.455**(_±_0.048)|**0.957**(_±_0.105)|
|VI: IAF|1e-3|0.797 (_±_0.065)|**0.485**(_±_0.556)|**1.169**(_±_0.313)|**0.619**(_±_0.036)|**0.469**(_±_0.064)|**0.989**(_±_0.124)|
|VI: IAF|1e-4|0.803 (_±_0.068)|**0.475**(_±_0.535)|**1.162**(_±_0.291)|**0.612**(_±_0.034)|**0.457**(_±_0.055)|**0.977**(_±_0.113)|



Table 10: Results of VI methods with different learning rates on 10 synthetic and 10 real-world datasets: Factor analysis with Gaussian priors on the weights and the latents and _K_ = 25 datapoints, _P_ = 5 features, and dimensionality of the latents **z** _dim_ = 3 (scenario 3). Comparison to HMC samples. All results within two standard errors of the best average result are marked in **bold** . 

|**Model**|**LR**|**S**|**ynthetic Evaluatio**|**n**|**Re**|**al-World Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Laplace Approximation|1e-2|1.000 (_±_0.000)|3.449(_±_0.821)|**1.773**(_±_0.539)|1.000 (_±_0.000)|2.703(_±_0.312)|**0.362**(_±_0.017)|
|Laplace Approximation|1e-3|1.000 (_±_0.000)|4.288(_±_0.853)|**2.263**(_±_0.732)|1.000 (_±_0.000)|2.896(_±_0.238)|**0.376**(_±_0.022)|
|Laplace Approximation|1e-4|1.000 (_±_0.000)|4.252(_±_0.611)|**2.122**(_±_0.430)|1.000 (_±_0.000)|2.805(_±_0.181)|**0.368**(_±_0.017)|
|VI: DiagonalNormal|1e-2|0.998 (_±_0.002)|2.880(_±_1.046)|**1.457**(_±_0.559)|0.944 (_±_0.008)|1.022(_±_0.067)|**0.230**(_±_0.010)|
|VI: DiagonalNormal|1e-3|0.998 (_±_0.002)|2.973(_±_0.834)|**1.465**(_±_0.540)|0.941 (_±_0.006)|0.997(_±_0.056)|**0.229**(_±_0.010)|
|VI: DiagonalNormal|1e-4|1.000 (_±_0.001)|3.416(_±_0.761)|**1.602**(_±_0.437)|0.943 (_±_0.009)|0.997(_±_0.057)|**0.229**(_±_0.010)|
|VI: MultivariateNormal|1e-2|0.993 (_±_0.007)|2.969(_±_1.089)|**1.506**(_±_0.659)|**0.929**(_±_0.007)|**0.957**(_±_0.048)|**0.224**(_±_0.010)|
|VI: MultivariateNormal|1e-3|0.996 (_±_0.004)|3.140(_±_0.910)|**1.570**(_±_0.625)|0.934 (_±_0.009)|**0.971**(_±_0.054)|**0.225**(_±_0.010)|
|VI: MultivariateNormal|1e-4|0.997 (_±_0.007)|3.464(_±_0.791)|**1.639**(_±_0.426)|0.934 (_±_0.005)|**0.962**(_±_0.049)|**0.225**(_±_0.010)|
|VI: Structured Normal|1e-2|0.998 (_±_0.002)|3.005(_±_0.871)|**1.481**(_±_0.504)|0.947 (_±_0.005)|1.003(_±_0.066)|**0.230**(_±_0.009)|
|VI: Structured Normal|1e-3|0.999 (_±_0.001)|3.244(_±_0.665)|**1.619**(_±_0.559)|0.948 (_±_0.007)|1.033(_±_0.078)|**0.232**(_±_0.009)|
|VI: Structured Normal|1e-4|0.999 (_±_0.001)|3.119(_±_0.612)|**1.487**(_±_0.400)|0.943 (_±_0.007)|0.998(_±_0.056)|**0.229**(_±_0.010)|
|VI: IAF|1e-2|**0.939**(_±_0.040)|**2.836**(_±_0.293)|**1.247**(_±_0.297)|0.944 (_±_0.008)|1.518(_±_0.048)|1.332 (_±_0.027)|
|VI: IAF|1e-3|**0.927**(_±_0.047)|**2.758**(_±_0.342)|**1.195**(_±_0.331)|0.949 (_±_0.009)|1.560(_±_0.031)|**1.392**(_±_0.024)|
|VI: IAF|1e-4|**0.842**(_±_0.038)|**2.862**(_±_0.296)|**1.281**(_±_0.292)|0.943 (_±_0.008)|1.493(_±_0.039)|1.302 (_±_0.039)|



22 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 11: Results of VI methods with different learning rates on 10 synthetic and 10 real-world datasets: Gaussian Mixture Model with _K_ = 50 datapoints, _L_ = 1 features (univariate case), _M_ = 5 components, _λ_ = 3, and _αdir_ = 1 (scenario 1). Comparison to HMC samples.All results within two standard errors of the best average result are marked in **bold** . 

|**Model**|**LR**|**S**|**ynthetic Evaluatio**|**n**|**Re**|**al-World Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Laplace Approximation|1e-2|1.000 (_±_0.000)|4.380(_±_1.386)|**4.838**(_±_1.521)|1.000 (_±_0.000)|4.588(_±_1.229)|**6.813**(_±_1.697)|
|Laplace Approximation|1e-3|1.000 (_±_0.000)|3.893(_±_1.433)|**4.010**(_±_1.233)|1.000 (_±_0.000)|4.699(_±_1.193)|**6.986**(_±_0.981)|
|Laplace Approximation|1e-4|1.000 (_±_0.000)|4.463(_±_1.117)|**4.610**(_±_1.027)|1.000 (_±_0.000)|4.710(_±_1.205)|**6.995**(_±_0.869)|
|VI: DiagonalNormal|1e-2|**0.979**(_±_0.138)|**1.370**(_±_1.394)|**3.522**(_±_1.634)|**0.985**(_±_0.030)|2.384(_±_1.318)|**6.202**(_±_1.747)|
|VI: DiagonalNormal|1e-3|**0.990**(_±_0.096)|**1.454**(_±_1.454)|**3.650**(_±_1.743)|0.999 (_±_0.002)|3.026(_±_0.977)|**6.959**(_±_0.890)|
|VI: DiagonalNormal|1e-4|1.000 (_±_0.001)|2.390(_±_1.177)|**4.903**(_±_1.278)|0.998 (_±_0.007)|2.830(_±_1.001)|**7.007**(_±_0.987)|
|VI: MultivariateNormal|1e-2|**0.978**(_±_0.119)|**1.351**(_±_1.410)|**3.474**(_±_1.604)|**0.987**(_±_0.024)|2.375(_±_1.304)|**6.189**(_±_1.761)|
|VI: MultivariateNormal|1e-3|**0.980**(_±_0.089)|**1.476**(_±_1.480)|**3.681**(_±_1.734)|0.997 (_±_0.008)|2.808(_±_1.014)|**6.964**(_±_0.944)|
|VI: MultivariateNormal|1e-4|1.000 (_±_0.001)|2.114(_±_1.140)|**4.532**(_±_1.187)|0.997 (_±_0.007)|2.799(_±_1.012)|**6.963**(_±_0.950)|
|VI: Structured Normal|1e-2|**0.958**(_±_0.129)|**1.246**(_±_1.615)|**3.225**(_±_1.701)|1.000 (_±_0.001)|2.911(_±_0.753)|**6.675**(_±_1.403)|
|VI: Structured Normal|1e-3|**0.979**(_±_0.092)|**1.593**(_±_1.561)|**3.395**(_±_1.440)|0.998 (_±_0.007)|2.882(_±_1.070)|**6.968**(_±_0.941)|
|VI: Structured Normal|1e-4|1.000 (_±_0.001)|2.270(_±_1.133)|**4.733**(_±_1.162)|0.997 (_±_0.009)|2.802(_±_1.012)|**6.953**(_±_0.948)|
|VI: IAF|1e-2|0.998 (_±_0.003)|**1.539**(_±_0.691)|**8.371**(_±_0.750)|**0.987**(_±_0.022)|**1.376**(_±_0.799)|**8.082**(_±_1.352)|
|VI: IAF|1e-3|0.997 (_±_0.004)|**1.443**(_±_0.564)|**8.517**(_±_0.820)|**0.988**(_±_0.020)|**1.304**(_±_0.855)|**8.425**(_±_1.281)|
|VI: IAF|1e-4|0.997 (_±_0.004)|**1.602**(_±_0.628)|**7.888**(_±_0.783)|**0.987**(_±_0.020)|**1.380**(_±_0.848)|**7.729**(_±_1.322)|



23 

**Can Transformers Learn Full Bayesian Inference In Context?** 

# **I. Detailed Experimental Results** 

In this section, we describe our experimental results in detail, discussing how different scenarios for GLMs, FA and GMMs affect the performance of different approaches. 

## **I.1. Generalized Linear Models** 



Figure 7: Density plots for first three the marginals of the posterior in a GLM with a gamma prior on the coefficients _β_ , and an inverse gamma prior on the variance _σ_<sup>2</sup> of the responses. The data is part of the Miami housing 2016 dataset. 

Table 45 contains detailed results regarding the performance of the proposed ICL and the reference VI approaches. In summary, we find that on the synthetic data, our ICL method has the overall best performance, or a performance not significantly worse than that of the best model, with respect to the C2ST metric.<sup>4</sup> More specifically, ICL significantly outperforms all other models in 5 out of seven cases w.r.t. the C2ST and also the MMD metric. While the _W_ 2 metric exhibits a larger variance, it also indicates that on the synthetic data, ICL yields the significantly best result in those 5 cases. 

On the real-world data, the differences between ICL and VI are less pronounced, and ICL attains the best average result without any other model within two standard errors in three scenarios in terms of the C2ST metric. ICL is among those models not significantly worse than the best in four cases with respect to the C2ST metric, in six cases in terms of the MMD metric, and also in six cases in terms of _W_ 2. 

In scenario 1, which is a linear regression scenario with a normal prior on the coefficients **_β_** and an inverse gamma prior on the variance _σ_<sup>2</sup> , ICL and HMC show a similarly large agreement with the analytical solution. Furthermore, the VI approaches with an ordinary multivariate normal distribution, a structured normal distribution as well as the approach based on inverse autoregressive flows also show a large agreement with the analytical solution, which is to be expected since scenario 1 is has a conjugate prior structure yielding a multivariate t-distribution for the posterior of the coefficients (Murphy, 2023). 

Scenario 2 and scenario 4 are those where an intercept is included in the generative structure of the GLM. The notably superior performance of the ICL approach in those two cases might be explained by its ability to model distributions with substantially different variances in different dimensions better than VI. Similarly, the posterior in scenario 5 is determined by the gamma prior on the coefficients leading to a (slightly) skewed posterior distribution, which might explain the good relative performance of ICL. See Figure 7 for a plot of the marginals of the posterior in this scenario on the Miami housing 2016 dataset. 

Finally, scenarios 6 and 7 demonstrate the versatility of the ICL method in terms of posterior inference for logistic regression and regression with a gamma response. 

> 4We refer to a difference that is larger than two standard deviations as “significant”. 

24 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 12: Generalized Linear Models: Evaluation on 50 synthetic and 17 real-world datasets for seven different scenarios. All results within two standard errors of the best average result for each scenario are marked in **bold** . 

|**Scenario**|**Model**||**Synthetic Evaluatio**|**n**|**R**|**eal-Wor**|**ld Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD|(_↓_)|_W_2(_↓_)|
||Laplace Approximation|1.000 (_±_0.000)|2.738(_±_0.721)|**0.825**(_±_0.279)|1.000 (_±_0.000)|2.150|(_±_0.323)|**0.642**(_±_0.124)|
||VI: DiagonalNormal|0.904 (_±_0.076)|1.452(_±_0.984)|**0.669**(_±_0.301)|0.797 (_±_0.083)|0.612|(_±_0.511)|**0.414**(_±_0.152)|
||VI: MultivariateNormal|**0.750**(_±_0.128)|**0.735**(_±_0.733)|**0.565**(_±_0.292)|**0.607**(_±_0.070)|**0.167**|(_±_0.196)|**0.301**(_±_0.123)|
|Scenario 1|VI: Structured Normal|**0.753**(_±_0.126)|<br>**0.736**(_±_0.737)|**0.570**(_±_0.310)|**0.600**(_±_0.070)|<br>**0.169**|<br>(_±_0.214)|**0.306**(_±_0.131)|
||VI: IAF|**0.777**(_±_0.122)|**0.864**(_±_0.844)|0.725 (_±_0.523)|0.683 (_±_0.132)|0.440|(_±_0.559)|0.503 (_±_0.383)|
||HMC|**0.745**(_±_0.130)|**0.722**(_±_0.732)|**0.569**(_±_0.301)|**0.595**(_±_0.075)|**0.173**|(_±_0.213)|**0.321**(_±_0.140)|
||**ICL (ours)**|**0.765**(_±_0.123)|**0.767**(_±_0.727)|**0.585**(_±_0.301)|**0.614**(_±_0.074)|**0.175**|(_±_0.219)|**0.310**(_±_0.138)|
||Laplace Approximation|1.000 (_±_0.000)|4.853(_±_2.333)|5.770 (_±_5.946)|1.000 (_±_0.000)|2.572|(_±_0.206)|0.809 (_±_0.149)|
||VI: DiagonalNormal|0.957 (_±_0.091)|3.906(_±_2.679)|5.628 (_±_6.092)|0.892 (_±_0.044)|0.847|(_±_0.389)|**0.530**(_±_0.175)|
||VI: MultivariateNormal|0.910 (_±_0.131)|3.407(_±_2.781)|5.584 (_±_6.104)|0.820 (_±_0.031)|0.243|(_±_0.148)|**0.408**(_±_0.118)|
|Scenario 2|VI: Structured Normal|0.908 (_±_0.119)|3.139(_±_2.763)|5.480 (_±_6.164)|0.824 (_±_0.023)|0.215|(_±_0.110)|**0.392**(_±_0.109)|
||VI: IAF|0.968 (_±_0.063)|4.416(_±_2.473)|7.474 (_±_6.235)|0.888 (_±_0.067)|0.921|(_±_0.860)|0.942 (_±_0.733)|
||**ICL (ours)**|**0.839**(_±_0.072)|**0.707**(_±_0.658)|**1.111**(_±_0.300)|**0.768**(_±_0.033)|**0.143**|(_±_0.089)|**0.411**(_±_0.094)|
||Laplace Approximation|1.000 (_±_0.000)|2.203(_±_0.997)|1.170 (_±_0.949)|1.000 (_±_0.000)|1.841|(_±_0.185)|0.729 (_±_0.175)|
||VI: DiagonalNormal|0.866 (_±_0.101)|1.069(_±_1.150)|0.846 (_±_0.747)|0.797 (_±_0.083)|0.526|(_±_0.361)|0.480 (_±_0.207)|
|Si 3|VI: MultivariateNormal|0.656 (_±_0.131)|0.445(_±_1.061)|0.660 (_±_0.737)|**0.560**(_±_0.035)|**0.032**|(_±_0.028)|**0.249**(_±_0.069)|
|cenaro|VI: Structured Normal|0.653 (_±_0.125)|0.421(_±_0.993)|0.659 (_±_0.736)|**0.552**(_±_0.028)|**0.027**|(_±_0.015)|**0.239**(_±_0.055)|
||VI: IAF<br>|0.751 (_±_0.148)<br>|0.939(_±_1.349)<br>|0.964 (_±_0.924)<br>|0.673 (_±_0.141)<br>|0.399 <br>|(_±_0.543)<br>|0.563 (_±_0.433)<br>|
||**ICL (ours)**|**0.611**(_±_0.070)|**0.089**(_±_0.114)|**0.423**(_±_0.348)|0.576 (_±_0.027)|**0.037**|(_±_0.026)|**0.257**(_±_0.044)|
||Laplace Approximation|1.000 (_±_0.000)|3.511(_±_2.025)|2.166 (_±_1.722)|1.000 (_±_0.000)|2.011|(_±_0.058)|0.993 (_±_0.144)|
||VI: DiagonalNormal|0.968 (_±_0.036)|2.798(_±_2.255)|2.065 (_±_1.745)|0.916 (_±_0.040)|0.928|(_±_0.339)|0.732 (_±_0.181)|
|i 4|VI: MultivariateNormal|0.855 (_±_0.123)|1.648(_±_2.052)|1.853 (_±_1.745)|0.771 (_±_0.017)|**0.087**|(_±_0.030)|**0.539**(_±_0.070)|
|Scenaro|VI: Structured Normal|0.847 (_±_0.116)|1.505(_±_1.978)|1.889 (_±_1.883)|0.769 (_±_0.012)|**0.083**|(_±_0.018)|**0.543**(_±_0.070)|
||VI: IAF|0.942 (_±_0.077)|3.029(_±_2.210)<br>|3.554 (_±_2.715)|0.833 (_±_0.069)|0.636 <br>|(_±_0.756)<br>|0.978 (_±_0.600)|
||**ICL (ours)**|**0.753**(_±_0.049)|**0.171**(_±_0.153)|**0.631**(_±_0.294)|**0.762**(_±_0.015)|**0.105**|(_±_0.046)|**0.597**(_±_0.104)|
||Laplace Approximation|1.000 (_±_0.000)|2.060(_±_0.472)|0.797 (_±_0.577)|1.000 (_±_0.000)|1.982|(_±_0.126)|0.623 (_±_0.084)|
||VI: DiagonalNormal|0.866 (_±_0.085)|0.954(_±_1.022)|0.651 (_±_0.549)|0.810 (_±_0.036)|0.441|(_±_0.252)|0.384 (_±_0.089)|
|Si 5|VI: MultivariateNormal|0.765 (_±_0.100)|0.537(_±_1.019)|0.633 (_±_1.067)|0.711 (_±_0.038)|0.148|(_±_0.093)|**0.279**(_±_0.056)|
|cenaro|VI: Structured Normal|0.758 (_±_0.098)|0.447(_±_0.818)|0.572 (_±_0.816)|0.705 (_±_0.032)|0.140|(_±_0.081)|**0.269**(_±_0.045)|
||VI: IAF|0.814 (_±_0.105)|<br>0.953(_±_1.165)|0.881 (_±_1.067)|0.777 (_±_0.106)|<br>0.684|<br>(_±_0.939)|0.625 (_±_0.525)|
||**ICL (ours)**|**0.621**(_±_0.063)|**0.067**(_±_0.080)|**0.299**(_±_0.195)|**0.610**(_±_0.045)|**0.046**|(_±_0.020)|**0.242**(_±_0.038)|
||Laplace Approximation|1.000 (_±_0.000)|2.026(_±_0.027)|1.612 (_±_0.162)|1.000 (_±_0.000)|1.993|(_±_0.032)|1.299 (_±_0.106)|
||VI: DiagonalNormal|0.724 (_±_0.060)|0.185(_±_0.082)|**0.787**(_±_0.078)|0.703 (_±_0.039)|0.147|(_±_0.063)|0.637 (_±_0.089)|
|Si 6|VI: MultivariateNormal|**0.534**(_±_0.018)|**0.014**(_±_0.006)|**0.581**(_±_0.074)|**0.538**(_±_0.019)|**0.016**|(_±_0.007)|**0.466**(_±_0.029)|
|cenaro|VI: Structured Normal|**0.536**(_±_0.016)|**0.014**(_±_0.005)|**0.583**(_±_0.071)|**0.536**(_±_0.019)|**0.017**|(_±_0.009)|**0.469**(_±_0.033)|
||VI: IAF|0.542 (_±_0.026)|<br>0.031(_±_0.031)|0.613 (_±_0.092)|**0.535**(_±_0.015)|<br>**0.015**|<br>(_±_0.006)|**0.467**(_±_0.031)|
||**ICL (ours)**|<br>**0.532**(_±_0.019)|<br>0.016(_±_0.008)|<br>**0.590**(_±_0.066)|<br>0.556 (_±_0.017)|<br>0.035|<br>(_±_0.015)|<br>**0.504**(_±_0.038)|
||Laplace Approximation|1.000 (_±_0.000)|3.559(_±_1.933)|1.347 (_±_1.067)|1.000 (_±_0.000)|2.016|(_±_0.080)|0.763 (_±_0.174)|
||<br>VI: DiagonalNormal|<br>0.938 (_±_0.074)|<br>2.536(_±_2.097)|<br>1.142 (_±_0.993)|<br>0.936 (_±_0.024)|<br>1.029|<br>(_±_0.255)|<br>0.579 (_±_0.181)|
||VI: MultivariateNormal|0.814 (_±_0.181)|1.999(_±_2.283)|1.033 (_±_0.969)|**0.741**(_±_0.020)|0.093|(_±_0.025)|**0.391**(_±_0.074)|
|Scenario 7|VI: Structured Normal|0.824 (_±_0.177)|<br>1.891(_±_2.127)|1.041 (_±_0.934)|**0.734**(_±_0.025)|<br>**0.072**|<br>(_±_0.019)|**0.385**(_±_0.065)|
||VI: IAF|0.939 (_±_0.091)|2.707(_±_1.712)|1.590 (_±_0.820)|0.864 (_±_0.093)|0.830|(_±_0.697)|1.064 (_±_0.616)|
||**ICL (ours)**|**0.700**(_±_0.116)|**0.317**(_±_0.355)|**0.400**(_±_0.286)|0.773 (_±_0.048)|**0.294**|(_±_0.457)|0.559 (_±_0.256)|



25 

**Can Transformers Learn Full Bayesian Inference In Context?** 

## **I.2. Factor Analysis** 

Table 46 contains detailed results regarding FA for 50 synthetic and 17 real-world datasets across 6 different scenarios. We find that overall the ICL method has a very high agreement with the gold standard HMC reference with scores of more than than 56 percent in five scenarios on the synthetic data. In comparison, the C2ST metric is almost saturated for all considered VI methods. For MMD and _W_ 2 the ICL method is again the best. 

The real-world datasets show a similar picture except for scenario 4 where C2ST and MMD indicate that VI with inverse autoregressive flows performs best. The _W_ 2 metric, however exhibits a relatively large variance in those cases and does not yield significant results regarding the best performance. 

Table 13: Factor Analysis: Evaluation on 50 synthetic and 17 real-world datasets for six different scenarios. All results within two standard errors of the best average result for each scenario are marked in **bold** . 

|**Scenario**|**Model**||**Synthetic Evaluatio**|**n**|**R**|**eal-Wor**|**ld Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD|(_↓_)|_W_2(_↓_)|
||Laplace Approximation|1.000 (_±_0.000)|3.459(_±_1.553)|1.987 (_±_1.363)|1.000 (_±_0.000)|2.487|(_±_0.454)|**0.875**(_±_0.036)|
||VI: DiagonalNormal|1.000 (_±_0.001)|4.695(_±_1.488)|2.865 (_±_1.681)|0.979 (_±_0.008)|1.283|(_±_0.225)|**0.625**(_±_0.058)|
||VI: MultivariateNormal|0.998 (_±_0.003)|4.163(_±_1.473)|2.603 (_±_1.959)|0.966 (_±_0.010)|1.213|(_±_0.260)|**0.608**(_±_0.047)|
|Scenario 1|VI: Structured Normal|0.997 (_±_0.004)|<br>4.655(_±_1.189)|2.700 (_±_1.333)|0.979 (_±_0.010)|<br>1.231|<br>(_±_0.132)|**0.611**(_±_0.041)|
||VI: IAF|0.953 (_±_0.104)|3.992(_±_2.089)|2.750 (_±_1.838)|0.849 (_±_0.075)|0.772|(_±_0.335)|**0.503**(_±_0.063)|
||**ICL (ours)**|<br>**0.552**(_±_0.028)|<br>**0.034**(_±_0.034)|<br>**0.289**(_±_0.083)|<br>**0.606**(_±_0.038)|<br>**0.068**|<br>(_±_0.069)|<br>0.265 (_±_0.078)|
||Laplace Approximation|1.000 (_±_0.000)|3.687(_±_1.661)|1.954 (_±_1.129)|1.000 (_±_0.000)|1.690|(_±_0.182)|**0.598**(_±_0.058)|
||VI: DiagonalNormal|0.998 (_±_0.002)|3.135(_±_1.482)|1.629 (_±_0.938)|0.975 (_±_0.010)|1.156|(_±_0.068)|**0.496**(_±_0.052)|
||<br>VI: MultivariateNormal|<br>0.989 (_±_0.009)|<br>2.945(_±_1.019)|<br>1.482 (_±_0.683)|<br>0.951 (_±_0.025)|<br>0.764|<br>(_±_0.053)|<br>**0.421**(_±_0.052)|
|Scenario 2|VI: Structured Normal|0.984 (_±_0.031)|<br>3.790(_±_1.572)|2.106 (_±_1.429)|0.958 (_±_0.025)|<br>1.001|<br>(_±_0.126)|**0.465**(_±_0.056)|
||VI: IAF|0.966 (_±_0.066)|<br>3.523(_±_1.340)|2.153 (_±_0.968)|0.799 (_±_0.058)|<br>0.462|<br>(_±_0.226)|**0.342**(_±_0.070)|
||**ICL (ours)**|**0.542**(_±_0.006)|**0.017**(_±_0.006)|**0.244**(_±_0.033)|**0.622**(_±_0.032)|**0.098**|(_±_0.039)|**0.287**(_±_0.046)|
||Laplace Approximation|1.000 (_±_0.000)|4.137(_±_0.932)|2.188 (_±_1.011)|1.000 (_±_0.000)|3.653|(_±_0.183)|**0.473**(_±_0.026)|
||VI: DiagonalNormal|0.999 (_±_0.002)|3.339(_±_0.985)|1.722 (_±_0.870)|0.951 (_±_0.007)|1.114|(_±_0.080)|**0.245**(_±_0.016)|
||VI: MultivariateNormal|0.994 (_±_0.007)|3.189(_±_0.960)|1.644 (_±_0.859)|0.945 (_±_0.007)|1.085|(_±_0.082)|**0.242**(_±_0.015)|
|Scenario 3|VI: Structured Normal|0.997 (_±_0.003)|<br>3.159(_±_0.968)|1.614 (_±_0.793)|0.942 (_±_0.009)|<br>1.084|<br>(_±_0.071)|**0.242**(_±_0.018)|
||VI: IAF|0.990 (_±_0.011)|3.145(_±_1.203)|1.705 (_±_0.990)|0.928 (_±_0.015)|1.022|(_±_0.093)|**0.235**(_±_0.018)|
||**ICL (ours)**|**0.537**(_±_0.023)|<br>**0.024**(_±_0.021)|**0.259**(_±_0.088)|**0.609**(_±_0.019)|<br>**0.124**|<br>(_±_0.037)|**0.179**(_±_0.018)|
||Laplace Approximation|1.000 (_±_0.000)|4.354(_±_0.572)|3.339 (_±_0.932)|1.000 (_±_0.000)|6.617|(_±_0.259)|0.598 (_±_0.135)|
||<br>VI: DiagonalNormal|1.000 (_±_0.000)|<br>3.396(_±_0.591)|2.420 (_±_0.720)|0.977 (_±_0.003)|<br>1.499|<br>(_±_0.066)|**0.096**(_±_0.003)|
||VI: MultivariateNormal|0.999 (_±_0.001)|3.447(_±_0.567)|2.479 (_±_0.848)|0.973 (_±_0.008)|1.484|(_±_0.097)|0.096 (_±_0.005)|
|Scenario 4|VI: Structured Normal|1.000 (_±_0.000)|3.421(_±_0.610)|2.481 (_±_0.884)|0.973 (_±_0.007)|1.474|(_±_0.078)|**0.095**(_±_0.004)|
||VI: IAF|0.999 (_±_0.001)|<br>3.269(_±_0.552)|2.307 (_±_0.779)|**0.961**(_±_0.018)|<br>**1.337**|<br>(_±_0.142)|**0.092**(_±_0.005)|
||**ICL (ours)**|**0.684**(_±_0.060)|**0.198**(_±_0.141)|**0.918**(_±_0.246)|0.988 (_±_0.003)|1.764|(_±_0.026)|1.248 (_±_0.008)|
||Laplace Approximation|1.000 (_±_0.000)|4.456(_±_0.785)|2.608 (_±_0.946)|1.000 (_±_0.000)|4.559|(_±_0.494)|0.663 (_±_0.127)|
||VI: DiagonalNormal|0.999 (_±_0.002)|3.520(_±_1.073)|2.012 (_±_0.886)|0.944 (_±_0.010)|1.007|(_±_0.129)|**0.261**(_±_0.036)|
||VI: MultivariateNormal|0.995 (_±_0.007)|<br>3.472(_±_1.021)|1.982 (_±_0.814)|0.930 (_±_0.017)|<br>0.964|<br>(_±_0.111)|**0.255**(_±_0.038)|
|Scenario 5|VI: Structured Normal|0.998 (_±_0.005)|3.369(_±_1.044)|1.916 (_±_0.852)|0.934 (_±_0.011)|0.996|(_±_0.133)|**0.259**(_±_0.035)|
||VI: IAF|<br>0.992 (_±_0.012)|<br>3.166(_±_0.967)|<br>1.761 (_±_0.671)|<br>0.910 (_±_0.011)|<br>**0.892**|<br>(_±_0.094)|<br>**0.247**(_±_0.037)|
||**ICL (ours)**|**0.535**(_±_0.016)|**0.021**(_±_0.011)|**0.279**(_±_0.060)|**0.886**(_±_0.017)|1.207|(_±_0.101)|**1.002**(_±_0.042)|
||Laplace Approximation|1.000 (_±_0.000)|3.942(_±_0.971)|2.624 (_±_1.682)|1.000 (_±_0.000)|3.319|(_±_0.196)|**0.377**(_±_0.020)|
||<br>VI: DiagonalNormal|<br>0.998 (_±_0.002)|<br>3.214(_±_1.072)|<br>2.209 (_±_1.543)|<br>0.949 (_±_0.008)|<br>1.196|<br>(_±_0.093)|<br>**0.210**(_±_0.011)|
||VI: MultivariateNormal|0.991 (_±_0.013)|<br>3.056(_±_1.237)|2.189 (_±_1.698)|0.938 (_±_0.009)|<br>1.121|<br>(_±_0.075)|**0.205**(_±_0.012)|
|Scenario 6|VI: Structured Normal|<br>0.997 (_±_0.005)|<br>3.279(_±_1.071)|<br>2.276 (_±_1.787)|<br>0.944 (_±_0.006)|<br>1.161|<br>(_±_0.066)|<br>**0.208**(_±_0.012)|
||VI: IAF|0.989 (_±_0.029)|3.027(_±_0.910)|1.936 (_±_1.060)|0.865 (_±_0.027)|0.822|(_±_0.106)|**0.179**(_±_0.015)|
||**ICL (ours)**|**0.543**(_±_0.021)|<br>**0.023**(_±_0.015)|**0.345**(_±_0.173)|**0.666**(_±_0.020)|<br>**0.200**|<br>(_±_0.034)|**0.224**(_±_0.014)|



26 

**Can Transformers Learn Full Bayesian Inference In Context?** 

## **I.3. Gaussian Mixture Models** 

We summarize the results of the ICL approach and the different VI methods regarding the GMM scenarios in Table 47. First, one can note that on the synthetic data, the ICL approach has a much lower C2ST score for scenario 1 and scenario 2 than the other methods. However, for scenarios 3 and 4, C2ST saturates, or at least almost saturates for all approaches. The MMD metric, however, shows that ICL not only has a high agreement with HMC in scenarios 1 and 2, but that it attains the significantly best result in scenarios 3 and 4 as well. This is supported by the _W_ 2 metric, which has the significantly lowest values for ICL in scenarios 2,3 and 4. 

Analogously, on the real-world data, MMD shows that ICL is the best approach in all four scenarios without any other model coming into the two standard-deviation range. While the C2ST score is the lowest in scenario 1 and scenario 2 for ICL, it saturates for cases 3 and 4. 

Table 14: Gaussian Mixture Models: Evaluation on 50 synthetic and 17 real-world datasets for six different scenarios. All results within two standard errors of the best average result for each scenario are marked in **bold** . 

|**Si**|**Mdl**||**Synthet**|**ic Evaluati**|**on**|**R**|**eal-World Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|---|
|**cenaro**|**oe**|C2ST (_↓_)|MMD|(_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
||Laplace Approximation|1.000 (_±_0.000)|3.367(|_±_1.030)|**4.341**(_±_2.018)|1.000 (_±_0.000)|3.374(_±_0.941)|**6.440**(_±_1.994)|
||VI: DiagonalNormal|0.988 (_±_0.013)|1.175(|_±_1.189)|**2.961**(_±_1.669)|0.995 (_±_0.006)|1.919(_±_1.217)|**5.145**(_±_2.489)|
||VI: MultivariateNormal|0.988 (_±_0.013)|1.135(|_±_1.149)|**2.926**(_±_1.651)|0.994 (_±_0.007)|2.007(_±_1.367)|**5.379**(_±_2.845)|
|Scenario 1|VI: Structured Normal|0.987 (_±_0.015)|1.126(|_±_1.145)|**2.944**(_±_1.663)|0.993 (_±_0.009)|1.943(_±_1.359)|**5.313**(_±_2.737)|
||VI: IAF|0.989 (_±_0.013)|<br>1.017(|<br> _±_1.036)|**3.104**(_±_1.523)|0.995 (_±_0.010)|<br>1.888(_±_1.051)|**5.402**(_±_2.310)|
||**ICL (ours)**|**0.760 (**_±_**0.092)**|**0.303**(|_±_0.548)|**2.095**(_±_1.692)|**0.847**(_±_0.082)|**0.486**(_±_0.623)|**4.054**(_±_2.782)|
||Laplace Approximation|1.000 (_±_0.000)|2.864(|_±_0.607)|5.407 (_±_2.320)|1.000 (_±_0.000)|2.928(_±_0.438)|**7.228**(_±_1.323)|
||VI: DiagonalNormal|0.989 (_±_0.024)|1.425(|_±_0.829)|4.933 (_±_2.379)|0.998 (_±_0.003)|1.525(_±_0.356)|**6.091**(_±_0.931)|
||VI: MultivariateNormal|0.991 (_±_0.021)|1.532(|_±_0.940)|5.119 (_±_2.521)|0.999 (_±_0.002)|1.619(_±_0.269)|**6.258**(_±_0.872)|
|Scenario 2|VI: Structured Normal|0.992 (_±_0.017)|<br>1.487(|<br> _±_0.899)|5.085 (_±_2.530)|0.999 (_±_0.002)|<br>1.580(_±_0.337)|**6.241**(_±_0.960)|
||VI: IAF|0.992 (_±_0.021)|1.319(|_±_0.854)|5.265 (_±_2.534)|0.998 (_±_0.004)|1.256(_±_0.320)|**6.201**(_±_0.892)|
||**ICL (ours)**|**0.812 (**_±_**0.061)**|**0.159**(|_±_0.154)|**2.314**(_±_0.926)|**0.937**(_±_0.041)|**0.282**(_±_0.131)|**3.947**(_±_1.055)|
||Laplace Approximation|1.000 (_±_0.000)|3.631(|_±_1.362)|16.387 (_±_19.604)|1.000 (_±_0.000)|3.009(_±_0.768)|**37.034**(_±_7.178)|
||<br>VI: DiagonalNormal|**0.996**(_±_0.011)|<br>2.127(|<br> _±_1.479)|16.864 (_±_19.301)|**0.992**(_±_0.018)|<br>2.429(_±_0.516)|**35.355**(_±_6.608)|
||VI: MultivariateNormal|0.997 (_±_0.009)|2.076(|_±_1.388)|16.938 (_±_19.636)|**0.993**(_±_0.016)|2.427(_±_0.510)|**35.312**(_±_6.655)|
|Scenario 3|VI: Structured Normal|**0.995**(_±_0.017)|<br>2.049(|<br> _±_1.462)|16.723 (_±_19.093)|**0.993**(_±_0.016)|<br>2.301(_±_0.549)|**34.217**(_±_5.461)|
||VI: IAF|**0.994**(_±_0.018)|1.675(|_±_1.049)|14.311 (_±_9.266)|**0.993**(_±_0.017)|2.148(_±_0.528)|**34.336**(_±_5.398)|
||**ICL (ours)**|1.000 (_±_0.000)|**0.582**(|_±_0.280)|**8.708**(_±_4.945)|1.000 (_±_0.000)|**1.869**(_±_0.342)|**33.230**(_±_8.095)|
||Laplace Approximation|1.000 (_±_0.000)|6.260(|_±_1.427)|13.497 (_±_29.702)|1.000 (_±_0.000)|5.924(_±_1.145)|**12.400**(_±_4.313)|
||VI: DiagonalNormal|**1.000**(_±_0.002)|3.958(|_±_1.641)|12.068 (_±_21.301)|1.000 (_±_0.000)|3.879(_±_1.061)|**11.080**(_±_3.341)|
||<br>VI: MultivariateNormal|**1.000**(_±_0.002)|<br>3.875(|<br> _±_1.691)|12.150 (_±_22.198)|1.000 (_±_0.000)|<br>3.896(_±_1.057)|**11.112**(_±_3.321)|
|Scenario 4|VI: Structured Normal|**1.000**(_±_0.001)|<br>3.661(|<br> _±_1.717)|12.195 (_±_22.874)|**0.996**(_±_0.016)|<br>3.822(_±_1.302)|**11.368**(_±_4.216)|
||VI: IAF|**1.000**(_±_0.002)|3.536(|_±_1.597)|12.015 (_±_20.884)|1.000 (_±_0.000)|3.471(_±_1.036)|**11.421**(_±_3.233)|
||**ICL (ours)**|1.000 (_±_0.000)|<br>**2.451**(|<br> _±_0.868)|**8.333**(_±_4.202)|**1.000**(_±_0.000)|<br>**2.518**(_±_**0.694)**|**11.938**(_±_2.956)|



27 

**Can Transformers Learn Full Bayesian Inference In Context?** 

# **J. Evaluating Predictive Performance** 

In this section, we discuss the results of our ICL approach in terms of predictive performance. In scenarios one to four, TabPFN gives the best overall performance, which is expected since it is not limited to the GLM structure. Besides that, the MAP approach obtains consistently the best results, while our ICL method performs on par (scenarios 1,2,3) or better than (scenario 4) compared to the fully Bayesian methods on the real-world data. On the synthetic data there is no significant difference to the other fully Bayesian methods, except for the real-world data in scenario where HMC is clearly the best method. In scenario 5 (gamma prior on the regression coefficients), the in-context learner performs significantly worse than all other methods, while this difference is less pronounced in scenario 7. In scenario 6, TabPFN also has a substantially better performance than all other methods. The MAP approach performs on average better than all fully Bayesian methods, which themselves do not differ significantly. 

Table 15: Evaluating the predictive performance across 50 synthetic and 17 real-world datasets for scenarios 1-4 in terms of Root Mean Squared Error (RMSE). The best result among all fully Bayesian methods is marked in **bold** . For the fully Bayesian approaches, we use the posterior mean as a point estimate for the response. MAP denotes the predictive performance of the model with the maximum a posteriori estimate for the latents. 

|**Scenario**|**Model**|**RMSE Real-World**(_↓_)|**RMSE Synthetic**(_↓_)|
|---|---|---|---|
||HMC|**0.591**(_±_0.023)|**0.510**(_±_0.040)|
||Laplace Approximation|**0.594**(_±_0.023)|**0.510**(_±_0.040)|
||VI: DiagonalNormal|**0.591**(_±_0.023)|**0.509**(_±_0.040)|
||VI: MultivariateNormal|**0.591**(_±_0.023)|**0.510**(_±_0.040)|
|Scenario 1|VI: Structured Normal|**0.629**(_±_0.017)|0.555 (_±_0.039)|
||VI: IAF|**0.593**(_±_0.023)|**0.510**(_±_0.040)|
||ICL (ours)|**0.593**(_±_0.020)|**0.524**(_±_0.038)|
||MAP|0.555 (_±_0.024)|0.491 (_±_0.038)|
||TabPFN|0.483 (_±_0.036)|0.453 (_±_0.036)|
||HMC|**0.559**(_±_0.023)|**0.556**(_±_0.049)|
||Laplace Approximation|**0.561**(_±_0.022)|**0.557**(_±_0.049)|
||VI: DiagonalNormal|**0.560**(_±_0.023)|**0.557**(_±_0.049)|
||VI: MultivariateNormal|**0.559**(_±_0.023)|**0.556**(_±_0.049)|
|Scenario 2|VI: Structured Normal|**0.604**(_±_0.016)|0.685 (_±_0.054)|
||VI: IAF|**0.563**(_±_0.023)|**0.557**(_±_0.049)|
||ICL (ours)|**0.561**(_±_0.019)|**0.653**(_±_0.049)|
||MAP|0.513 (_±_0.023)|0.522 (_±_0.048)|
||TabPFN|0.449 (_±_0.034)|0.498 (_±_0.047)|
||HMC|**0.684**(_±_0.027)|**0.512**(_±_0.040)|
||Laplace Approximation|**0.688**(_±_0.026)|0.516 (_±_0.040)|
||VI: DiagonalNormal|**0.686**(_±_0.027)|**0.513**(_±_0.040)|
||VI: MultivariateNormal|**0.685**(_±_0.027)|**0.512**(_±_0.040)|
|Scenario 3|VI: Structured Normal|**0.733**(_±_0.016)|0.607 (_±_0.043)|
||VI: IAF|**0.686**(_±_0.027)|**0.512**(_±_0.040)|
||ICL (ours)|**0.690**(_±_0.023)|**0.588**(_±_0.045)|
||MAP|0.646 (_±_0.028)|0.495 (_±_0.039)|
||TabPFN|0.556 (_±_0.041)|0.462 (_±_0.037)|
||HMC|**0.642**(_±_0.027)|**0.559**(_±_0.051)|
||Laplace Approximation|0.737 (_±_0.048)|2.457 (_±_0.493)|
||VI: DiagonalNormal|0.751 (_±_0.038)|2.046 (_±_0.399)|
||VI: MultivariateNormal|**0.690**(_±_0.037)|2.155 (_±_0.454)|
|Scenario 4|VI: Structured Normal|**0.686**(_±_0.015)|3.019 (_±_0.545)|
||VI: IAF|**0.643**(_±_0.027)|1.751 (_±_0.422)|
||ICL (ours)|**0.649**(_±_0.023)|1.464 (_±_0.151)|
||MAP|0.626 (_±_0.038)|2.377 (_±_0.529)|
||TabPFN|0.522 (_±_0.037)|0.496 (_±_0.047)|



28 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 16: Evaluating the predictive performance across 50 synthetic and 17 real-world datasets for scenarios 5 and 7 in terms of Root Mean Squared Error (RMSE). The best result among all fully Bayesian methods is marked in **bold** . For the fully Bayesian approaches, we use the posterior mean as a point estimate for the response. MAP denotes the predictive performance of the model with the maximum a posteriori estimate for the latents. 

|**Scenario**|**Model**|**RMSE Real-World**(_↓_)|**RMSE Synthetic**(_↓_)|
|---|---|---|---|
||HMC|**0.699**(_±_0.022)|**0.490**(_±_0.036)|
||Laplace Approximation|**0.699**(_±_0.022)|**0.491**(_±_0.036)|
||VI: DiagonalNormal|**0.702**(_±_0.022)|**0.491**(_±_0.036)|
||VI: MultivariateNormal|**0.698**(_±_0.021)|**0.491**(_±_0.036)|
|Scenario 5|VI: Structured Normal|1.507 (_±_0.089)|0.741 (_±_0.053)|
||VI: IAF|**0.699**(_±_0.022)|**0.490**(_±_0.036)|
||ICL (ours)|0.769 (_±_0.020)|0.701 (_±_0.049)|
||MAP|0.658 (_±_0.022)|0.471 (_±_0.035)|
||TabPFN|0.534 (_±_0.040)|0.442 (_±_0.035)|
||HMC|**0.953**(_±_0.015)|**0.719**(_±_0.041)|
||Laplace Approximation|**0.950**(_±_0.016)|**0.719**(_±_0.041)|
||VI: DiagonalNormal|**0.954**(_±_0.015)|**0.718**(_±_0.041)|
||VI: MultivariateNormal|**0.953**(_±_0.015)|**0.718**(_±_0.041)|
|Scenario 7|VI: Structured Normal|1.082 (_±_0.026)|1.028 (_±_0.118)|
||VI: IAF|**0.954**(_±_0.014)|**0.720**(_±_0.041)|
||ICL (ours)|1.019 (_±_0.017)|**0.765**(_±_0.041)|
||MAP|0.945 (_±_0.017)|0.686 (_±_0.048)|
||TabPFN|0.817 (_±_0.040)|0.654 (_±_0.039)|



Table 17: Evaluating the predictive performance across 50 synthetic and 17 real-world datasets for scenarios 5 and 7 in terms of accuracy (Acc.). The best result among all fully Bayesian methods is marked in **bold** . For the fully Bayesian approaches, we use the posterior mean as a point estimate for the response. MAP denotes the predictive performance of the model with the maximum a posteriori estimate for the latents. 

|**Scenario**|**Model**|**Acc. Real-World**(_↑_)|**Acc. Synthetic**(_↑_)|
|---|---|---|---|
||HMC|**0.694**(_±_0.028)|**0.546**(_±_0.015)|
||Laplace Approximation|**0.692**(_±_0.027)|**0.547**(_±_0.015)|
||VI: DiagonalNormal|**0.700**(_±_0.028)|**0.546**(_±_0.015)|
||VI: MultivariateNormal|**0.691**(_±_0.029)|**0.546**(_±_0.015)|
|Scenario 6|VI: Structured Normal|**0.686**(_±_0.028)|**0.546**(_±_0.015)|
||VI: IAF|**0.689**(_±_0.029)|**0.545**(_±_0.015)|
||ICL (ours)|**0.688**(_±_0.027)|**0.545**(_±_0.015)|
||MAP|0.723 (_±_0.025)|0.610 (_±_0.016)|
||TabPFN|0.862 (_±_0.021)|0.673 (_±_0.011)|



29 

**Can Transformers Learn Full Bayesian Inference In Context?** 

# **K. Ablation: Using a Gaussian Approximation** 

In this section, we present results on using a Gaussian approximation instead of Flow Matching to parameterize the approximation of the posterior. 

The key takeaway from these results is that the in-context learning approach performs substantially better with Flow Matching (Lipman et al., 2022) than when a Gaussian approximation of the posterior is employed. 

Table 18: Generalized Linear Models: Comparing the in-context learner with a Gaussian approximation, fitted via the forward KL-divergence, to the proposed flow matching method. Evaluation on 50 synthetic and 17 real-world datasets for seven different scenarios. If one method is by more than two standard errors better than the other, it is marked in **bold** . Overall, the ICL + Flow Matching method clearly outperforms the Gaussian approximation, fitted via the forward KL-divergence,: it yields significantly better results (according to the two-standard-error criterion) in 6 out of 7 scenarios on synthetic datasets and in all 7 scenarios on real-world datasets, across at least two of the three considered metrics (C2ST, MMD, or _W_ 2). In addition, the flow matching method consistently achieves lower or comparable standard errors, indicating more stable and reliable performance across datasets. 

|**Si**|**Mdl**|**S**|**ynthetic Evaluatio**|**n**|**R**|**eal-World Evaluati**|**on**|
|---|---|---|---|---|---|---|---|
|**cenaro**|**oe**|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Snri 1|ICL + Gaussian|**0.845**(_±_0.213)|**1.601**(_±_1.213)|2.024 (_±_0.874)|0.980 (_±_0.007)|1.715(_±_0.295)|1.976 (_±_0.238)|
|ceao|**ICL + Flow Matching**|**0.765**(_±_0.123)|**0.767**(_±_0.727)|**0.585**(_±_0.301)|**0.614**(_±_0.074)|**0.175**(_±_0.219)|**0.310**(_±_0.138)|
|Scenario 2|ICL + Gaussian<br>**ICL + Flow Matching**|**0.941**(_±_0.056)<br>**0.839**(_±_0.072)|**1.000**(_±_0.953)<br>**0.707**(_±_0.658)|1.943 (_±_0.657)<br>**1.111**(_±_0.300)|0.969 (_±_0.013)<br>**0.768**(_±_0.033)|1.490(_±_0.310)<br>**0.143**(_±_0.089)|2.068 (_±_0.259)<br>**0.411**(_±_0.094)|
|Scenario 3|ICL + Gaussian<br>**ICL + Flow Matching**|0.907 (_±_0.138)<br>**0.611**(_±_0.070)|1.779(_±_1.363)<br>**0.089**(_±_0.114)|4.713 (_±_1.560)<br>**0.423**(_±_0.348)|0.985 (_±_0.006)<br>**0.576**(_±_0.027)|1.526(_±_0.198)<br>**0.037**(_±_0.026)|4.144 (_±_0.438)<br>**0.257**(_±_0.044)|
|Scenario 4|ICL + Gaussian<br>**ICL + Flow Matching**|0.989 (_±_0.011)<br>**0.753**(_±_0.049)|3.544(_±_0.343)<br>**0.171**(_±_0.153)|23.035 (_±_6.549)<br>**0.631**(_±_0.294)|0.990 (_±_0.003)<br>**0.762**(_±_0.015)|3.858(_±_0.061)<br>**0.105**(_±_0.046)|13.601 (_±_0.427)<br>**0.597**(_±_0.104)|
|Scenario 5|ICL + Gaussian<br>**ICL + Flow Matching**|0.962 (_±_0.037)<br>**0.621**(_±_0.063)|1.444(_±_1.640)<br>**0.067**(_±_0.080)|3.299 (_±_1.614)<br>**0.299**(_±_0.195)|0.991 (_±_0.005)<br>**0.610**(_±_0.045)|1.666(_±_0.387)<br>**0.046**(_±_0.020)|2.963 (_±_0.239)<br>**0.242**(_±_0.038)|
|Scenario 6|ICL + Gaussian<br>**ICL + Flow Matching**|0.909 (_±_0.048)<br>**0.532**(_±_0.019)|1.020(_±_0.505)<br>**0.016**(_±_0.008)|1.515 (_±_0.358)<br>**0.590**(_±_0.066)|0.939 (_±_0.047)<br>**0.556**(_±_0.017)|1.799(_±_0.751)<br>**0.035**(_±_0.015)|1.904 (_±_0.541)<br>**0.504**(_±_0.038)|
|Si 7|ICL + Gaussian|0.970 (_±_0.030)|2.169(_±_1.473)|1.707 (_±_0.480)|0.993 (_±_0.006)|2.390(_±_0.414)|1.362 (_±_0.152)|
|cenaro|**ICL + Flow Matching**|**0.700**(_±_0.116)|**0.317**(_±_0.355)|**0.400**(_±_0.286)|**0.773**(_±_0.048)|**0.294**(_±_0.457)|**0.559**(_±_0.256)|



30 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 19: Factor Analysis: Comparing the in-context learner with a Gaussian approximation, fitted via the forward KLdivergence, to the proposed flow matching method. Evaluation on 50 synthetic and 17 real-world datasets for seven different scenarios. If one method is by more than two standard errors better than the other, it is marked in **bold** . The flow matching approach shows favorable performance in the majority of cases. Specifically, it achieves statistically significant improvements in all 6 scenarios on synthetic data and in 5 out of 6 scenarios on real-world data. Notably, it often reduces discrepancy measures such as MMD and Wasserstein-2 distance by a large margin. In addition, the variability of the flow matching estimates is generally lower, leading to more reliable and consistent results across different datasets. In scenario 4, the Gaussian in-context learner learned a singular covariance matrix. 

|**Scenario**|**Model**|**S**<br>|**ynthetic Evaluatio**<br>|**n**<br>|**R**<br>|**eal-World Evaluati**<br>|**on**<br>|
|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Scenario 1|ICL + Gaussian<br>**ICL + Flow Matching**|0.974 (_±_0.028)<br>**0.552**(_±_0.028)|1.838(_±_0.778)<br>**0.034**(_±_0.034)|1.450 (_±_0.607)<br>**0.289**(_±_0.083)|**0.589**(_±_0.015)<br>**0.606**(_±_0.038)|**0.080**(_±_0.010)<br>**0.068**(_±_0.069)|0.459 (_±_0.017)<br>**0.265**(_±_0.078)|
|Scenario 2|ICL + Gaussian<br>**ICL + Flow Matching**|0.835 (_±_0.040)<br>**0.542**(_±_0.006)|0.813(_±_0.276)<br>**0.017**(_±_0.006)|1.250 (_±_0.316)<br>**0.244**(_±_0.033)|0.889 (_±_0.027)<br>**0.622**(_±_0.032)|0.778(_±_0.109)<br>**0.098**(_±_0.039)|1.074 (_±_0.073)<br>**0.287**(_±_0.046)|
|Scenario 3|ICL + Gaussian<br>**ICL + Flow Matching**|0.826 (_±_0.035)<br>**0.537**(_±_0.023)|0.826(_±_0.226)<br>**0.024**(_±_0.021)|1.210 (_±_0.239)<br>**0.259**(_±_0.088)|0.942 (_±_0.008)<br>**0.609**(_±_0.019)|1.466(_±_0.078)<br>**0.124**(_±_0.037)|1.317 (_±_0.038)<br>**0.179**(_±_0.018)|
|Scenario 4|ICL + Gaussian<br>**ICL + Flow Matching**|0.870 (_±_0.043)<br>**0.684**(_±_0.060)|0.706(_±_0.218)<br>**0.198**(_±_0.141)|1.635 (_±_0.297)<br>**0.918**(_±_0.246)|0.999 (_±_0.001)<br>**0.988**(_±_0.003)|2.025(_±_0.017)<br>**1.764**(_±_0.026)|2.013 (_±_0.019)<br>**1.248**(_±_0.008)|
|Scenario 5|ICL + Gaussian<br>**ICL + Flow Matching**|0.838 (_±_0.029)<br>**0.535**(_±_0.016)|0.831(_±_0.219)<br>**0.021**(_±_0.011)|1.248 (_±_0.249)<br>**0.279**(_±_0.060)|0.944 (_±_0.009)<br>**0.886**(_±_0.017)|1.477(_±_0.073)<br>**1.207**(_±_0.101)|1.316 (_±_0.031)<br>**1.002**(_±_0.042)|
|Scenario 6|ICL + Gaussian<br>**ICL + Flow Matching**|0.837 (_±_0.030)<br>**0.543**(_±_0.021)|0.831(_±_0.219)<br>**0.023**(_±_0.015)|1.248 (_±_0.249)<br>**0.345**(_±_0.173)|0.944 (_±_0.008)<br>**0.666**(_±_0.020)|1.477(_±_0.073)<br>**0.200**(_±_0.034)|1.316 (_±_0.031)<br>**0.224**(_±_0.014)|



Table 20: Gaussian Mixture Models: Comparing the in-context learner with a Gaussian approximation, fitted via the forward KL-divergence, to the proposed flow matching method. Evaluation on 50 synthetic and 17 real-world datasets for seven different scenarios. If one method is by more than two standard errors better than the other, it is marked in **bold** . While the differences are less clear-cut than in the previous models, ICL + Flow Matching demonstrates favorable performance in several scenarios, particularly for the Wasserstein-2 distance and MMD. Notably, its advantage is most visible in lower-dimensional settings (Scenario 1 and 2), where it consistently improves upon the Gaussian approximation, fitted via the forward KL-divergence, across most metrics. However, as the dimensionality increases, the performance gap tends to narrow, and in some cases, the inherent variability of the datasets, especially for the Gaussian approximation, fitted via the forward KL-divergence,, makes it difficult to conclusively determine a clear winner. Nonetheless, the flow matching approach often achieves smaller standard errors and lower discrepancy measures, underlining its potential for more stable modeling. 

|**Scenario**|**Model**||**Synthetic Evaluat**|**ion**|**R**|**eal-World Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Scenario 1|ICL + Gaussian<br>**ICL + Flow Matching**|**0.926**(_±_0.029)<br>**0.760**(_±_0.092)|**0.555**(_±_0.452)<br>**0.303**(_±_0.548)|**2.586**(_±_0.560)<br>**2.095**(_±_1.692)|**0.957**(_±_0.034)<br>**0.847**(_±_0.082)|**0.765**(_±_0.958)<br>**0.486**(_±_0.623)|**3.717**(_±_1.709)<br>**4.054**(_±_2.782)|
|Scenario 2|ICL + Gaussian<br>**ICL + Flow Matching**|0.985 (_±_0.010)<br>**0.812**(_±_0.061)|0.761(_±_0.227)<br>**0.159**(_±_0.154)|5.022 (_±_0.945)<br>**2.314**(_±_0.926)|**0.999**(_±_0.001)<br>**0.937**(_±_0.041)|0.801(_±_0.256)<br>**0.282**(_±_0.131)|7.525 (_±_1.513)<br>**3.947**(_±_1.055)|
|Scenario 3|ICL + Gaussian<br>**ICL + Flow Matching**|**0.998**(_±_0.002)<br>**1.000**(_±_0.000)|**0.829**(_±_0.241)<br>**0.582**(_±_0.280)|**11.536**(_±_2.365)<br>**8.708**(_±_4.945)|**1.000**(_±_0.000)<br>**1.000**(_±_0.000)|**1.500**(_±_0.251)<br>**1.869**(_±_0.342)|**26.242**(_±_4.171)<br>**33.230**(_±_8.095)|
|Scenario 4|ICL + Gaussian|**0.998**(_±_0.001)|**6.314**(_±_0.449)|**13.404**(_±_0.609)|**0.997**(_±_0.001)|**2.770**(_±_1.201)|22.596 (_±_5.717)|
||**ICL + Flow Matching**|1.000 (_±_0.000)|**2.451**(_±_0.868)|**8.333**(_±_4.202)|1.000 (_±_0.000)|**2.518**(_±_0.694)|**11.938**(_±_2.956)|



31 

**Can Transformers Learn Full Bayesian Inference In Context?** 

# **L. Ablation: Using a Diffusion Objective** 

To validate choosing the flow matching objective with optimal transport (OT) paths resulting in the objective in equation Equation (7), we also conduct experiments using a diffusion-objective with variance preserving paths introduced by Song et al. (2020). We choose three selected GLM, FA and GMM scenarios with the same 50 synthetic and 17 real-world datasets for each scenario as in the other benchmarks. 

## **L.1. Diffusion with Flow-Matching** 

First, we use the diffusion objective learned via flow matching, as described in (Lipman et al., 2022), where we choose the same hyperparameters as (Lipman et al., 2022). 

Table 21: GLMs: Comparison of the OT flow matching and the VP diffusion objective on 50 synthetic and 17 real-world datasets for three different scenarios. All results within two standard errors of the best average result for each scenario are marked in **bold** . 

|**Scenario**|**Model**||**Synthetic Evaluati**|**on**|**R**|**eal-World Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Scenario 2|Diffusion paths + FM|0.961 (_±_0.040)|**1.525**(_±_0.777)|3.354 (_±_1.333)|0.961 (_±_0.016)|1.347(_±_0.365)|2.025 (_±_0.270)|
||**OT paths**|**0.839**(_±_0.072)|**0.707**(_±_0.658)|**1.111**(_±_0.300)|**0.768**(_±_0.033)|**0.143**(_±_0.089)|**0.411**(_±_0.094)|
|Si 3|Diffusion paths + FM|0.903 (_±_0.111)|1.080(_±_0.564)|1.733 (_±_0.408)|0.936 (_±_0.013)|1.002(_±_0.203)|1.442 (_±_0.103)|
|cenaro|**OT paths**|**0.611**(_±_0.070)|**0.089**(_±_0.114)|**0.423**(_±_0.348)|**0.576**(_±_0.027)|**0.037**(_±_0.026)|**0.257**(_±_0.044)|
|Si 5|Diffusion paths + FM|**0.691**(_±_0.074)|0.211(_±_0.143)|0.708 (_±_0.233)|**0.681**(_±_0.038)|0.182(_±_0.093)|**0.554 (**_±_**0.090)**|
|cenaro|**OT paths + FM**|**0.621**(_±_0.063)|**0.067**(_±_0.080)|**0.299**(_±_0.195)|**0.610**(_±_0.045)|**0.046**(_±_0.020)|**0.242**(_±_0.038)|



In summary, the empirical results demonstrate that using the OT paths consistently outperforms the VP diffusion objective across all scenarios for both GLMs and FAs. For GLMs, OT paths achieve significantly lower C2ST values in all scenarios. In Scenario 2, OT paths reduce C2ST from 0.961 to 0.839 on synthetic data and from 0.961 to 0.768 on real-world data. Similarly, in Scenario 3, OT paths achieve substantial improvements, with C2ST dropping from 0.903 to 0.611 on synthetic data and from 0.936 to 0.576 on real-world data. This trend is complemented by consistent improvements in other metrics such as _W_ 2, where OT paths often achieve reductions by over 50%. 

Table 22: FA: Comparison of the OT flow matching and the VP diffusion objective on 50 synthetic and 17 real-world datasets for three different scenarios. All results within two standard errors of the best average result for each scenario are marked in **bold** . 

|**Scenario**|**Model**||**Synthetic Evaluati**|**on**|**R**|**eal-World Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Si 1|Diffusion paths + FM|0.622 (_±_0.043)|0.207(_±_0.121)|0.692 (_±_0.192)|**0.595**(_±_0.012)|0.089(_±_0.011)|0.475 (_±_0.019)|
|cenaro|**OT paths + FM**|**0.552**(_±_0.028)|**0.034**(_±_0.034)|**0.289**(_±_0.083)|**0.606**(_±_0.038)|**0.068**(_±_0.069)|**0.265**(_±_0.078)|
|Scenario 2|Diffusion paths + FM|0.826 (_±_0.036)|0.768(_±_0.238)|1.219 (_±_0.276)|0.878 (_±_0.028)|0.793(_±_0.154)|1.056 (_±_0.084)|
||**OT paths + FM**|**0.542**(_±_0.006)|**0.017**(_±_0.006)|**0.244**(_±_0.033)|**0.622**(_±_0.032)|**0.098**(_±_0.039)|**0.287**(_±_0.046)|
|Snri 3|Diffusion paths + FM|0.751 (_±_0.048)|0.387(_±_0.216)|0.834 (_±_0.163)|0.944 (_±_0.008)|1.514(_±_0.056)|1.332 (_±_0.028)|
|ceao|**OT paths + FM**|**0.537**(_±_0.023)|**0.024**(_±_0.021)|**0.259**(_±_0.088)|**0.609**(_±_0.019)|**0.124**(_±_0.037)|**0.179**(_±_0.018)|



For FA, the performance gap in C2ST remains notable. In Scenario 1, OT paths achieve the best results on synthetic data, reducing C2ST from 0.622 to 0.552, while also delivering improvements in _W_ 2 (0.289 compared to 0.692). On real-world datasets, OT paths maintain competitive results, matching or exceeding the performance of diffusion paths. The advantage is even more pronounced in Scenario 2, where OT paths consistently lead across all metrics, with a particularly striking reduction in MMD on synthetic data (0.017 compared to 0.768) and strong results for C2ST on real-world data (0.622 vs. 0.878). Similarly, in Scenario 3, OT paths achieve the lowest C2ST values, with synthetic results improving from 0.751 to 0.537 and real-world results from 0.944 to 0.609. 

In the case of Gaussian Mixture Models (GMMs), the empirical results indicate that the OT paths generally outperform the VP diffusion objective across most scenarios and metrics, though the differences are not always statistically significant in pair-wise comparisons. For example, in Scenario 1, OT paths achieve notably better results for C2ST on both synthetic and real-world datasets, with reductions from 0.924 to 0.760 and from 0.958 to 0.847, respectively. Similarly, for _W_ 2, OT paths 

32 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 23: GMMs: Comparison of the OT flow matching and the VP diffusion objective on 50 synthetic and 17 real-world datasets for three different scenarios. All results within two standard errors of the best average result for each scenario are marked in **bold** . 

|**Scenario**|**Model**||**Synthetic Evaluat**|**ion**|**R**|**eal-World Evaluati**|**on**|
|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Scenario 1|Diffusion paths + FM|0.924 (_±_0.024)|**0.241**(_±_0.381)|**2.195**(_±_1.431)|0.958 (_±_0.030)|0.890(_±_0.912)|**5.328**(_±_2.544)|
||**OT paths + FM**|**0.760**(_±_0.092)|**0.303**(_±_0.548)|**2.095**(_±_1.692)|**0.847**(_±_0.082)|**0.486**(_±_0.623)|**4.054**(_±_2.782)|
|Scenario 2|Diffusion paths + FM|0.942 (_±_0.020)|**0.213**(_±_0.187)|**2.748**(_±_0.659)|**0.984**(_±_0.012)|**0.411**(_±_0.162)|**5.397**(_±_1.458)|
||**OT paths + FM**|**0.812**(_±_0.061)|**0.159**(_±_0.154)|**2.314**(_±_0.926)|**0.937**(_±_0.041)|**0.282**(_±_0.131)|**3.947**(_±_1.055)|
|Scenario 3|Diffusion paths + FM|**1.000**(_±_0.000)|0.582(_±_0.280)|**8.708**(_±_4.945)|**1.000**(_±_0.000)|1.869(_±_0.342)|**33.230**(_±_8.095)|
||**OT paths + FM**|**0.999**(_±_0.001)|**0.267**(_±_0.154)|**7.234**(_±_2.974)|**1.000**(_±_0.000)|**1.155**(_±_0.258)|**26.956**(_±_3.114)|





Figure 8: Marginal distribution for GLM scenario 2 (left) and GMM scenario 1 (right). The in-context learner is trained with a diffusion objective using VP paths. 

exhibit better performance on real-world data (4.054 vs. 5.328). In Scenario 2, OT paths maintain a consistent advantage in metrics such as C2ST and _W_ 2. For instance, synthetic data shows a C2ST improvement from 0.942 to 0.812, while real-world data improves from 0.984 to 0.937. The OT paths also achieve lower MMD on synthetic data (0.159 vs. 0.213), supporting their effectiveness in this scenario. For Scenario 3, OT paths achieve better results for _W_ 2 on both synthetic and real-world data, reducing it from 8.708 to 7.234 and from 33.230 to 26.956, respectively. 

## **L.2. Diffusion with Score-Matching** 

Second, we compare the results of using OT paths with flow matching to the results obtained when using VP paths and score matching. We use the score matching objective introduced by Song & Ermon (2019) and maintain the VP hyperparameters from Lipman et al. (2022) that we previously used for the diffusion objective with flow matching. 

We find that, across all three considered GLM scenarios, using OT paths and flow matching yields substantially better results than using Diffusion VP paths and score matching, where the score-matching objective sometimes yields results comparable to those obtained using a Laplace approximation. We observe similar overall results for FA and GMMs, although the effect is less pronounced. Note that the inferiority of score matching compared to flow matching is consistent with findings by Lipman et al. (2022) and Dax et al. (2024), who also report that flow matching produces more stable and less noisy training trajectories. 

The large quantity of noise in the diffusion objective might prevent the model from learning complex conditioning on datasets **_x_** , which is arguably the main challenge for performing in-context learning for the posteriors of latent variable 

33 

**Can Transformers Learn Full Bayesian Inference In Context?** 

models. We find visually that using the diffusion objective leads to a form of collapse where the model only learns a constant posterior distribution _Q_<sup>**_z_**</sup> _θ_<sup>_|_</sup><sup>**_x_**</sup> that has a relatively large variance and is centered around zero, while largely ignoring the conditioning on **_x_** (Please refer to figure 8). 

Table 24: GLMs: Comparison of the OT flow matching and the VP diffusion objective with score matching on 50 synthetic and 17 real-world datasets for three different scenarios. All results within two standard errors of the best average result for each scenario are marked in **bold** . 

|**Scenario**|**Model**||**Synthetic Evaluati**|**on**|**R**|**eal-World Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Scenario 2|Diffusion paths + SM<br>**OT paths + FM**|0.996 (_±_0.011)<br>**0.839**(_±_0.072)|4.121(_±_1.625)<br>**0.707**(_±_0.658)|8.761 (_±_4.415)<br>**1.111**(_±_0.300)|0.998 (_±_0.002)<br>**0.768**(_±_0.033)|1.574(_±_0.906)<br>**0.143**(_±_0.089)|8.483 (_±_1.580)<br>**0.411**(_±_0.094)|
|Si 3|Diffusion paths + SM|0.965 (_±_0.075)|2.466(_±_1.224)|3.947 (_±_1.323)|0.994 (_±_0.002)|2.018(_±_0.206)|3.301 (_±_0.260)|
|cenaro|**OT paths + FM**|**0.611**(_±_0.070)|**0.089**(_±_0.114)|**0.423**(_±_0.348)|**0.576**(_±_0.027)|**0.037**(_±_0.026)|**0.257**(_±_0.044)|
|Si 5|Diffusion paths + SM|0.998 (_±_0.002)|3.163(_±_0.651)|8.684 (_±_1.135)|0.999 (_±_0.001)|3.004(_±_0.056)|8.547 (_±_0.177)|
|cenaro|**OT paths + FM**|**0.621**(_±_0.063)|**0.067**(_±_0.080)|**0.299**(_±_0.195)|**0.610**(_±_0.045)|**0.046**(_±_0.020)|**0.242**(_±_0.038)|



Table 25: FA: Comparison of the OT flow matching and the VP diffusion objective with score matching on 50 synthetic and 17 real-world datasets for three different scenarios. All results within two standard errors of the best average result for each scenario are marked in **bold** . 

|**Scenario**|**Model**||**Synthetic Evaluati**|**on**|**R**|**eal-World Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Scenario 1|Diffusion paths + SM|0.880 (_±_0.024)|0.875(_±_0.134)|1.787 (_±_0.155)|0.906 (_±_0.007)|0.845(_±_0.026)|1.723 (_±_0.029)|
||**OT paths + FM**|**0.552**(_±_0.028)|**0.034**(_±_0.034)|**0.289**(_±_0.083)|**0.606**(_±_0.038)|**0.068**(_±_0.069)|**0.265**(_±_0.078)|
|Si 2|Diffusion paths + SM|0.932 (_±_0.022)|1.459(_±_0.128)|2.798 (_±_0.141)|0.980 (_±_0.008)|1.772(_±_0.065)|2.927 (_±_0.085)|
|cenaro|**OT paths + FM**|**0.542**(_±_0.006)|**0.017**(_±_0.006)|**0.244**(_±_0.033)|**0.622**(_±_0.032)|**0.098**(_±_0.039)|**0.287**(_±_0.046)|
|Si 3|Diffusion paths + SM|0.925 (_±_0.021)|1.747(_±_0.382)|3.028 (_±_0.646)|0.989 (_±_0.003)|2.101(_±_0.050)|2.882 (_±_0.050)|
|cenaro|**OT paths + FM**|**0.537**(_±_0.023)|**0.024**(_±_0.021)|**0.259**(_±_0.088)|**0.609**(_±_0.019)|**0.124**(_±_0.037)|**0.179**(_±_0.018)|



Table 26: GMMs: Comparison of the OT flow matching and the VP diffusion objective with score matching on 50 synthetic and 17 real-world datasets for three different scenarios. All results within two standard errors of the best average result for each scenario are marked in **bold** . 

|**Scenario**|**Model**||**Synthetic Evaluat**|**ion**|**R**|**eal-World Evaluati**|**on**|
|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Si 1|Diffusion paths + SM|1.000 (_±_0.001)|1.412(_±_0.365)|7.038 (_±_0.655)|0.998 (_±_0.002)|1.574(_±_0.906)|8.483 (_±_1.580)|
|cenaro|**OT paths + FM**|**0.760**(_±_0.092)|**0.303**(_±_0.548)|**2.095**(_±_1.692)|**0.847**(_±_0.082)|**0.486**(_±_0.623)|**4.054**(_±_2.782)|
|Si 2|Diffusion paths + SM|1.000 (_±_0.000)|1.275(_±_0.240)|6.621 (_±_1.091)|1.000 (_±_0.000)|1.032(_±_0.163)|7.931 (_±_0.748)|
|cenaro|**OT paths + FM**|**0.812**(_±_0.061)|**0.159**(_±_0.154)|**2.314**(_±_0.926)|**0.937**(_±_0.041)|**0.282**(_±_0.131)|**3.947**(_±_1.055)|
|Si 3|Diffusion paths + SM|**1.000**(_±_0.000)|1.337(_±_0.476)|**10.877**(_±_5.262)|**1.000**(_±_0.000)|2.277(_±_0.245)|**24.269**(_±_3.841)|
|cenaro|**OT paths + FM**|**0.999**(_±_0.001)|**0.267**(_±_0.154)|**7.234**(_±_2.974)|**1.000**(_±_0.000)|**1.155**(_±_0.258)|**26.956**(_±_3.114)|



34 

**Can Transformers Learn Full Bayesian Inference In Context?** 

# **M. Ablation: Using an MLP-based Encoder** 

To further justify choosing a transformer encoder in our ICL approach, we conduct an ablation study comparing the performance of our original ICL method with the performance obtained when the transformer encoder is replaced by an MLP with batch normalization (Ioffe, 2015) and skip-connections. To ensure a fair comparison, we use an MLP encoder with a hidden dimension of 1250 to give the overall model approximately the same number of parameters as in the transformer-based approach. Concretely, our MLP-approach has 43.3 million parameters compared to 43.1 million parameters with the transformer encoder. We choose three selected GLM, FA and GMM scenarios with 50 synthetic and 17 real-world datasets for each scenario. 

In summary, we find that the transformer encoder yields consistently better, results than the mlp encoder across all scenarios. While the difference is especially pronounced for the GLM scenarios, the difference become smaller for FA and GMMs. 

Table 27: GLMs: Comparison when using an MLP-based encoder and a transformer encoder on 50 synthetic and 17 real-world datasets for three different scenarios. 

|**Scenario**|**Te of Encoder**||**Synthetic Evaluati**|**on**|**R**|**eal-World Evaluati**|**on**|
|---|---|---|---|---|---|---|---|
||**yp**|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Scenario 2|MLP|0.942 (_±_0.093)|1.783(_±_1.048)|2.503 (_±_0.814)|0.968 (_±_0.012)|1.528(_±_0.394)|2.271 (_±_0.315)|
||**Transformer**|0.839 (_±_0.072)|0.707(_±_0.658)|1.111 (_±_0.300)|0.768 (_±_0.033)|0.143(_±_0.089)|0.411 (_±_0.094)|
|Snri 3|MLP|0.957 (_±_0.075)|2.236(_±_1.218)|2.681 (_±_1.130)|0.972 (_±_0.012)|1.658(_±_0.450)|2.076 (_±_0.427)|
|ceao|**Transformer**|0.611 (_±_0.070)|0.089(_±_0.114)|0.423 (_±_0.348)|0.576 (_±_0.027)|0.037(_±_0.026)|0.257 (_±_0.044)|
|Si 5|MLP|0.845 (_±_0.115)|1.066(_±_0.859)|1.166 (_±_0.996)|0.890 (_±_0.055)|1.223(_±_0.791)|1.102 (_±_0.383)|
|cenaro|**Transformer**|0.621 (_±_0.063)|0.067(_±_0.080)|0.299 (_±_0.195)|0.610 (_±_0.045)|0.046(_±_0.020)|0.242 (_±_0.038)|



In Table 27, the transformer encoder consistently outperforms the MLP encoder across all metrics and scenarios. In Scenario 2, C2ST drops from 0.942 (MLP) to 0.839 (Transformer) on synthetic data and from 0.968 to 0.768 on real-world data. Similarly, _W_ 2 improves significantly, decreasing from 2.503 to 1.111 on synthetic data and from 2.271 to 0.411 on real-world data. In Scenario 3, transformers achieve substantial improvements, reducing C2ST from 0.957 (MLP) to 0.611 on synthetic data and from 0.972 to 0.576 on real-world data. _W_ 2 also sees notable reductions, dropping from 2.681 to 0.423 on synthetic data and from 2.076 to 0.257 on real-world data. Finally, in Scenario 5, transformers maintain their superiority, achieving reductions in C2ST from 0.845 (MLP) to 0.621 on synthetic data and from 0.890 to 0.610 on real-world data. Improvements in _W_ 2 are similarly remarkable, with reductions from 1.166 to 0.299 on synthetic data and from 1.102 to 0.242 on real-world data. 

Table 28: FA: Comparison when using an MLP-based encoder and a transformer encoder on 50 synthetic and 17 real-world datasets for three different scenarios. 

|**Scenario**|**Te of Encoder**||**Synthetic Evaluati**|**on**|**R**|**eal-World Evaluati**|**on**|
|---|---|---|---|---|---|---|---|
||**yp**|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Snri 1|MLP|0.579 (_±_0.015)|0.017(_±_0.006)|0.364 (_±_0.029)|0.634 (_±_0.014)|0.013(_±_0.004)|0.331 (_±_0.010)|
|ceao|**Transformer**|0.552 (_±_0.028)|0.034(_±_0.034)|0.289 (_±_0.083)|0.606 (_±_0.038)|0.068(_±_0.069)|0.265 (_±_0.078)|
|Si 2|MLP|0.562 (_±_0.038)|0.037(_±_0.042)|0.308 (_±_0.097)|0.632 (_±_0.068)|0.182(_±_0.407)|0.339 (_±_0.174)|
|cenaro|**Transformer**|0.542 (_±_0.006)|0.017(_±_0.006)|0.244 (_±_0.033)|0.622 (_±_0.032)|0.098(_±_0.039)|0.287 (_±_0.046)|
|Scenario 3|MLP|0.539 (_±_0.025)|0.023(_±_0.022)|0.278 (_±_0.116)|0.680 (_±_0.019)|0.268(_±_0.044)|0.253 (_±_0.017)|
||**Transformer**|0.537 (_±_0.023)|0.024(_±_0.021)|0.259 (_±_0.088)|0.609 (_±_0.019)|0.124(_±_0.037)|0.179 (_±_0.018)|



For the factor analysis cases (Table 28), the transformer encoder still has better average performances even though the differences are substantially less pronounced than for the GLMs. In Scenario 1, transformers slightly outperform MLPs, reducing C2ST from 0.579 to 0.552 on synthetic data and from 0.634 to 0.606 on real-world data. _W_ 2 also sees moderate improvements, dropping from 0.364 to 0.289 on synthetic data and from 0.331 to 0.265 on real-world data. In Scenario 2, the advantage of the transformer encoder remains consistent, with C2ST decreasing from 0.562 (MLP) to 0.542 on synthetic data and from 0.632 to 0.622 on real-world data. _W_ 2 also improves slightly, dropping from 0.308 to 0.244 on synthetic data and from 0.339 to 0.287 on real-world data. Scenario 3 shows the smallest differences, where transformers marginally improve C2ST from 0.539 (MLP) to 0.537 on synthetic data and from 0.680 to 0.609 on real-world data. For _W_ 2, the reductions are minor but consistent, dropping from 0.278 to 0.259 on synthetic data and from 0.253 to 0.179 on real-world data. 

35 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 29: GMMs: Comparison when using an MLP-based encoder and a transformer encoder on 50 synthetic and 17 real-world datasets for three different scenarios. 

|**Scenario**|**Tpe of Encoder**||**Synthe**|**tic Evaluati**|**on**|**R**|**eal-World Evaluati**|**on**|
|---|---|---|---|---|---|---|---|---|
||**y**|C2ST (_↓_)|MMD|(_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Scenario 1|MLP|0.873 (_±_0.045)|0.242|(_±_0.363)|2.203 (_±_1.098)|0.917 (_±_0.067)|0.891(_±_1.150)|4.528 (_±_2.701)|
||**Transformer**|0.760 (_±_0.092)|0.303|(_±_0.548)|2.095 (_±_1.692)|0.847 (_±_0.082)|0.486(_±_0.623)|4.054 (_±_2.782)|
|Scenario 2|MLP|0.921 (_±_0.035)|0.291|(_±_0.205)|2.870 (_±_0.710)|0.992 (_±_0.005)|0.399(_±_0.127)|5.505 (_±_1.144)|
||**Transformer**|0.812 (_±_0.061)|0.159|(_±_0.154)|2.314 (_±_0.926)|0.937 (_±_0.041)|0.282(_±_0.131)|3.947 (_±_1.055)|
|Scenario 3|MLP|0.999 (_±_0.000)|0.438|(_±_0.181)|11.502 (_±_9.719)|1.000 (_±_0.000)|1.001(_±_0.149)|26.282 (_±_3.731)|
||**Transformer**|0.999 (_±_0.001)|0.267|(_±_0.154)|7.234 (_±_2.974)|1.000 (_±_0.000)|1.155(_±_0.258)|26.956 (_±_3.114)|



For the Gaussian Mixture Models (GMMs), the results indicate a more mixed performance where the transformer still performs slightly better (Table 29): In Scenario 1, transformer encoders slightly outperform MLPs on synthetic data, with C2ST improving from 0.873 (MLP) to 0.760 and _W_ 2 decreasing slightly from 2.203 to 2.095. However, on real-world data, MLPs perform marginally better in terms of MMD, reducing it from 0.486 to 0.242, while transformers show minor improvements in _W_ 2 from 4.528 to 4.054. In Scenario 2, transformers show a more noticeable advantage. On synthetic data, C2ST improves from 0.921 (MLP) to 0.812, and _W_ 2 decreases significantly from 2.870 to 2.314. On real-world data, transformers reduce C2ST from 0.992 to 0.937 and MMD from 0.399 to 0.282, along with a considerable improvement in _W_ 2 from 5.505 to 3.947. In Scenario 3, the differences between the two encoders are relatively small but still favor the transformers on synthetic data, with _W_ 2 decreasing from 11.502 (MLP) to 7.234. For real-world data, the results are nearly identical for C2ST (1.000 for both) but show a slight increase in _W_ 2 for the transformer from 26.282 to 26.956. Overall, for the GMMs, the transformer encoders demonstrate consistent improvements across scenarios for synthetic data, particularly in Scenarios 1 and 2. However, for real-world data, the performance differences are less pronounced. 

36 

**Can Transformers Learn Full Bayesian Inference In Context?** 



<!-- Start of picture text -->
1.00 1.6<br>6<br>0.95 1.4<br>5<br>1.2<br>0.90<br>1.0 4<br>0.85 0.8<br>3<br>0.80 0.6<br>2<br>0.4<br>0.75<br>0.56 0.60 0.64 0.68 0.72 0.76 0.80 0.84 0.88 0.56 0.60 0.64 0.68 0.72 0.76 0.80 0.84 0.88 0.56 0.60 0.64 0.68 0.72 0.76 0.80 0.84 0.88<br>Difference in data distribution (C2ST) Difference in data distribution (C2ST) Difference in data distribution (C2ST)<br>W2<br>C2ST MMD<br><!-- End of picture text -->

Figure 9: Out-of-distribution (OOD) performance of the ICL method in GLM Scenario 2. The _x_ -axis shows the distribution shift between training and test distributions, quantified by C2ST. The _y_ -axis displays the performance of the in-context learner, evaluated via C2ST, MMD, and _W_ 2 distances against HMC samples — where higher values indicate worse performance. OOD data is generated by gradually increasing the variance of the prior on the regression coefficients from an initial value of 1 _._ 0 to _√_ 2, 1 _._ 7, 2, 2 _._ 5, and 3 _._ 5. 

# **N. Robustness to Out-of-distribution Data** 

To investigate how our ICL approach behaves under mismatches between the distribution of synthetic training data and the data used to infer the posterior, we conduct an ablation study by changing aspects of the distribution of training and testing data. 

In summary, the results in Tables 31, 33 and 35 show that our ICL approach is, in most cases, capable of robustly generalizing beyond its specific pre-training distribution when various aspects of this distribution are changed. While the performance sometimes decreases when a mismatch between training and testing data occurs, the drops in performance are almost always modest and, in many cases, almost negligible. 

## **N.1. GLM Scenarios** 

For scenario 2, we change the variance of the prior on the covariates from a value of V( **_β_** _i,j_ ) = 1 to V( **_β_** _i,j_ ) = 2 for scenario 2.B and V( **_β_** _i,j_ ) = 4 for scenario 2.C. In scenarios 2.D and 2.E we change the scale parameter of the prior on the variance _σ_<sup>2</sup> of the noise—thereby changing its mean from E[ _σ_<sup>2</sup> ] = 0 _._ 5 to a value of E[ _σ_<sup>2</sup> ] _≈_ 0 _._ 7071 for 2.D and E[ _σ_<sup>2</sup> ] = 1 for 2.E. The variance is changed from V[ _σ_<sup>2</sup> ] _≈_ 0 _._ 0833 to V[ _σ_<sup>2</sup> ] _≈_ 0 _._ 1667 and V[ _σ_<sup>2</sup> ] _≈_ 0 _._ 333. 

For scenarios 3.B and 3.C, the variance of the coefficients is doubled from scenario 3 to scenario 3.B and from 3.B to 3.C again, analogously to scenarios 2.B and 2.C.0 

For scenario 5, the rate parameter of the gamma distribution is changed. This leads to a decrease in the variance from V( **_β_** _i,j_ ) = 1 to V( **_β_** _i,j_ ) = 0 _._ 5 for scenario 5.B and V( **_β_** _i,j_ ) = 0 _._ 25 for scenario 5.C. Notably, we also change the mean in the distribution of the covariates from mean from E[ **_β_** _i,j_ ] = 1 to a value of E[ **_β_** _i,j_ ] _≈_ 0 _._ 7071 for 2.D and E[ **_β_** _i,j_ ] = 0 _._ 5 for 2.E. 

Table 30 shows that our ICL approach only exhibits modest degradation in performance when the variance of the coefficients is doubled or quadruple while the mean stays the same (Scenarios 2.B, 2.C and 3.B, 3.C). Increasing the variance of the noise term by a factor of two only has a small effect while multiplying it by four causes a drop in C2ST by 9.3%. However, decreasing the variance of the gamma prior in scenario 5, combined with decreasing the mean, leads to a notable drop in performance across all metrics. 

## **N.2. FA Scenarios** 

To construct the mismatch between training and test distribution, we vary the variance of the factor loading _Wi,j,k_ for scenarios 1, 2 and 3. Concretely, the variance is doubled and quadrupled. 

For the FA cases (refer to Table 33), there is a notable drop in performance in the first scenario when OOD data is used. Please note that even in the most misspecified scenario (1.C), the performance, as measured in C2ST is still around ten 

37 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 30: Distribution of variables for the OOD analysis on GLM scenarios. 

|**Scenario**|_βi,j_|_βi,_0|_σ_<sup>2</sup><br>_i_|_yi,j|_(**_u_**_i,j,_ **_β_**_i, β_0_,i, σ_<sup>2</sup><br>_i_ <sup>)</sup><br><br>|
|---|---|---|---|---|
|Scenario 2|_N_(0_,_1)|_N_(0_,_9)|IG(5_,_2)|_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup><br><br>|
|Scenario 2.B|_N_(0_,_2)|_N_(0_,_9)|IG(5_,_2)|_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup><br>|
|Scenario 2.C|_N_(0_,_4)|_N_(0_,_9)|IG(5_,_2)|_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup>|
|Scenario 2.D|_N_(0_,_1)|_N_(0_,_9)|IG(5_,_2<br>_√_<br>2)|_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup>|
|Scenario 2.E|_N_(0_,_1)|_N_(0_,_9)|IG(5_,_4)|_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup>|
|Scenario 3|Laplace(0_,_1)|-|IG(5_,_2)|_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup>|
|Scenario 3.B|Laplace(0_,_<br>_√_<br>2)|-|IG(5_,_2)|_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup><br>|
|Scenario 3.C|Laplace(0_,_2)|-|IG(5_,_2)|_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup>|
|Scenario 5|Ga(1_,_1)|-|IG(5_,_2)|_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup>|
|Scenario 5.B|Ga(1_,_<br>_√_<br>2)|-|IG(5_,_2)|_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup>|
|Scenario 5.C|Ga(1_,_2)|-|IG(5_,_2)|_N_(**_u_**<sup>_⊤_</sup><br>_i,j_<sup>**_β_**</sup><sup>_i, σ_2</sup><br>_i_ <sup>)</sup>|



Table 31: OOD Performance: Evaluation on 50 synthetic datasets for 8 different GLM scenarios. All results within two standard errors of the non-OOD result for each scenario are marked in **bold** . 

|**Scenario**|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|---|---|---|---|
|Scenario 2|**0.839**(_±_0.072)|**0.707**(_±_0.658)|**1.111**(_±_0.300)|
|Scenario 2.B|**0.809**(_±_0.055)|**0.410**(_±_0.095)|**2.250**(_±_0.916)|
|Scenario 2.C|**0.857**(_±_0.105)|**0.634**(_±_0.318)|**3.067**(_±_1.759)|
|Scenario 2|**0.839**(_±_0.072)|**0.707**(_±_0.658)|**1.111**(_±_0.300)|
|Scenario 2.D|**0.840**(_±_0.109)|**0.916**(_±_1.123)|**4.007**(_±_3.261)|
|Scenario 2.E|**0.932**(_±_0.120)|**1.556**(_±_1.127)|**4.850**(_±_2.261)|
|Scenario 3|**0.611**(_±_0.070)|**0.089**(_±_0.114)|**0.423**(_±_0.348)|
|Scenario 3.B|**0.667**(_±_0.080)|**0.210**(_±_0.117)|1.172 (_±_0.258)|
|Scenario 3.C|**0.720**(_±_0.108)|**0.362**(_±_0.248)|1.891 (_±_0.678)|
|Scenario 5|**0.621**(_±_0.063)|**0.067**(_±_0.080)|**0.299**(_±_0.195)|
|Scenario 5.B|0.831 (_±_0.121)|0.479(_±_0.200)|1.762 (_±_0.541)|
|Scenario 5.C|0.920 (_±_0.064)|0.753(_±_0.424)|3.159 (_±_1.254)|



percent better than the best VI method in this scenario (Table 46). While the absolute difference between performance on the training distribution and the test distribution is very small for scenarios 2 and 3, the difference is still not within two standard errors of the non-OOD performance because the standard error itself is quite small. The performance on the OOD data is still better than all other VI methods (see Table 3). 

## **N.3. GMM Scenarios** 

To generate several distinct OOD scenarios based on the generative processes of GMMs, we vary scenario 2 in various ways. Note that the structure of the distributions is the same for all GMM scenarios—focusing on this specific scenario thus makes sense when considering OOD generalization. First, in scenario 2.B, we decrease the symmetric parameter of the Dirichlet prior on the assignments from 1 to 0.5 causing larger discrepancy in the number of points per cluster. In scenario 2.C we make the opposite change. 

In scenarios 2.D and 2.E we first double and then quadruple the variance of the prior on the per-component variances _σi,m,l_ . Finally, in scenarios 2.F and 2.G, the prior on the mean is made more dispersed compared to the training data. 

On the GMM scenarios (Table 35), the sample quality obtained via ICL is surprisingly stable under various changes to the data-generating process. It is relatively unsurprising that changing the Dirichlet prior, i.e., making the cluster more or less uniform in their number of samples, might lead to cases the ICL method can generalize to relatively easily, as demonstrated in scenarios 2.B and 2.C. The most pronounced drop in performance results from increasing the variance of the prior on the 

38 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 32: Distribution of variables for the OOD analysis on the FA scenarios. 

|**Scenario**|_K_|_P_|_µi,j_|Ψ_i,j,j_|_Wi,j,k_|_zi,j_|**_z_**_dim_|
|---|---|---|---|---|---|---|---|
|Scenario 1|50|3|_N_(0_,_1)|IG(5_,_1)|_N_(0_,_1)|_N_(0_,_1)|3|
|Scenario 1.B|50|3|_N_(0_,_1)|IG(5_,_1)|_N_(0_,_2)|_N_(0_,_1)|3|
|Scenario 1.C|50|3|_N_(0_,_1)|IG(5_,_1)|_N_(0_,_4)|_N_(0_,_1)|3|
|Scenario 2|50|3|_N_(0_,_0_._1)|IG(5_,_1)|Laplace(0_,_10)|_N_(0_,_1)|3|
|Scenario 2.B|50|3|_N_(0_,_0_._1)|IG(5_,_1)|Laplace(0_,_10_·_<br>_√_<br>2)|_N_(0_,_1)|3|
|Scenario 2.C|50|3|_N_(0_,_0_._1)|IG(5_,_1)|Laplace(0_,_20)|_N_(0_,_1)|3|
|Scenario 3|25|5|_N_(0_,_0_._1)|IG(5_,_2)|_N_(0_,_3)|_N_(0_,_1)|3|
|Scenario 3|25|5|_N_(0_,_0_._1)|IG(5_,_2)|_N_(0_,_3_·_<br>_√_<br>2)|_N_(0_,_1)|3|
|Scenario 3|25|5|_N_(0_,_0_._1)|IG(5_,_2)|_N_(0_,_6)|_N_(0_,_1)|3|



Table 33: OOD Performance: Evaluation on 50 synthetic datasets for 6 different FA scenarios. All results within two standard errors of the non-OOD result for each scenario are marked in **bold** . 

|**Scenario**|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|---|---|---|---|
|Scenario 1|**0.552**(_±_0.028)|**0.034**(_±_0.034)<br>|**0.289**(_±_0.083)|
|Scenario 1.B|0.826 (_±_0.066)|0.656(_±_0.384)<br>|0.929 (_±_0.321)|
|Scenario 1.C|0.855 (_±_0.060)|0.837(_±_0.494)|1.135 (_±_0.461)|
|Scenario 2|**0.542**(_±_0.006)|**0.017**(_±_0.006)<br>|**0.244**(_±_0.033)|
|Scenario 2.B|0.580 (_±_0.069)|0.087(_±_0.191)<br>|0.393 (_±_0.291)|
|Scenario 2.C|0.589 (_±_0.076)|0.089(_±_0.113)|0.446 (_±_0.233)|
|Scenario 3|**0.537**(_±_0.023)|**0.024**(_±_0.021)|**0.259**(_±_0.088)|
|Scenario 3.B|**0.544**(_±_0.028)|0.030(_±_0.021)|**0.285**(_±_0.094)|
|Scenario 3.C|**0.533**(_±_0.025)|0.021(_±_0.015)|**0.347**(_±_0.152)|



standard deviation of the components of the mixture model (scenario 2.E), while increasing the variance of the mean vector relative to the standard deviation of the components has a less pronounced effect. 

39 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 34: Distribution for the OOD analysis of the GMM scenarios. 

|**Scenario**|_K_|_M_|_L_|**_ϕ_**_i_|_σ_<sup>2</sup><br>_i,m,l_|_µi,m,l|σ_<sup>2</sup><br>_i,m,l_|
|---|---|---|---|---|---|---|
|Scenario 2|25|3|3|Dir(1)|IG(5_,_2)|_N_(0_,_3_σ_<sup>2</sup><br>_i,m,l_<sup>)</sup>|
|Scenario 2.B|25|3|3|Dir(0_._5)|IG(5_,_2)|_N_(0_,_3_σ_<sup>2</sup><br>_i,m,l_<sup>)</sup>|
|Scenario 2.C|25|3|3|Dir(2)|IG(5_,_2)|_N_(0_,_3_σ_<sup>2</sup><br>_i,m,l_<sup>)</sup>|
|Scenario 2.D|25|3|3|Dir(1)|IG(5_,_2_·_<br>_~~√~~_<br>2)|_N_(0_,_3_σ_<sup>2</sup><br>_i,m,l_<sup>)</sup>|
|Scenario 2.E|25|3|3|Dir(1)|IG(5_,_4)|_N_(0_,_3_σ_<sup>2</sup><br>_i,m,l_<sup>)</sup>|
|Scenario 2.F|25|3|3|Dir(1)|IG(5_,_2)|_N_(0_,_4_σ_<sup>2</sup><br>_i,m,l_<sup>)</sup>|
|Scenario 2.G|25|3|3|Dir(1)|IG(5_,_2)|_N_(0_,_5_σ_<sup>2</sup><br>_i,m,l_<sup>)</sup>|



Table 35: OOD Performance: Evaluation on 50 synthetic datasets for 6 different GMM scenarios. All results within two standard errors of the non-OOD result for each scenario are marked in **bold** . 

|**Scenario**|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|---|---|---|---|
|Scenario 2|**0.812**(_±_0.061)|**0.159**(_±_0.154)|**2.314**(_±_0.926)|
|Scenario 2.B|**0.829**(_±_0.050)|**0.233**(_±_0.161)|**2.595**(_±_0.998)|
|Scenario 2.C|**0.816**(_±_0.057)|0.149(_±_0.135)|2.272 (_±_0.654)|
|Scenario 2|**0.812**(_±_0.061)|**0.159**(_±_0.154)|**2.314**(_±_0.926)|
|Scenario 2.D|**0.812**(_±_0.076)|**0.148**(_±_0.091)|**2.557**(_±_0.837)|
|Scenario 2.E|**0.880**(_±_0.057)|**0.231**(_±_0.109)|**3.535**(_±_1.003)|
|Scenario 2|**0.812**(_±_0.061)|**0.159**(_±_0.154)|**2.314**(_±_0.926)|
|Scenario 2.F|**0.821**(_±_0.076)|**0.216**(_±_0.214)|**2.700**(_±_1.044|
|Scenario 2.G|**0.844**(_±_0.046)|**0.197**(_±_0.124)|**2.675**(_±_0.552)|



40 

**Can Transformers Learn Full Bayesian Inference In Context?** 

# **O. Ablation: Dimensionality of the Problem** 

In this section, the effect of the dimensionality _K_ of the latent variable **_z_** _∈_ R<sup>_K_</sup> for the GLM scenarios is investigated. 

In summary, our results show that forward-KL based VI and in particular MAP solutions perform strongly in terms of predictive performance, which is in line with results first presented by Mittal et al. (2025b;a). 

In table 36, we find that the advantages of the in-context learning approach to deteriorate for higher dimensionalities, with the variational inference methods using a Gaussian approximation performing well for 20 dimensions. This finding is line with work by Mittal et al. (2025b;a). For 50 dimensions we find that in many cases the used metrics do not allow to significantly discriminate the performance of the different approaches. Note that the randomness in the results, especially for higher dimensionalities, can in rare cases lead to better mean values. This is most likely not significant when taking the standard error into account. 

Similar to scenarios 1,2 and 3, we find that in scenarios 4,5 and 6 (Table 37) the advantages of the in-context learning approach to deteriorate for higher dimensionalities, with the variational inference methods using a Gaussian approximation performing well for 20 dimensions. For 50 dimensions we find that in many cases the used metrics do not allow to significantly discriminate the performance of the different approaches. Note that the randomness in the results, especially for higher dimensionalities, can in rare cases lead to better mean values. This is most likely not significant when taking the standard error into account. 

Finally, the results in Table 38 show that also for this scenario 7, the advantages of the in-context learning approach to deteriorate for higher dimensionalities. However, in this specific scenario the in-context learner is not significantly different from the VI methods in terms of C2ST and MMD for 20 dimensions. For 50 dimensions we find that the VI method using IAF performs well, together with the in-context learning approach in terms of MMD while the C2ST score does not indicate a clear winner and _W_ 2 favors the other methods. Note that the randomness in the results, especially for higher dimensionalities, can in rare cases lead to better mean values. This is most likely not significant when taking the standard error into account. 

41 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 36: Generalized Linear Models: Ablation with respect to the dimensionality of the problem on 50 synthetic and 17 real-world datasets for scenarios 1,2 and 3. All results within two standard errors of the best average result for each scenario are marked in **bold** . Due to the limitations of the number of features in the real-world data, we can only use 5 datasets for 20 and one dataset for 50 dimensions. 

|**Scenario**|**Dim**|**Model**||**Synthetic Evaluati**|**on**|**R**|**eal-World Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|---|
||**.**||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|||Laplace Approximation|1.000 (_±_0.000)|2.738(_±_0.721)|**0.825**(_±_0.279)|1.000 (_±_0.000)|2.150(_±_0.323)|**0.642**(_±_0.124)|
|||VI: DiagonalNormal<br>|0.904 (_±_0.076)<br>|1.452(_±_0.984)<br>|**0.669**(_±_0.301)<br>|0.797 (_±_0.083)<br>|0.612(_±_0.511)<br>|**0.414**(_±_0.152)<br>|
|Scenario 1|5|VI: MultivariateNormal<br>VI Sd Nl|**0.750**(_±_0.128)<br>**0753** _±_0126|**0.735**(_±_0.733)<br>**0736** _±_0737|**0.565**(_±_0.292)<br>**0570** _±_0310|**0.607**(_±_0.070)<br>**0600** _±_0070|**0.167**(_±_0.196)<br>**0169** _±_0214|**0.301**(_±_0.123)<br>**0306** _±_0131|
|||: tructure orma|**.**(.)|**.**(.)<br>|**.**(.)|**.**(.)|**.**(.)<br>|**.**(.)|
|||VI: IAF|**0.777**(_±_0.122)|**0.864**(_±_0.844)|0.725 (_±_0.523)|0.683 (_±_0.132)|0.440(_±_0.559)|0.503 (_±_0.383)|
|||HMC|<br>**0.745**(_±_0.130)|<br>**0.722**(_±_0.732)|<br>**0.569**(_±_0.301)|<br>**0.595**(_±_0.075)|<br>**0.173**(_±_0.213)|<br>**0.321**(_±_0.140)|
|||**ICL (ours)**|**0.765**(_±_0.123)|**0.767**(_±_0.727)|**0.585**(_±_0.301)|**0.614**(_±_0.074)|**0.175**(_±_0.219)|**0.310**(_±_0.138)|
|||Laplace Approximation|1.000 (_±_0.000)|2.237 (_±_0.024)|**3.252**(_±_1.172)|1.000 (_±_0.000)|2.206 (_±_0.021)|2.792 (_±_0.339)|
|||<br>VI: DiagonalNormal|**0.843**(_±_0.204)|1.056 (_±_0.869)|**2.976**(_±_0.927)|0.983 (_±_0.019)|1.217 (_±_0.463)|2.406 (_±_0.348)|
|||VI: MultivariateNormal|**0.789**(_±_0.140)|**0.714**(_±_0.351)|**2.774**(_±_0.995)|**0.768**(_±_0.120)|**0.180**(_±_0.159)|**2.064**(_±_0.306)|
|Scenario 1|20|VI: Structured Normal|<br>**0.792**(_±_0.109)|<br>**0.708**(_±_0.153)|<br>**2.703**(_±_1.069)|<br>**0.668**(_±_0.080)|<br>**0.168**(_±_0.071)|<br>**2.052**(_±_0.275)|
|||VI: IAF|**0.832**(_±_0.196)|**0.847**(_±_1.016)|**4.015**(_±_0.415)|0.939 (_±_0.024)|0.508 (_±_0.200)|3.140 (_±_0.290)|
|||ICL (ours)|**0.849**(_±_0.171)|**0.844**(_±_0.905)|4.564 (_±_0.622)|0.970 (_±_0.020)|0.724 (_±_0.287)|4.250 (_±_0.312)|
|||Laplace Approximation|1.000 (_±_0.000)|2.401 (_±_0.282)|**5.152**(_±_2.268)|1.000 (_±_nan)|2.339 (_±_nan)|6.642 (_±_nan)|
|||VI: DiagonalNormal|**0.812**(_±_0.197)|**0.956**(_±_1.016)|**5.541**(_±_2.356)|0.915 (_±_nan)|0.508 (_±_nan)|6.200 (_±_nan)|
|Scenario 1|50|VI: MultivariateNormal|**0.839**(_±_0.148)|**0.926**(_±_0.682)|**5.514**(_±_2.370)|0.905 (_±_nan)|0.790 (_±_nan)|6.258 (_±_nan)|
|||VI: Structured Normal|**0.823**(_±_0.160)|**0.844**(_±_0.480)|**5.752**(_±_2.098)|0.910 (_±_nan)|1.122 (_±_nan)|6.898 (_±_nan)|
|||VI: IAF|<br>**0.820**(_±_0.182)|<br>**0.814**(_±_0.987)|<br>**6.696**(_±_1.207)|<br>0.938 (_±_nan)|<br>0.256 (_±_nan)|<br>6.869 (_±_nan)|
|||ICL (ours)|**0.787**(_±_0.217)|**1.015**(_±_1.255)|**8.278**(_±_0.821)|0.979 (_±_nan)|0.413 (_±_nan)|8.368 (_±_nan)|
|||Laplace Approximation|1.000 (_±_0.000)|4.853(_±_2.333)|5.770 (_±_5.946)|1.000 (_±_0.000)|2.572(_±_0.206)|0.809 (_±_0.149)|
|||<br>VI: DiagonalNormal|<br>0.957 (_±_0.091)|<br>3.906(_±_2.679)|<br>5.628 (_±_6.092)|<br>0.892 (_±_0.044)|<br>0.847(_±_0.389)|<br>**0.530**(_±_0.175)|
|Si 2|5|VI: MultivariateNormal|0.910 (_±_0.131)|3.407(_±_2.781)|5.584 (_±_6.104)|0.820 (_±_0.031)|0.243(_±_0.148)|**0.408**(_±_0.118)|
|cenaro||VI: Structured Normal|0.908 (_±_0.119)|3.139(_±_2.763)|5.480 (_±_6.164)|0.824 (_±_0.023)|0.215(_±_0.110)|**0.392**(_±_0.109)|
|||VI: IAF|0.968 (_±_0.063)|4.416(_±_2.473)|7.474 (_±_6.235)|0.888 (_±_0.067)|0.921(_±_0.860)|0.942 (_±_0.733)|
|||ICL (ours)|**0.839**(_±_0.072)|<br>**0.707**(_±_0.658)|**1.111**(_±_0.300)|**0.768**(_±_0.033)|<br>**0.143**(_±_0.089)|**0.411**(_±_0.094)|
|||Laplace Approximation|1.000 (_±_0.000)|2.314 (_±_0.237)|3.069 (_±_1.168)|1.000 (_±_0.000)|2.222 (_±_0.018)|2.847 (_±_0.305)|
|||<br>VI: DiagonalNormal|0.904 (_±_0.168)|1.292 (_±_0.937)|**2.863**(_±_0.919)|0.990 (_±_0.009)|1.277 (_±_0.452)|2.483 (_±_0.318)|
|Scenario 2|20|VI: MultivariateNormal<br>|0.851 (_±_0.134)<br>|**0.492**(_±_0.547)<br>|**2.694**(_±_0.916)<br>|0.843 (_±_0.069)<br>|0.243 (_±_0.170)<br>|**2.166**(_±_0.266)<br>|
|||VI: Structured Normal|**0.697**(_±_0.065)|**0.070**(_±_0.099)|**2.497**(_±_0.993)|**0.655**(_±_0.031)|**0.029**(_±_0.025)|**2.191**(_±_0.271)|
|||VI: IAF|0.916 (_±_0.110)|1.062 (_±_1.076)|4.191 (_±_0.623)|0.952 (_±_0.025)|0.515 (_±_0.242)|3.331 (_±_0.371)|
|||ICL (ours)|0.955 (_±_0.057)|1.131 (_±_1.035)|4.945 (_±_0.836)|0.968 (_±_0.020)|0.724 (_±_0.278)|4.356 (_±_0.302)|
|||Laplace Approximation|**1.000**(_±_0.000)|2.437 (_±_0.271)|**5.728**(_±_1.358)|1.000 (_±_nan)|2.350 (_±_nan)|5.620 (_±_nan)|
|||VI: DiagonalNormal|**0.853**(_±_0.182)|0.787 (_±_0.687)|**6.224**(_±_1.225)|0.996 (_±_nan)|1.080 (_±_nan)|5.426 (_±_nan)|
|||VI: MultivariateNormal|**0.878**(_±_0.150)|**0.688**(_±_0.620)|**6.206**(_±_1.244)|0.994 (_±_nan)|0.791 (_±_nan)|5.305 (_±_nan)|
|Scenario 2|50|VI: Structured Normal|**0865**(_±_0081)|**0186**(_±_0169)|5874 (_±_1233)|0819 (_±_nan)|0093 (_±_nan)|5660 (_±_nan)|
|||<br>VI: IAF|**.** .<br>**0.909**(_±_0.130)|**.** .<br>0.649 (_±_0.650)|..<br>7.465 (_±_0.335)|. <br>0.985 (_±_nan)|. <br>0.426 (_±_nan)|. <br>6.426 (_±_nan)|
|||ICL (ours)|**0.972**(_±_0.039)|0.741 (_±_0.713)|8.313 (_±_0.608)|0.971 (_±_nan)|0.405 (_±_nan)|7.718 (_±_nan)|
|||Laplace Approximation|1.000 (_±_0.000)|2.203 (_±_0.997)|1.170 (_±_0.949)|1.000 (_±_0.000)|1.841 (_±_0.185)|0.729 (_±_0.175)|
|||VI: DiagonalNormal|0.866 (_±_0.101)|1.069 (_±_1.150)|0.846 (_±_0.747)|0.797 (_±_0.083)|0.526 (_±_0.361)|0.480 (_±_0.207)|
|||VI: MultivariateNormal|**0.656**(_±_0.131)|**0.445**(_±_1.061)|**0.660**(_±_0.737)|**0.560**(_±_0.035)|**0.032**(_±_0.028)|**0.249**(_±_0.069)|
|Scenario 3|5|VI: Structured Normal|**0.653**(_±_0.125)|**0.421**(_±_0.993)|**0.659**(_±_0.736)|**0.552**(_±_0.028)|**0.027**(_±_0.015)|**0.239**(_±_0.055)|
|||VI: IAF|0.751 (_±_0.148)|0.939 (_±_1.349)|0.964 (_±_0.924)|0.673 (_±_0.141)|0.399 (_±_0.543)|0.563 (_±_0.433)|
|||ICL (ours)|**0.611**(_±_0.070)|**0.089**(_±_0.114)|**0.423**(_±_0.348)|**0.576**(_±_0.027)|**0.037**(_±_0.026)|**0.257**(_±_0.044)|
|||Laplace Approximation|1.000 (_±_0.000)|2.726 (_±_1.116)|4.127 (_±_1.927)|1.000 (_±_0.000)|2.234 (_±_0.092)|3.589 (_±_0.519)|
|||<br>VI: DiagonalNormal<br>VI MliiNl|<br>**0.912**(_±_0.134)<br>**0863** _±_0113|<br>**1.704**(_±_1.467)<br>**0937** _±_1174|<br>**3.933**(_±_1.574)<br>**3754** _±_1650|<br>0.983 (_±_0.014)<br>**0796** _±_0099|<br>1.298 (_±_0.443)<br>**0268** _±_0226|<br>3.147 (_±_0.557)<br>**2645** _±_0466|
|Scenario 3|20|: utvarateorma<br>VI: Structured Normal|**.**(.)<br>**0.768**(_±_0.109)|**.**(.)<br>**0.302**(_±_0.518)|**.**(.)<br>**3.151**(_±_1.663)|**.**(.)<br>**0.722**(_±_0.073)|**.**(.)<br>**0.131**(_±_0.141)|**.**(.)<br>**2.579**(_±_0.399)|
|||VI: IAF|<br>**0.908**(_±_0.133)|<br>**1.657**(_±_1.476)|<br>5.543 (_±_1.120)|<br>0.936 (_±_0.041)|<br>0.548 (_±_0.341)|<br>3.678 (_±_0.670)|
|||ICL (ours)|<br>**0.902**(_±_0.076)|<br>**1.053**(_±_0.782)|<br>6.206 (_±_0.783)|<br>0.932 (_±_0.019)|<br>0.635 (_±_0.183)|<br>5.281 (_±_0.317)|
|||Laplace Approximation|**1.000**(_±_0.000)|2.700 (_±_0.789)|**8.841**(_±_1.691)|1.000 (_±_nan)|2.348 (_±_nan)|7.049 (_±_nan)|
|||<br>VI: DiagonalNormal|**0.870**(_±_0.127)|**1.154**(_±_1.321)|**9.180**(_±_1.513)|0.997 (_±_nan)|1.393 (_±_nan)|6.791 (_±_nan)|
|||VI: MultivariateNormal|**0.896**(_±_0.101)|**1.027**(_±_1.157)|**9.175**(_±_1.555)|0.998 (_±_nan)|1.092 (_±_nan)|6.667 (_±_nan)|
|Scenario 3|50|VI: Structured Normal|<br>**0.873**(_±_0.112)|<br>**0.539**(_±_0.667)|<br>**9.118**(_±_1.538)|<br>0.958 (_±_nan)|<br>0.420 (_±_nan)|<br>6.665 (_±_nan)|
|||VI: IAF|**0.869**(_±_0124)|**0.751**(_±_0939)|**9.917**(_±_0870)|0971 (_±_nan)|0417 (_±_nan)|7411 (_±_nan)|
|||<br>ICL (ours)|.<br>**0.931**(_±_0.062)|.<br>**0.784**(_±_0.884)|.<br>10.063 (_±_0.930)|. <br>0.965 (_±_nan)|. <br>0.347 (_±_nan)|. <br>8.482 (_±_nan)|



42 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 37: Generalized Linear Models: Ablation with respect to the dimensionality of the problem on 50 synthetic and 17 real-world datasets for scenarios 4, 5 and 6. All results within two standard errors of the best average result for each scenario are marked in **bold** . Due to the limitations of the number of features in the real-world data, we can only use 5 datasets for 20 and one dataset for 50 dimensions. 

|**Scenario**|**Dim**|**Model**||**Synthetic Evaluatio**|**n**|**R**|**eal-World Evalua**|**tion**|
|---|---|---|---|---|---|---|---|---|
||**.**||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|||Laplace Approximation|1.000 (_±_0.000)|3.511(_±_2.025)|2.166 (_±_1.722)|1.000 (_±_0.000)|2.011(_±_0.058)|0.993 (_±_0.144)|
|||VI: DiagonalNormal<br>|0.968 (_±_0.036)<br>|2.798(_±_2.255)<br>|2.065 (_±_1.745)<br>|0.916 (_±_0.040)<br>|0.928(_±_0.339)<br>|0.732 (_±_0.181)<br>|
|||VI: MultivariateNormal|0.855 (_±_0.123)|1.648(_±_2.052)|1.853 (_±_1.745)|0.771 (_±_0.017)|**0.087**(_±_0.030)|**0.539**(_±_0.070)|
|Scenario 4|5|VI: Structured Normal|0.847 (_±_0.116)|<br>1.505(_±_1.978)|1.889 (_±_1.883)|0.769 (_±_0.012)|<br>**0.083**(_±_0.018)|**0.543**(_±_0.070)|
|||VI: IAF|0.942 (_±_0.077)|<br>3.029(_±_2.210)|3.554 (_±_2.715)|0.833 (_±_0.069)|<br>0.636(_±_0.756)|0.978 (_±_0.600)|
|||**ICL (ours)**|**0.753**(_±_0.049)|**0.171**(_±_0.153)|**0.631**(_±_0.294)|**0.762**(_±_0.015)|**0.105**(_±_0.046)|**0.597**(_±_0.104)|
|||Laplace Approximation|1.000 (_±_0.000)|4.929(_±_1.611)|**8.863**(_±_3.796)|**1.000**(_±_0.000)|3.196(_±_0.841)|5.186 (_±_1.533)|
|||VI: DiagonalNormal|**0.988**(_±_0.060)|<br>4.418(_±_2.013)|**9.364**(_±_4.281)|0.997 (_±_0.007)|<br>3.095(_±_1.417)|6.098 (_±_2.435)|
|Scenario 4|20|VI: MultivariateNormal|**0.986**(_±_0.054)|3.388(_±_1.907)|**7.910**(_±_4.070)|0.893 (_±_0.087)|**0.534**(_±_0.469)|**3.175**(_±_0.751)|
|||VI: Structured Normal|**0.954**(_±_0.076)|**2.254**(_±_1.515)|**7.475**(_±_4.224)|**0.727**(_±_0.034)|**0.074**(_±_0.070)|**2.877**(_±_0.379)|
|||VI: IAF|<br>**0.987**(_±_0.059)|<br>3.258(_±_1.415)|<br>**9.865**(_±_3.515)|<br>0.955 (_±_0.030)|<br>0.629(_±_0.308)|<br>4.098 (_±_0.341)|
|||ICL (ours)|**0.978**(_±_0.038)|**1.185**(_±_0.720)|**11.335**(_±_1.378)|0.972 (_±_0.018)|0.668(_±_0.199)|9.937 (_±_0.466)|
|||Laplace Approximation|**1.000**(_±_0.000)|6.695 (_±_1.329)|**12.323**(_±_4.091)|1.000 (_±_nan)|5.491 (_±_nan)|7.518 (_±_nan)|
|||VI: DiagonalNormal|**0.965**(_±_0.084)|2.395 (_±_1.958)|**12.022**(_±_3.673)|0.996 (_±_nan)|4.368 (_±_nan)|6.951 (_±_nan)|
|Scenario 4|50|VI: MultivariateNormal<br>|**0.984**(_±_0.054)<br>|5.395 (_±_1.847)<br>|**12.141**(_±_3.079)<br>|1.000 (_±_nan)<br>|5.146 (_±_nan)<br>|9.002 (_±_nan)<br>|
|||VI: Structured Normal|**0.982**(_±_0.026)|**4.261**(_±_1.191)|**11.126**(_±_3.396)|0.869 (_±_nan)|3.181 (_±_nan)|7.065 (_±_nan)|
|||VI: IAF|**0.981**(_±_0.048)|**4.609**(_±_1.412)|**12.567**(_±_3.131)|0.988 (_±_nan)|3.558 (_±_nan)|7.849 (_±_nan)|
|||ICL (ours)|**0.960**(_±_0.045)|**3.792**(_±_0.758)|**14.071**(_±_0.894)|0.974 (_±_nan)|3.443 (_±_nan)|12.546 (_±_nan)|
|||Laplace Approximation|1.000 (_±_0.000)|2.060(_±_0.472)|0.797 (_±_0.577)|1.000 (_±_0.000)|1.982(_±_0.126)|0.623 (_±_0.084)|
|||<br>VI: DiagonalNormal|<br>0.866 (_±_0.085)|<br>0.954(_±_1.022)|<br>0.651 (_±_0.549)|<br>0.810 (_±_0.036)|<br>0.441(_±_0.252)|<br>0.384 (_±_0.089)|
|||VI: MultivariateNormal|0.765 (_±_0.100)|0.537(_±_1.019)|0.633 (_±_1.067)|0.711 (_±_0.038)|0.148(_±_0.093)|**0.279**(_±_0.056)|
|Scenario 5|5|VI: Structured Normal|0.758 (_±_0.098)|<br>0.447(_±_0.818)|0.572 (_±_0.816)|0.705 (_±_0.032)|<br>0.140(_±_0.081)|**0.269**(_±_0.045)|
|||VI: IAF|0.814 (_±_0.105)|0.953(_±_1.165)|0.881 (_±_1.067)|0.777 (_±_0.106)|0.684(_±_0.939)|0.625 (_±_0.525)|
|||ICL (ours)|<br>**0.621**(_±_0.063)|<br>**0.067**(_±_0.080)|<br>**0.299**(_±_0.195)|<br>**0.610**(_±_0.045)|<br>**0.046**(_±_0.020)|<br>**0.242**(_±_0.038)|
|||Laplace Approximation|**1.000**(_±_0.000)|2.367 (_±_0.555)|2.780 (_±_1.271)|**1.000**(_±_0.000)|2.200 (_±_0.041)|2.444 (_±_0.619)|
|||VI: DiagonalNormal|**0.938**(_±_0.098)|**1.153**(_±_0.954)|**2.552**(_±_1.147)|**0.967**(_±_0.012)|0.547 (_±_0.233)|**1.973**(_±_0.452)|
|Snri 5|20|VI: MultivariateNormal|**0.929**(_±_0.082)|**0.710**(_±_0.768)|**2.473**(_±_1.145)|**0.928**(_±_0.016)|**0.250**(_±_0.079)|**1.776**(_±_0.399)|
|ceao||VI: Structured Normal|**0.909**(_±_0.082)|**0.397**(_±_0.442)|**2.246**(_±_1.244)|**0.924**(_±_0.018)|**0.202**(_±_0.094)|**1.775**(_±_0.430)|
|||VI: IAF|<br>**0.934**(_±_0.092)|<br>1.325 (_±_1.161)|<br>4.899 (_±_1.320)|<br>**0.980**(_±_0.016)|<br>0.892 (_±_0.404)|<br>3.593 (_±_0.597)|
|||ICL (ours)|**0.961**(_±_0.046)|1.330 (_±_1.125)|5.084 (_±_1.297)|**0.981**(_±_0.014)|1.162 (_±_0.461)|4.804 (_±_0.578)|
|||Laplace Approximation|**1.000**(_±_0.000)|2.582 (_±_0.606)|**5.765**(_±_1.540)|1.000 (_±_nan)|2.322 (_±_nan)|3.485 (_±_nan)|
|||VI: DiagonalNormal|**0.925**(_±_0.074)|0.925 (_±_1.056)|**6.461**(_±_1.877)|0.972 (_±_nan)|0.186 (_±_nan)|3.251 (_±_nan)|
|||VI: MultivariateNormal|**0.934**(_±_0.064)|**0.825**(_±_0.972)|**6.404**(_±_1.882)|0.969 (_±_nan)|0.165 (_±_nan)|3.223 (_±_nan)|
|Scenario 5|50|VI: Structured Normal|<br>**0.927**(_±_0.068)|<br>**0.481**(_±_0.588)|<br>**6.420**(_±_1.970)|<br>0.961 (_±_nan)|<br>0.072 (_±_nan)|<br>3.324 (_±_nan)|
|||VI: IAF|**0.925**(_±_0.069)|**0.792**(_±_0.975)|8.458 (_±_0.864)|0.996 (_±_nan)|0.519 (_±_nan)|4.645 (_±_nan)|
|||ICL (ours)|**0.998**(_±_0.002)|**0.762**(_±_0.987)|8.195 (_±_0.820)|1.000 (_±_nan)|0.984 (_±_nan)|7.288 (_±_nan)|
|||Laplace Approximation|1.000 (_±_0.000)|2.026(_±_0.027)|1.612 (_±_0.162)|1.000 (_±_0.000)|1.993(_±_0.032)|1.299 (_±_0.106)|
|||<br>VI: DiagonalNormal|0.724 (_±_0.060)|<br>0.185(_±_0.082)|**0.787**(_±_0.078)|0.703 (_±_0.039)|<br>0.147(_±_0.063)|0.637 (_±_0.089)|
|Si 6|5|VI: MultivariateNormal|**0.534**(_±_0.018)|**0.014**(_±_0.006)|**0.581**(_±_0.074)|**0.538**(_±_0.019)|**0.016**(_±_0.007)|**0.466**(_±_0.029)|
|cenaro||VI: Structured Normal|**0.536**(_±_0.016)|**0.014**(_±_0.005)|**0.583**(_±_0.071)|**0.536**(_±_0.019)|**0.017**(_±_0.009)|**0.469**(_±_0.033)|
|||VI: IAF|0.542 (_±_0.026)|0.031(_±_0.031)|0.613 (_±_0.092)|**0.535**(_±_0.015)|**0.015**(_±_0.006)|**0.467**(_±_0.031)|
|||**ICL (ours)**|**0.532**(_±_0.019)|0.016(_±_0.008)|**0.590**(_±_0.066)|0.556 (_±_0.017)|0.035(_±_0.015)|**0.504**(_±_0.038)|
|||Laplace Approximation|1.000 (_±_0.000)|2.247 (_±_0.006)|4.158 (_±_0.243)|1.000 (_±_0.000)|2.240 (_±_0.007)|3.714 (_±_0.127)|
|||<br>VI: DiagonalNormal|**0.747**(_±_0.138)|**0.136**(_±_0.123)|**3.460**(_±_0.361)|**0.836**(_±_0.053)|**0.203**(_±_0.086)|**2.977**(_±_0.112)|
|||VI: MultivariateNormal|**0.621**(_±_0016)|**0.016**(_±_0002)|**3.564**(_±_0290)|**0.608**(_±_0017)|**0.015**(_±_0003)|**3.101**(_±_0115)|
|Scenario 6|20|<br>VI: Structured Normal|.<br>**0.599**(_±_0.015)|.<br>**0.012**(_±_0.002)|.<br>**3.592**(_±_0.267)|.<br>**0.584**(_±_0.028)|.<br>**0.012**(_±_0.002)|.<br>**3.120**(_±_0.107)|
|||VI: IAF|**0.625**(_±_0.040)|**0.019**(_±_0.009)|**3.572**(_±_0.266)|**0.636**(_±_0.021)|**0.020**(_±_0.005)|**3.106**(_±_0.128)|
|||ICL (ours)|<br>0.747 (_±_0.148)|<br>0.163 (_±_0.144)|<br>4.063 (_±_0.184)|<br>0.928 (_±_0.030)|<br>0.463 (_±_0.162)|<br>4.425 (_±_0.314)|
|||Laplace Approximation|1.000 (_±_0.000)|2.291 (_±_0.003)|**6.742**(_±_0.362)|1.000 (_±_nan)|2.293 (_±_nan)|6.587 (_±_nan)|
|||VI: DiagonalNormal|**0.761**(_±_0.138)|**0.087**(_±_0.083)|**6.909**(_±_0.743)|0.905 (_±_nan)|0.175 (_±_nan)|6.403 (_±_nan)|
|||VI: MultivariateNormal|**0.797**(_±_0.100)|**0.069**(_±_0.055)|**6.956**(_±_0.736)|0.891 (_±_nan)|0.110 (_±_nan)|6.473 (_±_nan)|
|Scenario 6|50|VI: Structured Normal|<br>**0.647**(_±_0.017)|<br>**0.013**(_±_0.002)|<br>7.218 (_±_0.506)|<br>0.654 (_±_nan)|<br>0.013 (_±_nan)|<br>6.890 (_±_nan)|
|||VI: IAF|**0639**(_±_0038)|**0014**(_±_0006)|7204 (_±_0463)|0692 (_±_nan)|0024 (_±_nan)|6887 (_±_nan)|
|||<br>ICL (ours)|**.** .<br>**0.742**(_±_0.178)|**.** .<br>0.115 (_±_0.124)|..<br>7.713 (_±_0.120)|. <br>0.935 (_±_nan)|. <br>0.203 (_±_nan)|. <br>7.846 (_±_nan)|



43 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 38: Generalized Linear Models: Ablation with respect to the dimensionality of the problem on 50 synthetic and 17 real-world datasets for scenario 7. All results within two standard errors of the best average result for each scenario are marked in **bold** . Due to the limitations of the number of features in the real-world data, we can only use 5 datasets for 20 and one dataset for 50 dimensions. 

|**Scenario**|**Dim.**|**Model**||**Synthetic Evaluatio**|**n**|**R**|**eal-World Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|---|
||||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|||Laplace Approximation|1.000 (_±_0.000)|3.559(_±_1.933)|1.347 (_±_1.067)|1.000 (_±_0.000)|2.016(_±_0.080)|0.763 (_±_0.174)|
|||VI: DiagonalNormal|0.938 (_±_0.074)|2.536(_±_2.097)|1.142 (_±_0.993)|0.936 (_±_0.024)|1.029(_±_0.255)|0.579 (_±_0.181)|
|Si 7|5|VI: MultivariateNormal|0.814 (_±_0.181)|1.999(_±_2.283)|1.033 (_±_0.969)|**0.741**(_±_0.020)|0.093(_±_0.025)|**0.391**(_±_0.074)|
|cenaro||VI: Structured Normal|0.824 (_±_0.177)|1.891(_±_2.127)|1.041 (_±_0.934)|**0.734**(_±_0.025)|**0.072**(_±_0.019)|**0.385**(_±_0.065)|
|||VI: IAF|0.939 (_±_0.091)|2.707(_±_1.712)|1.590 (_±_0.820)|0.864 (_±_0.093)|0.830(_±_0.697)|1.064 (_±_0.616)|
|||ICL (ours)|**0.700**(_±_0.116)|**0.317**(_±_0.355)|**0.400**(_±_0.286)|0.773 (_±_0.048)|**0.294**(_±_0.457)|0.559 (_±_0.256)|
|||Laplace Approximation|1.000 (_±_0.000)|3.581 (_±_2.147)|**3.365**(_±_1.583)|1.000 (_±_0.000)|2.213 (_±_0.024)|2.539 (_±_0.378)|
|||VI: DiagonalNormal|**0.887**(_±_0.184)|2.819 (_±_2.732)|3.637 (_±_1.371)|**0.996**(_±_0.005)|1.734 (_±_0.314)|**2.348**(_±_0.423)|
|Si 7|20|VI: MultivariateNormal|**0.881**(_±_0.164)|2.265 (_±_2.573)|**3.524**(_±_1.392)|**0.916**(_±_0.085)|0.766 (_±_0.535)|**2.043**(_±_0.516)|
|cenaro||VI: Structured Normal|**0.850**(_±_0.162)|**1.667**(_±_2.266)|**3.186**(_±_1.315)|**0.849**(_±_0.105)|**0.391**(_±_0.244)|**1.880**(_±_0.367)|
|||VI: IAF|**0.867**(_±_0.184)|**1.629**(_±_1.584)|4.875 (_±_1.239)|0.986 (_±_0.007)|0.895 (_±_0.361)|4.096 (_±_0.319)|
|||ICL (ours)|**0.867**(_±_0.185)|**1.428**(_±_1.352)|4.836 (_±_1.032)|**0.982**(_±_0.010)|**0.820**(_±_0.324)|4.177 (_±_0.368)|
|||Laplace Approximation|1.000 (_±_0.000)|4.768 (_±_1.171)|**6.573**(_±_1.038)|1.000 (_±_nan)|2.312 (_±_nan)|5.270 (_±_nan)|
|||VI: DiagonalNormal|**0.771**(_±_0.191)|3.263 (_±_1.853)|**6.919**(_±_1.257)|1.000 (_±_nan)|2.237 (_±_nan)|5.417 (_±_nan)|
|||VI: MultivariateNormal|**0.816**(_±_0.154)|3.245 (_±_1.793)|6.978 (_±_1.226)|0.997 (_±_nan)|2.117 (_±_nan)|5.781 (_±_nan)|
|Scenario 7|50|VI: Structured Normal|**0.795**(_±_0.171)|3.126 (_±_1.677)|**6.918**(_±_1.260)|1.000 (_±_nan)|1.879 (_±_nan)|5.461 (_±_nan)|
|||VI: IAF|**0.769**(_±_0.189)|**2.534**(_±_0.894)|7.895 (_±_0.843)|0.994 (_±_nan)|0.584 (_±_nan)|7.626 (_±_nan)|
|||ICL (ours)|**0.732**(_±_0.216)|**2.451**(_±_0.790)|7.787 (_±_0.661)|0.980 (_±_nan)|0.411 (_±_nan)|7.461 (_±_nan)|



# **P. Comparison to SGLD** 

Besides comparing the samples from our ICL approach to samples from various VI methods, we additionally compare it against samples generated via stochastic gradient Langevin dynamics (SGLD) (Welling & Teh, 2011). We run SGLD with a learning rate of 10<sup>_−_3</sup> for the GLM and GMM cases and a learning rate of 10<sup>_−_4</sup> for FA and use 1000 gradient steps for warmup and partition the data into ten minibatches. We implement the preconditioning method introduced by (Li et al., 2016) for more stable sampling behavior. Despite the preconditioning, SGLD consistently fails for GLMs scenario 7 because the sampler diverges causing singular covariance matrices. To facilitate running SGLD for the GMMs, which also include discrete variables, we marginalize over the discrete variables. 

In summary, we find that ICL yields samples with much higher quality than SGLD compared to the gold standard HMC samples across almost all scenarios on both synthetic and real-world data. The poor sample quality with SGLD is expected given that numerous theoretical and empirical findings confirm that, while SGLD is computationally very cheap, it is substantially outperformed by, for instance, HMC, in terms of sample quality, which is especially pronounced when the posterior distributions are complex and parameters are correlated (Chen et al., 2014; Mangoubi & Vishnoi, 2019; Izmailov et al., 2021; Brosse et al., 2018) . 

For GLMs (Table 42), ICL achieves significantly better results, with notable improvements in C2ST. In Scenario 1, synthetic C2ST drops from 0.992 to 0.765 and real-world C2ST from 0.980 to 0.614. Similarly, Scenario 3 shows substantial gains, with synthetic C2ST improving from 0.997 to 0.611 and real-world C2ST from 0.983 to 0.576. These trends extend to metrics like _W_ 2, where ICL yields consistent reductions. 

For FA (Table 43), ICL also achieves superior performance, particularly in Scenarios 1 and 2. For example, in Scenario 1, synthetic C2ST decreases from 0.996 to 0.552, accompanied by improvements in _W_ 2 from 1.776 to 0.289. Scenario 2 sees further enhancements, with synthetic MMD dropping from 2.950 to 0.017 and real-world C2ST improving from 0.995 to 0.622. 

For GMMs (Table 44), ICL demonstrates a clear advantage in most scenarios. In Scenario 1, ICL reduces synthetic C2ST from 1.000 to 0.760 and real-world _W_ 2 from 6.510 to 4.054. Scenario 2 shows synthetic C2ST improving from 1.000 to 0.812, and MMD from 3.046 to 0.159. While in scenarios 3, ICL has a singificantly lower MMD score on the synthetic data, the other differences are not significant. 

44 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 39: Evaluating the predictive performance across 50 synthetic and 17 real-world datasets in GLM scenario 2 for different dimensionalities. All results within two standard errors of the best average result for each scenario are marked in **bold** . Due to the limitations of the number of features in the real-world data, we can only use 5 datasets for 20 and one dataset for 50 dimensions. We find that the quality of the samples by the in-context learner, when evaluated based on predictive performance, decreases consistently with an increase in the dimensionality of the problem. Note that the randomness in the results, especially for higher dimensionalities, can in rare cases lead to better mean values. This is most likely not significant when taking the standard error into account. 

|**Scenario**|**Dim.**|**Model**|**RMSE Real-World**(_↓_)|**RMSE Synthetic**(_↓_)|
|---|---|---|---|---|
|||HMC|**0.559**(_±_0.023)|**0.556**(_±_0.049)|
|||Laplace Approximation|**0.561**(_±_0.022)|**0.557**(_±_0.049)|
|||VI: DiagonalNormal|**0.560**(_±_0.023)|**0.557**(_±_0.049)|
|||VI: MultivariateNormal|**0.559**(_±_0.023)|**0.556**(_±_0.049)|
|Scenario 2|5|VI: Structured Normal|**0.604**(_±_0.016)|0.685 (_±_0.054)|
|||VI: IAF|**0.563**(_±_0.023)|**0.557**(_±_0.049)|
|||ICL (ours)|**0.561**(_±_0.019)|**0.653**(_±_0.049)|
|||MAP|0.513 (_±_0.023)|0.522 (_±_0.048)|
|||TabPFN|0.449 (_±_0.034)|0.498 (_±_0.047)|
|||HMC|**0.682**(_±_0.029)|**0.536**(_±_0.041)|
|||Laplace Approximation|**0.682**(_±_0.030)|**0.538**(_±_0.040)|
|||VI: DiagonalNormal|**0.680**(_±_0.029)|**0.539**(_±_0.041)|
|||VI: MultivariateNormal|**0.685**(_±_0.029)|**0.537**(_±_0.041)|
|Scenario 2|20|VI: Structured Normal|0.746 (_±_0.019)|0.681 (_±_0.041)|
|||VI: IAF|**0.683**(_±_0.029)|**0.539**(_±_0.041)|
|||ICL (ours)|0.777 (_±_0.011)|1.122 (_±_0.078)|
|||MAP|0.578 (_±_0.025)|0.472 (_±_0.039)|
|||TabPFN|0.470 (_±_0.044)|0.446 (_±_0.038)|
|||HMC|0.669 (_±_nan)|**0.713**(_±_0.060)|
|||Laplace Approximation|0.594 (_±_nan)|0.878 (_±_0.068)|
|||VI: DiagonalNormal|0.582 (_±_nan)|0.870 (_±_0.065)|
|||VI: MultivariateNormal|0.729 (_±_nan)|**0.764**(_±_0.066)|
|Scenario 2|50|VI: Structured Normal|0.922 (_±_nan)|1.116 (_±_0.074)|
|||VI: IAF|0.695 (_±_nan)|**0.770**(_±_0.060)|
|||ICL (ours)|1.256 (_±_nan)|2.343 (_±_0.230)|
|||MAP|0.301 (_±_nan)|0.398 (_±_0.047)|
|||TabPFN|0.235 (_±_nan)|0.570 (_±_0.053)|



45 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 40: Evaluating the predictive performance across 50 synthetic and 17 real-world datasets in GLM scenario 2 for different dimensionalities. All results within two standard errors of the best average result for each scenario are marked in **bold** . Due to the limitations of the number of features in the real-world data, we can only use 5 datasets for 20 and one dataset for 50 dimensions. We find that the quality of the samples by the in-context learner, when evaluated based on predictive performance, decreases consistently with an increase in the dimensionality of the problem. 

|**Scenario**|**Dim.**|**Model**|**RMSE Real-World**(_↓_)|**RMSE Synthetic**(_↓_)|
|---|---|---|---|---|
|||HMC|**0.684**(_±_0.027)|**0.512**(_±_0.040)|
|||Laplace Approximation|**0.688**(_±_0.026)|**0.516**(_±_0.040)|
|||VI: DiagonalNormal|**0.686**(_±_0.027)|**0.513**(_±_0.040)|
|||VI: MultivariateNormal|**0.685**(_±_0.027)|**0.512**(_±_0.040)|
|Scenario 3|5|VI: Structured Normal|**0.733**(_±_0.016)|0.607 (_±_0.043)|
|||VI: IAF|**0.686**(_±_0.027)|**0.512**(_±_0.040)|
|||ICL (ours)|**0.690**(_±_0.023)|**0.588**(_±_0.045)|
|||MAP|0.646 (_±_0.028)|0.495 (_±_0.039)|
|||TabPFN|0.556 (_±_0.041)|0.462 (_±_0.037)|
|||HMC|**1.030**(_±_0.045)|**0.621**(_±_0.046)|
|||Laplace Approximation|**1.053**(_±_0.047)|0.755 (_±_0.052)|
|||VI: DiagonalNormal|**1.035**(_±_0.043)|0.734 (_±_0.053)|
|||VI: MultivariateNormal|**1.033**(_±_0.039)|**0.705**(_±_0.055)|
|Scenario 3|20|VI: Structured Normal|**1.095**(_±_0.045)|1.033 (_±_0.063)|
|||VI: IAF|**1.026**(_±_0.045)|**0.653**(_±_0.047)|
|||ICL (ours)|1.770 (_±_0.048)|2.160 (_±_0.217)|
|||MAP|0.861 (_±_0.038)|0.581 (_±_0.050)|
|||TabPFN|0.654 (_±_0.062)|0.475 (_±_0.039)|
|||HMC|0.858 (_±_nan)|**0.645**(_±_0.051)|
|||Laplace Approximation|0.866 (_±_nan)|0.865 (_±_0.083)|
|||VI: DiagonalNormal|0.788 (_±_nan)|0.870 (_±_0.084)|
|||VI: MultivariateNormal|0.819 (_±_nan)|0.778 (_±_0.066)|
|Scenario 3|50|VI: Structured Normal|0.812 (_±_nan)|1.040 (_±_0.103)|
|||VI: IAF|0.802 (_±_nan)|0.846 (_±_0.078)|
|||ICL (ours)|1.686 (_±_nan)|3.477 (_±_0.604)|
|||MAP|0.539 (_±_nan)|0.618 (_±_0.054)|
|||TabPFN|0.322 (_±_nan)|0.534 (_±_0.038)|



46 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 41: Evaluating the predictive performance across 50 synthetic and 17 real-world datasets in GLM scenario 2 for different dimensionalities. All results within two standard errors of the best average result for each scenario are marked in **bold** . Due to the limitations of the number of features in the real-world data, we can only use 5 datasets for 20 and one dataset for 50 dimensions. We find that the quality of the samples by the in-context learner, when evaluated based on predictive performance, decreases consistently with an increase in the dimensionality of the problem. 

|**Scenario**|**Dim.**|**Model**|**RMSE Real-World**(_↓_)|**RMSE Synthetic**(_↓_)|
|---|---|---|---|---|
|||HMC|**0.699**(_±_0.022)|**0.490**(_±_0.036)|
|||Laplace Approximation|**0.699**(_±_0.022)|**0.491**(_±_0.036)|
|||VI: DiagonalNormal|**0.702**(_±_0.022)|**0.491**(_±_0.036)|
|||VI: MultivariateNormal|**0.698**(_±_0.021)|**0.491**(_±_0.036)|
|Scenario 5|5|VI: Structured Normal|1.507 (_±_0.089)|0.741 (_±_0.053)|
|||VI: IAF|**0.699**(_±_0.022)|**0.490**(_±_0.036)|
|||ICL (ours)|<br>0.769 (_±_0.020)|<br>0.701 (_±_0.049)|
|||MAP|0.658 (_±_0.022)|0.471 (_±_0.035)|
|||TabPFN|0.534 (_±_0.040)|0.442 (_±_0.035)|
|||HMC|**1.527**(_±_0.055)|**0.553**(_±_0.044)|
|||Laplace Approximation|**1.585**(_±_0.065)|**0.586**(_±_0.043)|
|||VI: DiagonalNormal|**1.554**(_±_0.058)|**0.586**(_±_0.042)|
|||VI: MultivariateNormal|**1.530**(_±_0.058)|**0.564**(_±_0.043)|
|Scenario 5|20|VI: Structured Normal|2.109 (_±_0.156)|1.054 (_±_0.067)|
|||VI: IAF|**1.548**(_±_0.057)|**0.562**(_±_0.043)|
|||ICL (ours)|3.545 (_±_0.288)|1.626 (_±_0.140)|
|||MAP|1.254 (_±_0.027)|0.464 (_±_0.035)|
|||TabPFN|0.668 (_±_0.064)|0.413 (_±_0.032)|
|||HMC|1.626 (_±_nan)|**0.521**(_±_0.028)|
|||Laplace Approximation|1.541 (_±_nan)|0.655 (_±_0.040)|
|||VI: DiagonalNormal|1.576 (_±_nan)|0.639 (_±_0.041)|
|||VI: MultivariateNormal|1.659 (_±_nan)|0.592 (_±_0.035)|
|Scenario 5|50|VI: Structured Normal|2.076 (_±_nan)|1.018 (_±_0.102)|
|||VI: IAF|<br>1.706 (_±_nan)|<br>0.627 (_±_0.040)|
|||ICL (ours)|10.319 (_±_nan)|1.458 (_±_0.193)|
|||MAP|1.318 (_±_nan)|0.416 (_±_0.018)|
|||TabPFN|0.330 (_±_nan)|0.443 (_±_0.024)|



Table 42: SGLD vs. ICL: Evaluation on 50 synthetic and 17 real-world datasets for six different GLM scenarios. All results within two standard errors of the best average result for each scenario are marked in **bold** . 

|**Scenario**|**Model**||**Synthetic Evaluati**|**on**|**R**|**eal-World Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Scenario 1|SGLD<br>**ICL (ours)**|0.992 (_±_0.015)<br>**0.765**(_±_0.123)|2.846 (_±_1.411)<br>**0.767**(_±_0.727)|1.951 (_±_0.917)<br>**0.585**(_±_0.301)|0.980 (_±_0.013)<br>**0.614**(_±_0.074)|2.191 (_±_1.183)<br>**0.175**(_±_0.219)|0.865 (_±_0.438)<br>**0.310**(_±_0.138)|
|Scenario 2|SGLD<br>**ICL (ours)**|0.999 (_±_0.004)<br>**0.839**(_±_0.072)|5.650 (_±_1.762)<br>**0.707**(_±_0.658)|8.295 (_±_5.629)<br>**1.111**(_±_0.300)|0.994 (_±_0.006)<br>**0.768**(_±_0.033)|2.699 (_±_1.093)<br>**0.143**(_±_0.089)|1.289 (_±_0.454)<br>**0.411**(_±_0.094)|
|Si 3|SGLD|0.997 (_±_0.008)|3.320 (_±_1.595)|3.011 (_±_1.036)|0.983 (_±_0.013)|2.152 (_±_1.194)|0.935 (_±_0.523)|
|cenaro|**ICL (ours)**|**0.611**(_±_0.070)|**0.089**(_±_0.114)|**0.423**(_±_0.348)|**0.576**(_±_0.027)|**0.037**(_±_0.026)|**0.257**(_±_0.044)|
|Scenario 4|SGLD<br>**ICL (ours)**|1.000 (_±_0.000)<br>**0.753**(_±_0.049)|6.626 (_±_1.215)<br>**0.171**(_±_0.153)|15.674 (_±_8.100)<br>**0.631**(_±_0.294)|0.994 (_±_0.006)<br>**0.762**(_±_0.015)|2.927 (_±_1.564)<br>**0.105**(_±_0.046)|1.606 (_±_1.022)<br>**0.597**(_±_0.104)|
|Scenario 5|SGLD<br>**ICL (ours)**|0.999 (_±_0.003)<br>**0.621**(_±_0.063)|3.308 (_±_1.728)<br>**0.067**(_±_0.080)|2.216 (_±_1.247)<br>**0.299**(_±_0.195)|1.000 (_±_0.000)<br>**0.610**(_±_0.045)|4.012 (_±_1.413)<br>**0.046**(_±_0.020)|0.996 (_±_0.406)<br>**0.242**(_±_0.038)|
|Si 6|SGLD|0.998 (_±_0.001)|2.681 (_±_0.565)|2.419 (_±_0.510)|0.998 (_±_0.002)|2.845 (_±_0.590)|1.851 (_±_0.319)|
|cenaro|**ICL (ours)**|**0.532**(_±_0.019)|**0.016**(_±_0.008)|**0.590**(_±_0.066)|**0.556**(_±_0.017)|**0.035**(_±_0.015)|**0.504**(_±_0.038)|



47 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 43: SGLD vs. ICL: Evaluation on 50 synthetic and 17 real-world datasets for six different FA scenarios. All results within two standard errors of the best average result for each scenario are marked in **bold** . 

|**Si**|**Mdl**|**S**|**ynthetic Evaluatio**|**n**|**R**|**eal-World Evaluat**|**ion**|
|---|---|---|---|---|---|---|---|
|**cenaro**|**oe**|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Scenario 1|SGLD<br>**ICL (ours)**|0.996 (_±_0.006)<br>**0.552**(_±_0.028)|2.883 (_±_1.552)<br>**0.034**(_±_0.034)|1.776 (_±_0.694)<br>**0.289**(_±_0.083)|0.995 (_±_0.003)<br>**0.606**(_±_0.038)|2.676 (_±_0.710)<br>**0.068**(_±_0.069)|1.608 (_±_0.381)<br>**0.265**(_±_0.078)|
|Scenario 2|SGLD<br>**ICL (ours)**|0.997 (_±_0.003)<br>**0.542**(_±_0.006)|2.950 (_±_0.786)<br>**0.017**(_±_0.006)|1.892 (_±_0.533)<br>**0.244**(_±_0.033)|0.995 (_±_0.003)<br>**0.622**(_±_0.032)|2.517 (_±_0.583)<br>**0.098**(_±_0.039)|1.500 (_±_0.268)<br>**0.287**(_±_0.046)|
|Scenario 3|SGLD<br>**ICL (ours)**|0.998 (_±_0.005)<br>**0.537**(_±_0.023)|3.662 (_±_1.099)<br>**0.024**(_±_0.021)|2.086 (_±_0.919)<br>**0.259**(_±_0.088)|0.956 (_±_0.025)<br>**0.609**(_±_0.019)|1.580 (_±_0.819)<br>**0.124**(_±_0.037)|0.311 (_±_0.108)<br>**0.179**(_±_0.018)|
|Si 4|SGLD|1.000 (_±_0.000)|4.127 (_±_0.635)|3.047 (_±_0.972)|**0.950**(_±_0.021)|**1.520**(_±_0.512)|**0.141**(_±_0.031)|
|cenaro|**ICL (ours)**|**0.684**(_±_0.060)|**0.198**(_±_0.141)|**0.918**(_±_0.246)|0.988 (_±_0.003)|1.764 (_±_0.026)|1.248 (_±_0.008)|
|Si 5|SGLD|0.999 (_±_0.001)|3.465 (_±_0.939)|1.981 (_±_0.938)|0.962 (_±_0.024)|1.945 (_±_1.383)|**0.393**(_±_0.243)|
|cenaro|**ICL (ours)**|**0.535**(_±_0.016)|**0.021**(_±_0.011)|**0.279**(_±_0.060)|**0.886**(_±_0.017)|**1.207**(_±_0.101)|1.002 (_±_0.042)|
|Scenario 6|SGLD<br>**ICL (ours)**|0.997 (_±_0.004)<br>**0.543**(_±_0.021)|3.395 (_±_1.199)<br>**0.023**(_±_0.015)|2.358 (_±_1.458)<br>**0.345**(_±_0.173)|0.950 (_±_0.040)<br>**0.666**(_±_0.020)|2.177 (_±_1.643)<br>**0.200**(_±_0.034)|0.342 (_±_0.224)<br>**0.224**(_±_0.014)|



Table 44: SGLD vs. ICL: Evaluation on 50 synthetic and 17 real-world datasets for four different GMM scenarios. All results within two standard errors of the best average result for each scenario are marked in **bold** . 

|**Scenario**|**Model**||**Synthetic Evaluati**|**on**|**R**|**eal-World Evaluati**|**on**|
|---|---|---|---|---|---|---|---|
|||C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|C2ST (_↓_)|MMD (_↓_)|_W_2(_↓_)|
|Si 1|SGLD|1.000 (_±_0.001)|2.629 (_±_0.868)|3.279 (_±_1.330)|1.000 (_±_0.000)|3.421 (_±_0.877)|6.510 (_±_1.763)|
|cenaro|**ICL (ours)**|**0.760**(_±_0.092)|**0.303**(_±_0.548)|**2.095**(_±_1.692)|**0.847**(_±_0.082)|**0.486**(_±_0.623)|**4.054**(_±_2.782)|
|Si 2|SGLD|1.000 (_±_0.000)|3.046 (_±_1.041)|6.015 (_±_4.265)|1.000 (_±_0.000)|2.487 (_±_0.521)|6.858 (_±_1.618)|
|cenaro|**ICL (ours)**|**0.812**(_±_0.061)|**0.159**(_±_0.154)|**2.314**(_±_0.926)|**0.937**(_±_0.041)|**0.282**(_±_0.131)|**3.947**(_±_1.055)|
|Si 3|SGLD|1.000 (_±_0.000)|4.631 (_±_1.169)|23.247 (_±_30.646)|1.000 (_±_0.000)|2.655 (_±_0.437)|26.356 (_±_2.699)|
|cenaro|**ICL (ours)**|**1.000**(_±_0.000)|**0.582**(_±_0.280)|**8.708**(_±_4.945)|**1.000**(_±_0.000)|**1.869**(_±_0.342)|**33.230**(_±_8.095)|
|Scenario 4|SGLD|**1.000**(_±_0.000)|3.464(_±_1.098)|**6.995**(_±_5.554)|**1.000**(_±_0.000)|**2.555**(_±_0.494)|**9.477**(_±_3.432)|
||**ICL (ours)**|**1.000**(_±_0.000)|2.451 (_±_0.868)|**8.333**(_±_4.202)|1.000 (_±_0.000)|**2.518**(_±_0.694)|**11.938**(_±_2.956)|



48 

**Can Transformers Learn Full Bayesian Inference In Context?** 

# **Q. Evaluation the choice of classifier for the C2ST metric** 

In this section, we validate the choice of the classifier for the C2ST metric by comparing the ROC characteristic of a random forest (our default choice) and a neural network in distinguishing posterior samples. In summary, we find that despite minor differences, the two metrics yield the same overall results. Across all scenarios, both Random Forest (RF) and Neural Network NN classifiers yield quite consistent rankings of model performance with only insubstantial deviations in terms of the big picture. In particular, ICL is consistently among the top-performing approaches under both evaluation metrics. Out of the 14 total scenario–domain combinations (7 scenarios × 2 dataset types), the RF and NN metrics identify the same best-performing model in 12 cases. 

49 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 45: Generalized Linear Models: Comparison of C2ST scores with a Random Forest (RF) and a Neural Network (NN). For the NN we follow the setup of Lueckmann et al., 2021. Evaluation across seven distinct scenarios on 50 synthetic and 17 real-world datasets. All results within two standard errors of the best average result in each scenario are marked in **bold** . 

|**Scenario**|**Model**|**Synthetic**|**Evaluation**|**Real-World**|**Evaluation**|
|---|---|---|---|---|---|
|||C2ST RF (_↓_)|C2ST NN (_↓_)|C2ST RF (_↓_)|C2ST NN (_↓_)|
||Laplace Approximation|1.000 (_±_0.000)|0.998 (_±_0.000)|1.000 (_±_0.000)|0.998 (_±_0.000)|
||VI: DiagonalNormal|0.904 (_±_0.076)|0.857 (_±_0.001)|0.797 (_±_0.083)|0.803 (_±_0.004)|
||VI: MultivariateNormal|**0.750**(_±_0.128)|0.780 (_±_0.002)|**0.607**(_±_0.070)|0.713 (_±_0.004)|
|Scenario 1|VI: Structured Normal|<br>**0.753**(_±_0.126)|<br>0.781 (_±_0.002)|<br>**0.600**(_±_0.070)|<br>0.705 (_±_0.004)|
||VI: IAF|**0.777**(_±_0.122)|0.793 (_±_0.002)|0.683 (_±_0.132)|0.746 (_±_0.006)|
||HMC|**0.745**(_±_0.130)|0.777 (_±_0.002)|**0.595**(_±_0.075)|**0.702**(_±_0.004)|
||**ICL (ours)**|<br>**0.765**(_±_0.123)|<br>**0.712**(_±_0.002)|<br>**0.614**(_±_0.074)|<br>**0.701**(_±_0.004)|
||Laplace Approximation|1.000 (_±_0.000)|0.998 (_±_0.000)|1.000 (_±_0.000)|0.998 (_±_0.000)|
||<br>VI: DiagonalNormal|0.957 (_±_0.091)|0.883 (_±_0.002)|0.892 (_±_0.044)|0.851 (_±_0.003)|
||<br>VI: MultivariateNormal|<br>0.910 (_±_0.131)|<br>0.860 (_±_0.002)|<br>**0.820**(_±_0.031)|<br>0.815 (_±_0.003)|
|Scenario 2|VI: Structured Normal|0.908 (_±_0.119)|0.859 (_±_0.002)|**0.824**(_±_0.023)|0.817 (_±_0.003)|
||VI: IAF|0.968 (_±_0.063)|0.889 (_±_0.001)|0.888 (_±_0.067)|0.849 (_±_0.004)|
||**ICL (ours)**|**0.839**(_±_0.072)|**0.824**(_±_0.001)|**0.768**(_±_0.033)|**0.789**(_±_0.003)|
||Laplace Approximation|1.000 (_±_0.000)|0.998 (_±_0.000)|1.000 (_±_0.000)|0.998 (_±_0.000)|
||VI: DiagonalNormal|0.866 (_±_0.101)|0.838 (_±_0.002)|0.797 (_±_0.083)|0.803 (_±_0.004)|
||VI: MultivariateNormal|0.656 (_±_0.131)|0.733 (_±_0.002)|**0.590**(_±_0.035)|0.685 (_±_0.003)|
|Scenario 3|VI: Structured Normal|0.653 (_±_0.125)|0.731 (_±_0.002)|**0.582**(_±_0.028)|0.681 (_±_0.003)|
||VI: IAF|0.751 (_±_0.148)|0.780 (_±_0.002)|0.673 (_±_0.141)|0.741 (_±_0.006)|
||**ICL (ours)**|**0.611**(_±_0.070)|**0.710**(_±_0.001)|**0.576**(_±_0.027)|**0.693**(_±_0.003)|
||Laplace Approximation|1.000 (_±_0.000)|0.998 (_±_0.000)|1.000 (_±_0.000)|0.998 (_±_0.000)|
||VI: DiagonalNormal|0.968 (_±_0.036)|0.889 (_±_0.001)|0.916 (_±_0.040)|0.863 (_±_0.003)|
|i|VI: MultivariateNormal|0.855 (_±_0.123)|0.832 (_±_0.002)|0.771 (_±_0.017)|0.790 (_±_0.002)|
|Scenaro 4|VI: Structured Normal|0.847 (_±_0.116)|0.828 (_±_0.002)|0.769 (_±_0.012)|0.789 (_±_0.002)|
||VI: IAF|0.942 (_±_0.077)|0.876 (_±_0.001)|0.833 (_±_0.069)|0.821 (_±_0.004)|
||**ICL (ours)**|**0.753**(_±_0.049)|**0.781**(_±_0.001)|**0.762**(_±_0.015)|**0.786**(_±_0.002)|
||Laplace Approximation|1.000 (_±_0.000)|0.998 (_±_0.000)|1.000 (_±_0.000)|0.998 (_±_0.000)|
||VI: DiagonalNormal|0.866 (_±_0.085)|0.838 (_±_0.002)|0.810 (_±_0.036)|0.810 (_±_0.003)|
||VI: MultivariateNormal|0.765 (_±_0.100)|0.787 (_±_0.002)|0.711 (_±_0.038)|0.760 (_±_0.003)|
|Scenario 5|VI: Structured Normal|<br>0.758 (_±_0.098)|<br>0.784 (_±_0.002)|<br>0.705 (_±_0.032)|<br>0.757 (_±_0.003)|
||VI: IAF|<br>0.814 (_±_0.105)|<br>0.812 (_±_0.002)|<br>0.777 (_±_0.106)|<br>0.793 (_±_0.005)|
||**ICL (ours)**|**0.621**(_±_0.063)|**0.715**(_±_0.001)|**0.610**(_±_0.045)|**0.710**(_±_0.003)|
||Laplace Approximation|1.000 (_±_0.000)|0.998 (_±_0.000)|1.000 (_±_0.000)|0.998 (_±_0.000)|
||<br>VI: DiagonalNormal|<br>0.724 (_±_0.060)|<br>0.767 (_±_0.001)|<br>0.703 (_±_0.039)|<br>0.756 (_±_0.003)|
||<br>VI: MultivariateNormal|**0.534**(_±_0.018)|**0.672**(_±_0.001)|**0.538**(_±_0.019)|0.674 (_±_0.002)|
|Scenario 6|VI: Structured Normal|**0.536**(_±_0.016)|**0.673**(_±_0.001)|**0.536**(_±_0.019)|0.673 (_±_0.002)|
||VI: IAF|<br>0.542 (_±_0.026)|<br>0.676 (_±_0.001)|<br>**0.535**(_±_0.015)|<br>0.672 (_±_0.002)|
||**ICL (ours)**|<br>**0.532**(_±_0.019)|<br>**0.671**(_±_0.001)|<br>0.556 (_±_0.017)|<br>**0.653**(_±_0.002)|
||Laplace Approximation|1.000 (_±_0.000)|0.998 (_±_0.000)|1.000 (_±_0.000)|0.998 (_±_0.000)|
||<br>VI: DiagonalNormal|<br>0.938 (_±_0.074)|<br>0.874 (_±_0.001)|<br>0.936 (_±_0.024)|<br>0.873 (_±_0.003)|
||VI: MultivariateNormal|0.814 (_±_0.181)|0.812 (_±_0.002)|**0.741**(_±_0.020)|0.775 (_±_0.003)|
|Scenario 7|VI: Structured Normal|<br>0.824 (_±_0.177)|<br>0.817 (_±_0.002)|<br>**0.734**(_±_0.025)|<br>0.772 (_±_0.003)|
||VI: IAF|0.939 (_±_0.091)|0.874 (_±_0.002)|0.864 (_±_0.093)|0.837 (_±_0.005)|
||**ICL (ours)**|**0.700**(_±_0.116)|**0.721**(_±_0.002)|0.773 (_±_0.048)|**0.751**(_±_0.003)|



50 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 46: Factor Analysis: Comparison of C2ST scores using a Random Forest (RF) and a Neural Network (NN) classifier across six different scenarios on 50 synthetic and 17 real-world datasets. For the NN we follow the setup of Lueckmann et al., 2021. All results within two standard errors of the best average result in each scenario are marked in **bold** . 

|**Si**|**Mdl**|**Synthetic**|**Evaluation**|**Real-World**|**Evaluation**|
|---|---|---|---|---|---|
|**cenaro**|**oe**|C2ST RF (_↓_)|C2ST NN (_↓_)|C2ST RF (_↓_)|C2ST NN (_↓_)|
||Laplace Approximation|1.000 (_±_0.000)|0.997 (_±_0.000)|1.000 (_±_0.000)|0.997 (_±_0.000)|
||VI: DiagonalNormal|1.000 (_±_0.001)|0.997 (_±_0.000)|0.979 (_±_0.008)|0.950 (_±_0.001)|
||VI: MultivariateNormal|0.998 (_±_0.003)|0.960 (_±_0.000)|0.966 (_±_0.010)|0.944 (_±_0.001)|
|Scenario 1|VI: Structured Normal|0.997 (_±_0.004)|0.959 (_±_0.000)|0.979 (_±_0.010)|0.950 (_±_0.001)|
||VI: IAF|0.953 (_±_0.104)|0.937 (_±_0.001)|0.849 (_±_0.075)|0.885 (_±_0.003)|
||**ICL (ours)**|**0.552**(_±_0.028)|**0.737**(_±_0.000)|**0.606**(_±_0.038)|**0.764**(_±_0.001)|
||Laplace Approximation|1.000 (_±_0.000)|0.997 (_±_0.000)|1.000 (_±_0.000)|0.997 (_±_0.000)|
||VI: DiagonalNormal|0.998 (_±_0.002)|0.960 (_±_0.000)|0.975 (_±_0.010)|0.948 (_±_0.001)|
||VI: MultivariateNormal|0.989 (_±_0.009)|0.955 (_±_0.000)|0.951 (_±_0.025)|0.936 (_±_0.001)|
|Scenario 2|VI: Structured Normal|0.984 (_±_0.031)|0.953 (_±_0.000)|0.958 (_±_0.025)|0.940 (_±_0.001)|
||VI: IAF|0.966 (_±_0.066)|0.944 (_±_0.001)|0.799 (_±_0.058)|0.860 (_±_0.002)|
||**ICL (ours)**|**0.542**(_±_0.006)|**0.732**(_±_0.000)|**0.622**(_±_0.032)|**0.772**(_±_0.001)|
||Laplace Approximation|1.000 (_±_0.000)|0.997 (_±_0.000)|1.000 (_±_0.000)|0.997 (_±_0.000)|
||VI: DiagonalNormal|0.999 (_±_0.002)|0.960 (_±_0.000)|0.951 (_±_0.007)|0.936 (_±_0.001)|
||VI: MultivariateNormal|0.994 (_±_0.007)|0.958 (_±_0.000)|0.945 (_±_0.007)|0.933 (_±_0.001)|
|Scenario 3|VI: Structured Normal|0.997 (_±_0.003)|0.959 (_±_0.000)|0.942 (_±_0.009)|0.932 (_±_0.001)|
||VI: IAF|0.990 (_±_0.011)|0.987 (_±_0.000)|0.928 (_±_0.015)|0.925 (_±_0.001)|
||**ICL (ours)**|**0.537**(_±_0.023)|**0.729**(_±_0.000)|**0.609**(_±_0.019)|**0.765**(_±_0.001)|
||Laplace Approximation|1.000 (_±_0.000)|0.997 (_±_0.000)|1.000 (_±_0.000)|0.997 (_±_0.000)|
||VI: DiagonalNormal|1.000 (_±_0.000)|0.997 (_±_0.000)|0.977 (_±_0.003)|0.949 (_±_0.000)|
||VI: MultivariateNormal|0.999 (_±_0.001)|0.960 (_±_0.000)|0.973 (_±_0.008)|0.947 (_±_0.001)|
|Scenario 4|VI: Structured Normal|1.000 (_±_0.000)|0.997 (_±_0.000)|0.973 (_±_0.007)|0.947 (_±_0.001)|
||VI: IAF|0.999 (_±_0.001)|0.960 (_±_0.000)|**0.961**(_±_0.018)|0.941 (_±_0.001)|
||**ICL (ours)**|**0.684**(_±_0.060)|**0.803**(_±_0.001)|0.988 (_±_0.003)|**0.955**(_±_0.000)|
||Laplace Approximation|1.000 (_±_0.000)|0.997 (_±_0.000)|1.000 (_±_0.000)|0.997 (_±_0.000)|
||VI: DiagonalNormal|0.999 (_±_0.002)|0.960 (_±_0.000)|0.944 (_±_0.010)|0.933 (_±_0.001)|
||VI: MultivariateNormal|0.995 (_±_0.007)|0.958 (_±_0.000)|0.930 (_±_0.017)|0.926 (_±_0.001)|
|Scenario 5|VI: Structured Normal|0.998 (_±_0.005)|0.960 (_±_0.000)|0.934 (_±_0.011)|0.928 (_±_0.001)|
||VI: IAF|0.992 (_±_0.012)|0.957 (_±_0.000)|0.910 (_±_0.011)|0.916 (_±_0.001)|
||**ICL (ours)**|**0.535**(_±_0.016)|**0.728**(_±_0.000)|**0.886**(_±_0.017)|**0.904**(_±_0.001)|
||Laplace Approximation|1.000 (_±_0.000)|0.997 (_±_0.000)|1.000 (_±_0.000)|0.997 (_±_0.000)|
||VI: DiagonalNormal|0.998 (_±_0.002)|0.960 (_±_0.000)|0.949 (_±_0.008)|0.935 (_±_0.001)|
||VI: MultivariateNormal|0.991 (_±_0.013)|0.956 (_±_0.000)|0.938 (_±_0.009)|0.930 (_±_0.001)|
|Scenario 6|VI: Structured Normal|0.997 (_±_0.005)|0.959 (_±_0.000)|0.944 (_±_0.006)|0.933 (_±_0.001)|
||VI: IAF|0.989 (_±_0.029)|0.955 (_±_0.000)|0.865 (_±_0.027)|0.893 (_±_0.001)|
||**ICL (ours)**|**0.543**(_±_0.021)|**0.732**(_±_0.000)|**0.666**(_±_0.020)|**0.794**(_±_0.001)|



51 

**Can Transformers Learn Full Bayesian Inference In Context?** 

Table 47: Gaussian Mixture Models: Comparison of C2ST scores using a Random Forest (RF) and a Neural Network (NN) classifier across six distinct scenarios on 50 synthetic and 17 real-world datasets. All results within two standard errors of the best average result in each scenario are marked in **bold** . For the NN we follow the setup of Lueckmann et al., 2021. Both RF and NN classifiers yield consistent rankings, with ICL emerging as the top method in scenarios with more pronounced model mismatch. 

|||**Synthetic**|**Evaluation**|**Real-Worl**|**d Evaluation**|
|---|---|---|---|---|---|
|**Scenario**|**Model**|||||
|||C2ST RF (_↓_)|C2ST NN (_↓_)|C2ST RF (_↓_)|C2ST NN (_↓_)|
||Laplace Approximation|1.000 (_±_0.000)|0.997 (_±_0.000)|1.000 (_±_0.000)|0.997 (_±_0.000)|
||VI: DiagonalNormal|0.988 (_±_0.013)|1.012 (_±_0.000)|0.995 (_±_0.006)|0.996 (_±_0.001)|
||VI: MultivariateNormal|0.988 (_±_0.013)|1.012 (_±_0.000)|0.994 (_±_0.007)|0.993 (_±_0.001)|
|Scenario 1|VI: Structured Normal|0.987 (_±_0.015)|0.982 (_±_0.000)|0.993 (_±_0.009)|0.992 (_±_0.001)|
||VI: IAF|0.989 (_±_0.013)|0.983 (_±_0.000)|0.995 (_±_0.010)|0.996 (_±_0.001)|
||**ICL (ours)**|**0.760**(_±_0.092)|**0.825**(_±_0.001)|**0.847**(_±_0.082)|**0.869**(_±_0.003)|
||Laplace Approximation|1.000 (_±_0.000)|0.997 (_±_0.000)|1.000 (_±_0.000)|0.997 (_±_0.000)|
||VI: DiagonalNormal|0.989 (_±_0.024)|0.983 (_±_0.000)|0.998 (_±_0.003)|0.997 (_±_0.001)|
||VI: MultivariateNormal|0.991 (_±_0.021)|0.991 (_±_0.000)|0.999 (_±_0.002)|1.002 (_±_0.001)|
|Scenario 2|VI: Structured Normal|0.992 (_±_0.017)|0.988 (_±_0.000)|0.999 (_±_0.002)|1.002 (_±_0.001)|
||VI: IAF|0.992 (_±_0.021)|0.988 (_±_0.000)|0.998 (_±_0.004)|0.997 (_±_0.001)|
||**ICL (ours)**|**0.812**(_±_0.061)|**0.851**(_±_0.001)|**0.937**(_±_0.041)|**0.915**(_±_0.002)|
||Laplace Approximation|1.000 (_±_0.000)|0.997 (_±_0.000)|1.000 (_±_0.000)|0.997 (_±_0.000)|
||VI: DiagonalNormal|**0.996**(_±_0.011)|1.004 (_±_0.000)|**0.992**(_±_0.018)|0.988 (_±_0.001)|
||VI: MultivariateNormal|0.997 (_±_0.009)|1.007 (_±_0.000)|**0.993**(_±_0.016)|0.992 (_±_0.001)|
|Scenario 3|VI: Structured Normal|**0.995**(_±_0.017)|0.996 (_±_0.000)|**0.993**(_±_0.016)|0.992 (_±_0.001)|
||VI: IAF|**0.994**(_±_0.018)|0.993 (_±_0.000)|**0.993**(_±_0.017)|0.992 (_±_0.001)|
||**ICL (ours)**|1.000 (_±_0.000)|**0.997**(_±_0.000)|1.000 (_±_0.000)|**0.997**(_±_0.000)|
||Laplace Approximation|1.000 (_±_0.000)|0.997 (_±_0.000)|1.000 (_±_0.000)|0.997 (_±_0.000)|
||VI: DiagonalNormal|**1.000**(_±_0.002)|0.997 (_±_0.000)|1.000 (_±_0.000)|0.997 (_±_0.000)|
||VI: MultivariateNormal|**1.000**(_±_0.002)|0.997 (_±_0.000)|1.000 (_±_0.000)|0.997 (_±_0.000)|
|Scenario 4|VI: Structured Normal|**1.000**(_±_0.001)|0.997 (_±_0.000)|**0.996**(_±_0.016)|1.004 (_±_0.001)|
||VI: IAF|**1.000**(_±_0.002)|0.997 (_±_0.000)|1.000 (_±_0.000)|0.997 (_±_0.000)|
||**ICL (ours)**|1.000 (_±_0.000)|**0.997**(_±_0.000)|**1.000**(_±_0.000)|**0.997**(_±_0.000)|



52 

