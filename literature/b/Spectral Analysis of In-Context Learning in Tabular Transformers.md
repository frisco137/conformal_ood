# **Mechanistic Evidence for Spectral Structures in Prior-Data Fitted Networks** 

**Kaustubh Sharma**<sup>1</sup><sup>_∗_</sup> **, Srijan Tiwari**<sup>1</sup><sup>_∗_</sup> **, Ojasva Nema**<sup>2</sup> **,**<sup>_∗_</sup> **and Parikshit Pareek**<sup>1</sup><sup>_†_</sup> Indian Institute of Technology Roorkee (IIT Roorkee), Uttarakhand, India, `{pareek}@ee.iitr.ac.in` ; 

### **Abstract** 

Prior-Data Fitted Networks (PFNs) enable amortized Bayesian inference in a single forward pass, yet their internal representations remain opaque. It is unknown whether PFNs encode identifiable Bayesian structure or merely memorize inputoutput mappings. We provide mechanistic evidence that PFNs learn structured spectral representations and that these can be extracted as explicit kernels. First, probing experiments across three architectures, including the publicly released TabPFN, show that spectral information is linearly decodable from the latent attention score and organized along a dominant principal axis. Activation patching and targeted subspace interventions establish that this information is causally used for prediction and concentrated in a low-dimensional subspace, with spectral directions an order of magnitude more effective than random ones. Crucially, these properties hold on TabPFN with both synthetic out-of-distribution inputs and real-world time series (Airline Passengers, Milk Production), indicating they are emergent features of PFN-style amortization over continuous regression tasks rather than artifacts of training prior. Second, we introduce a Filter Bank Decoder that maps frozen PFN latents to explicit spectral densities, reconstructing stationary kernels via Bochner’s theorem. The resulting kernels support GP regression competitive with iterative baselines while requiring only a single forward pass, demonstrating that PFN priors are not merely implicit but are explicitly recoverable as portable Bayesian objects. 

### **1 Introduction** 

Bayesian Inference methods like Gaussian processes (GPs) provide a principled way to learn and reason under uncertainty. The explicit representation in terms of kernel is valuable wherever understanding the data-generating process is as important as making accurate predictions. However, inference with expressive kernels is computationally expensive, and restricted to fixed kernel families, limiting scalability and repeated use Rasmussen and Williams [2006]. 

Prior-Data Fitted Networks (PFNs) offer a resolution to this computational bottleneck Müller et al. [2022b]. A PFN is trained on a distribution of synthetic tasks sampled from a prior, learning to map context data directly to posterior predictive distributions (PPD) in a single forward pass. This enables efficient inference across new datasets drawn from the same prior family. 

This efficiency comes at the cost of **structural opacity** . In a classical GP, the kernel is an explicit, interpretable, portable object that can be inspected, composed Duvenaud et al. [2013], and transferred to new tasks. In a PFN, the corresponding structure is considered to be implicitly absorbed into the network’s weights and activations. The model produces approximate PPD, but it is not known if the hidden Bayesian object has any interpretable correlation with the input. It is also unknown whether 

> _∗_ Equal Contribution. _†_ Corresponding Author. 1Department of Electrical Engineering, 2Department of Metallurgical and Materials Engineering, The authors acknowledge the funding support provided by the ANRF PM Early Career Research Grant (ANRF/ECRG/2024/001962/ENS) and the IIT Roorkee Faculty Initiation Grant (IITR/SRIC/1431/FIG-101078) 

Preprint. 

any such object, like a kernel matrix, can be extracted from frozen PFN and provides comparative downstream performance as other kernel design methods like Wilson et al. [2015], Duvenaud et al. [2013], Lloyd et al. [2014], Wilson and Adams [2013]. 

Recently, Müller et al. [2025] surveyed the state of PFN research and identified several open challenges that must be addressed for PFNs to serve as general-purpose Bayesian tools. Among these, two stand out as particularly fundamental: **1) the lack of mechanistic transparency** : we do not understand _where_ or _how_ Bayesian structure is represented inside the network, nor whether it plays a causal role in prediction and **2) the absence of explicit, portable representations** of the latent priors learned during amortized inference. This work addresses both gaps by asking _two complementary questions_ : **(a)** _Do PFNs encode spectral information in a structured and interpretable form, and is this information causally used in prediction?_ and **(b)** _Can this structure be extracted as explicit kernel representations?_ 

Answering these questions requires mechanistic analysis tools that go beyond current practice. Mechanistic analysis methods have been predominantly developed for and validated on large language models Bills et al. [2023], Bricken et al. [2023], Belinkov [2022], with non-language foundation models comparatively underexplored El et al. [2025]. Even within that setting, analyses are typically descriptive as they reveal what information is present but do not extract representations that can be reused for downstream tasks. This gap is particularly relevant for models such as PFNs, as highlighted recently in Müller et al. [2025]. We address this gap by showing that mechanistic structure in PFNs can not only be identified but also _operationalized_ for better PFN design and downstream tasks. 

In this paper, to answer the aforementioned question **(a)** , we provide two complementary lines of evidence. First, probing experiments demonstrate that spectral information is _linearly decodable_ and cleanly organized within the latent attention score. Second, activation patching and targeted subspace interventions establish that this information is _causally used by the network_ and compressed into a low-dimensional subspace. Crucially, we show that these phenomena are not specific to a single architecture: spectral organization emerges in TabPFN Hollmann et al. [2025], a standard PFN with joint attention trained on tabular data. A GP-specialized model, DVA-PFN Sharma et al. [2025], exhibits significantly clearer spectral structure when trained directly on spectral priors. Importantly, these properties hold across 1D sinusoidal probing inputs, 5D RBF and Matérn GP inputs, and tabular inputs, confirming the generality of the analysis. We further validate these causal findings on _real-world time series_ converted to tabular regression via lag embedding. 

For question **(b)** , building on the mechanistic foundation, we introduce a _Filter Bank Decoder_ that maps frozen PFN representations to explicit spectral density estimates. The resulting kernels support GP regression competitive with iterative baselines while requiring only a single forward pass, and enable downstream tasks with performance competitive with kernels obtained via iterative and amortized kernel discovery methods like Wilson et al. [2015], Bitzer et al. [2023], Tancik et al. [2020]. In summary, our contributions are: 

- We provide first systematic mechanistic study of spectral structure in PFNs which is linearly decodable and causally relevant latent subspace. Also we show that it is a low-dimensional subspace, consistent across architectures, and validated on real-world observational data. 

- We introduce a Filter Bank Decoder that extracts explicit, portable kernels from frozen PFN representations, enabling competitive GP regression and downstream Bayesian tasks without test-time optimization. 

Together, our results show that PFNs not only approximate Bayesian inference but also learn structured and extractable representations of prior knowledge. We also provide some evidence in directions to improve PFN design based on these mechanistic analyses. 

### **2 Background** 

#### **2.1 Prior-Data Fitted Networks** 

Recently, Müller et al. [2022b] enabled amortized Bayesian inference by training a neural set-predictor on a vast distribution of synthetic datasets. Formally, given a prior _p_ ( _D_ ) over supervised learning tasks, we sample datasets _Dk_ = _{_ ( _xi, yi_ ) _}_<sup>_N_</sup> _i_ =1<sup>_∼p_(</sup><sup>_D_).Each dataset is partitioned into a context set</sup> _D_ ctx = ( _X_ ctx _, Y_ ctx) containing observed pairs, and a query set _D_ q = ( _X_ q _, Y_ q) containing targets to 

2 

predict. The model parameters _θ_ are optimized to minimize the expected Negative Log-Likelihood loss. Further, there has been growing interest in training these PFNs on different priors for specific tasks Müller et al. [2025]. Among them, _tabular prior_ based TabPFN Grinsztajn et al. [2025] has gained significant traction due to its ability to work with real-world tabular data, while only trained on synthetic dataset. Thus, solving the data availability problem for various applications Hollmann et al. [2025], Liu and Ye [2025], Feuer et al. [2024]. We select standard vanilla attention (VA) PFN Müller et al. [2022b] and TabPFN (Version 2.5) Grinsztajn et al. [2025] along with Decoupled-Value Attention (DVA) PFN presented in Sharma et al. [2025]. Together these three provide spectrum of attention mechanisms ranging from alternating row/column attention (TabPFN), to joint input-output attention (VA-PFN) and decoupled input-output attention forcing localization (DVA-PFN). 

#### **2.2 Mechanistic Analysis Methods** 

Mechanistic analysis aims to reverse-engineer the internal computations of neural networks by identifying interpretable features, and algorithms learned during training Olah et al. [2020], Elhage et al. [2021]. Techniques like activation patching and automated circuit discovery have enabled fine-grained analysis of individual components contribution Conmy et al. [2023]. A complementary approach is the use of _probing classifiers_ Alain and Bengio [2016], Belinkov [2022]. However, as highlighted in introduction, mechanistic analysis works are limited to language models mainly. 

**Probing classifiers.** Probing classifiers are lightweight models trained to predict _properties of interest_ from frozen intermediate representations of a neural network Alain and Bengio [2016] i.e, trained to test what information models encode. A linear probe lower bounds what is linearly accessible; a higher-capacity probe can recover nonlinearly entangled information, but risks learning the task itself [Hewitt and Liang, 2019, Belinkov, 2022]. Further, probing is purely correlational: high probe accuracy demonstrates that information is _present_ in a representation, but not that the network _uses_ it during inference Belinkov [2022]. This limitation motivates the causal methods. 

**Activation patching.** Activation patching Meng et al. [2022], Vig et al. [2020] tests causality by replacing a representation at layer _ℓ_ of input A with that of input B and measuring how far the output moves toward B’s prediction (the _causal effect)_ . This technique is also referred to as causal tracing Meng et al. [2022] or interchange intervention Geiger et al. [2021], has been used extensively in language models to localize factual knowledge and identify task-specific circuits Conmy et al. [2023], Wang et al. [2023] Patching different sites disentangles the causal roles of sub-modules. 

**Targeted subspace interventions.** Full activation patching replaces an entire representation vector, leaving open the question that whether the causally relevant information occupies the full ambient dimensionality or is concentrated in a compact subspace. Targeted subspace patching Geiger et al. [2024] replaces only the top-k principal components ranked by correlation with the property of interest, thus resulting _dose–response curve_ bounds both dimensionality and geometry of causal subspace. 

#### **2.3 Spectral Representation of Stationary Kernels** 

For a stationary covariance kernel _k_ ( _τ_ ), Bochner’s theorem Bochner [1959] establishes a one-toone correspondence between the kernel and a non-negative spectral density _S_ ( _ω_ ) via the Fourier transform: _k_ ( _τ_ ) = �R<sup>_S_(</sup><sup>_ω_)</sup><sup>_e_2</sup><sup>_πiωτ dω,S_(</sup><sup>_ω_)</sup><sup>_≥_0</sup><sup>_._Thisdualitymeansthatspecifyingakernelis</sup> equivalent to specifying a spectral density: smoothness, periodicity, and long-range correlation structure are all encoded in the shape of _S_ ( _ω_ ). Building on this correspondence, Wilson and Adams [2013] introduced the _Spectral Mixture_ (SM) kernel, which models the spectral density as a mixture of Gaussians: _S_ ( _ω_ ) =<sup>�</sup><sup>_Q_</sup> _q_ =1<sup>_wq N_</sup> � _ω | µq, σq_<sup>2</sup> � _, wq >_ 0, where ,each component is characterized by a center frequency _µq_ , a bandwidth _σq_ , and a mixture weight _wq_ . Substituting the density into Bochner theorem result yields the closed-form kernel 



The SM kernel is a universal approximator in the space of stationary kernels Wilson and Adams [2013]: any continuous stationary kernel on a compact domain can be approximated arbitrarily well by choosing a sufficient number of mixture components _Q_ . This expressiveness makes the 

3 

spectral density a natural target for kernel discovery—recovering _{_ ( _µq, σq, wq_ ) _}_<sup>_Q_</sup> _q_ =1<sup>from data is</sup> equivalent to recovering the full covariance structure of the underlying process. Importantly, there has been considerable work in kernel design and discovery Wilson and Adams [2013], Wilson et al. [2015], Tancik et al. [2020] most of which requires per-sample optimization to find kernel matrix or hyperparameters except notable works like Bitzer et al. [2023]. 

### **3 Locating Spectral Structure in PFN Representations** 

We first ask how a trained PFN organizes spectral information about input. We answer this with a ladder of probes of increasing capacity, following Alain and Bengio [2016] Belinkov [2022]: each rung adds parameters and tests a stricter notion of accessibility. Throughout, we compare the three architectures Section 2.1, which vary in attention design, training prior, and scale. VA-PFN and DVA-PFN are trained by us on spectral mixture priors; TabPFN is a publicly released frozen model Grinsztajn et al. [2025] trained on structural causal models not on spectral kernels, with a tabular task format. Agreement across all three is strong evidence that our findings reflect a general property of PFNs over continuous inputs rather than an artifact of an architecture or a prior. 

#### **3.1 Parameter-Free Probing: Frequency-Driven Geometry of** _H_<sup>¯</sup> 

The most conservative probe has zero trainable parameters. We ask a simple question: _when the generating frequency f of an input sinusoid changes, does the latent H_<sup>¯</sup> _change in a correspondingly structured way?_ We generate 500 sinusoids _y_ ( _t_ ) = sin(2 _πfit_ + _ϕ_ ) with _fi ∼U_ [0 _._ 5 _,_ 5 _._ 0] Hz and random phase _ϕ ∼U_ [0 _,_ 2 _π_ ], pass each through the frozen PFN, and mean-pool the final-layer attention score over positions ( _t_ values) to obtain _H_<sup>¯</sup> _∈_ R<sup>_d_</sup> . Here, _d_ is a PFN hyperparameter reflecting dimension into which input is projected while _H_ = _QK_<sup>_T_</sup> _/√d_ are attention scores. 

First, we attempt to answer _is the global geometry of H_<sup>¯</sup> _governed by f ?_ We calculate the Pearson correlation between two vectors of pairwise distances as _ρ_ ∆ = Corr( _{_ ∆ _fij}, {_ ∆ _Hij}_ ), which measure change in latent embedding (∆ _Hij_ = _∥H_<sup>¯</sup> _i − H_<sup>¯</sup> _j∥_ 2) with change in input frequency (∆ _fij_ = _|fi − fj|_ ). This measures whether frequency-close signals are embedding-close. Second, we ask _if the frequency sensitivity is dominant in a particular direction in H_<sup>¯</sup> _?_ . For this, we report the Pearson correlation _|r|_ PC0 between the generating frequency and _H_<sup>¯</sup> projected onto its first principal component in the original _d_ -dimensional space. Let PC0 = arg max _∥v∥_ =1 _v_<sup>_⊤_</sup> Σ _v_ with Σ = _N_ <u>1</u> � _Ni_ =1<sup>( ¯</sup><sup>_Hi−µ_)( ¯</sup><sup>_Hi−µ_)</sup><sup>_⊤_.Thenwecalculate</sup><sup>_|r|_</sup><sup>`PC`</sup> 0<sup>=</sup> �� Corr� _{zi}_<sup>_N_</sup> _i_ =1<sup>_,{fi}N_</sup> _i_ =1��� with _zi_ = PC<sup>_⊤_</sup> 0<sup>( ¯</sup><sup>_Hi−µ_).This isolates the dominant mode of variation rather than the full geometry.</sup> 

Table 1: Parameter-free frequency alignment metrics _<u>ρ</u>_ ∆ and _<u>|r|</u>_ PC0 <u>(mean</u> _±_ std over 5 seeds). 

|Metric|VA-PFN|TabPFN|DVA-PFN|
|---|---|---|---|
|**_ρ_∆**|0.615_±_0.038|0.765_±_0.021|0.852_±_0.015|
|**_|r|_**PC**0**|0.812_±_0.026|0.882_±_0.010|0.981_±_0.006|



Table 1 reports both numbers for all three PFNs. Most significantly, in TabPFN, which was trained on synthetic tabular priors, _H_<sup>¯</sup> tracks frequency at _ρ_ ∆ =0 _._ 76 and _|r|_ PC0 =0 _._ 88 shows that frequency sensitivity in a PFN’s latent does not require training on spectral kernel priors, it transfers to inputs well outside the training distribution. This suggests that PFN-style amortization over continuous regression inputs reliably gives rise to this representation across the architectures tested. Also, the ordering VA _<_ TabPFN _<_ DVA directly tracks the degree to which the attention head carries input information only separates the value stream from the query–key pair, as explained next. 

**Where Does Spectral Information Live?** Repeating the parameter-free measurement on the value stream _V_<sup>¯</sup> gives a sharp architectural split: _ρV_ = 0 _._ 19 (DVA), 0 _._ 53 (VA), 0 _._ 79 (TabPFN), with the t-SNE visualizations in Figures 4a and 4b (Appendix B) confirming the same ordering. This matches how each mechanism handles the value stream: DVA-PFN routes _y_ -information to _V_<sup>¯</sup> alone and confines frequency to _H_<sup>¯</sup> ; VA lets the joint ( _x, y_ ) embedding mix into all three projections, so frequency leaks into _V_<sup>¯</sup> ; and TabPFN’s alternating row/column attention repeatedly updates value representations alongside feature context, producing the strongest leakage. Thus, we read the _H_<sup>¯</sup> – _V_<sup>¯</sup> split as architectural evidence that attention design governs the _localization_ only not the _existence_ . 

4 

#### **3.2 How Accessible Is the Spectral Signal in** _H_<sup>¯</sup> **?** 

Geometric alignment (Sec. 3.1) tells us _H_<sup>¯</sup> is _organized_ by frequency, but not whether spectral quantities can be _read out_ in a simple form. Following the probe ladder of Alain and Bengio [2016] and Hewitt and Liang [2019], we train two probes of increasing capacity on the same frozen _H_<sup>¯</sup> : a linear probe (lower bound on accessibility) and a nonlinear MLP probe (upper bound). Our hypothesis is that if these two agree with high _R_<sup>2</sup> value, it implies MLP complexity is not required and the target is encoded in an approximately linearly separable form. (Appendix A.3 details of probes). 

**Control: random weights.** To confirm that linear accessibility reflects learned structure rather than trivial input geometry Hewitt and Liang [2019], we repeat the probing experiment on a randomly initialized (untrained) VA-PFN as a control. The MLP probe still succeeds ( _R_<sup>2</sup> _≥_ 0 _._ 99), but the linear probe collapses to _R_<sup>2</sup> = 0 _._ 18 (frequency) and 0 _._ 64 (weight), confirming that training specifically organizes _H_<sup>¯</sup> into a linearly separable form Belinkov [2022]. See Figure 7 in Appendix. 

**A linear read-out is enough.** Table 2 shows that a linear probe on _H_<sup>¯</sup> recovers both frequency and weight with _R_<sup>2</sup> _≥_ 0 _._ 93 across all three PFNs. The ordering in Table 1 repeats under probing: DVA _>_ TabPFN _>_ VA. In particular, an off the shelf TabPFN, supports _R_<sup>2</sup> =0 _._ 993 for frequency recovery, the clearest single-task evidence that spectral organization is a recurring property of the amortized Bayesian predictors tested here. Further, a we train a higher-capacity MLP probe, which can recover information that is present but nonlinearly entangled. The right-hand columns of Table 2 report the gap ∆= _R_ MLP<sup>2</sup><sup>_−R_</sup> linear<sup>2.The gap is within</sup><sup>_±_0</sup><sup>_._02 for every task and every architecture, and is slightly</sup> _negative_ on VA-PFN, consistent with mild over-fitting of the larger probe. This low gap along with the _R_<sup>2</sup> _→_ 1 for both probes show that PFNs internally solved the representation-learning problem for the scalar targets we probe and frequency and weights are exposed as approximately affine functions of the coordinates of _H_<sup>¯</sup> . This rules out the hypothesis that PFNs store spectral information in a form requiring nonlinear post-processing. See Figure 9-10 for alignment with true frequency. 

Table 2: Linear vs. nonlinear probing on _H_<sup>¯</sup> . For single-component signals, both quantities are linearly decodable across all three architectures, and added MLP capacity <u>gives no consistent improvement.</u> 

|||VA-|PFN|Tab|PFN|DVA|-PFN|
|---|---|---|---|---|---|---|---|
|Task|Probe|_R_<sup>2</sup>|∆|_R_<sup>2</sup>|∆|_R_<sup>2</sup>|∆|
|Frequency|Linear|0.967|–|0.993|–|0.998|–|
||MLP|0.946|_−_0_._021|0.991|_−_0_._002|1.000|+0_._002|
|Weight|Linear|0.934|–|0.982|–|0.997|–|
||MLP|0.913|_−_0_._021|0.976|_−_0_._006|0.999|+0_._002|



**Learned rectification enables mean-pooling** Since � sin( _ωt_ ) _dt →_ 0, recovering _f_ at _R_<sup>2</sup> _→_ 1 from mean-pooled _H_<sup>¯</sup> (Table 2) implies that the PFN’s MLP layers apply a rectifying nonlinearity before aggregation, a learned analogue of a classical periodogram’s square-and-average step. This accessibility weakens for multi-component signals: recovering all four parameters ( _f_ 1 _, f_ 2 _, a_ 1 _, a_ 2) drops to _R_<sup>2</sup> = 0 _._ 50 (Table 5; Figure 8; Table 4 in Appendix), a failure of uniform aggregation we address with multi-query attention pooling in Sec. 5. 

#### **3.3 Non-sinusoidal, Higher-dimensional Inputs** 

Beyond 1D sinusoids, we repeat the parameter-free probing analysis on 5D functions drawn from GP with RBF and Matérn-3/2 kernels. For each kernel family we sample 500 functions with lengthscales ( _ℓ_ ) log-uniformly distributed over [0 _._ 05 _,_ 10], pass them through frozen TabPFN, and compute the same two metrics against the characteristic spectral frequency _f_ char = 1 _/_ (2 _πℓ_ ) (or ~~�~~ 3 _/_ 2 _/_ (2 _πℓ_ ) for Matérn). We report _ρ_ ∆ = 0 _._ 815 and _|r|_ PC0 = 0 _._ 900 for RBF, and _ρ_ ∆ = 0 _._ 861 and _|r|_ PC0 = 0 _._ 928 for Matérn-3/2 (see Table 6 in Appendix). These values are comparable to those obtained on 1D sinusoids in Table 1. This along with t-SNE plots for both inputs (Figure 5 in Appendix), confirms that the spectral organization of _H_<sup>¯</sup> generalizes beyond 1D sinusoidal signals, further supporting our claim that structured spectral encoding is a general property of PFNs over continuous inputs. 

**Layer-Wise Spectral Refinement.** We next ask how the spectral representation evolves across depth by extracting _H_<sup>¯</sup> at every layer and computing the correlation _ρ_ ∆. Both DVA and TabPFN exhibit a characteristic _rise–plateau–decline trajectory_ (Figure 6, Appendix). The early saturation confirms 

5 

that a single cross-attention step suffices to fuse positional and value information into a spectrally meaningful latent, while the late-stage dip is consistent with the final layers reallocating capacity from geometric separation toward formatting the calibrated posterior predictive distribution. 

### **4 Is the Encoding Causal?** 

Previous section observations are correlational as a high probe _R_<sup>2</sup> shows that information is _present_ in the representation, not that the network _uses_ it during inference Belinkov [2022]. We close this gap with two _interventional experiments_ on the frozen PFNs: **(a) Activation patching** Meng et al. [2022], Vig et al. [2020] to show that _H_<sup>¯</sup> is a causal carrier of spectral identity. **b Targeted subspace patching** to see if causally-active information occupies the full latent or a compact subspace Geiger et al. [2021]. We than perform patching using tabular data for in-distribution assessment of TabPFN. 

#### **4.1 Activation Patching** 

We run two sinusoidal signals _A_ and _B_ (frequency _fA_ and _fB_ ), through the frozen DVA-PFN and cache intermediate representations at every layer. We then _patch_ (replace) the representation of signal _A_ with that of signal _B_ at layer _ℓ_ and continue the forward pass. The **Causal Effect** (CE, (2)) is the fraction by which the prediction shifts from _A_ toward _B_ and CE = 1 corresponds to a complete identity transfer. Three intervention sites isolate distinct causal pathways. The **H-patch** ( **h**<sup>(</sup> _A_<sup>_ℓ_)</sup><sup>_←_</sup><sup>**h**(</sup> _B_<sup>_ℓ_))</sup> tests our central claim, that the latent attention score causally carries spectral identity. The **V-patch** ( **v** _A ←_ **v** _B_ ) is a positive control: _V_ encodes only the raw function values in DVA-PFN so replacing it is equivalent to replacing the input data and CE should approach unity. The **K-patch** ( **k** _A ←_ **k** _B_ ) is a negative control: _A_ and _B_ share the same _t_ -grid in sin(2 _πft_ + _ϕ_ ), so _KA ≈ KB_ and intervention should have negligible effect. We evaluate _n_ = 50 random pairs ( _f ∈_ [0 _._ 5 _,_ 5 _._ 0] Hz, minimum gap _≥_ 1 _._ 0 Hz) across all six layers in DVA-PFN. 

Further, replacing _H_<sup>¯</sup> at any layer _ℓ ≥_ 2 shifts the prediction completely to match the donor signal (CE = 0 _._ 999, _p <_ 10<sup>_−_133</sup> against the K-patch baseline). Thus, solidifying the argument that _H_<sup>¯</sup> is causal carrier of spectral information (Table 9). Second, like Figure 6 for both DVA and TabPFN, spectral encoding emerges in the first attention step. The sharp _L_ 1 _→ L_ 2 transition (CE 0 _→_ 0 _._ 999) shows that a single cross-attention layer suffices to fuse spectral information into a spectrally-meaningful _H_<sup>¯</sup> , and subsequent layers maintain rather than build this representation. Third, _K_ carries no spectral information at any layer (CE = 0), corroborating the attention separation in DVA. 

#### **4.2 Targeted Subspace Patching** 

Full _H_<sup>¯</sup> -patching establishes that spectral information is causally read from the latent, but leaves open a structural question: is this information distributed diffusely across the _d_ -dimensional latent, or concentrated in a separable subspace? We localize the causally relevant directions and bound their dimensionality with a _dose–response curve_ for all three PFNs. We collect _H_<sup>¯</sup> representations at Layer 2 for 2 _,_ 500 signals (500 frequencies _×_ 5 phases) and run PCA on the resulting embeddings for all PFNs. Each principal component is ranked by _|r|_ , the absolute Pearson correlation between its projection and the generating frequency: the top- _k_ components define the _spectral subspace_ , the bottom- _k_ the _non-spectral subspace_ , and _k_ random orthogonal directions a baseline. We then patch only the selected _k_ PC dimensions of _H_<sup>¯</sup> _A_ with the corresponding components of _H_<sup>¯</sup> _B_ , leaving the remaining _d − k_ dimensions intact, and sweep _k_ from 1 to _d_ across all three architectures. 

The strongest test case here is TabPFN. Figure 1 (left) shows that patching just the top few spectral PCs of TabPFN’s _H_<sup>¯</sup> shifts predictions toward the donor signal, while patching an equal number of non-spectral or random directions changes predictions meaningfully only when _k >_ 100 out of 192. Note that Table 1 already revealed a dominant spectral axis in its latent _|r|_ PC0 = 0 _._ 882 (See Figure 12 in Appendix). This is a clear evidence that compact, causally-active spectral coding is not due to of our pretraining choices as it emerges in a model we did not train and inferred on signals well outside its training distribution i.e. an emergent property of PFNs. DVA-PFN and VA-PFN also replicate the pattern with CE _→_ 1 much faster compared to random and non-spectral baselines. See Table 10 for DVA-PFN’s quantitative results with Sinusoids and Figure 15 for Dose-curves with RBF-GP for DVA-PFN and single block patching of TabPFN. 

6 



<!-- Start of picture text -->
TabPFN VA-PFN DVA-PFN<br>1.00 1.00 1.00<br>0.75 0.75 0.75<br>0.50 0.50 0.50<br>Spectra l<br>0.25 0.25 0.25<br>Non-Spectral<br>Random (20 runs)<br>0.00 0.00 0.00<br>0 25 50 75 100 125 150 175 0 25 50 75 100 125 0 25 50 75 100 125<br>Dimension (k) Dimension (k) Dimension (k)<br>Causal Effect (CE)<br><!-- End of picture text -->

Figure 1: Causal dose–response under targeted subspace patching across architectures. Each panel sweeps patched directions _k_ from 1 to _d_ and reports the causal effect (CE) for spectral (red), nonspectral (blue), or random (green, 20 trials) subspaces. 

Together, these patching experiments yield two-level causal account. **(i)** _H_<sup>¯</sup> is a causal carrier of spectral information: replacing it transfers spectral identity in full (CE _≈_ 1). **(ii)** The information within _H_<sup>¯</sup> is compactly organized: it concentrates in a low-dimensional subspace whose dominant axis aligns with the generating frequency, and targeted intervention on this subspace is more effective, 1–2 orders of magnitude in the small-k regime, than random directions of the same size. This closes the correlational–causal loop opened in Sec. 3.2 and shows that _the spectral information probes recover from H_<sup>¯</sup> _is not a passive residue of the input but an organized representation that the network constructs in its first cross-attention step and uses for prediction._ 



<!-- Start of picture text -->
7.55.0 Airline Milk 1.0 0.98 0.89 1.00.8 SpectralNon-spectralRandom<br>2.5 0.8<br>0.0 0.6 Structured H-patch 0.6<br>Random H-patch (control) 0.4<br>2.5 0.4<br>5.0 0.2<br>0.2<br>7.5 20 10 0 10 20 0.0 0.01 0.00 0.0 48 16 32 64 128 192<br>t-SNE Dimension 1 Airline   Milk Milk   Airline Subspace Dimensionality k<br>(a) Latent t-SNE (Block 23). (b) Full-tensor patching. (c) Subspace dose-response.<br>t-SNE Dimension 2 Causal Effect (CE) Causal Effect (CE)<br><!-- End of picture text -->

Figure 2: **Causal patching of TabPFN on real-world data. Left:** t-SNE of mean-pooled _H_ embeddings. **Middle:** Full _H_ -replacement and random replacement control. **Right:** Dose-Response curve replacing only top- _k_ PCA directions (spectral), bottom- _k_ (non-spectral) and random. 

#### **4.3 Causal Structure in Real-World Time Series** 

Above sections use synthetic signals only. We now validate on real time series. We apply the same patching protocol to two classic time series: **a)** monthly _Airline Passengers_ (Box & Jenkins, 1949–1960) and **b)** _Milk Production_ (USDA, 1962–1975). These series share seasonal periodicity ( _≈_ 12 months) but differ in trend structure, amplitude dynamics, and noise profile, making them a discriminative pair for testing whether H encodes series-specific spectral identity rather than a generic seasonal template. Each dataset is converted to a 5D tabular regression task via lag ˆ embedding ( _Xt_ = [ _yt−_ 1 _, . . . , yt−_ 5] _, y_ = _yt_ ; see Appendix C.4 for details). The latent representations at TabPFN’s final block clearly separate these series in Figure 2a. Full-tensor _H_ -replacement at TabPFN’s last block achieves CE = 0 _._ 98 (Airline _←_ Milk) and 0 _._ 89 (reverse), while random replacement gives CE _≈_ 0 in Figure 2b. Targeted subspace patching in Figure 2c reveals that the top- _k_ PCA directions most correlated with series identity are roughly twice as causally efficient as the bottom- _k_ ( _k_ =64 i.e.<sup>1</sup> _/_ 3<sup>rd</sup> of _d_ ), the spectral subspace recovers CE=0 _._ 80 versus 0 _._ 43 for the non-spectral subspace. Further, we note that the tabular patching experiment (Appendix C.3, Figure 17) demonstrates that the causal role of _H_ extends beyond spectral identity to feature-relevance structure on TabPFN’s native 8D inputs, suggesting the spectral organization we identify is one manifestation of a broader structural encoding rather than an isolated phenomenon. 

### **5 Decoding Bayesian Structure: From Latents to Explicit Kernels** 

Last two sections established that _H_<sup>¯</sup> contains spectral information that is linearly accessible and causally used for prediction. Now we hypothesize that if _H_<sup>¯</sup> truly encodes input information in context of prior, kernel matrix should also be extractable. Kernel matrix is a fundamental Bayesian object providing both interpretability and downstream task capability. We pick a stationary kernel _k_ ( _τ_ ), a 

7 

spectral density _S_ ( _ω_ ) as targets of extraction. Note that here the proposed decoder is not designed to compete with iterative kernel discovery methods Lloyd et al. [2014], Duvenaud et al. [2013] (re-optimize per task) or amortized kerned discovery methods Bitzer et al. [2023]. Rather, it serves as a constructive proof that the spectral structure identified in Sections 3–4 is rich enough to reconstruct a functional covariance which is one of the most demanding read-out of the representation. 

**What is recoverable from data.** Two facts about identifiability shape our decoder design. _First_ , from a single variance-normalized realization from spectral prior, the spectral peak _locations_ and _bandwidths_ of _S_ ( _ω_ ) are recoverable, but the spectral _weights_ are identifiable only up to a common multiplicative constant even as _N →∞_ . The missing global scale _α_ admits an unbiased plug-in ˆ estimator _α_ = _∥_ **f** _∥_ 2<sup>2</sup><sup>_/_tr(</sup><sup>_Kpred_),soitcanberecoveredanalyticallyratherthanpredictedbythe</sup> network. _Second_ , with _M_ independent realizations from the same prior, the full set _{wq, µq, σq}_ becomes identifiable. We instantiate two decoder variants matched to these regimes (single-realization and multi-realization) with frozen PFN throughout. Detailed mathematical description and proofs are given in Appendix D. Figure 3 shows design of the proposed decoder. 

|Functions _{fm}_<sup>_M_</sup><br>_m_=1<br>or Single _f_<br>Frozen<br>PFN<br>Attention Head _H_<br>PPD<br>Encoded Value _V_<br>MQA Pooling<br>_zH_ =<br>MQA_H_(_H_)<br>_zV_ =<br>MQA_V_(_V_)<br>**PFN Stage**<br>`F`|Combine<br>MLP<br>_z_ =<br>Φ([_zH ∥zV_])<br>Ofset-BW<br>Regression<br>Bin Classifer<br>Weight<br>Regression<br>**Freq. ID &**<br>**Kernel**<br>**Constr.**<br>Analytic<br>scaling:<br>_α ⇒_<br>_K_pred=_αK_<br>`ilter Bank Decoder`|
|---|---|



Figure 3: The proposed **Filter Bank Decoder** . Both decoders pool _H_<sup>¯</sup> and _V_<sup>¯</sup> with multi-query attention (Table 4 in Appendix shows that it is necessary once spectral complexity exceeds a single component) and predict, for each frequency bin, an activation probability, a peak offset, a bandwidth, and (multi-realization only) a weight. The predicted parameters define a spectral mixture, which is converted to a stationary kernel via Bochner’s theorem (1). The PFN is never updated and the decoder is a diagnostic read-out of frozen features. Decoder details are given in Appendix E. 

**Predictive performance with no test-time optimization.** We evaluate the decoded kernel as a plugin covariance for GP regression on the standard kernel-cookbook benchmark (RBF, Periodic, product, Spectral Mixture with _Q ∈{_ 1 _,_ 4 _}_ ). Note that we evaluate not to claim state-of-the-art regression but to test whether the mechanistic structure above Sections 3-4.1 is rich enough to reconstruct a functional covariance which is one of the most demanding read-outs one could ask of any representation. **The decoder receives no information about which kernel family generated each task** : it sees only the context ( _X, y_ ), runs one forward pass through the frozen PFN, and returns an explicit _k_ ( _τ_ ). Table 3 shows that, despite this fully amortized setting, the decoded kernel matches or outperforms DKL and RFF on every family and substantially so on Periodic (1 _._ 5 _×_ 10<sup>_−_3</sup> vs. 6 _._ 6 _×_ 10<sup>_−_3</sup> ) and _K_ RP (2 _._ 7 _×_ 10<sup>_−_3</sup> vs. 4 _._ 9 _×_ 10<sup>_−_3</sup> ), while running _∼_ 250 _×_ faster, as those baselines re-optimize per task and our decoder does not. Further, the proposed decoded kernel outperforms these methods in GP-MSE, when tested on out-of-distribution kernels with varying context (Figure 16). 

Table 3: GP regression MSE on kernel cookbook with 50 context points ( _K_ RP denotes RBF _×_ Periodic). The decoder serves as a diagnostic extraction, not a general predictive replacement. 

|True kernel|Decoder (Ours)|PFN (Ours)|Amor-struct.|DKL|RFF|Oracle GP|
|---|---|---|---|---|---|---|
|RBF|1_._1_×_10<sup>_−_3</sup>|9_._7_×_10<sup>_−_4</sup>|4_._6_×_10<sup>_−_4</sup>|8_._4_×_10<sup>_−_4</sup>|1_._0_×_10<sup>_−_3</sup>|1_._7_×_10<sup>_−_4</sup>|
|Periodic|1_._5_×_10<sup>_−_3</sup>|1_._4_×_10<sup>_−_3</sup>|1_._5_×_10<sup>_−_3</sup>|6_._6_×_10<sup>_−_3</sup>|6_._4_×_10<sup>_−_3</sup>|8_._1_×_10<sup>_−_4</sup>|
|_K_RP|2_._7_×_10<sup>_−_3</sup>|1_._1_×_10<sup>_−_3</sup>|1_._6_×_10<sup>_−_3</sup>|4_._9_×_10<sup>_−_3</sup>|4_._6_×_10<sup>_−_3</sup>|9_._5_×_10<sup>_−_4</sup>|
|SM (_Q_= 1)|1_._3_×_10<sup>_−_3</sup>|4_._6_×_10<sup>_−_4</sup>|4_._3_×_10<sup>_−_4</sup>|6_._4_×_10<sup>_−_4</sup>|8_._0_×_10<sup>_−_4</sup>|2_._9_×_10<sup>_−_4</sup>|
|SM (_Q_= 4)|1_._5_×_10<sup>_−_3</sup>|6_._2_×_10<sup>_−_4</sup>|4_._4_×_10<sup>_−_4</sup>|1_._1_×_10<sup>_−_3</sup>|1_._4_×_10<sup>_−_3</sup>|1_._9_×_10<sup>_−_4</sup>|
|**Avg. Time (s)**|**0.0036**|**0.0020**|**0.0288**|**0.9180**|**0.7580**|**3.9460**|



The one amortized baseline that avoids per-task optimization but receives kernel family information, Amor-struct. Bitzer et al. [2023], does achieve lower MSE on several families, but at 8 _×_ higher latency and with an architecture explicitly designed for kernel regression rather than extracted as a diagnostic from a frozen, general-purpose predictor. Critically, the decoder is not designed to compete with the PFN’s own predictions, which retain access to the full latent; rather, it serves as a _constructive test_ of the mechanistic findings of Secs. 3.2–4: if the spectral structure we identified in _H_<sup>¯</sup> 

8 

is real and rich enough to matter, it should be possible to extract it as a functional kernel. The fact that a 9-parameter spectral-mixture object read out from frozen activations stays within a small constant factor of the PFN and beats iterative baselines that have no such structural constraint, confirms exactly this. A natural concern is that a classical FFT-based pipeline could recover _S_ ( _ω_ ) from raw ( _X, y_ ) without any PFN. We test this directly ( Table 14) by fitting spectral mixtures to the standard periodogram and Lomb–Scargle periodogram: both achieve very low kernel-MSE on the context set by overfitting the discrete samples, but their GP-MSE on unseen targets is 4–10 _×_ worse than ours across all four families. The decoded kernel does not memorize the context but it inherits the structural regularization of the PFN’s pretraining prior, which is exactly the property the mechanistic analysis identified (see Figure 22). Moreover, the decoded kernel supports downstream Bayesian tasks that the PFN itself cannot perform without continuous GPU support. We obtained competitive Bayesian optimization performance by decoded kernel on four component spectral mixture with 0 _._ 006 _±_ 0 _._ 029 are average regret while oracle GP got 0 _._ 004 _±_ 0 _._ 028. Further, decoded kernel only uses CPU (Appendix G, Tables 15). Figure 20 shows true and decoded kernel matrices pictorially. 

**Multi-realization setting.** With _M_ independent realizations from the same prior, Theorem 2 guarantees full identifiability of _S_ ( _ω_ ), and the decoder realizes this in practice: the Wasserstein distance between decoded and ground-truth densities decreases monotonically with _M_ across bandwidths (Fig. 18), with 1–4 component mixtures recovered faithfully (Fig. 19). On in-distribution spectral mixtures, decoded kernels match oracle GP MSE to within a factor of three (e.g. 1 _._ 14 _×_ 10<sup>_−_4</sup> vs. 1 _._ 45 _×_ 10<sup>_−_4</sup> on SM- _Q_ =2), and degrade gracefully on out-of-distribution families (4 _._ 8 _×_ 10<sup>_−_3</sup> on RBF, vs. catastrophic failure for the sparse-spectral assumption). The same pattern holds under additive kernels in 5D and 10D, where the decoder stays within _∼_ 3 _×_ of the oracle on every spectral mixture family (Appendix F, Tables 12, 13). See Figure **??** for pictorial depiction of decoded kernel matrix. 

### **6 Limitations and Implications** 

**Limitations.** Beyond the real-world time series validation (Section 4.3), tabular data patching (Section C.3) and 5D additive-GP experiments (Sec. 3.3, Appendix F), the probing experiments are restricted to stationary kernels representable as spectral mixtures and sinusoids, and the causal subspace analysis has not yet been extended to settings where noise, distributional shift, and active dimensionality interact simultaneously. The decoder is a diagnostic read-out, not a replacement for the PFN Müller et al. [2022a]: Table 3 shows a consistent MSE gap, indicating that some predictive information escapes the spectral-mixture parametrization. Our causal account identifies _where_ spectral information is stored and _that_ it is used, but not _how_ attention constructs it. We hypothesize that the MLP sub-layers apply a learned rectifying nonlinearity analogous to a periodogram’s square-andaverage step (Section 3.2), but verifying this via path patching remains future work. 

**Implications for PFN design.** If spectral structure is constructed in a single cross-attention step (Sec. 4.1) and concentrated in a low-dimensional causal subspace (Sec. 4.2), most of a PFN’s depth and width buys posterior formatting, not Bayesian capacity. The grid in Appendix K is consistent with this reading. Depth saturation tracks the _L_ 1 _→ L_ 2 emergence: at _d_ =128 the _L_ =2 _→ L_ =6 gain is only 2 _._ 2 _×_ , at _d_ =48 it shrinks to 1 _._ 2 _×_ and ceases to be monotone (Fig. 23a) — the regime where a narrow residual cannot absorb additional formatting capacity. Width plateaus by _d ≈_ 48–64, past which extra ambient dimensions do not enlarge the causal subspace of Sec. 4.2. The Pareto frontier (Fig. 23b) makes the consequence concrete: _d_ =64, _L_ =2, MQA reaches test MSE 7 _._ 7 _×_ 10<sup>_−_5</sup> , within 2 _._ 5 _×_ of our largest configuration ( _d_ =128, _L_ =6, Standard, 1 _._ 25M params) at 12 _×_ fewer parameters and 3 _._ 2 _×_ lower latency — the readout compresses, the construction does not. We read this as a sanity check that the mechanistic claims have design content, suggesting distillation and adapter-style tuning should target formatting layers rather than the first cross-attention step. 

### **7 Conclusion** 

This paper provided mechanistic evidence that PFNs encode spectral structure in a linearly decodable, low-dimensional, and causally active subspace of the mean-pooled latent attention score _H_<sup>¯</sup> . This structure emerges after a single attention step and generalizes across three architectures, including a frozen TabPFN probed with out-of-distribution inputs, indicating it is an emergent property of PFN-style amortization rather than because of any particular training prior, on regression tasks with 

9 

continuous covariates. The Filter Bank Decoder further demonstrates that this latent structure is rich enough to reconstruct explicit stationary kernels via Bochner’s theorem, yielding GP regression competitive with iterative baselines at lower latency. Together, these results show that PFN priors are not merely implicit: they are explicitly recoverable as portable Bayesian objects that support downstream tasks—including CPU-only Bayesian optimization, without re-invoking the network. 

### **References** 

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. _arXiv preprint arXiv:1610.01644_ , 2016. 

- Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. _Computational Linguistics_ , 48(1):207–219, 2022. 

- Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, and William Saunders. Language models can explain neurons in language models. _OpenAI Blog_ , 2023. 

- Matthias Bitzer, Mona Meister, and Christoph Zimmer. Amortized inference for Gaussian process hyperparameters of structured kernels. In Robin J. Evans and Ilya Shpitser, editors, _Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence_ , volume 216 of _Proceedings of Machine Learning Research_ , pages 184–194. PMLR, 31 Jul–04 Aug 2023. URL `https: //proceedings.mlr.press/v216/bitzer23a.html` . 

Salomon Bochner. _Lectures on Fourier Integrals_ . Princeton University Press, Princeton, NJ, 1959. 

- Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, et al. Towards monosemanticity: Decomposing language models with dictionary learning. _Transformer Circuits Thread_ , 2023. URL `https://transformer-circuits.pub/2023/monosemantic-features/index.html` . 

- Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià GarrigaAlonso. Towards automated circuit discovery for mechanistic interpretability. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL `https://openreview.net/ forum?id=89ia77nZ8u` . 

- David Duvenaud, James Robert Lloyd, Roger Grosse, Joshua B. Tenenbaum, and Zoubin Ghahramani. Structure discovery in nonparametric regression through compositional kernel search, 2013. URL `https://arxiv.org/abs/1302.4922` . 

- Batu El, Deepro Choudhury, Pietro Lio, and Chaitanya K. Joshi. Understanding information flow in graph transformers via attention graphs. In _ICLR 2025 Workshop: XAI4Science: From Understanding Model Behavior to Discovering New Scientific Knowledge_ , 2025. URL `https://openreview.net/forum?id=WpHMhLG0mj` . 

- Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. _Transformer Circuits Thread_ , 2021. URL `https://transformer-circuits.pub/ 2021/framework/index.html` . 

- Benjamin Feuer, Robin Tibor Schirrmeister, Valeriia Cherepanova, Chinmay Hegde, Frank Hutter, Micah Goldblum, Niv Cohen, and Colin White. Tunetables: context optimization for scalable priordata fitted networks. In _Proceedings of the 38th International Conference on Neural Information Processing Systems_ , Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9798331314385. 

- Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. Causal abstractions of neural networks. In _Advances in Neural Information Processing Systems_ , volume 34, 2021. 

- Atticus Geiger, Zhengxuan Wu, Christopher Potts, Thomas Icard, and Noah Goodman. Finding alignments between interpretable causal variables and distributed neural representations. In Francesco Locatello and Vanessa Didelez, editors, _Proceedings of the Third Conference on Causal Learning and Reasoning_ , volume 236 of _Proceedings of Machine Learning Research_ , pages 160– 187. PMLR, 01–03 Apr 2024. URL `https://proceedings.mlr.press/v236/geiger24a. html` . 

10 

- Léo Grinsztajn, Klemens Flöge, Oscar Key, Felix Birkel, Philipp Jund, Brendan Roof, Benjamin Jäger, Dominik Safaric, Simone Alessi, Adrian Hayler, Mihir Manium, Rosen Yu, Felix Jablonski, Shi Bin Hoo, Anurag Garg, Jake Robertson, Magnus Bühler, Vladyslav Moroshan, Lennart Purucker, Clara Cornu, Lilly Charlotte Wehrhahn, Alessandro Bonetto, Bernhard Schölkopf, Sauraj Gambhir, Noah Hollmann, and Frank Hutter. TabPFN-2.5: Advancing the state of the art in tabular foundation models, 2025. URL `https://arxiv.org/abs/2511.08667` . 

- John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 2733–2743, 2019. 

- Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. _Nature_ , 637(8045):319–326, 2025. 

- Siyang Liu and Han-Jia Ye. TabPFN unleashed: A scalable and effective solution to tabular classification problems. In _Forty-second International Conference on Machine Learning_ , 2025. URL `https://openreview.net/forum?id=5DD3RCcVcT` . 

- James Robert Lloyd, David Duvenaud, Roger Grosse, Joshua B. Tenenbaum, and Zoubin Ghahramani. Automatic construction and natural-language description of nonparametric regression models, 2014. URL `https://arxiv.org/abs/1402.4304` . 

- Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In _Advances in Neural Information Processing Systems_ , volume 35, 2022. 

- Samuel Müller, Sebastian Pineda Arango, Matthias Feurer, Josif Grabocka, and Frank Hutter. Bayesian optimization with a neural network meta-learned on synthetic data only. In _Sixth Workshop on Meta-Learning at the Conference on Neural Information Processing Systems_ , 2022a. URL `https://openreview.net/forum?id=9xCudkMSkC` . 

- Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do bayesian inference. In _International Conference on Learning Representations_ , 2022b. URL `https://openreview.net/forum?id=KSugKcbNf9` . 

- Samuel Müller et al. Position: The future of bayesian prediction is prior-fitted. In _Forty-second International Conference on Machine Learning Position Paper Track_ , 2025. 

- Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. _Distill_ , 2020. doi: 10.23915/distill.00024.001. 

- Carl Edward Rasmussen and Christopher K. I. Williams. _Gaussian Processes for Machine Learning_ . MIT Press, Cambridge, MA, 2006. 

- Kaustubh Sharma, Simardeep Singh, and Parikshit Pareek. Decoupled-value attention for prior-data fitted networks: Gp inference for physical equations, 2025. URL `https://arxiv.org/abs/ 2509.20950` . 

- Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains, 2020. URL `https://arxiv.org/abs/ 2006.10739` . 

- Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. Investigating gender bias in language models using causal mediation analysis. In _Advances in Neural Information Processing Systems_ , volume 33, 2020. 

- Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: A circuit for indirect object identification in GPT-2 small. In _International Conference on Learning Representations_ , 2023. 

- Andrew Gordon Wilson and Ryan Prescott Adams. Gaussian process kernels for pattern discovery and extrapolation. In _Proceedings of the 30th International Conference on Machine Learning (ICML)_ , pages 1067–1075, 2013. 

11 

Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P. Xing. Deep kernel learning, 2015. URL `https://arxiv.org/abs/1511.02222` . 

12 

## **Supplementary Material: Mechanistic Evidence for Spectral Structures in Prior-Data Fitted Networks** 

### **A Interpretability Experiments: Experimental Protocols** 

#### **A.1 Data Generation** 

For all experiments, we generate sinusoidal signals on _t ∈_ [ _−_ 1 _,_ 1] with 200 points. Frequencies are sampled uniformly from [0 _._ 5 _,_ 5 _._ 0] Hz with random phases _ϕ ∼U_ [0 _,_ 2 _π_ ]. For weighted signals, 

_y_ = _a ·_ sin(2 _πf_ 1 _t_ + _ϕ_ 1) + (1 _− a_ ) _·_ sin(2 _πf_ 2 _t_ + _ϕ_ 2) _._ 

#### **A.2 PFN Preprocessing** 

Following the PFN training protocol, we normalize inputs as: 



#### **A.3 Probe Architectures and Training** 

**Linear probe.** A single ridge regression ( _α_ = 1 _._ 0) with input _H_<sup>¯</sup> _∈_ R<sup>_d_</sup> and scalar target. No hidden layers, _d_ parameters. Inputs are standardized (StandardScaler) before fitting. Any _R_<sup>2</sup> this probe achieves is a lower bound on what is linearly encoded in _H_<sup>¯</sup> . 

**MLP probe.** A three-layer feed-forward network with hidden widths 256 _→_ 128 _→_ 64. Each layer uses LayerNorm, GELU activation, and dropout ( _p_ = 0 _._ 1). Optimized with AdamW (learning rate 10<sup>_−_3</sup> , weight decay 10<sup>_−_4</sup> ) using a cosine-annealing schedule (max 500 epochs, batch size 64) and early stopping on validation _R_<sup>2</sup> with patience 50. 

**Data and splits.** Each probing set consists of 2000 synthetic signals (1000 for VA-PFN), each with 200 observation points on _t ∈_ [ _−_ 1 _,_ 1]. Signals are passed through the frozen PFN and _H_<sup>¯</sup> = _N_ <u>1</u> � _i_<sup>_H_[</sup><sup>_i_] is mean-pooled over positions.Train /validation / test splits are 65% / 15% / 20%.</sup> 

### **B Additional Mechanistic Results** 



<!-- Start of picture text -->
RBF Matern<br>3.0<br>3.5<br>2.5 3.0<br>2.0 2.5<br>2.0<br>1.5<br>1.5<br>1.0<br>1.0<br>0.5<br>0.5<br>t-SNE Dimension 1 t-SNE Dimension 1<br>fchar fchar<br>t-SNE Dimension 2 t-SNE Dimension 2<br>Characteristic Frequency  Characteristic Frequency<br><!-- End of picture text -->

Figure 5: t-SNE projections of mean-pooled TabPFN embeddings _H_<sup>¯</sup> for 500 functions drawn from 5D GPs with RBF (left) and Matérn-3/2 (right) kernels, colored by characteristic frequency _f_ char = 1 _/_ (2 _πℓ_ ). Despite the absence of any sinusoidal structure in the generating process, embeddings organize smoothly by spectral scale, consistent with the quantitative metrics in Table 6. 

13 



<!-- Start of picture text -->
TabPFN VA-PFN DVA-PFN<br>10<br>4.5<br>5 2.5<br>0.0 3.0<br>0 0<br>2.5<br>1.5<br>5<br>5.0<br>10 0 10 20 20 10 0 10 5 0 5<br>t-SNE Dimension 1 t-SNE Dimension 1 t-SNE Dimension 1<br>(a) Mean-pooled attention weights H ¯ manifold.<br>TabPFN VA-PFN DVA-PFN<br>10 10 4.5<br>5<br>3.0<br>0<br>0 0<br>1.5<br>10 5<br>10<br>20 0 10 0 10 2.5 0.0 2.5 5.0<br>t-SNE Dimension 1 t-SNE Dimension 1 t-SNE Dimension 1<br>(b) Mean-pooled value encoding V ¯ manifolds.<br>Frequency (Hz)<br>t-SNE Dimension 2<br>Frequency (Hz)<br>t-SNE Dimension 2<br><!-- End of picture text -->

Figure 4: Manifold structures colored by frequency obtained via frequecy varying experiment described in Section 3. Note the tight spectral clusters in DVA-PFN compared to the smoother manifolds in VA and TabPFN due to diffusion of information inside attention mechanism. 



<!-- Start of picture text -->
DVA-PFN Manifold Correlation TabPFN Manifold Correlation<br>1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>Q L0 L1 L2 L3 L4 L5 B0 B4 B8 B12 B16 B20 B23<br>Layers Blocks<br>Correlation<br><!-- End of picture text -->

Figure 6: Layer-wise correlation _ρ_ ∆ between embedding distances and generating-frequency distances. DVA-PFN (left) peaks at L2 ( _ρ_ ∆ = 0 _._ 95) and TabPFN (right) plateaus near 0 _._ 93 by B12. Both architectures show a mild decline in the final layers, consistent with a shift toward posterior formatting. 

Table 4: Pooling ablation on _H_<sup>¯</sup> (MLP probe, _R_<sup>2</sup> ) on DVA-PFN. Mean pooling degrades sharply as spectral complexity grows, while multi-query attention pooling recovers an increasing fraction of the lost signal. _V_ -only probes yield _R_<sup>2</sup> =0 at every difficulty and are omitted. ∆ reports the gain of attention pooling over mean pooling on _H_ + _V_ . 

|Task|Mean(_H_)|Mean(_H_+_V_)|Attn(_H_)|Attn(_H_+_V_)|∆|
|---|---|---|---|---|---|
|Easy (1 param)|0.999|1.000|1.000|1.000|+0_._000|
|Medium (2 params)|0.982|0.983|0.991|0.991|+0_._008|
|Hard (4 params)|0.560|0.564|0.610|0.602|+0_._038|
|Very Hard (6 params)|0.333|0.342|0.404|0.399|+0_._057|



14 



<!-- Start of picture text -->
Frequency Probing (H) Weight Probing (H)<br>Trained Trained<br>1.0 0.998 1.000 1. 000Random 1.0 0.997 0.999 0. 998Random<br>0.8 0.8<br>0.639<br>0.6 0.6<br>0.4 0.4<br>0.2 0.181 0.2<br>0.0 0.0<br>Linear MLP Linear MLP<br>R² Score R² Score<br><!-- End of picture text -->

Figure 7: Control experiment: probing on trained vs. randomly initialized VA-PFN. Linear and MLP probes are trained to recover frequency (left) and mixing weight (right) from frozen _H_<sup>¯</sup> . On the trained network both probes succeed ( _R_<sup>2</sup> _≥_ 0 _._ 99). On the randomly initialized network, the MLP probe still recovers both targets—consistent with its capacity to learn the mapping itself Hewitt and Liang [2019]—but the linear probe fails ( _R_<sup>2</sup> = 0 _._ 18 for frequency, 0 _._ 64 for weight), confirming that linear accessibility is a consequence of learned representational structure, not of trivial input geometry. 

Table 5: Probing R<sup>2</sup> scores for spectral parameter extraction using **linear probes** on DVA-PFN. _H_ consistently dominates _V_ across all targets. 

|Target|_H_|_V_|_H_ +_V_|
|---|---|---|---|
|Single Frequency|0.98|0.21|0.99|
|Dual Frequencies|0.96|0.00|0.96|
|Full Spectral(_f_1_, f_2_, a_1_, a_2)|0.50|0.00|0.50|



Table 6: Parameter-free probing metrics for 5D GP functions on TabPFN. 



<!-- Start of picture text -->
RBF Matérn-3/2<br>ρ ∆ |r| PC0 ρ ∆ |r| PC0<br>TabPFN (5D) 0.815 0.900 0.861 0.928<br>f f a a<br>R²=0.851 R²=0.916 R²=0.067 R²=-0.108<br>5 5 0.7 0.70<br>0.65<br>4 4 0.6 0.60<br>3 0.55<br>3 0.5 0.50<br>2 2 0.4 0.45<br>0.40<br>1 1 0.3 0.35<br>0.30<br>1 2 3 4 5 1 2 3 4 5 0.3 0.4 0.5 0.6 0.7 0.3 0.4 0.5 0.6 0.7<br>True True True True<br>Predicted Predicted Predicted Predicted<br><!-- End of picture text -->

Figure 8: Probing performance on multi-component signals shows successful frequency extraction ( _f_ 1 _, f_ 2) but poor relative amplitude prediction ( _a_ 1 _, a_ 2) on TabPFN. 



<!-- Start of picture text -->
TabPFN TabPFN VA-PFN VA-PFN DVA-PFN DVA-PFN<br>Linear MLP Linear MLP Linear MLP<br>5 R²=0.993 5 R²=0.992 5 R²=0.967 5 R²=0.948 5 R²=0.998 5 R²=1.000<br>4 4 4 4 4 4<br>3 3 3 3 3 3<br>2 2 2 2 2 2<br>1 1 1 1 1 1<br>1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5<br>True Freq True Freq True Freq True Freq True Freq True Freq<br>Pred Freq<br><!-- End of picture text -->

Figure 9: Frequency Probing: Linear and MLP Relational Scatter Plots across PFN architectures. 

15 



<!-- Start of picture text -->
TabPFN TabPFN VA-PFN VA-PFN DVA-PFN DVA-PFN<br>Linear MLP Linear MLP Linear MLP<br>1.0 R²=0.982 1.0 R²=0.980 1.0 R²=0.934 1.0 R²=0.907 1.0 R²=0.997 1.0 R²=0.999<br>0.8 0.8 0.8 0.8 0.8 0.8<br>0.6 0.6 0.6 0.6 0.6 0.6<br>0.4 0.4 0.4 0.4 0.4 0.4<br>0.2 0.2 0.2 0.2 0.2 0.2<br>0.0 0.0 0.0 0.0 0.0 0.0<br>0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0<br>True Weight True Weight True Weight True Weight True Weight True Weight<br>Pred Weight<br><!-- End of picture text -->

Figure 10: Weight Probing: Linear and MLP Relational Scatter Plots across PFN architectures. 

### **C Mechanistic Analysis Results** 

#### **C.1 Probing Comparison (Frequency and Mixing Weights)** 

We quantify the accessibility of spectral information by training linear and MLP probes on frozen latent representations. Figure 9 show the scatter plots for frequency prediction across the three models. Similarly, Figure 10 and show the results for mixing weight estimation in dual-component signals. 

#### **C.2 Causal Evidence (Dose-Response Patching)** 

Functions are drawn from a zero-mean GP with RBF kernel _k_ ( _x, x_<sup>_′_</sup> ) = exp� _−_<sup><u>1</u></sup> 2<sup>(</sup><sup>_x −x′_)2</sup><sup>_/ℓ_2�</sup> on a fixed grid of _N_ =200 equally-spaced points in [ _−_ 1 _,_ 1]. Lengthscales are sampled log-uniformly: _ℓ ∼_ LogUniform(0 _._ 05 _,_ 2 _._ 0). All realisations use fixed random seeds, making the dataset fully deterministic and reproducible. Figure 11 shows representative pairs at the extremes of the _ℓ_ range used in the activation patching experiment. 



Figure 11: Representative RBF-GP signal pairs used for activation patching. Small _ℓ_ (red) produces high-frequency wiggly functions; large _ℓ_ (blue) produces smooth slowly-varying functions. The large visual separation ensures a strong baseline MSE(ˆ _yA,_ ˆ _yB_ ), making the causal effect measurement well-conditioned. 

Table 7 reports layer-wise causal effect (CE) for DVA-PFN under H-, V-, and K-patching, averaged over three signal pairs. CE is defined as 



where _y_ ˆ _A,_ ˆ _yB_ are the unpatched model predictions for contexts _yA_ and _yB_ respectively (using model predictions rather than raw GP realisations removes sample-noise from the denominator). 

Table 7: Layer-wise CE for DVA-PFN on RBF-GP pairs ( _ℓA_ = 0 _._ 05 _, ℓB_ = 2 _._ 0). H-patch reaches CE _≈_ 1 by layer 2; K-patch gives exactly 0. 

|Patch type|L1|L2|L3|L4|L5|L6|
|---|---|---|---|---|---|---|
|H (causal)|0.00|**1.00**|1.00|1.00|1.00|1.00|
|V (+control)|1.00|1.00|1.00|1.00|1.00|1.00|
|K (_−_control)|0.00|0.00|0.00|0.00|0.00|0.00|



Prior to the dose-response experiment, we verify that PCA identifies a meaningful spectral subspace. 

- **DVA-PFN** ( _d_ =128): top-5 PC correlations with _ℓ_ are [0 _._ 30 _,_ 0 _._ 28 _,_ 0 _._ 26 _,_ 0 _._ 19 _,_ 0 _._ 18]. Information is distributed across many PCs—consistent with DVA-PFN’s narrower training distribution—yet concentrated enough that the top-64 PCs account for all causal transfer (CEspec _, k_ =64 = 0 _._ 93). 

16 

- **TabPFN** ( _d_ =192): PC0 alone achieves _|r|_ =0 _._ 882 and explains 73 _._ 6% of embedding variance. The remaining PCs fall off rapidly ( _|r| ≤_ 0 _._ 21), indicating a strongly dominant single axis of structural variation. 

Table 8: Hyperparameters for non-sinusoidal patching experiments. 

|Parameter|Value|
|---|---|
|Input grid size_N_|200|
|Lengthscale range[_ℓ_min_, ℓ_max]|[0_._05_,_ 2_._0]|
|Probe set size|300 signals (log-uniform_ℓ_)|
|Patch pairs|30 (min gap_|ℓA −ℓB| ≥_0_._8)|
|Subspace dims_k_|_{_1_,_2_,_4_,_8_,_16_,_32_,_64_,_128_}_|
|DVA-PFN intervention layer|Cross-attention block 2 (of 6)|
|TabPFN intervention layer|Transformer block 24 (fnal)|
|Random seed|Fixed (all experiments deterministic)|



Table 9: Layer-wise Causal Effect (CE) for activation patching. Mean _±_ s.d. over _n_ = 50 pairs; _p_ -values from paired _t_ -tests (H vs. K). 

|**Layer**|**H-patch CE**|**V-patch CE**|**K-patch CE**|_p_**(H vs. K)**|
|---|---|---|---|---|
|L1|0_._000_±_0_._000|0_._999_±_0_._002|0_._000_±_0_._000|n.a.|
|L2|0_._999_±_0_._002|0_._999_±_0_._002|0_._000_±_0_._000|_<_10<sup>_−_133</sup>|
|L3–L6|0_._999_±_0_._002|0_._999_±_0_._002|0_._000_±_0_._000|_<_10<sup>_−_133</sup>|





<!-- Start of picture text -->
TabPFN: Spectral Subspace TabPFN: Non-Spectral Subspace<br>20<br>10 4 0.1 4<br>3 3<br>0.0<br>0<br>2 2<br>0.1<br>10 1 1<br>10 0 10 0.08 0.00 0.08<br>PC0 (|r|=0.920) PC164 (|r|=0.000)<br>Figure 12: PCA of TabPFN embeddings<br>VA-PFN: Spectral Subspace VA-PFN: Non-Spectral Subspace<br>0.04<br>6 4 4<br>3 3<br>0 0.00<br>2 2<br>6 1 0.04 1<br>15 0 15 0.08 0.00 0.08<br>PC0 (|r|=0.863) PC53 (|r|=0.000)<br>Frequency (Hz) Frequency (Hz)<br>PC1 (|r|=0.288)<br>PC150 (|r|=0.000)<br>Frequency (Hz) Frequency (Hz)<br>PC2 (|r|=0.364) PC81 (|r|=0.000)<br><!-- End of picture text -->

Figure 13: PCA of VA-PFN embeddings 

#### **C.3 Patching with Tabular Data** 

The patching experiments of Sections 4.1–4.2 use sinusoidal or GP-drawn signals that lie outside TabPFN’s pretraining distribution. To verify that _H_<sup>¯</sup> causally encodes structural information on inputs 

17 



<!-- Start of picture text -->
DVA-PFN: Spectral Subspace DVA-PFN: Non-Spectral Subspace<br>0.5<br>4 0.0004 4<br>0.0<br>3 3<br>0.0000<br>0.5<br>2 2<br>0.0004<br>1.0 1 1<br>0.5 0.0 0.5 0.0 0.5 1.0 1.5<br>PC0 (|r|=0.995) PC127 (|r|=0.000) 1e 7<br>Frequency (Hz) Frequency (Hz)<br>PC1 (|r|=0.055)<br>PC122 (|r|=0.000)<br><!-- End of picture text -->

Figure 14: PCA of DVA-PFN embeddings 

Table 10: Targeted subspace patching on DVA-PFN: causal effect as a function of subspace dimensionality _k_ <u>(out of</u> _d_ = 128). Selectivity = Spectral CE / Random CE. 

|_k_|**% of dims**|**Spectral CE**|**Non-spectral CE**|**Random CE**|**Selectivity**|
|---|---|---|---|---|---|
|1|0.78%|0_._026|_−_0_._000|0_._007|—|
|2|1.56%|**0**_._**592**|0_._008|0_._018|**33**_._**9**_×_|
|4|3.13%|**0**_._**676**|0_._017|0_._032|21_._3_×_|
|8|6.25%|0_._676|0_._023|0_._066|10_._3_×_|
|16|12.5%|0_._689|0_._115|0_._149|4_._6_×_|
|32|25.0%|**0**_._**977**|0_._135|0_._326|3_._0_×_|
|128|100%|0_._999|0_._999|0_._999|1_._0_×_|



TabPFN was designed for, we repeat the all-block patching protocol on **8-dimensional tabular data** sampled from _U_ [0 _,_ 1]<sup>8</sup> , consistent with TabPFN’s synthetic pretraining regime. We fix a shared feature matrix _X ∈_ R<sup>100</sup><sup>_×_8</sup> and construct two targets from disjoint feature subsets: _yA_ = _f_ ( _x_ 0 _, x_ 1) and _yB_ = _f_ ( _x_ 5 _, x_ 6), where _f_ ( _xi, xj_ ) = _xi_ sin(4 _xj_ ) + _x_<sup>2</sup> _i_<sup>+</sup><sup>_ε_is a shared nonlinear function.As</sup> a negative control, _yB_ is replaced with a random permutation _y_ rand that destroys feature-relevance structure while preserving marginal statistics. Figure 17 reports causal effect averaged over 30 pairs with all 24 blocks patched simultaneously. _H_<sup>¯</sup> -patching with a structurally different target achieves CE = 0 _._ 985 _±_ 0 _._ 014, confirming near-complete transfer of feature-relevance identity through _H_<sup>¯</sup> . The negative control yields CE = 0 _._ 415 _±_ 0 _._ 115 which is substantially lower, though nonzero. This residual is expected under all-block replacement: even activations from a random target form an internally consistent signal that coherently overrides the base computation, shifting predictions away from _y_ ˆ _A_ without specifically targeting _y_ ˆ _B_ . The decisive contrast where structured patching drives CE _→_ 1 while unstructured patching plateaus below 0 _._ 5, confirms that _H_<sup>¯</sup> causally encodes _which features drive the target_ , extending the spectral-identity results of Sections 4.1–4.2 to TabPFN’s native multi-feature regime. 

#### **C.4 Real-world Tabular Data Patching Details** 

**Data.** We use two publicly available monthly time series: _Airline Passengers_ (Box & Jenkins, 144 observations, 1949–1960) loaded via `statsmodels` , and _Milk Production per cow_ (USDA, 168 observations, 1962–1975). Both exhibit strong seasonality (period _≈_ 12 months) but differ in trend structure and amplitude dynamics. Each series is first-differenced to remove trend, then _z_ -normalized. ˆ A 5-lag embedding converts each to a tabular regression problem: _Xt_ = [ _yt−_ 1 _, . . . , yt−_ 5] _∈_ R<sup>5</sup> _, y_ = _yt_ , yielding 138 (Airline) and 162 (Milk) samples respectively. 

**Model.** We use `TabPFNRegressor` v2.5 on GPU with default hyperparameters. The model contains 24 transformer blocks; all interventions target block 23 (the final block). Baseline regression quality: _R_<sup>2</sup> = 0 _._ 81 (Airline) and _R_<sup>2</sup> = 0 _._ 94 (Milk) with 100 context and 30 test points. 

**Activation Patching (Exp. K2).** For each ordered pair (source, donor), we: 

1. Fit TabPFN on the source context and predict on 30 test points _→ y_ ˆ _A_ . 

2. Fit on the donor context and predict _→ y_ ˆ _B_ . 

18 



<!-- Start of picture text -->
Targeted Subspace Patching on Non-Sinusoidal Signals (RBF-GP, ℓ ∈ [0.05, 2.0])<br><!-- End of picture text -->



<!-- Start of picture text -->
DVA-PFN   RBF GP Functions TabPFN   RBF GP Functions<br>1.0 Spectral subspace 1.0 Spectral subspace<br>Non-spectral subspace Non-spectral subspace<br>Random subspace Random subspace<br>Uniform /d$ Uniform /d$<br>0.8 Full-$ CE$=0.94$ 0.8 Full-$ CE$=0.29$<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>0.0 0.0<br>124 8 16 32 64 128 124 8 16 32 64 128<br>Subspace Dimensionality $ Subspace Dimensionality $<br>Causal Effect (CE) Causal Effect (CE)<br><!-- End of picture text -->

Figure 15: **Targeted subspace patching on RBF-GP functions.** Causal Effect (CE) as a function of patching dimensionality _k_ for DVA-PFN ( _left_ ) and TabPFN ( _right_ ). **Red** : spectral subspace (PCs most correlated with _ℓ_ ). **Blue** : non-spectral subspace (bottom- _k_ PCs). **Grey** triangles: random _k_ -dimensional subspace. Dashed grey: uniform _k/d_ baseline. Dashed green: full- _H_ replacement ceiling. Shaded bands: 95% confidence intervals over 30 signal pairs. The spectral subspace dominates at every _k_ for both architectures, demonstrating that causal structural information is compactly organised beyond the sinusoidal training distribution. 



<!-- Start of picture text -->
In-Distribution Out-of-Distribution Computational Time<br>Ours 10 0 100.0<br>10 2 RFF<br>DKL 10.0<br>10 1<br>10 3 1.0<br>10 2 0.1<br>20 50 100 20 50 100 20 50 100<br>Context Length Context Length Context Length<br>MSE MSE<br>Time (s)<br><!-- End of picture text -->

Figure 16: GP-MSE versus context length on (left) in-distribution sinusoids and (right) out-ofdistribution triangle waves with random frequencies _∼U_ (1 _._ 0 _,_ 3 _._ 0). Decoded kernels match DKL/RFF on both, without per-task optimization. 

3. Cache the full output tensor _HB ∈_ R<sup>1</sup><sup>_×_194</sup><sup>_×_14</sup><sup>_×_192</sup> at block 23 during the donor forward pass. 

4. Re-fit on the source context, register a hook that replaces block 23’s output with _HB_ , and predict _→ y_ ˆpatch. 

5. CE = 1 _−_ MSE(ˆ _y_ patch _,_ ˆ _yB_ ) _/_ MSE(ˆ _yA,_ ˆ _yB_ ). 

The tensor dimensions correspond to batch (1), sequence positions (194 = context + test + padding), feature groups (14, internal to TabPFN), and model width ( _d_ =192). Shapes match across datasets because context size, test size, and input dimensionality are identical. 

Negative control: replacing _H_ with a random Gaussian tensor of the same shape. Results: 

|Direction|CEstruct|CErand|
|---|---|---|
|Airline_←_Milk|**0.979**|0.013|
|Milk_←_Airline|**0.891**|0.000|



**Subspace Patching (Exp. K3).** To identify the “spectral” subspace, we bootstrap 100 context subsets per series, extract the mean-pooled _H_<sup>¯</sup> _∈_ R<sup>192</sup> at block 23 for each, and fit PCA on the pooled 200-vector matrix. The leading PC achieves _|r|_ =0 _._ 975 with the binary series label and explains 65.6% of variance; subsequent PCs drop below _|r|_ =0 _._ 16. 

19 



<!-- Start of picture text -->
0.98<br>1.00<br>0.75<br>0.42<br>0.50<br>0.25<br>H¯-patch H¯-patch<br>(feature shift) (random y)<br><!-- End of picture text -->

Figure 17: TabPFN patching on in-distribution 8D synthetic tabular data, comparing structured feature-relevance transfer against a random target control. 

Dose-response patching operates on the _full tensor_ : for a rank- _k_ projection Π = _PP_<sup>_⊤_</sup> _∈_ R<sup>192</sup><sup>_×_192</sup> (with _P_ containing the top- _k_ or bottom- _k_ PCs), the hook computes _H_ patch = _HA − HA_ Π + _HB_ Π applied element-wise along the last dimension of the [1 _,_ 194 _,_ 14 _,_ 192] tensor. 

|_k_|Spectral|Non-spectral|Random|
|---|---|---|---|
|4|0.071|0.039|0.038|
|16|0.169|0.114|0.114|
|32|0.392|0.254|0.372|
|64|**0.798**|0.433|0.707|
|128|**0.964**|0.844|0.904|
|Full (192)||0.979||



The spectral subspace consistently outperforms the non-spectral subspace, achieving _≈_ 2 _×_ the CE at _k_ =64 and recovering 98.5% of the full- _H_ ceiling at _k_ =128. The random subspace tracks between the two, confirming that the PCA-identified directions carry disproportionate causal weight rather than the effect being purely dimensional. 

### **D Statistical Identifiability of Spectral Density** 

We characterize what is fundamentally retrievable about the spectral density _S_ ( _ω_ ) of a stationary GP from observed function data, in two regimes: single-realization and multi-realization. 

#### **D.1 Single-Realization Limit** 

**Theorem 1** (Single-function non-identifiability of spectral weights) **.** _Let f ∼GP_ (0 _, k_ ) _with continuous stationary kernel and spectral density S_ ( _ω_ ) =<sup>�</sup><sup>_Q_</sup> _q_ =1<sup>_wq N_(</sup><sup>_ω| µq, σ_</sup> _q_<sup>2)</sup><sup>_, wq>_0</sup><sup>_. Let {f_(</sup><sup>_xi_)</sup><sup>_}N_</sup> _i_ =1 _be a single realization on a fixed grid, normalized to unit empirical variance. Then {wq}_<sup>_Q_</sup> _q_ =1<sup>_is not_</sup> _identifiable from this single realization, except up to a common multiplicative constant, even in the limit N →∞._ 

_Proof._ Let **f** = ( _f_ ( _x_ 1) _, . . . , f_ ( _xN_ ))<sup>_⊤_</sup> denote the vector of observations. Since _f ∼GP_ (0 _, k_ ) is stationary and Gaussian, 



20 

By Bochner’s theorem, the kernel admits the spectral representation 



Consider any constant _c >_ 0 and define a rescaled spectral density 



with corresponding kernel 



Let _f_<sup>˜</sup> _∼GP_ (0 _, k_<sup>˜</sup> ). Then _f_<sup>˜</sup> and _f_ are related in distribution by 



Let<sup>˜</sup> **f** = ( _f_<sup>˜</sup> ( _x_ 1) _, . . . , f_<sup>˜</sup> ( _xN_ ))<sup>_⊤_</sup> . Under empirical variance normalization which normalization mirrors the preprocessing used in PFNs, making the model invariant to global rescaling of function values. 



Since<sup>˜</sup> **f** =<sup>_√_</sup> _<u>c</u>_ **f** , it follows immediately that 



Therefore, the normalized single-sample distribution induced by the kernel _k_ ( _τ_ ) is identical to that induced by _k_<sup>˜</sup> ( _τ_ ) = _ck_ ( _τ_ ). Because scaling the spectral density corresponds exactly to scaling all weights _{wq}_ by the same constant _c_ , no estimator operating on a single normalized realization can distinguish between _{wq}_ and _{cwq}_ .Consequently, the spectral weights are not identifiable from a single realization, except up to a common multiplicative constant, even as _N →∞_ . 

**Remark.** Frequency identifiability follows from classical spectral estimation theory: the discrete Fourier transform of a stationary process concentrates energy near the true frequencies, while global rescaling of the covariance affects only the magnitude, not the location, of spectral peaks. **Proposition 1** (Unbiased kernel-scale estimator) **.** _Let_ **f** _∼N_ (0 _, αK_ ) _for known PSD matrix K and_ ˆ _unknown α >_ 0 _. Then α_ = _∥_ **f** _∥_ 2<sup>2</sup><sup>_/_tr(</sup><sup>_Kpred_)</sup><sup>_satisfies_E[ˆ</sup><sup>_α_] =</sup><sup>_α._</sup> 

_Proof._ Let **f** = ( _f_ ( _x_ 1) _, . . . , f_ ( _xN_ ))<sup>_⊤_</sup> _∈_ R<sup>_N_</sup> denote the vector of function values evaluated at the fixed input locations _{xi}_<sup>_N_</sup> _i_ =1<sup>.Consider,</sup> 



where _K_ pred _∈_ R<sup>_N×N_</sup> is a fixed positive semi-definite matrix predicited using a single realization decoder and _α >_ 0 is an unknown scalar. 

We define the squared _ℓ_ 2 norm of **f** as _∥_ **f** _∥_ 2<sup>2=</sup><sup>**f**</sup><sup>_⊤_</sup><sup>**f**.All expectations below are taken with respect to</sup> the randomness of **f** induced by the Gaussian process, i.e. E[ _·_ ] = E **f** _∼N_ (0 _,αK_ pred)[ _·_ ]. 

Since **f** is zero-mean Gaussian, 



Taking the trace on both sides yields 



Therefore, for the estimator 

we obtain 



which proves that _α_ ˆ is an unbiased estimator of the kernel scale. 

21 

#### **D.2 Multi-Realization Guarantee** 

**Theorem 2** (Identifiability from multiple realizations) **.** _Let {fm}_<sup>_M_</sup> _m_ =1<sup>_be i.i.d. realizations from a_</sup> _zero-mean stationary GP with spectral density S_ ( _ω_ ) =<sup>�</sup><sup>_Q_</sup> _q_ =1<sup>_wq N_(</sup><sup>_ω| µq, σ_</sup> _q_<sup>2)</sup><sup>_, wq>_0</sup><sup>_, observed_</sup> _on a common fixed grid. As M →∞, the empirical covariance converges to k_ ( _τ_ ) _in probability, and {wq}_<sup>_Q_</sup> _q_ =1<sup>_becomes identifiable from second-order statistics._</sup> 

_Proof._ We proceed by showing that multiple independent realizations allow consistent estimation of the covariance function, which uniquely determines the spectral weights. 

For a zero-mean stationary Gaussian process, the covariance function 



fully characterizes the process. By Bochner’s theorem Rasmussen and Williams [2006], the covariance function _k_ ( _τ_ ) is in one-to-one correspondence with the spectral density _S_ ( _ω_ ). Therefore, identifying _k_ ( _τ_ ) is equivalent to identifying _S_ ( _ω_ ) =<sup>�</sup><sup>_Q_</sup> _q_ =1<sup>_wq N_(</sup><sup>_ω| µq, σ_</sup> _q_<sup>2) and its parameters</sup><sup>_{wq, µq, σq}_.</sup> We thus show that the empirical covariance converges to _k_ ( _τ_ ) as the number of realizations _M_ increases. 

Fix any pair of input locations ( _xi, xj_ ) and define the empirical covariance estimator across realizations: 



Since the realizations _{fm}_ are independent and identically distributed, each term _fm_ ( _xi_ ) _· fm_ ( _xj_ ) is an independent sample of a random variable with expectation 



where the expectation is taken with respect to the Gaussian process prior. 

Moreover, because _fm_ ( _xi_ ) _· fm_ ( _xj_ ) has finite second moment under the Gaussian process prior Rasmussen and Williams [2006], the Law of Large Numbers implies 



Since the input grid _{xi}_<sup>_N_</sup> _i_ =1<sup>isfixedandfinite,thisconvergenceholdsjointlyforallpairs(</sup><sup>_i, j_).</sup> Consequently, the entire empirical covariance matrix converges in probability to the true covariance matrix: 



Finally, the limiting covariance function _k_ ( _τ_ ) uniquely determines the spectral density _S_ ( _ω_ ) via Bochner’s theorem. In particular, for the spectral mixture form 



the parameters _{wq}_ are uniquely determined by _k_ ( _τ_ ). Therefore, as the number of independent realizations _M_ increases, the spectral weights _{wq}_ become identifiable from the empirical secondorder statistics of the observed functions. 

### **E Filter Bank Decoder: Architecture and Training** 

#### **E.1 Pipeline Overview** 

The decoder takes a context set _D_ ctx, processes it through the frozen PFN to obtain _H, V_ , and outputs explicit spectral parameters from which a stationary kernel is reconstructed via Bochner’s theorem. The PFN is never updated. 

22 

#### **E.2 Multi-Query Attention Pooling** 

For an input sequence _H ∈_ R<sup>_B×N×d_</sup> and learned queries _Q ∈_ R<sup>1</sup><sup>_×nq×d_</sup> , 



Independent MQA modules are applied to _H_ and _V_ , giving _zH_ = MQA _H_ ( _H_ ), _zV_ = MQA _V_ ( _V_ ), fused as _z_ = MLP([ _zH ∥ zV_ ]). 

#### **E.3 Spectral Parameter Heads** 

We discretize the frequency range [ _µ_ min _, µ_ max] into _B_ bins of width ∆= ( _µ_ max _− µ_ min) _/B_ . Three heads predict, per bin: (i) activation probability _pb_ , (ii) offset and bandwidth ( _δb, σb_ ) giving _µb_ = _µ_ min + ( _b_ + _δb_ )∆, and (iii) weight _wb_ (multi-realization only; the single-realization decoder uses uniform _wb_ = 1 following Theorem 1). 

#### **E.4 Kernel Reconstruction** 



ˆ with classification threshold _γ_ and analytical scale _α_ = _∥_ **f** _∥_ 2<sup>2</sup><sup>_/_tr(</sup><sup>_K_).</sup> 

#### **E.5 Loss and Curriculum** 

The decoder is trained with a composite loss 



with _w_ pos = 30 in BCE to handle sparse positive bins, and a curriculum that ramps _np_ (active components) from 1 to 4. Hyperparameters in Table 11. 

Table 11: Decoder training hyperparameters. 

|Parameter|Multi-Realization|Single-Realization|
|---|---|---|
|_n_samples|100,000|300,000|
|_n_points|200|200|
|_n_bins|50|50|
|_d_model|128|128|
|_d_ff|256|256|
|_n_queries|4|4|
|dropout|0.1|0.1|
|BCE positive weight|30.0|30.0|
|_λ_reg|5.0|5.0|
|learning rate|10<sup>_−_3</sup>|10<sup>_−_3</sup>|
|<br>weight decay|10<sup>_−_4</sup>|10<sup>_−_4</sup>|
|GP samples per task (_M_)|16|1|
|Frequency range (Hz)|[0_._5_,_3_._0]|[0_._5_,_3_._0]|
|_σ_range|[0_._01_,_0_._05]|[0_._01_,_0_._05]|
|Epochs (Phases 1 / 2 / 3)|1000 / 1000 / 2000|200 / 200 / 400|



#### **E.6 Training Data Generation** 

**Multi-Realization.** Spectral parameters: _µq ∼U_ [ _µ_ min _, µ_ max], _σq ∼U_ [ _σ_ min _, σ_ max], _wq ∼_ Gamma(2 _,_ 1). Kernel _K_ =<sup>�</sup> _q_<sup>_wq_exp(</sup><sup>_−_2</sup><sup>_π_2</sup><sup>_σ_</sup> _q_<sup>2</sup><sup>_τ_2) cos(2</sup><sup>_πµqτ_).GPsamples</sup><sup>_y_(</sup><sup>_m_)</sup><sup>_∼N_(0</sup><sup>_, K_),</sup> _m_ = 1 _, . . . , M_ . 

23 

**Single-Realization.** Random Fourier Feature (RFF) signals 



with _ωqj ∼N_ ( _µq, σq_<sup>2),</sup><sup>_ϕqj∼U_[0</sup><sup>_,_2</sup><sup>_π_],</sup><sup>_n_rff= 100.</sup> 

### **F Additional Decoder Results** 



<!-- Start of picture text -->
=0.01<br>0.5<br>=0.03<br>=0.05<br>0.4<br>0.3<br>0.2<br>2 4 6 8 10 12 14 16<br>Number of Functional Realizations M<br>Wasserstein Distance<br><!-- End of picture text -->

Figure 18: Wasserstein distance between true and decoded spectral densities as a function of the number of independent realizations _M_ , for three bandwidths _σ ∈{_ 0 _._ 01 _,_ 0 _._ 03 _,_ 0 _._ 05 _}_ . Monotone decrease is consistent with Theorem 2. 



<!-- Start of picture text -->
True True<br>1.6 1.0<br>Pred Pred<br>0.8 0.5<br>0.0 0.0<br>1 2 3 1 2 3<br>0.8 True True<br>Pred Pred<br>0.50<br>0.4<br>0.25<br>0.0 0.00<br>1 2 3 1 2 3<br><!-- End of picture text -->

Figure 19: Decoded vs. ground-truth spectral densities for 1–4 component mixtures. 

#### **F.1 High-Dimensional Data Generation** 

Functions are drawn from additive GP priors _K_ ( _x, x_<sup>_′_</sup> ) = _|D|_ <u>1</u> � _d∈D_<sup>_Kd_(</sup><sup>_xd, x_</sup> _d_<sup>_′_), where</sup><sup>_D_is a small</sup> active subset (2–4 in 5D; up to 6 in 10D under a curriculum) and each _Kd_ is RBF, Periodic, or SM. Inputs are drawn independently per dimension and sorted along each coordinate, removing combinatorial spatial variation while preserving kernel structure. 

24 



<!-- Start of picture text -->
Peaks = 1 Peaks = 2<br>1<br>0<br>-1<br>Peaks = 3 Peaks = 4<br>1<br>0<br>-1<br>True Pred True Pred<br><!-- End of picture text -->





Figure 20: Kernel reconstruction from **single function observations** . Global amplitude is ambiguous (Theorem 1); dominant periodic structure and lengthscales are recovered. 

Table 12: Decoder GP-MSE on a fixed support of 16 functions per kernel, evaluated on 20 unseen test functions from the same prior. RBF and Matérn families lie outside the sparse-spectral-mixture model class; the decoder shows graceful degradation rather than failure. 

|Kernel family|Oracle GP MSE|Decoder MSE|
|---|---|---|
|SM (_Q_=1)|1_._13_×_10<sup>_−_4</sup>|6_._15_×_10<sup>_−_4</sup>|
|SM (_Q_=2)|1_._45_×_10<sup>_−_4</sup>|1_._14_×_10<sup>_−_4</sup>|
|SM (_Q_=4)|1_._17_×_10<sup>_−_4</sup>|2_._99_×_10<sup>_−_4</sup>|
||_Out-of-distribution_||
|RBF|1_._34_×_10<sup>_−_4</sup>|4_._76_×_10<sup>_−_3</sup>|
|Matérn-1/2|9_._50_×_10<sup>_−_2</sup><br>|1_._78_×_10<sup>_−_1</sup><br>|
|Matérn-3/2|1_._14_×_10<sup>_−_1</sup><br>|2_._21_×_10<sup>_−_1</sup><br>|
|Matérn-5/2|1_._61_×_10<sup>_−_1</sup>|2_._94_×_10<sup>_−_1</sup>|



#### **Estimators.** 



where _τ_ is the standard frequency-dependent time delay. 

**Why classical methods fail at generalization.** Classical estimators achieve very low Ker-MSE on the context set — they memorize the discrete-sample artifacts, noise, and phase alignments of the specific realization. Theorem 1 predicts this: the weights are ambiguous from a single realization, and least-squares fits seize on whatever assignment best matches the observed samples. Once tested on unseen target locations, this overfit collapses, producing GP-MSE that is 4–10 _×_ worse than the PFN decoder across all four families. The decoder sidesteps this by inheriting the structural prior the PFN learned during pretraining: peak locations and bandwidths come from a frozen representation that has already integrated over many realizations, so they generalize to new target points rather than tracking the context. 

25 



<!-- Start of picture text -->
Confidence<br>1 MeanContext 10 1<br>True<br>10 2<br>10 3<br>10 4<br>0<br>0 20 40 60 80 100<br>0.0 0.5 1.0 Ncontext<br>Mse (log)<br><!-- End of picture text -->

Figure 21: **Single-realization** decoder performance. (a) Qualitative GP regression from a decoded kernel with _N_ = 20 context points. (b) GP-MSE versus context size _N_ context. The gap to the oracle remains roughly constant in _N_ , consistent with the single-realization weight ambiguity (Theorem 1). 

Table 13: Multi-realization decoder GP-MSE in 5D and 10D additive-kernel settings, _M_ = 16 functions per task. Predictions remain within a small constant factor of the oracle except on Periodic, which is hardest under additive structure. 

|Kernel (additive)|Oracle GP MSE|Decoder MSE|
|---|---|---|
|RBF (5D)|1_._30_×_10<sup>_−_4</sup>|2_._90_×_10<sup>_−_4</sup>|
|SM (_Q_=1, 5D)|1_._10_×_10<sup>_−_4</sup>|3_._00_×_10<sup>_−_4</sup>|
|SM (_Q_=2, 5D)|1_._20_×_10<sup>_−_4</sup>|2_._40_×_10<sup>_−_4</sup>|
|SM (_Q_=4, 5D)|1_._30_×_10<sup>_−_4</sup>|2_._90_×_10<sup>_−_4</sup>|
|Periodic (5D)|2_._60_×_10<sup>_−_4</sup>|9_._50_×_10<sup>_−_3</sup>|
|RBF (10D)|1_._30_×_10<sup>_−_4</sup>|3_._25_×_10<sup>_−_4</sup>|
|SM (_Q_=1, 10D)|1_._15_×_10<sup>_−_4</sup>|4_._01_×_10<sup>_−_4</sup>|
|SM (_Q_=2, 10D)|3_._19_×_10<sup>_−_4</sup>|5_._38_×_10<sup>_−_4</sup>|
|SM (_Q_=4, 10D)|2_._06_×_10<sup>_−_4</sup>|4_._56_×_10<sup>_−_4</sup>|
|Periodic (10D)|2_._71_×_10<sup>_−_4</sup>|2_._32_×_10<sup>_−_3</sup>|



### **G Downstream Use: Bayesian Optimization with the Decoded Kernel** 

The decoded kernel parameterizes a standard GP that can be evaluated and updated without reinvoking the PFN. We illustrate this with a Bayesian optimization pipeline: PFN _→_ decoder _→_ SM kernel _→_ GP posterior _→_ UCB acquisition. The PFN runs once on the initial context to extract _k_ ( _τ_ ); subsequent BO iterations use only the decoded kernel. 

**Setup.** 100 objective functions per kernel type (SM- _Q_ 1, SM- _Q_ 2, SM- _Q_ 4); _N_ 0 = 15 initial observations; 50 BO iterations; UCB acquisition. 

**What this shows.** The decoded kernel matches the oracle on simple mixtures and stays within one order of magnitude on _Q_ = 4, while the entire BO loop runs on CPU at _∼_ 5 ms per step. The PFN itself cannot be used this way: it produces predictive distributions at queried locations but does not expose a closed-form posterior, so it cannot supply the acquisition function or perform incremental posterior updates. The decoded kernel does both, validating the practical claim that the recovered Bayesian object is not just inspectable but reusable. 

### **H High-Dimensional Data Generation** 

Functions are generated from additive GP priors _K_ ( _x, x_<sup>_′_</sup> ) = _|D|_ <u>1</u> � _d∈D_<sup>_Kd_(</sup><sup>_xd, x_</sup> _d_<sup>_′_),where</sup><sup>_D_is</sup> the set of active dimensions and each _Kd_ is a 1D kernel (RBF, periodic, or spectral mixture). For each function only 2–4 dimensions are active in 5D and up to 6 in 10D, ensuring well-conditioned covariance matrices. Inputs _xd ∈_ [0 _,_ 1] are sampled uniformly per dimension and sorted before kernel evaluation, removing combinatorial spatial variation while preserving the structure each _Kd_ induces. 

26 

Table 14: Spectral recovery benchmark (50 context, 150 target points). Classical methods overfit the context set (low Ker-MSE) but fail at GP-MSE on unseen targets. The PFN-based decoder uses pretraining-induced regularization to generalize. We report Mean MSE over 100 samples. **Bold** marks best per column within each kernel family. 

|Kernel family|Method|Ker-MSE (_↓_)|Ker-CKA (_↑_)|GP-MSE (_↓_)|
|---|---|---|---|---|
||Periodogram|0.0060|**0.7316**|0.0859|
|RBF|Lomb–Scargle|**0.0031**|0.7177|0.0788|
||PFN Decoder (Ours)|0.0062|0.3648|0.0011|
||Periodogram|0.0028|**0.8768**|0.0964|
|Pidi|Lomb–Scargle|**0.0023**|0.7191|0.0468|
|eroc|PFN Decoder (Ours)|0.0031|0.5791|**0.0015**|
||Periodogram|0.0012|**0.7730**|0.0711|
|Locall Periodic|Lomb–Scargle|**0.0008**|0.7118|0.0467|
|y|PFN Decoder (Ours)|0.0012|0.6378|**0.0027**|
||Periodogram|0.0008|**0.7870**|0.1174|
|Sectral Mixture|Lomb–Scargle|**0.0005**|0.7676|0.0989|
|p|PFN Decoder (Ours)|0.0008|0.7007|**0.0017**|





<!-- Start of picture text -->
Our Decoder AHGP Periodogram Lomb-Scargle<br>True Function<br>1.5 Context<br>Predicted (Target)<br>0.0<br>1.5<br>0.6 0.0 0.6 0.6 0.0 0.6 0.6 0.0 0.6 0.6 0.0 0.6<br>x x x x<br>y<br><!-- End of picture text -->

Figure 22: GP predictive posterior on a complex spectral mixture from a singlerealization. Periodogram and Lomb–Scargle interpolate the 50 context points exactly but oscillate wildly between observations; the decoded kernel recovers the underlying structure and tracks the held-out target. 

Multiplicative kernel composition is avoided in high dimensions because it collapses covariance structure. 

The decoder architecture is unchanged across dimensions; in 10D we use a four-phase curriculum on the number of active dimensions (2 _→_ 3 _→_ 4 _→≤_ 6). PFN training in 5D and 10D uses the same architecture and hyperparameters as in 1D, with additive priors only. 

### **I High-Dimensional Scaling Details** 

Functions are generated from additive Gaussian process priors of the form 



where _D_ denotes the set of active dimensions and each _Kd_ is a one-dimensional kernel (RBF, periodic, or spectral mixture). Only a small subset of dimensions is active for any given function (between 2 and 4 in 5D), ensuring well-conditioned covariance matrices. 

### **J PFN Pre-Training Details** 

The DVA and VA PFN is trained on synthetic 1D functions drawn from a stationary GP with a random Spectral Mixture Kernel (SMK). The architecture is a Transformer-based Conditional Neural Process with 6 cross-attention blocks (4 heads, _d_ model = 128), trained using categorical cross-entropy over 100 discretized output bins. Our training pipeline is direct extention of the codes given by Müller et al. [2022b] and Sharma et al. [2025] 

27 

Table 15: Resource comparison for iterative inference on CPU. 

||PFN (per step)|Decoded Kernel + GP|
|---|---|---|
|Parameters|859,108|9|
|Memory|3.4 MB|72 B|
|Time / iteration (CPU)|24.8 ms|4.8 ms|
|GPU required|yes|no|



#### **J.1 Baseline Implementation Details** 

**Deep Kernel Learning (DKL).** Feature extractor: 3-layer MLP (1 _→_ 64 _→_ 64 _→_ 2), Tanh hidden activations, linear output layer. GP: `ScaleKernel(RBFKernel(ard_num_dims=2))` on the 2D latent. Joint MLP and GP optimization via Adam (lr= 0 _._ 01, 500 iterations, exact MLL). 

**Random Fourier Features (RFF).** Feature map: _ϕ_ ( _x_ ) = ~~�~~ 2 _/n_ cos( _xW_<sup>_⊤_</sup> _/ℓ_ + _b_ ), _n_ = 128. _W ∼N_ (0 _, I_ ) and _b ∼U_ [0 _,_ 2 _π_ ] are fixed (non-optimized) buffers. Learnable: softplus-constrained _ℓ_ , output scale, noise variance. GP: `ScaleKernel(LinearKernel())` on _ϕ_ ( _x_ ). Same optimizer, budget, and standardization as DKL. 

**GP Oracle.** Exact GP with ground-truth kernel family, `GaussianLikelihood` , and `SpectralMixtureKernel` initialized via `initialize_from_data` for SM- _Q_ tasks. Five restarts per task (Adam, lr= 0 _._ 1, 500 iterations); lowest MLL selected. 

### **K Architecture Ablation Validating Mechanistic Predictions** 

**Setup.** We sweep the architecture along three axes using DVA-PFN: width _d ∈ {_ 16 _,_ 32 _,_ 48 _,_ 64 _,_ 96 _,_ 128 _}_ , depth _L ∈{_ 2 _,_ 3 _,_ 6 _}_ , and attention variant (joint multi-head Standard vs. multi-query MQA), training each of the 36 combinations with 3 random seeds for a total of 108 runs. All runs share the training prior and pre-processing of Sec. K (hierarchical spectral mixture, sigmoid-normalized targets), the same optimizer (AdamW, lr = 10<sup>_−_4</sup> , 5-epoch warm-up, cosine schedule), the same training budget (200 epochs of 100 steps each, batch size 32, randomized context length _n_ ctx _∈_ [100 _,_ 110]), and the same frozen evaluation sets (500-function validation, 2000-function test) generated once with a fixed seed and loaded by every worker. Two architectural details are scaled with _d_ to keep cross-cell comparisons fair. The FFN hidden width is set to 4 _d_ rather than fixed at 256, which would silently inflate small models (a fixed-FFN ablation is reported below). The head count is chosen so head dimension is in [8 _,_ 16] regardless of _d_ , rather than fixed at 4, which would collapse head dimension at _d_ =16. Latency is measured in ms per batch of 32 on a single GPU after 30 warm-up forward passes. The full grid summary is reported in Tab. 16. 

**Mechanistic predictions tested.** Two findings of Secs. 3.2–4 generate concrete, falsifiable predictions about architecture. First, the abrupt _L_ 1 _→ L_ 2 emergence reported in Sec. 4.1 where the causal effect of patching _H_<sup>¯</sup> jumps from 0 to _≈_ 1 in a single cross-attention step — predicts that depth beyond two cross-attention layers should yield diminishing returns for spectrally-driven prediction, with the marginal benefit of additional layers attributable to posterior _formatting_ rather than spectral _construction_ . Second, the subspace concentration of Fig. 1, where roughly 10% of principal components account for the full causal effect, predicts that the residual-stream dimensionality required to carry the spectral computation is small in absolute terms, although learning to organize that one-dimensional signal in the right direction may demand a larger working budget than the signal itself occupies. 

**Depth saturates rapidly, consistent with** _L_ 1 _→ L_ 2 **emergence.** Holding width fixed and varying _L_ (Table 16), depth gains shrink as _d_ shrinks. At _d_ =128 (MQA), increasing _L_ from 2 to 6 roughly halves the test MSE (6 _._ 7 _×_ 10<sup>_−_5</sup> _→_ 3 _._ 1 _×_ 10<sup>_−_5</sup> , a factor of 2 _._ 2). At _d_ =64 the same 3 _×_ depth increase yields only 1 _._ 4 _×_ ; at _d_ =48 it yields 1 _._ 2 _×_ and the relationship is not monotone ( _L_ =3 outperforms both _L_ = 2 and _L_ = 6). We read this as the experimental dual of the manifold-correlation curves of Fig. 6: the first cross-attention layer produces the spectrally-organized _H_<sup>¯</sup> , and additional layers chiefly perform downstream formatting — useful but with diminishing returns, especially when the residual stream is narrow enough that the formatting capacity is already saturated. The fact that depth 

28 

helps _more_ at _d_ =128 than at _d_ =48 also supports the formatting interpretation: a wider stream gives later layers more usable capacity to refine. 

**Headline configuration and limitations.** The most efficient configuration that still approaches the full-capacity performance is _d_ =64, _L_ =2, MQA: test MSE 7 _._ 7 _×_ 10<sup>_−_5</sup> , _∼_ 105K parameters, 1 _._ 15 ms per batch of 32. Our largest configuration ( _d_ =128, _L_ =6, Standard) achieves 3 _._ 1 _×_ 10<sup>_−_5</sup> at _∼_ 1 _._ 25M parameters and 3 _._ 66 ms, so the small model trades 2 _._ 5 _×_ MSE for 12 _×_ fewer parameters and 3 _._ 2 _×_ lower latency. We note that the benchmark is one-dimensional synthetic data drawn from a fixed prior family; the model is a small DVA-PFN, not a production-scale TabPFN. The ratios reported here should not be ported uncritically to higher dimensions or to TabPFN-scale architectures. Therefore, we present these number in Table 16 and Figure 23 as evidence that the mechanistic story has design content, not as a prescription. 

Table 16: Full architecture grid (Appendix K). Six widths _×_ three depths _×_ two attention variants, three seeds each. Mean and standard deviation of test MSE across seeds; latency reported in ms per batch of 32 on a single GPU. 

|_d_|_L_|attn|params|MSE (mean)|MSE (std)|lat (ms)|
|---|---|---|---|---|---|---|
|16|2|MQA|8.6K|1_._09_×_10<sup>_−_2</sup><br>|1_._58_×_10<sup>_−_2</sup><br>|0.89|
|32|2|MQA|28.8K|6_._26_×_10<sup>_−_4</sup><br>|2_._17_×_10<sup>_−_4</sup><br>|0.80|
|48|2|MQA|61.6K|9_._57_×_10<sup>_−_5</sup><br>|6_._54_×_10<sup>_−_5</sup><br>|0.82|
|64|2|MQA|104.6K|7_._74_×_10<sup>_−_5</sup><br>|3_._83_×_10<sup>_−_5</sup><br>|1.15|
|96|2|MQA|229.1K|5_._65_×_10<sup>_−_5</sup>|2_._47_×_10<sup>_−_5</sup>|1.06|
|128|2|MQA|401.6K|6_._71_×_10<sup>_−_5</sup>|1_._37_×_10<sup>_−_5</sup>|1.20|
|16|3|MQA|11.6K|2_._45_×_10<sup>_−_3</sup><br>|1_._80_×_10<sup>_−_3</sup><br>|0.97|
|32|3|MQA|39.9K|2_._75_×_10<sup>_−_4</sup>|9_._81_×_10<sup>_−_5</sup>|1.20|
|48|3|MQA|86.4K|6_._02_×_10<sup>_−_5</sup>|1_._58_×_10<sup>_−_5</sup>|1.40|
|64|3|MQA|147.4K|6_._89_×_10<sup>_−_5</sup>|1_._09_×_10<sup>_−_5</sup>|1.57|
|96|3|MQA|324.7K|3_._78_×_10<sup>_−_5</sup>|1_._37_×_10<sup>_−_5</sup>|1.58|
|128|3|MQA|571.1K|3_._49_×_10<sup>_−_5</sup>|5_._73_×_10<sup>_−_6</sup>|1.95|
|16|6|MQA|20.6K|1_._06_×_10<sup>_−_3</sup>|6_._17_×_10<sup>_−_4</sup>|1.90|
|32|6|MQA|73.3K|2_._04_×_10<sup>_−_4</sup>|1_._44_×_10<sup>_−_4</sup>|2.44|
|48|6|MQA|160.7K|8_._20_×_10<sup>_−_5</sup>|2_._64_×_10<sup>_−_5</sup>|2.87|
|64|6|MQA|275.6K|5_._63_×_10<sup>_−_5</sup>|1_._71_×_10<sup>_−_5</sup>|3.46|
|96|6|MQA|611.5K|3_._35_×_10<sup>_−_5</sup>|7_._72_×_10<sup>_−_6</sup>|2.61|
|128|6|MQA|1079.5K|3_._08_×_10<sup>_−_5</sup>|7_._75_×_10<sup>_−_6</sup>|1.93|
|16|2|Standard|9.2K|1_._24_×_10<sup>_−_2</sup>|1_._48_×_10<sup>_−_2</sup>|0.75|
|32|2|Standard|32.1K|4_._01_×_10<sup>_−_4</sup>|1_._66_×_10<sup>_−_4</sup>|0.77|
|48|2|Standard|68.8K|1_._43_×_10<sup>_−_4</sup>|7_._92_×_10<sup>_−_5</sup>|0.97|
|64|2|Standard|119.3K|8_._50_×_10<sup>_−_5</sup><br>|3_._52_×_10<sup>_−_5</sup><br>|1.41|
|96|2|Standard|261.9K|7_._21_×_10<sup>_−_5</sup>|4_._87_×_10<sup>_−_5</sup>|1.22|
|128|2|Standard|459.7K|5_._13_×_10<sup>_−_5</sup>|3_._24_×_10<sup>_−_5</sup>|1.38|
|16|3|Standard|12.5K|3_._84_×_10<sup>_−_3</sup><br>|5_._05_×_10<sup>_−_3</sup><br>|1.12|
|32|3|Standard|44.8K|2_._05_×_10<sup>_−_4</sup><br>|9_._91_×_10<sup>_−_5</sup><br>|1.26|
|48|3|Standard|97.2K|7_._22_×_10<sup>_−_5</sup>|2_._74_×_10<sup>_−_5</sup>|1.33|
|64|3|Standard|169.4K|1_._11_×_10<sup>_−_4</sup><br>|4_._26_×_10<sup>_−_5</sup><br>|1.65|
|96|3|Standard|373.9K|5_._88_×_10<sup>_−_5</sup>|2_._70_×_10<sup>_−_5</sup>|1.74|
|128|3|Standard|658.3K|4_._23_×_10<sup>_−_5</sup>|2_._28_×_10<sup>_−_6</sup>|2.28|
|16|6|Standard|22.4K|2_._19_×_10<sup>_−_3</sup>|2_._47_×_10<sup>_−_3</sup>|2.23|
|32|6|Standard|83.1K|2_._14_×_10<sup>_−_4</sup>|7_._05_×_10<sup>_−_5</sup>|2.56|
|48|6|Standard|182.3K|1_._30_×_10<sup>_−_4</sup>|1_._82_×_10<sup>_−_5</sup>|3.12|
|64|6|Standard|319.8K|7_._44_×_10<sup>_−_5</sup><br>|1_._50_×_10<sup>_−_5</sup><br>|3.33|
|96|6|Standard|710.0K|2_._72_×_10<sup>_−_5</sup>|7_._18_×_10<sup>_−_6</sup>|3.66|
|128|6|Standard|1253.9K|3_._06_×_10<sup>_−_5</sup>|1_._06_×_10<sup>_−_5</sup>|3.66|



29 



<!-- Start of picture text -->
(a) Width×depth sweep (b) Parameter--MSE Pareto<br>10 2 Standard, L=2MQA, L=2 MQA, L=3Standard, L=6 10 2 Standard, L=2Standard, L=3 MQA, L=3MQA, L=6<br>Standard, L=3 MQA, L=6 Standard, L=6 Pareto<br>10 3 MQA, L=2<br>10 4 10 3<br>10 5<br>10 6 10 4<br>d=64, L=2,<br>10 7 MQA<br>16 32 48 64 96 128 10 4 10 5 10 6<br>dmodel Parameters<br>Test MSE Test MSE<br><!-- End of picture text -->

Figure 23: Architecture ablation supporting the mechanistic predictions of Secs. 3.2–4. Test MSE on the held-out 2000-function set across a 6 _×_ 3 _×_ 2 grid of widths _d_ , depths _L_ , and attention variants (3 seeds each). **(a)** MSE vs. _d_ model, with shaded _±_ 1 std bands across seeds. Width gains plateau by _d ≈_ 48–64, and the spread between depths shrinks as _d_ shrinks — consistent with the _L_ 1 _→ L_ 2 emergence of spectral coding (Sec. 4.1): once the first cross-attention step has constructed the spectrally-organized latent, additional layers contribute mainly posterior formatting, with diminishing returns. **(b)** Parameter–MSE Pareto frontier (dashed). The annotated configuration ( _d_ =64, _L_ =2, MQA) achieves test MSE 7 _._ 7 _×_ 10<sup>_−_5</sup> at _∼_ 105K parameters and 1 _._ 15 ms/batch — within 2 _._ 5 _×_ of the largest configuration ( _d_ =128, _L_ =6, Standard; 1 _._ 25M parameters, 3 _._ 66 ms) at 12 _×_ fewer parameters and 3 _._ 2 _×_ lower latency. 12 of 15 Pareto-optimal points use MQA. Full numerical table in Table 16. 

30 

