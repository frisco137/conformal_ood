**Transformers Can Learn Posterior Predictive Distributions In-Context** 

**Gyeonghun Kang**<sup>1</sup> **Changwoo J. Lee**<sup>1</sup><sup>_†_</sup> **Xiang Cheng**<sup>2</sup><sup>_†_</sup> 

# **Abstract** 

Prior-data fitted networks (PFNs) have recently emerged as a powerful approach for Bayesian prediction tasks, approximating the posterior predictive distribution (PPD) through in-context learning. Despite their strong empirical performance and ability to go beyond point predictions, theoretical understandings of the algorithmic capability of transformers to learn distributions in context are still lacking. Focusing on Gaussian process regression problems, we show by construction that transformers can implement a gradient descent algorithm targeting the posterior predictive mean and variance, followed by nonlinear mappings that yield binned probabilities of PPD. We study the error bounds of the approximated PPD in terms of attention depth and bin resolution. Based on these results, we further demonstrate the key role of normalization and the choice of attention depth in enabling the extrapolation abilities of transformers beyond the pretraining sample size range. We conduct simulations that corroborate our findings, providing insight into the expressivity of PFNs targeting PPDs and how architectural choices may influence generalization capabilities. 

# **1. Introduction** 

In-context learning (ICL) is the ability of a pre-trained model to adapt to new tasks at inference time by using examples provided in the input sequence without any parameter updates (Brown et al., 2020; Garg et al., 2022). This ability has been most prominently observed in transformer architectures, which have become the dominant modeling component in modern machine learning (Vaswani et al., 2017). Formally, we refer to ICL as the ability to learn a mapping from a set of input-output pairs _Dn_ = _{_ ( _xi, yi_ ) _}_<sup>_n_</sup> _i_ =1 

> _†_ Equal advising. 1Department of Statistical Science, Duke University, Durham, NC, USA<sup>2</sup> Department of Electrical and Computer Engineering, Duke University, Durham, NC, USA. Correspondence to: Gyeonghun Kang _<_ gyeonghun.kang@duke.edu _>_ . 

_Proceedings of the 43_<sup>_rd_</sup> _International Conference on Machine Learning_ , Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s). 



<!-- Start of picture text -->
PFN PPD Mean PFN PPD Lower 5% PFN PPD Upper 95%<br>38.65 38.65 38.65<br>38.60 38.60 38.60<br>38.55 38.55 38.55 240<br>38.50 38.50 38.50 220<br>38.45 38.45 38.45<br>200<br>38.40 38.40 38.40<br>121.50 121.45 121.40 121.35 121.50 121.45 121.40 121.35 121.50 121.45 121.40 121.35 180<br>GP PPD Mean GP PPD Lower 5% GP PPD Upper 95%<br>160<br>38.65 38.65 38.65<br>140<br>38.60 38.60 38.60<br>38.55 38.55 38.55 120<br>38.50 38.50 38.50<br>38.45 38.45 38.45<br>38.40 38.40 38.40<br>121.50 121.45 121.40 121.35 121.50 121.45 121.40 121.35 121.50 121.45 121.40 121.35<br>V<br><!-- End of picture text -->

_Figure 1._ Comparison of PPDs produced by a transformer-based PFN and an empirical-Bayes Gaussian process (GP) on the Sacramento home price dataset (Kuhn, 2008). The axes represent spatial coordinates (longitude and latitude), and the color scale represents price per square foot (V). The top row shows PFN posterior predictive summaries, and the bottom row shows the corresponding GP outputs. Columns correspond to the PPD mean (left), 5% quantile (center), and 95% quantile (right). Housing locations used as context are shown by blue _×_ marks in the left column. The PFN PPD tracks that of exact GP closely overall except in the regions with no nearby observations. See Section F for details. 

together with a new input _xn_ +1 to a prediction of _yn_ +1; see Dong et al. (2024) and references therein for a review. 

When the goal is probabilistic prediction of _yn_ +1, rather than point prediction, prior-data fitted networks (PFNs) (Muller¨ et al., 2022) have recently gained considerable attention as an ICL method for approximating the posterior predictive distribution (PPD) of _yn_ +1. By pre-training a transformer model on synthetic data from a joint prior predictive model of ( _x, y_ ), PFNs aim to learn a mapping from ( _Dn, xn_ +1) to the posterior predictive distribution _p_ ( _·|xn_ +1 _, Dn_ ). Compared to traditional approaches for computing PPDs such as Markov chain Monte Carlo (MCMC), PFNs offer substantial computational gains since model parameters are no longer updated after pre-training and also maintain robustness through pre-training on diverse datagenerating scenarios. Most notably, PFNs for tabular data have demonstrated strong empirical performance and excellent scalability across a wide range of benchmarks (Hollmann et al., 2023; 2025). PFNs have also been adopted in a broad range of application domains, such as RNA biophysics (Scheuer et al., 2025) and genomic prediction 

1 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

## (Ubbens et al., 2025). 

We are particularly motivated by PFN’s ability to quantify posterior predictive uncertainty. For instance, under Gaussian process regression settings, Muller et al.¨ (2022) reports that 95% credible intervals of _yn_ +1 produced by PFNs are virtually indistinguishable from the truth; see also Zhang et al. (2025), where PFN-based PPD intervals closely match nominal coverage under interpolation settings. Also, PFNs have been shown to successfully serve as flexible surrogate models for Bayesian optimization (Muller et al.¨ , 2023; Yu et al., 2026), accurately computing common acquisition functions such as expected improvement, which depend on both posterior predictive means and variances. See Figure 1 for an illustration of PFN’s ability to accurately capture the lower 5% and upper 95% PPD quantiles; details are in Section F. 

Despite its increasing popularity, the theoretical understanding of PFNs remains largely unexplored, especially regarding the role of the transformer in the pre-training stage. An increasing body of work suggests that ICL can be interpreted as implicitly performing gradient-based optimization. For a linear regression task, Akyurek et al.¨ (2023); Von Oswald et al. (2023); Bai et al. (2023) demonstrates that transformers and their recurrences are expressive enough to implement gradient descent updates in their forward pass; see also Fu et al. (2024); Vladymyrov et al. (2024). Furthermore, Ahn et al. (2023); Zhang et al. (2024); Mahankali et al. (2024) shows that the global optimum of a single layer transformer weights in the linear regression task implements a single iteration of preconditioned gradient descent. For PFNs, a natural research question concerns understanding how PFNs carry out in-context learning of PPD, and the mechanisms and conditions under which this capability arises. 

Taking a gradient-based optimization view of ICL, we are particularly interested in a principled understanding of how attention depth affects the approximation quality of PPDs. Recent state-of-the-art PFN models suggest that increasing network depth can substantially improve performance. For example, TabPFN-2.5 and TabPFN-3 (Grinsztajn et al., 2025; 2026) report markedly improved accuracy by increasing attention depth to 18 or 24 layers, a 1.5- or 2-fold increase over TabPFNv2 (Hollmann et al., 2025) with 12 layers. Another important aspect of PFN that affects the approximation quality of PPDs is the discretization resolution used for PPDs over continuous domains, a practice that is widespread in existing PFN implementations (Muller et al.¨ , 2022; Hollmann et al., 2025) but is seldom studied in the literature. 

Another interesting aspect of PFNs is their ability to generalize beyond the pre-training sample size. For instance, even though TabPFNv2 is pretrained on synthetic data only up to a sample size of 2048, it remains highly competitive 

for context size _n_ up to 10,000 (Hollmann et al., 2025). To reach even larger, recent works have adopted approaches such as fine-tuning (Feuer et al., 2024), boosting (Wang et al., 2025), and two-stage architectures (Qu et al., 2025). Adopting a gradient-based optimization view of ICL, we seek to provide a better understanding of when this generalization ability arises and how it can be further strengthened. Nagler (2023) attempts to explain this capability asymptotically, based on sensitivity to individual training samples and localization for a fixed architecture; our focus is instead on the relationship between attention depth, normalization, and generalization ability. 

**Contributions.** Focusing on GP regression problems, we establish theoretical support for PFNs’ capability to approximate the PPD and identify the mechanisms driving the generalization across different context size _n_ , referred to as _n-generalization_ , whereby models retain reliable point predictions and uncertainty quantification beyond the pretraining sample size range. 

1. We provide an explicit transformer construction that approximates the PPD in-context up to a prescribed tolerance: self-attention implements an iterative solver for the predictive mean and variance, and a shallow multilayer perceptron (MLP) maps these moments to a binned distribution. With binning and tail errors controlled, the principal remaining error arises in the attention block (Theorem 4.1), which inherits an exponential convergence rate in the number of layers _L_ (Theorem 3.1). This extends earlier ICL works focused on point estimation to distributional approximation assessment. 

2. We use our construction to characterize both the mechanisms that enable and the factors that limit _n_ - generalization in PFNs under a controlled setting. Using the spectral properties of the attention score matrix, we show that attention normalization is essential for achieving generalization (Theorem 5.1, Figure 4). We then demonstrate that extending generalization to substantially larger _n_ is intrinsically more difficult (Theorem 5.2), and explain how the increased attention depth could help mitigate this challenge (Theorem 5.3). 

3. Using our transformer pretrained in PFN fashion, we empirically validate our theory: PPD approximation error decreases with attention depth and bin resolution, in line with Theorem 4.1 (Figure 2). We further show that attention normalization is crucial for _n_ -generalization: unnormalized models fail outside the pretraining range, while normalized models remain stable (Figure 4). Finally, we confirm that generalization performance improves with both depth and pretraining range, and that solver errors grow with evaluation sample size, consistent with Theorem 5.3 (Figure 5, 10). 

2 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

# **2. Background and Problem Setup** 

**PFNs and in-context learning of PPD.** We begin by introducing the necessary background and notation for incontext learning of posterior predictive distributions (PPD) and prior-data fitted networks (PFN). Let _p_ ( _y | x, θ_ ) denote a conditional probability model of _y ∈Y ⊆_ R parametrized by _θ_ with a prior _p_ ( _θ_ ), and _x ∈_ R<sup>_d_</sup> is a covariate. Given i.i.d. data _Dn_ = _{_ ( _xi, yi_ ) _}_<sup>_n_</sup> _i_ =1<sup>, the PPD of</sup><sup>_y_at a query</sup><sup>_x_is</sup> 



where _p_ ( _θ | Dn_ ) is a posterior of _θ_ . The integral in (1) is often analytically intractable, and PPD is typically computed through Monte Carlo integration using samples from the posterior _p_ ( _θ | Dn_ ), such as those from MCMC. 

Without appealing to expression (1), prior-data fitted networks (PFNs) (Muller et al.¨ , 2022) take a radically different approach by directly approximating PPD through incontext learning. Assuming a joint prior predictive model iid ( _xi, yi_ ) _∼ p_ ( _x, y_ ) for all _i_ , PFN aims to learn a mapping ( _Dn, x_ ) _�→ qϑ_ ( _· | x, Dn_ ) that accepts a dataset _Dn_ and a query _x_ as input and outputs a probability distribution _qϑ_ that directly approximates PPD (1). The mapping, parameterized by _ϑ_ , corresponds to a forward pass of a neural net, and _ϑ_ is optimized such that 



where expectation is taken over the assumed joint prior iid predictive model ( _xi, yi_ ) _∼ p_ ( _x, y_ ). This is equivalent to minimizing the expected KL divergence between PPD and _qϑ_ ; see Muller¨ et al. (2022) for details. In practice, _ϑ_ is optimized such that the resulting mapping can approximate PPD for a broad range of context sizes _n_ . To achieve this, PFN is pre-trained on ensembles of synthetic data with varying sample sizes; see also Nagler (2023). 

**PFN outputs and population loss minimizer.** We focus on PFN based on transformer consisting of _L_ selfattention blocks and an MLP head with parameter _ϑ_ that yields a vector of finite length _C_ (“logits”), denoted as **_ℓ_** = ( _ℓ_ 1 _, ℓ_ 2 _, . . . , ℓC_ ), where its softmax sm( **_ℓ_** ) _c_ := exp( _ℓc_ ) _/_<sup>�</sup><sup>_C_</sup> _c_<sup>_′_</sup> =1<sup>exp(</sup><sup>_ℓc′_)and</sup><sup>_c_=1</sup><sup>_, . . . , C_representthe</sup> PPD output _qϑ_ . Specifically, we fix a partition Γ = _{a_ = _γ_ 1 _< γ_ 2 _< · · · < γC_ +1 = _b}_ of an interval ( _a, b_ ] _⊂Y_ , such that marginally P( _y ∈_ ( _a, b_ ]) = _ε_ for a small _ε >_ 0. Then, the logits ( _ℓ_ 1 _, . . . , ℓC_ ) induce a piecewise-constant probability density function of _y_ for _y ∈_ ( _a, b_ ]: 



where ∆ _c_ := _γc_ +1 _− γc_ . This setting coincides with a state-of-the-art practical implementation of PFNs that yields 

a piece-wise constant output distribution of _y_ (Hollmann et al., 2025). 

The network parameter _ϑ_ , whose architecture and details will be described in the following section, is pre-trained using simulated datasets from the assumed iid joint prior predictive model ( _x, y_ ) _∼ p_ ( _x, y_ ). It minimizes the (truncated) negative log likelihood (NLL) loss E _Dn∪{x}_ [E _y∼p_ ( _a,b_ ]( _·|x,Dn_ ) _{−_ log _qϑ_ ( _y | x, Dn_ ) _| x, Dn}_ ], and the population minimizer _q_<sup>_∗_</sup> is the truncated and discretized PPD (see Lemma B.7): 



Assuming _pn_ is Lipschitz smooth on ( _a, b_ ], one can show that E _Dn∪{x}_ [TV( _pn, q_<sup>_∗_</sup> )] = _O_ ( _|_ Γ _|_ ) + _ε_ , where TV is the total variation distance and _|_ Γ _|_ is the maximum bin width of Γ; see Lemma B.1 for details. 

**Transformer architecture.** We consider a transformer architecture consisting of _L_ self-attention blocks followed by an MLP head. Let _Z_<sup>(0)</sup> _∈_ R<sup>(</sup><sup>_d_+1)</sup><sup>_×_(</sup><sup>_n_+1)</sup> be an input token matrix where each column is a token: 



so the ( _n_ + 1)st token corresponds to the query point with a masked label. Denoting Attn a masked selfattention (see Section 3), we define an update for _Z_<sup>(</sup><sup>_l_+1)</sup> _∈_ R<sup>(</sup><sup>_d_+1)</sup><sup>_×_(</sup><sup>_n_+1)</sup> _, l_ = 0 _, . . . , L −_ 1 and the output logits **_ℓ_** as 





where _V_<sup>(</sup><sup>_l_)</sup> _, K_<sup>(</sup><sup>_l_)</sup> _, Q_<sup>(</sup><sup>_l_)</sup> _, S_<sup>(</sup><sup>_l_)</sup> _∈_ R<sup>(</sup><sup>_d_+1)</sup><sup>_×_(</sup><sup>_d_+1)</sup> are the perlayer parameters (attention projections and linear skip connections); _W_ 1 _∈_ R<sup>_C′×_(</sup><sup>_d_+1)</sup> , _W_ 2 _∈_ R<sup>_C×C′_</sup> , _h_ 1 _∈_ R<sup>_C′_</sup> , and _h_ 2 _∈_ R<sup>_C_</sup> are the MLP-head parameters, and act( _·_ ) is an activation function. The update (4) comprises a residual connection, a self-attention map, and an additional learnable skip term. We write _ϑ_ for the collection of all parameters. This abstraction is sufficient for our constructive proofs; additional components used in practice (e.g., token embeddings, multi-head attention, and layer normalization) can be viewed as architectural refinements that increase expressivity without altering the core mechanism. 

It has been shown that the attention blocks (4) are expressive enough to implement an iterative solver computing the predictive mean of _pn_ (Von Oswald et al., 2023; Ahn et al., 2023; Cheng et al., 2024). However, approximating an entire predictive distribution is more demanding: it requires matching not only the mean but also higher-order characteristics, such as variance. In the following sections, we address this gap via a constructive proof. 

3 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

**Gaussian process (GP) regression.** We consider the GP regression problem (Rasmussen & Williams, 2005) as a canonical example since the tractability of the PPD arising from the GP regression problem allows us to carefully examine the accuracy of the PFN output _qϑ_ . Let _ϕ_ : R<sup>_d_</sup> _→_ R be an unknown function, and _y_ 1 _, . . . , yn_ be a noisy realization of _ϕ_ based on a probabilistic model 



Also, let _Y_ := ( _y_ 1 _, . . . , yn_ )<sup>_⊤_</sup> be the training labels, _X_ := ( _x_<sup>_⊤_</sup> 1<sup>_, · · ·, x⊤_</sup> _n_<sup>)</sup><sup>_⊤_be the feature matrix, and let</sup><sup>_Z_:= (</sup><sup>_x, Dn_)</sup> denote the query-context pair. Assuming fixed error variance _σ_<sup>2</sup> , along with a mean zero GP prior on _ϕ_ with a covariance kernel _κ_ : R<sup>_d_</sup> _×_ R<sup>_d_</sup> _→_ R, the PPD _pn_ ( _y | x, Dn_ ) is also a normal distribution _N_ ( _y_ ; _µ_ ( _Z_ ) _, τ_ ( _Z_ )) where 



where _kx_ := � _κ_ ( _x_ 1 _, x_ ) _, . . . , κ_ ( _xn, x_ )� _⊤_ and _G ∈_ R _n×n_ with [ _G_ ] _i,j_ = _κ_ ( _xi, xj_ ) is a Gram matrix. Thus, both _µ_ ( _Z_ ) and the variance reduction term _kx_<sup>_⊤_(</sup><sup>_G_+</sup><sup>_σ_2</sup><sup>_In_)</sup><sup>_−_1</sup><sup>_kx_reduce</sup> to evaluating _kx_<sup>_⊤_(</sup><sup>_G_+</sup><sup>_σ_2</sup><sup>_In_)</sup><sup>_−_1</sup><sup>_v_for</sup><sup>_v∈{Y, kx}_.</sup> 

# **3. Attention Computes Moments of PPD** 

We exhibit an explicit weight configuration under which attention implements the iterative recursions for the posterior predictive mean _µ_ ( _Z_ ) and variance _τ_ ( _Z_ ). 

**KRR and Richardson iteration.** Our starting point is to view _µ_ ( _Z_ ) and the variance reduction term in _τ_ ( _Z_ ) as arising from kernel ridge regression (KRR) problems. Let _H_ be the reproducing kernel Hilbert space (RKHS) of _κ_ with norm _∥· ∥H_ . For _v ∈_ R<sup>_n_</sup> , define _u_<sup>_∗_</sup> _∈H_ as 



By the representer theorem (Kimeldorf & Wahba, 1970), _u_<sup>_∗_</sup> ( _x_ ) = _kx_<sup>_⊤_(</sup><sup>_G_+</sup><sup>_σ_2</sup><sup>_In_)</sup><sup>_−_1</sup><sup>_v_,andtherefore</sup><sup>_u∗_</sup> _X_ := ( _u_<sup>_∗_</sup> ( _x_ 1) _, . . . , u_<sup>_∗_</sup> ( _xn_ ))<sup>_⊤_</sup> is the solution of ( _G_ + _σ_<sup>2</sup> _In_ ) _u_<sup>_∗_</sup> _X_<sup>=</sup> _Gv_ . A Richardson iteration (Richardson, 1911) for this system can be written componentwise as, for _j_ = 1 _, . . . , n_ , 



initialized at _u_<sup>(0)</sup> ( _·_ ) _≡_ 0 and extended to a query _x_ by the same update, with _xj_ replaced by _x_ . 

Let _λ_ 1( _G_ ) and _λn_ ( _G_ ) denote the maximum and minimum eigenvalues of _G_ . If 0 _< η_<sup>(</sup><sup>_l_)</sup> _<_ 2 _/_ ( _λ_ 1( _G_ ) + _σ_<sup>2</sup> ) for 

all _l_ and<sup>�</sup><sup>_∞_</sup> _l_ =0<sup>_η_(</sup><sup>_l_)=</sup><sup>_∞_,thenas</sup><sup>_L→∞_,</sup><sup>_u_(</sup> _X_<sup>_l_)</sup> := ( _u_<sup>(</sup><sup>_l_)</sup> ( _x_ 1) _, · · · , u_<sup>(</sup><sup>_l_)</sup> ( _xn_ ))<sup>_⊤_</sup> _→ u_<sup>_∗_</sup> _X_<sup>, and</sup><sup>_u_(</sup><sup>_L_)(</sup><sup>_x_)</sup><sup>_→k_</sup> _x_<sup>_⊤_(</sup><sup>_G_+</sup> _σ_<sup>2</sup> _In_ )<sup>_−_1</sup> _v_ (see Theorem B.8). In particular, _u_<sup>(</sup><sup>_L_)</sup> ( _x_ ) _→ µ_ ( _Z_ ) when _v_ = _Y_ , and _u_<sup>(</sup><sup>_L_)</sup> ( _x_ ) _→ kx_<sup>_⊤_(</sup><sup>_G_+</sup><sup>_σ_2</sup><sup>_In_)</sup><sup>_−_1</sup><sup>_kx_</sup> when _v_ = _kx_ . Therefore, we can compute the moments of PPD by running the recursion (7) for two choices of the righthand side: _v_ = _Y_ and _v_ = _kx_ . 

**Unnormalized attention.** These recursions can be realized through attention layers. Let _Z_ = [ _z_ 1 _, . . . , zn, zn_ +1] _∈_ R<sup>_d′×_(</sup><sup>_n_+1)</sup> be a matrix of tokens, where _d_<sup>_′_</sup> is a token dimension. We define _unnormalized_ attention Attn _M,κ_ : R<sup>_d′×_(</sup><sup>_n_+1)</sup> _→_ R<sup>_d′×_(</sup><sup>_n_+1)</sup> as 



where _M ∈{_ 0 _,_ 1 _}_<sup>(</sup><sup>_n_+1)</sup><sup>_×_(</sup><sup>_n_+1)</sup> is a binary mask, and _hM,κ_ : R<sup>_d′×_(</sup><sup>_n_+1)</sup> _×_ R<sup>_d′×_(</sup><sup>_n_+1)</sup> _→_ R<sup>(</sup><sup>_n_+1)</sup><sup>_×_(</sup><sup>_n_+1)</sup> satisfies [ _hM,κ_ ( _KZ, QZ_ )] _ij_ = _κ_ ( _Kzi, Qzj_ ) _Mij_ for a positive semidefinite kernel _κ_ : R<sup>_d′_</sup> _×_ R<sup>_d′_</sup> _→_ R. Equivalently, the _j_ th output token is 



Examples of kernels include the _linear_ kernel _κ_ ( _x, x_<sup>_′_</sup> ) = _x_<sup>_⊤_</sup> _x_<sup>_′_</sup> and the _radial basis function (RBF)_ kernel _κ_ ( _x, x_<sup>_′_</sup> ) = exp( _−∥x − x_<sup>_′_</sup> _∥_<sup>2</sup> _/_ 2). 

**Attention computes** _µ_ ( _Z_ ) **and** _τ_ ( _Z_ ) **in-context.** The attention block in (4) can be configured to simultaneously carry out both recursions in parallel at each layer. The decay term ( _−η_<sup>(</sup><sup>_l_)</sup> _σ_<sup>2</sup> ) in (7) is implemented via the additional learnable connection, while the remaining term is realized by self-attention over the context tokens. Consequently, the transformer inherits the same convergence rate as the underlying Richardson iteration, which is exponential in _L_ . 

**Theorem 3.1** (Attention computes PPD moments) **.** _Define_ TF _ϑ,L_ ( _Z_<sup>(0)</sup> ) := _W_ [ _Z_<sup>(</sup><sup>_L_)</sup> ]: _,n_ +1 _as_ 



_where Mi,j_<sup>(0)= 1</sup><sup>_{i>n}, Mi,j_= 1</sup><sup>_{i≤n}are attention masks_</sup> _and W ∈_ R<sup>2</sup><sup>_×_(</sup><sup>_d_+4)</sup> _is a readout matrix. Then there exists ϑ, a set of all parameters (given in_ (23) _), such that_ 



_where ρ_ := 1 _− η_ ( _λn_ ( _G_ ) + _σ_<sup>2</sup> ) _∈_ (0 _,_ 1) _is the convergence factor, and η is the step size encoded in V_<sup>(</sup><sup>_l_)</sup> _and S_<sup>(</sup><sup>_l_)</sup> _._ 

4 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

Thus, TF _ϑ,L_ functions as an approximate solver with a finite iteration budget _L_ . This observation has direct implications for context size generalization, which we detail in Section 5. 

_Remark_ 3.2 (Extension to hierarchical GPs) _._ The construction in Theorem 3.1 can be potentially extended to fully Bayesian GP regression; see Section E for related experiments. Specifically, when the resulting PPD is a finite mixture of componentwise GP predictive distributions, one can enlarge the token dimension and use multi-head attention so that each head runs the Richardson-style recursion from Theorem 3.1 for one component. This would compute the componentwise predictive moments, while the mixture weights could, in principle, be represented by an additional readout module. A full treatment of this hierarchical extension is left for future work. 

# **4. From Moments to PPD** 

The MLP head in (5) can map the moments computed by TF _ϑ,L_ to a density. In particular, a shallow MLP followed by a softmax is naturally suited to density approximation via discretization. To make this concrete, consider a onedimensional exponential family density _fθ_ on _Y ⊆_ R with parameter _θ ∈_ Θ _⊆_ R<sup>_p_</sup> : 



where _ψ_ : Θ _→_ R<sup>_p′_</sup> is the natural parameter map, _T_ : _Y →_ R<sup>_p′_</sup> is the sufficient statistic map, _A_ ( _θ_ ) is the log normalizing constant, and _h_ is the base density. For example, for a Gaussian _N_ ( _µ, τ_ ), one may take _ψ_ ( _µ, τ_ ) = � _µ/τ, −_ 1 _/_ (2 _τ_ )� and _T_ ( _y_ ) = ( _y, y_<sup>2</sup> ). We further make _h_ a constant by expanding _ψ_ and _T_ to absorb the non-constant term in _h_ if any; see Section C.2 for details. 

**MLP for natural parameter conversion.** Let _K ⊂_ Θ be compact. By the universal approximation theorem (Hornik, 1991; Leshno et al., 1993), for any _δ_ MLP _>_ 0, there exists a one-hidden-layer MLP _ψ_<sup>˜</sup> : _K →_ R<sup>_p′_</sup> of the form 



with width _C_<sup>_′_</sup> and a non-polynomial activation act (e.g., ReLU), such that sup _θ∈K ∥ψ_ ( _θ_ ) _− ψ_<sup>˜</sup> ( _θ_ ) _∥_ 2 _≤ δ_ MLP. 

**Softmax discretization.** Let Γ = _{γc}_<sup>_C_</sup> _c_ =1<sup>+1be an equidis-</sup> tant partition of ( _a, b_ ] _⊂Y_ , with midpoints _ξc_ := ( _γc_ + _γc_ +1) _/_ 2. Define a readout matrix Ξ _∈_ R<sup>_C×p′_</sup> by Ξ _c,_ : = _T_ ( _ξc_ )<sup>_⊤_</sup> . Then Ξ _ψ_<sup>˜</sup> ( _θ_ ) is still an MLP output, corresponding to _W_ 2 = Ξ _W_<sup>˜</sup> 2, _W_ 1 = _W_<sup>˜</sup> 1 _W, h_ 1 = _h_<sup>˜</sup> 1, _h_ 2 = Ξ _h_<sup>˜</sup> 2 in expression (5) for GP regression, and (Ξ _ψ_<sup>˜</sup> ( _θ_ )) _c_ = _⟨ψ_<sup>˜</sup> ( _θ_ ) _, T_ ( _ξc_ ) _⟩_ becomes the inner product between the natural parameter and the sufficient statistics evaluated at _ξc_ . Moreover, softmax normalization removes the additive constant _A_ ( _θ_ ) in the logits. Thus, with a constant _h_ , the piecewise-constant 

density 



constitutes a midpoint discretization that can approximate _fθ_ arbitrarily well on ( _a, b_ ] as _C →∞_ (up to the MLP error; see Lemma B.2). 

Putting the pieces together, deeper attention depths ( _L_ ) control the convergence of the moments computation recursion, while larger _C_ controls the discretization error. The following theorem summarizes the resulting approximation guaranty (up to truncation mass outside ( _a, b_ ]); see Figure 2. **Theorem 4.1** (Transformer PPD approximation) **.** _Under the regularity conditions stated in Section C.2, there exists ϑ such that, for any context Z_<sup>(0)</sup> _, the total variation distance between the true posterior predictive pn_ ( _· | Z_<sup>(0)</sup> ) _and the Transformer output qϑ_ ( _· | Z_<sup>(0)</sup> ) _satisfies_ 



_where ρ ∈_ (0 _,_ 1) _is the convergence factor of the attention block and εtail_ ( _Z_<sup>(0)</sup> ) := P _y∼pn_ ( _·|Z_ (0))� _y ∈/_ ( _a, b_ ]� _._ 

Theorem 4.1 makes explicit how architectural choices relate to approximation accuracy. In practice, the attention stack is often the main architectural component of interest, whereas the truncation interval, _C_ , and the MLP width are typically chosen large enough (Muller¨ et al., 2022; Hollmann et al., 2025). In that regime, the attention mechanism is the primary bottleneck, a point developed in the next section. Theorem 4.1 also justifies the practice of interpreting the final MLP layer as binned probabilities in PFNs; the MLP weights are likely closely connected to the sufficient statistics of each bin needed to represent the target density. 

# **5. Generalizing Beyond Pretrain Sample Sizes** 

In this section, we provide an explanation based on our construction for what drives context size _n_ -generalization—i.e., retaining reliable point prediction and uncertainty quantification beyond the pretraining sample size range—and what ultimately limits it beyond computational cost. We cast _n_ - generalization as requiring an iterative solver that remains stable as the dimension of the underlying system varies with _n_ , and we show that attention normalization promotes generalization by preconditioning. We then show that, even with preconditioning, generalizing to substantially larger _n_ becomes intrinsically more demanding and explain how deeper attention is essential to mitigate this difficulty. 

In practice, PFN models are pretrained on varying _n_ by minimizing the loss over a _pretraining range n ∈_ [ _n_ min _, n_ max]: 



5 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 



<!-- Start of picture text -->
-0.77 -1.59 -2.88 -3.98 -4.19<br>-0.77 -1.56 -2.69 -3.41 -3.51<br>-0.77 -1.51 -2.38 -2.78 -2.82<br>-0.76 -1.38 -1.94 -2.11 -2.12<br>-0.72 -1.15 -1.41 -1.45 -1.46<br><!-- End of picture text -->





<!-- Start of picture text -->
-0.99 -2.60 -4.09 -4.14 -4.22<br>-0.99 -2.50 -3.46 -3.51 -3.48<br>-0.98 -2.31 -2.83 -2.82 -2.83<br>-0.96 -1.94 -2.15 -2.15 -2.14<br>-0.88 -1.42 -1.48 -1.48 -1.48<br><!-- End of picture text -->





<!-- Start of picture text -->
Theory<br>Learnable<br><!-- End of picture text -->



<!-- Start of picture text -->
Theory<br>Learnable<br><!-- End of picture text -->



<!-- Start of picture text -->
-0.67 -1.05 -2.00 -2.71 -2.86<br>-0.67 -0.99 -2.00 -2.69 -2.95<br>-0.67 -1.31 -1.97 -2.52 -2.68<br>-0.66 -1.24 -1.49 -2.18 -1.83<br>-0.65 -1.16 -1.47 -1.69 -1.61<br><!-- End of picture text -->





<!-- Start of picture text -->
-0.68 -1.45 -2.32 -2.53 -3.07<br>-0.68 -1.45 -2.30 -2.49 -2.92<br>-0.68 -1.43 -2.21 -2.34 -2.76<br>-0.67 -1.38 -2.00 -2.11 -2.32<br>-0.66 -1.24 -1.60 -1.65 -1.74<br><!-- End of picture text -->





<!-- Start of picture text -->
Theory<br>Learnable<br><!-- End of picture text -->



<!-- Start of picture text -->
Theory<br>Learnable<br><!-- End of picture text -->

_Figure 2._ **Verifying Theorem 4.1.** log E[TV( _p_ ( _a,b_ ] _, qϑ_ )] on the evaluation set as a function of depth _L ∈{_ 2 _,_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ and bin count _C ∈{_ 16 _,_ 32 _,_ 64 _,_ 128 _,_ 256 _}_ , for Bayesian linear regression (BLR) ( _d_ = 5, _n ∈_ [128 _,_ 512]) and GP regression with RBF kernel ( _d_ = 2, _n ∈_ [64 _,_ 128]). Left: heatmaps comparing _theory_ and _learnable_ parameterizations. Right: slices at fixed _L_ = 32 (varying _C_ ) and fixed _C_ = 256 (varying _L_ ), showing improvement with depth and bin resolution in both tasks. 



<!-- Start of picture text -->
Linear Kernel RBF Kernel<br>0.12 Simulation 0.025 Simulation<br>0.10 Ref (1/n) Ref (1/n)<br>0.020<br>0.08<br>0.015<br>0.06<br>0.010<br>0.04<br>0.02 0.005<br>200 400 600 800 1000 200 400 600 800 1000<br>Sample Size (n) Sample Size (n)<br>Admissible<br><!-- End of picture text -->

_Figure 3._ **Richardson step size varies with** _n_ **without normalization.** The upper bound of admissible step size of Richardson iteration solving (11), computed as (12), plotted against _n ∈{_ 100 _× i_ : _i ∈_ [10] _}_ for _d_ = 16 and _σ_<sup>2</sup> = 0 _._ 2. Results are averaged over 100 independent trials. 

moments. It must realize an iterative solver—with a single shared _ϑ_ —that approximately solves 



for any _n_ within a fixed iteration _L_ . If the learned parameters exactly match the data-generating parameters, the iteration converges as _L →∞_ for every _n_ . However, _L_ is finite, so convergence is not guaranteed uniformly over _n_ . Therefore, _ϑ_ becomes tuned to the spectrum of _G_ typical for _n ∈_ [ _n_ min _, n_ max]. 

For simplicity, assume a shared step size _η_<sup>(</sup><sup>_l_)</sup> = _η_ . The convergence of the recursion underlying TF _ϑ,L_ requires 



where the expectation over _n_ is taken with respect to a distribution on _{n_ min _, . . . , n_ max _}_ used in pretraining, such as discrete uniform. Models trained this way often exhibit nontrivial generalization beyond the pretraining range, although for sufficiently large _n_ , they are sometimes outperformed by non-ICL supervised methods (Hollmann et al., 2023; Muller¨ et al., 2025). This _n_ -generalization problem is pressing because the pretraining range is constrained by computational cost, yet a single pretrained network is routinely evaluated at test time on context sizes _n_ outside this range. While several works propose strategies to scale PFNs to larger samples (Feuer et al., 2024; Wang et al., 2025; Qu et al., 2025), a mechanistic understanding of which architectural components drive _n_ -generalization in PFNs—especially for both point prediction and PPD approximation—remains limited. 

**Step size decreases with** _n_ **.** Our construction renders these mechanisms explicit because the attention block TF _ϑ,L_ is designed to implement an iterative solver for the PPD 

(see Section A). Under the Gaussian design, for both linear and RBF kernels, the largest eigenvalue of _G_ grows linearly with _n_ as _λ_ 1( _G_ ) = Θ( _n_ ) with high probability (see Lemma B.3-B.4). The step size theorem then follows, as illustrated in Figure 3. 

iid **Theorem 5.1.** _With xi ∼N_ (0 _, Id/d_ ) _for fixed d, linear and RBF κ yield a Richardson step size bound of order_ 1 _/n._ 

A step size _η_ ˆ tuned to the range _n ∈_ [ _n_ min _, n_ max] must be small enough to satisfy (12) at _n_ max, yet large enough to yield fast convergence within the finite depth _L_ . Therefore, at inference, two failure modes could emerge: for _n_<sup>_′_</sup> _≪ n_ min, _η_ ˆ becomes overly conservative, and the recursion _under-converges_ within _L_ steps. For _n_<sup>_′_</sup> _≫ n_ max, _η_ ˆ violates (12), and the recursion may _diverge_ ; see Figure 4 (left). 

**Preconditioning ensures** _n_ **-robustness.** A standard remedy is _Jacobi (diagonal) preconditioning_ , which controls 

6 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 



<!-- Start of picture text -->
RBF Unnormalized RBF Normalized<br>0.5<br>1.0<br>1.5<br>2.0<br>2.5<br>50 100 150 200 50 100 150 200<br>Eval Sample Size (n) Eval Sample Size (n)<br>Log TV<br><!-- End of picture text -->

_Figure 4._ **Normalized attention enables sample size generalization (RBF,** _d_ = 8 **).** log E[TV( _p_ ( _a,b_ ] _, qϑ_ )] versus evaluation sample size _n_<sup>_′_</sup> _∈{_ 40 _,_ 50 _, . . . ,_ 200 _}_ for _learnable_ unnormalized TF _ϑ,L_ (left) and normalized TF<sup>pr</sup> _ϑ,L_<sup>(right),pretrainedon</sup> _n ∈_ [64 _,_ 128]. Normalization yields stable performance across _n_<sup>_′_</sup> . 

the spectrum of the iteration matrix uniformly over _n_ (Cutajar et al., 2016). Specifically, left-multiplying (11) by the inverse row-sum matrix _D_<sup>_−_1</sup> yields 



with the same fixed points as (11). As _D_<sup>_−_1</sup> _G_ is a row stochastic matrix, the maximum eigenvalue lies in _λ_ 1� _D_<sup>_−_1</sup> ( _G_ + _σ_<sup>2</sup> _In_ )� _∈_ [1 _,_ 1 + _σ_<sup>2</sup> ] (see Lemma B.5). Consequently, a sufficient step size condition for convergence can be chosen independent of _n_ . For the RBF kernel, a conservative range is 0 _< η <_ 2 _/_ (1 + _σ_<sup>2</sup> ). **Normalized attention implements preconditioning.** Jacobi preconditioning can be implemented with a minor modification of TF _ϑ,L_ . The Richardson iteration for (13) can be written coordinate-wise as 



where _sx_ =<sup>�</sup><sup>_n_</sup> _i_ =1<sup>_κ_(</sup><sup>_xi, x_)isthekernelaggregatefor</sup><sup>_x_.</sup> The preconditioner _sx_ is readily obtained from _normalized_ attention Attn<sup>na</sup> _M,κ_<sup>:R</sup><sup>_d′×_(</sup><sup>_n_+1)</sup><sup>_→_R</sup><sup>_d′×_(</sup><sup>_n_+1),wherethe</sup> (masked) attention weights are normalized by their sum: 



with _sj_ =<sup>�</sup><sup>_n_</sup> _i_<sup>_′_+1</sup> =1<sup>_κ_(</sup><sup>_Kzi′, Qzj_)</sup><sup>_Mi′j_.</sup> For strictly positive kernels where this normalization yields nonnegative weights (e.g., RBF), we define TF<sup>pr</sup> _θ,L_<sup>byreplacingeach</sup> head Attn _M,κ_ in (10) with Attn<sup>na</sup> _M,κ_<sup>andapplyingtoken-</sup> wise scaling 1 _/sxj_ to each token in the skip connection (the additive update implementing the _−ησ_<sup>2</sup> term), where _sxj_ is the attention normalizer for token _j_ (including the query token _x_ ). Under the same derivation as in the proof of Theorem 3.1, TF<sup>pr</sup> _ϑ,L_<sup>implements (14).This change is a</sup> lightweight local reweighting using attention output, rather 

than an additional learned module that materially departs from standard transformer components. 

**Larger** _n_ **demands more depth.** Regardless of Jacobi preconditioning, achieving a fixed accuracy with a finite-depth iterative solver becomes harder as _n_ grows. Under the optimal constant step size, the relative error of the Richardson iteration decays geometrically with the _convergence factor_ 



where cond( _·_ ) denotes the condition number (see Section A). A larger _ρ_<sup>_∗_</sup> close to 1 indicates slower error decay, requiring more iterations for a prescribed tolerance. 

The _ρ_<sup>_∗_</sup> captures the core difficulty of generalizing in _n_ . For the RBF kernel on R<sup>_d_</sup> , the eigenvalues of the population integral operator decay geometrically (Rasmussen & Williams, 2005, Section 4.3.1.). Using standard connections between the integral operator and the empirical Gram (e.g., Burt et al. (2019)), one can show that _G_ becomes increasingly ill-conditioned with _n_ (see Lemma B.4). While a ridge term _σ_<sup>2</sup> _In_ lifts small eigenvalues, it does not stop growth with _n_ (Lemma B.5); Jacobi preconditioning similarly bounds the spectrum but leaves the order unchanged. 



Thus, the convergence factor _ρ_<sup>_∗_</sup> _→_ 1 as _n_ grows, posing two key implications. First, the attention block—a solver with a single weight—must be trained on a sufficiently wide range (large _n_ max) so that the solver remains stable over a broader range of spectra. Second, because the convergence slows as _n_ increases, a deeper _L_ is essential to supply enough iteration budget to drive the solver error below a fixed tolerance. For the RBF kernel, its ill-conditioning makes the required depth grow essentially linearly in _n_ . 

**Theorem 5.3.** _Suppose the weight ϑ in_ TF<sup>pr</sup> _ϑ,L_<sup>_encodes_</sup> _the optimal constant step size for the Richardson solving_ (13) _. For RBF κ, for a context Z_<sup>(0)</sup> _of size n, to achieve_ �� TFpr _ϑ,L_<sup>(</sup><sup>_Z_(0))</sup><sup>_−_(</sup><sup>_µ_(</sup><sup>_Z_(0))</sup><sup>_, τ_(</sup><sup>_Z_(0)))</sup><sup>_⊤_��</sup> _∞_<sup>_< ϵ for ϵ >_0</sup><sup>_, it_</sup> _suffices that L_ ≳ _n_ log(1 _/ϵ_ ) _almost surely for large n. Remark_ 5.4 (Scalability of practical PFNs) _._ Theorem 5.3 gives a sufficient condition for a deliberately constrained construction with fixed geometry and a single step size. By contrast, the _learnable_ transformer used in Section 6 is more flexible, although it preserves key structural features of the construction: the _Q, K_ maps remain diagonal, and the sparsity pattern is inherited from the Richardson-style iteration. As shown in Section 6, this additional flexibility allows generalization to improve not only with depth _L_ but also with a wider pretraining sample size range, which exposes the model to a broader range of task eigenspectra. Thus, we view Theorem 5.3 as clarifying the role of depth in a simplified setting rather than claiming that its scaling law governs all practical PFNs. 

7 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 



<!-- Start of picture text -->
nmax = 128 nmax = 256 nmax = 512<br>0.4 0.4 0.4<br>L=4<br>0.6 L=8 0.6 0.6<br>L=16<br>0.8 L=32 0.8 0.8<br>1.0 1.0 1.0<br>1.2 1.2 1.2<br>1.4 1.4 1.4<br>0.925 True PPD 0.925 0.925<br>0.900 0.900 0.900<br>0.875 0.875 0.875<br>0.850 0.850 0.850<br>0.825 0.825 0.825<br>0.800 0.800 0.800<br>2.50 2.50 2.50<br>True PPD<br>2.25 2.25 2.25<br>2.00 2.00 2.00<br>1.75 1.75 1.75<br>1.50 1.50 1.50<br>200 400 600 800 1000 200 400 600 800 1000 200 400 600 800 1000<br>Eval Sample Size (n) Eval Sample Size (n) Eval Sample Size (n)<br>y<br>Log MSE<br>90% Coverage<br>90% Width<br><!-- End of picture text -->

_Figure 5._ **Depth and pretraining range improve generalization quality (RBF,** _d_ = 16 **).** Prediction MSE, 90% interval coverage, and 90% interval width versus evaluation sample size for _learnable_ normalized models with _C_ = 256, depths _L ∈{_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ , and pretraining ranges _n ∈_ [64 _, n_ max] with _n_ max _∈{_ 128 _,_ 256 _,_ 512 _}_ . Red dashed curves denote the corresponding true PPD interval width/nominal coverage. 

# **6. Numerical Studies** 

We validate our theoretical claims on two regression tasks: Bayesian linear regression (BLR; linear kernel) and nonlinear RBF regression (RBF kernel). For the unnormalized transformer TF _ϑ,L_ in Theorem 3.1, we consider two parameterization modes: _theory_ , where learnable weights are constrained as (23) and the nonzero entries of _K_<sup>(</sup><sup>_l_)</sup> = _Q_<sup>(</sup><sup>_l_)</sup> are learned per layer, and _learnable_ , where all nonzero parameters are learned independently. For the normalized transformer TF<sup>pr</sup> _ϑ,L_<sup>, we consider only the</sup><sup>_learnable_mode.</sup> 

We focus on moderate dimensions, consistent with common GP practice: as _d_ grows, substantially more samples are typically needed for hyperparameter estimation, making training more costly and less stable. In GP-based surrogate modeling, applications are therefore often limited to 10-20 dimensions unless additional structure is imposed (Frazier, 2018; Moriconi et al., 2020). 

**Pretraining.** For each pretraining instance, we draw a sample size _n ∼_ Unif[ _n_ min _, n_ max], and then sample ( _Dn, x, y_ ) from the corresponding GP model. Given a choice of hyperparameters ( _σ_<sup>2</sup> _,_ Σ _, α, ℓ_ ), we draw iniid puts _x_ 1 _, . . . , xn ∼N_ (0 _, Id/d_ ) and sample a latent function _ϕ ∼_ GP(0 _, κ_ ), where _κ_ blr( _x, x_<sup>_′_</sup> ) = _x_<sup>_⊤_</sup> Σ _x_<sup>_′_</sup> and _κ_ rbf ( _x, x_<sup>_′_</sup> ) = _α_<sup>2</sup> exp� _−_ 0 _._ 5 _∥x − x_<sup>_′_</sup> _∥_<sup>2</sup> _/ℓ_<sup>2�</sup> . The responses ind are generated as _yi ∼N_ ( _ϕ_ ( _xi_ ) _, σ_<sup>2</sup> ). For the test token, 

we independently draw _x ∼N_ (0 _, Id/d_ ). Conditioned on _Z_<sup>(0)</sup> = _Dn ∪{x}_ , we compute _µ_ ( _Z_<sup>(0)</sup> ) and _τ_ ( _Z_<sup>(0)</sup> ) and draw _y ∼N_ � _µ_ ( _Z_<sup>(0)</sup> ) _, τ_ ( _Z_<sup>(0)</sup> )� truncated to ( _a, b_ ], which is set based on Monte Carlo calibration over 2000 instances so that the probability outside the interval is 0 _._ 002. _Z_<sup>(0)</sup> is passed through the network to obtain a binned distribution _qϑ_ ( _· | Z_<sup>(0)</sup> ), and train by minimizing the (truncated) NLL. 

**Inference.** At inference time, we draw an evaluation sample size _n_<sup>_′_</sup> _∼_ Unif[ _n_<sup>_′_</sup> min<sup>_, n_</sup> max<sup>_′_]thatmayextendbeyond</sup> the pretraining range; all other settings match pretraining unless stated otherwise (e.g., covariate shift experiment in Section E). Given _Z_<sup>(0)</sup> , we let _p_ ( _a,b_ ] denote the true PPD truncated to ( _a, b_ ], and let _qϑ_ be the piecewise-constant density on ( _a, b_ ] induced by the model’s _C_ -bin predictive probabilities. Additional details on the choice of the MLP head and the hyperparameters ( _σ_<sup>2</sup> _,_ Σ _, α, ℓ_ ) can be found in Section D. Moreover, Section E contains additional simulation results with different choices of ( _σ_<sup>2</sup> _, ℓ_ ) and hierarchical GP regression with a finite discrete prior on random ( _σ_<sup>2</sup> _, ℓ_ ). 

## **6.1. Validation of Theorem 4.1** 

We verify Theorem 4.1 by training _theory_ and _learnable_ TF _ϑ,L_ over depths _L ∈{_ 2 _,_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ and bin sizes _C ∈ {_ 16 _,_ 32 _,_ 64 _,_ 128 _,_ 256 _}_ . We use _d_ = 5, ( _n_ min _, n_ max) = (128 _,_ 512) for BLR and _d_ = 2, ( _n_ min _, n_ max) = (64 _,_ 128) 

8 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

for RBF, and evaluate on the same ranges. For each ( _L, C_ ), we report the average TV distance E[TV( _p_ ( _a,b_ ] _, qϑ_ )] on the evaluation set. In Figure 2, shown on a log–log scale, log TV decreases linearly as the log of the bin size increases; it also decreases approximately according to a power law with respect to the log of the depth. 

## **6.2. Normalized vs. Unnormalized Attention** 

We compare _n_ -generalization of TF _ϑ,L_ and TF<sup>pr</sup> _ϑ,L_<sup>on</sup> the RBF task. First, for _d_ = 8 and pretraining range ( _n_ min _, n_ max) = (64 _,_ 128), we evaluate the _learnable_ unnormalized TF _ϑ,L_ and normalized TF<sup>pr</sup> _ϑ,L_<sup>with</sup><sup>_L_=32</sup> and _C_ = 256, reporting E[TV( _p_ ( _a,b_ ] _, qϑ_ )] for _n_<sup>_′_</sup> _∈ {_ 40 _,_ 50 _, . . . ,_ 200 _}_ . Figure 4 shows that the unnormalized model’s TV errors spike outside the pretraining range, while the normalized model remains stable. 

## **6.3. Validation of Theorem 5.3** 

We examine how depth _L_ and pretraining range _n_ max affect _n_ -generalization for _d_ = 16, fixing _n_ min = 64 and varying _n_ max _∈{_ 128 _,_ 256 _,_ 512 _}_ with _C_ = 256 and _L ∈{_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ , evaluated at _n_<sup>_′_</sup> _∈{_ 64 _× i_ : _i ∈_ [16] _}_ . We report metrics that align with downstream use: point prediction via E( _y − mϑ,_ 1)<sup>2</sup> and uncertainty quantification via 90% coverage P _{y ∈_ [ _lϑ, uϑ_ ] _}_ and mean width E( _uϑ − lϑ_ ) (with the true PPD interval width as a baseline). To directly probe solver accuracy, we additionally report E[TV( _p_ ( _a,b_ ] _, qϑ_ )] and moment MSEs E( _µ − mϑ,_ 1)<sup>2</sup> and E( _τ_ + _µ_<sup>2</sup> _− mϑ,_ 2)<sup>2</sup> , where _mϑ,_ 1( _Z_<sup>(0)</sup> ) := E(<sup>ˆ</sup> _y | Z_<sup>(0)</sup> _, ϑ_ ) and _mϑ,_ 2( _Z_<sup>(0)</sup> ) := E(<sup>ˆ</sup> _y_<sup>2</sup> _| Z_<sup>(0)</sup> _, ϑ_ ) are computed from the binned distribution _qϑ_ . 

Figure 5 shows that generalization improves with larger _L_ and _n_ max. Figure 10 in the Appendix shows that gains in prediction and interval metrics closely track corresponding reductions in solver convergence error (TV and moment errors) as _L_ and _n_ max increase, supporting the view that performance is primarily iteration-limited. However, consistent with Theorem 5.3, for fixed _L_ and _n_ max, the error of - the finite iteration solver increases as the evaluation sample size grows. Additional simulations (e.g., _d ∈{_ 4 _,_ 8 _}_ and covariate shift) are deferred to Section E. 

# **7. Discussion** 

This work provides a constructive account of how transformer-based PFNs can approximate PPDs in-context for GP regression, where the PPD is available in closed form. This setting allows us to make the target computation explicit, decompose the approximation error, and study how architectural choices such as depth, normalization, and output discretization affect distributional accuracy. While the results do not constitute a general theory of in-context learn- 

ing for PFNs under arbitrary priors, our results provide a concrete foundation for a broad class of GP regression models, including Bayesian linear regression, nonparametric regression, and Gaussian state-space models. 

Our contribution also provides insight into the expressive capabilities of PFNs in much broader settings. For example, in hierarchical GPs with a finite discrete prior over kernel and noise hyperparameters, the PPD becomes a finite mixture of componentwise GP predictive distributions; see Section E. This suggests a possible multi-head extension in which attention computes componentwise predictive moments in parallel while an additional readout module approximates the mixture weights. Another important class of models is latent GP models for non-Gaussian data, including GP classification and spatial/spatio-temporal models (Rue et al., 2009). Although exact PPDs are typically unavailable in closed form, approximate methods often rely on computation of latent GP posterior summaries that are closely related to GP regression predictive moments; see Chu & Ghahramani (2005, Section 4). Our decomposition suggests that attention mechanisms may be capable of implementing such iterative approximate posterior computations, while the output head maps the resulting summaries to a discretized approximation of the non-Gaussian PPD. 

A limitation of our work is that the theoretical results are primarily constructive existence results, similar to Akyurek¨ et al. (2023); Von Oswald et al. (2023), and do not imply that standard PFN pretraining necessarily discovers these mechanisms. Nevertheless, we believe that the construction is valuable from an interpretability perspective, as pretrained transformers are often difficult to interpret, and it is unclear whether the architecture can realize the PPD computation at all. We address this question by providing a simple and explicit configuration in which attention layers implement recursions for PPD moments, while a shallow MLP maps those summaries to binned predictive probabilities. The experiments partially bridge the gap between existence and learning by showing that learnable relaxations reproduce the predicted effects of depth, bin resolution, and normalization. These relaxations also motivate flexible but still interpretable model classes that help explain practical performance. Related work similarly interprets such relaxations as preconditioned gradient descent (Ahn et al., 2023) or anisotropic denoising (Rosu et al., 2026). That said, obtaining optimization guarantees for PFN pretraining and determining when trained transformers actually implement solver-like mechanisms remain important open problems. 

# **Software** 

The code is available at https://github.com/ hun-learning94/transformer-uq. 

9 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

# **Acknowledgements** 

The authors thank the anonymous reviewers for their insightful comments and constructive feedback, which greatly improved the manuscript. The research of Changwoo Lee was partially supported by NIH R01ES035625, NSF IIS2426762, and ONR N00014-24-1-2626. 

# **Impact Statement** 

This paper aims to advance the field of Machine Learning. While various societal impacts of our work are possible, none require specific emphasis here. 

# **References** 

- Ahn, K., Cheng, X., Daneshmand, H., and Sra, S. Transformers learn to implement preconditioned gradient descent for in-context learning. _Advances in Neural Information Processing Systems_ , 36:45614–45650, 2023. 

- Akyurek,¨ E., Schuurmans, D., Andreas, J., Ma, T., and Zhou, D. What learning algorithm is in-context learning? Investigations with linear models. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- Bai, Y., Chen, F., Wang, H., Xiong, C., and Mei, S. Transformers as statisticians: Provable in-context learning with in-context algorithm selection. _Advances in Neural Information Processing Systems_ , 36:57125–57211, 2023. 

- Boucheron, S., Lugosi, G., and Massart, P. _Concentration Inequalities: A Nonasymptotic Theory of Independence_ . Oxford University Press, 02 2013. ISBN 9780199535255. 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. _Advances in Neural Information Processing Systems_ , 33:1877–1901, 2020. 

- Burt, D., Rasmussen, C. E., and Van Der Wilk, M. Rates of convergence for sparse variational Gaussian process regression. In Chaudhuri, K. and Salakhutdinov, R. (eds.), _Proceedings of the 36th International Conference on Machine Learning_ , volume 97 of _Proceedings of Machine Learning Research_ , pp. 862–871. PMLR, 09–15 Jun 2019. 

- Cheng, X., Chen, Y., and Sra, S. Transformers implement functional gradient descent to learn non-linear functions in context. In Salakhutdinov, R., Kolter, Z., Heller, K., 

- Weller, A., Oliver, N., Scarlett, J., and Berkenkamp, F. (eds.), _Proceedings of the 41st International Conference on Machine Learning_ , volume 235 of _Proceedings of Machine Learning Research_ , pp. 8002–8037. PMLR, 21– 27 Jul 2024. 

- Chu, W. and Ghahramani, Z. Gaussian processes for ordinal regression. _Journal of Machine Learning Research_ , 6 (35):1019–1041, 2005. 

- Cutajar, K., Osborne, M., Cunningham, J., and Filippone, M. Preconditioning kernel matrices. In Balcan, M. F. and Weinberger, K. Q. (eds.), _Proceedings of The 33rd International Conference on Machine Learning_ , volume 48 of _Proceedings of Machine Learning Research_ , pp. 2529– 2538, New York, New York, USA, 20–22 Jun 2016. PMLR. 

- Davidson, K. R. and Szarek, S. J. Local operator theory, random matrices and Banach spaces. In _Handbook of the geometry of Banach spaces_ , volume 1, pp. 317–366. Elsevier, 2001. 

- Dong, Q., Li, L., Dai, D., Zheng, C., Ma, J., Li, R., Xia, H., Xu, J., Wu, Z., Chang, B., Sun, X., Li, L., and Sui, Z. A survey on in-context learning. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , pp. 1107–1128, Miami, Florida, USA, November 2024. Association for Computational Linguistics. 

- Feuer, B., Schirrmeister, R. T., Cherepanova, V., Hegde, C., Hutter, F., Goldblum, M., Cohen, N., and White, C. Tunetables: Context optimization for scalable priordata fitted networks. _Advances in Neural Information Processing Systems_ , 37:83430–83464, 2024. 

- Frazier, P. I. Bayesian optimization. In _Recent advances in optimization and modeling of contemporary problems_ , pp. 255–278. Informs, 2018. 

- Fu, D., Chen, T.-Q., Jia, R., and Sharan, V. Transformers learn to achieve second-order convergence rates for incontext linear regression. _Advances in Neural Information Processing Systems_ , 37:98675–98716, 2024. 

- Garg, S., Tsipras, D., Liang, P. S., and Valiant, G. What can transformers learn in-context? A case study of simple function classes. _Advances in Neural Information Processing Systems_ , 35:30583–30598, 2022. 

- Gneiting, T. and Raftery, A. E. Strictly proper scoring rules, prediction, and estimation. _Journal of the American Statistical Association_ , 102(477):359–378, 2007. 

10 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

- Grinsztajn, L., Floge, K., Key, O., Birkel, F., Jund, P., Roof,¨ B., Jager, B., Safaric, D., Alessi, S., Hayler, A., Manium,¨ M., Yu, R., Jablonski, F., Hoo, S. B., Garg, A., Robertson, J., Buhler,¨ M., Moroshan, V., Purucker, L., Cornu, C., Wehrhahn, L. C., Bonetto, A., Scholkopf, B., Gambhir,¨ S., Hollmann, N., and Hutter, F. Tabpfn-2.5: Advancing the state of the art in tabular foundation models. _arXiv preprint arXiv:2511.08667_ , 2025. 

- Grinsztajn, L., Floge, K., Key, O., Birkel, F., Jund, P., Roof,¨ B., Manium, M., Bin, S., Hoo, Buhler,¨ M., Garg, A., Safaric, D., Robertson, J., Jager, B., Alessi, S., Hayler, A.,¨ Moroshan, V., Purucker, L., Singer, P., Arazi, A., Siems, J., Metzen, J. H., Grab, G., Erickson, N., Guo, S., Kalfon, E., Bing, S., Salinas, D., Cornu, C., Wehrhahn, L. C., Kriuchkova, D., Kaya, K., Sidhoum, L., Salmon, M., Chen, J., Hulsebos, M., LeCun, Y., Muller, S., Sch¨ olkopf,¨ B., Gambhir, S., Hollmann, N., and Hutter, F. Tabpfn-3: Technical report. _arXiv preprint arXiv:2605.13986_ , 2026. 

- Hollmann, N., Muller, S., Eggensperger, K., and Hutter, F.¨ TabPFN: A transformer that solves small tabular classification problems in a second. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- Hollmann, N., Muller, S., Purucker, L., Krishnakumar, A.,¨ Korfer, M., Hoo, S. B., Schirrmeister, R. T., and Hutter,¨ F. Accurate predictions on small data with a tabular foundation model. _Nature_ , 637(8045):319–326, 2025. 

- Horn, R. A. and Johnson, C. R. _Matrix Analysis_ . Cambridge university press, 2012. 

- Hornik, K. Approximation capabilities of multilayer feedforward networks. _Neural Networks_ , 4(2):251–257, 1991. ISSN 0893-6080. 

- Isaaks, E. and Srivastava, R. _Applied Geostatistics_ . Oxford University Press, 1989. ISBN 9780195050134. 

- Kimeldorf, G. S. and Wahba, G. A correspondence between Bayesian estimation on stochastic processes and smoothing by splines. _The Annals of Mathematical Statistics_ , 41 (2):495–502, 1970. 

- Kuhn, M. Building predictive models in R using the caret package. _Journal of Statistical Software_ , 28(5):1–26, 2008. 

- Leshno, M., Lin, V. Y., Pinkus, A., and Schocken, S. Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. _Neural networks_ , 6(6):861–867, 1993. 

- Mahankali, A. V., Hashimoto, T., and Ma, T. One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Moriconi, R., Deisenroth, M. P., and Sesh Kumar, K. High-dimensional Bayesian optimization using lowdimensional feature spaces. _Machine Learning_ , 109(9): 1925–1943, 2020. 

- Muller, S., Hollmann, N., Arango, S. P., Grabocka, J., and¨ Hutter, F. Transformers can do Bayesian inference. In _International Conference on Learning Representations_ , 2022. 

- Muller,¨ S., Feurer, M., Hollmann, N., and Hutter, F. PFNs4BO: In-context learning for Bayesian optimization. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 25444–25470. PMLR, 23–29 Jul 2023. 

- Muller,¨ S., Reuter, A., Hollmann, N., Rugamer,¨ D., and Hutter, F. Position: The future of Bayesian prediction is prior-fitted. In Singh, A., Fazel, M., Hsu, D., LacosteJulien, S., Berkenkamp, F., Maharaj, T., Wagstaff, K., and Zhu, J. (eds.), _Proceedings of the 42nd International Conference on Machine Learning_ , volume 267 of _Proceedings of Machine Learning Research_ , pp. 81861–81875. PMLR, 13–19 Jul 2025. 

- Nagler, T. Statistical foundations of prior-data fitted networks. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 25660–25676. PMLR, 23–29 Jul 2023. 

- Pebesma, E., Graeler, B., and Pebesma, M. E. Package ‘gstat’. _Comprehensive R Archive Network (CRAN)_ , 2015. 

- Qu, J., Holzmuller,¨ D., Varoquaux, G., and Le Morvan, M. TabICL: A tabular foundation model for in-context learning on large data. In Singh, A., Fazel, M., Hsu, D., Lacoste-Julien, S., Berkenkamp, F., Maharaj, T., Wagstaff, K., and Zhu, J. (eds.), _Proceedings of the 42nd International Conference on Machine Learning_ , volume 267 of _Proceedings of Machine Learning Research_ , pp. 50817–50847. PMLR, 13–19 Jul 2025. 

- Rasmussen, C. E. and Williams, C. K. I. _Gaussian Processes for Machine Learning_ . The MIT Press, 11 2005. ISBN 9780262256834. 

- Richardson, L. F. IX. The approximate arithmetical solution by finite differences of physical problems involving differential equations, with an application to the stresses in a masonry dam. _Philosophical Transactions of the Royal Society of London, Series A: Containing Papers of a Mathematical or Physical Character_ , 210(459-470): 307–357, 01 1911. ISSN 0264-3952. 

11 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

- Rosu, P., Carin, L., and Cheng, X. From softmax to score: Transformers can effectively implement in-context denoising steps. _Advances in Neural Information Processing Systems_ , 38:142994–143017, 2026. 

- Rue, H., Martino, S., and Chopin, N. Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations. _Journal of the Royal Statistical Society Series B: Statistical Methodology_ , 71 (2):319–392, 2009. 

   - Zhang, R., Frei, S., and Bartlett, P. L. Trained transformers learn linear models in-context. _Journal of Machine Learning Research_ , 25(49):1–55, 2024. 

   - Zhu, H., Williams, C. K., Rohwer, R., and Morciniec, M. Gaussian regression and optimal finite dimensional linear models. _Aston University Technical Report_ , 1997. 

- Scheuer, D., Runge, F., Franke, J. K., Wolfinger, M. T., Flamm, C., and Hutter, F. KinPFN: Bayesian approximation of rna folding kinetics using prior-data fitted networks. _The Thirteenth International Conference on Learning Representations_ , 2025. 

- Ubbens, J., Stavness, I., and Sharpe, A. G. GPFN: Prior-data fitted networks for genomic prediction. _IEEE Transactions on Computational Biology and Bioinformatics_ , 22 (6):2642–2649, 2025. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. _Advances in Neural Information Processing Systems_ , 30, 2017. 

- Vladymyrov, M., Von Oswald, J., Sandler, M., and Ge, R. Linear transformers are versatile in-context learners. _Advances in Neural Information Processing Systems_ , 37: 48784–48809, 2024. 

- Von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., and Vladymyrov, M. Transformers learn in-context by gradient descent. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 35151–35174. PMLR, 23–29 Jul 2023. 

- Wang, Y., Jiang, B., Guo, Y., Gan, Q., Wipf, D., Huang, X., and Qiu, X. Prior-fitted networks scale to larger datasets when treated as weak learners. In Li, Y., Mandt, S., Agrawal, S., and Khan, E. (eds.), _Proceedings of The 28th International Conference on Artificial Intelligence and Statistics_ , volume 258 of _Proceedings of Machine Learning Research_ , pp. 1090–1098. PMLR, 03–05 May 2025. 

- Yu, R. T.-Y., Picard, C., and Ahmed, F. GIT-BO: Highdimensional Bayesian optimization with tabular foundation models. In _The Fourteenth International Conference on Learning Representations_ , 2026. 

- Zhang, Q., Tan, Y. S., Tian, Q., and Li, P. TabPFN: One model to rule them all? _arXiv preprint arXiv:2505.20003_ , 2025. 

12 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

# **A. Richardson Iteration** 

Consider a linear equation _Au_<sup>_∗_</sup> = _b_ where _A ∈_ R<sup>_n×n_</sup> is positive definite and _b ∈_ R<sup>_n_</sup> . Richardson iteration of solving the equation is given by 



whose fixed point is _u_<sup>_∗_</sup> . We call _In − ηA_ an iteration matrix and _η >_ 0 a step size. 

**Admissible Step Size for Convergence.** By subtracting _u_<sup>_∗_</sup> in both sides, we see that _e_<sup>(</sup><sup>_l_)</sup> := _u_<sup>(</sup><sup>_l_)</sup> _− u_<sup>_∗_</sup> evolves as 



Since _A ≻_ 0, we can write the iteration matrix as _In − ηA_ = _Q_ ( _In − η_ Λ) _Q_<sup>_⊤_</sup> with the spectral decomposition of _A_ . It follows that 



where _λ_ 1( _A_ ) _≥· · · ≥ λn_ ( _A_ ) _>_ 0 are eigenvalues of _A_ , and max _i |_ 1 _− ηλi_ ( _A_ ) _|_ is the convergence factor of _In − ηA_ . Therefore, _u_<sup>(</sup><sup>_l_)</sup> _→ u_<sup>_∗_</sup> if and only if max _i |_ 1 _− ηλi_ ( _A_ ) _| <_ 1, which translates to a condition on _η_ as 



Note that the smaller the convergence factor, the faster the convergence. 

**Optimal Step Size.** The optimal step size _η_<sup>_∗_</sup> minimizes the convergence factor max _i |_ 1 _− ηλi_ ( _A_ ) _|_ . Since max _i |_ 1 _− ηλi_ ( _A_ ) _|_ = max _{|_ 1 _− ηλ_ 1( _A_ ) _|, |_ 1 _− ηλn_ ( _A_ ) _|}_ , _η_<sup>_∗_</sup> satisfies _−_ 1 + _η_<sup>_∗_</sup> _λn_ ( _A_ ) = 1 _− η_<sup>_∗_</sup> _λ_ 1( _A_ ); hence, the optimal step size and the corresponding optimal convergence factor become 



where ( _λ_ 1 _/λn_ )( _A_ ) is the condition number of _A_ . 

**Convergence Rate.** Suppose 0 _< η <_ 1 _/λ_ 1( _A_ ). Then the convergence factor becomes _ρ_ := max _i |_ 1 _− ηλi_ ( _A_ ) _|_ = 1 _− ηλn_ ( _A_ ), and we can write, taking _u_<sup>(0)</sup> = 0, 



**Preconditioning.** Let _D ∈_ R<sup>_n×n_</sup> be positive definite. A preconditioned Richardson iteration solves _D_<sup>_−_1</sup> _Au_<sup>_∗_</sup> = _D_<sup>_−_1</sup> _b_ , and is given by 



It is clear that the above has the same fixed point as the unconditioned one. It naturally follows that 



Note that _D_<sup>_−_1</sup> _A_ and _D_<sup>_−_1</sup><sup>_/_2</sup> _AD_<sup>_−_1</sup><sup>_/_2</sup> are similar matrices, so they have the same characteristic polynomials, hence the same eigenvalues. Hence, the above quantities can be expressed in terms of _λj_ ( _D_<sup>_−_1</sup><sup>_/_2</sup> _AD_<sup>_−_1</sup><sup>_/_2</sup> ), _j ∈{_ 1 _, n}_ as well. 

# **B. Auxiliary Lemmas and Theorems** 

## **B.1. Approximation based on Discretization** 

**Lemma B.1.** _Let p_ ( _x_ ) _be a L_ Lip _-Lipschitz continuous density supported on_ R _. Fix a partition_ Γ = _{a_ = _γ_ 1 _< γ_ 2 _< · · · < γC_ +1 = _b} of an interval_ ( _a, b_ ] _such that_ P _{x ∈_ ( _a, b_ ] _}_ = _ε for ε >_ 0 _. Let_ ∆ _c_ := _γc_ +1 _− γc and |_ Γ _|_ := max1 _≤c≤C_ ∆ _c. Define p_ ˜( _x_ ) =<sup>�</sup><sup>_C_</sup> _c_ =1<sup>∆</sup> _c_<sup>_−_11</sup> [ _γc,γc_ +1)<sup>(</sup><sup>_x_)</sup> � _γγcc_ +1 _p_ ( _x_ ) _dx/_ (1 _− ε_ ) _. Then_ TV( _p,_ ˜ _p_ ) _≤ L_ Lip _|_ Γ _|_ ( _b − a_ ) _/_ 2 + _ε._ 

13 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

_Proof._ Note that 2 TV( _p,_ ˜ _p_ ) = � _|p − p_ ˜ _|dx_ where 



where _xc ∈_ [ _γc, γc_ +1] is the point in the interval whose density is equal to the mean value of density in the interval. By _L_ Lip-Lipschitz, 



which completes the proof. 

**Lemma B.2.** _Fix θ ∈_ R<sup>_k_</sup> _and let p_ ( _· | θ_ ) _be an exponential-family density on_ R _:_ 



_b Choose a < b such that p_ ( _· | θ_ ) _is continuous and positive on_ [ _a, b_ ] _and_ � _a_<sup>_p_(</sup><sup>_y|θ_)</sup><sup>_dy_=1</sup><sup>_−ϵforϵ>_0</sup><sup>_.Define_</sup> _p_ ( _a,b_ ]( _y | θ_ ) := _p_ ( _y | θ_ ) **1** ( _a,b_ ]( _y_ ) _/_ (1 _− ϵ_ ) _. For each C ≥_ 1 _, let_ Γ<sup>(</sup><sup>_C_)</sup> = _{a_ = _γ_ 1 _< γ_ 2 _< · · · < γC_ +1 = _b} be a partition with mesh size |_ Γ<sup>(</sup><sup>_C_)</sup> _|_ := max _c∈_ [ _C_ ]( _γc_ +1 _− γc_ ) _satisfying |_ Γ<sup>(</sup><sup>_C_)</sup> _| →_ 0 _as C →∞. Let ξc_ := ( _γc_ + _γc_ +1) _/_ 2 _be the midpoint for c ∈_ [ _C_ ] _and let_ ∆ _c_ := _γc_ +1 _− γc. Define two pdfs on_ ( _a, b_ ] _:_ 



_Letting ωp_ ( _δ_ ) := sup _{|p_ ( _y | θ_ ) _− p_ ( _y_<sup>_′_</sup> _| θ_ ) _|_ : _y, y_<sup>_′_</sup> _∈_ ( _a, b_ ] _, |y − y_<sup>_′_</sup> _| ≤ δ}, for every C,_ 



_In particular, if p_ ( _· | θ_ ) _is L_ Lip _-Lipschitz on_ ( _a, b_ ] _, then_ 



_Proof._ Define two pmfs on _{_ 1 _, . . . , C}_ : 



ˆ and note that TV ( _pC,_ ˆ _pC_ ) = TV ( _qC,_ ˆ _qC_ ) since _∥pC − pC∥L_ 1( _a,b_ ] =<sup>�</sup><sup>_C_</sup> _c_ =1<sup>_|qC_(</sup><sup>_c|θ_)</sup><sup>_−q_ˆ</sup><sup>_C_(</sup><sup>_c|θ_)</sup><sup>_|_.Since</sup><sup>_A_(</sup><sup>_θ_)isan</sup> additive constant of the log-density independent of _c_ , it cancels under sm. Therefore, 



14 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

_γc_ +1 ˆ Define the following: _g_ ˆ _c_ := � _γc p_ ( _y | θ_ ) _dy_ , _gc_ := _p_ ( _ξc | θ_ )∆ _c_ , _Z_ :=<sup>�</sup><sup>_C_</sup> _c_ =1<sup>_gc_=1</sup><sup>_−ϵ_, and</sup><sup>_Z_ˆ:=�</sup><sup>_C_</sup> _c_ =1<sup>_g_ˆ</sup><sup>_c_.Note that</sup> TV( _qC,_ ˆ _qC_ ) = _∥qC − qC∥_ 1 _/_ 2 where 



The last inequality holds since _|Z_<sup>ˆ</sup> _− Z|_ = _|_<sup>�</sup><sup>_C_</sup> _c_ =1<sup>(ˆ</sup><sup>_gc −gc_)</sup><sup>_| ≤_�</sup><sup>_C_</sup> _c_ =1<sup>_|g_ˆ</sup><sup>_c −gc|_.Now, consider the term</sup><sup>_|gc −g_ˆ</sup><sup>_c|_.Since</sup><sup>_p_is</sup> _L_ Lip-Lipschitz: 



Since _ξc_ is the midpoint of [ _γc, γc_ +1], the integral � _γγcc_ +1 _|y − ξc|dy_ represents the sum of the areas of two identical right-angled triangles, each with base ∆ _c/_ 2 and height ∆ _c/_ 2. Thus: 



Substituting this back into the sum: 



For the general continuous case (where Lipschitz continuity is not assumed), we retain the bound using the modulus of continuity _ωp_ ( _|_ Γ<sup>(</sup><sup>_C_)</sup> _|/_ 2), noting that _|gc − g_ ˆ _c| ≤_ ∆ _cωp_ ( _|_ Γ<sup>(</sup><sup>_C_)</sup> _|/_ 2). Combining these estimates with the TV bound yields the stated result. 

## **B.2. Spectral Scaling of Gram Matrix** 

iid Throughout this section, we consider _xi ∼N_ (0 _, Id/d_ ) for _i ∈_ [ _n_ ] where _n > d_ for a fixed _d_ . **Lemma B.3** (Linear kernel) **.** _Let G_ := _XX_<sup>_⊤_</sup> _, a rank d matrix, and λ_ 1 _≥· · · ≥ λd be its d nonzero eigenvalues. Then λj_ = �1 + _O_ P( _n_<sup>_−_1</sup><sup>_/_2</sup> )� _n/d for j ∈_ [ _d_ ] _._ 

iid _Proof._ Note that _X_ = _Z/√d_ where _Zij ∼N_ (0 _,_ 1). Letting _s_ 1 _≥· · · ≥ sd_ be the singular values of _Z_ , it holds that _sj_ = ~~�~~ _dλj_ . By Boucheron et al. (2013, Theorem 5.6), if _z ∼N_ (0 _, Id′_ ), _f_ : R<sup>_d′_</sup> _→_ R is _L_ -Lipschitz, then for any _t ≥_ 0, 



The mapping _Z �→ s_ 1( _Z_ ) is 1-Lipschitz as a function of vec( _Z_ ) with respect to _ℓ_ 2 norm, since _s_ 1( _Z_ ) = max _∥v∥_ 2=1 _∥Zv∥_ 2 = _∥Z∥_ 2 and 



where _∥Z∥F_ denotes the Frobenius norm. Moreover, since max _j |sj_ ( _Z_ ) _− sj_ ( _Z_<sup>_′_</sup> ) _| ≤∥Z − Z_<sup>_′_</sup> _∥_ 2, _s_ 1 _, · · · , sd_ are all 1-Lipschitz (see Horn & Johnson (2012, Theorem 7.4.9.1)). On the other hand, by Davidson & Szarek (2001, Theorem 2.13), 



15 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

Combining these two facts, we have, for _j ∈_ [ _d_ ], 



on the event _E_ with P( _E_ ) _≥_ 1 _−_ 2 _e_<sup>_−t_2</sup><sup>_/_2</sup> . Take _ϵn_ := _n/dλj_<sup>_−_1.Then</sup><sup>_√_</sup> _<u>nϵn</u>_ = _O_ P(1); hence, _λj_ = �1 + _O_ P( _n_<sup>_−_1</sup><sup>_/_2</sup> )� _n/d_ . 

**Lemma B.4** (RBF kernel) **.** _Define G ∈_ R<sup>_n×n_</sup> _as Gij_ = exp� _−_ 0 _._ 5 _∥xi − xj∥_<sup>2�</sup> _. Then λ_ 1( _G_ ) ≳ _n and there exists γ >_ 0 _where_ 



_almost surely for any large enough n._ 

_Proof._ The analytic eigenstructure of the Gaussian (RBF) kernel under a Gaussian input measure is available in closed form. In one dimension, Zhu et al. (1997) showed that the Mercer eigenvalues decay geometrically: _µi ≍ β_<sup>_i_</sup> for _i_ = 0 _,_ 1 _,_ 2 _, . . ._ with some _β ∈_ (0 _,_ 1). This form extends to isotropic multivariate Gaussians because, for a product Gaussian measure and an isotropic RBF kernel, the eigenfunctions factorize across coordinates, and the eigenvalues take a product form with combinatorial multiplicities; see Rasmussen & Williams (2005, Section 4.3.1). In particular, when the spectrum is grouped by total degree, the distinct eigenvalues still decay geometrically in that degree, while their multiplicities grow polynomially. This geometric decay of the population spectrum can be transferred to the empirical spectrum of the Gram matrix _G_ via a Mercer truncation argument: approximating the kernel by its first _T < n_ eigencomponents yields a rank _T_ Gram matrix _G<T_ , and the ( _T_ + 1)-st (hence smallest) sample eigenvalues is upper bounded through trace bounds and Markov inequality, following the technique in Burt et al. (2019). 

Let _κ_ ( _x, x_<sup>_′_</sup> ) = exp� _−_ 0 _._ 5 _∥x − x_<sup>_′_</sup> _∥_<sup>2�</sup> be the RBF kernel. By Mercer’s theorem, the spectral decomposition is _κ_ ( _x, x_<sup>_′_</sup> ) = � _j≥_ 1<sup>_µjϕj_(</sup><sup>_x_)</sup><sup>_ϕj_(</sup><sup>_x′_) where</sup><sup>_{ϕj}_are orthonormal basis and</sup><sup>_{µj}_are eigenvalues.</sup> 

**The largest eigenvalue.** Since _G ≻_ 0, _λ_ 1( _G_ ) = max _x_ =0 _x_<sup>_⊤_</sup> _Gx/x_<sup>_⊤_</sup> _x_ , hence 



Note that by the strong law of U-statistics, the average of off diagonal elements converges: 



Therefore, _λ_ 1( _G_ ) ≳ _n_ for any sufficiently large _n_ almost surely. 

**The smallest eigenvalue.** Fix 1 _≤ T < n_ and define a truncated kernel _κ≤T_ ( _x, x_<sup>_′_</sup> ) :=<sup>�</sup><sup>_T_</sup> _t_ =1<sup>_µtϕt_(</sup><sup>_x_)</sup><sup>_ϕt_(</sup><sup>_x′_) with the Gram</sup> matrix _G≤T ∈_ R<sup>_n×n_</sup> such that ( _G≤T_ ) _ij_ = _κ≤T_ ( _xi, xj_ ). We write _G_ = _G≤T_ + _R>T_ where _R>T_ is the residual matrix. Then we have 



where the first inequality is since _T < n_ , the second is from Weyl’s inequality (if _A, B ∈_ R<sup>_n×n_</sup> are symmetric, then _λi_ ( _A_ + _B_ ) _∈_ � _λi_ ( _A_ ) + _λn_ ( _B_ ) _, λi_ ( _A_ ) + _λ_ 1( _B_ )�), the equality is because _G≤T_ has a rank _T_ , and the final inequality holds since _R>T ≻_ 0. Now, 



16 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

and since E _ϕt_ ( _x_ )<sup>2</sup> = _∥ϕt∥_<sup>2</sup> _L_<sup>2= 1 from</sup><sup>_ϕt_being an orthonormal basis, we have E tr(</sup><sup>_R>T_) =</sup><sup>_n_�</sup> _t>T_<sup>_µt_.Then we can, as</sup> in the proof of Burt et al. (2019, Theorem 4), build a probabilistic bound using Markov’s inequality: for a sequence _an_ , 



If we take _an_ = _n_<sup>1+</sup><sup>_ϵ_</sup> for any _ϵ >_ 0 such that<sup>�</sup> _n≥_ 1<sup>1</sup><sup>_/an_issummable,thenbytheBorel-Cantellilemma,wehave</sup> _λn_ ( _G_ ) _≤ n_<sup>2+</sup><sup>_ϵ_�</sup> _t>T_<sup>_µt_almost surely for any sufficiently large</sup><sup>_n_.</sup> 

We now use the following fact (Zhu et al., 1997) to get an explicit form of the tail sum: if _x ∼N_ (0 _, σ_<sup>2</sup> ) and _κ_<sup>1D</sup> ( _x, x_<sup>_′_</sup> ) = exp � _−_ 0 _._ 5( _x − x_<sup>_′_</sup> )<sup>2</sup> _/ℓ_<sup>2�</sup> , then the eigenvalues _{µi}_ of _κ_<sup>1D</sup> decay geometrically in the following form: 



where 1 _/a_ = 4 _σ_<sup>2</sup> and 1 _/b_ = 2 _ℓ_<sup>2</sup> . This result transfers directly to the multivariate isotropic Gaussian measure, since the eigenvalues and eigenfunctions are also a product of the same univariate Gaussian. In isotropic case, an eigenfunction in _d_ -dimension is a tensor product of _d_ number of one-dimensional eigenfunctions, so it is indexed by a multi index ( _i_ 1 _, i_ 2 _, · · · , id_ ). Accordingly, eigenfunctions with the same value of _i_ 1 + _· · ·_ + _id_ (degree) share the same eigenvalue. Hence, the multiplicity of _µk_ equals the number of possible ways to distribute _k_ balls into _d_ bins. 

Specifically, let _µ_ ˜<sup>(</sup><sup>_k_)</sup> be an eigenvalue of _κ_ ( _x, x_<sup>_′_</sup> ) whose degree is _k_ and has a multiplicity _mk_ = � _k_ + _d−d−_ 1 1�. Each _µ_ ˜<sup>(</sup><sup>_k_)</sup> is a product of _µ_<sup>1D</sup> _i_ in a multi index ( _i_ 1 _, i_ 2 _, · · · , id_ ) such that _i_ 1 + _· · ·_ + _id_ = _k_ : 



Let _T<M_ := _|{_ ( _i_ 1 _, · · · , id_ ) : _i_ 1 + _· · ·_ + _id < M }|_ be the total number of eigenvalues whose degree _k_ is less than _M_ . Then 



The tail sum of eigenvalues of degree larger or equal to _M_ is 



Fix _q ∈_ ( _β,_ 1). Writing _γk_ := � _k_ + _d−d−_ 1 1� _β_<sup>_k_</sup> , we choose _M_ large enough so that 



Then<sup>�</sup> _k≥M_<sup>_γk≤γM_</sup> � _t≥_ 0<sup>_qt_=</sup><sup>_γM/_(1</sup><sup>_−q_); hence</sup> 

where _C_ is a constant that only depends on _α_ , _β_ , _q_ , and _d_ , since � _Md_ + _−d_ 1 _−_ 1� _∼ M_<sup>_d−_1</sup> _/_ ( _d −_ 1)! (note that _aM ∼ bM_ means _aM /bM →_ 1 as _M →∞_ ). 

For _n_ , choose a small _a >_ 0 such that after letting _M_ = _⌊an_<sup>1</sup><sup>_/d_</sup> _⌋_ , we have _n > T<M_ . Since _T<M ∼ M_<sup>_d_</sup> _/d_ !, one such choice could be _a_ ≲ ( _d_ !)<sup>1</sup><sup>_/d_</sup> . We assume _n_ is large enough so that _M_ satisfies (15). Then 



17 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

where _γ_ = _a_ log(1 _/β_ ) _>_ 0 (note that 0 _< β <_ 1) and _C_<sup>_′_</sup> is a constant. 

**Condition number.** Combining _λ_ 1( _G_ ) ≳ _n_ and _λn_ ( _G_ ) ≲ _n_<sup>2+</sup><sup>_ϵ_+</sup><sup>_d−_</sup> _d_<sup>1</sup> exp� _−γn_<sup>1</sup><sup>_/d_�</sup> , we have 



almost surely for any large enough _n_ , which completes the proof by taking _ϵ_ = 1. 

**Lemma B.5** (RBF kernel with ridge) **.** _Define G ∈_ R<sup>_n×n_</sup> _as Gij_ = exp� _−_ 0 _._ 5 _∥xi − xj∥_<sup>2�</sup> _. Let σ_<sup>2</sup> _>_ 0 _. Then_ 



## _almost surely for any large enough n._ 

_Proof._ Note that _λ_ 1( _G_ ) _≤_ tr( _G_ ) = _n_ and _λn_ ( _G_ ) _≤_ tr( _G_ ) _/n_ = 1. For large enough _n_ , almost surely we have 



where _ρ ∈_ (0 _,_ 1) is defined in the proof of Lemma B.4. 

**Lemma B.6** (RBF kernel with ridge and row normalization) **.** _Define G ∈_ R<sup>_n×n_</sup> _as Gij_ = exp� _−_ 0 _._ 5 _∥xi − xj∥_<sup>2�</sup> _. Let D_ := diag( _s_ 1 _, · · · , sn_ ) _where si_ :=<sup>�</sup><sup>_n_</sup> _j_ =1<sup>_Gij.Then_</sup> 



_almost surely for any large enough n._ 

_Proof._ We write _A ∼ B_ to denote that _A_ and _B_ are similar matrices, sharing the same eigenvalues. Then _D_<sup>_−_1</sup> ( _G_ + _σ_<sup>2</sup> _In_ ) _∼ D_<sup>_−_1</sup><sup>_/_2</sup> ( _G_ + _σ_<sup>2</sup> _In_ ) _D_<sup>_−_1</sup><sup>_/_2</sup> =: _A_<sup>˜</sup> + _σ_<sup>2</sup> _D_<sup>_−_1</sup> . Since _D_<sup>_−_1</sup> _G_ is a row stochastic matrix, we have _λ_ 1( _A_<sup>˜</sup> ) = _λ_ 1( _D_<sup>_−_1</sup> _G_ ) = 1. Also, by Weyl’s inequality, _λ_ 1( _A_<sup>˜</sup> + _σ_<sup>2</sup> _D_<sup>_−_1</sup> ) _≤_ 1 + _σ_<sup>2</sup> _/s_ min _≤_ 1 + _σ_<sup>2</sup> since _si ≥_ 1 for any _i ∈_ [ _n_ ]. Therefore, 



On the other hand, _λn_ ( _A_<sup>˜</sup> + _σ_<sup>2</sup> _D_<sup>_−_1</sup> ) _≥ σ_<sup>2</sup> _/s_ max, and for a basis vector _ei_ , _λn_ ( _A_<sup>˜</sup> + _σ_<sup>2</sup> _D_<sup>_−_1</sup> ) _≤_<sup>_e_</sup> _<u>i</u>_<sup>_⊤_</sup><sup><u>( ˜</u></sup> _A_ + _eσ_<sup>_⊤_</sup> _i_<sup>2</sup><sup>_e_</sup> _D_<sup>_i−_1</sup> <u>)</u> _ei ≤_<sup><u>1+</u></sup> _si_<sup>_<u>σ</u>_2.</sup> Since this holds for any _i ∈_ [ _n_ ], we have 



Combined, 



Trivially, _s_ max _≤ n_ since all elements are in [0 _,_ 1]. For any _i ∈_ [ _n_ ], consider 



Conditioned on _xi_ , _κ_ ( _xi, xj_ ) are iid with mean E _xκ_ ( _xi, x_ ) _∈_ (0 _,_ 1). Therefore, conditionally on _xi_ , by the law of large number, 



By taking expectation with respect to _xi_ , the convergence holds unconditionally. Note that E _xκ_ ( _xi, x_ ) and E _x,x′κ_ ( _x, x_<sup>_′_</sup> ) exist as a Gaussian integral. Therefore, for large enough _n_ , writing _ρ_ := E _x,x′κ_ ( _x, x_<sup>_′_</sup> ) _∈_ (0 _,_ 1), 



almost surely. Combined, _s_ max = Θ( _n_ ) almost surely, which completes the proof. 

18 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

## **B.3. Population Optimizer of NLL** 

**Theorem B.7.** _For each Z_<sup>(0)</sup> _, define the truncated conditional density_ 



_b assuming_ � _a_<sup>_p_(</sup><sup>_t | Z_(0))</sup><sup>_dt >_0</sup><sup>_almost surely.Consider the class of piecewise-constant conditional densities supported on_</sup> ( _a, b_ ] _:_ 



_Define the truncated population negative log-likelihood_ 



_Then the (almost surely unique) minimizer of L_ tr( _q_ ) _over Q is_ 



_Equivalently, q_<sup>_∗_</sup> ( _· | Z_<sup>(0)</sup> ) _is obtained by bin-averaging p_ ( _· | Z_<sup>(0)</sup> ) _on_ ( _a, b_ ] _and then normalizing by the total mass_ � _ab_<sup>_p_(</sup><sup>_t | Z_(0))</sup><sup>_dt, in addition to the_∆</sup><sup>_c factor._</sup> 

_Proof._ Condition on _Z_<sup>(0)</sup> . Any _q_ ( _· | Z_<sup>(0)</sup> ) _∈Q_ takes the form _q_ ( _y | Z_<sup>(0)</sup> ) = ∆<sup>_−_</sup> _c_<sup>1</sup><sup>_qc_(</sup><sup>_Z_(0)) for</sup><sup>_y∈_[</sup><sup>_γc, γc_+1).Therefore,</sup> 



The term<sup>�</sup><sup>_C_</sup> _c_ =1<sup>(log ∆</sup><sup>_c_)</sup><sup>_pc_(</sup><sup>_Z_(0)) does not depend on</sup><sup>_q_, so minimizing the conditional expected NLL over</sup><sup>_Q_is equivalent</sup> to minimizing<sup>�</sup><sup>_C_</sup> _c_ =1<sup>_pc_(</sup><sup>_Z_(0))(</sup><sup>_−_log</sup><sup>_qc_(</sup><sup>_Z_(0))) over the simplex</sup><sup>_{qc≥_0</sup><sup>_,_�</sup> _c_<sup>_qc_= 1</sup><sup>_}_.This is the categorical cross-entropy</sup> between _p_ ( _Z_<sup>(0)</sup> ) = ( _pc_ ( _Z_<sup>(0)</sup> ))<sup>_C_</sup> _c_ =1<sup>and</sup><sup>_q_(</sup><sup>_Z_(0)) = (</sup><sup>_qc_(</sup><sup>_Z_(0)))</sup><sup>_C_</sup> _c_ =1<sup>, and its unique minimizer is</sup><sup>_qc_(</sup><sup>_Z_(0)) =</sup><sup>_pc_(</sup><sup>_Z_(0)) for all</sup><sup>_c_.</sup> Thus, _qc_<sup>_∗_(</sup><sup>_Z_(0)) =</sup><sup>_pc_(</sup><sup>_Z_(0)) =</sup> � _γγcc_ +1 _p_ ( _a,b_ ]( _y | Z_<sup>(0)</sup> ) _dy_ , which yields the stated _q_<sup>_∗_</sup> . Taking expectation over _Z_<sup>(0)</sup> proves the result. 

## **B.4. Kernel Ridge Regression (KRR)** 

**Theorem B.8** (Standard KRR) **.** _Let {xi, yi}_ 1: _n with xi ∈X , yi ∈_ R _, and let κ_ : _X × X →_ R _be a positive semidefinite kernel of RKHS H with norm ∥·∥H. Let σ_<sup>2</sup> _>_ 0 _be the ridge parameter. Consider the kernel ridge regression (KRR) problem_ 



_Let G ∈_ R<sup>_n×n_</sup> _where Gij_ = _κ_ ( _xi, xj_ ) _be the kernel Gram matrix, and λ_ 1( _G_ ) _, λn_ ( _G_ ) _be its maximum and minimum eigenvalues. Suppose η_<sup>(</sup><sup>_l_)</sup> _>_ 0 _satisfy_ sup _l η_<sup>(</sup><sup>_l_)</sup> _<_ 2 _/{λ_ 1( _G_ ) + _σ_<sup>2</sup> _}, and_<sup>�</sup><sup>_∞_</sup> _l_ =0<sup>_η_(</sup><sup>_l_)=</sup><sup>_∞.Define u_(0)</sup><sup>_≡_0</sup><sup>_and_</sup> 



19 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

_Then for all i ∈_ [ _n_ ] _, u_<sup>(</sup><sup>_l_)</sup> ( _xi_ ) _→ u_<sup>_∗_</sup> ( _xi_ ) _, and for any fixed x ∈X , u_<sup>(</sup><sup>_l_)</sup> ( _x_ ) _→ u_<sup>_∗_</sup> ( _x_ ) _. Furthermore, if η_<sup>(</sup><sup>_l_)</sup> = _η ∈_ �0 _,_ 1 _/_ ( _λ_ 1( _G_ ) + _σ_<sup>2</sup> )� _for all l ≥_ 0 _, then for L ≥_ 1 _,_ 



_where ρ_ := 1 _− η_ ( _λn_ ( _G_ ) + _σ_<sup>2</sup> ) _is the convergence factor._ 

_Proof._ By the representer theorem, we know that _u_<sup>_∗_</sup> ( _·_ ) =<sup>�</sup><sup>_n_</sup> _i_ =1<sup>_αiκ_(</sup><sup>_xi, ·_) for some</sup><sup>_α_=[</sup><sup>_α_1</sup><sup>_, · · ·, αn_]</sup><sup>_⊤∈_R</sup><sup>_n_.There-</sup> fore, we can write _u_<sup>_∗_</sup> _X_<sup>:=[</sup><sup>_u∗_(</sup><sup>_x_1)</sup><sup>_, · · ·, u∗_(</sup><sup>_xn_)]</sup><sup>_⊤_=</sup><sup>_Gα_,andnotethat</sup><sup>_∥u∗∥_2</sup> _H_<sup>=</sup><sup>_⟨_�</sup> _i_<sup>_αiκ_(</sup><sup>_xi, ·_)</sup><sup>_,_�</sup> _i_<sup>_αjκ_(</sup><sup>_xj, ·_)</sup><sup>_⟩H_=</sup> � _ij_<sup>_αiαjκ_(</sup><sup>_xi, xj_) =</sup><sup>_α⊤Gα_.Combined, the KRR is equivalent to a quadratic optimization</sup> 



where _Y_ := [ _y_ 1 _, · · · , yn_ ]<sup>_⊤_</sup> . Solving the gradient _−_ 2 _G_ ( _Y −Gα_ )+2 _σ_<sup>2</sup> _Gα_ = 0, the minimum satisfies _G_ ( _G_ + _σ_<sup>2</sup> _In_ ) _α_ = _GY_ , and after rearranging, we have 



(since if two invertible matrices _A_ and _B_ commute, so do _A_ and _B_<sup>_−_1</sup> ). 

On the other hand, for a test input _x ∈X_ , its function value is _u_<sup>_∗_</sup> ( _x_ ) =<sup>�</sup> _i_<sup>_αiκ_(</sup><sup>_xi, x_) =</sup><sup>_k_</sup> _x_<sup>_⊤α_, where (</sup><sup>_kx_)</sup><sup>_i_=</sup><sup>_κ_(</sup><sup>_xi, x_) for</sup> _i ∈_ [ _n_ ]. We know from (16) that _α_ = ( _G_ + _σ_<sup>2</sup> _In_ )<sup>_−_1</sup> _Y_ + _δ_ where _δ ∈_ ker( _G_ ); hence _kx_<sup>_⊤α_=</sup><sup>_k_</sup> _x_<sup>_⊤_(</sup><sup>_G_+</sup><sup>_σ_2</sup><sup>_In_)</sup><sup>_−_1</sup><sup>_Y_+</sup><sup>_k_</sup> _x_<sup>_⊤δ_.</sup> Let _fδ_ ( _·_ ) :=<sup>�</sup><sup>_n_</sup> _i_ =1<sup>_δiκ_(</sup><sup>_xi, ·_)</sup><sup>_∈H_be the function in RKHS defined by</sup><sup>_δ_.Since</sup><sup>_δ∈_ker(</sup><sup>_G_), we have</sup><sup>_∥fδ∥_2</sup> _H_<sup>=</sup><sup>_δ⊤Gδ_= 0,</sup> which implies that for any _x ∈X_ , _fδ_ ( _x_ ) = _⟨fδ_ ( _·_ ) _, κ_ ( _x, ·_ ) _⟩H_ =<sup>�</sup><sup>_n_</sup> _i_ =1<sup>_δiκ_(</sup><sup>_xi, x_) =</sup><sup>_k_</sup> _x_<sup>_⊤δ_= 0.Therefore,</sup> _u_<sup>_∗_</sup> ( _x_ ) = _kx_<sup>_⊤α_=</sup><sup>_k_</sup> _x_<sup>_⊤_(</sup><sup>_G_+</sup><sup>_σ_2</sup><sup>_In_)</sup><sup>_−_1</sup><sup>_Y._</sup> (17) 

Note that _u_<sup>_∗_</sup> _X_<sup>is a solution to the linear equation (</sup><sup>_G_+</sup><sup>_σ_2</sup><sup>_In_)</sup><sup>_u∗_</sup> _X_<sup>=</sup><sup>_GY_.Define the corresponding Richardson iteration with</sup> _{η_<sup>(</sup><sup>_l_)</sup> _}l≥_ 0 as the _training recursion_ : 



Equivalently, for each coordinate _j ∈_ [ _n_ ], 



**Convergence of the training recursion.** We repeat the same argument given in Section A. Rearranging (18) in terms of the error _u_<sup>(</sup> _X_<sup>_l_)</sup><sup>_−u_</sup> _X_<sup>_∗_, and noting that (</sup><sup>_G_+</sup><sup>_σ_2</sup><sup>_In_)</sup><sup>_u∗_</sup> _X_<sup>=</sup><sup>_GY_, we have</sup> 



Using the spectral decomposition _G_ = _Q_ Λ _Q_<sup>_⊤_</sup> , we have _In − η_<sup>(</sup><sup>_l_)</sup> ( _G_ + _σ_<sup>2</sup> _In_ ) = _Q_ ( _In − η_<sup>(</sup><sup>_l_)</sup> (Λ + _σ_<sup>2</sup> _In_ )) _Q_<sup>_⊤_</sup> ; hence 

Therefore, (18) converges if 0 _< η_<sup>(</sup><sup>_l_)</sup> ( _λ_ 1( _G_ ) + _σ_<sup>2</sup> ) _<_ 2 for all _l ≥_ 0, sup _l η_<sup>(</sup><sup>_l_)</sup> _<_ 2 _/{λ_ 1( _G_ ) + _σ_<sup>2</sup> _}_ , and<sup>�</sup><sup>_∞_</sup> _l_ =0<sup>_η_(</sup><sup>_l_)=</sup><sup>_∞_.</sup> **Convergence rate of the training recursion.** Suppose _η_<sup>(</sup><sup>_l_)</sup> = _η ∈_ �0 _,_ ( _λ_ 1( _G_ ) + _σ_<sup>2</sup> )<sup>_−_1�</sup> is a constant. Then we have max _i_ ��1 _− η_ ( _λi_ ( _G_ ) + _σ_ 2)�� = 1 _− η_ ( _λn_ ( _G_ ) + _σ_ 2), and noting that _u_ (0) := 0, 



20 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

**Convergence of the testing recursion.** For each _u_<sup>(</sup> _X_<sup>_l_), define the</sup><sup>_testing recursion u_(</sup><sup>_l_)(</sup><sup>_x_) as</sup><sup>_u_(0)(</sup><sup>_x_) := 0 and for</sup><sup>_l ≥_0,</sup> 



If 0 _< η_<sup>(</sup><sup>_l_)</sup> ( _λ_ 1( _G_ ) + _σ_<sup>2</sup> ) _<_ 2 and<sup>�</sup><sup>_∞_</sup> _l_ =0<sup>_η_(</sup><sup>_l_)=</sup><sup>_∞_, we know that</sup><sup>_u_(</sup><sup>_l_)</sup><sup>_→u∗_, and we can see that (20) still holds if we plug</sup> _u_<sup>_∗_</sup> _X_<sup>into</sup><sup>_u_(</sup> _X_<sup>_l_)and</sup><sup>_u∗_(</sup><sup>_x_) into</sup><sup>_u_(</sup><sup>_l_+1)(</sup><sup>_x_) and</sup><sup>_u_(</sup><sup>_l_)(</sup><sup>_x_):</sup> 



Subtracting (21) from (20) and writing _e_<sup>(</sup> _X_<sup>_l_):=</sup><sup>_u_(</sup> _X_<sup>_l_)</sup><sup>_−u_</sup> _X_<sup>_∗_and</sup><sup>_e_(</sup><sup>_l_)(</sup><sup>_x_) :=</sup><sup>_u_(</sup><sup>_l_)(</sup><sup>_x_)</sup><sup>_−u∗_(</sup><sup>_x_), we have</sup> 



The first term is a contraction since _|_ 1 _− η_<sup>(</sup><sup>_l_)</sup> _σ_<sup>2</sup> _| <_ 1 under our assumption, and the second term vanishes as _u_<sup>_∗_</sup> converges. Therefore, _e_<sup>(</sup><sup>_l_)</sup> ( _x_ ) _→_ 0, i.e., _u_<sup>(</sup><sup>_l_)</sup> ( _x_ ) _→ u_<sup>_∗_</sup> ( _x_ ). 

**Convergence rate of the testing recursion.** Since _u_<sup>(0)</sup> _X_<sup>:= 0 and</sup><sup>_u_(0)(</sup><sup>_x_) := 0, we can write</sup> 



Suppose _η_<sup>(</sup><sup>_l_)</sup> = _η ∈_ �0 _,_ ( _λ_ 1( _G_ ) + _σ_<sup>2</sup> )<sup>_−_1�</sup> . Define _a_ := 1 _− ησ_<sup>2</sup> , _A_ := _I − η_ ( _G_ + _σ_<sup>2</sup> _In_ ) = _aIn − ηG_ . Since the training error proceeds as _e_<sup>(</sup> _X_<sup>_l_)=</sup><sup>_Ale_(0)</sup> _X_<sup>, accumulating (22), we see that the test recursion error rolls out as</sup> 



Note that _ηG_ = _aIn − A_ by the definition, and 



Therefore, _e_<sup>(</sup><sup>_l_)</sup> ( _x_ ) = _−kx_<sup>_⊤_</sup> � _I − η_ ( _G_ + _σ_<sup>2</sup> _In_ )� _l_ ( _G_ + _σ_ 2 _In_ ) _−_ 1 _Y_ . Taking absolute value, we see that for _L ≥_ 0, 



which completes the proof. 

# **C. Proof of Main Theorems** 

Theorem 5.1 follows from Lemma B.3-B.4, and Theorem 5.2 is a restatement of Lemma B.6. 

21 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

## **C.1. Proof of Theorem 3.1** 

_Proof._ Set token dimension _d_<sup>_′_</sup> = _d_ + 4. Let _ϑ_ be (unspecified elements are 0), for _l_ = 0 _, · · · , L −_ 1, 



**Step 1 (First attention layer output)** . For _l_ = 0 _, · · · , L_ , we write _K_<sup>(</sup><sup>_l_)</sup> _Z_<sup>(</sup><sup>_l_)</sup> = [ _k_ 1<sup>(</sup><sup>_l_)</sup><sup>_, · · ·, k_</sup> _n_<sup>(</sup><sup>_l_</sup> +1<sup>)],</sup><sup>_Q_(</sup><sup>_l_)</sup><sup>_Z_(</sup><sup>_l_)=</sup> [ _q_ 1<sup>(</sup><sup>_l_)</sup><sup>_, · · ·, q_</sup> _n_<sup>(</sup><sup>_l_</sup> +1<sup>)], and</sup><sup>_V_(</sup><sup>_l_)</sup><sup>_Z_(</sup><sup>_l_)= [</sup><sup>_v_</sup> 1<sup>(</sup><sup>_l_)</sup><sup>_, · · ·, v_</sup> _n_<sup>(</sup><sup>_l_</sup> +1<sup>)].The first attention layer in (9) is expressed as, for</sup><sup>_j_= 1</sup><sup>_, . . . , n_+ 1,</sup> 



where second equality follows from _Mi,j_<sup>(0)=1</sup><sup>_{i>n}_andthirdequalityfollowsfrom</sup><sup>_v_</sup> _n_<sup>(0)</sup> +1<sup>=(0</sup><sup>_⊤_</sup> _d_ +1<sup>_,_1</sup><sup>_,_0</sup><sup>_,_0)</sup><sup>_⊤_(dueto</sup> _Vd_<sup>(0)</sup> +2 _,d_ +1<sup>= 1) and</sup><sup>_K_(0)=</sup><sup>_Q_(0)= diag((1</sup><sup>_⊤_</sup> _d_<sup>_,_0</sup> 4<sup>_⊤_)).Therefore,</sup> 



## **Step 2 (Richardson iteration implementation)** 

For _l ≥_ 1, (10) gives us, for _j_ = 1 _, . . . , n_ + 1, 



where recall that _Mi,j_ = 1 _{i≤n}_ . The last two rows of _Z_<sup>(</sup><sup>_l_)</sup> , denoted as [ _Z_<sup>(</sup><sup>_l_)</sup> ] _d_ +3 _,_ : and [ _Z_<sup>(</sup><sup>_l_)</sup> ] _d_ +4 _,_ :, correspond to the following recursions: initiating from _f_<sup>(0)</sup> ( _·_ ) _≡_ 0 and _g_<sup>(0)</sup> ( _·_ ) _≡_ 0, 



Assuming _η_<sup>(</sup><sup>_l_)</sup> _∈_ �0 _,_ 2 _/_ ( _λ_ 1( _G_ ) + _σ_<sup>2</sup> )� for all _l ≥_ 1 and<sup>�</sup><sup>_∞_</sup> _l_ =0<sup>_η_(</sup><sup>_l_)=</sup><sup>_∞_, as</sup><sup>_l →∞_, by Theorem B.8,</sup> 



22 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

Therefore, for any column _j ∈_ [ _n_ + 1], the rows [ _Z_<sup>(</sup><sup>_l_+1)</sup> ] _d_ +3: _,j_ and [ _Z_<sup>(</sup><sup>_l_+1)</sup> ] _d_ +4: _,j_ evolve as (24) and (25) respectively, each starting from 0. For large enough _L_ , [ _Z_<sup>(</sup><sup>_L_)</sup> ] _d_ +3: _,n_ +1 _≈ µ_ ( _Z_<sup>(0)</sup> ), and _σ_<sup>2</sup> + [ _Z_<sup>(</sup><sup>_l_+1)</sup> ] _d_ +2: _,n_ +1 _−_ [ _Z_<sup>(</sup><sup>_l_+1)</sup> ] _d_ +4: _,n_ +1 _≈ τ_ ( _Z_<sup>(0)</sup> ). The convergence rates follow from Theorem B.8. 

## **C.2. Proof of Theorem 4.1** 

_Proof._ We first state the regularity conditions. For GP regression, the posterior predictive distribution is Gaussian with _τ_ ( _Z_<sup>(0)</sup> ) _≥ σ_<sup>2</sup> _>_ 0, so these conditions hold on any fixed truncation interval as long as the predictive mean and variance remain in a compact subset. We assume, uniformly over the context _Z_<sup>(0)</sup> , 

1. � _µ_ ( _Z_<sup>(0)</sup> ) _, τ_ ( _Z_<sup>(0)</sup> )� _∈K_ for some _K ⊂_ R _×_ (0 _, ∞_ ) and _ψ_ is approximated uniformly on _K_ by MLP _ψ_<sup>˜</sup> with a non-polynomial activation, 

2. the target PPD belongs to a one-dimensional exponential family _fθ_ ( _y_ ), 

3. _fθ_ ( _y_ ) is Lipschitz continuous on ( _a, b_ ] and the grid is equidistant. 

Fix a compact set _K ⊂_ R _×_ (0 _, ∞_ ) on which the mapping _ψ_ : _K →_ R<sup>2</sup> is defined as _ψ_ : ( _µ, τ_ ) _�→_ � _<u>µτ</u>_<sup>_, −_</sup> 2<sup><u>1</u></sup> _τ_ �. Such _K_ is plausible given that _y ∈_ ( _a, b_ ] and _τ_ ( _Z_<sup>(0)</sup> ) _∈_ [ _σ_<sup>2</sup> _, σ_<sup>2</sup> + _κ_ ( _x, x_ )] for any _Z_<sup>(0)</sup> . By the universal approximation theorem (Hornik, 1991; Leshno et al., 1993), for any _δ_ MLP _>_ 0, there exists a single layer MLP _ψ_<sup>˜</sup> : _K →_ R<sup>2</sup> of the form 



with a non-polynomial activation function act (e.g., ReLU), such that sup _θ∈K ∥ψ_ ( _θ_ ) _− ψ_<sup>˜</sup> ( _θ_ ) _∥_ 2 _≤ δ_ MLP. For a Lipschitz activation, _ψ_<sup>˜</sup> is also Lipschitz; for any _x, x_<sup>_′_</sup> _∈K_ , _∥ψ_<sup>˜</sup> ( _x_ ) _− ψ_<sup>˜</sup> ( _x_<sup>_′_</sup> ) _∥_ 2 _≤ Lψ_ ˜<sup>_∥x −x′∥_2.</sup> 

Define the midpoint _ξc_ := ( _γc_ + _γc_ +1) _/_ 2 and Ξ _∈_ R<sup>_C×_2</sup> as Ξ _c,_ : = [ _ξc, ξc_<sup>2]</sup><sup>_⊤_for</sup><sup>_c∈_[</sup><sup>_C_].Fix</sup><sup>_Z_(0).Let TF</sup><sup>_L,θ_(</sup><sup>_Z_(0))=:</sup> � _µL_ ( _Z_<sup>(0)</sup> ) _, τL_ ( _Z_<sup>(0)</sup> )� _⊤_ . We suppress the notation _Z_ (0) and define the logits _ℓ_ 1 _, ℓ_ 2 _, ℓ_ 3 _∈_ R _C_ as 

_ℓ_ 1 := Ξ _ψ_<sup>˜</sup> ( _µL, τL_ ) Transformer logit _, ℓ_ 2 := Ξ _ψ_<sup>˜</sup> ( _µ, τ_ ) exact moments _µ_ , _τ, ℓ_ 3 := Ξ _ψ_ ( _µ, τ_ ) exact natural parameter conversion _ψ._ 

Recall that by Theorem 3.1, _∥_ ( _µ, τ_ ) _−_ ( _µL, τL_ ) _∥∞_ ≲ exp( _−_ (1 _− ρ_ ) _L_ ) which implies _∥_ ( _µ, τ_ ) _−_ ( _µL, τL_ ) _∥_ 2 ≲ exp( _−_ (1 _− ρ_ ) _L_ ). Therefore, 



where _∥_ Ξ _∥_ 2 is the fixed, largest singular value of Ξ. Moreover, by the definition of _ψ_<sup>˜</sup> , it holds that _∥ℓ_ 2 _− ℓ_ 3 _∥_ 2 _≤∥_ Ξ _∥_ 2 _δ_ MLP. Combined, we have 



Define probability vectors _pl_ := sm( _ℓl_ ) for _l ∈{_ 1 _,_ 3 _}_ , and write ∆ as an upper bound of the _ℓ∞_ difference of the logits _∥ℓ_ 1 _− ℓ_ 3 _∥∞ ≤_ ∆. This implies that, for any _c ∈_ [ _C_ ], 











23 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

Since ∆ _>_ 0 and ( _p_ 3) _c ≤_ 1, this implies that, for a small ∆, 



We construct a piecewise continuous density on ( _a, b_ ] with probability vectors _p_ 1 and _p_ 3: 



By Lemma B.1-B.2, _∥pn − q_ 3 _∥L_ 1( _a,b_ ] ≲ 1 _/C_ + _ε_ tail( _Z_<sup>(0)</sup> ), and since _∥qϑ − q_ 3 _∥L_ 1( _a,b_ ] = _∥p_ 1 _− p_ 3 _∥_ 1, we conclude that TV( _pn, qϑ_ ) ≲ exp( _−_ (1 _− ρ_ ) _L_ ) + _δ_ MLP + 1 _/C_ + _ε_ tail( _Z_<sup>(0)</sup> ) _,_ 

which completes the proof. 

Although we focus on GP regression settings where PPD is Gaussian and _h_ is a constant, if the target exponential family distribution contains non-constant _h_ terms, the parameter mapping _ψ_ and data mapping _T_ can be appropriately expanded such that _h_ becomes a constant on its support. For example, if the target exponential density is an inverse Gaussian distribution parameterized by mean _µ_ and variance _τ_ so that _fθ_ ( _y_ ) = � _<u>µ</u>_ 2 _πy_<sup>3</sup> _<u>/τ</u>_<sup>3exp</sup> � _−_<sup>_<u>µ</u>_</sup><sup><u>(</u></sup><sup>_<u>y</u>_</sup> 2<sup>_−_</sup> _yτ_<sup>_<u>µ</u>_</sup><sup><u>)2</u></sup> � 1 _{y>_ 0 _}_ = exp � _⟨_ ( _−_ 0 _._ 5 _µ/τ, −_ 0 _._ 5 _µ_<sup>3</sup> _/τ_ ) _,_ ( _y,_ 1 _/y_ ) _⟩_ +<sup>_<u>µ</u>_</sup> _τ_<sup>2+</sup><sup><u>1</u></sup> 2<sup>(log</sup><sup>_<u>µ</u>_</sup> _τ_<sup>3</sup><sup>_−_log(2</sup><sup>_π_))</sup> � _y_<sup>_−_3</sup><sup>_/_2</sup> 1 _{y>_ 0 _}_ , we parametrize it as _fθ_ ( _y_ ) = exp( _⟨ψ_ ( _µ, τ_ ) _, T_ ( _y_ ) _⟩_ +<sup>_<u>µ</u>_</sup> _τ_<sup>2+</sup> 2<sup><u>1</u>(log</sup><sup>_<u>µ</u>_</sup> _τ_<sup>3</sup><sup>_−_log(2</sup><sup>_π_)))</sup><sup>_h_(</sup><sup>_y_)with</sup><sup>_T_(</sup><sup>_y_)=(</sup><sup>_y,_1</sup><sup>_/y,_log</sup><sup>_y_),</sup> _ψ_ ( _µ, τ_ ) = ( _−µ/_ (2 _τ_ ) _, −µ_<sup>3</sup> _/_ (2 _τ_ ) _, −_ 3 _/_ 2), and _h_ ( _y_ ) = 1 _{y>_ 0 _}_ . 

## **C.3. Proof of Theorem 5.3** 

_Proof._ Under the assumption, TF<sup>pr</sup> _ϑ,L_<sup>implements</sup><sup>_L_iterations of the preconditioned Richardson iteration solving (13).As</sup> such, the theorem addresses bounding the finite iteration convergence error of the two testing recursions to obtain the linear readouts _µ_ ( _Z_<sup>(0)</sup> ) and the variance reduction term, which, from the proof of Theorem B.8, has the same convergence factor as the training recursion. Since the error is stated in terms of the maximum of the two, it suffices to show for one recursion, say, to obtain _µ_ ( _Z_<sup>(0)</sup> ). 

Without loss of generality, assume _∥µ_ ( _Z_<sup>(0)</sup> ) _∥_ 2 = 1. Let _cn_ := cond( _D_<sup>_−_1</sup> ( _G_ + _σ_<sup>2</sup> _In_ )). The convergence factor is, under the assumption of optimal step size, given by _ρ_ = 1 _−_ 2 _/_ ( _cn_ +2) (see Section A). Let _u_<sup>(</sup><sup>_L_)</sup> := TF<sup>pr</sup> _ϑ,L_<sup>(</sup><sup>_Z_(0)) and</sup><sup>_u∗_:=</sup><sup>_µ_(</sup><sup>_Z_(0)).</sup> Following Section A, suppose we have _ϵ >_ 0 such that 



Rearranging and noting that _ρ ∈_ (0 _,_ 1), 



where we used the fact that log(1 _− x_ ) _≤−x_ around 0. The rest follows from the fact that _cn_ = Θ( _n_ ) almost surely for large _n_ (see Lemma B.6). 

# **D. Details of Experiments** 

**Default hyperparameter choice.** The effects of lengthscale _ℓ_ and variance _σ_<sup>2</sup> are both-sided. For _ℓ_ , a smaller _ℓ_ makes the GP rougher and more localized, making predictions harder, especially in higher _d_ . However, it makes the associated linear system easier to solve. On the other hand, a larger _ℓ_ makes the GP smoother but can worsen conditioning due to correlations. Larger _σ_ makes the prediction harder but improves the conditioning via regularization. 

24 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

Throughout our experiments, we fix the noise variance at _σ_<sup>2</sup> = 0 _._ 2. For the BLR task with linear kernel, we use the diagonal covariance Σ is segmented by input dimension _d_ : 



For the RBF task with RBF kernel, we set the amplitude at _α_ = 1 and bandwidth _ℓ_ = 0 _._ 8. Additionally, for _theory_ mode for RBF task, we allow the step sizes of the drift term and the data residual terms (computed by the attention) to be different. 

**MLP head.** To isolate the effect of the attention layers, we fix the conversion module _ψ_ ( _µ, τ_ ) by hard-coding the analytic map from predicted moments ( _µ, τ_ ) to the natural parameters of a Gaussian density. In _learnable_ mode, we additionally introduce two learned scalar rescaling _s_ 1 and _s_ 2 applied to the outputs of _ψ_ ( _µ, τ_ ). We also fix the linear readout matrix Ξ to its theoretical value. 

**Pretraining details.** We optimize our models using the Adam optimizer with a base learning rate of LRbase and no weight decay. We employ a cosine learning rate decay schedule with a linear warmup phase. The warmup period lasts for the first 5% of the total training steps, after which the learning rate follows a cosine decay to a minimum value of 0 _._ 1 _×_ LRbase. Gradients are clipped at a global norm of 1 _._ 0 to ensure stability. The pretraining hyperparameters are summarized in Table 1. All the metrics in the figures are evaluated on 4096 evaluation samples. 

|**Task**|**Figure**|LRbase|_N_|
|---|---|---|---|
|BLR (theory), trainTF_ϑ,L_|Fig.2|2_×_10<sup>_−_4</sup><br>|10<sup>4</sup><br>|
|BLR (learnable), trainTF_ϑ,L_|Fig.2|2_×_10<sup>_−_4</sup><br>|2_×_10<sup>4</sup><br>|
|RBF (theory), trainTF_ϑ,L_|Fig.2|1_×_10<sup>_−_3</sup>|5_×_10<sup>4</sup>|
|RBF (learnable), trainTF_ϑ,L_|Fig.2|2_×_10<sup>_−_4</sup>|10<sup>5</sup>|
|RBF (learnable), trainTF_ϑ,L_|Fig.4|2_×_10<sup>_−_4</sup><br>|10<sup>5</sup>|
|RBF (learnable), trainTF<sup>pr</sup><br>_ϑ,L_|Fig.4|2_×_10<sup>_−_4</sup>|10<sup>5</sup>|
|RBF (learnable), trainTF<sup>pr</sup><br>_ϑ,L_|Fig.5-14|2_×_10<sup>_−_4</sup>|10<sup>5</sup>|



_Table 1._ Pretraining hyperparameters for BLR and RBF tasks. All runs use batch size 128 for _N_ iterations. 

# **E. Additional Experimental Results** 

**Sensitivity to lengthscale.** We conducted additional simulations to assess the sensitivity of our findings to the RBF lengthscale. We fix _σ_<sup>2</sup> = 0 _._ 2 as in Section 6.3, set _n_ max = 256 and _d_ = 16, and vary _ℓ ∈{_ 0 _._ 4 _,_ 0 _._ 8 _,_ 1 _._ 6 _}_ , including the main-paper setting _ℓ_ = 0 _._ 8 for comparison. In addition to interval coverage, we report the continuous ranked probability score (CRPS), a proper scoring rule for predictive cumulative distribution functions (Gneiting & Raftery, 2007); lower CRPS indicates better predictive distributions, although its absolute scale depends on the scale of the response. For each row, entries are ordered by attention depth _L ∈{_ 8 _,_ 16 _,_ 32 _}_ . In this setting, CRPS decreases as _ℓ_ increases, suggesting that 

|_ℓ_|_n_<sup>_′_</sup>|CRPS|50%|90%|95%|
|---|---|---|---|---|---|
|0_._4|256|_._538_, ._547_, ._539|_._504_, ._487_, ._505<br>_._904_, _|_._899_, ._904|_._953_, ._946_, ._953|
||1024|_._540_, ._540_, ._538|_._495_, ._485_, ._499<br>_._896_, _|_._887_, ._894|_._947_, ._943_, ._947|
|0_._8|256|_._318_, ._310_, ._307|_._488_, ._499_, ._509<br>_._890_, _|_._901_, ._905|_._942_, ._951_, ._954|
||1024|_._271_, ._262_, ._258|_._444_, ._478_, ._505<br>_._848_, _|_._877_, ._897|_._908_, ._932_, ._950|
|1_._6|256|_._157_, ._151_, ._149|_._500_, ._498_, ._506<br>_._901_, _|_._900_, ._908|_._950_, ._948_, ._952|
||1024|_._147_, ._137_, ._135|_._481_, ._479_, ._503<br>_._883_, _|_._884_, ._908|_._938_, ._943_, ._952|



_Table 2._ Sensitivity to RBF lengthscale _ℓ_ with _d_ = 16, _σ_<sup>2</sup> = 0 _._ 2, _n_ max = 256, and _L ∈{_ 8 _,_ 16 _,_ 32 _}_ . Entries in each metric column are ordered by _L_ = 8 _,_ 16 _,_ 32. CRPS denotes the continuous ranked probability score; lower is better. 

25 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 

smoother GP draws reduce the statistical difficulty of the prediction problem. Across all lengthscales, increasing depth generally improves CRPS, with the clearest improvement at _ℓ_ = 0 _._ 8. The coverage values remain reasonably close to nominal even at _n_<sup>_′_</sup> = 1024 _> n_ max, especially for larger depths. 

**Hierarchical GP prior.** We also evaluate a hierarchical GP setting with random hyperparameters. Let _θ ∼_<sup>�</sup><sup>_H_</sup> _h_ =1<sup>_πhδθ_</sup> _h_ iid for _θh_ = ( _ℓh, σh_ ). Conditional on _θ_ = _θh_ , _f ∼_ GP(0 _, κℓh_ ), _yi_ = _f_ ( _xi_ ) + _ϵi_ where _ϵi ∼N_ (0 _, σh_<sup>2).In this setting, the</sup> PPD is tractable as a finite mixture of Gaussian predictive distributions. For the experiment, we use the uniform prior over Θ = _{_ 0 _._ 4 _,_ 0 _._ 8 _,_ 1 _._ 2 _} × {_ 0 _._ 1 _,_ 0 _._ 2 _,_ 0 _._ 3 _}_ . We set _n_ max = 256, _d_ = 16, and otherwise match the setup of Section 6.3. In each cell, entries are ordered by attention depth _L ∈{_ 8 _,_ 16 _,_ 32 _}_ . 

|_n_<sup>_′_</sup><br>|CRPS|50%|90%|95%|
|---|---|---|---|---|
|256<br>_._380_, _|_._378_, ._377|_._573_, ._575_, ._590|_._895_, ._897_, _|_._901<br>_._937_, ._937_, ._942|
|512<br>_._374_, _|_._371_, ._367|_._570_, ._568_, ._590|_._889_, ._891_, _|_._903<br>_._926_, ._925_, ._932|
|1024<br>_._358_, _|_._353_, ._346|_._560_, ._563_, ._598|_._887_, ._887_, _|_._902<br>_._930_, ._929_, ._940|



_Table 3._ Hierarchical GP experiment with random hyperparameters _θ_ = ( _ℓ, σ_ ) _∈{_ 0 _._ 4 _,_ 0 _._ 8 _,_ 1 _._ 2 _} × {_ 0 _._ 1 _,_ 0 _._ 2 _,_ 0 _._ 3 _}_ , _d_ = 16, _n_ max = 256, and _L ∈{_ 8 _,_ 16 _,_ 32 _}_ . Entries in each metric column are ordered by _L_ = 8 _,_ 16 _,_ 32. 

The results are consistent with the main findings. CRPS improves with depth _L_ , while coverage remains reasonably calibrated for _n_<sup>_′_</sup> _> n_ max, especially at the 90% and 95% levels. This suggests that the depth effect and normalized-attention mechanism identified in the fixed-hyperparameter GP setting remain relevant in this more complex hierarchical setting. 

**Additional generalization results.** We repeat the generalization experiment from Figure 5 and 10 for dimensions _d ∈{_ 4 _,_ 8 _}_ , with results shown in Figure 6-9. Across dimensions, the same pattern emerges: generalization improves with deeper _L_ and larger _n_ max, except in the case _d_ = 8, where the _L_ = 16 model exhibits higher error than shallower counterparts, likely due to a training failure. 

**Covariate shifts.** We replicate the generalization experiment under a covariate shift, where the evaluation inputs are drawn iid from _xi ∼_ Unif[ _−_ 1 _/√d,_ 1 _/√d_ ] for _d ∈{_ 4 _,_ 8 _}_ The results shown in Figure 11-14. Despite the shift in covariates, we observe that the generalization performance is comparable to the true baseline for _n_ max = 512 and _L ≥_ 16, with _L_ = 16 model at _d_ = 8 being an exception. 

# **F. Details of Real Data Case Studies** 

We present a real data case study on the Sacramento home price dataset (Kuhn, 2008) and the Walker Lake dataset (Isaaks & Srivastava, 1989). Note that this case study is intended as an illustrative validation of the proposed mechanism rather than as a comprehensive benchmark. 

**Sacramento.** The dataset is from the R package caret version 7.0.1 (Kuhn, 2008), from which we selected data corresponding to the cities of Sacramento and Elk Grove. We use the two spatial coordinates, longitude and latitude, as input features and denote the response by _V_ . The spatial coordinates are centered and standardized coordinate-wise, and then divided by a constant 0 _._ 3; the response is centered and standardized with the scaling constant 1. This preprocessing is used consistently for empirical-Bayes fitting, PFN pretraining, and evaluation. Following Rasmussen & Williams (2005, Chapter 5), we first fit an empirical-Bayes anisotropic RBF GP to the Sacramento data. Specifically, we use the ARD kernel 



together with Gaussian observation noise. The fitted hyperparameters are _α_ = 0 _._ 666, _ℓ_ = (1 _._ 226 _,_ 0 _._ 770) and _σ_ = 0 _._ 836. Using these hyperparameters, we pretrain the normalized transformer TF<sup>pr</sup> _ϑ,L_<sup>on synthetic data generated from the fitted</sup> prior. We use _C_ = 256 bins, depth _L_ = 32, and sample pretraining context sizes from _n_<sup>_′_</sup> _∼_ Unif[128 _,_ 552]. The truncation interval ( _a, b_ ] for the binned output distribution is chosen by Monte Carlo calibration under the fitted prior, as in the synthetic experiments. 

In Figure 1, we provide the trained PFN with the full Sacramento context and evaluate its PPD on a regular 100 _×_ 100 spatial grid over the region [ _−_ 121 _._ 53 _, −_ 121 _._ 33] _×_ [38 _._ 38 _,_ 38 _._ 69]. From the PFN output probabilities, we compute the predictive 

26 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 



<!-- Start of picture text -->
nmax = 128 nmax = 256 nmax = 512<br>1.5 L=4 1.5 1.5<br>L=8<br>L=16<br>2.0 L=32 2.0 2.0<br>2.5 2.5 2.5<br>0.925 0.925 0.925<br>0.900 0.900 0.900<br>0.875 True PPD 0.875 0.875<br>0.850 0.850 0.850<br>0.825 0.825 0.825<br>True PPD<br>1.4 1.4 1.4<br>1.2 1.2 1.2<br>1.0 1.0 1.0<br>0.8 0.8 0.8<br>200 400 600 800 1000 200 400 600 800 1000 200 400 600 800 1000<br>Eval Sample Size (n) Eval Sample Size (n) Eval Sample Size (n)<br>y<br>Log MSE<br>90% Coverage<br>90% Width<br><!-- End of picture text -->

_Figure 6._ ( _d_ = 4) Prediction MSE, 90% interval coverage, and 90% interval width versus evaluation sample size for _learnable_ normalized models with _C_ = 256, depths _L ∈{_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ , and pretraining ranges _n ∈_ [64 _, n_ max] with _n_ max _∈{_ 128 _,_ 256 _,_ 512 _}_ . Red dashed curves denote the corresponding true PPD interval width/nominal coverage. 

mean and the 5% and 95% predictive quantiles. As a baseline, we fit an empirical-Bayes GP to the same standardized context and evaluate its predictive mean and Gaussian predictive quantiles on the same grid. Finally, all predictions are transformed back to the original response scale. 

**Walker Lake.** The dataset is from the R package gstat version 2.1-6 (Pebesma et al., 2015), whose spatial coordinates and response (V) are standardized in the same manner as Sacramento data. We estimate anisotropic RBF hyperparameters from a randomly sampled subset of the training data with _n_ = 200 by maximizing the marginal log likelihood, obtaining _α_ = 0 _._ 782, _ℓ_ 1 = 0 _._ 146, _ℓ_ 2 = 0 _._ 252, and _σ_ = 0 _._ 611. We then pretrain TF<sup>pr</sup> _ϑ,L_<sup>with</sup><sup>_C_= 256 and</sup><sup>_L_= 32 on synthetic data</sup> from the fitted prior with context sizes _n_<sup>_′_</sup> _∼_ Unif[128 _,_ 384]. Given a separate context of size _n_ = 200, the PFN PPD is evaluated on the full spatial grid and compared with an empirical-Bayes GP fitted to the same context. The resulting PFN and GP predictive means and 5%/95% quantiles show qualitatively similar spatial patterns (Figure 15). 

27 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 



<!-- Start of picture text -->
nmax = 128 nmax = 256 nmax = 512<br>1.75 1.75 1.75<br>2.00 2.00 2.00<br>2.25 L=4 2.25 2.25<br>L=8<br>2.50 L=16 2.50 2.50<br>L=32<br>2.75 2.75 2.75<br>3 3 3<br>4 4 4<br>5 5 5<br>1 1 1<br>2 2 2<br>3 3 3<br>200 400 600 800 1000 200 400 600 800 1000 200 400 600 800 1000<br>Eval Sample Size (n) Eval Sample Size (n) Eval Sample Size (n)<br>Log TV<br>Log MSE<br>2]<br>Y<br>Log MSE [<br><!-- End of picture text -->

_Figure 7._ ( _d_ = 4) log E[TV( _p_ ( _a,b_ ] _, qϑ_ )] and moment MSEs E( _µ − mϑ,_ 1)<sup>2</sup> and E( _τ_ + _µ_<sup>2</sup> _− mϑ,_ 2)<sup>2</sup> versus evaluation sample size _n_<sup>_′_</sup> for _learnable_ normalized models with _C_ = 256, depths _L ∈{_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ , and pretraining ranges _n ∈_ [64 _, n_ max] with _n_ max _∈{_ 128 _,_ 256 _,_ 512 _}_ . 



<!-- Start of picture text -->
nmax = 128 nmax = 256 nmax = 512<br>L=4<br>1.0 L=8 1.0 1.0<br>L=16<br>L=32<br>1.5 1.5 1.5<br>2.0 2.0 2.0<br>True PPD<br>0.90 0.90 0.90<br>0.85 0.85 0.85<br>0.80 0.80 0.80<br>True PPD<br>2.0 2.0 2.0<br>1.5 1.5 1.5<br>1.0 1.0 1.0<br>200 400 600 800 1000 200 400 600 800 1000 200 400 600 800 1000<br>Eval Sample Size (n) Eval Sample Size (n) Eval Sample Size (n)<br>y<br>Log MSE<br>90% Coverage<br>90% Width<br><!-- End of picture text -->

_Figure 8._ ( _d_ = 8) Prediction MSE, 90% interval coverage, and 90% interval width versus evaluation sample size for _learnable_ normalized models with _C_ = 256, depths _L ∈{_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ , and pretraining ranges _n ∈_ [64 _, n_ max] with _n_ max _∈{_ 128 _,_ 256 _,_ 512 _}_ . Red dashed curves denote the corresponding true PPD interval width/nominal coverage. 

28 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 



<!-- Start of picture text -->
nmax = 128 nmax = 256 nmax = 512<br>1.0 1.0 1.0<br>1.5 1.5 1.5<br>2.0 L=4 2.0 2.0<br>L=8<br>2.5 L=16 2.5 2.5<br>L=32<br>2.0 2.0 2.0<br>2.5 2.5 2.5<br>3.0 3.0 3.0<br>3.5 3.5 3.5<br>4.0 4.0 4.0<br>4.5 4.5 4.5<br>1 1 1<br>2 2 2<br>3 3 3<br>200 400 600 800 1000 200 400 600 800 1000 200 400 600 800 1000<br>Eval Sample Size (n) Eval Sample Size (n) Eval Sample Size (n)<br>Log TV<br>Log MSE<br>2]<br>Y<br>Log MSE [<br><!-- End of picture text -->

_Figure 9._ ( _d_ = 8) log E[TV( _p_ ( _a,b_ ] _, qϑ_ )] and moment MSEs E( _µ − mϑ,_ 1)<sup>2</sup> and E( _τ_ + _µ_<sup>2</sup> _− mϑ,_ 2)<sup>2</sup> versus evaluation sample size _n_<sup>_′_</sup> for _learnable_ normalized models with _C_ = 256, depths _L ∈{_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ , and pretraining ranges _n ∈_ [64 _, n_ max] with _n_ max _∈{_ 128 _,_ 256 _,_ 512 _}_ . 



<!-- Start of picture text -->
nmax = 128 nmax = 256 nmax = 512<br>1.5 1.5 1.5<br>2.0 2.0 2.0<br>2.5 L=4 2.5 2.5<br>L=8<br>3.0 L=16 3.0 3.0<br>L=32<br>2 2 2<br>3 3 3<br>4 4 4<br>5 5 5<br>1 1 1<br>2 2 2<br>3 3 3<br>4 4 4<br>200 400 600 800 1000 200 400 600 800 1000 200 400 600 800 1000<br>Eval Sample Size (n) Eval Sample Size (n) Eval Sample Size (n)<br>Log TV<br>Log MSE<br>2]<br>Y<br>Log MSE [<br><!-- End of picture text -->

_Figure 10._ ( _d_ = 16) log E[TV( _p_ ( _a,b_ ] _, qϑ_ )] and moment MSEs E( _µ − mϑ,_ 1)<sup>2</sup> and E( _τ_ + _µ_<sup>2</sup> _− mϑ,_ 2)<sup>2</sup> versus evaluation sample size _n_<sup>_′_</sup> for _learnable_ normalized models with _C_ = 256, depths _L ∈{_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ , and pretraining ranges _n ∈_ [64 _, n_ max] with _n_ max _∈{_ 128 _,_ 256 _,_ 512 _}_ . Errors decrease with larger _L_ and _n_ max and grow with _n_<sup>_′_</sup> , mirroring the trends in prediction and interval metrics in Figure 5. Consistent with Theorem 5.3, for fixed _L_ and _n_ max, the error of the finite-iteration solver increases as the evaluation sample size grows. 

29 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 



<!-- Start of picture text -->
nmax = 128 nmax = 256 nmax = 512<br>2.6 L=4 2.6 2.6<br>2.7 L=8 2.7 2.7<br>L=16<br>2.8 L=32 2.8 2.8<br>2.9 2.9 2.9<br>3.0 3.0 3.0<br>3.1 3.1 3.1<br>0.90 0.90 0.90<br>0.85 True PPD 0.85 0.85<br>0.80 0.80 0.80<br>True PPD<br>0.8 0.8 0.8<br>0.7 0.7 0.7<br>0.6 0.6 0.6<br>200 400 600 800 1000 200 400 600 800 1000 200 400 600 800 1000<br>Eval Sample Size (n) Eval Sample Size (n) Eval Sample Size (n)<br>y<br>Log MSE<br>90% Coverage<br>90% Width<br><!-- End of picture text -->

_Figure 11._ ( _d_ = 4, covariate shift) Prediction MSE, 90% interval coverage, and 90% interval width versus evaluation sample size for _learnable_ normalized models with _C_ = 256, depths _L ∈{_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ , and pretraining ranges _n ∈_ [64 _, n_ max] with _n_ max _∈ {_ 128 _,_ 256 _,_ 512 _}_ . Red dashed curves denote the corresponding true PPD interval width/nominal coverage. 



<!-- Start of picture text -->
nmax = 128 nmax = 256 nmax = 512<br>1.75 1.75 1.75<br>2.00 2.00 2.00<br>2.25 2.25 2.25<br>2.50 L=4 2.50 2.50<br>L=8<br>2.75 L=16 2.75 2.75<br>L=32<br>3.00 3.00 3.00<br>4.0 4.0 4.0<br>4.5 4.5 4.5<br>5.0 5.0 5.0<br>5.5 5.5 5.5<br>6.0 6.0 6.0<br>2.0 2.0 2.0<br>2.5 2.5 2.5<br>3.0 3.0 3.0<br>3.5 3.5 3.5<br>200 400 600 800 1000 200 400 600 800 1000 200 400 600 800 1000<br>Eval Sample Size (n) Eval Sample Size (n) Eval Sample Size (n)<br>Log TV<br>Log MSE<br>2] Y<br>Log MSE [<br><!-- End of picture text -->

_Figure 12._ ( _d_ = 4, covariate shift) log E[TV( _p_ ( _a,b_ ] _, qϑ_ )] and moment MSEs E( _µ − mϑ,_ 1)<sup>2</sup> and E( _τ_ + _µ_<sup>2</sup> _− mϑ,_ 2)<sup>2</sup> versus evaluation sample size _n_<sup>_′_</sup> for _learnable_ normalized models with _C_ = 256, depths _L ∈{_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ , and pretraining ranges _n ∈_ [64 _, n_ max] with _n_ max _∈{_ 128 _,_ 256 _,_ 512 _}_ . 

30 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 



<!-- Start of picture text -->
nmax = 128 nmax = 256 nmax = 512<br>1.75 L=4 1.75 1.75<br>L=8<br>2.00 L=16 2.00 2.00<br>L=32<br>2.25 2.25 2.25<br>2.50 2.50 2.50<br>2.75 2.75 2.75<br>0.90 0.90 0.90<br>0.85 0.85 0.85<br>True PPD<br>0.80 0.80 0.80<br>0.75 0.75 0.75<br>1.4 1.4 1.4<br>True PPD<br>1.2 1.2 1.2<br>1.0 1.0 1.0<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>200 400 600 800 1000 200 400 600 800 1000 200 400 600 800 1000<br>Eval Sample Size (n) Eval Sample Size (n) Eval Sample Size (n)<br>y<br>Log MSE<br>90% Coverage<br>90% Width<br><!-- End of picture text -->

_Figure 13._ ( _d_ = 8, covariate shift) Prediction MSE, 90% interval coverage, and 90% interval width versus evaluation sample size for _learnable_ normalized models with _C_ = 256, depths _L ∈{_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ , and pretraining ranges _n ∈_ [64 _, n_ max] with _n_ max _∈ {_ 128 _,_ 256 _,_ 512 _}_ . Red dashed curves denote the corresponding true PPD interval width/nominal coverage. 



<!-- Start of picture text -->
nmax = 128 nmax = 256 nmax = 512<br>1.0 1.0 1.0<br>1.5 1.5 1.5<br>2.0 L=4 2.0 2.0<br>L=8<br>L=16<br>2.5 2.5 2.5<br>L=32<br>3 3 3<br>4 4 4<br>5 5 5<br>1.0 1.0 1.0<br>1.5 1.5 1.5<br>2.0 2.0 2.0<br>2.5 2.5 2.5<br>3.0 3.0 3.0<br>3.5 3.5 3.5<br>200 400 600 800 1000 200 400 600 800 1000 200 400 600 800 1000<br>Eval Sample Size (n) Eval Sample Size (n) Eval Sample Size (n)<br>Log TV<br>Log MSE<br>2]<br>Y<br>Log MSE [<br><!-- End of picture text -->

_Figure 14._ ( _d_ = 8, covariate shift) log E[TV( _p_ ( _a,b_ ] _, qϑ_ )] and moment MSEs E( _µ − mϑ,_ 1)<sup>2</sup> and E( _τ_ + _µ_<sup>2</sup> _− mϑ,_ 2)<sup>2</sup> versus evaluation sample size _n_<sup>_′_</sup> for _learnable_ normalized models with _C_ = 256, depths _L ∈{_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ , and pretraining ranges _n ∈_ [64 _, n_ max] with _n_ max _∈{_ 128 _,_ 256 _,_ 512 _}_ . 

31 

**Transformers Can Learn Posterior Predictive Distributions In-Context** 



<!-- Start of picture text -->
PFN PPD Mean PFN PPD Lower 5% PFN PPD Upper 95%<br>1.0 1.0 1.0<br>0.5 0.5 0.5<br>0.0 0.0 0.0 700<br>0.5 0.5 0.5 600<br>1.0 1.0 1.0<br>500<br>1 0 1 1 0 1 1 0 1<br>400<br>GP PPD Mean GP PPD Lower 5% GP PPD Upper 95%<br>300<br>1.0 1.0 1.0<br>200<br>0.5 0.5 0.5<br>100<br>0.0 0.0 0.0<br>0.5 0.5 0.5<br>1.0 1.0 1.0<br>1 0 1 1 0 1 1 0 1<br>Mineral Grade (ppm)<br><!-- End of picture text -->

_Figure 15._ Illustration of similarities between PPDs produced by a transformer (PFN) and a Gaussian process (GP) on the Walker Lake dataset (Isaaks & Srivastava, 1989). The _x_ , _y_ axes represent standardized spatial coordinates, while the color scale shows estimated mineral (V) concentrations in ppm. The top row displays PPD outputs from a transformer, while the bottom row shows PPD based on a GP. Columns correspond to the PPD mean (left), 5% quantile (center), 95% quantile (right) of PPD; both PPDs are based on the same 200 observations shown in blue _×_ mark on the left column. 

32 

