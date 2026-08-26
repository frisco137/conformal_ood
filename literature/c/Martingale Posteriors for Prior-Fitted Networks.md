# **Uncertainty Quantification for Prior-Data Fitted Networks using Martingale Posteriors** 

**Thomas Nagler David Rügamer** 

Department of Statistics, LMU Munich Munich Center for Machine Learning (MCML) `{t.nagler,david.ruegamer}@lmu.de` 

## **Abstract** 

Prior-data fitted networks (PFNs) have emerged as promising foundation models for prediction from tabular datasets, achieving state-of-the-art performance on small to moderate data sizes without tuning. While PFNs are motivated by Bayesian ideas, they do not provide any uncertainty quantification for predictive means, quantiles, or similar quantities. We propose a principled, efficient, and tuning-free sampling procedure to construct Bayesian posteriors for such estimates based on martingale posteriors, and prove its convergence. Several simulated and real-world data examples showcase the efficiency and calibration of our method in inference applications. 

## **1 Introduction** 

Prior-data fitted networks (PFNs) are foundation models (Hollmann et al., 2023; Müller et al., 2022) that allow for in-context learning, i.e., the ability to learn at inference time without any parameter updates (Garg et al., 2022). TabPFN, a transformer pre-trained on synthetic data for in-context learning on tabular datasets, has recently attracted a lot of interest. TabPFN (Hollmann et al., 2023, 2025) and related variants such as TuneTables (Feuer et al., 2024), LocalPFN (Thomas et al., 2024), or TabICL (Qu et al., 2025, 2026) have been shown to achieve state-of-the-art performance on tabular benchmarks by pre-training on purely synthetic data. Since PFNs and extensions learn in-context, there is no need for further model (fine-)tuning on the inference task. 

Recent extensions of PFNs allow their applicability to large datasets (Feuer et al., 2024), the use of PFN “priors” for latent variable models (Reuter et al., 2025), and simultaneously minimizing bias and variance to improve their performance (Liu and Ye, 2025). PFNs are also related to simulation-based inference and amortized inference, but have slightly different goals and do not amortize across a single but multiple datasets (Reuter et al., 2025). While introduced as a Bayesian method and approximation to the posterior predictive, PFNs can also be interpreted as pre-tuned untrained predictors (Nagler, 2023). This also relates to the question of what uncertainty PFN models can provide. 

PFNs approximate the posterior predictive distribution for the label given some feature values. Despite the name, this only yields point estimates of the most relevant predictive quantities, such as the conditional mean or quantiles. Due to the complex nature of PFNs, it is difficult to assess the uncertainty of these point estimates. This explains why, despite the practical relevance, methods for such uncertainty assessment are currently lacking. 

In this article, we propose a principled and efficient method to construct Bayesian posteriors for such estimates using the idea of _Martingale Posteriors_ (MPs; Fong et al., 2023). 

Preprint. 



<!-- Start of picture text -->
Optimal AMP with TabPFN AMP with TabICL<br>Coverage = 0.903; Length = 0.565 Coverage = 0.935; Length = 0.841 Coverage = 0.903; Length = 0.776<br>1<br>0<br>1<br>2 1 0 1 2 2 1 0 1 2 2 1 0 1 2<br>x1 x1 x1<br>True median Pred. median 90% CI<br>y<br><!-- End of picture text -->

Figure 1: Comparison of estimated median function (orange line) and credible intervals (shaded blue area) for a non-linear feature effect (dashed blue line) in a Bayesian additive model based on different methods (facets). Numbers below each method state the average coverage and interval length across all data points. “Optimal” corresponds to the posterior induced by the chosen prior and likelihood, which in this case can be analytically derived (see Section 4.1 for details). Our approach (AMP) can be defined for any PFN that allows access to the learned predictive distribution (here, TabPFN and TabICL). 

### **Our contributions** 

1. We introduce a formulation of the MP framework for inference of predictive quantities conditional on a specific feature value _x_ . 

2. We propose an efficient, nonparametric resampling scheme yielding an approximate posterior for the point estimates derived from a PFN, and prove its convergence. 

3. We adapt existing learning rate schedules and discuss the role of contraction rates in light of nonparametric PFN estimators. 

4. We illustrate the new method in several simulated and real-world data applications. 

5. We perform a variety of ablation studies providing insights into the driving factors of our proposal’s efficacy and failure modes of alternative methods. 

Our work provides an essential tool for principled inference with the increasingly popular PFN methods. Our approximate martingale posterior (AMP) algorithm provides an uncertainty quantification layer that aligns perfectly with the strengths of PFNs: a tuning-free method yielding well-calibrated credible intervals in a matter of seconds. 

**Related work** There are two works that are closely related to ours, but target different quantities. Ng et al. (2025) propose martingale posteriors for unconditional, rather than predictive summaries of the data. Fortini et al. (2026) propose a Gaussian approximation of the joint posterior of a finite number of event probabilities, but do not provide uncertainty estimates for, e.g., conditional means or quantiles considered in this article. Additionally, the corresponding algorithms require many repeated calls to the PFN, making them orders of magnitude slower than ours. 

## **2 Background** 

We consider a tabular prediction task with labels _y ∈_ R and features _x ∈_ R<sup>_d_</sup> drawn from a joint distribution _P_ . A typical problem in such tasks is to estimate predictive quantities such as conditional means E[ _y|x_ ], conditional probabilities _P_ ( _y|x_ ), or conditional quantiles _P_<sup>_−_1</sup> ( _α|x_ ). Because the true distribution _P_ is unknown and only a finite amount of data _Dn_ = ( _yi, xi_ )<sup>_n_</sup> _i_ =1<sup>is available, estimates</sup> of such quantities bear some uncertainty. Our goal is to quantify this uncertainty. 

### **2.1 Prior-data fitted networks** 

Prior-data fitted networks are foundation models trained to approximate the posterior predictive density PPD( _y|x_ ) = _p_ ( _y|x, Dn_ ), which quantifies the likelihood of observing label _y_ given that the feature is _x_ and _Dn_ has been observed. The PPD is a Bayesian concept and implicitly involves a prior 

2 

over the distributions _P_ that could have generated the data. To approximate the PPD with a PFN, a deep neural network—typically a transformer—is pre-trained on simulated datasets with diverse characteristics. After pre-training, the network weights are fixed, and the approximate PPD for a new training set can be computed through a single forward pass without additional training or tuning. 

The PPD quantifies uncertainty about the label _y_ . However, it mixes the aleatoric and epistemic components of uncertainty (Hüllermeier and Waegeman, 2021). From the PPD alone, it is impossible to disentangle these parts. Consequently, PFNs do _not_ provide uncertainty estimates for predictive summaries, such as conditional mean, probabilities, and quantiles. 

### **2.2 Bayesian inference** 

In classical Bayesian inference, the set of possible distributions _P_ = _Pθ_ is indexed by some parameter _θ_ . A _prior_ distribution _π_ ( _θ_ ) is elicited to quantify our beliefs about the likelihood of the possible values of _θ_ before seeing any data. After observing _Dn_ , this belief is updated to a _posterior π_ ( _θ|Dn_ ) of the parameter _θ_ given the data. For predictive inference, the PPD can be computed as 





for any _A ⊆_ R. PFNs neither provide an explicit model for _pθ_ nor an explicit prior _π_ ( _θ_ ), although both may be implicit in the PPD. The following shows how Bayesian posterior inference can be approached when only the PPD is available. 

### **2.3 Martingale posteriors** 

Martingale posteriors were recently introduced by Fong et al. (2023) as a new method for Bayesian uncertainty quantification. Its core idea is to reverse the direction of the Bayesian inference. In classical Bayesian inference, the posterior is derived from a prior and likelihood, which then implicitly leads to the PPD. MP inference starts from the PPD and leaves the prior _π_ ( _θ_ ) implicit. An appropriate sampling scheme and Doob’s theorem then allow us to derive posteriors for virtually all quantities of interest (e.g., the conditional mean _µ_ ( _x_ )). 

To simplify our outline of the approach, consider the case where there are no features, and we are interested in unconditional inference. An extension to our predictive inference setting will be made explicit in Section 3.1. Suppose we have observed data _y_ 1: _n_ = ( _y_ 1 _, . . . , yn_ ). 

The MP approach involves iteratively sampling 



_N_ times, which yields a sample _y_ ( _n_ +1):( _n_ + _N_ ) drawn from the predictive joint distribution 



Observe, however, that the samples are not independent. As a consequence, the long-run empirical distribution of the obtained sample, 



is a random function and comes out differently whenever the sampling procedure is repeated. Denote by Π( _F∞|Dn_ ) the distribution of this function (which depends on the data _Dn_ we start with). For any parameter _θ_ = _θ_ ( _P_ ) of interest, the martingale posterior is now given as 



where _A_ is any Borel set on the space where the parameter _θ_ lives. Furthermore, Doob’s theorem (Doob, 1949) implies that Π( _θ|Dn_ ) coincides with the classical Bayes posterior for the prior _π_ ( _θ_ ) implicit in the PPD (Fong et al., 2023). 

3 

## **3 Efficient martingale posteriors for prior-data fitted networks** 

Martingale posteriors allow for Bayesian inference directly from the PPD. PFNs approximate the PPD, so using PFNs to construct a martingale posterior seems natural. However, there are two problems. First, Falck et al. (2024) found that modern transformer-based models substantially deviate from the _martingale property_ 



Second, modern PFNs are based on transformer architectures that require Ω( _n_<sup>2</sup> ) operations per forward pass on a training set of size _n_ . Iteratively computing _p_ ( _y|y_ 1:( _n_ + _k_ )) for _k_ = 1 _, . . . , N_ thus has complexity Ω( _N_<sup>3</sup> ), which is prohibitive. 

We propose to use the PPD implied by the PFN only as a starting point for the MP sampling scheme. After obtaining the PFN’s PPD, we then iteratively update it using a nonparametric forward sampling scheme that ensures the martingale property. The resulting martingale posterior effectively treats the PFN output as a strong, informed prior, without requiring the PFN to provide coherent posterior updates or incurring computational overhead from iterative model evaluations. 

### **3.1 Martingale posteriors for conditional inference** 

We extend the unconditional sampling scheme outlined in the previous section to the conditional inference setting. Fong et al. (2023) already proposed one such extension. Their scheme involves forward sampling of the features _x_ ( _n_ +1):( _n_ + _N_ ). The distribution of the features isn’t of primary interest, but it complicates the sampling procedure and slows its convergence exponentially, making it impractical beyond about 5 feature dimensions. To alleviate this, we propose to sample only the labels _y_ ( _n_ +1):( _n_ + _N_ ) conditional on the event that _xn_ + _k_ = _x_ , for a fixed value of _x_ and all _k_ = 1 _, . . . , N_ . 

Set _xn_ + _k_ = _x_ for all _k ≥_ 1, and define 



and _Pk_ as the corresponding CDF. Applying Bayes’ rule recursively gives 



which suggests that we can iteratively sample 



Denote the long-run empirical distribution of the obtained sample by _F∞,x_ ( _y_ ) = lim _N →∞ N_<sup>_−_1 �</sup><sup>_N_</sup> _i_ =1<sup>**1**(</sup><sup>_yn_+</sup><sup>_i≤y_), which is again a random function, even in the limit.Repeating</sup> the iterative sampling procedure gives us its distribution Π( _F∞,x|Dn_ ). For any conditional parameter _θ_ ( _x_ ) = _θ_ ( _P_ ( _· |x_ )) of interest, the martingale posterior is now given as 



Common examples of the parameter _θ_ ( _x_ ) are the conditional mean _θ_ ( _x_ ) = � _y dP_ ( _y|x_ ) _dy_ or a conditional _φ_ -quantile _θ_ ( _x_ ) = _P_<sup>_−_1</sup> ( _φ|x_ ). 

### **3.2 Efficient PPD updates based on the Gaussian copula** 

Observe that _p_ 0( _y_ ) = _p_ ( _y|x, Dn_ ) is the PPD approximated by the PFN. However, the following update distributions _p_ 1 _, p_ 2 _, . . ._ are generally intractable. To alleviate this, we propose the following computationally efficient surrogate updates: 



where _Pk_ is the CDF corresponding to _pk_ , _αi_ a learning rate, and _Hρ_ ( _u, v_ ) = Φ((Φ<sup>_−_1</sup> ( _u_ ) _− ρ_ Φ<sup>_−_1</sup> ( _v_ )) _/_ �1 _− ρ_<sup>2</sup> ) _,_ with Φ the standard normal cumulative distribution function. 

4 

The updates have a similar form to those proposed in the unconditional context by Fong et al. (2023), who derived them a nonparametric density estimator based on Dirichlet Process Mixture Models (DPMMs) and a copula decomposition of the conditional _pk_ . We propose several modifications to their procedure in the following sections to make it amenable to PFN inference. 

The procedure has a hyperparameter _ρ_ , corresponding to a bandwidth that smoothes the updates. For _ρ →_ 1, the _Hρ_ -term converges to the CDF of a Dirac measure at _yn_ + _k_ . The update then amounts to sampling from _Pk−_ 1 and then mixing _Pk−_ 1 with this measure proportional to _an_ + _k_ . This shows similarities to the Bayesian bootstrap, although starting from a different _P_ 0. When _ρ <_ 1, the Dirac measure is smoothed on the scale of probability integral transforms. Fong et al. (2023) proposed to tune _ρ_ by maximizing the likelihood of the updated densities over the observed data. This is mainly to appropriately tune their initial estimate of the PPD. Since we use a different initial PPD, it is more appropriate to remain agnostic and choose _ρ ≈_ 1, and our experiments use _ρ_ = 0 _._ 99. 

Fong et al. (2023) suggested a learning rate _αi ∼_ ( _i_ + 1)<sup>_−_1</sup> for the unconditional context, where distribution functions can be estimated with convergence rate _n_<sup>_−_1</sup><sup>_/_2</sup> . As we shall see in Proposition 3.3, the learning rate _αi_ determines the contraction rate of the posterior. Our setting, however, is conditional on _x ∈_ R<sup>_d_</sup> , where nonparametric methods such as PFNs must converge more slowly, with rates depending on the dimension of the estimation problem (see, e.g., Stone, 1982). To accommodate such settings, we use a more general parameterization of the learning rate: 

_αi_ = 2<sup>_β_</sup> ( _i_ + 1)<sup>_−β_</sup> _, β >_ 1 _/_ 2 _._ (3) 

The factor 2<sup>_β_</sup> is chosen such that _αi_ = 1 if only a single observation has been observed ( _n_ = 1). The choice of _β_ is discussed in Section 3.4. 

### **3.3 Theoretical properties** 

Despite the simplicity of the updates given in (2), they provide essential theoretical guarantees. The following is a direct consequence of Theorem 3 by Fong et al. (2023). 

**Proposition 3.1.** _It holds_ ( _yN_ +1 _, yN_ +2 _, . . ._ ) _→d_ ( _z_ 1 _, z_ 2 _, . . ._ ) _as N →∞ where_ ( _z_ 1 _, z_ 2 _, . . ._ ) _has an exchangeable distribution._ 

By de Finetti’s theorem (e.g., Theorem 1.49 in Schervish, 2012), it then follows that there is a random variable Θ such that ( _z_ 1 _, z_ 2 _, . . ._ ) is conditionally _iid_ given Θ. The distribution of Θ can be interpreted as an implicit prior. In our setting, this prior depends both on the initial PPD implied by the PFN and the Gaussian copula updates specified by (2). In particular, the following result follows from Proposition 3.1 above and Theorem 2.2 of Berti et al. (2004). 

**Proposition 3.2.** _Suppose that P_ 0 _is absolutely continuous with respect to the Lebesgue measure. Then there exists a random probability distribution P∞,x such that_ lim _N →∞ PN_ ( _y_ ) = _P∞,x_ ( _y_ ) = _F∞,x_ ( _y_ ) _almost surely._ 

The proposition implies that the (random) limit _P∞,x_ is well defined and that the iterative sampling scheme is a valid way to draw from its distribution. We can be more precise: 

**Proposition 3.3.** _For any y ∈_ R _, β >_ 1 _/_ 2 _, the following holds with probability at least_ 1 _− δ:_ 



The proof, given in Section A, is based on a time-uniform version of the Azuma-Hoeffding concentration inequality (Howard et al., 2020), the martingale property of the copula updates, and the learning rate schedule. Importantly, time-uniform martingale concentration allows us to directly bound the deviation of interest _|P∞,x − P_ 0 _|_ as opposed to _|PM − P_ 0 _|_ for fixed _M < ∞_ as in Fong et al. (2023). 

Proposition 3.3 quantifies how much the random limit _P∞,x_ fluctuates around _P_ 0, our initial point estimate of the PPD. The spread of these fluctuations characterizes the spread of the martingale posterior for _θ_ ( _P_ ( _·|x_ )). Consequently, the contraction rate of the posterior is _n_<sup>_−β_+1</sup><sup>_/_2</sup> . 

### **3.4 The role of** _β_ 

The case _β_ = 1 in (3) corresponds to the choice _αi ∼_ ( _i_ + 1)<sup>_−_1</sup> proposed by Fong et al. (2023) in the unconditional context with contraction rate of _n_<sup>_−_1</sup><sup>_/_2</sup> . Contraction rates for nonparametric regression 

5 

**Algorithm 1** Approximate Martingale Posteriors (AMP) 

1: **Input:** Estimated PPD(<sup>�</sup> _y|x_ ) obtained from the PFN. 2: **for** _b_ = 1 _, . . . , B_ **do** 3: Initialize _P_ 0<sup>(</sup><sup>_b_)</sup> _←_ PPD(<sup>�</sup> _y|x_ ). 4: **for** _k_ = 1 _, . . . , N_ **do** 5: Sample _yn_<sup>(</sup><sup>_b_</sup> +<sup>)</sup> _k_<sup>_∼P_</sup> _k_<sup>(</sup> _−_<sup>_b_)</sup> 1<sup>.</sup> 6: Update ( _Pk_<sup>(</sup> _−_<sup>_b_)</sup> 1<sup>_, y_</sup> _n_<sup>(</sup><sup>_b_</sup> +<sup>)</sup> _k_<sup>)</sup><sup>_→P_</sup> _k_<sup>(</sup><sup>_b_)</sup> as in (2) with _αi_ as in (4). 7: **end for** 8: Compute _P_<sup>�</sup> _N_<sup>(</sup><sup>_b_)(</sup><sup>_y_) =</sup> _N_<sup><u>1</u></sup> � _Ni_ =1<sup>**1**</sup> � _yn_<sup>(</sup><sup>_b_</sup> +<sup>)</sup> _i_<sup>_≤y_</sup> �. _b_ ) 9: Set _θ_<sup>(</sup><sup>_b_)</sup> ( _x_ ) _← θ_ �� _P_ ( _N_ �. 10: **end for** 11: Compute the estimated Martingale Posterior: � _B_ Π _θ_ ( _x_ ) _∈ A|Dn_ = _B_<sup><u>1</u></sup> <u>�</u> _b_ =1<sup>**1**</sup> _θ_<sup>(</sup><sup>_b_)</sup> ( _x_ ) _∈ A_ . <u>� � � �</u> 

problems are typically much slower than _n_<sup>_−_1</sup><sup>_/_2</sup> (Ghosal and Van der Vaart, 2017, Chapter 9), however. To accommodate such slow contraction rates, it is therefore crucial to pick a learning rate with _β <_ 1. The choice _β_ = 1 _/_ 2 + 2 _/_ ( _d_ + 4) corresponds to the best possible contraction rate we can expect in a nonparametric regression problem with _d_ features and twice continuously differentiable regression curve. In nonparametric regression problems, there is, however, an unavoidable estimation bias that is also reflected in the posterior. To account for this bias and achieve asymptotically correct coverage, Bayesian credible intervals have to be inflated, typically using a slowly diverging blow-up factor (Szabó et al., 2015). This motivates a slightly smaller choice for _β_ , such as _β_ = 1 _/_ 2 + 2 _/_ (1 _._ 1 _d_ + 4) _,_ leading to slightly wider credible intervals. 

### **3.5 The role of initialization** 

The following corollary clarifies the role of the initialization in our algorithm. 

**Corollary 3.1.** _Let P_ 0 _,_ 1 _and P_ 0 _,_ 2 _be two PPDs to initialize our algorithm with, and P∞,x,_ 1 _and P∞,x,_ 2 _be the corresponding limit posteriors. For any x, y, it holds with probability at least_ 1 _−_ 2 _δ:_ 



This is a direct result of Proposition 3.3 and shows that, up to the contraction rate of the posteriors, the difference in martingale posteriors is as large as the difference in the initializations. Hence, the choice of initialization is the dominating factor for the difference in the martingale posteriors. Our approach, therefore, relies crucially on the quality and calibration of the initial PPD estimate, making full use of the excellent empirical performance of modern PFNs. It then adds an additional computational layer that allows for disentangling the epistemic uncertainty about a predictive summary (e.g., a conditional quantile) from the aleatoric uncertainty in the label _Y_ . 

### **3.6 Computation** 

The theoretical analysis above assumes that we run the update schemes indefinitely. In practice, we can only sample finite sequences, which incurs a truncation error. A closer inspection of the proof of Proposition 3.3 reveals that the posterior spread is determined by the square root of the quantity _Sn,N_<sup>2:= �</sup><sup>_n_</sup> _i_ =<sup>+</sup> _n_<sup>_N−_1</sup> _αi_<sup>2.For</sup><sup>_N< ∞_, the posterior spread is too small by the multiplicative factor</sup> 



Even for a relatively large number of forward samples _N_ , this can be detrimental. For example, with _n_ = 200, _N_ = 1000, and _d_ = 10, we get _Cn,N ≈_ 0 _._ 62 and the credible intervals are almost 40% too small. Knowing this, we can implement a simple fix by appropriately blowing up the intervals, achieved by multiplying the learning rate by _Cn,N_<sup>_−_1.Our proposed learning rate is therefore</sup> 



6 



<!-- Start of picture text -->
J = 1, d = 1 J = 5, d = 10 J = 10, d = 10 J = 10, d = 20 J = 20, d = 20<br>1.0<br>0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>200 400 600 800 200 400 600 800 200 400 600 800 200 400 600 800 200 400 600 800<br>J = 1, d = 1 J = 5, d = 10 J = 10, d = 10 J = 10, d = 20 J = 20, d = 20<br>4<br>3<br>2<br>1<br>200 400 600 800 200 400 600 800 200 400 600 800 200 400 600 800 200 400 600 800<br>Training set size<br>Method Analytic TabPFN + AMP TabICL + AMP Target quantile 0.5 0.9<br>Coverage<br>Interval width<br><!-- End of picture text -->

Figure 2: Coverage (target: 90%) and CI width for data simulated from a Bayesian additive model, comparing our approximate martingale posterior (AMP) credible sets for TabPFN and TabICL with analytic credible sets using the true DGP as prior. All widths are scaled by _d_<sup>_−_1</sup><sup>_/_2</sup> for better visibility. 

### The procedure is summarized in Algorithm 1. 

The run-time complexity of the algorithm is _O_ ( _BN_ ), where _B_ is the number of replications, and _N_ is the length of one ‘chain’. The computations are independent across the outer loop ( _b_ = 1 _, . . . , B_ ) and straightforward to parallelize or vectorize; the inner loop ( _k_ = 1 _, . . . , N_ ) must be run sequentially. Except for the initial PPD computation, the runtime is independent of the training set size _n_ . 

## **4 Numerical experiments** 

To demonstrate the efficacy of our approach, we conduct various experiments. Code is made available at `https://anonymous.4open.science/r/UQ_for_PFNs` . Our experiments use both TabPFN and TabICL to demonstrate the algorithm’s versatility. Unless stated otherwise, AMP uses _N_ = 50 forward samples and _B_ = 50 independent ‘chains’. 

### **4.1 Simulation study** 

We first assess the calibration of the algorithm in a simulation study using Bayesian additive model DGPs as ground-truth. We generate the true underlying feature effects _fj_ ( _xj_ ) = _B_ ( _xj_ )<sup>_⊤_</sup> _θj_ for feature _xj_ by drawing parameters _θj ∼N_ (0 _,_ 2 _I_ ) and using those parameters as spline coefficients for a B-spline basis _Bj_ ( _xj_ ) _∈_ R<sup>20</sup> of degree 3 with 20 basis functions. A training data set of size _n ∈{_ 50 _,_ 100 _,_ 200 _,_ 400 _,_ 800 _}_ is then generated by the law _Y |x, θ ∼N_ (<sup>�</sup><sup>_J_</sup> _j_ =1<sup>_fj_(</sup><sup>_xj_)</sup><sup>_,_1).We use</sup> _d ∈{_ 1 _,_ 10 _,_ 20 _}_ features from which _J ∈{⌈d/_ 2 _⌉, d}_ have a non-zero signal while the rest act as noise features with no contribution to _Y |x_ . Using the true DGP as a Bayesian prior, the resulting posterior is analytically available and can be used to construct credible intervals for conditional quantiles (referred to as _Optimal_ ). This allows for comparing the quality of epistemic uncertainty to a gold-standard baseline. For each ( _n, d, J_ ) combination, we simulate 20 data sets and evaluate all methods on 100 random test points. 

**Results** Figure 2 shows the results of our simulation experiment with average coverage in the top row and average interval width in the bottom row. First, we observe that all methods have approximately correct coverage, with the lowest points at around 80% for _n_ = 50 in some settings. For _J < d_ , the PFNs appear over-conservative for the median estimates. Our routine is unaware that some features are irrelevant, but the PFN may exploit this to yield better-than-expected point 

7 



<!-- Start of picture text -->
1.0 1.00<br>0.9 Training set size Type<br>0.75<br>50 Ours (TabPFN)<br>0.8<br>100 Initial PPD: scale by 0.8<br>0.50<br>0.7 200 Initial PPD: scale by 1.25<br>400 Schedule: beta = 1<br>0.25<br>0.6 800 Schedule: no blowup<br>0.5 0.00<br>25 50 75 100 200 400 600 800<br>Number of forward samples Training set size<br>Coverage Coverage<br><!-- End of picture text -->

Figure 3: Ablation studies using TabPFN on simulated data with _d_ = 10 _, J_ = 10. The left panel shows coverage for a given number of forward samples _N_ ; the right panel varies the initial PPD estimate and the learning rate schedule. 

estimates. The interval widths decrease with the training set size, as expected. The widths for the analytic method decrease much faster for _d >_ 1, because this method exploits full knowledge of additivity, irrelevant features, and the noise variance. The AMP method also decreases width with training set size, but appropriately adapts to the actual uncertainty in the PFN estimators. 

### **4.2 Ablation studies** 

In Figure 3, we assess the role of other factors affecting performance using TabPFN on simulated data with _d_ = 10 and _J_ = 10 as an example; extended results are given in Appendix Section B.1. 

**Number of forward samples** Our main experiments used _N_ = 50 forward samples as default. The left panel of Figure 3 shows the effect of this parameter on the coverage. The coverage appears to stabilize already at around _N_ = 20 forward samples thanks to the blow-up factor introduced in Section 3.6, so _N_ = 50 is already a conservative choice. For _d_ = 20, _N_ = 50 appears appropriate (see Section B.1). 

**Initialization** To corroborate the need for a well-calibrated initial PPD estimate, we compress and widen the PPD (relative to the conditional median) by factors of 0.8 and 1.25, respectively. In the right panel of Figure 3, we see a direct effect on coverage: the compressed initial PPD yields lower coverage, while the widened version yields higher coverage. 

**Learning rate schedule** The right panel of Figure 3 further shows two alternatives to our learning rate schedule (4). As expected from our theoretical arguments in Section 3.4 and Section 3.6, using _β_ = 1 rather than _β_ = 1 _/_ 2 + 2 _/_ (1 _._ 1 _d_ + 4) or removing the blow-up factor _Cn,N_<sup>_−_1from the schedule</sup> leads to severe undercoverage. 

### **4.3 Benchmarks on real data** 

We next assess our method on a suite of real-world data from the UCI repository (Dua and Graff, 2017); see Appendix B.2 for a description of the data sets. Since the true conditional quantile is not known on real data sets, we use the following surrogate procedure. We use 20% of the data for training and compute predicted quantiles along with credible intervals for the held-out test data. We then check whether the predicted quantile from a separate PFN trained and evaluated only on the test data falls within the intervals. Since the latter PFN is trained on a much larger data set, it serves as a reasonable ‘oracle‘ surrogate for the true conditional quantile. While the training set sizes are rather small, this provides an ideal benchmark for PFNs, which remain limited in their scalability to large datasets. 

We compare against a bootstrap approach as a frequentist natural baseline, as it also assesses epistemic uncertainty for PFN estimates without knowledge of the pre-training pipeline. The bootstrap draws samples with replacement from the training dataset, uses TabPFN to make quantile predictions on the 

8 

Table 1: Benchmark results for 90%-quantile regression on various UCI data sets, reporting averages over ten random 20-80 train-test splits with _±_ 2se. The target coverage is 90%. 

|||Metric|airfoil|boston|concrete|diabetes|energy|fsh|forest|real|
|---|---|---|---|---|---|---|---|---|---|---|
||FN|Coverage|0.79_±_0.01|0.87_±_0.04|0.86_±_0.02|0.94_±_0.04|0.89_±_0.01|0.89_±_0.03|0.99_±_0.01|0.88_±_0.03|
||bP|Width|0.56_±_0.03|1.01_±_0.06|0.88_±_0.02|2.31_±_0.10|0.15_±_0.01|1.61_±_0.09|2.37_±_0.09|2.15_±_0.35|
|MP|Ta|Time (s)|20.5_±_0.2|7.1_±_0.8|13.7_±_1.1|5.7_±_0.4|9.7_±_0.6|12.6_±_0.8|7.6_±_0.6|5.1_±_0.1|
|A|CL|Coverage|0.72_±_0.01|0.89_±_0.03|0.85_±_0.02|0.89_±_0.05|0.84_±_0.02|0.91_±_0.03|0.86_±_0.05|0.87_±_0.03|
||TabI|Width<br>Time (s)|0.55_±_0.02<br>20.6_±_0.2|1.03_±_0.06<br>6.3_±_0.4|0.82_±_0.03<br>13.8_±_1.0|2.26_±_0.08<br>5.8_±_0.3|0.13_±_0.01<br>10.1_±_0.9|1.66_±_0.09<br>13.3_±_1.1|2.90_±_0.11<br>7.1_±_0.5|2.43_±_0.29<br>5.0_±_0.0|
||FN|Coverage|0.60_±_0.02|0.71_±_0.03|0.57_±_0.01|0.85_±_0.04|0.77_±_0.02|0.70_±_0.03|0.90_±_0.05|0.70_±_0.03|
|p|bP|Width|0.57_±_0.03|0.77_±_0.07|0.63_±_0.02|1.12_±_0.07|0.17_±_0.01|0.98_±_0.07|1.75_±_0.13|0.59_±_0.06|
|stra|Ta|Time (s)|76.3_±_30.1|17.4_±_5.9|53.0_±_35.3|41.9_±_23.9|67.0_±_25.1|53.4_±_14.6|33.7_±_8.5|10.3_±_1.1|
|oot|CL|Coverage|0.50_±_0.02|0.45_±_0.03|0.43_±_0.02|0.81_±_0.04|0.80_±_0.02|0.40_±_0.04|0.72_±_0.09|0.55_±_0.04|
|B|bI|Width|0.57_±_0.03|0.69_±_0.07|0.55_±_0.01|0.84_±_0.04|0.10_±_0.01|0.69_±_0.05|1.21_±_0.12|0.52_±_0.05|
||Ta|Time (s)|70.0_±_30.1|65.7_±_29.8|54.7_±_26.9|41.7_±_13.1|39.0_±_17.2|38.2_±_19.9|22.5_±_5.2|13.5_±_0.2|



test set, and computes an empirical confidence interval across _B_ = 50 replications. In all cases, we construct a 90% credible/confidence interval (CI). 

**Results** Table 1 shows the results of our coverage benchmark for predicting the conditional 90%quantile; results for median regression are given in Appendix B.3. Our AMP method provides approximately correct coverage across most datasets, though it can be slightly overconservative for TabPFN and conditional medians. The bootstrap produces narrower intervals and undercovers in most cases for the same reason that requires inflating credible sets: it ignores the unavoidable estimation bias in nonparametric regression problems (Hall and Horowitz, 2013). Notably, the bootstrap is several times slower than our AMP method. While AMP fits the PFN only once on the training data, the bootstrap requires _B_ refits, each costing _O_ ( _n_<sup>2</sup> ). The AMP method computes CIs for all test points in a few seconds, making it an ideal companion for modern PFNs. 

## **5 Discussion** 

This work proposes an efficient and principled Bayesian uncertainty quantification method for estimates derived from prior-data fitted networks. The method provides posterior uncertainty for predictive summaries, such as conditional means or quantiles, which is not available from the PFN posterior predictive distribution alone. Compared with more classical approaches such as deep ensembles (Lakshminarayanan et al., 2017), the proposed procedure is particularly well suited to pretrained models: it requires only one PFN evaluation on the training data and does not need access or modifications of the pre-training pipeline. By providing tuning-free, well-calibrated credible intervals in seconds, it serves as an ideal uncertainty layer playing to the strengths of modern PFNs. 

**Limitations** Our theory and experiments currently focus on iid tabular prediction problems. Although the same martingale posterior idea may be applicable beyond this setting, extensions to non-tabular or dependent data require additional assumptions and analysis. A second limitation is that the current implementation computes posterior credible intervals for each query point separately. Thus, the computational cost scales with the size of the test set, although the method is sufficiently fast in our experiments to evaluate hundreds of query points within seconds. 

The pointwise nature of the current procedure also means that it does not provide uniform credible sets for joint posterior inference. A full joint posterior would require modeling the dependence across feature values, so the distribution of _x_ 1: _n_ can no longer be ignored. Fong et al. (2023) proposed a general joint update of the PPDs over all values of _x_ , but this update is intractable in practice, and the proposed heuristic simplifications are neither especially simple nor theoretically justified. A possible direction is to combine the marginal posteriors _P∞,x_ 1 _, . . . , P∞,xK_ through a dependence model, for example, a multivariate Gaussian copula with a covariance kernel depending on distances between query points, or the more flexible vine-copula construction proposed by Huk et al. (2024). Developing such joint posterior extensions is a promising direction for future work. 

9 

## **References** 

- Berti, P., Pratelli, L., and Rigo, P. (2004). Limit theorems for a class of identically distributed random variables. _The Annals of Probability_ , 32(3):2029 – 2052. 

- Doob, J. L. (1949). Application of the theory of martingales. _Le calcul des probabilites et ses applications_ , pages 23–27. 

Dua, D. and Graff, C. (2017). UCI Machine Learning Repository. 

- Efron, B., Hastie, T., Johnstone, I., and Tibshirani, R. (2004). Least angle regression. _The Annals of Statistics_ , 32(2):407–499. 

- Falck, F., Wang, Z., and Holmes, C. C. (2024). Is in-context learning in large language models bayesian? A martingale perspective. In _Proceedings of the 41st International Conference on Machine Learning_ , volume 235 of _Proceedings of Machine Learning Research_ , pages 12784–12805. PMLR. 

- Feuer, B., Schirrmeister, R. T., Cherepanova, V., Hegde, C., Hutter, F., Goldblum, M., Cohen, N., and White, C. (2024). Tunetables: Context optimization for scalable prior-data fitted networks. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_ . 

- Fong, E., Holmes, C., and Walker, S. G. (2023). Martingale posterior distributions. _Journal of the Royal Statistical Society Series B: Statistical Methodology_ , 85(5):1357–1391. 

- Fortini, S., Ng, K., Petrone, S., Rousseau, J., and Wei, S. (2026). A principled framework for uncertainty decomposition in tabpfn. _arXiv preprint arXiv:2602.04596_ . 

- Garg, S., Tsipras, D., Liang, P. S., and Valiant, G. (2022). What can transformers learn in-context? a case study of simple function classes. _Advances in Neural Information Processing Systems_ , 35:30583–30598. 

- Ghosal, S. and Van der Vaart, A. W. (2017). _Fundamentals of nonparametric Bayesian inference_ , volume 44. Cambridge University Press. 

- Hall, P. and Horowitz, J. (2013). A simple bootstrap method for constructing nonparametric confidence bands for functions. _The Annals of Statistics_ , pages 1892–1921. 

- Hollmann, N., Müller, S., Eggensperger, K., and Hutter, F. (2023). TabPFN: A transformer that solves small tabular classification problems in a second. In _The Eleventh International Conference on Learning Representations_ . 

- Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., and Hutter, F. (2025). Accurate predictions on small data with a tabular foundation model. _Nature_ , 637(8045):319–326. 

- Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. (2020). Time-uniform chernoff bounds via nonnegative supermartingales. _Probability Surveys_ , 17:257–317. 

- Huk, D., Zhang, Y., Dutta, R., and Steel, M. (2024). Quasi-Bayes meets vines. _Advances in Neural Information Processing Systems_ , 37:40359–40392. 

- Hüllermeier, E. and Waegeman, W. (2021). Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods. _Machine learning_ , 110(3):457–506. 

- Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. In _Proceedings of the 31st Conference on Neural Information Processing Systems_ . 

- Liu, S.-Y. and Ye, H.-J. (2025). TabPFN Unleashed: A Scalable and Effective Solution to Tabular Classification Problems. In _Forty-second International Conference on Machine Learning_ . 

- Müller, S., Hollmann, N., Arango, S. P., Grabocka, J., and Hutter, F. (2022). Transformers can do Bayesian Inference. In _International Conference on Learning Representations_ . 

10 

- Nagler, T. (2023). Statistical foundations of prior-data fitted networks. In _International Conference on Machine Learning_ , pages 25660–25676. PMLR. 

- Ng, K., Fong, E., Frazier, D. T., Knoblauch, J., and Wei, S. (2025). TabMGP: Martingale posterior with TabPFN. _arXiv:2510.25154_ . 

- Qu, J., Holzmüller, D., Varoquaux, G., and Le Morvan, M. (2025). TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. In _International Conference on Machine Learning_ , pages 50817–50847. PMLR. 

- Qu, J., Holzmüller, D., Varoquaux, G., and Morvan, M. L. (2026). TabICLv2: A better, faster, scalable, and open tabular foundation model. _arXiv preprint arXiv:2602.11139_ . 

- Reuter, A., Rudner, T. G. J., Fortuin, V., and Rügamer, D. (2025). Can transformers learn full bayesian inference in context? In _Forty-second International Conference on Machine Learning_ . 

- Schervish, M. J. (2012). _Theory of statistics_ . Springer Science & Business Media. 

- Stone, C. J. (1982). Optimal global rates of convergence for nonparametric regression. _The Annals of Statistics_ , pages 1040–1053. 

- Szabó, B., van der Vaart, A., and van Zanten, J. (2015). Frequentist coverage of adaptive nonparametric bayesian credible sets. _The Annals of Statistics_ , 43(4):1391–1428. 

- Thomas, V., Ma, J., Hosseinzadeh, R., Golestan, K., Yu, G., Volkovs, M., and Caterini, A. L. (2024). Retrieval & fine-tuning for in-context tabular models. _Advances in Neural Information Processing Systems_ , 37:108439–108467. 

- Tsanas, A. and Xifara, A. (2012). Accurate Quantitative Estimation of Energy Performance of Residential Buildings Using Statistical Machine Learning Tools. _Energy and Buildings_ , 49. 

- Yeh, I.-C. (1998). Modeling of Strength of High-Performance Concrete Using Artificial Neural Networks. _Cement and Concrete research_ , 28(12). 

## **A Proof of Proposition 3.3** 

Let _P_ 0 be an arbitrary probability distribution on R. We first show that _PM_ ( _y_ ) _, M ≥_ 0 _,_ is a martingale. It holds 



= (1 _− αn_ + _M −_ 1) _PM −_ 1( _y_ ) + _αn_ + _M −_ 1E _yn_ + _M ∼PM −_ 1[ _Hρ_ ( _PM −_ 1( _y_ ) _, PM −_ 1( _yn_ + _M_ )] _._ 

By the probability integral transform, it holds _PM −_ 1( _yn_ + _M_ ) _∼_ Uniform[0 _,_ 1]. Thus, 



by the properties of the Gaussian copula. Hence, 



which implies that _PM_ ( _y_ ) is a martingale adapted to the filtration _σ_ ( _y_ ( _n_ +1):( _n_ + _M_ )) _M ≥_ 1. 

Next, we apply the uniform-in-time version of the Hoeffding-Azuma inequality given in Corollary 1 (a) of Howard et al. (2020). It holds 

_|PM_ ( _y_ ) _− PM −_ 1( _y_ ) _| ≤ αn_ + _M −_ 1 _≤_ 2<sup>_β_</sup> ( _n_ + _M_ + 1)<sup>_−β_</sup> _,_ for all _y ∈_ R _,_ 

and, for all _M ≥_ 0, 



where _Cβ_ = 2<sup>2</sup><sup>_β_</sup> _/_ (2 _β −_ 1). 

11 

Now Corollary 1 (a) of Howard et al. (2020) with _m_ = _Cβn_<sup>1</sup><sup>_−_2</sup><sup>_β_</sup> yields 



Setting _ϵ_ = ~~�~~ 2 _Cβ_ log(2 _/δ_ ) _n_<sup>1</sup><sup>_/_2</sup><sup>_−β_</sup> gives, Pr( _∃M ≥_ 0: _|PM_ ( _y_ ) _− P_ 0( _y_ ) _| ≥ ϵ_ ) _≤ δ._ 

Now the claim follows since _PM_ ( _y_ ) _→ P∞,x_ ( _y_ ) almost surely (Proposition 3.2). 

## **B Further experimental results and details** 

### **B.1 Additional ablation results** 



<!-- Start of picture text -->
J = 1, d = 1 J = 5, d = 10 J = 10, d = 10 J = 10, d = 20 J = 20, d = 20<br>1.0<br>0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>1.0<br>0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>25 50 75 100 25 50 75 100 25 50 75 100 25 50 75 100 25 50 75 100<br>Number of forward samples<br>Training set size 50 100 200 400 800<br>Target quantile: 0.5<br>Coverage<br>Target quantile: 0.9<br><!-- End of picture text -->

Figure 4: Ablation results for the number of forward samples _N_ on all simulation settings. 



<!-- Start of picture text -->
J = 1, d = 1 J = 5, d = 10 J = 10, d = 10 J = 10, d = 20 J = 20, d = 20<br>1.00<br>0.75<br>0.50<br>0.25<br>0.00<br>1.00<br>0.75<br>0.50<br>0.25<br>0.00<br>200 400 600 800 200 400 600 800 200 400 600 800 200 400 600 800 200 400 600 800<br>Training set size<br>Type Ours (TabPFN) Initial PPD: scale by 0.8 Initial PPD: scale by 1.25 Schedule: beta = 1 Schedule: no blowup<br>Target quantile: 0.5<br>Coverage<br>Target quantile: 0.9<br><!-- End of picture text -->

Figure 5: Ablation results for initialization and schedule on all simulation settings. 

12 

### **B.2 Real data sets** 

We use the following data sets from the UCI repository (Dua and Graff, 2017): 

- `airfoil` ( _n_ = 1503, _d_ = 5), 

- `boston` ( _n_ = 252, _d_ = 15), 

- `concrete` ( _n_ = 1030, _d_ = 8, Yeh, 1998), 

- `diabetes` ( _n_ = 442, _d_ = 10, Efron et al., 2004), 

- `energy` ( _n_ = 768, _d_ = 8, Tsanas and Xifara, 2012), 

- `fish` ( _n_ = 908, _d_ = 6), 

- `forest_fire` ( _n_ = 516, _d_ = 13), 

- `real` ( _n_ = 413, _d_ = 7). 

### **B.3 Additional benchmark results** 

Table 2: Benchmark results for median regression on various UCI data sets, reporting averages over ten random 20-80 train-test splits with _±_ 2se. The target coverage is 90%. 

|||Metric|airfoil|boston|concrete|diabetes|energy|fsh|forest|real|
|---|---|---|---|---|---|---|---|---|---|---|
||bPFN|Coverage<br>Width|0.86_±_0.01<br>0.65_±_0.03|0.93_±_0.03<br>1.01_±_0.06|0.90_±_0.01<br>0.95_±_0.03|1.00_±_0.00<br>2.21_±_0.10|0.95_±_0.01<br>0.16_±_0.01|0.98_±_0.01<br>1.72_±_0.08|0.70_±_0.24<br>2.37_±_0.09|0.98_±_0.01<br>3.08_±_0.28|
|MP|Ta|Time (s)|20.4_±_0.1|6.1_±_0.3|13.7_±_0.7|5.9_±_0.5|9.1_±_0.1|11.3_±_0.3|6.8_±_0.6|5.3_±_0.3|
|A|CL|Coverage|0.84_±_0.02|0.88_±_0.03|0.89_±_0.02|1.00_±_0.00|0.93_±_0.01|0.94_±_0.01|0.96_±_0.03|0.96_±_0.02|
||TabI|Width<br>Time (s)|0.64_±_0.02<br>21.1_±_0.5|1.03_±_0.06<br>6.2_±_0.3|0.89_±_0.03<br>14.2_±_1.2|2.15_±_0.08<br>6.0_±_0.4|0.14_±_0.01<br>9.3_±_0.2|1.75_±_0.07<br>11.2_±_0.2|2.90_±_0.11<br>6.7_±_0.5|3.08_±_0.28<br>5.2_±_0.4|
||N|Coverage|0.67_±_0.01|0.71_±_0.03|0.62_±_0.01|0.86_±_0.04|0.78_±_0.01|0.71_±_0.02|0.87_±_0.07|0.75_±_0.04|
|p|bPF|Width|0.48_±_0.02|0.55_±_0.03|0.51_±_0.01|1.14_±_0.04|0.11_±_0.01|0.79_±_0.03|1.09_±_0.10|1.09_±_0.12|
|stra|Ta|Time (s)|83.1_±_32.5|35.0_±_17.0|56.2_±_40.3|67.2_±_34.1|57.9_±_27.0|17.8_±_4.2|20.5_±_4.5|15.9_±_4.9|
|oot|L|Coverage|0.69_±_0.02|0.58_±_0.02|0.58_±_0.01|0.82_±_0.03|0.80_±_0.02|0.48_±_0.02|0.52_±_0.11|0.66_±_0.02|
|B|bIC|Width|0.51_±_0.02|0.50_±_0.03|0.46_±_0.01|0.81_±_0.02|0.09_±_0.01|0.53_±_0.02|0.81_±_0.08|0.99_±_0.13|
||Ta|Time (s)|64.4_±_27.6|32.4_±_12.5|48.3_±_26.4|24.8_±_5.5|48.8_±_18.9|58.0_±_14.5|40.3_±_11.6|33.2_±_13.2|



### **B.4 Computational environment** 

All computations were performed on a user PC with a GeForce RTX 4070 Ti GPU using Python 3.12.7. The total run time of the experiments does not exceed 24 hours. All experiments are run with versions `tabpfn==2.0.5` ( `n_estimators=8` ) and `tabicl==2.0.3` ( `n_estimators=2` ). 

13 

