# **LLMs are Bayesian in Expectation, Not Realization** 

**Leon Chlon** Hassana Labs `leo@hassana.io` 

**Zein Khamis Fatima Sheaib** Hassana Labs Hassana Labs 

**Maggie Chlon** Hassana Labs 

**Mahdi El Zein MarcAntonio M. Awada** Hassana Labs Harvard University `mawada@hbs.edu` 

## **Abstract** 

Bayesian accounts of in-context learning appear to face a direct objection: exact posterior predictives for exchangeable data are invariant to task-preserving order, while transformers often change next-token probabilities when the same examples are serialized differently. We show that this objection targets a structural invariant rather than the operational quantity used to score online prediction. For any Bayesian predictive reference, excess prequential code length is exactly cumulative predictive KL. For unordered support sets that must be serialized, the expected regret of a single admissible ordering decomposes into the regret of the orderaveraged predictor plus an order-averaging gain. Exchangeability violations are therefore not binary refutations; they are priced by log loss. We instantiate the theory with KT/Dirichlet finite-alphabet prediction, candidate-event decompositions, safe-code floors, and coarsened Bayesian linear-regression (BLR) predictive distributions. On Qwen2.5-7B/14B, floored candidate distributions at support 256 have one-step excess code lengths of 0 _._ 020 _/_ 0 _._ 011 bits for Bernoulli prediction and 0 _._ 039 _/_ 0 _._ 022 bits for four-way categorical prediction, with candidate mass above 0 _._ 999; coarsened BLR continuations increasingly match the posterior-predictive digit distribution as support grows. A frequentist plug-in baseline sharpens the reading: the predictive distributions sit closer to the Bayesian posterior predictive than to the maximum-likelihood plug-in, by a margin largest at small support—where the plug-in is degenerate—and vanishing as the references converge. Position interventions and a from-scratch encoding ablation separate task-preserving from semantic order and localize it to the positional encoding; activation patching tests whether decoded sufficient statistics causally affect predictions; and permutation mixtures quantify the log-loss cost of arbitrary orderings downstream. The result is a quantitative reconciliation: transformers need not realize exchangeable posterior predictives for every serialization to be Bayes-competitive prequential predictors. 

## **1 Introduction** 

Bayesian interpretations of in-context learning give few-shot prompting a clean semantics: the support examples identify a latent task, and the continuation is the posterior predictive under that task [1, 2, 4, 3]. This view predicts a concrete object: the conditional distribution for the next token. It also appears to collide with a concrete fact about transformers. A Bayesian posterior predictive for exchangeable observations is invariant to task-preserving reordering; an autoregressive transformer reads a sequence with positions. 

Recent martingale and exchangeability diagnostics make this collision explicit. Pretrained language models do not, in general, satisfy the identities of exact Bayesian prediction on exchangeable data: 

Preprint. 

permuting an unordered support set can change next-token probabilities, and posterior-predictive martingale tests can fail [9, 10]. These results are important. They rule out the strongest slogan version of Bayesian ICL: a transformer is not an exact exchangeable posterior-predictive distribution for every serialization of the same support set. 

They do not rule out the operational claim that matters for online prediction. A sequence predictor is scored by prequential log loss, 



where _qt_ ( _· | y<t_ ) is the conditional distribution actually used at prefix _t_ . This score does not directly penalize failure of invariance at unused serializations. It penalizes probability assigned to the next observation. If the model’s conditional distributions stay close to a Bayesian reference in predictive KL along the evaluated path, the cumulative code length can remain close to the Bayesian code even when exact exchangeability fails. 

The central contribution is therefore a change of estimand. Structural exchangeability is a property of an exact posterior-predictive law. Prequential regret is the operational quantity paid by a sequence predictor. The bridge between them is predictive KL. We prove this first as an exact prequential comparison identity. We then show how arbitrary serialization of an unordered information state contributes an additional, measurable order-averaging gain. This turns exchangeability from a yes/no test into a code-length budget. 

**Thesis.** Exchangeability failures refute exact posterior-predictive equivalence. They do not, by themselves, refute Bayes-competitive prequential prediction. The operational question is how much cumulative predictive KL the violations spend. 

**Contributions.** First, we give a robust formalization of the evaluation target: excess prequential code length against any Bayesian reference is cumulative predictive KL. Second, we introduce a task-preserving ordering decomposition: the regret of a single serialization equals the regret of the order-averaged predictor plus a nonnegative order-averaging gain. Third, we connect language-model scoring to native probabilities through a candidate-event decomposition and safe-code floors with explicit overhead. Fourth, we audit pretrained transformers in closed-form Bayesian settings, coarsened BLR continuations, position interventions, causal statistic-use interventions, and downstream order-averaging experiments. Fifth, we add a non-Bayesian baseline—the maximum-likelihood and Laplace plug-in predictors—and report the model-to-Bayes minus model-to-plug-in predictive code length, which tests whether a frequentist sequential predictor accounts for the same conditionals. The experiments are not presented as evidence that transformers are literal posterior samplers. They estimate the code-length terms that determine whether order sensitivity is operationally costly. 

**Relation to prior work.** Bayesian and latent-task accounts explain ICL as approximate posterior prediction [1, 2, 3]; algorithmic accounts show that transformers can implement learned predictors, gradient-descent-like updates, or algorithm selection in context [5, 6, 7, 4, 8]; positional-encoding work studies the sequence mechanisms that make exact exchangeability nontrivial [15, 16, 17, 18]; and prompt-order work documents practical sensitivity to support order and calibration [11, 12, 13, 14]. The martingale critiques make exact posterior-predictive equivalence untenable for present LLMs [9, 10]. We keep that negative result and replace the binary invariance question by a prequential regret question. 

## **2 Prequential prediction under task-preserving orderings** 

We distinguish two notions that are often conflated. _Structural Bayesianity_ asks whether every realized posterior-predictive distribution satisfies invariances implied by the probabilistic model, such as exchangeability under permutations of an unordered support set. _Operational Bayes-competitiveness_ asks how much extra prequential code length a predictor pays relative to a Bayesian predictive reference. These notions are not equivalent. 

2 

### **2.1 Prequential comparison** 

Let _P_ , _B_ , and _Q_ be sequential processes on a common dominated observation space, with conditional predictive distributions 

_pt_ ( _· | ht_ ) _, bt_ ( _· | ht_ ) _, qt_ ( _· | ht_ ) _, ht_ = _y<t._ 

Think of _P_ as the evaluation distribution, _B_ as the Bayesian reference, and _Q_ as the model being scored. 

**Theorem 1** (Prequential comparison identity) **.** _Assume the relevant absolute-continuity and integrability conditions. Then_ 



_Equivalently,_ 





The identity is elementary but decisive. Under a Bayesian reference process, excess code length is exactly cumulative predictive KL. Exchangeability violations matter operationally only through the predictive KL they accumulate. This identity is the paper’s formal replacement for binary martingale rejection as an evaluation target. 

### **2.2 Serialization of unordered information states** 

An unordered statistical object must be serialized before a transformer can process it. Let _Zt_ denote the task-relevant information state at prefix _t_ . A _task-preserving ordering_ is a serialization that leaves _Zt_ unchanged. For Bernoulli prediction, _Zt_ = ( _t, St_ ); for categorical prediction, _Zt_ = ( _t, c_ 1 _, . . . , cK_ ); for conjugate Bayesian linear regression (BLR), _Zt_ consists of the sufficient statistics and query covariate; for evidence-grounded QA, it is the evidence set under the admissible retrieval-order distribution. If reordering changes the label or the task semantics, it is not task-preserving and should not be averaged. 

Let _µt_ ( _· | Zt_ ) be any distribution over task-preserving orderings. Uniform permutations are a special case; banded retrieval-preserving permutations are another. Let _qt_<sup>_g_(</sup><sup>_· | Zt_) be the model’s predictive</sup> distribution after serialization _g_ , and define the order-averaged predictor 

_q_ ¯ _t_ ( _· | Zt_ ) = E _g∼µt_ ( _· | Zt_ ) _qt_<sup>_g_(</sup><sup>_· | Zt_)</sup><sup>_._</sup> (5) 

**Theorem 2** (Order-averaging decomposition) **.** _Suppose the Bayesian reference predictive distribution bt_ ( _· | Zt_ ) _depends only on Zt and is therefore invariant to task-preserving orderings. Then, for each information state,_ 



_where_ 



_Consequently, the expected prequential regret of a sampled single ordering decomposes as_ 



Thus exact exchangeability corresponds to zero order-averaging gain. Nonzero gain is the extra log loss paid by using one admissible serialization instead of the averaged predictive distribution. The theorem does not make order sensitivity free. It prices it. 

**Corollary 1** (Persistent order sensitivity has a predictive price) **.** _In natural-log units, on a shared finite candidate space,_ 

E _g,g′_ TV( _qt_<sup>_g, q_</sup> _t_<sup>_g′_)2</sup><sup>_≤_2 E</sup><sup>_g_KL(</sup><sup>_bt∥q_</sup> _t_<sup>_g_)</sup><sup>_._</sup> (9) 

_Therefore a predictor with small average predictive KL cannot have persistent large task-preserving order dispersion on the scored event space._ 

3 

### **2.3 Candidate events and safe predictors** 

Language-model evaluations often score a fixed set of answer events rather than the entire vocabulary. Scoring a fixed answer set is valid only if the mass outside the event set is reported or decomposed. For single-token labels the events are tokens; for multi-token answers they are the continuation events defined by the chosen verbalizer. The following identity makes the decomposition exact. 

**Proposition 1** (Candidate-event decomposition) **.** _Fix a prefix and mutually exclusive candidate events E_ 1 _, . . . , Em, with E_ = _∪iEi. Suppose the reference distribution is supported on these events, bi_ = _B_ ( _Ei | h_ ) _and_<sup>�</sup> _i_<sup>_bi_= 1</sup><sup>_. Let the model assign event probabilities Q_(</sup><sup>_Ei | h_)</sup><sup>_and total candidate_</sup> _mass_ 



_When MQ_ ( _E | h_ ) _>_ 0 _, define the normalized candidate distribution q_ ¯ _i_ = _Q_ ( _Ei | h_ ) _/MQ_ ( _E | h_ ) _. Then_ 



Candidate normalization therefore does not hide probability mass: it separates predictive-shape error from outside-candidate mass. We report both quantities in the controlled audits. 

**Lemma 1** (Safe-code floor) **.** _Let q be a predictive distribution on m candidate events and let u be uniform. For λ ∈_ (0 _,_ 1) _define_ 



_Then qi_<sup>_λ≥λ/m for all i.Moreover, whenever_KL(</sup><sup>_b∥q_)</sup><sup>_< ∞,_</sup> 



_and always_ 



_A fixed floor gives an explicit finite-horizon safe code; a vanishing floor schedule with_<sup>�</sup> _t≤n_<sup>_−_log(1</sup><sup>_−_</sup> _λt_ ) = _o_ ( _n_ ) _preserves asymptotic Bayes-competitiveness._ 

### **2.4 Closed-form instantiations** 

The theory is not Bernoulli-specific. Bernoulli and categorical tasks are the cleanest finite-alphabet settings because the Bayesian reference distribution is closed form. 

**Corollary 2** (Finite-alphabet KT/Dirichlet certificate) **.** _Let A be finite and let K_ KT _be the Krichevsky– Trofimov (KT) Dirichlet_ (1 _/_ 2 _, . . . ,_ 1 _/_ 2) _predictive code. For any i.i.d. source p ∈_ ∆( _A_ ) _,_ 



_For any predictor Q,_ 



_Thus if the cumulative comparison term is at most Rn, then Q has expected code length at most nH_ ( _p_ ) + _O_ (log _n_ ) + _Rn. The finite-support approximation and floor conditions reported in Table 2 are one sufficient way to obtain Rn_ = _O_ (<sup>_√_</sup> _<u>n</u>_ <u>)</u> _._ 

**Corollary 3** (Coarsened continuous predictive distributions) **.** _Let B be a Bayesian predictive process with conditional density for a real-valued observation, and let V be a finite quantizer or verbalizer with cells E_ 1 _, . . . , Em. The induced distributions_ 



_satisfy the same prequential comparison identity for the discrete process V_ ( _Yt_ ) _. If full densities for B and Q exist, data processing gives_ KL( _b_<sup>_V_</sup> _t_<sup>_∥q_</sup> _t_<sup>_V_)</sup><sup>_≤_KL(</sup><sup>_bt∥qt_)</sup><sup>_._</sup> 

4 

Table 1: Theory-to-evidence map. Each empirical quantity estimates a code-length term, a mass term, or a task-preservation condition in the theory. 

|Formal quantity|Interpretation|Main evidence|
|---|---|---|
|E_B_[_Ln_(_Q_)_−Ln_(_B_)]<br>|Excess prequential code length|Table 2|
|KL(_bt∥qt_)<br>|One-step predictive regret|Tables 2and3|
|KL(_bt∥_¯_qt_)|Regret after averaging task-preserving or-<br>derings|Tables 9and15|
|_Jt_|Log-loss cost of using one admissible or-<br>dering|NLL/order-averaging<br>gains;Section 4.4|
|_MQ_(_E_)|Probability assigned to scored candidate<br>events|candidate-mass column|
|_−_log(1_−λ_)|Safe-code overhead from fooring|native/foored<br>compari-<br>son|
|Task-preserving_Zt_|Whether reordering preserves the statisti-<br>cal task|Table 6andsection 4.4|
|Statistic intervention effect|Whether decoded statistics causally affect<br>predictions|Table 8|
|KL(_bt∥qt_)_−_KL(_ft∥qt_)|Separation from a frequentist plug-in _ft_<br>(MLE/Laplace/OLS)|Tables 4and5|



## **3 Experimental protocol** 

All experiments score next-token probabilities. The main models are Qwen2.5-7B/14B-Instruct and Qwen2.5-7B base for position and activation interventions; Llama-3.1-8B appears in probe and evidence-permutation comparisons [22, 23]. Apart from a controlled from-scratch ablation that isolates the positional-encoding channel (Section 4.4), all experiments are inference-only on pretrained checkpoints. Downstream order-averaging audits use MMLU, GSM8K, BBH, FEVER, HotpotQA, NQ-Open, and PopQA slices [31, 32, 33, 34, 35, 36, 37]. For discrete conjugate tasks, the prefix is an unordered sequence of symbols and the target is the next symbol. For BLR, the prefix is a raw numeric continuation task: _x_ 1 _, y_ 1 _x_ 2 _, y_ 2 _· · · x⋆,_ for one covariate, or _x_ 1 _, y_ 1 _, z_ 1 _· · · x⋆, y⋆,_ for two covariates. The BLR reference is the Gaussian posterior predictive distribution integrated over the digit bins used for token scoring. 

**Native, candidate, and floored distributions.** Native probabilities measure model behavior. When a task specifies candidate events, we report the normalized candidate distribution and the candidate mass, as justified by Proposition 1. For the finite-alphabet discrete audit we also score a floored safe predictor, 



where _UA_ is uniform over the candidate alphabet. The floored predictor is not used as evidence about native model probabilities. It is an evaluated safe code with explicit finite-horizon overhead under Lemma 1. 

**Task-preserving versus semantic order.** Order averaging is justified only for serializations that preserve the information state _Zt_ . The Bayesian prediction task presents an exchangeable support set, so demonstration order is task-preserving. The semantic-first control makes the first demonstration semantically relevant, so position is part of the task. This distinction is central: the intervention is not to average away every order effect, but to price the effects of arbitrary serialization when order is not part of the task. 

## **4 Empirical evidence** 

The experiments follow the theory. We first audit closed-form Bayesian predictive distributions. We then ask whether order effects are task-preserving or semantic, whether decoded sufficient statistics are causally used, and how much log loss is saved by averaging admissible orderings. The controlled predictive-distribution audits are the main evidence for Bayes-competitiveness. The downstream mixture experiments are predictive-score diagnostics, not claims of large accuracy gains. 

5 

Table 2: Discrete Bayesian predictive-distribution audit. Candidate distributions use the floored safe predictor _qθ,λ_ with _λ_ = 0 _._ 01. KL is in bits and equals one-step excess prequential code length under the Bayesian predictive distribution. 

|Family|Model|Support|Contexts|KL bits|_ℓ_1|_ϵ∞,A_|�_CA_|_c_min_,A_|Cand. mass|
|---|---|---|---|---|---|---|---|---|---|
|Beta–Bernoulli,0_/_1|7B|256|256|0_._020|0_._081|0_._243|3_._902|0_._192|0_._9998|
|Dirichlet-categorical,0_/_1_/_2_/_3|7B|256|256|0_._039|0_._154|0_._300|4_._805|0_._171|0_._9996|
|Beta–Bernoulli,0_/_1|14B|256|512|0_._011|0_._067|0_._227|3_._632|0_._329|0_._9998|
|Dirichlet-categorical,0_/_1_/_2_/_3|14B|256|128|0_._022|0_._124|0_._183|2_._937|0_._401|0_._9994|



Table 3: Coarsened BLR continuation distributions on Qwen2.5-7B. The model is scored on the next digit after raw pair or tuple sequences. The full Bayesian posterior predictive becomes a better distributional match as support grows; in the two-covariate task it beats the best misspecified control at support 100. 

|Task|Support|corr(E_θ,_E_B_)|Dist. corr.|Best controlEcorr.|Best control dist. corr.|KL bits|
|---|---|---|---|---|---|---|
|Pair-next-number|25|0_._935|0_._808|n/a|n/a|0_._388|
|Pair-next-number|50|0_._974|0_._886|n/a|n/a|0_._238|
|Pair-next-number|100|0_._988|0_._934|n/a|n/a|0_._131|
|Tuple-next-number|25|0_._880|0_._713|0_._891|0_._667|0_._669|
|Tuple-next-number|50|0_._935|0_._789|0_._902|0_._713|0_._439|
|Tuple-next-number|100|0_._965|0_._849|0_._881|0_._689|0_._318|



### **4.1 Controlled Bayesian predictive distributions** 

Table 2 reports the finite-alphabet quantities covered by Corollary 2. The Bernoulli and four-way categorical distributions are scored as floored KT/Dirichlet safe predictors. The KL column is onestep excess prequential code length in bits under the Bayesian reference distribution. The envelope constant _C_<sup>�</sup> _A_ and the floor ratio _c_ min _,A_ report finite-support sufficient conditions for the _O_ (<sup>_√_</sup> _<u>n</u>_ <u>)</u> certificate; candidate mass reports how much native probability lies on the scored events. 

Table 3 audits coarsened continuous prediction. BLR is continuous, so we score the induced digit distribution rather than claiming to measure the full posterior density. This is exactly the setting of Corollary 3. The posterior-predictive distribution becomes a better match as support grows, and in the two-covariate task it beats the best misspecified control by support 100. 

The finite discrete distributions give the cleanest certificate: they have small predictive KL, candidate mass above 0 _._ 999, and audited floor ratios well above the deterministic floor. The BLR distributions support the same operational interpretation on induced digit events: at support 100, Qwen2.5-7B tracks the one-covariate posterior-predictive expectation with correlation 0 _._ 988 and the two-covariate full BLR expectation with correlation 0 _._ 965. Additional 14B BLR predictive-regret sweeps are in Section C.5. 

### **4.2 A frequentist plug-in does not explain the predictive distributions** 

The audits above report small predictive KL against a Bayesian reference, but a Bayesian reading is only forced if a non-Bayesian sequential predictor does not fit the same conditionals equally well. The natural null is the frequentist plug-in: predict the next symbol from the running maximum-likelihood estimate—the empirical frequency—or with Laplace add-one smoothing; for the continuous task it is the ordinary-least-squares (OLS) point estimate. We score these predictors on the same contexts and under the same _λ_ = 0 _._ 01 floor as the model, and report the difference of predictive code lengths ∆= KL(model _∥_ Bayes) _−_ KL(model _∥_ plug-in), so that ∆ _<_ 0 certifies the model’s conditional is closer to the Bayesian predictive than to the frequentist one (Tables 4 and 5). 

Two properties separate the model from the plug-in. First, the maximum-likelihood plug-in is a _degenerate_ code on short prefixes: it assigns zero probability to any symbol not yet observed and pays infinite prequential code length whenever such a symbol occurs. This is not a small-constant effect—it is every single-example categorical context and 75% of contexts at support 16—while the ¯ order-averaged predictor _q_ reserves posterior mass for unseen symbols (up to 0 _._ 68 at support one) and holds its excess over the Bayesian reference below 0 _._ 2 bits throughout. The plug-in is inadmissible precisely where the model is admissible. Second, where the plug-in is non-degenerate the model 

6 

Table 4: Frequentist plug-in baseline for the discrete predictive audit (companion to Table 2). The order-averaged model predictive _q_ ¯ is compared to the KT/Jeffreys Bayesian reference and to two frequentist plug-ins—the MLE (empirical frequency) and Laplace (add-one)—across a support sweep; all predictors carry the _λ_ =0 _._ 01 safe-code floor (Lemma 1). ∆MLE = KL(¯ _q∥_ Bayes) _−_ KL(¯ _q∥_ MLE) and likewise ∆Lap; ∆ _<_ 0 means the model is closer to the Bayesian posterior predictive than to that plug-in. “MLE deg.” is the fraction of contexts on which the unfloored MLE assigns zero probability to a possible symbol (infinite code length). KL in bits. These use the order-averaged predictor _q_ ¯ of Theorem 2, so the model–Bayes column lies below the single-ordering values in Table 2. 

|Family|Model|Supp.|KL(¯_q∥_B)|∆MLE|∆Lap|MLE deg.|Cand. mass|
|---|---|---|---|---|---|---|---|
|Beta–Bernoulli|Qwen2.5-7B|4|0_._0111|_−_0_._134|_−_0_._017|52%|0_._9071|
|||16|0_._0152|_−_0_._007|_−_0_._004|25%<br>|0_._9858|
|||64|0_._0094|_−_0_._002|0_._000|12%|0_._9980|
|||256|0_._0022|_−_0_._000|0_._000|8%|0_._9996|
||Qwen2.5-14B|4|0_._0186|_−_0_._147|_−_0_._013|52%|0_._9045|
|||16|0_._0122|_−_0_._007|_−_0_._003|25%|0_._9923|
|||64|0_._0086|_−_0_._002|0_._001|12%|0_._9988|
|||256|0_._0010|_−_0_._000|_−_0_._000|8%|0_._9998|
|Dirichlet-cat. (0_/_1_/_2_/_3)|Qwen2.5-7B|4|0_._1159|_−_0_._762|_−_0_._012|98%|0_._8630|
|||16|0_._0504|_−_0_._084|_−_0_._004|75%|0_._9690|
|||64|0_._0150|_−_0_._005|_−_0_._002|53%|0_._9964|
|||256|0_._0049|_−_0_._001|0_._000|28%|0_._9988|
||Qwen2.5-14B|4|0_._1316|_−_0_._860|0_._003|98%|0_._8439|
|||16|0_._0499|_−_0_._083|_−_0_._004|75%|0_._9776|
|||64|0_._0182|_−_0_._005|_−_0_._002|53%|0_._9974|
|||256|0_._0046|_−_0_._000|_−_0_._000|28%|0_._9993|



still sits on the Bayesian side: ∆MLE is _−_ 3 _._ 3 bits (7B) and _−_ 3 _._ 8 bits (14B) for four-way categorical prediction at a single example, _−_ 0 _._ 76 bits at support four, and _−_ 0 _._ 001 bits by support 256, where the Bayesian and plug-in predictives provably coincide and no predictor is distinguishable. The separation from the _Laplace_ plug-in is an order of magnitude smaller and of inconsistent sign; this is the expected signature rather than a weakness, because add-one smoothing is itself the Dirichlet(1) posterior predictive—the comparison is then against a member of the model’s own conjugate family, and _q_ ¯ lands near the Jeffreys value rather than at the maximum-likelihood corner. 

The coarsened BLR continuations show the same separation against the OLS plug-in (Table 5). The point estimate reproduces the Bayesian posterior _mean_ , so the plug-in’s entire deficit is the querydependent predictive-variance inflation _σ_<sup>2</sup> + _x_<sup>_⊤_</sup> _⋆_<sup>Σ</sup><sup>_nx⋆_that a posterior carries and a point estimate</sup> discards. Accordingly ∆= KL(Bayes _∥q_ ) _−_ KL(OLS _∥q_ ) is negative on 85–99% of contexts—by 0 _._ 65 bits at support four, shrinking with support—and the model’s per-query predictive variance correlates with the Bayesian one at 0 _._ 57–0 _._ 83. The pattern is identical for one and two covariates and replicates on Llama-3.1-8B. In every setting the separation concentrates in the small-data regime, where Bayesian and frequentist prediction genuinely diverge, and closes asymptotically—the behavior the prequential comparison identity (Theorem 1) predicts for a Bayes-competitive predictor, and the reason the support-256 cells alone cannot adjudicate the question. 

### **4.3 Position interventions separate task-preserving order from semantic order** 

We hold Qwen2.5-7B’s weights and tokens fixed and change only `position_ids` at inference time. The normal condition uses monotone positions. The demo-local condition resets positions inside each demonstration block, preserving within-demonstration token order while removing the absolute demonstration slot. The random-offset condition preserves within-demonstration order but assigns each demonstration a random positional offset. We evaluate 64 items with 16 support permutations per item for each task and position mode and report 95% bootstrap confidence intervals. 

This intervention estimates whether ordering is task-preserving or semantic. In the Bayesian prediction task, absolute demonstration position explains much of the serialization variation. In the semantic-first control, removing that channel damages performance because position carries task signal. A reviewer should therefore not read order sensitivity as one phenomenon: the same positional machinery can be task-preserving in one task and signal in another. 

7 

Table 5: Frequentist plug-in baseline for the BLR predictive audit (companion to Table 3). The model’s next-digit distribution is compared to the full Bayesian posterior predictive and to the OLS point-estimate plug-in (same homoscedastic noise, no posterior-variance inflation). ∆= _−_ KL(Bayes _∥q_ ) KL(OLS _∥q_ ) in bits; “closer” is the fraction of contexts on which the model is nearer the full Bayesian predictive than the plug-in; “var. corr” is the correlation between the model’s and the Bayesian per-query predictive variance. The plug-in’s deficit is the missing query-dependent variance inflation; ∆ _<_ 0 means the model is closer to Bayes. 

|BLR|Model|Supp.|KL(B_∥q_)|KL(OLS_∥q_)|∆[|95% CI]|closer|var. corr|
|---|---|---|---|---|---|---|---|---|
|1-cov. (pair)|Qwen2.5-7B|4|0_._7744|1_._4265<br>_−_0_._652|[_−_0_._749_, _|_−_0_._566]|95%|0_._571|
|||8|0_._5913|0_._8854<br>_−_0_._294|[_−_0_._335_, _|_−_0_._252]|93%|0_._618|
|||16|0_._4726|0_._6254<br>_−_0_._153|[_−_0_._183_, _|_−_0_._124]|86%|0_._736|
||Qwen2.5-14B|4|0_._7453|1_._3821<br>_−_0_._637|[_−_0_._715_, _|_−_0_._562]|95%|0_._705|
|||8|0_._5262|0_._8080<br>_−_0_._282|[_−_0_._321_, _|_−_0_._244]|93%|0_._750|
|||16|0_._3852|0_._5093<br>_−_0_._124|[_−_0_._147_, _|_−_0_._102]|87%|0_._796|
|2-cov. (tuple)|Qwen2.5-7B|20|0_._6230|0_._8234<br>_−_0_._200|[_−_0_._215_, _|_−_0_._186]|95%|0_._826|
||Qwen2.5-14B|20|0_._6777|0_._8807<br>_−_0_._203|[_−_0_._217_, _|_−_0_._190]|95%|0_._770|



Table 6: Pretrained position intervention. Resetting demonstration-local positions reduces Bayesian prediction order variance by about 21 _×_ , while the semantic-order control degrades. Positive ∆ NLL (negative log-likelihood) means higher loss than normal positions. 

|Task|Position mode|Order|variance|NLL|∆NLL vs. normal|
|---|---|---|---|---|---|
|Bayesian prediction|normal|5_._108 [4_._480_,_5_._780] <br>|_×_10<sup>_−_3</sup><br>|0_._6157[0_._6090_,_0_._6221]|n/a|
|Bayesian prediction|demo-local|2_._420 [2_._136_,_2_._715] <br>|_×_10<sup>_−_4</sup><br>|0_._6358[0_._6297_,_0_._6417]|+0_._0201[0_._0168_,_0_._0232]|
|Bayesian prediction|random offsets|5_._343 [4_._833_,_5_._855]|_×_10<sup>_−_3</sup>|0_._6831[0_._6792_,_0_._6868]|+0_._0674[0_._0614_,_0_._0734]|
|Semantic frst|normal|2_._174 [2_._005_,_2_._360] <br>|_×_10<sup>_−_3</sup><br>|0_._1236[0_._1205_,_0_._1269]|n/a|
|Semantic frst|demo-local|1_._049 [0_._959_,_1_._139] <br>|_×_10<sup>_−_2</sup><br>|0_._6599[0_._6468_,_0_._6730]|+0_._5363[0_._5229_,_0_._5491]|
|Semantic frst|random offsets|2_._020 [1_._816_,_2_._228]|_×_10<sup>_−_2</sup>|0_._3793[0_._3639_,_0_._3947]|+0_._2557[0_._2407_,_0_._2707]|



### **4.4 Controlled positional-encoding ablation** 

The pretrained intervention shows that position carries the order channel, but it cannot remove that channel from a fixed model. To test whether positional encoding is the _source_ of task-preserving order sensitivity, we train small transformers from scratch on i.i.d. Bernoulli sequences and vary only the positional-encoding scheme: none, learned absolute, sinusoidal, RoPE, and ALiBi, with a pooled readout that is permutation-invariant when no encoding is supplied. Training and evaluation settings are in Table 16. 

The schemes separate sharply. With no positional encoding, the within-prefix order variance is _≈_ 3 _._ 7 _×_ 10<sup>_−_16</sup> , the exact-exchangeability endpoint to numerical precision: the predictive distribution does not move across serializations, so _Jt_ is zero at every state. Learned absolute, sinusoidal, RoPE, and ALiBi instead sit at 10<sup>_−_8</sup> to 10<sup>_−_6</sup> , an eight-to-ten order-of-magnitude increase. Exchangeability failure is thus a property of the encoding a transformer is given, not of the architecture; Section A.8 gives one stylized channel by which a fixed encoding produces this dispersion. 

These models also let us read the order-averaging decomposition against a closed-form reference. Scoring cumulative regret relative to the KT predictive code (Corollary 2) at _n_ = 200, the permutationaveraged regret stays between 0 _._ 0017 and 0 _._ 0022 bits/token across all five schemes, while the worst sampled ordering costs 0 _._ 0049 to 0 _._ 0098 bits/token. This is Equation (8) read off a controlled system: _Rn_<sup>avg</sup> is a few thousandths of a bit against the Bayesian code, and the single-ordering penalty<sup>�</sup> _t_<sup>E[</sup><sup>_Jt_]</sup> is the worst-case-minus-average gap. Order sensitivity is real and bounded away from zero, and it is also small in the units that score prediction. 

### **4.5 Decoded statistics are available and causally used** 

High probe accuracy alone would not establish Bayesian prediction; it only shows that a statistic is available [24, 25]. We therefore use probes as a localization diagnostic and activation patching as the causal-use test [26, 27, 28]. Ridge probes on residual streams recover the joint sufficient statistic ( _St, t_ ) with _R_<sup>2</sup> = 0 _._ 9998 on Qwen2.5-7B and Llama-3.1-8B; categorical joint counts reach 

8 

Table 7: Linear-probe availability diagnostic. Count and position information is linearly available; Table 8 tests whether the information is used in the next-token distribution. 

|Model|Joint(_St, t_)_R_<sup>2</sup>|Emp. freq. _R_<sup>2</sup>|Pseudo-count_R_<sup>2</sup>|Cat. joint_R_<sup>2</sup>|
|---|---|---|---|---|
|Qwen2.5-7B|0_._9998|0_._904|0_._926|0_._9993|
|Llama-3.1-8B|0_._9998|0_._779|0_._801|0_._9992|



Table 8: Layer-specific activation-patching audit on Qwen2.5-7B. Values are restoration scores with 95% bootstrap CIs. Learned subspace patching restores task logits far above matched random controls. 

|Task|Layer|Full|vector<br>Lea|rned subspace|Random subspace|Random vector|
|---|---|---|---|---|---|---|
|Keybinding|24|0_._510[0_._503_,_|0_._518]<br>0_._254|[0_._237_,_0_._270]|0_._008[0_._006_,_0_._011]|_−_0_._001[_−_0_._004_,_0_._001]|
|Keybinding|26|0_._756[0_._744_,_|0_._769]<br>0_._338|[0_._315_,_0_._361]|0_._015[0_._011_,_0_._018]|0_._006[0_._002_,_0_._009]|
|Bernoulli-next|24|0_._887[0_._838_,_|0_._939]<br>0_._880|[0_._814_,_0_._946]|_−_0_._047[_−_0_._090_, −_0_._005]|_−_0_._014[_−_0_._064_,_0_._034]|
|Bernoulli-next|26|0_._928[0_._887_,_|0_._969]<br>0_._854|[0_._799_,_0_._910]|_−_0_._051[_−_0_._105_,_0_._003]|_−_0_._027[_−_0_._079_,_0_._024]|



_R_<sup>2</sup> = 0 _._ 9993 and 0 _._ 9992 respectively. Empirical-frequency and pseudo-count probes are weaker, consistent with the model carrying count and position features before readout. 

The probe result motivates an intervention. We form clean/corrupt prompt pairs with different sufficient statistics, patch a learned low-rank activation subspace from clean into corrupt examples, and measure restoration of the candidate-logit difference. Matched random subspaces and matched random vectors control for intervention norm and rank. 

The learned subspace restores the candidate logit difference well above matched random controls, especially for Bernoulli-next. This closes the gap left by probes: beyond being decodable, the statistic can be intervened on to move the next-token distribution. 

### **4.6 Order averaging as a predictive-score diagnostic** 

When support order is task-preserving, averaging predictive distributions across sampled serializations is the empirical version of Theorem 2. Concavity of log loss supplies the guaranteed comparison to the average sampled ordering; the magnitude of the gain estimates how much log loss arbitrary serialization spends. On MMLU, the effect is statistically precise but practically modest. That is consistent with the theory: the measured order-averaging gain is small for this benchmark and prompt distribution. 

The mean per-item standard deviation of the gold-answer probability across random orderings is 0 _._ 0158 (95% CI [0 _._ 0151 _,_ 0 _._ 0165]), so order changes the predictive distribution even when it usually does not change the argmax. Among 359 mixture-wrong items, 319 are wrong under every sampled order and only 40 have any sampled order with the gold answer on top. Uniform averaging smooths the predictive distribution; it does not search for a favorable order. GSM8K-derived multiple choice shows the same calibration geometry at a larger scale: averaging random demonstration orders improves NLL from 1 _._ 5424 at _k_ = 1 to 1 _._ 4948 at _k_ = 8, Brier from 0 _._ 8052 to 0 _._ 7922, and expected calibration error (ECE) from 0 _._ 3204 to 0 _._ 3097, while accuracy remains near 0 _._ 43 (Table 14). 

### **4.7 Evidence-grounded QA with exchangeable evidence** 

A practical version of the same problem arises when unordered evidence chunks must be serialized in a prompt. On a five-benchmark factuality slice of 3 _,_ 059 items with 3–60 chunks per item, we form banded evidence permutations rather than all _n_ ! permutations, reflecting a non-uniform admissible ordering distribution. Qwen2.5-7B-Instruct shows a dispersion slope of 0 _._ 377 versus log _n_ and an order-averaging gain of 0 _._ 1041 nats/token; Llama-3.1-8B-Instruct shows a slope of 0 _._ 147 and a gain of 0 _._ 00982 nats/token. Uniform weights are within the measured optimization tolerance in both cases (Table 15). This is the applied order-averaging principle: if the evidence set is exchangeable but the model must read a sequence, score or average admissible serializations rather than treating one arbitrary order as the task. 

9 

Table 9: MMLU random-permutation audit on Qwen2.5-7B. With 1200 items, 8 demonstrations, and 16 unique random permutations per item, the same-sample predictive mixture improves NLL over the mean sampled single ordering. CIs are paired bootstraps; _p_ -values are sign-flip tests. 

|Comparison|B|aseline<br>|Mixture (_k_=16)|Paired delta|_p_|
|---|---|---|---|---|---|
|Canonical accuracy|0_._6983[0_._6717_,_|0_._7242]<br>0_._7008[|0_._6750_,_0_._7275]|+0_._0025[_−_0_._0042_,_0_._0100]|0_._654|
|Canonical NLL|0_._7243[0_._6705_,_|0_._7768]<br>0_._7212[|0_._6694_,_0_._7741]|_−_0_._0031[_−_0_._0074_,_0_._0009]|0_._144|
|Mean single-order accuracy|0_._6987[0_._6728_,_|0_._7231]<br>0_._7008[|0_._6750_,_0_._7275]|+0_._0021[_−_0_._0024_,_0_._0069]|0_._378|
|Mean single-order NLL|0_._7235[0_._6717_,_|0_._7762]<br>0_._7212[|0_._6694_,_0_._7741]|_−_0_._00230[_−_0_._00262_, −_0_._00201]|0_._00020|



## **5 Boundary conditions and interpretation** 

The theory is broad, but each application must specify the predictive reference, the candidate events being scored, and the orderings that preserve the task. Bernoulli and categorical prediction use closedform KT/Dirichlet references. BLR uses the induced digit distribution of a continuous posterior predictive, not the full continuous density. MMLU, GSM8K, and evidence QA are downstream order-averaging diagnostics unless an explicit Bayesian reference is defined. 

The results do not show that transformers instantiate literal posterior distributions, nor that all order effects are harmless. Corollary 1 states the opposite: persistent large movement of the scored predictive distribution has a log-loss price. The point is that the price is quantitative. Exact exchangeability is sufficient for zero order-averaging gain, but Bayes-competitive prequential prediction only requires the accumulated predictive KL to be small. 

The floored discrete predictor is an evaluated safe code over the candidate alphabet and should be distinguished from native model behavior. Fixed flooring is a finite-horizon device with explicit overhead; asymptotic claims require native lower bounds or a vanishing floor schedule. Candidate normalization is interpreted through Proposition 1, which is why candidate mass is reported. Finally, order averaging is appropriate only when order is task-preserving. For language modeling, chronology, chain-of-thought trajectories, or tasks where the first example changes the label, averaging orderings changes the task. 

**Broader impacts.** The work is primarily diagnostic. A positive impact is more reliable evaluation of in-context predictors: order sensitivity can be priced by log loss rather than treated as a binary failure. A possible negative impact is that order averaging may improve calibration of systems used for automated adjudication without addressing factual, fairness, or deployment risks. We therefore frame the evidence-QA experiment as a predictive-score audit, not as a deployment recommendation. 

## **6 Conclusion** 

The apparent paradox is resolved by separating structural invariance from operational regret. Transformers are not, in general, exact exchangeable posterior-predictive machines. But online prediction is scored by prequential log loss, and the excess over a Bayesian reference is cumulative predictive KL. When an unordered support set must be serialized, the additional cost of using one admissible ordering is the order-averaging gain. The experiments estimate these quantities: controlled discrete tasks give small KT/Dirichlet predictive KL, BLR continuations track coarsened posterior predictives, position interventions and a from-scratch encoding ablation separate arbitrary serialization from signal and bound its worst-case cost against the KT code, activation patching tests causal statistic use, and downstream mixtures measure serialization cost. A frequentist plug-in baseline separates the predictive distributions from a non-Bayesian sequential predictor, most sharply in the small-data regime and vanishingly as the references converge. Exchangeability failures therefore refute exact posterior equivalence, not Bayes-competitive prequential prediction. 

## **References** 

> [1] S. M. Xie, A. Raghunathan, P. Liang, and T. Ma. An explanation of in-context learning as implicit Bayesian inference. _International Conference on Learning Representations_ , 2022. 

> [2] S. Müller, N. Hollmann, S. P. Arango, J. Grabocka, and F. Hutter. Transformers can do Bayesian inference. _arXiv:2112.10510_ , 2021. 

10 

- [3] S. Müller, N. Hollmann, and F. Hutter. Bayes’ power for explaining in-context learning generalizations. _arXiv:2310.01388_ , 2024. 

- [4] Y. Bai, F. Chen, H. Wang, C. Xiong, and S. Mei. Transformers as statisticians: Provable in-context learning with in-context algorithm selection. _Advances in Neural Information Processing Systems_ , 36, 2023. 

- [5] S. Garg, D. Tsipras, P. S. Liang, and G. Valiant. What can transformers learn in-context? A case study of simple function classes. _arXiv:2208.01066_ , 2022. 

- [6] J. von Oswald et al. Transformers learn in-context by gradient descent. _International Conference on Machine Learning_ , 2023. 

- [7] E. Akyürek, D. Schuurmans, J. Andreas, T. Ma, and D. Zhou. What learning algorithm is in-context learning? Investigations with linear models. _International Conference on Learning Representations_ , 2023. 

- [8] Y. Li, M. E. Ildiz, D. Papailiopoulos, and S. Oymak. Transformers as algorithms: Generalization and stability in in-context learning. _International Conference on Machine Learning_ , 2023. 

- [9] F. Falck, Z. Wang, and C. C. Holmes. Is in-context learning in large language models Bayesian? A martingale perspective. _Proceedings of the 41st International Conference on Machine Learning_ , PMLR 235:12784–12805, 2024. 

- [10] A. Jesson, N. Beltran-Vélez, and D. Blei. Can generative AI solve your in-context learning problem? A martingale perspective. _arXiv:2412.06033_ , 2024. 

- [11] Y. Lu, M. Bartolo, A. Moore, S. Riedel, and P. Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. _arXiv:2104.08786_ , 2021. 

- [12] Z. Zhao, E. Wallace, S. Feng, D. Klein, and S. Singh. Calibrate before use: Improving few-shot performance of language models. _International Conference on Machine Learning_ , 2021. 

- [13] A. Kazemnejad, I. Padhi, K. N. Ramamurthy, P. Das, and S. Reddy. The impact of positional encoding on length generalization in transformers. _Advances in Neural Information Processing Systems_ , 2023. 

- [14] O. Golovneva, T. Wang, J. Weston, and S. Sukhbaatar. Contextual position encoding: Learning to count what’s important. _arXiv:2309.08553_ , 2024. 

- [15] A. Vaswani et al. Attention is all you need. _Advances in Neural Information Processing Systems_ , 30, 2017. 

- [16] P. Shaw, J. Uszkoreit, and A. Vaswani. Self-attention with relative position representations. _NAACL-HLT_ , 2018. 

- [17] J. Su et al. RoFormer: Enhanced transformer with rotary position embedding. _arXiv:2104.09864_ , 2021. 

- [18] O. Press, N. A. Smith, and M. Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. _arXiv:2108.12409_ , 2021. 

- [19] J. Rissanen. Modeling by shortest data description. _Automatica_ , 14(5):465–471, 1978. 

- [20] P. D. Grünwald. _The Minimum Description Length Principle_ . MIT Press, 2007. 

- [21] R. E. Krichevsky and V. K. Trofimov. The performance of universal encoding. _IEEE Transactions on Information Theory_ , IT-27(2):199–207, 1981. 

- [22] Qwen Team. Qwen2.5 technical report. _arXiv:2412.15115_ , 2024. 

- [23] A. Grattafiori et al. The Llama 3 herd of models. _arXiv:2407.21783_ , 2024. 

- [24] J. Hewitt and P. Liang. Designing and interpreting probes with control tasks. _EMNLP-IJCNLP_ , 2019. 

- [25] A. Ravichander, Y. Belinkov, and E. Hovy. Probing the probing paradigm: Does probing accuracy entail task relevance? _arXiv:2005.00719_ , 2020. 

- [26] A. Geiger, H. Lu, T. Icard, and C. Potts. Causal abstractions of neural networks. _Advances in Neural Information Processing Systems_ , 34, 2021. 

- [27] A. Geiger, Z. Wu, H. Lu, J. Rozner, E. Kreiss, T. Icard, N. Goodman, and C. Potts. Inducing causal structure for interpretable neural networks. _International Conference on Machine Learning_ , 2022. 

- [28] K. Meng, D. Bau, A. Andonian, and Y. Belinkov. Locating and editing factual associations in GPT. _Advances in Neural Information Processing Systems_ , 35, 2022. 

- [29] L. Kuhn, Y. Gal, and S. Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. _International Conference on Learning Representations_ , 2023. 

- [30] J. Geng et al. A survey of confidence estimation and calibration in large language models. _NAACL_ , 2024. 

- [31] D. Hendrycks et al. Measuring massive multitask language understanding. _International Conference on Learning Representations_ , 2021. 

- [32] K. Cobbe, V. Kosaraju, M. Bavarian, J. Hilton, R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems. _arXiv:2110.14168_ , 2021. 

11 

- [33] A. Srivastava et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _Transactions on Machine Learning Research_ , 2023. 

- [34] J. Thorne, A. Vlachos, C. Christodoulopoulos, and A. Mittal. FEVER: A large-scale dataset for fact extraction and verification. _NAACL-HLT_ , 2018. 

- [35] Z. Yang et al. HotpotQA: A dataset for diverse, explainable multi-hop question answering. _EMNLP_ , 2018. 

- [36] T. Kwiatkowski et al. Natural questions: A benchmark for question answering research. _Transactions of the Association for Computational Linguistics_ , 2019. 

- [37] A. Mallen, A. Asai, V. Zhong, R. Das, D. Khashabi, and H. Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. _ACL_ , 2023. 

## **A Proofs and auxiliary formal statements** 

### **A.1 Proof of the prequential comparison identity** 

For any realized sequence, 



Taking expectation under _P_ and conditioning on _Ht_ = _Y<t_ gives Equation (2). For the KL form, use the chain rule 



and the analogous identity with _B_ in place of _Q_ . Subtracting expectations under _P_ gives Equation (3). If _P_ = _B_ , then _pt_ = _bt_ , and the conditional expectation at each prefix is exactly KL( _bt_ ( _· | Ht_ ) _∥qt_ ( _· | Ht_ )). 

### **A.2 Proof of the order-averaging decomposition** 

Fix _Zt_ and suppress it from notation. Since _q_ ¯( _y_ ) = E _gq_<sup>_g_</sup> ( _y_ ), 



The first term is KL( _b∥q_ ¯) and the second is _Jt_ ( _Zt_ ). Non-negativity follows from Jensen’s inequality, because log E _gq_<sup>_g_</sup> ( _y_ ) _≥_ E _g_ log _q_<sup>_g_</sup> ( _y_ ) for each _y_ . Summing the identity over _t_ and taking expectation over _Zt_ gives Equation (8). 

### **A.3 Proof of the dispersion-price corollary** 

Pinsker’s inequality in natural-log units gives TV( _b, q_<sup>_g_</sup> )<sup>2</sup> _≤_<sup><u>1</u></sup> 2<sup>KL(</sup><sup>_b∥qg_).By the triangle inequality,</sup> 



so ( _a_ + _b_ )<sup>2</sup> _≤_ 2 _a_<sup>2</sup> + 2 _b_<sup>2</sup> gives 



Taking expectation over independent _g, g_<sup>_′_</sup> and applying Pinsker proves 



### **A.4 Proof of the candidate-event decomposition** 

Because _Q_ ( _Ei | h_ ) = _MQ_ ( _E | h_ )¯ _qi_ , 



12 

### **A.5 Proof of the safe-code floor lemma** 

The lower bound _qi_<sup>_λ≥λ/m_is immediate.Since</sup><sup>_q_</sup> _i_<sup>_λ≥_(1</sup><sup>_−λ_)</sup><sup>_qi_,</sup> 



Also _qi_<sup>_λ≥λui_gives</sup> 



Summing the first overhead bound over _t_ gives the stated condition for a vanishing floor schedule. 

### **A.6 Proof of the finite-alphabet KT certificate** 

For a finite alphabet, the KT/Dirichlet(1 _/_ 2 _, . . . ,_ 1 _/_ 2) sequential code satisfies the standard redundancy bound 



for any i.i.d. source _p_ [19, 21, 20]. For any predictor _Q_ , 



Taking expectation gives Equation (14). If the comparison term is bounded above by _Rn_ , the code-length bound follows. 

**One sufficient finite-support condition.** Let _qt_<sup>_B_</sup> = _K_ KT( _· | Y<t_ ) and suppose a predictor _qt_ satisfies on the evaluated support 



The same argument as in the KT redundancy proof bounds the expected one-step replacement cost by _O|A|,C,c_ (( _t_ + 1)<sup>_−_1</sup><sup>_/_2</sup> ): for the realized symbol, 



and the Dirichlet-1 _/_ 2 predictive denominator gives a bounded expectation of _pa/qt_<sup>_B_(</sup><sup>_a_) after summing</sup> over symbols. Summing over _t_ yields _Rn_ = _O_ (<sup>_√_</sup> _<u>n</u>_ <u>) for fixed alphabet and constants.</u> This is the finite-support certificate reported in Table 2. 

### **A.7 Proof of the coarsened-predictive corollary** 

The induced variables _V_ ( _Zt_ ) form a discrete sequential process with predictive distributions _b_<sup>_V_</sup> _t_ and _qt_<sup>_V_, so Theorem 1 applies directly.If full densities exist, the partition map</sup><sup>_V_is a measurable</sup> coarsening. Data processing for KL gives KL( _b_<sup>_V_</sup> _t_<sup>_∥q_</sup> _t_<sup>_V_)</sup><sup>_≤_KL(</sup><sup>_bt∥qt_).</sup> 

### **A.8 Stylized positional mechanism** 

The main theorem does not assume a transformer mechanism for order sensitivity. The following calculation, adapted from a Bernoulli positional model, illustrates one sufficient source. Fix a length- _t_ Bernoulli prefix with _St_ ones and draw a uniform ordering _τ_ . Suppose the next-token logit depends on the ordering through 



and is _Lt_ -Lipschitz in this summary. Let 

13 

Then the standard deviation, over within-prefix orderings preserving ( _t, St_ ), of the expected log score is bounded by 



This is not used as a transformer theorem in the main text. It only shows how positional encodings can create task-preserving order dispersion while the prequential theory above prices whatever dispersion is actually observed. Both constants are observable on white-box models: _σ_ PE _,t_ is computed directly from the encoding and the context length, and _Lt_ can be estimated from the gradient norm of the candidate logit with respect to _ut,τ_ , so on the open-weight models audited here the bound is a measurable quantity rather than a free constant. The _t_<sup>_−_1</sup><sup>_/_2</sup> factor predicts that within-prefix order dispersion shrinks as the prefix grows; this is the controlled-microscope counterpart of the finitesupport order variances reported for the pretrained model in Table 6 and for the trained-from-scratch models in Section 4.4. 

## **B Mechanistic sketch for implicit pseudo-counts** 

Self-attention can implement the implicit pseudo-count picture in Table 7 by pooling token embeddings in a way that depends mainly on the multiset of observed symbols. If queries and keys separate ˆ _xi_ = 1 from _xi_ = 0, the head output approximates a function of _St_ (or _pt_ = _St/t_ ). Given features that track ( _t, St_ ), an MLP readout can approximate the Beta–Bernoulli predictive mean for some effective pseudo-counts ( _α_ 0 _, β_ 0): 



This sketch explains why the predictive-distribution audit target is plausible in a controlled architecture. The causal intervention in Section 4.5 tests whether a pretrained transformer uses the statistic. 

## **C Additional diagnostic and semantic-order results** 

### **C.1 Prefix-difference coefficients** 

The probes in Table 7 define availability and prefix-difference diagnostics on a real model. The relevant object is a prefix-state update. Let _rℓ_ ( _p_ ) denote the chosen readout residual at layer _ℓ_ for prefix _p_ . Let _G_ be the matrix whose entry _Gc,k_ is the projected prefix finite difference, conditioned on appending the observed symbol _c_ : 



An additive count diagnostic predicts _G ≈ I_ for Dirichlet-categorical, slope-1 for Poisson, and _G ≈_ **1** for Bernoulli. Define the additivity coefficient _α_ add := diag _G −_ max _i_ = _j |Gij|_ , and the concentration drift _β_ conc := median( _α_ eff _/α_ true) where _α_ eff is the effective prior concentration that minimises KL( _qB∥qM_ ). Operationally, the reported probes instantiate _rℓ_ at a fixed readout position and compare the distribution reached before and after appending the new observation. On Qwen2.5-7B Dirichlet ( _K_ = 3) at residual layer _ℓ_ = 3 the diagonal of _G_ is [1 _._ 016 _,_ 0 _._ 986 _,_ 1 _._ 002] with max _i_ = _j |Gij|_ = 0 _._ 020, giving _α_ add = 0 _._ 981; on Qwen2.5-7B Poisson–Gamma with _α_ true = 2 _._ 0 the per-sequence median fitted prior is _α_ eff = 3 _._ 5 ( _β_ conc = 1 _._ 75, with mean-correlation 0 _._ 923 between model and reference predictives indicating concentration drift with preserved direction). These coefficients are planning diagnostics: they indicate when count summaries or concentration errors are the next object to test. 

### **C.2 Activation-patching audit details** 

Table 8 reports Qwen2.5-7B activation patching with layer-specific learned subspaces. For each clean/corrupt pair, we compute the clean and corrupt candidate logit difference, filter pairs with small denominators, and score restoration as 



14 

Subspaces are fit on separate calibration pairs at each patched layer. The matched random-subspace control samples a random basis orthogonalised against the learned subspace and rescales its component to the learned component norm; the matched random-vector control uses a random vector with the same norm. Keybinding uses 256 evaluation pairs. Bernoulli-next generates 256 evaluation pairs and retains 229 after the denominator filter. 

The threshold-question BLR audit is a negative control for interface choice. When the prompt asks whether _y_ exceeds a threshold and scores Yes/No logits, the reference-threshold control does not align with the analytic survival-logit distribution even though final-layer hidden replacement reproduces source model distributions. This identifies the measurement as an interface mismatch. The sequencecompletion BLR audits below use raw continuation distributions and track posterior-predictive expectations directly. 

### **C.3 BLR pair-next-number audit** 

The BLR pair audit samples one-dimensional integer covariates _x ∈{_ 0 _, . . . ,_ 9 _}_ , generates noisy linear responses, rounds and clips _y_ to digits 0 _, . . . ,_ 9, and presents the support as a raw sequence 

### _x_ 1 _, y_ 1 _x_ 2 _, y_ 2 _· · · xm, ym x⋆,_ 

with no instruction and no chat template. The model distribution is the normalized next-token distribution over digits 0 _, . . . ,_ 9. The reference is the Gaussian BLR posterior predictive distribution integrated over the same digit bins. Table 10 reports the support sweep. 

|Support|Contexts|corr(E_θ_[_y_]_,_E_B_[_y_])|Dist. corr.|Top-1 match|E[_y_]MAE|
|---|---|---|---|---|---|
|25|256|0_._935|0_._808|0_._480|0_._459|
|50|256|0_._974|0_._886|0_._590|0_._299|
|100|512|0_._988|0_._934|0_._676|0_._207|



Table 10: **BLR pair-next-number audit.** The model is scored on the next digit after a raw pair sequence. Expectation and distributional correlations increase with support. 

### **C.4 Multivariate BLR tuple-next-number audit** 

The multivariate BLR audit uses the same next-token setup with two digit covariates. Each context samples _x, y ∈{_ 0 _, . . . ,_ 9 _}_ , generates a noisy linear response from a model with design vector [1 _, x, y_ ] after digit normalization, rounds and clips _z_ to digits 0 _, . . . ,_ 9, and presents the support as 

### _x_ 1 _, y_ 1 _, z_ 1 _x_ 2 _, y_ 2 _, z_ 2 _· · · xm, ym, zm x⋆, y⋆,_ 

again with no instruction and no chat template. The model distribution is the normalized next-token distribution over _z_ digits. The main reference is the Gaussian BLR posterior predictive distribution for the full [1 _, x, y_ ] design, integrated over the same digit bins. Misspecified posterior controls use x-only, y-only, and intercept-only designs. Table 11 reports the support sweep. 

|Support|Contexts|corr(E_θ_[_z_|]_,_E_B_[_z_])|Dist. corr.|Best controlE[_z_]corr.|Best control dist. corr.|Top-1 match|
|---|---|---|---|---|---|---|---|
|25|256||0_._880|0_._713|0_._891|0_._667|0_._465|
|50|256||0_._935|0_._789|0_._902|0_._713|0_._535|
|100|512||0_._965|0_._849|0_._881|0_._689|0_._557|



Table 11: **Multivariate BLR tuple-next-number audit.** The model’s next- _z_ digit distribution is compared with the full [1 _, x, y_ ] posterior predictive and the best misspecified control. At support 100, the full two-covariate reference is the best match. 

### **C.5 BLR predictive-regret support sweep** 

This sweep reports raw full-vocabulary predictive regret for the same pair and tuple continuations used in Tables 10 and 11. The KL column is KL( _KB∥Kθ_ ) in bits. The _ℓ_ 1 and _ϵ∞,A_ columns use raw candidate probabilities, with candidate-set leakage included in _ℓ_ 1. These distributions support the BLR reference-comparison claim; the floored discrete KT distribution is reported in Table 2. 

15 

|Model|Family|Support|Contexts|KL bits|_ℓ_1|_ϵ∞,A_|_c_min_,A_|
|---|---|---|---|---|---|---|---|
|7B|BLR pair-next-number|25|256|0_._388|0_._538|0_._539|0_._0571|
|7B|BLR pair-next-number|50|256|0_._238|0_._398|0_._453|0_._0296|
|7B|BLR pair-next-number|100|256|0_._131|0_._288|0_._320|0_._0092|
|7B|BLR tuple-next-number|25|256|0_._669|0_._719|0_._775|0_._0183|
|7B|BLR tuple-next-number|50|256|0_._439|0_._549|0_._801|0_._0069|
|7B|BLR tuple-next-number|100|256|0_._318|0_._462|0_._659|0_._0151|
|14B|BLR pair-next-number|25|256|0_._309|0_._468|0_._459|0_._0357|
|14B|BLR pair-next-number|50|256|0_._251|0_._398|0_._516|0_._0107|
|14B|BLR pair-next-number|100|256|0_._160|0_._303|0_._385|0_._0102|
|14B|BLR tuple-next-number|25|256|0_._761|0_._769|0_._769|0_._0218|
|14B|BLR tuple-next-number|50|256|0_._456|0_._556|0_._772|0_._0143|
|14B|BLR tuple-next-number|100|256|0_._325|0_._470|0_._594|0_._0145|



Table 12: **BLR predictive-regret support sweep.** One-covariate and two-covariate BLR continuations are scored by raw next-token probability against the posterior predictive digit distribution. Increasing support reduces KL regret for both models and both BLR families. 

### **C.6 Semantic-order and calibration observations** 

Table 13 applies the same averaging operation to semantic-order settings. These cells measure semantic-distribution and trajectory-style variance reduction, separated from the exchangeability experiments in the main text. 

|Model|Task|Acc_k_=1|Acc_k_=16|NLL_k_=1|NLL_k_=16|
|---|---|---|---|---|---|
|Qwen2.5-7B-Instruct|BBH-navigate|0_._650|0_._644|1_._862|1_._468|
|Qwen2.5-7B-Instruct|BBH-logical-deduction|0_._730|n/a|0_._836|n/a|
|Qwen2.5-7B-Instruct|BBH-tracking|0_._260|0_._266|2_._628|2_._471|
|Qwen2.5-7B-Instruct|GSM8K-CoT-reorder|0_._990|1_._000|0_._052|0_._014|



Table 13: **Semantic-order controls.** These cells separate semantic-distribution and trajectory-style variance reduction from the exchangeable, task-preserving ordering experiments. 

Calibration and accuracy are tracked separately from predictive-mixture NLL. In the randompermutation 1200-item MMLU audit, canonical accuracy is 0 _._ 6983 and mixture accuracy is 0 _._ 7008, while the same-sample NLL comparison improves from 0 _._ 7235 to 0 _._ 7212. In the GSM8K-derived multiple-choice run, NLL, Brier, and ECE improve monotonically from _k_ = 1 to _k_ = 8 while accuracy stays near 0 _._ 43. In distributional terms, this is a margin/entropy transformation of _Kθ_ ( _· | p_ ): the top token can remain stable while probability mass and logit gaps move under averaging. This is consistent with prior calibration observations for instruction-tuned models [11, 12, 29, 30]. 

## **D Full task-preserving ordering mixture tables** 

|_k_|Accuracy|NLL|Brier|ECE|
|---|---|---|---|---|
|1|0_._4324_±_0_._0033|1_._5424_±_0_._0056|0_._8052_±_0_._0023|0_._3204_±_0_._0023|
|2|0_._4263_±_0_._0028|1_._5161_±_0_._0050|0_._7983_±_0_._0019|0_._3159_±_0_._0029|
|4|0_._4295_±_0_._0026|1_._5018_±_0_._0026|0_._7940_±_0_._0012|0_._3096_±_0_._0028|
|8|0_._4292_±_0_._0017|1_._4948_±_0_._0018|0_._7922_±_0_._0008|0_._3097_±_0_._0014|



Table 14: GSM8K-derived few-shot multiple choice. Averaging random demonstration orderings improves NLL, Brier, and ECE while leaving accuracy nearly flat, consistent with order-averaged predictive scoring. 

16 

||Qwen2.5-7B-Instruct|Llama-3.1-8B-Instruct|
|---|---|---|
|Dispersion slope_b_(vs. log_n_)|0_._377([0_._319_,_ 0_._435])|0_._147([0_._109_,_ 0_._184])|
|Order-averaging gain (nats/token)|0_._1041<br>|0_._00982<br>|
|Mixture optimality gap (nats/token)|_<_10<sup>_−_4</sup>|_≤_5_._3_×_10<sup>_−_5</sup>|



Table 15: Evidence-grounded QA under evidence permutations. Both instruction-tuned models show order dispersion and positive order-averaging gains from uniform permutation mixtures. Uniform weights are within the measured optimization tolerance. 

## **E Reproducibility** 

Table 16 summarizes the experimental configuration. All experiments use bfloat16 inference on a single A100 (40 GB) or H100 (80 GB) GPU per cell, with a fixed prompt-sampling seed ( `SEED=0` ). For reproducibility, a complete supplementary package should include anonymized code, configurations, prompt templates, seeds, and per-cell JSON outputs; any de-anonymized release should be deferred until camera-ready. Dataset manifests for MMLU, BBH, GSM8K are used under their published licenses; FEVER, HotpotQA, NQ-Open, and PopQA are used under CC-BY-SA 4.0 / CC-BY 4.0 / MIT respectively. 

## **F Statistical analysis** 

Bootstrap confidence intervals use the percentile method, resampling independent units: items for the pretrained position intervention and MMLU audit, paired examples for activation patching, sequences for Bernoulli gap experiments, and tasks for ICL _k_ -sweeps. MMLU paired deltas use sign-flip _p_ -values over item-level deltas. Model comparison on gap-vs- _n_ uses weighted least squares with inverse-variance weights _wn_ = _Nn/_ ∆<sup>ˆ2</sup> _n_<sup>.</sup> 

## **G Computational resources** 

The reported open-weight experiments use bfloat16 or 4-bit single-GPU inference on A100 40 GB, H100 80 GB, or equivalent 7B–14B workers. The position intervention and causal-patching cells require model-internal access; scoring cells require saved next-token log probabilities on the audited candidate sets. Resampling is performed over saved per-item or per-example outputs, so bootstrap and sign-flip intervals do not require model re-execution. 

17 

|Component|Confguration|
|---|---|
|PE intervention (Table 6)|Qwen2.5-7B<br>with<br>fxed<br>pretrained<br>weights;<br>inference-only<br>`position_ids` modes {normal,<br>demo-local,<br>random demo off-<br>sets}; 64items_×_16permutations per task and mode; support size16;<br>Bayesian prediction and semantic-frst tasks; 2,000 bootstrap resamples<br>over independent items.|
|From-scratch PE ablation<br>(Section 4.4)|<br>Small transformers trained on i.i.d. Bernoulli sequences,_d_=128,4layers,<br>4heads, pooled`[CLS]`readout (permutation-invariant with no encoding);<br>positional schemes {none, learned absolute, sinusoidal, RoPE, ALiBi};<br>AdamW,2_,_000steps, batch128, learning rate3_×_10<sup>_−_4</sup>, prefx lengths<br>sampled up to the context, Bernoulli rates from a Jeffreys Beta( <sup>1</sup><br>2<sup>_,_ 1</sup><br>2<sup>)</sup><br>prior; within-prefx order variance from500permutations at_t ∈{_20_,_40_}_;<br>KT regret at_n_=200with60sequences and20permutations per sequence<br>across_p ∈{_0_._05_,_0_._1_,_0_._3_,_0_._5_,_0_._7_,_0_._9_,_0_._95_}_.|
|Probes (Table 7)|<br>scikit-learn`Ridge`(_α_=1);_N_seq _∈{_150_,_300_}_prompts of length_T_=100;<br>75_/_25train/val split; per-layer_R_<sup>2</sup>, best layer reported.|
|Bayesian and BLR distribu-<br>tion audits (Tables 2and12)|<br>Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct; symbol, pair, and tu-<br>ple continuations; raw full-vocabulary candidate-token probabilities<br>and conditional candidate distributions saved; foored discrete distri-<br>butions use _qθ,λ_ with _λ_ = 0_._01 candidate-uniform fooring; supports<br>1_,_2_,_4_,_8_,_16_,_32_,_64_,_128_,_256for Bernoulli and categorical distributions<br>and25_,_50_,_100for BLR distributions; reported metrics areKL(_KB∥qθ,λ_)<br>for foored discrete distributions,KL(_KB∥Kθ_)for BLR distributions, full-<br>distribution_ℓ_1,_ϵ∞,A_, <sup>�</sup>_CA_,_c_min_,A_, candidate mass, and support-binned<br>JSON outputs.|
|BLR pair-next-number (Ta-<br>ble 10)|<br>Qwen2.5-7B-Instruct,<br>bfoat16/4-bit<br>inference;<br>raw<br>prompts<br>_x_1_, y_1<br>_x_2_, y_2<br>_· · ·_<br>_x⋆,_ with no instruction and no chat template;<br>_y_ candidates are digit tokens 0_, . . . ,_9; supports 25_,_50_,_100 with<br>|
||256_,_256_,_512contexts respectively.|
|Multivariate BLR tuple-next-<br>number (Table 11)|<br>Qwen2.5-7B-Instruct,<br>bfoat16/4-bit<br>inference;<br>raw<br>prompts<br>_x_1_, y_1_, z_1<br>_· · ·_<br>_x⋆, y⋆,_ with no instruction and no chat template;<br>_z_ candidates are digit tokens 0_, . . . ,_9;<br>supports 25_,_50_,_100 with<br>256_,_256_,_512 contexts respectively; x-only, y-only, and intercept-only<br>posterior controls are scored against the same model distributions.|
|Causal statistic use (Table 8)|<br>Qwen2.5-7B activation patching; layer-specifc rank-4 clean-minus-<br>corrupt subspaces;<br>full-vector, learned-subspace, matched-random-<br>subspace, and matched-random-vector interventions; layers 12_,_24_,_26<br>swept with layers24_,_26reported; keybinding_n_=256evaluation pairs;<br>Bernoulli-next256generated and229retained after the denominator flter;<br>2000 bootstra resamles over aired examles|
|Cross-task<br>(Table<br>9,<br>Ta-<br>ble 14)|, p p  p p.<br>MMLU random-permutation audit on Qwen2.5-7B:1200items,8demon-<br>strations, 16 true random unique demonstration permutations per item,<br>raw per-order probabilities saved, paired item bootstrap CIs and sign-fip<br>_p_-values. GSM8K-derived multiple choice uses_k ∈{_1_,_2_,_4_,_8_}_random<br>demonstration orderings on Qwen2.5-7B-Instruct.|
|Evidence QA (Table 15)|<br>Banded permutations with_B_=6contiguous bands;_m_= 12unique per-<br>mutations per item for the 7B model and _m_ = 16 for the 8B model;<br>uniform mixture vs. optimized convex weights.|



Table 16: **Per-component reproducibility configuration.** 

18 

