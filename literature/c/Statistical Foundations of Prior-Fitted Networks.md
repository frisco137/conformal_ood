**Statistical Foundations of Prior-Data Fitted Networks** 

## **Thomas Nagler**<sup>1 2</sup> 

# **Abstract** 

Prior-data fitted networks (PFNs) were recently proposed as a new paradigm for machine learning. Instead of training the network to an observed training set, a fixed model is pre-trained offline on small, simulated training sets from a variety of tasks. The pre-trained model is then used to infer class probabilities in-context on fresh training sets with arbitrary size and distribution. Empirically, PFNs achieve state-of-the-art performance on tasks with similar size to the ones used in pre-training. Surprisingly, their accuracy further improves when passed larger data sets during inference. This article establishes a theoretical foundation for PFNs and illuminates the statistical mechanisms governing their behavior. While PFNs are motivated by Bayesian ideas, a purely frequentistic interpretation of PFNs as pre-tuned, but untrained predictors explains their behavior. A predictor’s variance vanishes if its sensitivity to individual training samples does and the bias vanishes only if it is appropriately localized around the test feature. The transformer architecture used in current PFN implementations ensures only the former. These findings shall prove useful for designing architectures with favorable empirical behavior. 

# **1. Introduction** 

## **1.1. PFNs in a Nutshell** 

Prior-data fitted networks (PFNs) were proposed by Muller¨ et al. (2022) as a new approach to machine learning, motivated by ideas from Bayesian nonparametrics and metalearning. The goal is to compute a posterior predictive distribution (PPD) for a test feature given observed training data. To approximate the PPD, a transformer network is 

> 1Department of Statistics, LMU Munich, Munich, Germany 

> 2Munich Center for Machine Learning, Munich, Germany. Correspondence to: Thomas Nagler _<_ t.nagler@lmu.de _>_ . 

_Proceedings of the 40_<sup>_th_</sup> _International Conference on Machine Learning_ , Honolulu, Hawaii, USA. PMLR 202, 2023. Copyright 2023 by the author(s). 

trained offline through meta-learning. After simulating several training data sets from a variety of tasks, a transformer network imitating the PPD on these sets is trained. After this pre-training phase, the network is treated as fixed. In the inference phase, a fresh training set and some test features are passed to the pre-trained network, which computes predictions for the test labels in a single forward pass. 

This approach is different from usual machine learning methods. Here, one would set up a model for the relationship between label and feature, and train the model parameters on a specific data set. The main benefit of PFNs is that no training or tuning is necessary at the inference stage, and predictions are delivered in split seconds. 

## **1.2. Empirical Findings** 

Empirically, Muller et al.¨ (2022) found that PFNs can indeed approximate a given PPD and perform well on real prediction tasks. While this pilot study was limited to tiny data sets, the TabPFN model of Hollmann et al. (2022) made a leap forward to classification tasks on moderately large tabular data sets. In particular, they pre-train a network with simulated data sets of size up to _n ≈_ 1000 and report state-of-the-art performance on several benchmarks. And surprisingly, the network’s predictions continue to improve at the inference stage when fed data sets with more than 1000 samples. This is an example of _in-context learning (ICL)_ : a pre-trained network learns from the context provided in the prompt (here: the fresh training data) without updating its parameters. 

## **1.3. Summary** 

The main contribution of this work is establishing the theoretical foundation for PFNs and identifying statistical mechanisms explaining their empirical behavior. 

- **Theoretical framework** (Section 2): As a preliminary step, we give precise definitions of the PPD, the statistical model behind it, and its PFN approximation. 

- **When PPDs can learn** (Section 3): Since PPDs are the main motivation behind PFNs, we first ask when a PPD can learn from training data. This can be approached from the perspective of Bayesian nonparametrics (Ghosal & van der Vaart, 2017). If the prior 

1 

**Statistical Foundations of Prior-Data Fitted Networks** 

- has large enough support and does not concentrate too much away from the true hypothesis, one can guarantee that the PPD converges to a close approximation of the true predictive distribution. 

- **How PFNs approximate PPDs** (Section 4): The optimal PFN approximation is characterized by a KullbackLeibler criterion. To allow for accurate approximation of the conditional class probabilities, we need sufficiently complex PFN models and prior. Practically, a PFN is trained on simulated data sets. The larger these data sets are, the more complex the PPD we approximate. The training set size can therefore be understood as a regularizer on the expected complexity of the network. The Monte-Carlo approximation of the optimal PFN is discussed briefly, but rather uneventful and of minor importance for PFNs’ inference behavior. 

- **Why PFNs can learn** (Section 5): The most intriguing question is: Why can a pre-trained network still learn in the inference phase? Although we know now why a PPD does, a PFN is not a valid PPD, and it is only trained to approximate the PPD for limited training sizes. The learning phenomenon can be understood through a purely frequentistic interpretation of PFNs as untrained predictors with many hyperparameters. During pre-training, these hyperparameters are tuned to be optimal for a set of tasks defined by the ‘prior’. Whether the PFN predictor can learn at inference depends on its structural properties. We show that the variance of a fixed network vanishes if its sensitivity to individual samples does, and that the network’s bias can only vanish if it is sufficiently localized. 

- **Insights on specific PFNs** (Section 6): We look at some specific PFN models: window smoothers, classification trees, and transformer networks. The examples cover cases where the bias is constant, increasing, or decreasing with _n_ . We show that if the model is well-designed, it can implicitly select or average over sub-models, making the bias decrease with the sample size. Transformer networks allow for vanishing variance and model selection through multi-head attention, but not for localization. However, TabPFN’s bias can be improved further with a simple post-hoc localization method. 

Section 7 concludes with suggestions for future research. All proofs are given in Appendix A. 

a sequence prediction task on a large corpus of text. When deployed, large models show the ability to solve tasks that they haven’t seen during pre-training (e.g., mathematical reasoning problems), only from the prompt context (Brown et al., 2020; Wei et al., 2022). In particular, no parameter updates are conducted after deployment. ICL has become a new paradigm for natural language processing and is intimately linked to the transformer architecture (Vaswani et al., 2017). Dong et al. (2023) provide an up-to-date survey of the large body of LLM-related research on in-context learning. 

**In-context-learning on numeric data** ICL has also been observed in more classical statistical learning tasks: classification and regression from tabular data. Muller et al.¨ (2022) proposed the concept of PFNs and illustrated the abilities of a transformer model on toy examples. The TabPFN model of Hollmann et al. (2022) implements a matured version of this idea and shows superb performance on benchmarks with small tabular data. Concurrently, Nguyen & Grover (2022) proposed Transformer Neural Processes following essentially the same idea. They also consider non- _iid_ settings and show promising results in applications to image completion, contextual bandits, and Bayesian Optimization. This paper illuminates the statistical foundations of such models in the _iid_ -setting and disentangles the prior from the model architecture. 

**Mechanics of transformer-based ICL** Garg et al. (2022) show that transformers can learn target functions generated from linear models, two-layer neural networks, and decision trees. Several other works provide arguments and experiments on how in-context learning emerges through implicit gradient descent (Dai et al., 2022; von Oswald et al., 2022; Akyurek et al.¨ , 2023). Olsson et al. (2022) identify a pattern of several attention heads working together, closely related to the discussion after Theorem 6.3 in this paper. Kirsch et al. (2022) experimentally investigates other architectural features (layers, memory, etc.). This work sheds further light on the mechanisms and architectural features enabling ICL. 

Overall, the current work complements the existing ICL literature, by providing a theoretical foundation for the empirical findings and deriving new insights from the perspective of statistical generalization theory. 

# **2. Theoretical Framework** 

## **2.1. Statistical Model** 

## **1.4. Related Work** 

**In-context-learning in large language models** The recent interest in ICL was spurred by the success of _large language models (LLMs)_ . These models are pre-trained on 

Consider a classification problem with class label _Y ∈Y_ and features **_X_** _∈X ⊆_ R<sup>_d_</sup> . Suppose we have _iid_ training data _Dn_ = ( _Yi,_ **_X_** _i_ )<sup>_n_</sup> _i_ =1<sup>from some distribution</sup><sup>_p_0. The goal</sup> is to predict the conditional class probabilities _p_ 0( _y |_ **_x_** ) = 

2 

**Statistical Foundations of Prior-Data Fitted Networks** 

P( _Y_ = _y |_ **_X_** = **_x_** ). From the perspective of Bayesian nonparametrics, we view _p_ 0 as a realization of a random, infinite-dimensional parameter _p ∈P_ with distribution Π. The distribution Π is called _prior_ and expresses our beliefs about _p_ before seeing any data. Under this model, data sets _Dn ∪_ ( _Y,_ **_X_** ) are generated by the following mechanism: 

1. Draw _p ∼_ Π. 

2. Draw _iid_ samples _Dn_ = ( _Yi,_ **_X_** _i_ )<sup>_n_</sup> _i_ =1<sup>and (</sup><sup>_Y,_</sup><sup>**_X_**) from</sup> model _p_ . 

## **2.2. Posterior Predictive Distribution** 

For every _n_ , the statistical model gives the tuple ( _Dn ∪_ ( _Y,_ **_X_** ) _, p_ ) a well-defined joint distribution. For every _n_ , we can then approximate _p_ 0( _y |_ **_x_** ) by the posterior predictive distribution (PPD) 



This defines a family of PPDs indexed by _n_ . If the prior Π factorizes into independent parts for _p_ ( _y |_ **_x_** ) and _p_ ( **_x_** ), the PPD can be written as 



where the _posterior_ Π( _· | Dn_ ) is the conditional distribution of _p_ given the data _Dn_ . The PPD _π_ ( _y |_ **_x_** _, Dn_ ) is then simply the posterior mean over conditional distributions _p_ ( _y |_ **_x_** ). 

_Remark_ 2.1 _._ In their implementation of PFNs, Muller et al.¨ (2022) and Hollmann et al. (2022) use priors that factorize as above, but do not mention it explicitly to justify (1). Priors that do not factorize this way would lead to a different form of _π_ ( _y |_ **_x_** _, D_ ): 



Here, observing the test feature **_x_** would be informative about the conditional distribution _p_ ( _y |_ **_x_** ), which is unintuitive. 

## **2.3. PFNs** 

A PFN is a numerical approximation of the family of PPDs. It is based on the insight that, for all _n_ , the PPDs maximize the expected conditional likelihood 



where EΠ is an expectation over ( _Y,_ **_X_** ) _∪Dn_ generated as in Section 2.1 (see, M¨uller et al., 2022, Section 3). 





_denote the set of all conditional probability functions. Then π in_ (1) _satisfies_ 



_Remark_ 2.3 _._ Maximizing (2) can also be interpreted as minimizing expected KL divergence between _q_ ( _· |_ **_X_** _, Dn_ ) and _π_ ( _· |_ **_X_** _, Dn_ ). 

To approximate the PPDs, we train a model _q_ **_θ_** parametrized by **_θ_** . To be precise, for every parameter value **_θ_** , there is an entire family of functions 



but we shall not make this explicit in notation. To find the best parameters for given PPDs, Muller et al.¨ (2022) propose to solve 



where Π _N_ is a probability distribution over the sample size _N_ . The expectation over _N_ makes _q_ **_θ_** _∗_ mimic the _family_ of PPDs, not just its _n_ th element. 

The model _q_ **_θ_** will normally be misspecified; that is, there is no parameter **_θ_** such that _q_ **_θ_** equals _π_ . In this case, (3) defines a KL-optimal approximation of _π_ over the class _{q_ **_θ_** : **_θ_** _∈_ Θ _}_ . In practice, the expectation in (3) is approximated by Monte-Carlo integration, i.e., averaging over _iid_ data sets ( _Yj,_ **_X_** _j_ ) _∪D_<sup>(</sup><sup>_j_)</sup> of size _Nj_ + 1 generated as in Section 2.1 and _Nj ∼_ Π _N_ . We approximate **_θ_**<sup>_∗_</sup> by solving 



This is of course an idealization of the training process. Sophisticated PFNs are large and usually trained in a single epoch. The maximum in (4) is never reached. This does not affect the main results of the following sections, which largely consider arbitrary **_θ_** . 

_Remark_ 2.4 _._ Hollmann et al. (2022) use a transformer network (Vaswani et al., 2017) for _q_ **_θ_** . For such architectures, any fixed network _q_ **_θ_** accepts an arbitrary number of feature vectors **_x_** 1 _, . . . ,_ **_x_** _n_ test and a data set _Dn_ of arbitrary length. The output _q_ **_θ_** ( _· |_ **_x_** 1 _, . . . ,_ **_x_** _n_ test _, Dn_ ) are _n_ test vectors of conditional class probabilities. Each vector contains predictions for the conditional class probabilities _p_ ( _· |_ **_x_** _j_ ). The test size _n_ test is irrelevant in what follows, so we take _n_ test = 1 for simplicity. 

# **3. When PPDs can Learn** 

The PPDs 



3 

**Statistical Foundations of Prior-Data Fitted Networks** 

are fully characterized by the prior Π. If _Dn_ is a data set generated from _p_ 0, we hope that Π( _p |_ **_x_** _, Dn_ ) concentrates around _p_ 0 as the size of _Dn_ increases. Setting a good prior is tricky in a nonparametric context. Finding a prior supporting a large enough subset of possible functions isn’t trivial. And even if, the prior may wash out very slowly or not at all if it puts too much mass in unfavorable regions (see, Ghosal & van der Vaart, 2017, Sections 1.2–1.3). But also if _p_ 0 is outside the support _P_ = _{p_ : Π( _p_ ) _>_ 0 _}_ of Π, PPDs can learn from data if the prior is sufficiently well-behaved: 

**Theorem 3.1.** _Under conditions (A1) and (A2), there is p_<sup>_∗_</sup> _∈P such that_ 



_for P_ 0 _-almost every_ ( _y,_ **_x_** ) _. Moreover, p_<sup>_∗_</sup> _is a KL-optimal approximation of p_ 0 _in P._ 

Exact conditions and a proof are given in Appendix A.2. If _P_ is sufficiently large, the KL-optimal _p_<sup>_∗_</sup> _∈P_ is close to _p_ 0. This explains why PPDs can learn when fed more data. If this was not the case, trying to approximate them with PFNs would be pointless. And the better we choose Π, the more attractive the PPDs become as an approximation target. 

# **4. PFN Approximation of the PPD** 

Four factors influence the PFN approximation (4): the data prior Π, the size prior Π _N_ , the model _q_ **_θ_** , and the MonteCarlo size _m_ . Since a PFN is pre-trained, the model class _{q_ **_θ_** : **_θ_** _∈_ Θ _}_ can be considered fixed relative to the number of Monte-Carlo sets _m_ . The approximation quality of **_θ_**<sup>�</sup> then follows from standard results on empirical risk minimization. In particular, we can expect **_θ_**<sup>�</sup> = **_θ_**<sup>_∗_</sup> + _Op_ ( _m_<sup>_−_1</sup><sup>_/_2</sup> ), see Appendix B for more details. The other factors are more interesting. 

If Π consists of only simple models, the optimal PFN _q_ **_θ_** _∗_ likely also produces only simple functions of ( _y,_ **_x_** ). Conversely, simple models _{q_ **_θ_** : **_θ_** _∈_ Θ _}_ cannot benefit much from complex Π. For the pre-trained PFN to work well on diverse tasks, we need sufficient capacity in both _{q_ **_θ_** : **_θ_** _∈_ Θ _}_ and Π. 

When pre-training the PFN via (4), we sample data sets _D_<sup>(</sup><sup>_j_)</sup> with random sample size _Nj_ . Let us define the KL-optimal parameter **_θ_** _n_<sup>_∗_for a given sample size:</sup> 



The PPD _π_ ( _y |_ **_x_** _, Dn_ ) we are trying to approximate changes with _n_ . Hence, the KL-optimal parameter **_θ_** _n_<sup>_∗_may change</sup> with _n_ as well. Seen as a function of ( _y,_ **_x_** ), we should expect the complexity of _π_ ( _y |_ **_x_** _, Dn_ ) to increase in _n_ . Similarly, we should expect the parameter **_θ_** _n_<sup>_∗_to favor more</sup> complex models. At the other extreme, _n_ = 1, the true PPD 

is close to the average model in our prior and normally close to constant. The training set sizes _Nj_ can thus be seen as a regularizer on model complexity. By optimizing an average over random sizes _Nj_ , **_θ_** _∗_ also averages **_θ_** _N_<sup>_∗_</sup> _j_<sup>.Thedistri-</sup> bution Π _N_ lets us emphasize some ranges of sample sizes more than others. The TabPFN of Hollmann et al. (2022) was trained with a uniform distribution over _{_ 1 _, . . . ,_ 1023 _}_ for Π _N_ . The restriction to small sample sizes has computational reasons: the cost of evaluating a transformer network scales quadratically in _Nj_ . 

Since TabPFN has never seen sample sizes larger than around 1000 during pre-training, it is curious that it improves its predictions when fed larger data sets. Whether such behavior occurs depends in a non-obvious way on the family _{q_ **_θ_** �( _· | ·, Dn_ ) _, n ∈_ N _}_ . The family learned by TabPFN seems to have some structure that allows extrapolating nicely to larger _n_ . This structure may come from the architecture of the network _q_ **_θ_** or from learning **_θ_**<sup>_∗_</sup> for a given Π. The following section looks closer into the mechanisms at play. 

# **5. Why PFNs can Learn In-Context** 

There is no reason to believe the PFN behaves like a PPD family for some (implicit) prior when encountering sample sizes never seen in pre-training. So even though PPDs serve as a theoretical motivation for PFNs, Theorem 3.1 does not apply to _q_ **_θ_** �. So why does a PFN _q_ **_θ_** � pre-trained on up to 1000 samples improve when feeding larger data sets during inference? 

## **5.1. PFNs as Untrained Predictors** 

To understand what is going on, we take a frequentist perspective. For any data size _n_ encountered at inference, we may treat the pre-tuned network _q_ **_θ_** �( _y |_ **_x_** _, ·_ ) as an untrained predictor for _p_ 0( _y |_ **_x_** ), i.e., a function ( _Y × X_ )<sup>_n_</sup> _→PY |_ **_X_** that maps a data set _Dn_ to an element of the space _PY |_ **_X_** of conditional distribution functions. In that view, **_θ_** is a collection of tuning parameters of the predictor, selected through meta-learning in the pre-training phase. Further, the ‘priors’ Π and Π _N_ are simply distributions over tasks for which we want the predictor to do well. 

Now decompose the estimation error into bias and variance components: 



Empirically, the error above decreases with _n_ . So what structural features of PFNs can explain this? 

4 

**Statistical Foundations of Prior-Data Fitted Networks** 

## **5.2. Symmetry** 

Standard transformers are symmetric functions of the individual samples in _Dn_ . If the samples in _Dn_ are _iid_ , this is most natural. 

**Lemma 5.1.** _Let f_ : ( _Y × X_ )<sup>_n_</sup> _→PY |_ **_X_** _be any predictor. Then there is a symmetrized version f_<sup>�</sup> _of f such that, for every probability measure P ,_ 



Thus, using symmetric _f_ is optimal in an MSE sense. However, symmetry itself does not have any meaningful consequences for learning. For example, _q_ ( _y |_ **_x_** _, Dn_ ) = 1 _/|Y|_ is a symmetric function that is incapable of learning anything. 

## **5.3. Variance and Diminishing Sensitivity** 

There is other structure we can reasonably expect from _q_ **_θ_** . When passed larger data sets _Dn_ , the influence of individual elements should diminish. This allows to bound the variance of the predictor _q_ **_θ_** . Formally, suppose there are _α >_ 0 and _L < ∞_ , such that for large enough _n_ and almost all data sets _Dn, Dn_<sup>_′_differing only in one sample,</sup> 



**Theorem 5.2.** _If_ (5) _holds, then_ 



_with high probability._ 

If _α >_ 1 _/_ 2, we get lim _n→∞ n_<sup>1</sup><sup>_/_2</sup><sup>_−α_</sup> = 0, so the variance caused by _Dn_ vanishes. In that case, the difference above vanishes almost surely. 

**Lemma 5.3.** _If_ (5) _holds with α >_ 1 _/_ 2 _, then_ 



This only partially explains how pre-trained PFNs can still learn at inference. The remaining error is due to bias. 

## **5.4. Bias and the Need for Locality** 

The bias is determined by the behavior of the sequence E[ _q_ **_θ_** ( _y |_ **_x_** _, Dn_ )]. It is reasonable to assume that 



for some function _<u>q</u>_ **_θ_** . Without a specific model _q_ **_θ_** at hand, we cannot say much more. In Section 6 we shall see examples where the bias is constant, and other examples where the bias decreases or increases with _n_ . 

We can give necessary conditions for a vanishing bias, however. A predictor that has vanishing bias on a sufficiently rich class of functions must be _local_ : asymptotically, only samples ( _Yi,_ **_X_** _i_ ) _∈Dn_ with **_X_** _i_ close to **_x_** should contribute to _q_ **_θ_** ( _y |_ **_x_** _, Dn_ ). 

**Theorem 5.4.** _Let P be a set of distributions. Suppose that for every p ∈P,_ 



_If_ (5) _holds, there is a sequence ϵn →_ 0 _for every p_ � _∈P, such that almost surely,_ 





So if _q_ **_θ_** is unbiased for rich enough _P_ , we can flip the labels of samples away from **_x_** almost arbitrarily without changing the behavior of _q_ **_θ_** ; only samples ( _Yi,_ **_X_** _i_ ) with **_X_** _i_ close to **_x_** matter. 

The result bears little meaning if the class _P_ is too small, and meaningless if _P_ contains only one _p_ . Even for rich _P_ , it only provides necessary conditions for a vanishing bias. A constant predictor _q_ **_θ_** = 1 is local in the sense of (7), but its bias does not change with _n_ . However, if _P_ is rich and the bias does vanish for all _p ∈P_ , the predictor _q_ **_θ_** can effectively only use _ϵnn_ = _o_ ( _n_ ) samples, so (5) is unlikely to hold with _α_ = 1. This is in line with the lower bounds on the bias-variance trade-off derived in Derumigny & Schmidt-Hieber (2020). 

# **6. Insights on Specific PFNs** 

We now consider some concrete examples of PFNs _q_ **_θ_** , to shed further light on the factors facilitating learning in the inference phase. Before turning to transformer networks, we briefly discuss two simpler models to illustrate some key mechanisms. The following result will be helpful. 

**Lemma 6.1.** _Let g be a function bounded by K < ∞ and_ 



_for some sequence An_ ( **_x_** ) _⊂X . If_ 



_for some η ∈_ (0 _,_ 1 _/_ 2) _and c >_ 0 _, then q_ **_θ_** _satisfies the conditions of Theorem 5.2 with α_ = 1 _− η and L_ = 4 _K/c._ 

5 

**Statistical Foundations of Prior-Data Fitted Networks** 

## **6.1. Window Smoother** 

For _θ ∈_ (0 _, ∞_ ), define 



This corresponds to a window smoother with bandwidth _θ_ . Then _θ_<sup>_∗_</sup> is the KL-optimal bandwidth for datasets from the prior. The fitted PFN is therefore just a window smoother with its hyperparameter tuned to such data sets. According to Lemma 6.1, _qθ_ satisfies (5) with _α_ = 1. The bias 



is constant, but optimized for data sizes from Π _×_ Π _N_ . Despite constant bias, the PFN learns from more data at inference, but only through reducing its variance. 

Now consider some sequence ( _an_ ) _n∈_ N and 



Lemma 6.1 yields that Theorem 5.2 holds with _α_ = 1. The bias is 



which is independent of _n_ . To reduce the bias as in the previous example, we would need to grow the number _S_ of split locations with _n_ . But the model is considered fixed in the inference phase and we cannot change the number of parameters. 

Instead, we could set up an ensemble of classification trees with Bayesian model averaging. Let _q_ **_θ_** 1 _, . . . , q_ **_θ_** _K_ be classification trees as above, and 



where 



and 

If _an_ increases with _n_ , the width of the smoothing window does too. This choice of _qθ_ isn’t sensible, of course, as the bias 



typically increases with _n_ . If we instead choose _an →_ 0 and _p_ 0 is sufficiently smooth, we can get rid of the bias. The scaling _an_ = _n_<sup>_−_1</sup><sup>_/_(4+</sup><sup>_d_)</sup> is known to be optimal in an MSE sense (e.g., Wand & Jones, 1994), with squared bias and variance decreasing at rate _n_<sup>_−_4</sup><sup>_/_(4+</sup><sup>_d_)</sup> . Indeed, we have _an_ P0( _∥_ **_X_** _−_ **_x_** _∥ < anθ_ ) _→ p_ 0( **_x_** ), so this is the convergence rate implied by Lemma 6.1 and Theorem 5.2. The hyperparameter _θ_<sup>_∗_</sup> reduces to a prefactor, tuned to be (asymptotically) optimal for data sets generated from Π of arbitrary size. 

### 

To keep the notation simple, suppose for the moment that _X ⊆_ R. Define 



as a classification tree with parameters **_θ_** _∈_ R<sup>_S_</sup> and _θ_ 0 = _−∞, θS_ +1 = + _∞_ by convention. The (hyper-)parameters are the split locations of the tree. The split locations are trained offline, to work best on sets from the prior Π _×_ Π _N_ . 



As _n_ grows, the model _q_ **_θ_** drifts towards a weighted average of the best-performing ensemble members. We can thus expect the bias to reduce with _n_ , approaching the bias of the best ensemble members. The role of the hyperparameters **_θ_** is the same as for a single tree. But now, the KL-optimal parameter **_θ_**<sup>_∗_</sup> likely induces more complex and diverse ensemble members. 

## **6.3. Transformer Networks** 

We now consider a transformer network with one layer. Let _Dn_ = _{_ **_V_** _i}_<sup>_n_</sup> _i_ =1<sup>with</sup><sup>**_V_**</sup><sup>_i_=(</sup><sup>_Yi,_</sup><sup>**_X_**</sup><sup>_i_)</sup><sup>_∈{_0</sup><sup>_,_1</sup><sup>_} ×_R</sup><sup>_d_and</sup> **_v_** = (0 _,_ **_x_** ). Similar<sup>1</sup> to Thickstun (2021), define 



1Some scaling and redundancies in the parametrization have been deliberately removed. They help for training the network, but not its theoretical analysis. 

6 

**Statistical Foundations of Prior-Data Fitted Networks** 

where _Wq_<sup>(</sup><sup>_h_)</sup> _, Wv_<sup>(</sup><sup>_h_)</sup> _∈_ R<sup>(</sup><sup>_d_+1)</sup><sup>_×_(</sup><sup>_d_+1)</sup> , _Wr,_ 1 _, Wr,_<sup>_⊤_</sup> 2 _∈_ R<sup>_m×_(</sup><sup>_d_+1)</sup> , and _Wo ∈_ R<sup>_|Y|×_(</sup><sup>_d_+1)</sup> . The parameter **_θ_** collects all these matrices. The SoftMax, LayerNorm, and ReLu operations are defined as 



with exp and max acting componentwise on vectors. Here and in everything that follows, the norm _∥· ∥_ is understood as _∥· ∥_ 2 for both vectors and matrices. 

The first two equations describe an _attention mechanism_ with _H_ heads (Vaswani et al., 2017). By definition, _a_<sup>(</sup> 1<sup>_h_)</sup> + _· · ·_ + _a_<sup>(</sup> _n_<sup>_h_)</sup> = 1. The idea is that within every head, the attention weights _a_<sup>(</sup> _j_<sup>_h_)</sup> emphasize specific samples **_V_** _j ∈ Dn_ . Emphasis is put on those **_V_** _j_ that are ‘similar’ to **_v_** in a sense measured by **_v_**<sup>_⊤_</sup> _Wq_<sup>(</sup><sup>_h_)</sup> **_V_** _j_ . Each attention head allows for a different definition of similarity. With the help of Theorem 5.2 and Theorem A.1, we can show that the variance of this predictor vanishes. 

**Theorem 6.2.** _For X_ = _{_ **_x_** : _∥_ **_x_** _∥ ≤ K} and ∥Wq_<sup>(</sup><sup>_h_)</sup> _∥, ∥Wv_<sup>(</sup><sup>_h_)</sup> _∥, ∥Wr,_ 1 _∥, ∥Wr,_ 2 _∥, ∥Wo∥ < ∞, it holds_ 



## _with high probability._ 

The variance vanishes irrespective of the parameter **_θ_** . This is because the attention mechanism necessarily gives diminishing weight to individual samples (see the proof in Appendix A.8). 

The bias depends heavily on the choice of **_θ_** . 

**Theorem 6.3.** _Under the assumptions of Theorem 6.2,_ 



_where_ _<u>qθ</u>_ ( _· |_ **_x_** ) _is defined as q_ **_θ_** ( _· |_ **_x_** _, Dn_ ) _, but with_ **_u_**<sup>_′_</sup> _replaced by_ 



The measure _gh_ is an exponentially tilted version of _p_ 0 (Siegmund, 1976). The tilt can be understood as an infinitesimal 

form of the attention mechanism. Relative to _p_ 0, the tilted measure lifts the likelihood of values **_s_** that are similar to **_v_** (with similarity measured by **_v_**<sup>_⊤_</sup> _Wq_<sup>(</sup><sup>_h_)</sup> **_s_** ) and discounts the likelihood of others. Each attention head _h_ assesses a certain aspect of the unknown distribution _p_ 0 (characterized by _Wq_<sup>(</sup><sup>_h_)</sup> ). If the matrices ( _Wq_<sup>(</sup><sup>_h_)</sup> )<sup>_H_</sup> _h_ =1<sup>arespecifiedwell,</sup> the aspect views distinguish distinct feature values. This localizes the predictor to some degree, but not in the sense of Theorem 5.4. Although we upweight the influence of samples “similar to” **_v_** , there always remains an influence of samples away from **_v_** . We cannot flip the labels of such samples without changing the predictor (asymptotically), so we should not expect the bias to vanish. 

Nevertheless, the limiting bias _<u>qθ</u>_ ( _y |_ **_x_** ) _− p_ 0( _y |_ **_x_** ) may be small if the remaining network processes _the sum_ of aspect summaries _Wv_<sup>(</sup><sup>_h_)</sup> E **_V_** _∼gh_ [ **_V_** ] into a good approximation of _p_ 0. The relevance of individual aspect summaries depends on the true measure _p_ 0, and less relevant aspects may also contribute less to the sum. On small samples, this effect is milder. At the extreme end, _n_ = 1, all attention weights _a_<sup>(</sup> _j_<sup>_h_)</sup> equal 1, so all aspects contribute equally. This suggests that the bias of the transformer network may decrease — provided the hyperparameters downstream make meaningful use of the aspect views. For example, Olsson et al. (2022) identified powerful patterns of several attention heads working together. This effect is similar to that of the model averaging layer in Section 6.2. Key to this is the presence of multiple attention heads ( _H >_ 1). However, this applies only to sample sizes the parameter **_θ_** has been tuned to. For larger sample sizes, there is no reason to expect a tuned network’s bias to decrease. 

## **6.4. Localized PFNs** 

According to Theorem 5.4, we need to localize the network to make its bias decrease. A simple post-hoc approach applicable to any pre-trained network is the following. To predict the label at a new feature **_x_** : 

1. Construct a reduced training set _Dn_ ( **_x_** ) by excluding all but the _kn_ nearest neighbors of **_x_** from _Dn_ . 

2. Predict the label that maximizes _q_ **_θ_** ( _· |_ **_x_** _, Dn_ ( **_x_** )). 

Intuitively, restricting to a neighborhood is like stretching/flattening the target _p_ ( _y | ·_ ) at the cost of a reduction in sample size. Flatter functions are easier to approximate. This is the mechanism behind the window smoother from Section 6.1. If the model _qθ_ approximates constant functions well, localization should improve the bias. 

## **6.5. Numerical Validation** 

Since the key mechanism acting on _Dn_ remains intact if we add more layers to the network, the findings likely transfer to 

7 

**Statistical Foundations of Prior-Data Fitted Networks** 



<!-- Start of picture text -->
average squared bias average variance<br>0.006<br>0.02<br>0.004<br>0.01<br>0.002<br>0.000 0.00<br>0 1000 2000 3000 4000 0 1000 2000 3000 4000<br>n<br>method TabPFN Localized TabPFN<br><!-- End of picture text -->

_Figure 1._ Average squared bias and variance of the pre-trained TabPFN of Hollmann et al. (2022) on simulated data sets. 

larger networks. The main predictions from our theoretical considerations are: ( _i_ ) the variance vanishes at rate 1 _/n_ , ( _ii_ ) the bias does not vanish, but decreases until _n ≈_ 1000. To confirm this empirically, we simulate 500 data sets _Dn_ from the model _p_ 0(1 _|_ **_X_** ) = 1 _/_ 2 + sin( **1**<sup>_⊤_</sup> **_X_** ) _/_ 2 with _Y ∈ {_ 0 _,_ 1 _}_ , **_X_** _∼N_ ( **0** _, I_ 5), and run the pre-trained TabPFN of Hollmann et al. (2022, pip version 0.1.8).<sup>2</sup> We compute the average squared bias and variance over 100 samples **_X_** test _∼N_ ( **0** _, I_ 5). The results in Figure 1 confirm that the variance indeed decreases at rate 1 _/n_ and that the bias decreases until _n ≈_ 1000, but does not vanish. 

The analysis shows that, for larger sample sizes at inference, TabPFN learns mainly through decreasing its variance. This variance reduction is a consequence of the transformer architecture and takes place irrespective of the tuned parameters **_θ_**<sup>�</sup> . Figure 1 also shows the results of a localized version of TabPFN (as in Section 6.4 with _kn_ = min _{_ 500 _, ⌈n_<sup>4</sup><sup>_/_(</sup><sup>_d_+4)</sup> _⌉}_ ). Here, the bias continues to decrease beyond _n_ = 1000 at the cost of a slightly larger variance. 

# **7. Discussion** 

As explained in Section 5.1, the prior Π characterizes tasks we want the predictor to do well on. Hollmann et al. (2022) propose a new kind of prior based on structural causal models (Pearl, 2009) that is interesting on its own. Their intuitive idea is that the pair ( _Y,_ **_X_** ) is generated by some 

> 2An R script to reproduce the results can be found at https://gist.github.com/tnagler/ 62f6ce1f996333c799c81f1aef147e72. 

noisy, causal mechanism (not necessarily in the direction **_X_** _→ Y_ ). Because the mechanisms can be arbitrarily complex, the prior is essentially nonparametric. The rate at which corresponding posteriors contract is a complex issue (Ghosal & van der Vaart, 2017, Chapters 8–9) and poses an interesting open question. 

The key factors driving PFNs capability to learn are their sensitivity to individual samples, their ability to choose submodels, and localization. These insights may help to inform architecture design. In Section 6.1 and Section 6.4, we found a way to make the bias vanish at the cost of increased sensitivity to the training instances. This was achieved by introducing a scaling of the tuning parameters adapted to the training set size _n_ . Whether this is possible and what a good scaling is depends on the model architecture. The localization approach in Section 6.4 is simple and can be applied post-hoc to any pre-trained network _qθ_ . More serious architecture design should account for the entire training pipeline and computational efficiency. Additional improvements can be expected if localization is incorporated into pre-training. Thinking about ways to adapt the transformer architecture appropriately could be a promising path. Another possible improvement is to augment the architecture with a Bayesian averaging mechanism similar to Section 6.2. 

Hollmann et al. (2022) acknowledge constraints on the feature dimension and sample size as a major limitation of current PFN implementations. Owing to the standard transformer architecture, the maximal feature size is fixed, and the algorithm scales quadratically in the number of samples. To mitigate this, several works proposed scalable modifications of the transformer architecture (Beltagy et al., 2020; 

8 

**Statistical Foundations of Prior-Data Fitted Networks** 

Zaheer et al., 2020; Kitaev et al., 2020). Hollmann et al. (2022) rightfully point out that PFNs are quick enough to be used as ensemble members. The size constraints could therefore be overcome by boosting and bagging techniques akin to random forests or boosted trees. 

The full potential of PFNs is yet to be explored. 

# **Acknowledgements** 

The author is grateful for several helpful comments by Samuel Muller, Noah Hollmann, Thibault Vatter, and two¨ anonymous referees. 

# **References** 

- Akyurek,¨ E., Schuurmans, D., Andreas, J., Ma, T., and Zhou, D. What learning algorithm is in-context learning? investigations with linear models. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum? id=0g0X4H8yN4I. 

- Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer. _arXiv preprint arXiv:2004.05150_ , 2020. 

- Boucheron, S., Lugosi, G., and Massart, P. _Concentration inequalities: A nonasymptotic theory of independence_ . Oxford university press, 2013. 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), _Advances in Neural Information Processing Systems_ , volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper_files/paper/2020/file/ 

- 1457c0d6bfcb4967418bfb8ac142f64a-Paper. pdf. 

- Dai, D., Sun, Y., Dong, L., Hao, Y., Sui, Z., and Wei, F. Why can gpt learn in-context? language models secretly perform gradient descent as meta optimizers. _arXiv preprint arXiv:2212.10559_ , 2022. 

- De Blasi, P. and Walker, S. G. Bayesian asymptotics with misspecified models. _Statistica Sinica_ , pp. 169–187, 2013. 

- Derumigny, A. and Schmidt-Hieber, J. On lower bounds for the bias-variance trade-off. _arXiv preprint arXiv:2006.00278_ , 2020. 

- Dong, Q., Li, L., Dai, D., Zheng, C., Wu, Z., Chang, B., Sun, X., Xu, J., Li, L., and Sui, Z. A survey on in-context learning, 2023. 

- Gao, B. and Pavel, L. On the properties of the softmax function with application in game theory and reinforcement learning. _arXiv preprint arXiv:1704.00805_ , 2017. 

- Garg, S., Tsipras, D., Liang, P. S., and Valiant, G. What can transformers learn in-context? a case study of simple function classes. _Advances in Neural Information Processing Systems_ , 35:30583–30598, 2022. 

- Ghosal, S. and van der Vaart, A. _Fundamentals of nonparametric Bayesian inference_ , volume 44. Cambridge University Press, 2017. 

- Hollmann, N., Muller, S., Eggensperger, K., and Hutter, F.¨ TabPFN: A transformer that solves small tabular classification problems in a second. In _NeurIPS 2022 First Table Representation Workshop_ , 2022. URL https: //openreview.net/forum?id=eu9fVjVasr4. 

- Kirsch, L., Harrison, J., Sohl-Dickstein, J., and Metz, L. General-purpose in-context learning by meta-learning transformers, 2022. 

- Kitaev, N., Kaiser, L., and Levskaya, A. Reformer: The efficient transformer. _arXiv preprint arXiv:2001.04451_ , 2020. 

- McDiarmid, C. _On the method of bounded differences_ , pp. 148–188. London Mathematical Society Lecture Note Series. Cambridge University Press, 1989. doi: 10.1017/CBO9781107359949.008. 

- Muller, S., Hollmann, N., Arango, S. P., Grabocka, J., and¨ Hutter, F. Transformers can do bayesian inference. In _International Conference on Learning Representations_ , 2022. URL https://openreview.net/forum? id=KSugKcbNf9. 

- Nguyen, T. and Grover, A. Transformer neural processes: Uncertainty-aware meta learning via sequence modeling. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), _Proceedings of the 39th International Conference on Machine Learning_ , volume 162 of _Proceedings of Machine Learning Research_ , pp. 16569–16594. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/ v162/nguyen22b.html. 

- Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, 

9 

**Statistical Foundations of Prior-Data Fitted Networks** 

- A., Conerly, T., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., and Olah, C. In-context learning and induction heads. _Transformer Circuits Thread_ , 2022. https://transformer-circuits.pub/2022/in-contextlearning-and-induction-heads/index.html. 

Pearl, J. _Causality_ . Cambridge university press, 2009. 

- Siegmund, D. Importance Sampling in the Monte Carlo Study of Sequential Tests. _The Annals of Statistics_ , 4(4):673 – 684, 1976. doi: 10.1214/aos/ 1176343541. URL https://doi.org/10.1214/ aos/1176343541. 

- Thickstun, J. The transformer model in equations. _University of Washington, Tech. Rep_ , 2021. URL https://johnthickstun.com/docs/ transformers.pdf. 

- Vaart, A. W. and Wellner, J. A. Weak convergence. In _Weak convergence and empirical processes_ , pp. 16–28. Springer, 1996. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., and Vladymyrov, M. Transformers learn in-context by gradient descent. _arXiv preprint arXiv:2212.07677_ , 2022. 

- Wand, M. P. and Jones, M. C. _Kernel smoothing_ . CRC press, 1994. 

- Wei, J., Bosma, M., Zhao, V., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V. Finetuned language models are zero-shot learners. In _International Conference on Learning Representations_ , 2022. URL https: //openreview.net/forum?id=gEZrGCozdqR. 

- Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., et al. Big bird: Transformers for longer sequences. _Advances in Neural Information Processing Systems_ , 33:17283–17297, 2020. 

10 

**Statistical Foundations of Prior-Data Fitted Networks** 

# **A. Proofs** 

## **A.1. Proof of Theorem 2.2** 

By definition of the KL-divergence, 



or, equivalently, 



Since this holds for any ( **_x_** _, D_ ) it must also hold if we take expectations over random draws of ( **_x_** _, D_ ). Taking expectation with respect to ( **_x_** _, D_ ) _∼_ Π on both sides, the law of iterated expectations yields 



## **A.2. Proof of Theorem 3.1** 

Write KL( _f | f_<sup>_′_</sup> ) for the KL-divergence of _f_ relative to _f_<sup>_′_</sup> , and _H_ ( _f, f_<sup>_′_</sup> ) for their Hellinger distance. We need the following assumptions: 

(A1) There is a unique _p_<sup>_∗_</sup> _∈P_ with _p_<sup>_∗_</sup> = arg min _p_ KL( _p_<sup>_∗_</sup> _| p_ 0) and KL( _p_<sup>_∗_</sup> _| p_ 0) _< ∞_ . 

(A2) For every _α ∈_ (0 _,_ 1 _/_ 2), there are sets _B_ 1 _, . . . , BJ_ ( _α_ ) with 



Now let Π _n_ ( _A_ ) = � _A_<sup>_d_Π(</sup><sup>_p | Dn_) be the posterior measure.From our assumptions and Corollary 1 of De Blasi & Walker</sup> (2013) it follows that for all _ϵ >_ 0, 



with probability 1 over sequences _Dn_ . For some _δ >_ 0 and an arbitrary set _A ⊆Y × X_ with _µA_ = _P_<sup>_∗_</sup> ( _A_ ) _>_ 0, define 



For any _p ∈ Sδ,A_ , it holds 



where TV( _f, f_<sup>_′_</sup> ) is the total variation distance. Together with (8), this implies 



11 

**Statistical Foundations of Prior-Data Fitted Networks** 

We then get 



Since _δ_ and _A_ were arbitrary, we have shown that 



with probability 1 for _P_<sup>_∗_</sup> -almost every ( _y,_ **_x_** ). Since KL( _p_<sup>_∗_</sup> _| p_ 0) _< ∞_ by (A1), convergence must also take place for _P_ 0-almost every ( _y,_ **_x_** ). 

## **A.3. Proof of Lemma 5.1** 

Let _Rn_ be the set of permutations _ρ_ : ( _Y × X_ )<sup>_n_</sup> _→_ ( _Y × X_ )<sup>_n_</sup> . Define the symmetrized function _f_<sup>�</sup> as 



If the elements in _Dn_ are _iid_ , it holds E _P_ [ _f_ ( _Dn_ )] = E _P_ [( _f ◦ ρ_ )( _Dn_ )] and Var _P_ [ _f_ ( _Dn_ )] = Var _P_ [( _f ◦ ρ_ )( _Dn_ )] for any _ρ ∈ Rn_ . Therefore, E _P_ [ _f_ ( _Dn_ )] = E _P_ [ _f_<sup>�</sup> ( _Dn_ )] and 



with equality for all _P_ if and only if _f_ is symmetric. 

## **A.4. Proof of Theorem 5.2** 

McDiarmid’s inequality (McDiarmid, 1989) yields 



Choosing _ϵ_<sup>2</sup> = log( _δ_ ) _L_<sup>2</sup> _n_<sup>1</sup><sup>_−_2</sup><sup>_α_</sup> _/_ 2, we get 



## **A.5. Proof of Lemma 5.3** 

Using the first inequality from the previous proof, we get 



The Borel-Cantelli lemma then implies that, almost surely, 



Since _ϵ_ was arbitrary the claim follows. 

12 

**Statistical Foundations of Prior-Data Fitted Networks** 

## **A.6. Proof of Theorem 5.4** 

Condition (6) implies 



for all _p,_ � _p ∈P_ with _p_ ( _y |_ **_s_** ) = _p_ �( _y |_ **_s_** ) for _∥_ **_s_** _−_ **_x_** _∥ < ϵ_ . Lemma 5.3 then implies 



Since _ϵ_ was arbitrary, convergence must also hold for some sequence _ϵn →_ 0. 

## **A.7. Proof of Lemma 6.1** 

Let _Dn_ = ( _Yi,_ **_X_** _i_ )<sup>_n_</sup> _i_ =1<sup>and</sup><sup>_D_</sup> _n_<sup>_′_=(</sup><sup>_Y_</sup> _i_<sup>_′,_</sup><sup>**_X_**</sup> _i_<sup>_′_)</sup><sup>_n_</sup> _i_ =1<sup>suchthat(</sup><sup>_Yi,_</sup><sup>**_X_**</sup><sup>_i_)=(</sup><sup>_Y_</sup> _i_<sup>_′,_</sup><sup>**_X_**</sup> _i_<sup>_′_)forall</sup><sup>_i>_1.Hoeffdings’sinequality</sup> (Boucheron et al., 2013, Theorem 2.8) gives 



Since _η <_ 1 _/_ 2, 



and the Borell-Cantelli lemma implies that for large _n_ , 



The remaining inequalities are understood almost surely, for large enough _n_ . Because _n_<sup>_η_</sup> P _{_ **_X_** _i ∈ An_ ( **_x_** ) _} → c_ , we get 



which implies 



Using this bound, 



which proves the claim. 

## **A.8. Proof of Theorem 6.2** 

The theorem is a consequence of Theorem 5.2 and the following result. (The norm bounds on _∥_ **_x_** _∥_ and the weight matrices are arbitrary and can be relaxed.) 

**Theorem A.1.** _Let X_ = _{_ **_x_** : _∥_ **_x_** _∥≤_ 1 _} and ∥Wq_<sup>(</sup><sup>_h_)</sup> _∥, ∥Wv_<sup>(</sup><sup>_h_)</sup> _∥, ∥Wr,_ 1 _∥, ∥Wr,_ 2 _∥, ∥Wo∥≤_ 1 _. Then the network q_ **_θ_** _satisfies_ (5) _with α_ = 1 _and L_ = _O_ ( _H|γ_ 1 _|/|γ_ 2 _|_ ) _._ 

13 

**Statistical Foundations of Prior-Data Fitted Networks** 

_Proof._ Let _D_<sup>�</sup> _n_ = ( _Dn \_ **_V_** _n_ ) _∪_ **_V_**<sup>�</sup> _n_ and define � _a_<sup>(</sup> _j_<sup>_h_)</sup> , **_u_** �<sup>_′_</sup> , **_u_** �, **_z_** �<sup>_′_</sup> , **_z_** � accordingly. Because SoftMax is 1-Lipschitz (Gao & Pavel, 2017, Proposition 4), 



Using Lemma A.2 below, we further get 



Because also ReLu is 1-Lipschitz, 



Using Lemma A.2 again, 



The last displays together yield 



Defining **_V_**<sup>�</sup> _i_ = **_V_** _i_ for _i < n_ , we obtain 



Let _si_ = **_v_**<sup>_⊤_</sup> _Wq_<sup>(</sup><sup>_h_)</sup> **_V_** _i_ and note that _|si| ≤∥_ **_v_** _∥∥Wq_<sup>(</sup><sup>_h_)</sup> _∥_ max _j ∥_ **_V_** _j∥≤_ 4. 



Further, 



14 

**Statistical Foundations of Prior-Data Fitted Networks** 

The first term on the right is zero if _j_ = _n_ . For _j_ = _n_ , it can be bounded by 2 _e_<sup>8</sup> _/n_ as before. Using the same argument for the second term above, we get 



Accordingly, 



Combining (9) and (10) proves the claim. 

**Lemma A.2.** _For any two vectors_ **_a_** _,_ **_b_** _∈_ R<sup>_d_</sup> _it holds_ 



_Proof._ Let us first assume avg( **_a_** ) = avg( **_b_** ) = **0** and, without loss of generality, _∥_ **_a_** _∥≥∥_ **_b_** _∥_ . It holds, 



If avg( **_a_** ) = 0 or avg( **_b_** ) = 0, we get 



because 



where we used the triangle inequality in the second step and Jensen’s inequality in the last. 

## **A.9. Proof of Theorem 6.3** 

We have 



15 

**Statistical Foundations of Prior-Data Fitted Networks** 

The law of large numbers implies **_u_**<sup>_′_</sup> _→_ **_<u>u</u>_**<sup>_~~′~~_</sup> almost surely, where 



Now, observe that 



Since all following operations on **_u_**<sup>_′_</sup> are continuous (see the proof of Theorem A.1), it also holds 



Because the sequence _|q_ **_θ_** ( _· |_ **_x_** _, Dn_ ) _− q_ **_θ_** ( _· |_ **_x_** ) _|_ is uniformly bounded by 2, convergence is also in _L_ 1. This implies lim _n→∞_ E[ _q_ **_θ_** ( _· |_ **_x_** _, Dn_ )] = _<u>qθ</u>_ ( _· |_ **_x_** ) _._ 

# **B. Approximation Results for the PFN Parameter** 

Suppose that the parameters **_θ_** live in a subset of R<sup>_p_</sup> with _p_ fixed and finite. This assumption would be questionable in the usual machine learning setting, but our situation is different. A PFN is pre-trained offline, using _m_ Monte-Carlo samples from data sets _D_<sup>(</sup><sup>_j_)</sup> . Given enough computing power, we can take _m_ as large as we want. 

Given sufficient regularity, the following theorems are direct applications of existing results. To keep additional concepts and notation to a minimum, we omit detailed conditions and proofs and refer to the original works for specifics. We start with the behavior of **_θ_**<sup>�</sup> . 

## **Theorem B.1.** _It holds:_ 





_Proof._ See Corollary 3.2.3 and Example 3.2.12 in Vaart & Wellner (1996). 

The first part shows that **_θ_**<sup>�</sup> is a valid approximation of **_θ_**<sup>_∗_</sup> , the second quantifies its accuracy. Theorem B.1 has direct implications for the approximated model _q_ **_θ_** �. 

## **Theorem B.2.** _It holds:_ 



- _(ii)_<sup>_√_</sup> _<u>m</u>_ <u>(</u> _q_ **_θ_** � _− q_ **_θ_** _∗_ ) _converges weakly to a mean-zero Gaussian process with_ 



_Proof._ Part ( _i_ ) follows from Theorem B.1 and the continuous mapping theorem (Vaart & Wellner, 1996, Theorem 1.11.1 ), part ( _ii_ ) from the delta method (Vaart & Wellner, 1996, Theorem 3.9.4). 

16 

**Statistical Foundations of Prior-Data Fitted Networks** 

From the second part, we see that the variance of _q_ **_θ_** �( _y |_ **_x_** _, D_ ) is approximately 



Intuitively, the variance depends on the accuracy of **_θ_**<sup>�</sup> (through Σ **_θ_** _∗_ ), and the sensitivity of _q_ **_θ_** _∗_ with respect to **_θ_**<sup>_∗_</sup> (through _∇_ **_θ_** _q_ **_θ_** _∗_ ). Model complexity normally works against us in both parts. Hence, more complex models need to be trained with more Monte-Carlo samples to limit the variance. 

17 

