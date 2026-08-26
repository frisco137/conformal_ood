# **Linear Transformers are Versatile In-Context Learners** 

### **Max Vladymyrov** 

Google Research `mxv@google.com` 

### **Johannes von Oswald** 

Google, Paradigms of Intelligence Team `jvoswald@google.com` 

**Mark Sandler** Google Research `sandler@google.com` 

**Rong Ge** 

Duke University `rongge@cs.duke.edu` 

## **Abstract** 

Recent research has demonstrated that transformers, particularly linear attention models, implicitly execute gradient-descent-like algorithms on data provided incontext during their forward inference step. However, their capability in handling more complex problems remains unexplored. In this paper, we prove that each layer of a linear transformer maintains a weight vector for an implicit linear regression problem and can be interpreted as performing a variant of preconditioned gradient descent. We also investigate the use of linear transformers in a challenging scenario where the training data is corrupted with different levels of noise. Remarkably, we demonstrate that for this problem linear transformers discover an intricate and highly effective optimization algorithm, surpassing or matching in performance many reasonable baselines. We analyze this algorithm and show that it is a novel approach incorporating momentum and adaptive rescaling based on noise levels. Our findings show that even linear transformers possess the surprising ability to discover sophisticated optimization strategies. 

## **1 Introduction** 

The transformer architecture (Vaswani et al., 2017) has revolutionized the field of machine learning, driving breakthroughs across various domains and serving as a foundation for powerful models (Anil et al., 2023; Achiam et al., 2023; Team et al., 2023; Jiang et al., 2023). However, despite their widespread success, the mechanisms that drive their performance remain an active area of research. A key component of their success is attributed to in-context learning (ICL, Brown et al., 2020) – an emergent ability of transformers to make predictions based on information provided within the input sequence itself, without explicit parameter updates. 

Recently, several papers (Garg et al., 2022; Akyürek et al., 2022; von Oswald et al., 2023a) have suggested that ICL might be partially explained by an implicit meta-optimization of the transformers that happens on input context (aka mesa-optimization Hubinger et al., 2019). They have shown that transformers with linear self-attention layers (aka linear transformers) trained on linear regression tasks can internally implement gradient-based optimization. 

Specifically, von Oswald et al. (2023a) demonstrated that linear transformers can execute iterations of an algorithm similar to the gradient descent algorithm (which they call GD<sup>++</sup> ), with each attention layer representing one step of the algorithm. Later, Ahn et al. (2023); Zhang et al. (2023) further characterized this behavior, showing that the learned solution is a form of preconditioned GD, and this solution is optimal for one-layer linear transformers. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 

In this paper, we continue to study linear transformers trained on linear regression problems. We prove that each layer of every linear transformer maintains a weight vector for an underlying linear regression problem. Under some restrictions, the algorithm it runs can be interpreted as a complex variant of preconditioned gradient descent with momentum-like behaviors. 

While maintaining a linear regression model (regardless of the data) might seem restrictive, we show that linear transformers can discover powerful optimization algorithms. As a first example, we prove that in case of GD<sup>++</sup> , the preconditioner results in a second order optimization algorithm. 

Furthermore, we demonstrate that linear transformers can be trained to uncover even more powerful and intricate algorithms. We modified the problem formulation to consider mixed linear regression with varying noise levels<sup>1</sup> (inspired by Bai et al., 2023). This is a harder and non-trivial problem with no obvious closed-form solution, since it needs to account for various levels of noise in the input. 

Our experiments with two different noise variance distributions (uniform and categorical) demonstrate the remarkable flexibility of linear transformers. Training a linear transformer in these settings leads to an algorithm that outperforms GD<sup>++</sup> as well as various baselines derived from the exact closedform solution of the ridge regression. We discover that this result holds even when training a linear transformer with diagonal weight matrices. 

Through a detailed analysis, we reveal key distinctions from GD<sup>++</sup> , including momentum-like term and adaptive rescaling based on the noise levels. 

Our findings contribute to the growing body of research where novel, high-performing algorithms have been directly discovered through the reverse-engineering of transformer weights. This work expands our understanding of the implicit learning capabilities of attention-based models and highlights the remarkable versatility of even simple linear transformers as in-context learners. We demonstrate that transformers have the potential to discover effective algorithms that may advance the state-of-the-art in optimization and machine learning in general. 

## **2 Preliminaries** 

In this section we introduce notations for linear transformers, data, and type of problems we consider. 

### **2.1 Linear transformers and in-context learning** 

Given input sequence _e_ 1 _, e_ 2 _, ..., en ∈_ R<sup>_d_+1</sup> , a single head in a linear self-attention layer is usually parameterized by four matrices, key _WK_ , query _WQ_ , value _WV_ and projection _WP_ . The output of the non-causal layer at position _i_ is _ei_ + ∆ _ei_ where ∆ _ei_ is computed as 



Equivalently, one can use parameters _P_ = _WP WV_ and _Q_ = _WK_<sup>_⊤WQ_, and the equation becomes</sup> 



Multiple heads ( _P_ 1 _, Q_ 1) _,_ ( _P_ 2 _, Q_ 2) _, ...,_ ( _Ph, Qh_ ) simply sum their effects 



We define a _linear transformer_ as a multi-layer neural network composed of _L_ linear self-attention layers parameterized by _θ_ = _{Q_<sup>_l_</sup> _k_<sup>_, P_</sup> _k_<sup>_l}_for</sup><sup>_k_= 1</sup><sup>_. . . H, l_= 1</sup><sup>_. . . L_.To isolate the core mechanisms,</sup> we consider a simplified decoder-only architecture, excluding MLPs and LayerNorm components. This architecture was also used in previous work (von Oswald et al., 2023a; Ahn et al., 2023). 

We consider two versions of linear transformers: FULL with the transformer parameters represented by full matrices and DIAG, where the parameters are restricted to diagonal matrices only. 

Inspired by von Oswald et al. (2023a), in this paper we consider regression data as the token sequence. Each token _ei_ = ( _xi, yi_ ) _∈_ R<sup>_d_+1</sup> consists of a feature vector _xi ∈_ R<sup>_d_</sup> and its corresponding output 

1We consider a model where each sequence contains data with the same noise level, while different sequences have different noise levels. 

2 

_yi ∈_ R. Additionally, we append a query token _en_ +1 = ( _xt,_ 0) to the sequence, where _xt ∈_ R<sup>_d_</sup> represents test data. The goal of in-context learning is to predict _yt_ for the test data _xt_ . We constrain the attention to only focus on the first _n_ tokens of the sequence so that it ignores the query token. 

We use ( _x_<sup>_l_</sup> _i_<sup>_, y_</sup> _i_<sup>_l_) to denote the</sup><sup>_i_-th token in the transformer’s output at layer</sup><sup>_l_. The initial layer is simply</sup> the input: ( _x_<sup>0</sup> _i_<sup>_, y_</sup> _i_<sup>0) = (</sup><sup>_xi, yi_).For a model with parameters</sup><sup>_θ_, we read out the prediction by taking the</sup> ˆ negative<sup>2</sup> of the last coordinate of the final token in the last layer as _yθ_ ( _{e_ 1 _, ..., en}, en_ +1) = _−yn_<sup>_L_</sup> +1<sup>.</sup> Let’s also define the following notation to be used throughout the paper 



### **2.2 Noisy regression model** 

As a model problem, we consider data generated from a noisy linear regression model. For each input sequence _τ_ , we sample a ground-truth weight vector _wτ ∼ N_ (0 _, I_ ), and generate _n_ data points as _xi ∼ N_ (0 _, I_ ) and _yi_ = _⟨wτ , xi⟩_ + _ξi_ , with noise _ξi ∼ N_ (0 _, στ_<sup>2).</sup> 

Note that each sequence can have different ground-truth weight vectors _wτ_ , but every data point in the sequence shares the same _wτ_ and _στ_ . The query is generated as _xt ∼ N_ (0 _, I_ ) and _yt_ = _⟨wτ , xt⟩_ (since the noise is independent, whether we include noise in _yq_ will only be an additive constant to the final objective). 

We further define an ordinary least square (OLS) loss as 



The OLS solution is _w_<sup>_∗_</sup> := Σ<sup>_−_1</sup> _α_ with residuals _ri_ := _yi −⟨w_<sup>_∗_</sup> _, xi⟩_ . 

In the presence of noise _στ_ , _w_<sup>_∗_</sup> in general is not equal to the ground truth _wτ_ . For a _known_ noise level _στ_ , the best estimator for _wτ_ is provided by ridge regression: 



_−_ 1 with solution _wσ_<sup>_∗_2:=</sup> �Σ + _στ_<sup>2</sup><sup>_I_</sup> � _α_ . Of course, in reality the variance of the noise is not known and has to be estimated from the data. 

### **2.3 Fixed vs. mixed noise variance problems** 

We consider two different problems within the noisy linear regression framework. 

**Fixed noise variance.** In this scenario, the variance _στ_ remains constant for all the training data. Here, the in-context loss is: 



where _ei_ = ( _xi, yi_ ) and _yi_ = _⟨wτ , xi⟩_ + _ξi_ . This problem was initially explored by Garg et al. (2022). Later, von Oswald et al. (2023a) have demonstrated that a linear transformer (6) converges to a form of a gradient descent solution, which they called GD<sup>++</sup> . We define this in details later. **Mixed noise variance.** In this case, the noise variance _στ_ is drawn from some fixed distribution _p_ ( _στ_ ) for each sequence. The in-context learning loss becomes: 



2 _l_ We set the actual prediction to _−yn_ +1<sup>, similar to von Oswald et al. (2023a), because it’s easier for linear</sup> transformers to predict _−yt_ . 

3 

In other words, each training sequence _τ_ has a fixed noise level _στ_ , but different training sequences have different noise levels sampled from a specified distribution _p_ ( _στ_ ). This scenario adds complexity because the model must predict _wτ_ for changing noise distribution, and the optimal solution likely would involve some sort of noise estimation. We have found that empirically, GD<sup>++</sup> fails to model this noise variance and instead converges to a solution which can be interpreted as a single noise variance estimate across all input data. 

## **3 Related work** 

**In-context Learning as Gradient Descent** Our work builds on research that frames in-context learning as (variants of) gradient descent (Akyürek et al., 2022; von Oswald et al., 2023a). For 1-layer linear transformer, several works Zhang et al. (2023); Mahankali et al. (2023); Ahn et al. (2023) characterized the optimal parameters and training dynamics. More recent works extended the ideas to auto-regressive models (Li et al., 2023; von Oswald et al., 2023b) and nonlinear models (Cheng et al., 2023). Fu et al. (2023) noticed that transformers perform similarly to second-order Newton methods on linear data, for which we give a plausible explanation in Theorem 5.1. 

**In-context Learning in LLMs** There are also many works that study how in-context learning works in pre-trained LLMs (Kossen et al., 2023; Wei et al., 2023; Hendel et al., 2023; Shen et al., 2023). Due to the complexity of such models, the exact mechanism for in-context learning is still a major open problem. Several works (Olsson et al., 2022; Chan et al., 2022; Akyürek et al., 2024) identified induction heads as a crucial mechanism for simple in-context learning tasks, such as copying, token translation and pattern matching. 

**Other theories for training transformers** Other than the setting of linear models, several other works (Garg et al., 2022; Tarzanagh et al., 2023; Li et al., 2023; Huang et al., 2023; Tian et al., 2023a,b) considered optimization of transformers under different data and model assumptions. Wen et al. (2023) showed that it can be difficult to interpret the “algorithm” performed by transformers without very strong restrictions. 

**Mixed Linear Models** Several works observed that transformers can achieve good performance on a mixture of linear models (Bai et al., 2023; Pathak et al., 2023; Yadlowsky et al., 2023). While these works show that transformers _c_ an implement many variants of model-selection techniques, our result shows that linear transformers solve such problems by discovering interesting optimization algorithm with many hyperparameters tuned during the training process. Such a strategy is quite different from traditional ways of doing model selection. Transformers are also known to be able to implement strong algorithms in many different setups (Guo et al., 2023; Giannou et al., 2023). 

**Effectiveness of linear and kernel-like transformers** A main constraint on transformer architecture is that it takes _O_ ( _N_<sup>2</sup> ) time for a sequence of length _N_ , while for a linear transformer this can be improved to _O_ ( _N_ ). Mirchandani et al. (2023) showed that even linear transformers are quite powerful for many tasks. Other works (Katharopoulos et al., 2020; Wang et al., 2020; Schlag et al., 2021; Choromanski et al., 2020) uses ideas similar to kernel/random features to improve the running time to almost linear while not losing much performance. 

## **4 Linear transformers maintain linear regression model at every layer** 

While large, nonlinear transformers can model complex relationship, we show that linear transformers are restricted to maintaining a linear regression model based on the input, in the sense that the _l_ -th layer output is always a linear function of the input with latent (and possibly nonlinear) coefficients. 

4 

**Theorem 4.1.** _Suppose the output of a linear transformer at l-th layer is_ ( _x_<sup>_l_</sup> 1<sup>_, y_</sup> 1<sup>_l_)</sup><sup>_,_(</sup><sup>_x_</sup> 2<sup>_l, y_</sup> 2<sup>_l_)</sup><sup>_, ...,_(</sup><sup>_xl_</sup> _n_<sup>_, y_</sup> _n_<sup>_l_)</sup><sup>_,_(</sup><sup>_x_</sup> _t_<sup>_l, y_</sup> _t_<sup>_l_)</sup><sup>_, then there exists matrices M l, vectors ul, wland scalars al_</sup> _such that_ 



Note that _M_<sup>_l_</sup> , _u_<sup>_l_</sup> _, w_<sup>_l_</sup> and _a_<sup>_l_</sup> are not linear in the input, but this still poses restrictions on what the linear transformers can do. For example we show that it cannot represent a quadratic function: 

**Theorem 4.2.** _Suppose the input to a linear transformer is_ ( _x_ 1 _, y_ 1) _,_ ( _x_ 2 _, y_ 2) _, ...,_ ( _xn, yn_ ) _where xi ∼ N_ (0 _, I_ ) _and yi_ = _w_<sup>_⊤_</sup> _xi, let the l-th layer output be_ ( _x_<sup>_l_</sup> 1<sup>_, y_</sup> 1<sup>_l_)</sup><sup>_,_(</sup><sup>_xl_</sup> 2<sup>_, y_</sup> 2<sup>_l_)</sup><sup>_, ...,_(</sup><sup>_xl_</sup> _n_<sup>_, y_</sup> _n_<sup>_l_)</sup><sup>_andlet_</sup> _y_<sup>_l_</sup> = ( _y_ 1<sup>_l, ..., y_</sup> _n_<sup>_l_)</sup><sup>_and y∗_= (</sup><sup>_x_1(1)2</sup><sup>_, x_2(1)2</sup><sup>_, ..., xn_(1)2)</sup><sup>_(here xi_(1)</sup><sup>_is just the first coordinate of xi),_</sup> _then when n ≫ d with high probability the cosine similarity of y_<sup>_∗_</sup> _and y_<sup>_l_</sup> _is at most 0.1._ 

Theorem 4.1 implies that the output of linear transformer can always be explained as linear combinations of input with latent weights _a_<sup>_l_</sup> and _w_<sup>_l_</sup> . The matrices _M_<sup>_l_</sup> , vectors _u_<sup>_l_</sup> _, w_<sup>_l_</sup> and numbers _a_<sup>_l_</sup> are not linear and can in fact be quite complex, which we characterize below: 

**Lemma 4.3.** _In the setup of Theorem 4.1, if we let_ 



_then one can recursively compute matrices M_<sup>_l_</sup> _, vectors u_<sup>_l_</sup> _, w_<sup>_l_</sup> _and numbers a_<sup>_l_</sup> _for every layer using_ 



_with the init. condition a_<sup>0</sup> = 1 _, w_<sup>0</sup> = 0 _, M_<sup>0</sup> = _I, u_<sup>0</sup> = 0 _._ 

The updates to the parameters are complicated and nonlinear, allowing linear transformers to implement powerful algorithms, as we will later see in Section 5. In fact, even with diagonal _P_ and _Q_ , they remain flexible. The updates in this case can be further simplified to a more familiar form: 

**Lemma 4.4.** _In the setup of Theorem 4.1 with diagonal parameters, u_<sup>_l_</sup> _, w_<sup>_l_</sup> _are updated as_ 



_Here_ Λ<sup>_l_</sup> _,_ Γ<sup>_l_</sup> _, s_<sup>_l_</sup> _,_ Π<sup>_l_</sup> _,_ Φ<sup>_l_</sup> _are matrices and numbers that depend on M_<sup>_l_</sup> _, u_<sup>_l_</sup> _, a_<sup>_l_</sup> _, w_<sup>_l_</sup> _in Lemma 4.3._ 

Note that Σ � _a_<sup>_l_</sup> _w_<sup>_∗_</sup> _− w_<sup>_l_�</sup> is (proportional to) the gradient of a linear model _f_ ( _w_<sup>_l_</sup> ) =<sup>�</sup><sup>_n_</sup> _i_ =1<sup>(</sup><sup>_alyi −_</sup> _⟨w_<sup>_l_</sup> _, xi⟩_ )<sup>2</sup> . This makes the updates similar to a gradient descent with momentum: 



Of course, the formula in Lemma 4.4 is still much more complicated with matrices in places of _β_ and _η_ , and also including a gradient term for the update of _w_ . 

## **5 Power of diagonal attention matrices** 

Although linear transformers are constrained, they can solve complex in-context learning problems. Empirically, we have found that they are able to very accurately solve linear regression with mixed noise variance (7), with final learned weights that are very diagonal heavy with some low-rank component (see Fig. 4). Surprisingly, the final loss remains remarkably consistent even when their _Q_ and _P_ matrices (3) are diagonal. Here we will analyze this special case and explain its effectiveness. 

5 

Since the elements of _x_ are permutation invariant, a diagonal parameterization reduces each attention heads to just four parameters: 



It would be useful to further reparametrize the linear transformer (3) using: 



This leads to the following diagonal layer updates: 



Four variables _ωxx_<sup>_l_,</sup><sup>_ω_</sup> _xy_<sup>_l_,</sup><sup>_ω_</sup> _yx_<sup>_l_,</sup><sup>_ω_</sup> _yy_<sup>_l_represent information flow between the data and the labels across</sup> layers. For instance, the term controlled by _ωxx_<sup>_l_measures information flow from</sup><sup>_xl_to</sup><sup>_xl_+1,</sup><sup>_ω_</sup> _yx_<sup>_l_</sup> measures the flow from _x_<sup>_l_</sup> to _y_<sup>_l_+1</sup> and so forth. Since the model can always be captured by these 4 variables, having many heads does not significantly increase its representation power. When there is only one head the equation _ωxx_<sup>_lω_</sup> _yy_<sup>_l_=</sup><sup>_ω_</sup> _xy_<sup>_lω_</sup> _yx_<sup>_l_is always true, while models with more than one head</sup> do not have this limitation. However empirically even models with one head is quite powerful. 

### **5.1 GD**<sup>++</sup> **and least squares solver** 

GD<sup>++</sup> , introduced in von Oswald et al. (2023a), represents a linear transformer that is trained on a fixed noise variance problem (6). It is a variant of a diagonal linear transformer, with all the heads satisfying _qy,k_<sup>_l_= 0.Dynamics are influenced only by</sup><sup>_ω_</sup> _xx_<sup>_l_and</sup><sup>_ω_</sup> _yx_<sup>_l_, leading to simpler updates:</sup> 



The update on _x_ acts as preconditioning and the update on _y_ performs gradient descent on the data. 

While existing analysis by Ahn et al. (2023) has not yielded fast convergence rates for GD<sup>++</sup> , we show here that it is actually a second-order optimization algorithm for the least squares problem (4): **Theorem 5.1.** _Given_ ( _x_ 1 _, y_ 1) _, ...,_ ( _xn, yn_ ) _,_ ( _xt,_ 0) _where_ Σ _has eigenvalues in the range_ [ _ν, µ_ ] _with a condition number κ_ = _ν/µ. Let w_<sup>_∗_</sup> _be the optimal solution to least squares problem_ (4) _, then there exists hyperparameters for GD_<sup>++</sup> _algorithm that outputs_ ˆ _y with accuracy |y_ ˆ _−⟨xt, w_<sup>_∗_</sup> _⟩| ≤ ϵ∥xt∥∥w_<sup>_∗_</sup> _∥ in l_ = _O_ (log _κ_ +log log 1 _/ϵ_ ) _steps. In particular that implies there exists an l-layer linear transformer that can solve this task._ 

The convergence rate of _O_ (log log 1 _/ϵ_ ) is typically achieved only by second-order algorithms such as Newton’s method. 

### **5.2 Understanding** _ωyy_ **: adaptive rescaling** 

If a layer only has _ωyy_<sup>_l_= 0, it has a rescaling effect.The amount of scaling is related to the amount</sup> of noise added in a model selection setting. The update rule for this layer is: 



This rescales every _y_ by a factor that depends on _λ_<sup>_l_</sup> . When _ωyy_<sup>_l<_0, this shrinks of the output based</sup> on the norm of _y_ in the previous layer. This is useful for the mixed noise variance problem, as ridge regression solution scales the least squares solution by a factor that depends on the noise level. 

Specifically, assuming Σ _≈_ E[Σ] = _nI_ , the ridge regression solution becomes _wσ_<sup>_∗_2</sup><sup>_≈_</sup> _n_ + _<u>nσ</u>_<sup>2</sup><sup>_w∗_,</sup> which is exactly a scaled version of the OLS solution. Further, when noise is larger, the scaled factor is smaller, which agrees with the behavior of a negative _ωyy_ . 

We can show that using adaptive scaling _ωyy_ even a 2-layer linear transformer can be enough to solve a simple example of categorical mixed noise variance problem _στ ∈{σ_ 1 _, σ_ 2 _}_ and _n →∞_ : 

6 



<!-- Start of picture text -->
σmax  = 0 σmax  = 1 σmax  = 2 σmax  = 3<br>10 0 10 0 10 0 10 0<br>10 −1<br>10 −5 10 −1<br>10 −1<br>10 −2<br>10 −10 10 −2<br>10 0 σmax  = 4 10 0 σmax  = 5 σmax  = 6 σmax  = 7<br>10 −1 10 −1 10 −1 10 −1<br>1 3 5 7 1 3 5 7 1 3 5 7 1 3 5 7<br>Number of layers Number of layers Number of layers Number of layers<br>GD + + Diag Full ConstRR AdaRR TunedRR<br>Adj. eval loss<br>Adj. eval loss<br><!-- End of picture text -->

Figure 1: In-context learning performance for noisy linear regression problem across models with different number of layers and _σmax_ for _στ ∼ U_ (0 _, σmax_ ). Each marker corresponds to a separately trained model with a given number of layers. Models with diagonal attention weights (DIAG) match those with full attention weights (FULL). Models specialized on a fixed noise (GD<sup>++</sup> ) perform poorly, similar to a Ridge Regression solution with a constant noise (CONSTRR). Among the baselines, only tuned exact Ridge Regression solution (TUNEDRR) is comparable with linear transformers. 

**Theorem 5.2.** _Suppose the input to the transformer is_ ( _x_ 1 _, y_ 1) _,_ ( _x_ 2 _, y_ 2) _, ...,_ ( _xn, yn_ ) _,_ ( _xq,_ 0) _, where xi ∼ N_ (0 _, n_<sup><u>1</u></sup><sup>_I_)</sup><sup>_,yi_=</sup><sup>_w⊤xi_+</sup><sup>_ξi.Here ξi∼N_(0</sup><sup>_, σ_2)</sup><sup>_is the noise whose noise level σcan take_</sup> _one of two values: σ_ 1 _or σ_ 2 _. Then as n goes to_ + _∞, there exists a set of parameters for two-layer linear transformers such that the implicit w_<sup>2</sup> _of the linear transformer converges to the optimal ridge regression results (and the output of the linear transformer is −⟨w_<sup>2</sup> _, xq⟩). Further, the first layer only has ωyx being nonzero and the second layer only has ωyy being nonzero._ 

### **5.3 Understanding** _ωxy_ **: adapting step-sizes** 

The final term in the diagonal model, _ωxy_ , has a more complicated effect. Since it changes only the _x_ -coordinates, it does not have an immediate effect on _y_ . To understand how it influences the _y_ we consider a simplified two-step process, where the first step only has _ωxy_ = 0 and the second step only has _ωyx_ = 0 (so the second step is just doing one step of gradient descent). In this case, the first layer will update the _xi_ ’s as: 



There are two effects of the _ωxy_ term, one is a multiplicative effect on _xi_ , and the other is an additive term that makes _x_ -output related to the residual _ri_ . The multiplicative step in _xi_ has an unknown preconditioning effect. For simplicity we assume the multiplicative term is small, that is: 

_x_<sup>1</sup> _i_<sup>_≈xi_+</sup><sup>_ωxyri_Σ</sup><sup>_w∗_;</sup> _x_<sup>1</sup> _t_<sup>_≈xt._</sup> 

The first layer does not change _y_ , so _yt_<sup>1=</sup><sup>_yt_and</sup><sup>_y_</sup> _i_<sup>1=</sup><sup>_yi_.For this set of</sup><sup>_xi_, we can write down the</sup> output on _y_ in the second layer as 



7 



<!-- Start of picture text -->
0.010 σmax  = 0 σmax  = 1 σmax  = 2 0.10 σmax  = 3<br>0.05<br>0.01<br>0.005 0.05<br>0.000 0.00 0.00 0.00<br>0.0 0.5 1.0 0 1 2 0 1 2 3 0 1 2 3 4<br>σmax  = 4 σmax  = 5 σmax  = 6 0.2 σmax  = 7<br>0.1 0.1 0.1<br>0.1<br>0.0 0.0 0.0 0.0<br>0 1 2 3 4 5 0 1 2 3 4 5 6 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 8<br>Variance  σ Variance  σ Variance  σ Variance  σ<br>0.20 2 layers 3 layers 4 layers 5 layers 6 layers 7 layers<br>0.15<br>0.10<br>0.05<br>0.00<br>0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6<br>Variance  σ Variance  σ Variance  σ Variance  σ Variance  σ Variance  σ<br>GD + + Diag Full ConstRR AdaRR TunedRR<br>Adj. eval loss<br>Adj. eval loss<br>Adj. eval loss<br><!-- End of picture text -->

Figure 2: Per-variance profile of models behavior for uniform noise variance _στ ∼ U_ (0 _, σmax_ ). _Top two rows:_ 7-layer models with varying _σmax_ . _Bottom row:_ models with varying numbers of layers, fixed _σmax_ = 5. In-distribution noise is shaded gray. 

Here we used the properties of residual _ri_ (in particular<sup>�</sup> _i_<sup>_yixi_=Σ</sup><sup>_w∗_, and �</sup> _i_<sup>_yiri_=�</sup> _i_<sup>_r_</sup> _i_<sup>2).</sup> Note that (Σ _w_<sup>_∗_</sup> )<sup>_⊤_</sup> _xt_ is basically what a gradient descent step on the original input should do. Therefore effectively, the two-layer network is doing gradient descent, but the step size is the product of _−ωyx_ and (1 + _ωxy_ � _i_<sup>_r_</sup> _i_<sup>2).The factor (1 +</sup><sup>_ωxy_</sup> � _i_<sup>_r_</sup> _i_<sup>2) depends on the level of noise, and when</sup> _ωxy, ωyx <_ 0, the effective step size is smaller when there is more noise. This is especially helpful in the model selection problem, because intuitively one would like to perform early-stopping (small step sizes) when the noise is high. 

## **6 Experiments** 

In this section, we investigate the training dynamics of linear transformers when trained with a mixed noise variance problem (7). We evaluate three types of single-head linear transformer models: 

- FULL. Trains full parameter matrices. 

- DIAG. Trains diagonal parameter matrices (10). 

- GD<sup>++</sup> . An even more restricted diagonal variant defined in (11). 

For each experiment, we train each linear transformer modifications with a varying number of layers (1 to 7) using using Adam optimizer for 200 000 iterations with a learning rate of 0 _._ 0001 and a batch size of 2 048. In some cases, especially for a large number of layers, we had to adjust the learning rate to prevent stability issues. We report the best result out of 5 runs with different training seeds. We used _N_ = 20 in-context examples in _D_ = 10 dimensions. We evaluated the algorithm using 100 000 novel sequences. All the experiments were done on a single H100 GPU with 80GB of VRAM. It took on average 4–12 hours to train a single algorithm, however experimenting with different weight decay parameters, better optimizer and learning rate schedule will likely reduce this number dramatically. 

We use _adjusted evaluation loss_ as our main performance metric. It is calculated by subtracting the oracle loss from the predictor’s loss. The oracle loss is the closed-form solution of the ridge regression loss (5), assuming the noise variance _στ_ is known. The adjusted evaluation loss allows for direct model performance comparison across different noise variances. This is important because higher noise significantly degrades the model prediction. Our adjustment does not affect the model’s optimization process, since it only modifies the loss by an additive constant. 

**Baseline estimates.** We evaluated the linear transformer against a closed-form solution to the ridge regression problem (5). We estimated the noise variance _στ_ using the following methods: 

- _Constant Ridge Regression (_ CONSTRR _)._ The noise variance is estimated using a single scalar value for all the sequences, tuned separately for each mixed variance problem. 

8 



<!-- Start of picture text -->
στ ∈{ 1 ,  3 } στ ∈{ 1 ,  3 ,  5 }<br>10 0 0.2 10 0 0.2<br>10 −1 0.1 10 −1 0.1<br>10 −2 0.0 10 −2 0.0<br>1 3 5 7 0 1 2 3 4 5 6 1 3 5 7 0 1 2 3 4 5 6<br>Number of layers Noise variance Number of layers Noise variance<br>0.2 2 layers 3 layers 4 layers 5 layers 6 layers 7 layers<br>0.1<br>0.0<br>0.2<br>0.1<br>0.0<br>0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6<br>Variance  σ Variance  σ Variance  σ Variance  σ Variance  σ Variance  σ<br>GD + + Diag Full ConstRR AdaRR TunedRR<br>Adj. eval loss Adj. eval loss Adj. eval loss Adj. eval loss<br>}<br> 31 ,<br>∈{<br>τ Adj. eval loss<br>σ<br>}<br> 5 31 ,,<br>Adj. eval loss<br>∈{<br>τ<br>σ<br><!-- End of picture text -->

Figure 3: In-context learning performance for noisy linear regression across models with varying number of layers for conditional noise variance _στ ∈{_ 1 _,_ 3 _}_ and _στ ∈{_ 1 _,_ 3 _,_ 5 _}_ . _Top:_ loss for models with various number of layers and per-variance profile for models with 7 layers. _Bottom:_ Per-variance profile of the model across different numbers of layers. In-distribution noise is shaded gray. 

- _Adaptive Ridge Regression (_ ADARR _)._ Estimate the noise variance via unbiased estimator (Cherkassky & Ma, 2003) _σ_ est<sup>2=</sup> _n−_ <u>1</u> _d_ � _nj_ =1<sup>(</sup><sup>_yj−y_ˆ</sup><sup>_j_)2, where</sup><sup>_y_ˆ</sup><sup>_j_represents the solution to</sup> the ordinary least squares (4), found in a closed-form. 

- _Tuned Adaptive Ridge Regression (_ TUNEDRR _)._ Same as above, but after the noise is estimated, we tuned two additional parameters to minimize the evaluation loss: (1) a max. threshold value for the estimated variance, (2) a multiplicative adjustment to the noise estimator. These values are tuned separately for each problem. 

Notice that all the baselines above are based on ridge regression, which is a closed-form, non-iterative solution. Thus, they have an algorithmic advantage over linear transformers that do not have access to matrix inversion. These baselines help us gauge the best possible performance, establishing an upper bound rather than a strictly equivalent comparison. 

A more faithful comparison to our method would be an iterative version of the ADARR that does not use matrix inversion. Instead, we can use gradient descent to estimate the noise and the solution to the ridge regression. However, in practice, this gradient descent estimator converges to ADARR only after _≈_ 100 iterations. In contrast, linear transformers typically converge in fewer than 10 layers. 

We consider two choices for the distribution of _στ_ : 

- _Uniform. στ ∼ U_ (0 _, σmax_ ) drawn from a uniform distribution bounded by _σmax_ . We tried multiple scenarios with _σmax_ ranging from 0 to 7. 

- _Categorical. στ ∈ S_ chosen from a discrete set _S_ . We tested _S_ = _{_ 1 _,_ 3 _}_ and _S_ = _{_ 1 _,_ 3 _,_ 5 _}_ . 

Our approach generalizes the problem studied by Bai et al. (2023), who considered only categorical variance selection and show experiments only with two _στ_ values. 

**Uniform noise variance.** For the uniform noise variance, Fig. 1 shows that FULL and DIAG achieve comparable performance across different numbers of layers and different _σmax_ . On the other hand, GD<sup>++</sup> converges to a higher value, closely approaching the performance of the CONSTRR baseline. As _σmax_ grows, linear transformers show a clear advantage over the baselines. With 4 layers, they outperform the closed-form solution ADARR for _σmax_ = 4 and larger. Models with 5 or more layers match or exceed the performance of TUNEDRR. 

The top of Fig. 2 offers a detailed perspective on performance of 7-layer models and the baselines. Here, we computed per-variance profiles across noise variance range from 0 to _σmax_ + 1. We can see that poor performance of GD<sup>++</sup> comes from its inability to estimate well across the full noise variance range. Its performance closely mirrors to CONSTRR, suggesting that GD<sup>++</sup> under the hood might also be estimating a single constant variance for all the data. 

ADARR perfectly estimates problems with no noise, but struggles more as noise variance increases. TUNEDRR slightly improves estimation by incorporating _σmax_ into its tunable parameters, yet its 

9 





Figure 4: Weights for 4 layer linear transformer with FULL parametrization trained with categorical noise _στ ∈{_ 1 _,_ 3 _}_ . _Top:_ weights for _Q_<sup>_l_</sup> matrix, _bottom:_ weights for _P_<sup>_l_</sup> matrix. 

prediction suffers in the mid-range. FULL and DIAG demonstrate comparable performance across all noise variances. While more research is needed to definitively confirm or deny their equivalence, we believe that these models are actually not identical despite their similar performance. 

At the bottom of Fig. 2 we set the noise variance to _σmax_ = 5 and display a per-variance profile for models with varying layers. Two-layer models for FULL and DIAG behave akin to GD<sup>++</sup> , modeling only a single noise variance in the middle. However, the results quickly improve across the entire noise spectrum for 3 or more layers. In contrast, GD<sup>++</sup> quickly converges to a suboptimal solution. 

**Categorical noise variance.** Fig. 3 shows a notable difference between DIAG and FULL models for categorical noise variance _στ ∈{_ 1 _,_ 3 _}_ . This could stem from a bad local minima, or suggest a fundamental difference between the models for this problem. Interestingly, from per-variance profiling we see that DIAG extrapolates better for variances not used for training, while FULL, despite its lower in-distribution error, performs worse on unseen variances. Fig. 4 shows learned weights of the 4 layer linear transformer with FULL parametrization. The weights are very diagonal heavy, potentially with some low-rank component. 

For _στ ∈{_ 1 _,_ 3 _,_ 5 _}_ , examining the per-variance profile at the bottom of Fig. 3 reveals differences in their behaviors. FULL exhibits a more complex per-variance profile with more fluctuations than the diagonal model, suggesting greater representational capacity. Surprisingly, it did not translate to better loss results compared to DIAG. 

For easy comparison, we compile the results of all methods and baselines in Table 1 in the Appendix. 

## **7 Conclusions** 

Our research reveals the surprising ability of linear transformers to tackle challenging in-context learning problems. We show that each layer maintains an implicit linear regression model, akin to a complex variant of preconditioned gradient descent with momentum-like behavior. 

Remarkably, when trained on noisy linear regression problems with unknown noise variance, linear transformers not only outperform standard baselines but also uncover a sophisticated optimization algorithm that incorporates noise-aware step-size adjustments and rescaling. This discovery highlights the potential of linear transformers to automatically discover novel optimization algorithms when presented with the right problems, opening exciting avenues for future research, including automated algorithm discovery using transformers and generalization to other problem domains. 

While our findings demonstrate the impressive capabilities of linear transformers in learning optimization algorithms, we acknowledge limitations in our work. These include the focus on simplified linear models, analysis of primarily diagonal attention matrices, and the need for further exploration into the optimality of discovered algorithms, generalization to complex function classes, scalability with larger datasets, and applicability to more complex transformer architectures. We believe these limitations present valuable directions for future research and emphasize the need for a deeper understanding of the implicit learning mechanisms within transformer architectures. 

10 

## **8 Acknowledgements** 

The authors would like to thank Nolan Miller and Andrey Zhmoginov for their valuable suggestions and feedback throughout the development of this project. Part of this work was done while Rong Ge was visiting Google Research. Rong Ge’s research is supported in part by NSF Award DMS-2031849 and CCF-1845171 (CAREER). 

## **References** 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to implement preconditioned gradient descent for in-context learning. _arXiv preprint arXiv:2306.00297_ , 2023. 

- Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. _arXiv preprint arXiv:2211.15661_ , 2022. 

- Ekin Akyürek, Bailin Wang, Yoon Kim, and Jacob Andreas. In-Context language learning: Architectures and algorithms. _arXiv preprint arXiv:2401.12973_ , 2024. 

- Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. _arXiv preprint arXiv:2305.10403_ , 2023. 

- Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians: Provable in-context learning with in-context algorithm selection. _arXiv preprint arXiv:2306.04637_ , 2023. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- Stephanie Chan, Adam Santoro, Andrew Lampinen, Jane Wang, Aaditya Singh, Pierre Richemond, James McClelland, and Felix Hill. Data distributional properties drive emergent in-context learning in transformers. _Advances in Neural Information Processing Systems_ , 35:18878–18891, 2022. 

- Xiang Cheng, Yuxin Chen, and Suvrit Sra. Transformers implement functional gradient descent to learn non-linear functions in context. _arXiv preprint arXiv:2312.06528_ , 2023. 

- Vladimir Cherkassky and Yunqian Ma. Comparison of model selection for regression. _Neural computation_ , 15(7):1691–1714, 2003. 

- Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. _arXiv preprint arXiv:2009.14794_ , 2020. 

- Deqing Fu, Tian-Qi Chen, Robin Jia, and Vatsal Sharan. Transformers learn higher-order optimization methods for in-context learning: A study with linear models. _arXiv preprint arXiv:2310.17086_ , 2023. 

- Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. _Advances in Neural Information Processing Systems_ , 35:30583–30598, 2022. 

- Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. _arXiv preprint arXiv:2301.13196_ , 2023. 

- Tianyu Guo, Wei Hu, Song Mei, Huan Wang, Caiming Xiong, Silvio Savarese, and Yu Bai. How do transformers learn in-context beyond simple functions? a case study on learning with representations. _arXiv preprint arXiv:2310.10616_ , 2023. 

11 

- Roee Hendel, Mor Geva, and Amir Globerson. In-context learning creates task vectors. _arXiv preprint arXiv:2310.15916_ , 2023. 

- Yu Huang, Yuan Cheng, and Yingbin Liang. In-context convergence of transformers. _arXiv preprint arXiv:2310.05249_ , 2023. 

- Evan Hubinger, Chris van Merwijk, Vladimir Mikulik, Joar Skalse, and Scott Garrabrant. Risks from learned optimization in advanced machine learning systems. _arXiv preprint arXiv:1906.01820_ , 2019. 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. _arXiv preprint arXiv:2310.06825_ , 2023. 

- Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In _International conference on machine learning_ , pp. 5156–5165. PMLR, 2020. 

- Jannik Kossen, Tom Rainforth, and Yarin Gal. In-context learning in large language models learns label relationships but is not conventional learning. _arXiv preprint arXiv:2307.12375_ , 2023. 

- Yingcong Li, Muhammed Emrullah Ildiz, Dimitris Papailiopoulos, and Samet Oymak. Transformers as algorithms: Generalization and stability in in-context learning. In _International Conference on Machine Learning_ , pp. 19565–19594. PMLR, 2023. 

- Arvind Mahankali, Tatsunori B Hashimoto, and Tengyu Ma. One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. _arXiv preprint arXiv:2307.03576_ , 2023. 

- Suvir Mirchandani, Fei Xia, Pete Florence, Brian Ichter, Danny Driess, Montserrat Gonzalez Arenas, Kanishka Rao, Dorsa Sadigh, and Andy Zeng. Large language models as general pattern machines. _arXiv preprint arXiv:2307.04721_ , 2023. 

- Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. _arXiv preprint arXiv:2209.11895_ , 2022. 

- Reese Pathak, Rajat Sen, Weihao Kong, and Abhimanyu Das. Transformers can optimally learn regression mixture models. _arXiv preprint arXiv:2311.08362_ , 2023. 

- Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In _International Conference on Machine Learning_ , pp. 9355–9366. PMLR, 2021. 

- Lingfeng Shen, Aayush Mishra, and Daniel Khashabi. Do pretrained transformers really learn in-context by gradient descent? _arXiv preprint arXiv:2310.08540_ , 2023. 

- Davoud Ataee Tarzanagh, Yingcong Li, Xuechen Zhang, and Samet Oymak. Max-margin token selection in attention mechanism. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. 

- Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

- Yuandong Tian, Yiping Wang, Beidi Chen, and Simon Du. Scan and snap: Understanding training dynamics and token composition in 1-layer transformer. _arXiv preprint arXiv:2305.16380_ , 2023a. 

- Yuandong Tian, Yiping Wang, Zhenyu Zhang, Beidi Chen, and Simon Du. Joma: Demystifying multilayer transformers via joint dynamics of mlp and attention. In _NeurIPS 2023 Workshop on Mathematics of Modern Machine Learning_ , 2023b. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

12 

- Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In _International Conference on Machine Learning_ , pp. 35151–35174. PMLR, 2023a. 

- Johannes von Oswald, Eyvind Niklasson, Maximilian Schlegel, Seijin Kobayashi, Nicolas Zucchet, Nino Scherrer, Nolan Miller, Mark Sandler, Max Vladymyrov, Razvan Pascanu, et al. Uncovering mesa-optimization algorithms in transformers. _arXiv preprint arXiv:2309.05858_ , 2023b. 

Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. _arXiv preprint arXiv:2006.04768_ , 2020. 

- Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, et al. Larger language models do in-context learning differently. _arXiv preprint arXiv:2303.03846_ , 2023. 

- Kaiyue Wen, Yuchen Li, Bingbin Liu, and Andrej Risteski. Transformers are uninterpretable with myopic methods: a case study with bounded dyck grammars. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. 

- Steve Yadlowsky, Lyric Doshi, and Nilesh Tripuraneni. Pretraining data mixtures enable narrow model selection capabilities in transformer models. _arXiv preprint arXiv:2311.00871_ , 2023. 

- Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear models in-context. _arXiv preprint arXiv:2306.09927_ , 2023. 

13 

## **A Proofs from Sections 4 and 5** 

### **A.1 Proof of Theorem 4.1** 

We first give the proof for Theorem 4.1. In the process we will also prove Lemma 4.3, as Theorem 4.1 follows immediately from an induction based on the lemma. 

_Proof._ We do this by induction. At _l_ = 0, it’s easy to check that we can set _a_<sup>(0)</sup> = 1 _, w_<sup>(0)</sup> = 0 _, M_<sup>(0)</sup> = _I, u_<sup>(0)</sup> = 0. 

Suppose this is true for some layer _l_ , if the weights of layer _l_ are ( _P_ 1<sup>_l, Ql_</sup> 1<sup>)</sup><sup>_, ...,_(</sup><sup>_P l_</sup> _k_<sup>_, Ql_</sup> _k_<sup>) for</sup><sup>_k_heads,</sup> at output of layer _l_ + 1 we have: 



Note that the same equation is true for _i_ = _n_ + 1 just by letting _yn_ +1 = 0. Let the middle matrix has the following structure: 



Then one can choose the parameters of the next layer as in Lemma 4.3 



One can check that this choice satisfies (12). 

### **A.2 Proof of Lemma 4.4** 

This lemma is in fact a corollary of Lemma 4.3. We first give a more detailed version which explicitly state the unknown matrices Λ<sup>_l_</sup> _,_ Γ<sup>_l_</sup> _,_ Π<sup>_l_</sup> _,_ Φ<sup>_l_</sup> : 

**Lemma A.1.** _In the setup of Theorem 4.1 with diagonal parameters_ (9) _, one can recursively compute matrices u_<sup>_l_</sup> _, w_<sup>_l_</sup> _using the following formula_ 





_Proof._ First, we compute the following matrix that appeared in Lemma 4.3 for the specific diagonal case: 



14 

This implies that _A_<sup>_l_</sup> = _ωxx_<sup>_l_Σ</sup><sup>_l_,</sup><sup>_bl_=</sup><sup>_ω_</sup> _xy_<sup>_lαl_,</sup><sup>_cl_=</sup><sup>_ω_</sup> _yx_<sup>_lαl_and</sup><sup>_dl_=</sup><sup>_ω_</sup> _yy_<sup>_lλl_.Next we rewrite</sup><sup>_αl_:</sup> 



Here the first step is by Theorem 4.1, the second step replaces _yi_ with _⟨w_<sup>_∗_</sup> _, xi⟩_ + _ri_ , the third step uses the fact that<sup>�</sup><sup>_n_</sup> _i_ =1<sup>_rixi_= 0 to get rid of the cross terms.</sup> 

The remaining proof just substitutes the formula for _α_<sup>_l_</sup> into Lemma 4.3. 

Now Lemma A.1 implies Lemma 4.4 immediately by setting Λ<sup>_l_</sup> = _−ωxy_<sup>_l_(</sup><sup>_al_)2</sup><sup>_ρI−ω_</sup> _xx_<sup>_l_Σ</sup><sup>_l_,</sup> Γ<sup>_l_</sup> = _a_<sup>_l_</sup> _ωxy_<sup>_l_</sup> � _M_<sup>_l_</sup> + _a_<sup>_l_</sup> _u_<sup>_l_</sup> ( _w_<sup>_∗_</sup> )<sup>_⊤_�</sup> , _s_<sup>_l_</sup> = _tuωyyλ_<sup>_l_</sup> , Π<sup>_l_</sup> = _ωyx_<sup>_l_(</sup><sup>_M l_)</sup><sup>_⊤_(</sup><sup>_M l_+</sup><sup>_alul_(</sup><sup>_w∗_)</sup><sup>_⊤_)andΦ</sup><sup>_l_=</sup> _a_<sup>_l_</sup> _ρωyx_<sup>_l_(</sup><sup>_M l_)</sup><sup>_⊤_.</sup> 

### **A.3 Proof for Theorem 4.2** 

_Proof._ By Theorem 4.1, we know _yi_<sup>_l_=</sup><sup>_⟨wl, xi⟩_for some</sup><sup>_wl_.When</sup><sup>_n ≫d_, with high probability</sup> the norm of _y_<sup>_l_</sup> is on the order of Θ(<sup>_√_</sup> _<u>n</u>_ <u>)</u> _∥w_<sup>_l_</sup> _∥_ , and the norm of _y_<sup>_∗_</sup> is Θ(<sup>_√_</sup> _<u>n</u>_ <u>).</u> Therefore we only need to bound the correlation. The correlation is equal to 



We know with high probability _|_<sup>�</sup><sup>_n_</sup> _i_ =1<sup>_x_</sup> _i_<sup>3</sup><sup>_|_=</sup><sup>_O_(</sup><sup>_√_</sup> _<u>n</u>_ <u>) because E[</u> _x_<sup>3</sup> _i_<sup>] = 0.The second term can be</sup> written as _⟨w_<sup>_l_</sup> _, v⟩_ where _v_ is a vector whose coordinates are _v_ 1 = 0 and _vj_ =<sup>�</sup><sup>_n_</sup> _i_ =1<sup>_xi_(1)2</sup><sup>_xi_(</sup><sup>_j_) for</sup> 2 _≤ j ≤ d_ , therefore with high probability _∥v∥_ = _O_ ( _√nd_ ). Therefore, with high probability the cosine similarity is at most 



When _n ≫ d_ this can be made smaller than any fixed constant. 

### **A.4 Proof for Theorem 5.1** 

In this section we prove Theorem 5.1 by finding hyperparameters for GD<sup>++</sup> algorithm that solves least squares problems with very high accuracy. The first steps in the construction iteratively makes the data _xi_ ’s better conditioned, and the last step is a single step of gradient descent. The proof is based on several lemma, first we observe that if the data is very well-conditioned, then one-step gradient descent solves the problem accurately: 

15 

**Lemma A.2.** _Given_ ( _x_ 1 _, y_ 1) _, ...,_ ( _xn, yn_ ) _where_ Σ :=<sup>�</sup><sup>_n_</sup> _i_ =1<sup>_xix_</sup> _i_<sup>_⊤has eigenvalues between_1</sup><sup>_and_</sup> _n_ 1 + _ϵ. Let w_<sup>_∗_</sup> := arg min _w_ � _i_ =1<sup>(</sup><sup>_yi −⟨w, xi⟩_)2</sup><sup>_be the optimal least squares solution, thenw_ˆ=</sup> � _ni_ =1<sup>_yixisatisfies ∥w_ˆ</sup><sup>_−w∗∥≤ϵ∥w∗∥._</sup> 

_Proof._ We can write _yi_ = _⟨xi, w_<sup>_∗_</sup> _⟩_ + _ri_ . By the fact that _w_<sup>_∗_</sup> is the optimal solution we know _ri_ ’s satisfy<sup>�</sup><sup>_n_</sup> _i_ =1<sup>_rixi_= 0.Therefore</sup><sup>_w_ˆ= �</sup><sup>_n_</sup> _i_ =1<sup>_yixi_= �</sup><sup>_n_</sup> _i_ =1<sup>_⟨xi, w∗⟩xi_= Σ</sup><sup>_w∗_.This implies</sup> ˆ _∥w − w_<sup>_∗_</sup> _∥_ = _∥_ (Σ _− I_ ) _w_<sup>_∗_</sup> _∥≤∥_ Σ _− I∥∥w_<sup>_∗_</sup> _∥≤ ϵ∥w_<sup>_∗_</sup> _∥._ 

Next we show that by applying just the preconditioning step of GD<sup>++</sup> , one can get a well-conditioned _x_ matrix very quickly. Note that the Σ matrix is updated as Σ _←_ ( _I − γ_ Σ)Σ( _I − γ_ Σ), so an eigenvalue of _λ_ in the original Σ matrix would become _λ_ (1 _− γλ_ )<sup>2</sup> . The following lemma shows that this transformation is effective in shrinking the condition number 

**Lemma A.3.** _Suppose ν/µ_ = _κ ≥_ 1 _._ 1 _, then there exists an universal constant c <_ 1 _such that choosing γν_ = 1 _/_ 3 _implies_ 



_On the other hand, if ν/µ_ = _κ ≤_ 1 + _ϵ where ϵ ≤_ 0 _._ 1 _, then choosing γν_ = 1 _/_ 3 _implies_ 



The first claim shows that one can reduce the condition number by a constant factor in every step until it’s a small constant. The second claim shows that once the condition number is small (1 + _ϵ_ ), each iteration can bring it much closer to 1 (to the order of 1 + _O_ ( _ϵ_<sup>2</sup> )). 

Now we prove the lemma. 

_Proof._ First, notice that the function _f_ ( _x_ ) = _x_ (1 _− γx_ )<sup>2</sup> is monotonically nondecreasing for _x ∈_ [0 _, ν_ ] if _γν_ = 1 _/_ 3 (indeed, it’s derivative _f_<sup>_′_</sup> ( _x_ ) = (1 _− γx_ )(1 _−_ 3 _γx_ ) is always nonnegative). Therefore, the max is always achieved at _x_ = _ν_ and the min is always achieved at _x_ = _µ_ . The new ratio is therefore 



4 _<u>/</u>_ 9 4 _<u>/</u>_ 9 When _κ ≥_ 1 _._ 1 the ratio (1 _−_ 1 _/_ 3 _κ_ )<sup>2is always below</sup> (1 _−_ 1 _/_ 3 _._ 3)<sup>2which is a constant bounded away</sup> from 1. 

When _κ_ = 1 + _ϵ <_ 1 _._ 1, we can write down the RHS in terms of _ϵ_ 



Note that by the careful choice of _γ_ , the RHS has the following Taylor expansion: 



One can then check the RHS is always upperbounded by 2 _ϵ_<sup>2</sup> when _ϵ <_ 0 _._ 1. 

With the two lemmas we are now ready to prove the main theorem: 

_Proof._ By Lemma A.3 we know in _O_ (log _κ_ + log log 1 _/ϵ_ ) iterations, by assigning _κ_ in the way of Lemma A.3 one can reduce the condition number of _x_ to _κ_<sup>_′_</sup> _≤_ 1 + _ϵ/_ 2 _κ_ (we chose _ϵ/_ 2 _κ_ here to give some slack for later analysis). 

Let Σ<sup>_′_</sup> be the covariance matrix after these iterations, and _ν_<sup>_′_</sup> _, µ_<sup>_′_</sup> be the upper and lowerbound for its eigenvalues. The data _xi_ ’s are transformed to a new data _x_<sup>_′_</sup> _i_<sup>=</sup><sup>_Mxi_forsomematrix</sup> 

16 



<!-- Start of picture text -->
GD + + Diag Full<br>1 layers<br>2 layers<br>10 0 10 0 10 0<br>3 layers<br>4 layers<br>10 −1 10 −1 10 −1 5 layers<br>6 layers<br>0 2 4 6 8 0 2 4 6 8 0 2 4 6 8 7 layers<br>Prediction after K layers Prediction after K layers Prediction after K layers<br>Adjusted eval loss<br><!-- End of picture text -->

Figure 5: Linear transformer models show a consistent decrease in error per layer when trained on data with mixed noise variance _στ ∼ U_ (0 _,_ 5). The error bars measure variance over 5 training seeds. 

_M_ . Let _M_ = _A_ Σ<sup>_−_1</sup><sup>_/_2</sup> , then since _M_<sup>_′_</sup> = _AA_<sup>_⊤_</sup> we know _A_ is a matrix with singular values between<sup>_√_</sup> _µ_<sup>_<u>′</u>_</sup> and _√ν_<sup>_′_</sup> . The optimal solution ( _w_<sup>_∗_</sup> )<sup>_′_</sup> = _M_<sup>_−⊤_</sup> _w_<sup>_∗_</sup> has norm at most<sup>_<u>√</u>_</sup> _<u>ν/</u>_<sup>_√_</sup> _µ_<sup>_<u>′</u>_</sup> _∥w_<sup>_∗_</sup> _∥_ . Therefore by Lemma A.2 we know the one-step gradient step with _w_ ˆ =<sup>�</sup><sup>_n_</sup> _i_ =1 _µ_ <u>1</u><sup>_′yixi_satisfy</sup> ˆ _∥w −_ ( _w_<sup>_∗_</sup> )<sup>_′_</sup> _∥≤_ ( _κ_<sup>_′_</sup> _−_ 1)<sup>_√_</sup> _<u>ν/</u>_<sup>_√_</sup> _µ_<sup>_<u>′</u>_</sup> _∥w_<sup>_∗_</sup> _∥_ . The test data _xt_ is also transformed to _x_<sup>_′_</sup> _t_<sup>=</sup><sup>_A_Σ</sup><sup>_−_1</sup><sup>_/_2</sup><sup>_xt_, and</sup> the algorithm outputs _⟨w, x_ ˆ<sup>_′_</sup> _t_<sup>_⟩_, so the error is at most</sup><sup>_√_</sup> _<u>ν∥w</u>_<sup>_∗_</sup> _∥∗∥x_<sup>_′_</sup> _t_<sup>_∥≤_(</sup><sup>_κ′ −_1)</sup><sup>_√_</sup> _<u>κ√κ</u>_<sup>_′_</sup> _∥w_<sup>_∗_</sup> _∥∗∥xt∥_ . By the choice of _κ_<sup>_′_</sup> we can check that RHS is at most _ϵ∥w_<sup>_∗_</sup> _∥∥xt∥_ . 

### **A.5 Proof of the Theorem 5.2** 

_Proof._ The key observation here is that when _n → ∞_ , under the assumptions we have _n_ lim _n→∞_ � _i_ =1<sup>_xix_</sup> _i_<sup>_⊤_</sup> = _I_ . Therefore the ridge regression solutions converge to _wσ_<sup>_∗_2</sup> = 1+1 _σ_<sup>2</sup> � _ni_ =1<sup>_yixi_and the desired output is</sup><sup>_⟨w_</sup> _σ_<sup>_∗_2</sup><sup>_, xq⟩_.</sup> 

_n_ By the calculations before, we know after the first-layer, the implicit _w_ is _w_<sup>1</sup> = _ωyx_ � _i_ =1<sup>_yixi_.As</sup> long as _ωyx_ is a constant, when _n →∞_ we know _n_<sup><u>1</u></sup> � _ni_ =1<sup>(</sup><sup>_y_</sup> _i_<sup>1)2=</sup><sup>_σ_2 (as the part of</sup><sup>_y_that depend</sup> on _x_ is negligible compared to noise), therefore the output of the second layer satisfies 



Therefore, as long as we choose _ωyx_ and _ωyy_ to satisfy (1 + _nσ_<sup>2</sup> _ωyy_ ) _ωyx_ = 1+1 _σ_<sup>2when</sup><sup>_σ_=</sup><sup>_σ_1 or</sup> _σ_ 2 (notice that these are two linear equations on _ωyx_ and _nωyxωyy_ , so they always have a solution), then we have lim _n→∞ w_<sup>2</sup> = _wσ_<sup>_∗_2for the two noise levels.</sup> 

## **B More experiments** 

Here we provide results of additional experiments that did not make it to the main text. 

Fig. 6 shows an example of unadjusted loss. Clearly, it is virtually impossible to compare the methods across various noise levels this way. 

Fig. 7 shows per-variance profile of intermediate predictions of the network of varying depth. It appears that GD<sup>++</sup> demonstrates behavior typical of GD-based algorithms: early iterations model higher noise (similar to early stopping), gradually converging towards lower noise predictions. DIAG exhibits this patter initially, but then dramatically improves, particularly for lower noise ranges. Intriguingly, FULL displays the opposite trend, first improving low-noise predictions, followed by a decline in higher noise prediction accuracy, especially in the last layer. 

Finally, Table 1 presents comprehensive numerical results for our experiments across various mixed noise variance models. For each model variant (represented by a column), the best-performing result is highlighted in bold. 

17 



<!-- Start of picture text -->
6 GD + +<br>Diag<br>4 Full<br>ConstRR<br>AdaRR<br>2<br>TunedRR<br>0<br>0 2 4 6<br>Variance  σ<br>Unadjusted eval loss<br><!-- End of picture text -->

Figure 6: Example of unadjusted loss given by directly minimizing (7). It is pretty hard to see variation between comparable methods using this loss directly. 

|Method|||Unifo|rm_στ ∼_|(0_, σma_|_x_)|||Categor|ical_στ ∈S_|
|---|---|---|---|---|---|---|---|---|---|---|
||0|1|2|3|4|5|6|7|{1,3}|{1,3,5}|
||||||1 layer||||||
|GD<sup>++</sup>|1.768|1.639|1.396|1.175|1.015|0.907|0.841|0.806|1.007|0.819|
|DIAG|1.767|1.639|1.396|1.175|1.015|0.906|0.841|0.806|1.007|0.819|
|FULL|1.768|1.640|1.397|1.176|1.016|0.907|0.842|0.806|1.008|0.820|
|||||2|layers||||||
|GD<sup>++</sup>|0.341|0.295|0.243|0.265|0.347|0.366|0.440|0.530|0.305|0.427|
|DIAG|0.265|0.214|0.173|0.188|0.219|0.242|0.254|0.259|0.201|0.246|
|FULL|0.264|0.215|0.173|0.188|0.220|0.245|0.259|0.263|0.202|0.276|
|||||3|layers||||||
|GD<sup>++</sup>|0.019|0.021|0.071|0.161|0.259|0.344|0.454|0.530|0.222|0.422|
|DIAG|0.013|0.015|0.048|0.087|0.109|0.118|0.121|0.123|0.098|0.119|
|FULL|0.012|0.015|0.049|0.075|0.101|0.117|0.124|0.127|0.076|0.113|
|||||4|layers||||||
|GD<sup>++</sup>|9.91e-05|0.014|0.066|0.160|0.258|0.344|0.454|0.530|0.222|0.422|
|DIAG|1.19e-04|0.006|0.024|0.041|0.050|0.059|0.065|0.073|0.043|0.062|
|FULL|1.63e-04|0.005|0.021|0.038|0.052|0.065|0.068|0.076|0.032|0.061|
|||||5|layers||||||
|GD<sup>++</sup>|1.14e-07|0.014|0.066|0.161|0.265|0.344|0.454|0.530|0.222|0.422|
|DIAG|1.81e-07|0.004|0.016|0.029|0.041|0.051|0.058|0.062|0.026|0.051|
|FULL|1.79e-07|**0.002**|0.015|0.026|0.038|0.048|0.059|0.065|0.016|0.048|
|||||6|layers||||||
|GD<sup>++</sup>|2.37e-10|0.009|0.066|0.161|0.265|0.344|0.454|0.530|0.222|0.422|
|DIAG|2.57e-10|0.003|0.014|0.028|0.040|0.048|0.054|0.059|0.020|0.047|
|FULL|2.71e-10|**0.002**|0.014|0.025|0.036|0.044|0.052|0.059|0.011|0.043|
|||||7|layers||||||
|GD<sup>++</sup>|2.65e-12|0.009|0.066|0.161|0.265|0.344|0.454|0.530|0.222|0.422|
|DIAG|2.50e-12|**0.002**|0.014|0.027|0.040|0.047|0.052|0.059|0.018|0.046|
|FULL|2.50e-12|**0.002**|**0.010**|0.025|**0.035**|**0.047**|**0.050**|**0.057**|**0.010**|**0.035**|
|||||B|aselines||||||
|CONSTRR|**0**|0.009|0.066|0.161|0.265|0.365|0.454|0.530|0.222|0.422|
|ADARR|**0**|0.003|0.016|0.034|0.053|0.068|0.081|0.092|0.051|0.084|
|TUNEDRR|**0**|**0.002**|**0.010**|**0.023**|0.037|0.049|0.060|0.068|0.021|0.054|



Table 1: Adjusted evaluation loss for models with various number of layers with uniform noise variance _στ ∼ U_ (0 _, σmax_ ). We highlight in bold the best results for each problem setup (i.e. each column). 

18 



<!-- Start of picture text -->
1.00 GD + + Diag Full<br>0.75<br>0.50<br>0.25<br>0.00<br>1.00<br>0.75<br>0.50<br>0.25<br>0.00<br>1.00<br>0.75<br>0.50<br>0.25<br>0.00<br>1.00<br>0.75<br>0.50<br>0.25<br>0.00<br>1.00<br>0.75<br>0.50<br>0.25<br>0.00<br>1.00<br>0.75<br>0.50<br>0.25<br>0.00<br>1.00<br>0.75<br>0.50<br>0.25<br>0.00<br>0 2 4 6 8 10 0 2 4 6 8 10 0 2 4 6 8 10<br>Variance  σ Variance  σ Variance  σ<br>After 1 layer After 3 layers After 5 layers After 7 layers<br>After 2 layers After 4 layers After 6 layers After 8 layers<br>2 layers model Adj. eval loss<br>3 layers model Adj. eval loss<br>4 layers model Adj. eval loss<br>5 layers model Adj. eval loss<br>6 layers model Adj. eval loss<br>7 layers model Adj. eval loss<br>8 layers model Adj. eval loss<br><!-- End of picture text -->

Figure 7: Layer by layer prediction quality for different models with _στ ∼ U_ (0 _,_ 5). The error bars measure std over 5 training seeds. 

19 

