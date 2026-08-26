# **Transformers learn to implement preconditioned gradient descent for in-context learning** 

**Kwangjun Ahn**<sup>_∗_</sup> **Xiang Cheng**<sup>_∗_</sup> **Hadi Daneshmand**<sup>_∗_</sup> **Suvrit Sra** MIT EECS/LIDS MIT LIDS MIT LIDS/FODSI TU Munich / MIT `kjahn@mit.edu chengx@mit.edu hdanesh@mit.edu suvrit@mit.edu` 

## **Abstract** 

Several recent works demonstrate that transformers can implement algorithms like gradient descent. By a careful construction of weights, these works show that multiple layers of transformers are expressive enough to simulate iterations of gradient descent. Going beyond the question of expressivity, we ask: _Can transformers learn to implement such algorithms by training over random problem instances?_ To our knowledge, we make the first theoretical progress on this question via an analysis of the loss landscape for linear transformers trained over random instances of linear regression. For a single attention layer, we prove the global minimum of the training objective implements a single iteration of preconditioned gradient descent. Notably, the preconditioning matrix not only adapts to the input distribution but also to the variance induced by data inadequacy. For a transformer with _L_ attention layers, we prove certain critical points of the training objective implement _L_ iterations of preconditioned gradient descent. Our results call for future theoretical studies on learning algorithms by training transformers. 

## **1 Introduction** 

In-context learning (ICL) is the striking capability of large language models: Given a prompt containing examples and a query, the transformer produces the correct output based on the context provided by the examples, _without adapting its parameters_ (Brown et al., 2020; Lieber et al., 2021; Rae et al., 2021; Black et al., 2022). This property has become the focus of body of recent research that aims to shed light on the underlying mechanism of large language models (Garg et al., 2022; Akyürek et al., 2022; von Oswald et al., 2023; Li and Malik, 2017; Min et al., 2021; Xie et al., 2021; Elhage et al., 2021; Olsson et al., 2022). 

A line of research studies ICL via the expressive power of transformers. Transformer architectures are powerful Turing machines, capable of implementing various algorithms (Pérez et al., 2021; Wei et al., 2022). Given an in-context prompt, Edelman et al. (2022); Olsson et al. (2022) argue that transformers are able to implement algorithms through the recurrence of multi-head attentions to extract coarse information from raw input prompts. Akyürek et al. (2022); von Oswald et al. (2023) assert that transformers can implement gradient descent on linear regression encoded in a given input prompt. It is thought provoking that transformers can implement such algorithms. 

Although transformers are universal machines to implement algorithms, they need specific parameter configurations for achieving these implementations. In practice, their parameters are adjusted via training using non-convex optimization over random problem instances. Hence, it remains unclear whether this non-convex optimization can be used to learn algorithms. The present paper investigates _the possibility of learning algorithms via training over random problem instances._ 

> _∗_ Equal contribution, alphabetical order. 

37th Conference on Neural Information Processing Systems (NeurIPS 2023). 

More specifically, we investigate the learning of gradient-based methods. It is hard to mathematically formulate what it means to learn gradient descent for general functions with transformers. Yet, Garg et al. (2022) elegantly examine it in the specific setting of ICL for learning functions. Empirical evidence suggests that transformers indeed learn to implement gradient descent, after training on random instances of linear regression (Garg et al., 2022; Akyürek et al., 2022; von Oswald et al., 2023). Motivated by these observations, we theoretically investigate the loss landscape of a simple transformer architecture based on **_attention without softmax_** (Schlag et al., 2021; von Oswald et al., 2023) (see Section 2 for details). 

**_Summary of our main results._** Our main contributions are the following: 

- We provide a complete characterization of the global optimum of a single-layer linear transformer. In particular, we observe that, with the optimal parameters, the transformer implements a single step of preconditioned gradient descent. Notably, the preconditioning matrix not only adapts to the distribution of input data but also to the variance caused by data inadequacy. We present this result in Theorem 1 in Section 3. 

- Next, we focus on a subset of the transformer parameter space, defined by a special sparsity condition (8). Such a parameter configuration allows us to formulate training transformers as a search over _k-step adaptive gradient-based algorithms_ . Theorem 2 characterizes the global minimizers of the training objective of a two-layer linear transformer over isotropic regression instances, and shows that the optima correspond to gradient descent with adaptive stepsizes. For multilayer transformers, Theorem 3 demonstrates that gradient descent, with a data-dependent preconditioning, can be derived from a critical point of the training objective. 

- Finally, we study the loss landscape in the absence of the sparsity condition (8), which goes beyond searching over conventional gradient-based optimization methods. In this case, we prove and interpret the structure of a critical point of the training objective. We show that a certian critical point in parameter space leads to an intriguing gradient-based algorithm that simultaneously takes gradient steps preconditioned by data covariance, and applies a linear transformation to further improve the conditioning. In the specific case when data covariance is isotropic, this algorithm corresponds to the GD++ algorithm of von Oswald et al. (2023) which is experimentally observed to be the outcome of training. 

We empirically validate the critical points analyzed in Theorem 3 and Theorem 4. For a transformer with three layers, our experimental results confirm the structural of critical points. Furthermore, we observed the objective value associated with these critical points is close to 0, suggesting that the critical points might be global optima. These experiments substantiate our theoretical analysis and suggests that our theory indeed _aligns with practice_ . Code for our experiments is available at `https://github.com/chengxiang/LinearTransformer` . 

### **1.1 Related works** 

The ability of neural network architectures to implement algorithms has been investigated in various context. The seminal work by Siegelmann and Sontag (1992) investigate the Turing completeness of recurrent neural networks. Despite this computational power, training recurrent networks remains a challenge. Graves et al. (2014) design an alternative neural architecture known as the _neural Turing machine_ , building on _attention layers_ introduced by Hochreiter and Schmidhuber (1997). Leveraging attention, Vaswani et al. (2017) propose transformers as powerful neural architectures, capable of solving various tasks in natural language processing (Devlin et al., 2019). This capability inspired a line of research that examines the algorithmic power of transformers (Pérez et al., 2021; Wei et al., 2022; Giannou et al., 2023; Akyürek et al., 2022; Olsson et al., 2022). What sets transformers apart from conventional neural networks is their impressive performance after training. In this work, we focus on understanding _how transformers learn to implement algorithms_ by training over problem instances. 

A line of research investigates how deep neural networks process data across their layers. The seminal work by Jastrzebski et al. (2018) observes that hidden representations across the layers of deep neural networks approximately implement gradient descent. Recent observations provide novel insights into the working mechanism of ICL for large language models, showing they can implement optimization algorithms across their layers (Garg et al., 2022; Akyürek et al., 2022; von Oswald et al., 2023). Moreover, Zhao et al. (2023); Allen-Zhu and Li (2023) observe transformer perform 

2 

dynamic programming to generate text. In this work, we theoretically study how transformer learns gradient-based algorithms for ICL. 

We discuss here two related works (Zhang et al., 2023; Mahankali et al., 2023) that appeared shortly after publication of our original draft. Both of these studies focus on a single layer attention network (see Section 3). Zhang et al. (2023) prove the global convergence of gradient descent to the global optimum whose structure is analyzed independently from this study and it the same as that in Theorem 1. Mahankali et al. (2023) also characterize the global minimizer of a single layer attention without softmax for a different data distribution. In addition to results for a single-layer attention, we analyze the landscape of two and multi-layer transformers. 

## **2 Setting: training linear transformers over random linear regression** 

In order to understand the mechanism of ICL, we consider the setting of training transformers over the random instances of linear regression, following (Garg et al., 2022; Akyürek et al., 2022; von Oswald et al., 2023). In particular, the random instances of linear regression are formalized as follows. 

**_Data distribution: random linear regression instances._** Let _x_<sup>(</sup><sup>_i_)</sup> _∈_ R<sup>_d_</sup> be the covariates drawn i.i.d. from a distribution _DX_ , and _w⋆ ∈_ R<sup>_d_</sup> be drawn from _DW_ . Let _X ∈_ R<sup>(</sup><sup>_n_+1)</sup><sup>_×d_</sup> be the matrix of covariates whose row _i_ contains tokens _x_<sup>(</sup><sup>_i_)</sup> . Given _x_<sup>(</sup><sup>_i_)</sup> ’s and _w⋆_ , the responses are defined as _y_ = [ _⟨x_<sup>(1)</sup> _, w⋆⟩, . . . , ⟨x_<sup>(</sup><sup>_n_)</sup> _, w⋆⟩_ ] _∈_ R<sup>_n_</sup> . Define the **_input matrix_** _Z_ 0 as 



where zero in the above matrix is used to replace the unknown response variable corresponding to _x_<sup>(</sup><sup>_n_+1)</sup> . Then, our goal is to predict _w⋆_<sup>_⊤x_(</sup><sup>_n_+1)given</sup><sup>_Z_0.In other words, the training data consists of</sup> pairs ( _Z_ 0 _, w⋆_<sup>_⊤x_(</sup><sup>_n_+1)) for</sup><sup>_x_(</sup><sup>_i_)</sup><sup>_∼DX_and</sup><sup>_w⋆∼DW_.We then consider training transformers over</sup> this data distribution. 

**_Self-attention layer without softmax._** Following (Schlag et al., 2021; von Oswald et al., 2023), we consider the linear self-attention layer. To motivate, we first briefly review the standard self-attention layer (Vaswani et al., 2017). Letting _Z ∈_ R<sup>(</sup><sup>_d_+1)</sup><sup>_×_(</sup><sup>_n_+1)</sup> be the input matrix with _n_ + 1 tokens in R<sup>_d_+1</sup> , a single-head self-attention layer denoted by Attn<sup>smax</sup> is a parametric map defined as 



where _Wv, Wk, Wq ∈_ R<sup>(</sup><sup>_d_+1)</sup><sup>_×_(</sup><sup>_d_+1)</sup> are the (value, key and query) weight matrices, and smax( _·_ ) is the softmax operator which applies softmax operation to each column of the input matrix. Note that the prompt is asymmetric since the label for _x_<sup>(</sup><sup>_n_+1)</sup> is excluded from the input. To reflect this asymmetric structure, the mask matrix _M_ is included in the attention. In our setting, we consider the self-attention layer that omits the softmax operation in (2). In particular, we reparameterize weights as _P_ := _Wv ∈_ R<sup>(</sup><sup>_d_+1)</sup><sup>_×_(</sup><sup>_d_+1)</sup> and _Q_ := _Wk⊤Wq ∈_ R( _d_ +1) _×_ ( _d_ +1) and consider 



At first glance, the omission of the softmax operation (3) might seem over-simplified. But, (von Oswald et al., 2023) proves such attention can implement gradient descent, and we will prove in Lemma 1 that it can also implement various algorithms to solve linear regression in-context. 

**_Architecture for prediction._** We now present the neural network architecture that will be used through- <u>out this paper. For the number of layers</u> _L_ , we define an _L_ **_-layer transformer_** as a stack of _L_ linear self-attention blocks. Formally, denoting by _Zℓ_ the output of the _ℓ_<sup>th</sup> layer attention, we define 



The scaling factor<sup>1</sup> _/n_ is used only for ease of notation and does not influence the expressive power of the transformer. Given _ZL_ , we define TF _L_ ( _Z_ 0; _{Pℓ, Qℓ}ℓ_ =0 _,_ 1 _,...L−_ 1) = _−_ [ _ZL_ ]( _d_ +1) _,_ ( _n_ +1), i.e., the ( _d_ + 1 _, n_ + 1)-th entry of _ZL_ . The reason for the minus sign is to be consistent with (von Oswald 

3 

et al., 2023), and we will clarify such a choice in Lemma 1. For training, the parameters are optimized to minimize in-context loss as 



**_Goal: the landscape analysis of the training objective functions._** We are interested in understanding how the optimization of _f_ leads to in-context learning. We investigate this question by analyzing its loss landscape. Such analysis is challenging due to two major reasons: _(i) f is non-convex in parameters {Pi, Qi} even for a single layer transformer. (ii) The cross-product structures in attention makes f a highly nonlinear function in its parameters._ Hence, we analyze a spectrum of settings from single-layer transformers to multi-layer transformers. For simpler settings such as single-layer transformers, we prove stronger results such as the full characterization of the global minimizers. For networks with more layers, we characterize the structure of critical points. Furthermore, we provide algorithmic interpretations of the critical points. Table 1 summarizes our results for various parameteric models. 

|Results|_x_<sup>(</sup><sup>_i_)</sup>|_w⋆_|Setting|Guarantees|
|---|---|---|---|---|
|Theorem 1|_N_(0_,_Σ)|_N_(0_, I_)|single-layer|global minimizers|
|Theorem 2|<br>_N_(0_, I_)|<br>_N_(0_, I_)|two-layer + symmetric (8)|<br>global minimizers|
|Theorem 3|_N_(0_,_Σ)|_N_(0_,_Σ<sup>_−_1</sup>)|<br>multi-layer + (8)|<br>critical points|
|Theorem 4|<br>_N_(0_,_Σ)|_N_(0_,_Σ<sup>_−_1</sup>)|multi-layer + (11)|critical points|
|Theorem 5|<br>_N_(0_, I_)|_N_(0_, I_)|<br>single-layer + ReLU activation|<br>global minimizers|



Table 1: Summary of our analyses for various models and input distributions. The additional conditions (8) and (11) are about the sparsity structure of parameters. In addition, “symmetric (8)” means we additionally impose the weights to be symmetric. 

**Remark 1** ( **_Optimizing_** (5) **_vs. practical transformer optimization_** ) **.** _Interestingly, a recent work by Ahn et al. (2023) reports that common optimization algorithms such as SGD/ADAM behave remarkably similarly on the (linear Transformers + linear regression) problem as they do on (practical transformers + real language modeling tasks). In particular, they reproduce several distinctive features of transformer optimization under a simple shallow linear transformer. This work suggests that (linear transformer + linear regression) may serve as a good proxy for understanding practical transformer optimization._ 

## **3 The global optimum for a single-layer transformer** 

For the single layer case of _L_ = 1, the following result characterizes the optimal parameters _P_ 0 and _Q_ 0 for the in-context loss (5). 

**Theorem 1** ( **Single-layer; non-isotropic data** ) **.** _Assume that vector x_<sup>(</sup><sup>_i_)</sup> _is sampled from N_ (0 _,_ Σ) _, i.e., a Gaussian with covariance_ Σ = _U_ Λ _U_<sup>_⊤_</sup> _where_ Λ = diag( _λ_ 1 _, . . . , λd_ ) _. Moreover, assume that w⋆ is sampled from N_ (0 _, Id_ ) _. Then, the following choice of parameters_ 



_is a global minimizer of f_ ( _P, Q_ ) _up to re-scaling, i.e., P_ 0 _← γP_ 0 _and Q_ 0 _← γ_<sup>_−_1</sup> _Q_ 0 _for a scalar γ._ 

See Appendix A for the proof of Theorem 1. In the specific case when the Gaussian is isotropic, i.e., Σ = _Id_ , the optimal _Q_ 0 has the following simple form 



Up to scaling, the above parameter configuration is equivalent to the parameters used by von Oswald et al. (2023) to perform one step of gradient descent. Thus, in the single-layer setting, the in-context loss is indeed minimized by a transformer that implements the gradient descent algorithm. 

4 

More generally, when the in-context samples are non-isotropic, the transformer learns to implement one step of a _preconditioned_ gradient descent as we shall detail in Lemma 1. Here the “preconditioning matrix” given in (6) has interesting properties: 

- When the number of samples _n_ is large, the first _d × d_ submatrix of _Q_ 0 approximates Σ<sup>_−_1</sup> , the inverse of the data covariance matrix, which is also close to the Gram matrix formed from _x_<sup>(1)</sup> _, . . . , x_<sup>(</sup><sup>_n_)</sup> . Hence the preconditioning can lead to considerably faster convergence rate when Σ is ill-conditioned. 

- Moreover, _n_<sup><u>1</u></sup> � _k_<sup>_λk_in (6) acts as a regularizer.It becomes more significant when</sup><sup>_n_is small and</sup> variance of the _x_<sup>(</sup><sup>_i_)</sup> ’s is high. Such an adjustment resembles structural risk minimization (Vapnik, 1999) where the regularization strength is adapted to the sample size. 

## **4 Multi-layer transformers with sparse parameters** 

Theorem 1 proves a single layer of linear attention can implement a single step of preconditioned gradient descent. Inspired by this result, we investigate the algorithmic power of the linear transformer architecture. We show that the model can implement various optimization methods even under sparsity constraints. In particular, we impose the following restrictions on the parameters: 



The next lemma proves that a forward-pass of a _L_ -layer transformer, with the parameter configuration (8) is the same as taking _L_ steps of gradient descent, preconditioned by _Aℓ_ . 

**Lemma 1** ( **_Forward pass as a preconditioned gradient descent_** ) **.** _Consider the L-layer linear transfomer parameterized by A_ 0 _, . . . , AL−_ 1 _as in_ (8) _. Let yℓ_<sup>(</sup><sup>_n_+1)</sup> _be the_ ( _d_ + 1 _, n_ + 1) _-th entry of the ℓ-th layer output, i.e., yℓ_<sup>(</sup><sup>_n_+1)</sup> = [ _Zℓ_ ]( _d_ +1) _,_ ( _n_ +1) _for ℓ_ = 1 _, . . . , L. Then, it holds that yℓ_<sup>(</sup><sup>_n_+1)</sup> = _−⟨x_<sup>(</sup><sup>_n_+1)</sup> _, wℓ_<sup>gd</sup><sup>_⟩where {w_</sup> _ℓ_<sup>gd</sup><sup>_} is defined as w_</sup> 0<sup>gd= 0</sup><sup>_and as follows for ℓ_= 1</sup><sup>_, . . . , L −_1</sup><sup>_:_</sup> 



See Subsection C.1 for a proof. The iterative scheme (9) includes various optimization methods including gradient descent with _Aℓ_ = _γℓId_ , and (adaptive) preconditioned gradient descent, where the preconditioner _Aℓ_ depends on the time step. In the upcoming sections, we characterize how the optimal _{Aℓ}_ are linked to the input distribution. 

### **4.1 Warm-up: optimal two-layer transformer with symmetric weights** 

For the rest of this section, we will study the optimal parameters for the in-context loss under the constraint of Eq. (8). Later in Section 5, we analyze the optimal model for a more general parameters. For a two-layer transformer, the next Theorem proves the optimal in-context loss obtains the simple gradient descent with adaptive coordinate-wise stepsizes. 

**Theorem 2** (Global optimality for the two-layer (symmetric) transformer) **.** _Consider the optimization of in-context loss for a two-layer transformer with the parameter configuration in Eq._ (8) _, and additionally assume that A_ 1 _, A_ 2 _are symmetric matrices. More formally, consider_ 



_Assume x_<sup>(</sup><sup>_i_)</sup><sup>_i.i.d._</sup> _∼ N_ (0 _, Id_ ) _and w⋆ ∼ N_ (0 _, Id_ ) _; then, there are diagonal matrices A_ 1 _and A_ 2 _that are a global minimizer of f ._ 

Combining the above result with Lemma 1 concludes that the two iterations of gradient descent with _coordinate-wise adaptive stepsizes_ achieve the minimal in-context loss for isotropic Gaussian inputs. Gradient descent with adaptive stepsizes such as Adagrad (Duchi et al., 2011) are widely used in machine learning. While Adagrad adjusts its stepsize based on the individual problem instance, the algorithm learned adjusts its stepsize to the underlying data distribution. 

5 

### **4.2 Multi-layer transformers** 

We now turn to the setting of general _L_ -layer transformers, for any positive integer _L_ . The next theorem proves that certain critical points of the in-context loss effectively implement a specific preconditioned gradient algorithm, where the preconditioning matrix is the inverse covariance of the input distribution. Before stating this result, let us first consider a motivating scenario in which the data-covariance matrix is non-identity: 

**_Linear regression with distorted view of the data:_** Suppose that _<u>w⋆</u> ∼N_ (0 _, I_ ) and the _latent_ covariates are _<u>x</u>_<sup>~~(~~1)</sup> _, . . . ,_ _<u>x</u>_<sup>~~(~~</sup><sup>_n_+1)</sup> , drawn i.i.d from _N_ (0 _, I_ ). We are given _y_<sup>(1)</sup> _, . . . , y_<sup>(</sup><sup>_n_)</sup> , with _y_<sup>(</sup><sup>_i_)</sup> = � _<u>x</u>_<sup>~~(~~</sup><sup>_i_)</sup> _,_ _<u>w⋆</u>_ �. However, we _do not observe_ the latent covariates _<u>x</u>_<sup>~~(~~</sup><sup>_i_)</sup> . Instead, we observe the _distorted_ covariates _x_<sup>(</sup><sup>_i_)</sup> = _W_ _<u>x</u>_<sup>~~(~~</sup><sup>_i_)</sup> , where _W ∈_ R<sup>_d×d_</sup> is a distortion matrix. Thus the prompt consists of ( _x_<sup>(1)</sup> _, y_<sup>(1)</sup> ) _, . . . ,_ ( _x_<sup>(</sup><sup>_n_)</sup> _, y_<sup>(</sup><sup>_n_)</sup> ), as well as _x_<sup>(</sup><sup>_n_+1)</sup> . The goal is still to predict _y_<sup>(</sup><sup>_n_+1)</sup> . Note that this setting is quite common in practice, when covariates are often represented in an arbitrary basis. Assume that Σ := _WW_<sup>_⊤_</sup> _≻_ 0. We verify from our definitions that for _w⋆_ := Σ<sup>_−_1</sup><sup>_/_2</sup> _<u>w⋆</u>_ , _y_<sup>(</sup><sup>_i_)</sup> = � _x_<sup>(</sup><sup>_i_)</sup> _, w⋆_ �. Furthermore, _x_<sup>(</sup><sup>_i_)</sup> _∼N_ (0 _,_ Σ) and _w⋆ ∼N_ (0 _,_ Σ<sup>_−_1</sup> ). From Lemma 1, the transformer with weight matrices _{A_ 0 _, . . . , AL−_ 1 _}_ implements preconditioned gradient descent with respect to <u>1</u> _Rw⋆_ ( _w_ ) = 2 _n_<sup>(</sup><sup>_w −w⋆_)</sup><sup>_T XX⊤_(</sup><sup>_w −w⋆_), with</sup><sup>_X_=</sup> � _x_<sup>(1)</sup> _, . . . , x_<sup>(</sup><sup>_n_)�</sup> . Under this loss, the Hessian matrix _∇_<sup>2</sup> _Rw⋆_ ( _w_ ) = 21 _n_<sup>_XX⊤_(atleastinthecaseoflarge</sup><sup>_n_).Foranyfixedprompt,Newton’s</sup> method corresponds to _Ai ∝_ � _XX_<sup>_⊤_�</sup><sup>_−_1</sup> , which makes the problem well-conditioned even if Σ is very degenerate. As we will see in Theorem 3 below, the choice of _Ai ∝_ Σ<sup>_−_1</sup> = E � _XX_<sup>_⊤_�</sup><sup>_−_1</sup> appears to be a _stationary point_ of the loss landscape, in expectation over prompts. 

Before stating the theorem, we introduce the following simplified notation: let _A_ := _{Ai}_<sup>_L_</sup> _i_ =0<sup>_−_1</sup><sup>_∈_</sup> R<sup>_L×d×d_</sup> . We use _f_ ( _A_ ) to denote the in-context loss of _f_ � _{Pi, Qi}_<sup>_L_</sup> _i_ =0<sup>_−_1</sup> � as defined in (5), when _Qi_ depends on _Ai_ , and _Pi_ is a constant matrix, as described in (8). 

**Theorem 3.** _Assume that x_<sup>(</sup><sup>_i_)</sup><sup>_iid_</sup> _∼N_ (0 _,_ Σ) _and w⋆ ∼N_ (0 _,_ Σ<sup>_−_1</sup> ) _, for i_ = 1 _, . . . , n, and for some_ Σ _≻_ 0 _. Consider the optimization of in-context loss for a k-layer transformer with the the parameter configuration in Eq._ (8) _given by:_ 



_Let S ⊂_ R<sup>_L×d×d_</sup> _be defined as follows: A ∈S if and only if for all i_ = 0 _, . . . , L −_ 1 _, there exists scalars ai ∈_ R _such that Ai_ = _ai_ Σ<sup>_−_1</sup> _. Then_ 



_where ∇Aif denotes derivative wrt the Frobenius norm ∥Ai∥F ._ 

As discussed in the motivation above, under the setting of _Ai_ = _ai_ Σ<sup>_−_1</sup> , the linear transformer implements an algorithm that is reminiscent of Newton’s method (as well as a number of other adaptive algorithms such as the full-matrix variant of Adagrad); these can converge significantly faster than vanilla gradient descent when the problem is ill-conditioned. The proposed parameters _Ai_ in Theorem 3 are also similar to _Ai_ ’s in Theorem 1 when _n_ is large. However, in contrast to Theorem 1, there is no trade-off with statistical robustness; this is because _w⋆_ has covariance matrix Σ<sup>_−_1</sup> in the Theorem 3, while Theorem 1 has isotropic _w⋆_ . 

Unlike our prior results, Theorem 3 only guarantees that the set _S_ of transformer prameters satisfying � _Ai ∝_ Σ<sup>_−_1�</sup><sup>_L_</sup> _i_ =0<sup>_−_1</sup><sup>_essentially_2contains critical points of the in-context loss.However, in the next</sup> section, we show experimentally that this choice of _Ai_ ’s does indeed seem to be recovered by training. 

We defer the proof of Theorem 3 to Subsection B.2. Due to the complexity of the transformer function, even verifying critical points can be challenging. We show that the in-context loss can be equivalently written as (roughly) a matrix polynomial involving the weights at each layer. By 

> 2A subtle issue is that the infimum may not be attained, so it is possible that _S_ contains points with arbitrarily small gradient, but does not contain a point with exactly 0 gradient. 

6 

exploiting invariances in the underlying distribution of prompts, we construct a flow, contained entirely in _S_ , whose objective value decreases as fast as gradient flow. Since _f_ is lower bounded, we conclude that there must be points in _S_ whos gradient is arbitrarily small. 

### **4.3 Experimental validations for Theorem 3** 

We present here an empirical verification of our results in Theorem 3. We consider the ICL loss for linear regression. The dimension is _d_ = 5, and the number of training samples in the prompt is _n_ = 20. Both _x_<sup>(</sup><sup>_i_)</sup> _∼N_ (0 _,_ Σ) and _w⋆ ∼N_ (0 _,_ Σ<sup>_−_1</sup> ), where Σ = _U_<sup>_T_</sup> _DU_ , where _U_ is a uniformly random orthogonal matrix, and _D_ is a fixed diagonal matrix with entries (1 _,_ 1 _,_ 0 _._ 25 _,_ 0 _._ 0625 _,_ 1). 

We optimizes _f_ for a three-layer linear transformer using ADAM, where the matrices _A_ 0 _, A_ 1 _,_ and _A_ 2 are initialized by i.i.d. Gaussian matrices. Each gradient step is computed from a minibatch of size 20000, and we resample the minibatch every 100 steps. We clip the gradient of each matrix to 0.01. All plots are averaged over 5 runs with different _U_ (i.e. Σ) sampled each time. 

Figure 1d plots the average loss. We observe that the training converges to an almost 0 value, suggesting the convergence to global minimum. The parameters at convergence match the stationary point introduced in Theorem 3, and indeed appear to be globally optimal. 

To quantify the similarity between _A_ 0 _, A_ 1 _, A_ 2 and Σ<sup>_−_1</sup> (up to scaling), we use the _normalized Frobe-_ _<u>∥M</u> −α·I∥ d nius norm distance_ : Dist( _M, I_ ) := min _α ∥M ∥F_ , (equivalent to choosing _α_ := _d_<sup><u>1</u></sup> � _i_ =1<sup>_M_[</sup><sup>_i, i_]).</sup> This is essentially the projection distance of<sup>_M_</sup> _/∥M ∥F_ onto the space of scaled identity matrices. 

We plot Dist ( _Ai, I_ ), averaged over 5 runs, against iteration in Figures 1a,1b,1c. In each plot, the blue line represents Dist(Σ<sup>1</sup><sup>_/_2</sup> _Ai_ Σ<sup>1</sup><sup>_/_2</sup> _, I_ ), and we verify that the optimal parameters are converging to the critical point introduced in Theorem 3, which implements preconditioned gradient descent. The red line represents Dist( _Ai, I_ ); it remains constant indicating that the trained transformer is not implementing plain gradient descent. Figures 2a–2c visualize each Σ<sup>1</sup><sup>_/_2</sup> _Ai_ Σ<sup>1</sup><sup>_/_2</sup> matrix at the end of training to further validate that the learned parameter is as described in Theorem 3. 



<!-- Start of picture text -->
1.0 1.0 1.0<br>1.5<br>0.8 0.8 0.8 1.0<br>0.5<br>0.6 A0 0.6 A1 0.6 0.0<br>0.4 1/2A0 1/2 0.4 1/2A1 1/2 0.4 0.5<br>0.2 0.2 0.2 A1/22 A2 1/2 1.01.5<br>0.0 0.0 0.0 2.0<br>0 5000 10000 15000 20000 25000 30000 0 5000 10000 15000 20000 25000 30000 0 5000 10000 15000 20000 25000 30000 0 5000 10000 15000 20000 25000 30000<br>Iteration Iteration Iteration Iteration<br>(a) Dist(Σ 1 / 2 A 0Σ 1 / 2 , I ) (b) Dist(Σ 1 / 2 A 1Σ 1 / 2 , I ) (c) Dist(Σ 1 / 2 A 2Σ 1 / 2 , I ) (d) log(Loss)<br>log(Loss)<br>Distance to Id Distance to Id Distance to Id<br><!-- End of picture text -->

Figure 1: Plots for verifying convergence of general linear transformer, defined in Theorem 3. Figure (d) shows convergence of loss to 0. Figures (a),(b),(c) illustrate convergence of _Ai_ ’s to identity. More specifically, the blue line represents Dist(Σ<sup>1</sup><sup>_/_2</sup> _Ai_ Σ<sup>1</sup><sup>_/_2</sup> _, I_ ), which measures the convergence to the critical point introduced in Theorem 3 (corresponding to Σ<sup>_−_1</sup> -preconditioned gradient descent). The red line represents Dist( _Ai, I_ ); it remains constant indicating that the trained transformer is not implementing plain gradient descent. 

## **5 Multi-layer transformers beyond standard optimization methods** 

In this section, we study the more general setting of 



Note that _Ai, Bi_ are not constrained to be symmetric. Similar to Section 4, we introduce the following simplified notation: let _A_ := _{Ai}_<sup>_L_</sup> _i_ =0<sup>_−_1</sup><sup>_∈_R</sup><sup>_L×d×d_and</sup><sup>_B_:=</sup><sup>_{Bi}L_</sup> _i_ =0<sup>_−_1</sup><sup>_∈_R</sup><sup>_L×d×d_.We use</sup><sup>_f_(</sup><sup>_A, B_)</sup> to denote the in-context loss of _f_ � _{Pi, Qi}_<sup>_L_</sup> _i_ =0<sup>_−_1</sup> � as defined in (5), when _Pi_ and _Qi_ depend on _Bi_ and _Ai_ as described in (11). 

7 



<!-- Start of picture text -->
-0.69 -0.00 -0.00 -0.00 -0.00<br>-0.00 -0.69 -0.00 -0.00 -0.00<br>-0.00 -0.00 -0.69 -0.00 -0.00<br>0.00 -0.00 -0.00 -0.69 -0.00<br>0.00 -0.00 -0.00 0.00 -0.69<br><!-- End of picture text -->





<!-- Start of picture text -->
0 -0.42 -0.01 0.00 -0.00 0.00<br>1 -0.00 -0.43 -0.00 0.00 -0.00<br>2 0.00 -0.00 -0.43 0.00 -0.00<br>3 -0.00 0.00 -0.00 -0.42 -0.00<br>4 -0.00 0.00 -0.00 -0.00 -0.43<br><!-- End of picture text -->





<!-- Start of picture text -->
0 -1.71 -0.01 0.00 -0.01 -0.00<br>1 -0.00 -1.72 -0.01 -0.00 -0.00<br>2 0.01 -0.00 -1.72 0.01 -0.00<br>3 0.00 -0.00 0.00 -1.69 0.00<br>4 -0.00 0.00 -0.01 0.01 -1.72<br><!-- End of picture text -->



(a) Visualization of Σ<sup>1</sup><sup>_/_2</sup> _A_ 0Σ<sup>1</sup><sup>_/_2</sup> (b) Visualization of Σ<sup>1</sup><sup>_/_2</sup> _A_ 1Σ<sup>1</sup><sup>_/_2</sup> (c) Visualization of Σ<sup>1</sup><sup>_/_2</sup> _A_ 2Σ<sup>1</sup><sup>_/_2</sup> 

Figure 2: Visualization of learned weights for the setting of Theorem 3. We visualize each Σ<sup>1</sup><sup>_/_2</sup> _Ai_ Σ<sup>1</sup><sup>_/_2</sup> matrix at the end of training. Note that the optimized weights match the stationary point discussed in Theorem 3. 

With this relaxed parameter configuration, it turns out transformers can learn algorithms beyond the conventional preconditioned gradient descent. The next theorem asserts the possibility of learning a novel preconditioned gradient method. Let _L_ be a fixed but arbitrary number of layers. 

> <sup>_iid_</sup> **Theorem 4.** _Let_ Σ _denote any PSD matrix. Assume that x_<sup>(</sup><sup>_i_)</sup> _∼N_ (0 _,_ Σ) _and w⋆ ∼N_ (0 _,_ Σ<sup>_−_1</sup> ) _, for i_ = 1 _, . . . , n, and for some_ Σ _≻_ 0 _. Consider the optimization of in-context loss for a L-layer linear transformer with the the parameter configuration in Eq._ (11) _given by:_ 



_Let S ⊂_ R<sup>2</sup><sup>_×L×d×d_</sup> _be defined as follows:_ ( _A, B_ ) _∈S if and only if for all i ∈{_ 0 _, . . . , k}, there exists scalars ai, bi ∈_ R _such that Ai_ = _ai_ Σ<sup>_−_1</sup> _and Bi_ = _biI. Then_ 



_where ∇Aif denotes derivative wrt the Frobenius norm ∥Ai∥F ._ 

In words, parameter matrices in _S_ implement the following algorithm: � _Ai_ = _ai_ Σ<sup>_−_1�</sup><sup>_L_</sup> _i_ =0<sup>_−_1plays the</sup> role of a distribution-dependent preconditioner for the gradient steps. At the same time, _Bi_ = _biI_ transforms the covariates themselves to make the Gram matrix have better condition number with each iteration. When the Σ = _I_ , the algorithm implemented by _Ai ∝ I, bi ∝ I_ is exactly the GD++ algorithm proposed in (von Oswald et al., 2023) (up to stepsize). 

The result in (12) says that the set _S essentially_<sup>3</sup> contains critical points of the in-context loss _f_ ( _A, B_ ). In the next section, we provide empirical evidence that the trained transformer parameters do in fact converge to a point in _S_ . 

### **5.1 Experimental validations for Theorem 4** 

The experimental setup is similar to Subsection 4.3: we consider ICL for linear regression with _n_ = 10 _, d_ = 5, with _x_<sup>(</sup><sup>_i_)</sup> _∼N_ (0 _,_ Σ) and _w⋆ ∼N_ (0 _,_ Σ<sup>_−_1</sup> ), where Σ = _U_<sup>_T_</sup> _DU_ , where _U_ is a uniformly random orthogonal matrix, and _D_ is a fixed diagonal matrix with entries (1 _,_ 1 _,_ 0 _._ 25 _,_ 0 _._ 0625 _,_ 1). We train a three-layer linear transformer, under the constraints in (11) which is less restrictive than (8) in Subsection 4.3. We train the matrices _A_ 0 _, A_ 1 _, A_ 2 _, B_ 0 _, B_ 1<sup>4</sup> using ADAM with the same setup as in Section Subsection 4.3. We repeat this experiment 5 times with different random seeds, each time we sample a different _U_ (i.e. Σ). 

> 3Once again, similar to the case of Theorem 3, the infimum may not be attained, so it is possible that _S_ contains points with arbitrarily small gradient, but does not contain a point with exactly 0 gradient. 

> 4Note that the objective function does not depend on _B_ 2. 

8 

In Figure 3c, we plot the in-context loss through the iterations of L-BFGS; the loss appears to be converging to 0, suggesting that parameters are converging to the global minimum. 



<!-- Start of picture text -->
1.0 1.0 2<br>B0 B1<br>0.8 0.8 0<br>0.6 0.6 2<br>0.4 0.4 4<br>0.2 0.2 6<br>0.0 0 20000 40000Iteration60000 80000 100000 0.0 0 20000 40000Iteration60000 80000 100000 0 20000 40000Iteration60000 80000 100000<br>(a) Dist( B 0 , I ) (b) Dist( B 1 , I ) (c) log(Loss)<br>1.0 1.0 1.0<br>0.8 0.8 0.8<br>0.6 A0 0.6 0.6<br>0.4 1/2A0 1/2 0.4 0.4<br>0.2 0.2 A1 0.2 A2<br>1/2A1 1/2 1/2A2 1/2<br>0.0 0.0 0.0<br>0 20000 40000 60000 80000 100000 0 20000 40000 60000 80000 100000 0 20000 40000 60000 80000 100000<br>Iteration Iteration Iteration<br>(d) Distances for  A 0 (e) Distances for  A 1 (f) Distances for  A 2<br>log(Loss)<br>Distance to Id Distance to Id<br>Distance to Id Distance to Id Distance to Id<br><!-- End of picture text -->

Figure 3: Plots for verifying convergence of general linear transformer, defined in Theorem 4. Figure (c) shows convergence of loss to 0. Figures (a),(b) illustrate convergence of _B_ 0 _, B_ 1 to identity. Figures (d),(e),(f) illustrate convergence of _Ai_ ’s to Σ<sup>_−_1</sup> . 

We next verify that the parameters at convergence are consistent with Theorem 4. We will once again use Dist( _M, I_ ) to measure the distance from _M_ to the identity matrix, up to scaling (see Subsection 4.3 for definition of _Dist_ ). Figures 3a and 3b show that _B_ 0 and _B_ 1 are close to identity, as Dist( _Bi, I_ ) appears to be decreasing to 0. Figures 3d, 3e and 3f plot Dist( _Ai, I_ ) (red line) and Dist(Σ<sup>1</sup><sup>_/_2</sup> _Ai_ Σ<sup>1</sup><sup>_/_2</sup> _, I_ ) (blue line); the results here suggest that _Ai_ is converging to Σ<sup>_−_1</sup> , up to scaling. In Figures 3a and 3b, we observe that _B_ 0 and _B_ 1 also converge to the identity matrix ( _without_ left and right multiplication by Σ<sup>1</sup><sup>_/_2</sup> ), consistent with Theorem 4. 

We visualize each of _B_ 0 _, B_ 1 in Figure 4 and _A_ 0 _, A_ 1 _, A_ 2 in Figure 5a-5c at the end of training. We highlight two noteworthy observations: 

1. Let _Xk ∈_ R<sup>_d×n_</sup> denote the first _d_ rows of _Zk_ , which are the output at layer _k −_ 1 defined in (4). Then the update to _Xk_ is _Xk_ +1 = _Xk_ + _BkXkMXk_<sup>_TAkXk≈Xk_+1=</sup> _Xk_ � _I −|akbk|MXk_<sup>_TXk_</sup> �, where _M_ is a mask defined in (2). As noted by von Oswald et al. (2023), this may be motivated by curvature correction. 

2. As seen in Figures 5a-5c in the Appendix, _∥A_ 0 _∥≤∥A_ 1 _∥≤∥A_ 2 _∥_ that implies the transformer implements gradient descent with a small stepsize at the beginning and a large stepsize at the end. This makes intuitive sense as _X_ 2 is better-conditioned compared to _X_ 1, due to the choice of _B_ 0 _, B_ 1. This can be contrasted with the plots in Figures (2a)-(2c), where similar trends are not as pronounced because _Bi_ ’s are constrained to be 0. 

## **6 Discussion** 

We take a first step toward proving that transformers can learn algorithms when trained over a set of random problem instances. Specifically, we investigate the possibility of learning gradient based methods when training on the in-context loss for linear regression. For a single layer transformer, we prove that the global minimum corresponds to a single iteration of preconditioned gradient descent. For multiple layers, we show that certain parameters that correspond to the critical points of the in-context loss can be interpreted as a broad family of adaptive gradient-based algorithms. 

9 







<!-- Start of picture text -->
0.8 0.5<br>0 0.90 -0.00 0.00 0.00 -0.00 0 0.56 -0.00 0.00 0.00 -0.00<br>0.4<br>1 0.00 0.91 0.00 -0.01 -0.00 0.6 1 -0.00 0.57 -0.00 -0.00 0.00<br>0.3<br>2 0.01 0.01 0.90 -0.02 -0.01 2 -0.00 -0.00 0.57 0.00 0.00<br>0.4<br>3 -0.00 -0.00 0.00 0.90 -0.01 3 0.00 0.00 -0.00 0.56 0.00 0.2<br>0.2<br>4 0.00 -0.00 -0.00 0.01 0.91 4 -0.00 -0.00 0.00 0.00 0.57 0.1<br>0 1 2 3 4 0 1 2 3 4<br>0.0 0.0<br><!-- End of picture text -->

Figure 4: Visualization of optimized weight matrices _B_ 0 (left) and _B_ 1 (right). One can see that the weight pattern matches the stationary point analyzed in Theorem 4. Matrices _A_ 0, _A_ 1 and _A_ 2 are similar to Figure 2, and are visualized in Figure 5 in Appendix D. 

We discuss below two interesting future directions. 

**Beyond linear attention.** The standard transformer architecture comes with nonlinear activations in attention. Hence, the natural question here is to ask the effect of nonlinear activations for our main results. Empirically, von Oswald et al. (2023) have observed that for linear regression task, softmax activations generally degrade the prediction performance, and in particular, softmax transformers typically need more attention heads to match their performance with that of linear transformers. 

As a first step analysis, we consider the nonlinear attention defined as 

Attn<sup>_σ_</sup> _P,Q_<sup>(</sup><sup>_Z_) :=</sup><sup>_PZMσ_(</sup><sup>_Z⊤QZ_)</sup> where _σ_ : R _→_ R is applied entry-wise. 

The following result is an analog of Theorem 1 for single-layer nonlinear attention. It characterizes a global minimizer for this setting with ReLU activation. Here, our choice of ReLU activation was motivated by Wortsman et al. (2023) who observed that ReLU attention matches the performance of softmax attention for vision transformers. 

**Theorem 5.** _Consider the single layer nonlinear attention setting with σ_ = ReLU _. Assume that vector x_<sup>(</sup><sup>_i_)</sup> _is sampled from N_ (0 _, Id_ ) _. Moreover, assume that w⋆ is sampled from N_ (0 _, Id_ ) _. Consider the parameter configuration P_ 0 _, Q_ 0 _where we additionally assume that the last row of Q_ 0 _is zero. Then, the following parameters form a global minimizer of the corresponding in-context loss:_ 



The proof of Theorem 5 involves an instructive argument and leverages tools from (Erdogdu et al., 2016); we defer it to Subsection A.4. Thus, for isotropic Gaussian data, the structure of global minimum under ReLU attention is similar to the global minimum with linear attention, established in Theorem 1 (specifically the minimizer for the isotropic date given in (7)). 

**Refined landscape analysis for multilayer transformer.** Theorem 4 proves that a stationary point of the in-context loss corresponds to implementing a preconditioned gradient method. However, we do not prove that all critical points of the non-convex objective lead to similar optimization methods. In fact, in Lemma 4 in Appendix B, we prove that the in-context loss can have multiple critical points. It will be interesting to analyze the set of all critical points and try to understand their algorithmic interpretations, as well as quantify their (sub)optimality. 

## **Acknowledgments and Disclosure of Funding** 

We thank Ekin Akyürek, Johannes von Oswald, Alex Gu and Joshua Robinson for helpful discussions. Kwangjun Ahn was supported by the ONR grant (N00014-20-1-2394) and MIT-IBM Watson as well as a Vannevar Bush fellowship from Office of the Secretary of Defense. Kwangjun Ahn also acknowledges support from the Kwanjeong Educational Foundation. Xiang Cheng acknowledges support from NSF CCF-2112665 (TILOS AI Research Institute). Hadi Daneshmand acknowledges support from NSF TRIPODS program (award DMS-2022448). Suvrit Sra acknowledges support from an NSF CAREER grant (1846088), and NSF CCF-2112665 (TILOS AI Research Institute). 

10 

## **References** 

- Kwangjun Ahn, Xiang Cheng, Minhak Song, Chulhee Yun, Ali Jadbabaie, and Suvrit Sra. Linear attention is (maybe) all you need (to understand transformer optimization). _arXiv preprint arXiv:2310.01082_ , 2023. 

- Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. _International Conference on Learning Representations_ , 2022. 

- Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 1, context-free grammar. _arXiv preprint arXiv:2305.13673_ , 2023. 

- Noga Alon and Joel H Spencer. _The probabilistic method_ . John Wiley & Sons, 2016. 

- Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. Gpt-neox-20b: An open-source autoregressive language model. _Proceedings of BigScience – Workshop on Challenges & Perspectives in Creating Large Language Models_ , 2022. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Neural Information Processing Systems_ , 2020. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1_ , 2019. 

- John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. _Journal of machine learning research_ , 12(7), 2011. 

- Benjamin L Edelman, Surbhi Goel, Sham Kakade, and Cyril Zhang. Inductive biases and variable creation in self-attention mechanisms. In _International Conference on Machine Learning (ICML)_ , 2022. 

- Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. _Transformer Circuits Thread_ , 2021. https://transformer-circuits.pub/2021/framework/index.html. 

- Murat A Erdogdu, Lee H Dicker, and Mohsen Bayati. Scaled least squares estimator for glms in large-scale problems. _Advances in Neural Information Processing Systems_ , 29, 2016. 

- Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. _Advances in Neural Information Processing Systems_ , 35:30583–30598, 2022. 

- Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. _arXiv preprint arXiv:2301.13196_ , 2023. 

- Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines. _arXiv preprint arXiv:1410.5401_ , 2014. 

- Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. _Neural computation_ , 1997. 

- Stanisław Jastrzebski, Devansh Arpit, Nicolas Ballas, Vikas Verma, Tong Che, and Yoshua Bengio. Residual connections encourage iterative inference. In _International Conference on Learning Representations_ , 2018. URL `https://openreview.net/forum?id=SJa9iHgAZ` . 

- Ke Li and Jitendra Malik. Learning to optimize. In _International Conference on Learning Representations_ , 2017. 

11 

Opher Lieber, Or Sharir, Barak Lenz, and Yoav Shoham. Jurassic-1: Technical details and evaluation. _White Paper. AI21 Labs_ , 2021. 

- Arvind Mahankali, Tatsunori B Hashimoto, and Tengyu Ma. One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. _arXiv preprint arXiv:2307.03576_ , 2023. 

- Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. Metaicl: Learning to learn in context. _Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , 2021. 

- Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads. _Transformer Circuits Thread_ , 2022. 

- Jorge Pérez, Pablo Barceló, and Javier Marinkovic. Attention is turing complete. _The Journal of Machine Learning Research_ , 2021. 

- Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. _arXiv preprint arXiv:2112.11446_ , 2021. 

- Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In _International Conference on Machine Learning_ , pages 9355–9366. PMLR, 2021. 

- Hava T Siegelmann and Eduardo D Sontag. On the computational power of neural nets. In _Proceedings of Workshop on Computational learning theory_ , 1992. 

- Vladimir Vapnik. _The nature of statistical learning theory_ . Springer science & business media, 1999. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 2017. 

- Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In _International Conference on Machine Learning_ , pages 35151–35174. PMLR, 2023. 

- Colin Wei, Yining Chen, and Tengyu Ma. Statistically meaningful approximation: a case study on approximating turing machines with transformers. _Advances in Neural Information Processing Systems_ , 35:12071–12083, 2022. 

- Mitchell Wortsman, Jaehoon Lee, Justin Gilmer, and Simon Kornblith. Replacing softmax with relu in vision transformers. _arXiv preprint arXiv:2309.08586_ , 2023. 

- Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. _International Conference on Learning Representations_ , 2021. 

- Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear models in-context. _arXiv preprint arXiv:2306.09927_ , 2023. 

- Haoyu Zhao, Abhishek Panigrahi, Rong Ge, and Sanjeev Arora. Do transformers parse while predicting the masked word? _arXiv preprint arXiv:2303.08117_ , 2023. 

12 

# **Appendix** 

|**A **|**Proo**|**fs for the single layer case**|**13**|
|---|---|---|---|
||A.1|Rewriting the loss function . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>13|
||A.2|Warm-up: proof for the isotropic data<br>. . . . . . . . . . . . . . . . . . . . .|. . .<br>14|
||A.3|Proof for the non-isotropic case. . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>16|
||A.4|Proof for non-linear attentions (Theorem 5) . . . . . . . . . . . . . . . . . .|. . .<br>18|
|**B**|**Proo**|**fs for the multi-layer case**|**21**|
||B.1|Proof of Theorem 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>21|
||B.2|Proof of Theorem 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>21|
||B.3|Proof of Theorem 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>24|
||B.4|Equivalence under permutation . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>33|
|**C **|**Auxi**|**liary Lemmas**|**33**|
||C.1|Proof of Lemma 1 (Equivalence to Preconditioned Gradient Descent)<br>. . . .|. . .<br>33|
||C.2|Reformulating the in-context loss. . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>35|
|**D **|**Add**|**itional experimental results**|**36**|



## **A Proofs for the single layer case** 

In this section, we prove our characterization of global minima for the single layer case (Theorem 1). We begin by simplifying the loss into a more concrete form. Throughout the proof, we will write _P, Q_ instead of _P_ 0 _, Q_ 0 for brevity. 

### **A.1 Rewriting the loss function** 

Recall the in-context loss (5) for the single layer case _f_ ( _P, Q_ ) is defined as: 



From the definition of attention given in (3), one can further spell out the expression _Z_ 0 + <u>1</u> _n_<sup>Attn</sup><sup>_P,Q_(</sup><sup>_Z_0) using the notation</sup><sup>_Z_0= [</sup><sup>_z_(1)</sup><sup>_z_(2)</sup><sup>_· · ·z_(</sup><sup>_n_+1)] as follows:</sup> 



Thus, the last column of the above matrix can be expressed as 



where note that the summation is for _i_ = 1 _,_ 2 _, . . . , n_ due to the mask matrix _M_ . Therefore, letting _b_<sup>_⊤_</sup> be the last row of _P_ , and _A ∈_ R<sup>_d_+1</sup><sup>_,d_</sup> be the first _d_ columns of _Q_ (as we did in (14)), then � _Z_ 0 + _n_<sup><u>1</u>Attn</sup><sup>_P,Q_(</sup><sup>_Z_0)</sup> �( _d_ +1) _,_ ( _n_ +1)<sup>can be written as</sup> 



13 

in other words, _f_ ( _P, Q_ ) only depends on the parameter _b_ and _A_ . Henceforth, we will write _f_ ( _P, Q_ ) as _f_ ( _b, A_ ). Let us summarize our conclusion so far since it’s crucial for the analysis to follow. 

**Conclusion so far:** A careful inspection reveals that the in-context loss only depends on the last row of _P_ and the first _d_ columns of _Q_ . Thus, consider the following parametrization 



Now with this parametrization, the in-context loss can be written as _f_ ( _b, A_ ) := _f_ ([0 _b_ ]<sup>_⊤_</sup> _,_ [ _A_ 0]). 

Now, let us spell out _f_ ( _b, A_ ) based on (13) as follows: 



where we used the notation G := _n_<sup><u>1</u></sup> � _i_<sup>_z_(</sup><sup>_i_)</sup><sup>_z_(</sup><sup>_i_)</sup><sup>_⊤_to simplify.We now analyze the global minima of</sup> this loss function. To illustrate the proof idea clearly, we begin with the proof for the simpler case of isotropic data. 

### **A.2 Warm-up: proof for the isotropic data** 

As a warm-up, we first prove the result for the special case where _x_<sup>(</sup><sup>_i_)</sup> is sampled from _N_ (0 _, Id_ ). 

**_1. Decomposing the loss function into components._** Writing _A_ = [ _a_ 1 _a_ 1 _· · · ad_ ], and use the fact that E[ _x_<sup>(</sup><sup>_n_+1)</sup> [ _j_ ] _x_<sup>(</sup><sup>_n_+1)</sup> [ _j_<sup>_′_</sup> ]] = 0 for _j_ = _j_<sup>_′_</sup> and E[ _x_<sup>(</sup><sup>_n_+1)</sup> [ _j_ ]<sup>2</sup> ] = 1, we get 



The key idea is to characterize the global minima of each component in the summation separately. Another key idea is to reparametrize the cost function given the following identity: 



where we use the notation _⟨X, Y ⟩_ := Tr( _XY_<sup>_⊤_</sup> ) for two matrices _X_ and _Y_ here and below. Given the above identity, we define each component in the summation as follows. 



**_2. Characterizing_** **_<u>global minima of each component.</u>_** To characterize the global minima of each objective, we prove the following result. 

**Lemma 2** ( **Global minima of each component** ) **.** _Suppose that x_<sup>(</sup><sup>_i_)</sup> _is sampled from N_ (0 _, Id_ ) _and w⋆ is sampled from N_ (0 _, Id_ ) _. Consider the following objective (here, ⟨X, Y ⟩_ := Tr( _XY_<sup>_⊤_</sup> ) _for two matrices X and Y )_ 



_Then a global minimum is given as_ 



_where Ei_ 1 _,i_ 2 _is the matrix whose_ ( _i_ 1 _, i_ 2) _-th entry is_ 1 _, and the other entries are zero._ 

14 

**Proof of Lemma 2.** Note first that _fj_ is convex in _X_ . Hence, in order to show that matrix _Xj_ is the global optimum of _fj_ , it suffices to show that the gradient vanishes at that point, in other words, _∇fj_ ( _Xj_ ) = 0. To verify this, let us compute the gradient of _fj_ : for a matrix _X_ , 

_∇fj_ ( _X_ ) = 2 E [ _⟨_ G _, X⟩_ G] + 2 E [ _w⋆_ [ _j_ ] G] _,_ 

where we recall that G is defined as 



To verify that the gradient is equal to zero, let us first compute E [ _w⋆_ [ _j_ ] G]. For each _i_ = 1 _, . . . , n_ , note that E[ _w⋆_ [ _j_ ] _x_<sup>(</sup><sup>_i_)</sup> _x_<sup>(</sup><sup>_i_)</sup><sup>_⊤_</sup> ] = _O_ because E[ _w⋆_ ] = 0. Moreover, E[ _w⋆_ [ _j_ ] ( _y_<sup>(</sup><sup>_i_)</sup> )<sup>2</sup> ] = 0 because _w⋆_ is symmetric, i.e., _w⋆_ = _d −w⋆_ , and _y_ ( _i_ ) = _⟨w⋆, x_ ( _i_ ) _⟩_ . Lastly, for _k_ = 1 _,_ 2 _, . . . , d_ , we have 

E[ _w⋆_ [ _j_ ] _y_<sup>(</sup><sup>_i_)</sup> _x_<sup>(</sup><sup>_i_)</sup> [ _k_ ]] = E[ _w⋆_ [ _j_ ] _⟨w⋆, x_<sup>(</sup><sup>_i_)</sup> _⟩ x_<sup>(</sup><sup>_i_)</sup> [ _k_ ]] = E � _w⋆_ [ _j_ ]<sup>2</sup> _x_<sup>(</sup><sup>_i_)</sup> [ _j_ ] _x_<sup>(</sup><sup>_i_)</sup> [ _k_ ]� = 1 [ _j_ = _k_ ] (16) because E[ _w⋆_ [ _i_ ] _w⋆_ [ _j_ ]] = 0 for _i_ = _j_ . Combining the above calculations, it follows that 



In order to compute E [ _⟨_ G _, X⟩_ G], let us compute E [ _⟨_ G _, Ei,i′⟩_ G] for _i, i_<sup>_′_</sup> = 1 _, . . . , d_ + 1. Without loss of generality, _i ≥ i_<sup>_′_</sup> . First of all We now compute compute E [ _⟨_ G _, Ed_ +1 _,j⟩_ G]. Note first that 

_⟨_ G _, Ed_ +1 _,j⟩_ = � _⟨w⋆, x_<sup>(</sup><sup>_i_)</sup> _⟩ x_<sup>(</sup><sup>_i_)</sup> [ _j_ ] _. i_ 

Hence, it holds that 



because E[ _w⋆_ ] = 0. Next, we have 

_d_ because _w⋆_ = _−w⋆_ . Lastly, we compute 



To that end, note that for _j_ = _j_<sup>_′_</sup> , 



and 

where the last case follows from the fact that the fourth moment of Gaussian is 3 and 



Combining the above calculations together, we arrive at 



Therefore, combining (17) and (19), the results follows. 

15 

### **_3. Combining_** **_<u>global minima of each component.</u>_** From Lemma 2, it follows that 



is the unique global minimum of _fj_ . Hence, _b_ and _A_ = [ _a_ 1 _a_ 1 _· · · ad_ ] achieve the global minimum of _f_ ( _b, A_ ) =<sup>�</sup><sup>_d_</sup> _j_ =1<sup>_fj_(</sup><sup>_ba⊤_</sup> _j_<sup>) if they satisfy</sup> 



This can be achieve by the following choice: 



where **e** _j_ is the _j_ -th coordinate vector. This choice precisely corresponds to 



We next move on to the non-isotropic case. 

### **A.3 Proof for the non-isotropic case** 

**_1. Diagonal covariance case._** We first consider the case where _x_<sup>(</sup><sup>_i_)</sup> is sampled from _N_ (0 _,_ Λ) where Λ = diag( _λ_ 1 _, . . . , λd_ ) and _w⋆_ is sampled from _N_ (0 _, Id_ ). We prove the following generalization of Lemma 2. 

**Lemma 3.** _Suppose that x_<sup>(</sup><sup>_i_)</sup> _is sampled from N_ (0 _,_ Λ) _where_ Λ = diag( _λ_ 1 _, . . . , λd_ ) _and w⋆ is sampled from N_ (0 _, Id_ ) _. Consider the following objective_ 



_Then a global minimum is given as_ 



_where Ei_ 1 _,i_ 2 _is the matrix whose_ ( _i_ 1 _, i_ 2) _-th entry is_ 1 _, and the other entries are zero._ 

**Proof of Lemma 3.** Similarly to the proof of Lemma 2, it suffices to check that 



where we recall that G is defined as 



A similar calculation as the proof of Lemma 2 yields 



Here the factor of _λj_ comes from the following generalization of (16): 



Next, we compute E [ _⟨_ G _, Ed_ +1 _,j⟩_ G]. Again, we follow a similar calculation to the proof of Lemma 2 except that this time we use the following generalization of (18): 



16 

where the last line follows since 



Therefore, we have 



Therefore, combining (20) and (21), the results follows. 

Now we finish the proof. From Lemma 2, it follows that 



is the unique global minimum of _fj_ . Hence, _b_ and _A_ = [ _a_ 1 _a_ 1 _· · · ad_ ] achieve the global minimum of _f_ ( _b, A_ ) =<sup>�</sup><sup>_d_</sup> _j_ =1<sup>_fj_(</sup><sup>_b, Aj_) if they satisfy</sup> 



This can be achieve by the following choice: 



where **e** _j_ is the _j_ -th coordinate vector. This choice precisely corresponds to 



**_2. Non-diagonal covariance case (the setting of Theorem 1)._** We finally prove the general result of Theorem 1, namely _x_<sup>(</sup><sup>_i_)</sup> is sampled from a Gaussian with covariance Σ = _U_ Λ _U_<sup>_⊤_</sup> where Λ = diag( _λ_ 1 _, . . . , λd_ ) and _w⋆_ is sampled from _N_ (0 _, Id_ ). The proof works by reducing this case to the � � � previous case. For each _i_ , define _x_<sup>(</sup><sup>_i_)</sup> := _U_<sup>_T_</sup> _x_<sup>(</sup><sup>_i_)</sup> . Then E[ _x_<sup>(</sup><sup>_i_)</sup> ( _x_<sup>(</sup><sup>_i_)</sup> )<sup>_⊤_</sup> ] = E[ _U_<sup>_⊤_</sup> ( _U_ Λ _U_<sup>_⊤_</sup> ) _U_ ] = Λ. � Now let us write the loss function (15) with this new coordinate system: since _x_<sup>(</sup><sup>_i_)</sup> = _U x_<sup>(</sup><sup>_i_)</sup> , we have 



Hence, let us consider the vector ( _b_<sup>_⊤_</sup> G _A_ + _w⋆_<sup>_⊤_)</sup><sup>_U_.By definition of G, we have</sup> 



17 

_U_ 0 _U_<sup>_⊤_</sup> 0 � where we define<sup>�</sup> _b_<sup>_⊤_</sup> := _b_<sup>_⊤_</sup> 0 1 , _A_<sup>�</sup> := 0 1 _AU_ , and _w⋆_ := _U_<sup>_⊤_</sup> _w⋆_ . By the rotational symmetry, _w_ � _⋆_ is also distributed as� _N_ � (0 _, Id_ ).�Hence, this reduces to the previous case, and a global� minimum is given as 



From the definition of<sup>�</sup> _b, A_<sup>�</sup> , it thus follows that a global minimum is given by 



as desired. 

### **A.4 Proof for non-linear attentions (Theorem 5)** 

As mentioned in Theorem 5, we focus on the setting where the last row of _Q_ is zero, i.e., let 



We first rewrite the loss function and simplify it following Subsection A.1. Moreover, for simple notation we will often write _z, x_ instead of _z_<sup>(</sup><sup>_n_+1)</sup> _, x_<sup>(</sup><sup>_n_+1)</sup> . 

### **_1. Rewriting loss function._** 

Following Subsection A.1, let us write down the in-context loss (5). for the single-layer nonlinear attention denoted by _f_ ( _P, Q_ ): 



Recalling the definition of the ReLU attention Attn<sup>_σ_</sup> _P,Q_<sup>(</sup><sup>_Z_):=</sup><sup>_PZMσ_(</sup><sup>_Z⊤QZ_), the data matrix</sup> _Z_ := [ _z_<sup>(1)</sup> _· · · z_<sup>(</sup><sup>_n_+1)</sup> ], and the mask matrix _M_ , the term Attn<sup>_σ_</sup> _P,Q_<sup>(</sup><sup>_Z_0) can be written as:</sup> 



Hence, it follows that the ( _d_ + 1 _, n_ + 1)-th entry of Attn<sup>_σ_</sup> _P,Q_<sup>(</sup><sup>_Z_0)isequaltotheproductofthe</sup> ( _d_ + 1)-th row of _PZ_ 0, the mask matrix _M_ , and the ( _n_ + 1)-th column of _σ_ � _Z_ 0<sup>_⊤QZ_0</sup> �. Hence, let us write them down explicitly: 

- Letting _b_<sup>_⊤_</sup> be the last row of the matrix _P_ , it holds that the ( _d_ + 1)-th row of _PZ_ 0 is equal to [ _⟨b, z_<sup>(</sup><sup>_i_)</sup> _⟩_ ] _i_ =1 _,...,n_ +1. 

- The ( _n_ + 1)-th column of _σ_ � _Z_ 0<sup>_⊤QZ_0</sup> � is equal to � _σ_ �( _z_<sup>(</sup><sup>_i_)</sup> )<sup>_⊤_</sup> _Qz_<sup>(</sup><sup>_n_+1)��</sup> _i_ =1 _,...,n_ +1<sup>.Letting</sup><sup>_A_</sup> the first _d_ columns of _Q_ , this vector is equal to � _σ_ �( _x_<sup>(</sup><sup>_i_)</sup> )<sup>_⊤_</sup> _Ax_<sup>(</sup><sup>_n_+1)��</sup> _i_ =1 _,...,n_ +1<sup>because the last</sup> row of _Q_ is zero and the last row of _z_<sup>(</sup><sup>_n_+1)</sup> is zero (since ( _z_<sup>(</sup><sup>_n_+1)</sup> )<sup>_⊤_</sup> = [( _x_<sup>(</sup><sup>_n_+1)</sup> )<sup>_⊤_</sup> 0]). 

Thus, the product of [ _⟨b, z_<sup>(</sup><sup>_i_)</sup> _⟩_ ] _i_ =1 _,...,n_ +1, the mask matrix _M_ , and � _σ_ �( _x_<sup>(</sup><sup>_i_)</sup> )<sup>_⊤_</sup> _Ax_<sup>(</sup><sup>_n_+1)��</sup> _i_ =1 _,...,n_ +1 results in the following expression of the attention (writing _z, x_ instead of _z_<sup>(</sup><sup>_n_+1)</sup> _, x_<sup>(</sup><sup>_n_+1)</sup> ): 



18 

Since [ _Z_ 0] _d_ +1 _,n_ +1 = 0, we therefore have 



Therefore, it follows that the in-context loss _f_ ( _P, Q_ ) only depends on _b_ and _A_ . Henceforth, let us write _f_ ( _b, A_ ) instead of _f_ ( _P, Q_ ) following Subsection A.1. In particular, writing _b_<sup>_⊤_</sup> = [ _b_<sup>_⊤_</sup> 0<sup>_, b_1] for</sup> _b_ 0 _∈_ R<sup>_d_</sup> and _b_ 1 _∈_ R, the loss function can be expressed as 



### **_2. Simplifying the loss function with symmetry._** 

Now, we use the fact that both _x_<sup>(</sup><sup>_i_)</sup> ’s and _w⋆_ are sampled from the isotropic Gaussian, i.e., _N_ (0 _, Id_ ) in order to further simplify the loss function in (22). In particular, we use the following facts: 

- ( _a_ ) For orthonormal matrices _U, V ∈_ R<sup>_d×d_</sup> , it holds that _Ux_<sup>(</sup><sup>_i_)</sup> , _V x_ and _Uw⋆_ have the same distributions as _N_ (0 _, Id_ ). 

- ( _b_ ) Moreover, for a diagonal matrix Ξ = diag( _ξi_ ) _∈_ R<sup>_d×d_</sup> with the diagonal entries being random signs _ξi ∼{±_ 1 _}_ , it holds that Ξ _x_<sup>(</sup><sup>_i_)</sup> , Ξ _x_ and Ξ _w⋆_ have the same distributions as _N_ (0 _, Id_ ). 

Now let us fix a matrix _A ∈_ R<sup>_d×d_</sup> and _b_<sup>_⊤_</sup> = [ _b_<sup>_⊤_</sup> 0<sup>_, b_1] for</sup><sup>_b_0</sup><sup>_∈_R</sup><sup>_d_and</sup><sup>_b_1</sup><sup>_∈_R.Letting</sup><sup>_A_=</sup><sup>_U_Σ</sup><sup>_V⊤_</sup> be the SVD of the matrix _A_ , it follows that 





where in the third line we use the fact that Ξ<sup>_⊤_</sup> ΣΞ = ΞΣΞ = Σ; and the fourth line follows from the Jensen’s inequality. Hence for the remainder of the proof, we will characterize the global minimizer of the lower bound, i.e., _f_ lower and then we will connect it back to the original objective. 

### **_3. Computation of the lower bound_** _<u>f</u>_ lower **_._** 

Let us now explicitly compute _f_ lower. Let us rewrite the definition of _f_ lower. In fact since, _σ_ = ReLU is homogenous, one can further simplify the lower bound by pushing the constant _b_ 1 inside and write _b_ 1Σ as Σ. Hence, for two diagonal matrices Σ _, D ∈_ R<sup>_d×d_</sup> , _f_ lower is defined as: 



In particular, _D_ is constrained to be the diagonal part of an orthogonal matrix (since _D_ = diag( _U_<sup>_⊤_</sup> _V_ ) in the above derivation). Now we focus on characterizing the global minimizers of _f_ lower. 

19 

The main part of the argument is inspired by the elegant observation of Erdogdu et al. (2016), which says that the solution of least squares and generalized linear models are collinear for Gaussian inputs. We leverage the same proof technique ( _à la_ Stein’s Lemma) to prove that the presence of ReLU only changes the scaling of global optimum. 

First, since _w⋆_ is isotropic Gaussian, we can take the expectation over _w⋆_ to obtain 



which after a careful expansion becomes 

In order to compute (23), we will rely on the aforementioned argument of Erdogdu et al. (2016). In particular, from integration by parts, or Stein’s lemma (Erdogdu et al., 2016) (since _x ∼N_ (0 _, Id_ )), we have 

E _x_ [ _σ_ ( _x_<sup>_⊤_</sup> _v_ ) _x_ ] = E _x_ [ _σ_<sup>_′_</sup> ( _x_<sup>_⊤_</sup> _v_ )] _v_ for a fixed _v ∈_ R<sup>_d_</sup> . 

We use this to compute all the terms in (23) as follows: 

- We first apply Stein’s lemma to the first term of (23) for _i_ = _j_ . This results in 



Using the fact that _x_<sup>(</sup><sup>_i_)</sup> is a symmetric random variable, one can compute the expectation above as follows: one the one hand, we know E _x_ ( _i_ )[ _σ_<sup>_′_</sup> ( _x_<sup>_⊤_</sup> Σ _x_<sup>(</sup><sup>_i_)</sup> )] = E _x_ ( _i_ )[ _σ_<sup>_′_</sup> ( _−x_<sup>_⊤_</sup> Σ _x_<sup>(</sup><sup>_i_)</sup> )]. On the other hand, we also know that for any scalar _α_ , _σ_<sup>_′_</sup> ( _−α_ ) + _σ_<sup>_′_</sup> ( _α_ ) = 1. Therefore, we conclude that E _x_ ( _i_ )[ _σ_<sup>_′_</sup> ( _x_<sup>_⊤_</sup> Σ _x_<sup>(</sup><sup>_i_)</sup> )] = 1 _/_ 2. Thus, applying this technique twice, we obtain the following 



- Similarly, we can use Stein’s lemma to the second term of (23) to conclude 



- Lastly, the computation of the first term of (23) for _i_ = _j_ is straightforward. Using the fact that _∀α ∈_ R, _σ_<sup>2</sup> ( _α_ ) + _σ_<sup>2</sup> ( _−α_ ) = _a_<sup>2</sup> , we get 



Putting things all together (and ignoring the constant part in (23)), we have 



### **_4. Connecting back to the original loss function._** 

One can in fact write (24) solely in terms of _b_ 1 and _A_ as follows: 



Since the latter is a convex function in the matrix _b_ 1 _A_ , it follows that the minimizer corresponds to 



In fact the choice _b_ 1 = 1, _b_ 0 = 0, and _A_ = _−_ 2( _d_ +2)+(2 _<u>n</u> n−_ 1)<sup>_· Id_achieves this, and more crucially,</sup> satisfies the property that _f_ lower = _f_ for the corresponding parameters. Therefore, this shows that such choice is a global minimizer. 

20 

## **B Proofs for the multi-layer case** 

### **B.1 Proof of Theorem 2** 

The proof is based on probabilistic methods (Alon and Spencer, 2016). According to Lemma 5, the objective function can be written as (for more details check the derivations in (25)) 



where we use the isotropy of _w⋆_ and the linearity of trace to get the last equation. Suppose that _A_<sup>_∗_</sup> 0 and _A_<sup>_∗_</sup> 1<sup>denote the global minimizer of</sup><sup>_f_over symmetric matrices.Since</sup><sup>_A∗_</sup> 1<sup>is a symmetric matrix,</sup> it admits the spectral decomposition _A_ 1 = _UD_ 1 _U_<sup>_⊤_</sup> where _D_ 1 is a diagonal matrix and _U_ is an orthogonal matrix. Remarkably, the distribution of _X_ 0 is invariant to a linear transformation by an orthogonal matrix, i.e, _X_ 0 has the same distribution as _X_ 0 _U_<sup>_⊤_</sup> . This invariance yields 



Thus, we can assume _A_<sup>_∗_</sup> 1<sup>is diagonal without loss of generality.To prove</sup><sup>_A∗_</sup> 2<sup>is also diagonal, we</sup> leverage a probabilistic proof technique. Consider the random diagonal matrix _S_ whose diagonal elements are either 1 or _−_ 1 with probability<sup><u>1</u></sup> 2<sup>.Since the input distribution is invariant to orthogonal</sup> transformations, we have 



Note that we use _SD_ 1 _S_ = _D_ 1 in the last equation, which holds due to _D_ 1 and _S_ are diagonal matrices and _S_ has diagonal elements in _{_ +1 _, −_ 1 _}_ . Since _f_ is convex in _A_ 2, a straightforward application of Jensen’s inequality yields 

_f_ ( _D_ 1 _, A_<sup>_∗_</sup> 2<sup>) = E [</sup><sup>_f_(</sup><sup>_D_1</sup><sup>_, SA_</sup> 2<sup>_∗S_)]</sup><sup>_≥f_(</sup><sup>_D_1</sup><sup>_,_E [</sup><sup>_SA_</sup> 2<sup>_∗S_]) =</sup><sup>_f_(</sup><sup>_D_1</sup><sup>_,_diag(</sup><sup>_A_</sup> 2<sup>_∗_))</sup><sup>_._</sup> Thus, there are diagonal _D_ 1 and diag( _A_<sup>_∗_</sup> 2<sup>) for which</sup><sup>_f_(</sup><sup>_D_1</sup><sup>_,_diag(</sup><sup>_A∗_</sup> 2<sup>))</sup><sup>_≤f_(</sup><sup>_A∗_</sup> 1<sup>_, A∗_</sup> 2<sup>) holds for an</sup> optimal _A_<sup>_∗_</sup> 1<sup>and</sup><sup>_A∗_</sup> 2<sup>.This concludes the proof.</sup> 

### **B.2 Proof of Theorem 3** 

Let us drop the factor of<sup>1</sup> _/n_ which was present in the original update (56). This is because the constant 1 _/n_ can be absorbed into _Ai_ ’s. Doing so does not change the theorem statement, but reduces notational clutter. 

Let us consider the reformulation of the in-context loss _f_ presented in Lemma 5. Specifically, let _Z_ 0 be defined as 



where _y_<sup>(</sup><sup>_n_+1)</sup> = � _w⋆, x_<sup>(</sup><sup>_n_+1)�</sup> . Let _Z i_ denote the output of the ( _i−_ 1)<sup>_th_</sup> layer of the linear transformer (as defined in (56), initialized at _Z_ 0). For the rest of this proof, we will drop the bar, and simply denote _Z i_ by _Zi_ .<sup>5</sup> Let _Xi ∈_ R<sup>_d×_(</sup><sup>_n_+1)</sup> denote the first _d_ rows of _Zi_ and let _Yi ∈_ R<sup>1</sup><sup>_×_(</sup><sup>_n_+1)</sup> denote the ( _d_ +1)<sup>_th_</sup> row of _Zk_ . Under the sparsity pattern enforced in (8), we verify that, for any _i ∈{_ 0 _, . . . , k}_ , 





> 5This use of _Zi_ differs the original definition in (1). But we will not refer to the original definition anywhere in this proof. 

21 



We adopt the shorthand _A_ = _{Ai}_<sup>_k_</sup> _i_ =0<sup>.Let</sup><sup>_S⊂_R(</sup><sup>_k_+1)</sup><sup>_×d×d_,and</sup><sup>_A∈S_ifandonlyifforall</sup> _i ∈{_ 0 _, . . . , k}_ , there exists scalars _ai ∈_ R such that _Ai_ = _ai_ Σ<sup>_−_1</sup> and _Bi_ = _biI_ . We use _f_ ( _A_ ) to refer to the in-context loss of Theorem 3, that is, 



Throughout this proof, we will work with the following formulation of the _in-context loss_ from Lemma 5: 



The theorem statement is equivalent to the following: 



where _∇Aif_ denotes derivative wrt the Frobenius norm _∥Ai∥F_ . Towards this end, we establish the following intermediate result: if _A ∈S_ , then for any _R ∈_ R<sup>(</sup><sup>_k_+1)</sup><sup>_×d×d_</sup> , there exists _R_<sup>˜</sup> _∈S_ , such that, at _t_ = 0, 



In fact, we show that _R_<sup>˜</sup> _i_ := _riI_ , for _ri_ = _d_<sup><u>1</u>Tr</sup> �Σ<sup>1</sup><sup>_/_2</sup> _Ri_ Σ<sup>1</sup><sup>_/_2�</sup> . This implies (27) via the following simple argument: Consider the " _S_ -constrained gradient flow": let _A_ ( _t_ ) : R<sup>+</sup> _→_ R<sup>(</sup><sup>_k_+1)</sup><sup>_×d×d_</sup> be defined as 



for _i_ = 0 _, . . . , k_ . By (28), we verify that 



We verify from its definition that _f_ ( _A_ ) _≥_ 0; if the infimum in (27) fails to be zero, then inequality (29) will ensure unbounded descent as _t →∞_ , contradicting the fact that _f_ ( _A_ ) is lower-bounded. This concludes the proof. 

**_Proof outline._** The remainder of the proof will be devoted to showing (28), which we outline as follows: 

- In Step 1, we reduce the condition in (29) to a more easily verified _layer-wise_ condition. Specifically, we only need to verify (29) when _Ri_ are all zero except for _Rj_ for some fixed _j_ (see (30)) At the end of Step 1, we set up some additional notation, and introduce an important matrix _G_ , which is roughly "a product of attention layer matrices". In (31), we study the evolution of _f_ ( _A_ ( _t_ )) when _A_ ( _t_ ) moves in the direction of _R_ , as _X_ 0 is (roughly speaking) randomly transformed. 

- In Step 2, we use the results of Step 2 to to study _G_ (see (32)) and _dt_<sup>_<u>d</u>G_(</sup><sup>_A_(</sup><sup>_t_)) (see (33)) under</sup> random transformation of _X_ 0. The idea in (33) is that "randomly transforming _X_ 0" has the same effect as "randomly transforming _S_ " (recall _S_ is the perturbation to _B_ ). 

- In Step 3, we apply the result from Step 2 to the expression of _dt_<sup>_<u>d</u>f_(</sup><sup>_A_(</sup><sup>_t_)) in (31).We verify that</sup><sup>_R_˜</sup> in (28) is exactly the expected matrix after "randomly transforming _S_ ". This concludes our proof. 

**_1. Reduction to layer-wise condition._** To prove (28), it suffices to show the following simpler condition: Let _j ∈{_ 0 _, . . . , k}_ . Let _Rj ∈_ R<sup>_d×d_</sup> be arbitrary matrices. For _C ∈_ R<sup>_d×d_</sup> , let _A_ ( _tC, j_ ) denote 

22 

the collection of matrices, where [ _A_ ( _tC, j_ )] _j_ = _Aj_ + _tC_ , and for _i_ = _j_ , _A_ ( _tC, j_ ) _i_ = _Ai_ . We show that for all _j ∈{_ 0 _, . . . , k} , Rj ∈_ R<sup>_d×d_</sup> , there exists _R_<sup>˜</sup> _j_ = _rj_ Σ<sup>_−_1</sup> , such that, at _t_ = 0, 



We can verify that (28) is equivalent to (30) by noticing that for any _R_ , at _t_ = 0, _dt_<sup>_<u>d</u>f_(</sup><sup>_A_+</sup><sup>_tR_)=</sup> � _kj_ =0 _dtd_<sup>_f_(</sup><sup>_A_(</sup><sup>_tRj, j_)).We will now work towards proving (30) for some index</sup><sup>_j_that is arbitrarily</sup> chosen but fixed throughout. 





The second equality follows from plugging in (25). For the rest of this proof, let _U_ denote a uniformly _d_ randomly sampled orthogonal matrix. Let _U_ Σ := Σ<sup>1</sup><sup>_/_2</sup> _U_ Σ<sup>_−_1</sup><sup>_/_2</sup> . Using the fact that _X_ 0 = _U_ Σ _X_ 0, we can verify 



**_2._** _G_ **_and_** _<u>dt</u>_<sup>_<u>d</u>G_</sup><sup>**_under random transformation of_**</sup><sup>_X_0</sup><sup>**_._**</sup> We will now verify that _G_ ( _U_ Σ _X_ 0 _, Aj_ ) = _U_ Σ _G_ ( _X_ 0 _, Aj_ ): 



where we use the fact that _U_ Σ<sup>_⊤AiU_Σ=</sup><sup>_U_</sup> Σ<sup>_⊤_(</sup><sup>_ai_Σ</sup><sup>_−_1)</sup><sup>_U_Σ=</sup><sup>_Ai_.Next, we verify that</sup> 



where the first equality again uses the fact that _U_ Σ<sup>_⊤AiU_Σ=</sup><sup>_Ai_.</sup> 

**_3. Putting everything together._** Let us continue from (31). Plugging (32) and (33) into (31), 



23 



where _rj_ := _d_<sup><u>1</u>Tr</sup> �Σ<sup>1</sup><sup>_/_2</sup> _Rj_ Σ<sup>1</sup><sup>_/_2�</sup> . In the above, ( _i_ ) uses 1. (32) and (33), as well as the fact that _U_ Σ<sup>_⊤_Σ</sup><sup>_−_1</sup><sup>_U_Σ=Σ</sup><sup>_−_1.(</sup><sup>_ii_) uses the fact that</sup> _dtd_<sup>_G_(</sup><sup>_X_0</sup><sup>_, Aj_+</sup><sup>_tC_)</sup> �� _t_ =0<sup>is affine in</sup><sup>_C_.To see this, one</sup> can verify from the definition of _G_ , e.g. using similar algebra as (33), that _dt_<sup>_<u>d</u>G_(</sup><sup>_X_0</sup><sup>_, Aj_+</sup><sup>_C_) is affine</sup> in _C_ . Thus E _U_ � _G_ ( _X_ 0 _, Aj_ + _tU_ Σ<sup>_⊤RjU_Σ)</sup> � = _G_ ( _X_ 0 _, Aj_ + _t_ E _U_ � _U_ Σ<sup>_⊤RjU_Σ)</sup> �. 

### **B.3 Proof of Theorem 4** 

The proof of Theorem 4 is similar to that of Theorem 3, and with a similar setup. However to keep the proof self-contained, we will restate the setup. Once again, we drop the factor of _n_<sup><u>1</u>which was</sup> present in the original update (56). This is because the constant 1 _/n_ can be absorbed into _Ai_ ’s. Doing so does not change the theorem statement, but reduces notational clutter. 

Let us consider the reformulation of the in-context loss _f_ presented in Lemma 5. Specifically, let _Z_ 0 be defined as 



where _y_<sup>(</sup><sup>_n_+1)</sup> = � _w⋆, x_<sup>(</sup><sup>_n_+1)�</sup> . Let _Z i_ denote the output of the ( _i−_ 1)<sup>_th_</sup> layer of the linear transformer <u>(as defined in (56), initialized at</u> _Z_ 0). For the rest of this proof, we will drop the bar, and simply denote _Z i_ by _Zi_ .<sup>6</sup> Let _Xi ∈_ R<sup>_d×n_+1</sup> denote the first _d_ rows of _Zi_ and let _Yi ∈_ R<sup>1</sup><sup>_×n_+1</sup> denote the ( _d_ + 1)<sup>_th_</sup> row of _Zk_ . Under the sparsity pattern enforced in (11), we verify that, for any _i ∈{_ 0 _, . . . , k}_ , 



We adopt the shorthand _A_ = _{Ai}_<sup>_k_</sup> _i_ =0<sup>and</sup><sup>_B_=</sup><sup>_{Bi}k_</sup> _i_ =0<sup>.Let</sup><sup>_S⊂_R2</sup><sup>_×_(</sup><sup>_k_+1)</sup><sup>_×d×d_, and (</sup><sup>_A, B_)</sup><sup>_∈S_</sup> if and only if for all _i ∈{_ 0 _, . . . , k}_ , there exists scalars _ai, bi ∈_ R such that _Ai_ = _ai_ Σ<sup>_−_1</sup> and _Bi_ = _biI_ . Throughout this proof, we will work with the following formulation of the _in-context loss_ from Lemma 5: 



(note that the only randomness in _Z_ 0 comes from _X_ 0 as _Y_ 0 is a deterministic function of _X_ 0). The theorem statement is equivalent to the following: 



> 6This use of _Zi_ differs the original definition in (1). But we will not refer to the original definition anywhere in this proof. 

24 

where _∇Aif_ denotes derivative wrt the Frobenius norm _∥Ai∥F_ . 

Our goal is to show that, if ( _A, B_ ) _∈S_ , then for any ( _R, S_ ) _∈_ R<sup>2</sup><sup>_×_(</sup><sup>_k_+1)</sup><sup>_×d×d_</sup> , there exists ( _R,_<sup>˜</sup> _S_<sup>˜</sup> ) _∈ S_ , such that, at _t_ = 0, 



In fact, we show that _R_<sup>˜</sup> _i_ := _riI_ , for _ri_ = _d_ <u>1</u><sup>Tr</sup> �Σ<sup>1</sup><sup>_/_2</sup> _Ri_ Σ<sup>1</sup><sup>_/_2�</sup> and _S_<sup>˜</sup> _i_ = _siI_ , for _si_ = _d_ <u>1</u><sup>Tr</sup> �Σ<sup>_−_1</sup><sup>_/_2</sup> _Si_ Σ<sup>1</sup><sup>_/_2�</sup> . This implies (36) via the following simple argument: Consider the " _S_ - constrained gradient flow": let _A_ ( _t_ ) : R<sup>+</sup> _→_ R<sup>(</sup><sup>_k_+1)</sup><sup>_×d×d_</sup> and _B_ ( _t_ ) : R<sup>+</sup> _→_ R<sup>(</sup><sup>_k_+1)</sup><sup>_×d×d_</sup> be defined as 



for _i_ = 0 _, . . . , k_ . By (37), we verify that 

We verify from its definition that _f_ ( _A, B_ ) _≥_ 0; if (36) does not hold then (38) will ensure unbounded descent as _t →∞_ , contradicting the fact that _f_ ( _A, B_ ) is lower-bounded. This concludes the proof. 

**_Proof outline._** The remainder of the proof will be devoted to showing (37), which we outline as follows: 

- In Step 1, we reduce the condition in (37) to a more easily verified _layer-wise_ condition. Specifically, we only need to verify (37) in one of the two cases: (I) when _Ri, Si_ are all zero except for _Rj_ for some fixed _j_ (see (40)), or (II) when _Ri, Si_ are all zero except for _Sj_ for some fixed _j_ (see (39)). We focus on the proof of (II), as the proof of (I) is almost identical. At the end of Step 1, we set up some additional notation, and introduce an important matrix _G_ , which is roughly "a product of attention layer matrices". In (41), we study the evolution of _f_ ( _A, B_ ( _t_ )) when _B_ ( _t_ ) moves in the direction of _S_ , as _X_ 0 is (roughly speaking) randomly transformed. This motivates the subsequent analysis in Steps 2 and 3 below. 

- In Step 2, we study how outputs of each layer (34) changes when _X_ 0 is randomly transformed. There are two main results here: First we provide the expression for _Xi_ in (42). Second, we provide the expression for _dt_<sup>_<u>d</u>Xi_(</sup><sup>_B_(</sup><sup>_t_)) in (43).</sup> 

- In Step 3, we use the results of Step 2 to to study _G_ (see (47)) and _dt_<sup>_<u>d</u>G_(</sup><sup>_B_(</sup><sup>_t_)) (see (48)) under</sup> random transformation of _X_ 0. The idea in (48) is that "randomly transforming _X_ 0" has the same effect as "randomly transforming _S_ " (recall _S_ is the perturbation to _B_ ). 

- In Step 4, we use the results from Steps 2 and 3 to the expression of _dt_<sup>_<u>d</u>f_(</sup><sup>_A, B_(</sup><sup>_t_)) in (41).We</sup> verify that _S_<sup>˜</sup> in (37) is exactly the expected matrix after "randomly transforming _S_ ". This concludes our proof of (II). 

- In Step 5, we sketch the proof of (I), which is almost identical to Steps 2-4. 

**_1. Reduction to layer-wise condition._** To prove (37), it suffices to show the following simpler condition: Let _j ∈{_ 0 _, . . . , k}_ . Let _Rj, Sj ∈_ R<sup>_d×d_</sup> be arbitrary matrices. For _C ∈_ R<sup>_d×d_</sup> , let _A_ ( _tC, j_ ) denote the collection of matrices, where _A_ ( _tC, j_ ) _j_ = _Aj_ + _tC_ , and for _i_ = _j_ , _A_ ( _tC, j_ ) _i_ = _Ai_ . Define _B_ ( _tC, j_ ) analogously. We show that for all _j ∈{_ 0 _, . . . , k}_ and all _Rj, Sj ∈_ R<sup>_d×d_</sup> , there exists _R_<sup>˜</sup> _j_ = _rj_ Σ<sup>_−_1</sup> and _S_<sup>˜</sup> _j_ = _sj_ Σ<sup>_−_1</sup> , such that, at _t_ = 0, 



25 

We can verify that (37) is equivalent to (39)+(40) by noticing that for any ( _R, S_ ) _∈_ R<sup>2</sup><sup>_×_(</sup><sup>_k_+1)</sup><sup>_×d×d_</sup> , at _t_ = 0, _dt_<sup>_<u>d</u>f_(</sup><sup>_A_+</sup><sup>_tR, B_+</sup><sup>_tS_) = �</sup><sup>_k_</sup> _j_ =0 � _dtd_<sup>_f_(</sup><sup>_A_(</sup><sup>_tRj, j_)</sup><sup>_, B_) +</sup> _dt_<sup>_<u>d</u>f_(</sup><sup>_A, B_(</sup><sup>_tSj, j_))</sup> �. 

We will first focus on proving (40) (the proof of (39) is similar, and we present it in Step 5 at the end), for some index _j_ that is arbitrarily chosen but fixed throughout. Notice that _Xi_ and _Yi_ in (34) are in fact functions of _A, B_ and _X_ 0. For most of our subsequent discussion, _Ai_ (for all _i_ ) and _Bi_ (for all _i_ = _j_ ) can be treated as constant matrices. We will however make the dependence on _X_ 0 and _Bj_ explicit (as we consider the curve _Bj_ + _tS_ ), i.e. we use _Xi_ ( _X, C_ ) (resp _Yi_ ( _X, C_ )) to denote the value of _Xi_ (resp _Yi_ ) from (34), with _X_ 0 = _X_ , and _Bj_ = _C_ . 

By (35) and (34), 



where _G_ ( _X, C_ ) := _X_<sup>�</sup><sup>_k_</sup> _i_ =0 � _I − MXi_ ( _X, C_ )<sup>_T_</sup> _AiXi_ ( _X, C_ )�. The second equality follows from plugging in (34). 

For the rest of this proof, let _U_ denote a uniformly randomly sampled orthogonal matrix. Let _d U_ Σ := Σ<sup>1</sup><sup>_/_2</sup> _U_ Σ<sup>_−_1</sup><sup>_/_2</sup> . Using the fact that _X_ 0 = _U_ Σ _X_ 0, we can verify 



**_2._** _Xi_ **_and_** _<u>dt</u>_<sup>_<u>d</u>Xi_</sup><sup>**_under random transformation of_**</sup><sup>_X_0</sup><sup>**_._**</sup> In this step, we prove that when _X_ 0 is transformed by _U_ Σ, _Xi_ for _i ≥_ 1 are likewise transformed in a simple manner. The first goal of this step is to show 



We will prove this by induction. When _i_ = 0, this clearly holds by definition. Suppose that (42) holds for some _i_ . Then 







= _U_ Σ _Xi_ +1( _X_ 0 _, Bj_ ) 

where the second equality uses the inductive hypothesis, and the fact that _Ai_ = _ai_ Σ<sup>_−_1</sup> , so that _U_ Σ<sup>_TAiU_Σ=</sup><sup>_Ai_,andthefactthat</sup><sup>_Bi_=</sup><sup>_biI_,fromthedefinitionof</sup><sup>_S_andourassumptionthat</sup> ( _A, B_ ) _∈S_ . This concludes the proof of (42). 

We now present the second main result of this step. Let _U_ Σ<sup>_−_1</sup> := Σ<sup>1</sup><sup>_/_2</sup> _U_<sup>_T_</sup> Σ<sup>_−_1</sup><sup>_/_2</sup> , so that it satisfies _U_ Σ _U_ Σ<sup>_−_1</sup> = _U_ Σ<sup>_−_1</sup><sup>_U_Σ=</sup><sup>_I_.For all</sup><sup>_i_,</sup> 



To reduce notation, we will not write _·|t_ =0 explicitly in the subsequent proof. We first write down the dynamics for the right-hand-side term of (43): From (34), for any _ℓ ≤ j_ , and for any _i ≥ j_ + 1, and 

26 

for any _C ∈_ R<sup>_d×d_</sup> , 



We are now ready to prove (43) using induction. For the base case, we verify that for _ℓ ≤ j_ , _U_ Σ<sup>_−_1</sup> _dtd_<sup>_Xℓ_(</sup><sup>_U_Σ</sup><sup>_X_0</sup><sup>_, Bk_+</sup><sup>_tSj_) = 0 =</sup> _dt_<sup>_<u>d</u>Xℓ_</sup> � _X_ 0 _, Bj_ + _tU_ Σ<sup>_−_1</sup><sup>_SjU_Σ</sup> � (see first equation in (44)). For index _j_ + 1, we verify that 



where we use two facts: 1. _Xi_ ( _U_ Σ _X_ 0 _, Bj_ ) = _U_ Σ _Xi_ ( _X_ 0 _, Bj_ ) from (42), 2. _Ai_ = _ai_ Σ<sup>_−_1</sup> , so that _U_ Σ<sup>_⊤AiU_Σ</sup> = _Ai_ . We verify by comparison to the second equation in (44) that _U_ Σ<sup>_−_1</sup> _dtd_<sup>_Xj_(</sup><sup>_U_Σ</sup><sup>_X_0</sup><sup>_, Bj_+</sup><sup>_tSj_)=0=</sup> _dtd_<sup>_Xj_</sup> � _X_ 0 _, Bj_ + _tU_ Σ<sup>_−_1</sup><sup>_SjU_Σ</sup> �. These conclude the proof of the base case. 

Now suppose that (43) holds for some _i_ . We will now prove (43) holds for _i_ + 1. From (34), 



27 



In ( _i_ ) above, we crucially use the following facts: 1. _Bi_ = _biI_ so that _U_ Σ<sup>_−_1</sup><sup>_Bi_=</sup><sup>_BiU_</sup> Σ<sup>_−_1,2.</sup> _Xi_ ( _U_ Σ _X_ 0 _, Bj_ ) = _U_ Σ _Xi_ ( _X_ 0 _, Bj_ ) from (42), 3. _Ai_ = _ai_ Σ<sup>_−_1</sup> , so that _U_ Σ<sup>_⊤AiU_Σ=</sup><sup>_Ai_, 4.</sup><sup>_U_Σ</sup><sup>_U_</sup> Σ<sup>_−_1</sup> = _U_ Σ<sup>_−_1</sup><sup>_U_Σ=</sup><sup>_I_.(</sup><sup>_ii_)followsfromourinductivehypothesis.Theinductiveproofiscompleteby</sup> verifying that (46) exactly matches the third equation of (44) when _C_ = _U_ Σ<sup>_−_1</sup><sup>_SU_Σ.</sup> 

**_3._** _G_ **_and_** _<u>dt</u>_<sup>_<u>d</u>G_</sup><sup>**_under random transformation of_**</sup><sup>_X_0</sup><sup>**_._**</sup> We now verify that _G_ ( _U_ Σ _X_ 0 _, Bj_ ) = _U_ Σ _G_ ( _X_ 0 _, Bj_ ). This is a straightforward consequence of (42) as 

_G_ ( _U_ Σ _X_ 0 _, Bj_ ) 





To see this, we can expand 



28 



In ( _i_ ) above, we the following facts: 1. _Xi_ ( _U_ Σ _X_ 0 _, Bj_ ) = _U_ Σ _Xi_ ( _X_ 0 _, Bj_ ) from (42), 2. _Ai_ = _ai_ Σ<sup>_−_1</sup> , so that _U_ Σ<sup>_⊤AiU_Σ=</sup><sup>_Ai_, 3.</sup><sup>_U_Σ</sup><sup>_U_</sup> Σ<sup>_−_1</sup> = _U_ Σ<sup>_−_1</sup><sup>_U_Σ=</sup><sup>_I_.(</sup><sup>_ii_) follows from (43).(</sup><sup>_iii_) is by definition of</sup> _G_ . 

**_4. Putting everything together._** Let us now continue from (41). We can now plug (47) and (48) into (41): 



where _sj_ := _d_<sup><u>1</u>Tr</sup> �Σ<sup>_−_1</sup><sup>_/_2</sup> _Sj_ Σ<sup>1</sup><sup>_/_2�</sup> . In the above, ( _i_ ) uses 1. (47) and (48), as well as the fact that _U_ Σ<sup>_⊤_Σ</sup><sup>_−_1</sup><sup>_U_Σ=Σ</sup><sup>_−_1.(</sup><sup>_ii_) uses the fact that</sup> _dtd_<sup>_G_(</sup><sup>_X_0</sup><sup>_, Bj_+</sup><sup>_tC_)</sup> �� _t_ =0<sup>is affine in</sup><sup>_C_.To see this, one</sup> can verify from (44), using a simple induction argument, that _dt_<sup>_<u>d</u>Xi_(</sup><sup>_X_0</sup><sup>_, Bj_+</sup><sup>_tC_) is affine in</sup><sup>_C_for</sup> all _i_ . We can then verify from the definition of _G_ , e.g. using similar algebra as the proof of (48), that _dtd_<sup>_G_(</sup><sup>_X_0</sup><sup>_, Bj_+</sup><sup>_C_)isaffinein</sup> _dtd_<sup>_Xi_(</sup><sup>_X_0</sup><sup>_, Bj_+</sup><sup>_tC_).ThusE</sup><sup>_U_</sup> � _G_ ( _X_ 0 _, Bj_ + _tU_ Σ<sup>_−_1</sup><sup>_SjU_Σ)</sup> � = _G_ ( _X_ 0 _, Bj_ + _t_ E _U_ � _U_ Σ<sup>_−_1</sup><sup>_SjU_Σ)</sup> �. 

With this, we conclude our proof of (40). 

**_5. Proof of_** <u>(39)</u> **_._** We will now prove (39) for fixed but arbitrary _j_ , i.e. there is some _rj_ such that 



The proof is very similar to the proof of (40) that we just saw, and we will essentially repeat the same steps from Step 2-4 above. 

Since we now consider perturbations to _A_ instead of to _B_ , we will need to redefine some notation: let _Xi_ ( _X, C_ ) (resp _Yi_ ( _X, C_ )) to denote the value of _Xi_ (resp _Yi_ ) from (34), with _X_ 0 = _X_ , and _Aj_ = _C_ (previously it was with _Bj_ = _C_ ). Let _G_ ( _X, Aj_ + _C_ ) := _X_<sup>�</sup><sup>_i_</sup> _i_ =0 � _I_ + _M_ � _Xi_ ( _X, Aj_ + _C_ )<sup>_T_</sup> _A_ ( _C, j_ ) _iXi_ ( _X, Aj_ + _C_ )��, where recall that _A_ ( _C, j_ ) := _Aj_ + _C_ , and _A_ ( _C, j_ ) _ℓ_ := _Aℓ_ for all _ℓ ∈{_ 0 _...k} \ {j}_ . 

We first verify that 



29 

The proofs are identical to the proofs of (42) and (47) so we omit them. Next, we show that for all _i_ , 



We establish the dynamics for the right-hand-side of (50): 



Similar to (45), we show that for _i ≤ j_ , 



and 



Finally, for the inductive step, we follow identical steps leading up to (46) to show that 



The inductive proof is complete by verifying that (52) exactly matches the third equation of (51) when _C_ = _U_ Σ<sup>_−_1</sup><sup>_SU_Σ.This concludes the proof of (50).</sup> 

Next, we study the time derivative of _G_ ( _U_ Σ _X_ 0 _, Aj_ + _tRj_ ) and show that 



This proof differs significantly from that of (48) in a few places, so we provide the whole derivation below. By chain-rule, we can write 



30 

where 

and 



We will separately simplify _♠_ and _♡_ , and verify at the end that summing them recovers the righthand-side of (53). We begin with _♠_ , and the steps are almost identical to the proof of (48). 



31 

In ( _i_ ) above, we the following facts: 1. _Xi_ ( _U_ Σ _X_ 0 _, Bj_ ) = _U_ Σ _Xi_ ( _X_ 0 _, Bj_ ) from (49), 2. _Ai_ = _ai_ Σ<sup>_−_1</sup> , so that _U_ Σ<sup>_⊤AiU_Σ=</sup><sup>_Ai_, 3.</sup><sup>_U_Σ</sup><sup>_U_</sup> Σ<sup>_−_1</sup> = _U_ Σ<sup>_−_1</sup><sup>_U_Σ=</sup><sup>_I_.(</sup><sup>_ii_) follows from (50).</sup> We will now simplify _♡_ . 



where ( _i_ ) uses the fact that _Xi_ ( _U_ Σ _X_ 0 _, Bj_ ) = _U_ Σ _Xi_ ( _X_ 0 _, Bj_ ) from (49) and the fact that _Ai_ = _ai_ Σ<sup>_−_1</sup> . 

By expanding _dt_<sup>_<u>d</u>G_(</sup><sup>_X_0</sup><sup>_, Aj_+</sup><sup>_tU_</sup> Σ<sup>_⊤RjU_Σ), we verify that</sup> 



this concludes the proof of (53). 

The remainder of the proof is similar to what was done in (41) in Step 4: 



where _rj_ := _d_<sup><u>1</u>Tr</sup> �Σ<sup>1</sup><sup>_/_2</sup> _Rj_ Σ<sup>1</sup><sup>_/_2�</sup> . In the above, ( _i_ ) uses 1. (49) and (53), as well as the fact that _U_ Σ<sup>_⊤_Σ</sup><sup>_−_1</sup><sup>_U_Σ=Σ</sup><sup>_−_1.(</sup><sup>_ii_) uses the fact that</sup> _dtd_<sup>_G_(</sup><sup>_X_0</sup><sup>_, Aj_+</sup><sup>_tC_)</sup> �� _t_ =0<sup>is affine in</sup><sup>_C_.To see this, one</sup> can verify using a simple induction argument, that _dtd_<sup>_Xi_(</sup><sup>_X_0</sup><sup>_, Aj_+</sup><sup>_tC_)isaffinein</sup><sup>_C_forall</sup><sup>_i_.</sup> We can then verify from the definition of _G_ , e.g. using similar algebra as the proof of (53), that _dtd_<sup>_G_(</sup><sup>_X_0</sup><sup>_, Aj_+</sup><sup>_C_) is affine in</sup> _dt_<sup>_<u>d</u>Xi_(</sup><sup>_X_0</sup><sup>_, Aj_+</sup><sup>_tC_) and</sup><sup>_C_.Thus E</sup><sup>_U_</sup> � _G_ ( _X_ 0 _, Aj_ + _tU_ Σ<sup>_⊤RjU_Σ)</sup> � = _G_ ( _X_ 0 _, Aj_ + _t_ E _U_ � _U_ Σ<sup>_⊤RjU_Σ</sup> �). 

This concludes the proof of (39), and hence of the whole theorem. 

32 

### **B.4 Equivalence under permutation** 

**Lemma 4.** _Consider the same setup as Theorem 3. Let A_ = _{Ai}_<sup>_k_</sup> _i_ =0<sup>_, with Ai_=</sup><sup>_ai_Σ</sup><sup>_−_1</sup><sup>_.Let_</sup> 



_Let i, j ∈{_ 0 _, . . . , k} be any two arbitrary indices, and let A_<sup>˜</sup> _i_ = _Aj, A_<sup>˜</sup> _j_ = _Ai, and let A_<sup>˜</sup> _ℓ_ = _Aℓ for all ℓ ∈{_ 0 _, . . . , k} \ {i, j}. Then f_ ( _A_ ) = _f_ ( _A_<sup>˜</sup> ) 

_Proof._ Following the same setup leading up to (26) in the proof of Theorem 3, we verify that the in-context loss is 



Consider any fixed index _ℓ_ . We will show that 



The lemma can then be proven by repeatedly applying the above, so that indices of _Ai_ and _Aj_ are swapped. 

To prove the above equality, 



This concludes the proof. Notice that we crucially used the fact that _Aℓ_ and _Aℓ_ +1 are the same matrix up to scaling. 

## **C Auxiliary Lemmas** 

### **C.1 Proof of Lemma 1 (Equivalence to Preconditioned Gradient Descent)** 

Consider fixed samples _x_<sup>(1)</sup> _, . . . , x_<sup>(</sup><sup>_n_)</sup> , and fixed _w⋆_ . Let _P_ = _{Pi}_<sup>_k_</sup> _i_ =0<sup>_, Q_=</sup><sup>_{Qi}k_</sup> _i_ =0<sup>denote</sup> fixed weights. Let _Zi_ evolve as described in (4). Let _Xi_ denote the first _d_ rows of _Zk_ (under (8), _Xi_ = _X_ 0 for all _I_ ) and let _Yi_ denote the ( _d_ + 1)<sup>_th_</sup> row of _Zi_ . Let _g_ ( _x, y, k_ ) : R<sup>_d_</sup> _×_ R _×_ Z _→_ R be a function defined as follows: let _x_<sup>_n_+1</sup> = _x_ and let _y_ 0<sup>_n_+1</sup> = _y_ , then _g_ ( _x, y, k_ ) := _yk_<sup>_n_+1</sup> . Note that _yk_<sup>_n_+1</sup> = [ _Yk_ ] _n_ +1. 

We verify that, under (8), the formula for updating _yk_<sup>(</sup><sup>_n_+1)</sup> is given by 



_I_ 0 where _M_ is a mask given by . We can verify the following facts 0 0 � � 

1. _g_ ( _x, y, k_ ) = _g_ ( _x,_ 0 _, k_ ) + _y_ . To see this, notice first that for all _i ∈{_ 1 _, . . . , n}_ , 



33 

In other words, _yk_<sup>(</sup><sup>_i_)</sup> does not depend on _yt_<sup>(</sup><sup>_n_+1)</sup> for any _t_ . Next, for _yk_<sup>(</sup><sup>_n_+1)</sup> itself, 



which depends on _yk_<sup>_n_+1</sup> only additively. We can verify under a simple induction that _g_ ( _x, y, k_ + 1) _− y_ = _g_ ( _x, y, k_ ) _− y_ . 

2. _g_ ( _x,_ 0 _, k_ ) is linear in _x_ . To see this, notice first that for _j_ = _n_ + 1, _yk_<sup>(</sup><sup>_j_)</sup> is does not depend on _x_<sup>(</sup> _t_<sup>_n_+1)</sup> for all _t, j, k_ . Consequently, the update formula for _yk_<sup>(</sup><sup>_n_</sup> +1<sup>+1)</sup> depends only linearly on _x_<sup>(</sup><sup>_n_+1)</sup> and _yk_<sup>(</sup><sup>_n_+1)</sup> . Finally, _y_ 0<sup>(</sup><sup>_n_+1)</sup> = 0 is linear in _x_ , so the conclusion follows by induction. 

With these two facts in mind, we verify that for each _k_ , there exists a _θk ∈_ R<sup>_d_</sup> , such that 

_g_ ( _x, y, k_ ) = _g_ ( _x,_ 0 _, k_ ) + _y_ = _⟨θk, x⟩_ + _y_ 

for all _x, y_ . It follows from definition that _g_ ( _x, y,_ 0) = _y_ , so that _⟨θ_ 0 _, x⟩_ = _g_ ( _x, y,_ 0) _− y_ = 0, so that _θ_ 0 = 0. 

We now turn our attention to the third crucial fact: for all _i_ , 



To see this, suppose that we let _x_<sup>(</sup><sup>_n_+1)</sup> := _x_<sup>(</sup><sup>_i_)</sup> for some _i ∈_ 1 _, . . . , n_ . Then 

thus _yk_<sup>(</sup><sup>_i_</sup> +1<sup>)=</sup><sup>_y_</sup> _k_<sup>(</sup><sup>_n_</sup> +1<sup>+1)</sup> if _yk_<sup>(</sup><sup>_i_)</sup> = _yk_<sup>(</sup><sup>_n_+1)</sup> , and the induction proof is completed by noting that _y_ 0<sup>(</sup><sup>_i_)</sup> = _y_ 0<sup>(</sup><sup>_n_+1)</sup> by definition. Let _X_<sup>¯</sup> _∈ R_<sup>_d×n_</sup> be the matrix whose columns are _x_<sup>(1)</sup> _, . . . , x_<sup>(</sup><sup>_n_)</sup> , leaving out _x_<sup>(</sup><sup>_n_+1)</sup> . Let _Y_<sup>¯</sup> _k ∈_ R<sup>1</sup><sup>_×n_</sup> denote the vector of _yk_<sup>(1)</sup><sup>_, . . . , y_</sup> _k_<sup>(</sup><sup>_n_).Then it follows that</sup> ¯ ¯ _Yk_ = _Y_ 0 + _θk_<sup>_TX._¯</sup> 



Since the choice of _x_<sup>(</sup><sup>_n_+1)</sup> is arbitrary, we get the more general update formula 







This concludes the proof. 

34 

### **C.2 Reformulating the in-context loss** 

In this section, we will develop a re-formulation in-context loss, defined in (5), in a more convenient form (see Lemma 5). 

For the entirety of this section, we assume that the transformer parameters _{Pi, Qi}_<sup>_k_</sup> _i_ =0<sup>are of the</sup> form defined in (11), which we reproduce below for ease of reference: 



Recall the update dynamics in (4), which we reproduce below: 



where _M_ is a mask matrix given by _M_ := _In×n_ 0 . Let _Xk ∈_ R<sup>_d×n_+1</sup> denote the first _d_ rows 0 0 � � of _Zk_ and let _Yk ∈_ R<sup>1</sup><sup>_×n_+1</sup> denote the ( _d_ + 1)<sup>_th_</sup> (last) row of _Zk_ . Then the dynamics in (56) is equivalent to 



We present below an equivalent form for the in-context loss from (5): 

**Lemma 5.** _Let px and pw denote distributions over_ R<sup>_d_</sup> _. Let x_<sup>(1)</sup> _, . . . , x_<sup>(</sup><sup>_n_+1)</sup><sup>_iid_</sup> _∼ px and w⋆ ∼ pw. Let Z_ 0 _∈_ R<sup>_d_+1</sup><sup>_×n_+1</sup> _be as defined in_ (1) _:_ 



_Let Zk denote the output of the_ ( _k −_ 1)<sup>_th_</sup> _layer of the linear transformer (as defined in_ (56) _, initialized at Z_ 0 _). Let f_ � _{Pi, Qi}_<sup>_k_</sup> _i_ =0� _denote the in-context loss defined in_ (5) _, i.e._ 



_Let Z_ 0 _be defined as_ 



_where y_<sup>(</sup><sup>_n_+1)</sup> = � _w⋆, x_<sup>(</sup><sup>_n_+1)�</sup> _. Let Z k denote the output of the_ ( _k −_ 1)<sup>_th_</sup> _layer of the linear transformer (as defined in_ (56) _, initialized at Z_ 0 _). Assume {Pi, Qi}_<sup>_k_</sup> _i_ =0<sup>_be of the form in_(11)</sup><sup>_.Then_</sup> _the loss in_ (5) _has the equivalent form_ 



_where Y k_ +1 _∈_ R<sup>1</sup><sup>_×n_+1</sup> _is the_ ( _d_ + 1)<sup>_th_</sup> _row of Z k._ 

Before proving Lemma 5, we first establish an intermediate result (Lemma 6 below). To facilitate discussion, let us define a function _FX {Ai, Bi}_<sup>_k_</sup> _i_ =0<sup>_, X_0</sup><sup>_, Y_0</sup> and _FY {Ai, Bi}_<sup>_k_</sup> _i_ =0<sup>_, X_0</sup><sup>_, Y_0</sup> to � � � � be the outputs, after _k_ layers of linear transformers respectively. I.e. 



as defined in (57), given initialization _X_ 0 _, Y_ 0. 

We now prove a useful lemma showing that [ _Y_ 0] _n_ +1 = _y_<sup>(</sup><sup>_n_+1)</sup> influences _Xi, Yi_ in a very simple manner: 

35 

**Lemma 6.** _Let Xi, Yi follow the dynamics in_ (57) _. Then_ 

_1._ [ _Xi_ ] _is are independent of_ [ _Y_ 0] _n_ +1 _._ 





_In other words, for C_ := [0 _,_ 0 _,_ 0 _, . . . , ,_ 0 _, c_ ] _∈_ R<sup>1</sup><sup>_×_(</sup><sup>_n_+1)</sup> _,_ 



_Proof of Lemma 6._ The first and second items follows directly from observing that the dynamics for _Xi_ and _Yi_ in (57) do not involve [ _Yi_ ] _n_ +1, due to the effect of _M_ . 

The third item again uses the fact that [ _Yi_ +1 _− Yi_ ] _n_ +1 does not depend on [ _Yi_ ] _n_ +1. 

We are now ready to prove Lemma 5 

_Proof of Lemma 5._ Let _Z_ 0, _Zk_ , _Z_ 0, _Z k_ be as defined in the lemma statement. Let _X k_ and _Y k_ denote first _d_ rows and last row of _Z k_ . Then by Lemma 6, _X k_ +1 = _Xk_ +1 and _Y k_ +1 = _Yk_ +1 + �0 0 _· · ·_ 0 � _w⋆, x_<sup>(</sup><sup>_n_+1)��</sup> . Therefore, (58) is equivalent to 



This concludes the proof. 

## **D Additional experimental results** 

In this section, we present a few addition experimental results. We first present in Figure 5 a visualization of learned weights _A_ 0 _, A_ 1 _, A_ 2 for the setting of Theorem 4. One can see that the weight pattern matches the stationary point analyzed in Theorem 4; hence, combining Figure 4 and Figure 5, we corroborate our results from Theorem 4. Interestingly, it appears that the transformer implements a tiny gradient step using _X_ 0 (as _A_ 0 is small), and a large gradient step using _X_ 2 (as _A_ 2 is large). We believe that this is due to _X_ 2 being better-conditioned than _X_ 1, due to the effects of _B_ 0 _, B_ 1. 



<!-- Start of picture text -->
0 -0.38 0.00 0.01 -0.01 -0.00<br>1 -0.00 -0.38 0.01 -0.00 0.00<br>2 -0.01 -0.01 -0.38 -0.01 -0.00<br>3 0.01 0.00 0.01 -0.38 0.01<br>4 0.00 -0.00 0.00 -0.01 -0.38<br><!-- End of picture text -->



<!-- Start of picture text -->
0.00<br><!-- End of picture text -->



<!-- Start of picture text -->
-2.40 -0.00 0.00 -0.00 -0.00<br>0.00 -2.40 0.01 0.01 0.00<br>0.00 0.00 -2.39 0.01 0.00<br>-0.00 0.01 0.01 -2.38 0.01<br>0.00 0.00 0.00 0.01 -2.42<br><!-- End of picture text -->





<!-- Start of picture text -->
-9.59 -0.01 -0.01 -0.03 0.02<br>-0.01 -9.54 0.01 0.05 0.03<br>-0.00 0.01 -9.55 0.05 0.01<br>-0.03 0.06 0.04 -9.46 0.02<br>0.01 0.03 0.02 0.02 -9.65<br><!-- End of picture text -->





<!-- Start of picture text -->
(a) Visualization of Σ 1 / 2 A 0Σ 1 / 2 (b) Visualization of Σ 1 / 2 A 1Σ 1 / 2 (c) Visualization of Σ 1 / 2 A 2Σ 1 / 2<br><!-- End of picture text -->

Figure 5: Visualization of learned weights for the setting of Theorem 4. One can see that the weight pattern matches the stationary point analyzed in Theorem 4. 

36 

We next present some additional experiments that investigates the properties of the learned predictors of various algorithms. First, we plot the **test losses against the number of examples provided in the prompt** (“the number of ICL examples”). We compare four different algorithms: (i) the predictor learned by a three-layered of linear transformer, (ii) three steps of GD, (iii) three steps of preconditioned GD, and (iv) the ordinary least-squared solution (OLS). For GD and preconditioned GD, the optimal stepsizes are found by gridsearch. For preconditioned GD, preconditioner is fixed to be Σ<sup>_−_1</sup> for comparison. In all cases, the dimension _d_ = 5, and for each _N_ , the linear Transformer is trained using Adam. The result is presented in Figure 6. 



<!-- Start of picture text -->
3-Step GD<br>3-Step Preconditioned GD<br>4<br>3-Layer Linear Transformer<br>OLS<br>3<br>2<br>1<br>0<br>2 6 10 14 18 22<br>Number of ICL Examples<br>Loss<br><!-- End of picture text -->

Figure 6: Test loss comparison between (i) the predictor learned by a three-layered of linear transformer, (ii) three steps of GD, (iii) three steps of preconditioned GD, and (iv) the ordinary leastsquared solution (OLS). 

Lastly, in Figure 7, we plot the **test losses against the number of layer** _L_ (or the number of steps in the case of gradient-based algorithms). For _L_ = 1 _,_ 2 _,_ 3 _,_ 4, we compare between (i) the predictor learned by _L_ -linear transformer and (i) _L_ -steps of GD, (ii) _L_ -steps of preconditioned GD. Again, the optimal stepsize is found by gridsearch, and for preconditioned GD, the preconditioner is fixed to be Σ<sup>_−_1</sup> . In all cases, the dimension _d_ = 5, and context length _N_ = 20. The linear transformer is trained with Adam. 



<!-- Start of picture text -->
0<br>2<br>4<br>6<br>GD<br>8 Preconditioned GD<br>Linear Transformer<br>1 2 3 4<br>Number of Layers/Steps<br>log(Loss)<br><!-- End of picture text -->

Figure 7: Test loss comparison between (i) the predictor learned by a _L_ -layered linear transformer and (i) _L_ -steps of GD, (ii) _L_ -steps of preconditioned GD, for _L_ = 1 _,_ 2 _,_ 3 _,_ 4. 

37 

