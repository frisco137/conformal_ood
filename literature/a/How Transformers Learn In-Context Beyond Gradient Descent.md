# **How Well Can Transformers Emulate In-context Newton’s Method?** 

Angeliki Giannou<sup>∗</sup> 

Liu Yang<sup>†</sup> Tianhao Wang<sup>‡</sup> Dimitris Papailiopoulos<sup>§</sup> 

Jason D. Lee<sup>¶</sup> 

March 6, 2024 

##### **Abstract** 

Transformer-based models have demonstrated remarkable in-context learning capabilities, prompting extensive research into its underlying mechanisms. Recent studies have suggested that Transformers can implement first-order optimization algorithms for in-context learning and even second order ones for the case of linear regression. In this work, we study whether Transformers can perform higher order optimization methods, beyond the case of linear regression. We establish that linear attention Transformers with ReLU layers can approximate second order optimization algorithms for the task of logistic regression and achieve _ϵ_ error with only a logarithmic to the error more layers. As a by-product we demonstrate the ability of even linear attention-only Transformers in implementing a single step of Newton’s iteration for matrix inversion with merely two layers. These results suggest the ability of the Transformer architecture to implement complex algorithms, beyond gradient descent. 

## **1 Introduction** 

Transformer networks have had a significant impact in machine learning, particularly in tasks related to natural language processing and computer vision (Vaswani et al., 2017; Khan et al., 2022; Yuan et al., 2021; Dosovitskiy et al., 2020). A key building block of Transformers is the self-attention mechanism, which enables the model to weigh the significance of different parts of the input data with respect to each other. This allows the model to capture long-range dependencies and learn complex patterns of the data, yielding state-of-the-art performance across a several tasks, including but not limited to language translation, text summarization, and conversational agents (Vaswani et al., 2017; Kenton and Toutanova, 2019). 

It has been long observed that Transformers are able to perform various downstream tasks at inference without any parameter updates Brown et al. (2020); Lieber et al. (2021); Black et al. (2022). This ability, known as _in-context learning_ , has attracted the interest of the community, resulting in a line of works aiming to interpret and understand it. Towards this direction, Garg et al. (2022) were the first to consider a setting, in which the “language” component of the problem is removed from the picture, allowing the authors to study the ability of Transformers to learn how to learn 

> ∗ University of Wisconsin-Madison. Email: `giannou@wisc.edu` . 

> †University of Wisconsin-Madison. Email: `liu.yang@wisc.edu` . 

> ‡Yale University. Email: `tianhao.wang@yale.edu` . 

> §University of Wisconsin-Madison. Email: `dimitris@ece.wisc.edu` . 

> ¶Princeton University. Email: `jasonlee@princeton.edu` . 

1 

in regression settings. However, even for this simple setting, understanding the mechanics of the architecture that would allow such capability, is not yet very well understood. 

Following research has presented constructive methods to explain what type of algorithms these models can implement internally, by designing model weights that lead to a model that implements specific meta-algorithms. Akyürek et al. (2022) constructed Transformers that implement one step of gradient descent with _O_ (1) layers. Other works have focused on the linear attention (removing the softmax) and have shown empirically (von Oswald et al., 2022) and theoretically (Ahn et al., 2023; Mahankali et al., 2023) that the optimum for one layer, is in essence one step of preconditioned gradient descent. Ahn et al. (2023) also showed that the global minimizer in the two-layer case corresponds to gradient descent with adaptive step size, but the optimality is restricted to only a class of sparse weights configurations. Beyond these, it still remains open how to characterize the global minimizer of the in-context loss for Transformers with multiple layers. 

Another approach is to approximate the closed-form solution of linear regression, instead of minimizing the loss. For that purpose, one needs to be able to perform matrix inversion. There are multiple approaches to matrix inversion and in terms of iterative algorithms, one of the most popular ones is Newton’s iteration (Schulz, 1933), which is a second-order method. Specifically, the method has a warm-up state with logarithmic, to the condition number of the matrix, steps and afterwards quadratic rate of convergence to arbitrary accuracy. The work of Giannou et al. (2023) showed that Transformers can implement second-order methods, like Newton’s iteration<sup>1</sup> , for matrix inversion. Fu et al. (2023) implemented Newton’s iteration with Transformers for in-context linear regression; the authors also performed an empirical study to argue that Newton’s iteration is closer to the trained models output rather than gradient descent. 

Newton’s iteration for matrix inversion is of the form **X** _t_ +1 = **X** _t_ (2I _−_ **AX** _t_ ), where **A** is the matrix we want to invert, and **X** _t_ is the approximation of the inverse. The implementation of matrix inversion, serves as a stepping stone towards answering the following question: 

#### “ _How well can Transformers implement higher-order optimization methods?_ ” 

In pursuit of a concrete answer, we focus on the well-known Newton’s method, a second-order optimization algorithm. For minimizing a function _f_ , the method can be described as **x** _t_ +1 = **x** _t − η_ ( **x** _t_ )( _∇_<sup>2</sup> _f_ ( **x** _t_ ))<sup>_−_1</sup> _∇f_ ( **x** _t_ ) for some choice of step-size _η_ ( _·_ ). To implement such updates, one essentially needs to first compute the step size _η_ ( **x** _t_ ), then invert the Hessian _∇_<sup>2</sup> _f_ ( **x** _t_ ), and finally multiply them together with the gradient _∇f_ ( **x** _t_ ). In general, the step size _η_ ( **x** _t_ ) may also depend on quantities computed from _∇f_ ( **x** _t_ ) and _∇_<sup>2</sup> _f_ ( **x** _t_ ). It is relatively straightforward for Transformers to perform operations like matrix transposition and multiplication, while the challenge is to devise an organic combination of all the above components to deliver an efficient implementation of Newton’s method with convergence guarantees. In particular, it further requires rigorous convergence analysis to verify the effectiveness of the construction in concrete examples. 

**Main contributions.** In this work we tackle the challenge from the perspective of in-context learning for the tasks of linear and logistic regression. We consider Transformers with linear attention layers and position-wise feed-forward layers with the ReLU activation. We provide concrete constructions of Transformers to solve the two tasks, and derive explicit upper bounds on the depth and width of the model with respect to the targeted error threshold. At a high level, our main findings are summarized in the following informal theorem. 

**Theorem 1.1** (Informal) **.** _Transformers can efficiently perform matrix inversion via Newton’s iteration, based on which they can further 1) compute the least-square solution for linear regression,_ 

> 1We use Newton’s iteration for the matrix inversion algorithm and Newton’s method for the optimization algorithm. 

2 

_and 2) perform Newton’s method to efficiently optimize the regularized logistic loss for logistic regression. In particular, in the latter case, only_ log log(1 _/ϵ_ ) _many layers and_ 1 _/ϵ_<sup>8</sup> _width are required for the Transformer to implement Newton’s method on the regularized logistic loss to achieve ϵ error._ 

We also corroborate our results with experimental evidence. Interestingly, trained Transformers seem to outperform Newton’s method for the initial layers/steps. To understand what the models are actually learning we also train models with different number of layers for the simpler task of linear regression. We observe a significant difference in performance for models trained with or without layer norm. We compare the Transformers with variations of Newton’s iteration with different order of convergence. 



<!-- Start of picture text -->
Logistic Regression Loss<br>TF<br>Newton Method<br>1.0<br>0.6<br>0 10 20 30 40<br>layers / steps<br><!-- End of picture text -->

## **2 Related work** 

It has been observed that Transformer-based models have the ability of performing in-context learning, as well as the ability of algorithmic reasoning (Brown et al., 2020; Nye et al., 2021; Wei et al., 2022a,b; Dasgupta et al., 2022; Zhou et al., 2022). 

Figure 1: The logistic regression loss of a trained Transformer with 1-40 layers and corresponding steps of Newton’s method. 

Recently, Garg et al. (2022) initiated the mathematical formulation of the in-context learning problem, studying empirically the controlled setting of linear regression. Transformers were able to perform in-context linear regression, given only ( **x** _i, yi_ ) pairs, generated as _yi_ = **w** _∗_<sup>_⊤_</sup><sup>**x**</sup><sup>_i_,which</sup> were not seen during training. Later on, other works studied this setting both empirically and theoretically (Akyürek et al., 2022; von Oswald et al., 2022; Bai et al., 2023; Li et al., 2023; Guo et al., 2023; Chen et al., 2024). 

Towards explaining this capability, Akyürek et al. (2022); von Oswald et al. (2023a) showed by construction that Transformers can emulate gradient descent for the task of linear regression. von Oswald et al. (2023a), also observed that empirically, one layer of linear attention Transformer had very similar performance with one step of gradient descent. Indeed, Ahn et al. (2023); Mahankali et al. (2023) proved that the global minimizer of the in-context learning loss for linear regression corresponds to one step of (preconditioned) gradient descent. 

Related to our work is also the work of Bai et al. (2023), which is the only work - to the best of our knowledge - that provides a construction of gradient based algorithms for various in-context learning tasks, including logistic regression; they also demonstrate the ability of Transformer based models to perform algorithm selection. In the case of learning non-linear functions, Cheng et al. (2023) showed that Transformers can perform functional gradient descent. 

Focusing on performing linear regression through matrix inversion, the recent work of von Oswald et al. (2023b) is of interest. The authors approximate the inverse of a matrix using Neumman series. For the task of linear regression this approach requires less memory compared to Newton’s method, but it has a linear rate of convergence (Wu et al., 2014). 

Considering higher order optimization methods, Giannou et al. (2023) first implemented matrix inversion using Newton iteration, their construction though is sub-optimal, since it uses thirteen layers to perform just one step of the method and it is given in a general template. In a recent work by Fu et al. (2023), the authors use Newton’s iteration for matrix inversion to approximate the closed form solution of linear regression; they compare it with a 12-layer trained transformer, by linear-probing each layer’s output and comparing it with steps of the iterative algorithm. They furthermore conclude that Transformers are closer to Newton’s iteration rather than gradient descent. 

3 

In terms of the optimization dynamics, Zhang et al. (2023) proved for one-layer linear attention the convergence of gradient flow to the global minimizer of the population loss given suitable initialization. Huang et al. (2023) showed the convergence of gradient descent for training softmaxattention Transformer under certain orthogonality assumption on the data features. 

## **3 Preliminaries** 

**Notation.** We use lowercase bold letters for vectors e.g., **x** _,_ **y** _,_ and by convention we consider them to be column vectors; for matrices we use uppercase bold letters e.g., **A** _,_ **B** . We use _λ_ ( **A** ) _, σ_ ( **A** ) to denote the eigenvalues and singular values of a matrix **A** respectively; we use _κ_ ( **A** ) = _σ_ max( **A** ) _/σ_ min( **A** ) to denote the condition number of **A** . For a Positive Symmetric Definite (PSD) matrix **A** , we denote _∥_ **x** _∥_ **A** = _√_ **x**<sup>_⊤_</sup> **Ax** . We use **0** _d_ and I _d_ to denote a _d × d_ matrix of zeros and the _d × d_ identity matrix, respectively. If not specified otherwise, we use _∗_ to denote inconsequential values. 

### **3.1 The Transformer architecture** 

There are multiple variations of the Transformer architecture, depending on which type of attention (e.g., softmax, ReLU, and linear) is used. In this work, we focus on Transformers with linear self-attention described as follows: For each layer, let **H** _∈_ R<sup>_d×n_</sup> be the input, where each column is a _d_ -dimensional embedding vector for each token. Let _H_ be the number of attention heads, and for each head _i ∈_ [ _H_ ], we denote by **W** _K_<sup>(</sup><sup>_i_)</sup><sup>_,_</sup><sup>**W**</sup> _Q_<sup>(</sup><sup>_i_)</sup><sup>_,_</sup><sup>**W**</sup> _V_<sup>(</sup><sup>_i_)</sup><sup>_∈_R</sup><sup>_d×d_thekey,query,andvalueweightmatrices,</sup> respectively. Further let **W** 1 _∈_ R<sup>_D×d_</sup> and **W** 2 _∈_ R<sup>_d×D_</sup> be the weights of the feed-forward network, then the output of this layer is given by computing consecutively, 





Here _σ_ ( _·_ ) denotes the ReLU activation. Consistent with previous literature, the first equation (3.1a) represents the attention layer, combining which with the feed-forward layer in (3.1b) yields a single Transformer layer. We note here that the only difference with standard Transformer architecture is the elimination of the softmax operation and attention mask in the attention layer. 

A Transformer model can contain multiple Transformer layers defined as above, and the output of the whole model would be the composition of multiple layers. For ease of presentation, we omit the details here. From now on, we refer to Transformers with linear attention layers as _linear Transformers_ . 

### **3.2 In-context learning using Transformers** 

In this work we consider two different settings for in-context learning. The first one is that of linear regression, which is commonly studied in the literature (Akyürek et al., 2022; Bai et al., 2023), while we also go one step further and investigate the more difficult task of logistic regression. Our target would be to use the Transformer architecture to emulate in-context known algorithms for solving these tasks. 

#### **3.2.1 Linear Regression** 

For the task of linear regression, let the pairs _{_ ( **a** _i, yi_ ) _}_<sup>_n_</sup> _i_ =1<sup>begivenasinputtotheTransformer,</sup> where **a** _i ∈_ R<sup>_d_</sup> and _yi ∈_ R for all _i_ = 1 _, . . . , n_ . We assume that for each such sequence of pairs, 

4 

there is a weight vector **w** _∗_ , such that _yi_ = **w** _∗_<sup>_⊤_</sup><sup>**a**</sup><sup>_i_+</sup><sup>_ϵi_forall</sup><sup>_i_=1</sup><sup>_, . . . , n_,where</sup><sup>_ϵi_issomenoise.</sup> Given these samples, we want the Transformer to approximate the weight vector **w** _∗_ or make a new prediction on a test point **a** _test_ . 

Define **y** = ( _y_ 1 _, . . . , yn_ )<sup>_⊤_</sup> _∈_ R<sup>_n_</sup> and **A** = [ **a** 1 _, . . . ,_ **a** _n_ ]<sup>_⊤_</sup> _∈_ R<sup>_n×d_</sup> . The standard least-square solution is given by 



As a minimizer of the square loss _ℓ_ ( **w** ) =<sup>�</sup><sup>_n_</sup> _i_ =1<sup>(</sup><sup>_yi −_</sup><sup>**w**</sup><sup>_⊤_</sup><sup>**a**</sup><sup>_i_)2,</sup><sup>**w**ˆcanalsobeobtainedbyminimizing</sup> _ℓ_ ( **w** ) using, e.g., gradient descent. Indeed, it has been shown that Transformers can perform gradient descent to solve linear regression in Akyürek et al. (2022) and later in Bai et al. (2023) with explicit bounds on width and depth requirements. Specifically, in existing works, the number of steps (or equivalently, depth of the Transformer) required for convergence up to _ϵ_ error is of order _O_ ( _κ_ ( **A**<sup>_⊤_</sup> **A** ) log(1 _/ϵ_ )). This suggests that the required number of layers increases linearly with the condition number. 

Another approach is to compute directly the closed form solution (3.2), which involves the matrix inversion. One choice is Newton’s method, an iterative method that approximates the inverse of a matrix with logarithmic dependence on the condition number and quadratic convergence to the accuracy improving upon gradient descent. 

**Newton’s iteration for matrix inversion.** The iteration can be described as follows: Suppose we want to invert a matrix **A** , then with initialization **X** 0 = _α_ **A**<sup>_⊤_</sup> , we compute 



For _α ∈_ (0 _, λ_ max(2 **A**<sup>_⊤_</sup> **A** )<sup>),itcanbeshownthattheestimateis</sup><sup>_ϵ_-accurateafter</sup><sup>_O_(log2</sup><sup>_κ_(</sup><sup>**A**)+</sup> log2 log2(1 _/ϵ_ )) steps (Ogden, 1969; Pan and Schreiber, 1991). 

One interesting generalization of the well-known Newton’s iteration for matrix inversion is the following family of algorithms (Li and Li, 2010): Initialized at **X** 0 = _α_ **A**<sup>_⊤_</sup> , 



We can see that for _n_ = 2 we get the standard Newton’s iteration described in Equation (3.3). For any fixed _n ≥_ 2, the corresponding algorithm has an _n_ -th order convergence to the inverse matrix, i.e., (I _−_ **X** _k_ +1 **A** ) = (I _−_ **X** _k_ **A** )<sup>_n_</sup> , suggesting that the error decays exponentially fast in an order of _n_ . This results in the improvement of the convergence rate by changing the logarithm basis from log2 to log _n_ . More importantly, the initial overhead of constant steps is also reduced, which is the main overhead of Newton’s iteration. This would become clear in the Section 6, where we compare these methods against the Transformer architecture. 

#### **3.2.2 Logistic Regression** 

For in-context learning of logistic regression, we consider pairs of examples _{_ ( **a** _i, yi_ ) _}_<sup>_n_</sup> _i_ =1<sup>where</sup> each **a** _i ∈_ R<sup>_d_</sup> is the covariate vector and _yi ∈{−_ 1 _,_ 1 _}_ is the corresponding label. We assume that _yi_ = sign( **a**<sup>_⊤_</sup> _i_<sup>**w**</sup><sup>_∗_)forsomevector</sup><sup>**w**</sup><sup>_∗∈_R</sup><sup>_d_.Ourtargetistofindavector</sup><sup>**w**ˆ=argmin</sup> **w** _∈_ R<sup>_d f_(</sup><sup>**w**)</sup> where _f_ : R<sup>_d_</sup> _→_ R is the regularized logistic loss defined as 



5 

As in the setting of linear regression, we can use the vector **w** ˆ to make a new prediction on some ˆ point **a** _test_ by calculating 1 _/_ (1 + exp( _−_ **w**<sup>_⊤_</sup> **a** _test_ )). 

The _L_ 2 penalty term is needed to ensure that the loss function is self-concordant in the following sense. 

**Definition 3.1.** [Self-concordant function; Definition 5.1.1, Nesterov et al. 2018] Let _f_ : R<sup>_d_</sup> _→_ R be a closed convex function that is 3 times continuously differentiable on its domain dom( _f_ ) := _{_ **x** _∈_ R<sup>_d_</sup> _| f_ ( **x** ) _< ∞}_ . For any fixed **x** _,_ **u** _∈_ R<sup>_d_</sup> and _t ∈_ R, define _ϕ_ ( _t_ ; **x** _,_ **u** ) := _f_ ( **x** + _t_ **u** ) as a function of _t_ . Then we say _f_ is _self-concordant_ if there exists a constant _Mf_ such that, for all **x** _∈_ dom( _f_ ) and **u** _∈_ R<sup>_d_</sup> with **x** + _t_ **u** _∈_ dom( _f_ ) for all sufficiently small _t_ , 



We say _f_ is _standard self-concordant_ when _Mf_ = 1. 

In particular, the regularized logistic loss is a self-concordant function under a mild assumption on the data. 

**Assumption 3.2.** For the data _{_ ( **a** _i, yi_ ) _}_<sup>_n_</sup> _i_ =1<sup>in(3.5),itholdsthat</sup><sup>_∥_</sup><sup>**a**</sup><sup>_i∥_</sup> 2<sup>_≤_1forall</sup><sup>_i_= 1</sup><sup>_, . . . , n_.</sup> 

**Proposition 3.3** (Lemma 2, Zhang and Xiao 2015) **.** _For µ >_ 0 _, the regularized logistic loss f_ ( _·_ ) _defined in_ (3.5) _is self-concordant with Mf_ = 1 _/_<sup>_√_</sup> _<u>µ</u> under assumption 3.2. Furthermore, the function f/_ 4 _µ is standard self-concordant._ 

Self-concordance ensures rapid convergence of second-order optimization algorithms such as Newton’s method. As in the case of matrix inversion, the rate of convergence is quadratic after a constant number of steps that depend on how close the algorithm was initialized to the minimum. 

**Newton’s method.** Given the initialization **x** 0 _∈_ R<sup>_d_</sup> , the Newton’s method updates as follows: 



Different choices of the step-size _η_ ( **x** _t_ ) lead to different variants of the algorithm: For _<u>η</u>_ = 1 we have <u>1</u> the “classic” Newton’s method; for _η_ ( **x** _t_ ) = 1+ _λ_ ( **x** _t_ )<sup>with</sup><sup>_λ_(</sup><sup>**x**</sup><sup>_t_) =</sup> � _∇f_ ( **x** _t_ )<sup>_⊤_</sup> [ _∇_<sup>2</sup> _f_ ( **x** _t_ )]<sup>_−_1</sup> _∇f_ ( **x** _t_ ), we have the so-called damped Newton’s method (see, e.g., Section 4 in Nesterov et al. (2018)). 

For self-concordant functions, the damped Newton’s method has guarantees for global convergence, which contains two phases: In the initial phase, the quantity _λ_ ( **x** ) is decreased until it drops below the threshold of 1 _/_ 4. While _λ_ ( **x** ) _≥_ 1 _/_ 4, there is a constant decrease per step of the self-concordant function _g_ of at least 0 _._ 02. Then, the second phase begins once _λ_ ( **x** ) drops below the 1 _/_ 4 threshold, afterwards it will decay with a quadratic rate, implying a quadratic convergence of the objective value. All together, this implies that _c_ + log log(1 _/ϵ_ ) steps are required to reach _ϵ_ accuracy. For the analysis of the exact Newton’s method for self-concordant functions one may refer to Nesterov et al. (2018). 

## **4 Main Results for linear regression** 

We now present our main results on in-context learning for linear regression using Transformers. The corresponding proofs of results in this section can be found in Appendix A. 

Notice that in order to obtain the least-square solution in (3.2), it suffices to perform the operations of matrix inversion, matrix multiplications and matrix transposition. The linear Transformer architecture can trivially perform the last two operations, while our first result in Lemma 4.1 below shows that it can also efficiently approximate matrix inversion via Newton’s iteration. 

6 

**Lemma 4.1.** _For any dimension d, there exists a linear Transformer consisting of 2 linear attention layers, each of which has 2 attention heads and width_ 4 _d, such that it can perform one step of Newton’s iteration for any target matrix_ **A** _∈_ R<sup>_d×d_</sup> _. Specifically, the Transformer realizes the following mapping from input to output for any_ **X** 0 _∈_ R<sup>_d×d_</sup> _:_ 



_where_ **X** 1 = **X** 0(2I _d −_ **AX** 0) _, corresponding to one step of Newton’s iteration in Equation_ (3.3) _. Furthermore, if restricted to only symmetric_ **A** _, then_ 1 _layer suffices._ 

Built upon the construction from the above lemma, one can implement multiple steps of Newton’s iteration by repeatedly applying such constructed layers. This gives rise to the following main result on how linear Transformer can solve linear regression in-context. 

**Theorem 4.2** (Linear regression) **.** _For any dimension d, n and index T >_ 0 _, there exists a linear Transformer consisting of_ 3 + _T layers, where each layer has 2 attention heads and width equal to_ 4 _d_ + 3 _, such that it realizes the following mapping from input to output:_ 



ˆ _where_ **A** = ( **a** 1 _, . . . ,_ **a** _n_ )<sup>_⊤_</sup> _,_ **y** = ( _y_ 1 _, . . . , yn_ )<sup>_⊤_</sup> _, and y_ = **a**<sup>_⊤_</sup> test<sup>**X**</sup><sup>_T_</sup><sup>**A**</sup><sup>_⊤_</sup><sup>**y**</sup><sup>_isthepredictionwith_</sup><sup>**X**</sup><sup>_Tbeing_</sup> _the output of T steps of Newton’s iteration for inversion on the matrix_ **A**<sup>_⊤_</sup> **A** _, where the initialization is_ **X** 0 = _ϵ_ **A**<sup>_⊤_</sup> **A** _for some ϵ ∈_ (0 _, λ_<sup>2</sup> max(2 **A**<sup>_⊤_</sup> **A** )<sup>)</sup><sup>_._</sup> 

_Remark_ 4.3 _._ In Theorem 4.2, the three extra layers are used to create the matrix **A**<sup>_⊤_</sup> **A** , perform the multiplication **AX** _T_ **a** _test_ , and execute the final multiplication with **y**<sup>_⊤_</sup> . 

_Remark_ 4.4 _._ We note here that our construction corresponds to the solution of ridgeless regression, and it can be easily extended to the case of ridge regression, which amounts to inverting **A**<sup>_⊤_</sup> **A** + _µ_ I _d_ for some regularization parameter _µ_ . 

_Remark_ 4.5 _._ Considering previous results of implementing gradient descent for this task (e.g., Akyürek et al. 2022; Bai et al. 2023), one may observe the memory trade-off between the two methods. In the case of gradient descent, there is no need of any extra memory, and each step can be described as **w** _t_<sup>_⊤_</sup> +1<sup>=</sup><sup>**w**</sup> _t_<sup>_⊤−η_(</sup><sup>**yA**</sup><sup>_−_</sup><sup>**w**</sup> _t_<sup>_⊤_</sup><sup>**A**</sup><sup>_⊤_</sup><sup>**A**)forsomestep-size</sup><sup>_η_.Thuswecanupdatethevector</sup> **w** _t_ without storing the intermediate result **A**<sup>_⊤_</sup> **A** . While applying Newton’s iteration requires to store the intermediate results, as well as the initialization which results to the need of width equal to 4 _d_ . Nevertheless, the latter achieves better dependence on the condition number (logarithmic instead of linear) and quadratic decay of the error instead of linear. The recent work by Fu et al. (2023) studies also Newton’s iteration as a baseline to compare with Transformers in the setting of linear regression. They showed that Transformers can implement Newton’s iteration, the stated result needs _O_ ( _T_ ) layers and _O_ ( _d_ ) width to perform _T_ steps of Newton’s iteration on matrices of size _d_ . 

Below in Section 6, we will compare the predictions made by different orders of Newton’s iteration Equation (3.4) with the Transformer architecture, as well as the acquired loss. 

7 

## **5 Main Results for logistic regression** 

In this section we will present constructive arguments showing that Transformers can approximately implement Newton’s method for logistic regression. To the best of our knowledge, the only result prior to our work for logistic regression is that of Bai et al. (2023) in which they implement gradient descent on the logistic loss using transformers. Newton’s method can achieve a quadratic rate of convergence instead of linear with respect to the achieved accuracy. 

As presented in Section 3.2.2, we seek to minimize the regularized logistic loss defined in (3.5), which is self-concordant by Proposition 3.3. In a nutshell, our main result in this case shows that linear Transformer can efficiently emulate Newton’s method to minimize the loss. This is summarized in the following theorem. We remind that _f_ is the regularized logistic loss of Equation (3.5). 

**Theorem 5.1.** _For any dimension d, consider the regularized logistic loss defined in_ (3.5) _with regularization parameter µ >_ 0 _, and define κf_ = max **x** _∈_ R _d κ_ ( _∇_<sup>2</sup> _f_ ( **x** )) _. Then for any T >_ 0 _and ϵ >_ 0 _, there exists a linear Transformer that can approximate T iterations of Newton’s method on the regularized logistic loss up to error ϵ per iteration. In particular, the width of such a linear Transformer can be bounded by O_ ( _d_ (1+ _µ_ )<sup>6</sup> _/ϵ_<sup>4</sup> _µ_<sup>8</sup> ) _, and its depth can be bounded by T_ (11+2 _k_ ) _, where k ≤_ 2 log _κf_ + log log<sup><u>(1</u></sup> _ϵ_<sup><u>+2</u></sup> _µ_<sup>_<u>µ</u>_2)3</sup><sup>_.Furthermore,thereisaconstantc >_0</sup><sup>_dependingonµsuchthatifϵ < c,_</sup> ˜ ˜ ˆ _then the output of the Transformer provides a_ **w** _satisfying that ∥_ **w** _−_ **w** _∥_ 2 _≤ O_ ( ~~�~~ _ϵ_ (1 + _µ_ ) _/_ (4 _µ_ )) _, where_ **w** ˆ _is the global minimizer of the loss._ 

The proof of Theorem 5.1 contains two main components: The approximate implementation of Newton’s method by Transformer and the convergence analysis of the resulting inexact Newton’s method. Below we will address these two parts separately in Section 5.1 and Section 5.2. 

### **5.1 Transformer can implement Newton’s method for logisitic regression** 

To analyze the convergence properties of Newton’s method on _f_ , we actually implement the updates on _g_ = _f/_ 4 _µ_ , which based on Proposition 3.3 is standard self-concordant. Specifically, the targeted update is 



where the second equality follows directly from the definition _g_ = _f/_ 4 _µ_ . Our first result is that a linear Transformer can approximately implement the update above. 

**Theorem 5.2.** _Under the setting of Theorem 5.1, there exists a Transformer consisting of linear attention with ReLU layers that can approximately perform damped Newton’s method on the regularized logistic loss as follows_ 



_where_ **_ε_** _is an error term. For any ϵ >_ 0 _, to achieve that ∥_ **_ε_** _∥_ 2 _≤ ϵ, the width of such a Transformer can be bounded by O_ ( _d_<sup><u>(</u></sup> _ϵ_<sup>14+</sup> _µ_<sup>_<u>µ</u>_10)8)</sup><sup>_,anditsdepthcanbeboundedby_11+2</sup><sup>_kwherek≤_2 log</sup><sup>_κf_+log log</sup><sup><u>(1</u></sup> _ϵ_<sup><u>+2</u></sup> _µ_<sup>_<u>µ</u>_2)3</sup><sup>_._</sup> 

_Remark_ 5.3 _._ Here we omit the details of input and output format for ease of presentation. See Appendix C for details. Roughly speaking, the input to the Transformer is of dimension (6 _d_ + 4) _× n_ and contains the matrix **A** , the labels **y** , and the initialization **x** 0. 

8 

Below we provide the main idea of the proof by describing main steps of our construction. 

_Proof sketch of Theorem 5.2._ To construct a Transformer that implements Newton’s method to optimize the regularized logistic loss, we implement first each one of the required components, including gradient _∇f_ ( **x** ), Hessian _∇_<sup>2</sup> _f_ ( **x** ), and the damping parameter _λ_ ( **x** ). The gradient and Hessian of _f_ are 



where each _pi_ := exp( _−yi_ **x**<sup>_⊤_</sup> **a** _i_ ) _/_ (1 + exp( _−yi_ **x**<sup>_⊤_</sup> **a** _i_ )) and **D** := diag( _p_ 1(1 _− p_ 1) _, . . . , pn_ (1 _− pn_ )). 

**Step 1: Approximate gradient and Hessian.** We use the linear attention layer to perform matrix multiplications, and we use the ReLU network to approximate the following quantities: 

- The values _pi_ . 

- The diagonal elements of the matrix **D** . 

• The “hidden” dot product of the diagonal elements of **D** and the matrix **A** , since the _i_ -th row of **DA** is _di_ **a**<sup>_⊤_</sup> _i_<sup>.</sup> Theseˆ approximationsˆ give rise to the approximated gradient and Hessian of _f_ , which we denote by _∇f_ ( **x** ) _, ∇_<sup>2</sup> _f_ ( **x** ). 

**Step 2: Invert the Hessian.** For the next step, we use the construction presented in Lemma 4.1 to invert the matrix _∇_<sup>ˆ2</sup> _f_ ( **x** ). We leverage classical results in matrix perturbation theory to show that ( _∇_<sup>ˆ2</sup> _f_ ( **x** ))<sup>_−_1</sup> = ( _∇_<sup>2</sup> _f_ ( **x** ))<sup>_−_1</sup> + **E** 1 _,t_ , where **E** 1 _,t_ is an error matrix whose norm can be bounded based on the number of iterations performed for the inversion. 

**Step 3: Approximate the step size.** Next, we need to approximate the step size 2<sup>_~~√~~_</sup> _<u>µ</u>_ 2<sup>_<u>√</u>_</sup> + _<u>µλ</u>_ ( **x** )<sup>.</sup> This is done using the ReLU layers. Recall that _λ_ ( **x** )<sup>2</sup> = _∇f_ ( **x** )<sup>_⊤_</sup> ( _∇_<sup>2</sup> _f_ ( **x** ))<sup>_−_1</sup> _∇f_ ( **x** ), which can be approximated using _∇_<sup>ˆ</sup> _f_ ( **x** ) _, ∇_<sup>2</sup> _f_ ( **x** ) from the previous steps. To get the step size, we need to further approximate the function _g_ ( _z_ ) = 1 _/_ (1 +<sup>_√_</sup> _<u>z</u>_ <u>)</u> and evaluate it at _λ_ ( **x** )<sup>2</sup> . Thus, any error in the approximation of _λ_ ( **x** )<sup>2</sup> translates to a square root error in the calculation of _g_ ( _λ_ ( **x** )<sup>2</sup> ). To see this, consider the derivative of the function _g_ , _g_<sup>_′_</sup> ( _z_ ) = _−_ 2<sup>_√_</sup> _<u>z/</u>_ (1 +<sup>_√_</sup> _<u>z</u>_ <u>),</u> and observe that for _z_ of constant order, if _z_ changes by _ϵ_ , the corresponding change in _g_ ( _z_ ) would be of order<sup>_√_</sup> _<u>ϵ</u>_ <u>.</u> This leads to the quadratic requirement of width with respect to the desired error threshold. 

**Step 4: Aggregate all approximations.** Finally, we aggregate all the error terms induced by each approximation step and get the desired approximation to one step of Newton’s method. ■ 

### **5.2 Convergence of inexact Newton’s method** 

Theorem 5.2 shows that linear Transformers can approximately implement damped Newton’s method for logistic regression, while providing width requirements in order to control the approximation error. In complement to this, we further provide convergence analysis for the resulting algorthm. 

9 

**Theorem 5.4.** _Under Assumption 3.2, for the regularized logistic loss f defined in_ (3.5) _with regularization parameter µ >_ 0 _, consider a sequence of iterates {_ **x** _t}t≥_ 0 _satisfying_ 



_where ∥_ **_ε_** _t∥_ 2 _≤ ϵ for all t ≥_ 0 _. Then there exists constants c, C_ 1 _, C_ 2 _depending only on µ such that for any ϵ ≤ c, it holds that g_ ( **x** _t_ ) _− g_ ( **x**<sup>_∗_</sup> ) _≤ ϵ for all t ≥ C_ 1 + _C_ 2 log log<sup><u>1</u></sup><sup>_._</sup> 

_ϵ_<sup>_._</sup> 

The proof of Theorem 5.4 follows the proof presented in Nesterov et al. (2018) but accounts for the error term. Similar analysis has been performed in the past in Sun et al. (2020). Our proof consists of the following steps: 

- **Feasibility** : We first show by induction that there exists a constant depending on _µ_ such that _∥_ **x** _t∥_ 2 _≤ C_ for all _t ≥_ 0. This is a consequence of the bounded norm assumption in Assumption 3.2 and the strongly convex _L_ 2 regularizer. 

- **Constant decrease** : We then show constant decreasing of the loss value per step when _λ_ ( **x** _t_ ) _≥_ 1 _/_ 6. This is achieved by upper bounding the suboptimality of _g_ ( **x** _t_ ) a function of _λ_ ( **x** _t_ ), which is guaranteed by the self-concordance of _g_ . 

- **Quadratic convergence** : In the last step, we show that when _λ_ ( **x** _t_ ) _<_ 1 _/_ 6, the inequality _λ_ ( **x** _t_ +1) _≤ cλ_ ( **x** _t_ )<sup>2</sup> + _ϵ_<sup>_′_</sup> holds, which implies further quadratic convergence to achieving _O_ ( _ϵ_ ) error. 

See Appendix B.2 for the complete proof of Theorem 5.4. Finally, combining the construction in Theorem 5.2 and the convergence analysis in Theorem 5.4 yields the performance guarantee of the constructed Transformer. 

## **6 Experiments** 

In this section, we corroborate our theoretical findings with empirical evidence. Specifically, we aim to empirically demonstrate the effectiveness of the linear self-attention model when it is trained to solve linear regression as well as logistic regression using encoder-based models. Our code is available here<sup>2</sup> . 

**Linear regression.** For the task of linear regression, we train models consisting of linear selfattention (LSA) with and without LayerNorm (LN) to learn in-context. We use an embedding dimension of 64 and 4 attention heads. The data dimension is 10, and the input contains 50 in-context samples. 

We train models with LSA, having from 1 to 6 layers, employing the training technique of von Oswald et al. (2023a) to stabilize the training process. Consistent with the findings reported in von Oswald et al. (2023a), we observe that after four layers, there is no significant improvement in performance by adding more layers. We attribute this to limitations in the training process, and the optimal training method for linear Transformers beyond 5 _−_ 6 layers remains unidentified. We furthermore train LSA models with LN to be able to test more than 6 layers, but to also test how LN affects the capabilities of these models. 

> 2 `https://anonymous.4open.science/r/transformer_higher_order-B80B/` 

10 



<!-- Start of picture text -->
Linear Regression Error Linear Regression Error<br>10 1 10 0<br>10 2 10 3<br>10 5 10 6<br>LSA LSA w/ LN<br>10 8 Newton(order=2) 10 9 Newton(order=4)<br>Newton(order=3) Newton(order=6)<br>10 11 Newton(order=4) 10 12 Newton(order=8)<br>Newton(order=5) Newton(order=10)<br>1 2 3 4 5 6 1 2 3 4 5 6<br>layers / steps layers / steps<br><!-- End of picture text -->

Figure 2: _Loss of LSA, LSA with layernorm and different order Newton iteration for linear regression error._ 

In Figure 2, we plot the in-distribution loss (meaning we keep the same sampling method as in the training process) on new test samples. We observe that models having 4 or more layers have almost the same performance in terms of the loss. 

To get a clearer picture of what the models are learning, we further provide plots illustrating the actual outputs of the trained Transformers, as shown in Figures 3 and 5. Here we test the trained models in the following set-up: we first sample a batch of 5000 examples, i.e., _{_ ( **x** _i, yi_ ) _}_<sup>_n_</sup> _i_ =1<sup>data</sup> pairs that all share the same underlying weight vector _yi_ = **w** _∗_<sup>_⊤_</sup><sup>**x**</sup><sup>_i_,forall</sup><sup>_i_=1</sup><sup>_, . . . , n_andallthe</sup> batches. We then pick a specific **x** _test_ sample, which we keep the same for all the batches. For this test sample, the value of its first coordinate varies across [ _−a, a_ ] for different values of _a_ . 

We test two different cases: 1) the value _a_ has been encountered in the training set with probability at least 0 _._ 99 and 2) the value is an outlier. We calculate this value to be [ _−_ 14 _._ 9 _,_ 14 _._ 9], which is the range of the ‘in-distribution’ plots in Figures 3 and 5 (range in between the dashed line), while in Figures 4 and 6 we have the ‘out-of-distribution’ ones. 

Looking at Figure 3, one may observe that the performance of the model lies between the secondand third-order Newton’s iteration. Intuitively, Transformers do have slighter higher order capacity than Newton’s iteration. To see that, assume that the model is given in the input a symmetric matrix **A** , then in the first layer from Equation (3.1a) the model can create up to the third power of the matrix **A** , i.e., all powers for 1 _−_ 3. Now in the second layer the higher-order term can be up to power of 9; in contrast the second-order Newton’s iteration can reach up to order 7 (the first iteration **X** 1 _∼_ **A**<sup>3</sup> , and in the second one **X** 2 _∼_ **X** 1 **AX** 1 _∼_ **A**<sup>7</sup> . 

The gap between the possible powers of the matrix, between second-order Newton’s and Transformers will increase as the number of layers increases. On the other hand, third order Newton’s has up to power **A**<sup>5</sup> in the fist step **X** 1. While in second one the maximum power is **X** 1( **AX** 1)<sup>2</sup> _∼_ **A**<sup>17</sup> . Thus, the LSA Transformer will not be able to outperform it. 

**Logistic regression.** We further analyze the Transformer’s ability to solve logistic regression tasks. To simplify the setting, we focus on training the Transformer to predict the logistic regression parameter _w_ . Since predicting the true weight directly is a hard task and is not necessarily the solution of the minimization problem described in Equation (3.5), we instead opt to train the Transformer to output a solution comparable to that given by Newton’s Method. 

11 



<!-- Start of picture text -->
20 1 20 2 20 3 20 4 20 5 20 6<br>0 0 0 0 0 0<br>20 20 20 20 20 20<br>20 0 20 20 0 20 20 0 20 20 0 20 20 0 20 20 0 20<br>20 20 20 20 20 20<br>0 0 0 0 0 0<br>20 20 20 20 20 20<br>20 0 20 20 0 20 20 0 20 20 0 20 20 0 20 20 0 20<br>20 20 20 20 20 20<br>0 0 0 0 0 0<br>20 20 20 20 20 20<br>20 0 20 20 0 20 20 0 20 20 0 20 20 0 20 20 0 20<br>LSA<br>Ord. 2<br>Ord. 3<br><!-- End of picture text -->

Figure 3: _Output of LSA without LayerNorm and Newton’s iteration by keeping the test sample fixed and changing one of its coordinates, within in-distribution. We plot_ 1 _−_ 6 _layers against_ 1 _−_ 6 _steps of second and third order Newton’s iteration ; we observe that the model lies between second and third order._ 



<!-- Start of picture text -->
100 1 100 2 100 3 100 4 100 5 100 6<br>0 0 0 0 0 0<br>100 1 00 1 00 1 00 1 00 1 00<br>100 0 100 100 0 100 100 0 100 100 0 100 100 0 100 100 0 100<br>100 100 100 100 100 100<br>0 0 0 0 0 0<br>100 1 00 1 00 1 00 1 00 1 00<br>100 0 100 100 0 100 100 0 100 100 0 100 100 0 100 100 0 100<br>100 100 100 100 100 100<br>0 0 0 0 0 0<br>100 1 00 1 00 1 00 1 00 1 00<br>100 0 100 100 0 100 100 0 100 100 0 100 100 0 100 100 0 100<br>LSA<br>Ord. 2<br>Ord. 3<br><!-- End of picture text -->

Figure 4: _Same setting as above, but the models/algorithms are tested with out-of-distribution values as well. We observe that the model does not actually learn the underlying linear function._ 



<!-- Start of picture text -->
Logistic Regression Loss Training Error<br>TF TF<br>Newton Method 10 1<br>1.0<br>0.6 10 2<br>0 10 20 30 40 0 10 20 30 40<br>layers / steps layers<br><!-- End of picture text -->

Figure 7: _Performance of Transformer on logistic regression tasks. (Left)_ The logistic regression loss for the Transformer (TF) and the Newton Method, with a regularization _µ_ = 0 _._ 1. According to our theoretical construction, a single step of the Newton Method can be implemented by at least 11 layers, therefore, we have scaled the Newton Method plot to 13 layers per step. The Transformer is shown to approximate the method more effectively within a few layers. _(Right)_ The training error of the Transformer when it is trained to predict the solution derived from Newton’s Method. 

12 



<!-- Start of picture text -->
1 3 5 7 10 12<br>20 20 20 20 20 20<br>0 0 0 0 0 0<br>20 20 20 20 20 20<br>20 0 20 20 0 20 20 0 20 20 0 20 20 0 20 20 0 20<br>20 20 20 20 20 20<br>0 0 0 0 0 0<br>20 20 20 20 20 20<br>20 0 20 20 0 20 20 0 20 20 0 20 20 0 20 20 0 20<br>20 20 20 20 20 20<br>0 0 0 0 0 0<br>20 20 20 20 20 20<br>20 0 20 20 0 20 20 0 20 20 0 20 20 0 20 20 0 20<br>LSA w/ LN<br>Ord. 3<br>Ord. 5<br><!-- End of picture text -->

Figure 5: _Output of LSA with LayerNorm and Newton’s iteration by keeping the test sample fixed and changing one of its coordinates, within in-distribution. We observe that LN improves the performance of the model. The spikes are noted in the out-of-distribution range._ 



<!-- Start of picture text -->
100 1 100 3 100 5 100 7 100 10 100 12<br>0 0 0 0 0 0<br>100 1 00 1 00 1 00 1 00 1 00<br>100 0 100 100 0 100 100 0 100 100 0 100 100 0 100 100 0 100<br>100 100 100 100 100 100<br>0 0 0 0 0 0<br>100 1 00 1 00 1 00 1 00 1 00<br>100 0 100 100 0 100 100 0 100 100 0 100 100 0 100 100 0 100<br>100 100 100 100 100 100<br>0 0 0 0 0 0<br>100 1 00 1 00 1 00 1 00 1 00<br>100 0 100 100 0 100 100 0 100 100 0 100 100 0 100 100 0 100<br>LSA w/ LN<br>Ord. 3<br>Ord. 5<br><!-- End of picture text -->

Figure 6: _Same setting as above, but the models/algorithms are tested with out-of-distribution values as well._ 5 _layers seem to perform better for out-of-distribution than_ 7 _,_ 10 _,_ 12 _, not though in-distribution._ 

We train on data dimension 5, and the input contains 26 in-context samples, and embedding dimension 32, with 4 heads. We set the regularization parameter _µ_ = 0 _._ 1 in the regularized logistic loss in (3.5). In Figure 7, we compare the loss value of the trained Transformer across different number of layers and the loss value of different iterates of Newton’s method in the left plot; we further plot the corresponding training error in the right plot. According to our construction in Theorem 5.2, a single step of the Newton Method is equivalent to 13 layers of a standard Transformer. However, when trained, the Transformer can approximate the Newton Method even more effectively as shown in Figure 7. 

## **7 Discussion** 

In this paper we provide explicit construction of Transformers that can efficiently perform the matrix inversion operation. Built upon this, we further show that Transformers can compute the least-square 

13 

solution to solve the linear regression task in-context. Moreover, for in-context learning of logistic regression, it also gives rise to construction of Transformers that can perform Newton’s method to optimize the regularized logistic loss. We provide concrete convergence analysis of the inexact Newton’s method emulated by the constructed Transformer. We further examine and compare the empirical performance of the trained Transformers with those of the higher-order methods. 

## **References** 

- Ahn, K., Cheng, X., Daneshmand, H. and Sra, S. (2023). Transformers learn to implement preconditioned gradient descent for in-context learning. 

- Akyürek, E., Schuurmans, D., Andreas, J., Ma, T. and Zhou, D. (2022). What learning algorithm is in-context learning? investigations with linear models. _arXiv preprint arXiv:2211.15661_ . 

- Bai, Y., Chen, F., Wang, H., Xiong, C. and Mei, S. (2023). Transformers as statisticians: Provable in-context learning with in-context algorithm selection. 

- Black, S., Biderman, S., Hallahan, E., Anthony, Q., Gao, L., Golding, L., He, H., Leahy, C., McDonell, K., Phang, J. et al. (2022). Gpt-neox-20b: An open-source autoregressive language model. _arXiv preprint arXiv:2204.06745_ . 

- Boyd, S. P. and Vandenberghe, L. (2004). _Convex optimization_ . Cambridge university press. 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A. et al. (2020). Language models are few-shot learners. _Advances in neural information processing systems_ **33** 1877–1901. 

- Chen, S., Sheen, H., Wang, T. and Yang, Z. (2024). Training dynamics of multi-head softmax attention for in-context learning: Emergence, convergence, and optimality. _arXiv preprint arXiv:2402.19442_ . 

- Cheng, X., Chen, Y. and Sra, S. (2023). Transformers implement functional gradient descent to learn non-linear functions in context. 

- Dasgupta, I., Lampinen, A. K., Chan, S. C., Creswell, A., Kumaran, D., McClelland, J. L. and Hill, F. (2022). Language models show human-like content effects on reasoning. _arXiv preprint arXiv:2207.07051_ . 

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_ . 

- Fu, D., Chen, T.-Q., Jia, R. and Sharan, V. (2023). Transformers learn higher-order optimization methods for in-context learning: A study with linear models. 

- Garg, S., Tsipras, D., Liang, P. S. and Valiant, G. (2022). What can transformers learn in-context? a case study of simple function classes. _Advances in Neural Information Processing Systems_ **35** 30583–30598. 

14 

- Giannou, A., Rajput, S., Sohn, J.-y., Lee, K., Lee, J. D. and Papailiopoulos, D. (2023). Looped transformers as programmable computers. _arXiv preprint arXiv:2301.13196_ . 

- Guo, T., Hu, W., Mei, S., Wang, H., Xiong, C., Savarese, S. and Bai, Y. (2023). How do transformers learn in-context beyond simple functions? a case study on learning with representations. _arXiv preprint arXiv:2310.10616_ . 

- Huang, Y., Cheng, Y. and Liang, Y. (2023). In-context convergence of transformers. _arXiv preprint arXiv:2310.05249_ . 

- Kenton, J. D. M.-W. C. and Toutanova, L. K. (2019). Bert: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of NAACL-HLT_ . 

- Khan, S., Naseer, M., Hayat, M., Zamir, S. W., Khan, F. S. and Shah, M. (2022). Transformers in vision: A survey. _ACM computing surveys (CSUR)_ **54** 1–41. 

- Li, W. and Li, Z. (2010). A family of iterative methods for computing the approximate inverse of a square matrix and inner inverse of a non-square matrix. _Applied Mathematics and Computation_ **215** 3433–3442. 

- Li, Y., Ildiz, M. E., Papailiopoulos, D. and Oymak, S. (2023). Transformers as algorithms: Generalization and stability in in-context learning. _International Conference on Machine Learning_ . 

- Lieber, O., Sharir, O., Lenz, B. and Shoham, Y. (2021). Jurassic-1: Technical details and evaluation. _White Paper. AI21 Labs_ **1** 9. 

- Mahankali, A., Hashimoto, T. B. and Ma, T. (2023). One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. _arXiv preprint arXiv:2307.03576_ . 

Nesterov, Y. et al. (2018). _Lectures on convex optimization_ , vol. 137. Springer. 

- Nye, M., Andreassen, A. J., Gur-Ari, G., Michalewski, H., Austin, J., Bieber, D., Dohan, D., Lewkowycz, A., Bosma, M., Luan, D. et al. (2021). Show your work: Scratchpads for intermediate computation with language models. _arXiv preprint arXiv:2112.00114_ . 

- Ogden, H. C. (1969). Iterative methods of matrix inversion . 

- Pan, V. and Schreiber, R. (1991). An improved newton iteration for the generalized inverse of a matrix, with applications. _SIAM Journal on Scientific and Statistical Computing_ **12** 1109–1130. 

- Schulz, G. (1933). Iterative berechung der reziproken matrix. _ZAMM - Journal of Applied Mathematics and Mechanics / Zeitschrift für Angewandte Mathematik und Mechanik_ **13** 57–59. 

- Stewart, G. W. and guang Sun, J. (1990). Matrix perturbation theory. 

- Sun, T., Necoara, I. and Tran-Dinh, Q. (2020). Composite convex optimization with global and local inexact oracles. _Computational Optimization and Applications_ **76** 69–124. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. and Polosukhin, I. (2017). Attention is all you need. _Advances in neural information processing systems_ **30** . 

15 

- von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A. and Vladymyrov, M. (2022). Transformers learn in-context by gradient descent. _arXiv preprint arXiv:2212.07677_ . 

- von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A. and Vladymyrov, M. (2023a). Transformers learn in-context by gradient descent. 

- von Oswald, J., Niklasson, E., Schlegel, M., Kobayashi, S., Zucchet, N., Scherrer, N., Miller, N., Sandler, M., y Arcas, B. A., Vladymyrov, M., Pascanu, R. and Sacramento, J. (2023b). Uncovering mesa-optimization algorithms in transformers. 

- Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D. et al. (2022a). Emergent abilities of large language models. _arXiv preprint arXiv:2206.07682_ . 

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Chi, E., Le, Q. and Zhou, D. (2022b). Chain of thought prompting elicits reasoning in large language models. _arXiv preprint arXiv:2201.11903_ . 

- Wu, M., Yin, B., Wang, G., Dick, C., Cavallaro, J. R. and Studer, C. (2014). Large-scale mimo detection for 3gpp lte: Algorithms and fpga implementations. _IEEE Journal of Selected Topics in Signal Processing_ **8** 916–929. 

- Yuan, L., Chen, Y., Wang, T., Yu, W., Shi, Y., Jiang, Z.-H., Tay, F. E., Feng, J. and Yan, S. (2021). Tokens-to-token vit: Training vision transformers from scratch on imagenet. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ . 

- Zhang, R., Frei, S. and Bartlett, P. L. (2023). Trained transformers learn linear models in-context. _arXiv preprint arXiv:2306.09927_ . 

- Zhang, Y. and Xiao, L. (2015). Communication-efficient distributed optimization of self-concordant empirical loss. 

- Zhou, H., Nova, A., Larochelle, H., Courville, A., Neyshabur, B. and Sedghi, H. (2022). Teaching algorithmic reasoning via in-context learning. _arXiv preprint arXiv:2211.09066_ . 

16 

## **A Constructions of Transformers for linear regression** 

In this section we provide the exact constructions for implementing Newton’s Iteration for matrix inversion. We then use this construction to approximate the closed form solution of linear regression in-context. 

### **A.1 Newton-Raphson Method for Matrix Inverse** 

The Newton-Raphson method (Schulz, 1933) for inverting a matrix **A** _∈_ R<sup>_d×d_</sup> is defined by the following update rule: 



An initialization for this method that guarantees convergence is **X** 0 = _ϵ_ **A**<sup>_⊤_</sup> , where _ϵ ∈_ (0 _, λ_ 1( **AA** <u>2</u><sup>_⊤_</sup> )<sup>).</sup> We prove below that linear-attention transformers can emulate the above update. 

**Lemma 4.1.** _For any dimension d, there exists a linear Transformer consisting of 2 linear attention layers, each of which has 2 attention heads and width_ 4 _d, such that it can perform one step of Newton’s iteration for any target matrix_ **A** _∈_ R<sup>_d×d_</sup> _. Specifically, the Transformer realizes the following mapping from input to output for any_ **X** 0 _∈_ R<sup>_d×d_</sup> _:_ 



_where_ **X** 1 = **X** 0(2I _d −_ **AX** 0) _, corresponding to one step of Newton’s iteration in Equation_ (3.3) _. Furthermore, if restricted to only symmetric_ **A** _, then_ 1 _layer suffices. Proof._ Assume that we are given the following input 



For the first layer, we choose 



so that 

Then the output of the first layer is 



17 

Next, we use two attention heads for the second layers. For the first head, we choose 



so that 

For the second head, we choose 

so that 

Combining the outputs of the two heads, we get the output of the second layer as 



This completes the proof. For the case where the matrix **A** is symmetric, we give the construction as part of the proof for linear regression. See the proof of Theorem 4.2 in Appendix A.2. ■ 

### **A.2 Linear regression through matrix inversion.** 

Given the least-square solution, the prediction for a fresh test point **a** test is 



Below we give a construction of Transformer that can emulate the Newton-Raphson method to approximate the inverse of **A**<sup>_⊤_</sup> **A** and then multiply the result with the necessary quantities to obtain an approximation of _y_ test. Notice that here the matrix we want to invert, **A**<sup>_⊤_</sup> **A** , is symmetric. 

18 

**Theorem 4.2** (Linear regression) **.** _For any dimension d, n and index T >_ 0 _, there exists a linear Transformer consisting of_ 3 + _T layers, where each layer has 2 attention heads and width equal to_ 4 _d_ + 3 _, such that it realizes the following mapping from input to output:_ 



ˆ _where_ **A** = ( **a** 1 _, . . . ,_ **a** _n_ )<sup>_⊤_</sup> _,_ **y** = ( _y_ 1 _, . . . , yn_ )<sup>_⊤_</sup> _, and y_ = **a**<sup>_⊤_</sup> test<sup>**X**</sup><sup>_T_</sup><sup>**A**</sup><sup>_⊤_</sup><sup>**y**</sup><sup>_isthepredictionwith_</sup><sup>**X**</sup><sup>_Tbeing_</sup> _the output of T steps of Newton’s iteration for inversion on the matrix_ **A**<sup>_⊤_</sup> **A** _, where the initialization is_ **X** 0 = _ϵ_ **A**<sup>_⊤_</sup> **A** _for some ϵ ∈_ (0 _, λ_<sup>2</sup> max(2 **A**<sup>_⊤_</sup> **A** )<sup>)</sup><sup>_._</sup> 

_Proof._ We denote by **H** 0 the input to the Transformer. To prove the desired result we present in steps the construction. 

**Step 1: Initialize (1 layer).** We consider one Transformer layer with 2 heads. For the first head, we choose 



19 

Then combining the above two heads, we have 



**Step 2: Implement** _T_ **steps of Newton-Raphson (** _T_ **layers).** We now define **X** 0 = _ϵ_ **A**<sup>_⊤_</sup> **A** and **R** = **A**<sup>_⊤_</sup> **A** . The input matrix to the next layer is in the following form (for _t_ = 0) 



We will show that one Transformer laye with two heads can yield the following output 



Here we choose the weight matrices for the first head to be 



20 

so that 



Combining these two heads, we obtain 



21 

Repeating the above for _T_ many layers yields 



**Step 3: Output (2 layers).** We now create first the matrix **y**<sup>_⊤_</sup> **AX** _T_ with one Transformer layer and then the final prediction output with another layer. For the first layer, we choose 



The output of this layer is 



Now for the final step, we construct a Transformer layer with two heads. For the first head, we choose 



22 

For the second head, we choose 



ˆ Notice that the last element of the last row is the desired quantity _y_ = **y**<sup>_⊤_</sup> **AX** _t_ **a** test. This completes the proof. ■ 

## **B Convergence of inexact damped Newton’s method for regularized logistic regression** 

We first review some basics for self-concordant functions in Appendix B.1, and then provide the proof of Theorem 5.4 in Appendix B.2. 

### **B.1 Preliminaries on self-concordant functions** 

We review some results on self-concordant functions that will be useful for the next sections. Most of these theorems can be found in Boyd and Vandenberghe (2004); Nesterov et al. (2018). 

First recall the definition of self-concordant functions. 

**Definition 3.1.** [Self-concordant function; Definition 5.1.1, Nesterov et al. 2018] Let _f_ : R<sup>_d_</sup> _→_ R be a closed convex function that is 3 times continuously differentiable on its domain dom( _f_ ) := _{_ **x** _∈_ R<sup>_d_</sup> _| f_ ( **x** ) _< ∞}_ . For any fixed **x** _,_ **u** _∈_ R<sup>_d_</sup> and _t ∈_ R, define _ϕ_ ( _t_ ; **x** _,_ **u** ) := _f_ ( **x** + _t_ **u** ) as a function of _t_ . Then we say _f_ is _self-concordant_ if there exists a constant _Mf_ such that, for all **x** _∈_ dom( _f_ ) and **u** _∈_ R<sup>_d_</sup> with **x** + _t_ **u** _∈_ dom( _f_ ) for all sufficiently small _t_ , 



We say _f_ is _standard self-concordant_ when _Mf_ = 1. 

23 

For the regularized logistic regression problem that we consider, the objective function is strongly convex, and the theorems provided below use the strong convexity. Nonetheless, these theorems have more general forms, as detailed in Chapter 5 of Nesterov et al. (2018). 

In the sequel, let _f_ be a self-concordant function. 

**Definition B.1** (Dikin ellipsoid) **.** For a function _f_ : R<sup>_d_</sup> _→_ R, consider the following sets for any **x** _∈_ R<sup>_d_</sup> and _r >_ 0: 



where _cl_ ( _·_ ) defines the closure of a set. This set is called the _Dikin ellipsoid_ of the function _f_ at **x** . **Theorem B.2** (Theorem 5.1.5 in Nesterov et al. (2018)) **.** _Let f_ : R<sup>_d_</sup> _→_ R _be a self-concordant function. Then for any_ **x** _∈_ dom( _f_ ) _, it holds that W_ ( **x** ; 1 _/Mf_ ) _⊆_ dom( _f_ ) _._ 

**Theorem B.3** (Theorem 5.1.8 & 5.1.9 in Nesterov et al. (2018)) **.** _Let f_ : R<sup>_d_</sup> _→_ R _be a self-concordant function. Then for any_ **x** _,_ **y** _∈_ dom( _f_ ) _, it holds that_ 



_where ω_ ( _t_ ) = _t −_ ln(1 + _t_ ) _, ω∗_ ( _t_ ) = _−t −_ ln(1 _− t_ ) _._ 

**Lemma B.4** (Lemma 5.1.5 in Nesterov et al. (2018)) **.** _For any t ≥_ 0 _, the functions ω_ ( _t_ ) _, ω∗_ ( _t_ ) _in Theorem B.3 satisfy:_ 



**Theorem B.5** (Theorem 5.1.7, Nesterov et al. 2018) **.** _Let f_ : R<sup>_d_</sup> _→_ R _be a self-concordant function. Then for any_ **x** _∈_ dom( _f_ ) _and any_ **y** _∈W_ ( **x** ; 1 _/Mf_ ) _, it holds that_ 



_where r_ = _∥_ **y** _−_ **x** _∥∇_ 2 _f_ ( **x** ) _._ 

A useful corollary of the above theorem is 

**Corollary B.6** (Corollary 5.1.5, Nesterov et al. 2018) **.** _Let f_ : R<sup>_d_</sup> _→_ R _be a self-concordant function. Then for any_ **x** _∈_ dom( _f_ ) _and any_ **y** _such that r_ = _∥_ **y** _−_ **x** _∥∇_ 2 _f_ ( **x** ) _<_ 1 _/Mf , it holds that_ 



A key quantity in the analysis of Newton method for self-concordant functions is the so-called Newton decrement: 



The following theorem characterizes the sub-optimality gap in terms of the Newton decrement for a strongly convex self-concordant function. 

**Theorem B.7** (Theorem 5.1.13 in Nesterov et al. (2018)) **.** _Let f_ : R<sup>_d_</sup> _→_ R _be a strongly convex self-concordant function with_ **x**<sup>_∗_</sup> = argmin **x** _f_ ( **x** ) _. Suppose λf_ ( **x** ) _<_ 1 _/Mf for some_ **x** _∈_ dom( _f_ ) _, then the following holds:_ 



_where ω∗_ ( _·_ ) _is the function defined in Theorem B.3._ 

24 

**B.2 Convergence analysis of inexact damped Newton’s method** In this section we consider the inexact damped Newton’s method for optimizing the regularized logistic loss: 



where **_ε_** _t_ is an error term. For simplicity, we also define 



ˆ ˆ In other words, we have **x** _t_ +1 = **x** _t_ + **∆** _t_ . 

Recall that the regularized logistic loss defined in Equation (3.5) is self-concordant with _Mf_ = 1 _/_<sup>_√_</sup> _<u>µ</u>_ <u>,</u> as a consequence of Proposition 3.3. 

**Lemma B.8.** _Under Assumption 3.2, let f be the regularized logistic loss defined in Equation_ (3.5) _with regularization parameter µ >_ 0 _. Then the following bounds hold_ 





_Proof of Lemma B.8._ For _|f_ ( **x** ) _|_ , note that for each _i ∈_ [ _n_ ], we have _−yi_ **x**<sup>_⊤_</sup> **a** _i ≤∥_ **x** _∥_ 2 because _yi ∈{−_ 1 _,_ 1 _}_ and _∥_ **a** _i∥_ 2 _≤_ 1. This implies the first bound on _|f_ ( **x** ) _|_ . 

Next, recall the gradient and Hessian of _f_ : 



where each _pi_ := exp( _−yi_ **x**<sup>_⊤_</sup> **a** _i_ ) _/_ (1 + exp( _−yi_ **x**<sup>_⊤_</sup> **a** _i_ )) and **D** := diag( _p_ 1(1 _− p_ 1) _, . . . , pn_ (1 _− pn_ )). By triangle inequality, we have 



where the last inequality follows from Assumption 3.2. Similarly, for the Hessian, we have 



Finally, the lower bound on _∥∇_<sup>2</sup> _f_ ( **x** ) _∥_ op follows from the fact that _n_<sup><u>1</u></sup><sup>**A**</sup><sup>_⊤_</sup><sup>**DA**ispositivedefinite.This</sup> completes the proof. ■ 

**Lemma B.9.** _Under Assumption 3.2, let f be the regularized logistic loss defined in Equation_ (3.5) _with regularization parameter µ >_ 0 _. Then there exists some constant C >_ 0 _depending on µ such that for any_ **_ε_** _∈_ R<sup>_d_</sup> _with ∥_ **_ε_** _∥_ 2 _≤ µ_<sup>2</sup> _, if ∥_ **x** _∥_ 2 _≥ C, then_ 



25 

_Proof of Lemma B.9._ For simplicity, denote 

It follows that 



where we used the fact that _∥_ **_ε_** _∥_ 2 _≤ µ_<sup>2</sup> and triangle inequality. Plugging in the expression for _∇f_ ( **x** ), we have 



where the first inequality follows from Assumption 3.2 and triangle inequality, and the second inequality is due to Lemma B.8. Similarly, we also have 



Combining this with Equations (B.4) and (B.5), we obtain (after rearrangement of the terms) 



Further applying the bounds on _λ_ ( **x** ) from Lemma B.8, with direct computation, we have 



Observe that for sufficiently large _∥_ **x** _∥_ 2, we have _I_ 1 _≈−_ 6 _µ_<sup>2</sup> _∥_ **x** _∥_ 2 and _I_ 2 = _O_ (1). This implies that _∥_ **x**<sup>_′_</sup> _∥_ 2 _≤∥_ **x** _∥_ 2 for sufficiently large _∥_ **x** _∥_ 2, and thus completes the proof. ■ 

26 

**Lemma B.10.** _Under Assumption 3.2, let f be the regularized logistic loss defined in Equation_ (3.5) _with regularization parameter µ >_ 0 _. Consider a sequence of iterates {_ **x** _t}t≥_ 0 _satisfying_ 



_where ∥_ **_ε_** _t∥_ 2 _≤ µ_<sup>2</sup> _for all t ≥_ 0 _. Then there exists a constant C depending on µ such that ∥_ **x** _t∥_ 2 _≤ C for all t ≥_ 0 _._ 

_Proof of Lemma B.10._ First, there exists a constant _C_ 1 (given by Lemma B.9) such that if _∥_ **x** _t∥_ 2 _≥ C_ 1, then _∥_ **x** _t_ +1 _∥_ 2 _≤∥_ **x** _t∥_ 2. Then we define 



Note that by Lemma B.8, _C_ 2 is a constant depending only on _C_ 1 and the regularization parameter _µ_ . Finally, we choose _C_ = max _{∥_ **x** 0 _∥_ 2 _, C_ 1 _, C_ 2 _}_ , and the result follows. ■ 

Our target is to prove convergence of the inexact damped Newton’s method up to some error threshold, which depends on the bound of the error terms _{_ **_ε_** _t}_ . At a high level, the proof strategy is as follows: 

- **Step 1** : Show by induction that if **x** _t_ is feasible, then for sufficiently small error **_ε_** _t_ , so is the next iteration. This leads to some constraint on the error term **_ε_** _t_ . 

- **Step 2** : Show constant decrease of the loss function when _λ_ (ˆ **x** _t_ ) _≥_ 1 _/_ 6. 

- **Step 3** : Show that when _λ_ ( **x** _t_ ) _<_ 1 _/_ 6, we have _λ_ ( **x** _t_ +1) _≤ cλ_<sup>2</sup> _t_<sup>+</sup><sup>_ϵ′_forsomeconstant</sup><sup>_c >_0,sowe</sup> enter the regime of quadratic convergence, which is maintained up to error _ε_ . 

**Theorem 5.4.** _Under Assumption 3.2, for the regularized logistic loss f defined in_ (3.5) _with regularization parameter µ >_ 0 _, consider a sequence of iterates {_ **x** _t}t≥_ 0 _satisfying_ 



_where ∥_ **_ε_** _t∥_ 2 _≤ ϵ for all t ≥_ 0 _. Then there exists constants c, C_ 1 _, C_ 2 _depending only on µ such that for any ϵ ≤ c, it holds that g_ ( **x** _t_ ) _− g_ ( **x**<sup>_∗_</sup> ) _≤ ϵ for all t ≥ C_ 1 + _C_ 2 log log<sup><u>1</u></sup> _ϵ_<sup>_._</sup> 

_Remark_ B.11 _._ In more detail the error obtained is _α_ + _ε_ , where _α ≥_ ~~�~~ _ϵ_ (1 + _µ_ ) _/_ (4 _µ_ ), given that the 1 algorithm performs _T_ = _c_ + log log<sup>where</sup><sup>_c_denotestheinitialstepsofconstant</sup> 1 _/_ 2 + 3 _α_<sup>+ log log 1</sup> _ε_<sup>,</sup> decrease of the function _g_ . By picking _ε_ =<sup>_√_</sup> _<u>ϵ</u>_ we get the desired result. 

_Proof._ First by Lemma B.10, we know that there exists a constant _C_ such that _∥_ **x** _t∥_ 2 _≤ C_ for all _t ≥_ 0. We first derive an upper bound for _δt_ := _∥_ **∆** _t∥∇_ 2 _g_ ( **x** _t_ ). By definition, we have 



27 

It follows from Lemma B.8 that for all _t ≥_ 0, _|_ **_ε_**<sup>_⊤_</sup> _t_<sup>_∇g_(</sup><sup>**x**</sup><sup>_t_)</sup><sup>_| ≤ϵ_(1 +</sup><sup>_µ∥_</sup><sup>**x**</sup><sup>_t∥_2)</sup><sup>_/_(4</sup><sup>_µ_)</sup><sup>_≤ϵ_(1 +</sup><sup>_Cµ_)</sup><sup>_/_(4</sup><sup>_µ_),</sup> and also _|_ **_ε_**<sup>_⊤_</sup> _t_<sup>_∇_2</sup><sup>_g_(</sup><sup>**x**</sup><sup>_t_)</sup><sup>_−_1</sup><sup>**_ε_**</sup><sup>_t| ≤ϵ_2</sup><sup>_/_4.Therefore,wehave</sup> 



Note that by Lemma B.8, we have _λ_ ( **x** _t_ ) _≤_ (1 + _µ∥_ **x** _t∥_ 2) _/_<sup>_√_</sup> _<u>µ ≤</u>_ (1 + _Cµ_ ) _/_ (2<sup>_√_</sup> _<u>µ</u>_ <u>).</u> Then there exists some constant _c_ 1 depending only on _µ_ such that _δt <_ 1 when _ϵ ≤ c_ 1. 

Now, we proceed to show that there is a constant decrease of the loss value up to the point that _λ_ ( **x** ) _≤_ 1 _/_ 6, after which we enter the regime of quadratic convergence. 

**Phase I: Constant decrease of the loss function.** Suppose _λ_ ( **x** _t_ ) _≥_ 1 _/_ 6 (note that if _λ_ ( **x** 0) _<_ 1 _/_ 6, then we can directly proceed to the next phase). Since _g_ is standard self-concordant, it follows from Theorem B.3 that 



where we applied the definition of _λ_ ( **x** _t_ ) and _η_ ( **x** _t_ ) in the last equality. Combining the above two equations, we have 



Recall that _w∗_ ( _x_ ) = _−x −_ log(1 _− x_ ), and we view the right-hand side as a function of _λ_ ( **x** _t_ ) while regarding _c ≡_ **_ε_**<sup>_⊤_</sup> _t_<sup>_∇g_(</sup><sup>**x**</sup><sup>_t_)and</sup><sup>_c′≡_</sup><sup>**_ε_**</sup><sup>_⊤_</sup> _t_<sup>_∇_2</sup><sup>_g_(</sup><sup>**x**</sup><sup>_t_)</sup><sup>_−_1</sup><sup>**_ε_**</sup><sup>_t_asconstants,yielding</sup> 



Since _|c| ≤_ 0 _._ 06 by our assumption on **_ε_** _t_ , it can be verified that _h_ ( _x_ ) is decreasing in _x_ for _x ≥_ 1 _/_ 6, and moreover _h_ (1 _/_ 6) _≤_ 0 _._ 01 by our assumption on **_ε_** _t_ (see Appendix E.1). Therefore, we have _g_ ( **x** _t_ +1) _− g_ ( **x** _t_ ) _≤−_ 0 _._ 01 for all _t_ such that _λ_ ( **x** _t_ ) _≥_ 1 _/_ 6. 

**Phase II: Quadratic convergence.** Now suppose<sup>_√_</sup> _<u>ϵ < λ</u>_ ( **x** _t_ ) _<_ 1 _/_ 6 (again, if _λ_ ( **x** 0) _<_<sup>_√_</sup> _<u>ϵ</u>_ <u>,</u> then we are done). Note that there exists some constant _C_ 1 such that _t ≤ C_ 1 due to the constant decrease of the loss function in the previous phase. Then by Theorem B.7 and Lemma B.4, we have 



where **x**<sup>_∗_</sup> is the global minimizer of _g_ . Thus it suffices to characterize the decrease of _λ_ ( **x** _t_ ). Applying Theorem B.5, we get 



28 

1 Note that _∇g_ ( **x** _t_ +1) = _∇g_ ( **x** _t_ )+�0<sup>_∇_2</sup><sup>_g_(</sup><sup>**x**</sup><sup>_t_+</sup><sup>_s_</sup><sup>**∆**</sup><sup>_t_)</sup><sup>**∆**</sup><sup>_t_d</sup><sup>_s_.Also, we have</sup><sup>_∇g_(</sup><sup>**x**</sup><sup>_t_) =</sup><sup>_−_(1+</sup><sup>_λ_(</sup><sup>**x**</sup><sup>_t_))</sup><sup>_∇_2</sup><sup>_g_(</sup><sup>**x**</sup><sup>_t_)</sup><sup>**∆**</sup><sup>_t_+</sup> (1 + _λ_ ( **x** _t_ )) _∇_<sup>2</sup> _g_ ( **x** _t_ ) **_ε_** _t_ . Combining these two equations, we obtain 



where we introduced the notation **G** _t_ for the integral term. Therefore, by triangle inequality, 



By Corollary B.6, it holds that 



By direct computation, it can be verified that 0 _≤ δt/_ (1 _− δt_ ) _− λ_ ( **x** _t_ ) _≤ δt_ + _λ_ ( **x** _t_ ). This implies that _∥∇_<sup>2</sup> _g_ ( **x** _t_ )<sup>_−_1</sup><sup>_/_2</sup> **G** _t∥_ op _≤_ ( _δt_ + _λ_ ( **x** _t_ )) _∥∇_<sup>2</sup> _g_ ( **x** _t_ )<sup>_−_1</sup><sup>_/_2</sup> _∥_ op and _∥∇_<sup>2</sup> _g_ ( **x** _t_ )<sup>_−_1</sup><sup>_/_2</sup> **G** _t∇_<sup>2</sup> _g_ ( **x** _t_ )<sup>_−_1</sup><sup>_/_2</sup> _∥_ op _≤ δt_ + _λ_ ( **x** _t_ ). Applying these bounds to Equation (B.9), we obtain 



where the last inequality follows from Lemma B.8. Plugging this into Equation (B.8), we obtain 



Note that there exists some constant _c_ 2 depending only on _µ_ such that if _ϵ < c_ 2, then when _λ_ ( **x** _t_ ) _<_ 1 _/_ 6, we have _δt ≤_ 1 _/_ 5 by Equation (B.6). Then, for some constant _C_<sup>_′_</sup> _>_ 0, we further have 



Let _α_ = ~~�~~ _C_<sup>_′_</sup> _ϵ/_ 3 _≤_ 1 _/_ 6. We can rewrite the above inequality as 



Telescoping this inequality, we obtain that for some constant _C_ 2 _>_ 0, _λ_ ( **x** _t_ ) _≤ C_<sup>_′′√_</sup> _<u>ϵ</u>_ when _t ≥ C_ 1 + _C_ 2 log log<sup><u>1</u></sup> _ϵ_<sup>.Combining this with (B.7), we see that for any such</sup><sup>_t_, we have</sup><sup>_g_(</sup><sup>**x**</sup><sup>_t_)</sup><sup>_−g_(</sup><sup>**x**</sup><sup>_∗_)</sup><sup>_≤_</sup><sup><u>3</u></sup><sup>_<u>C</u>_</sup> 5<sup>_′′ϵ_.</sup> Finally, choosing _c_ = min _{c_ 1 _, c_ 2 _}_ completes the proof. ■ 

29 

## **C Transformers for logistic regression** 

In this section, we present the explicit construction of a linear Transformer that can emulate Newton’s method on the regularized logistic loss, and we further provide its error analysis. Recall the regularized logistic loss defined as 



and its gradient and Hessian 



where each _pi_ = exp( _−yi_ **x**<sup>_⊤_</sup> **a** _i_ ) _/_ (1 + exp( _−yi_ **x**<sup>_⊤_</sup> **a** _i_ )) and **D** = diag( _p_ 1(1 _− p_ 1) _, . . . , pn_ (1 _− pn_ )). Letting _g_ ( **x** ) _≡ f_ ( **x** ) _/_ (4 _µ_ ), Newton’s method on _g_ updates as follows: 



where _λg_ ( **x** ) = ~~�~~ _∇g_ ( **x** )<sup>_⊤_</sup> ( _∇_<sup>2</sup> _g_ ( **x** ))<sup>_−_1</sup> _∇g_ ( **x** ) and _λf_ ( **x** ) = ~~�~~ _∇f_ ( **x** )<sup>_⊤_</sup> ( _∇_<sup>2</sup> _f_ ( **x** ))<sup>_−_1</sup> _∇f_ ( **x** ). To emulate the above update, we need to approximate the following components: 

1. The values of the diagonal entries of the matrix **D** , resulting an error vector **u** 1 _∈_ R<sup>_n_</sup> . 

2. The multiplication of **DA** , which incurs an error **U** 2 _∈_ R<sup>_n×d_</sup> . 

3. The inversion of the Hessian matrix, which incurs an error **E** 2 _∈_ R<sup>_d×d_</sup> . 

4. The values of **p** = ( _p_ 1 _, . . . , pn_ )<sup>_⊤_</sup> _∈_ R<sup>_n_</sup> , which incurs an error **u** 3 _∈_ R<sup>_n_</sup> . 

5. The step-size 2<sup>_√_</sup> _<u>µ/</u>_ (2<sup>_√_</sup> _<u>µ</u>_ + _λf_ ( **x** _t_ )), which incurs an error _ϵ_ 4. Notice that to do this, we need to first approximate the value of _λf_ ( **x** _t_ ). 

For simplicity, we drop the subscript _f_ from _λf_ , and we just write _λ_ from now on. The resulting update for one step admits the following form: 



where **E** 1 = **A**<sup>_⊤_</sup> diag( **u** 1) **A** + **A**<sup>_⊤_</sup> **U** 2. 

30 

**Input format.** We consider the following input format: 



Here the first identity matrix will be used to store the updates for calculating the inverse, the second one for the initialization, while the third one for required computation. The initialization **x** 0 is copied _n_ times The second to last line is used to store the parameter _η_ , the step-size. Below we will provide explicit construction of Transformers that can realize the following map from input to output: 



where _K_ denotes the number of Transformer layers. 

**Theorem 5.2.** _Under the setting of Theorem 5.1, there exists a Transformer consisting of linear attention with ReLU layers that can approximately perform damped Newton’s method on the regularized logistic loss as follows_ 



_where_ **_ε_** _is an error term. For any ϵ >_ 0 _, to achieve that ∥_ **_ε_** _∥_ 2 _≤ ϵ, the width of such a Transformer can be bounded by O_ ( _d_<sup><u>(</u></sup> _ϵ_<sup>14+</sup> _µ_<sup>_<u>µ</u>_10)8)</sup><sup>_,anditsdepthcanbeboundedby_11+2</sup><sup>_kwherek≤_2 log</sup><sup>_κf_+log log</sup><sup><u>(1</u></sup> _ϵ_<sup><u>+2</u></sup> _µ_<sup>_<u>µ</u>_2)3</sup><sup>_._</sup> 

For clarity, we split the proof into two parts: the first part is to construct the approximate updates of the Newton’s algorithm, and the second part is devoted to the error analysis. 

_Proof of Theorem 5.2: Construction of the approximate updates._ For ease of presentation, in the following proof we omit the explicit expressions of the weights and describe instead the operations they induce. The constructions are straightforward to be determined, similar to the proofs in Appendix A. Notice that each of the value, key and query weight matrices can always make any row selection by zero-padding and ignoring the rest of the rows of the input matrix. 

31 

**Step 1 - Create d.** We start by creating the diagonal entries of **D** . We choose **W** _V ,_ **W** _K,_ **W** _Q_ such that 



Note that here the value weight matrix **W** _V_ just picks the first row of the identity matrix, zeroes out any other row, and performs a row permutation to place the result in the second to last line. Then we have 



we then use the ReLU network to approximate the values _di_ = _Dii_ = _p_ ( **x**<sup>_⊤_</sup> 0<sup>**a**</sup><sup>_i, yi_)(1</sup><sup>_−p_(</sup><sup>**x**</sup><sup>_⊤_</sup> 0<sup>**a**</sup><sup>_i, yi_))</sup> where _p_ ( _x, y_ ) = exp( _−yx_ ) _/_ (1+ exp( _−yx_ )). We zero out all the rows except for the rows of **y**<sup>_⊤_</sup> _,_ **x**<sup>_⊤_</sup> 0<sup>**A**</sup><sup>_⊤_</sup> and construct the weights of the ReLU network to approximate the function _p_ ( _x, y_ )(1 _− p_ ( _x, y_ )). Notice that this function takes values between [0 _,_ 1] and it domain contains _x ∈_ R and _y ∈{−_ 1 _,_ 1 _}_ in this case. Also note that it is symmetric with respect to the value _y_ . Therefore, it suffices to approximate the function _f_ ( _x_ ) = _e_<sup>_x_</sup> _/_ (1 + _e_<sup>_x_</sup> )<sup>2</sup> , which is increasing for _x ≤_ 0 and decreasing for _x ≥_ 0. We approximate it by splitting [0 _,_ 1] into 1 _/ϵ_ intervals and deal with each of these intervals seperately for _x ≥_ 0 and _x ≤_ 0 using a total number of 4 _/ϵ_ ReLU neurons. This gives rise to 

where **d**<sup>ˆ</sup> = **d** + **u** 1. Notice that 



where _N_ is the width of the ReLU layer. 

**Step 2 - Approximate**<sup>1</sup> In the attention of the second layer, we compute **d**<sup>ˆ</sup> _/n_ by _n_<sup>**D**</sup><sup>_⊙_</sup><sup>**A**</sup><sup>_⊤_</sup><sup>**.**</sup> setting 



32 

and we use one more head to subtract the residual by letting 



Thus, 



The next ReLU layer approximates the multiplication of the diagonal matrix _n_<sup><u>1</u></sup><sup>**D**ˆwiththematrix</sup> **A** , which is the same as creating the vectors _d_<sup>ˆ</sup> 1 **a** 1 _/n, d_<sup>ˆ</sup> 2 **a** 2 _/n, . . . d_<sup>ˆ</sup> _n_ **a** _n/n_ , or equivalently, the matrix _n_ <u>1</u><sup>**D**ˆ</sup><sup>_⊙_</sup><sup>**A**</sup><sup>_⊤_.ThiscanbeimplementedwithoneReLUlayersincetheelementswillbeprocessed</sup> serially and the information needed is on the same column. This approximation incurs an error matrix **U** 2. This yields the output of the second Transformer layer: 



Notice that we can easily subtract the identity matrix since we have another copy of it. Using Proposition A.1 in Bai et al. (2023), we have that _s_ = 2 and each _d_<sup>ˆ</sup> _i ∈_ [ _−_ 0 _._ 1 _,_ 1 _._ 1] thus it requires a total number of _O_ (1 _/_ ( _n_<sup>2</sup> _ϵ_<sup>2</sup> )) ReLUs to achieve _ϵ_ accuracy for each entry. Moreover, 

and thus 



where _N_ is the width of the ReLU layer and we omit logarithmic factors in _O_<sup>�</sup> ( _·_ ). 

33 

**Step 3 - Create the matrix** _n_<sup><u>1</u></sup><sup>**A**</sup><sup>_⊤_</sup><sup>**DA**+</sup><sup>_µ_I</sup><sup>**.**</sup> For this step, we first use the attention layer to implement 



Notice that ( **W** _K_<sup>(1)</sup><sup>**H**2)</sup><sup>_⊤_=</sup> _n_<sup><u>1</u></sup><sup>**DA**ˆ+</sup><sup>**U**</sup> 2<sup>_⊤_.Wealsousetwoextraheadssuchthat</sup> 



Combining these three heads, we get 



For simplicity, we denote **B** := _n_<sup><u>1</u></sup><sup>**A**</sup><sup>_⊤_</sup><sup>**D**ˆ</sup><sup>**A**+</sup><sup>**A**</sup><sup>_⊤_</sup><sup>**U**</sup> 2<sup>_⊤_+</sup><sup>_µ_I</sup><sup>_d_.Wethenuseonemoreattentionlayer</sup> with two head as follows: For the first head, 



The second head is used to subtract the residual, so that the output of this layer is 



34 

We keep track of the errors in the following way: 



where **E** 1 can be controlled as 



since _∥_ **a** _i∥_<sup>2</sup> _≤_ 1 for all _i_ = 1 _, . . . , n_ by Assumption 3.2. 

**Step 4 - Invert the matrix.** Next, we implement Newton’s iteration as in the previous section for _k_ steps and we get 



**Step 5 - Create the** _{pi}_<sup>_n_</sup> _i_ =1<sup>**.**</sup> We repeat Step 1 to get 



35 

We then use the ReLU layer to approximate _{pi}_<sup>_n_</sup> _i_ =1<sup>andwehave</sup> 



The function approximation here is the same as the one in Step 1, and it requires 2 _/ϵ_ ReLUs to achieve _ϵ_ accuracy. Then 



where _N_ is the width of the ReLU layer. 

**Step 6 - Calculate the values** _{yipi/n}_<sup>_n_</sup> _i_ =1<sup>**.**</sup> In the attention layer, we multiply each _pi_ with 1 _/n_ as we did in Step 2, and we have 



We then use the ReLU layer to approximate the inner product between _n_<sup><u>1</u></sup><sup>**p**and</sup><sup>**y**.Todoso,notice</sup> that each _pi ∈_ (0 _,_ 1), and _y_ = _{−_ 1 _,_ 1 _}_ and consider the following sets of ReLUs: 



Suppose _x ∈_ (0 _,_ 1), then if _y_ = 1, the outputs are _o_ 1 = _x_ , _o_ 2 = 0, and for _y_ = _−_ 1, the outputs are _o_ 1 = 0 and _o_ 2 = _−x_ , thus _o_ 1 + _o_ 2 = _xy_ . Consequently, 



36 

**Step 7 - Calculate the gradient.** Next, we want to calculate the quantity _− n_<sup><u>1</u></sup> � _ni_ =1<sup>_yipi_</sup><sup>**a**</sup><sup>_i_+</sup><sup>_µ_</sup><sup>**x**0.</sup> We first set the weight matrices of the attention layer such that 



and we need a third head to add _µ_ **x** 0 by setting 



Thus, we place the result in the second block of the matrix and we have 



where **E** 2 is the error incurred by running Newton’s method for the inversion of the matrix and _n n n_ **b** = _− n_<sup><u>1</u></sup> � _i_ =1<sup>(</sup><sup>_yipi_+</sup><sup>_yiu_3</sup><sup>_i_)</sup><sup>**a**</sup><sup>_i_+</sup><sup>_µ_</sup><sup>**x**0=</sup><sup>_−_</sup> _n_<sup><u>1</u></sup> � _i_ =1<sup>_yipi_</sup><sup>**a**</sup><sup>_i_+</sup><sup>_µ_</sup><sup>**x**0 +</sup><sup>**_ϵ_**3,</sup><sup>**_ϵ_**3=</sup><sup>_−_</sup> _n_<sup><u>1</u></sup> � _i_ =1<sup>_yiu_3</sup><sup>_i_</sup><sup>**a**</sup><sup>_i_.Thus,</sup> 



where is the width of the ReLU layer used to approximate the error **u** 3 (Equation (C.16)). 

**Step 8 - Calculate the stepsize.** We proceed to approximate _λ_<sup>ˆ</sup> ( **x** )<sup>2</sup> = _∇_<sup>_⊤_</sup> _f_ ( **x** )( _∇_<sup>2</sup> _f_ ( **x** ))<sup>_−_1</sup> _∇f_ ( **x** ) = **b**<sup>_⊤_</sup> ( **B**<sup>_−_1</sup> + **E** 2) **b** . We first create the quantity ( **B**<sup>_−_1</sup> + **E** 2) **b** which we also need for the update, store it, and then calculate in the next layer the parameter _λ_ ( **x** )<sup>2</sup> . For the first layer we set 



37 

Then we get 



ˆ We use another Transformer layer to calculate the quantity _λ_ ( **x** _t_ )<sup>2</sup> . We set the attention layer as follows 



where we place the **e**<sup>_⊤_</sup> 1<sup>inthesecondtolastposition.Wealsouseasbeforeanextraheadtoremove</sup> the residual. Then we get 



In the ReLU layer, we approximate the function 2<sup>_~~√~~_</sup> 2 _<u>µ</u>_<sup>_<u>√</u>_</sup> <u>+</u> _<u>µ</u>_<sup>_~~√~~_</sup> _<u>x</u>_<sup>.Noticethatthisfunctiontakevaluesin</sup> (0 _,_ 1]. The first derivative of the function is _−_ 2(2<sup>_~~√~~_</sup> _<u>µ</u>_ 2+<sup>_<u>√</u>_</sup><sup>_~~√~~_</sup> _<u>µx</u>_ <u>)</u><sup>2</sup><sup>_~~√~~_</sup> _<u>x</u>_<sup>_<_0,soitisamonotonicallydecreasing</sup> function. Thus, using the same argument with previous steps we can approximate it up to error _ϵ_ with 2 _/ϵ_ ReLUs. This yields an error _ϵ_ 4 



38 

ˆ where _N_ is the width of the ReLU layer and we can write the stepsize as _η_ ( **x** 0) = 2<sup>_√_</sup> _<u>µ/</u>_ (2<sup>_√_</sup> _<u>µ</u>_ + _λ_ ( **x** 0)) + _ϵ_ 4. Thus, 



where _∗_ denotes inconsequential values. 

**Step 9 - Update** For the last step, we use two more attention layers. We set the weights for the first layer such that 



We also use an extra head to remove the residual by setting 



where the values [ _η_ ˆ( **x** 0) _∗_ ] are placed in the second to last row. Combining these two heads, we get 



where _∗_ are again inconsequential values. Notice that when subtracting the residual with the second head, we correct only the first _d_ columns. 

Finally, we perform the update with one more attention layer: 



39 

Another head is used to restore the matrix in its initial form: 



As a result, we get 



We further use the ReLUs to zero out the second to last row. Note that the inconsequential values wer create when approximating the function 1 _/_ (1 +<sup>_√_</sup> _<u>x</u>_ ), so they are close to one. Thus, we can use the following simple ReLU to zero out this line: 



where we use as _x_ the elements of the second to last row and _y_ , the last row and all the other connections are zero. The final output is 



Thus, for the next step the input is in the correct form and we can repeat the steps above for the next iteration. ■ 

Next, we present the error analysis of the emulated iterates. The target is to show that the updates described in Equation (C.1) can be recasted to the following updates: 



where the error term **_ε_** _t_ is controlled to satisfy the considitions of Theorem 5.4. 

_Proof of Theorem 5.2: Error analysis._ For **x** 0, let _C_ be the constant given by Lemma B.10. From the previous step of weight constructions, we have the following update 



40 

where 





_ϵ_ 4 = Approximation error for the quantity _λ_ . 

where the error terms admit the following bounds 



where _N_ is the width. These bounds were derived in Equations (C.6), (C.10), (C.16) and (C.20) for **u** 1 _,_ **U** 2 **_ϵ_** 3 and _ϵ_ 4 respectively. The error term _ϵ_ 2 is controlled by the number of layers, for _k_ = _c_ + 2 log log(1 _/ϵ_ ) layers we achieve error less than _ϵ_ . 

Applying Corollary E.2, we have ( _∇_<sup>2</sup> _f_ ( **x** 0) + **E** 1)<sup>_−_1</sup> = _∇_<sup>2</sup> _f_ ( **x** 0)<sup>_−_1</sup> + **E**<sup>_′_</sup> 1<sup>where</sup><sup>**E**</sup><sup>_′_</sup> 1<sup>satisfies</sup> _∥_ **E**<sup>_′_</sup> 1<sup>_∥_</sup> 2<sup>_≤∥_</sup><sup>**E**1</sup><sup>_∥_</sup> 2<sup>_/_(</sup><sup>_µ_(</sup><sup>_µ −∥_</sup><sup>**E**1</sup><sup>_∥_</sup> 2<sup>)).Furtherwriting</sup><sup>**E**</sup><sup>_′_</sup> 2<sup>:=</sup><sup>**E**</sup> 1<sup>_′_+</sup><sup>**E**2,thenby(C.24),itholdsthat</sup> 



Now we can rewrite (C.23) as 



Next we analyze the error involved in the quantity _λ_<sup>ˆ2</sup> (ˆ **x** _t_ ). Recall that 



By triangle inequality and the bounds from Lemma B.8, we can bound _ϵ_<sup>_′_</sup> 4<sup>asfollows:</sup> 



Now, as long as it holds that 



we would have 





Then by Appendix E.2 we have that if the condition above holds 



This implies that 



for some _ϵ_ ˜4, such that _|ϵ_ ˜4 _| ≤ ϵ_ 4, since we also account for the approximation error from the ReLUs. Then it follows that 



where the error term **_ε_** satisfies 



Using Equation (C.27) we have that 

Notice now if 



then 



and consequently, _∥_ **_ε_** _∥_ 2 _≤ ϵ_ . 

We will now study how we can bound the error term **E**<sup>_′_</sup> 2<sup>.NoticethatfromEquations(C.13)</sup> and (C.25), the following conditions hold: 



Now, given that 



42 

it is straightforward to verify that indeed **E**<sup>_′_</sup> 2<sup>satisfiesEquation(C.27).Fortheboundon</sup><sup>**E**1,it</sup> suffices to have 



It remains to determine the necessary width and depth for these error bounds to be achieved. We combine the bounds from Equations (C.24), (C.31), (C.33) and (C.34) to get that 



Finally, the error term _∥_ **E** 2 _∥_ 2 is controlled by the depth of the network and scales as _k ∼_ 2 log _κ_ ( _n_<sup><u>1</u></sup><sup>**A**</sup><sup>_⊤_</sup><sup>**D**ˆ</sup><sup>**A**+</sup><sup>_µ_I +</sup><sup>**E**1) + log log (1 +</sup><sup>_Cµ_</sup><sup><u>)3</u></sup> , where the condition number of<sup>1</sup> _ϵµ_<sup>2</sup> _n_<sup>**A**</sup><sup>_⊤_</sup><sup>**D**ˆ</sup><sup>**A**+</sup><sup>_µ_I +</sup><sup>**E**1</sup> should be close to the condition number max **x** _κ_ ( _∇_<sup>2</sup> _f_ ( **x** )) ( For a formal proof of this statement see Appendix E.4). This completes the proof. 



### **C.1 Main Result** 

We are now able to state our main result with explicit bounds. 

**Theorem 5.1.** _For any dimension d, consider the regularized logistic loss defined in_ (3.5) _with regularization parameter µ >_ 0 _, and define κf_ = max **x** _∈_ R _d κ_ ( _∇_<sup>2</sup> _f_ ( **x** )) _. Then for any T >_ 0 _and ϵ >_ 0 _, there exists a linear Transformer that can approximate T iterations of Newton’s method on the regularized logistic loss up to error ϵ per iteration. In particular, the width of such a linear Transformer can be bounded by O_ ( _d_ (1+ _µ_ )<sup>6</sup> _/ϵ_<sup>4</sup> _µ_<sup>8</sup> ) _, and its depth can be bounded by T_ (11+2 _k_ ) _, where k ≤_ 2 log _κf_ + log log<sup><u>(1</u></sup> _ϵ_<sup><u>+2</u></sup> _µ_<sup>_<u>µ</u>_2)3</sup><sup>_.Furthermore,thereisaconstantc >_0</sup><sup>_dependingonµsuchthatifϵ < c,_</sup> ˜ ˜ ˆ _then the output of the Transformer provides a_ **w** _satisfying that ∥_ **w** _−_ **w** _∥_ 2 _≤ O_ ( ~~�~~ _ϵ_ (1 + _µ_ ) _/_ (4 _µ_ )) _, where_ **w** ˆ _is the global minimizer of the loss._ 

_Remark_ C.1 _._ Both _T_ and _k_ are actually have just one extra additive constant term. In the first case, depends on the number of steps needed for _λ_ ( **x** ) to become less than 1 _/_ 6, while for _k_ the constant term is 2 � log _κ_ ( _∇_<sup>2</sup> _f_ ( **x** )) �. 

_Proof._ The proof follows by combining Theorem 5.4 and Theorem 5.2. ■ 

## **D Experiments** 

During training, we utilize the Adam optimizer, with a carefully planned learning rate schedule. The task is learned through curriculum learning Garg et al. (2022), where we begin training with a small problem dimension and gradually increase the difficulty of the tasks. When implementing 

43 



<!-- Start of picture text -->
Num Cond=5, noise=0.0 Num Cond=5, noise=0.1 Num Cond=10, noise=0.0 Num Cond=10, noise=0.1<br>10 1 10 1 10 1<br>10 0<br>10 1 10 0 10 0<br>10 3 10 1 10 2 10 1<br>1010 57 LSA w/ LN Newton(order=2) Newton(order=3) 10 2 LSA w/ LNNewton(orderNewton(order=3)=2) 10 4 LSA w/ LN Newton(order=2) Newton(order=3) 10 2 LSA w/ LNNewton(orderNewton(order=3)=2)<br>Newton(order=4) 10 3 Newton(order=4) Newton(order=4) 10 3 Newton(order=4)<br>1 2 3 4 5 6 1 2 3 4 5 6 1 2 3 4 5 6 1 2 3 4 5 6<br>layers / steps layers / steps layers / steps layers / steps<br><!-- End of picture text -->

Figure 8: _Loss of LSA and different order Newton iteration for linear regression with different input conditions._ 



<!-- Start of picture text -->
Num Cond=5, noise=0.0 Num Cond=10, noise=0.0 Num Cond=20, noise=0.0 Num Cond=50, noise=0.0 Num Cond=100, noise=0.0<br>10 1 10 1 10 0 10 1<br>10 2 10 2 10 3 10 2 10 0<br>10 5 10 5 10 6 10 5 10 2<br>1010 118 LSA w/ LNNewton(order=4)Newton(order=6)Newton(order=8) 1010 118 LSA w/ LN Newton(order=4) Newton(order=6) Newton(order=8) 1010 129 LSA wNewton(order=4)NewtonNewton(order=8)/ LN(order=6) 1010 118 LSA w/ LN Newton(order=4) Newton(order=6) Newton(order=8) 10 4 LSA w/ LNNewtonNewton(order=6)Newton(order=8)(order=4)<br>1 2 3 4 5 6 1 2 3 4 5 6 1 2 3 4 5 6 1 2 3 4 5 6 1 2 3 4 5 6<br>layers / steps layers / steps layers / steps layers / steps layers / steps<br>Num Cond=5, noise=0.1 Num Cond=10, noise=0.1 Num Cond=20, noise=0.1 Num Cond=50, noise=0.1 Num Cond=100, noise=0.1<br>10 1 LSA w/ LN 10 1 LSA w/ LN LSA w/ LN 10 0 LSA w/ LN 10 1 LSA w/ LN<br>10 0 Newton(order=4)Newton(order=6) 10 0 Newton(order=4) Newton(order=6) 10 0 Newton(order=4)Newton(order=6) Newton(order=4)Newton(order=6) 10 0 Newton(order=4) Newton(order=6)<br>10 1 Newton(order=8) 10 1 Newton(order=8) 10 1 Newton(order=8) 10 1 Newton(order=8) 10 1 Newton(order=8)<br>10 2 10 2 10 2 10 2 10 2<br>10 3 10 3 10 3 10 3 10 3<br>1 2 3 4 5 6 1 2 3 4 5 6 1 2 3 4 5 6 1 2 3 4 5 6 1 2 3 4 5 6<br>layers / steps layers / steps layers / steps layers / steps layers / steps<br><!-- End of picture text -->

Figure 9: _Loss of LSA w/ layernorm and different order Newton iteration for linear regression with different input conditions._ 

the Transformer backbone, we adhere to the structure of NanoGPT2<sup>3</sup> , modifying only the causal attention to full attention. 

For both the task of linear and logistic regression, we sample the data points as follows: We sample a random matrix **A** and create its SVD decomposition, i.e., **A** = **USV**<sup>_⊤_</sup> . We then sample the maximum eigenvalue _λmax_ uniformly random in the range of [1 _,_ 100]. We then set the minimum eigenvalue as _λmin_ = _λmax/κ_ , where _κ_ is the condition number of the problem at hand and it is fixed. Finally, we sample the rest of the eigenvalues uniformly random between [ _λmin, λmax_ ] and recreate the matrix **S** , with the new eigenvalues. We then create our covariance matrix as Σ = **USU**<sup>_⊤_</sup> . We use Σ, to sample the data samples **x** _i_ from a multivariate Gaussian with mean 0 and covariance matrix Σ. 

In Figures 8 and 9, we present comprehensive results for the LSA and LSA with layernorm models, trained on linear regression tasks across various condition numbers and noise levels. The trained LSA and LSA with layernorm consistently find higher-order methods when attempting to solve ill-conditioned linear regression problems. 

## **E Auxiliary results** 

We collect here some auxiliary results that are used in the proofs of the main results. 

> 3https://github.com/karpathy/nanoGPT/blob/master/model.py 

44 

### **E.1 Auxiliary result for constant decrease of the logistic loss** 

Let 



where _δ_<sup>˜</sup> = ~~�~~ _x_<sup>2</sup> _/_ (1 + _x_ )<sup>2</sup> _−_ 2 _c/_ (1 + _x_ ) + _c_<sup>_′_</sup> and _c, c_<sup>_′_</sup> are constants with respect to _x_ . Mainly, we see the RHS of the previous inequality as a function of _λ_ and consider **E**<sup>_⊤_</sup> _t_<sup>_∇g_(</sup><sup>**x**ˆ</sup><sup>_t_)</sup><sup>_,_</sup><sup>**E**</sup><sup>_⊤_</sup> _t_<sup>_∇_2</sup><sup>_g_(</sup><sup>**x**ˆ</sup><sup>_t_)</sup><sup>**E**</sup><sup>_t_as</sup> constants. Then we have that 



We use mathematica to plot this function (see Figure 10) and the max value it can attain, when _x ∈_ [1 _/_ 6 _,_ 1] and _|c| ≤_ 0 _._ 06 and we have that the maximum value of _g_<sup>_′_</sup> is approximately _−_ 0 _._ 02. This<sup>_<u>µ</u>_</sup><sup><u>)</u></sup> implies that _h_ is decreasing, since we have that _|c| ≤∥_ **E** _t∥∥∇f_ (ˆ **x** _t_ ) _∥≤_<sup>_ϵ_</sup><sup><u>(1 +</u></sup> _≤_ 0 _._ 06. Thus, we 4 _µ_ have 



where _y_ = **E**<sup>_⊤_</sup> _t_<sup>_∇g_(ˆ</sup><sup>**x**</sup><sup>_t_)and</sup><sup>_z_=</sup><sup>**E**</sup><sup>_⊤_</sup> _t_<sup>_∇_2</sup><sup>_g_(ˆ</sup><sup>**x**</sup><sup>_t_)</sup><sup>**E**</sup><sup>_t_.Noticenowthat</sup> 



Since _ϵ ≤_ min _{_ 0 _._ 01 _,_ 0 _._ 04 _µ/_ (1 + _µ_ ) _}_ . Given these bounds we have that 



We again use mathematica and plot this function for _|y| ≤_ 0 _._ 012, which can be viewed in Figure 10 and we see that we get a constant decrease of at least 0 _._ 01. 

### **E.2 Auxiliary result for controlling the error** 

To control the change that this quantity can evoke, we note that we have approximated the function _g_ ( _x_ ) = 2<sup>_√_</sup> _<u>µ/</u>_ (2<sup>_√_</sup> _<u>µ</u>_ +<sup>_√_</sup> _<u>x</u>_ <u>)</u> by discretizing (0 _,_ 1], so whenever _x ≤ α_<sup>2</sup> , _g_ ( _x_ ) _≥_ 2<sup>_√_</sup> _<u>µ/</u>_ (2<sup>_√_</sup> _<u>µ</u>_ + _α_ ). Thus, if _x ≤_ 4 _µϵ_<sup>2</sup> 4<sup>_/_(1</sup><sup>_−ϵ_4)2wehavethat</sup><sup>_g_(</sup><sup>_x_)</sup><sup>_≥_1</sup><sup>_−ϵ_4,similarlyif</sup><sup>_x≥_4</sup><sup>_µ_(1</sup><sup>_−ϵ_4)2</sup><sup>_/ϵ_2</sup> 4<sup>wehavethat</sup> 

45 





<!-- Start of picture text -->
Constant decrease<br>-0.010 -0.005 -0.0112 0.005 0.010<br>-0.0114<br>-0.0116<br>-0.0118<br>-0.0120<br><!-- End of picture text -->

Figure 10: Left: The derivative of _h_ as a function of both _x_ and _c_ for _x ∈_ [1 _/_ 6 _,_ 1] and _|c| ≤_ 0 _._ 06. Right: We see that the function is decreasing at least _−_ 0 _._ 01 at each step. 

˜ _g_ ( _x_ ) _≤ ϵ_ 4. Now notice that given _x,_ ˜ _x_ such that max _{x, x} ≥_ 4 _µϵ_<sup>2</sup> 4<sup>_/_(1</sup><sup>_−ϵ_4)2(otherwisewehave</sup> already covered the case) we have 



Thus, if it holds that 



then the function is always less than _ϵ_ 4. 

### **E.3 Perturbation bounds** 

**Theorem E.1** (Corollary 2.7, p. 119 in Stewart and guang Sun (1990)) **.** _Let κ_ ( **A** ) = _∥_ **A** _∥_ 2 �� **A** _−_ 1��2 _be the condition number of_ **A** _. If_ **A**<sup>˜</sup> = **A** + **E** _is non-singular, then_ 



_If in addition κ_ ( **A** )<sup>_<u>∥</u>_</sup><sup>**E**</sup><sup>_<u>∥</u>_</sup><sup><u>2</u></sup> _<_ 1 _then ∥_ **A** _∥_ 2 



46 

_and thus_ 



**Corollary E.2.** _Let f be the regularized logistic loss defined in_ (3.5) _with regularization parameter µ >_ 0 _. For the matrix_ **B** = ( _∇_<sup>2</sup> _f_ ( **x** ) + **E** 1) _, it holds that_ 





_Proof._ From Theorem E.1 we have that 



### **E.4 Condition number of perturbed matrix** 

To show that the condition number of the matrix **B** = _∇f_ ( **x** ) + **E** is close to the condition number of _∇f_ ( **x** ) we will use Weyl’s inequality. 

**Lemma E.3** (Weyl’s Corollary 4.9 in Stewart and guang Sun (1990)) **.** _Let λi be the eigenvalues of a matrix_ **A** _with λ_ 1 _≥ . . . ≥ λn, λ_<sup>˜</sup> _i be the eigenvalues of a perturbed matrix_ **A**<sup>˜</sup> = **A** + **E** _and finally let ϵ_ 1 _≥ . . . ϵm be the eigenvalues of_ **E** _. For i_ = 1 _, . . . , n it holds that_ 



Thus, for the matrix **B** we have that the condition number of the eigenvalues of **B** can be bounded as follows 





For _∥_ **E** _∥_ 2 _< µ_ we have that 



And since _∥_ **E** _∥_ 2 is small the two condition numbers are close. 

47 

