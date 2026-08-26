Published as a conference paper at ICLR 2023 

# - - WHAT LEARNING ALGORITHM IS IN CONTEXT LEARN ING? INVESTIGATIONS WITH LINEAR MODELS 

**Ekin Aky¨urek**<sup>1</sup><sup>_,_2</sup><sup>_,a_</sup> _._ **Dale Schuurmans**<sup>1</sup> **Jacob Andreas**<sup>_∗_2</sup> **Tengyu Ma**<sup>_∗_1</sup><sup>_,_3</sup><sup>_,b_</sup> **Denny Zhou**<sup>_∗_1</sup> 

1Google Research 2MIT CSAIL 3 Stanford University _∗_ collaborative advising 

## ABSTRACT 

Neural sequence models, especially transformers, exhibit a remarkable capacity for _in-context learning_ . They can construct new predictors from sequences of labeled examples ( _x, f_ ( _x_ )) presented in the input without further parameter updates. We investigate the hypothesis that transformer-based in-context learners implement standard learning algorithms _implicitly_ , by encoding smaller models in their activations, and updating these implicit models as new examples appear in the context. Using linear regression as a prototypical problem, we offer three sources of evidence for this hypothesis. First, we prove by construction that transformers can implement learning algorithms for linear models based on gradient descent and closed-form ridge regression. Second, we show that trained in-context learners closely match the predictors computed by gradient descent, ridge regression, and exact least-squares regression, transitioning between different predictors as transformer depth and dataset noise vary, and converging to Bayesian estimators for large widths and depths. Third, we present preliminary evidence that in-context learners share algorithmic features with these predictors: learners’ late layers non-linearly encode weight vectors and moment matrices. These results suggest that in-context learning is understandable in algorithmic terms, and that (at least in the linear case) learners may rediscover standard estimation algorithms. 

## 1 INTRODUCTION 

One of the most surprising behaviors observed in large neural sequence models is **in-context learning** (ICL; Brown et al., 2020). When trained appropriately, models can map from sequences of ( _x, f_ ( _x_ )) pairs to accurate predictions _f_ ( _x_<sup>_′_</sup> ) on novel inputs _x_<sup>_′_</sup> . This behavior occurs both in models trained on collections of few-shot learning problems (Chen et al., 2022; Min et al., 2022) and surprisingly in large language models trained on open-domain text (Brown et al., 2020; Zhang et al., 2022; Chowdhery et al., 2022). ICL requires a model to implicitly construct a map from in-context examples to a predictor without any updates to the model’s parameters themselves. How can a neural network with fixed parameters to learn a new function from a new dataset on the fly? 

This paper investigates the hypothesis that some instances of ICL can be understood as _implicit_ implementation of known learning algorithms: in-context learners encode an implicit, contextdependent model in their hidden activations, and train this model on in-context examples in the course of computing these internal activations. As in recent investigations of empirical properties of ICL (Garg et al., 2022; Xie et al., 2022), we study the behavior of transformer-based predictors (Vaswani et al., 2017) on a restricted class of learning problems, here linear regression. Unlike in past work, our goal is not to understand _what_ functions ICL can learn, but _how_ it learns these functions: the specific inductive biases and algorithmic properties of transformer-based ICL. 

In Section 3, we investigate theoretically what learning algorithms transformer decoders can implement. We prove by construction that they require only a modest number of layers and hidden units to train linear models: for _d_ -dimensional regression problems, with _O_ ( _d_ ) hidden size and constant depth, a transformer can implement a single step of gradient descent; and with _O_ ( _d_<sup>2</sup> ) hidden size 

> _a_ Correspondence to akyurek@mit.edu. Ekin is a student at MIT, and began this work while he was intern at Google Research. Code and reference implementations are released at this web page 

> _b_ The work is done when Tengyu Ma works as a visiting researcher at Google Research. 

1 

Published as a conference paper at ICLR 2023 

and constant depth, a transformer can update a ridge regression solution to include a single new observation. Intuitively, _n_ steps of these algorithms can be implemented with _n_ times more layers. 

In Section 4, we investigate empirical properties of trained in-context learners. We begin by constructing linear regression problems in which learner behavior is under-determined by training data (so different valid learning rules will give different predictions on held-out data). We show that model predictions are closely matched by existing predictors (including those studied in Section 3), and that they _transition_ between different predictors as model depth and training set noise vary, behaving like Bayesian predictors at large hidden sizes and depths. Finally, in Section 5, we present preliminary experiments showing how model predictions are computed algorithmically. We show that important intermediate quantities computed by learning algorithms for linear models, including parameter vectors and moment matrices, can be decoded from in-context learners’ hidden activations. 

A complete characterization of which learning algorithms are (or could be) implemented by deep networks has the potential to improve both our theoretical understanding of their capabilities and limitations, and our empirical understanding of how best to train them. This paper offers first steps toward such a characterization: some in-context learning appears to involve familiar algorithms, discovered and implemented by transformers from sequence modeling tasks alone. 

## 2 PRELIMINARIES 

Training a machine learning model involves many decisions, including the choice of model architecture, loss function and learning rule. Since the earliest days of the field, research has sought to understand whether these modeling decisions can be automated using the tools of machine learning itself. Such “meta-learning” approaches typically treat learning as a **bi-level optimization** problem (Schmidhuber et al., 1996; Andrychowicz et al., 2016; Finn et al., 2017): they define “inner” and “outer” models and learning procedures, then train an outer model to set parameters for an inner procedure (e.g. initializer or step size) to maximize inner model performance across tasks. 

Recently, a more flexible family of approaches has gained popularity. In **in-context learning** (ICL), meta-learning is reduced to ordinary supervised learning: a large sequence model (typically implemented as a transformer network) is trained to map from sequences [ _x_ 1 _, f_ ( _x_ 1) _, x_ 2 _, f_ ( _x_ 2) _, ..., xn_ ] to predictions _f_ ( _xn_ ) (Brown et al., 2020; Olsson et al., 2022; Laskin et al., 2022; Kirsch & Schmidhuber, 2021). ICL does not specify an explicit inner learning procedure; instead, this procedure exists only implicitly through the parameters of the sequence model. ICL has shown impressive results on synthetic tasks and naturalistic language and vision problems (Garg et al., 2022; Min et al., 2022; Zhou et al., 2022; Hollmann et al., 2022). 

Past work has characterized _what_ kinds of functions ICL can learn (Garg et al., 2022; Laskin et al., 2022; M¨uller et al., 2021) and the distributional properties of pretraining that can elicit in-context learning (Xie et al., 2021; Chan et al., 2022). But _how_ ICL learns these functions has remained unclear. What learning algorithms (if any) are implementable by deep network models? Which algorithms are actually discovered in the course of training? This paper takes first steps toward answering these questions, focusing on a widely used model architecture (the transformer) and an extremely well-understood class of learning problems (linear regression). 

### 2.1 THE TRANSFORMER ARCHITECTURE 

**Transformers** (Vaswani et al., 2017) are neural network models that map a sequence of input vectors **_x_** = [ _x_ 1 _, . . . , xn_ ] to a sequence of output vectors **_y_** = [ _y_ 1 _, . . . , yn_ ]. Each **layer** in a transformer maps a matrix _H_<sup>(</sup><sup>_l_)</sup> (interpreted as a sequence of vectors) to a sequence _H_<sup>(</sup><sup>_l_+1)</sup> . To do so, a transformer layer processes each column **_h_**<sup>(</sup> _i_<sup>_l_)</sup> of _H_<sup>(</sup><sup>_l_)</sup> in parallel. Here, we are interested in _autoregressive_ (or “decoder-only”) transformer models in which each layer first computes a **self-attention** : 



where each **_b_** is the response of an “attention head” defined by: 



2 

Published as a conference paper at ICLR 2023 

then applies a **feed-forward transformation:** 



Here _σ_ denotes a nonlinearity, e.g. a Gaussian error linear unit (GeLU; Hendrycks & Gimpel, 2016): 



and _λ_ denotes layer normalization (Ba et al., 2016): 



where the expectation and variance are computed across the entries of **_x_** . To map from **_x_** to **_y_** , a transformer applies a sequence of such layers, each with its own parameters. We use _θ_ to denote a model’s full set of parameters (the complete collection of _W_ matrices across layers). The three main factors governing the computational capacity of a transformer are its **depth** (the number of layers), its **hidden size** (the dimension of the vectors **_h_** ), and the number of **heads** (denoted _m_ above). 

### 2.2 TRAINING FOR IN-CONTEXT LEARNING 

We study transformer models directly trained on an ICL objective. (Some past work has found that ICL also “emerges” in models trained on general text datasets; Brown et al., 2020.) To train a transformer _T_ with parameters _θ_ to perform ICL, we first define a class of functions _F_ , a distribution _p_ ( _f_ ) supported on _F_ , a distribution _p_ ( _x_ ) over the domain of functions in _F_ , and a loss function _L_ . We then choose _θ_ to optimize the auto-regressive objective, where the resulting _Tθ_ is an **in-context learner** : 



### 2.3 LINEAR REGRESSION 

Our experiments focus on **linear regression** problems. In these problems, _F_ is the space of linear functions _f_ ( **_x_** ) = **_w_**<sup>_⊤_</sup> **_x_** where **_w_** _,_ **_x_** _∈_ R<sup>_d_</sup> , and the loss function is the squared error _L_ ( _y, y_<sup>_′_</sup> ) = ( _y − y_<sup>_′_</sup> )<sup>2</sup> . Linear regression is a model problem in machine learning and statistical estimation, with diverse algorithmic solutions. It thus offers an ideal test-bed for understanding ICL. Given a dataset with inputs _X_ = [ **_x_** 1 _, . . . ,_ **_x_** _n_ ] and **_y_** = [ _y_ 1 _, . . . , yn_ ], the (regularized) linear regression objective: 





With _λ_ = 0, this objective is known as **ordinary least squares regression** (OLS); with _λ >_ 0, it is known as **ridge regression** (Hoerl & Kennard, 1970). (As discussed further in Section 4, ridge regression can also be assigned a Bayesian interpretation.) To present a linear regression problem to a transformer, we encode both _x_ and _f_ ( _x_ ) as _d_ + 1-dimensional vectors: _x_ ˜ _i_ = [0 _, xi_ ], _y_ ˜ _i_ = [ _yi,_ **0** _d_ ], where **0** _d_ denotes the _d_ -dimensional zero vector. 

## 3 WHAT LEARNING ALGORITHMS CAN A TRANSFORMER IMPLEMENT? 

For a transformer-based model to solve Eq. (9) by implementing an explicit learning algorithm, that learning algorithm must be implementable via Eq. (1) and Eq. (4) with some fixed choice of transformer parameters _θ_ . In this section, we prove constructively that such parameterizations exist, giving concrete implementations of two standard learning algorithms. These proofs yield upper bounds on how many layers and hidden units suffice to implement (though not necessarily learn) each algorithm. Proofs are given in Appendices A and B. 

3 

Published as a conference paper at ICLR 2023 

### 3.1 PRELIMINARIES 

It will be useful to first establish a few computational primitives with simple transformer implementations. Consider the following four functions from R<sup>_H×T_</sup> _→_ R<sup>_H×T_</sup> : 

**mov** ( _H_ ; _s, t, i, j, i_<sup>_′_</sup> _, j_<sup>_′_</sup> ): selects the entries of the _s_<sup>th</sup> column of _H_ between rows _i_ and _j_ , and copies them into the _t_<sup>th</sup> column ( _t ≥ s_ ) of _H_ between rows _i_<sup>_′_</sup> and _j_<sup>_′_</sup> , yielding the matrix: 



**mul** ( _H_ ; _a, b, c,_ ( _i, j_ ) _,_ ( _i_<sup>_′_</sup> _, j_<sup>_′_</sup> ) _,_ ( _i_<sup>_′′_</sup> _, j_<sup>_′′_</sup> )): in _each_ column **_h_** of _H_ , interprets the entries between _i_ and _j_ as an _a × b_ matrix _A_ 1, and the entries between _i_<sup>_′_</sup> and _j_<sup>_′_</sup> as a _b × c_ matrix _A_ 2, multiplies these matrices together, and stores the result between rows _i_<sup>_′′_</sup> and _j_<sup>_′′_</sup> , yielding a matrix in which each column has the form [ **_h_** : _i′′−_ 1 _, A_ 1 _A_ 2 _,_ **_h_** _j′′_ :]<sup>_⊤_</sup> . 

**div** ( _H_ ; ( _i, j_ ) _, i_<sup>_′_</sup> _,_ ( _i_<sup>_′′_</sup> _, j_<sup>_′′_</sup> )): in each column **_h_** of _H_ , divides the entries between _i_ and _j_ by the absolute value of the entry at _i_<sup>_′_</sup> , and stores the result between rows _i_<sup>_′′_</sup> and _j_<sup>_′′_</sup> , yielding a matrix in which every column has the form [ **_h_** : _i_<sup>_′′_</sup> _−_ 1 _,_ **_h_** _i_ : _j/|_ **_h_** _i_<sup>_′_</sup> _|,_ **_h_** _j_<sup>_′′_</sup> :]<sup>_⊤_</sup> . 

**aff** ( _H_ ; ( _i, j_ ) _,_ ( _i_<sup>_′_</sup> _, j_<sup>_′_</sup> ) _,_ ( _i_<sup>_′′_</sup> _, j_<sup>_′′_</sup> ) _, W_ 1 _, W_ 2 _, b_ ): in each column **_h_** of _H_ , applies an affine transformation to the entries between _i_ and _j_ and _i_<sup>_′_</sup> and _j_<sup>_′_</sup> , then stores the result between rows _i_<sup>_′′_</sup> and _j_<sup>_′′_</sup> , yielding a matrix in which every column has the form [ **_h_** : _i′′−_ 1 _, W_ 1 **_h_** _i_ : _j_ + _W_ 2 **_h_** _i′_ : _j′_ + _b,_ **_h_** _j′′_ :]<sup>_⊤_</sup> . 

**Lemma 1.** _Each of_ mov _,_ mul _,_ div _and_ aff _can be implemented by a single transformer decoder layer: in Eq._ (1) _and Eq._ (4) _, there exist matrices W_<sup>_Q_</sup> _, W_<sup>_K_</sup> _, W_<sup>_V_</sup> _, W_<sup>_F_</sup> _, W_ 1 _and W_ 2 _such that, given a matrix H as input, the layer’s output has the form of the corresponding function output above._<sup>1</sup> 

With these operations, we can implement building blocks of two important learning algorithms. 

### 3.2 GRADIENT DESCENT 

Rather than directly solving linear regression problems by evaluating Eq. (10), a standard approach to learning exploits a generic loss minimization framework, and optimizes the ridge-regression objective in Eq. (9) via gradient descent on parameters **_w_** . This involves repeatedly computing updates: 



for different examples ( **_x_** _i, yi_ ), and finally predicting **_w_**<sup>_′⊤_</sup> **_x_** _n_ on a new input _xn_ . A step of this gradient descent procedure can be implemented by a transformer: 

**Theorem 1.** _A transformer can compute Eq._ (11) _(i.e. the prediction resulting from single step of gradient descent on an in-context example) with constant number of layers and O_ ( _d_ ) _hidden space, where d is the problem dimension of the input x. Specifically, there exist transformer parameters θ such that, given an input matrix of the form:_ 



_the transformer’s output matrix H_<sup>(</sup><sup>_L_)</sup> _contains an entry equal to_ **_w_**<sup>_′⊤_</sup> **_x_** _n (Eq._ (11) _) at the column index where xn is input._ 

### 3.3 CLOSED-FORM REGRESSION 

Another way to solve the linear regression problem is to directly compute the closed-form solution Eq. (10). This is somewhat challenging computationally, as it requires inverting the regularized covariance matrix _X_<sup>_⊤_</sup> _X_ + _λI_ . However, one can exploit the Sherman–Morrison formula (Sherman & Morrison, 1950) to reduce the inverse to a sequence of rank-one updates performed example-byexample. For any invertible square _A_ , 



> 1We omit the trivial size preconditions, e.g. **mul:** ( _i − j_ = _a ∗ b, i′ − j′_ = _b ∗ c, i′′ − j′′_ = _c ∗ d_ ). 

4 

Published as a conference paper at ICLR 2023 

Because the covariance matrix _X_<sup>_⊤_</sup> _X_ in Eq. (10) can be expressed as a sum of rank-one terms each involving a single training example **_x_** _i_ , this can be used to construct an iterative algorithm for computing the closed-form ridge-regression solution. 

**Theorem 2.** _A transformer can predict according to a single Sherman–Morrison update:_ 



_with constant layers and O_ ( _d_<sup>2</sup> ) _hidden space. More precisely, there exists a set of transformer parameters θ such that, given an input matrix of the form in Eq._ (12) _, the transformer’s output matrix H_<sup>(</sup><sup>_L_)</sup> _contains an entry equal to_ **_w_**<sup>_′⊤_</sup> _xn (Eq._ (14) _) at the column index where xn is input._ 

**Discussion.** There are various existing universality results for transformers (Yun et al., 2020; Wei et al., 2021), and for neural networks more generally (Hornik et al., 1989). These generally require very high precision, very deep models, or the use of an external “tape”, none of which appear to be important for in-context learning in the real world. Results in this section establish sharper upper bounds on the necessary capacity required to implement learning algorithms specifically, bringing theory closer to the range where it can explain existing empirical findings. Different theoretical constructions, in the context of meta-learning, have been shown for linear self-attention models (Schlag et al., 2021), or for other neural architectures such as recurrent neural networks (Kirsch & Schmidhuber, 2021). We emphasize that Theorem 1 and Theorem 2 each show the implementation of a single step of an iterative algorithm; these results can be straightforwardly generalized to the multi-step case by “stacking” groups of transformer layers. As described next, it is these iterative algorithms that capture the behavior of real learners. 

## 4 WHAT COMPUTATION DOES AN IN-CONTEXT LEARNER PERFORM? 

The previous section showed that the building blocks for two specific procedures—gradient descent on the least-squares objective and closed-form computation of its minimizer—are implementable by transformer networks. These constructions show that, in principle, fixed transformer parameterizations are expressive enough to simulate these learning algorithms. When trained on real datasets, however, in-context learners might implement other learning algorithms. In this section, we investigate the empirical properties of trained in-context learners in terms of their _behavior_ . In the framework of Marr’s (2010) “levels of analysis”, we aim to explain ICL at the **computational** level by identifying the _kind of algorithms_ to regression problems that transformer-based ICL implements. 

### 4.1 BEHAVIORAL METRICS 

Determining which learning algorithms best characterize ICL predictions requires first quantifying the degree to which two predictors agree. We use two metrics to do so: 

**Squared prediction difference.** Given any learning algorithm _A_ that maps from a set of input– output pairs _D_ = [ **_x_** 1 _, y_ 1 _, . . . ,_ **_x_** _n, yn_ ] to a predictor _f_ ( **_x_** ) = _A_ ( _D_ )( **_x_** ), we define the squared prediction difference (SPD): 



where _D_ is sampled as in Eq. (8). SPD measures agreement at the _output_ level, regardless of the algorithm used to compute this output. 

**Implicit linear weight difference.** When ground-truth predictors all belong to a known, parametric function class (as with the linear functions here), we may also investigate the extent to which different learners agree on the parameters themselves. Given an algorithm _A_ , we sample a _context_ dataset _D_ as above, and an additional collection of unlabeled test inputs _DX ′_ = _{_ **_x_**<sup>_′_</sup> _i_<sup>_}_.We</sup> then compute _A_ ’s prediction on each _x_<sup>_′_</sup> _i_<sup>,yielding a</sup><sup>_predictor-specific datasetDA_=</sup><sup>_{_(</sup><sup>**_x_**</sup><sup>_′_</sup> _i_<sup>_,_ˆ</sup><sup>_yi_)</sup><sup>_}_=</sup> �� **_x_** _i, A_ ( _D_ )( **_x_**<sup>_′_</sup> _i_<sup>)</sup> �� encapsulating the function learned by _A_ . Next we compute the implied parameters: 



5 

Published as a conference paper at ICLR 2023 



<!-- Start of picture text -->
0.8 0.35 (OLS, ICL)<br>0.30 (Ridge(0.1), ICL)(GD(0.01), ICL)<br>(SGD(0.01), ICL)<br>0.6 0.25 (GD(0.02), ICL)<br>(SGD(0.03), ICL)<br>(OLS, Y)<br>0.20 (KNN(3, weighted), ICL)<br>0.4 0.15 (KNN(3, uniform), ICL)((Ridge(0.1), Y)OLS, Y)<br>(ICL, Y)<br>0.10<br>0.2<br>0.05<br>0.0 0.00<br>1 2 4 6 8 10 12 14 3 5 7 9 11 13<br>#exemplars #exemplars<br>(a) Predictor–ICL fit w.r.t. prediction differences. (b) Predictor–ICL fit w.r.t implicit weights.<br>)2<br>1, )2<br>d SPD(1/ ILWD(1,<br><!-- End of picture text -->

Figure 1: **Fit between ICL and standard learning algorithms:** We plot (dimension normalized) SPD and ILWD values between textbook algorithms and ICL on noiseless linear regression with _d_ = 8. GD( _α_ ) denotes one step of batch gradient descent and SGD( _α_ ) denotes one pass of stochastic gradient descent with learning rate _α_ . Ridge( _λ_ ) denotes Ridge regression with regularization parameter _λ_ . Under both evaluations, in-context learners _agree_ closely with ordinary least squares, and are significantly less well approximated by other solutions to the linear regression problem. 

We can then quantify agreement between two predictors _A_ 1 and _A_ 2 by computing the distance between their implied weights in expectation over datasets: 



When the predictors are not linear, ILWD measures the difference between the closest linear predictors (in the sense of Eq. (16)) to each algorithm. For algorithms that have linear hypothesis space (e.g. Ridge regression), we will use the actual value of **_w_** ˆ _A_ instead of the estimated value. 

### 4.2 EXPERIMENTAL SETUP 

We train a Transformer decoder autoregresively on the objective in Eq. (8). For all experiments, we perform a hyperparameter search over depth _L ∈{_ 1 _,_ 2 _,_ 4 _,_ 8 _,_ 12 _,_ 16 _}_ , hidden size _W ∈{_ 16 _,_ 32 _,_ 64 _,_ 256 _,_ 512 _,_ 1024 _}_ and heads _M ∈{_ 1 _,_ 2 _,_ 4 _,_ 8 _}_ . Other hyper-parameters are noted in Appendix D. For our main experiments, we found that _L_ = 16 _, H_ = 512 _, M_ = 4 minimized loss on a validation set. We follow the training guidelines in Garg et al. (2022), and trained models for 500 _,_ 000 iterations, with each in-context dataset consisting of 40 ( **_x_** _, y_ ) pairs. For the main experiments we generate data according to _p_ ( **_w_** ) = _N_ (0 _, I_ ) and _p_ ( **_x_** ) = _N_ (0 _, I_ ). 

### 4.3 RESULTS 

**ICL matches ordinary least squares predictions on noiseless datasets.** We begin by comparing a ( _L_ = 16 _, H_ = 512 _, M_ = 4) transformer against a variety of reference predictors: 

- _k_ **-nearest neighbors** : In the _uniform_ variant, models predict _y_ ˆ _i_ =<sup><u>1</u></sup> 3 � _j_<sup>_yj_, where</sup><sup>_j_is the</sup> top-3 closest data point to _xi_ where _j < i_ . In the _weighted_ variant, a weighted average _y_ ˆ _i ∝_ 3<sup><u>1</u></sup> � _j_<sup>_|xi −xj|−_2</sup><sup>_yj_is calculated, normalized by the total weights of the</sup><sup>_yj_s.</sup> 

- **One-pass stochastic gradient descent** : _y_ ˆ _i_ = **_w_** _i_<sup>_⊤xi_where</sup><sup>**_w_**</sup><sup>_i_isobtainedbystochastic</sup> gradient descent on the previous examples with batch-size equals to 1: **_w_** _i_ = **_w_** _i−_ 1 _−_ 2 _α_ ( _x_<sup>_⊤_</sup> _i−_ 1<sup>**_w_**</sup> _i_<sup>_⊤_</sup> _−_ 1<sup>_xi−_1</sup><sup>_−x⊤_</sup> _i−_ 1<sup>_yi−_1 +</sup><sup>_λwi−_1).</sup> 

- **One-step batch gradient descent** : _y_ ˆ _i_ = **_w_** _i_<sup>_⊤xi_where</sup><sup>**_w_**</sup><sup>_i_is obtained by one of step gradi-</sup> ent descent on the batch of previous examples: **_w_** _i_ = **_w_** 0 _−_ 2 _α_ ( _X_<sup>_⊤_</sup> **_w_**<sup>_⊤_</sup> _X − X_<sup>_⊤_</sup> _Y_ + _λw_ 0). 

- **Ridge regression** : We compute _y_ ˆ _i_ = **_w_**<sup>_′⊤_</sup> _xi_ where **_w_**<sup>_′⊤_</sup> = ( _X_<sup>_⊤_</sup> _X_ + _λI_ )<sup>_−_1</sup> _X_<sup>_⊤_</sup> _Y_ . We denote the case of _λ_ = 0 as **OLS** . 

The agreement between the transformer-based ICL and these predictors is shown in Fig. 1. As can be seen, there are clear differences in fit to predictors: for almost any number of examples, normalized SPD and ILWD are small between the transformer and OLS predictor (with squared error less than 0.01), while other predictors (especially nearest neighbors) agree considerably less well. 

6 

Published as a conference paper at ICLR 2023 



<!-- Start of picture text -->
(Lstsq, ICL) 1.25e-05 1.34e-04 3.96e-04 1.51e-03 4.13e-03<br>(Ridge(1/16), ICL) 1.10e-04 3.29e-05 1.12e-04 8.24e-04 2.92e-03<br>(Ridge(1/9), ICL) 3.49e-04 9.65e-05 3.86e-05 4.50e-04 2.15e-03<br>(Ridge(1/4), ICL) 1.69e-03 8.64e-04 4.39e-04 3.30e-05 6.81e-04<br>(Ridge(4/9), ICL) 4.83e-03 3.09e-03 2.21e-03 7.52e-04 6.10e-05<br>(0.0/1.0) 2 =0 (0.25/1.0) 2 =1/16 (0.5/1.5)2/ 2 2=1/9 (0.5/1.0) 2 =1/4 (0.5/0.75) 2 =4/9<br>)(AA21,<br><!-- End of picture text -->

Figure 2: **ICL under uncertainty:** With problem dimension _d_ = 8, and for different values of prior variance _τ_<sup>2</sup> and data noise _σ_<sup>2</sup> , we display (dimension-normalized) MSPD values for each predictor pair, where MSPD is the average SPD value over underdetermined region of the linear problem. Brightness is proportional with MSPD1<sup>.ICLmostcloselyfollowstheminimum-Bayes-riskRidge</sup> regression output for all<sup>_<u>σ</u>_</sup> _τ_<sup>22values.</sup> 

When the number of examples is less than the input dimension _d_ = 8, the linear regression problem is under-determined, in the sense that multiple linear models can exactly fit the in-context training dataset. In these cases, OLS regression selects the _minimum-norm_ weight vector, and (as shown in Fig. 1), the in-context learner’s predictions are reliably consistent with this minimum-norm predictor. Why, when presented with an ambiguous dataset, should ICL behave like this particular predictor? One possibility is that, because the weights used to generate the training data are sampled from a Gaussian centered at zero, ICL learns to output the _minimum Bayes risk_ solution when predicting under uncertainty (see M¨uller et al. (2021)). Building on these initial findings, our next set of experiments investigates whether ICL is behaviorally equivalent to Bayesian inference more generally. 

**ICL matches the minimum Bayes risk predictor on noisy datasets.** To more closely examine the behavior of ICL algorithms under uncertainty, we add noise to the training data: now we present the in-context dataset as a sequence: [ **_x_** 1 _, f_ ( **_x_** 1) + _ϵ_ 1 _, . . . ,_ **_x_** _n, f_ ( **_x_** _n_ ) + _ϵn_ ] where each _ϵi ∼N_ (0 _, σ_<sup>2</sup> ). Recall that ground-truth weight vectors are themselves sampled from a Gaussian distribution; together, this choice of prior and noise mean that the learner cannot be certain about the target function with any number of examples. 

Standard Bayesian statistics gives that the optimal predictor for minimizing the loss in Eq. (8) is: 



This is because, conditioned on _x_ and _D_ , the scalar ˆ _y_ ( _x, D_ ) := E[ _y|x, D_ ] is the minimizer of the loss E[( _y − y_ ˆ)<sup>2</sup> _|x, D_ ], and thus the estimator ˆ _y_ is the minimzier of E[( _y − y_ ˆ)<sup>2</sup> ] = E _x,D_ [E[( _y − y_ ˆ)<sup>2</sup> _|x, D_ ]]. For linear regression with Gaussian priors and Gaussian noise, the Bayesian estimator in Eq. (18) has a closed-form expression: 



Note that this predictor has the same form as the ridge predictor from Section 2.3, with the regularization parameter set to<sup>_<u>σ</u>_</sup> _τ_<sup>22.Inthepresenceofnoisylabels,doesICLmatchthisBayesian</sup> predictor? We explore this by varying both the dataset noise _σ_<sup>2</sup> and the prior variance _τ_<sup>2</sup> (sampling **_w_** _∼N_ (0 _, τ_<sup>2</sup> )). For these experiments, the SPD values between the in-context learner and various regularized linear models is shown in Fig. 2. As predicted, as variance increases, the value of the ridge parameter that best explains ICL behavior also increases. For all values of _σ_<sup>2</sup> _, τ_<sup>2</sup> , the ridge parameter that gives the best fit to the transformer behavior is also the one that minimizes Bayes risk. These experiments clarify the finding above, showing that ICL in this setting behaviorally matches minimum-Bayes-risk predictor. We also note that when the noise level _σ →_ 0<sup>+</sup> , the Bayes predictor converges to the ordinary least square predictor. Therefore, the results on noiseless datasets studied in the beginning paragraph of this subsection can be viewed as corroborating the finding here in the setting with _σ →_ 0<sup>+</sup> . 

**ICL exhibits algorithmic phase transitions as model depth increases.** The two experiments above evaluated extremely high-capacity models in which (given findings in Section 3) computational constraints are not likely to play a role in the choice of algorithm implemented by ICL. But 

7 

Published as a conference paper at ICLR 2023 



<!-- Start of picture text -->
0.4<br>(OLS, ICL)<br>(Ridge(0.1), ICL)<br>0.3 (Ridge(0.5), ICL)<br>(KNN(3, weighted), ICL)<br>(KNN(3, uniform), ICL)<br>0.2 (SGD(0.03), ICL)<br>(GD(0.02), ICL)<br>0.1 (SGD(0.01), ICL)<br>(GD(0.01), ICL)<br>0.0<br>2 0 2 1 2 2 2 3 2 4 2 5 2 7 2 9<br>L (depth) H (hidden size)<br>(a) Linear regression problem with  d  = 8<br>0.4<br>(OLS, ICL)<br>(Ridge(0.1), ICL)<br>0.3 (Ridge(0.5), ICL)<br>(KNN(3, weighted), ICL)<br>(KNN(3, uniform), ICL)<br>0.2 (SGD(0.03), ICL)<br>(GD(0.02), ICL)<br>0.1 (SGD(0.01), ICL)<br>(GD(0.01), ICL)<br>0.0<br>2 0 2 1 2 2 2 3 2 4 2 5 2 7 2 9<br>L (depth) H (hidden size)<br>(b) Linear regression problem with  d  = 16<br>)2<br>1,<br>MSPD(<br>)2<br>1,<br>MSPD(<br><!-- End of picture text -->

Figure 3: **Computational constraints on ICL:** We show SPD averaged over underdetermined region of the linear regression problem. In-context learners behaviorally match ordinary least squares predictors if there is enough number of layers and hidden sizes. When varying model depth (left background), algorithmic “phases” emerge: models transition between being closer to gradient descent, (red background), ridge regression (green background), and OLS regression (blue). 

what about smaller models—does the size of an in-context learner play a role in determining the learning algorithm it implements? To answer this question, we run two final behavioral experiments: one in which we vary the _hidden size_ (while optimizing the depth and number of heads as in Section 4.2), then vary the _depth_ of the transformer (while optimizing the hidden size and number of heads). These experiments are conducted without dataset noise. 

Results are shown in Fig. 3. When we vary the depth, learners occupy three distinct regimes: very shallow models (1L) are best approximated by a single step of gradient descent (though not wellapproximated in an absolute sense). Slightly deeper models (2L-4L) are best approximated by ridge regression, while the deepest (+8L) models match OLS as observed in Fig. 3. Similar phase shift occurs when we vary hidden size in a 16D problem. Interestingly, we can read hidden size requirements to be close to ridge-regression-like solutions as _H ≥_ 16 and _H ≥_ 32 for 8D and 16D problems respectively, suggesting that ICL discovers more efficient ways to use available hidden state than our theoretical constructions requiring _O_ ( _d_<sup>2</sup> ). Together, these results show that ICL _does not necessarily_ involve minimum-risk prediction. However, even in models too computationally constrained to perform Bayesian inference, alternative interpretable computations can emerge. 

## 5 DOES ICL ENCODE MEANINGFUL INTERMEDIATE QUANTITIES? 

Section 4 showed that transformers are a good fit to standard learning algorithms (including those constructed in Section 3) at the _computational_ level. But these experiments leave open the question of how these computations are implemented at the **algorithmic** level. How do transformers arrive at the solutions in Section 4, and what quantities do they compute along the way? Research on extracting precise algorithmic descriptions of learned models is still in its infancy (Cammarata et al., 2020; Mu & Andreas, 2020). However, we can gain insight into ICL by inspecting learners’ intermediate states: asking _what_ information is encoded in these states, and _where_ . 

To do so, we identify two intermediate quantities that we expect to be computed by gradient descent and ridge-regression variants: the **moment vector** _X_<sup>_⊤_</sup> _Y_ and the (min-norm) least-square estimated **weight vector** **_w_** OLS, each calculated after feeding _n_ exemplars. We take a trained in-context learner, freeze its weights, then train an auxiliary **probing** model (Alain & Bengio, 2016) to attempt to 

8 

Published as a conference paper at ICLR 2023 



<!-- Start of picture text -->
X T Y probe error vs layer Layer-8's attention heatmap over time<br>0.06<br>Linear ProbeMLP Probe 0.75<br>0.04<br>0.50<br>0.02 0.25<br>0.00<br>0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75<br>layer position<br>wOLS probe errors vs layer Layer-12's attention heatmap over time<br>0.04<br>Linear ProbeMLP Probe 0.75<br>0.50<br>0.02<br>0.25<br>0.00<br>0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75<br>layer position<br>0<br>MSE 12<br>Target Index<br>3<br>0<br>MSE 12<br>Target Index<br>3<br><!-- End of picture text -->

Figure 4: **Probing results on** _d_ = 4 **problem:** Both moments _X_<sup>_⊤_</sup> _Y_ (top) and least-square solution **_w_** OLS (middle) are recoverable from learner representations. Plots in the left column show the accuracy of the probe for each target in different model layers. Dashed lines show the best probe accuracies obtained on a _control task_ featuring a fixed weight vector _w_ = **1** . Plots in the right column show the attention heatmap for the best layer’s probe, with the number of input examples on the x-axis. The value of the target after _n_ exemplars is decoded primarily from the representation of _yn_ , or, after _n_ = _d_ examplars, uniformly from _yn≥_ 4. 

recover the target quantities from the learner’s hidden representations. Specifically, the probe model takes hidden states at a layer _H_<sup>(</sup><sup>_l_)</sup> as input, then outputs the prediction for target variable. We define a probe with **position-attention** that computes (Appendix E): 



We train this probe to minimize the squared error between the predictions and targetsˆ **_v_** : _L_ ( **_v_** _,_ ˆ **_v_** ) = _|_ **_v_** _−_ **_v_** _|_<sup>2</sup> . The probe performs two functions simultaneously: its prediction error on held-out representations determines the _extent to which the target quantity is encoded_ , while its attention mask, **_α_** identifies the _location in which the target quantity is encoded_ . For the FF term, we can insert the function approximator of our choosing; by changing this term we can determine the _manner in which the target quantity is encoded_ —e.g. if FF is a linear model and the probe achieves low error, then we may infer that the target is encoded linearly. 

For each target, we train a separate probe for the value of the target on _each prefix of the dataset_ : i.e. one probe to decode the value of **_w_** computed from a single training example, a second probe to decode the value for two examples, etc. Results are shown in Fig. 4. For both targets, a 2- layer MLP probe outperforms a linear probe, meaning that these targets are encoded nonlinearly (unlike the constructions in Section 3). However, probing also reveals similarities. Both targets are decoded accurately deep in the network (but inaccurately in the input layer, indicating that probe success is non-trivial.) Probes attend to the correct timestamps when decoding them. As in both constructions, _X_<sup>_⊤_</sup> _Y_ appears to be computed first, becoming predictable by the probe relatively early in the computation (layer 7); while **_w_** becomes predictable later (around layer 12). For comparison, we additionally report results on a _control task_ in which the transformer predicts _y_ s generated with a fixed weight vector _w_ = **1** (so no ICL is required). Probes applied to these models perform significantly worse at recovering moment matrices (see Appendix E for details). 

## 6 CONCLUSION 

We have presented a set of experiments characterizing the computations underlying in-context learning of linear functions in transformer sequence models. We showed that these models are capable in theory of implementing multiple linear regression algorithms, that they empirically implement this range of algorithms (transitioning between algorithms depending on model capacity and dataset noise), and finally that they can be probed for intermediate quantities computed by these algorithms. 

While our experiments have focused on the linear case, they can be extended to many learning problems over richer function classes—e.g. to a network whose initial layers perform a non-linear 

9 

Published as a conference paper at ICLR 2023 

feature computation. Even more generally, the experimental methodology here could be applied to larger-scale examples of ICL, especially language models, to determine whether their behaviors are also described by interpretable learning algorithms. While much work remains to be done, our results offer initial evidence that the apparently mysterious phenomenon of in-context learning can be understood with the standard ML toolkit, and that the solutions to learning problems discovered by machine learning researchers may be discovered by gradient descent as well. 

## ACKNOWLEDGEMENTS 

We thank Evan Hernandez, Andrew Drozdov, Ed Chi for their feedback on the early drafts of this paper. At MIT, Ekin Aky¨urek is supported by an MIT-Amazon ScienceHub fellowship and by the MIT-IBM Watson AI Lab. 

## REFERENCES 

- Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. _ArXiv preprint_ , abs/1610.01644, 2016. URL https://arxiv.org/abs/1610.01644. 

- Marcin Andrychowicz, Misha Denil, Sergio Gomez Colmenarejo, Matthew W. Hoffman, David Pfau, Tom Schaul, and Nando de Freitas. Learning to learn by gradient descent by gradient descent. In Daniel D. Lee, Masashi Sugiyama, Ulrike von Luxburg, Isabelle Guyon, and Roman Garnett (eds.), _Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain_ , pp. 3981–3989, 2016. URL https://proceedings.neurips.cc/paper/2016/hash/ fb87582825f9d28a8d42c5e5e5e8b23d-Abstract.html. 

- Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. _ArXiv preprint_ , abs/1607.06450, 2016. URL https://arxiv.org/abs/1607.06450. 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html. 

- Nick Cammarata, Shan Carter, Gabriel Goh, Chris Olah, Michael Petrov, Ludwig Schubert, Chelsea Voss, Ben Egan, and Swee Kiat Lim. Thread: circuits. _Distill_ , 5(3):e24, 2020. 

- Stephanie CY Chan, Adam Santoro, Andrew K Lampinen, Jane X Wang, Aaditya Singh, Pierre H Richemond, Jay McClelland, and Felix Hill. Data distributional properties drive emergent fewshot learning in transformers. _ArXiv preprint_ , abs/2205.05055, 2022. URL https://arxiv.org/ abs/2205.05055. 

- Chi Chen, Maosong Sun, and Yang Liu. Mask-align: Self-supervised neural word alignment. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pp. 4781–4791, Online, 2021. Association for Computational Linguistics. doi: 10.18653/v1/ 2021.acl-long.369. URL https://aclanthology.org/2021.acl-long.369. 

- Yanda Chen, Ruiqi Zhong, Sheng Zha, George Karypis, and He He. Meta-learning via language model in-context tuning. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 719–730, Dublin, Ireland, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.53. URL https://aclanthology.org/2022.acl-long.53. 

10 

Published as a conference paper at ICLR 2023 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. _ArXiv preprint_ , abs/2204.02311, 2022. URL https://arxiv.org/abs/2204.02311. 

- Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Doina Precup and Yee Whye Teh (eds.), _Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017_ , volume 70 of _Proceedings of Machine Learning Research_ , pp. 1126–1135. PMLR, 2017. URL http://proceedings.mlr.press/v70/finn17a.html. 

- Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. _ArXiv_ , abs/2208.01066, 2022. 

- Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). _ArXiv preprint_ , abs/1606.08415, 2016. URL https://arxiv.org/abs/1606.08415. 

- Arthur E Hoerl and Robert W Kennard. Ridge regression: Biased estimation for nonorthogonal problems. _Technometrics_ , 12(1):55–67, 1970. 

- Noah Hollmann, Samuel M¨uller, Katharina Eggensperger, and Frank Hutter. Tabpfn: A transformer that solves small tabular classification problems in a second, 2022. URL https://arxiv.org/ abs/2207.01848. 

- Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. _Neural networks_ , 2(5):359–366, 1989. 

- Louis Kirsch and J¨urgen Schmidhuber. Meta learning backpropagation and improving it. _Advances in Neural Information Processing Systems_ , 34:14122–14134, 2021. 

- Michael Laskin, Luyu Wang, Junhyuk Oh, Emilio Parisotto, Stephen Spencer, Richie Steigerwald, DJ Strouse, Steven Hansen, Angelos Filos, Ethan Brooks, et al. In-context reinforcement learning with algorithm distillation. _ArXiv preprint_ , abs/2210.14215, 2022. URL https://arxiv.org/ abs/2210.14215. 

- David Marr. _Vision: A computational investigation into the human representation and processing of visual information_ . MIT press, 2010. 

- Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. MetaICL: Learning to learn in context. In _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 2791–2809, Seattle, United States, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. naacl-main.201. URL https://aclanthology.org/2022.naacl-main.201. 

- Jesse Mu and Jacob Andreas. Compositional explanations of neurons. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6- 12, 2020, virtual_ , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ c74956ffb38ba48ed6ce977af6727275-Abstract.html. 

- Samuel M¨uller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do bayesian inference. _arXiv preprint arXiv:2112.10510_ , 2021. 

- Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, T. J. Henighan, Benjamin Mann, Amanda Askell, Yushi Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, John Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom B. Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Christopher Olah. In-context learning and induction heads. 2022. 

- Imanol Schlag, Kazuki Irie, and J¨urgen Schmidhuber. Linear transformers are secretly fast weight programmers. In Marina Meila and Tong Zhang (eds.), _Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event_ , volume 139 of _Proceedings of Machine Learning Research_ , pp. 9355–9366. PMLR, 2021. URL http://proceedings.mlr.press/v139/schlag21a.html. 

11 

Published as a conference paper at ICLR 2023 

Juergen Schmidhuber, Jieyu Zhao, and Marco A Wiering. Simple principles of metalearning. 1996. 

- Jack Sherman and Winifred J Morrison. Adjustment of an inverse matrix corresponding to a change in one element of a given matrix. _The Annals of Mathematical Statistics_ , 21(1):124–127, 1950. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (eds.), _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA_ , pp. 5998–6008, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html. 

- Colin Wei, Yining Chen, and Tengyu Ma. Statistically meaningful approximation: a case study on approximating turing machines with transformers. _ArXiv preprint_ , abs/2107.13163, 2021. URL https://arxiv.org/abs/2107.13163. 

- Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. _ArXiv preprint_ , abs/2111.02080, 2021. URL https: //arxiv.org/abs/2111.02080. 

- Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. _ArXiv_ , abs/2111.02080, 2022. 

- Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J. Reddi, and Sanjiv Kumar. Are transformers universal approximators of sequence-to-sequence functions? In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net, 2020. URL https://openreview.net/forum?id=ByxRM0Ntvr. 

- Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. Opt: Open pre-trained transformer language models, 2022. 

- Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for visionlanguage models. _International Journal of Computer Vision_ , 130(9):2337–2348, 2022. 

12 

Published as a conference paper at ICLR 2023 

## A THEOREM 1 

The operations for 1-step SGD with single exemplar can be expressed as following chain (please see proofs for the Transformer implementation of these operations (Lemma 1) in Appendix C): 

- mov(; 1 _,_ 0 _,_ (1 _,_ 1 + _d_ ) _,_ (1 _,_ 1 + _d_ )) (move **_x_** ) 

- • aff(; (1 _,_ 1 + _d_ ) _,_ () _,_ (1 + _d,_ 2 + _d_ ) _, W_ 1 = **_w_** ) ( **_w_**<sup>_⊤_</sup> **_x_** ) • aff(; (1 + _d,_ 2 + _d_ ) _,_ (0 _,_ 1) _,_ (2 + _d,_ 3 + _d_ ) _, W_ 1 = _I, W_ 2 = _−I_ ) ( **_w_**<sup>_⊤_</sup> **_x_** _− y_ ) • mul(; _d,_ 1 _,_ 1 _,_ (1 _,_ 1 + _d_ ) _,_ (2 + _d,_ 3 + _d_ ) _,_ (3 + _d,_ 3 + 2 _d_ )) ( **_x_** ( **_w_**<sup>_⊤_</sup> **_x_** _− y_ )) • aff(; () _,_ () _,_ (3 + 2 _d,_ 3 + 3 _d_ ) _, b_ = **_w_** _,_ ) (write **_w_** ) • aff(; (3+ _d,_ 3+2 _d_ ) _,_ (3+2 _d,_ 3+3 _d_ ) _,_ (3+3 _d,_ 3+4 _d_ ) _, W_ 1 = _I, W_ 2 = _−λ_ ) ( **_x_** ( **_w_**<sup>_⊤_</sup> **_x_** _− y_ ) _− λ_ **_w_** ) • aff(; (3 + 2 _d,_ 3 + 3 _d_ ) _,_ (3 + 3 _d,_ 3 + 4 _d_ ) _,_ (3 + 2 _d,_ 3 + 3 _d_ ) _, W_ 1 = _I, W_ 2 = _−_ 2 _α,_ ) ( **_w_**<sup>_′_</sup> ) • mov(; 2 _,_ 1 _,_ (3 + 2 _d,_ 3 + 3 _d_ ) _,_ (3 + 2 _d,_ 3 + 3 _d_ )) (move **_w_**<sup>_′_</sup> ) • mul(; 1 _, d,_ 1 _,_ (3 + 2 _d,_ 3 + 3 _d_ ) _,_ (1 _,_ 1 + _d_ ) _,_ (3 + 3 _d,_ 4 + 3 _d_ )) ( **_w_**<sup>_′⊤_</sup> _x_ 2) 

This will map: 



We can verify the chain of operator step-by-step. In each step, we show only the non-zero rows. 

- mov(; 1 _,_ 0 _,_ (1 _,_ 1 + _d_ ) _,_ (1 _,_ 1 + _d_ )) 



















13 

Published as a conference paper at ICLR 2023 



- aff(; (3+ _d,_ 3+2 _d_ ) _,_ (3+2 _d,_ 3+3 _d_ ) _,_ (3+3 _d,_ 3+4 _d_ ) _, W_ 1 = _I, W_ 2 = _−_ 2 _λ_ ) ( **_x_** ( **_w_**<sup>_⊤_</sup> **_x_** _−y_ ) _−_ 2 _λ_ **_w_** ) 



- aff(; (3 + 2 _d,_ 3 + 3 _d_ ) _,_ (3 + 3 _d,_ 3 + 4 _d_ ) _,_ (3 + 2 _d,_ 3 + 3 _d_ ) _, W_ 1 = _I, W_ 2 = _−_ 2 _α,_ ) 





- mul(; 1 _, d,_ 1 _,_ (3 + 2 _d,_ 3 + 3 _d_ ) _,_ (1 _,_ 1 + _d_ ) _,_ (3 + 3 _d,_ 4 + 3 _d_ )) 

( **_w_**<sup>_′⊤_</sup> _x_ 2) 

14 

Published as a conference paper at ICLR 2023 



We obtain the updated prediction in the last hidden unit of the third time-step. 

**Generalizing to multiple steps of SGD.** Since **_w_**<sup>_′_</sup> is written in the hidden states, we may repeat ˆ this iteration to obtain _y_ 3 = **_w_**<sup>_′′⊤_</sup> **_x_** 3 where **_w_**<sup>_′′_</sup> is the one step update **_w_**<sup>_′_</sup> _−_ 2 _α_ ( **_x_** 2 **_w_**<sup>_′⊤_</sup> **_x_** 2 _−_ **_y_** 2 **_x_** 2 + _λ_ **_w_** , requiring a total of _O_ ( _n_ ) layers for a single pass through the dataset where _n_ is the number of examplers. 

As an empirical demonstration of this procedure, the accompanying code release contains a reference implementation of SGD defined in terms of the base primitive provided in an anymous links https://icl1.s3.us-east-2.amazonaws.com/theory/ _{_ primitives,sgd,ridge _}_ .py (to preserve the anonymity we did not provide the library dependencies). This implementation predicts ˆ _yn_ = **_w_** _n_<sup>_⊤_</sup><sup>**_x_**</sup><sup>_n_,where</sup><sup>**_w_**</sup><sup>_n_istheweightvectorresultingfrom</sup><sup>_n −_1consecutiveSGDupdateson</sup> previous examples. It can be verified there that the procedure requires _O_ ( _n_ + _d_ ) hidden space. Note that, it is not _O_ ( _nd_ ) because we can reuse spaces for the next iteration for the intermediate variables, an example of this performed in ( **_w_**<sup>_′_</sup> ) step above highlighted with blue color. 

## B THEOREM 2 

We provide a similar construction to Theorem 1 (please see proofs for the Transformer implementation of these operations in Appendix C, specifically for div see Appendix C.6) 

|• mov(; 1_,_0_,_(1_,_1 +_d_)_,_(1_,_1 +_d_))<br>(move**_x_**1)|
|---|
|• mul(;_d,_1_,_1_,_(1_,_1 +_d_)_,_(0_,_1)_,_(1 +_d,_1 + 2_d_))<br>(**_x_**1_y_)|
|• aff(; ()_,_()_,_(1 + 2_d,_1 + 2_d_+_d_<sup>2</sup>)_, b_= <sup>_I_</sup><br>_λ_<sup>)</sup><br>(_A_<sup>_−_1</sup><br>0<br>= <sup>_I_</sup><br>_λ_<sup>)</sup>|
|• mul(;_d, d,_1_,_(1 + 2_d,_1 + 2_d_+_d_<sup>2</sup>)_,_(1_,_1 +_d_)_,_(1 + 2_d_+_d_<sup>2</sup>_,_1 + 3_d_+_d_<sup>2</sup>))<br>(_A_<sup>_−_1</sup><br>0 <sup>**_u_** =</sup> <sup>_I_</sup><br>_λ_<sup>**_x_**1)</sup>|
|• mul(; 1_, d, d,_(1_,_1 +_d_)_,_(1 + 2_d,_1 + 2_d_+_d_<sup>2</sup>)_,_(1 + 3_d_+_d_<sup>2</sup>_,_1 + 4_d_+_d_<sup>2</sup>))<br>(**_v_**_A_<sup>_−_1</sup><br>0<br>=**_x_**<sup>_⊤_</sup><br>1<br>_I_<br>_λ_<sup>)</sup>|
|• mul(;_d,_1_, d,_(1 + 2_d_+_d_<sup>2</sup>_,_1 + 3_d_+_d_<sup>2</sup>)_,_(1 + 3_d_+_d_<sup>2</sup>_,_1 + 4_d_+_d_<sup>2</sup>)_,_(1 + 4_d_+_d_<sup>2</sup>_,_1 + 4_d_+ 2_d_<sup>2</sup>))<br>(_A_<sup>_−_1</sup><br>0 <sup>**_uv_**</sup><sup>_A−_1</sup><br>0<br>= <sup>_I_</sup><br>_λ_<sup>**_x_**1</sup><sup>**_x_**</sup><sup>_⊤_</sup><br>1<br>_I_<br>_λ_<sup>)</sup>|
|• mul(; 1_, d,_1_,_(1+3_d_+_d_<sup>2</sup>_,_1+4_d_+_d_<sup>2</sup>)_,_(1_,_1+_d_)_,_(1+4_d_+2_d_<sup>2</sup>_,_2+4_d_+2_d_<sup>2</sup>))(**_v_**<sup>_⊤_</sup>_A_<sup>_−_1</sup><br>0 <sup>**_u_** =</sup><sup>**_x_**</sup><sup>_⊤_</sup><br>1<br>_I_<br>_λ_<sup>**_x_**1)</sup>|
|• aff(; (1+4_d_+2_d_<sup>2</sup>_,_2+4_d_+2_d_<sup>2</sup>)_,_()_,_(1+4_d_+2_d_<sup>2</sup>_,_2+4_d_+2_d_<sup>2</sup>)_, W_1 = 1_, b_= 1_,_)(1+**_v_**<sup>_⊤_</sup>_A_<sup>_−_1</sup><br>0 <sup>**_u_** =</sup><br>1 +**_x_**<sup>_⊤_</sup><br>1<br>_I_<br>_λ_<sup>**_x_**1)</sup>|
|• div(; (1 + 4_d_+_d_<sup>2</sup>_,_1 + 4_d_+ 2_d_<sup>2</sup>)_,_1 + 4_d_+ 2_d_<sup>2</sup>_,_(2 + 4_d_+ 2_d_<sup>2</sup>_,_2 + 4_d_+ 3_d_<sup>2</sup>))<br>(right term)|
|• aff(; (1+2_d,_1+2_d_+_d_<sup>2</sup>)_,_(2+4_d_+2_d_<sup>2</sup>_,_2+4_d_+3_d_<sup>2</sup>)_,_(1+2_d,_1+2_d_+_d_<sup>2</sup>)_, W_1 =_I, W_2 =_−I_)<br>(_A_<sup>_−_1</sup><br>1 <sup>)</sup>|
|• mul(;_d, d,_1_,_(1 + 2_d,_1 + 2_d_+_d_<sup>2</sup>)_,_(1_,_1 +_d_)_,_(2 + 4_d_+ 3_d_<sup>2</sup>_,_2 + 5_d_+ 3_d_<sup>2</sup>))<br>(_A_<sup>_−_1</sup><br>1 <sup>**_x_**1)</sup>|
|• mul(;_d,_1_,_1_,_(2 + 4_d_+ 3_d_<sup>2</sup>_,_2 + 5_d_+ 3_d_<sup>2</sup>)_,_(0_,_1)_,_(2 + 4_d_+ 3_d_<sup>2</sup>_,_2 + 5_d_+ 3_d_<sup>2</sup>))<br>(_A_<sup>_−_1</sup><br>1 <sup>**_x_**1</sup><sup>_y_1)</sup>|



15 

Published as a conference paper at ICLR 2023 

• mov(; 2 _,_ 1 _,_ (2 + 4 _d_ + 3 _d_<sup>2</sup> _,_ 2 + 5 _d_ + 3 _d_<sup>2</sup> ) _,_ (2 + 4 _d_ + 3 _d_<sup>2</sup> _,_ 2 + 5 _d_ + 3 _d_<sup>2</sup> )) (move **_w_**<sup>_′_</sup> ) • mul(; _d,_ 1 _,_ 1(2 + 4 _d_ + 3 _d_<sup>2</sup> _,_ 2 + 5 _d_ + 3 _d_<sup>2</sup> ) _,_ (1 _,_ 1 + _d_ ) _,_ (2 + 5 _d_ + 3 _d_<sup>2</sup> _,_ 3 + 5 _d_ + 3 _d_<sup>2</sup> )) ( **_w_**<sup>_′⊤_</sup> _x_ 2) 

Note that, in contrast to Appendix A, we need _O_ ( _d_<sup>2</sup> ) space to implement matrix multiplications. Therefore over-all required hidden size is _O_ ( _d_<sup>2</sup> ) 

As Theorem 1, generalizing it to multiple iterations will at least require _O_ ( _n_ ) layers, as we repeat the process for the next examplar. 

## C LEMMA 1 

All of the operators mentioned in this lemma share a common computational structure, and can in fact be implemented as special cases of a “base primitive” we call RAW (for Read-Arithmetic-Write). This operator may also be useful for future work aimed at implementing other algorithms. 

The structure of our proof of Lemma 1 is as follows: 

   1. Motivation of the base primitive RAW. 

   2. RAW. 

   3. Definition of dot, aff, mov in terms of RAW. 

   4. Implementation of RAW in terms of transformer parameters. 

   5. Brief discussion of how to parallelize RAW, making it possible to implement mul. 

   6. Seperate proof for div by utilizing layer norm. 

- C.1 RAW OPERATOR: INTUITION 

At a high level, all of the primitives in Lemma 1 involve a similar sequence of operations: 

**1) Operators read some hidden units from the current or previous timestep:** dot and aff read from two subsets of indices in the current hidden state **_h_** _t_<sup>2</sup> , while mov reads from a previous hidden state **_h_** _t′_ . This selection is straightforwardly implemented using the attention component of a transformer layer. 

We may notate this reading operation as follows: 



Here **r** denotes a list of indice to read from, and _K_ denotes a map from _current timesteps_ to _target timesteps_ . For convenience, we use Numpy-like notation to denote indexing into a vector with another vector: 

**Definition C.1** (Bracket) **.** **_x_** [ _._ ] is Python index notation where the resulting vector, **_x_**<sup>_′_</sup> = **_x_** [r]: 



The first step of our proof below shows that the attention output **_a_**<sup>(</sup><sup>_l_)</sup> can compute the expression above. 

**2) Operators perform element-wise arithmetic between the quantity read in step 1 and another set of entries from the current timestep:** This step takes different forms for aff and mul (mov ignores values at the current timestep altogether). 

> 2For notational convenience, we will use **_h_** to refer to sequence of hidden states (instead of _H_ in Eq. (1).), **_h_** _t′_ will be the hidden state at time step _t_<sup>_′_</sup> 

16 

Published as a conference paper at ICLR 2023 



The second step of the proof below computes these operations inside the MLP component of the transformer layer. 

**3) Operators reduce, then write to the current hidden state** Once the underlying element-wise operation calculated, the operator needs to **write** these values to the some indices in current hidden state, defined by a list of indices **w** . Writing might be preceded by a **reduction** state (e.g. for computing dot products), which can be expressed generically as a linear operator _Wo_ . The final form of the computation is thus: 



Here, _←_ means that the other indices _i ∈/ w_ are copied from _h_<sup>_l−_1</sup> . 

### C.2 RAW OPERATOR DEFINITION 

We denote this “master operator” as RAW: 

**Definition C.2.** RAW( **_h_** ; _⃝⋆,_ s _,_ r _,_ w _, Wo, Wa, W, K_ ) is a function R<sup>_H×T_</sup> _�→_ R<sup>_H×T_</sup> . It is parameterized by an elementwise operator _⃝⋆ ∈{_ + _, ⊙}_ , three matrices _W ∈_ R<sup>_d×|_s</sup><sup>_|_</sup> , _Wa ∈_ R<sup>_d×|_r</sup><sup>_|_</sup> _, Wo ∈_ R<sup>_|w|×d_</sup> , three index sets s, r, and w, and a timestep map _K_ : Z<sup>+</sup> _�→_ (Z<sup>+</sup><sup>_∗_</sup> ). Given an input matrix **_h_** , it outputs a matrix with entries: 



We additionally require that _j ∈ K_ ( _i_ ) = _⇒ j < i_ (since self-attention is causal.) 

(For simplicity, we did not include a possible bias term in linear projections _Wo_ , _Wa_ , _W_ , we can always assume the accompanying bias parameters **_b_** 0 _,_ **_b_** _a,_ **_b_** when needed) 

### C.3 REDUCING LEMMA 1 OPERATORS TO RAW OPERATOR 

Given this operator, we can define each primitive in Lemma 1 using a single RAW operator, except the **mul** and **div** . Instead of the matrix multiplication operator **mul** , we will first show the dot product **dot** (a special case of **mul** ), then later in the proof, we will argue that we can parallelize these dot products in Appendix C.5 to obtain **mul** . We will show how to implement div separately in Appendix C.6. 

17 

Published as a conference paper at ICLR 2023 

**Lemma 2.** _We can define_ **mov** _,_ **aff** _operator, and the dot product case of_ **mul** _in Lemma 1 by using a single RAW operator_ 

**dot** ( **_h_** ; ( _i, j_ ) _,_ ( _i_<sup>_′_</sup> _, j_<sup>_′_</sup> ) _,_ ( _i_<sup>_′′_</sup> _, j_<sup>_′′_</sup> )) = **mul** ( **_h_** ; 1 _, |i − j|,_ 1 _,_ ( _i, j_ ) _,_ ( _i_<sup>_′_</sup> _, j_<sup>_′_</sup> ) _,_ ( _i_<sup>_′′_</sup> _, i_<sup>_′′_</sup> + 1)) 



**aff** ( **_h_** ; ( _i, j_ ) _,_ ( _i_<sup>_′_</sup> _, j_<sup>_′_</sup> ) _,_ ( _i_<sup>_′′_</sup> _, j_<sup>_′′_</sup> ) _, W_ 1 _, W_ 2 _, b_ ) 



**mov** ( **_h_** ; _s, t,_ ( _i, j_ ) _,_ ( _i_<sup>_′_</sup> _, j_<sup>_′_</sup> )) 



_Proof._ Follows immediately by substituting parameters into Eq. (26). 

### C.4 IMPLEMENTING RAW 

It remains only to show: 

**Lemma 3.** _A single transformer layer can implement the_ RAW _operator: there exist settings of transformer parameters such that, given an arbitrary hidden matrix_ **_h_** _as input, the transformer computes_ **_h_**<sup>_′_</sup> _(Eq._ (26) _) as output._ 

Our proof proceeds in stages. We begin by providing specifying initial embedding and positional embedding layers, constructing inputs to the main transformer layer with necessary positional information and scratch space. Next, we prove three useful procedures for bypassing (or exploiting) non-linearities in the feed-forward component of the transformer. Finally, we provide values for remaining parameters, showing that we can implement the _Elementwise_ and _Reduction_ steps described above. 

### C.4.1 EMBEDDING LAYERS 

**Embedding Layer for Initialization:** Rather than inserting the input matrix **_h_** directly into the transformer layer, we assume (as is standard) the existence of a linear **embedding** layer. We can set this layer to pad the input, providing extra scratch space that will be used by later steps of our implementation. 

We define the embedding matrix _We_ as: 



Then, the embedded inputs will be 



**Position Embeddings for Attention Manipulation:** Implementing RAW ultimately requires controlling which position attends to which position in each layer. For example, we may wish to have layers in which each position attends to the previous position only, or in which even positions attends to other even positions. We can utilize position embeddings, **_p_** _i_ , to control attention weights. In a standard transformer, the position embedding matrix is a constant matrix that is added to the inputs of the transformer after embedding layer (before the first layer), so the actual input to to the transformer is: 



We will use these position embeddings to encode the timestep map K. To do this, we will use 2 _p_ units per layer ( _p_ will be defined momentarily). _p_ units will be used to encode attention _keys_ **k** _i_ , and the other _p_ will be used to encode _queries_ **q** _i_ . 

We define the position embedding matrix as follows: 



18 

Published as a conference paper at ICLR 2023 

With _K_ encoded in positional embeddings, the transformer matrices _WQ_ and _WK_ are easy to define: they just need to retrieve the corresponding embedding values: 



The constructions used in this paper rely on two specific timestep maps _K_ , each of which can be implemented compactly in terms of **_k_** and **_q_** : 

**Case 1: Attend to previous token.** This can be constructed by setting: 



where _N_ is a sufficiently large number. In this case, the output of the attention mechanism will be: 



**Case 2: Attend to a single token.** For simpler patterns, such as attention to a specific token: 



only 1 hidden unit is required. We set: 

from which it can be verified (using the same procedure as in Case 1) that the desired attention pattern is produced. 

**Intricacy: How can K be empty?** We can also cause _K_ ( _i_ ) to attend to an empty set by assuming the softmax has extra (“imaginary”) timestep obtained by prepending a 0 to attention vector pot-hoc (Chen et al., 2021). 

Cumulatively, the parameter matrices defined in this subsection implement the _Read with Attention_ component of the RAW operator. 

### C.4.2 HANDLING & UTILIZING NONLINEARITIES 

The mul operator requires elementwise multiplication of quantities stored in hidden states. While transformers are often thought of as only straightforwardly implementing _affine_ transformations on hidden vectors, their nonlinearities in fact allow elementwise multiplication to a high degree of approximation. We begin by observing the following property of the GeLU activation function in the MLP layers of the Transformer network: 

**Lemma 4.** _The GeLU nonlinearity can be used to perform multiplication: specifically,_ 



19 

Published as a conference paper at ICLR 2023 

_Proof._ A standard implementation of the GeLU nonlinearity is defined as follows: 



Thus 







For small _x_ and _y_ , the third-order term vanishes. By scaling inputs down by a constant before the GeLU layer, and scaling them up afterwards, models may use the GeLU operator to perform elementwise multiplication. 

We can generalize this proof to other smooth functions as we discussed further in [TODO REF]. Previous work also shows, in practice, Transformers with ReLU activation utilize non-linearities to get the multiplication in other settings. 

When implementing the aff operator, we have the opposite problem: we would like the output of addition to be transmitted _without_ nonlinearities to the output of the transformer layer. Fortunately, for large inputs, the GeLU nonlinearity is very close to linear; to bypass it it suffices to add to inputs a large _N_ : 

**Lemma 5.** _The GeLU nonlinearity can be bypassed: specifically,_ 



_Proof._ 



For all verions of the RAW operator, it is additionally necessary to bypass the LayerNorm operation. The following formula will be helpful for this: 

**Lemma 6.** _Let N be a large number and λ the LayerNorm function. Then the following approxi-_ 



_Proof._ 









20 

Published as a conference paper at ICLR 2023 

By adding a large number _N_ to two padding locations and sum the part of the hidden state that we are interested to pass through LayerNorm, we make _x_ to the output of LayerNorm pass through. This addition can be done in the transformer’s feed-forward computation (with parameter _W_<sup>_F_</sup> ) prior 

to layer norm. This multiplication of ~~�~~ _L_ <u>2</u><sup>_N_can be done in first layer of MLP back, then linear layer</sup> can output/use **_x_** . For convenience, we will henceforth omit the LayerNorm operation when it is not needed. 

We may make each of these operations as precise as desired (or allowed by system precision). With them defined, we are ready to specify the final components of the RAW operator. 

### C.4.3 PARAMETERIZING RAW 

We want to show a layer of Transformer defined in above, hence parameterized by _θ_ = _{W_ f _, W_ 1 _, W_ 2 _,_ ( _W_<sup>_Q_</sup> _, W_<sup>_K_</sup> _, W_<sup>_v_</sup> ) _m}_ , can well-approximate the RAW operator defined in Eq. (25). We will provide step by step constructions and define the parameters in _θ_ . Begin by recalling the transformer layer definition: 





**Attention Output** We will only use _m_ = 2 attention heads for this construction. We show in Eq. (32) that we can control attentions to uniformly attend with a pattern by setting _key_ and _query_ matrices. Assume that the first head parameters _W_ 1<sup>_Q, W_</sup> 1<sup>_K_</sup> have been set in the described way to obtain the pattern function **K** . Now we will set remaining attention parameters _W_ 1<sup>_V, W_</sup> 2<sup>_Q, W_</sup> 2<sup>_K, W_</sup> 2<sup>_V_andshowhatwecanmake</sup> the **_a_** _i_ + **_h_**<sup>(</sup> _i_<sup>_l_)</sup> term in Eq. (4) to contain the corresponding term in Eq. (25), in some unused indices t such that: 



Then the term on the RAW operator can be obtained by the first head’s output. In order to achieve that, we will set _Wa_ as a part of actual attention value network such that _W_ 1<sup>_V_issparsematrix0</sup> everywhere expect: 



Now our first heads stores the right term in Eq. (53) in the indicies _t_ . However, when we add the residual term **_h_**<sup>(</sup> _i_<sup>_l_), this will change.To remove the residual term, we will use another head to output</sup> **_h_**<sup>(</sup> _i_<sup>_l_), by setting</sup><sup>_W Q_</sup> 2<sup>_, W_</sup> 2<sup>_K_such that</sup><sup>_K_(</sup><sup>_i_) =</sup><sup>_i_, and</sup><sup>_W V_</sup> 2<sup>(similar to Eq. (34)):</sup> 



Then, _W_<sup>f</sup> _∈_ R<sup>_H×_2</sup><sup>_H_</sup> is zero otherwise: 





We already defined ( _W_<sup>_Q_</sup> _, W_<sup>_K_</sup> _, W_<sup>_V_</sup> )1 _,_ 2 and _W_<sup>f</sup> and obtained the first term in the Eq. (25) in ( **_a_** _i_ + **_h_**<sup>(</sup> _i_<sup>_l_))</sup><sup>_t′∈_t.</sup> 

21 

Published as a conference paper at ICLR 2023 

**Arithmetic term** Now we want to calculate the term inside the parenthesis Eq. (25). We will calculate it through the MLP layer and store in **_m_** _i_ and substract the first term. Let’s denote the input to the MLP as **_x_** _i_ = ( **_a_** _i_ + **_h_**<sup>(</sup> _i_<sup>_l_)), the output of the first layer</sup><sup>**_u_**</sup><sup>_i_, the output of the non-linearity</sup> as **_a_** _i_ , and the final output as **_m_** _i_ . The entries of **_m_** _i_ will be: 



We will define the MLP layer to operate the attention term calculated above with a part of the current hidden state by defining _W_ 1 and _W_ 2. Let’s assume we bypass the LayerNorm by using Lemma 6. Let’s show this seperately for + and _⊙_ operators. 

**RAW** (+ _, ._ ) If the operator, _⃝⋆_ = +, first layer of the MLP will calculate the second term in Eq. (25) and overwrite the space where the attention output term Eq. (53) is written, and add a large positive bias term _N_ to by pass GeLU as explained in Lemma 4. We will use an available space<sup>ˆ</sup> t in the _xi_ same size as t. 



This can be done by setting _W_ 1 (weight term of the first layer of the MLP) to zero except the below indices: 



and the bias vector **_b_** 1 to 



Note the second term is added to make unused indices t _∪_ w _∪_ t<sup>ˆ</sup> become zero after the gelu, which outputs zero for large negative values. Since we added a large positive term, we make sure gelu behaved like a linear layer. Thus we have, 



Now, we need to set _W_ 2, to simulate _Wo ∈_ R<sup>_|w|×|t|_</sup> , 



22 

Published as a conference paper at ICLR 2023 



Therefore, **_m_** _i_ [ _w_ ] = _Wo_ **_x_** _i_ [t] + _W_ 0 _W_ **_h_**<sup>_l_</sup> _i_<sup>[s]</sup><sup>_−_</sup><sup>**_x_**</sup><sup>_i_[w]equalstowhatwepromisedinEq.(60)for</sup> + case. If we sum this with the residual **_x_** _i_ term back Eq. (53), so the output of this layer can be written as: 



**RAW** ( _⊙, ._ ) If the operator, _⃝⋆_ = _⊙_ , we need to use three extra hidden units the same size as _|_ t _|_ , let’s name the extra indices as t _a_ , t _b_ , t _c_ , and output _w_ space. The ( **_u_** _i_ ) will get below entries to be able to use [], where _N_ is a large number: 



All of this operations are linear, can be done _W_ 1 zero except the below entries: 



and **_b_** 1 to: 



The resulting **_v_** with the approximations become: 



23 

Published as a conference paper at ICLR 2023 

Now, we can use the GeLU trick in Lemma 4, by setting _W_ 2 



We then set _b_ 2: 





We have used 4 _|t|_ space for internal computation of this operation, and finally used _|_ w _|_ space to write the final result. We show RAW operator is implementable by setting the parameters of a Transformer. 

### C.5 PARALLELIZING THE RAW OPERATOR 

**Lemma 7.** _With the conditions that K is constant, the operators are independent (i.e_ ( _ri ∪ si ∪ wi_ ) _∩ wj_ = _i_ = _∅), and there is_<sup>�</sup> _k_<sup>(4</sup><sup>_|_t</sup><sup>_k|_+</sup><sup>_|_w</sup><sup>_k|_)</sup><sup>_availablespaceinthehiddenstate,thena_</sup> _Transformer layer can apply k such_ RAW _operation in parallel by setting different regions of W_ 1 _, W_ 2 _, Wf and_ ( _W_<sup>_V_</sup> ) _k matrices._ 

_Proof._ From the construction above, it is straightforward to modify the definition of the RAW operator to perform _k_ operations as all the indices of matrices that we use in Appendix C.4.3 do not overlap with the given conditions in the lemma. 

This makes it possible to construct a Transformer layer not only to implement vector-vector dot products, but general matrix-matrix products, as required by **mul** . With this, we show that we can implement **mul** by using single layer of a Transformer. 

### C.6 LAYERNORM FOR DIVISION 

Let say we have the input [ _c,_ **_y_** _,_ **0** ]<sup>_⊤_</sup> calculated before the attention output in Eq. (53), and we want to divide **_y_** to _c_ . This trick is very similar to the on in Lemma 6. We can use the following formula: **Lemma 8.** _using LayerNorm for division. Let N, M to be large numbers, λ LayerNorm function, the following approximation holds:_ 



_Proof._ 



24 

Published as a conference paper at ICLR 2023 

Then, 



To get the input to the format used in this Lemma, we can easily use _Wf_ to convert the head outputs. Then, after the layer norm, we can use _W_ 1 to pull the<sup>**_<u>y</u>_**</sup> _c_<sup>back and write it to the attention output.By</sup> this way, we can approximate scalar division in one layer. 

**Lemma 1** By Lemmas 2, 3, 3, 7 and 8; we constructed the operators in Lemma 1 using single layer of a Transformer, thus proved Lemma 1 

## D DETAILS OF TRANSFORMER ARHITECTURE AND TRAINING 

We perform these experiments using the Jax framework on P100 GPUs. The major hyperparameters used in these experiments are presented in Table 1. The code repository used for reproducing these experiments will be open sourced at the time of publication. Most of the hyperparameters adapted from previous work Garg et al. (2022) to be compatible, and we adapted the Transformer architecture details. We use Adam optimizer with cosine learning rate scheduler with warmup where number of warmup steps set to be 1/5 of total iterations. We use larned absolute position embeddings. 

|**Parameter**|**Search Range**|
|---|---|
|Number of heads|1, 2,**4**, 8 s|
|Number of layers|1, 2, 12,**16**|
|Hidden size|16, 32, 64, 256,**512**, 1024|
|Batch size|64|
|Maximum number of epochs|500.000|
|Initial Learning rate (_lri_)|**1e-4**, 2.5e-4|
|Weight decay|**0**, 1e-5|
|Bias initialization|**uniform scaling**, normal(1.0)|
|Weight initialization|**uniform scaling**, normal(1.0)|
|Position embedding initialization|**uniform scaling**, normal(1.0)|



Table 1: Hyperparameters used in the ICL. The best parameter for each hyperparameter is highlighted. 

In the phase shift plots in Fig. 3, we keep the value in the x-axis constant and used the best setting over the parameters: _{_ number of layers, hidden size, number of heads and learning rate _}_ . 

## E DETAILS OF PROBE 

We will use the terms _probe model_ and _task model_ to distinguish probe from ICL. Our probe is 





The position scores **_s_** _v ∈_ R<sup>_T_</sup> are learned parameters where _T_ is the max input sequence length ( _T_ = 80 in our experiments). The softmax of position scores attention weights **_α_** for each position and for each target variable. This enables us to learn input-independent, optimal target locations for each target (displayed on the right side of Fig. 4). We then average hidden states by using these attention weights. A linear projection, _Wv ∈_ R<sup>_T ×H′_</sup> , is applied before averaging. FF is either a linear layer or a 2-layer MLP (hidden size=512) with a GeLU activation function. For each layer, we train a different probe with different parameters using stochastic gradient descent. _H_<sup>_′_</sup> equals to the 512. The probe is trained using an Adam optimizer with a learning rate of 0.001 (chosen from among _{_ 0 _._ 01 _,_ 0 _._ 001 _,_ 0 _._ 0001 _}_ on validation data). 

25 

Published as a conference paper at ICLR 2023 



<!-- Start of picture text -->
X T Y probe error vs layer Layer-8's attention heatmap over time<br>0.06<br>Linear Probe 0.6<br>MLP Probe<br>0.04 0.4<br>0.02 0.2<br>0.00<br>0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75<br>layer position<br>wOLS probe errors vs layer Layer-12's attention heatmap over time<br>0.04<br>Linear Probe<br>MLP Probe<br>0.4<br>0.02<br>0.2<br>0.00<br>0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75<br>layer position<br>Figure 5: Detailed error values of the control probe displayed in Fig. 4.<br>1.0<br>R2<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>1 2 4 6 8 10 12<br>#exemplars<br>0<br>MSE 12<br>Target Index<br>3<br>0<br>MSE 12<br>Target Index<br>3<br> 2R<br><!-- End of picture text -->

Figure 6: _R_<sup>2</sup> of linear weight estimation on _d_ = 8 problem 

**Control Experiments** In Fig. 4, dashed lines show probing results with a task model trained on a _control task_ , in which _w_ is always the all-ones **1** . This problem structurally resembles our main experiment setup, but does not require in-context learning. During probing, we feed this model data generated by _w_ sampled form normal distribution as in the original task model. We observe that the control probe has a significantly higher error rate, showing that the probing accuracy obtained with actual task model is non-trivial. We present detailed error values of the control probe in Fig. 5. 

## F LINEARITY OF ICL 

In Fig. 1b, we compare implicit linear weight of the ICL against the linear algorithms using ILWD measure. Note that this measure do not assume predictors to be linear: when the predictors are not linear, ILWD measures the difference between closest linear predictors (in Eq. (16) sense) to each algorithm. 

To gain more insight to ICL’s algorithm, we can measure how linear ICL in different regimes of the linear problem (underdetermined, determined) by using _R_<sup>2</sup> (coefficient of determination) measure. So, instead of asking what’s the best linear fit in Eq. (16), we can ask how good is the linear fit, which is the _R_<sup>2</sup> of the estimator. Interestingly, even though our model matches min-norm least square solution in both metrics in Section 4.3, we show that ICL is becoming gradually linear in the under-determined regime Fig. 6. This is an important result, enables us to say the in-context learner’s hypothesis class is not purely linear. 

26 

Published as a conference paper at ICLR 2023 







(c) A piece-wise linear approximation to _x_<sup>2</sup> by using ReLU, Eq. (141). 

Figure 7: Approximations of multiplication via various non-linearities. 

## G MULTIPLICATIVE INTERACTIONS WITH OTHER NON-LINEARITIES 

We can show that for a real-valued and smooth non-linearity _f_ ( _x_ ), we can apply the same trick in in the paper body. In particular, we can write Taylor expansion as: 



which converges for some sufficiently small neighborhood: _X ∈_ [ _−ϵ, ϵ_ ]. First, assume that the second order term _a_ 2 dominates higher-order terms in this domain such that: 

_a_ 2 _x_<sup>2</sup> _≫ ai>_ 2 _x_<sup>_i_</sup> where _x ∈X_ 

. 

It’s is easy to verify that the following is true: 



So, given the expansion for GeLU in Eq. (37), we can use this generic formula to obtain the multiplication approximation: 



We plot this approximation against _x_<sup>2</sup> for [0 _._ 1 _, −_ 0 _._ 1] range in Fig. 7a. 

In the case of _a_ 2 is zero, we cannot get any second order term, and in the case of _a_ 2 is negligible _O_ ( _x_<sup>3</sup> + _y_<sup>3</sup> ) will dominate the Eq. (135), so we cannot obtain a good approximation of _xy_ . In this case, we can resort to numerical derivatives and utilize the _a_ 3 term: 



27 

Published as a conference paper at ICLR 2023 

If _a_ 3 is not negligible, _a_ 3 _x_<sup>2</sup> _≪ ai>_ 3 _x_<sup>_i_</sup> in the same domain, we can use numerical derivatives to get a multiplication term: 



For example, tanh has no second order term in its Taylor expansion: 



Using above formula we can obtain the following expression: 



Similar to our construction in Eq. (110), we can construct a Transformer layer that calculates these quantities (noting that _δ_ is a small, input-independent scalar). 

We plot this approximation against _x_<sup>2</sup> for [0 _._ 1 _, −_ 0 _._ 1] range in Fig. 7b. Note that, if we use this approximation in our constructions we will need more hidden space as there are 6 different tanh term as opposed to 3 GeLU term in Eq. (110). 

**Non-smooth non-linearities** ReLU is another commonly used non-linearity that is not differentiable. With ReLU, we can only hope to get piece-wise linear approximations. For example, we can try to approximate _x_<sup>2</sup> with the following function: 



We plot this approximation against _x_<sup>2</sup> for [0 _._ 1 _, −_ 0 _._ 1] range in Fig. 7c. 

## H EMPIRICAL SCALING ANALYSIS WITH DIMENSIONALITY 

In Figs. 3a and 3b, we showed that ICL needs different hidden sizes to enter the “Ridge regression phase” (orange background) or “OLS phase” (green background) depending on the dimensionality _d_ of inputs _x_ . However, we cannot reliably read the actual relations between size requirements and the dimension of the problem from only two dimensions. To better understand size requirements, we ask the following empirical question for each dimension: how many layer/hidden size/heads are needed to better fit the least-squares solution than the Ridge( _λ_ = _ϵ_ ) regression solution (the green phase in Figs. 3a and 3b)? 

To answer this important question, we experimented with _d_ = _{_ 1 _,_ 2 _,_ 4 _,_ 8 _,_ 12 _,_ 16 _,_ 20 _}_ and run an experiment sweep for each dimension over: 

- number of layers (L): _{_ 1 _,_ 2 _,_ 4 _,_ 8 _,_ 12 _,_ 16 _}_ , 

- hidden size (H): _{_ 16 _,_ 32 _,_ 64 _,_ 256 _,_ 512 _,_ 1024 _}_ , 

- number of heads (M): _{_ 1 _,_ 2 _,_ 4 _,_ 8 _}_ , 

- learning rate: _{_ 1e-4, 2.5e-4 _}_ . 

For each feature that affects computational capacity of transformer ( _L_ , _H_ , _M_ ), we optimize other features and find the minimum value for the feature that satisfies SPD(OLS _,_ ICL) _<_ SPD(Ridge( _λ_ = _ϵ_ ) _,_ ICL). We plot our experiment with _ϵ_ = 0 _._ 1 in Appendix H. We find that single head is enough for all problem dimensions, while other parameters exhibit a step-functionlike dependence on input size. 

Please note that other hyperparameters discussed in Appendix D (e.g weight initialization) were not optimized for each dimension independently. 

28 

Published as a conference paper at ICLR 2023 



<!-- Start of picture text -->
8<br>30<br>6<br>25<br>4<br>20<br>2<br>1 2 4 8 16 20 1 2 4 8 16 20<br>Problem dimension (d) Problem dimension (d)<br>(a) hidden size requirements. (b) layer requirements.<br>1.04<br>1.02<br>1.00<br>0.98<br>0.96<br>1 2 4 8 16 20<br>Problem dimension (d)<br>)H )L<br>Layers (<br>hidden size (<br>)M<br>Heads (<br><!-- End of picture text -->



<!-- Start of picture text -->
(c) number of head requirements.<br><!-- End of picture text -->

Figure 8: Empirical requirements on model parameters to satisfy SPD(Ridge( _λ_ = 0 _._ 1) _,_ ICL) _>_ SPD(OLS _,_ ICL) when other parameters optimized. 

29 

