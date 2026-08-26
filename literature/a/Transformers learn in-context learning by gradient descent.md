# **Transformers Learn In-Context by Gradient Descent** 

**Johannes von Oswald**<sup>1 2</sup> **Eyvind Niklasson**<sup>2</sup> **Ettore Randazzo**<sup>2</sup> **Jo˜ao Sacramento**<sup>1</sup> **Alexander Mordvintsev**<sup>2</sup> **Andrey Zhmoginov**<sup>2</sup> **Max Vladymyrov**<sup>2</sup> 

## **Abstract** 

## **1. Introduction** 

At present, the mechanisms of in-context learning in Transformers are not well understood and remain mostly an intuition. In this paper, we suggest that training Transformers on auto-regressive objectives is closely related to gradient-based metalearning formulations. We start by providing a simple weight construction that shows the equivalence of data transformations induced by 1) a single linear self-attention layer and by 2) gradientdescent (GD) on a regression loss. Motivated by that construction, we show empirically that when training self-attention-only Transformers on simple regression tasks either the models learned by GD and Transformers show great similarity or, remarkably, the weights found by optimization match the construction. Thus we show how trained Transformers become mesa-optimizers i.e. learn models by gradient descent in their forward pass. This allows us, at least in the domain of regression problems, to mechanistically understand the inner workings of in-context learning in optimized Transformers. Building on this insight, we furthermore identify how Transformers surpass the performance of plain gradient descent by learning an iterative curvature correction and learn linear models on deep data representations to solve non-linear regression tasks. Finally, we discuss intriguing parallels to a mechanism identified to be crucial for in-context learning termed _induction-head_ (Olsson et al., 2022) and show how it could be understood as a specific case of in-context learning by gradient descent learning within Transformers. 

In recent years Transformers (TFs; Vaswani et al., 2017) have demonstrated their superiority in numerous benchmarks and various fields of modern machine learning, and have emerged as the _de-facto_ neural network architecture used for modern AI (Dosovitskiy et al., 2021; Yun et al., 2019; Carion et al., 2020; Gulati et al., 2020). It has been hypothesised that their success is due in part to a phenomenon called _in-context learning_ (Brown et al., 2020; Liu et al., 2021): an ability to flexibly adjust their prediction based on additional data given _in context_ (i.e. in the input sequence itself). In-context learning offers a seemingly different approach to few-shot and meta-learning (Brown et al., 2020), but as of today the exact mechanisms of how it works are not fully understood. It is thus of great interest to understand what makes Transformers pay attention to their context, what the mechanisms are, and under which circumstances, they come into play (Chan et al., 2022b; Olsson et al., 2022). 

In this paper, we aim to bridge the gap between in-context and meta-learning, and show that in-context learning in Transformers can be an emergent property approximating gradient-based few-shot learning within its forward pass, see Figure 1. For this to be realized, we show how Transformers **(1)** construct a loss function dependent on the data given in sequence and **(2)** learn based on gradients of that loss. We will first focus on the latter, the more elaborate learning task, in sections 2 and 3, after which we provide evidence for the former in section 4. 

We summarize our contributions as follows<sup>1</sup> : 

- We construct explicit weights for a linear self-attention layer that induces an update identical to a single step of gradient descent (GD) on a mean squared error loss. Additionally, we show how several self-attention layers can iteratively perform curvature correction improving on plain gradient descent. 

> 1Department of Computer Science, ETH Zurich,¨ Zurich,¨ Switzerland<sup>2</sup> Google Research. Correspondence to: Johannes von Oswald _<_ voswaldj@ethz.ch _>_ . 

_Proceedings of the 40_<sup>_th_</sup> _International Conference on Machine Learning_ , Honolulu, Hawaii, USA. PMLR 202, 2023. Copyright 2023 by the author(s). 

- When optimized on linear regression datasets, we demonstrate that linear self-attention-only Transform- 

> 1Main experiments can be reproduced with notebooks provided under the following link: https://github.com/ google-research/self-organising-systems/ tree/master/transformers_learn_icl_by_gd 

1 

**Transformers Learn In-Context by Gradient Descent** 



























<!-- Start of picture text -->
Gradient descent<br>0.2 Trained Transformer<br>0.1<br>0.0<br>0 20 40<br>GD Steps / Transformer Layers<br>Loss<br><!-- End of picture text -->

_Figure 1._ **Illustration of our hypothesis: gradient-based optimization and attention-based in-context learning are equivalent.** _Left_ : Learning a neural network output layer by gradient descent on a dataset D<sup>train</sup> . The task-shared meta-parameters _θ_ are obtained by meta-learning with the goal that after adjusting the neural network output layer, the model generalizes well on unseen data. _Center_ : Illustration of a Transformer that adjusts its query prediction on the data given in-context i.e. _tθ_ ( _x_ query; D<sup>context</sup> ). The weights of the Transformer are optimized to predict the next token _y_ query. _Right_ : Our results confirm the hypothesis that learning with _K_ steps of gradient descent on a dataset D<sup>train</sup> ( _green part of the left plot_ ) matches trained Transformers with _K_ linear self-attention layers ( _central plot_ ) when given D<sup>train</sup> as in-context data D<sup>context</sup> . 

- ers either converge to our weight construction and therefore implement gradient descent, or generate linear models that closely align with models trained by GD, both in in- and out-of-distribution validation tasks. 

- By incorporating multi-layer-perceptrons (MLPs) into the Transformer architecture, we enable solving _non_ linear regression tasks within Transformers by showing its equivalence to learning a linear model on deep representations. We discuss connections to kernel regression as well as nonparametric kernel smoothing methods. Empirically, we compare meta-learned MLPs and a single step of GD on its output layer with trained Transformers and demonstrate striking similarities between the identified solutions. 

- We resolve the dependency on the specific token construction by providing evidence that learned Transformers first encode incoming tokens into a format amenable to the in-context gradient descent learning that occurs in the later layers of the Transformer. 

These findings allow us to connect _learning_ Transformer weights and the concept of _meta-learning_ a learning algorithm (Schmidhuber, 1987; Hinton & Plaut, 1987; Bengio et al., 1990; Chalmers, 1991; Schmidhuber, 1992; Thrun & Pratt, 1998; Hochreiter et al., 2001; Andrychowicz et al., 2016; Ba et al., 2016; Kirsch & Schmidhuber, 2021). In this extensive research field, meta-learning is typically regarded as learning that takes place on various time scales namely fast and slow. The slowly changing parameters control and prepare for fast adaptation reacting to sudden changes in the incoming data by e.g. a context switch. Notably, we build heavily on the concept of fast weights (Schmidhuber, 1992) which has shown to be equivalent to linear self-attention (Schlag et al., 2021) and show how optimized Transformers implement interpretable learning algorithms within their weights. 

Another related meta-learning concept, termed MAML (Finn et al., 2017), aims to meta-learn a deep neural network 

initialization which allows for fast adaptation on novel tasks. It has been shown that in many circumstances, the solution found can be approximated well when only adapting the output layer i.e. learning a linear model on a meta-learned deep data representations (Finn et al., 2017; Finn & Levine, 2018; Gordon et al., 2019; Lee et al., 2019; Rusu et al., 2019; Raghu et al., 2020; von Oswald et al., 2021). In section 3, we show the equivalence of this framework to in-context learning implemented in a common Transformer block i.e. when combining self-attention layers with a multi-layerperceptron. 

In the light of meta-learning we show how optimizing Transformer weights can be regarded as learning on two time scales. More concretely, we find that solely through the pressure to predict correctly Transformers discover learning algorithms inside their forward computations, effectively meta-learning a learning algorithm. Recently, this concept of an emergent optimizer within a learned neural network, such as a Transformer, has been termed “mesa-optimization” (Hubinger et al., 2019). We find and describe one possible realization of this concept and hypothesize that the in-context learning capabilities of language models emerge through mechanisms similar to the ones we discuss here. 

Transformers come in different “shapes and sizes”, operate on vastly different domains, and exhibit varying forms of phase transitions of in-context learning (Kirsch et al., 2022; Chan et al., 2022a), suggesting variance and significant complexity of the underlying learning mechanisms. As a result, we expect our findings on linear self-attention-only Transformers to only explain a limited part of a complex process, and it may be one of many possible methods giving rise to in-context learning. Nevertheless, our approach provides an intriguing perspective on, and novel evidence for, an incontext learning mechanism that significantly differs from existing mechanisms based on associative memory (Ramsauer et al., 2020), or by the copying mechanism termed _induction heads_ identified by (Olsson et al., 2022). We, therefore, state the following 

2 

**Transformers Learn In-Context by Gradient Descent** 

**Hypothesis 1** (Transformers learn in-context by gradient descent) **.** _When training Transformers on auto-regressive tasks, in-context learning in the Transformer forward pass is implemented by gradient-based optimization of an implicit auto-regressive inner loss constructed from its in-context data._ 

We acknowledge work done in parallel, investigating the same hypothesis. Akyurek¨ et al. (2023) puts forward a weight construction based on a chain of Transformer layers (including MLPs) that together implement a single step of gradient descent with weight decay. Similar to work done by Garg et al. (2022), they then show that trained Transformers match the performance of models obtained by gradient descent. Nevertheless, it is not clear that optimization finds Transformer weights that coincide with their construction. 

Here, we present a much simpler construction that builds on Schlag et al. (2021) and _only requires a single linear selfattention layer_ to implement a step of gradient descent. This allows us to (1) show that optimizing self-attention-only Transformers finds weights that match our weight construction (Proposition 1), demonstrating its practical relevance, and (2) explain in-context learning in shallow two layer Transformers intensively studied by Olsson et al. (2022). Therefore, although related work provides comprehensive empirical evidence that Transformers indeed seem to implement gradient descent based learning on the data given in-context, we will in the following present mechanistic verification of this hypothesis and provide compelling evidence that our construction, which implements GD in a Transformer forward pass, is found in practice. 

## **2. Linear self-attention** **_can_ emulate gradient descent on a linear regression task** 

We start by reviewing a standard multi-head self-attention (SA) layer with parameters _θ_ . A SA layer updates each element _ej_ of a set of tokens _{e_ 1 _, . . . , eN }_ according to 



with _Ph, Vh, Kh_ the projection, value and key matrices, respectively, and _qh,i_ the query, all for the _h_ -th head. To simplify the presentation, we omit bias terms here and throughout. The columns of the value _Vh_ = [ _vh,_ 1 _, . . . , vh,N_ ] and key _Kh_ = [ _kh,_ 1 _, . . . , kh,N_ ] matrices consist of vectors _vh,i_ = _Wh,V ei_ and _kh,i_ = _Wh,Kei_ ; likewise, the query is produced by linearly projecting the tokens, _qh,j_ = _Wh,Qej_ . The parameters _θ_ = _{Ph, Wh,V , Wh,K, Wh,Q}h_ of a SA layer consist of all the projection matrices, of all heads. 

The self-attention layer described above corresponds to the one used in the standard Transformer model. Follow- 

ing Schlag et al. (2021), we now introduce our first (and only) departure from the standard model, and omit the softmax operation in equation 1, leading to the _linear_ selfattention (LSA) layer _ej ← ej_ + LSA _θ_ ( _j, {e_ 1 _, . . . , eN }_ ) = _ej_ +<sup>�</sup> _h_<sup>_PhVhK_</sup> _h_<sup>_Tqh,j_We next show that with some sim-</sup> ple manipulations we can relate the update performed by an LSA layer to one step of gradient descent on a linear regression loss. 

### **Data transformations induced by gradient descent** 

We now introduce a reference linear model _y_ ( _x_ ) = _Wx_ parameterized by the weight matrix _W ∈_ R<sup>_Ny×Nx_</sup> , and a training dataset _D_ = _{_ ( _xi, yi_ ) _}_<sup>_N_</sup> _i_ =1<sup>comprising of input</sup> samples _xi ∈_ R<sup>_Nx_</sup> and respective labels _yi ∈_ R<sup>_Ny_</sup> . The goal of learning is to minimize the squared-error loss: 



One step of gradient descent on _L_ with learning rate _η_ yields the weight change 



Considering the loss after changing the weights, we obtain 



where we introduced the transformed targets _yi −_ ∆ _yi_ with ∆ _yi_ = ∆ _Wxi_ . Thus, we can view the outcome of a gradient descent step as an update to our regression loss (equation 2), where data, and not weights, are updated. Note that this formulation is closely linked to predicting based on nonparametric kernel smoothing, see Appendix A.8 for a discussion. 

Returning to self-attention mechanisms and Transformers, we consider an in-context learning problem where we are given _N_ context tokens together with an extra query token, indexed by _N_ + 1. In terms of our linear regression problem, the _N_ context tokens _ej_ = ( _xj, yj_ ) _∈_ R<sup>_Nx_+</sup><sup>_Ny_</sup> correspond to the _N_ training points in _D_ , and the _N_ +1-th token _eN_ +1 = ( _xN_ +1 _, yN_ +1) = ( _x_ test _,_ ˆ _y_ test) = _e_ test to the test input _x_ test and the corresponding prediction _y_ ˆtest. We use the terms training and in-context data interchangeably, as well as query and test token/data, as we establish their equivalence now. 

3 

**Transformers Learn In-Context by Gradient Descent** 

### **Transformations induced by gradient descent and a linear self-attention layer can be equivalent** 

We have re-cast the task of learning a linear model as directly modifying the data, instead of explicitly computing and returning the weights of the model (equation 4). We proceed to establish a connection between self-attention and gradient descent. We provide a construction where learning takes place simultaneously by directly updating all tokens, including the test token, through a linear self-attention layer. In other words, the token produced in response to a query (test) token is transformed from its initial value _W_ 0 _x_ test, where _W_ 0 is the initial value of _W_ , to the post-learning prediction ˆ _y_ = ( _W_ 0 +∆ _W_ ) _x_ test obtained after one gradient descent step. 

**Proposition 1.** _Given a 1-head linear attention layer and the tokens ej_ = ( _xj, yj_ ) _, for j_ = 1 _, . . . , N , one can construct key, query and value matrices WK, WQ, WV as well as the projection matrix P such that a Transformer step on every token ej is identical to the gradient-induced dynamics ej ←_ ( _xj, yj_ ) + (0 _, −_ ∆ _Wxj_ ) = ( _xj, yj_ ) + _P V K_<sup>_T_</sup> _qj such that ej_ = ( _xj, yj −_ ∆ _yj_ ) _. For the test data token_ ( _xN_ +1 _, yN_ +1) _the dynamics are identical._ 

The simple construction can be found in Appendix A.1 and we denote the corresponding self-attention weights by _θ_ GD. 

Below, we provide some additional insights on what is needed to implement the provided LSA-layer weight construction, and further details on what it can achieve: 

- **Full self-attention.** Our dynamics model training is based on in-context tokens only, i.e., only _e_ 1 _, . . . , eN_ are used for computing key and value matrices; the query token _eN_ +1 (containing test data) is excluded. This leads to a linear function in _x_ test as well as to the correct ∆ _W_ , induced by gradient descent on a loss consisting only of the training data. This is a minor deviation from full self-attention. In practice, this modification can be dropped, which corresponds to assuming that the underlying initial weight matrix is zero, _W_ 0 _≈_ 0, which makes ∆ _W_ in equation 8 independent of the test token even if incorporating it in the key and value matrices. In our experiments, we see that these assumptions are met when initializing the attention weights _θ_ to small values. 

- **Reading out predictions.** When initializing the _y_ - entry of the test-data token with _−W_ 0 _xN_ +1, i.e. _e_ test = ˆ 

- ( _x_ test _, −W_ 0 _x_ test), the test-data prediction _y_ can be easily read out by simply multiplying again by _−_ 1 the updated token, since _−yN_ +1 + ∆ _yN_ +1 = _−_ ( _yN_ +1 _−_ ∆ _yN_ +1) = _yN_ +1 + ∆ _WxN_ +1. This can easily be done by a final projection matrix, which incidentally is usually found in Transformer architectures. Importantly, we see that a single head of self-attention is 

sufficient to transform our training targets as well as the test prediction simultaneously. 

- **Uniqueness.** We note that the construction is not unique; in particular, it is only required that the products _PWV_ as well as _WKWQ_ match the construction. Furthermore, since no nonlinearity is present, any rescaling _s_ of the matrix products, i.e., _PWV s_ and _WKWQ/s_ , leads to an equivalent result. If we correct for these equivalent formulations, we can experimentally verify that weights of our learned Transformers indeed match the presented construction. 

- **Meta-learned task-shared learning rates.** When training self-attention parameters _θ_ across a family of in-context learning tasks _τ_ , where the data ( _xτ,i, yτ,i_ ) follows a certain distribution, the learning rate can be implicitly (meta-)learned such that an optimal loss reduction (averaged over tasks) is achieved given a fixed number of update steps. In our experiments, we find this to be the case. This kind of meta-learning to improve upon plain gradient descent has been leveraged in numerous previous approaches for deep neural networks (Li et al., 2017; Lee & Choi, 2018; Park & Oliva, 2019; Zhao et al., 2020; Flennerhag et al., 2020). 

- **Task-specific data transformations.** A self-attention layer is in principle further capable of exploiting statistics in the current training data samples, beyond modeling task-shared curvature information in _θ_ . More concretely, a LSA layer updates an input sample according to a data transformation _xj ← xj_ + ∆ _xj_ = ( _I_ + _P_ ( _X_ ) _V_ ( _X_ ) _K_ ( _X_ )<sup>_T_</sup> _WQ_ ) _xj_ = _Hθ_ ( _X_ ) _xj_ , with _X_ the _Nx × N_ input training data matrix, when neglecting influences by target data _yi_ . Through _Hθ_ ( _X_ ), a LSA layer can encode in _θ_ an algorithm for carrying out data transformations which depend on the actual input training samples in _X_ . In our experiments, we see that trained self-attention learners employ a simple form of _H_ ( _X_ ) and that this leads to substantial speed ups in for GD and TF learning. 

## **3. Trained Transformers** **_do_ mimic gradient descent on linear regression tasks** 

We now experimentally investigate whether trained attention-based models implement gradient-based incontext learning in their forward passes. We gradually build up from single linear self-attention layers to multi-layer nonlinear models, approaching full Transformers. In this section, we follow the assumption of Proposition 1 tightly and construct our tokens by concatenating input and target data, _ej_ = ( _xj, yj_ ) for 1 _≤ j ≤ N_ , and our query token by concatenating the test input and a zero vector, _eN_ +1 = ( _x_ test _,_ 0). We show how to lift this assumption in the last section of the 

4 

**Transformers Learn In-Context by Gradient Descent** 



<!-- Start of picture text -->
GD<br>Trained TF<br><!-- End of picture text -->



<!-- Start of picture text -->
Preds diff Model cos<br>Model diff<br><!-- End of picture text -->





<!-- Start of picture text -->
GD<br>Interpolated<br>Trained TF<br>0.5 1.0 1.5 2.0<br>   where x   U( )<br><!-- End of picture text -->

_Figure 2._ **Comparing one step of GD with a trained single linear self-attention layer.** _Outer left:_ Trained single LSA layer performance is identical to the one of gradient descent. _Center left_ : Almost perfect alignment of GD and the model generated by the SA layer after training, measured by cosine similarity and the L2 distance between models as well as their predictions. _Center right:_ Identical loss of GD, the LSA layer model as well as the model obtained by interpolating between the construction and the optimized LSA layer weights for different _N_ = _Nx_ . _Outer right:_ The trained LSA layer, gradient descent and their interpolation show identically loss (in log-scale) when provided input data different than during training i.e. with scale of 1. We display the mean/std. or the single runs of 5 seeds. 

paper. The prediction _y_ ˆ _θ_ ( _{eτ,_ 1 _, . . . , eτ,N }, eτ,N_ +1) of the attention-based model, which depends on all tokens and on the parameters _θ_ , is read-out from the _y_ -entry of the updated _N_ + 1-th token as explained in the previous section. 

The objective of training, visualized in Figure 1, is to minimize the expected squared prediction error, averaged over ˆ tasks min _θ_ E _τ_ [ _||yθ_ ( _{eτ,_ 1 _, . . . , eτ,N }, eτ,N_ +1) _− yτ,_ test _||_<sup>2</sup> ]. We achieve this by minibatch online minimization (by Adam (Kingma & Ba, 2014)): At every optimization step, we construct a batch of novel training tasks and take a step of stochastic gradient descent on the loss function: 



where each task (context) _τ_ consists of in-context training data _Dτ_ = _{_ ( _xτ,i, yτ,i_ ) _}_<sup>_N_</sup> _i_ =1<sup>andtestpoint</sup> ( _xτ,N_ +1 _, yτ,N_ +1), which we use to construct our tokens _{eτ,i}_<sup>_N_</sup> _i_ =1<sup>+1as described above.We denote the optimal pa-</sup> rameters found by this optimization process by _θ_<sup>_∗_</sup> . In our setup, finding _θ_<sup>_∗_</sup> may be thought of as meta-learning, while learning a particular task _τ_ corresponds to simply evaluating the model _y_ ˆ _θ_ ( _{eτ,_ 1 _, . . . , eτ,N }, eτ,N_ +1). Note that we therefore never see the exact same training task twice during training. See Appendix A.12, especially Figure 16 for an analyses when using a fixed dataset size which we cycle over during training. 

We focus on solvable tasks and similarly to Garg et al. (2022) generate data for each task using a teacher model with parameters _Wτ ∼N_ (0 _, I_ ). We then sample _xτ,i ∼ U_ ( _−_ 1 _,_ 1)<sup>_nI_</sup> and construct targets using the task-specific teacher model, _yτ,i_ = _Wτ xτ,i_ . In the majority of our experiments we set the dimensions to _N_ = _nI_ = 10 and _nO_ = 1. Since we use a noiseless teacher for simplicity, we can expect our regression tasks to be well-posed and analytically solvable as we only compute a loss on the Transformers last token, which stands in contrast to usual autoregressive training and the training setup of Garg et al. (2022). Full details and results for training with a fixed training set size may be found in Appendix A.12. 

### **One-step of gradient descent vs. a single trained self-attention layer** 

Our first goal is to investigate whether a trained single, linear self-attention layer can be explained by the provided weight construction that implements GD. To that end, we compare the predictions made by a LSA layer with trained weights _θ_<sup>_∗_</sup> (which minimize equation 5) and with constructed weights _θ_ GD (which satisfy Proposition 1). 

ˆ Recall that a LSA layer yields the prediction _yθ_ ( _x_ test) = _eN_ +1 + LSA _θ_ ( _{e_ 1 _, . . . , eN }, eN_ +1) = ∆ _Wθ,Dx_ test, which is linear in _x_ test. We denote by ∆ _Wθ,D_ the matrix generated by the LSA layer following the construction provided in Proposition 1, with query token _eN_ +1 set such that the initial prediction is set to zero, _y_ ˆtest = 0. We compare _y_ ˆ _θ_ ( _x_ test) to the prediction of the control LSA _y_ ˆ _θ_ GD( _x_ test), which under our token construction corresponds to a linear model trained by one step of gradient descent starting from _W_ 0 = 0. For this control model, we determine the optimal learning rate _η_ by minimizing _L_ ( _η_ ) over a training set of 10<sup>4</sup> tasks through line search, with _L_ ( _η_ ) defined analogously to equation 5. 

More concretely, to compare trained and constructed LSA layers, we sample _T_ val = 10<sup>4</sup> validation tasks and record the following quantities, averaged over validation tasks: (1) the difference in predictions measured with the L2 norm, ˆ ˆ _∥yθ_ ( _xτ,_ test) _− yθ_ GD( _xτ,∂_ test _y_ ˆ _θ_ <u>GD)</u> _∥_ <u>(,</u> _x_ (2) _τ,_ test)the cosine similarity between the sensitivities and<sup>_∂y_ˆ</sup><sup>_θ_</sup><sup><u>(</u></sup><sup>_xτ,_test)</sup> as well as _∂y_ ˆ _θ_ <u>GD (</u> _∂xxτ,_ testtest) _−_<sup>_∂y_ˆ</sup><sup>_θ_</sup><sup><u>(</u></sup><sup>_xτ,_test</sup> _∂x_<sup><u>)</u></sup> test (3) their difference _∥ ∂x_ test _∂x_ test _∥_ again according to the L2 norm, which in both cases yields the explicit models computed by the algorithm. We show the results of these comparisons in Figure 2. We find an excellent agreement between the two models over a wide range of hyperparameters. We note that as we do not have direct access to the initialization of _W_ in the attention-based learners (it is hidden in _θ_ ), we cannot expect the models to agree exactly. 

Although the above metrics are important to show similarities between the resulting learned models (in-context 

5 

**Transformers Learn In-Context by Gradient Descent** 

vs. gradient-based), the underlying algorithms could still be different. We therefore carry out an extended set of analyses: 

1. **Interpolation.** We take inspiration on recent work (Benzing et al., 2022; Entezari et al., 2021) that showed approximate equivalence of models found by SGD after permuting weights within the trained neural networks. Since our models are deep linear networks with respect to _x_ test we only correct for scaling mismatches between the two models – in this case the construction that implements GD and the trained weights. As shown in Figure 2, we observe (and can actually inspect by eye, see Appendix Figure 9) that a simple scaling correction on the trained weights is enough to recover the weight construction implementing GD. This leads to an identical loss of GD, the trained Transformer and the linearly interpolated weights _θ_ I = ( _θ_ + _θ_ GD) _/_ 2. See details in Appendix A.3 on how our weight correction and interpolation is obtained. 

2. **Out-of-distribution validation tasks.** To test if our in-context learner has found a generalizable update rule, we investigate how GD, the trained LSA layer and its interpolation behave when providing in-context data in regimes different to the ones used during training. We therefore visualize the loss increase when (1) sampling the input data from _U_ ( _−α, α_ )<sup>_Nx_</sup> or (2) scaling the teacher weights by _α_ as _αW_ when sampling validation tasks. For both cases, we set _α_ = 1 during training. We again observe that when training a single linear self-attention Transformer, for both interventions, the Transformer performs equally to gradient descent outside of this training setups, see Figure 2 as well Appendix Figure 6. Note that the loss obtained through gradient descent also starts degrading quickly outside the training regime. Since we tune the learning rate for the input range [ _−_ 1 _,_ 1] and one gradient step, tasks with larger input range will have higher curvature and the optimal learning rate for smaller ranges will lead to divergence and a drastic increase in loss also for GD. 

3. **Repeating the LSA update.** Since we claim that a single trained LSA layer implements a GD-like learning rule, we further test its behavior when applying it repeatedly, not only once as in training. After we correct the learning rate of both algorithms, i.e. for GD and the trained Transformer with a dampening parameter _λ_ = 0 _._ 75 (details in Appendix A.6), we see an identical loss decrease of both GD and the Transformer, see Figure 1. 

To conclude, we present evidence that optimizing a single LSA layer to solve linear regression tasks finds weights 

that (approximately) coincide with the LSA-layer weight construction of Proposition 1, hence implementing a step of gradient descent, leading to the same learning capabilities on in- and out-of-distribution tasks. We comment on the random seed dependent phase transition of the loss during training in Appendix A.11. 

### **Multiple steps of gradient descent vs. multiple layers of self-attention** 

We now turn to deep linear self-attention-only Transformers. The construction we put forth in Proposition 1, can be immediately stacked up over _K_ layers; in this case, the final prediction can be read out from the last layer as before by negating the _y_ -entry of the last test token: _−yN_ +1 + _K_ � _k_ =1<sup>∆</sup><sup>_yk,N_+1=</sup><sup>_−_(</sup><sup>_yN_+1</sup><sup>_−_�</sup><sup>_K_</sup> _k_ =1<sup>∆</sup><sup>_yk,N_+1) =</sup><sup>_yN_+1 +</sup> � _Kk_ =1<sup>∆</sup><sup>_WkxN_+1, where</sup><sup>_yk,N_+1 are the test token values</sup> at layer _k_ , and ∆ _yk,N_ +1 the change in the _y_ -entry of the test token after applying the _k_ -th step of self-attention, and ∆ _Wk_ the _k_ -th implicit change in the underlying linear model parameters _W_ . When optimizing such Transformers with _K_ layers, we observe that these models generally outperform _K_ steps of plain gradient descent, see Figure 3. Their behavior is however well described by a variant of gradient descent, for which we tune a single parameter _γ_ defined through the transformation function _H_ ( _X_ ) which transforms the input data according to _xj ← H_ ( _X_ ) _xj_ , with _H_ ( _X_ ) = ( _I − γXX_<sup>_T_</sup> ). We term this gradient descent variant GD<sup>++</sup> which we explain and analyze in Appendix A.10. 

To analyze the effect of adding more layers to the architecture, we first turn to the arguably simplest extension of a single SA layer and analyze a _recurrent_ or _looped_ 2-layer LSA model. Here, we simply repeatably apply the same layer (with the same weights) multiple times i.e. drawing the analogy to learning an iterative algorithm that applies the same logic multiple times. 

Somewhat surprisingly, we find that the trained model surpasses plain gradient descent, which also results in decreasing alignment between the two models (see center left column), and the recurrent Transformer realigns perfectly with GD<sup>++</sup> while matching its performance on in- and out-of distribution tasks. Again, we can interpolate between the Transformer weights found by optimization and the LSAweight construction with learned _η, γ_ , see Figure 3 & 6. 

We next consider deeper, non-recurrent 5-layer LSA-only Transformers, with different parameters per layer (i.e. no weight tying). We see that a different GD learning rate as well as _γ_ per step (layer) need to be tuned to match the Transformer performance. This slight modification leads again to almost perfect alignment between the trained TF and GD<sup>++</sup> with in this case 10 additional parameters and 

6 

**Transformers Learn In-Context by Gradient Descent** 

### **(a) Comparing two steps of gradient descent with trained** **_recurrent_ two-layer Transformers.** 



<!-- Start of picture text -->
0.40 GD vs trained TF GD + +  vs trained TF Test on larger inputs<br>GD 2.5 2.5 10 4<br>0.35 GD + + Model cos Model cos GD<br>Trained TF 2.0 1.0 2.0 1.0 10 3 GD + +<br>0.300.25 1.5 Preds diffModel diff 0.90.8 1.5 Preds diffModel diff 0.90.8 1010 21 InterpolatedTrained TF<br>0.20 1.0 0.7 1.0 0.7<br>10 0<br>0.15 0.5 0.6 0.5 0.6 10 1<br>0.10 0.0 0.5 0.0 0.5<br>0 1000 2000 3000 0 1000 2000 3000 0 1000 2000 3000 0.5 1.0 1.5 2.0<br>Training steps Training steps Training steps    where x   U( , )<br>(b) Comparing five steps of gradient descent with trained five-layer Transformers.<br>0.4 GD vs trained TF GD + +  vs trained TF Test on larger inputs<br>0.3 GDGDTrained TF + + 5 steps 2.01.5 Model cos 1.051.00 2.01.5 Model cos 1.051.00 10 1 GDGDTrained TF + +<br>0.2 1.0 Preds diffModel diff 0.95 1.0 Preds diffModel diff 0.95 10 0<br>0.90 0.90 10 1<br>0.1 0.5 0.5<br>0.85 0.85 10 2<br>0.0 0.80 0.0 0.80 0.5 1.0 1.5 2.0<br>0 20000 40000 0 20000 40000 0 20000 40000<br>   where x   U( , )<br>Training steps Training steps Training steps<br>Loss Loss<br>L2 Norm L2 Norm<br>Cosine sim Cosine sim<br>Loss Loss<br>L2 Norm L2 Norm<br>Cosine sim Cosine sim<br><!-- End of picture text -->

_Figure 3. Far left column:_ The trained TF performance surpasses standard GD but matches GD<sup>++</sup> , our GD variant with simple iterative data transformation. On both cases, we tuned the gradient descent learning rates as well as the scalar _γ_ which governs the data transformation _H_ ( _X_ ). _Center left & center right columns_ : We measure the alignment between the GD as well as the GD<sup>++</sup> models and the trained TF. In both cases the TF aligns well with GD in the beginning of training but aligns much better with GD<sup>++</sup> after training. _Far right column:_ TF performance (in log-scale) mimics the one of GD<sup>++</sup> well when testing on OOD tasks ( _α_ = 1). 

loss close to 0, see Figure 3. Nevertheless, we see that the naive correction necessary for model interpolation used in the aforementioned experiments is not enough to interpolate without a loss increase. We leave a search for better weight corrections to future work. We further study Transformers with different depths for recurrent as well as non-recurrent architectures with multiple heads and equipped with MLPs, and find qualitatively equivalent results, see Appendix Figure 7 and Figure 8. Additionally, in Appendix A.9, we provide results obtained when using softmax SA layers as well as LayerNorm, thus essentially retrieving the standard Transformer architecture. We again observe and are able to explain (after slight architectural modifications) good learning performance and as well as alignment with the construction of Proposition 1, though worse than when using linear self-attention. These findings suggest that the incontext learning abilities of the standard Transformer with these common architecture choices can be explained by the gradient-based learning hypothesis explored here. Our findings also question the ubiquitous use of softmax attention, and suggest further investigation is warranted into the performance of linear vs. softmax SA layers in real-world learning tasks, as initiated by Schlag et al. (2021). 

### **Transformers solve nonlinear regression tasks by gradient descent on deep data representations** 

It is unreasonable to assume that the astonishing in-context learning flexibility observed in large Transformers is ex- 

plained by gradient descent on linear models. We now show that this limitation can be resolved by incorporating one additional element of fully-fledged Transformers: preceding self-attention layers by MLPs enables learning linear models by gradient descent on deep representations which motivates our illustration in Figure 1. Empirically, we demonstrate this by solving non-linear sine-wave regression tasks, see Figure 4. Experimental details can be found in Appendix A.7. We state 

**Proposition 2.** _Given a Transformer block i.e. a MLP m_ ( _e_ ) _which transforms the tokens ej_ = ( _xj, yj_ ) _followed by an attention layer, we can construct weights that lead to gradient descent dynamics descending_ 21 _N_ � _Ni_ =1<sup>_||Wm_(</sup><sup>_xi_)</sup><sup>_−yi||_2</sup><sup>_._</sup> _Iteratively applying Transformer blocks therefore can solve kernelized least-squares regression problems with kernel function k_ ( _x, y_ ) = _m_ ( _x_ )<sup>_⊤_</sup> _m_ ( _y_ ) _induced by the MLP m_ ( _·_ ) _._ 

A detailed discussion on this form of kernel regression as well as kernel smoothing w/wo softmax nonlinearity through gradient descent on the data can be found in Appendix A.8. The way MLPs transform data in Transformers diverges from the standard meta-learning approach, where a task-shared _input_ embedding network is optimized by backpropagation-through-training to improve the learning performance of a task-specific readout (e.g., Raghu et al., 2020; Lee et al., 2019; Bertinetto et al., 2019). On the other hand, given our token construction in Proposition 1, MLPs in Transformers intriguingly process both inputs _and_ targets. The output of this transformation is then processed by a sin- 

7 

**Transformers Learn In-Context by Gradient Descent** 

gle linear self-attention layer, which, according to our theory, is capable of implementing gradient descent learning. We compare the performance of this Transformer model, where all weights are learned, to a control Transformer where the final LSA weights are set to the construction _θ_ GD which is therefore identical to training an MLP by backpropagation through a GD updated output layer. 

Intriguingly, both obtained functions show again surprising similarity on (1) the initial (meta-learned) prediction, read out after the MLP, and (2) the final prediction, after altering the output of the MLP through GD or the self-attention layer. This is again reflected in our alignment measures that now, since the obtained models are nonlinear w.r.t. _x_ test, only represent the two first parts of the Taylor approximation of the obtained functions. Our results serve as a first demonstration of how MLPs and self-attention layers can interplay to support nonlinear in-context learning, allowing to fine-tune deep data representations by gradient descent. Investigating the interplay between MLPs and SA-layer in deep TFs is left for future work. 

## **4. Do self-attention layers build regression tasks?** 

The construction provided in Proposition 1 and the previous experimental section relied on a token structure where both input and output data are concatenated into a single token. This design is different from the way tokens are typically built in most of the related work dealing with simple few-shot learning problems as well as in e.g. language modeling. We therefore ask: Can we overcome the assumption required in Proposition 1 and allow a Transformer to build the required token construction on its own? This motivates 

**Proposition 3.** _Given a 1-head linear or softmax attention layer and the token construction e_ 2 _j_ = ( _xj_ ) _, e_ 2 _j_ +1 = (0 _, yj_ ) _with a zero vector_ 0 _of dim Nx − Ny and concatenated positional encodings, one can construct key, query and value matrix WK, WQ, WV as well as the projection matrix P such that all tokens ej are transformed into tokens equivalent to the ones required in Proposition 1._ 

before the Transformer performance jumps to the one of GD, token _ej_ transformed by the first self-attention layer becomes notably dependant on the neighboring token _ej_ +1 while staying independent on the others which we denote as _e_ other in Figure 5. 



<!-- Start of picture text -->
0.55 3.5<br>GD 1 step<br>0.50 TF 2 layers 3.0<br>0.45 2.5<br>0.40 2.0 t(ej)/ ej<br>t(ej)/ ej + 1<br>0.35 1.5 t(ej)/ eother<br>0.30 1.0<br>0.25 0.5<br>0.20 0.0<br>0 10000 20000 30000 40000 0 10000 20000 30000 40000<br>Training steps Training steps<br>Loss<br>Norm part. derivatives<br><!-- End of picture text -->

_Figure 5._ **Training a two layer SA-only Transformer using the standard token construction.** _Left:_ The loss of trained TFs matches one step of GD, not two, and takes an order of magnitude longer to train. _Right_ : Norm of the partial derivatives of the output of the first self-attention layer w.r.t. input tokens. Before the Transformer performance jumps to the one of GD, the first layer becomes highly sensitive to the next token. 

We interpret this as evidence for a copying mechanism of the Transformer’s first layer to merge input and output data into single tokens as required by Proposition 1. Then, in the second layer the Transformer performs a single step of GD. Notably, we were not able to train the Transformer with linear self-attention layers, but had to incorporate the softmax operation in the first layer. These preliminary findings support the study of Olsson et al. (2022) showing that softmax self-attention layers easily learn to copy; we confirm this claim, and further show that such copying allows the Transformer to proceed by emulating gradient-based learning in the second or deeper attention layers. 

We conclude that copying through (softmax) attention layers is the second crucial mechanism for in-context learning in Transformers. This operation enables Transformers to merge data from different tokens and then to compute dot products of input and target data downstream, allowing for in-context learning by gradient descent to emerge. 

## **5. Discussion** 

The construction and its discussion can be found in Appendix A.5. To provide evidence that copying is performed in trained Transformers, we optimize a two-layer self-attention circuit on in-context data where alternating tokens include input or output data i.e. _e_ 2 _j_ = ( _xj_ ) and _e_ 2 _j_ +1 = (0 _, yj_ ). We again measure the loss as well as the mean of the norm of the partial derivative of the first layer’s output w.r.t. the input tokens during training, see Figure 5. First, the training speeds are highly variant given different training seeds, also reported in Garg et al. (2022). Nevertheless, the Transformer is able to match the performance of a _single_ (not two) step gradient descent. Interestingly, 

Transformers show remarkable in-context learning behavior. Mechanisms based on attention, associative memory and copying by induction heads are currently the leading explanations for this remarkable feature of learning within the Transformer forward pass. In this paper, we put forward the hypothesis, similar to Garg et al. (2022) and Akyurek et al.¨ (2023), that Transformer’s in-context learning is driven by gradient descent, in short – _Transformers learn to learn by gradient descent based on their context_ . Viewed through the lens of meta-learning, learning Transformer weights corresponds to the outer-loop which then enables the forward 

8 

#### **Transformers Learn In-Context by Gradient Descent** 



<!-- Start of picture text -->
0.006 0.08<br>0.6 GT GD init Tr. TF init GD Partial cosine<br>Data GD step 1 Tr. TF step 1 0.005 Trained TF 1.0<br>0.4 0.06<br>0.004 0.9<br>0.2<br>0.003 0.04 0.8<br>0.0<br>0.002 0.7<br>0.2 0.02<br>0.001 Preds diff 0.6<br>0.4 Partial diff<br>0.000 0.00 0.5<br>4 2 0 2 4 0 20000 40000 0 20000 40000<br>x Training steps Training steps<br>y<br>Loss<br>L2 Norm<br>Cosine sim<br><!-- End of picture text -->

_Figure 4._ **Sine wave regression: comparing trained Transformers with meta-learned MLPs for which we adjust the output layer with one step of gradient descent.** _Left:_ Plots of the learned initial functions as well as the adjusted functions through either a layer of self-attention or a step of GD. We observe similar initial functions as well as solutions for the trained TF compared fine-tuning a meta-learned MLP. _Center_ : The performance of the trained Transformer is matched by meta-learned MLPs. _Left_ : We observe strong alignment when comparing the prediction as well as the partial derivatives of the the meta-learned MLP and the trained Transformer. 

### pass to transform tokens by gradient-based optimization. 

To provide evidence for this hypothesis, we build on Schlag et al. (2021) that already provide a linear self-attention layer variant with (fast-)inner loop learning by the error-correcting delta rule (Widrow & Hoff, 1960). We diverge from their setting and focus on (in-context) learning where we specifically construct a dataset by considering neighboring elements in the input sequence as input- and target training pairs, see assumptions of Proposition 1. This construction could be realized, for example, due to the model learning to implement a copying layer, see section 4 and proposition 3, and allows us to provide a simple and different construction to Schlag et al. (2021) that solely is built on the standard linear, and approximately softmax, self-attention layer but still implements gradient descent based learning dynamics. We, therefore, are able to explain gradient descent based learning in these standard architectures. Furthermore, we extend this construction based on a single self-attention layer and provide an explanation of how deeper K-layer Transformer models implement principled K-step gradient descent learning, which deviates again from Schlag et al. and allows us to identify that deep Transformers implement GD++, an accelerated version of gradient descent. 

We highlight that our construction of gradient descent and GD++ is not suggestive but when training multi-layer selfattention-only Transformers on simple regression tasks, we provide strong evidence that the construction is actually found. This allows us, at least in our restricted problems settings, to explain mechanistically in-context learning in trained Transformers and its close resemblance to GD observed by related work. Further work is needed to incorporate regression problems with noisy data and weight regularization into our hypothesis. We speculate aspects of learning in these settings are meta-learned – e.g., the weight magnitudes to be encoded in the self-attention weights. Additionally, we did not analyze logistic regression for which one possible weight construction is already presented in Zhmoginov et al. (2022). 

Our refined understanding of in-context learning based on 

gradient descent motives us to investigate how to improve it. We are excited about several avenues of future research. First, to exceed upon a single step of gradient descent in every self-attention layer it could be advantageous to incorporate so called _declarative_ nodes (Amos & Kolter, 2017; Bai et al., 2019; Gould et al., 2021; Zucchet & Sacramento, 2022) into Transformer architectures. This way, we would treat a single self-attention layer as the solution of a fully optimized regression loss leading to possibly more efficient architectures. Second, our findings are restricted to small Transformers and simple regression problems. We are excited to delve deeper into research trying to understand how further mechanistic understanding of Transformers and incontext learning in larger models is possible and to what extend. Third, we are excited about targeted modifications to Transformer architectures, or their training protocols, leading to improved gradient descent based learning algorithms or allow for alternative in-context learners to be implemented within Transformer weights, augmenting their functionality, as e.g. in Dai et al. (2023). Finally, it would be interesting to analyze in-context learning in HyperTransformers (Zhmoginov et al., 2022) that produce weights for target networks and already offer a different perspective on merging Transformers and meta-learning. There, Transformers transform weights instead of data and could potentially allow for gradient computations of weights deep inside the target network lifting the limitation of GD on linear models analyzed here. 

### **Acknowledgments** 

Joao Sacramento and Johannes von Oswald deeply thank˜ Angelika Steger for her support and guidance. The authors also thank Seijin Kobayashi, Marc Kaufmann, Nicolas Zucchet, Yassir Akram, Guillaume Obozinski and Mark Sandler for many valuable insights throughout the project and Dale Schuurmans and Timothy Nguyen for their valuable comments on the manuscript. Joao Sacramento was supported˜ by an Ambizione grant (PZ00P3 ~~1~~ 86027) from the Swiss National Science Foundation and an ETH Research Grant (ETH-23 21-1). 

9 

**Transformers Learn In-Context by Gradient Descent** 

## **References** 

- Akyurek,¨ E., Schuurmans, D., Andreas, J., Ma, T., and Zhou, D. What learning algorithm is in-context learning? investigations with linear models. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum? id=0g0X4H8yN4I. 

- Amos, B. and Kolter, J. Z. Optnet: Differentiable optimization as a layer in neural networks. In _International Conference on Machine Learning_ , 2017. 

- Andrychowicz, M., Denil, M., Gomez, S., Hoffman, M. W., Pfau, D., Schaul, T., Shillingford, B., and de Freitas, N. Learning to learn by gradient descent by gradient descent. In _Advances in Neural Information Processing Systems_ , 2016. 

- Ba, J., Hinton, G. E., Mnih, V., Leibo, J. Z., and Ionescu, C. Using fast weights to attend to the recent past. In _Advances in Neural Information Processing Systems 29_ , 2016. 

- Bai, S., Kolter, J. Z., and Koltun, V. Deep equilibrium models. _Advances in Neural Information Processing Systems_ , 2019. 

- Bengio, Y., Bengio, S., and Cloutier, J. Learning a synaptic learning rule. Technical report, Universite´ de Montreal, D´ epartement d’Informatique et de Recherche´ op´erationnelle, 1990. 

- Benzing, F., Schug, S., Meier, R., von Oswald, J., Akram, Y., Zucchet, N., Aitchison, L., and Steger, A. Random initialisations performing above chance and how to find them. _OPT2022: 14th Annual Workshop on Optimization for Machine Learning_ , 2022. 

- Bertinetto, L., Henriques, J. F., Torr, P. H. S., and Vedaldi, A. Meta-learning with differentiable closed-form solvers. In _International Conference on Learning Representations_ , 2019. 

- Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. _arXiv preprint arXiv:2005.14165_ , 2020. 

- Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., and Zagoruyko, S. End-to-end object detection with transformers. In _Computer Vision – ECCV 2020_ . Springer International Publishing, 2020. 

- Chalmers, D. J. The evolution of learning: an experiment in genetic connectionism. In Touretzky, D. S., Elman, J. L., Sejnowski, T. J., and Hinton, G. E. (eds.), _Connectionist Models_ , pp. 81–90. Morgan Kaufmann, 1991. 

- Chan, S. C. Y., Dasgupta, I., Kim, J., Kumaran, D., Lampinen, A. K., and Hill, F. Transformers generalize differently from information stored in context vs in weights. _arXiv preprint arXiv:2210.05675_ , 2022a. 

- Chan, S. C. Y., Santoro, A., Lampinen, A. K., Wang, J. X., Singh, A., Richemond, P. H., McClelland, J., and Hill, F. Data distributional properties drive emergent in-context learning in transformers. _Advances in Neural Information Processing Systems_ , 2022b. 

- Choromanski, K. M., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J. Q., Mohiuddin, A., Kaiser, L., Belanger, D. B., Colwell, L. J., and Weller, A. Rethinking attention with performers. In _International Conference on Learning Representations_ , 2021. URL https://openreview.net/forum? id=Ua6zuk0WRH. 

- Dai, D., Sun, Y., Dong, L., Hao, Y., Ma, S., Sui, Z., and Wei, F. Why can GPT learn in-context? language models implicitly perform gradient descent as meta-optimizers. In _ICLR 2023 Workshop on Mathematical and Empirical Understanding of Foundation Models_ , 2023. URL https: //openreview.net/forum?id=fzbHRjAd8U. 

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_ , 2021. URL https:// openreview.net/forum?id=YicbFdNTTy. 

- Entezari, R., Sedghi, H., Saukh, O., and Neyshabur, B. The role of permutation invariance in linear mode connectivity of neural networks. _arXiv preprint arXiv:2110.06296_ , 2021. 

- Finn, C. and Levine, S. Meta-learning and universality: Deep representations and gradient descent can approximate any learning algorithm. In _International Conference on Learning Representations_ , 2018. URL https: //openreview.net/forum?id=HyjC5yWCW. 

- Finn, C., Abbeel, P., and Levine, S. Model-agnostic metalearning for fast adaptation of deep networks. In _International Conference on Machine Learning_ , 2017. 

- Flennerhag, S., Rusu, A. A., Pascanu, R., Visin, F., Yin, H., and Hadsell, R. Meta-learning with warped gradient descent. In _International Conference on Learning Representations_ , 2020. 

10 

**Transformers Learn In-Context by Gradient Descent** 

- Garg, S., Tsipras, D., Liang, P., and Valiant, G. What can transformers learn in-context? a case study of simple function classes. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), _Advances in Neural Information Processing Systems_ , 2022. URL https: //openreview.net/forum?id=flNZJ2eOet. 

- Gordon, J., Bronskill, J., Bauer, M., Nowozin, S., and Turner, R. Meta-learning probabilistic inference for prediction. In _International Conference on Learning Representations_ , 2019. URL https://openreview. net/forum?id=HkxStoC5F7. 

- Gould, S., Hartley, R., and Campbell, D. J. Deep declarative networks. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2021. 

- Gulati, A., Qin, J., Chiu, C.-C., Parmar, N., Zhang, Y., Yu, J., Han, W., Wang, S., Zhang, Z., Wu, Y., and Pang, R. Conformer: Convolution-augmented transformer for speech recognition. _arXiv preprint arXiv:2005.08100_ , 2020. 

- Hendrycks, D. and Gimpel, K. Gaussian error linear units (gelus). _arXiv preprint arXiv:1606.08415_ , 2016. 

- Hinton, G. E. and Plaut, D. C. Using fast weights to deblur old memories. 1987. 

- Hochreiter, S., Younger, A. S., and Conwell, P. R. Learning to learn using gradient descent. In Dorffner, G., Bischof, H., and Hornik, K. (eds.), _Artificial Neural Networks — ICANN 2001_ , pp. 87–94, Berlin, Heidelberg, 2001. Springer Berlin Heidelberg. ISBN 978-3-540-44668-2. 

- Hubinger, E., van Merwijk, C., Mikulik, V., Skalse, J., and Garrabrant, S. Risks from learned optimization in advanced machine learning systems. _arXiv [cs.AI]_ , Jun 2019. URL http://arxiv.org/abs/1906. 01820. 

- Irie, K., Schlag, I., Csordas, R., and Schmidhuber, J.´ Going beyond linear transformers with recurrent fast weight programmers. _CoRR_ , abs/2106.06295, 2021. URL https://arxiv.org/abs/2106.06295. 

- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization, 2014. 

- Kirsch, L. and Schmidhuber, J. Meta learning backpropagation and improving it. In Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), _Advances in Neural Information Processing Systems_ , 2021. URL https: //openreview.net/forum?id=hhU9TEvB6AF. 

- Kirsch, L., Harrison, J., Sohl-Dickstein, J., and Metz, L. General-purpose in-context learning by meta-learning transformers. In _Sixth Workshop on Meta-Learning at the_ 

- _Conference on Neural Information Processing Systems_ , 2022. URL https://openreview.net/forum? id=t6tA-KB4dO. 

- Lee, K., Maji, S., Ravichandran, A., and Soatto, S. Metalearning with differentiable convex optimization. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2019. 

- Lee, Y. and Choi, S. Gradient-based meta-learning with learned layerwise metric and subspace. In _International Conference on Machine Learning_ , 2018. 

- Li, Z., Zhou, F., Chen, F., and Li, H. Meta-SGD: Learning to learn quickly for few shot learning. _arXiv preprint arXiv:1707.09835_ , 2017. 

- Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., and Neubig, G. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. _arXiv preprint arXiv:2107.13586_ , 2021. 

- Nadaraya, E. A. On estimating regression. _Theory of Probability & its Applications_ , 9(1):141–142, 1964. 

- Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., and Olah, C. Incontext learning and induction heads. _arXiv preprint arXiv:2209.11895_ , 2022. 

- Park, E. and Oliva, J. B. Meta-curvature. In _Advances in Neural Information Processing Systems_ , 2019. 

- Power, A., Burda, Y., Edwards, H., Babuschkin, I., and Misra, V. Grokking: Generalization beyond overfitting on small algorithmic datasets. abs/2201.02177, 2022. 

- Raghu, A., Raghu, M., Bengio, S., and Vinyals, O. Rapid learning or feature reuse? Towards understanding the effectiveness of MAML. In _International Conference on Learning Representations_ , 2020. 

- Ramsauer, H., Schafl,¨ B., Lehner, J., Seidl, P., Widrich, M., Adler, T., Gruber, L., Holzleitner, M., Pavlovic, M.,´ Sandve, G. K., Greiff, V., Kreil, D., Kopp, M., Klambauer, G., Brandstetter, J., and Hochreiter, S. Hopfield networks is all you need. _arXiv preprint arXiv:2008.02217_ , 2020. 

- Rusu, A. A., Rao, D., Sygnowski, J., Vinyals, O., Pascanu, R., Osindero, S., and Hadsell, R. Meta-learning with latent embedding optimization. In _International Conference on Learning Representations_ , 2019. 

- Schlag, I., Irie, K., and Schmidhuber, J. Linear transformers are secretly fast weight programmers. In _ICML_ , 2021. 

11 

**Transformers Learn In-Context by Gradient Descent** 

- Schmidhuber, J. _Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta-... hook_ . Diploma thesis, Institut fur Informatik, Technische¨ Universit¨at M¨unchen, 1987. 

- Schmidhuber, J. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. _Neural Computation_ , 4(1):131–139, 1992. doi: 10.1162/neco. 1992.4.1.131. 

- Thrun, S. and Pratt, L. _Learning to learn_ . Springer US, 1998. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need, 2017. 

- von Oswald, J., Zhao, D., Kobayashi, S., Schug, S., Caccia, M., Zucchet, N., and Sacramento, J. Learning where to learn: Gradient sparsity in meta and continual learning. In _Advances in Neural Information Processing Systems_ , 2021. 

- Watson, G. S. Smooth regression analysis. _Sankhya:¯ The Indian Journal of Statistics, Series A_ , pp. 359–372, 1964. 

- Widrow, B. and Hoff, M. E. Adaptive switching circuits. In _1960 IRE WESCON Convention Record, Part 4_ , pp. 96–104, New York, 1960. IRE. 

- Yun, S., Jeong, M., Kim, R., Kang, J., and Kim, H. J. Graph transformer networks. In Wallach, H., Larochelle, H., Beygelzimer, A., dAlch<sup>´</sup> e-Buc, F., Fox, E., and Garnett,´ R. (eds.), _Advances in Neural Information Processing Systems_ , 2019. 

- Zhang, A., Lipton, Z. C., Li, M., and Smola, A. J. Dive into deep learning. _arXiv preprint arXiv:2106.11342_ , 2021. 

- Zhao, D., Kobayashi, S., Sacramento, J., and von Oswald, J. Meta-learning via hypernetworks. In _NeurIPS Workshop on Meta-Learning_ , 2020. 

- Zhmoginov, A., Sandler, M., and Vladymyrov, M. HyperTransformer: Model generation for supervised and semi-supervised few-shot learning. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), _Proceedings of the 39th International Conference on Machine Learning_ , volume 162 of _Proceedings of Machine Learning Research_ , pp. 27075–27098. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr. press/v162/zhmoginov22a.html. 

- Zucchet, N. and Sacramento, J. Beyond backpropagation: bilevel optimization through implicit differentiation and equilibrium propagation. _Neural Computation_ , 34(12), December 2022. 

12 

**Transformers Learn In-Context by Gradient Descent** 

## **A. Appendix** 

### **A.1. Proposition 1** 

First, we highlight the dependency on the tokens _ei_ of the linear self-attention operation 



with _⊗_ the outer product between two vectors. With this we can now easily draw connections to one step of gradient descent on _L_ ( _W_ ) = 21 _N_ � _Ni_ =1<sup>_∥Wxi −yi∥_2 with learning rate</sup><sup>_η_which yields weight change</sup> 



### We first restate 

**Proposition 1.** _Given a 1-head linear attention layer and the tokens ej_ = ( _xj, yj_ ) _, for j_ = 1 _, . . . , N , one can construct key, query and value matrices WK, WQ, WV as well as the projection matrix P such that a Transformer step on every token ej is identical to the gradient-induced dynamics ej ←_ ( _xj, yj_ ) + (0 _, −_ ∆ _Wxj_ ) = ( _xi, yi_ ) + _P V K_<sup>_T_</sup> _qj such that ej_ = ( _xj, yj −_ ∆ _yj_ ) _. For the test data token_ ( _xN_ +1 _, yN_ +1) _the dynamics are identical._ 

_Ix_ 0 We provide the weight matrices in block form: _WK_ = _WQ_ = 0 0 with _Ix_ and _Iy_ the identity matrices of size _Nx_ and � � 0 0 _Ny_ respectively. Furthermore, we set _WV_ = � _W_ 0 _−Iy_ � with the weight matrix _W_ 0 _∈_ R<sup>_Ny×Nx_</sup> of the linear model we wish to train and _P_ =<sup>_<u>η</u>_with identity matrix of size</sup><sup>_Nx_+</sup><sup>_Ny_.With this simple construction we obtain the following</sup> _N_<sup>_I_</sup> dynamics 



for every token _ej_ = ( _xj, yj_ ) including the query token _eN_ +1 = _e_ test = ( _x_ test _, −W_ 0 _x_ test) which will give us the desired result. 

### **A.2. Comparing the out-of-distribution behavior of trained Transformers and GD** 

We provide more experimental results when comparing GD with tuned learning rate _η_ and data transformation scalar _γ_ and the trained Transformer on other data distributions than provided during training, see Figure 6. We do so by changing the in-context data distribution and measure the loss of both methods averaged over 10.000 tasks when either changing _α_ that 1) affects the input data range _x ∼ U_ ( _−α, α_ )<sup>_Nx_</sup> or 2) the teacher by _αW_ with _W ∼N_ (0 _, I_ ). This setups leads to results shown in the main text, in the first two columns of Figure 6 and in the corresponding plots of Figure 7. Although the match for deeper architectures starts to become worse, overall the trained Transformers behaves remarkably similar to GD and GD<sup>++</sup> for layer depth greater than 1. 

Furthermore, we try GD and the trained Transformer on input distributions that it never has seen during training. Here, we chose by chance of 1 _/_ 3 either a normal, exponential or Laplace distribution (with JAX default parameters) and depict the average loss value over 10.000 tasks where the _α_ value now simply scales the input values that are sampled from one of the distributions _αx_ . The teacher scaling is identical to the one described above. See for results the two right columns of Figure 6, where we see almost identical behavior for recurrent architectures with less good match for deeper non-recurrent 

13 

**Transformers Learn In-Context by Gradient Descent** 



<!-- Start of picture text -->
(a) Comparing one step of gradient descent with trained one layer Transformers on OOD data.<br><!-- End of picture text -->



<!-- Start of picture text -->
Test on larger inputs Test on larger targets Test on larger inputs Test on larger targets<br>GD 10 1 10 0<br>Interpolated<br>Trained TF<br>10 0 10 0<br>10 0<br>10 1<br>GD 10 1 GD GD<br>10 1 10 1 InterpolatedTrained TF 10 2 InterpolatedTrained TF 10 2 InterpolatedTrained TF<br>0.5 1.0 1.5 2.0 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5<br>   where x   U( , ) W   where W   N(0, I)    where  x W   where W   N(0, I)<br>(b) Comparing two steps of gradient descent with trained  recurrent  two layer Transformers on OOD data.<br>Test on larger inputs Test on larger targets Test on larger inputs Test on larger targets<br>GD GD<br>10 1 Interpolated 10 1 GD + +<br>Trained TF 10 0 10 0 InterpolatedTrained TF 10 1<br>10 0<br>10 1 GDGD + + 10 1 GDGD + +<br>10 1 Interpolated 10 2 Interpolated<br>Trained TF 10 2 Trained TF<br>10 2<br>0.5 1.0 1.5 2.0 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5<br>   where x   U( , ) W   where W   N(0, I)    where  x W   where W   N(0, I)<br>(c) Comparing five steps of gradient descent with trained five layer Transformers on OOD data.<br>Test on larger inputs Test on larger targets Test on larger inputs Test on larger targets<br>10 1<br>GD GD<br>10 1 GD + + 10 1 GD + + 10 0<br>Trained TF 10 0 Trained TF<br>10 0 10 0 10 1<br>10 1 10 1 10 1 10 2<br>GD GD<br>10 2 10 2 GDTrained TF + + 1010 23 10 3 GDTrained TF + +<br>0.5 1.0 1.5 2.0 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5<br>   where x   U( , ) W   where W   N(0, I)    where  x W   where W   N(0, I)<br>Loss Loss Loss Loss<br>Loss Loss Loss Loss<br>Loss Loss Loss Loss<br><!-- End of picture text -->

_Figure 6. Left & center left column:_ Comparing Transformers, GD and their weight interpolation on rescaled training distributions. In all setups, the trained Transformer behaves remarkably similar to GD or GD<sup>++</sup> . _Right & center right_ : Comparing Transformers, GD and their weight interpolation on data distributions never seen during training. Again, in all setups, the trained Transformer behaves remarkably similar to GD or GD<sup>++</sup> with less good match for deep non-recurrent Transformers far away from training regimes. 

architectures far away from the training range of _α_ = 1. Note that for deeper Transformers ( _K >_ 2) the corresponding GD and GD<sup>++</sup> version, see for more experimental details Appendix section A.12, we include a harsh clipping of the token values after every step of transformation between [ _−_ 10 _,_ 10] (for the trained TF and GD) to improve training stability. Therefore, the loss increase is restricted to a certain value and plateaus. 

### **A.3. Linear mode connectivity between the weight construction of Prop 1 and trained Transformers** 

In order to interpolate between the construction _θ_ GD and the trained weights of the Transformer _θ_ , we need to correct for some scaling ambiguity. For clarification, we restate here the linear self-attention operation for a single head 



Now, to match the weight construction of Prop. 1 we have the aim for the matrix product _WKQ_ to match an identify matrix (except for the last diagonal entry) after re-scaling. Therefore we compute the mean of the diagonal of the matrix product of the trained Transformer weights _WKQ_ which we denote by _β_ . After resealing both operations i.e. _WKQ ← WKQ/β_ and _WP V ← WP V β_ we interpolate linearly between the matrix products of GD as well as these rescaled trained matrix products i.e. _WI,KQ_ = ( _WGD,KQ_ + _WT F,KQ_ ) _/_ 2 as well as _WI,P V_ = ( _WGD,P V_ + _WT F,P V_ ) _/_ 2. We use these parameters to obtain results throughout the paper denote with _Interpolated_ . We do so for GD as well as GD<sup>++</sup> when comparing to 

14 

**Transformers Learn In-Context by Gradient Descent** 



<!-- Start of picture text -->
0.4 GD vs trained TF GD + +  vs trained TF<br>GD 2.0 1.05 2.0 1.05<br>GD + + Model cos Model cos<br>0.3 Trained TF 1.5 1.00 1.5 1.00<br>0.2 1.0 Preds diff 0.95 1.0 Preds diff 0.95<br>Model diff Model diff<br>0.90 0.90<br>0.1 0.5 0.5<br>0.85 0.85<br>0.0 0.80 0.0 0.80<br>0 5000 10000 15000 0 5000 10000 15000 0 5000 10000 15000<br>Training steps Training steps Training steps<br>Test on larger inputs Test on larger targets Test on larger inputs Test on larger targets<br>GD GD<br>10 1 Interpolated 10 1 GD + +<br>Trained TF 10 0 10 0 InterpolatedTrained TF 10 1<br>10 0<br>10 1 GDGD + + 10 1 GDGD + +<br>10 1 Interpolated 10 2 Interpolated<br>Trained TF 10 2 Trained TF<br>10 2<br>0.5 1.0 1.5 2.0 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5<br>   where x   U( , ) W   where W   N(0, I)    where  x W   where W   N(0, I)<br>Loss<br>L2 Norm L2 Norm<br>Cosine sim Cosine sim<br>Loss Loss Loss Loss<br><!-- End of picture text -->

_Figure 7._ **Comparing ten steps of gradient descent with trained** **_recurrent_ ten-layer Transformers.** Results comparable to recurrent Transformer with two layers, see Figure 3, but now with 10 repeated layers. We again observe for deeper recurrent linear self-attention only Transformers that overall GD<sup>++</sup> and the trained Transformer align very well with one another and are again interpolatable leading to very similar behavior insight as well as outside training situations. Note the inferior performance to the non-recurrent five-layer Transformer which highlights the importance on specific learning rate as well _γ_ parameter per layer/step. 







_Figure 8._ **Comparing twelve steps of GD**<sup>++</sup> **with a trained twelve-layer Transformers with MLPs and 4 headed linear self-attention layer.** Results comparable to the deep recurrent Transformer, see Figure 7, but now with 12 independent Transformer blocks including MLPs and 4-head linear self-attention. We omit LayerNorm. We again observe a close resemblance of the trained Transformers and GD<sup>++</sup> . We hypotheses that even when equipped with multiple heads and MLPs, Transformers approximate GD<sup>++</sup> . 

15 

#### **Transformers Learn In-Context by Gradient Descent** 



<!-- Start of picture text -->
Weights of WK TWV<br><!-- End of picture text -->









<!-- Start of picture text -->
Weights of WK TWV<br><!-- End of picture text -->







_Figure 9._ **Visualizing the weight matrices of trained Transformers** . _Left & outer left:_ Weight matrix products of a trained single linear self-attention layer. We see (after scalar correction) a perfect resemblance of our construction. _Right & outer right:_ Weight matrix products of a trained 3-layer recurrent linear self-attention Transformer. Again, we see (after scalar correction) a perfect resemblance of our construction and an additional curvature correction i.e. diagonal values in _PWV_ of the same magnitude except the last entry that functions as the learning rate. 

recurrent Transformers. Note that for non-recurrent Transformers, we face more ambiguity that we have to correct for since e.g. scalings influence each other across layer. We also see this in practice and are not able (only for some seeds) to interpolate between weights with our simple correction from above. We leave the search for more elaborate corrections for future work. 

### **A.4. Visualizing the trained Transformer weights** 

The simplicity of our construction enables us to visually compare trained Transformers and the construction put forward in Proposition A.1 in weight space. As discussed in the previous section A.3 there is redundancy in the way the trained Transformer can construct the matrix products leading to the weights corresponding to gradient descent. We therefore visualize _WKQ_ = _WK_<sup>_TWQ_as well as</sup><sup>_WP V_=</sup><sup>_PKWV_in Figure 9.</sup> 

### **A.5. Proof and discussion of Proposition 3** 

We state here again Proposition 3, provide the necessary construction and a short discussion. 

**Proposition 3.** _Given a 1-head linear- or softmax attention layer and the token construction e_ 2 _j_ = ( _xj_ ) _, e_ 2 _j_ +1 = (0 _, yj_ ) _with a zero vector_ 0 _of dim Nx − Ny and concatenated positional encodings, one can construct key, query and value matrix WK, WQ, WV as well as the projection matrix P such that all tokens ej are transformed into tokens equivalent to the ones required in proposition 1._ 

To get a simple and clean construction, we choose wlog _xj ∈_ R<sup>2</sup><sup>_N_+1</sup> and (0 _, yj_ ) _∈_ R<sup>2</sup><sup>_N_+1</sup> as well as model the positional encodings as unit vectors _pj ∈_ R<sup>2</sup><sup>_N_+1</sup> and concatenate them to the tokens i.e. _ej_ = ( _xj/_ 2 _, pj_ ). We wish for a construction that realizes 



This means that a token replaces its own positional encoding by coping the target data of the next token to itself leading to _ej_ = ( _xj/_ 2 _,_ 0 _, yj/_ 2+1), with slight abusive of notation. This can simply be realized by (for example) setting _P_ = _I_ , 0 0 0 0 0 0 _WV_ = � _Ix −Ix,off_ � _, WK_ = �0 _Ix_ � and _WQ_ = �0 _Ix,off_<sup>_T_</sup> � with _Ix,off_ the lower diagonal identity matrix fo size _Nx_ . Note that then simply _K_<sup>_T_</sup> _WQej_ = _pj_ +1 i.e. it chooses the _j_ + 1 element of _V_ which stays _pj_ +1 if we apply the softmax operation on _K_<sup>_T_</sup> _qj_ . Since the _j_ + 1 entry of _V_ is (0 _, yj/_ 2+1 _− pj_ ) we obtain the desired result. 

For the (toy-)regression problems considered in this manuscript, the provided result would give _N/_ 2 tokens for which we also copy (parts) of _xj_ underneath _yj_ . This is desired for modalities such as language where every two tokens could be considered an in-and output pair for the implicit autoregressive inner-loop loss. These tokens do not have be necessarily next to each other, see for this behavior experimental findings presented in (Olsson et al., 2022). For the experiments conducted here, one solution is to zero out these tokens which could be constructed by a two-head self-attention layer that given uneven _j_ simply subtracts itself resulting in a zero token. For all even tokens, we use the construction from above which effectively coincides with the token construction required in Proposition 1. 

16 

**Transformers Learn In-Context by Gradient Descent** 

### **Rolling out experiment with different dampening strength** 



<!-- Start of picture text -->
Dampening = 1 Dampening = 0.875 Dampening = 0.75<br>GD GD GD<br>0.2 0.2 0.2<br>Trained TF Trained TF Trained TF<br>0.1 0.1 0.1<br>0.0 0.0 0.0<br>0 10 20 30 40 50 0 10 20 30 40 50 0 20 40<br>GD Steps / Transformer Layers GD Steps / Transformer Layers GD Steps / Transformer Layers<br>Loss Loss Loss<br><!-- End of picture text -->

_Figure 10._ **Roll-out experiments: applying a trained single linear self-attention layer multiple times.** We observe that different dampening strengths affect the generalization of both methods with slightly better robustness for GD which matching performance for 50 steps when _λ_ = 0 _._ 75. 

### **A.6. Dampening the self-attention layer** 

As an additional out-of-distribution experiment, we test the behavior when repeating a single LSA-layer trained to lower our objective, see equation 5, with the aim to repeat the learned learning/update rule. Note that GD as well as the selfattention layer were optimized to be optimal for one step. For GD we line search the otpimal learning rate _η_ on 10.000 task. Interestingly, for both methods we observe quick divergence when applied multiple times, see left plot of Figure 10. Nevertheless, both of our update functions are described by a linear self-attention layer for which we can control the norm, post training, by a simple scale which we denote as _λ_ . This results in the new update _y_ test + _λ_ ∆ _Wx_ test for GD and _y_ test + _λPV K_<sup>_T_</sup> _WQx_ test for the trained self-attention layer which effectively re-tunes the learning rate for GD and the trained self-attention layer. Intriguingly, both methods do generalize similarly well (or poorly) on this out-of-distribution experiment when changing _λ_ , see again Figure 10. We show in Figure 1 the behavior for _λ_ = 0 _._ 75 for which we see both methods steadily decreasing the loss within 50 steps. 

### **A.7. Sine wave regression** 

For the sine wave regression tasks, we follow (Finn et al., 2017) and other meta-learning literature and sample for each task an amplitude _a ∼ U_ (0 _._ 1 _,_ 5) and a phase _ρ ∼ U_ (0 _, π_ ). Each tasks consist of _N_ = 10 data points where inputs are sampled _x ∼ U_ ( _−_ 5 _,_ 5) and targets computed by _y_ = _a_ sin( _ρ_ + _x_ ). We choose here for the first time, for GD as well as for the Transformer, an input embedding emb that maps tokens _ei_ = ( _xi, yi_ ) into a 40 dimensional space emb( _ei_ ) = _W_ emb _ei_ through an affine projection without bias. We skip the first self-attention layer but, as usually done in Transformers, then transform the embedded tokens through an MLP _m_ with a single hidden layer, widening factor of 4 (160 hidden neuros) and GELU nonlinearity (Hendrycks & Gimpel, 2016) i.e. _ej ← m_ (emb( _ej_ )) + emb( _ej_ ). 

We interpret the last entry of the transformed tokens as the (transformed) targets and the rest as a higher-dimensional input data representation on which we train a model with a single gradient descent step. We compare the obtained meta-learned GD solution with training a Transformer on the same token embeddings but instead learn a self-attention layer. Note that the embeddings of the tokens, including the transformation through the MLP, are not dependent on an interplay between the tokens. Furthermore, the initial transformation is dependent on _ei_ = ( _xi, yi_ ), i.e., input as well as on the target data except for the query token for which _y_ test = 0. This means that this construction is, except for the additional dependency on targets, close to a large corpus of meta-learning literature that aims to find a deep representation optimized for (fast) fine tuning and few-shot learning. In order to compare the meta-training of the MLP and the Transformer, we choose the same seed to initialize the network weights for the MLPs and the input embedding trained by meta-learning i.e. backprop through training or the Transformer. This leads to the plots and almost identical learned initial function and updated functions shown in Figure 4. 

### **A.8. Proposition 2 and connections between gradient descent, kernelized regression and kernel smoothing** 

Let’s consider the data transformation induced by an MLP _m_ ˜ ( _x_ ) and a residual connection commonly used in Transformer blocks i.e. _ej ← ej_ + _m_ ˜ ( _ej_ ) = ( _xj, yj_ ) + ( ˜ _m_ ( _xj_ ) _,_ 0) = ( _m_ ( _xj_ ) _, yj_ ) with _m_ ( _xj_ ) = _xj_ + _m_ ˜ ( _xj_ ) and _m_ ˜ not changing the targets _y_ . When simply applying Proposition 1, it is easy to see that given this new token construction, a linear self-attention layer can induce the token dynamics _ej ←_ ( _m_ ( _xj_ ) _, yj_ )+(0 _, −_ ∆ _Wm_ ( _xj_ )) with ∆ _W_ = _−η∇L_ ( _W_ ) given the loss function _L_ ( _W_ ) = 21 _N_ � _Ni_ =1<sup>_||Wm_(</sup><sup>_xi_)</sup><sup>_−yi||_2.</sup> 

17 

**Transformers Learn In-Context by Gradient Descent** 

Interestingly, for the test token _e_ test = ( _x_ test _,_ 0) this induces, after a multiplication with _−_ 1, an initial prediction after a single Transformer block given by 



with _m_ ( _xi_ )<sup>_T_</sup> _m_ ( _x_ test) = _k_ ( _xi, x_ test) _∈_ R interpreted as a kernel function. Concluding, we see that the combination of MLPs and a _single_ self-attention layer can lead to dynamics induced when descending a kernelized regression (squared error) loss with a _single_ step of gradient-descent. 

Interestingly, when choosing _W_ 0 = 0, we furthermore see that a single self-attention layer or Transformer block can be ˆ regarded as doing nonparametric kernel smoothing _y_ =<sup>�</sup><sup>_N_</sup> _i_ =1<sup>_yik_(</sup><sup>_xi, x_test) based on the data given in-context (Nadaraya,</sup> 1964; Watson, 1964). Note that we made a particular choice of kernel function here and that this view still holds when _m_ ( _xj_ ) = 1 i.e. consider Transformers without MLPs or leverage the well-known view of softmax self-attention layer as a kernel function used to measure similarity between tokens (e.g. Choromanski et al., 2021; Zhang et al., 2021). Thus, implementing one step of gradient descent through a self-attention layer (w/wo softmax nonlinearity) is equivalent to performing kernel smoothing estimation. We however argue that this nonparametric kernel smoothing view of in-context learning is limited, and arises from looking only at a _single_ self-attention layer. When considering deeper Transformer architectures, we see that multiple Transformer blocks can iteratively transform the targets based on multiple steps of gradient descent leading to minimization of a kernelized squared error loss _L_ ( _W_ ). One way to obtain a suitable construction is by neglecting MLPs everywhere except in the first Transformer block. We leave the study of the exact mechanics, especially how the Transformer makes use of possibility transforming the targets through the MLPs, and the possibility of iteratively changing the kernel function throughout depth for future study. 

### **A.9. Linear vs. softmax self-attention as well LayerNorm Transformers** 

Although linear Transformers and their variants have been shown to be competitive with their softmax counterpart (Irie et al., 2021), the removal of this nonlinearity is still a major departure from classic Transformers and more importantly from the Transformers used in related studies analyzing in-context learning. In this section we investigate whether and when gradient-based learning emerges in trained softmax self-attention layers, and we provide an analytical argument to back our findings. 

First, we show, see Figure 12, that a single layer of softmax self-attention is not able to match GD performance. We tuned the learning rate as well as the weight initialization but found no significant difference over the hyperparameters we used througout this study. In general, we hypothesize that GD is an optimal update given the limited capacity of a single layer of (single-head) self-attention. We therefore argue that the softmax induces (at best) a linear offset of the matrix product of training data and query vector 







proportional to a factor dependent on all _{xτ,i}_<sup>_N_</sup> _i_ =1<sup>+1.We speculate that the dependency on the specific task</sup><sup>_τ_, for large</sup> _Nx_ vanishes or that the _x_ -dependent value matrix could introduce a correcting effect. In this case the softmax operation introduces an additive error w.r.t. to the optimal GD update. To overcome this disadvantageous offset, the Transformer can (approximately) introduce a correction with a second self-attention head by a simple subtraction i.e. 



18 

**Transformers Learn In-Context by Gradient Descent** 



<!-- Start of picture text -->
1 W 1, KQ<br><!-- End of picture text -->





<!-- Start of picture text -->
2 W 2, KQ<br><!-- End of picture text -->





<!-- Start of picture text -->
1 W 1, KQ + 2 W 2, KQ<br><!-- End of picture text -->



_Figure 11._ **Visualizing the correction to the softmax operation when training Transformers on regression tasks.** The left and center plot show the matrix product _WKQ_ = _WK_<sup>_TWQ_including its scaling by</sup><sup>_η_induced through</sup><sup>_PWV_of the two heads of the trained softmax</sup> self-attention layer. We observe that both of the matrices are approximate diagonal almost perfect sign reversed values on the off-diagonal terms. After adding the matrices (right plot), we observe a diagonal matrix and therefore to much improved approximation of our construction and therefore gradient descent dynamics. 

Here we assume that _PV_ 1) subsumes the dividing factor of the softmax and that 2) is the same (up to scaling) for each head. Note that if ( _W_ 1 _,KQ − W_ 2 _,KQ_ ) is diagonal, and _P_ and _V_ chosen as in the Proposition of Appendix A.1, we recover our gradient descent construction. 

We base this derivation on empirical findings, see Figure 12, that, first of all, show the softmax self-attention performance increases drastically when using two heads instead of one. Nevertheless, the self-attention layer has difficulties to match the loss values of a model trained with GD. Furthermore, this architecture change leads to a very much improved alignment of the trained model and GD. Second, we can observe that when training a two-headed softmax self-attention layer on regression tasks the correction proposed above is actually observed in weight space, see Figure 11. Here, we visualize the matrix product within the softmax operation _Wh,KQ_ per head which we scale with the last diagonal entry of _PhWh,V_ which we denote by _ηh_ = _PhWh,V_ ( _−_ 1 _, −_ 1). Intriguingly, this results in an almost perfect cancellation (right plot) of the off-diagonal terms and therefore in sum to an improved approximation of our construction, see the derivation above. 

We would like to reiterate that the stronger inductive bias for copying data of the softmax layer remains, and is not invalidated by the analysis above. Therefore, even for our shallow and simple constructions they indeed fulfill an important role in support for our hypotheses: The ability to merge or copy input and target data into single tokens allowing for their dot product computation necessary for the construction in Proposition 1, see Section 4 in the main text. 

We end this section by analysing Transformers equipped with LayerNorm which we apply as usually done before the self-attention layer: Overall, we observe qualitatively similar results to Transformers with softmax self-attention layer i.e. a decrease in performance compared to GD accompanied with a decrease in alignment between models generated by the Transformer and models trained with GD, see Figure 14. Here, we test again a single linear self-attention layer succeeding LayerNorm as well as two layers where we skip the first LayerNorm and only include a LayerNorm between the two. Including more heads does not help substantially. We again assume the optimality of GD and argue that information of targets and inputs present in the tokens is lost by averaging when applying LayerNorm. This naturally leads to decreasing performance compared to GD, see first row of Figure 14. Although the alignment to GD and GD<sup>++</sup> , especially for two layers, is high, we overall see inferior performance to one or two steps of GD or two steps of GD<sup>++</sup> . Nevertheless, we speculate that LayerNorm might not only stabilize Transformer training but could also act as some form of data normalization procedure that implicitly enables better generalization for larger inputs as well as targets provided in-context, see OOD experiments in Figure 14. 

Overall we conclude that common architecture choices like softmax and LayerNorm seem supoptimal for the constructed in-context learning settings when comparing to GD or linear self-attention. Nevertheless, we speculate that the potentially small performance drops of in-context learning are negligible when turning to deep and wide Transformers for which these architecture choices have empirically proven to be superior. 

19 

**Transformers Learn In-Context by Gradient Descent** 

### **(a) Comparing one step of GD with a trained** **_softmax one_ -headed self-attention layer.** 



<!-- Start of picture text -->
0.40 3.5 Test on larger inputs Test on larger targets<br>GD Preds diff Model cos 10 1<br>Trained TF 3.0 Model diff 1.0 GD<br>0.35 2.5 0.8 Trained TF<br>2.0 10 0 10 0<br>0.30 0.6<br>1.5<br>0.4<br>0.25 1.0<br>0.5 0.2 10 1 10 1 GDTrained TF<br>0.20 0.0 0.0<br>0 2000 4000 0 1000 2000 3000 4000 5000 0.5 1.0 1.5 2.0 1 2 3 4 5<br>Training steps Training steps    where x   U( , ) W   where W   N(0, I)<br>(b) Comparing one step of GD with a trained  softmax two -headed self-attention layer.<br>0.40 3.5 Test on larger inputs Test on larger targets<br>GD Preds diff Model cos<br>Trained TF 3.0 Model diff 1.0 GD<br>0.35 2.5 0.8 Trained TF<br>2.0 10 0 10 0<br>0.30 0.6<br>1.5<br>0.4<br>0.25 1.0<br>0.5 0.2 10 1 10 1 GDTrained TF<br>0.20 0.0 0.0<br>0 2500 5000 7500 10000 0 2000 4000 6000 8000 10000 0.5 1.0 1.5 2.0 1 2 3 4 5<br>Training steps Training steps    where x   U( , ) W   where W   N(0, I)<br>Loss Loss Loss<br>L2 Norm<br>Cosine sim<br>Loss Loss Loss<br>L2 Norm<br>Cosine sim<br><!-- End of picture text -->

- **(b) Comparing one step of GD with a trained** **_softmax two_ -headed self-attention layer.** 

_Figure 12._ **Comparing trained two-headed and one-headed single-layer** **_softmax_ self-attention with 1 step of gradient descent on linear regression tasks.** _Left column:_ Softmax self-attention is not able to match gradient descent performance with hand-tuned learning rate, but adding a second attention head significantly reduces the gap, as expected by our analytical argument. _Center left_ : The alignment suffers significantly for single-head softmax SA. We observe good but not as precise alignment when compared to linear Transformers for the two-headed softmax SA layer. _Center right & right:_ The two-headed self-attention compared to the single-head layer shows similar robust out-of-distribution behavior compared to gradient descent. 

### **A.10. Details of curvature correction** 

We give here a precise construction showing how to implement in a single head, a step of GD and the discussed data transformation, resulting in GD<sup>++</sup> . Recall again the linear self-attention operation with a single head 



We provide again the weight matrices in block form of the construction of Prop. 1 but now enabling additionally our _Ix_ 0 described data transformation: _WK_ = _WQ_ = 0 0 with _Ix_ the identity matrix of size _Nx_ , _Iy_ od size _Ny_ resp. � � _Ix_ 0 Furthermore, we set _WV_ = � _W −Iy_ � with the weight matrix _W ∈_ R<sup>_Ny×Nx_</sup> of the linear model we wish to train and _P_ = _−γI_ 0 _x_ 0 _<u>η</u>_ . This leads to the following update � _N_ � 



for every token _ej_ = ( _xj, yj_ ) including the query token _eN_ +1 = _e_ test = ( _x_ test _,_ 0) which will give us the desired result. 

**Why does GD**<sup>++</sup> **perform better?** We give here one possible explanation of the superior performance of GD<sup>++</sup> compared to GD. Note that there is a close resemblance of the GD transformation and a heavily truncated Neuman series approximation of the inverse _XX_<sup>_T_</sup> . We provide here a more heuristic explanation for the observed acceleration. 

Given _γ ∈_ R, GD<sup>++</sup> transforms every input according to _xi ← xi − γXX_<sup>_T_</sup> _xi_ = ( _I − γXX_<sup>_T_</sup> ) _xi_ . We can therefore look _N_ at the change of squared regression loss _L_ ( _W_ ) =<sup><u>1</u></sup> 2 � _i_ =0<sup>(</sup><sup>_Wxi −yi_)2induced by this transformation i.e.</sup><sup>_L_++(</sup><sup>_W_)=</sup> 

20 

**Transformers Learn In-Context by Gradient Descent** 







_Figure 13._ **GD**<sup>++</sup> **analyses** . _Left:_ We visualize the change of the eigenspectrum induced by the input data transformation of GD<sup>++</sup> for different _γ_ observed in practice. _Center:_ Given we know the maximum and minimum of eigenvalues _λ_ 1 _, λn_ of the loss Hessian _XX_<sup>_T_</sup> with _X_ = ( _x_ 0 _, . . . , xN_ ) for different _N_ , we compare the original condition number (depicted by *’s at _γ_ = 0) and the condition number (in log scale) of the GD<sup>++</sup> altered loss Hessian when varying _γ_ . We plot in dotted lines the _γ_ values that we observe in practice which are close the optimal ones i.e. the local minimum derived through our analysis. _Right:_ For _N_ = 25, we plot for different _γ_ values the distribution of condition numbers _κ_ = _λ_ 1 _/λn_ for 10000 tasks and observe favorable _κ_ values close to 1 when approaching the _γ_ = 0 _._ 099 value was found in practice. The _κ_ values quickly explode for _γ >_ 0 _._ 1. 

<u>1</u> _N_ 2 � _i_ =0<sup>(</sup><sup>_W_(</sup><sup>_I −γXXT_)</sup><sup>_xi −yi_)2=</sup><sup><u>1</u></sup> 2<sup>(</sup><sup>_W_(</sup><sup>_I −γXXT_)</sup><sup>_X −Y_)2 which in turn leads to a change of the loss Hessian from</sup> _∇_<sup>2</sup> _L_ = _XX_<sup>_T_</sup> to _∇_<sup>2</sup> _L_<sup>++</sup> = ( _I − γXX_<sup>_T_</sup> ) _X_ (( _I − γXX_<sup>_T_</sup> ) _X_ )<sup>_T_</sup> . 

Given the original Hessian _H_ = _XX_<sup>_T_</sup> = _U_ Σ _U_<sup>_T_</sup> with it’s set of sorted eigenvalues _{λ_ 1 _, . . . , λn}_ and _λi ≥_ 0 on the diagonal matrix Σ we can express the new Hessian through _U,_ Σ i.e. _H_<sup>++</sup> = ( _I − γXX_<sup>_T_</sup> ) _X_ (( _I − γXX_<sup>_T_</sup> ) _X_ )<sup>_T_</sup> = ( _I − γU_ Σ _U_<sup>_T_</sup> ) _U_ Σ _U_<sup>_T_</sup> ( _I − γU_ Σ _U_<sup>_T_</sup> )<sup>_T_</sup> . 

We can simplify _H_<sup>++</sup> further as 



Given the eigenspectrum _{λ_ 1 _, . . . , λn}_ of _H_ , we obtain an (unsorted) eigenspecturm for _H_<sup>++</sup> with _{λ_ 1 _−_ 2 _γλ_<sup>2</sup> 1<sup>+</sup> _γ_<sup>2</sup> _λ_<sup>3</sup> 1<sup>_, . . . , λn−_2</sup><sup>_γλ_2</sup> _n_<sup>+</sup><sup>_γ_2</sup><sup>_λ_3</sup> _n_<sup>_}_whichwevisualizeinFigure13fordifferent</sup><sup>_γ_observedinpractice.Wehypotheses</sup> that the Transformer chooses _γ_ in a way that on average, across the distribution of tasks, the data transformation (iteratively) decreases the condition number _λ_ 1 _/λn_ leading to accelerated learning. This could be achieved, for example, by keeping the smallest eigenvalue _λn ≈ λ_<sup>++</sup> _n_ fixed and choosing _γ_ such that the largest eigenvalue of the transformed data _λ_<sup>++</sup> 1 is reduced, while the original _λ_ 1 stays within [ _λ_<sup>++</sup> 1 _, λ_<sup>++</sup> _n_<sup>].</sup> 

To support our hypotheses empirically, we computed the minimum and maximum eigenvalues of _XX_<sup>_T_</sup> across 10000 tasks while changing the number of datapoints _N ∈_ [10 _,_ 25 _,_ 50 _,_ 100] i.e. _X_ = ( _x_ 0 _, . . . , xN_ ) leading to better conditioned loss Hessians i.e. [1e _−_ 10 _,_ 0 _._ 097 _,_ 0 _._ 666 _,_ 2 _._ 870] and [4 _._ 6 _,_ 7 _._ 712 _,_ 10 _._ 845 _,_ 17 _._ 196] as the minimum and maximum eigenvalues of _XX_<sup>_T_</sup> across all tasks where we cut the smallest eigenvalue for _N_ = 10 at 1e _−_ 10. Furthermore, we extract the _γ_ values from the weights of optimized recurrent 2-layer Transformers trained on different task distributions and obtain _γ_ values of [0 _._ 179 _,_ 0 _._ 099 _,_ 0 _._ 056 _,_ 0 _._ 029], see again Figure 13. Note that the observed eigenvalues stay within [0 _,_ 1 _/γ_ ] i.e. the two roots of _f_ ( _λ, γ_ ) = _λ −_ 2 _γλ_<sup>2</sup> + _γ_<sup>2</sup> _λ_<sup>3</sup> . 

Given the derived function of eigenvalue change _f_ ( _λ, γ_ ), we compute the condition number of _H_<sup>++</sup> by dividing the novel maximum eigenvalues _λ_<sup>++</sup> 1 = _f_ (1 _/_ (3 _γ_ ) _, γ_ ) where _λ_ = 1 _/_ (3 _γ_ ) as the local maximum of _f_ ( _λ, γ_ ), for fixed _γ_ , and the novel minimum eigenvalue _λ_<sup>++</sup> _n_ = min( _f_ ( _λ_ 1 _, γ_ ) _, f_ ( _λn, γ_ )). Note that with too small _γ_ , we move the original _λn_ closer to the root of _f_ ( _λ, γ_ ) i.e. _λ_ = 1 _/γ_ and therefore can change the smallest eigenvalue. 

Given the task distribution and its corresponding eigenvalue distribution, we see that choosing _γ_ reduces the new condition number _κ_<sup>++</sup> = _λ_<sup>++</sup> 1 _/λ_<sup>++</sup> _n_ which leads to better conditioned learning, see center plot of Figure 13. Note that the optimal _γ_ based on our derivation above is based on the maximum and minimum eigenvalue across all tasks and does not take the change of the eigenvalue distribution into account. We argue therefore that the simplicity of the arguments above does not capture the task statistics and distribution shifts entirely and therefore obtains a slightly larger _γ_ as an optimal value. 

21 

**Transformers Learn In-Context by Gradient Descent** 

### **(a) Comparing one step of GD with a single-layer LSA Transformer with LayerNorm.** 



<!-- Start of picture text -->
0.40 3.5 Test on larger inputs Test on larger targets<br>GD Preds diff Model cos 10 1 10 1<br>Trained TF 3.0 Model diff 1.0 GD<br>0.35 2.5 0.8 Trained TF<br>0.30 2.0 0.6 10 0 10 0<br>1.5<br>0.4<br>0.25 1.0<br>0.5 0.2 10 1 10 1 GDTrained TF<br>0.20 0.0 0.0<br>0 5000 10000 15000 0 5000 10000 15000 0.5 1.0 1.5 2.0 1 2 3 4 5<br>Training steps Training steps    where x   U( , ) W   where W   N(0, I)<br>(b) Comparing two steps of GD with a two-layer LSA Transformer with LayerNorm.<br>0.30 GD vs trained TF GD + +  vs trained TF Test on larger inputs<br>GD<br>GD + + Model cos Model cos 1.0 10 3 GD<br>0.25 Trained TF 1.5 1.0 1.5 GD + +<br>0.9 0.9 10 2 Trained TF<br>0.20 1.0 Preds diffModel diff 0.8 1.0 Preds diffModel diff 0.8 10 1<br>0.15 0.5 0.7 0.5 0.7 10 0<br>0.6 0.6 10 1<br>0.10 0.0 0.5 0.0 0.5<br>0 5000 10000 15000 0 5000 10000 15000 0 5000 10000 15000 0.5 1.0 1.5 2.0<br>Training steps Training steps Training steps    where x   U( , )<br>Loss Loss Loss<br>L2 Norm<br>Cosine sim<br>Loss Loss<br>L2 Norm L2 Norm<br>Cosine sim Cosine sim<br><!-- End of picture text -->

_Figure 14._ **Comparing trained 1-layer and 2-layer Transformers with** **_LayerNorm_ and 1 step or 2 steps of gradient descent resp.** _Left column:_ The Transformers is not able to match the gradient descent performance with hand-tuned learning rate. _Alignment plots_ : The alignment suffers significantly when comparing to linear self-attention layers although still reasonable alignment is obtained which decreases slightly when comparing to GD<sup>++</sup> for the two-layer Transformer. _Center right & right:_ The LayerNorm Transformer outperforms when GD when providing training input data that is significantly larger than the data provided during training. 

We furthermore visualize the condition number change for _N_ = 25 and 10000 tasks in the right plot of Figure 13 and observe the distribution moving to desirable _κ_ values close to 1. For _γ_ values larger than 0.1 the distribution quickly exhibits exploding condition numbers. 

### **A.11. Phase transitions** 

We comment shorty on the curiously looking phase transitions of the training loss observed in many of our experiments, see Figure 2. Nevertheless, simply switching from a single-headed self-attention layer to a two-headed self-attention layer mitigates the random seed dependent training instabilities in our experiments presented in the main text, see left and center plot of Figure 15. 

Furthermore, these transitions look reminiscent of the recently observed ”grokking” behaviour (Power et al., 2022). Interestingly, when carefully tuning the learning rate and batchsize we can also make the Transformers trained in these linear regression tasks _grokk_ . For this, we train a single Transformer block (self-attention layer and MLP) on a limited amount of data (8192 tasks), see right plot of Figure 15, and observe grokking like train and test loss phase transitions where test set first increases drastically before experiencing a sudden drop in loss almost matching the desired GD loss of 0 _._ 2. We leave a thorough investigation of these phenomena for future study. 

### **A.12. Experimental details** 

We use for most experiments identical hyperparameters that were tuned by hand which we list here 

- Optimizer: Adam (Kingma & Ba, 2014) with default parameters and learning rate of 0.001 for Transformer with depth _K <_ 3 and 0.0005 otherwise. We use a batchsize of 2048 and applied gradient clipping to obtain gradients with global norm of 10. We used the Optax library. 

- Haiku weight initialisation (fan-in) with truncated normal and std 0 _._ 002 _/K_ where _K_ the number of layers. 

- We did not use any regularisation and observed for deeper Transformers with _K >_ 2 instabilities when reaching GD performance. We speculate that this occurs since the GD performance is, for the given training tasks, already close to divergence as seen when providing tasks with larger input ranges. Therefore, training Transformers also becomes 

22 

**Transformers Learn In-Context by Gradient Descent** 







_Figure 15._ **Phase transitions during training** . _Left:_ Loss based on 10 different random seeds when optimizing a single-headed selfattention layer. We observe for some seeds very long initial phases of virtually zero progress after which the loss drops suddenly to the desired GD loss. _Center:_ The same experiment but optimizing a _two_ -headed self-attention layer. We observe fast and robust convergence to the loss of GD. _Right:_ Training a single Transformer block i.e. a self-attention layer with MLP and a reduced training set size of 8192 tasks. We observe grokking like train and test loss phase transitions where test set first increases drastically before experiencing a sudden drop in loss almost matching the desired GD loss of 0 _._ 2. 

instable when we approach GD with an optimal learning rate. In order to stabilize training, we simply clipped the token values to be in the range of [ _−_ 10 _,_ 10]. 

- When applicable we use standard positional encodings of size 20 which we concatenated to all tokens. 

- For simplicity, and to follow the provided weight construction closely, we did use square key, value and query parameter matrix in all experiments. 

- The training length varied throughout our experimental setups and can be read off our training plots in the article. 

- When training meta-parameters for gradient descent i.e. _η_ and _γ_ we used an identical training setup but usually training required much less iterations. 

- In all experiments we choose inital _W_ 0 = 0 for gradient descent trained models. 

Inspired by (Garg et al., 2022), we additionally provide results when training a single linear self-attention layer on a fixed number of training tasks. Therefore, we iterate over a single fixed batch of size _B_ instead of drawing new batch of tasks at every iteration. Results can be found in Figure 16. Intriguingly, we find that (meta-)gradient descent finds Transformer weights that align remarkable well with the provided construction and therefore gradient descent even when provided with an arguably very small number of training tasks. We argue that this again highlights the strong inductive bias of the LSA-layer to match (approximately) gradient descent learning in its forward pass. 

23 

**Transformers Learn In-Context by Gradient Descent** 



<!-- Start of picture text -->
(a) Comparing 1 step of gradient descent with training a LSA-layer on 128 tasks.<br>5 Test on larger inputs Test on larger targets<br>4 GDTrained TF 108 Model cos 1.0 GDInterpolated<br>3 6 Preds diff 0.8 Trained TF<br>Model diff 0.6<br>2 4 0.4<br>GD<br>1 2 0.2 Interpolated<br>0 0 1000 2000 3000 4000 50000.0 0.1 0.1 Trained TF<br>0 1000 2000 3000 4000 5000 Training steps 0.5 1.0 1.5 2.0 1 2 3 4 5<br>Training steps    where x   U( , ) W   where W   N(0, I)<br>(b) Comparing 1 step of gradient descent with training a LSA-layer on 512 tasks.<br>0.40 Test on larger inputs Test on larger targets<br>GDTrained TF 3.5 Model cos GD<br>0.35 3.0 1.0 Interpolated<br>2.5 0.8 Trained TF<br>0.30 2.01.5 Preds diffModel diff 0.6<br>0.4<br>1.0 GD<br>0.25 0.5 0.2 Interpolated<br>0.1<br>0.0 0.0 0.1 Trained TF<br>0.20 0 2000 4000 0 1000 Training steps2000 3000 4000 5000 0.5 1.0 1.5 2.0 1 2 3 4 5<br>Training steps    where x   U( , ) W   where W   N(0, I)<br>(c) Comparing 1 step of gradient descent with training a LSA-layer on 2048 tasks.<br>0.40 Test on larger inputs Test on larger targets<br>GDTrained TF 3.5 Model cos GD<br>0.35 3.0 1.0 Interpolated<br>2.5 0.8 Trained TF<br>0.30 2.01.5 Preds diffModel diff 0.6<br>0.4<br>1.0 GD<br>0.25 0.5 0.2 0.1 Interpolated<br>0.0 0.0 0.1 Trained TF<br>0.20 0 2000 4000 0 1000 Training steps2000 3000 4000 5000 0.5 1.0 1.5 2.0 1 2 3 4 5<br>Training steps    where x   U( , ) W   where W   N(0, I)<br>(d) Comparing 1 step of gradient descent training a LSA-layer on 8192 tasks.<br>0.40 Test on larger inputs Test on larger targets<br>GDTrained TF 3.5 Model cos GD<br>0.35 3.0 1.0 Interpolated<br>2.5 0.8 Trained TF<br>0.30 2.01.5 Preds diffModel diff 0.6<br>0.4<br>1.0 GD<br>0.25 0.5 0.2 0.1 Interpolated<br>0.0 0.0 0.1 Trained TF<br>0.20 0 2000 4000 0 1000 Training steps2000 3000 4000 5000 0.5 1.0 1.5 2.0 1 2 3 4 5<br>Training steps    where x   U( , ) W   where W   N(0, I)<br>Loss L2 Norm Cosine sim Loss Loss<br>Loss L2 Norm Cosine sim Loss Loss<br>Loss L2 Norm Cosine sim Loss Loss<br>Loss L2 Norm Cosine sim Loss Loss<br><!-- End of picture text -->

_Figure 16._ **Comparing trained Transformers with GD and their weight interpolation when training the Transformer on a fixed training set size** _B_ . Across our alignment measures as well as our tests on out-of-training behaviour, trained Transformers fail to align with GD when provided with a very small amount of tasks. Nevertheless, we see already almost perfect alignment in our base setting _N_ = _Nx_ = 10 when provided with _B >_ 2048 tasks. In all settings, we train the Transformer on (non-stochastic) gradient descent iterating over a single batch of tasks of size _B_ equal to the number provided in the Figure titles (128, 512, 2048, 8192). 

24 

