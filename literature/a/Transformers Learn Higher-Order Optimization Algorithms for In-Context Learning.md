# Transformers Learn to Achieve Second-Order Convergence Rates for In-Context Linear Regression 

**Deqing Fu Tian-Qi Chen Robin Jia Vatsal Sharan** Department of Computer Science University of Southern California 

{deqingfu,tchen939,robinjia,vsharan}@usc.edu 

##### **Abstract** 

Transformers excel at _in-context learning_ (ICL)—learning from demonstrations without parameter updates—but how they do so remains a mystery. Recent work suggests that Transformers may internally run Gradient Descent (GD), a first-order optimization method, to perform ICL. In this paper, we instead demonstrate that Transformers learn to approximate second-order optimization methods for ICL. For in-context linear regression, Transformers share a similar convergence rate as _Iterative Newton’s Method_ ; both are exponentially faster than GD. Empirically, predictions from successive Transformer layers closely match different iterations of Newton’s Method _linearly_ , with each middle layer roughly computing 3 iterations; thus, Transformers and Newton’s method converge at roughly the same rate. In contrast, Gradient Descent converges _exponentially_ more slowly. We also show that Transformers can learn in-context on ill-conditioned data, a setting where Gradient Descent struggles but Iterative Newton succeeds. Finally, to corroborate our empirical findings, we prove that Transformers can implement _k_ iterations of Newton’s method with _k_ + _O_ (1) layers. 

## **1 Introduction** 

Transformer neural networks [Vaswani et al., 2017] have become the default architecture for natural language processing [Devlin et al., 2019, Brown et al., 2020, OpenAI, 2023]. As first demonstrated by GPT-3 [Brown et al., 2020], Transformers excel at _in-context learning_ (ICL)—learning from prompts consisting of input-output pairs, without updating model parameters. Through in-context learning, Transformer-based Large Language Models (LLMs) can achieve state-of-the-art few-shot performance across a variety of downstream tasks [Rae et al., 2022, Smith et al., 2022, Thoppilan et al., 2022, Chowdhery et al., 2022]. 

Given the importance of Transformers and ICL, many prior efforts have attempted to understand how Transformers perform in-context learning. Prior work suggests Transformers can approximate various linear functions well in-context [Garg et al., 2022]. Specifically to linear regression tasks, prior work has tried to understand the ICL mechanism, and the dominant hypothesis is that Transformers learn in-context by running optimization internally through gradient-based algorithms [von Oswald et al., 2022, 2023, Ahn et al., 2023, Dai et al., 2023, Mahankali et al., 2024]. 

This paper presents strong evidence for a competing hypothesis: Transformers trained to perform in-context linear regression learn a strategy much more similar to a second-order optimization 

> Our codes are available at https://github.com/DeqingFu/transformers-icl-second-order. 

1 

method than a first-order method like Gradient Descent (GD). In particular, Transformers approximately implement a second-order method with a convergence rate very similar to Newton-Schulz’s Method, also known as the _Iterative Newton’s Method_ , which iteratively improves an estimate of the inverse of the data matrix to compute the optimal weight vector. Across many Transformer layers, subsequent layers approximately compute more and more iterations of Newton’s Method, with increasingly better predictions; both eventually converge to the optimal minimum-norm solution found by ordinary least squares (OLS). Interestingly, this mechanism is specific to Transformers: LSTMs do not learn these same second-order methods, as their predictions do not even improve across layers. 

We present both empirical and theoretical evidence for our claims. Empirically, Transformer layers demonstrate a similar rate of convergence to the OLS solution as second-order methods such as Iterative Newton, which is substantially faster than the rate of convergence of GD (Figure 2). The predictions made by the Transformer at successive layers closely match the predictions made by Iterative Newton after a proportional number of iterations, showing that they progress in similar ways at the same rate. In contrast, to match the Transformer’s predictions after _k_ layers, GD would have to run for exponential in _k_ many steps (Figure 3). Some individual Transformer layers make progress equivalent to hundreds of GD steps: these layers must be doing something more sophisticated than GD. Furthermore, a crucial aspect of second-order methods is that they can handle ill-conditioned problems by correcting the curvature. We find that the convergence rate of Transformers is not significantly affected by ill-conditioning, which again matches Iterative Newton but not GD. To provide theoretical grounding to our empirical results, we show that Transformer circuits can efficiently implement Iterative Newton: one transformer layer can compute one Newton iteration (given _O_ (1) pre/post-processing layers), and requires hidden states of dimension _O_ ( _d_ ) for a _d_ -dimensional linear regression problem. Overall, our work provides a mechanistic account of how Transformers perform ICL that explains model behavior better than previous hypotheses, and hints at why Transformers are well-suited for ICL relative to other architectures. 

## **2 Related Work** 

**In-context learning by large language models.** GPT-3 [Brown et al., 2020] first showed that Transformer-based large language models can “learn” to perform new tasks from in-context demonstrations (i.e., input-output pairs). Since then, a large body of work in NLP has studied in-context learning, for instance by understanding how the choice and order of demonstrations affects results [Lu et al., 2022, Liu et al., 2022, Rubin et al., 2022, Su et al., 2023, Chang and Jia, 2023, Nguyen and Wong, 2023], studying the effect of label noise [Min et al., 2022c, Yoo et al., 2022, Wei et al., 2023], and proposing methods to improve ICL accuracy [Zhao et al., 2021, Min et al., 2022a,b]. 

**In-context learning beyond natural language.** Inspired by the phenomenon of ICL by large language models, subsequent work has studied how Transformers learn in-context beyond NLP tasks. Garg et al. [2022] first investigated Transformers’ ICL abilities for various classical machine learning problems, including linear regression. We largely adopt their linear regression setup in this work. Li et al. [2023] formalize in-context learning as an algorithm learning problem. Han et al. [2023] suggests that Transformers learn in-context by performing Bayesian inference on prompts, which can be asymptotically interpreted as kernel regression. Other work has analyzed how Transformers do in-context classification [Tarzanagh et al., 2023a,b, Zhang et al., 2023], the role of pertaining data [Raventós et al., 2023], and the relationship between model architecture and ICL [Lee et al., 2023]. 

2 

**Do Transformers implement Gradient Descent?** A growing body of work has suggested that Transformers learn in-context by implementing gradient descent within their internal representations. Akyürek et al. [2022] summarize operations that Transformers can implement, such as multiplication and affine transformations, and show that Transformers can implement gradient descent for linear regression using these operations. Concurrently, von Oswald et al. [2022] argue that Transformers learn in-context via gradient descent, where one layer performs one gradient update. In subsequent work, von Oswald et al. [2023] further argue that Transformers are strongly biased towards learning to implement gradient-based optimization routines. Ahn et al. [2023] extend the work of von Oswald et al. [2022] by showing Transformers can learn to implement preconditioned Gradient Descent, where the pre-conditioner can adapt to the data. Bai et al. [2023] provide detailed constructions for how Transformers can implement a range of learning algorithms via gradient descent. Finally, Dai et al. [2023] conduct experiments on NLP tasks and conclude that Transformer-based language models performing ICL behave similarly to models fine-tuned via gradient descent; however, concurrent work [Shen et al., 2023b] argues that real-world LLMs do not perform ICL via gradient descent. Mahankali et al. [2024] showed that implementing gradient descent is a global minima for single layer linear self-attention. However, we study deeper models in this work, which can behave differently from single-layer models. In this paper, we argue that Transformers actually learn to perform in-context learning by implementing a second-order . optimization method, not gradient descent<sup>1</sup> 

**Mechanistic interpretability for Transformers.** Our work attempts to understand the mechanism through which Transformers perform in-context learning. Prior work has studied other aspects of Transformers’ internal mechanisms, including reverse-engineering language models [Wang et al., 2022], the grokking phenomenon [Power et al., 2022, Nanda et al., 2023], manipulating attention maps [Hassid et al., 2022], and circuit finding [Conmy et al., 2023]. 

**Theoretical Expressivity of Transformers.** Giannou et al. [2023] provide a construction of looped transformers to implement Iterative Newton’s method for solving pseudo-inverse, and each Newton iteration can be implemented by 13 looped Transformer layers. In contrast, our construction needs only one Transformer layer to compute one Newton iteration. 

## **3 Problem Setup** 



<!-- Start of picture text -->
!𝑦! !𝑦" !𝑦# !𝑦#$!<br>Transformers<br>………<br>𝑥𝑥!!⋮!"𝑥!!⋮!"!!⋮!"!⋮!"⋮!" 00⋮0⋮ 𝑥𝑥""⋮!"𝑥""⋮!"""⋮!""⋮!"⋮!" 00⋮0⋮ In-Context ExamplesExamples 𝑥𝑥$$⋮!"𝑥$$⋮!"$$⋮!"$⋮!"⋮!" 00⋮0⋮ 𝑥𝑥$%!$%!⋮!"𝑥$%!$%!⋮!"$%!$%!⋮!"$%!⋮!"⋮!" 00⋮0⋮<br># # # #<br>𝑥!!# 𝑦!! 𝑥""# 𝑦"" 𝑥$$# 𝑦## 𝑥$%!$%!# 𝑦#$!#$!<br><!-- End of picture text -->

In this paper, we focus on the following linear regres! " # #$! sion task. The task involves _n_ examples _{_ **_x_** _i, yi}_<sup>_n_</sup> _i_ =1 where **_x_** _i ∈_ R<sup>_d_</sup> and _yi ∈_ R. The examples are generTransformers ated from the following data generating distribution _PD_ , parameterized by a distribution _D_ over ……… ( _d × d_ ) positive semi-definite matrices. For each sequence of _n_ in-context examples, we first sample a !"" ⋮ !"" ⋮ !"" ⋮ !"" ⋮ 𝑥𝑥!!⋮!"𝑥!!⋮!"!!⋮!"!⋮!"⋮!" 00⋮0⋮ 𝑥𝑥""⋮!"𝑥""⋮!"""⋮!""⋮!"⋮!" 00⋮0⋮ In-Context ExamplesExamples 𝑥𝑥$$⋮!"𝑥$$⋮!"$$⋮!"$⋮!"⋮!" 00⋮0⋮ 𝑥𝑥$%!$%!⋮!"𝑥$%!$%!⋮!"$%!$%!⋮!"$%!⋮!"⋮!" 00⋮0⋮ ground-truth weight vector **_w_**<sup>_⋆_i.i.d.</sup> _∼N_ ( **0** _,_ **_I_** ) _∈_ R<sup>_d_</sup> 𝑥!!# 𝑦!! 𝑥""# 𝑦"" 𝑥$$# 𝑦## 𝑥$%!$%!# 𝑦#$!#$! and a matrix **Σ**<sup>i.i.d.</sup> _∼D_ . For _i ∈_ [ _n_ ], we sample each i.i.d. Figure 1: Illustration of how Transformers **_x_** _i ∼N_ ( **0** _,_ **Σ** ). The label _yi_ for each **_x_** _i_ is given by _yi_ = **_w_**<sup>_⋆⊤_</sup> **_x_** _i_ . Note that for much of our experiments are trained to do in-context linear regression. _D_ is only supported on the identity matrix **_I_** and hence **Σ** = **_I_** , but we also consider some distribu- 

1After an initial version of this paper, Vladymyrov et al. [2024] found that a variant of Gradient Descent can mimic Iterative Newton by approximating the inverse implicitly and getting second-order rates, which also supports our claim. 

3 

tions over ill-conditioned matrices, which give rise to ill-conditioned regression problems. Most of our results are on this noiseless setup and results with the noisy setup are in Appendix A.3.2. 

### **3.1 Standard Methods for Solving Linear Regression** 

Our central research question is: 

#### **_What convergence rate does the algorithm Transformers learn for linear regression achieve?_** 

To investigate this question, we first discuss various known algorithms for linear regression. We then compare them with Transformers empirically in §4 and theoretically in §5, to evaluate if Transformers are more similar to first-order or second-order methods. We care particularly about algorithms’ convergence rates (the number of steps required to reach an _ϵ_ error). 

For any time step _t_ , let **_X_**<sup>(</sup><sup>_t_)</sup> = � **_x_** 1 _· · ·_ **_x_** _t_ � _⊤_ be the data matrix and **_y_** ( _t_ ) = � _y_ 1 _· · · yt_ � _⊤_ be the labels for all the datapoints seen so far. Note that since _t_ can be smaller than the data dimension _d_ , **_X_**<sup>(</sup><sup>_t_)</sup> can be singular. We now consider various algorithms for making predictions for **_x_** _t_ +1 based on **_X_**<sup>(</sup><sup>_t_)</sup> and **_y_**<sup>(</sup><sup>_t_)</sup> . When it is clear from context, we drop the superscript and refer to **_X_**<sup>(</sup><sup>_t_)</sup> and **_y_**<sup>(</sup><sup>_t_)</sup> as **_X_** and **_y_** , where **_X_** and **_y_** correspond to all the datapoints seen so far. 

**Ordinary Least Squares.** This method finds the minimum-norm solution to the objective: 



The Ordinary Least Squares (OLS) solution has a closed form given by the Normal Equations: 



where **_S_** := **_X_**<sup>_⊤_</sup> **_X_** and **_S_**<sup>_†_</sup> is the pseudo-inverse [Moore, 1920] of **_S_** . 

**Gradient Descent.** Gradient descent (GD) is a first-order method which finds the weight vector ˆ ˆ **_w_**<sup>GD</sup> with initialization **_w_** 0<sup>GD</sup> = **0** using the iterative update rule: 



It is known that GD requires _O_ ( _κ_ ( **_S_** ) log(1 _/ϵ_ )) steps to converge to an _ϵ_ error where _κ_ ( **_S_** ) =<sup>_λ_</sup> _λ_<sup>max</sup> min(<sup><u>(</u></sup> **_S_**<sup>**_S_**</sup> )<sup><u>)</u></sup> is the _condition number_ . Thus, when _κ_ ( **_S_** ) is large, GD converges slowly [Boyd and Vandenberghe, 2004]. 

**Online Gradient Descent.** While GD computes the gradient with respect to the full data matrix **_X_** at each iteration, Online Gradient Descent (OGD) is an online algorithm that only computes gradients on the newly received data point _{_ **_x_** _k, yk}_ at step _k_ : 



Picking _ηk_ = _∥_ **_x_** <u>1</u> _k∥_<sup>2</sup> 2<sup>ensures that the new weight vector</sup><sup>**_w_**ˆ</sup> _k_<sup>OGD</sup> +1<sup>makes zero error on</sup><sup>_{_</sup><sup>**_x_**</sup><sup>_k, yk}_.</sup> 

**Iterative Newton’s Method.** This is a second-order method which finds the weight vector **_w_** ˆ<sup>Newton</sup> by iteratively apply Newton’s method to finding the pseudo inverse of **_S_** = **_X_**<sup>_⊤_</sup> **_X_** [Schulz, 1933, Ben-Israel, 1965]. 



4 

This computes an approximation of the psuedo inverse using the moments of **_S_** . In contrast to GD, the Iterative Newton’s method only requires _O_ (log _κ_ ( **_S_** ) + log log(1 _/ϵ_ )) steps to converge to an _ϵ_ error [Soderstrom and Stewart, 1974, Pan and Schreiber, 1991]. Note that this is exponentially faster than the convergence rate of GD. We discuss additional algorithms such as Conjugate Gradient, BFGS, and L-BFGS in the Appendix A.2.3. 

### **3.2 Solving Linear Regression with Transformers** 

We will use neural network models such as Transformers to solve this linear regression task. As shown in Figure 1, at time step _t_ + 1, the model sees the first _t_ in-context examples _{_ **_x_** _i, yi}_<sup>_t_</sup> _i_ =1<sup>, and</sup> then makes predictions for **_x_** _t_ +1, whose label _yt_ +1 is not observed by the Transformers model. 

We randomly initialize our models and then train them on the linear regression task to make predictions for every number of in-context examples _t_ , where _t ∈_ [ _n_ ]. Training and test data are both drawn from _PD_ . To make the input prompts contain both **_x_** _i_ and _yi_ , we follow same the setup as Garg et al. [2022]’s to zero-pad _yi_ ’s, and use the same GPT-2 model [Radford et al., 2019] with softmax activation and causal attention mask (discussed later in Definition 3.1). 

We now present the key mathematical details for the Transformer architecture, and how they can be used for in-context learning. First, the causal attention mask enforces that attention heads can only attend to hidden states of previous time steps, and is defined as follows. **Definition 3.1** (Causal Attention Layer) **.** A **causal** attention layer with _M_ heads and activation function _σ_ is denoted as Attn on any input sequence **_H_** = � **_h_** 1 _, · · · ,_ **_h_** _N_ � _∈_ R<sup>_D×N_</sup> , where _D_ is the dimension of hidden states and _N_ is the sequence length. In the vector form, 



Vaswani et al. [2017] originally proposed the Transformer architecture with the Softmax activation function for the attention layers. Later works have found that replacing Softmax( _·_ ) with <u>1</u> _t_<sup>ReLU(</sup><sup>_·_) does not hurt model performance [Cai et al., 2022, Shen et al., 2023a, Wortsman et al.,</sup> 2023]. The Transformers architecture is defined by putting together attention layers with feed forward layers: 

**Definition 3.2** (Transformers) **.** An _L_ -layer decoder-based transformer with Causal Attention Layers is denoted as TF **_θ_** and is a composition of a MLP Layer (with a skip connection) and a Causal Attention Layers. For input sequence **_H_**<sup>(0)</sup> , the transformers _ℓ_ -th hidden layer is given by 



In particular for the linear regression task, Transformers perform in-context learning as follows 

**Definition 3.3** (Transformers for Linear Regression) **.** Given in-context examples _{_ **_x_** 1 _, y_ 1 _, . . . ,_ **_x_** _t, yt}_ , Transformers make predictions on a query example **_x_** _t_ +1 through a readout layer parameterized as ˆ **_θ_** readout = _{_ **_u_** _, v}_ , and the prediction _yt_<sup>TF</sup> +1<sup>is given by</sup> 



5 



<!-- Start of picture text -->
10 0 Transformer Errors v.s. # Layers 10 0 Iterative Newton Errors v.s. # Steps 10 0 Gradient Descent Errors v.s. # Steps<br>10 1 10 1 10 1<br>10 2 10 2 10 2<br># In-Context Examples = 05 # In-Context Examples = 05 # In-Context Examples = 05<br>10 3 # In-Context Examples = 10# In-Context Examples = 15 10 3 # In-Context Examples = 10# In-Context Examples = 15 10 3 # In-Context Examples = 10# In-Context Examples = 15<br># In-Context Examples = 20 # In-Context Examples = 20 # In-Context Examples = 20<br># In-Context Examples = 22 # In-Context Examples = 22 # In-Context Examples = 22<br># In-Context Examples = 25 # In-Context Examples = 25 # In-Context Examples = 25<br># In-Context Examples = 30 # In-Context Examples = 30 # In-Context Examples = 30<br>10 4 # In-Context Examples = 35 10 4 # In-Context Examples = 35 10 4 # In-Context Examples = 35<br>1 2 3 4 5 6 7 8 9 10 11 12 1 3 5 7 9 11 13 15 17 19 21 23 1 30 80 160<br>Transformer Layer Index Iterative Newton Steps Gradient Descent Steps<br>(a) Transformers (b) Iterative Newton’s Method (c) Gradient Descent<br>Errors Errors Errors<br><!-- End of picture text -->

Figure 2: **Convergence of Algorithms. Similar to Iterative Newton and GD, Transformer’s performance improve over the layer index** _ℓ_ **. When** _n > d_ **, the Transformer model, from layers 3 to 8, demonstrates a superlinear convergence rate, similar to Iterative Newton, while GD, with fixed step size, is sublinear. Later layers of Transformers show a slower convergence rate, and we hypothesize they have little incentive to implement the algorithm precisely since the error is already very small. A 24-layer Transformer model exhibits the same superlinear convergence (Figure 25 in §A.4.2).** 

To compare the rate of convergence of iterative algorithms to that of Transformers, we treat the layer index _ℓ_ of Transformers as analogous to the iterative step _k_ of algorithms discussed in §3.1. Note that for Transformers, we need to re-train the ReadOut layer for every layer index _ℓ_ so that they can improve progressively (see §4.1 and for experimental details) for linear regression tasks. 

### **3.3 Measuring Algorithmic Similarity** 

We propose two metrics to measure the similarity between linear regression algorithms. 

**Similarity of Errors.** This metric aims to measure similarity of algorithms through comparing prediction errors. For a linear regression algorithm _A_ , let _A_ ( **_x_** _t_ +1 _| {_ **_x_** _i, yi}_<sup>_t_</sup> _i_ =1<sup>) denote its prediction</sup> on the ( _t_ + 1)-th in-context example **_x_** _t_ +1 after observing the first _t_ examples (see Figure 1). We write _A_ ( **_x_** _t_ +1) := _A_ ( **_x_** _t_ +1 _| {_ **_x_** _i, yi}_<sup>_t_</sup> _i_ =1<sup>) for brevity.Errors (i.e., residuals) on the sequence are:2</sup> 



The similarity of errors for two algorithms _Aa_ and _Ab_ is the expected cosine similarity of their errors on a randomly sampled data sequence: 



Here _C_ ( **_u_** _,_ **_v_** ) = _∥_ **_u_** _<u>⟨∥</u>_ **_u_** 2 _<u>,∥</u>_ **_vv_** _<u>⟩∥</u>_ 2<sup>is the cosine similarity,</sup><sup>_n_is the total number of in-context examples,</sup> and _PD_ is the data generation process discussed previously. 

**Similarity of Induced Weights.** All standard algorithms for linear regression estimate a weight ˆ vector **_w_** . While neural ICL models like Transformers do not explicitly learn such a weight vector, 

> 2the indices start from 2 to _n_ + 1 because we evaluate all cases where _t_ can choose from 1 _, · · · , n_ . 

6 

similar to Akyürek et al. [2022], we can _induce_ an implicit weight vector **_w_** ˜ learned by any algorithm _A_ by fitting a weight vector to its predictions. We can then measure similarity of algorithms by ˜ comparing the induced **_w_** . To do this, for any fixed sequence of _t_ in-context examples _{_ **_x_** _i, yi}_<sup>_t_</sup> _i_ =1<sup>,</sup> ˜ i.i.d. we sample _T ≫ d_ query examples **_x_** _k ∼N_ ( **0** _,_ **Σ** ), where _k ∈_ [ _T_ ]. For this fixed sequence of in-context examples _{_ **_x_** _i, yi}_<sup>_t_</sup> _i_ =1<sup>, we create</sup><sup>_T_in-context prediction tasks and use the algorithm</sup><sup>_A_to</sup> make predictions _A_ (˜ **_x_** _k | {_ **_x_** _i, yi}_<sup>_t_</sup> _i_ =1<sup>).We define the induced data matrix and labels as</sup> 



The induced weight vector for _A_ and these _t_ examples is: 



The similarity of induced weights between two algorithms _Aa_ and _Ab_ is the expected average cosine similarity<sup>3</sup> of induced weights **_w_** ˜ _t_ ( _Aa_ ) and **_w_** ˜ _t_ ( _Ab_ ) over all possible 1 _≤ t ≤ n_ , on a randomly sampled data sequence: 



**Matching steps between algorithms.** Each algorithm converges to its predictions after several **steps** — for example the number of iterations for Iterative Newton and GD, and the number of layers for Transformers (see Section 4.1). When comparing two algorithms, given a choice of steps for the first algorithm, we match it with the steps for the second algorithm that maximize similarity. 

**Definition 3.4** (Best-matching Steps) **.** Let _M_ be the metric for evaluating similarities between two algorithms _Aa_ and _Ab_ , which have steps _pa ∈_ [0 _, Ta_ ] and _pb ∈_ [0 _, Tb_ ], respectively. For a given choice of _pa_ , we define the best-matching number of steps of algorithm _Ab_ for _Aa_ as: 



In our experiments, we chose _Ta, Tb_ be large enough integers so the algorithms converge. The matching processes can be visualized as heatmaps as shown in Figure 3, where best-matching steps are highlighted. This enables us to compare the rate of convergence of algorithms. In particular, if two algorithms converge at the same rate, the best matching steps between the two algorithms should follow a linear trend. We will discuss these results in §4. See Figure 26 on how best-matching steps help compare the convergence rates. 

## **4 Experimental Evidence** 

We primarily study the Transformers-based GPT-2 model with 12 layers and 8 heads per layer. Alternative configurations with fewer heads per layer, or with more layers, also support our findings; we defer them to §A.4.1 and §A.4.2. We initially focus on isotropic cases where **Σ** = **_I_** and 

> 3Alternative metrics such as _ℓ_ 2 distance gives the same observation. Here cosine similarity is better since errors usually have small magnitudes, and directions of induced weights are meaningful. 

7 

later consider ill-conditioned **Σ** in §4.3. Our training setup is exactly the same as Garg et al. [2022]: models are trained with at most _n_ = 40 in-context examples for _d_ = 20 (with the same learning rate, batch size etc.). 

We claim that Transformers learn high-order optimization methods in-context. We provide evidence that Transformers improve themselves with more layers in §4.1; Transformers share the same rate of convergence as Iterative Newton, exponentially faster than that of GD, in §4.2; and they also perform well on ill-conditioned problems in §4.3. Finally, we contrast Transformers with LSTMs in §4.5. 

### **4.1 Transformers improve progressively over layers** 

Many known algorithms for linear regression, including GD, OGD, and Iterative Newton, are _iterative_ : their performance progressively improves as they perform more iterations, eventually converging to a final solution. How can a Transformer implement such an iterative algorithm? von Oswald et al. [2022] propose that deeper _layers_ of the Transformer may correspond to more iterations; in particular, they show that there exist Transformer parameters such that each attention layer performs one step of GD. 

Following this intuition, we first investigate whether the predictions of a trained Transformer improve as the layer index _ℓ_ increases. For each layer of hidden states **_H_**<sup>(</sup><sup>_ℓ_)</sup> (see Definition 3.2), we re-train the ReadOut to predict _yt_ for each _t_ ; the new predictions are given by ReadOut<sup>(</sup><sup>_ℓ_) �</sup> **_H_**<sup>(</sup><sup>_ℓ_)�</sup> . Thus for each input prompt, there are _L_ Transformer predictions parameterized by layer index _ℓ_ . All parameters besides the ReadOut layer parameters are kept frozen. 

As shown in Figure 2(a) (and Figure 7(a) in the Appendix), as we increase the layer index _ℓ_ , the prediction performance improves progressively. Hence, Transformers progressively improve their predictions over layers _ℓ_ , similar to how iterative algorithms improve over steps. Such observations are consistent with language tasks where Transformers-based language models also improve their predictions along with layer progressions [Geva et al., 2022, Chuang et al., 2023]. 

### **4.2 Transformers are more similar to second-order methods, such as Iterative Newton** 

We now test the more specific hypothesis that the iterative updates performed across Transformer layers are similar to the iterative updates for known iterative algorithms. First, Figure 2 shows that the middle layers of Transformers converge at a rate similar to Iterative Newton, and faster than GD. In particular, the Transformer and Iterative Newton both converge at a superlinear rate, while GD converges at a sublinear rate. 

Next, we analyze whether each layer _ℓ_ of the Transformer corresponds to performing _k_ steps of some iterative algorithm, for some _k_ depending on _ℓ_ . We focus here on GD and Iterative Newton’s Method; we will discuss online algorithms in Section 4.5, and additional optimization methods in Appendix A.2.3. We will discuss results on noisy linear regression tasks in Appendix A.3.2. 

For each layer _ℓ_ of the Transformer, we measure the best-matching similarity (see Def. 3.4) with candidate iterative algorithms with the optimal choice of the number of steps _k_ . As shown in Figure 3, the Transformer has very high error similarity with Iterative Newton’s method at all layers. Moreover, we see a clear _linear_ trend between layer 3 and layer 9 of the Transformer, where each layer appears to compute roughly 3 additional iterations of Iterative Newton’s method. This trend only stops at the last few layers because both algorithms converge to the OLS solution; Newton is known to converge to OLS (see §3.1), and we verify in Appendix A.2 that the last few layers of the Transformer also basically compute OLS (see Figure 14 in the Appendix). We observe 

8 



<!-- Start of picture text -->
Similarity of Errors  (Transformers v.s. Iterative Newton)<br>Transformer Layer Index<br>1 2 3 4 5 6 7 8 9 10 11 12<br>1 .920 .920 .912<br>2 .876 .876 .929<br>3 .927<br>4 .916<br>5 .923<br>6 .916<br>7 .926<br>8 .927<br>9 .919 .949<br>10 .954<br>11 .953<br>12<br>13 .979<br>14 .980<br>15 .979<br>16 .988<br>17 .988<br>18 .988<br>19 .992<br>20 .993 .993 .993 .993<br>21 .992 .993 .994 .994<br>22 .993 .993 .993<br>23<br> Iterative newton Steps<br><!-- End of picture text -->



<!-- Start of picture text -->
Similarity of Errors  (Transformers v.s. Gradient Descent)<br>Transformer Layer Index<br>1 2 3 4 5 6 7 8 9 10 11 12<br>1 .954 .953 .870 .770 .692 .645<br>50 .578 .577 .733 .831 .905 .946<br>100 .795 .883 .944<br>150 .973<br>200 .974<br>250 .974<br>300<br>350<br>400<br>450<br>500<br>550<br>600<br>650 .982<br>700 .982<br>750 .982<br>800<br>850<br>900<br>950<br>1000<br>1050<br>1100<br>1150<br>1200 .986 .986 .986 .986<br>1250 .986 .986 .987 .987<br>1300 .986 .986 .986 .986<br> Gradient descent Steps<br><!-- End of picture text -->

Figure 3: **Heatmaps of Similarity.** The best matching steps are highlighted in yellow. Transformers layers show a linear trend with Iterative Newton steps but an exponential trend with GD. This suggests Transformers and Iterative Newton have the same convergence rate that is exponentially faster than GD. See Figure 10 for an additional heatmap where GD’s steps are shown in log scale: on that plot there is a linear correspondence between Transformers and GD’s steps. This further strengthens the claim that Transformers have an exponentially faster rate of convergence than GD. 

the same trends when using similarity of induced weights as our similarity metric (see Figure 9 in the Appendix).Figure 11 in the Appendix shows that there is a similar _linear_ trend between Transformer and BFGS, an alternative quasi-Newton method. This is perhaps not surprising, given that BFGS also gets a superlinear convergence rate for linear regression Nocedal and Wright [1999]. Thus, we do not claim that Transformers specifically implement Iterative Newton, only that they (approximately) implement some second-order method. 

In contrast, even though GD has a comparable similarity with the Transformers at later layers, their best matching follows an _exponential_ trend. As discussed in the Section 3.1, for well-conditioned problems where _κ ≈_ 1, to achieve _ϵ_ error, the rate of convergence of GD is _O_ (log(1 _/ϵ_ )) while the rate of convergence of Iterative Newton is _O_ (log log(1 _/ϵ_ )). Therefore the rate of convergence of Iterative Newton is exponentially faster than GD. Transformer’s _linear_ correspondence with Iterative Newton and its _exponential_ correspondence with GD provides strong evidence that the rate of convergence of Transformers is similar to Iterative Newton, i.e., _O_ (log log(1 _/ϵ_ )). We also note that it is not possible to significantly improve GD’s convergence rate without using second-order methods: Nemirovski and Yudin [1983] showed a Ω� log(1 _/ϵ_ )� lower bound on the convergence rate of gradient-based methods for smooth and strongly convex problems, and Arjevani et al. [2016] shows a similar lower bound specifically for quadratic problems. 

In the Appendix, we show that limited-memory BFGS Liu and Nocedal [1989] and conjugate gradient (see Figure 12), which do not use full-second order information, also converge slower than Transformers. This provides further evidence for the usage of second-order information by Transformers. We also show more evidence by investigating alternative function classes such as linear regression with noises in Appendix A.3.2 and 2-layer neural network with ReLU or Tanh 

9 



<!-- Start of picture text -->
OGD 0.25 OGD<br>1.0 Transformers Transformers<br>LSTM LSTM<br>Convergence on Ill-Conditioned Data Newton's Method (5 steps) 0.20 Newton's Method (5 steps)<br>10 0 0.8<br>Transformer<br>10 1 Iterative NewtonGradient Descent 0.6 0.15<br>0.10<br>10 2 0.4<br>0.2 0.05<br>10 3<br>0.0 0.00<br>10 4 10 0 10 1 10 2 10 3 0 5 Number of In-Context Examples10 15 20 25 30 35 40 19 15 Time Stamp Gap11 7 3<br>Steps<br>Errors Mean Square Errors Mean Square Errors<br><!-- End of picture text -->

Figure 4: Transformers performance on ill-conditioned data. Given 40 in-context examples, Transformers and Iterative Newton converge similarly and they both can converge to the OLS solution quickly whereas GD suffers. 

Figure 5: In the left figure, we measure model predictions with normalized MSE. Though LSTM is seemingly most similar to Newton’s Method with only 5 steps, neither algorithm converges yet. OGD also has a similar trend as LSTM. In the right figure, we measure the model’s error rate on example **_x_** _n−g_ after seeing _n_ examples, for different values of the time stamp gap _g_ (see Appendix A.6) 

, and find both Transformers and not-converged Newton have better memorization than LSTM and OGD. 

activation function in Appendix A.3.3. 

Overall, we conclude that a Transformer trained to perform in-context linear regression learns to implement an algorithm that is very similar to second-order methods, such as Iterative Newton’s method, not GD. Starting at layer 3, subsequent layers of the Transformer compute more and more iterations of Iterative Newton’s method. This algorithm successfully solves the linear regression problem, as it converges to the optimal OLS solution in the final layers. 

### **4.3 Transformers perform well on ill-conditioned data** 

i.i.d. We repeat the same experiments with data **_x_** _i ∼N_ ( **0** _,_ **Σ** ) sampled from an ill-condition covariance matrix **Σ** with condition number _κ_ ( **Σ** ) = 100, and eigenbasis chosen uniformly at random. The first _d/_ 2 eigenvalues of **Σ** are 100, and the last _d/_ 2 are 1. Note that choosing the eigenbasis uniformly at random for _each_ sequence ensures that there is a different covariance matrix **Σ** for each sequence of datapoints. 

As shown in Figure 4, the Transformer model’s performance still closely matches Iterative Newton’s Method with 21 iterations, same as when **Σ** = **_I_** (see layer 10-12 in Figure 3). The convergence of second-order methods has a mild logarithmic dependence on the condition number since they correct for the curvature. On the other hand, GD’s convergence is affected polynomially by conditioning. As _κ_ ( **Σ** ) increase from 1 to 100, the number steps required for GD’s convergence increases significantly (see Fig. 4 where GD requires 2,000 steps to converge), making it impossible for a 12-layer Transformers to implement these many gradient updates. We also note that preconditioning the data by ( **_X_**<sup>_⊤_</sup> **_X_** )<sup>_†_</sup> can make the data well-conditioned, but since the eigenbasis is chosen uniformly at random, with high probability there is no sparse pre-conditioner or any fixed pre-conditioner which works across the data distribution. Computing ( **_X_**<sup>_⊤_</sup> **_X_** )<sup>_†_</sup> appears to be as hard as computing the OLS solution (Eq. 1)—in fact Sharan et al. [2019] conjecture that firstorder methods such as gradient descent and its variants cannot avoid polynomial dependencies 

10 



<!-- Start of picture text -->
Transformers with Various Hidden Sizes<br>1.2 Transformers (Hidden Size=8)<br>Transformers (Hidden Size=16)<br>1.0 Transformers (Hidden Size=32)<br>Transformers (Hidden Size=64)<br>0.8 Least Squares<br>0.6<br>0.4<br>0.2<br>0.0<br>0 5 10 15 20 25 30 35 40<br>in-context examples<br>squared error<br><!-- End of picture text -->

Figure 6: Ablation on Transformer’s Hidden Size. For linear regression problems with _d_ = 20, Transformers need _O_ ( _d_ ) hidden dimension to mimic OLS solutions. 

in condition number in the ill-conditioned case.<sup>4</sup> See Appendix A.3.1 for detailed experiments on ill-conditioned problems. These experiments further strengthen our thesis that Transformers learn to perform second-order optimization methods in-context, not first-order methods such as GD. 

### **4.4 Transformers Require** _O_ ( _d_ ) **Hidden Dimension** 

We ablate 12-layer 1-head Transformers with various hidden sizes on _d_ = 20 problems. As shown in Figure 6, we observe that Transformers can mimic OLS solution when the hidden size is 32 or 64, but fail with smaller sizes. This resonates with our theoretical results on _O_ ( _d_ ) hidden dimension in Theorem 5.1, and in this case, the theorem ensures a construction of transformers to implement Iterative Newton’s method. 

### **4.5 LSTM is more similar to OGD than Transformers** 

As discussed in §A.1, LSTM is an alternative auto-regressive model widely used before the introduction of Transformers. Thus, a natural research question is: _If Transformers can learn in-context, can LSTMs do so as well? If so, do they learn the same algorithms?_ To answer this question, we train a LSTM model in an identical manner to the Transformers studied in the previous sections. 

Figure 5 plots the error of Transformers, LSTMs, and other standard methods as a function of the number of in-context (i.e., training) examples provided. While LSTMs can also learn linear regression in-context, they have much higher mean-squared error than Transformers. Their error rate is similar to Iterative Newton’s Method after only 5 iterations, a point where it is far from converging to the OLS solution. 

Finally, we show that LSTMs behave more like an online learning algorithm than Transformers. In particular, its predictions are biased towards getting more recent training examples correct, as opposed to earlier examples, as shown in Figure 5. This property makes LSTMs similar to online GD. In contrast, five steps of Newton’s method has the same error on average for recent and early examples, showing that the LSTM implements a very different algorithm from a few iterations of Newton. 

> 4Regarding preconditioning, we also note that—even for well-conditioned instances—preconditioned GD still gets a linear rate of convergence, whereas Transformers and Iterative Newton get superlinear rates. 

11 

We hypothesize that since LSTMs have limited memory, they must learn in a roughly online fashion; in contrast, Transformer’s attention heads can access the entire sequence of past examples, enabling it to learn more complex algorithms. See §A.1 for more discussions. 

## **5 Theoretical Justification** 

Our empirical evidence demonstrates that Transformers behave much more similarly to Iterative Newton’s than to GD. Iterative Newton is a second-order optimization method, and is algorithmically more involved than GD. We begin by first examining this difference in complexity. As discussed in Section 3, the updates for Iterative Newton are of the form, 



and **_M_** 0 = _α_ **_S_** for some _α >_ 0 _._ We can express **_M_** _k_ in terms of powers of **_S_** by expanding iteratively, for example **_M_** 1 = 2 _α_ **_S_** _−_ 4 _α_<sup>2</sup> **_S_**<sup>3</sup> _,_ **_M_** 2 = 4 _α_ **_S_** _−_ 12 _α_<sup>2</sup> **_S_**<sup>3</sup> + 16 _α_<sup>3</sup> **_S_**<sup>5</sup> _−_ 16 _α_<sup>4</sup> **_S_**<sup>7</sup> , and in general **_M_** _k_ =<sup>�2</sup> _s_ =1<sup>_k_+1</sup><sup>_−_1</sup> _βs_ **_S_**<sup>_s_</sup> for some _βs ∈_ R (see Appendix B.3 for detailed calculations). Note that _k_ steps of Iterative Newton’s requires computing Ω(2<sup>_k_</sup> ) moments of **_S_** . Let us contrast this with GD. GD updates for linear regression take the form, 



Like Iterative Newton, we can express **_w_** ˆ _k_<sup>GD</sup> in terms of powers of **_S_** and **_X_**<sup>_⊤_</sup> **_y_** . However, after _k_ steps of GD, the highest power of **_S_** is only _O_ ( _k_ ). This exponential separation is consistent with the exponential gap in terms of the parameter dependence in the convergence rate— _O_ ( _κ_ ( **_S_** ) log(1 _/ϵ_ )) for GD vs. _O_ (log _κ_ ( **_S_** ) + log log(1 _/ϵ_ )) for Iterative Newton. Therefore, a natural question is whether Transformers can actually as complicated of a method such as Iterative Newton with only polynomially many layers? Theorem 5.1 shows that this is indeed possible. 

**Theorem 5.1.** _For any k, there exist Transformer weights such that on any set of in-context examples_ ˆ _{_ **_x_** _i, yi}_<sup>_n_</sup> _i_ =1<sup>_and test point_</sup><sup>**_x_**test</sup><sup>_, the Transformer predicts on_</sup><sup>**_x_**test</sup><sup>_using_</sup><sup>**_x_**</sup><sup>_⊤_</sup> test<sup>**_w_**ˆ</sup> _k_<sup>Newton</sup> _. Here_ **_w_** _k_<sup>Newton</sup> _are_ ˆ _the Iterative Newton updates given by_ **_w_** _k_<sup>Newton</sup> = **_M_** _k_ **_X_**<sup>_⊤_</sup> **_y_** _where_ **_M_** _j is updated as_ 



_for some α >_ 0 _and_ **_S_** = **_X_**<sup>_⊤_</sup> **_X_** _. The dimensionality of the hidden layers is O_ ( _d_ ) _, and the number of layers is k_ + 8 _. One transformer layer computes one Newton iteration. 3 initial transformer layers are needed for initializing_ **_M_** 0 _and 5 layers at the end are needed to read out predictions from the computed pseudo-inverse_ **_M_** _k._ 

Here we provide a skecth of the proof. We note that our proof uses full attention instead of causal attention and ReLU activations for the self-attention layers. The definitions of these and the B. full proof appear in Appendix 

### **5.1 Proof Sketch for Theorem 5.1** 

The constructive proof leverages some key operators which Akyürek et al. [2022] showed a single B.4 . Transformers layer can implement. We summarize these in Proposition We mainly use the mov operator, which can copy parts of the hidden state from time stamp _i_ to time stamp _j_ for any _j ≥ i_ ; the mul operator which can do multiplications; and the aff operator, which can be used to 

12 

do addition and subtraction. **Transformers Implement Initialization** **_T_**<sup>(0)</sup> = _α_ **_S_ .** Given input sequence **_H_** = _{_ **_x_** 1 _, · · · ,_ **_x_** _n}_ , we can first use the mov operator from Proposition B.4 so that the **_x_** 1 _· · ·_ **_x_** _n_ input sequence becomes . We call each column **_h_** _j_ . With an full attention layer and � **_x_** 1 _· · ·_ **_x_** _n_ � normalized ReLU activations, one can construct two heads with query and value matrices of the **_I_** _d×d_ **_O_** _d×d_ **_I_** _d×d_ **_O_** _d×d_ form **_Q_**<sup>_⊤_</sup> 1<sup>**_K_**1=</sup><sup>_−_</sup><sup>**_Q_**</sup><sup>_⊤_</sup> 2<sup>**_K_**2=</sup> � **_O_** _d×d_ **_O_** _d×d_ � and value matrices **_V_** _m_ = _nα_ � **_O_** _d×d_ **_O_** _d×d_ � for some _α ∈_ R.<sup>5</sup> Combining the attention layer and skip connections we end up with **_h_** _t ←_ **_x_** _t_ + _α_ **_Sx_** _t_ . � **_x_** _t_ � Applying the aff operator from Proposition B.4 to do subtraction, we can have each column of the form _α_ **_Sx_** _t_ . We denote **_T_**<sup>(0)</sup> := _α_ **_S_** so that Transformers and Iterative Newton have the similar � **_x_** _t_ � initialization and we call these columns **_h_**<sup>(0)</sup> _t_<sup>.</sup> 

**Transformers implement Newton Iteration.** We claim that we can construct layer _ℓ_ ’s hidden states to be of the form 



We prove by induction that assuming our claim is true for _ℓ_ , we work on _ℓ_ + 1: Let **_Q_** _m_ = **_Q_** ˜ _m_ **_O_** _d −_<sup>_<u>n</u>_</sup> 2<sup>**_I_**</sup><sup>_d_</sup> _,_ **_K_** _m_ = **_K_** ˜ _m_ **_I_** _d_ **_O_** _d_ where **_Q_**<sup>˜</sup><sup>_⊤_</sup> 1 **_K_**<sup>˜</sup> 1 := **_I_** , **_Q_**<sup>˜</sup><sup>_⊤_</sup> 2 **_K_**<sup>˜</sup> 2 := _−_ **_I_** and **_V_** 1 = **_V_** 2 = � **_O_** _d_ **_O_** _d_ � � **_O_** _d_ **_O_** _d_ � **_I_** _d_ **_O_** _d_ . A 2-head self-attention layer gives � **_O_** _d_ **_O_** _d_ � 



Note that all **_T_**<sup>(</sup><sup>_ℓ_)</sup> are symmetric. Passing over an MLP layer gives 



We denote **_T_**<sup>(</sup><sup>_ℓ_+1)</sup> := 2 **_T_**<sup>(</sup><sup>_ℓ_)</sup> _−_ **_T_**<sup>(</sup><sup>_ℓ_)</sup> **_ST_**<sup>(</sup><sup>_ℓ_)</sup> and this is the exactly same form as Iterative Newton updates. 

ˆ **Transformers can implement** **_w_** _ℓ_<sup>TF</sup> = **_T_**<sup>(</sup><sup>_ℓ_)</sup> **_X_**<sup>_⊤_</sup> **_y_ .** We insert columns with [0 _,_ 0 _, · · · , yj_ ]<sup>_⊤_</sup> after each **_x_** _j_ (See Figure 1 for illustration) and keep them unchanged until reaching layer _ℓ_ . Applying mov **_ξ_** and mul, we have columns where **_ξ_** are irrelevant quantities. Apply Lemma B.3 for � **_T_**<sup>(</sup><sup>_ℓ_)</sup> _yj_ **_x_** _j_ � summation, we can gather<sup>�</sup><sup>_n_</sup> _j_ =1<sup>**_T_**(</sup><sup>_ℓ_)</sup><sup>_yj_</sup><sup>**_x_**</sup><sup>_j_=</sup><sup>**_T_**(</sup><sup>_ℓ_)</sup><sup>**_X_**</sup><sup>_⊤_</sup><sup>**_y_**, which is again the same as Iterative Newton</sup> ˆ and we call this **_w_**<sup>TF</sup> . _ℓ_ 

ˆ TF **Transformers can make predictions on** **_x_** _test_ **by** � **_w_** _ℓ ,_ **_x_** test� **.** 

Now we can make predictions on text query **_x_** test: 



> 5 The value matrices contain the number of in-context examples _n_ . There are ways to avoid this by applying Proposition B.4 to count. 

13 

ˆ TF A final readout layer can extract the prediction � **_w_** _ℓ ,_ **_x_** test�. 

Now we complete the proof that Transformers can perform exactly Iterative Newton. Finally, we count the number of layers and the dimension of hidden states. We can see all operations in the proof require a linear amount of hidden state dimensions, which are _O_ ( _d_ ). There are 3 Transformer layers needed to compute the Newton initialization and 5 layers needed for reading out predictions. Operations from Transformers index _ℓ_ to _ℓ_ + 1 require 1 Transformer layer. Hence, to perform _k_ Iterative Newton updates, Transformers require _k_ + 8 layers. This implies that the rate of convergence of Transformers to solve linear regression in-context is the same as Iterative Newton’s: _O_ (log log(1 _/ϵ_ )) and this is consistent with our experimental results in §4. 

## **6 Conclusion and Discussion** 

In this work, we studied how Transformers perform in-context learning for linear regression. In contrast with the hypothesis that Transformers learn in-context by implementing gradient descent, our experimental results show that different Transformer layers match iterations of Iterative Newton _linearly_ and Gradient Descent _exponentially_ . This suggests that Transformers share a similar rate of convergence to Iterative Newton but not to Gradient Descent. Moreover, Transformers can perform well empirically on ill-conditioned linear regression, whereas first-order methods such as Gradient Descent struggle. This empirical evidence — when combined with existing lower bounds in optimization — suggests that Transformers use second-order information for solving linear regression, and we also prove that Transformers can indeed represent second-order methods. 

An interesting direction is to explore a wider range of second-order methods that Transformers can implement. It also seems promising to extend our analysis to classification problems, especially given recent work showing that Transformers resemble SVMs in classification tasks [Li et al., 2023, Tarzanagh et al., 2023a]. Finally, a natural question is to understand the differences in the model architecture that make Transformers better in-context learners than LSTMs. Based on our investigations with LSTMs, we hypothesize that Transformers can implement more powerful algorithms because of having access to a longer history of examples. Investigating the role of this additional memory in learning appears to be an intriguing direction. 

## **Acknowledgement** 

We would like to thank the USC NLP Group and Center for AI Safety for providing compute resources. DF would like to thank Oliver Liu and Ameya Godbole for their extensive discussions. DF and RJ were supported by a Google Research Scholar Award. RJ was also supported by an Open Philanthropy research grant. VS was supported by NSF CAREER Award CCF-2239265 and an Amazon Research Award. 

## **References** 

- Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to implement preconditioned gradient descent for in-context learning. _ArXiv_ , abs/2306.00297, 2023. 1, 2 

- Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. _ArXiv_ , abs/2211.15661, 2022. 2, 3.3, 5.1, B.1, B.4, B.2, B.5 

14 

- Yossi Arjevani, Shai Shalev-Shwartz, and Ohad Shamir. On lower and upper bounds in smooth and strongly convex optimization. _Journal of Machine Learning Research_ , 17(126):1–51, 2016. URL http://jmlr.org/papers/v17/15-106.html. 4.2 

- Yu Bai, Fan Chen, Haiquan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians: Provable in-context learning with in-context algorithm selection. _ArXiv_ , abs/2306.04637, 2023. 2 

- Adi Ben-Israel. An iterative method for computing the generalized inverse of an arbitrary matrix. _Mathematics of Computation_ , 19(91):452–455, 1965. ISSN 00255718, 10886842. URL http://www. jstor.org/stable/2003676. 3.1 

- Stephen P Boyd and Lieven Vandenberghe. _Convex optimization_ . Cambridge university press, 2004. 3.1 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, _Advances in Neural Information Processing Systems_ , volume 33, pages 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/ file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf. 1, 2 

- Han Cai, Chuang Gan, and Song Han. Efficientvit: Enhanced linear attention for high-resolution low-computation visual recognition. _ArXiv_ , abs/2205.14756, 2022. 3.2 

- Ting-Yun Chang and Robin Jia. Data curation alone can stabilize in-context learning. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 8123–8144, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.452. URL https://aclanthology.org/2023.acl-long.452. 2 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways, 2022. 1 

- Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. Dola: Decoding by contrasting layers improves factuality in large language models, 2023. 4.1 

15 

- Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability, 2023. 2 

- Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Zhifang Sui, and Furu Wei. Why can gpt learn in-context? language models secretly perform gradient descent as meta-optimizers. _ArXiv_ , abs/2212.10559, 2023. 1, 2 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https: //aclanthology.org/N19-1423. 1 

- Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. _ArXiv_ , abs/2208.01066, 2022. 1, 2, 3.2, 4, A.3.3 

- Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 30–45, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.3. URL https://aclanthology.org/2022.emnlp-main.3. 4.1 

- Angeliki Giannou, Shashank Rajput, Jy-Yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 11398–11442. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/giannou23a.html. 2, B.5 

- Chi Han, Ziqi Wang, Han Zhao, and Heng Ji. In-context learning of large language models explained as kernel regression, 2023. 2 

- Michael Hassid, Hao Peng, Daniel Rotem, Jungo Kasai, Ivan Montero, Noah A. Smith, and Roy Schwartz. How much does attention actually attend? questioning the importance of attention in pretrained transformers, 2022. 2 

- Sepp Hochreiter and Jürgen Schmidhuber. Long Short-Term Memory. _Neural Computation_ , 9(8): 1735–1780, 11 1997. ISSN 0899-7667. doi: 10.1162/neco.1997.9.8.1735. URL https://doi.org/ 10.1162/neco.1997.9.8.1735. A.1 

- Ivan Lee, Nan Jiang, and Taylor Berg-Kirkpatrick. Exploring the relationship between model architecture and in-context learning ability, 2023. 2 

- Yingcong Li, Muhammed Emrullah Ildiz, Dimitris Papailiopoulos, and Samet Oymak. Transformers as algorithms: Generalization and stability in in-context learning. In _International Conference on Machine Learning_ , 2023. 2, 6 

- Dong C. Liu and Jorge Nocedal. On the limited memory bfgs method for large scale optimization. _Mathematical Programming_ , 45:503–528, 1989. URL https://api.semanticscholar.org/ CorpusID:5681609. 4.2 

16 

- Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for GPT-3? In _Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures_ , pages 100–114, Dublin, Ireland and Online, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.deelio-1.10. URL https://aclanthology.org/2022.deelio-1. 10. 2 

- Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 8086–8098, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.556. URL https://aclanthology.org/2022.acl-long.556. 2 

- Arvind V. Mahankali, Tatsunori Hashimoto, and Tengyu Ma. One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/ forum?id=8p3fu56lKc. 1, 2 

- Sewon Min, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Noisy channel language model prompting for few-shot text classification. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 5316–5330, Dublin, Ireland, May 2022a. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.365. URL https://aclanthology.org/2022.acl-long.365. 2 

- Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. MetaICL: Learning to learn in context. In _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 2791–2809, Seattle, United States, July 2022b. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.201. URL https://aclanthology.org/2022.naacl-main.201. 2 

- Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 11048–11064, Abu Dhabi, United Arab Emirates, December 2022c. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.759. URL https://aclanthology.org/ 2022.emnlp-main.759. 2 

- E.H Moore. On the reciprocal of the general algebraic matrix. _Bulletin of American Mathematical Society_ , 26:394–395, 1920. 3.1 

- Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability, 2023. 2 

- A.S. Nemirovski and D.B Yudin. Problem complexity and method efficiency in optimization. 1983. 4.2 

- Tai Nguyen and Eric Wong. In-context example selection with influences. _arXiv preprint arXiv:2302.11042_ , 2023. 2 

- Jorge Nocedal and Stephen J Wright. _Numerical optimization_ . Springer, 1999. 4.2, A.2.3 

17 

- OpenAI. Gpt-4 technical report, 2023. URL http://arxiv.org/abs/2303.08774v3. 1 

- Victor Y. Pan and Robert S. Schreiber. An improved newton iteration for the generalized inverse of a matrix, with applications. _SIAM J. Sci. Comput._ , 12:1109–1130, 1991. 3.1 

- Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets, 2022. 2 

- Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. 3.2 

- Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training gopher, 2022. 1 

- Allan Raventós, Mansheej Paul, Feng Chen, and Surya Ganguli. Pretraining task diversity and the emergence of non-bayesian in-context learning for regression, 2023. 2 

- Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. In _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 2655–2671, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.191. URL https://aclanthology.org/2022.naacl-main.191. 2 

- Günther Schulz. Iterative berechung der reziproken matrix. _Zeitschrift für Angewandte Mathematik und Mechanik (Journal of Applied Mathematics and Mechanics)_ , 13:57–59, 1933. 3.1 

- Vatsal Sharan, Aaron Sidford, and Gregory Valiant. Memory-sample tradeoffs for linear regression with small error. In _Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing_ , pages 890–901, 2019. 4.3 

- Kai Shen, Junliang Guo, Xu Tan, Siliang Tang, Rui Wang, and Jiang Bian. A study on relu and softmax in transformer, 2023a. 3.2 

- Lingfeng Shen, Aayush Mishra, and Daniel Khashabi. Do pretrained transformers really learn in-context by gradient descent?, 2023b. 2 

- Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, Elton Zhang, Rewon 

18 

Child, Reza Yazdani Aminabadi, Julie Bernauer, Xia Song, Mohammad Shoeybi, Yuxiong He, Michael Houston, Saurabh Tiwary, and Bryan Catanzaro. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model, 2022. 1 

- Torsten Soderstrom and G. W. Stewart. On the numerical properties of an iterative method for computing the moore- penrose generalized inverse. _SIAM Journal on Numerical Analysis_ , 11(1): 61–74, 1974. ISSN 00361429. URL http://www.jstor.org/stable/2156431. 3.1 

- Hongjin Su, Jungo Kasai, Chen Henry Wu, Weijia Shi, Tianlu Wang, Jiayi Xin, Rui Zhang, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Tao Yu. Selective annotation makes language models better few-shot learners. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id=qY1hlv7gwg. 2 

- Davoud Ataee Tarzanagh, Yingcong Li, Christos Thrampoulidis, and Samet Oymak. Transformers as support vector machines. _ArXiv_ , abs/2308.16898, 2023a. 2, 6 

- Davoud Ataee Tarzanagh, Yingcong Li, Xuechen Zhang, and Samet Oymak. Max-margin token selection in attention mechanism, 2023b. 2 

- Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Vincent Zhao, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, Pranesh Srinivasan, Laichee Man, Kathleen Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran, Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Aguera-Arcas, Claire Cui, Marian Croak, Ed Chi, and Quoc Le. Lamda: Language models for dialog applications, 2022. 1 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, _Advances in Neural Information Processing Systems_ , volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_files/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf. 1, 3.2 

- Max Vladymyrov, Johannes von Oswald, Mark Sandler, and Rong Ge. Linear transformers are versatile in-context learners, 2024. 1 

- Johannes von Oswald, Eyvind Niklasson, E. Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In _International Conference on Machine Learning_ , 2022. 1, 2, 4.1 

- Johannes von Oswald, Eyvind Niklasson, Maximilian Schlegel, Seijin Kobayashi, Nicolas Zucchet, Nino Scherrer, Nolan Miller, Mark Sandler, Blaise Agüera y Arcas, Max Vladymyrov, Razvan Pascanu, and Joao Sacramento. Uncovering mesa-optimization algorithms in transformers. _ArXiv_ , abs/2309.05858, 2023. 1, 2 

- Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small, 2022. 2 

19 

- Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, and Tengyu Ma. Larger language models do in-context learning differently, 2023. 2 

- Mitchell Wortsman, Jaehoon Lee, Justin Gilmer, and Simon Kornblith. Replacing softmax with relu in vision transformers, 2023. 3.2 

- Kang Min Yoo, Junyeob Kim, Hyuhng Joon Kim, Hyunsoo Cho, Hwiyeol Jo, Sang-Woo Lee, Sang-goo Lee, and Taeuk Kim. Ground-truth labels matter: A deeper look into inputlabel demonstrations. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 2422–2437, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.155. URL https://aclanthology.org/2022.emnlp-main.155. 2 

- Ruiqi Zhang, Spencer Frei, and Peter L. Bartlett. Trained transformers learn linear models in-context. _ArXiv_ , abs/2306.09927, 2023. 2 

- Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models. In _International Conference on Machine Learning_ , pages 12697–12706. PMLR, 2021. 2 

20 

## **Appendix** 

|**A Add**|**itional Experimental Results**|**21**|
|---|---|---|
|A.1|Contrast with LSTMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>21|
|A.2|Additional Results on Isotropic Data without Noise . . . . . . . . . . . . . . .|. . . .<br>22|
||A.2.1<br>Progression of Algorithms . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>22|
||A.2.2<br>Heatmaps . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>22|
||A.2.3<br>Comparison with Other Second-Order Methods . . . . . . . . . . . . .|. . . .<br>25|
||A.2.4<br>Additional Results on Comparison over Transformer Layers . . . . . .|. . . .<br>27|
||A.2.5<br>Additional Results on Similarity of Induced Weights<br>. . . . . . . . . .|. . . .<br>27|
|A.3|Varying Data Distribution or Function Class. . . . . . . . . . . . . . . . . . . .|. . . .<br>28|
||A.3.1<br>Experiments on Ill-Conditioned Problems . . . . . . . . . . . . . . . . .|. . . .<br>28|
||A.3.2<br>Experiments with Noisy Linear Regression . . . . . . . . . . . . . . . .|. . . .<br>31|
||A.3.3<br>Experiments with a Non-Linear Function Class (2-Layer MLP). . . . .|. . . .<br>32|
|A.4|Varying Transformer Architecture. . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>34|
||A.4.1<br>Experiments on Transformers of Fewer Heads . . . . . . . . . . . . . .|. . . .<br>34|
||A.4.2<br>Experiments on Transformers with More Layers . . . . . . . . . . . . .|. . . .<br>35|
|A.5|Heatmaps with Best-Matching Steps Help Compare Convergence Rates<br>. . .|. . . .<br>36|
|A.6|Defnitions for Evaluating Forgetting . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>37|
|**B**<br>**Det**|**ailed Proofs for Section 5**|**38**|
|B.1|Helper Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>38|
|B.2|Proof of Theorem 5.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . .<br>39|
|B.3|Iterative Newton as a Sum of Moments Method<br>. . . . . . . . . . . . . . . . .|. . . .<br>43|
|B.4|Estimated weight vectors lie in the span of previous examples . . . . . . . . .|. . . .<br>44|



## **A Additional Experimental Results** 

### **A.1 Contrast with LSTMs** 

While our primary goal is to analyze Transformers, we also consider LSTMs [Hochreiter and Schmidhuber, 1997] to understand whether Transformers learn different algorithms than other neural sequence models trained to do linear regression. In particular, we train a unidirectional _L_ -layer LSTM, which generates a sequence of hidden states **_H_**<sup>(</sup><sup>_ℓ_)</sup> for each layer _ℓ_ , similarly to an _L_ -layer Transformer. As with Transformers, we add a readout layer that predicts the _y_ ˆ _t_<sup>LSTM</sup> +1 from **_H_**<sup>(</sup><sup>_L_)</sup> the final hidden state at the final layer, : _,_ 2 _t_ +1<sup>.</sup> 

We train a 10-layer LSTM model, with 5.3M parameters, in an identical manner to the Transformers (with 9.5M parameters) studied in the previous sections.<sup>6</sup> 

> 6While the LSTM has fewer parameters than the Transformer, we found in preliminary experiments that increasing the size of the LSTM would not substantively change our results. 

21 

||Transformers|LSTM|
|---|---|---|
|Newton|**0.991**|0.920|
|GD|**0.957**|0.916|
|OGD|0.806|**0.954**|



Table 1: **Similarity of errors between algorithms.** Transformers are more similar to full-observation methods such as Newton and GD; and LSTMs are more similar to online methods such as OGD. 

LSTMs’ inferior performance to Transformers can be explained by the inability of LSTMs to use deeper layers to improve their predictions. Figure 7 shows that LSTM performance does not improve across layers—a readout head fine-tuned for the first layer makes equally good predictions as the full 10-layer model. Thus, LSTMs seem poorly equipped to fully implement iterative algorithms. Similarly, Table 1 shows that LSTMs are more similar to OGD than Transformers are, whereas Transformers are more similar to Newton and GD than LSTMs. 

### **A.2 Additional Results on Isotropic Data without Noise** 

#### **A.2.1 Progression of Algorithms** 



<!-- Start of picture text -->
Errors v.s. # In-Context Examples Errors v.s. # In-Context Examples Errors v.s. # In-Context Examples<br>10 0 10 0 10 0<br>10 1 10 1 10 1<br>Transformers Layer #01 Iterative Newton #01 (1 steps)<br>Transformers Layer #02 Iterative Newton #02 (1 steps)<br>Transformers Layer #03 Iterative Newton #03 (3 steps) LSTM Layer #01<br>Transformers Layer #04 Iterative Newton #04 (5 steps) LSTM Layer #02<br>10 2 Transformers Layer #05Transformers Layer #06 10 2 Iterative Newton #05 (8 steps)Iterative Newton #06 (10 steps) 10 2 LSTM Layer #03LSTM Layer #04<br>Transformers Layer #07 Iterative Newton #07 (14 steps) LSTM Layer #05<br>Transformers Layer #08 Iterative Newton #08 (17 steps) LSTM Layer #06<br>Transformers Layer #09 Iterative Newton #09 (20 steps) LSTM Layer #07<br>Transformers Layer #10 Iterative Newton #10 (21 steps) LSTM Layer #08<br>Transformers Layer #11 Iterative Newton #11 (21 steps) LSTM Layer #09<br>Transformers Layer #12 Iterative Newton #12 (21 steps) LSTM Layer #10<br>10 3 0 1 2 3 4 5 6 7 # In-Context Examples8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 10 3 0 1 2 3 4 5 6 7 # In-Context Examples8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 10 3 0 1 2 3 4 5 6 7 # In-Context Examples8 9 10 11 12 13 14 15 16 17 18 19 20 21 22<br>(a) Transformers (b) Iterative Newton’s Method (c) LSTM<br>Errors Errors Errors<br><!-- End of picture text -->

Figure 7: **Progression of Algorithms.** (a) Transformer’s performance improves over the layer index _ℓ_ . (b) Iterative Newton’s performance improves over the number of iterations _k_ , in a way that closely resembles the Transformer. We plot the best-matching _k_ to Transformer’s _ℓ_ following Definition 3.4. (c) In contrast, LSTM’s performance does not improve from layer to layer. 

#### **A.2.2 Heatmaps** 

We present heatmaps with all values of similarities. 

22 



<!-- Start of picture text -->
Similarity of Errors  (Transformers v.s. Iterative Newton)<br>Trasformer Layer Index Similarity of Errors  (Transformers v.s. Gradient Descent)<br>1 2 3 4 5 6 7 8 9 10 11 12 Trasformer Layer Index<br>1 2 3 4 5 6 7 8 9 10 11 12<br>1 .920 .920 .912 .816 .716 .662 .634 .623 .618 .613 .620 .616<br>1 .954 .953 .870 .770 .692 .645 .620 .610 .606 .600 .607 .603<br>2 .876 .876 .929 .858 .760 .702 .672 .660 .655 .651 .656 .652<br>50 .578 .577 .733 .831 .905 .946 .954 .946 .941 .939 .939 .938<br>3 .829 .828 .927 .893 .805 .745 .713 .700 .694 .690 .695 .692 100 .543 .543 .694 .795 .883 .944 .970 .967 .963 .961 .962 .961<br>4 .781 .781 .911 .916 .848 .789 .755 .741 .735 .732 .735 .733 150 .531 .531 .679 .781 .871 .939 .973 .974 .972 .970 .970 .969<br>5 .734 .734 .883 .923 .886 .834 .798 .783 .777 .774 .777 .774 200 .525 .524 .672 .772 .863 .935 .974 .977 .976 .974 .974 .974<br>6 .691 .691 .850 .916 .912 .875 .840 .824 .817 .814 .817 .815 250 .521 .521 .667 .767 .858 .932 .974 .979 .978 .977 .977 .977<br>300 .518 .518 .664 .763 .855 .929 .973 .980 .980 .979 .979 .979<br>7 .652 .652 .814 .898 .926 .910 .878 .862 .855 .852 .854 .852<br>350 .516 .516 .661 .761 .852 .927 .973 .981 .981 .980 .980 .980<br>8 .619 .619 .779 .874 .927 .935 .911 .895 .888 .885 .886 .885 400 .515 .515 .660 .759 .850 .926 .972 .982 .983 .981 .981 .981<br>9 .591 .591 .748 .849 .919 .949 .937 .921 .915 .913 .913 .913 450 .514 .514 .658 .757 .849 .924 .972 .982 .983 .982 .982 .982<br>10 .569 .569 .723 .826 .907 .954 .956 .942 .936 .934 .935 .934 500 .514 .514 .657 .756 .847 .923 .971 .982 .984 .983 .983 .983<br>11 .552 .552 .703 .807 .894 .953 .968 .958 .953 .950 .951 .950 550 .512 .512 .656 .755 .846 .922 .970 .982 .984 .983 .983 .983<br>600 .512 .512 .655 .753 .845 .921 .970 .982 .984 .983 .984 .984<br>12 .539 .539 .688 .792 .882 .949 .976 .969 .965 .962 .962 .962 650 .512 .511 .655 .753 .844 .921 .970 .982 .985 .984 .984 .984<br>13 .530 .530 .677 .780 .871 .944 .979 .977 .973 .971 .971 .971 700 .511 .511 .654 .752 .844 .920 .969 .982 .985 .984 .985 .985<br>14 .524 .524 .669 .771 .863 .938 .980 .983 .980 .978 .978 .978 750 .511 .510 .653 .752 .843 .920 .969 .982 .985 .985 .985 .985<br>15 .520 .519 .664 .765 .857 .933 .979 .986 .985 .983 .983 .983 800 .510 .510 .652 .751 .842 .919 .968 .982 .985 .984 .985 .985<br>850 .510 .510 .652 .750 .841 .919 .968 .982 .985 .985 .985 .985<br>16 .517 .517 .660 .760 .852 .929 .977 .988 .988 .987 .986 .987<br>900 .510 .510 .652 .750 .841 .918 .968 .982 .986 .985 .986 .986<br>17 .515 .515 .657 .757 .848 .926 .975 .988 .990 .989 .989 .989 950 .509 .509 .652 .750 .841 .918 .968 .982 .986 .986 .986 .986<br>18 .513 .513 .655 .754 .846 .924 .973 .988 .992 .991 .991 .991 1000 .509 .508 .651 .749 .840 .917 .967 .982 .986 .985 .986 .986<br>19 .512 .512 .653 .752 .843 .921 .972 .988 .992 .992 .992 .993 1050 .509 .509 .651 .749 .840 .917 .967 .981 .986 .985 .986 .986<br>20 .511 .511 .652 .751 .842 .920 .970 .987 .993 .993 .993 .993 1100 .510 .509 .651 .749 .840 .916 .967 .981 .986 .986 .986 .986<br>1150 .509 .508 .650 .748 .839 .916 .966 .981 .986 .986 .986 .986<br>21 .511 .511 .651 .750 .840 .918 .969 .986 .992 .993 .994 .994<br>1200 .508 .508 .650 .748 .839 .916 .966 .981 .986 .986 .986 .986<br>22 .510 .510 .649 .749 .839 .917 .967 .984 .991 .993 .993 .993 1250 .508 .508 .650 .748 .839 .916 .966 .981 .986 .986 .987 .987<br>23 .508 .508 .646 .746 .835 .913 .963 .981 .988 .989 .990 .990 1300 .508 .508 .650 .748 .838 .915 .966 .981 .986 .986 .986 .986<br>Figure 8: Similarity of Errors. The best matching steps are highlighted in yellow.<br>Similarity of Induced Weight w (Transformers v.s. Iterative Newton)<br>Trasformer Layer Index Similarity of Induced Weight w (Transformers v.s. Gradient Descent)<br>1 2 3 4 5 6 7 8 9 10 11 12 Trasformer Layer Index<br>1 2 3 4 5 6 7 8 9 10 11 12<br>1 -.000 .001 .859 .811 .742 .719 .714 .711 .711 .711 .711 .712<br>1 .069 -.002 .771 .731 .695 .683 .674 .671 .671 .669 .670 .670<br>2 .000 .001 .872 .856 .795 .769 .763 .760 .760 .760 .760 .761<br>50 .020 .004 .772 .880 .934 .958 .965 .963 .964 .962 .964 .964<br>3 .001 .001 .870 .890 .844 .816 .809 .806 .806 .805 .806 .806 100 .020 .005 .757 .866 .927 .959 .971 .970 .971 .969 .971 .971<br>4 .002 .001 .857 .909 .883 .857 .849 .845 .845 .845 .845 .846 150 .020 .005 .752 .861 .923 .958 .972 .973 .974 .972 .974 .974<br>5 .003 .000 .838 .915 .911 .889 .881 .877 .877 .876 .877 .877 200 .020 .005 .749 .858 .921 .957 .973 .974 .975 .973 .975 .975<br>6 .004 -.000 .819 .912 .928 .914 .906 .902 .902 .901 .902 .902 250 .020 .005 .748 .856 .919 .956 .973 .974 .976 .974 .976 .976<br>300 .019 .005 .747 .855 .918 .955 .973 .975 .976 .974 .976 .976<br>7 .005 -.000 .801 .903 .937 .932 .926 .922 .921 .921 .922 .922<br>350 .020 .005 .746 .854 .917 .954 .973 .975 .976 .975 .977 .977<br>8 .005 -.001 .785 .893 .939 .944 .941 .937 .936 .936 .937 .937 400 .020 .005 .745 .854 .917 .954 .972 .975 .977 .975 .977 .977<br>9 .005 -.001 .773 .883 .936 .951 .952 .948 .947 .947 .948 .948 450 .020 .005 .745 .853 .916 .953 .972 .975 .977 .975 .977 .977<br>10 .006 -.001 .763 .874 .932 .953 .959 .955 .955 .955 .956 .956 500 .020 .005 .744 .853 .916 .953 .972 .975 .977 .976 .977 .977<br>11 .006 -.000 .756 .867 .927 .954 .963 .961 .961 .960 .961 .961 550 .020 .005 .744 .852 .915 .953 .972 .975 .977 .976 .977 .977<br>600 .019 .005 .744 .852 .915 .953 .972 .975 .977 .976 .978 .977<br>12 .006 -.000 .750 .862 .923 .953 .966 .965 .965 .964 .965 .965 650 .020 .005 .744 .852 .915 .952 .972 .975 .977 .976 .978 .978<br>13 .006 -.000 .747 .858 .920 .952 .967 .967 .968 .967 .968 .968 700 .020 .005 .743 .851 .915 .952 .972 .975 .977 .976 .978 .978<br>14 .006 .000 .744 .855 .918 .950 .968 .969 .969 .969 .970 .970 750 .020 .005 .743 .851 .914 .952 .971 .975 .977 .976 .978 .978<br>15 .006 .000 .742 .853 .916 .949 .967 .970 .971 .970 .971 .971 800 .020 .005 .743 .851 .914 .952 .971 .975 .977 .976 .978 .978<br>850 .020 .005 .743 .851 .914 .952 .971 .975 .977 .976 .978 .978<br>16 .006 .000 .741 .851 .914 .948 .967 .970 .972 .971 .972 .972<br>900 .020 .005 .743 .851 .914 .952 .971 .975 .977 .976 .978 .978<br>17 .007 .000 .740 .850 .913 .947 .966 .970 .972 .972 .973 .973 950 .020 .005 .743 .851 .914 .952 .971 .975 .977 .976 .978 .978<br>18 .007 .000 .739 .849 .912 .946 .966 .970 .973 .972 .973 .974 1000 .020 .005 .743 .851 .914 .951 .971 .975 .977 .976 .978 .978<br>19 .007 .000 .739 .849 .911 .945 .966 .970 .973 .973 .974 .974 1050 .020 .005 .743 .851 .914 .951 .971 .975 .978 .976 .978 .978<br>20 .007 .000 .738 .848 .911 .945 .965 .970 .973 .973 .974 .974 1100 .020 .005 .742 .850 .914 .951 .971 .975 .978 .976 .978 .978<br>1150 .020 .005 .742 .851 .914 .951 .971 .975 .978 .976 .978 .978<br>21 .007 .000 .738 .848 .911 .944 .965 .970 .973 .973 .974 .974<br>1200 .020 .005 .742 .850 .913 .951 .971 .975 .978 .976 .978 .978<br>22 .007 .000 .738 .848 .910 .944 .965 .970 .973 .973 .974 .974 1250 .020 .005 .742 .850 .913 .951 .971 .975 .978 .976 .978 .978<br>23 .007 .000 .738 .848 .910 .944 .965 .970 .973 .973 .974 .974 1300 .019 .005 .742 .850 .913 .951 .971 .975 .978 .976 .978 .978<br> Iterative newton Steps  Gradient descent Steps<br> Iterative newton Steps  Gradient descent Steps<br><!-- End of picture text -->



<!-- Start of picture text -->
.954 .953 .870 .770 .692 .645 .620 .610 .606 .600 .607 .603<br>.578 .577 .733 .831 .905 .946 .954 .946 .941 .939 .939 .938<br>.543 .543 .694 .795 .883 .944 .970 .967 .963 .961 .962 .961<br>.531 .531 .679 .781 .871 .939 .973 .974 .972 .970 .970 .969<br>.525 .524 .672 .772 .863 .935 .974 .977 .976 .974 .974 .974<br>.521 .521 .667 .767 .858 .932 .974 .979 .978 .977 .977 .977<br>.518 .518 .664 .763 .855 .929 .973 .980 .980 .979 .979 .979<br>.516 .516 .661 .761 .852 .927 .973 .981 .981 .980 .980 .980<br>.515 .515 .660 .759 .850 .926 .972 .982 .983 .981 .981 .981<br>.514 .514 .658 .757 .849 .924 .972 .982 .983 .982 .982 .982<br>.514 .514 .657 .756 .847 .923 .971 .982 .984 .983 .983 .983<br>.512 .512 .656 .755 .846 .922 .970 .982 .984 .983 .983 .983<br>.512 .512 .655 .753 .845 .921 .970 .982 .984 .983 .984 .984<br>.512 .511 .655 .753 .844 .921 .970 .982 .985 .984 .984 .984<br>.511 .511 .654 .752 .844 .920 .969 .982 .985 .984 .985 .985<br>.511 .510 .653 .752 .843 .920 .969 .982 .985 .985 .985 .985<br>.510 .510 .652 .751 .842 .919 .968 .982 .985 .984 .985 .985<br>.510 .510 .652 .750 .841 .919 .968 .982 .985 .985 .985 .985<br>.510 .510 .652 .750 .841 .918 .968 .982 .986 .985 .986 .986<br>.509 .509 .652 .750 .841 .918 .968 .982 .986 .986 .986 .986<br>.509 .508 .651 .749 .840 .917 .967 .982 .986 .985 .986 .986<br>.509 .509 .651 .749 .840 .917 .967 .981 .986 .985 .986 .986<br>.510 .509 .651 .749 .840 .916 .967 .981 .986 .986 .986 .986<br>.509 .508 .650 .748 .839 .916 .966 .981 .986 .986 .986 .986<br>.508 .508 .650 .748 .839 .916 .966 .981 .986 .986 .986 .986<br>.508 .508 .650 .748 .839 .916 .966 .981 .986 .986 .987 .987<br>.508 .508 .650 .748 .838 .915 .966 .981 .986 .986 .986 .986<br><!-- End of picture text -->



<!-- Start of picture text -->
-.000 .001 .859 .811 .742 .719 .714 .711 .711 .711 .711 .712<br>.000 .001 .872 .856 .795 .769 .763 .760 .760 .760 .760 .761<br>.001 .001 .870 .890 .844 .816 .809 .806 .806 .805 .806 .806<br>.002 .001 .857 .909 .883 .857 .849 .845 .845 .845 .845 .846<br>.003 .000 .838 .915 .911 .889 .881 .877 .877 .876 .877 .877<br>.004 -.000 .819 .912 .928 .914 .906 .902 .902 .901 .902 .902<br>.005 -.000 .801 .903 .937 .932 .926 .922 .921 .921 .922 .922<br>.005 -.001 .785 .893 .939 .944 .941 .937 .936 .936 .937 .937<br>.005 -.001 .773 .883 .936 .951 .952 .948 .947 .947 .948 .948<br>.006 -.001 .763 .874 .932 .953 .959 .955 .955 .955 .956 .956<br>.006 -.000 .756 .867 .927 .954 .963 .961 .961 .960 .961 .961<br>.006 -.000 .750 .862 .923 .953 .966 .965 .965 .964 .965 .965<br>.006 -.000 .747 .858 .920 .952 .967 .967 .968 .967 .968 .968<br>.006 .000 .744 .855 .918 .950 .968 .969 .969 .969 .970 .970<br>.006 .000 .742 .853 .916 .949 .967 .970 .971 .970 .971 .971<br>.006 .000 .741 .851 .914 .948 .967 .970 .972 .971 .972 .972<br>.007 .000 .740 .850 .913 .947 .966 .970 .972 .972 .973 .973<br>.007 .000 .739 .849 .912 .946 .966 .970 .973 .972 .973 .974<br>.007 .000 .739 .849 .911 .945 .966 .970 .973 .973 .974 .974<br>.007 .000 .738 .848 .911 .945 .965 .970 .973 .973 .974 .974<br>.007 .000 .738 .848 .911 .944 .965 .970 .973 .973 .974 .974<br>.007 .000 .738 .848 .910 .944 .965 .970 .973 .973 .974 .974<br>.007 .000 .738 .848 .910 .944 .965 .970 .973 .973 .974 .974<br><!-- End of picture text -->



<!-- Start of picture text -->
.069 -.002 .771 .731 .695 .683 .674 .671 .671 .669 .670 .670<br>.020 .004 .772 .880 .934 .958 .965 .963 .964 .962 .964 .964<br>.020 .005 .757 .866 .927 .959 .971 .970 .971 .969 .971 .971<br>.020 .005 .752 .861 .923 .958 .972 .973 .974 .972 .974 .974<br>.020 .005 .749 .858 .921 .957 .973 .974 .975 .973 .975 .975<br>.020 .005 .748 .856 .919 .956 .973 .974 .976 .974 .976 .976<br>.019 .005 .747 .855 .918 .955 .973 .975 .976 .974 .976 .976<br>.020 .005 .746 .854 .917 .954 .973 .975 .976 .975 .977 .977<br>.020 .005 .745 .854 .917 .954 .972 .975 .977 .975 .977 .977<br>.020 .005 .745 .853 .916 .953 .972 .975 .977 .975 .977 .977<br>.020 .005 .744 .853 .916 .953 .972 .975 .977 .976 .977 .977<br>.020 .005 .744 .852 .915 .953 .972 .975 .977 .976 .977 .977<br>.019 .005 .744 .852 .915 .953 .972 .975 .977 .976 .978 .977<br>.020 .005 .744 .852 .915 .952 .972 .975 .977 .976 .978 .978<br>.020 .005 .743 .851 .915 .952 .972 .975 .977 .976 .978 .978<br>.020 .005 .743 .851 .914 .952 .971 .975 .977 .976 .978 .978<br>.020 .005 .743 .851 .914 .952 .971 .975 .977 .976 .978 .978<br>.020 .005 .743 .851 .914 .952 .971 .975 .977 .976 .978 .978<br>.020 .005 .743 .851 .914 .952 .971 .975 .977 .976 .978 .978<br>.020 .005 .743 .851 .914 .952 .971 .975 .977 .976 .978 .978<br>.020 .005 .743 .851 .914 .951 .971 .975 .977 .976 .978 .978<br>.020 .005 .743 .851 .914 .951 .971 .975 .978 .976 .978 .978<br>.020 .005 .742 .850 .914 .951 .971 .975 .978 .976 .978 .978<br>.020 .005 .742 .851 .914 .951 .971 .975 .978 .976 .978 .978<br>.020 .005 .742 .850 .913 .951 .971 .975 .978 .976 .978 .978<br>.020 .005 .742 .850 .913 .951 .971 .975 .978 .976 .978 .978<br>.019 .005 .742 .850 .913 .951 .971 .975 .978 .976 .978 .978<br><!-- End of picture text -->

Figure 9: **Similarity of Induced Weight Vectors.** The best matching steps are highlighted in yellow. 

23 

||||Sim|ilarity|of Err|Trasfo<br>ors  (|rmer<br>Transfo|Layer<br>rmers|Index<br>v.s. Gr|adient|Desce|nt)||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||1|2|3|4|5<br>|6<br>|7<br>|8<br>|9|10|11|12|
||1|.953|.953|.870|.771|.692|.645|.620|.609|.605|.599|.606|.602|
||2|.910|.910|.903|.826|.750|.703|.676|.665|.660|.655|.661|.657|
||4|.842|.841|.913|.878|.816|.773|.746|.733|.728|.724|.728|.725|
||8|.759|.759|.886|.905|.876|.846|.820|.807|.801|.798|.801|.799|
|eps|16|.678|.677|.831|.895|.910|.903|.886|.873|.867|.865|.867|.865|
|ent St|32|.610|.610|.768|.858|.914|.938|.934|.924|.918|.916|.917|.916|
|desc|64|.563|.563|.717|.817|.897|.947|.961|.954|.950|.948|.948|.947|
|adient|128|.536|.535|.685|.786|.875|.941|.972|.971|.968|.966|.967|.966|
|Gr|256|.521|.521|.666|.766|.858|.932|.973|.979|.978|.977|.977|.977|
||512|.513|.513|.656|.755|.847|.923|.971|.982|.984|.982|.983|.983|
|1|024|.509|.509|.652|.749|.840|.917|.967|.982|.986|.985|.986|.986|
|2|048|.507|.507|.648|.745|.836|.913|.964|.980|.986|.986|.987|.987|
|4|096|.506|.505|.646|.744|.834|.911|.962|.979|.985|.987|.988|.988|





<!-- Start of picture text -->
Similarity of Errors  (Transformers v.s. Gradient Descent)<br>Trasformer Layer Index<br>1 2 3 4 5 6 7 8 9 10 11 12<br>.953 .953 .870 .771 .692 .645 .620 .609 .605 .599 .606 .602<br>.910 .910 .903 .826 .750 .703 .676 .665 .660 .655 .661 .657<br>.842 .841 .913 .878 .816 .773 .746 .733 .728 .724 .728 .725<br>.759 .759 .886 .905 .876 .846 .820 .807 .801 .798 .801 .799<br>.678 .677 .831 .895 .910 .903 .886 .873 .867 .865 .867 .865<br>.610 .610 .768 .858 .914 .938 .934 .924 .918 .916 .917 .916<br>.563 .563 .717 .817 .897 .947 .961 .954 .950 .948 .948 .947<br>.536 .535 .685 .786 .875 .941 .972 .971 .968 .966 .967 .966<br>.521 .521 .666 .766 .858 .932 .973 .979 .978 .977 .977 .977<br>.513 .513 .656 .755 .847 .923 .971 .982 .984 .982 .983 .983<br>.509 .509 .652 .749 .840 .917 .967 .982 .986 .985 .986 .986<br>.507 .507 .648 .745 .836 .913 .964 .980 .986 .986 .987 .987<br>.506 .505 .646 .744 .834 .911 .962 .979 .985 .987 .988 .988<br><!-- End of picture text -->

Figure 10: **Similarity of Errors of Gradient Descent in Log Scale.** The best matching steps are highlighted in yellow. Putting the number of steps of Gradient Descent in log scale further verifies the claim that Transformer’s rate of covergence is exponentially faster than that of Gradient Descent. 

24 

#### **A.2.3 Comparison with Other Second-Order Methods** 

In this section, we ablate with alternative second-order methods, such as Conjugate Gradient, BFGS, and its limited memory variant, L-BFGS. 

**Conjugate Gradient Method.** For linear regression problems, the Conjugate Gradient (CG) method solves the linear system 



CG finds the weight vector **_w_** ˆ<sup>_CG_</sup> with initialization **_w_** 0 by maintain a set of conjugate gradient _{_ ∆ **_w_** 1 _, · · · ,_ ∆ **_w_** _k}_ . It follows the iterative update rule 



The conjugate Gradient method requires _O_ (<sup>_√_</sup> _<u>κ</u>_ log(1 _/ϵ_ )) steps to converge to an _ϵ_ error on quadratic objectives such as linear regression. 

**BFGS.** Broyden– Fletcher–Goldfarb–Shanno (BFGS) is a Quasi-Newton method, designed to approximate the inverse Hessian **_B_** _k_ : _≈∇_<sup>2</sup> _L_ ( **_w_** _k_ )<sup>_−_1</sup> . The BFGS updates are given by 



where 



When _k_ is large, **_B_** _k_ approximates the inverse Hessian well. 

**L(imited-memory)-BFGS.** L-BFGS is a limited-memory version of BFGS. Instead of the inverse Hessian **_B_** _k_ , L-BFGS maintains a history of past _m_ updates (where _m_ is usually small). Recall the iterative update rule of **_B_** _k_ in BFGS 



Unlike BFGS, which recursively unroll to an initialization **_B_** 0, L-BFGS only unroll to **_B_** _k−m_ but replacing **_B_** _k−m_ with **_B_** init. In this regard, running _n_ steps of L-BFGS only requires _O_ ( _mn_ ) memory, which is more memory-efficient than BFGS who requires _O_ ( _n_<sup>2</sup> ) memory. The trade-off is that L- BFGS won’t have a good estimate of the inverse Hessian when _m < d_ , where _d_ is the dimensionality of the quadratic problem. In this regard, it will converge slower than full BFGS. 

In Figure 11 and Figure 12, we compare Transformers with BFGS, L-BFGS, and Conjugate Gradient method on the metric of similarity of errors. We find that Transformers have a similar _linear_ correspondence with BFGS. This is perhaps not surprising, given that BFGS also gets a superlinear convergence rate for linear regression Nocedal and Wright [1999]. Meanwhile, Transformers show a substantially faster convergence rate than L-BFGS and CG. 

25 



<!-- Start of picture text -->
Similarity of Errors  (Transformers v.s. Bfgs)<br>1 2 3 4 Transformer Layer Index5 6 7 8 9 10 11 12<br>1 .977 .977 .863 .759 .675 .628 .601 .587 .583 .578 .586 .581<br>2 .911 .911 .826 .825 .779 .735 .706 .690 .684 .679 .686 .682<br>3 .801 .801 .719 .783 .817 .801 .775 .760 .754 .750 .755 .752<br>4 .687 .687 .616 .706 .805 .829 .816 .802 .796 .793 .797 .794<br>5 .585 .585 .537 .633 .767 .834 .839 .829 .823 .820 .823 .822<br>6 .505 .505 .498 .591 .730 .829 .859 .852 .848 .847 .848 .847<br>7 .467 .467 .512 .602 .722 .828 .876 .876 .873 .872 .872 .872<br>8 .471 .471 .558 .647 .744 .838 .892 .895 .893 .892 .892 .892<br>9 .491 .491 .608 .698 .781 .858 .907 .912 .910 .908 .908 .908<br>10 .512 .512 .650 .742 .818 .880 .921 .926 .925 .923 .923 .923<br>11 .526 .526 .676 .771 .846 .900 .934 .939 .938 .935 .936 .936<br>12 .532 .532 .687 .784 .864 .917 .946 .951 .950 .947 .948 .948<br>13 .530 .530 .687 .787 .871 .930 .958 .963 .962 .959 .960 .960<br>14 .525 .525 .681 .782 .872 .936 .968 .973 .972 .970 .970 .970<br>15 .518 .518 .671 .772 .864 .936 .973 .979 .979 .977 .978 .978<br>16 .511 .511 .662 .762 .857 .933 .976 .984 .984 .982 .983 .983<br>17 .506 .506 .656 .756 .850 .929 .976 .986 .987 .985 .986 .986<br>18 .504 .504 .653 .753 .847 .927 .976 .987 .989 .988 .988 .988<br>19 .504 .504 .653 .752 .846 .925 .975 .988 .991 .990 .990 .990<br>20 .504 .504 .652 .751 .846 .924 .974 .989 .993 .992 .992 .992<br>21 .503 .503 .652 .751 .845 .923 .973 .989 .993 .993 .994 .994<br>22 .503 .503 .651 .750 .843 .922 .972 .988 .994 .994 .994 .995<br>23 .502 .502 .650 .748 .842 .920 .971 .987 .994 .995 .995 .996<br>24 .501 .501 .649 .747 .841 .919 .969 .987 .994 .995 .996 .996<br>25 .500 .500 .648 .747 .840 .918 .969 .986 .993 .995 .996 .996<br>26 .500 .500 .648 .746 .839 .918 .968 .986 .993 .995 .996 .996<br>27 .500 .500 .647 .746 .839 .917 .968 .985 .993 .995 .996 .996<br>28 .500 .500 .647 .745 .839 .917 .967 .985 .993 .995 .996 .996<br>29 .500 .500 .647 .745 .839 .917 .967 .985 .992 .995 .996 .996<br>30 .500 .500 .647 .745 .839 .917 .967 .985 .992 .995 .996 .996<br>31 .500 .500 .647 .745 .839 .917 .967 .985 .992 .995 .996 .996<br>32 .500 .500 .647 .745 .838 .917 .967 .984 .992 .995 .996 .996<br>33 .500 .500 .647 .745 .838 .917 .967 .984 .992 .995 .996 .996<br>34 .500 .500 .647 .745 .838 .917 .967 .984 .992 .995 .996 .996<br>35 .500 .500 .647 .745 .838 .916 .967 .984 .992 .995 .996 .996<br>36 .500 .500 .647 .745 .838 .916 .967 .984 .992 .995 .996 .996<br>37 .500 .500 .647 .745 .838 .916 .967 .984 .992 .995 .996 .996<br>38 .500 .500 .647 .745 .838 .916 .967 .984 .992 .994 .995 .996<br>39 .500 .500 .647 .745 .838 .916 .967 .984 .992 .994 .995 .996<br>40 .500 .500 .647 .745 .838 .916 .967 .984 .992 .994 .995 .996<br> BFGS Steps<br><!-- End of picture text -->



<!-- Start of picture text -->
Similarity of Errors  (Transformers v.s. L-Bfgs)<br>1 2 3 4 Transformer Layer Index5 6 7 8 9 10 11 12<br>1 .979 .979 .863 .762 .678 .629 .604 .591 .588 .581 .589 .585<br>2 .783 .783 .924 .913 .846 .797 .767 .753 .747 .742 .747 .744<br>3 .722 .722 .894 .928 .893 .851 .821 .807 .801 .796 .800 .798<br>4 .626 .626 .806 .899 .929 .916 .891 .877 .870 .867 .869 .868<br>5 .594 .594 .766 .870 .930 .940 .922 .908 .901 .899 .900 .899<br>6 .589 .589 .754 .858 .929 .949 .936 .923 .917 .914 .915 .915<br>7 .579 .579 .739 .841 .921 .956 .951 .939 .933 .930 .932 .931<br>8 .565 .565 .722 .824 .911 .959 .963 .952 .947 .945 .946 .945<br>9 .549 .549 .705 .807 .899 .959 .973 .964 .960 .957 .958 .958<br>10 .537 .537 .691 .794 .888 .955 .978 .972 .968 .965 .966 .966<br>11 .532 .532 .685 .788 .882 .953 .980 .976 .973 .970 .971 .971<br>12 .528 .528 .681 .784 .878 .951 .982 .980 .976 .974 .975 .974<br>13 .525 .525 .677 .779 .874 .947 .982 .983 .980 .978 .978 .978<br>14 .522 .522 .673 .774 .869 .944 .982 .985 .983 .981 .981 .981<br>15 .519 .519 .669 .771 .865 .941 .982 .987 .985 .983 .984 .984<br>16 .517 .517 .667 .768 .863 .939 .982 .988 .987 .985 .985 .985<br>17 .516 .516 .665 .766 .860 .937 .981 .988 .988 .986 .986 .986<br>18 .514 .514 .663 .764 .859 .936 .980 .989 .989 .987 .988 .988<br>19 .513 .513 .662 .763 .857 .934 .980 .989 .990 .988 .989 .989<br>20 .512 .512 .661 .762 .856 .933 .979 .989 .991 .989 .989 .989<br>21 .511 .511 .660 .761 .855 .932 .978 .990 .991 .990 .990 .990<br>22 .511 .511 .659 .760 .854 .931 .977 .990 .992 .991 .991 .991<br>23 .510 .510 .658 .759 .853 .930 .977 .989 .992 .991 .992 .992<br>24 .509 .509 .657 .758 .852 .929 .976 .989 .993 .992 .992 .992<br>25 .509 .509 .657 .757 .851 .928 .975 .989 .993 .992 .993 .993<br>26 .508 .508 .656 .756 .850 .927 .975 .989 .993 .993 .993 .993<br>27 .508 .508 .656 .756 .849 .927 .974 .989 .993 .993 .993 .994<br>28 .507 .507 .655 .755 .849 .926 .974 .989 .993 .994 .994 .994<br>29 .507 .507 .655 .754 .848 .925 .973 .989 .993 .994 .994 .994<br>30 .507 .507 .654 .754 .848 .925 .973 .988 .994 .994 .994 .994<br>31 .506 .506 .654 .754 .847 .924 .973 .988 .994 .994 .994 .995<br>32 .506 .506 .654 .753 .847 .924 .972 .988 .993 .994 .995 .995<br>33 .506 .506 .653 .753 .846 .924 .972 .988 .993 .994 .995 .995<br>34 .506 .506 .653 .753 .846 .923 .972 .987 .993 .995 .995 .995<br>35 .506 .505 .653 .752 .846 .923 .971 .987 .993 .995 .995 .995<br>36 .505 .505 .653 .752 .845 .923 .971 .987 .993 .995 .995 .996<br>37 .505 .505 .652 .752 .845 .922 .970 .987 .993 .995 .995 .996<br>38 .505 .505 .652 .751 .845 .922 .970 .986 .993 .995 .995 .996<br>39 .505 .505 .652 .751 .845 .922 .970 .986 .993 .995 .995 .996<br>40 .505 .504 .652 .751 .845 .922 .970 .986 .993 .995 .996 .996<br> L-BFGS Steps<br><!-- End of picture text -->

Figure 11: **Similarity of Errors between Transformers and BFGS or L-BFGS.** The best matching steps are highlighted in yellow. We find that Transformer, from layers 6 to 11, has a linear correspondence with BFGS. For L-BFGS, due to its limited memory, it approximates second-order information more slowly and results in a slower convergence rate than Transformers. 



<!-- Start of picture text -->
Similarity of Errors  (Transformers v.s. Conjugate Gradient)<br>Transformer Layer Index<br>1 2 3 4 5 6 7 8 9 10 11 12<br>1 .841 .841 .937 .858 .767 .718 .689 .675 .670 .665 .670 .667<br>2 .731 .731 .893 .923 .871 .827 .795 .779 .772 .768 .772 .770<br>3 .665 .665 .835 .914 .917 .890 .860 .844 .837 .833 .836 .834<br>4 .618 .619 .787 .885 .930 .927 .904 .888 .881 .878 .880 .878<br>5 .586 .586 .751 .855 .925 .947 .934 .919 .912 .909 .910 .910<br>6 .565 .565 .726 .831 .914 .955 .952 .939 .932 .930 .931 .930<br>7 .552 .552 .711 .816 .904 .957 .964 .952 .947 .944 .945 .944<br>8 .544 .544 .701 .806 .896 .956 .971 .962 .956 .954 .955 .954<br>9 .538 .538 .694 .798 .890 .954 .976 .969 .964 .961 .962 .962<br>10 .534 .534 .689 .793 .885 .952 .979 .974 .969 .967 .967 .967<br>11 .530 .530 .685 .788 .881 .950 .980 .977 .974 .971 .972 .971<br>12 .527 .527 .681 .784 .877 .948 .982 .980 .977 .974 .975 .975<br>13 .524 .524 .678 .780 .874 .946 .982 .982 .979 .977 .978 .978<br>14 .522 .522 .675 .777 .871 .945 .982 .984 .982 .979 .980 .980<br>15 .520 .520 .672 .774 .868 .943 .982 .985 .983 .981 .982 .982<br>16 .518 .518 .670 .772 .866 .941 .982 .986 .985 .983 .983 .983<br>17 .517 .517 .668 .770 .864 .940 .982 .987 .986 .984 .984 .984<br>18 .515 .515 .667 .768 .863 .939 .982 .988 .987 .985 .985 .985<br>19 .514 .514 .666 .767 .861 .938 .981 .988 .988 .986 .986 .986<br>20 .513 .513 .664 .765 .860 .937 .981 .989 .989 .987 .987 .987<br>21 .512 .513 .663 .764 .859 .936 .981 .989 .989 .988 .988 .988<br>22 .512 .512 .662 .763 .858 .935 .980 .989 .990 .988 .989 .989<br>23 .511 .511 .661 .762 .857 .934 .980 .989 .990 .989 .989 .989<br>24 .510 .510 .660 .761 .856 .933 .979 .990 .991 .990 .990 .990<br>25 .510 .510 .660 .760 .855 .932 .979 .990 .991 .990 .990 .990<br>26 .509 .509 .659 .759 .854 .932 .979 .990 .992 .990 .991 .991<br>27 .509 .509 .658 .759 .853 .931 .978 .990 .992 .991 .991 .991<br>28 .508 .508 .658 .758 .852 .930 .978 .990 .992 .991 .991 .992<br>29 .508 .508 .657 .757 .852 .930 .978 .990 .992 .991 .992 .992<br>30 .508 .508 .657 .757 .851 .929 .977 .990 .993 .992 .992 .992<br>31 .507 .507 .656 .756 .851 .929 .977 .990 .993 .992 .992 .992<br>32 .507 .507 .656 .756 .850 .928 .977 .990 .993 .992 .992 .993<br>33 .507 .507 .655 .755 .850 .928 .976 .990 .993 .992 .993 .993<br>34 .506 .506 .655 .755 .849 .927 .976 .990 .993 .993 .993 .993<br>35 .506 .506 .655 .754 .849 .927 .976 .989 .993 .993 .993 .993<br>36 .506 .506 .654 .754 .848 .927 .975 .989 .993 .993 .993 .994<br>37 .505 .505 .654 .754 .848 .926 .975 .989 .994 .993 .993 .994<br>38 .505 .505 .654 .753 .847 .926 .975 .989 .994 .993 .994 .994<br>39 .505 .505 .654 .753 .847 .926 .975 .989 .994 .993 .994 .994<br>40 .505 .505 .653 .753 .847 .925 .974 .989 .994 .994 .994 .994<br> Conjugate Gradient Steps<br><!-- End of picture text -->

Figure 12: **Similarity of Errors between Transformers and Conjugate Gradient.** Transformer’s convergence rate is still faster than conjugate gradient methods. 

26 

#### **A.2.4 Additional Results on Comparison over Transformer Layers** 



<!-- Start of picture text -->
1.0 1.0<br>0.9 0.8<br>0.8 0.6<br>0.7 0.4<br>0.6 0.2<br>SimE(Transformers, Newton) SimW(Transformers, Newton)<br>SimE(Transformers, GD) SimW(Transformers, GD)<br>0.5 SimE(Transformers, OLS) 0.0 SimW(Transformers, OLS)<br>2 4 6 8 10 12 2 4 6 8 10 12<br>Layer Index Layer Index<br>(a) Similarity of Errors (b) Similarity of Induced Weights<br>Cosine Similarity Cosine Similarity<br><!-- End of picture text -->

Figure 13: Similarities between Transformer and candidate algorithms. Transformers resemble _Iterative Newton’s Method_ the most. 

#### **A.2.5 Additional Results on Similarity of Induced Weights** 

We present more details line plots for how the similarity of weights changes as the models see more in-context observations _{_ **_x_** _i, yi}_<sup>_n_</sup> _i_ =1<sup>, i.e., as</sup><sup>_n_increases.We fix the number of Transformers layers</sup><sup>_ℓ_</sup> and compare with other algorithms with their best-match steps to _ℓ_ in Figure 14. 



<!-- Start of picture text -->
1.0 Layer 2 1.0 Layer 3 1.0 Layer 12<br>CosSim(wTF, wNewton) CosSim(wTF, wNewton)<br>CosSim(wTF, wGD) CosSim(wTF, wGD)<br>0.8 CosSim(wTF, wOLS) 0.8 0.8 CosSim(wTF, wOLS)<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0.0 0.0 CosSim(wTF, wNewton) 0.0<br>CosSim(wTF, wGD)<br>CosSim(wTF, wOLS)<br>0.2 0.2 0.2<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br>Number of In-Context Examples Number of In-Context Examples Number of In-Context Examples<br>Cosine Similarity Cosine Similarity Cosine Similarity<br><!-- End of picture text -->

Figure 14: _Similarity of induced weights_ over varying number of in-context examples, on three layer indices of Transformers, indexed as 2, 3 and 12. We find that initially at layer 2, the Transformers model hasn’t learned so it has zero similarity to all candidate algorithms. As we progress to the next layer number 3, we find that Transformers start to learn, and when provided few examples, Transformers are more similar to OLS but soon become most similar to the Iterative Newton’s Method. Layer 12 shows that Transformers in the later layers converge to the OLS solution when provided more than 1 example. We also find there is a dip around _n_ = _d_ for similarity between Transformers and OLS but not for Transformers and Newton, and this is probably because OLS has a more prominent double-descent phenomenon than Transformers and Newton. 

27 

### **A.3 Varying Data Distribution or Function Class** 

#### **A.3.1 Experiments on Ill-Conditioned Problems** 

In this section, we repeat the same experiments as we did on isotropic data in the main text and in Appendix A.2, and we change the covariance matrix to be ill-conditioned such that _κ_ ( **Σ** ) = 100. 



<!-- Start of picture text -->
Errors v.s. # In-Context Examples Errors v.s. # In-Context Examples<br>10 0 10 0<br>10 1 10 1<br>Transformers Layer #01 Iterative Newton #01 (1 steps)<br>Transformers Layer #02 Iterative Newton #02 (1 steps)<br>Transformers Layer #03 Iterative Newton #03 (3 steps)<br>10 2 Transformers Layer #04 10 2 Iterative Newton #04 (5 steps)<br>Transformers Layer #05 Iterative Newton #05 (8 steps)<br>Transformers Layer #06 Iterative Newton #06 (10 steps)<br>Transformers Layer #07 Iterative Newton #07 (14 steps)<br>Transformers Layer #08 Iterative Newton #08 (17 steps)<br>Transformers Layer #09 Iterative Newton #09 (20 steps)<br>Transformers Layer #10 Iterative Newton #10 (21 steps)<br>Transformers Layer #11 Iterative Newton #11 (21 steps)<br>10 3 0 1 2Transformers Layer #123 4 5 6 7 # In-Context Examples8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 10 3 0 1 2Iterative Newton #12 (21 steps)3 4 5 6 7 # In-Context Examples8 9 10 11 12 13 14 15 16 17 18 19 20 21 22<br>(a) Transformers (b) Iterative Newton’s Method<br>Errors Errors<br><!-- End of picture text -->

Figure 15: **Progression of Algorithms on Ill-Conditioned Data.** Transformer’s performance still improves over the layer index _ℓ_ ; Iterative Newton’s Method’s performance improves over the number of iterations _t_ and we plot the best-matching _t_ to Transformer’s _ℓ_ following Definition 3.4. 

We also present the heatmaps to find the best-matching steps and conclude that Transformers are similar to Newton’s method than GD in ill-conditioned data. 

28 



<!-- Start of picture text -->
Similarity of Errors  (Transformers v.s. Iterative Newton)<br>Trasformer Layer Index Similarity of Errors  (Transformers v.s. Gradient Descent)<br>1 2 3 4 5 6 7 8 9 10 11 12 Trasformer Layer Index<br>1 2 3 4 5 6 7 8 9 10 11 12<br>1 .885 .886 .829 .713 .598 .557 .535 .529 .528 .530 .532 .529<br>1 .990 .990 .709 .548 .469 .440 .420 .413 .413 .416 .418 .413<br>2 .814 .814 .848 .780 .662 .615 .593 .587 .585 .587 .589 .586 100 .502 .503 .686 .870 .941 .921 .896 .889 .886 .887 .887 .886<br>3 .736 .736 .842 .838 .733 .679 .656 .650 .649 .650 .652 .650 200 .451 .451 .633 .839 .953 .958 .936 .929 .927 .927 .927 .926<br>4 .661 .662 .811 .878 .805 .745 .722 .716 .714 .715 .716 .715 300 .433 .433 .612 .821 .950 .970 .952 .945 .943 .943 .943 .943<br>400 .422 .423 .600 .809 .945 .975 .960 .954 .952 .952 .952 .952<br>5 .593 .593 .765 .893 .867 .808 .783 .777 .775 .775 .777 .775 500 .417 .418 .593 .802 .941 .977 .966 .960 .958 .958 .958 .958<br>6 .536 .537 .715 .887 .913 .862 .834 .828 .825 .826 .827 .826 600 .413 .413 .588 .796 .937 .978 .970 .964 .962 .962 .962 .962<br>700 .410 .410 .584 .791 .933 .978 .973 .967 .965 .965 .966 .966<br>7 .493 .494 .672 .868 .940 .903 .873 .866 .864 .864 .865 .864 800 .408 .408 .581 .788 .930 .978 .975 .970 .968 .968 .968 .968<br>8 .464 .464 .640 .847 .951 .933 .902 .894 .892 .893 .894 .893 900 .405 .406 .578 .785 .927 .977 .977 .972 .970 .970 .970 .971<br>9 .444 .445 .617 .828 .953 .953 .923 .915 .913 .913 .914 .913 1000 .404 .405 .576 .782 .925 .977 .978 .974 .972 .972 .972 .972<br>1100 .402 .403 .574 .780 .923 .976 .979 .975 .974 .974 .974 .974<br>10 .431 .432 .601 .812 .948 .966 .938 .930 .928 .928 .929 .928 1200 .401 .402 .573 .778 .921 .975 .980 .976 .975 .975 .975 .976<br>11 .422 .423 .590 .800 .942 .973 .949 .940 .939 .939 .939 .939 1300 .400 .400 .572 .776 .919 .975 .981 .977 .976 .976 .976 .977<br>1400 .399 .400 .571 .775 .918 .974 .981 .978 .977 .977 .977 .978<br>12 .416 .416 .582 .791 .935 .976 .958 .949 .947 .947 .948 .948 1500 .399 .400 .570 .774 .917 .974 .982 .980 .978 .979 .979 .979<br>13 .411 .412 .576 .784 .928 .977 .965 .956 .954 .954 .955 .956 1600 .398 .398 .569 .772 .915 .973 .982 .980 .979 .979 .979 .980<br>14 .407 .408 .572 .778 .923 .976 .971 .963 .961 .962 .962 .963 1700 .397 .398 .568 .771 .913 .972 .982 .981 .979 .980 .980 .980<br>1800 .397 .397 .567 .770 .913 .971 .983 .982 .980 .981 .981 .981<br>15 .404 .404 .567 .772 .916 .973 .976 .970 .968 .968 .969 .970 1900 .396 .396 .567 .769 .912 .971 .983 .982 .981 .981 .981 .982<br>16 .400 .400 .563 .766 .910 .970 .980 .975 .974 .974 .975 .976 2000 .395 .396 .566 .768 .910 .970 .983 .982 .981 .982 .982 .982<br>2100 .395 .395 .565 .767 .909 .970 .983 .983 .982 .982 .982 .983<br>17 .397 .397 .559 .760 .904 .966 .981 .979 .978 .979 .979 .980<br>2200 .394 .394 .564 .766 .908 .969 .983 .983 .982 .982 .983 .983<br>18 .394 .394 .555 .756 .898 .962 .982 .983 .982 .982 .983 .984 2300 .394 .395 .564 .766 .908 .969 .984 .984 .982 .983 .983 .984<br>19 .392 .392 .552 .752 .894 .958 .981 .985 .984 .985 .986 .986 2400 .393 .393 .563 .765 .907 .968 .983 .984 .983 .983 .983 .984<br>2500 .393 .394 .563 .765 .907 .968 .984 .985 .984 .984 .984 .985<br>20 .390 .390 .549 .748 .890 .954 .979 .985 .985 .986 .987 .988 2600 .393 .394 .563 .764 .905 .967 .984 .985 .984 .984 .984 .985<br>21 .389 .389 .548 .746 .887 .951 .977 .985 .985 .986 .987 .988 2700 .393 .394 .562 .763 .905 .967 .984 .985 .984 .984 .984 .985<br>22 .387 .388 .545 .743 .883 .947 .973 .983 .983 .984 .985 .986 2800 .392 .392 .562 .763 .904 .966 .983 .985 .984 .984 .984 .985<br>2900 .392 .392 .561 .762 .903 .965 .983 .985 .984 .984 .985 .985<br>23 .384 .385 .538 .733 .872 .935 .962 .972 .972 .973 .974 .975 3000 .391 .392 .561 .762 .903 .965 .984 .985 .984 .985 .985 .986<br>Figure 16: Similarity of Errors on Ill-Conditioned Data. The best matching steps are highlighted<br>in yellow.<br>Similarity of Induced Weight w (Transformers v.s. Iterative Newton)<br>Trasformer Layer Index Similarity of Induced Weight w (Transformers v.s. Gradient Descent)<br>1 2 3 4 5 6 7 8 9 10 11 12 Trasformer Layer Index<br>1 2 3 4 5 6 7 8 9 10 11 12<br>1 .003 .023 .646 .739 .747 .721 .650 .626 .617 .615 .612 .608<br>1 .010 -.062 .292 .337 .346 .333 .294 .287 .280 .284 .273 .274<br>2 .003 .024 .659 .778 .793 .765 .690 .664 .654 .653 .649 .645 100 .009 .010 .625 .829 .913 .907 .831 .807 .798 .796 .790 .787<br>3 .003 .024 .662 .808 .834 .805 .726 .699 .688 .687 .683 .679 200 .008 .011 .611 .821 .916 .924 .858 .839 .830 .828 .822 .820<br>4 .002 .024 .655 .827 .868 .838 .755 .728 .717 .715 .711 .707 300 .008 .012 .602 .812 .912 .931 .874 .858 .850 .848 .843 .841<br>400 .008 .012 .595 .804 .906 .934 .884 .873 .864 .862 .858 .857<br>5 .002 .024 .644 .836 .893 .864 .779 .751 .740 .738 .734 .729 500 .008 .012 .589 .796 .899 .935 .892 .883 .875 .873 .870 .868<br>6 .002 .024 .632 .838 .907 .881 .795 .766 .755 .753 .749 .744 600 .008 .013 .583 .789 .893 .934 .897 .892 .884 .882 .879 .878<br>700 .008 .013 .577 .782 .886 .932 .901 .898 .891 .889 .886 .885<br>7 .002 .023 .622 .835 .915 .892 .805 .777 .765 .764 .760 .755 800 .008 .013 .572 .775 .880 .930 .904 .904 .896 .895 .892 .892<br>8 .001 .023 .615 .831 .919 .900 .814 .785 .773 .772 .768 .763 900 .008 .014 .568 .769 .875 .928 .906 .908 .901 .899 .898 .897<br>9 .002 .023 .610 .827 .920 .906 .820 .792 .780 .779 .775 .770 1000 .008 .014 .564 .764 .870 .926 .908 .912 .905 .903 .902 .901<br>1100 .007 .014 .560 .759 .865 .924 .909 .915 .908 .907 .906 .905<br>10 .002 .023 .606 .824 .920 .911 .828 .800 .788 .787 .783 .778 1200 .007 .014 .557 .755 .860 .922 .910 .918 .911 .910 .909 .909<br>11 .002 .023 .603 .821 .919 .917 .837 .810 .798 .797 .793 .789 1300 .007 .014 .554 .750 .856 .919 .910 .920 .914 .912 .912 .911<br>1400 .007 .014 .551 .747 .852 .917 .911 .922 .916 .914 .914 .914<br>12 .002 .023 .599 .816 .917 .924 .851 .826 .813 .812 .809 .804 1500 .007 .014 .548 .743 .848 .915 .911 .923 .918 .916 .916 .916<br>13 .002 .023 .592 .807 .910 .930 .868 .846 .834 .832 .829 .825 1600 .007 .014 .546 .740 .845 .913 .911 .925 .919 .918 .918 .918<br>14 .002 .022 .579 .791 .896 .932 .885 .868 .856 .855 .853 .849 1700 .007 .015 .544 .737 .842 .911 .911 .926 .921 .920 .920 .920<br>1800 .007 .015 .542 .734 .839 .909 .911 .927 .922 .921 .921 .922<br>15 .002 .021 .562 .768 .873 .926 .897 .889 .878 .877 .876 .873 1900 .008 .015 .540 .731 .836 .907 .910 .928 .923 .922 .923 .923<br>16 .002 .021 .544 .744 .849 .914 .904 .906 .895 .894 .895 .893 2000 .007 .015 .538 .729 .834 .906 .910 .929 .924 .923 .924 .924<br>2100 .007 .015 .536 .726 .831 .904 .910 .930 .925 .924 .925 .926<br>17 .002 .020 .528 .722 .826 .900 .905 .918 .909 .908 .910 .909<br>2200 .007 .015 .534 .724 .829 .902 .910 .931 .926 .925 .926 .927<br>18 .002 .020 .515 .704 .807 .886 .902 .925 .918 .918 .922 .921 2300 .007 .015 .533 .722 .827 .901 .909 .931 .927 .926 .927 .928<br>19 .003 .019 .505 .690 .792 .873 .897 .929 .924 .925 .929 .929 2400 .007 .015 .531 .720 .825 .900 .909 .932 .928 .927 .928 .929<br>2500 .007 .015 .530 .718 .822 .898 .909 .932 .928 .927 .929 .929<br>20 .003 .019 .498 .680 .781 .863 .891 .931 .926 .928 .933 .933 2600 .007 .015 .529 .716 .821 .897 .908 .933 .929 .928 .929 .930<br>21 .003 .019 .492 .672 .772 .854 .885 .930 .927 .929 .935 .935 2700 .007 .015 .528 .715 .819 .896 .908 .933 .929 .929 .930 .931<br>22 .003 .019 .488 .666 .766 .848 .880 .929 .926 .929 .935 .935 2800 .007 .015 .527 .713 .817 .894 .908 .934 .930 .929 .931 .932<br>2900 .007 .015 .525 .712 .816 .893 .907 .934 .930 .929 .931 .932<br>23 .003 .019 .485 .662 .761 .843 .877 .927 .925 .927 .934 .935 3000 .007 .015 .524 .710 .814 .892 .907 .934 .931 .930 .932 .933<br> Iterative newton Steps  Gradient descent Steps<br> Iterative newton Steps  Gradient descent Steps<br><!-- End of picture text -->



<!-- Start of picture text -->
.990 .990 .709 .548 .469 .440 .420 .413 .413 .416 .418 .413<br>.502 .503 .686 .870 .941 .921 .896 .889 .886 .887 .887 .886<br>.451 .451 .633 .839 .953 .958 .936 .929 .927 .927 .927 .926<br>.433 .433 .612 .821 .950 .970 .952 .945 .943 .943 .943 .943<br>.422 .423 .600 .809 .945 .975 .960 .954 .952 .952 .952 .952<br>.417 .418 .593 .802 .941 .977 .966 .960 .958 .958 .958 .958<br>.413 .413 .588 .796 .937 .978 .970 .964 .962 .962 .962 .962<br>.410 .410 .584 .791 .933 .978 .973 .967 .965 .965 .966 .966<br>.408 .408 .581 .788 .930 .978 .975 .970 .968 .968 .968 .968<br>.405 .406 .578 .785 .927 .977 .977 .972 .970 .970 .970 .971<br>.404 .405 .576 .782 .925 .977 .978 .974 .972 .972 .972 .972<br>.402 .403 .574 .780 .923 .976 .979 .975 .974 .974 .974 .974<br>.401 .402 .573 .778 .921 .975 .980 .976 .975 .975 .975 .976<br>.400 .400 .572 .776 .919 .975 .981 .977 .976 .976 .976 .977<br>.399 .400 .571 .775 .918 .974 .981 .978 .977 .977 .977 .978<br>.399 .400 .570 .774 .917 .974 .982 .980 .978 .979 .979 .979<br>.398 .398 .569 .772 .915 .973 .982 .980 .979 .979 .979 .980<br>.397 .398 .568 .771 .913 .972 .982 .981 .979 .980 .980 .980<br>.397 .397 .567 .770 .913 .971 .983 .982 .980 .981 .981 .981<br>.396 .396 .567 .769 .912 .971 .983 .982 .981 .981 .981 .982<br>.395 .396 .566 .768 .910 .970 .983 .982 .981 .982 .982 .982<br>.395 .395 .565 .767 .909 .970 .983 .983 .982 .982 .982 .983<br>.394 .394 .564 .766 .908 .969 .983 .983 .982 .982 .983 .983<br>.394 .395 .564 .766 .908 .969 .984 .984 .982 .983 .983 .984<br>.393 .393 .563 .765 .907 .968 .983 .984 .983 .983 .983 .984<br>.393 .394 .563 .765 .907 .968 .984 .985 .984 .984 .984 .985<br>.393 .394 .563 .764 .905 .967 .984 .985 .984 .984 .984 .985<br>.393 .394 .562 .763 .905 .967 .984 .985 .984 .984 .984 .985<br>.392 .392 .562 .763 .904 .966 .983 .985 .984 .984 .984 .985<br>.392 .392 .561 .762 .903 .965 .983 .985 .984 .984 .985 .985<br>.391 .392 .561 .762 .903 .965 .984 .985 .984 .985 .985 .986<br><!-- End of picture text -->

Figure 16: **Similarity of Errors on Ill-Conditioned Data.** The best matching steps are highlighted in yellow. 



<!-- Start of picture text -->
.003 .023 .646 .739 .747 .721 .650 .626 .617 .615 .612 .608<br>.003 .024 .659 .778 .793 .765 .690 .664 .654 .653 .649 .645<br>.003 .024 .662 .808 .834 .805 .726 .699 .688 .687 .683 .679<br>.002 .024 .655 .827 .868 .838 .755 .728 .717 .715 .711 .707<br>.002 .024 .644 .836 .893 .864 .779 .751 .740 .738 .734 .729<br>.002 .024 .632 .838 .907 .881 .795 .766 .755 .753 .749 .744<br>.002 .023 .622 .835 .915 .892 .805 .777 .765 .764 .760 .755<br>.001 .023 .615 .831 .919 .900 .814 .785 .773 .772 .768 .763<br>.002 .023 .610 .827 .920 .906 .820 .792 .780 .779 .775 .770<br>.002 .023 .606 .824 .920 .911 .828 .800 .788 .787 .783 .778<br>.002 .023 .603 .821 .919 .917 .837 .810 .798 .797 .793 .789<br>.002 .023 .599 .816 .917 .924 .851 .826 .813 .812 .809 .804<br>.002 .023 .592 .807 .910 .930 .868 .846 .834 .832 .829 .825<br>.002 .022 .579 .791 .896 .932 .885 .868 .856 .855 .853 .849<br>.002 .021 .562 .768 .873 .926 .897 .889 .878 .877 .876 .873<br>.002 .021 .544 .744 .849 .914 .904 .906 .895 .894 .895 .893<br>.002 .020 .528 .722 .826 .900 .905 .918 .909 .908 .910 .909<br>.002 .020 .515 .704 .807 .886 .902 .925 .918 .918 .922 .921<br>.003 .019 .505 .690 .792 .873 .897 .929 .924 .925 .929 .929<br>.003 .019 .498 .680 .781 .863 .891 .931 .926 .928 .933 .933<br>.003 .019 .492 .672 .772 .854 .885 .930 .927 .929 .935 .935<br>.003 .019 .488 .666 .766 .848 .880 .929 .926 .929 .935 .935<br>.003 .019 .485 .662 .761 .843 .877 .927 .925 .927 .934 .935<br><!-- End of picture text -->



<!-- Start of picture text -->
.010 -.062 .292 .337 .346 .333 .294 .287 .280 .284 .273 .274<br>.009 .010 .625 .829 .913 .907 .831 .807 .798 .796 .790 .787<br>.008 .011 .611 .821 .916 .924 .858 .839 .830 .828 .822 .820<br>.008 .012 .602 .812 .912 .931 .874 .858 .850 .848 .843 .841<br>.008 .012 .595 .804 .906 .934 .884 .873 .864 .862 .858 .857<br>.008 .012 .589 .796 .899 .935 .892 .883 .875 .873 .870 .868<br>.008 .013 .583 .789 .893 .934 .897 .892 .884 .882 .879 .878<br>.008 .013 .577 .782 .886 .932 .901 .898 .891 .889 .886 .885<br>.008 .013 .572 .775 .880 .930 .904 .904 .896 .895 .892 .892<br>.008 .014 .568 .769 .875 .928 .906 .908 .901 .899 .898 .897<br>.008 .014 .564 .764 .870 .926 .908 .912 .905 .903 .902 .901<br>.007 .014 .560 .759 .865 .924 .909 .915 .908 .907 .906 .905<br>.007 .014 .557 .755 .860 .922 .910 .918 .911 .910 .909 .909<br>.007 .014 .554 .750 .856 .919 .910 .920 .914 .912 .912 .911<br>.007 .014 .551 .747 .852 .917 .911 .922 .916 .914 .914 .914<br>.007 .014 .548 .743 .848 .915 .911 .923 .918 .916 .916 .916<br>.007 .014 .546 .740 .845 .913 .911 .925 .919 .918 .918 .918<br>.007 .015 .544 .737 .842 .911 .911 .926 .921 .920 .920 .920<br>.007 .015 .542 .734 .839 .909 .911 .927 .922 .921 .921 .922<br>.008 .015 .540 .731 .836 .907 .910 .928 .923 .922 .923 .923<br>.007 .015 .538 .729 .834 .906 .910 .929 .924 .923 .924 .924<br>.007 .015 .536 .726 .831 .904 .910 .930 .925 .924 .925 .926<br>.007 .015 .534 .724 .829 .902 .910 .931 .926 .925 .926 .927<br>.007 .015 .533 .722 .827 .901 .909 .931 .927 .926 .927 .928<br>.007 .015 .531 .720 .825 .900 .909 .932 .928 .927 .928 .929<br>.007 .015 .530 .718 .822 .898 .909 .932 .928 .927 .929 .929<br>.007 .015 .529 .716 .821 .897 .908 .933 .929 .928 .929 .930<br>.007 .015 .528 .715 .819 .896 .908 .933 .929 .929 .930 .931<br>.007 .015 .527 .713 .817 .894 .908 .934 .930 .929 .931 .932<br>.007 .015 .525 .712 .816 .893 .907 .934 .930 .929 .931 .932<br>.007 .015 .524 .710 .814 .892 .907 .934 .931 .930 .932 .933<br><!-- End of picture text -->

Figure 17: **Similarity of Induced Weights on Ill-Conditioned Data.** The best matching steps are highlighted in yellow. 

29 



<!-- Start of picture text -->
Similarity of Errors  (Transformers v.s. Bfgs)<br>Transformer Layer Index<br>1 2 3 4 5 6 7 8 9 10 11 12<br>1 .925 .857 .725 .619 .562 .542 .533 .530 .531 .529 .527 .528<br>2 .819 .774 .794 .727 .661 .638 .627 .625 .625 .623 .621 .621<br>3 .720 .660 .757 .782 .726 .700 .689 .688 .687 .686 .685 .684<br>4 .650 .586 .705 .801 .768 .743 .733 .732 .731 .729 .729 .728<br>5 .592 .544 .667 .804 .805 .782 .772 .771 .770 .769 .769 .768<br>6 .546 .528 .649 .800 .839 .819 .810 .809 .807 .806 .806 .805<br>7 .507 .534 .657 .803 .870 .857 .848 .847 .845 .844 .844 .844<br>8 .473 .548 .679 .813 .889 .890 .882 .882 .879 .878 .878 .878<br>9 .451 .565 .711 .833 .905 .917 .911 .910 .907 .907 .906 .907<br>10 .440 .582 .733 .854 .919 .938 .935 .934 .931 .930 .930 .931<br>11 .433 .593 .747 .871 .934 .954 .952 .952 .949 .948 .948 .949<br>12 .427 .593 .750 .878 .946 .968 .967 .967 .964 .964 .964 .965<br>13 .423 .593 .749 .878 .953 .977 .978 .978 .976 .975 .975 .976<br>14 .419 .588 .743 .874 .954 .982 .984 .985 .983 .982 .982 .983<br>15 .416 .585 .739 .869 .952 .984 .988 .989 .987 .987 .987 .987<br>16 .414 .583 .738 .868 .950 .985 .990 .991 .990 .989 .989 .990<br>17 .413 .582 .736 .866 .948 .984 .991 .992 .992 .991 .991 .992<br>18 .412 .580 .734 .864 .946 .983 .992 .993 .993 .992 .993 .994<br>19 .411 .578 .731 .861 .944 .982 .992 .993 .993 .993 .994 .994<br>20 .410 .577 .730 .860 .942 .980 .992 .993 .994 .994 .994 .995<br>21 .409 .577 .729 .858 .941 .979 .991 .993 .994 .994 .994 .995<br>22 .409 .576 .728 .858 .940 .979 .991 .992 .994 .994 .994 .995<br>23 .409 .576 .728 .857 .940 .978 .990 .992 .993 .994 .994 .995<br>24 .408 .575 .728 .857 .939 .978 .990 .992 .993 .993 .994 .995<br>25 .408 .575 .728 .857 .939 .978 .990 .991 .993 .993 .994 .995<br>26 .408 .575 .727 .856 .939 .977 .990 .991 .993 .993 .994 .995<br>27 .408 .575 .727 .856 .939 .977 .990 .991 .993 .993 .994 .995<br>28 .408 .575 .727 .856 .938 .977 .989 .991 .993 .993 .993 .994<br>29 .408 .575 .727 .856 .938 .977 .989 .991 .993 .993 .993 .994<br>30 .408 .575 .727 .856 .938 .977 .989 .991 .992 .993 .993 .994<br>31 .408 .575 .727 .856 .938 .976 .989 .991 .992 .993 .993 .994<br>32 .408 .575 .727 .855 .938 .976 .989 .990 .992 .993 .993 .994<br>33 .408 .575 .727 .855 .938 .976 .989 .990 .992 .992 .993 .994<br>34 .408 .575 .727 .855 .938 .976 .989 .990 .992 .992 .993 .994<br>35 .408 .574 .726 .855 .938 .976 .989 .990 .992 .992 .993 .994<br>36 .408 .574 .726 .855 .937 .976 .989 .990 .992 .992 .993 .994<br>37 .408 .574 .726 .855 .937 .976 .988 .990 .992 .992 .993 .994<br>38 .408 .574 .726 .855 .937 .976 .988 .990 .992 .992 .993 .994<br>39 .408 .574 .726 .855 .937 .976 .988 .990 .992 .992 .993 .994<br>40 .408 .574 .726 .855 .937 .976 .988 .990 .992 .992 .993 .994<br> BFGS Steps<br><!-- End of picture text -->



<!-- Start of picture text -->
Similarity of Errors  (Transformers v.s. L-Bfgs)<br>Transformer Layer Index<br>1 2 3 4 5 6 7 8 9 10 11 12<br>1 .926 .856 .723 .617 .561 .541 .532 .529 .530 .528 .526 .526<br>2 .699 .881 .873 .779 .712 .689 .678 .676 .675 .674 .672 .674<br>3 .607 .832 .906 .852 .784 .758 .747 .745 .744 .743 .741 .742<br>4 .527 .747 .898 .916 .857 .828 .817 .816 .814 .813 .813 .813<br>5 .494 .694 .869 .939 .898 .870 .859 .858 .856 .855 .855 .856<br>6 .484 .671 .848 .945 .923 .895 .884 .883 .881 .880 .880 .881<br>7 .481 .662 .832 .945 .941 .914 .904 .904 .901 .900 .900 .901<br>8 .472 .652 .816 .939 .955 .931 .921 .920 .918 .917 .917 .917<br>9 .458 .637 .801 .930 .965 .946 .936 .935 .933 .932 .932 .933<br>10 .447 .625 .787 .919 .970 .957 .948 .947 .945 .944 .944 .945<br>11 .439 .616 .779 .912 .973 .965 .957 .956 .953 .953 .952 .953<br>12 .436 .611 .773 .906 .973 .971 .964 .963 .960 .959 .959 .960<br>13 .433 .607 .769 .901 .972 .975 .969 .968 .965 .965 .965 .965<br>14 .432 .604 .765 .897 .970 .978 .973 .972 .970 .969 .969 .970<br>15 .430 .602 .761 .893 .969 .980 .976 .975 .972 .972 .972 .973<br>16 .429 .600 .758 .890 .967 .982 .978 .978 .975 .974 .974 .975<br>17 .427 .598 .755 .887 .965 .983 .980 .980 .977 .976 .976 .977<br>18 .424 .595 .752 .884 .963 .984 .982 .982 .979 .978 .978 .979<br>19 .423 .594 .750 .882 .962 .985 .984 .983 .981 .980 .980 .981<br>20 .422 .592 .749 .880 .960 .986 .985 .985 .982 .981 .981 .982<br>21 .421 .591 .747 .878 .959 .986 .986 .986 .983 .983 .983 .984<br>22 .420 .589 .746 .876 .958 .986 .987 .987 .985 .984 .984 .985<br>23 .418 .588 .744 .874 .956 .986 .988 .988 .986 .985 .986 .986<br>24 .417 .587 .742 .873 .954 .986 .989 .989 .987 .987 .987 .987<br>25 .417 .586 .741 .871 .953 .986 .989 .990 .988 .987 .987 .988<br>26 .416 .585 .740 .870 .952 .986 .990 .990 .988 .988 .988 .989<br>27 .415 .584 .739 .869 .951 .985 .990 .991 .989 .989 .989 .990<br>28 .414 .584 .738 .868 .951 .985 .991 .991 .990 .989 .989 .990<br>29 .414 .583 .737 .867 .950 .985 .991 .992 .990 .990 .990 .991<br>30 .413 .582 .736 .866 .949 .985 .991 .992 .991 .990 .990 .991<br>31 .413 .582 .736 .866 .948 .984 .991 .992 .991 .991 .991 .992<br>32 .412 .581 .735 .865 .948 .984 .991 .992 .991 .991 .991 .992<br>33 .412 .581 .735 .865 .947 .984 .992 .992 .992 .991 .992 .992<br>34 .412 .581 .734 .864 .947 .984 .992 .993 .992 .992 .992 .993<br>35 .412 .580 .734 .863 .946 .983 .992 .993 .992 .992 .992 .993<br>36 .411 .580 .733 .863 .946 .983 .992 .993 .992 .992 .992 .993<br>37 .411 .579 .733 .863 .945 .983 .992 .993 .993 .992 .993 .994<br>38 .411 .579 .732 .862 .945 .983 .992 .993 .993 .993 .993 .994<br>39 .411 .579 .732 .862 .944 .982 .992 .993 .993 .993 .993 .994<br>40 .411 .579 .732 .861 .944 .982 .992 .993 .993 .993 .993 .994<br> L-BFGS Steps<br><!-- End of picture text -->

Figure 18: **Similarity of Errors on Ill-Conditioned Data with Quasi-Newton Methods.** The best matching steps are highlighted in yellow. Transformer also matches BFGS linearly, from layers 4 to 11. L-BFGS still suffers due to its limited memory but still better than Gradient Descentbecause L-BFGS also attempts to approximate second-order information. 

30 

#### **A.3.2 Experiments with Noisy Linear Regression** 

We repeat the same experiments on noisy linear regression tasks with _y_ = **_w_**<sup>_⊤_</sup> **_x_** + _ε_ where _ε ∼ N_ (0 _, σ_<sup>2</sup> ) with noise level _σ_ = 0 _._ 1. As shown in Figure 19, Transformers still show superlinear convergence on noisy linear regression tasks. Since the predictor is **_w_** ˆ = � **_X_**<sup>_⊤_</sup> **_X_** + _λ_ **_I_** � _†_ **_X_** _⊤_ **_y_** for some _λ_ , the iterative newton’s method is applied to **_S_** = **_X_**<sup>_⊤_</sup> **_X_** + _λ_ **_I_** . Iterative Newton’s method still keeps the same superlinear convergence rates. As it’s also shown in Figure 19, Transformers and Iternative Newton’s rates match linearly, as in the noiseless linear regression tasks. 



<!-- Start of picture text -->
Transformer Errors v.s. # Layers<br>10 0<br>10 1<br>10 2<br># In-Context Examples = 05<br># In-Context Examples = 10<br>10 3 # In-Context Examples = 15<br># In-Context Examples = 20<br># In-Context Examples = 22<br># In-Context Examples = 25<br># In-Context Examples = 30<br># In-Context Examples = 35<br>10 4<br>1 2 3 4 5 6 7 8 9 10 11 12<br>Transformer Layer Index<br>Errors<br><!-- End of picture text -->



<!-- Start of picture text -->
Similarity of Errors  (Transformers v.s. Iterative Newton)<br>Transformer Layer Index Similarity of Errors  (Transformers v.s. Gradient Descent)<br>1 2 3 4 5 6 7 8 9 10 11 12 Transformer Layer Index<br>1 .957 .903 .786 .670 .615 .586 .589 .576 .590 .583 .586 .584 1 2 3 4 5 6 7 8 9 10 11 12<br>2 .922 .921 .829 .712 .651 .620 .622 .611 .623 .617 .619 .618 1 .962 .887 .788 .692 .638 .607 .610 .598 .611 .604 .607 .606<br>3 .878 .925 .869 .761 .691 .659 .660 .650 .661 .655 .658 .657<br>4 .829 .914 .899 .812 .736 .702 .701 .692 .701 .696 .699 .698 2 .918 .906 .839 .757 .699 .667 .668 .657 .669 .662 .665 .665<br>5 .778 .890 .915 .861 .781 .745 .742 .735 .743 .739 .741 .740<br>4 .848 .903 .885 .832 .774 .740 .739 .730 .740 .734 .737 .737<br>6 .729 .858 .917 .901 .826 .788 .784 .777 .784 .781 .783 .782<br>7 .685 .822 .905 .929 .867 .829 .824 .818 .824 .821 .823 .822 8 .761 .868 .905 .897 .850 .817 .813 .807 .814 .810 .812 .812<br>8 .646 .786 .884 .944 .904 .867 .861 .856 .860 .859 .860 .859<br>9 .611 .752 .859 .946 .933 .901 .893 .889 .893 .892 .893 .892 16 .677 .810 .889 .936 .912 .885 .880 .875 .880 .878 .879 .879<br>10 .582 .722 .833 .939 .953 .929 .921 .917 .920 .919 .920 .920<br>32 .607 .747 .850 .940 .952 .936 .929 .926 .929 .928 .929 .929<br>11 .558 .696 .809 .926 .964 .951 .942 .939 .941 .941 .941 .941<br>12 .540 .676 .789 .912 .967 .966 .958 .955 .957 .957 .957 .957 64 .558 .696 .807 .922 .967 .966 .961 .959 .960 .960 .961 .961<br>13 .527 .662 .773 .899 .967 .976 .969 .967 .968 .968 .968 .969<br>14 .518 .652 .763 .889 .964 .982 .977 .976 .976 .976 .976 .977 128 .529 .664 .775 .900 .967 .981 .978 .977 .978 .978 .978 .978<br>15 .512 .645 .755 .881 .960 .985 .983 .982 .982 .982 .982 .983<br>16 .507 .640 .749 .875 .956 .986 .986 .986 .986 .987 .986 .987 256 .513 .646 .756 .882 .960 .986 .987 .986 .987 .987 .987 .988<br>17 .504 .636 .744 .870 .952 .985 .989 .989 .989 .990 .989 .990<br>512 .505 .637 .745 .871 .953 .985 .990 .990 .991 .991 .991 .992<br>18 .502 .633 .741 .866 .949 .984 .990 .990 .991 .991 .991 .992<br>19 .501 .631 .738 .863 .946 .983 .990 .991 .992 .992 .992 .993 1024 .501 .631 .739 .864 .947 .983 .991 .992 .992 .993 .993 .993<br>20 .499 .629 .736 .861 .944 .981 .990 .991 .992 .993 .992 .993<br>21 .498 .628 .735 .859 .942 .980 .990 .991 .992 .993 .992 .993 2048 .498 .628 .735 .860 .943 .980 .990 .991 .992 .993 .993 .993<br>22 .498 .627 .734 .858 .941 .979 .989 .991 .991 .992 .992 .993<br>23 .497 .626 .733 .857 .940 .978 .989 .990 .991 .992 .992 .993 4096 .497 .626 .733 .857 .940 .978 .989 .990 .991 .992 .992 .993<br> iterative newton Steps<br> gradient descent (log scale) Steps<br><!-- End of picture text -->

Figure 19: Experiment results on **Noisy Linear Regression** . **(Top)** Transformers have superlinear convergence rate. **(Bottom)** Transformers match Iterative Newton’s rate and are exponentially faster than Gradient Descent. 

31 

#### **A.3.3 Experiments with a Non-Linear Function Class (2-Layer MLP)** 

To extend our experiments to non-linear cases, we adopt the same 2-layer ReLU neural network studied by Garg et al. [2022]: see Fig. 5(c) in their paper. For any prompt ( **_x_** 1 _, y_ 1 _, · · · ,_ **_x_** _t, yt_ ), instead of generating labels _yk_ = **_w_**<sup>_⋆⊤_</sup> **_x_** as mainly studied in the paper, we study a 2-layer neural network function class parameterized by **_W_** _∈_ R<sup>_d_hidden</sup><sup>_×d_</sup> , **_v_** _∈_ R<sup>_d_hidden</sup> , **_a_** _∈_ R<sup>_d_hidden</sup> , and _b ∈_ R, so that 



Then we repeat the same probing experiments as in the main paper. As shown in Figure 20, even on 2-layer neural network tasks with ReLU activation, Transformer shows superlinear convergence rates. Transformer shows an exponentially faster convergence rate than Gradient Descent’s, because Gradient Descent’s steps are shown in log scale and the trend is linear – similar to Figure 9 in the main paper. 



<!-- Start of picture text -->
Transformer Errors v.s. # Layers Similarity of Errors  (Transformers v.s. Gradient Descent)<br>2-Layer MLP with ReLU activation 1 2 3 4 Transformer Layer Index5 6 7 8 9 10 11 12<br>10 0 1 .850 .851 .850 .858 .675 .662 .661 .619 .617 .620 .627 .612<br>2 .860 .861 .860 .867 .684 .671 .669 .626 .624 .627 .633 .619<br>4 .881 .883 .882 .890 .706 .691 .688 .647 .644 .646 .654 .638<br>8 .913 .916 .916 .922 .750 .729 .723 .684 .682 .682 .690 .673<br>16 .944 .949 .949 .952 .804 .777 .767 .732 .725 .722 .731 .714<br>32 .941 .947 .947 .955 .870 .838 .826 .796 .788 .783 .790 .777<br>64 .892 .897 .898 .915 .920 .901 .893 .864 .856 .852 .856 .845<br>128 .801 .805 .805 .834 .913 .929 .934 .917 .916 .914 .915 .910<br># In-Context Examples = 25<br># In-Context Examples = 40<br># In-Context Examples = 80 256 .733 .734 .734 .764 .873 .919 .936 .938 .943 .942 .945 .948<br># In-Context Examples = 100<br>10 1<br>1 2 3 4 5 6 7 8 9 10 11 12 512 .699 .701 .702 .738 .856 .905 .925 .935 .947 .947 .949 .955<br>Transformer Layer Index<br>Errors<br> gradient descent (log scale) Steps<br><!-- End of picture text -->

Figure 20: Empirical Results on 2-Layer Neural Network Regression with ReLU activation function. Transformers have superlinear convergence rates and match Gradient Descent’s convergence rate exponentially 

It would be interesting to ablate the activation function used in Equation (19). We further consider the case when it’s using the Tanh activation instead of ReLU, i.e. 



Repeating the same experiments as before, as shown in Figure 21, we find that Transformers use the entire first 5 layers to pre-process and then only in the next few layers show exponentially faster convergence rate compared to Gradient Descent. We further note that in both Figure 20 and Figure 21, the cosine similarities between Transformers and Gradient Descent are significantly lower than the experiments with linear regression tasks. This might due to the over-parameterization of the function class and Transformers and Gradient Descent may arrive at different optima. 

32 



<!-- Start of picture text -->
Transformer Errors v.s. # Layers<br>Similarity of Errors  (Transformers v.s. Gradient Descent)<br>10 1 2-Layer MLP with Tanh activation 1 2 3 4 Transformer Layer Index5 6 7 8 9 10 11 12<br>1 .980 .980 .980 .980 .978 .739 .675 .666 .663 .667 .659 .662<br>1 .980 .980 .980 .980 .978 .739 .675 .666 .663 .667 .659 .662<br>6 × 10 2 3 .982 .982 .982 .982 .981 .744 .680 .671 .669 .672 .665 .667<br>5 .981 .981 .981 .981 .979 .745 .679 .670 .668 .672 .665 .667<br>10 .981 .981 .981 .981 .979 .750 .683 .675 .672 .675 .668 .669<br>4 × 10 2<br>18 .980 .980 .980 .980 .977 .761 .698 .688 .684 .687 .680 .681<br>34 .973 .973 .973 .973 .970 .772 .712 .703 .699 .703 .696 .697<br>3 × 10 2<br>61 .962 .962 .962 .962 .959 .791 .736 .728 .722 .725 .718 .718<br>110 .943 .943 .943 .943 .940 .821 .771 .762 .756 .758 .751 .750<br>2 × 10 2 198 .889 .889 .889 .888 .886 .859 .831 .824 .818 .820 .814 .812<br>357 .807 .807 .808 .808 .807 .877 .876 .873 .869 .870 .864 .863<br>642 .717 .717 .717 .717 .716 .851 .888 .894 .892 .893 .890 .889<br># In-Context Examples = 25<br># In-Context Examples = 40 1156 .662 .661 .661 .662 .661 .824 .880 .894 .895 .898 .898 .898<br># In-Context Examples = 80<br>10 2 # In-Context Examples = 100 2082 .618 .618 .618 .618 .617 .795 .863 .883 .885 .888 .888 .887<br>1 2 3 4 5 6 7 8 9 10 11 12<br>Transformer Layer Index 3748 .616 .616 .616 .616 .615 .792 .853 .872 .876 .880 .879 .879<br>Errors<br> gradient descent (log scale) Steps<br><!-- End of picture text -->

Figure 21: Empirical Results on 2-Layer Neural Network Regression with Tanh activation function. Transformers have superlinear convergence rates and match Gradient Descent’s convergence rate exponentially 

It would be interesting for future research to explore further this function class of 2-layer MLP to understand fully how Transformer solve the regression problem in-context and whether it achieves a different optimum compared to alternative algorithms such as (Stochastic) Gradient Descent. 

33 

### **A.4 Varying Transformer Architecture** 

#### **A.4.1 Experiments on Transformers of Fewer Heads** 

In this section, we present experimental results from an alternative model configurations than the main text. We show in the main text that Transformers learn second-order optimization methods in-context where the experiments are using a GPT-2 model with 12 layers and 8 heads per layer. In this section, we present experiments with a GPT-2 model with 12 layers but only 1 head per layer. 



<!-- Start of picture text -->
Similarity of Errors  (Transformers v.s. Iterative Newton)<br>Trasformer Layer Index Similarity of Errors  (Transformers v.s. Gradient Descent)<br>1 2 3 4 5 6 7 8 9 10 11 12 Trasformer Layer Index<br>1 .920 .920 .911 .909 .861 .785 .707 .671 .647 .631 .626 .619 1 2 3 4 5 6 7 8 9 10 11 12<br>1 .954 .955 .915 .885 .840 .757 .685 .655 .630 .615 .609 .604<br>2 .876 .876 .892 .912 .879 .823 .749 .709 .685 .667 .663 .655<br>50 .584 .585 .645 .703 .764 .870 .923 .943 .945 .943 .944 .939<br>3 .829 .829 .864 .901 .887 .859 .791 .750 .726 .706 .702 .694 100 .552 .552 .610 .668 .729 .841 .911 .945 .955 .964 .966 .962<br>4 .780 .780 .829 .877 .884 .887 .832 .792 .768 .746 .743 .735 150 .539 .539 .596 .653 .713 .826 .902 .941 .956 .970 .973 .971<br>5 .733 .733 .791 .845 .872 .906 .867 .832 .810 .787 .784 .776 200 .532 .532 .588 .645 .705 .819 .897 .938 .955 .973 .977 .975<br>6 .690 .690 .753 .811 .853 .913 .896 .869 .849 .825 .823 .816 250 .528 .528 .583 .640 .700 .813 .892 .936 .954 .974 .979 .978<br>300 .525 .525 .581 .637 .697 .810 .889 .934 .953 .975 .980 .979<br>7 .654 .654 .719 .777 .829 .910 .916 .900 .884 .861 .860 .852<br>350 .522 .522 .578 .635 .694 .807 .887 .932 .952 .975 .980 .980<br>8 .624 .624 .688 .746 .805 .900 .927 .924 .912 .894 .893 .885 400 .520 .520 .576 .632 .692 .804 .885 .931 .951 .976 .981 .982<br>9 .598 .598 .661 .719 .780 .885 .930 .939 .934 .920 .920 .913 450 .519 .520 .575 .631 .691 .803 .884 .930 .950 .976 .981 .983<br>10 .576 .576 .637 .695 .757 .867 .926 .947 .947 .941 .942 .935 500 .519 .519 .574 .630 .689 .801 .882 .928 .950 .976 .981 .983<br>11 .559 .559 .619 .676 .738 .851 .918 .948 .955 .956 .957 .951 550 .518 .518 .573 .629 .688 .801 .881 .928 .949 .976 .981 .983<br>600 .517 .517 .572 .628 .687 .799 .880 .927 .948 .976 .982 .984<br>12 .546 .546 .605 .662 .723 .837 .910 .947 .958 .966 .968 .963 650 .516 .516 .572 .628 .687 .799 .880 .926 .948 .976 .982 .985<br>13 .537 .537 .595 .651 .712 .826 .903 .944 .959 .973 .976 .972 700 .516 .516 .571 .627 .686 .798 .879 .926 .948 .976 .981 .985<br>14 .530 .530 .587 .644 .704 .817 .896 .940 .958 .977 .981 .979 750 .516 .516 .570 .626 .686 .797 .878 .925 .947 .976 .981 .985<br>15 .525 .525 .582 .638 .698 .810 .890 .936 .957 .980 .984 .984 800 .516 .516 .571 .626 .685 .797 .878 .925 .947 .976 .982 .985<br>850 .515 .515 .570 .626 .685 .796 .877 .924 .947 .976 .982 .985<br>16 .522 .522 .578 .634 .693 .806 .886 .933 .955 .981 .986 .987<br>900 .514 .514 .569 .625 .684 .795 .876 .924 .946 .976 .982 .985<br>17 .519 .519 .576 .631 .690 .802 .882 .930 .953 .981 .987 .989 950 .513 .514 .568 .625 .684 .795 .876 .923 .946 .976 .981 .986<br>18 .518 .517 .573 .629 .688 .799 .880 .928 .951 .981 .987 .991 1000 .514 .514 .569 .624 .683 .795 .876 .923 .946 .976 .982 .986<br>19 .516 .516 .572 .627 .686 .798 .878 .926 .949 .980 .986 .992 1050 .513 .513 .568 .624 .683 .795 .875 .923 .946 .976 .982 .986<br>20 .515 .515 .571 .626 .685 .796 .876 .924 .948 .979 .986 .992 1100 .513 .513 .568 .624 .683 .794 .875 .922 .945 .975 .981 .986<br>1150 .513 .513 .568 .624 .683 .794 .875 .922 .945 .975 .981 .986<br>21 .514 .514 .570 .625 .684 .795 .874 .923 .946 .978 .985 .992<br>1200 .513 .513 .567 .623 .682 .794 .874 .922 .945 .975 .981 .986<br>22 .513 .513 .569 .624 .683 .793 .872 .920 .945 .976 .983 .991 1250 .513 .513 .567 .623 .682 .794 .875 .922 .945 .975 .981 .986<br>23 .510 .510 .565 .620 .679 .787 .865 .914 .938 .969 .976 .984 1300 .513 .513 .568 .623 .682 .793 .874 .921 .944 .975 .981 .986<br>Similarity of Errors on an alternative Transformers Configuration.<br>steps are highlighted in yellow.<br>Similarity of Induced Weight w (Transformers v.s. Iterative Newton)<br>Trasformer Layer Index Similarity of Induced Weight w (Transformers v.s. Gradient Descent)<br>1 2 3 4 5 6 7 8 9 10 11 12 Trasformer Layer Index<br>1 .000 -.003 .581 .756 .740 .757 .734 .717 .713 .710 .711 .712 1 2 3 4 5 6 7 8 9 10 11 12<br>1 .001 .119 .522 .684 .689 .702 .688 .680 .673 .669 .668 .669<br>2 .001 -.003 .590 .767 .770 .802 .782 .766 .762 .759 .760 .761<br>50 .003 .020 .517 .675 .758 .885 .936 .951 .955 .961 .961 .963<br>3 .002 -.002 .588 .764 .789 .840 .827 .811 .807 .805 .806 .807 100 .002 .019 .508 .662 .746 .876 .933 .952 .959 .968 .968 .970<br>4 .002 -.002 .580 .751 .797 .868 .864 .850 .847 .844 .845 .846 150 .002 .019 .503 .657 .741 .871 .930 .952 .959 .970 .971 .973<br>5 .003 -.002 .567 .734 .796 .885 .891 .881 .878 .876 .877 .877 200 .003 .019 .502 .655 .739 .869 .928 .951 .959 .971 .972 .974<br>6 .003 -.002 .553 .716 .789 .893 .911 .905 .903 .901 .902 .902 250 .003 .019 .501 .653 .737 .867 .927 .950 .959 .971 .972 .975<br>300 .002 .019 .500 .652 .736 .866 .927 .950 .958 .971 .972 .975<br>7 .003 -.001 .541 .700 .780 .894 .923 .922 .921 .921 .922 .922<br>350 .003 .019 .499 .652 .735 .865 .926 .950 .958 .972 .973 .976<br>8 .003 -.001 .531 .686 .770 .891 .930 .934 .935 .936 .937 .937 400 .003 .019 .499 .651 .735 .865 .925 .949 .958 .972 .973 .976<br>9 .003 -.001 .522 .675 .761 .886 .932 .941 .944 .947 .948 .948 450 .003 .019 .498 .651 .734 .864 .925 .949 .958 .972 .973 .976<br>10 .002 -.000 .515 .666 .753 .880 .932 .945 .949 .954 .955 .955 500 .003 .019 .498 .650 .734 .864 .925 .949 .958 .972 .973 .976<br>11 .003 -.000 .510 .660 .746 .875 .930 .946 .952 .959 .961 .961 550 .003 .019 .498 .650 .734 .863 .924 .948 .957 .972 .973 .976<br>600 .003 .019 .498 .650 .733 .863 .924 .948 .957 .972 .973 .976<br>12 .003 .000 .506 .655 .741 .870 .928 .947 .954 .963 .964 .965 650 .002 .019 .498 .650 .733 .863 .924 .948 .957 .972 .973 .977<br>13 .003 .000 .503 .652 .738 .867 .926 .946 .954 .965 .967 .968 700 .003 .019 .497 .649 .733 .863 .924 .948 .957 .972 .973 .977<br>14 .003 .001 .501 .650 .736 .864 .924 .945 .954 .966 .968 .969 750 .003 .019 .497 .649 .733 .862 .923 .948 .957 .972 .973 .977<br>15 .003 .001 .500 .648 .734 .862 .922 .944 .954 .967 .969 .971 800 .003 .019 .497 .649 .733 .862 .923 .948 .957 .972 .973 .977<br>850 .003 .019 .497 .649 .733 .862 .923 .948 .957 .972 .973 .977<br>16 .003 .001 .499 .647 .732 .861 .921 .943 .953 .968 .970 .972<br>900 .003 .019 .497 .649 .732 .862 .923 .948 .957 .972 .973 .977<br>17 .003 .001 .498 .646 .731 .860 .920 .942 .953 .968 .970 .972 950 .003 .019 .497 .649 .732 .862 .923 .947 .957 .972 .973 .977<br>18 .003 .001 .498 .645 .731 .859 .919 .942 .952 .968 .970 .973 1000 .003 .019 .497 .649 .732 .862 .923 .947 .957 .972 .973 .977<br>19 .003 .001 .498 .645 .730 .858 .919 .941 .952 .968 .970 .973 1050 .003 .019 .497 .649 .732 .862 .922 .947 .957 .972 .973 .977<br>20 .003 .001 .497 .644 .730 .858 .918 .941 .951 .967 .970 .973 1100 .003 .019 .497 .649 .732 .862 .923 .947 .957 .972 .973 .977<br>1150 .003 .019 .497 .649 .732 .862 .923 .947 .956 .972 .973 .977<br>21 .003 .001 .497 .644 .729 .858 .918 .941 .951 .967 .970 .973<br>1200 .003 .019 .496 .648 .732 .861 .922 .947 .956 .972 .973 .977<br>22 .003 .001 .497 .644 .729 .857 .918 .940 .951 .967 .970 .973 1250 .003 .019 .497 .648 .732 .861 .922 .947 .956 .972 .973 .977<br>23 .003 .001 .497 .644 .729 .857 .918 .940 .951 .967 .969 .973 1300 .003 .019 .496 .648 .732 .861 .922 .947 .956 .972 .973 .977<br> Iterative newton Steps  Gradient descent Steps<br> Iterative newton Steps  Gradient descent Steps<br><!-- End of picture text -->



<!-- Start of picture text -->
.954 .955 .915 .885 .840 .757 .685 .655 .630 .615 .609 .604<br>.584 .585 .645 .703 .764 .870 .923 .943 .945 .943 .944 .939<br>.552 .552 .610 .668 .729 .841 .911 .945 .955 .964 .966 .962<br>.539 .539 .596 .653 .713 .826 .902 .941 .956 .970 .973 .971<br>.532 .532 .588 .645 .705 .819 .897 .938 .955 .973 .977 .975<br>.528 .528 .583 .640 .700 .813 .892 .936 .954 .974 .979 .978<br>.525 .525 .581 .637 .697 .810 .889 .934 .953 .975 .980 .979<br>.522 .522 .578 .635 .694 .807 .887 .932 .952 .975 .980 .980<br>.520 .520 .576 .632 .692 .804 .885 .931 .951 .976 .981 .982<br>.519 .520 .575 .631 .691 .803 .884 .930 .950 .976 .981 .983<br>.519 .519 .574 .630 .689 .801 .882 .928 .950 .976 .981 .983<br>.518 .518 .573 .629 .688 .801 .881 .928 .949 .976 .981 .983<br>.517 .517 .572 .628 .687 .799 .880 .927 .948 .976 .982 .984<br>.516 .516 .572 .628 .687 .799 .880 .926 .948 .976 .982 .985<br>.516 .516 .571 .627 .686 .798 .879 .926 .948 .976 .981 .985<br>.516 .516 .570 .626 .686 .797 .878 .925 .947 .976 .981 .985<br>.516 .516 .571 .626 .685 .797 .878 .925 .947 .976 .982 .985<br>.515 .515 .570 .626 .685 .796 .877 .924 .947 .976 .982 .985<br>.514 .514 .569 .625 .684 .795 .876 .924 .946 .976 .982 .985<br>.513 .514 .568 .625 .684 .795 .876 .923 .946 .976 .981 .986<br>.514 .514 .569 .624 .683 .795 .876 .923 .946 .976 .982 .986<br>.513 .513 .568 .624 .683 .795 .875 .923 .946 .976 .982 .986<br>.513 .513 .568 .624 .683 .794 .875 .922 .945 .975 .981 .986<br>.513 .513 .568 .624 .683 .794 .875 .922 .945 .975 .981 .986<br>.513 .513 .567 .623 .682 .794 .874 .922 .945 .975 .981 .986<br>.513 .513 .567 .623 .682 .794 .875 .922 .945 .975 .981 .986<br>.513 .513 .568 .623 .682 .793 .874 .921 .944 .975 .981 .986<br><!-- End of picture text -->

Figure 22: **Similarity of Errors on an alternative Transformers Configuration.** The best matching steps are highlighted in yellow. 



<!-- Start of picture text -->
.000 -.003 .581 .756 .740 .757 .734 .717 .713 .710 .711 .712<br>.001 -.003 .590 .767 .770 .802 .782 .766 .762 .759 .760 .761<br>.002 -.002 .588 .764 .789 .840 .827 .811 .807 .805 .806 .807<br>.002 -.002 .580 .751 .797 .868 .864 .850 .847 .844 .845 .846<br>.003 -.002 .567 .734 .796 .885 .891 .881 .878 .876 .877 .877<br>.003 -.002 .553 .716 .789 .893 .911 .905 .903 .901 .902 .902<br>.003 -.001 .541 .700 .780 .894 .923 .922 .921 .921 .922 .922<br>.003 -.001 .531 .686 .770 .891 .930 .934 .935 .936 .937 .937<br>.003 -.001 .522 .675 .761 .886 .932 .941 .944 .947 .948 .948<br>.002 -.000 .515 .666 .753 .880 .932 .945 .949 .954 .955 .955<br>.003 -.000 .510 .660 .746 .875 .930 .946 .952 .959 .961 .961<br>.003 .000 .506 .655 .741 .870 .928 .947 .954 .963 .964 .965<br>.003 .000 .503 .652 .738 .867 .926 .946 .954 .965 .967 .968<br>.003 .001 .501 .650 .736 .864 .924 .945 .954 .966 .968 .969<br>.003 .001 .500 .648 .734 .862 .922 .944 .954 .967 .969 .971<br>.003 .001 .499 .647 .732 .861 .921 .943 .953 .968 .970 .972<br>.003 .001 .498 .646 .731 .860 .920 .942 .953 .968 .970 .972<br>.003 .001 .498 .645 .731 .859 .919 .942 .952 .968 .970 .973<br>.003 .001 .498 .645 .730 .858 .919 .941 .952 .968 .970 .973<br>.003 .001 .497 .644 .730 .858 .918 .941 .951 .967 .970 .973<br>.003 .001 .497 .644 .729 .858 .918 .941 .951 .967 .970 .973<br>.003 .001 .497 .644 .729 .857 .918 .940 .951 .967 .970 .973<br>.003 .001 .497 .644 .729 .857 .918 .940 .951 .967 .969 .973<br><!-- End of picture text -->



<!-- Start of picture text -->
.001 .119 .522 .684 .689 .702 .688 .680 .673 .669 .668 .669<br>.003 .020 .517 .675 .758 .885 .936 .951 .955 .961 .961 .963<br>.002 .019 .508 .662 .746 .876 .933 .952 .959 .968 .968 .970<br>.002 .019 .503 .657 .741 .871 .930 .952 .959 .970 .971 .973<br>.003 .019 .502 .655 .739 .869 .928 .951 .959 .971 .972 .974<br>.003 .019 .501 .653 .737 .867 .927 .950 .959 .971 .972 .975<br>.002 .019 .500 .652 .736 .866 .927 .950 .958 .971 .972 .975<br>.003 .019 .499 .652 .735 .865 .926 .950 .958 .972 .973 .976<br>.003 .019 .499 .651 .735 .865 .925 .949 .958 .972 .973 .976<br>.003 .019 .498 .651 .734 .864 .925 .949 .958 .972 .973 .976<br>.003 .019 .498 .650 .734 .864 .925 .949 .958 .972 .973 .976<br>.003 .019 .498 .650 .734 .863 .924 .948 .957 .972 .973 .976<br>.003 .019 .498 .650 .733 .863 .924 .948 .957 .972 .973 .976<br>.002 .019 .498 .650 .733 .863 .924 .948 .957 .972 .973 .977<br>.003 .019 .497 .649 .733 .863 .924 .948 .957 .972 .973 .977<br>.003 .019 .497 .649 .733 .862 .923 .948 .957 .972 .973 .977<br>.003 .019 .497 .649 .733 .862 .923 .948 .957 .972 .973 .977<br>.003 .019 .497 .649 .733 .862 .923 .948 .957 .972 .973 .977<br>.003 .019 .497 .649 .732 .862 .923 .948 .957 .972 .973 .977<br>.003 .019 .497 .649 .732 .862 .923 .947 .957 .972 .973 .977<br>.003 .019 .497 .649 .732 .862 .923 .947 .957 .972 .973 .977<br>.003 .019 .497 .649 .732 .862 .922 .947 .957 .972 .973 .977<br>.003 .019 .497 .649 .732 .862 .923 .947 .957 .972 .973 .977<br>.003 .019 .497 .649 .732 .862 .923 .947 .956 .972 .973 .977<br>.003 .019 .496 .648 .732 .861 .922 .947 .956 .972 .973 .977<br>.003 .019 .497 .648 .732 .861 .922 .947 .956 .972 .973 .977<br>.003 .019 .496 .648 .732 .861 .922 .947 .956 .972 .973 .977<br><!-- End of picture text -->

Figure 23: **Similarity of Induced Weights on an alternative Transformers Configuration.** The best matching steps are highlighted in yellow. 

We conclude that our experimental results are not restricted to a specific model configurations, 

34 

smaller models such as GPT-2 with 12 layers and 1 head each layer also suffice in implementing the Iterative Newton’s method, and more similar than gradient descents, in terms of rate of convergence. 

#### **A.4.2 Experiments on Transformers with More Layers** 

In this section, we investigate whether deeper models would behave similarly or differently. We work on Transformers with 24 layers and 8 heads each. 



<!-- Start of picture text -->
Similarity of Errors  (Transformers v.s. Iterative Newton)<br>Trasformer Layer Index Similarity of Errors  (Transformers v.s. Gradient Descent)<br>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 Trasformer Layer Index<br>1 .918 .918 .918 .918 .915 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24<br>1 .953 .953 .952 .953 .886 .840 .771 .744 .702 .681 .644<br>2 .872 .873 .873 .873 .922 .920 50 .577 .577 .577 .577 .712 .776 .833 .866 .899 .920 .948<br>3 .913 .931 100 .796 .833 .874 .899 .945 .961 .966<br>4 .926 .919 150 .961 .967<br>5 .924 .928 200 .960 .966<br>6 .915 .930 .921 250<br>300<br>7 .920 .928 .936<br>350<br>8 .924 .939 400<br>9 .933 .952 450 .976<br>10 .955 500 .976<br>11 .953 .964 .969 550 .976<br>600<br>12 .966 .972 650 .982<br>13 .965 .972 700 .982<br>14 750 .982<br>15 .981 800<br>16 .981 850<br>900<br>17 .981 .987 950<br>18 .987 1000<br>19 .987 .988 .992 1050<br>20 .988 .993 .993 .992 .992 .991 .991 .992 .993 1100<br>21 .988 .993 .993 .992 .992 .992 .992 .992 .993 1150<br>1200 .984<br>22 .992 .992 .992 .991 .991 .992 .993 1250 .984 .987 .987 .986 .986 .986 .986 .986 .987<br>23 1300 .984 .987 .987 .986 .986 .986 .986 .986 .987<br>Similarity of Errors on a 24-layer Transformers Configuration.<br>are highlighted in yellow.<br>Transformer Errors v.s. # Layers<br>10 0<br>10 1<br>10 2<br># In-Context Examples = 05<br># In-Context Examples = 10<br>10 3 # In-Context Examples = 15<br># In-Context Examples = 20<br># In-Context Examples = 22<br># In-Context Examples = 25<br># In-Context Examples = 30<br># In-Context Examples = 35<br>10 4<br>1 3 5 7 9 11 13 15 17 19 21 23<br>Transformer Layer Index<br> Iterative newton Steps  Gradient descent Steps<br>Errors<br><!-- End of picture text -->



<!-- Start of picture text -->
.953 .953 .952 .953 .886 .840 .771 .744 .702 .681 .644<br>.577 .577 .577 .577 .712 .776 .833 .866 .899 .920 .948<br>.796 .833 .874 .899 .945 .961 .966<br>.961 .967<br>.960 .966<br>.976<br>.976<br>.976<br>.982<br>.982<br>.982<br>.984<br>.984 .987 .987 .986 .986 .986 .986 .986 .987<br>.984 .987 .987 .986 .986 .986 .986 .986 .987<br><!-- End of picture text -->

Figure 24: **Similarity of Errors on a 24-layer Transformers Configuration.** The best matching steps are highlighted in yellow. 

Figure 25: Transformers with 24 layers also converge superlinearly, similar to Iterative Newton. 

35 

### **A.5 Heatmaps with Best-Matching Steps Help Compare Convergence Rates** 

In this section, we show the heatmaps with best-matching steps among _known algorithms._ 



<!-- Start of picture text -->
Iterative Newton Steps<br>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23<br>1 .974 .955 .923 .883 .838 .794 .754 .719 .691 .670<br>50 .697 .738 .782 .828 .873 .915 .948 .971 .984 .987 .985 .979<br>100 .830 .874 .913 .945 .967 .980 .987 .989<br>150 .982 .988 .989<br>200 .989 .989<br>250 .988 .990<br>300 .989 .990<br>350 .990<br>400 .989<br>450 .990<br>500 .990<br>550 .989 .990<br>600 .990<br>650 .990<br>700<br>750 .990<br>800 .990<br>850 .990<br>900<br>950<br>1000<br>1050<br>1100<br>1150<br>1200 .990 .990 .990 .989 .984<br>1250 .991 .990 .990 .989 .985<br>1300 .990 .990 .990 .989 .984<br>Gradient Descent Steps<br><!-- End of picture text -->



<!-- Start of picture text -->
Iterative Newton Steps<br>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23<br>1 1.000 .991<br>2 .991 1.000 .990<br>3 .990 1.000 .989<br>4 .989 1.000 .989<br>5 .989 1.000 .990<br>6 .990 1.000 .990<br>7 .990 1.000 .992<br>8 .992 1.000 .993<br>9 .993 1.000 .995<br>10 .995 1.000 .996<br>11 .996 1.000 .997<br>12 .997 1.000 .998<br>13 .998 1.000 .998<br>14 .998 1.000 .999<br>15 .999 1.000 .999<br>16 .999 1.000 1.000<br>17 1.000 1.000 1.000<br>18 1.000 1.000 1.000<br>19 1.000 1.000 1.000<br>20 1.000 1.000 1.000<br>21 1.000 1.000 1.000<br>22 1.000 1.000 .999<br>23 .999 1.000<br>Iterative Newton Steps<br><!-- End of picture text -->



<!-- Start of picture text -->
Iterative Newton Steps<br>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23<br>1 .985 .966 .934 .896 .858 .822<br>2 .925 .916 .906 .896 .885 .871<br>3 .817 .838<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12 .852<br>13 .857 .883 .903<br>14 .855 .883 .908 .927<br>15 .877 .906 .929 .946 .958<br>16 .928 .949 .964<br>17 .945 .964 .978 .988<br>18 .979 .989 .994<br>19 .979 .989 .995 .997<br>20 .994 .997 .998 .999<br>21 .997 .998 .999 .999 .999<br>22 .998 .999 .999 1.0001.000<br>23 .999 1.0001.0001.000<br>24 1.0001.0001.000<br>25 1.0001.000<br>BFGS Steps<br><!-- End of picture text -->

Figure 26: Best-Matching Steps on Similarity of Residuals Help Compare Convergence Rates. (a: top-left) When comparing Iterative Newton and Gradient Descent, there is an exponential trend – showing Iterative Newton converges exponentially faster than Gradient Descent. (b: top-right) When Iterative Newton is compared with itself in sub-figure, there is a linear trend – showing they have the same convergence rate. (c: bottom) When Iterative Newton is compared to BFGS in sub-figure, there a linear trend after there are enough steps for BFGS to approximate second-order information – showing Iterative Newton and BFGS share a similar convergence rate after sufficient BFGS steps. 

36 

### **A.6 Definitions for Evaluating Forgetting** 

We measure the phenomenon of model forgetting by reusing an in-context example within _{_ **_x_** _i, yi}_<sup>_n_</sup> _i_ =1<sup>as the test example</sup><sup>**_x_**test.In experiments of Figure 5, we fix</sup><sup>_n_= 20 and reuse</sup><sup>**_x_**test=</sup><sup>**_x_**</sup><sup>_i_.</sup> We denote the “Time Stamp Gap” as the distance the reused example index _i_ from the current time stamp _n_ = 20. We measure the forgetting of index _i_ as 



Note: the further away _i_ is from _n_ , the more possible algorithm _A_ forgets. 

37 

## **B Detailed Proofs for Section 5** 

In this section, we work on full attention layers with normalized ReLU activation _σ_ ( _·_ ) = _n_<sup><u>1</u>ReLU(</sup><sup>_·_)</sup> given _n_ examples. 

**Definition B.1.** A full attention layer with _M_ heads and ReLU activation is also denoted as Attn on any input sequence **_H_** = � **_h_** 1 _, · · · ,_ **_h_** _N_ � _∈_ R<sup>_D×N_</sup> , where _D_ is the dimension of hidden states and _N_ is the sequence length. In the vector form, 



_Remark_ B.2 _._ This is slightly different from the **causal** attention layer (see Definition 3.1) in that at each time stamp _t_ , the attention layer in Definition B.1 has full information of all hidden states _j ∈_ [ _n_ ], unlike causal attention layer which requires _j ∈_ [ _t_ ]. 

### **B.1 Helper Results** 

We begin by constructing a useful component for our proof, and state some existing constructions from Akyürek et al. [2022]. 

**Lemma B.3.** _Given hidden states {_ **_h_** 1 _, · · · ,_ **_h_** _n}, there exists query, key and value matrices_ **_Q_** _,_ **_K_** _,_ **_V_** _respectively such that one attention layer can compute_<sup>�</sup><sup>_n_</sup> _j_ =1<sup>**_h_**</sup><sup>_j._</sup> 







We apply one attention layer to these 1-padded hidden states and we have 



38 

**Proposition B.4** (Akyürek et al., 2022) **.** _Each of_ mov _,_ aff _,_ mul _,_ div _can be implemented by a single transformer layer. These four operations are mappings_ R<sup>_D×N_</sup> _→_ R<sup>_D×N_</sup> _, expressed as follows,_ 

mov( **_H_** ; _s, t, i, j, i_<sup>_′_</sup> _, j_<sup>_′_</sup> ) _: selects the entries of the s-th column of_ **_H_** _between rows i and j, and copies them into the t-th column (t ≥ s) of_ **_H_** _between rows i_<sup>_′_</sup> _and j_<sup>_′_</sup> _._ 

mul( **_H_** ; _a, b, c,_ ( _i, j_ ) _,_ ( _i_<sup>_′_</sup> _, j_<sup>_′_</sup> ) _,_ ( _i_<sup>_′′_</sup> _, j_<sup>_′′_</sup> )) _: in each column_ **_h_** _of_ **_H_** _, interprets the entries between i and j as an a × b matrix_ **_A_** 1 _, and the entries between i_<sup>_′_</sup> _and j_<sup>_′_</sup> _as a b × c matrix_ **_A_** 2 _, multiplies these matrices together, and stores the result between rows i_<sup>_′′_</sup> _and j_<sup>_′′_</sup> _, yielding a matrix in which each column has the form ⊤_ � **_h_** : _i′′−_ 1 _,_ **_A_** 1 **_A_** 2 _,_ **_h_** _j′′_ :� _. This allows the layer to implement inner products._ div( **_H_** ; ( _i, j_ ) _, i_<sup>_′_</sup> _,_ ( _i_<sup>_′′_</sup> _, j_<sup>_′′_</sup> )) _: in each column_ **_h_** _of_ **_H_** _, divides the entries between i and j by the absolute value of the entry at i_<sup>_′_</sup> _, and stores the result between rows i_<sup>_′′_</sup> _and j_<sup>_′′_</sup> _, yielding a matrix in which every column ⊤ has the form_ � **_h_** : _i′′−_ 1 _,_ **_h_** _i_ : _j/|_ **_h_** _i′|,_ **_h_** _j′′_ :� _._ aff( **_H_** ; ( _i, j_ ) _,_ ( _i_<sup>_′_</sup> _, j_<sup>_′_</sup> ) _,_ ( _i_<sup>_′′_</sup> _, j_<sup>_′′_</sup> ) _,_ **_W_** 1 _,_ **_W_** 2 _,_ **b** ) _: in each column_ **_h_** _of_ **_H_** _, applies an affine transformation to the entries between i and j and i_<sup>_′_</sup> _and j_<sup>_′_</sup> _, then stores the result between rows i_<sup>_′′_</sup> _and j_<sup>_′′_</sup> _, yielding a ⊤ matrix in which every column has the form_ � **_h_** : _i′′−_ 1 _,_ **_W_** 1 **_h_** _i_ : _j_ + **_W_** 2 **_h_** _i′_ : _j′_ + **b** _,_ **_h_** _j′′_ :� _. This allows the layer to implement subtraction by setting_ **_W_** 1 = **_I_** _and_ **_W_** 2 = _−_ **_I_** _._ 

### **B.2 Proof of Theorem 5.1** 

**Theorem 5.1.** _For any k, there exist Transformer weights such that on any set of in-context examples_ ˆ _{_ **_x_** _i, yi}_<sup>_n_</sup> _i_ =1<sup>_and test point_</sup><sup>**_x_**test</sup><sup>_, the Transformer predicts on_</sup><sup>**_x_**test</sup><sup>_using_</sup><sup>**_x_**</sup><sup>_⊤_</sup> test<sup>**_w_**ˆ</sup> _k_<sup>Newton</sup> _. Here_ **_w_** _k_<sup>Newton</sup> _are_ ˆ _the Iterative Newton updates given by_ **_w_** _k_<sup>Newton</sup> = **_M_** _k_ **_X_**<sup>_⊤_</sup> **_y_** _where_ **_M_** _j is updated as_ 



_for some α >_ 0 _and_ **_S_** = **_X_**<sup>_⊤_</sup> **_X_** _. The dimensionality of the hidden layers is O_ ( _d_ ) _, and the number of layers is k_ + 8 _. One transformer layer computes one Newton iteration. 3 initial transformer layers are needed for initializing_ **_M_** 0 _and 5 layers at the end are needed to read out predictions from the computed pseudo-inverse_ **_M_** _k._ 

#### _Proof._ We break the proof into parts. 

**Transformers Implement Initialization** **_T_**<sup>(0)</sup> = _α_ **_S_ .** Given input sequence **_H_** := _{_ **_x_** 1 _, · · · ,_ **_x_** _n}_ , with **_x_** _i ∈_ R<sup>_d_</sup> , we first apply the mov operations given by Proposition B.4 (similar to Akyürek et al. [2022], we show only non-zero rows when applying these operations): 



We call each column after mov as **_h_** _j_ . With an full attention layer, one can construct two heads **_I_** _d×d_ **_O_** _d×d_ with query and value matrices of the form **_Q_**<sup>_⊤_</sup> 1<sup>**_K_**1=</sup><sup>_−_</sup><sup>**_Q_**</sup><sup>_⊤_</sup> 2<sup>**_K_**2=</sup> such that for any � **_O_** _d×d_ **_O_** _d×d_ � _t ∈_ [ _n_ ], we have 



39 

**_I_** _d×d_ **_O_** _d×d_ Let all value matrices **_V_** _m_ = _nα_ for some _α ∈_ R. Combining the skip connections, � **_O_** _d×d_ **_O_** _d×d_ � we have 



Now we can use the aff operator to make subtractions and then 



We call this transformed hidden states as **_H_**<sup>(0)</sup> and denote **_T_**<sup>(0)</sup> = _α_ **_S_** : 



Notice that **_S_** is symmetric and thereafter **_T_**<sup>(0)</sup> is also symmetric. 

**Transformers implement Newton Iteration.** Let the input prompt be the same as Equation (28), 



We claim that the _ℓ_ ’s hidden states can be of the similar form 



We prove by induction that assuming our claim is true for _ℓ_ , we work on _ℓ_ + 1: 





where 



40 



Now we pass over an MLP layer with 



Now we denote the iteration 



We find that **_T_**<sup>(</sup><sup>_ℓ_+1)</sup><sup>_⊤_</sup> = **_T_**<sup>(</sup><sup>_ℓ_+1)</sup> since **_T_**<sup>(</sup><sup>_ℓ_)</sup> and **_S_** are both symmetric. It reduces to 



This is exactly the same as the Newton iteration. 

ˆ **Transformers can implement** **_w_** _ℓ_<sup>TF</sup> = **_T_**<sup>(</sup><sup>_ℓ_)</sup> **_X_**<sup>_⊤_</sup> **_y_ .** Going back to the empirical prompt format _{_ **_x_** 1 _, y_ 1 _, · · · ,_ **_x_** _n, yn}_ . We can let parameters be zero for positions of _y_ ’s and only rely on the skip **_T_**<sup>(</sup><sup>_ℓ_)</sup> **_x_ 0** _n j_ connection up to layer _ℓ_ , and the **_H_**<sup>(</sup><sup>_ℓ_)</sup> is then  **_x_** _j_ **0**  . We again apply operations from  0 _yj_  _j_ =1 Proposition B.4: 



Now we apply Lemma B.3 over all even columns in Equation (37) and we have 



where **_ξ_** denotes irrelevant quantities. Note that the resulting **_T_**<sup>(</sup><sup>_ℓ_)</sup> **_X_**<sup>_⊤_</sup> **_y_** is also the same as Iterative ˆ ˆ Newton’s predictor **_w_** _k_ = **_M_** _k_ **_X_**<sup>_⊤_</sup> **_y_** after _k_ iterations. We denote **_w_** _ℓ_<sup>TF</sup> = **_T_**<sup>(</sup><sup>_ℓ_)</sup> **_X_**<sup>_⊤_</sup> **_y_** . ˆ TF **Transformers can make predictions on** **_x_** _test_ **by** � **_w_** _ℓ ,_ **_x_** test� **.** 

41 

Now we can make predictions on text query **_x_** test: 



Finally, we can have an readout layer **_β_** ReadOut = _{_ **_u_** _, v}_ applied (see Definition 3.3) with **_u_** = � **0** 3 _d_ 1� _⊤_ and _v_ = 0 to extract the prediction � **_w_** ˆ _ℓ_ TF _,_ **_x_** test� at the last location, given by **_x_** test. This is exactly how Iterative Newton makes predictions. 

**To Perform** _k_ **steps of Newton’s iterations, Transformers need** _O_ ( _k_ ) **layers.** 

Let’s count the layers: 

- **Initialization** : mov needs _O_ (1) layer; gathering _α_ **_S_** needs _O_ (1) layer; and aff needs _O_ (1) layer. In total, Transformers need _O_ (1) layers for initialization. 

- **Newton Iteration** : each exact Newton’s iteration requires _O_ (1) layer. Implementing _k_ iterations requires _O_ ( _k_ ) layers. 

- **Implementing** **_w_** ˆ _ℓ_<sup>TF</sup> : We need one operation of mov and mul each, requiring _O_ (1) layer each. Apply Lemma B.3 for summation also requires _O_ (1) layer. 

- **Making prediction on test query** : We need one operation of mov and mul each, requiring _O_ (1) layer each. 

Hence, in total, Transformers can implement _k_ -step Iterative Newton and make predictions accordingly using _O_ ( _k_ ) layers. 

_Remark_ B.5 _._ We note that Giannou et al. [2023] used 13 layers to compute one Newton Iteration, and in our construction, we need only one Transformer layer (with one attention layer and one MLP layer) to compute one Newton Iteration. At the same time, we didn’t use Akyürek et al. [2022] for constructing Newton Iterations. Akyürek et al. [2022] is applied to initialize Newton and for reading out the prediction. 

In our construction, only the initialization and read-out prediction components use causal attention and softmax because Akyürek et al. [2022]’s construction is applied. To be more specific, those are the first 3 layers in initializing Iterative Newton and the last 5 layers in reading out the predictions from the computed pseudo-inverse. All the layers corresponding to the Iterative Newton updates are using full attention and normalized ReLU activations. 

_Remark_ B.6 _._ We note that our proof can be extended to causal attention for _n_ sufficiently larger than _d_ . Under causal attention (see Definition 3.1) with normalized ReLU activation, Equation (33) can **_O_** _d −_<sup><u>1</u></sup> 2<sup>**_I_**</sup><sup>_d_</sup> be rewritten as follows, given _t > d_ , we first choose **_G_** = , where the coefficient on the � **_O_** _d_ **_O_** _d_ � 

42 

upper right block is _−_<sup><u>1</u></sup> 2<sup>instead of</sup><sup>_−_</sup><sup>_<u>n</u>_</sup> 2<sup>originally.Then</sup> 



where **Σ**<sup>ˆ</sup> =<sup><u>1</u></sup> _t_ � _tj_ =1<sup>**_x_**</sup><sup>_j_</sup><sup>**_x_**</sup><sup>_⊤_</sup> _j_<sup>is the estimate of the covariance matrix given seen in-context examples</sup> _{_ **_x_** _j, yj}_<sup>_t_</sup> _j_ =1<sup>so far.Since</sup><sup>_t > d_,</sup><sup>**Σ**ˆis an unbiased estimate for</sup><sup>**Σ**</sup><sup>_≈_</sup> _n_<sup><u>1</u></sup><sup>**_S_**if</sup><sup>_n_is sufficiently large.The</sup> rest of the proof follows similarly, up to the perturbation introduced by the error in the estimate of **Σ** ˆ . We also note when _t < d_ , the estimate **Σ**<sup>ˆ</sup> =<sup><u>1</u></sup> _t_ � _tj_ =1<sup>**_x_**</sup><sup>_j_</sup><sup>**_x_**</sup><sup>_⊤_</sup> _j_<sup>is no longer a valid covariance matrix</sup> since it’s singular. Then this gives different **_T_**<sup>(</sup><sup>_ℓ_+1)</sup> for different time stamp _t < d_ and such error may propagate in our proof. Hence, a formal extension to causal models requires extensive analysis of the error bounds and it is beyond the scope of this work. Nonetheless, we provide a plausible direction of such an extension. 

### **B.3 Iterative Newton as a Sum of Moments Method** 

Recall that Iterative Newton’s method finds **_S_**<sup>_†_</sup> as follows 



We can expand the iterative equation to moments of **_S_** as follows. 



Let’s do this one more time for **_M_** 2. 



We can see that **_M_** _k_ are summations of moments of **_S_** , with respect to some pre-defined coefficients from the Newton’s algorithm. Hence Iterative Newton is a special of an algorithm which computes an approximation of the inverse using second-order moments of the matrix, 



43 

with coefficients _βs ∈_ R. 

We note that Transformer circuits can represent other sum of moments other than Newton’s method. We can introduce different coefficients _βi_ than in the proof of Theorem 5.1 by scaling the value matrices or through the MLP layers. 

### **B.4 Estimated weight vectors lie in the span of previous examples** 

What properties can we infer and verify for the weight vectors which arise from Newton’s method? A straightforward one arises from interpreting any sum of moments method as a kernel method. We can expand **_S_**<sup>_s_</sup> as follows 



Then we have 



where **_X_** is the data matrix, **_β_** are coefficients of moments given by the sum of moments method and _ϕt_ ( _·_ ) is some function which assigns some weight to the _i_ -th datapoint, based on all other datapoints. Therefore if the Transformer implements a sum of moments method (such as Newton’s method), then its induced weight vector **_w_** ˜ _t_ (Transformers _| {_ **_x_** _i, yi}_<sup>_t_</sup> _i_ =1<sup>)afterseeingin-context</sup> examples _{_ **_x_** _i, yi}_<sup>_t_</sup> _i_ =1<sup>should lie in the span of the examples</sup><sup>_{_</sup><sup>**_x_**</sup><sup>_i}t_</sup> _i_ =1<sup>:</sup> 



We test this hypothesis. Given a sequence of in-context examples _{_ **_x_** _i, yi}_<sup>_t_</sup> _i_ =1<sup>,we fit coefficients</sup> _{ai}_<sup>_t_</sup> _i_ =1<sup>in Equation (47) to minimize MSE loss:</sup> 



44 

We then measure the quality of this fit across different number of in-context examples _t_ , and visualize the residual error in Figure 27. We find that even when _t < d_ , Transformers’ induced weights still lie close to the span of the observed examples **_x_** _i_ ’s. This provides an additional validation of our proposed mechanism. 



<!-- Start of picture text -->
Linearity Error  vs. # In-Context Examples<br>Transformers<br>0.00012 OLS<br>0.00010<br>0.00008<br>0.00006<br>0.00004<br>0.00002<br>0.00000<br>1 5 10 15 20 25 30 35 40<br># of In-Context Examples<br>Linearity Error (MSE)<br><!-- End of picture text -->

Figure 27: Verification of hypothesis that the Transformers induced weight vector **_w_** lies in the span of observed examples _{_ **_x_** _i}_ . 

45 

