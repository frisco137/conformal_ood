# Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection 

Yu Bai<sup>∗§</sup> Fan Chen<sup>†§</sup> Huan Wang<sup>∗</sup> Caiming Xiong<sup>∗</sup> Song Mei<sup>‡§</sup> 

July 7, 2023 

##### **Abstract** 

Neural sequence models based on the transformer architecture have demonstrated remarkable _incontext learning_ (ICL) abilities, where they can perform new tasks when prompted with training and test examples, without any parameter update to the model. This work advances the understandings of the strong ICL abilities of transformers. We first provide a comprehensive statistical theory for transformers to perform ICL by deriving end-to-end quantitative results for the expressive power, in-context prediction power, and sample complexity of pretraining. Concretely, we show that transformers can implement a broad class of standard machine learning algorithms in context, such as least squares, ridge regression, Lasso, convex risk minimization for generalized linear models (such as logistic regression), and gradient descent on two-layer neural networks, with near-optimal predictive power on various in-context data distributions. Using an efficient implementation of in-context gradient descent as the underlying mechanism, our transformer constructions admit mild bounds on the number of layers and heads, and can be learned with polynomially many pretraining sequences. 

Building on these “base” ICL algorithms, intriguingly, we show that transformers can implement more complex ICL procedures involving _in-context algorithm selection_ , akin to what a statistician can do in real life—A _single_ transformer can adaptively select different base ICL algorithms—or even perform qualitatively different tasks—on different input sequences, without any explicit prompting of the right algorithm or task. We both establish this in theory by explicit constructions, and also observe this phenomenon experimentally. In theory, we construct two general mechanisms for algorithm selection with concrete examples: (1) Pre-ICL testing, where the transformer determines the right task for the given sequence (such as choosing between regression and classification) by examining certain summary statistics of the input sequence; (2) Post-ICL validation, where the transformer selects—among multiple base ICL algorithms (such as ridge regression with multiple regularization strengths)—a near-optimal one for the given sequence using a train-validation split. As an example, we use the post-ICL validation mechanism to construct a transformer that can perform nearly Bayes-optimal ICL on a challenging task—noisy linear models with mixed noise levels. Experimentally, we demonstrate the strong in-context algorithm selection capabilities of standard transformer architectures. 

## **1 Introduction** 

Large neural sequence models have demonstrated remarkable _in-context learning_ (ICL) capabilities [12], where models can make accurate predictions on new tasks when prompted with training examples from the same task, in a zero-shot fashion without any parameter update to the model. A prevalent example is large language models based on the transformer architecture [81], which can perform a diverse range of tasks in context when trained on enormous text [12, 87]. Recent models in this paradigm such as GPT-4 achieve surprisingly impressive ICL performance that makes them akin to a general-purpose agent in many 

> ∗Salesforce AI Research. Email: `{yu.bai,huan.wang,cxiong}@salesforce.com` 

> †Peking University. Email: `chern@pku.edu.cn` 

> ‡UC Berkeley. Email: `songmei@berkeley.edu` 

> §Equal technical and directional contributions. Code for our experiments is available at `https://github.com/allenbai01/transformers-as-statisticians` . 

1 



<!-- Start of picture text -->
Example 1: Ridge with different  λ Mechanism 1: Post-ICL Validation Example 2: Regression + Classification Mechanism 2: Pre-ICL Testing<br>yN +1 : 𝖱𝗂𝖽𝗀𝖾 λ 1( D ,  xN +1) 𝖱𝗂𝖽𝗀𝖾 λ 2( D ,  xN +1) yN +1 = 𝖠𝗅𝗀 k ⋆( D train,  xN +1) yN +1 : 𝖫𝗂𝗇𝖱𝖾𝗀( D ,  xN +1) 𝖫𝗈𝗀𝖱𝖾𝗀( D ,  xN +1) yN +1 = 𝖠𝗅𝗀 k ⋆( D ,  xN +1)<br>Which loss is the smallest?<br>𝖠𝗅𝗀1 …… 𝖠𝗅𝗀 K<br>Transformer 𝖫𝗈𝗌𝗌1( D val) …… 𝖫𝗈𝗌𝗌 K ( D val) Transformer<br>dist 1 dist K<br>𝖠𝗅𝗀1( D train) …… 𝖠𝗅𝗀 K ( D train) Which distribution?<br>Data 1  Data 2  Train-validation split Data 1  Data 2<br>(Reg w/noise  σ 1) (Reg w/noise  σ 2) Data TF (Regression) (Classification) Data TF<br><!-- End of picture text -->



<!-- Start of picture text -->
Example 1: Ridge with different  λ<br><!-- End of picture text -->

Figure 1: **Illustration of in-context algorithm selection, and two mechanisms constructed in our theory.** _Left, middle-left_ : A single transformer can perform ridge regression with different _λ_ ’s on input sequences with different observation noise; we prove this by the **post-ICL validation** mechanism (Section 4.1). _Middle-right, right_ : A single transformer can perform linear regression on regression data and logistic regression on classification data; we prove this via the **pre-ICL testing** mechanism (Section 4.2). 

aspects [62, 14]. Such strong capabilities call for better understandings, which a recent line of work tackles from various aspects [48, 90, 28, 69, 15, 56, 61]. 

Recent pioneering work of Garg et al. [31] proposes an interpretable and theoretically amenable setting for understanding ICL in transformers. They perform ICL experiments where input tokens are real-valued (input, label) pairs generated from standard statistical models such as linear models (and the sparse version), neural networks, and decision trees. Garg et al. [31] find that transformers can learn to perform ICL with prediction power (and fitted functions) matching standard machine learning algorithms for these settings, such as least squares for linear models, and Lasso for sparse linear models. Subsequent work further studies the internal mechanisms [2, 83, 18], expressive power [2, 32], and generalization [46] of transformers in this setting. However, these works only showcase simple mechanisms such as regularized regression [31, 2, 46] or gradient descent [2, 83, 18], which are arguably only a small subset of what transformers are capable of in practice; or expressing universal function classes not specific to ICL [86, 32]. This motivates the following question: 

#### _How do transformers learn in context beyond implementing simple algorithms?_ 

This paper makes steps on this question by making two main contributions: (1) We **unveil a general mechanism—** **_in-context algorithm selection_** —by which a _single_ transformer can adaptively _select different “base” ICL algorithms_ to use on _different ICL instances_ , without any explicit prompting of the right algorithm to use in the input sequence. For example, a transformer may choose to perform ridge regression with regularization _λ_ 1 on ICL instance 1, and _λ_ 2 on ICL instance 2 (Figure 2); or perform regression on ICL instance 1 and classification on ICL instance 2 (Figure 5). This adaptivity allows transformers to achieve much stronger ICL performance than the base ICL algorithms. We both prove this in theory, and demonstrate this phenomenon empirically on standard transformer architectures. (2) Along the way, equally importantly, we present a first comprehensive theory for ICL in transformers by establishing end-to-end quantitative guarantees for the **expressive power, in-context prediction performance, and sample complexity of pretraining** . These results add upon the recent line of work on the statistical learning theory of transformers [93, 86, 27, 38], and lay out a foundation for the intriguing special case where the _learning targets are themselves ICL algorithms_ . 

#### **Summary of contributions and paper outline** 

- We prove that transformers can implement a broad class of standard machine learning algorithms in context, such as least squares and ridge regression (Section 3.1), convex risk minimization for learning generalized linear models (such as logistic regression; Section 3.2), Lasso (Section 3.3), and gradient descent for two-layer neural networks (Section 3.4 & Appendix G). Our constructions admit mild bounds on the number of layers, heads, and weight norms, and achieve near-optimal prediction power on many in-context data distributions. 

- Technically, the above transformer constructions build on a new efficient implementation of in-context 

2 

- (a) Noisy linear reg with noise _σ_ 1 (b) Noisy linear reg with noise _σ_ 2 (c) Task 1 vs. task 2 at token 20 



<!-- Start of picture text -->
1.0 TF_alg_select TF_noise_1 1.4 TFTF_noise_1_alg_select 1.4 TF_alg_selectTF_noise_1<br>0.8 TF_noise_2 1.2 TF_noise_2 TF_noise_2<br>ridge_lam_1 ridge_lam_1 1.2 ridge_lam_1<br>0.6 ridge_lam_2 1.0 ridge_lam_2 ridge_lam_2<br>ridge analytical<br>0.4 0.8 1.0 Bayes Bayes_err_noise_2_ err _ noise _ 1<br>0.2 0.6 0.8<br>0.0 0.4 0.6<br>0 10 20 30 40 0 10 20 30 40 0.10 0.15 0.20 0.25 0.30<br>in-context examples in-context examples noisy_reg_noise_1<br>square loss<br>noisy_reg_noise_2<br><!-- End of picture text -->

Figure 2: In-context algorithm selection on two separate noisy linear regression tasks with noise ( _σ_ 1 _, σ_ 2) = (0 _._ 1 _,_ 0 _._ 5). _(a,b)_ A **single transformer** `TF_alg_select` **simultaneously approaches the performance of the two individual Bayes predictors** `ridge_lam_1` on task 1 and `ridge_lam_2` on task 2. _(c)_ At token 20 (using example _{_ 0 _, . . . ,_ 19 _}_ for training), `TF_alg_select` approaches the Bayes error on two tasks simultaneously, and **outperforms ridge regression with any fixed** _λ_ . _(a,b,c)_ Note that transformers pretrained on a single task ( `TF_noise_1` , `TF_noise_2` ) perform near-optimally on that task but suboptimally on the other task. More details about the setup and training method can be found in Section 6.2. 

gradient descent (Section 3.5), which could be broaderly applicable. For a broad class of smooth convex empirical risks over the in-context training data, we construct an ( _L_ +1)-layer transformer that approximates _L_ steps of gradient descent. Notably, the approximation error accumulates only _linearly_ in _L_ , utilizing a stability-like property of smooth convex optimization. 

- We prove that transformers can perform in-context algorithm selection (Section 4). We construct two algorithm selection mechanisms: Post-ICL validation (Section 4.1), and Pre-ICL testing (Section 4.2). For both mechanisms, we provide general constructions as well as concrete examples. Figure 1 provides a pictorial illustration of the two mechanisms. 

- As a concrete application, using the post-ICL validation mechanism, we construct a transformer that can perform nearly Bayes-optimal ICL on noisy linear models with _mixed_ noise levels (Section 4.1.1), a more complex task than those considered in existing work. 

- We provide the first line of results for _pretraining_ transformers to perform the various ICL tasks above, from polynomially many training sequences (Section 5). 

- Experimentally, we find that learned transformers indeed exhibit strong in-context algorithm selection capabilities in the settings considered in our theory (Section 6). For example, Figure 2 shows that a _single_ transformer can approach the individual Bayes risks (the optimal risk among all possible algorithms) simultaneously on two noisy linear models with different noise levels. 

**Transformers as statisticians** We humbly remark that the typical toolkit of a statistician contains much more beyond those covered in this work, including and not limited to inference, uncertainty quantification, and theoretical analysis. This work merely aims to show the algorithm selection capability of transformers, akin to what a statistician _can_ do. 

### **1.1 Related work** 

**In-context learning** The in-context learning (ICL) capability of large language models (LLMs) has gained significant attention since demonstrated on GPT-3 Brown et al. [12]. A number of subsequent empirical studies have contributed to a better understanding of the capabilities and limitations of ICL in LLM systems, which include but are not limited to [48, 54, 55, 49, 96, 71, 69, 28, 44, 88]. For an overview of ICL, see the survey by Dong et al. [24] which highlights some key findings and advancements in this direction. 

A line of recent work investigates why and how LLMs perform ICL [90, 31, 83, 2, 18, 32, 46, 67]. In particular, Xie et al. [90] propose a Bayesian inference framework explaining how ICL works despite formatting differences between training and inference distributions. Garg et al. [31] show empirically that transformers 

3 

could be trained from scratch to perform ICL of linear models, sparse linear models, two-layer neural networks, and decision trees. Li et al. [46] analyze the generalization error of trained ICL transformers from a stability viewpoint. They also experimentally show that transformers could perform “in-context model selection” (conceptually similar to in-context algorithm selection considered in this work) in specific tasks and presented related theoretical hypotheses. However, they do not provide concrete mechanisms or constructions for in-context model selection. A recent work [95] shows that pretrained transformers can perform Bayesian inference in latent variable models, which may also be interpreted as a mechanism for ICL. Our experimental findings extend these results by unveiling and demonstrating the in-context algorithm selection capabilities of transformers. 

Closely related to our theoretical results are [83, 2, 18, 32], which show (among many things) that transformers can perform ICL by simulating gradient descent. However, these results do not provide quantitative error bounds for simulating multi-step gradient descent, and only handle linear regression models or their simple variants. Among these works, Akyürek et al. [2] showed that transformers can implement learning algorithms for linear models based on gradient descent and closed-form ridge regression; it also presented preliminary evidence that learned transformers perform ICL similar to Bayes-optimal ridge regression. Our work builds upon and substantially extends this line of work by (1) providing a more efficient construction for in-context gradient descent; (2) providing an end-to-end theory with additional results for pretraining and statistical power; (3) analyzing a broader spectrum of ICL algorithms, including least squares, ridge regression, Lasso, convex risk minimization for generalized linear models, and gradient descent on two-layer neural networks; and (4) constructing more complex ICL procedures using in-context algorithm selection. 

When in-context data are generated from a prior, the Bayes risk is a theoretical lower bound for the risk of any possible ICL algorithm, including transformers. Xie et al. [90], Akyürek et al. [2] observe that learned transformers behave closely to the Bayes predictor on a variety of tasks such as hidden Markov models [90] and noisy linear regression with a fixed noise level [2, 46]. Using the in-context algorithm selection mechanism (more precisely the post-ICL validation mechanism), we show that transformers can perform nearly-Bayes optimal ICL in noisy linear models with mixed noise levels (a strictly more challenging task than considered in [2, 46]), with both concrete theoretical guarantees (Section 4.1.1) and empirical evidence (Figure 2 & 4b). 

**Transformers and its theory** The transformer architecture, introduced by [81], has revolutionized natural language processing and been adopted in most of the recently developed large language models such as BERT and GPT [65, 21, 12]. Broaderly, transformers have demonstrated remarkable performance in many other fields of artificial intelligence such as computer vision, speech, graph processing, reinforcement learning, and biological applications [23, 25, 50, 66, 92, 16, 40, 70, 62, 14]. Towards a better theoretical understanding, recent work has studied the capabilities [93, 64, 36, 91, 11, 94, 47], limitations [33, 10], and internal workings [28, 76, 89, 27, 61] of transformers. 

We remark that the transformer architecture used in our theoretical constructions differs from the standard one by replacing the softmax activation (in the attention layers) with a (normalized) ReLU function. Transformers with ReLU activations is experimentally studied in the recent work of Shen et al. [75], who find that they perform as well as the standard softmax activation in many NLP tasks. 

**Meta-learning** Training models (such as transformers) to perform ICL can be viewed as an approach for the broader problem of learning-to-learn or meta-learning [74, 58, 78]. A number of other approaches has been studied extensively for this problem, including (and not limited to) training a meta-learner on how to update the parameters of a downstream learner [9, 45], learning parameter initializations that quickly adapt to downstream tasks [29, 68], learning latent embeddings that allow for effective similarity search [77]. Most relevant to the ICL setting are approaches that directly take as input examples from a downstream task and a query input and produce the corresponding output [34, 57, 72, 43]. For a comprehensive overview, see the survey [35]. 

Theoretical aspects of meta-learning have received significant recent interest [7, 51, 26, 80, 19, 30, 42, 39, 85, 20, 5, 73, 17, 97]. In particular, [51, 26, 80] analyzed the benefit of multi-task learning through a representation learning perspective, and [85, 20, 5, 73, 97] studied the statistical properties of learning the parameter initialization for downstream tasks. 

4 

**Techniques** We build on various existing techniques from the statistics and learning theory literature to establish our approximation and generalization guarantees for transformers. For the approximation component, we rely on a technical result of Bach [4] on the approximation power of ReLU networks. We use this result to show that transformers can approximate gradient descent (GD) on a broad range of loss functions, substantially extending the results of [83, 2, 18] who primarily consider the square loss. The recent work of Giannou et al. [32] also approximates GD with general loss functions by transformers, though using a different technique of forcing the softmax activations to act as sigmoids. Our analyses of Lasso and generalized linear models build on [84, 59, 1, 53]. Our generalization bound for transformers (used in our pretraining results) build on a chaining argument [84]. 

## **2 Preliminaries** 

We consider a sequence of _N_ input vectors _{_ **h** _i}_<sup>_N_</sup> _i_ =1<sup>_⊂_R</sup><sup>_D_,writtencompactlyasaninputmatrix</sup><sup>**H**=</sup> [ **h** 1 _, . . . ,_ **h** _N_ ] _∈_ R<sup>_D×N_</sup> , where each **h** _i_ is a column of **H** (also a _token_ ). Throughout this paper, we let _σ_ ( _t_ ) := ReLU( _t_ ) = max _{t,_ 0 _}_ denote the standard relu activation. 

### **2.1 Transformers** 

We consider transformer architectures that process any input sequence **H** _∈_ R<sup>_D×N_</sup> by applying (encodermode<sup>1</sup> ) attention layers and MLP layers formally defined as follows. 

**Definition 1** (Attention layer) **.** _A (self-)attention layer with M heads is denoted as_ Attn **_θ_** ( _·_ ) _with parameters_ **_θ_** = _{_ ( **V** _m,_ **Q** _m,_ **K** _m_ ) _}m∈_ [ _M_ ] _⊂_ R<sup>_D×D_</sup> _. On any input sequence_ **H** _∈_ R<sup>_D×N_</sup> _,_ 



_where σ_ : R _→_ R _is the ReLU function. In vector form,_ 



Above, (1) uses a normalized ReLU activation _t �→ σ_ ( _t_ ) _/N_ in place of the standard softmax activation, which is for technical convenience and does not affect the essence of our study<sup>2</sup> . 

**Definition 2** (MLP layer) **.** _A (token-wise) MLP layer with hidden dimension D_<sup>_′_</sup> _is denoted as_ MLP **_θ_** ( _·_ ) _with parameters_ **_θ_** = ( **W** 1 _,_ **W** 2) _∈_ R<sup>_D′×D_</sup> _×_ R<sup>_D×D′_</sup> _. On any input sequence_ **H** _∈_ R<sup>_D×N_</sup> _,_ 



_where σ_ : R _→_ R _is the ReLU function. In vector form, we have_ **h**<sup>�</sup> _i_ = **h** _i_ + **W** 2 _σ_ ( **W** 1 **h** _i_ ) _._ 

We consider a transformer architecture with _L ≥_ 1 transformer layers, each consisting of a self-attention layer followed by an MLP layer. 

**Definition 3** (Transformer) **.** _An L-layer transformer, denoted as_ TF **_θ_** ( _·_ ) _, is a composition of L self-attention layers each followed by an MLP layer:_ **H**<sup>(</sup><sup>_L_)</sup> = TF **_θ_** ( **H**<sup>(0)</sup> ) _, where_ **H**<sup>(0)</sup> _∈_ R<sup>_D×N_</sup> _is the input sequence, and_ 



_Above, the parameter_ **_θ_** = ( **_θ_** `attn`<sup>(1:</sup><sup>_L_)</sup><sup>_,_</sup><sup>**_θ_**</sup> `mlp`<sup>(1:</sup><sup>_L_))</sup><sup>_consists of the attention layers_</sup><sup>**_θ_**</sup> `attn`<sup>(</sup><sup>_ℓ_)=</sup><sup>_{_(</sup><sup>**V**</sup> _m_<sup>(</sup><sup>_ℓ_)</sup><sup>_,_</sup><sup>**Q**(</sup> _m_<sup>_ℓ_)</sup><sup>_,_</sup><sup>**K**(</sup> _m_<sup>_ℓ_))</sup><sup>_}_</sup> _m∈_ [ _M_ ( _ℓ_ )]<sup>_⊂_</sup> R<sup>_D×D_</sup> _and the MLP layers_ **_θ_** `mlp`<sup>(</sup><sup>_ℓ_)= (</sup><sup>**W**</sup> 1<sup>(</sup><sup>_ℓ_)</sup><sup>_,_</sup><sup>**W**</sup> 2<sup>(</sup><sup>_ℓ_))</sup><sup>_∈_R</sup><sup>_D_(</sup><sup>_ℓ_)</sup><sup>_×D×_R</sup><sup>_D×D_(</sup><sup>_ℓ_)</sup><sup>_.We will frequently consider “_attention-</sup> only _” transformers with_ **W** 1<sup>(</sup><sup>_ℓ_)</sup><sup>_,_</sup><sup>**W**</sup> 2<sup>(</sup><sup>_ℓ_)</sup> = **0** _, which we denote as_ TF<sup>0</sup> **_θ_**<sup>(</sup><sup>_·_)</sup><sup>_for shorthand,with_</sup><sup>**_θ_**=</sup><sup>**_θ_**(1:</sup><sup>_L_):=</sup><sup>**_θ_**</sup> `attn`<sup>(1:</sup><sup>_L_)</sup><sup>_._</sup> 

> 1Many of our results can be generalized to decoder-based architectures; see Appendix B for a discussion. 

> 2For each query index _i_ , the attention weights _{σ_ ( _⟨_ **Q** _m_ **h** _i,_ **K** _m_ **h** _j ⟩_ ) _/N }j∈_ [ _N_ ] is also a set of non-negative weights that sum to _O_ (1) (similar as a softmax probability distribution) in typical scenarios. 

5 

We additionally define the following norm of a transformer TF **_θ_** : 



In (2), the choices of the operator norm and max/sums are for convenience only and not essential, as our results (e.g. for pretraining) depend only logarithmically on _|||_ **_θ_** _|||_ . 

### **2.2 In-context learning** 

iid In an in-context learning (ICL) instance, the model is given a dataset _D_ = _{_ ( **x** _i, yi_ ) _}i∈_ [ _N_ ] _∼_ P and a new test input **x** _N_ +1 _∼_ P **x** for some data distribution P, where _{_ **x** _i}i∈_ [ _N_ ] _⊆_ R<sup>_d_</sup> are the input vectors, _{yi}i∈_ [ _N_ ] _⊆_ R are the corresponding labels (e.g. real-valued for regression, or _{_ 0 _,_ 1 _}_ -valued for binary classification), and **x** _N_ +1 is the test input on which the model is required to make a prediction. Different from standard supervised learning, in ICL, each instance ( _D,_ **x** _N_ +1) is in general drawn from a different distribution P _j_ , such as a linear model with a new ground truth coefficient **w** _⋆,j ∈_ R<sup>_d_</sup> . Our goal is to construct _fixed_ transformer to perform ICL on a large set of P _j_ ’s. 

We consider using transformers to perform ICL, in which we encode ( _D,_ **x** _N_ +1) into an input sequence **H** _∈_ R<sup>_D×_(</sup><sup>_N_+1)</sup> . In our theory, we use the following format, where the first two rows contain ( _D,_ **x** _N_ +1) (zero at the location for _yN_ +1), and the third row contains fixed vectors _{_ **p** _i}i∈_ [ _N_ +1] with ones, zeros, and indicator for being the train token (similar to a positional encoding vector): 



We will choose _D_ = Θ( _d_ ), so that the hidden dimension of **H** is at most a constant multiple of _d_ . We then feed **H** into a transformer to obtain the output **H**<sup>�</sup> = TF **_θ_** ( **H** ) _∈_ R<sup>_D×_(</sup><sup>_N_+1)</sup> with the same shape, and � _read out_ the prediction _yN_ +1 from the ( _d_ + 1 _, N_ + 1)-th entry of **H**<sup>�</sup> = [ **h**<sup>�</sup> _i_ ] _i∈_ [ _N_ +1] (the entry corresponding � � to the missing test label): _yN_ +1 = ready( **H**<sup>�</sup> ) := ( **h**<sup>�</sup> _N_ +1) _d_ +1. The goal is to predict _yN_ +1 that is close to _yN_ +1 _∼_ P _y|_ **x** _N_ +1 measured by proper losses. 

**Generalization to predicting at every token using a decoder architecture** We emphasize that the setting above considers predicting only at the last token **x** _N_ +1, which is without much loss of generality. Our constructions may be generalized to predicting at every token, by using a decoder architecture and a corresponding input format (cf. Appendix B). Our theory focuses on predicting at the last token only, which simplifies the setting. Our experiments test both settings. 

**Miscellaneous setups** We assume bounded features and labels throughout the paper (unless otherwise specified, e.g. when **x** _i_ is Gaussian): _∥_ **x** _i∥_ 2 _≤ Bx_ and _|yi| ≤ By_ with probability one. We use the standard notation **X** = [ **x**<sup>_⊤_</sup> 1<sup>;</sup><sup>_. . ._;</sup><sup>**x**</sup><sup>_⊤_</sup> _N_<sup>]</sup><sup>_∈_R</sup><sup>_N×d_and</sup><sup>**y**=[</sup><sup>_y_1;</sup><sup>_. . ._;</sup><sup>_yN_]</sup><sup>_∈_R</sup><sup>_N_todenotethematrixof</sup> inputs and vector of labels, respectively. To prevent the transformer from blowing up on tail events, in all our results concerning (statistical) in-context prediction powers, we consider a clipped prediction � _yN_ +1 = read<sup>�</sup> y( **H**<sup>�</sup> ) := clip _R_ (( **h**<sup>�</sup> _N_ +1) _d_ +1), where clip _R_ ( _t_ ) := Proj[ _−R,R_ ]( _t_ ) is the standard clipping operator with (a suitably large) radius _R ≥_ 0 that varies in different problems. 

We will often use shorthand _yi_<sup>_′∈_R defined as</sup><sup>_y_</sup> _i_<sup>_′_=</sup><sup>_yi_for</sup><sup>_i ∈_[</sup><sup>_N_] and</sup><sup>_y_</sup> _N_<sup>_′_</sup> +1<sup>= 0 to simplify our notation, with</sup> which the input sequence **H** _∈_ R<sup>_D×_(</sup><sup>_N_+1)</sup> can be compactly written as **h** _i_ = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>**p**</sup><sup>_i_] = [</sup><sup>**x**</sup><sup>_i_;</sup><sup>_y_</sup> _i_<sup>_′_;</sup><sup>**0**</sup><sup>_D−d−_3; 1;</sup><sup>_ti_]</sup> for _i ∈_ [ _N_ + 1], where _ti_ := 1 _{i < N_ + 1 _}_ is the indicator for the training examples. 

## **3 Basic in-context learning algorithms** 

We begin by constructing transformers that approximately implement a variety of standard machine learning algorithms in context, with mild size bounds and near-optimal prediction power on many standard in-context 

6 

data distributions. 

### **3.1 In-context ridge regression and least squares** 

Consider the standard ridge regression estimator over the in-context training examples _D_ with regularization _λ ≥_ 0 (reducing to least squares at _λ_ = 0 and _N ≥ d_ ): 



We show that transformers can approximately implement (ICRidge) (proof in Appendix D.1). 

**Theorem 4** (Implementing in-context ridge regression) **.** _For any λ ≥_ 0 _,_ 0 _≤ α ≤ β with κ_ := _α_<sup>_<u>β</u>_</sup><sup><u>+</u></sup> +<sup>_λ_</sup> _λ_<sup>_,Bw>_0</sup><sup>_,_</sup> _and ε < BxBw/_ 2 _, there exists an L-layer attention-only transformer_ TF<sup>0</sup> **_θ_**<sup>_with_</sup> 



_(with R_ := max _{BxBw, By,_ 1 _}) such that the following holds. On any input data_ ( _D,_ **x** _N_ +1) _such that the problem (ICRidge) is well-conditioned and has a bounded solution:_ 







Theorem 4 presents the first quantitative construction for end-to-end in-context ridge regression up to arbitrary precision, and improves upon Akyürek et al. [2] whose construction does not give (or directly imply) an explicit error bound like (6). Further, the bounds on the number of layers and heads in (4) are mild (constant heads and logarithmically many layers). 

**Near-optimal in-context prediction power for linear problems** Combining Theorem 4 with standard analyses of linear regression yields the following corollaries (proofs in Appendix D.3 & D.4). 

**Corollary 5** (Near-optimal linear regression with transformers by approximating least squares) **.** _For any N ≥ O_<sup>�</sup> ( _d_ ) _, there exists an O_ ( _κ_ log( _κN/σ_ )) _-layer transformer_ **_θ_** _, such that on any_ P _satisfying standard statistical assumptions for least squares (Assumption A), its ICL prediction y_ � _N_ +1 _achieves_ 



Assumption A requires only generic tail properties such as sub-Gaussianity, and _not_ realizability (i.e., P follows a true linear model); _κ, σ_ above denote the covariance condition number and the noise level therein. The _O_<sup>�</sup> ( _dσ_<sup>2</sup> _/N_ ) excess risk is known to be rate-optimal for linear regression [37], and Corollary 5 achieves this in context with a transformer with only logarithmically many layers. 

Next, consider Bayesian linear models where each in-context data distribution P = P<sup>lin</sup> **w** _⋆_<sup>isdrawnfroma</sup> Gaussian prior _π_ : **w** _⋆ ∼_ N(0 _,_ **I** _d/d_ ), and ( **x** _, y_ ) _∼_ P<sup>lin</sup> **w** _⋆_<sup>issampledas</sup><sup>**x**</sup><sup>_∼_N(</sup><sup>**0**</sup><sup>_,_</sup><sup>**I**</sup><sup>_d_),</sup><sup>_y_=</sup><sup>_⟨_</sup><sup>**w**</sup><sup>_⋆,_</sup><sup>**x**</sup><sup>_⟩_+ N(0</sup><sup>_, σ_2).It</sup> is a standard result that the Bayes estimator of _yN_ +1 given ( _D,_ **x** _N_ +1) is given by ridge regression (ICRidge): _y_ � _N_<sup>Bayes</sup> +1<sup>:=</sup><sup>_⟨_</sup><sup>**w**</sup> ridge<sup>_λ,_</sup><sup>**x**</sup><sup>_N_+1</sup><sup>_⟩_with</sup><sup>_λ_=</sup><sup>_dσ_2</sup><sup>_/N_.Weshowthattransformersachievenearly-Bayesriskforthis</sup> problem, and we use 



to denote the Bayes risk of this problem under prior _π_ . 

**Corollary 6** (Nearly-Bayes linear regression with transformers by approximating ridge regression) **.** _Under the Bayesian linear model above with N ≥_ max _{d/_ 10 _, O_ (log(1 _/ε_ )) _}, there exists a L_ = _O_ (log(1 _/ε_ )) _-layer_ <u>1</u> _transformer such that_ E **w** _⋆,_ ( _D,_ **x** _N_ +1 _,yN_ +1)� 2<sup>(</sup><sup>_y_�</sup><sup>_N_+1</sup><sup>_−yN_+1)2�</sup> _≤_ BayesRisk _π_ + _ε._ 

7 

### **3.2 In-context learning of generalized linear models** 

As a natural generalization of linear regression, we now show that transformers can recover learn generalized linear models (GLMs) [52] (which includes logistic regression for linear classification as an important special case), by implementing the corresponding convex risk minimization algorithm in context, and achieve nearoptimal excess risk under standard statistical assumptions. 

Let _g_ : R _→_ R be a link function that is non-decreasing and _C_<sup>2</sup> -smooth. We consider the following convex empirical risk minimization (ERM) problem 



_t_ where _ℓ_ ( _t, y_ ) := _−yt_ + �0<sup>_g_(</sup><sup>_s_)</sup><sup>_ds_istheconvex(integral)lossassociatedwith</sup><sup>_g_.</sup> A canonical example of (ICGLM) is logistic regression, in which _g_ ( _t_ ) = _σ_ log( _t_ ) := (1 + _e_<sup>_−t_</sup> )<sup>_−_1</sup> is the sigmoid function, and the resulting _ℓ_ ( _t, y_ ) = _ℓ_ log( _t, y_ ) = _−yt_ + log(1 + _e_<sup>_t_</sup> ) is the logistic loss. 

The following result (proof in Appendix E.1) shows that, as long as the empirical risk _L_<sup>�</sup> _N_ satisfies strong convexity and bounded solution conditions (similar as in Theorem 4), transformers can approximately implement the ERM predictor _g_ ( _⟨_ **x** _N_ +1 _,_ **w** GLM _⟩_ ), with **w** GLM given by (ICGLM). 

**Theorem 7** (Implementing convex risk minimization for GLMs) **.** _For any_ 0 _< α < β with κ_ := _α_<sup>_<u>β</u>,Bw>_</sup> 0 _, Bx >_ 0 _, κw_ := _LgBx_<sup>2</sup><sup>_/α_+ 1</sup><sup>_andε < Bw/_2</sup><sup>_,thereexistsanattention-onlytransformer_TF0</sup> **_θ_**<sup>_with_</sup> 



_(where Lg_ := sup _t |g_<sup>_′_</sup> ( _t_ ) _|, R_ := max _{BxBw, By,_ 1 _}, and Cg >_ 0 _is a constant that depends only on R and the C_<sup>2</sup> _-smoothness of g within_ [ _−R, R_ ] _), such that the following holds. On any input data_ ( _D,_ **x** _N_ +1) _such that_ 





In Theorem 7, the number of heads scales as _O_<sup>�</sup> (1 _/ε_<sup>2</sup> ) as opposed to Θ(1) as in ridge regression (Theorem 4), due to the fact that the gradient of the loss is in general a smooth function that can be only _approximately_ expressed as a sum-of-relus (cf. Definition 12 & Lemma A.5) rather than exactly expressed as in the case for the square loss. 

**In-context prediction power** We next show that (proof in Appendix E.2) the transformer constructed in Theorem 7 achieves desirable statistical power if the in-context data distribution satisfies standard statistical assumptions for learning GLMs. Let _L_ P( **w** ) := E( **x** _,y_ ) _∼_ P[ _ℓ_ ( _⟨_ **w** _,_ **x** _⟩ , y_ )] denote the corresponding population risk for any distribution P of ( **x** _, y_ ). When P is _realizable_ by a generalized linear model of link function _g_ and parameter **_β_** in the sense that EP[ _y|_ **x** ] = _g_ ( _⟨_ **_β_** _,_ **x** _⟩_ ), it is a standard result that **_β_** is indeed a minimizer of _L_ P [41] (see also [6, Appendix A.3]). **Theorem 8** (Statistical guarantee for generalized linear models) **.** _For any fixed set of parameters defined in Assumption B, there exists a transformer_ **_θ_** _with L ≤O_ (log( _N_ )) _layers and_ max _ℓ∈_ [ _L_ ] _M_<sup>(</sup><sup>_ℓ_)</sup> _≤ O_<sup>�</sup> � _d_<sup>3</sup> _N_ � _, such that for any distribution_ P _satisfying Assumption B with those parameters, as long as N ≥O_ ( _d_ ) _,_ � � _that outputs yN_ +1 = read<sup>�</sup> y(TF **_θ_** ( **H** )) _and_ **w** = read<sup>�</sup> w(TF **_θ_** ( **H** )) _∈_ R<sup>_d_</sup> _(for another read-out function_ read<sup>�</sup> w _) satisfying the following._ 



8 

_(b) (Realizable setting) If there exists a_ **_β_** _∈_ R<sup>_d_</sup> _such that under_ P _,_ E[ _y|_ **x** ] = _g_ ( _⟨_ **_β_** _,_ **x** _⟩_ ) _almost surely, then_ E( _D,_ **x** _N_ +1 _,yN_ +1) _∼_ P�( _y_ � _N_ +1 _− yN_ +1)<sup>2�</sup> _≤_ E( **x** _N_ +1 _,yN_ +1) _∼_ P�( _g_ ( _⟨_ **_β_** _,_ **x** _N_ +1 _⟩_ ) _− yN_ +1)<sup>2�</sup> + _O_ ( _d/N_ ) _,_ (9) � _or equivalently,_ E[( _yN_ +1 _−_ E[ _yN_ +1 _|_ **x** _N_ +1])<sup>2</sup> ] _≤O_ ( _d/N_ ) _._ 

Above, _O_ ( _·_ ) hides constants that depend polynomially on the parameters in Assumption B. Similar as in Corollary 5, the _O_ ( _d/N_ ) excess risk obtained here matches the optimal (fast) rate for typical learning problems with _d_ parameters and _N_ samples [84]. 

Applying Theorem 8 to logistic regression, we have the following result as a direct corollary. Below, the Gaussian input assumption is for convenience only and can be genearalized to e.g. sub-Gaussian input. 

**Corollary 9** (In-context logistic regression) **.** _Consider any in-context data distribution_ P _satisfying_ 



_For the link function g_ = _σ_ log _and Bw_<sup>_⋆_=</sup><sup>_O_(1)</sup><sup>_,wecanchooseBw, Bµ, µg, Lg, µx, Kx, Ky_=Θ (1)</sup><sup>_sothat_</sup> _Assumption B holds. In that case, when N ≥O_ ( _d_ ) _, there exists a transformer_ **_θ_** _with L_ = _O_ (log( _N_ )) _layers, such that for any_ P _considered above,_ 

� _(a) The estimation_ **w** = read<sup>�</sup> w(TF **_θ_** ( **H** )) _outputted by_ **_θ_** _achieves excess risk bound (8)._ 

- _(b) (Realizable setting) Consider the logistic in-context data distribution_ 



_Then, for any distribution_ P = P<sup>log</sup> **_β_** _with ∥_ **_β_** _∥_ 2 _≤ Bw_<sup>_⋆,thepredictiony_�</sup><sup>_N_+1=</sup> read<sup>�</sup> y(TF **_θ_** ( **H** )) _of_ **_θ_** _additionally achieves the square loss excess risk (9)._ 

### **3.3 In-context Lasso** 

Consider the standard Lasso estimator [79] which minimizes an _ℓ_ 1-regularized linear regression loss _L_<sup>�</sup> lasso over the in-context training examples _D_ : 



We show that transformers can also approximate in-context Lasso with a mild number of layers, and can perform sparse linear regression in standard sparse linear models (proofs in Appendix F). 

**Theorem 10** (Implementing in-context Lasso) **.** _For any λN ≥_ 0 _, β >_ 0 _, Bw >_ 0 _, and ε >_ 0 _, there exists a L-layer transformer_ TF **_θ_** _with_ 



_(where R_ := max _{BxBw, By,_ 1 _}) such that the following holds. On any input data_ ( _D,_ **x** _N_ +1) _such that λ_ max( **X**<sup>_⊤_</sup> **X** _/N_ ) _≤ β and ∥_ **w** lasso _∥_ 2 _≤ Bw/_ 2 _,_ TF **_θ_** ( **H**<sup>(0)</sup> ) _approximately implements (ICLasso), in that it_ � � � _outputs yN_ +1 = _⟨_ **x** _N_ +1 _,_ **w** _⟩ with L_<sup>�</sup> lasso( **w** ) _− L_<sup>�</sup> lasso( **w** lasso) _≤ ε._ 

**Theorem 11** (Near-optimal sparse linear regression with transformers by approximating Lasso) **.** _For any d, N ≥_ 1 _, δ >_ 0 _, Bw_<sup>_⋆, σ>_0</sup><sup>_,thereexistsa_</sup> _O_<sup>�</sup> (( _Bw_<sup>_⋆_)2</sup><sup>_/σ_2</sup><sup>_×_(1 + (</sup><sup>_d/N_)))</sup><sup>_-layertransformer_</sup><sup>**_θ_**</sup><sup>_suchthatthe_</sup> _following holds: For any s and N ≥O_ ( _s_ log( _d/δ_ )) _, suppose that_ P _is an s-sparse linear model:_ **x** _i ∼_ N(0 _,_ **I** _d_ ) _, yi_ = _⟨_ **w** _⋆,_ **x** _i⟩_ + N(0 _, σ_<sup>2</sup> ) _for any ∥_ **w** _⋆∥_ 2 _≤ Bw_<sup>_⋆and∥_</sup><sup>**w**</sup><sup>_⋆∥_</sup> 0<sup>_≤s,thenwithprobabilityatleast_1</sup><sup>_−δ(overthe_</sup> _randomness of D), the transformer output y_ � _N_ +1 _achieves_ 



The _O_<sup>�</sup> ( _s_ log _d/N_ ) excess risk obtained in Theorem 11 is optimal up to log factors [59, 84]. We remark that Theorem 11 is not a direct corollary of Theorem 10; Rather, the bound on the number of layers in Theorem 11 requires a sharper convergence analysis of the (ICLasso) problem under sparse linear models (Appendix F.2), similar to [1]. 

9 

### **3.4 Gradient descent on two-layer neural networks** 

Thus far, we have focused on convex risks with (generalized) linear predictors of the form **x** _�→⟨_ **w** _,_ **x** _⟩_ . To move beyond both restrictions, as a primary example, we show that transformers can approximate in-context gradient descent on two-layer neural networks (NNs). 

Precisely, we consider a two-layer NN pred( **x** ; **w** ) :=<sup>�</sup><sup>_K_</sup> _k_ =1<sup>_ukr_(</sup><sup>**v**</sup> _k_<sup>_⊤_</sup><sup>**x**)parameterizedby</sup><sup>**w**=[</sup><sup>**v**</sup><sup>_k_;</sup><sup>_uk_]</sup><sup>_k∈_[</sup><sup>_K_]</sup><sup>_∈_</sup> R<sup>_K_(</sup><sup>_d_+1)</sup> , and the following associated ERM problem: 



where _W ⊂_ R<sup>_K_(</sup><sup>_d_+1)</sup> is a bounded domain. 

In Theorem G.1, we show that under suitable assumptions on ( _r, ℓ, W_ ), an 2 _L_ -layer transformer can implement _L_ -steps of _inexact_ gradient descent on (10). The construction itself is more sophisticated than the convex case (Theorem 13) due to the two-layer structure, even though the inexact gradient guarantee is similar as the convex case (Proposition C.2). As a corollary, the same transformer can approximate exact gradient descent (Corollary G.1), though the guarantee is expectedly much weaker than the convex case due to the non-convexity of the objective. See Appendix G for details. Compared with the result of [32, Algorithm 11] which implements SGD on two-layer neural networks using looped transformers, our construction is more quantitative (admits explicit size bounds) and arguably simpler. 

### **3.5 Mechanism: In-context gradient descent** 

Technically, the constructions in Section 3.1-3.3 rely on a new efficient construction for transformers to implement in-context gradient descent and its variants, which we present as follows. We begin by presenting the result for implementing (vanilla) gradient descent on convex empirical risks. 

_N_ **Gradient descent on empirical risk** Let _ℓ_ : R<sup>2</sup> _→_ R be a loss function. Let _L_<sup>�</sup> _N_ ( **w** ) := _N_<sup><u>1</u></sup> � _i_ =1<sup>_ℓ_(</sup><sup>**w**</sup><sup>_⊤_</sup><sup>**x**</sup><sup>_i, yi_)</sup> denote the empirical risk with loss function _ℓ_ on dataset _{_ ( **x** _i, yi_ ) _}i∈_ [ _N_ ], and 



denote the gradient descent trajectory on _L_<sup>�</sup> _N_ with initialization **w** GD<sup>0</sup><sup>_∈_R</sup><sup>_d_andlearningrate</sup><sup>_η>_0.</sup> We require the partial derivative of the loss _∂sℓ_ : ( _s, t_ ) _�→ ∂sℓ_ ( _s, t_ ) (as a bivariate function) to be approximable by a sum of relus, defined as follows. 

**Definition 12** (Approximability by sum of relus) **.** _A function g_ : R<sup>_k_</sup> _→_ R _is_ ( _ε_ approx _, R, M, C_ ) _-_ approximable by sum of relus _, if there exists a “_ ( _M, C_ ) _-sum of relus” function_ 



_such that_ sup **z** _∈_ [ _−R,R_ ] _k |g_ ( **z** ) _− fM,C_ ( **z** ) _| ≤ ε_ approx _._ 

Definition 12 is known to contain broad class of functions. For example, any mildly smooth _k_ -variate function is approximable by a sum of relus for any ( _ε_ approx _, R_ ), with mild bounds on ( _M, C_ ) (Proposition A.1, building on results of Bach [4]). Also, any function that is a ( _M, C_ )-sum of relus itself (which includes all piecewise linear functions) is by definition (0 _, ∞, M, C_ )-approximable by sum of relus. 

We show that _L_ steps of (ICGD) can be approximately implemented by an ( _L_ + 1)-layer transformer. 

**Theorem 13** (Convex ICGD) **.** _Fix any Bw >_ 0 _, L >_ 1 _, η >_ 0 _, and ε ≤ Bw/_ (2 _L_ ) _. Suppose that_ 

_1. The loss ℓ_ ( _·, ·_ ) _is convex in the first argument;_ 

_2. ∂sℓ is_ ( _ε, R, M, C_ ) _-approximable by sum of relus with R_ = max _{BxBw, By,_ 1 _}._ 

10 



<!-- Start of picture text -->
1 N M<br>Attention hN t +1 +1 = hN t +1 + N ∑ i =1 m ∑=1 σ ( ⟨ Qm hN t +1 , Km hi t ⟩) × Vm hi t<br>constructionWeight  hi t = wyx ii t , Qm w * * t = bcmm  × × w 1 t , Km wyx ii t = xy ii , Vm wyx ii t = − ηa m × x 00 i<br>w yx t +1 ii = wyx ii t − Nη ∑ i =1 N m ∑ M =1 am σ ( bm ⟨ x i , w t ⟩+ cm y i ) × x 00 i<br>approximationUniversal  ∂ sℓ ( s ,  t ) = ∑ M m =1 am  ⋅ σ ( bm  ⋅ s + cm  ⋅ t )<br>N<br>Gradient descent w t +1 = w t − Nη ∑ i =1 ∂ s ℓ ( ⟨ x i , w t ⟩, y i ) × x i<br><!-- End of picture text -->

Figure 3: Illustration of our main mechanism for implementing basic ICL algorithms: One attention layer implements a single (ICGD) iterate (Proposition C.2 & Theorem 13). Top: the attention mechanism as in Definition 1. Bottom: A single (ICGD) iterate. Middle: Linear algebraic illustration of the attention layer for implementing a GD update. 

_Then, there exists an attention-only transformer_ TF<sup>0</sup> **_θ_**<sup>_with_(</sup><sup>_L_+ 1)</sup><sup>_layers,_max</sup> _ℓ∈_ [ _L_ ]<sup>_M_(</sup><sup>_ℓ_)</sup><sup>_≤Mheadswithin_</sup> _the first L layers, and M_<sup>(</sup><sup>_L_+1)</sup> = 2 _such that for_ any input data ( _D,_ **x** _N_ +1) _such that_ 



_1. (Parameter space) For every ℓ ∈_ [ _L_ ] _, the ℓ-th layer’s output_ **H**<sup>(</sup><sup>_ℓ_)</sup> = TF<sup>0</sup> **_θ_**<sup>(1:</sup><sup>_ℓ_)(</sup><sup>**H**(0))</sup><sup>_approximatesℓsteps_</sup> _of (ICGD): We have_ **h**<sup>(</sup> _i_<sup>_ℓ_)</sup> = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>**w**�</sup><sup>_ℓ_;</sup><sup>**0**</sup><sup>_D−_2</sup><sup>_d−_3; 1;</sup><sup>_ti_]</sup><sup>_foreveryi ∈_[</sup><sup>_N_+ 1]</sup><sup>_,where_</sup> ��� **w** _ℓ −_ **w** GD _ℓ_ ��2<sup>_≤ε ·_(</sup><sup>_ℓηBx_)</sup><sup>_._</sup> 

_Note that the bound scales as O_ ( _ℓ_ ) _, a_ linear _error accumulation._ 



_Further, the transformer admits norm bound |||_ **_θ_** _||| ≤_ 2 + _R_ + 2 _ηC._ 

The proof can be found in Appendix C.3. Theorem 13 substantially generalizes that of von Oswald et al. [83] (which only does GD on square losses with a _linear_ self-attention), and is simpler than the ones in Akyürek et al. [2] and Giannou et al. [32]. See Figure 3 for a pictorial illustration of the basic component of the construction, which implements a single step of gradient descent using a single attention layer (Proposition C.2). 

Technically, we utilize the stability of _convex_ gradient descent as in the following lemma (proof in Appendix C.4) to obtain the _linear_ error accumulation in Theorem 13; the error accumulation will become _exponential_ in _L_ in the non-convex case in general; see Lemma G.1(b). 

**Lemma 14** (Composition of error for approximating convex GD) **.** _Suppose f_ : R<sup>_d_</sup> _→_ R _is a convex function. Let_ **w**<sup>_⋆_</sup> _∈_ arg min **w** _∈_ R _d f_ ( **w** ) _, R ≥_ 2 _∥_ **w**<sup>_⋆_</sup> _∥_ 2 _, and assume that ∇f is Lf -smooth on_ B<sup>_d_</sup> 2<sup>(</sup><sup>_R_)</sup><sup>_.Letsequences_</sup> _{_ **w** �<sup>_ℓ_</sup> _}ℓ≥_ 0 _⊂_ R<sup>_d_</sup> _and {_ **w** GD<sup>_ℓ}ℓ≥_0</sup><sup>_⊂_R</sup><sup>_dbegivenby_</sup><sup>**w**�0=</sup><sup>**w**</sup> GD<sup>0=</sup><sup>**0**</sup><sup>_,_</sup> 



11 

_L L for all ℓ ≥_ 0 _. Then as long as η ≤_ 2 _/Lf , for any_ 0 _≤ L ≤ R/_ (2 _ε_ ) _, it holds that_ ��� **w** _−_ **w** GD��2<sup>_≤Lεand_</sup> � _∥_ **w**<sup>_L_</sup> _∥_ 2 _≤_<sup>_<u>R</u>_</sup> 2<sup>+</sup><sup>_Lε ≤R._</sup> 

**Proximal gradient descent** We also provide a result for implementing proximal gradient descent [63], a widely-studied variant of gradient descent that is suitable for handling regularized losses. In particular, we use it to elegantly handle the non-smooth ( _ℓ_ 1-norm) regularizer in (ICLasso). The construction utilizes the MLP layers within transformers to implement proximal operators; see Appendix C.1. 

## **4 In-context algorithm selection** 

We now show that transformers can perform various kinds of _in-context algorithm selection_ , which allows them to implement more complex ICL procedures by adaptively selecting different “base” algorithms on different input sequences. We construct two general mechanisms: _Post-ICL validation_ , and _Pre-ICL testing_ ; See Figure 1 for a pictorial illustration. 

### **4.1 Post-ICL validation mechanism** 

In our first mechanism, post-ICL validation, the transformer begins by implementing a _train-validation split D_ = ( _D_ train _, D_ val), and running _K base_ ICL algorithms on _D_ train. Let _{fk}k∈_ [ _K_ ] _⊂_ (R<sup>_d_</sup> _→_ R) denote the _K_ learned predictors, and 



denote the validation loss of any predictor _f_ . 

We show that (proof in Appendix H.1) a 3-layer transformer can output a predictor _f_<sup>�</sup> that achieves nearly the smallest validation loss, and thus nearly optimal expected loss if _L_<sup>�</sup> val concentrates around the expected loss _L_ . Below, the input sequence **H** uses a generalized positional encoding **p** _i_ := [ **0** _D−_ ( _d_ +3); 1; _ti_ ] in (3), where _ti_ := 1 for _i ∈D_ train, _ti_ := _−_ 1 for _i ∈D_ val, and _tN_ +1 := 0. 

**Proposition 15** (In-context algorithm selection via train-validation split) **.** _Suppose that ℓ_ ( _·, ·_ ) _in_ (11) _is approximable by sum of relus (Definition 12, which includes all C_<sup>3</sup> _-smooth bivariate functions). Then there exists a 3-layer transformer_ TF **_θ_** _that maps (recalling yi_<sup>_′_=</sup><sup>_yi_1</sup><sup>_{i < N_+ 1</sup><sup>_})_</sup> 



_where the predictor f_<sup>�</sup> : R<sup>_d_</sup> _→_ R _is a convex combination of {fk_ : _L_<sup>�</sup> val( _fk_ ) _≤_ min _k⋆∈_ [ _K_ ] _L_<sup>�</sup> val( _fk⋆_ ) + _γ}. As a corollary, for any convex risk L_ : (R<sup>_d_</sup> _→_ R) _→_ R _, f_<sup>�</sup> _satisfies_ 



**Ridge regression with in-context regularization selection** As an example, we use Proposition 15 to construct a transformer to perform in-context ridge regression with regularization selection according to <u>1</u> the _unregularized_ validation loss _L_<sup>�</sup> val( **w** ) := 2 _|D_ val _|_ �( _xi,yi_ ) _∈D_ val<sup>(</sup><sup>_⟨_</sup><sup>**w**</sup><sup>_,_</sup><sup>**x**</sup><sup>_i⟩−yi_)2(proofinAppendixH.2).Let</sup> _λ_ 1 _, . . . , λK ≥_ 0 be _K_ fixed regularization strengths. 

**Theorem 16** (Ridge regression with in-context regularization selection) **.** _There exists a transformer with O_ (log(1 _/ε_ )) _layers and O_ ( _K_ ) _heads such that the following holds: On any_ ( _D,_ **x** _N_ +1) _well-conditioned (cf. (5))_ � � _for all {λk}k∈_ [ _K_ ] _, it outputs yN_ +1 = _⟨_ **w** _,_ **x** _N_ +1 _⟩, where_ 



> _Above,_ **w** � ridge<sup>_λ_</sup> _,_ train<sup>_denotes the solution to (ICRidge) on the training split D_train</sup><sup>_,and γ′_:= 2(</sup><sup>_BxBw_+</sup><sup>_By_)</sup><sup>_Bxε_+</sup> _γ, where Bx, Bw, By are the bounds in the well-conditioned assumption (5)._ 

12 

#### **4.1.1 Nearly Bayes-optimal ICL on noisy linear models with mixed noise levels** 

We build on Theorem 16 to show that transformers can perform nearly Bayes-optimal ICL when data come from noisy linear models with a _mixture of K different noise levels σ_ 1 _, . . . , σK >_ 0. 

Concretely, consider the following data generating model, where we first sample P = P **w** _⋆,σk ∼ π_ from iid _k ∼_ Λ _∈_ ∆([ _K_ ]), **w** _⋆ ∼_ N( **0** _,_ **I** _d/d_ ), and then sample data _{_ ( **x** _i, yi_ ) _}i∈_ [ _N_ +1] _∼_ P **w** _⋆,σk_ as 



For any fixed ( _N, d_ ), consider the Bayes risk for predicting _yN_ +1 under this model: 



By standard Bayesian calculations, the above Bayes risk is attained when _A_ is a certain _mixture of K ridge regressions_ with regularization _λk_ = _dσk_<sup>2</sup><sup>_/N_; however, the mixing weights depend on</sup><sup>_D_in a highly non-trivial</sup> fashion (see Appendix I.2 for a derivation). By using the post-ICL validation mechanism in Theorem 16, we construct a transformer that achieves nearly the Bayes risk. 

**Theorem 17** (Nearly Bayes-optimal ICL; Informal version of Theorem I.1) **.** _For sufficiently large N, d, there exists a transformer with O_ (log _N_ ) _layers and O_ ( _K_ ) _heads such that on the above model, it outputs a prediction y_ � _N_ +1 _that is nearly Bayes-optimal:_ 



In particular, Theorem 17 applies in the _proportional setting_ where _N, d_ are large and _N/d_ = Θ(1) [22], in which case BayesRisk _π_ = Θ(1), and thus the transformer achieves vanishing excess risk relative to the Bayes risk at large _N_ . This substantially strengthens the results of Akyürek et al. [2], who empirically find that transformers can achieve nearly Bayes risk under any _fixed_ noise level. By contrast, Theorem 17 shows that a _single_ transformer can achieve nearly Bayes risk even under a mixture of _K_ noise levels, with quantitative guarantees. Also, our proof in fact gives a stronger guarantee: The transformer approaches the _individual Bayes risks on all K noise levels simultaneously_ (in addition to the overall Bayes risk for _k ∼_ Λ as in Theorem 17). We demonstrate this empirically in Section 6 (cf. Figure 4b & 2). 

**Exact Bayes predictor vs. Post-ICL validation mechanism** As BayesRisk _π_ is the theoretical lower bound for the risk of any possible ICL algorithm, Theorem 17 implies that our transformer performs similarly as the exact Bayes estimator<sup>3</sup> . Notice that our construction builds on the (generic) post-ICL validation mechanism, rather than a direct attempt of approximating the exact Bayes predictor, whose structure may vary significantly case-by-case. This highlights post-ICL validation as a promising mechanism for approximating the Bayes predictor on broader classes of problems beyond noisy linear models, which we leave as future work. 

**Generalized linear models with adaptive link function selection** As another example of the postICL validation mechanism, we construct a transformer that can learn a generalized linear model with adaptively chosen link function for the particular ICL instance; see Theorem I.2. 

### **4.2 Pre-ICL testing mechanism** 

In our second mechanism, pre-ICL testing, the transformer runs a _distribution testing_ procedure on the input sequence to determine the right ICL algorithm to use. While the test (and thus the mechanism itself) could in principle be general, we focus on cases where the test amounts to computing some simple summary statistics of the input sequence. 

> 3By the Bayes risk decomposition for square loss, (12) implies that E[( _y_ � _N_ +1 _− y_ � _N_<sup>Bayes</sup> +1<sup>)2]</sup><sup>_≤O_((log</sup><sup>_K/N_)1</sup><sup>_/_3).</sup> 

13 

To showcase pre-ICL testing, we consider the toy problem of selecting between in-context regression and in-context classification, by running the following _binary type check_ on the input labels _{yi}i∈_ [ _N_ ]. 



**Lemma 18.** _There exists a single attention layer with 6 heads that implements_ Ψ<sup>binary</sup> _exactly._ 

Using this test, we construct a transformer that performs logistic regression when labels are binary, and linear regression with high probability if the label admits a continuous distribution. 

**Proposition 19** (Adaptive regression or classification; Informal version of Proposition H.4) **.** _There exists a transformer with O_ (log(1 _/ε_ )) _layers such that the following holds: On any D such that yi ∈{_ 0 _,_ 1 _}, it outputs y_ � _N_ +1 _that ε-approximates the prediction of in-context logistic regression._ 

_By contrast, for any distribution_ P _whose marginal distribution of y is not concentrated around {_ 0 _,_ 1 _}, with high probability (over D), y_ � _N_ +1 _ε-approximates the prediction of in-context least squares._ 

The proofs can be found in Appendix H.3. We additionally show that transformers can implement more complex tests such as a _linear correlation test_ , which can be useful in certain scenarios such as “confident linear regression” (predict only when the signal-to-noise ratio is high); see Appendix H.4. 

## **5 Analysis of pretraining** 

Thus far, we have established the existence of transformers for performing various ICL tasks with good in-context statistical performance. We now analyze the sample complexity of pretraining these transformers from a finite number of training ICL instances. 

### **5.1 Generalization guarantee for pretraining** 

**Setup** At pretraining time, each training ICL instance has form **Z** := ( **H** _, yN_ +1), where **H** := **H** ( _D,_ **x** _N_ +1) _∈_ R<sup>_D×_(</sup><sup>_N_+1)</sup> denote the input sequence formatted as in (3). We consider the square loss between the in-context prediction and the ground truth label: 



Above, clip _By_ ( _t_ ) := max _{_ min _{t, By}, −By}_ is the standard clipping operator onto [ _−By, By_ ], and TF<sup>_R_</sup> **_θ_**<sup>the</sup> transformer architecture as in Definition 3 with clipping operators after each layer: let **H**<sup>(0)</sup> = clipR( **H** ), 



The clipping operator is used to control the Lipschitz constant of TF **_θ_** with respect to **_θ_** , and we typically choose a sufficiently large clipping radius R so that it does not modify the behavior of the transformer on any input sequence of our concern. 

We draw ICL instances **Z** := ( **H** _, yN_ +1) = ( _D,_ ( **x** _N_ +1 _, yN_ +1)) from a (meta-)distribution denoted as _π_ , which iid first sample an in-context data distribution P _∼ π_ , then sample iid examples ( **x** _i, yi_ )<sup>_N_</sup> _i_ =1<sup>+1</sup> _∼_ P<sup>_⊗_(</sup><sup>_N_+1)</sup> and form _D_ = _{_ ( **x** _i, yi_ ) _}i∈_ [ _N_ ]. Our pretraining loss is the average ICL loss on _n_ pretraining instances **Z**<sup>(1:</sup><sup>_n_)iid</sup> _∼ π_ , and we consider the corresponding test ICL loss on a new test instance: 



14 

Our pretraining algorithm is to solve a standard constrained empirical risk minimization (ERM) problem over transformers with _L_ layers, _M_ heads, and norm bound _B_ (recall the definition of the _|||·|||_ norm in (2)): 



**Generalization guarantee** By standard uniform concentration analysis via chaining arguments (Proposition A.4; see also [84, Chapter 5] for similar arguments), we have the following excess loss guarantee for (TF-ERM). The proof can be found in Appendix J.2. 

**Theorem 20** (Generalization for pretraining) **.** _With probability at least_ 1 _− ξ (over the pretraining instances {_ **Z**<sup>_j_</sup> _}j∈_ [ _n_ ] _), the solution_ **_θ_**<sup>�</sup> _to (TF-ERM) satisfies_ 



_where ι_ = log(2 + max _{B,_ R _, By}_ ) _is a log factor._ 

### **5.2 Examples of pretraining for in-context regression problems** 

In Theorem 20, the comparator inf **_θ_** _∈_ Θ _L,M,D′ ,B L_ icl( **_θ_** ) is simply the smallest expected ICL loss for ICL instances drawn from _π_ , among all transformers within the norm ball Θ _L,M,D′,B_ . Using our constructions in Section 3 & 4, we show that this comparator loss is small on various (meta-)distribution _π_ ’s, by which we obtain end-to-end guarantees for pretraining transformers with small ICL loss at test time. Here we showcase this argument on several representative regression problems. 

**Linear regression** For any in-context data distribution P, let **w** P<sup>_⋆_:=EP[</sup><sup>**xx**</sup><sup>_⊤_]</sup><sup>_−_1EP[</sup><sup>**x**</sup><sup>_y_]denotethebest</sup> linear predictor for P. We show that with mild choices of _L, M, B_ , the learned transformer can perform in-context linear regression with near-optimal statistical power, in that on the sampled P _∼ π_ and ICL iid instance _{_ ( **x** _i, yi_ ) _}i∈_ [ _N_ +1] _∼_ P, it competes with the best linear predictor **w** P<sup>_⋆forthisparticular_P.The proof</sup> follows directly by on combining Corollary 5 with Theorem 20, and can be found in Appendix J.3. **Theorem 21** (Pretraining transformers for in-context linear regression) **.** _Suppose_ P _∼ π is almost surely_ well-posed� for in-context linear regression _(Assumption A) with the canonical parameters. Then, for N ≥ O_ ( _d_ ) _, with probability at least_ 1 _− ξ (over the training instances_ **Z**<sup>(1:</sup><sup>_n_)</sup> _), the solution_ **_θ_**<sup>�</sup> _of (TF-ERM) with L_ = _O_ ( _κ_ log( _κN/σ_ )) _layers, M_ = 3 _heads, D_<sup>_′_</sup> = 0 _(attention-only), and B_ = _O_ ( _√κd_ ) _achieves small excess ICL risk over_ **w** P<sup>_⋆:_</sup> 



_where O_<sup>�</sup> ( _·_ ) _only hides polylogarithmic factors in κ, N,_ 1 _/σ._ 

To our best knowledge, Theorem 21 offers the first end-to-end result for pretraining a transformer to perform in-context linear regression with explicit excess loss bounds. The _O_<sup>�</sup> (� _κ_<sup>2</sup> _d_<sup>2</sup> _/n_ ) term originates from the generalization of pretraining (Theorem 20), where as the _O_<sup>�</sup> ( _dσ_<sup>2</sup> _/N_ ) term agrees with the standard fast rate for the excess loss of linear regression [37]. Further, as long as _n ≥ O_<sup>�</sup> ( _κ_<sup>2</sup> _N/σ_<sup>2</sup> ), the excess risk achieves the optimal rate _O_<sup>�</sup> ( _dσ_<sup>2</sup> _/N_ ) (up to log factors). 

**Additional examples** By similar arguments as in the proof of Theorem 21, we can directly turn most of our other expressivity results into results on the pretrained transformers. Here we present three such additional examples (proofs in Appendix J.4-J.6). The first example is for the sparse linear regression problem considered in Theorem 11. 

15 

**Theorem 22** (Pretraining transformers for in-context sparse linear regression) **.** _Suppose each_ P _∼ π is almost surely an instance of the sparse linear model specified in Theorem 11 with parameters Bw_<sup>_⋆andσ._</sup> _Suppose N ≥ O_<sup>�</sup> ( _s_ log(( _d ∨ N_ ) _/σ_ )) _and let κ_ := _Bw_<sup>_⋆/σ._</sup> 

_Then with probability at least_ 1 _− ξ (over the training instances_ **Z**<sup>(1:</sup><sup>_n_)</sup> _), the solution_ **_θ_**<sup>�</sup> _of (TF-ERM) with L_ = _O_<sup>�</sup> ( _κ_<sup>2</sup> (1 + _d/N_ )) _layers, M_ = 2 _heads, D_<sup>_′_</sup> = 2 _d, and B_ = _O_<sup>�</sup> (poly( _d, Bw_<sup>_⋆, σ_))</sup><sup>_achievessmallexcessICL_</sup> _risk:_ 



_where O_<sup>�</sup> ( _·_ ) _only hides polylogarithmic factors in d, N,_ 1 _/σ._ 

Our next example is for the problem of noisy linear regression with mixed noise levels considered in Theorem 17 and Theorem I.1. There, the constructed transformer uses the post-ICL validation mechanism to perform ridge regression with an adaptive regulariation strength depending on the particular input sequence. 

**Theorem 23** (Pretraining transformers for in-context noisy linear regression with algorithm selection) **.** _Suppose π is the data generating model (noisy linear model with mixed noise levels) considered in Theorem I.1, with σ_ max _≤O_ (1) _. Let N ≥ d/_ 10 _._ 

_Then, with probability at least_ 1 _− ξ (over the training instances_ **Z**<sup>(1:</sup><sup>_n_)</sup> _), the solution_ **_θ_**<sup>�</sup> _of (TF-ERM) with input dimension D_ = Θ( _dK_ ) _, L_ = _O_ ( _σ_ min<sup>_−_2log(</sup><sup>_N/σ_min))</sup><sup>_layers,M_=</sup><sup>_O_(</sup><sup>_K_)</sup><sup>_heads,D′_=</sup><sup>_O_(</sup><sup>_K_2)</sup><sup>_,and_</sup> _B_ = _O_ (poly( _K, σ_ min<sup>_−_1</sup><sup>_, d, N_))</sup><sup>_achievessmallexcessICLrisk:_</sup> 



_where O_<sup>�</sup> ( _·_ ) _only hides polylogarithmic factors in d, N, K,_ 1 _/σ_ min _._ Our final example is for in-context logistic regression. For simplicity we consider the realizable case. **Theorem 24** (Pretraining transformers for in-context logistic regression; square loss guarantee) **.** _Suppose for_ P _∼ π,_ P _is almost surely a realizable logistic model (i.e._ P = P<sup>log</sup> **_β_** _with ∥_ **_β_** _∥_ 2 _≤ Bw_<sup>_⋆asinCorollary9)._</sup> _Suppose that Bw_<sup>_⋆_=</sup><sup>_O_(1)</sup><sup>_andN≥O_(</sup><sup>_d_)</sup><sup>_._</sup> 

_Then, with probability at least_ 1 _− ξ (over the training instances_ **Z**<sup>(1:</sup><sup>_n_)</sup> _), the solution_ **_θ_**<sup>�</sup> _of (TF-ERM) with L_ = _O_ (log( _N_ )) _layers, M_ = _O_<sup>�</sup> � _d_<sup>3</sup> _N_ � _heads, D_<sup>_′_</sup> = 0 _, and B_ = _O_ (poly( _d, N_ )) _achieves small excess ICL risk:_ 



_where O_<sup>�</sup> ( _·_ ) _only hides polylogarithmic factors in d, N ._ 

**Remark on generality of transformer** All results above are established by the expressivity results in Section 3 & 4 for transformers to implement various ICL procedures (such as least squares, Lasso, GLM, and ridge regression with in-context algorithm selection), combined with the generalization bound (Theorem 20). However, the transformer itself was not specified to encode any actual structure about the problem at hand in any result above, other than having sufficiently large number of layers, number of heads, and weight norms, which illustrates the flexibility of the transformer architecture. 

## **6 Experiments** 

### **6.1 In-context learning and algorithm selection** 

We test our theory by studying the ICL and in-context algorithm selection capabilities of transformers, using the encoder-based architecture in our theoretical constructions (Definition 3). Additional experimental 

16 



<!-- Start of picture text -->
(a) Base ICL capabilities (b) Noisy reg with two noises (c) Reg + Classification<br>linear_regression TransformerLeast Squares 1.4 TF_alg_select TF _ noise _ 1 0.35 TF TF_reg_ alg _ select<br>noisy_reg_noise_1 Averaging3-NNridge_lam_1 1.2 TF_noise_2ridge_lam_1ridge_lam_2 0.30 TF_cls Least Squares Averaging<br>noisy_reg_noise_2 ridge_lam_2 ridge analytical 3-NN<br>Lasso_lam=1 1.0 Bayes_err_noise_1 0.25<br>Lasso_lam=0.1 Bayes_err_noise_2<br>sparse_reg Lasso_lam=0.01Lasso_lam=0.001 0.8 0.20<br>linear_classification Logistic Regression 0.6<br>0.00 0.25 0.50 0.75 1.00 1.25 1.50 0.10 0.15 0.20 0.25 0.30 0.0 0.5 1.0 1.5<br>Loss noisy_reg_noise_1 regression_square_loss<br>noisy_reg_noise_2 classification_error<br><!-- End of picture text -->

Figure 4: ICL capabilities of the transformer architecture used in our theoretical constructions. _(a)_ On five representative base tasks, transformers approximately match the best baseline algorithm for each task, when pretrained on the corresponding task. _(b,c)_ A **single transformer** `TF_alg_select` **simultaneously approaches the performance of the strongest baseline algorithm** on two separate tasks: _(b)_ noisy linear regression with two different noise levels _σ ∈{_ 0 _._ 1 _,_ 0 _._ 5 _}_ , and _(c)_ adaptively selecting between regression and classification. 

details can be found in Appendix K.1. 

**Training data distributions and evaluation** We train a 12-layer transformer, with two modes for the training sequence (instance) distribution _π_ . In the “base” mode, similar to [31, 2, 83, 46], we sample the training instances from _one_ of the following base distributions (tasks), where we first sample P = P **w** _⋆ ∼ π_ iid iid by sampling **w** _⋆ ∼_ N( **0** _,_ **I** _d/d_ ), and then sample _{_ ( **x** _i, yi_ ) _}i∈_ [ _N_ +1] _∼_ P **w** _⋆_ as **x** _i ∼_ N( **0** _,_ **I** _d_ ), and _yi_ from one of the following models studied in Section 3: 

1. Linear model: _yi_ = _⟨_ **w** _⋆,_ **x** _i⟩_ ; 

2. Noisy linear model: _yi_ = _⟨_ **w** _⋆,_ **x** _i⟩_ + _σzi_ , where _σ >_ 0 is a fixed noise level, and _zi ∼_ N(0 _,_ 1). 

3. Sparse linear model: _yi_ = _⟨_ **w** _⋆,_ **x** _i⟩_ with _∥_ **w** _⋆∥_ 0 _≤ s_ , where _s < d_ is a fixed sparsity level, and in this case we sample **w** _⋆_ from a special prior supported on _s_ -sparse vectors; 

4. Linear classification model: _yi_ = sign( _⟨_ **w** _⋆,_ **x** _i⟩_ ). 

These base tasks have been empirically investigated by Garg et al. [31], though we remark that our architecture (used in our theory) differs from theirs in several aspects, such as encoder-based architecture instead of decoder-based, and ReLU activation instead of softmax. All experiments use _d_ = 20. We choose _σ ∈{σ_ 1 _, σ_ 2 _}_ = _{_ 0 _._ 1 _,_ 0 _._ 5 _}_ and _N_ = 20 for noisy linear regression, _s_ = 3 and _N_ = 10 for sparse linear regression, and _N_ = 40 for linear regression and linear classification. 

In the “mixture” mode, _π_ is the uniform _mixture of two or more base distributions_ . We consider two representative mixture modes studied in Section 4: 

- Linear model + linear classification model; 

- Noisy linear model with four noise levels _σ ∈{_ 0 _._ 1 _,_ 0 _._ 25 _,_ 0 _._ 5 _,_ 1 _}_ . 

Transformers trained with the mixture mode will be evaluated on _multiple_ base distributions simultaneously. When the base distributions are sufficiently diverse, a transformer performing well on all of them will _likely_ be performing some level of in-context algorithm selection. We evaluate transformers against standard machine learning algorithms in context (for each task respectively) as baselines. 

**Results** Figure 4a shows the ICL performance of transformers on five base tasks, within each the transformer is trained on the same task. Transformers match the best baseline algorithm in four out of the five cases, except for the sparse regression task where the Transformer still outperforms least squares and matches Lasso with some choices of _λ_ (thus utilizing sparsity to some extent). This demonstrates the strong ICL capability of the transformer architecture considered in our theory. 

Figure 4b & 4c examine the in-context algorithm selection capability of transformers, on noisy linear regression with two different noise levels (Figure 4b), and regression + classification (Figure 4c). In both figures, the transformer trained in the mixture mode ( `TF_alg_select` ) approaches the best baseline algorithm on 

17 



<!-- Start of picture text -->
(a) Linear regression (b) Linear classification (c) Reg vs. cls at token 40<br>0.40<br>1.0 TF_alg_select TF_alg_select 0.35 TF _ alg _ select<br>0.8 TF_reg TF _ cls 0.35 TFTF_cls_reg TF_regTF_cls<br>Least Squares Logistic Regression 0.30 Least Squares<br>0.6 0.30 Averaging<br>3-NN<br>0.4 0.25 0.25<br>0.2<br>0.20 0.20<br>0.0<br>0.15<br>0 10 20 30 40 0 10 20 30 40 0 1 2 3 4<br>in-context examples in-context examples regression_square_loss<br>error<br>square loss<br>classification_error<br><!-- End of picture text -->

Figure 5: In-context algorithm selection abilities of transformers between linear regression and linear classification. _(a,b)_ On these two tasks, a **single transformer** `TF_alg_select` **simultaneously approaches the performance of the strongest baseline algorithm** `Least Squares` on linear regression and `Logistic Regression` on linear classification. _(c)_ At token 40 (using example _{_ 0 _, . . . ,_ 39 _}_ for training), `TF_alg_select` matches the performance of the best baseline algorithm for both tasks. _(a,b,c)_ Note that transformers pretrained on a single task ( `TF_reg` , `TF_cls` ) perform near-optimally on their pretraining task but suboptimally on the other task. 

both tasks simultaneously. By contrast, transformers trained in the base mode for one of the tasks perform well on that task but behave suboptimally on the other task as expected. The existence of `TF_alg_select` showcases a single transformer that performs well on multiple tasks simultaneously (and thus has to perform in-context algorithm selection to some extent), supporting our theoretical results in Section 4. 

### **6.2 Decoder-based architecture & details for Figure 2** 

ICL capabilities have also been demonstrated in the literature for decoder-based architectures [31, 2, 46]. There, the transformer can do in-context predictions at every token **x** _i_ using past tokens _{_ ( **x** _j,_ **y** _j_ ) _}j≤i−_ 1 as training examples. Here we show that such architectures is also able to perform in-context algorithm selection _at every token_ ; For results for this architecture on “base” ICL tasks (such as those considered in Figure 4a), we refer the readers to Garg et al. [31]. 

**Setup** Our setup is the same as the two “mixture” modes (linear model + linear classification model, and noisy linear models with two different noise levels) as in Section 6.1, except that the architecture is GPT-2 following Garg et al. [31], and the input format is changed to (15) (so that the input sequence has 2 _N_ + 1 tokens) without positional encodings. For every _i ∈_ [ _N_ + 1], we extract the prediction _y_ � _i_ using a linear read-out function applied on output token 2 _i −_ 1, and the (learnable) linear read-out function is the same across all tokens, similar as in Section 6.1. The rest of the setup (optimization, training, and evaluation) is the same as in Section 6.1 & K.1. Note that we also train on the objective (48) for all tokens averaged, instead of for the last test token as in Section 6.1. 

**Result** Figure 2 shows the results for noisy linear models with two different noise levels, and Figure 5 shows the results for linear model + linear classification model. We observe that at every token, In both cases, `TF_alg_select` nearly matches the strongest baseline for both tasks simultaneously, whereas transformers trained on a single task perform suboptimally on the other task. Further, this phenomenon consistently shows up at every token. For example, in Figure 2a & 2b, `TF_alg_select` matches ridge regression with the optimal _λ_ on all tokens _i ∈{_ 1 _, . . . , N }_ ( _N_ = 40). In Figure 5a & 5b, `TF_alg_select` matches least squares on the regression task and logistic regression on the classification task on all tokens _i ∈_ [ _N_ ]. This demonstrates the in-context algorithm selection capabilities of standard decoder-based transformer architectures. 

## **7 Conclusion** 

This work shows that transformers can perform complex in-context learning procedures with strong incontext algorithm selection capabilties, by both explicit theoretical constructions and experiments. We 

18 

believe our work opens up many exciting directions, such as (1) more mechanisms for in-context algorithm selection; (2) Bayes-optimal ICL on other problems by either the post-ICL validation mechanism or new approaches; (3) understanding the internal workings of transformers performing in-context algorithm selection; (4) other mechanisms for implementing complex ICL procedures beyond in-context algorithm selection; (5) further statistical analyses, e.g. of pretraining. 

## **Acknowledgment** 

The authors would like to thank Tengyu Ma and Jason D. Lee for the many insightful discussions. S. Mei is supported in part by NSF DMS-2210827 and NSF CCF-2315725. 

## **References** 

- [1] A. Agarwal, S. Negahban, and M. J. Wainwright. Fast global convergence rates of gradient methods for high-dimensional statistical recovery. _Advances in Neural Information Processing Systems_ , 23, 2010. 

- [2] E. Akyürek, D. Schuurmans, J. Andreas, T. Ma, and D. Zhou. What learning algorithm is in-context learning? investigations with linear models. _arXiv preprint arXiv:2211.15661_ , 2022. 

- [3] J. L. Ba, J. R. Kiros, and G. E. Hinton. Layer normalization. _arXiv preprint arXiv:1607.06450_ , 2016. 

- [4] F. Bach. Breaking the curse of dimensionality with convex neural networks. _The Journal of Machine Learning Research_ , 18(1):629–681, 2017. 

- [5] Y. Bai, M. Chen, P. Zhou, T. Zhao, J. Lee, S. Kakade, H. Wang, and C. Xiong. How important is the train-validation split in meta-learning? In _International Conference on Machine Learning_ , pages 543–553. PMLR, 2021. 

- [6] Y. Bai, S. Mei, H. Wang, and C. Xiong. Don’t just blame over-parametrization for over-confidence: Theoretical analysis of calibration in binary classification. In _International Conference on Machine Learning_ , pages 566–576. PMLR, 2021. 

- [7] J. Baxter. A model of inductive bias learning. _Journal of artificial intelligence research_ , 12:149–198, 2000. 

- [8] A. Beck and M. Teboulle. Gradient-based algorithms with applications to signal recovery. _Convex optimization in signal processing and communications_ , pages 42–88, 2009. 

- [9] S. Bengio, Y. Bengio, J. Cloutier, and J. Gescei. On the optimization of a synaptic learning rule. In _Optimality in Biological and Artificial Networks?_ , pages 281–303. Routledge, 2013. 

- [10] S. Bhattamishra, K. Ahuja, and N. Goyal. On the ability and limitations of transformers to recognize formal languages. _arXiv preprint arXiv:2009.11264_ , 2020. 

- [11] S. Bhattamishra, A. Patel, and N. Goyal. On the computational power of transformers and its implications in sequence modeling. _arXiv preprint arXiv:2006.09286_ , 2020. 

- [12] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- [13] S. Bubeck. Convex optimization: Algorithms and complexity. _Foundations and Trends® in Machine Learning_ , 8(3-4):231–357, 2015. 

- [14] S. Bubeck, V. Chandrasekaran, R. Eldan, J. Gehrke, E. Horvitz, E. Kamar, P. Lee, Y. T. Lee, Y. Li, S. Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. _arXiv preprint arXiv:2303.12712_ , 2023. 

19 

- [15] S. Chan, A. Santoro, A. Lampinen, J. Wang, A. Singh, P. Richemond, J. McClelland, and F. Hill. Data distributional properties drive emergent in-context learning in transformers. _Advances in Neural Information Processing Systems_ , 35:18878–18891, 2022. 

- [16] L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch. Decision transformer: Reinforcement learning via sequence modeling. _Advances in neural information processing systems_ , 34:15084–15097, 2021. 

- [17] K. Chua, Q. Lei, and J. D. Lee. How fine-tuning allows for effective meta-learning. _Advances in Neural Information Processing Systems_ , 34:8871–8884, 2021. 

- [18] D. Dai, Y. Sun, L. Dong, Y. Hao, Z. Sui, and F. Wei. Why can gpt learn in-context? language models secretly perform gradient descent as meta optimizers. _arXiv preprint arXiv:2212.10559_ , 2022. 

- [19] G. Denevi, C. Ciliberto, D. Stamos, and M. Pontil. Incremental learning-to-learn with statistical guarantees. _arXiv preprint arXiv:1803.08089_ , 2018. 

- [20] G. Denevi, C. Ciliberto, D. Stamos, and M. Pontil. Learning to learn around a common mean. _Advances in Neural Information Processing Systems_ , 31, 2018. 

- [21] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ , 2018. 

- [22] E. Dobriban and S. Wager. High-dimensional asymptotics of prediction: Ridge regression and classification. _The Annals of Statistics_ , 46(1):247–279, 2018. 

- [23] L. Dong, S. Xu, and B. Xu. Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition. In _2018 IEEE international conference on acoustics, speech and signal processing (ICASSP)_ , pages 5884–5888. IEEE, 2018. 

- [24] Q. Dong, L. Li, D. Dai, C. Zheng, Z. Wu, B. Chang, X. Sun, J. Xu, and Z. Sui. A survey for in-context learning. _arXiv preprint arXiv:2301.00234_ , 2022. 

- [25] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_ , 2020. 

- [26] S. S. Du, W. Hu, S. M. Kakade, J. D. Lee, and Q. Lei. Few-shot learning via learning the representation, provably. _arXiv preprint arXiv:2002.09434_ , 2020. 

- [27] B. L. Edelman, S. Goel, S. Kakade, and C. Zhang. Inductive biases and variable creation in self-attention mechanisms. In _International Conference on Machine Learning_ , pages 5793–5831. PMLR, 2022. 

- [28] N. Elhage, N. Nanda, C. Olsson, T. Henighan, N. Joseph, B. Mann, A. Askell, Y. Bai, A. Chen, T. Conerly, et al. A mathematical framework for transformer circuits. _Transformer Circuits Thread_ , 2021. 

- [29] C. Finn, P. Abbeel, and S. Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In _International conference on machine learning_ , pages 1126–1135. PMLR, 2017. 

- [30] C. Finn, A. Rajeswaran, S. Kakade, and S. Levine. Online meta-learning. In _International Conference on Machine Learning_ , pages 1920–1930. PMLR, 2019. 

- [31] S. Garg, D. Tsipras, P. S. Liang, and G. Valiant. What can transformers learn in-context? a case study of simple function classes. _Advances in Neural Information Processing Systems_ , 35:30583–30598, 2022. 

- [32] A. Giannou, S. Rajput, J.-y. Sohn, K. Lee, J. D. Lee, and D. Papailiopoulos. Looped transformers as programmable computers. _arXiv preprint arXiv:2301.13196_ , 2023. 

- [33] M. Hahn. Theoretical limitations of self-attention in neural sequence models. _Transactions of the Association for Computational Linguistics_ , 8:156–171, 2020. 

20 

- [34] S. Hochreiter, A. S. Younger, and P. R. Conwell. Learning to learn using gradient descent. In _Artificial Neural Networks—ICANN 2001: International Conference Vienna, Austria, August 21–25, 2001 Proceedings 11_ , pages 87–94. Springer, 2001. 

- [35] T. Hospedales, A. Antoniou, P. Micaelli, and A. Storkey. Meta-learning in neural networks: A survey. _IEEE transactions on pattern analysis and machine intelligence_ , 44(9):5149–5169, 2021. 

- [36] J. Hron, Y. Bahri, J. Sohl-Dickstein, and R. Novak. Infinite attention: Nngp and ntk for deep attention networks. In _International Conference on Machine Learning_ , pages 4376–4386. PMLR, 2020. 

- [37] D. Hsu, S. M. Kakade, and T. Zhang. Random design analysis of ridge regression. In _Conference on learning theory_ , pages 9–1. JMLR Workshop and Conference Proceedings, 2012. 

- [38] S. Jelassi, M. E. Sander, and Y. Li. Vision transformers provably learn spatial structure. _arXiv preprint arXiv:2210.09221_ , 2022. 

- [39] K. Ji, J. D. Lee, Y. Liang, and H. V. Poor. Convergence of meta-learning with task-specific adaptation over partial parameters. _Advances in Neural Information Processing Systems_ , 33:11490–11500, 2020. 

- [40] J. Jumper, R. Evans, A. Pritzel, T. Green, M. Figurnov, O. Ronneberger, K. Tunyasuvunakool, R. Bates, A. **v** Zídek, A. Potapenko, et al. Highly accurate protein structure prediction with alphafold. _Nature_ , 596(7873):583–589, 2021. 

- [41] S. M. Kakade, V. Kanade, O. Shamir, and A. Kalai. Efficient learning of generalized linear and single index models with isotonic regression. _Advances in Neural Information Processing Systems_ , 24, 2011. 

- [42] M. Khodak, M.-F. F. Balcan, and A. S. Talwalkar. Adaptive gradient-based meta-learning methods. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- [43] L. Kirsch and J. Schmidhuber. Meta learning backpropagation and improving it. _Advances in Neural Information Processing Systems_ , 34:14122–14134, 2021. 

- [44] L. Kirsch, J. Harrison, J. Sohl-Dickstein, and L. Metz. General-purpose in-context learning by metalearning transformers. _arXiv preprint arXiv:2212.04458_ , 2022. 

- [45] K. Li and J. Malik. Learning to optimize. _arXiv preprint arXiv:1606.01885_ , 2016. 

- [46] Y. Li, M. E. Ildiz, D. Papailiopoulos, and S. Oymak. Transformers as algorithms: Generalization and implicit model selection in in-context learning. _arXiv preprint arXiv:2301.07067_ , 2023. 

- [47] B. Liu, J. T. Ash, S. Goel, A. Krishnamurthy, and C. Zhang. Transformers learn shortcuts to automata. _arXiv preprint arXiv:2210.10749_ , 2022. 

- [48] J. Liu, D. Shen, Y. Zhang, B. Dolan, L. Carin, and W. Chen. What makes good in-context examples for gpt-3? _arXiv preprint arXiv:2101.06804_ , 2021. 

- [49] Y. Lu, M. Bartolo, A. Moore, S. Riedel, and P. Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. _arXiv preprint arXiv:2104.08786_ , 2021. 

- [50] A. Madani, B. McCann, N. Naik, N. S. Keskar, N. Anand, R. R. Eguchi, P.-S. Huang, and R. Socher. Progen: Language modeling for protein generation. _arXiv preprint arXiv:2004.03497_ , 2020. 

- [51] A. Maurer, M. Pontil, and B. Romera-Paredes. The benefit of multitask representation learning. _Journal of Machine Learning Research_ , 17(81):1–32, 2016. 

- [52] P. McCullagh. _Generalized linear models_ . Routledge, 2019. 

- [53] S. Mei, Y. Bai, and A. Montanari. The landscape of empirical risk for nonconvex losses. _The Annals of Statistics_ , 46(6A):2747–2774, 2018. 

- [54] S. Min, M. Lewis, H. Hajishirzi, and L. Zettlemoyer. Noisy channel language model prompting for few-shot text classification. _arXiv preprint arXiv:2108.04106_ , 2021. 

21 

- [55] S. Min, M. Lewis, L. Zettlemoyer, and H. Hajishirzi. Metaicl: Learning to learn in context. _arXiv preprint arXiv:2110.15943_ , 2021. 

- [56] S. Min, X. Lyu, A. Holtzman, M. Artetxe, M. Lewis, H. Hajishirzi, and L. Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? _arXiv preprint arXiv:2202.12837_ , 2022. 

- [57] N. Mishra, M. Rohaninejad, X. Chen, and P. Abbeel. A simple neural attentive meta-learner. _arXiv preprint arXiv:1707.03141_ , 2017. 

- [58] D. K. Naik and R. J. Mammone. Meta-neural networks that learn by learning. In _[Proceedings 1992] IJCNN International Joint Conference on Neural Networks_ , volume 1, pages 437–442. IEEE, 1992. 

- [59] S. N. Negahban, P. Ravikumar, M. J. Wainwright, and B. Yu. A unified framework for high-dimensional analysis of m-estimators with decomposable regularizers. 2012. 

- [60] Y. Nesterov. _Lectures on convex optimization_ , volume 137. Springer, 2018. 

- [61] C. Olsson, N. Elhage, N. Nanda, N. Joseph, N. DasSarma, T. Henighan, B. Mann, A. Askell, Y. Bai, A. Chen, et al. In-context learning and induction heads. _arXiv preprint arXiv:2209.11895_ , 2022. 

- [62] OpenAI. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- [63] N. Parikh, S. Boyd, et al. Proximal algorithms. _Foundations and trends® in Optimization_ , 1(3): 127–239, 2014. 

- [64] J. Pérez, J. Marinković, and P. Barceló. On the turing completeness of modern neural network architectures. _arXiv preprint arXiv:1901.03429_ , 2019. 

- [65] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, et al. Improving language understanding by generative pre-training. 2018. 

- [66] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 

- [67] A. Raventos, M. Paul, F. Chen, and S. Ganguli. The effects of pretraining task diversity on in-context learning of ridge regression. In _ICLR 2023 Workshop on Mathematical and Empirical Understanding of Foundation Models_ , 2023. 

- [68] S. Ravi and H. Larochelle. Optimization as a model for few-shot learning. In _International conference on learning representations_ , 2017. 

- [69] Y. Razeghi, R. L. Logan IV, M. Gardner, and S. Singh. Impact of pretraining term frequencies on few-shot reasoning. _arXiv preprint arXiv:2202.07206_ , 2022. 

- [70] S. Reed, K. Zolna, E. Parisotto, S. G. Colmenarejo, A. Novikov, G. Barth-Maron, M. Gimenez, Y. Sulsky, J. Kay, J. T. Springenberg, et al. A generalist agent. _arXiv preprint arXiv:2205.06175_ , 2022. 

- [71] O. Rubin, J. Herzig, and J. Berant. Learning to retrieve prompts for in-context learning. _arXiv preprint arXiv:2112.08633_ , 2021. 

- [72] A. Santoro, S. Bartunov, M. Botvinick, D. Wierstra, and T. Lillicrap. Meta-learning with memoryaugmented neural networks. In _International conference on machine learning_ , pages 1842–1850. PMLR, 2016. 

- [73] N. Saunshi, A. Gupta, and W. Hu. A representation learning perspective on the importance of trainvalidation splitting in meta-learning. In _International Conference on Machine Learning_ , pages 9333– 9343. PMLR, 2021. 

- [74] J. Schmidhuber. _Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta-... hook_ . PhD thesis, Technische Universität München, 1987. 

22 

- [75] K. Shen, J. Guo, X. Tan, S. Tang, R. Wang, and J. Bian. A study on relu and softmax in transformer. _arXiv preprint arXiv:2302.06461_ , 2023. 

- [76] C. Snell, R. Zhong, D. Klein, and J. Steinhardt. Approximating how single head attention learns. _arXiv preprint arXiv:2103.07601_ , 2021. 

- [77] J. Snell, K. Swersky, and R. Zemel. Prototypical networks for few-shot learning. _Advances in neural information processing systems_ , 30, 2017. 

- [78] S. Thrun and L. Pratt. _Learning to learn_ . Springer Science & Business Media, 2012. 

- [79] R. Tibshirani. Regression shrinkage and selection via the lasso. _Journal of the Royal Statistical Society: Series B (Methodological)_ , 58(1):267–288, 1996. 

- [80] N. Tripuraneni, M. Jordan, and C. Jin. On the theory of transfer learning: The importance of task diversity. _Advances in neural information processing systems_ , 33:7852–7862, 2020. 

- [81] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- [82] R. Vershynin. _High-dimensional probability: An introduction with applications in data science_ , volume 47. Cambridge university press, 2018. 

- [83] J. von Oswald, E. Niklasson, E. Randazzo, J. Sacramento, A. Mordvintsev, A. Zhmoginov, and M. Vladymyrov. Transformers learn in-context by gradient descent. _arXiv preprint arXiv:2212.07677_ , 2022. 

- [84] M. J. Wainwright. _High-dimensional statistics: A non-asymptotic viewpoint_ , volume 48. Cambridge university press, 2019. 

- [85] X. Wang, S. Yuan, C. Wu, and R. Ge. Guarantees for tuning the step size using a learning-to-learn approach. In _International Conference on Machine Learning_ , pages 10981–10990. PMLR, 2021. 

- [86] C. Wei, Y. Chen, and T. Ma. Statistically meaningful approximation: a case study on approximating turing machines with transformers. _arXiv preprint arXiv:2107.13163_ , 2021. 

- [87] J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, et al. Emergent abilities of large language models. _arXiv preprint arXiv:2206.07682_ , 2022. 

- [88] J. Wei, J. Wei, Y. Tay, D. Tran, A. Webson, Y. Lu, X. Chen, H. Liu, D. Huang, D. Zhou, et al. Larger language models do in-context learning differently. _arXiv preprint arXiv:2303.03846_ , 2023. 

- [89] G. Weiss, Y. Goldberg, and E. Yahav. Thinking like transformers. In _International Conference on Machine Learning_ , pages 11080–11090. PMLR, 2021. 

- [90] S. M. Xie, A. Raghunathan, P. Liang, and T. Ma. An explanation of in-context learning as implicit bayesian inference. _arXiv preprint arXiv:2111.02080_ , 2021. 

- [91] S. Yao, B. Peng, C. Papadimitriou, and K. Narasimhan. Self-attention networks can process bounded hierarchical languages. _arXiv preprint arXiv:2105.11115_ , 2021. 

- [92] C. Ying, T. Cai, S. Luo, S. Zheng, G. Ke, D. He, Y. Shen, and T.-Y. Liu. Do transformers really perform badly for graph representation? _Advances in Neural Information Processing Systems_ , 34:28877–28888, 2021. 

- [93] C. Yun, S. Bhojanapalli, A. S. Rawat, S. J. Reddi, and S. Kumar. Are transformers universal approximators of sequence-to-sequence functions? _arXiv preprint arXiv:1912.10077_ , 2019. 

- [94] Y. Zhang, A. Backurs, S. Bubeck, R. Eldan, S. Gunasekar, and T. Wagner. Unveiling transformers with lego: a synthetic reasoning task. _arXiv preprint arXiv:2206.04301_ , 2022. 

- [95] Y. Zhang, B. Liu, Q. Cai, L. Wang, and Z. Wang. An analysis of attention via the lens of exchangeability and latent variable models. _arXiv preprint arXiv:2212.14852_ , 2022. 

23 

- [96] Z. Zhao, E. Wallace, S. Feng, D. Klein, and S. Singh. Calibrate before use: Improving few-shot performance of language models. In _International Conference on Machine Learning_ , pages 12697–12706. PMLR, 2021. 

- [97] X. Zuo, Z. Chen, H. Yao, Y. Cao, and Q. Gu. Understanding train-validation split in meta-learning with neural networks. In _The Eleventh International Conference on Learning Representations_ , 2023. URL `https://openreview.net/forum?id=JVlyfHEEm0k` . 

|**Cont**|**ents**||
|---|---|---|
|**1**<br>**Intr**|**oduction**|**1**|
|1.1|Related work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>3|
|**2**<br>**Pre**|**liminaries**|**5**|
|2.1|Transformers<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>5|
|2.2|In-context learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>6|
|**3**<br>**Bas**|**ic in-context learning algorithms**|**6**|
|3.1|In-context ridge regression and least squares . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>7|
|3.2|In-context learning of generalized linear models . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>8|
|3.3|In-context Lasso<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>9|
|3.4|Gradient descent on two-layer neural networks<br>. . . . . . . . . . . . . . . . . . . . . . .|. . .<br>10|
|3.5|Mechanism: In-context gradient descent . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>10|
|**4**<br>**In-c**|**ontext algorithm selection**|**12**|
|4.1|Post-ICL validation mechanism . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>12|
||4.1.1<br>Nearly Bayes-optimal ICL on noisy linear models with mixed noise levels<br>. . . .|. . .<br>13|
|4.2|Pre-ICL testing mechanism . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>13|
|**5**<br>**Ana**|**lysis of pretraining**|**14**|
|5.1|Generalization guarantee for pretraining . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>14|
|5.2|Examples of pretraining for in-context regression problems . . . . . . . . . . . . . . . . .|. . .<br>15|
|**6**<br>**Exp**|**eriments**|**16**|
|6.1|In-context learning and algorithm selection<br>. . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>16|
|6.2|Decoder-based architecture & details for Figure 2 . . . . . . . . . . . . . . . . . . . . . .|. . .<br>18|
|**7**<br>**Con**|**clusion**|**18**|
|**A Tec**|**hnical tools**|**26**|
|A.1|Concentration inequalities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>26|
|A.2|Approximation theory . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>26|
|A.3|Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>28|
|A.4|Uniform convergence . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>29|
|A.5|Useful properties of transformers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>31|
|**B Ext**|**ension to decoder-based architecture**|**32**|
|B.1|Decoder-based transformers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>32|
|B.2|In-context learning with decoder-based transformers . . . . . . . . . . . . . . . . . . . .|. . .<br>33|
|B.3|Results. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>33|
|B.4|Proof of Proposition B.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>34|
|**C Pro**|**ofs for Section 3.5 and additional results**|**36**|
|C.1|Proximal gradient descent for regularized convex losses . . . . . . . . . . . . . . . . . . .|. . .<br>36|
|C.2|Approximating a single GD step<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . .<br>37|



24 

|C.3 <br>|Proof of Theorem 13 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . <br>|. . . . . . .<br>38<br><br>|
|---|---|---|
|C.4|Proof of Lemma 14 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>40|
|C.5|Convex ICGD with _ℓ_2 regularization . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>40|
|**D Pro**|**ofs for Section 3.1**|**41**|
|D.1|Proof of Theorem 4. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>41|
|D.2|Statistical analysis of in-context least squares . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>42|
|D.3|Proof of Corollary 5<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>45|
|D.4|Proof of Corollary 6<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>46|
|**E Pro**|**ofs for Section 3.2**|**48**|
|E.1|Proof of Theorem 7. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>48|
|E.2|Proof of Theorem 8. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>49|
|E.3|Proof of Theorem E.1 (a)<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>52|
|E.4|Proof of Theorem E.1 (b) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>53|
|E.5|Proof of Theorem E.1 (c)<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>54|
|E.6|Proof of Theorem E.1 (d) & (e) . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>55|
|**F Pro**|**ofs for Section 3.3**|**55**|
|F.1|Proof of Theorem 10 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>55|
|F.2|Sharper convergence analysis of proximal gradient descent for Lasso<br>. . . . . . .|. . . . . . .<br>56|
|F.3|Basic properties for Lasso . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>57|
|F.4|Proof of Theorem F.1<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>58|
|F.5|Proof of Theorem 11 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>60|
|**G Gra**|**dient descent on two-layer neural networks**|**62**|
|G.1|Proof of Theorem G.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>63|
|G.2|Proof of Lemma G.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>66|
|**H Pro**|**ofs for Section 4**|**67**|
|H.1|Proof of Proposition 15<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>67|
|H.2|Proof of Theorem 16 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>72|
||H.2.1<br>Proof of Theorem H.3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . <br>|. . . . . . .<br>73|
|H.3|Proofs for Section 4.2<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>74|
||H.3.1<br>Proof of Lemma 18 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>74|
||H.3.2<br>Formal statement and proof of Proposition 19 . . . . . . . . . . . . . . . .|. . . . . . .<br>75|
|H.4|Linear correlation test and application . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>77|
|**I**<br>**Pro**|**of of Theorem 17: Noisy linear model with mixed noise levels**|**79**|
|I.1|Proof of Theorem I.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>80|
|I.2|Derivation of the exact Bayes predictor<br>. . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>82|
|I.3|Useful lemmas<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>83|
|I.4|Generalized linear models with adaptive link function selection . . . . . . . . . .|. . . . . . .<br>84|
|**J**<br>**Pro**|**ofs for Section 5**|**86**|
|J.1|Lipschitzness of transformers<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>86|
|J.2|Proof of Theorem 20 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>89|
|J.3|Proof of Theorem 21 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>90|
|J.4|Proof of Theorem 22 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>90|
|J.5|Proof of Theorem 23 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>91|
|J.6|Proof of Theorem 24 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>92|
|**K Exp**|**erimental details**|**92**|
|K.1|Additional details for Section 6.1 . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>92|
|K.2|Computational resource . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>93|



25 

## **A Technical tools** 

**Additional notation for proofs** We say a random variable _X_ is _σ_<sup>2</sup> -sub-Gaussian (or SG( _σ_ ) interchangeably) if E[exp( _X_<sup>2</sup> _/σ_<sup>2</sup> )] _≤_ 2. A random vector **x** _∈_ R<sup>_d_</sup> is _σ_<sup>2</sup> -sub-Gaussian if _⟨_ **v** _,_ **x** _⟩_ is _σ_<sup>2</sup> -sub-Gaussian for all _∥_ **v** _∥_ 2 = 1. A random variable _X_ is _K_ -sub-Exponential (or SE( _K_ ) interchangeably) if E[exp( _|X| /K_ )] _≤_ 2. 

### **A.1 Concentration inequalities** 

**Lemma A.1.** _Let_ **_β_** _∼_ N( **0** _,_ **I** _d/d_ ) _. Then we have_ 



**Lemma A.2** (Theorem 6.1 of [84]) **.** _Let X_ = [ _Xij_ ] _∈_ R<sup>_n×d_</sup> _be a Gaussian random matrix with Xij ∼_ N(0 _,_ 1) _. Let σ_ min( _X_ ) _and σ_ min( _X_ ) _be the minimum and maximum singular value of X, respectively. Then we have_ 



The following lemma is a standard result of covariance concentration, see e.g. [82, Theorem 4.6.1]. 

**Lemma A.3.** _Suppose that_ **x** 1 _, · · · ,_ **x** _N are independent d-dimensional K-sub-Gaussian random vectors. Then as long as N ≥ C_ 0 _d, with probability at least_ 1 _−_ exp( _−N/C_ 0) _we have_ 



_where C_ 0 _is a universal constant._ 

iid iid **Lemma A.4.** _For random matrix_ **X** = [ _xij_ ] _∈_ R<sup>_N×d_</sup> _with xij ∼_ N(0 _,_ 1) _and_ **_ε_** = [ _εi_ ] _∈_ R<sup>_N_</sup> _with εi ∼_ N(0 _, σ_<sup>2</sup> ) _, it holds that_ 



_Proof._ We consider **u** _j_ := [ _xij_ ] _i ∈_ R<sup>_N_</sup> , then �� **X** _⊤_ **_ε_** �� _∞_<sup>= max</sup><sup>_i∈_[</sup><sup>_d_]</sup><sup>_|⟨_</sup><sup>**u**</sup><sup>_j,_</sup><sup>**_ε_**</sup><sup>_⟩|_.Noticethattherandomvariables</sup> _⟨_ **u** 1 _,_ **_ε_** _⟩ , · · · , ⟨_ **u** _d,_ **_ε_** _⟩_ are independent N(0 _, ∥_ **_ε_** _∥_ 2<sup>2),andhence</sup> 



### **A.2 Approximation theory** 

For any signed measure _µ_ over a space _W_ , let TV( _µ_ ) := � _W_<sup>_|dµ_(</sup><sup>**w**)</sup><sup>_| ∈_[0</sup><sup>_, ∞_] denote its total measure.Recall</sup> _σ_ ( _·_ ) = ReLU( _·_ ) is the standard relu activation, and B<sup>_k_</sup> _∞_<sup>(</sup><sup>_R_)=[</sup><sup>_−R, R_]</sup><sup>_k_denotesthestandard</sup><sup>_ℓ∞_ballinR</sup><sup>_k_</sup> with radius _R >_ 0. 

**Definition A.1** (Sufficiently smooth _k_ -variable function) **.** _We say a function g_ : R<sup>_k_</sup> _→_ R _is_ ( _R, Cℓ_ ) _-smooth, if for s_ = _⌈_ ( _k −_ 1) _/_ 2 _⌉_ + 2 _, g is a C_<sup>_s_</sup> _function on_ B<sup>_k_</sup> _∞_<sup>(</sup><sup>_R_)</sup><sup>_,and_</sup> 



_for all i ∈{_ 0 _,_ 1 _, . . . , s}, with_ max0 _≤i≤s LiR_<sup>_i_</sup> _≤ Cℓ._ 

26 

The following result for expressing smooth functions as a random feature model with relu activation is adapted from Bach [4, Proposition 5]. 

**Lemma A.5** (Expressing sufficiently smooth functions by relu random features) **.** _Suppose function g_ : _R_<sup>_k_</sup> _→_ R _is_ ( _R, Cℓ_ ) _smooth. Then there exists a signed measure µ over W_ = _{_ **w** _∈_ R<sup>_k_+1</sup> : _∥_ **w** _∥_ 1 = 1 _} such that_ 



_and_ TV( _µ_ ) _≤ C_ ( _k_ ) _Cℓ, where C_ ( _k_ ) _< ∞ is a constant that only depends on k._ 

**Lemma A.6** (Uniform finite-neuron approximation) **.** _Let X be a space equipped with a distance function dX_ ( _·, ·_ ) : _X × X →_ R _≥_ 0 _. Suppose function g_ : _X →_ R _is given by_ 



_where ϕ_ ( _·_ ; _·_ ) : _X × W →_ [ _−B, B_ ] _is L-Lipschitz (in dX ) in the first argument, and µ is a signed measure over W with finite total measure A_ = TV( _µ_ ) _< ∞. Then for any ε >_ 0 _, there exists α_ 1 _, · · · , αK ∈{±_ 1 _},_ **w** 1 _, · · · ,_ **w** _K ∈W with K_ = _O_ ( _A_<sup>2</sup> _B_<sup>2</sup> log _N_ ( _X , dX ,_ 3 _ALε_<sup>)</sup><sup>_/ε_2)</sup><sup>_,suchthat_</sup> 



_where N_ ( _X , dX ,_ 3 _ALε_<sup>)</sup><sup>_denotesthe_(</sup> 3 _ALε_<sup>)</sup><sup>_-coveringnumberofXindX ._</sup> 

_Proof._ Let _α_ ( **w** ) := sign( _dµ_ ( **w** )) _∈{±_ 1 _}_ denote the sign of the density _dµ_ ( **w** ). We have 



Note that _|dµ_ ( **w** ) _|/A_ is the density of a probability distribution over _W_ . Thus for any **x** _∈X_ , as long as iid _K ≥O_ ( _A_<sup>2</sup> _B_<sup>2</sup> log(1 _/δ_ ) _/ε_<sup>2</sup> ), we can sample **w** 1 _, . . . ,_ **w** _K ∼|dµ_ ( _·_ ) _|/A_ , and obtain by Hoeffding’s inequality that with probability at least 1 _− δ_ , 



Letwe _N_ have( 3 _AL_ with _<u>ε</u>_<sup>) :=</sup> probability<sup>_N_(</sup><sup>_X, dX ,_</sup> at3 _ALε_ least<sup>) for shorthand.</sup> 1 _− δ_ that for<sup>By union bound, as long as</sup> every **x** � in the covering set<sup>_K_</sup> corresponding<sup>_≥O_(</sup><sup>_A_2</sup><sup>_B_2 log(</sup> to<sup>_N_</sup> _N_<sup>(</sup> 3( _AL_ 3 _<u>εALε</u>_<sup>)</sup><sup>_/δ_),)</sup><sup>_/ε_2),</sup> 



Taking _δ_ = 1 _/_ 2 (for which _K_ = _O_ ( _A_<sup>2</sup> _B_<sup>2</sup> log _N_ ( 3 _ALε_<sup>)</sup><sup>_/ε_2)),bytheprobabilisticmethod,thereexistsa</sup> deterministic set _{_ **w** _i}i∈_ [ _K_ ] _⊂W_ and _{αi_ := _α_ ( **w** _i_ ) _}i∈_ [ _K_ ] _∈{±_ 1 _}_ such that the above holds. 

Next, note that bothfor any **x** _∈X_ , taking _g_ (by **x** � to (13)be the) and the functionpoint in the covereing **x** _�→ K_<sup>_<u>A</u>_</sup> �set _Ki_ =1with<sup>_α_(</sup><sup>**w**</sup> _d_<sup>_i_</sup> _X_<sup>)</sup> (<sup>_ϕ_</sup> **x**<sup>(</sup><sup>**x**</sup> _,_ � **x**<sup>;</sup><sup>**w**</sup> ) _≤_<sup>_i_) are</sup> 3 _ALε_<sup>(,</sup><sup>_AL_we)-Lipschitz.haveTherefore,</sup> 



This proves the lemma. 

27 

**Proposition A.1** (Approximating smooth _k_ -variable functions) **.** _For any ε_ approx _>_ 0 _, R ≥_ 1 _, Cℓ >_ 0 _, we have the following: Any_ ( _R, Cℓ_ ) _-smooth function (Definition A.1) g_ : R<sup>_k_</sup> _→_ R _is_ ( _ε_ approx _, R, M, C_ ) _- approximable by sum of relus (Definition 12) with M ≤ C_ ( _k_ ) _Cℓ_<sup>2log(1+</sup><sup>_Cℓ/ε_approx)</sup><sup>_/ε_2</sup> approx<sup>)</sup><sup>_and C≤C_(</sup><sup>_k_)</sup><sup>_Cℓ,_</sup> _where C_ ( _k_ ) _>_ 0 _is a constant that depends only on k. In other words, there exists_ 



_such that_ sup **z** _∈_ [ _−R,R_ ] _k |f_ ( **z** ) _− g_ ( **z** ) _| ≤ ε_ approx _._ 

_Proof._ As function _g_ : B<sup>_k_</sup> _∞_<sup>(</sup><sup>_R_)</sup><sup>_→_Ris(</sup><sup>_R, Cℓ_)-smooth,wecanapplyLemmaA.5toobtainthatthereexists</sup> a signed measure _µ_ over _W_ := _{_ **w** _∈_ R<sup>_k_+1</sup> : _∥_ **w** _∥_ 1 _≤_ 1 _}_ such that 



and _A_ = TV( _µ_ ) _≤ C_ ( _k_ ) _Cℓ_ where _C_ ( _k_ ) _>_ 0 denotes a constant depending only on _k_ . 

We now apply Lemma A.6 to approximate the above random feature by finitely many neurons. Let **x** := [ **z** ; _R_ ] _∈X_ := [ _−R, R_ ]<sup>_k_</sup> _× {R}_ . Then, the function _ϕ_ ( **x** ; **w** ) := _R_ <u>1</u><sup>_σ_(</sup><sup>**w**</sup><sup>_⊤_</sup><sup>**x**)=</sup><sup>_σ_(</sup> _R_<sup><u>1</u></sup><sup>**w**</sup><sup>_⊤_[</sup><sup>**z**;</sup><sup>_R_])isboundedby</sup> _B_ = 1 and (1 _/R_ )-Lipschitz in **x** (in the standard _ℓ∞_ -distance). Further, we have log _N_ ( _X , ∥· −·∥∞ ,_<sup>_ε_</sup> 3<sup>a</sup> _A/R_<sup><u>pprox</u>)</sup><sup>_≤_</sup> _O_ ( _k_ log(1 + _A/ε_ approx)). We can thus apply Lemma A.6 to obtain that, for 



there exists **_α_** = _{αm}m∈_ [ _M_ ] _⊂{±_ 1 _}_ and **W** = _{_ **w** _m}m∈_ [ _M_ ] _⊂W_ = _{_ **w** _∈_ R<sup>_k_+1</sup> : _lone_ **w** = 1 _}_ such that 



where (recalling **z** = [ _s_ ; _t_ ]) 



Note that we have<sup>�</sup><sup>_M_</sup> _m_ =1<sup>_|cm|_=</sup><sup>_A ≤C_(</sup><sup>_k_)</sup><sup>_Cℓ_,and</sup><sup>_∥_</sup><sup>**a**</sup><sup>_m∥_</sup> 1<sup>_≤∥_</sup><sup>**w**</sup><sup>_m∥_</sup> 1<sup>= 1.Thisisthedesiredresult.</sup> 

### **A.3 Optimization** 

The following convergence result for minimizing a smooth and strongly convex function is standard from the convex optimization literature, see e.g. Bubeck [13, Theorem 3.10]. 

**Proposition A.2** (Gradient descent for smooth and strongly convex functions) **.** _Suppose L_ : R<sup>_d_</sup> _→_ R _is α-strongly convex and β-smooth for some_ 0 _< α ≤ β. Then, the gradient descent iterates_ **w** GD<sup>_t_+1:=</sup> **w** GD<sup>_t−η∇L_(</sup><sup>**w**</sup> GD<sup>_t_)</sup><sup>_withlearningrateη_= 1</sup><sup>_/βandinitialization_</sup><sup>**w**</sup> GD<sup>0</sup><sup>_∈_R</sup><sup>_dsatisfiesforanyt ≥_1</sup><sup>_,_</sup> 



_where κ_ := _β/α is the condition number of L, and_ **w**<sup>_⋆_</sup> := arg min **w** _∈_ R _d L_ ( **w** ) _is the minimizer of L._ 

The following convergence result of proximal gradient descent (PGD) on convex composite minimization problem is also standard, see e.g. [8]. 

28 

**Proposition A.3** (Proximal gradient descent for convex function) **.** _Suppose L_ = _f_ + _h, f_ : R<sup>_d_</sup> _→_ R _is convex and β-smooth for some β >_ 0 _, h_ : R<sup>_d_</sup> _→_ R _is a simple convex function. Then, the proximal gradient descent iterates_ **w** PGD<sup>_t_+1:=</sup><sup>**prox**</sup> _ηh_<sup>(</sup><sup>**w**</sup> PGD<sup>_t−η∇f_(</sup><sup>**w**</sup> PGD<sup>_t_))</sup><sup>_withlearningrateη_=1</sup><sup>_/βandinitialization_</sup><sup>**w**</sup> GD<sup>0</sup><sup>_∈_R</sup><sup>_d_</sup> _satisfies the following for any t ≥_ 1 _:_ 

_1. {L_ ( **w** PGD<sup>_t_)</sup><sup>_}isadecreasingsequence._</sup> 

_2. For any minimizer_ **w**<sup>_⋆_</sup> _∈_ arg min **w** _∈_ R _d L_ ( **w** ) _,_ 



_and hence ∥_ **w** PGD<sup>_t−_</sup><sup>**w**</sup><sup>_⋆∥_2</sup> 2 _is also a decreasing sequence._ � � _3. For k ≥_ 1 _, t ≥_ 0 _, it holds that_ 



### **A.4 Uniform convergence** 

The following result is shown in [84, Section 5.6]. 

**Theorem A.1.** _Suppose that ψ_ : [0 _,_ + _∞_ ) _→_ [0 _,_ + _∞_ ) _is a convex, non-decreasing function that satisfies ψ_ ( _x_ + _y_ ) _≥ ψ_ ( _x_ ) _ψ_ ( _y_ ) _. For any random variable X, we consider the Orlicz norm induced by ψ: ∥X∥ψ_ := inf _{K >_ 0 : E _ψ_ ( _|X| /K_ ) _} ≤_ 1 _._ 

_Suppose that {Xθ}θ is a zero-mean random process indexed by θ ∈_ Θ _such that ∥Xθ − Xθ′∥ψ ≤ ρ_ ( _θ, θ_<sup>_′_</sup> ) _for some metric ρ on the space_ Θ _. Then it holds that_ 



_where D is the diameter of the metric space_ (Θ _, ρ_ ) _, and the generalized Dudley entropy integral J is given by_ 



_where N_ ( _δ_ ; Θ _, ρ_ ) _is the δ-covering number of_ (Θ _, ρ_ ) _._ 

As a corollary of Theorem A.1, we have the following result. 

**Proposition A.4** (Uniform concentration bound by chaining) **.** _Suppose that {Xθ}θ∈_ Θ _is a zero-mean random process given by_ 



_where z_ 1 _, · · · , zN are i.i.d samples from a distribution_ P _z such that the following assumption holds:_ 

- _(a) The index set_ Θ _is equipped with a distance ρ and diameter D. Further, assume that for some constant A, for any ball_ Θ<sup>_′_</sup> _of radius r in_ Θ _, the covering number admits upper bound_ log _N_ ( _δ_ ; Θ<sup>_′_</sup> _, ρ_ ) _≤ d_ log(2 _Ar/δ_ ) _for all_ 0 _< δ ≤_ 2 _r._ 

- _(b) For any fixed θ ∈_ Θ _and z sampled from_ P _z, the random variable f_ ( _z_ ; _θ_ ) _is a_ SG( _B_<sup>0</sup> ) _-sub-Gaussian random variable._ 

- _(c) For any θ, θ_<sup>_′_</sup> _∈_ Θ _and z sampled from_ P _z, the random variable f_ ( _z_ ; _θ_ ) _− f_ ( _z_ ; _θ_<sup>_′_</sup> ) _is a_ SG( _B_<sup>1</sup> _ρ_ ( _θ, θ_<sup>_′_</sup> )) _-subGaussian random variable._ 

29 

_Then with probability at least_ 1 _− δ, it holds that_ 



_where C is a universal constant, and we denote κ_ = 1 + _B_<sup>1</sup> _D/B_<sup>0</sup> _._ 

_Furthermore, if we replace the_ SG _in assumption (b) and (c) by_ SE _, then with probability at least_ 1 _− δ, it holds that_ 



_Proof._ Fix a _D_ 0 _∈_ (0 _, D_ ] to be specified later. We pick a ( _D_ 0 _/_ 2)-covering Θ0 of Θ so that log _|_ Θ0 _| ≤ d_ log(2 _AD/D_ 0). Then, by the standard uniform covering of independent sub-Gaussian random variables, we have with probability at least 1 _− δ/_ 2, 



Assume that Θ0 = _{θ_ 1 _, · · · , θn}_ . For each _j ∈_ [ _n_ ], we consider Θ _j_ is the ball centered at _θj_ of radius _D_ 0 in (Θ _, ρ_ ). Then _θ ∈_ Θ _j_ has diameter _D_ 0 and admits covering number bound log _N_ (Θ _j, δ_ ) _≤ d_ log( _AD_ 0 _/δ_ ). Hence, we can apply Theorem A.1 with the process _{Xθ}θ∈_ Θ _j_ , then 



and a simple calculation yields 



Therefore, we can let _t ≤_ �log(2 _n/δ_ ) _/N_ in the above inequality and taking the union bound over _j ∈_ [ _n_ ], and hence with probability at least 1 _− δ/_ 2, it holds that for all _j ∈_ [ _n_ ], 



Notice that for each _θ ∈_ Θ _,_ there exists _j ∈_ [ _n_ ] such that _θ ∈_ Θ _j_ , and hence 



Thus, with probability at least 1 _− δ_ , it holds 



Taking _D_ 0 = _D/κ_ completes the proof of SG case. 

We next consider the SE case. The idea is the same as the SG case, but in this case we need to consider the following Orlicz-norm: 



Then Bernstein’s inequality of SE random variables yields 



30 

for some universal constant _C_ 0. Therefore, we can repeat the argument above to deduce that with probability at least 1 _− δ_ , it holds 



Taking _D_ 0 = _D/κ_ completes the proof. 

### **A.5 Useful properties of transformers** 

The following result can be obtained immediately by “joining” the attention heads and MLP layers of two single-layer transformers. 

**Proposition A.5** (Joining parallel single-layer transformers) **.** _Suppose that P_ 1 : R<sup>(</sup><sup>_D_0+</sup><sup>_D_1)</sup><sup>_×N_</sup> _→_ R<sup>_D_1</sup><sup>_×N_</sup> _, P_ 2 : R<sup>(</sup><sup>_D_0+</sup><sup>_D_2)</sup><sup>_×N_</sup> _→_ R<sup>_D_2</sup><sup>_×N_</sup> _are two sequence-to-sequence functions that are implemented by single-layer transformers, i.e. there exists_ **_θ_** 1 _,_ **_θ_** 2 _such that_ 



_Then, there exists_ **_θ_** _such that for_ **H**<sup>_′_</sup> _that takes form_ **h**<sup>_′_</sup> _i_<sup>=[</sup><sup>**h**(0)</sup> _i_<sup>;</sup><sup>**h**(1)</sup> _i_<sup>;</sup><sup>**h**(2)</sup> _i_<sup>]</sup><sup>_,with_</sup><sup>**h**(0)</sup> _i ∈_ R<sup>_D_0</sup> _,_ **h**<sup>(1)</sup> _i ∈_ R<sup>_D_1</sup> _,_ **h**<sup>(2)</sup> _i ∈_ R<sup>_D_2</sup> _, we have_ 



_Further,_ **_θ_** _has at most M ≤ M_ 1 + _M_ 2 _heads, D_<sup>_′_</sup> _≤ D_ 1<sup>_′_+</sup><sup>_D_</sup> 2<sup>_′hiddendimensioninitsMLPlayer,andnorm_</sup> _bound |||_ **_θ_** _||| ≤|||_ **_θ_** 1 _|||_ + _|||_ **_θ_** 2 _|||._ 

**Proposition A.6** (Joining parallel multi-layer transformers) **.** _Suppose that P_ 1 : R<sup>(</sup><sup>_D_0+</sup><sup>_D_1)</sup><sup>_×N_</sup> _→_ R<sup>_D_1</sup><sup>_×N_</sup> _, P_ 2 : R<sup>(</sup><sup>_D_0+</sup><sup>_D_2)</sup><sup>_×N_</sup> _→_ R<sup>_D_2</sup><sup>_×N_</sup> _are two sequence-to-sequence functions that are implemented by multi-layer transformers, i.e. there exists_ **_θ_** 1 _,_ **_θ_** 2 _such that_ 



_Then, there exists_ **_θ_** _such that for_ **H**<sup>_′_</sup> _that takes form_ **h**<sup>_′_</sup> _i_<sup>=[</sup><sup>**h**(0)</sup> _i_<sup>;</sup><sup>**h**(1)</sup> _i_<sup>;</sup><sup>**h**(2)</sup> _i_<sup>]</sup><sup>_,with_</sup><sup>**h**(0)</sup> _i ∈_ R<sup>_D_0</sup> _,_ **h**<sup>(1)</sup> _i ∈_ R<sup>_D_1</sup> _,_ **h**<sup>(2)</sup> _i ∈_ R<sup>_D_2</sup> _, we have_ 



_Further,_ **_θ_** _has at most L ≤_ max _{L_ 1 _, L_ 2 _} layers,_ max _ℓ∈_ [ _L_ ] _M_<sup>(</sup><sup>_ℓ_)</sup> _≤_ max _ℓ∈_ [ _L_ ] � _M_ 1<sup>(</sup><sup>_ℓ_)</sup> + _M_ 2<sup>(</sup><sup>_ℓ_)</sup> � _heads,_ max _ℓ∈_ [ _L_ ] _D_<sup>(</sup><sup>_ℓ_)</sup> _≤_ max _ℓ∈_ [ _L_ ] � _D_ 1<sup>(</sup><sup>_ℓ_)</sup> + _D_ 2<sup>(</sup><sup>_ℓ_)</sup> � _hidden dimension in its MLP layer (understanding the size of the empty layers as 0), and norm bound |||_ **_θ_** _||| ≤|||_ **_θ_** 1 _|||_ + _|||_ **_θ_** 2 _|||._ 

31 

_Proof._ When _L_ 1 = _L_ 2 ( **_θ_** 1 and **_θ_** 2 have the same number of layers), the result follows directly by applying Proposition A.5 repeatedly for all _L_ 1 layers and the definition of the norm (2). 

If (without loss of generality) _L_ 1 _< L_ 2, we can augment **_θ_** 1 to _L_ 2 layers by adding ( _L_ 2 _− L_ 1) layers with zero attention heads, and zero MLP hidden dimension (note that this does not change _M_ 1, _D_ 1<sup>_′_,and</sup><sup>_|||_</sup><sup>**_θ_**1</sup><sup>_|||_).Due</sup> to the residual structure, the transformer maintains the output _P_ 1( **H** 1) throughout layer _L_ 1 +1 _, . . . , L_ 2, and it reduces to the case _L_ 1 = _L_ 2. 

## **B Extension to decoder-based architecture** 

Here we briefly discuss how our theoretical results can be adapted to decoder-based architectures (henceforth decoder TFs). Adopting the setting as in Section 2, we consider a sequence of _N_ input vectors _{_ **h** _i}_<sup>_N_</sup> _i_ =1<sup>_⊂_R</sup><sup>_D_,</sup> written compactly as an input matrix **H** = [ **h** 1 _, . . . ,_ **h** _N_ ] _∈_ R<sup>_D×N_</sup> . Recall that _σ_ ( _t_ ) := ReLU( _t_ ) = max _{t,_ 0 _}_ denotes the standard relu activation. 

### **B.1 Decoder-based transformers** 

Decoder TFs are the same as encoder TFs, except that the attention layers are replaced by masked attention layers with a specific decoder-based (causal) attention mask. 

**Definition B.1** (Masked attention layer) **.** _A masked attention layer with M heads is denoted as_ MAttn **_θ_** ( _·_ ) _with parameters_ **_θ_** = _{_ ( **V** _m,_ **Q** _m,_ **K** _m_ ) _}m∈_ [ _M_ ] _⊂_ R<sup>_D×D_</sup> _. On any input sequence_ **H** _∈_ R<sup>_D×N′_</sup> _with N_<sup>_′_</sup> _≤ N ,_ 



_where ◦ denotes the entry-wise (Hadamard) product of two matrices, and_ MSK _∈_ R<sup>_N×N_</sup> _is the mask matrix given by_ 

 01 11 _//_ 22 11 _//_ 33 _· · ·· · ·_ 11 _/N/N_  MSK = 0 0 1 _/_ 3 _· · ·_ 1 _/N . · · · · · · · · · · · · · · ·_  0 0 0 _· · ·_ 1 _/N_  

_In vector form, we have_ 



Notice that standard masked attention definitions use the pre-activation additive masks (with mask value _−∞_ ) [81]. The post-activation multiplicative masks we use is equivalent to the pre-activation additive masks, and the modified presentation is for notational convenience. We also use a normalized ReLU activation _t �→ σ_ ( _t_ ) _/i_ in place of the standard softmax activation to be consistent with Definition 1. Note that the normalization 1 _/i_ is to ensure that the attention weights _{σ_ ( _⟨_ **Q** _m_ **h** _i,_ **K** _m_ **h** _j⟩_ ) _/i}j∈_ [ _i_ ] is a set of non-negative weights that sum to _O_ (1). The motivation of masked attention layer is to ensure that, when processing a sequence of tokens, the computations at any token do not see any later token. 

We next define the decoder-based transformers with _L ≥_ 1 transformer layers, each consisting of a masked attention layer (c.f. Definition B.1) followed by an MLP layer (c.f. Definition 2). This definition is similar to the definition of encoder-based transformers (c.f., Definition 3), except that we replace the attention layers by masked attention layers. 

**Definition B.2** (Decoder-based Transformer) **.** _An L-layer decoder-based transformer, denoted as_ DTF **_θ_** ( _·_ ) _, is a composition of L self-attention layers each followed by an MLP layer:_ **H**<sup>(</sup><sup>_L_)</sup> = DTF **_θ_** ( **H**<sup>(0)</sup> ) _, where_ **H**<sup>(0)</sup> _∈_ R<sup>_D×N_</sup> _is the input sequence, and_ 



32 

_Above, the parameter_ **_θ_** = ( **_θ_** `mattn`<sup>(1:</sup><sup>_L_)</sup><sup>_,_</sup><sup>**_θ_**</sup> `mlp`<sup>(1:</sup><sup>_L_))</sup><sup>_istheparameterconsistingoftheattentionlayers_</sup><sup>**_θ_**</sup> `mattn`<sup>(</sup><sup>_ℓ_)=</sup> _{_ ( **V** _m_<sup>(</sup><sup>_ℓ_)</sup><sup>_,_</sup><sup>**Q**(</sup> _m_<sup>_ℓ_)</sup><sup>_,_</sup><sup>**K**(</sup> _m_<sup>_ℓ_))</sup><sup>_}_</sup> _m∈_ [ _M_ ( _ℓ_ )]<sup>_⊂_R</sup><sup>_D×DandtheMLPlayers_</sup><sup>**_θ_**</sup> `mlp`<sup>(</sup><sup>_ℓ_)=(</sup><sup>**W**</sup> 1<sup>(</sup><sup>_ℓ_)</sup><sup>_,_</sup><sup>**W**</sup> 2<sup>(</sup><sup>_ℓ_))</sup><sup>_∈_R</sup><sup>_D_(</sup><sup>_ℓ_)</sup><sup>_×D×_R</sup><sup>_D×D_(</sup><sup>_ℓ_)</sup><sup>_.We_</sup> _will frequently consider “_ attention-only _” decoder-based transformers with_ **W** 1<sup>(</sup><sup>_ℓ_)</sup><sup>_,_</sup><sup>**W**</sup> 2<sup>(</sup><sup>_ℓ_)</sup> = **0** _, which we denote as_ DTF<sup>0</sup> **_θ_**<sup>(</sup><sup>_·_)</sup><sup>_forshorthand,with_</sup><sup>**_θ_**=</sup><sup>**_θ_**(1:</sup><sup>_L_):=</sup><sup>**_θ_**</sup> `mattn`<sup>(1:</sup><sup>_L_)</sup><sup>_._</sup> We also use (2) to define the norm of DTF **_θ_** . 

### **B.2 In-context learning with decoder-based transformers** 

We consider using decoder-based TFs to perform ICL. We encode ( _D,_ **x** _N_ +1), which follows the generating rule as described in Section 2.2, into an input sequence **H** _∈_ R<sup>_D×_(2</sup><sup>_N_+1)</sup> . In our theory, we use the following format, where the first two rows contain ( _D,_ **x** _N_ +1) which alternating between [ **x** _i_ ; 0] _∈_ R<sup>_d_+1</sup> and [ **0** _d×_ 1; _yi_ ] _∈_ R<sup>_d_+1</sup> (the same setup as adopted in [31, 2]); The third row contains fixed vectors _{_ **p** _i}i∈_ [ _N_ +1] with ones, zeros, the example index, and indicator for being the covariate token (similar to a positional encoding vector): 



(15) is different from out input format (3) for encoder-based TFs. The main difference is that ( **x** _i, yi_ ) are in different tokens in (15), whereas ( **x** _i, yi_ ) are in the same token in (3). The reason for the former (i.e., different tokens in decoder) is that we want to avoid every [ **x** _i_ ; 0] token seeing the information of _yi_ , since we will evaluate the loss at every token. The reason for the latter (i.e., the same token in encoder) is for presentation convenience: since we only evaluate the loss at the last token, it is not necessary to alternate between [ **x** _i_ ; 0] and [ **0** ; _yi_ ] to avoid information leakage. 

We then feed **H** into a decoder TF to obtain the output **H**<sup>�</sup> = DTF **_θ_** ( **H** ) _∈_ R<sup>_D×_(2</sup><sup>_N_+1)</sup> with the same � shape, and _read out_ the prediction _yN_ +1 from the ( _d_ + 1 _,_ 2 _N_ + 1)-th entry of **H**<sup>�</sup> = [ **h**<sup>�</sup> _i_ ] _i∈_ [2 _N_ +1] (the entry � � corresponding to the last missing test label): _yN_ +1 = ready( **H**<sup>�</sup> ) := ( **h**<sup>�</sup> 2 _N_ +1) _d_ +1. The goal is to predict _yN_ +1 that is close to _yN_ +1 _∼_ P _y|_ **x** _N_ +1 measured by proper losses. 

The benefit of using the decoder architecture is that, during the pre-training phase, one can construct the � � training loss function by using all the predictions _{yj}j∈_ [ _N_ +1], where _yj_ gives the ( _d_ + 1 _,_ 2 _j −_ 1)-th entry of **H**<sup>�</sup> = [ **h**<sup>�</sup> _i_ ] _i∈_ [2 _N_ +1] for each _j ∈_ [ _N_ + 1] (the entry corresponding to the missing test label of the 2 _j −_ 1’th � token): _yj_ = ready _,j_ ( **H**<sup>�</sup> ) := ( **h**<sup>�</sup> 2 _j−_ 1) _d_ +1. Given a loss function _ℓ_ : R _×_ R _→_ R associated to a single response, the training loss associated to the whole input sequence can be defined by _ℓ_ ( **H** ) =<sup>�</sup><sup>_N_</sup> _j_ =1<sup>+1</sup><sup>_ℓ_(</sup><sup>_yj,_�</sup><sup>_yj_).This</sup> potentially enables less training sequences in the pre-training stage, and some generalization bound analysis justifying this benefit was provided in [46]. 

### **B.3 Results** 

We discuss how our theoretical results upon encoder TFs can be converted to those of the decoder TFs. Taking the implementation of (ICGD) (a key mechanism that enables most basic ICL algorithms such as ridge regression; cf. Section 3.5) as an example, this conversion is enabled by the following facts: (a) the input format (15) of decoders can be converted to the input format (3) of encoders by a 2-layer decoder TF; (b) the encoder TF that implements (ICGD) with input format (3), by a slight parameter modification, can be converted to a decoder TF that implements the (ICGD) algorithm with a converted input format. 

**Input format conversion** Despite the difference between the input format (15) and (3), we show that there exists a 2-layer decoder TF that can convert the input format (15) to format (3). The proof can be found in Appendix B.4. 

**Proposition B.1** (Input format conversion) **.** _There exists a 2-layer decoder TF_ DTF _with_ 3 _heads per layer, hidden dimension_ 2 _and |||_ **_θ_** _||| ≤_ 12 _such that upon taking input_ **H** _of format_ (15) _, it outputs_ **H**<sup>�</sup> = DTF( **H** ) 

33 

_with_ 



_In particular, format (16) contains format (3) as a submatrix, by restricting to the {_ 1 _,_ 2 _, . . . , D −_ 1 _, D −_ 2 _, D} rows and {_ 2 _,_ 4 _, . . . ,_ 2 _N −_ 2 _,_ 2 _N,_ 2 _N_ + 1 _} columns._ 

**Generalization TF constructions to decoder architecture** The construction in Theorem 13 can be generalized to using the input format (16) along with a decoder TF, by using the scratch pad within the last token to record the gradient descent iterates. Further, if we slightly change the normalization in MSK from 1 _/i_ to 1 _/_ (( _i −_ 1) _∨_ 1), then the same construction performs (ICGD) (with training examples _{_ 1 _, . . . , j}_ ) at every token _i_ = 2 _j_ +1 (corresponding to predicting at **x** _j_ +1). Building on this extension, all our constructions in Section 3 and Section 4.2 can be generalized to decoder TFs. 

### **B.4 Proof of Proposition B.1** 

For the simplicity of presentation, we write _ci_ = _⌈i/_ 2 _⌉ , ti_ = mod( _i_ +1 _,_ 2), **u** _i_ = **h** _i_ [1 : _d_ ] _∈_ R<sup>_d_+1</sup> be the vector of first _d_ entries of **h** _i_<sup>4</sup> , and let _vi_ = **h** _i_ [ _d_ + 1] be the ( _d_ + 1)-th entry of **h** _i_ . With such notations, the input sequence **H** = [ **h** _i_ ] _i_ can be compactly written as 



In the following, we construct the desired **_θ_** = ( **_θ_**<sup>(1)</sup> _,_ **_θ_**<sup>(2)</sup> ) as follows. 

**Step 1:** construction of **_θ_**<sup>(1)</sup> = ( **_θ_** `mattn`<sup>(1)</sup><sup>_,_</sup><sup>**_θ_**</sup> `mlp`<sup>(1)),sothatMLP</sup> **_θ_** `mlp`<sup>(1)</sup><sup>_◦_MAttn</sup><sup>**_θ_**</sup> `mattn`<sup>(1)maps</sup> 



For _m ∈{_ 0 _,_ 1 _}_ , we define matrices **Q** _m_<sup>(1)</sup><sup>_,_</sup><sup>**K**(1)</sup> _m_<sup>_,_</sup><sup>**V**</sup> _m_<sup>(1)</sup><sup>_∈_R</sup><sup>_D×D_suchthat</sup> 



for all _i, j_ . By the structure of **h** _i_ , these matrices indeed exist, and further it is straightforward to check that they have norm bounds 



Now, for every _i_ , 



Notice that _ti_ = 0 only when 2 _| i_ , we then compute for _i_ = 2 _k_ that 



> 4In other words, when 2 ∤ _i_ , **u** _i_ = **x** ( _i−_ 1) _/_ 2; when 2 _| i_ , **u** _i_ = **0** _d_ . 

34 

Therefore, the **_θ_** `mattn`<sup>(1)=</sup><sup>_{_(</sup><sup>**Q**</sup> _m_<sup>(1)</sup><sup>_,_</sup><sup>**K**(1)</sup> _m_<sup>_,_</sup><sup>**V**</sup> _m_<sup>(1)</sup><sup>_∈_R</sup><sup>_D×D_)</sup><sup>_}_</sup> _m∈{_ 0 _,_ 1 _}_<sup>weconstructaboveisindeedthedesiredatten-</sup> tion layer. The existence of the desired **_θ_** `mlp`<sup>(1)isclear,and</sup><sup>**_θ_**</sup> `mlp`<sup>(1)= (</sup><sup>**W**</sup> 1<sup>(1)</sup><sup>_,_</sup><sup>**W**</sup> 2<sup>(1))canfurtherbechosensothat</sup> _∥_ **W** 1<sup>(1)</sup><sup>_∥_op</sup><sup>_≤_1</sup><sup>_, ∥_</sup><sup>**W**</sup> 2<sup>(1)</sup><sup>_∥_op</sup><sup>_≤_1.</sup> **Step 2:** construction of **_θ_**<sup>(2)</sup> . For every _m ∈{−_ 1 _,_ 0 _,_ 1 _}_ , we define matrices **Q** _m_<sup>(2)</sup><sup>_,_</sup><sup>**K**(2)</sup> _m_<sup>_,_</sup><sup>**V**</sup> _m_<sup>(2)</sup><sup>_∈_R</sup><sup>_D×D_such</sup> that 



for all _i, j_ . By the structure of **h**<sup>(1)</sup> _i_<sup>,thesematricesindeedexist,andfurtheritisstraightforwardtocheck</sup> that they have norm bounds 



Now, for every _i, j_ , we have 

where the last equality follows from the fact that 



Therefore, 



Therefore, the **_θ_** `mattn`<sup>(2)=</sup><sup>_{_(</sup><sup>**Q**</sup> _m_<sup>(2)</sup><sup>_,_</sup><sup>**K**(2)</sup> _m_<sup>_,_</sup><sup>**V**</sup> _m_<sup>(2)</sup><sup>_∈_R</sup><sup>_D×D_)</sup><sup>_}_</sup> _m∈{−_ 1 _,_ 0 _,_ 1 _}_<sup>weconstructabovemaps</sup> 



Finally, we only need to take a MLP layer **_θ_** `mlp`<sup>(2)= (</sup><sup>**W**</sup> 1<sup>(2)</sup><sup>_,_</sup><sup>**W**</sup> 2<sup>(2))withhiddendimension2thatmaps</sup> 



which clearly exists and can be chosen so that _∥_ **W** 1<sup>(2)</sup><sup>_∥_op</sup><sup>_≤_1</sup><sup>_, ∥_</sup><sup>**W**</sup> 2<sup>(2)</sup><sup>_∥_op</sup><sup>_≤_1.</sup> Combining the two steps above, we complete the proof of Proposition B.1. 

35 

## **C Proofs for Section 3.5 and additional results** 

### **C.1 Proximal gradient descent for regularized convex losses** 

Proximal gradient descent (PGD) is a variant of gradient descent that is suitable for minimizing regularized risks [63], in particular those with a non-smooth regularizer such as the _ℓ_ 1 norm. In this section, we show that transformers can approximate PGD with similar quantitative guarantees as for GD in Section 3.5. 

Let _ℓ_ ( _·, ·_ ) : R<sup>2</sup> _→_ R be a loss function. Let _L_<sup>�</sup> _N_ ( **w** ) := _N_ <u>1</u> � _Ni_ =1<sup>_ℓ_(</sup><sup>**w**</sup><sup>_⊤_</sup><sup>**x**</sup><sup>_i, yi_) +</sup><sup>_R_(</sup><sup>**w**)denotetheregularized</sup> empirical risk with loss function _ℓ_ on dataset _{_ ( **x** _i, yi_ ) _}i∈_ [ _N_ ] and regularizer _R_ . To minimize _L_<sup>�</sup> _N_ , we consider the proximal gradient descent trajectory on _L_<sup>�</sup> _N_ with initialization **w** GD<sup>0=</sup><sup>**0**</sup><sup>_∈_R</sup><sup>_d_andlearningrate</sup><sup>_η>_0:</sup> 



where we denote _L_<sup>�0</sup> _N_<sup>(</sup><sup>**w**) :=</sup> _N_<sup><u>1</u></sup> � _Ni_ =1<sup>_ℓ_(</sup><sup>**w**</sup><sup>_⊤_</sup><sup>**x**</sup><sup>_i, yi_).</sup> 

To approximate (ICPGD) by transformers, in addition to the requirement on the loss _ℓ_ as in Theorem 13, we additionally require the the proximal operator **prox** _ηR_ ( _·_ ) to be approximable by an MLP layer (as a vector-valued analog of Definition 12) defined as follows. 

**Definition C.1** (Approximability by MLP) **.** _An operator P_ : R<sup>_d_</sup> _→_ R<sup>_d_</sup> _is_ ( _ε, R, D, C_ ) _-_ approximable by MLP _, if there exists a there exists a MLP_ **_θ_** `mlp` = ( **W** 1 _,_ **W** 2) _∈_ R<sup>_D×d_</sup> _×_ R<sup>_d×D_</sup> _with hidden dimension D, ∥_ **W** 1 _∥_ op + _∥_ **W** 2 _∥_ op _≤ C_<sup>_′_</sup> _, such that_ sup _∥_ **w** _∥_ 2 _≤R_ �� _P_ ( **w** ) _−_ MLP **_θ_** `mlp` ( **w** )��2<sup>_≤ε._</sup> 

The definition above captures the proximal operator **prox** _ηR_ for a broad class of regularizers, such as the (commonly-used) _L_ 1 and _L_ 2 regularizer listed in the following proposition, for all of which one can directly check that they can be exactly implemented by an MLP as stated below. 

**Proposition C.1** (Proximal operators for commonly-used regularizers) **.** _For regularizer R in {λ ∥·∥_ 1 _,_<sup>_<u>λ</u>_</sup> 2<sup>_∥·∥_</sup> 2<sup>2</sup><sup>_,_IB</sup> _∞_<sup>(</sup><sup>_B_)(</sup><sup>_·_)</sup><sup>_},_</sup> _the operator_ **prox** _ηR_ : R<sup>_d_</sup> _→_ R<sup>_d_</sup> _is_ exactly _approximable by MLP. More concretely, we have_ 

_1. For R_ = _λ ∥·∥_ 1 _,_ **prox** _ηR is_ (0 _,_ + _∞,_ 4 _d,_ 4 + 2 _ηλ_ ) _-approximable by MLP._ 

_2. For R_ =<sup>_<u>λ</u>_</sup> 2<sup>_∥·∥_</sup> 2<sup>2</sup><sup>_,_</sup><sup>**prox**</sup> _ηR_<sup>_is_(0</sup><sup>_,_+</sup><sup>_∞,_2</sup><sup>_d,_2 + 2</sup><sup>_ηλ_)</sup><sup>_-approximablebyMLP._</sup> 

_3. For R_ = IB _∞_ ( _B_ )( _·_ ) _,_ **prox** _ηR_ = ProjB _∞_ ( _B_ ) _is_ (0 _,_ + _∞,_ 2 _d,_ 2 + 2 _B_ ) _-approximable by MLP._ 

- **Theorem C.1** (Convex ICPGD) **.** _Fix any Bw >_ 0 _, L >_ 1 _, η >_ 0 _, and ε_ + _ε_<sup>_′_</sup> _≤ Bw/_ (2 _L_ ) _. Suppose that_ 

_1. The loss ℓ_ ( _·, ·_ ) _is convex in the first argument;_ 

_2. ∂sℓ is_ ( _ε, R, M, C_ ) _-approximable by sum of relus with R_ = max _{BxBw, By,_ 1 _}._ 

_3. R convex, and the proximal operator_ **prox** _ηR_ ( **w** ) _is_ ( _ηε_<sup>_′_</sup> _, R_<sup>_′_</sup> _, D_<sup>_′_</sup> _, C_<sup>_′_</sup> ) _-approximable by MLP with R_<sup>_′_</sup> = sup _∥_ **w** _∥_ 2 _≤Bw_ �� **w** _η_ +��2<sup>+</sup><sup>_ηε._</sup> 

_Then there exists a transformer_ TF **_θ_** _with_ ( _L_ + 1) _layers,_ max _ℓ∈_ [ _L_ ] _M_<sup>(</sup><sup>_ℓ_)</sup> _≤ M heads within the first L layers, M_<sup>(</sup><sup>_L_+1)</sup> = 2 _, and hidden dimension D_<sup>_′_</sup> _such that, for_ any input data ( _D,_ **x** _N_ +1) _such that_ 



TF **_θ_** ( **H**<sup>(0)</sup> ) _approximately implements (ICGD):_ 

_1. (Parameter space) For every ℓ ∈_ [ _L_ ] _, the ℓ-th layer’s output_ **H**<sup>(</sup><sup>_ℓ_)</sup> = TF **_θ_** (1: _ℓ_ )( **H**<sup>(0)</sup> ) _approximates ℓ steps of (ICGD): We have_ **h**<sup>(</sup> _i_<sup>_ℓ_)</sup> = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>**w**�</sup><sup>_ℓ_;</sup><sup>**0**</sup><sup>_D−_2</sup><sup>_d−_3; 1;</sup><sup>_ti_]</sup><sup>_foreveryi ∈_[</sup><sup>_N_+ 1]</sup><sup>_,where_</sup> ��� **w** _ℓ −_ **w** PGD _ℓ_ ��2<sup>_≤_(</sup><sup>_ε_+</sup><sup>_ε′_)</sup><sup>_·_(</sup><sup>_LηBx_)</sup><sup>_._</sup> 

36 



_Further, the weight matrices have norm bounds |||_ **_θ_** _||| ≤_ 3 + _R_ + 2 _ηC_ + _C_<sup>_′_</sup> _._ 

The proof of Theorem C.1 is essentially similar to the proof of Theorem 13, using the following generalized version of Lemma 14. 

**Lemma C.1** (Composition of error for approximating convex PGD) **.** _Suppose f_ : R<sup>_d_</sup> _→_ R _is a convex function and R is a convex regularizer. Let_ **w**<sup>_⋆_</sup> _∈_ arg min **w** _∈_ R _d f_ ( **w** ) + _R_ ( **w** ) _, R ≥_ 2 _∥_ **w**<sup>_⋆_</sup> _∥_ 2 _, and assume that ∇f is Lf -smooth on_ B<sup>_d_</sup> 2<sup>(</sup><sup>_R_)</sup><sup>_.Letsequences{_</sup><sup>**w**�</sup><sup>_ℓ}ℓ≥_0</sup><sup>_⊂_R</sup><sup>_dand{_</sup><sup>**w**</sup> GD<sup>_ℓ}ℓ≥_0</sup><sup>_⊂_R</sup><sup>_dbegivenby_</sup><sup>**w**�0=</sup><sup>**w**</sup> GD<sup>0=</sup><sup>**0**</sup><sup>_,_</sup> 



_L L for all ℓ ≥_ 0 _. Then as long as η ≤_ 2 _/Lf , for any_ 0 _≤ L ≤ R/_ (2 _ε_ ) _, it holds that_ ��� **w** _−_ **w** GD��2<sup>_≤Lεand_</sup> � _∥_ **w**<sup>_L_</sup> _∥_ 2 _≤_<sup>_<u>R</u>_</sup> 2<sup>+</sup><sup>_Lε ≤R._</sup> 

The proof of the above lemma is done by utilizing the non-expansiveness of the PGD operator **w** _�→_ **prox** _ηR_ ( **w** _− η∇f_ ( **w** )) and otherwise following the same arguments as for Lemma 14. 

### **C.2 Approximating a single GD step** 

**Proposition C.2** (Approximating a single GD step by a single attention layer) **.** _Let ℓ_ ( _·, ·_ ) : R<sup>2</sup> _→_ R _be a lossL_ � _N_ ( **w** _function_ ) := _N_<sup><u>1</u></sup> � _suchNi_ =1 _that_<sup>_ℓ_(</sup><sup>**w**</sup><sup>_⊤_</sup> _∂_<sup>**x**</sup> 1<sup>_i_</sup> _ℓ_<sup>_, y_</sup> _is_<sup>_i_)</sup> (<sup>_denote_</sup> _ε, R, M, C_<sup>_the_</sup> ) _-approximable_<sup>_empiricalriskwith_</sup> _by sum_<sup>_loss_</sup> _of_<sup>_function_</sup> _relus with_<sup>_ℓon_</sup> _R_<sup>_dataset_</sup> = max _{_<sup>_{_</sup> _B_<sup>(</sup><sup>**x**</sup> _x_<sup>_i_</sup> _B_<sup>_, y_</sup> _w_<sup>_i_)</sup> _, B_<sup>_}_</sup> _i∈y,_ [ _N_ 1 _}_ ]<sup>_._</sup> _. Let_ 

_Then, for any ε >_ 0 _, there exists an attention layer_ **_θ_** = _{_ ( **Q** _m,_ **K** _m,_ **V** _m_ ) _}m∈_ [ _M_ ] _with M heads such that,_ **h** _for_ � _i_ = [Attn _any input_ **_θ_** ( **H** )] _sequencei_ = [ **x** _i_ ; _ythati_<sup>_′_;</sup><sup>**w**�;</sup><sup>**0**</sup> _takes_<sup>_D−_2</sup><sup>_d_</sup> _form_<sup>_−_3; 1;</sup><sup>_t_</sup> **h**<sup>_i_</sup> _i_<sup>]</sup><sup>_for_</sup> = [<sup>_all_</sup> **x** _i_ ; _y_<sup>_i ∈_</sup> _i_<sup>_′_;</sup><sup>**w**[</sup><sup>_N_;</sup><sup>**0**+ 1]</sup><sup>_D−_2</sup><sup>_,d−where_3; 1;</sup><sup>_ti_]</sup><sup>_with∥_</sup><sup>**w**</sup><sup>_∥_</sup> 2<sup>_≤Bw,itgivesoutput_</sup> 



_Further, |||_ **_θ_** _||| ≤_ 2 + _R_ + 2 _ηC._ 

_Proof of Proposition C.2._ As _∂sℓ_ is ( _ε, R, M, C_ )-approximable by sum of relus, there exists a function _f_ : [ _−R, R_ ]<sup>2</sup> _→_ R of form 



such that sup( _s,t_ ) _∈_ [ _−R,R_ ]2 _|f_ ( _s, t_ ) _− ∂sℓ_ ( _s, t_ ) _| ≤ ε_ . 

Next, for every _m ∈_ [ _M_ ], we define matrices **Q** _m,_ **K** _m,_ **V** _m ∈_ R<sup>_D×D_</sup> such that 



for all _i, j ∈_ [ _N_ + 1]. As the input has structure **h** _i_ = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>**w**;</sup><sup>**0**</sup><sup>_D−_2</sup><sup>_d−_3; 1;</sup><sup>_ti_],thesematricesindeedexist,</sup> and further it is straightforward to check that they have norm bounds 



37 

Consequently, _|||_ **_θ_** _||| ≤_ 2 + _R_ + 2 _ηC_ . Now, for every _i, j ∈_ [ _N_ + 1], we have 



where the last equality follows from the bound 



so that the above relu equals 0 if _tj ≤_ 0. Therefore, 

Thus letting the attention layer **_θ_** = _{_ ( **V** _m,_ **Q** _m,_ **K** _m_ ) _}m∈_ [ _M_ ], we have 



where the error vector **_ε_** _∈_ R<sup>_d_</sup> satisfies 



This is the desired result. 

### **C.3 Proof of Theorem 13** 

We first prove part (a), which requires constructing the first _L_ layers of **_θ_** . Note that by our precondition _L ≤ Bw/_ (2 _ε_ ). 

> By our precondition, the partial derivative of the loss _∂sℓ_ is ( _ε, R, M, C_ )-approximable by sum of relus. Therefore we can apply Proposition C.2 to obtain that, there exists a single attention layer **_θ_**<sup>(1)</sup> = _{_ ( **Q** _m,_ **K** _m,_ **V** _m_ ) _}m∈_ [ _M_ ] with _M_ heads (and norm bounds specified in Proposition C.2), such that for any 

38 

**w** with _∥_ **w** _∥_ 2 _≤ Bw_ , the attention layer Attn **_θ_** (1) maps the input **h** _i_ = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>**w**;</sup><sup>**0**</sup><sup>_D−_2</sup><sup>_d−_3; 1;</sup><sup>_ti_]tooutput</sup> **h**<sup>_′_</sup> _i_<sup>= [</sup><sup>**x**</sup><sup>_i_;</sup><sup>_y_</sup> _i_<sup>_′_;</sup><sup>**w**�;</sup><sup>**0**</sup><sup>_D−_2</sup><sup>_d−_3; 1;</sup><sup>_ti_]forall</sup><sup>_i ∈_[</sup><sup>_N_+ 1],where</sup> 



Consider the _L_ -layer transformer **_θ_**<sup>1:</sup><sup>_L_</sup> = ( **_θ_**<sup>(1)</sup> _, . . . ,_ **_θ_**<sup>(1)</sup> ) which stacks the same attention layer **_θ_**<sup>(1)</sup> for _L_ times, and for the given input **h**<sup>(0)</sup> _i_ = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>**w**0;</sup><sup>**0**</sup><sup>_D−_2</sup><sup>_d−_3; 1;</sup><sup>_ti_], its</sup><sup>_ℓ_-th layer’s output</sup><sup>**h**(</sup> _i_<sup>_ℓ_)</sup> = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>**w**�</sup><sup>_ℓ_;</sup><sup>**0**</sup><sup>_D−_2</sup><sup>_d−_3; 1;</sup><sup>_ti_].</sup> We now inductively show that ��� **w** _ℓ_ ��2<sup>_≤Bw_and</sup> ��� **w** _ℓ −_ **w** GD _ℓ_ ��2<sup>_≤ℓε_forall</sup><sup>_ℓ∈_[</sup><sup>_L_].Thebasecaseof</sup><sup>_ℓ_=0</sup> � is trivial. Suppose the claim holds for _ℓ_ . Then for _ℓ_ + 1 _≤ L ≤ Bw/_ (2 _ε_ ), the sequence _{_ **w**<sup>_i_</sup> _}i≤ℓ_ +1 and _{_ **w** GD<sup>_i}_</sup> _i≤ℓ_ +1<sup>satisfies the precondition of the error composition lemma (Lemma 14) with error bound</sup><sup>_ε_, from</sup> which we obtain ��� **w** _ℓ_ +1��2<sup>_≤Bw_and</sup> 



This finishes the induction, and gives the following approximation guarantee for all _ℓ ∈_ [ _L_ ]: 



which proves part (a). 

We now prove part (b), which requires constructing the last attention layer **_θ_**<sup>(</sup><sup>_L_+1)</sup> . Recall **h**<sup>(</sup> _i_<sup>_L_)</sup> = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>**w**�</sup><sup>_L_;</sup><sup>**0**</sup><sup>_D−_2</sup><sup>_d−_3; 1;</sup><sup>_ti_]</sup> forthatallfor _i ∈_ every[ _N_ + 1] _i, j ∈_ . We[ _N_ + 1]construct, a 2-head attention layer **_θ_**<sup>(</sup><sup>_L_+1)</sup> = _{_ ( **Q** _m_<sup>(</sup><sup>_L_+1)</sup> _,_ **K**<sup>(</sup> _m_<sup>_L_+1)</sup> _,_ **V** _m_<sup>(</sup><sup>_L_+1)</sup> ) _}m_ =1 _,_ 2 such 



Note that the weight matrices have norm bound 



Then we have 



Above, (i) uses the identity _t_ = _σ_ ( _t_ ) _− σ_ ( _−t_ ). Further by part (a) we have 



This proves part (b), and also finishes the proof Theorem 13 where the overall ( _L_ + 1)-layer attention-only transformer is given by TF<sup>0</sup> **_θ_**<sup>with</sup> 



39 

### **C.4 Proof of Lemma 14** 

As _f_ is a convex, _Lf_ smooth function on B<sup>_d_</sup> 2<sup>(</sup><sup>_R_),themapping</sup><sup>_Tη_:</sup><sup>**w**</sup><sup>_�→_</sup><sup>**w**</sup><sup>_−η∇f_(</sup><sup>**w**)isnon-expansivein</sup> _∥·∥_ 2: Indeed, for any **w** _,_ **w**<sup>_′_</sup> _∈_ B<sup>_d_</sup> 2<sup>(</sup><sup>_R_)wehave</sup> 



<u>1</u> Above, (i) uses the property _⟨_ **w** _−_ **w**<sup>_′_</sup> _, ∇f_ ( **w** ) _−∇f_ ( **w**<sup>_′_</sup> ) _⟩≥ Lf_<sup>_∥∇f_(</sup><sup>**w**)</sup><sup>_−∇f_(</sup><sup>**w**</sup><sup>_′_)</sup><sup>_∥_</sup> 2<sup>2forsmoothconvexfunc-</sup> tions [60, Theorem 2.1.5]; (ii) uses the precondition that _η ≤_ 2 _/Lf_ . 

The lemma then follows directly by induction on _L_ . The base case of _L_ = 0 follows directly by assumption � that **w**<sup>0</sup> = **w** GD<sup>0</sup><sup>_∈_B</sup> 2<sup>_d_(</sup><sup>_R/_2).Supposetheclaimholdsforiterate</sup><sup>_L_.Foriterate</sup><sup>_L_+ 1</sup><sup>_≤R/_(2</sup><sup>_ε_),wehave</sup> 



Above, (i) uses the non-expansiveness, and (ii) uses the inductive hypothesis. Similarly, by our assumption **w**<sup>_⋆_</sup> = _Tη_ ( **w**<sup>_⋆_</sup> ), 



This finishes the induction. 

### **C.5 Convex ICGD with** _ℓ_ 2 **regularization** 

In the same setting as Theorem 13, consider the ICGD dynamics over an _ℓ_ 2-regularized empirical risk: 



with initialization **w** GD<sup>0</sup><sup>_∈_R</sup><sup>_d_andlearningrate</sup><sup>_η>_0,where</sup><sup>_L_�</sup><sup>_λ_</sup> _N_<sup>(</sup><sup>**w**):=</sup><sup>_L_�</sup><sup>_N_(</sup><sup>**w**)+</sup><sup>_<u>λ</u>_</sup> 2<sup>_∥_</sup><sup>**w**</sup><sup>_∥_</sup> 2<sup>2denotesthe</sup> _ℓ_ 2-regularized empirical risk. 

**Corollary C.1** (Convex ICGD with _ℓ_ 2 regularization) **.** _Fix any Bw >_ 0 _, L >_ 1 _, η >_ 0 _, and ε < BxBw. Suppose the loss ℓ_ ( _·, ·_ ) _is convex in the first argument, and ∂sℓ is_ ( _ε, R, M, C_ ) _-approximable by sum of relus with R_ = max _{BxBw, By,_ 1 _}._ 

_Then, there exists an attention-only transformer_ TF<sup>0</sup> **_θ_**<sup>_with_(</sup><sup>_L_+ 1)</sup><sup>_layers,_max</sup> _ℓ∈_ [ _L_ ]<sup>_M_(</sup><sup>_ℓ_)</sup><sup>_≤M_+ 1</sup><sup>_heads_</sup> _within the first L layers, and M_<sup>(</sup><sup>_L_+1)</sup> = 2 _such that for_ any input data ( _D,_ **x** _N_ +1) _with_ 



TF<sup>0</sup> **_θ_**<sup>(</sup><sup>**H**(0))</sup><sup>_approximatelyimplements(ICGD-ℓ_2</sup><sup>_):_</sup> 

_1. (Parameter space) For every ℓ ∈_ [ _L_ ] _, the ℓ-th layer’s output_ **H**<sup>(</sup><sup>_ℓ_)</sup> = TF **_θ_** (1: _ℓ_ )( **H**<sup>(0)</sup> ) _approximates ℓ steps of (ICGD-ℓ_ 2 _): We have_ **h**<sup>(</sup> _i_<sup>_ℓ_)</sup> = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>**w**�</sup><sup>_ℓ_;</sup><sup>**0**</sup><sup>_D−_2</sup><sup>_d−_3; 1;</sup><sup>_ti_]</sup><sup>_foreveryi ∈_[</sup><sup>_N_+ 1]</sup><sup>_,where_</sup> ��� **w** _ℓ −_ **w** GD _ℓ_ ��2<sup>_≤ε ·_(2</sup><sup>_LηBx_)</sup><sup>_._</sup> 

_2. (Prediction space) The final output_ **H**<sup>(</sup><sup>_L_+1)</sup> = TF **_θ_** ( **H**<sup>(0)</sup> ) _approximates the prediction of L steps of (ICGDℓ_ 2 _): We have_ **h**<sup>(</sup> _N_<sup>_L_</sup> +1<sup>+1)= [</sup><sup>**x**</sup><sup>_N_+1; �</sup><sup>_yN_+1;</sup><sup>**w**�</sup><sup>_L_;</sup><sup>**0**</sup><sup>_D−_2</sup><sup>_d−_3; 1; 0]</sup><sup>_,where_</sup> 



40 

_Further, the transformer admits norm bound |||_ **_θ_** _||| ≤_ 2 + _R_ + (2 _C_ + _λ_ ) _η._ 

_Proof._ This construction is the same as in the proof of Theorem 13, except that within each layer _ℓ ∈_ [ _L_ ], we add one more attention head ( **Q**<sup>(</sup><sup>_ℓ_)</sup> _,_ **K**<sup>(</sup><sup>_ℓ_)</sup> _,_ **V**<sup>(</sup><sup>_ℓ_)</sup> ) _⊂_ R<sup>_D×D_</sup> which when acting on its input **h**<sup>(</sup> _i_<sup>_ℓ−_1)</sup> = [ _∗_ ; _∗_ ; **w** �<sup>_ℓ−_1</sup> ; 1; _∗_ ] gives 



for all _i, j ∈_ [ _N_ + 1]. Note that �� **Q** ( _ℓ_ )��op<sup>=</sup> �� **K** ( _ℓ_ )��op<sup>= 1,and</sup> �� **V** ( _ℓ_ )��op<sup>=</sup><sup>_ηλ_.Further,itisstraightforward</sup> to check that the output of this attention head on every **h**<sup>(</sup> _i_<sup>_ℓ_)</sup> is 

Adding this onto the original output of the _ℓ_ -th layer exactly implements the gradient of the regularizer **w** _�→_<sup>_<u>λ</u>_</sup> 2<sup>_∥_</sup><sup>**w**</sup><sup>_∥_</sup> 2<sup>2.TherestoftheprooffollowsbyrepeatingtheargumentofTheorem13,andcombiningthe</sup> norm bound for the additional attention head here with the norm bound therein. 

## **D Proofs for Section 3.1** 

### **D.1 Proof of Theorem 4** 

Fix _λ ≥_ 0, 0 _≤ α ≤ β_ with _κ_ := _αβ_ <u>++</u> _λλ_<sup>,and</sup><sup>_Bw>_0,andconsideranyin-contextdata</sup><sup>_D_suchthatthe</sup> precondition of Theorem 4 holds. Let 



denote the ridge regression loss in (ICRidge), so that **w** ridge<sup>_λ_= arg min</sup> **w** _∈_ R<sup>_d L_ridge(</sup><sup>**w**).Itis a standardresult</sup> that _∇_<sup>2</sup> _L_ ridge( **w** ) = **X**<sup>_⊤_</sup> **X** _/N_ + _λ_ **I** _d_ , so that _L_ ridge is ( _α_ + _λ_ )-strongly convex and ( _β_ + _λ_ )-smooth over R<sup>_d_</sup> . Consider the gradient descent algorithm on the ridge loss 



with initialization, learning rate, and number of steps 



By standard convergence results for strongly convex and smooth functions (Proposition A.2), we have for all _t ≥_ 1 that 



Further, we have 

It remains to construct a transformer to approximate **w** GD<sup>_T_.Noticethattheproblem(ICRidge)corresponds</sup> to an _ℓ_ 2-regularized ERM with the square loss _ℓ_ ( _s, t_ ) := 2<sup><u>1</u>(</sup><sup>_s −t_)2,whosepartialderivative</sup><sup>_∂sℓ_(</sup><sup>_s, t_) =</sup><sup>_s −t_</sup> is exactly a sum of two relus: 



41 

In particular, this shows that _∂sℓ_ ( _s, t_ ) is (0 _, R,_ 2 _,_ 4)-approximable for any _R >_ 0, in particular for _R_ = max _{BxBw, By,_ 1 _}_ . 

Therefore, we can apply Corollary C.1 with the square loss _ℓ_ , learning rate _η_ , regularization strength _λ_ and accuracy parameter _ε_ = 0 to obtain that there exists an attention-only transformer TF<sup>0</sup> **_θ_**<sup>with(</sup><sup>_T_+ 1) :=</sup><sup>_L_</sup> layers such that the final output **h**<sup>(</sup> _N_<sup>_L_</sup> +1<sup>)= [</sup><sup>**x**</sup><sup>_N_+1; �</sup><sup>_yN_+1;</sup><sup>_∗_]with</sup> 



and number of heads _M_<sup>(</sup><sup>_ℓ_)</sup> = 3 for all _ℓ ∈_ [ _L −_ 1] (can be taken as 2 in the unregularized case _λ_ = 0 directly by Theorem 13), and _M_<sup>(</sup><sup>_L_)</sup> = 2. Further, **_θ_** admits norm bound _|||_ **_θ_** _||| ≤_ 2+ _R_ + _β_<sup><u>8+</u></sup> +<sup>_<u>λ</u>_</sup> _λ_<sup>_≤_3</sup><sup>_R_+8(</sup><sup>_β_+</sup><sup>_λ_)</sup><sup>_−_1 +1</sup><sup>_≤_</sup> 4 _R_ + 8( _β_ + _λ_ )<sup>_−_1</sup> . 

Combining (18) and (19), we obtain that 



Further, we have readw( **h**<sup>_T_</sup> _i_<sup>)=</sup><sup>**w**</sup> GD<sup>_T_forall</sup><sup>_i∈_[</sup><sup>_N_+ 1],wherereadw(</sup><sup>**h**):=</sup><sup>**h**(</sup><sup>_d_+2):(2</sup><sup>_d_+1)(cf.CorollaryC.1),</sup> so that _∥_ readw( **h**<sup>_T_</sup> _i_<sup>)</sup><sup>_−_</sup><sup>**w**</sup> ridge<sup>_λ∥_2</sup><sup>_≤ε/Bx_asshownabove.Thisfinishestheproof.</sup> 

### **D.2 Statistical analysis of in-context least squares** 

Consider the standard least-squares algorithm _A_ LS and least-squares estimator **w** � LS _∈_ R<sup>_d_</sup> defined as 



For any distribution P over ( **x** _, y_ ) _∈_ R<sup>_d_</sup> _×_ R and any estimator **w** _∈_ R<sup>_d_</sup> , let 



denote the expected risk of **w** over a new test example ( **x**<sup>_′_</sup> _, y_<sup>_′_</sup> ) _∼_ P. 

**Assumption A** (Well-posedness for learning linear predictors) **.** _We say a distribution_ P _on_ R<sup>_d_</sup> _×_ R _is well-posed for learning linear predictors, if_ ( **x** _, y_ ) _∼_ P _satisfies_ 

- _(1) ∥_ **x** _∥_ 2 _≤ Bx and |y| ≤ By almost surely;_ 

- _(2) The covariance_ **Σ** P := EP[ **xx**<sup>_⊤_</sup> ] _satisfies λ_ min **I** _d ⪯_ **Σ** P _⪯ λ_ max **I** _d, with_ 0 _< λ_ min _≤ λ_ max _, and κ_ := _λ_ max _/λ_ min _._ 

- _(3) The whitened vector_ **Σ**<sup>_−_</sup> P<sup>1</sup><sup>_/_2</sup> **x** _is K_<sup>2</sup> _-sub-Gaussian for some K ≥_ 1 _._ 



- _(5) We have_ E[( _y −⟨_ **x** _,_ **w** P<sup>_⋆⟩_)2</sup><sup>_|_</sup><sup>**x**]</sup><sup>_≤σ_2</sup><sup>_withprobabilityone(over_</sup><sup>**x**</sup><sup>_)._</sup> 

_Further, we say_ P _is well-posed with_ canonical parameters _if_ 



_where_ Θ( _·_ ) _and O_ ( _·_ ) _only hides absolute constants._ 

The following result bounds the excess risk of least squares under Assumption A with a clipping operation on the predictor; the clipping allows the result to only depend on the second moment of the noise (cf. Assumption A(5)) instead of e.g. its sub-Gaussianity, and also makes the result convenient to be directly translated to a result for transformers. 

**Proposition D.1** (Guarantees for in-context least squares) **.** _Suppose distribution_ P _satisfies Assumption A. Then as long as N ≥O_ ( _dK_<sup>4</sup> log(1 _/δ_ )) _, we have the following:_ 

42 

- _(a) The (clipped) least squares predictor achieves small expected excess risk (fast rate) over the best linear predictor: For any clipping radius R ≥ By,_ 







_Proof._ We first show P( _E_ cov) _≥_ 1 _− δ/_ 20. Let **Σ**<sup>�</sup> := _N_<sup><u>1</u></sup> � _Ni_ =1<sup>**x**</sup><sup>_i_</sup><sup>**x**</sup> _i_<sup>_⊤_, and let the whitened covariance and noise</sup> variables be denoted as 



Also let _zi_ := _yi −⟨_ **x** _i,_ **w** P<sup>_⋆⟩_denotethe“noise”variables.Notethat</sup> 



� � � is exactly a covariance concentration of the whitened vectors _{_ **x** _i}i∈_ [ _N_ ]. Recall that E[ **x** _i_ **x**<sup>_⊤_</sup> _i_<sup>]=</sup><sup>**I**</sup><sup>_d_,and</sup><sup>**x**�</sup><sup>_i_</sup> are _K_<sup>2</sup> -sub-Gaussian by assumption. Therefore, we can apply [82, Theorem 4.6.1], we have with probability at least 1 _− δ/_ 10 that 



Setting _N ≥O_ ( _K_<sup>4</sup> ( _d_ + log(1 _/δ_ ))) ensures that the right-hand side above is at most 1 _/_ 2, on which event we have 



i.e. _E_ cov holds. This shows that P( _E_ cov<sup>_c_)</sup><sup>_≤δ/_10.</sup> 

Next, we show (21). Using _E_ cov, we decompose the risk as 



43 

Above, (i) follows by assumption that _|yN_ +1 _| ≤ By ≤ R_ almost surely, so that removing the clipping can only potentiallysurely in the second term; (ii) follows by the fact thatincrease the distance in the first term, and E **x** the _N_ +1 _,y_ square _N_ +1[ _⟨_ **w** �lossLS _−_ is **w** upperP<sup>_⋆,_</sup><sup>**x**</sup><sup>_N_+1</sup> bounded<sup>_⟩_(</sup><sup>_⟨_</sup><sup>**w**</sup> P<sup>_⋆,_</sup><sup>**x**</sup> by<sup>_N_+11</sup> 2<sup>_·⟩−_(2</sup><sup>_RyN_)2+1almost)] = 0</sup> by the definition of **w** P<sup>_⋆_,aswellasthefactthat1</sup><sup>_{E_cov</sup><sup>_}_isindependentof(</sup><sup>**x**</sup><sup>_N_+1</sup><sup>_, yN_+1).</sup> 

<u>1</u> It thus remains to bound E _D_ � 2<sup>_∥_</sup><sup>**w**�LS</sup><sup>_−_</sup><sup>**w**</sup> P<sup>_⋆∥_2</sup> **Σ** P<sup>1</sup><sup>_{E_cov</sup><sup>_}_</sup> �. Note that on the event _E_ cov, we have 



Therefore, 



Note that E[ **x** � _izi_ ] = **Σ**<sup>_−_</sup> P<sup>1</sup><sup>_/_2</sup> E[ **x** _i_ ( _yi −⟨_ **w** P<sup>_⋆,_</sup><sup>**x**</sup><sup>_i⟩_)] = 0.Therefore,takingexpectationontheabove(over</sup><sup>_D_),we</sup> get 



Above, (i) follows by conditioning on **x** 1 and using Assumption A(5). Combining with (25), we obtain 



This proves (21). 

Finally, we show P( _E_ cov _∩ Ew_ ) _≥_ 1 _− δ/_ 10. Using (26) and **Σ** P _⪰ λ_ min **I** _d_ by assumption, we get 



Therefore, using an argument similar to Chebyshev’s inequality, 



This implies that 



This is the desired result. 

44 

### **D.3 Proof of Corollary 5** 

The proof follows by first checking the well-conditionedness of the data _D_ (cf. (5)) with high probability, then invoking Theorem 4 (for approximation least squares) and Proposition D.1 (for the statistical power of least squares). 

First, as P satisfies Assumption A, by Proposition D.1, as long as _N ≥O_ ( _K_<sup>4</sup> ( _d_ + log(1 _/δ_ ))), we have with probability at least 1 _− δ/_ 10 that event _E_ cov _∩ Ew_ holds. On this event, we have 



and thus the dataset _D_ is well-conditioned (in the sense of (5)) with parameters _α_ = _λ_ min _/_ 2, _β_ = 2 _λ_ max, and _Bw_ defined as above. Note that the condition number of **Σ**<sup>�</sup> is upper bounded by _β/α_ = 4 _λ_ max _/λ_ min _≤_ 4 _κ_ , where _κ_ is the upper bound on the condition number of **Σ** P as in Assumption A(c). Define parameters 



Note that _Bw ≤O_ ( _Bw_<sup>_⋆_+</sup> � _By_<sup>2</sup> _/λ_ min) by the above choice of _δ_ . 

We can thus apply Theorem 4 in the unregularized case ( _λ_ = 0) to obtain that, there exists a transformer **_θ_** with max _ℓ∈_ [ _L_ ] _M_<sup>(</sup><sup>_ℓ_)</sup> _≤_ 3, _|||_ **_θ_** _||| ≤_ 4 _R_ + 4 _/λ_ max (with _R_ = max _{BxBw, By,_ 1 _}_ ), and number of layers 



such that on _E_ cov _∩ Ew_ (so that _D_ is well-conditioned), we have (choosing the clipping radius in read<sup>�</sup> y( _·_ ) = clip _By_ (ready( _·_ )) to be _By_ ): 



We now bound the excess risk of the above transformer. Combining Proposition D.1 and (29), we have 



Above, (i) uses the approximation guarantee (29) as well as Proposition D.1(a) (with clipping radius _By_ ). This proves the desired excess risk guarantee. 

45 

Finally, under the canonical choice of parameters (20), the bounds for _L, M, |||_ **_θ_** _|||_ simplify to 



and the requirement for _N_ simplifies to _N ≥O_ ( _d_ + log(1 _/δ_ )) = _O_<sup>�</sup> ( _d_ ) (as _K_ = Θ(1)). This proves the claim about the required _N_ and _L_ . 

### **D.4 Proof of Corollary 6** 

Fix parameters _δ, ε >_ 0 to be specified later and a large universal constant _C_ 0. Let us set 



Consider the following good events (below **_ε_** = [ _εi_ ] _i∈_ [ _N_ ] _∈_ R<sup>_N_</sup> is given by _εi_ = _yi −⟨_ **w** _⋆,_ **x** _i⟩_ ) 



and we define _E_ := _Eπ ∩Ew ∩Eb ∩Eb,N_ +1. Under the event _E_ , the problem (ICRidge) is well-conditioned and _∥_ **w** ridge<sup>_λ∥≤Bw/_2(byLemmaD.1).</sup> 

Therefore, Theorem 4 implies that for _κ_ =<sup>_<u>α</u>_</sup> _β_ +<sup><u>+</u></sup> _λ_<sup>_<u>λ</u>_,thereexistsa</sup><sup>_L_=</sup><sup>_⌈_2</sup><sup>_κ_log(</sup><sup>_Bw/ε_</sup> <u>)</u> _⌉_ + 1-layer transformer � **_θ_** with prediction _yN_ +1 := read<sup>�</sup> y(TF<sup>0</sup> **_θ_**<sup>(</sup><sup>**H**))(clippedby</sup><sup>_By_),suchthatunderthegoodevent</sup><sup>_E_,wehave</sup> � � � _yN_ +1 = clip _By_ ( _⟨_ **x** _N_ +1 _,_ **w** _⟩_ ) and _∥_ **w** _−_ **w** ridge<sup>_λ∥≤_</sup><sup>_<u>ε</u>_</sup> <u>.</u> 

In the following, we show that **_θ_** is indeed the desired transformer (when _<u>ε</u>_ and _δ_ is suitably chosen). Notice that we have 



and we analyze these two parts separately. 

**Prediction risk under good event** _E_ **.** We first note that 



where the inequality is because _yN_ +1 _∈_ [ _−By, By_ ] under the good event _E_ . Notice that by our construction, � � � under the good event _E_ , **w** = **w** ( _D_ ) depends only on the dataset _D_<sup>5</sup> . Therefore, we have _∥_ **w** ( _D_ ) _−_ **w** ridge<sup>_λ_(</sup><sup>_D_)</sup><sup>_∥≤_</sup> _<u>ε</u>_ as long as the event _E_ 0 := _Eπ ∩Ew ∩Eb_ holds for ( **w** _⋆, D_ ). Thus, under _E_ 0, 



> 5We need this, as on _E c_ , the transformer output at this location could in principle depend additionally on **x** _N_ +1, as (17) may not hold due to the potential unbounededness of its input. A similar fact will also appear in later proofs (for generalized linear models and Lasso). 

46 



and we also have 





**Prediction risk under bad event** _E_<sup>_c_</sup> **.** Notice that 



We can upper bound P( _E_<sup>_c_</sup> ) = P( _Eπ_<sup>_c∪E_</sup> _w_<sup>_c∪E_</sup> _b_<sup>_c∪E_</sup> _b,N_<sup>_c_</sup> +1<sup>)byLemmaA.1,LemmaA.2andthesub-Gaussian</sup> tail bound: 



Thus, as long as _N ≥_ 8 log(12 _/δ_ ), we have P( _E_<sup>_c_</sup> ) _≤ δ_ . Further, a simple calculation yields 



Notice that _yN_ +1 _|_ **w** _⋆ ∼_ N(0 _, ∥_ **w** _⋆∥_ 2<sup>2+</sup><sup>_σ_2),henceE</sup><sup>_y_</sup> _N_<sup>4</sup> +1<sup>= 3E(</sup><sup>_∥_</sup><sup>**w**</sup><sup>_⋆∥_</sup> 2<sup>2+</sup><sup>_σ_2)2</sup><sup>_≤_3(3 + 2</sup><sup>_σ_2 +</sup><sup>_σ_4)</sup><sup>_≤B_</sup> _y_<sup>4.Thus,</sup> we can conclude that 



**Choosing** _<u>ε</u>_ **and** _δ_ **.** Combining the inequalities above, we have 



To ensure<sup><u>1</u></sup> 2<sup>E(</sup><sup>_y_�</sup><sup>_N_+1</sup><sup>_−yN_+1)2</sup><sup>_≤_BayesRisk</sup><sup>_π_+</sup><sup>_ε_,weonlyneedtotake</sup><sup><u>(</u></sup><sup>_<u>ε</u>_</sup> _<u>, δ</u>_ ) so that the following constraints are satisfied: 







our choice of _<u>ε</u>_ and _δ_ is feasible. Note that _κ ≤O_ �1 + _σ_<sup>_−_2�</sup> , and hence under such choice of <u>(</u> _<u>ε, δ</u>_ ), we have _L_ = _O_ (log(1 _/ε_ )) and _|||_ **_θ_** _|||_ = _O_<sup>�</sup> _√d_ . This is the desired result. � <u>�</u> 

**Lemma D.1.** _Under the event Eπ ∩Ew, we have_ 



_Proof of Lemma D.1._ By the definition of **w** ridge<sup>_λ_andrecallthat</sup><sup>_λ_=</sup><sup>_dσ_2</sup><sup>_/N_,wehave</sup><sup>**w**</sup> ridge<sup>_λ_=(</sup><sup>**X**</sup><sup>_⊤_</sup><sup>**X**+</sup> _dσ_<sup>2</sup> **I** _d_ )<sup>_−_1</sup> **X**<sup>_⊤_</sup> **y** . 

47 

Therefore, we only need to prove the following fact: for any _γ >_ 0 and **_β_**<sup>�</sup> = ( **X**<sup>_⊤_</sup> **X** + _dγ_ **I** _d_ )<sup>_−_1</sup> **X**<sup>_⊤_</sup> **y** , we have 



We now prove (31). Note that we have 



where **B** 1 = **X**<sup>_⊤_</sup> **X** ( **X**<sup>_⊤_</sup> **X** + _dγ_ **I** _d_ )<sup>_−_1</sup> , **B** 2 = ( **X**<sup>_⊤_</sup> **X** + _dγ_ **I** _d_ )<sup>_−_1</sup> **X**<sup>_⊤_</sup> . Note that _∥_ **B** 1 _∥_ op _≤_ 1 clearly holds, and under _Eπ_ we also have _∥_ **_ε_** _∥_ 2 _≤_ 2 _√Nσ_ . Therefore, it remains to bound the term _∥_ **B** 2 _∥_ op. 

Consider the SVD decomposition of **X** = _U_ Σ _V_ , Σ = diag( _λ_ 1 _, · · · , λd_ ), and _U ∈_ R<sup>_N×d_</sup> _, V ∈_ R<sup>_d×d_</sup> are orthonormal matrices. Then **B** 2 = _V_<sup>_⊤_</sup> (Σ<sup>2</sup> + _dγ_ **I** _d_ )<sup>_−_1</sup> Σ _U_<sup>_⊤_</sup> , and hence 



When _N ≤_ 36 _d_ , we directly have _∥_ **B** 2 _∥_ op _≤_ 2<sup><u>1</u>(</sup><sup>_dγ_)</sup><sup>_−_1</sup><sup>_/_2</sup><sup>_≤_3(</sup><sup>_Nγ_)</sup><sup>_−_1</sup><sup>_/_2.Otherwise,wehave</sup><sup>_N≥_36</sup><sup>_d_,and</sup> then for each _i ∈_ [ _d_ ], _λi ≥_ ~~�~~ _λ_ min( **X**<sup>_⊤_</sup> **X** ) _≥ √αN ≥ √N/_ 3. Hence, in this case we also have _∥_ **B** 2 _∥_ op _≤_ max _i λ_<sup>_−_</sup> _i_<sup>1</sup> _≤_ 3 _N_<sup>_−_1</sup><sup>_/_2</sup> . Combining the both cases completes the proof of (31). 

## **E Proofs for Section 3.2** 

We begin by stating our assumptions on the well-posedness of the generalized linear models. 

**Assumption B** (Well-posedness for learning GLMs) **.** _We assume that there is some Bµ >_ 0 _such that for any t ∈_ [ _−Bµ, Bµ_ ] _, g_<sup>_′_</sup> ( _t_ ) _≥ µg >_ 0 _._ 

_We also assume that for each i ∈_ [ _N_ + 1] _,_ ( **x** _i, yi_ ) _is independently sampled from_ P _such that the following holds._ 

_(a) Under the law_ ( **x** _, y_ ) _∼_ P _, We have_ **x** _∼_ SG( _Kx_ ) _, y ∼_ SG( _Ky_ ) _and g_ ( _⟨_ **w** _,_ **x** _⟩_ ) _∼_ SG( _Ky_ ) _∀_ **w** _∈_ B2( _Bw_ ) _. (b) For some µx >_ 0 _, it holds that_ 



_(c) For_ **_β_**<sup>_⋆_</sup> = arg min _L_ P _, it holds ∥_ **_β_**<sup>_⋆_</sup> _∥_ 2 _≤ Bw/_ 4 _._ 

### **E.1 Proof of Theorem 7** 

Let us fix parameters _εg >_ 0 and _T >_ 0 (that we specify later in proof). Define _R_ = max _{BxBw, By,_ 1 _}_ and 



By Proposition A.1, _g_ is ( _εg, M, R, C_ ) with 



Therefore, we can invoke Theorem 13 to obtain that, as long as 2 _Tεg ≤ Bw_ , there exists a _T_ -layer attentiononly transformer **_θ_**<sup>(1:</sup><sup>_T_)</sup> with _M_ heads per layer, such that for any input **H** of format (3) and satisfies (7), its last layer outputs **h**<sup>(</sup> _i_<sup>_T_)</sup> = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>**w**�</sup><sup>_T_;</sup><sup>**0**</sup><sup>_D−_2</sup><sup>_d−_3; 1;</sup><sup>_ti_],suchthat</sup> 



48 

where _{_ **w** GD<sup>_ℓ}ℓ∈_[</sup><sup>_L_]isthesequenceofgradientdescentiterateswithstepsize</sup><sup>_β−_1andinitialization</sup><sup>**w**</sup> GD<sup>0=</sup><sup>**0**.</sup> Notice that Proposition A.2 implies (with _κ_ := _β/α_ ) 



Furthermore, we can show that (similar to the proof of Theorem 13 (b)), there exists a single atten� tion layer **_θ_**<sup>(</sup><sup>_T_+1)</sup> with _M_ heads such that it outputs **h**<sup>(</sup> _N_<sup>_T_</sup> +1<sup>+1)</sup> = [ **x** _N_ +1; � _yN_ +1; **w**<sup>_T_</sup> ; **0** _D−_ 2 _d−_ 3; 1; 0], where � ��� _yN_ +1 _− g_ (� **x** _N_ +1 _,_ **w**<sup>_T_�</sup> )�� _≤ εg_ . 

In the following, we show that for suitably chosen ( _T, εg_ ), **_θ_** = ( **_θ_**<sup>(1:</sup><sup>_T_)</sup> _,_ **_θ_**<sup>(</sup><sup>_T_+1)</sup> ) is the desired transformer. � First notice that its output **h**<sup>(</sup> _N_<sup>_T_</sup> +1<sup>+1)</sup> = [ **x** _N_ +1; � _yN_ +1; **w**<sup>_T_</sup> ; **0** _D−_ 2 _d−_ 3; 1; 0] satisfies 



Therefore, for any fixed _ε >_ 0, we can take 



so that the **_θ_** we construct above ensures _|y_ � _N_ +1 _− g_ ( _⟨_ **x** _N_ +1 _,_ **w** GLM _⟩_ ) _| ≤ ε_ for any input **H** that satisfies (7). The upper bound on _|||_ **_θ_** _|||_ follows immediately from Theorem 13. 

### **E.2 Proof of Theorem 8** 

We summarize some basic and useful facts about GLM in the following theorem. Its proof is presented in Appendix E.3 - E.6. 

**Theorem E.1.** _Under Assumption B, the following statements hold with universal constant C_ 0 _and constant C_ 1 _, C_ 2 _that depend only on the parameters_ ( _Kx, Ky, Bµ, Bw, µx, Lg, µg_ ) _._ 

_(a) As long as N ≥ C_ 1 _· d, the following event happens with probability at least_ 1 _−_ 2 _e_<sup>_−N/C_1</sup> _:_ 



_(b) For any δ >_ 0 _, we have with probability at least_ 1 _− δ that_ 



_where we denote ι_ = log(2 + _LgKx_<sup>2</sup><sup>_Bw/Ky_)</sup><sup>_._</sup> 

_(c) Condition on (a) holds and N ≥ C_ 2 _· d, the event Er_ := _{∥_ **w** GLM _∥_ 2 _≤ Bw/_ 2 _} happens with probability at least_ 1 _− e_<sup>_N/C_2</sup> _._ 

_(d) For any_ **w** _∈_ B2( _Bw_ ) _, it holds that_ 



_(e) (Realizable setting) As long as_ **w** GLM _∈_ B2( _Bw_ ) _, it holds that_ 



49 

Therefore, we can set 



Consider the following good events 



Under the event _E_ and our choice of _α, β_ , the problem (ICGLM) is well-conditioned (i.e. (7) holds). 

Theorem 7 implies that there exists a transformer **_θ_** such that for any input **H** of the form (3), TF **_θ_** outputs **h**<sup>_′_</sup> _N_ +1<sup>=[</sup><sup>**x**</sup><sup>_N_+1; �</sup><sup>_yN_+1;</sup><sup>**w**�;</sup><sup>**0**</sup><sup>_D−_2</sup><sup>_d−_3; 1; 0],suchthattheoutputisgivenby</sup><sup>_y_�</sup><sup>_N_+1=</sup> read<sup>�</sup> y(TF **_θ_** ( **H** )) = � � � clip _By_ ( _yN_ +1) and **w** = read<sup>�</sup> w(TF **_θ_** ( **H** )) := ProjB2( _Bw_ )( **w** ), and the following holds on the good event _E_ : 

- 

- (a) _yN_ +1 = _fD_ ( **x** _N_ +1), where _fD_ = _A_ ( _D_ ) is a predictor such that _|fD_ ( **x** ) _− g_ ( _⟨_ **x** _,_ **w** GLM _⟩_ ) _| ≤ ε_ for all **x** _∈_ B2( _Bx_ ). 

- � 

- (b) **w** �� _∇_ = _L_ � _N_ **w** ( **w** (� _D_ )��) 2 _∈_<sup>_≤_</sup> B _L_ 2 _g_ ( _<u>βBBε</u> ww_<sup>.</sup> ) depends only on _D_ (by the proof of Theorem 7 and Theorem 13), such that 

In the following, we show that **_θ_** constructed above fulfills both (a) & (b) of Theorem 8. The bounds on number of layers and heads and _|||_ **_θ_** _|||_ follows from plugging our choice of _Bx, By_ in our proof of Theorem 7. 

� � � **Proof of Theorem 8 (a).** Notice that under the good event _E_ , we have **w** = **w** = **w** ( _D_ ) depends only on _D_ . Then we have 



Thus, we can consider _E_ 0 = _Er ∩Ew ∩Eb_ , and then 





� � where the second equality follows from _Lp_ ( **w** ( _D_ )) = E( **x** _N_ +1 _,yN_ +1) _|Dℓ_ ( _⟨_ **x** _N_ +1 _,_ **w** ( _D_ ) _⟩ , yN_ +1). Therefore, 



where the last line follows from Cauchy inequality and the fact P( _E_<sup>_c_</sup> ) = _O_ � _N_<sup>_−_10�</sup> , and _Bℓ_ is defined in Lemma E.1. 

Notice that by Theorem E.1 (d), we have 



50 

and by Theorem E.1 (b) and taking integration over _δ >_ 0, we have 



Also, we have inf _Lp_ = _Lp_ ( **_β_**<sup>_⋆_</sup> ) _≤ Bℓ_ by Lemma E.1. Therefore, we can conclude that 



Taking _ε_<sup>2</sup> _≤ BKwy_<sup>2</sup> _K_<sup>_ι_</sup> _x_<sup>2</sup> _Nd_<sup>completestheproof.</sup> 

**Proof of Theorem 8 (b).** Similar to the proof of Corollary 6, we have 



where the inequality follows from _yN_ +1 _∈_ [ _−By, By_ ] on event _E_ . For the first part, we have 





For the second part, we know P( _E_<sup>_c_</sup> ) = _O_ � _N_<sup>_−_10�</sup> and 



In conclusion, we have 



Taking _ε_<sup>2</sup> _≤ LgµKxx_<sup>2</sup> _µ_<sup>_K_</sup> _g_ _<u>y</u>_<sup>2</sup><sup>_ι_</sup> _Nd_<sup>completestheproof.</sup> 

**Lemma E.1.** _Suppose that_ **x** _∼_ SG( _Kx_ ) _, y ∼_ SG( _Ky_ ) _, and_ **w** _is a (possibly random) vector such that ∥_ **w** _∥_ 2 _≤ Bw. Then_ 



_Proof._ Notice that by our assumption, _|g_ (0) _| ≤_ 2 _Ky_ . Therefore, by the definition of _ℓ_ , 



The proof is then done by bounding the moment by E _|y|_<sup>8</sup> _≤O_ � _Ky_<sup>8</sup> � and E _|⟨_ **x** _,_ **w** _⟩|_<sup>8</sup> _≤ Bw_<sup>8E</sup><sup>_∥_</sup><sup>**x**</sup><sup>_∥_8</sup> 2<sup>_≤_</sup> _O_ ( _√dBwKx_ )<sup>8�</sup> , which is standard (by utilizing the tail bound of sub-Gaussian/sub-Exponential random � variable). 

51 

### **E.3 Proof of Theorem E.1 (a)** 

We begin with the upper bound on _λ_ max( _∇_<sup>2</sup> _L_<sup>�</sup> _N_ ( **w** )). By Lemma A.3, as long as _N ≥ C_ 0 _· d_ , the following event 



happens with probability at least 1 _−_ exp( _−N/C_ 0). By the assumption that sup _|g_<sup>_′_</sup> _| ≤ Lg_ , it is clear that when _Ew,_ 0 holds, we have _λ_ max( _∇_<sup>2</sup> _L_<sup>�</sup> _N_ ( **w** )) _≤_ 8 _LgKx_<sup>2</sup><sup>_∀_</sup><sup>**w**</sup><sup>_∈_R</sup><sup>_d_.</sup> 

In the following, we analyze the quantity _λ_ max( _∇_<sup>2</sup> _L_<sup>�</sup> _N_ ( **w** )). We have to invoke the following covering argument (see e.g. [82, Section 4.1.1]). 

**Lemma E.2.** _Suppose that V is a ε-covering of_ S<sup>_d−_1</sup> _with ε ∈_ [0 _,_ 1) _. Then the following holds:_ 



<u>1</u> _2. For any vector_ **x** _∈_ R<sup>_d_</sup> _, ∥_ **x** _∥_ 2 _≤_ 1 _−ε_<sup>max</sup><sup>**v**</sup><sup>_∈V |⟨_</sup><sup>**v**</sup><sup>_,_</sup><sup>**x**</sup><sup>_⟩|._</sup> Notice that 



Therefore, we can define _h_ ( _t_ ) := ( _Bµ −|t|_ )+ (which is a 1-Lipschitz function), and we have 



In the following, we pick a _ε_ **v** -covering _V_ of S<sup>_d−_1</sup> such that _|V| ≤_ (3 _/ε_ **v** )<sup>_d_</sup> (we will specify _ε_ **v** later in proof). Then for any **w** _∈_ B2( _Bw_ ), 



By our definition of _A_ ( **w** ), we have (for any fixed _Bxv_ ) 



By Lemma E.3, we can choose _Bxv_ = _Kx_ (15+log( _Kx_<sup>2</sup><sup>_/µx_)), and then E[</sup><sup>_U_</sup><sup>**v**(</sup><sup>**w**)]</sup><sup>_≥_3</sup><sup>_Bµµx/_8.Thus, combining</sup> the inequalities above, we can take _ε_ **v** =<sup>128</sup> _µx_<sup>_K_</sup> _<u>x</u>_<sup>2</sup> in the following, so that under event _Ew,_ 0, 



52 

In the following, we consider the random process � _U_ **v** ( **w** ) := _U_ **v** ( **w** ) _−_ E[ _U_ **v** ( **w** )]� **w**<sup>,whichiszero-mean</sup> and indexed by **w** _∈_ B2( _Bw_ ). For any fixed **v** , consider applying Proposition A.4 to the random process � _U_ **v** ( **w** )� **w**<sup>.Weneedtoverifythepreconditions:</sup> 

(a) With norm _ρ_ ( **w** _,_ **w**<sup>_′_</sup> ) = _∥_ **w** _−_ **w**<sup>_′_</sup> _∥_ 2, log _N_ (B _ρ_ ( **w** _, r_ ) _, δ_ ) _≤ d_ log(2 _Ar/δ_ ) with constant _A_ = 2; 

(b) Let _f_ ( **x** ; **w** ) := _h_ ( _⟨_ **w** _,_ **x** _i⟩_ ) min _⟨_ **v** _,_ **x** _i⟩_<sup>2</sup> _, Bxv_<sup>2</sup> , then _|f_ ( **x** ; **w** ) _| ≤ BµBxv_<sup>2andhenceinSG(</sup><sup>_CBµB_</sup> _xv_<sup>2)for</sup> � � any random **x** ; 

(c) For **w** _,_ **w**<sup>_′_</sup> _∈W_ , we have _|h_ ( _⟨_ **w** _,_ **x** _i⟩_ ) _− h_ ( _⟨_ **w**<sup>_′_</sup> _,_ **x** _i⟩_ ) _| ≤|⟨_ **w** _−_ **w**<sup>_′_</sup> _,_ **x** _i⟩|_ . Hence, because **x** _∼_ SG( _Kx_ ), the random variable _h_ ( _⟨_ **w** _,_ **x** _⟩_ ) _− h_ ( _⟨_ **w**<sup>_′_</sup> _,_ **x** _⟩_ ) is SG( _CKx∥_ **w** _−_ **w**<sup>_′_</sup> _∥_ 2), and the random variable _f_ ( **x** ; **w** ) _− f_ ( **x** ; **w**<sup>_′_</sup> ) is SG( _CKxBxv_<sup>2</sup><sup>_∥_</sup><sup>**w**</sup><sup>_−_</sup><sup>**w**</sup><sup>_′∥_2).</sup> 

Therefore, we can apply Proposition A.4 to obtain that with probability 1 _− δ_ 0, it holds 



where we denote _κg_ = 1 + _KxBw/Bµ_ . Setting _δ_ 0 = _δ/ |V|_ and taking the union bound over **v** _∈V_ , we obtain that with probability at least 1 _− δ_ , 



where we use log _|V| ≤ d_ log(4 _/ε_ **v** ). Therefore, we plug in the definition of _ε_ **v** and _Bxv_ to deduce that, if we set 



then as long as _N ≥ C_ 1 _· d_ , it holds max **v** _∈V_ E[ _U_ **v** ( **w** )] _− U_ **v** ( **w** ) _≤_ _<u>µx</u>_ 16 _Bµ_ with probability at least 1 _−_ exp( _−N/C_ 1). This is the desired result. 

**Lemma E.3.** _Under Assumption B, for Bxv_ = _Kx_ (15 + log( _Kx_<sup>2</sup><sup>_/µx_))</sup><sup>_,itholds_</sup> 



_Proof._ For any fixed **w** _∈_ B2( _Bw_ ) _,_ **v** _∈_ S<sup>_d−_1</sup> , 



Because **x** _∼_ SG( _Kx_ ), **x**<sup>_⊤_</sup> **v** _∼_ SG( _Kx_ ), and a simple calculation yields 



Taking _t_ = 15 + log( _Kx_<sup>2</sup><sup>_/µx_)givesE[(</sup><sup>**x**</sup><sup>_⊤_</sup><sup>**v**)21</sup><sup>_{|_</sup><sup>**x**</sup><sup>_⊤_</sup><sup>**v**</sup><sup>_| > Bxv}_]</sup><sup>_≤µx/_4,whichcompletestheproof.</sup> 

### **E.4 Proof of Theorem E.1 (b)** 

Notice that 



53 

In the following, we pick a minimal 1 _/_ 2-covering of S<sup>_d−_1</sup> (so _|V| ≤_ 5<sup>_d_</sup> ). Then by Lemma E.2, it holds 



Fix a **v** _∈_ S<sup>_d−_1</sup> and set _δ_<sup>_′_</sup> = _δ/|V|_ . We proceed to bound sup **w** _|X_ **v** ( **w** ) _|_ by applying Proposition A.4 to the random process _{X_ **v** ( **w** ) _}_ **w** . We need to verify the preconditions: 

(a) With norm _ρ_ ( **w** _,_ **w**<sup>_′_</sup> ) = _∥_ **w** _−_ **w**<sup>_′_</sup> _∥_ 2, log _N_ ( _δ_ ; B _ρ_ ( _r_ ) _, ρ_ ) _≤ d_ log(2 _Ar/δ_ ) with constant _A_ = 2; 

(b) For **z** = [ **x** ; _y_ ], we let _f_ ( **z** ; **w** ) := ( _g_ ( _⟨_ **w** _,_ **x** _⟩_ ) _− y_ ) _⟨_ **x** _,_ **v** _⟩_ , then _f_ ( **z** ; **w** ) _∼_ SE( _CKxKy_ ) for any **w** by our assumption on ( **x** _, y_ ); 

(c) For **w** _,_ **w**<sup>_′_</sup> _∈W_ , we have _|g_ ( _⟨_ **w** _,_ **x** _⟩_ ) _− g_ ( _⟨_ **w**<sup>_′_</sup> _,_ **x** _⟩_ ) _| ≤ Lg |⟨_ **w** _−_ **w**<sup>_′_</sup> _,_ **x** _⟩|_ . Hence, because **x** _∼_ SG( _Kx_ ), the random variable _g_ ( _⟨_ **w** _,_ **x** _i⟩_ ) _− g_ ( _⟨_ **w**<sup>_′_</sup> _,_ **x** _i⟩_ ) is sub-Gaussian in SG( _KxLg∥_ **w** _−_ **w**<sup>_′_</sup> _∥_ 2). Thus, _f_ ( **z** ; **w** ) _− f_ ( **z** ; **w**<sup>_′_</sup> ) is sub-exponential in SE( _CKx_<sup>2</sup><sup>_Lg∥_</sup><sup>**w**</sup><sup>_−_</sup><sup>**w**</sup><sup>_′∥_2).</sup> 

Therefore, we can apply Proposition A.4 to obtain that with probability 1 _− δ_ 0, it holds 



where we denote _κy_ = 1 + _LgKx_<sup>2</sup><sup>_Bw/Ky_.Setting</sup><sup>_δ_0=</sup><sup>_δ/ |V|_andtakingtheunionboundover</sup><sup>**v**</sup><sup>_∈V_,we</sup> obtain that with probability at least 1 _− δ_ , 



This is the desired result. 

### **E.5 Proof of Theorem E.1 (c)** 

In the following, we condition on (a) holds, i.e. _L_<sup>�</sup> _N_ is _α_ -strongly-convex and _β_ smooth over B2( _Bw_ ) with _α_ = _µxµg/_ 8 and _β_ = 8 _LgKx_<sup>2.Wedefine</sup> 



Then by standard convex analysis, we have 



Notice that �� _∇L_ � _N_ ( **_β_** _⋆_ )��2<sup>_≤ε_stat,wecanconcludethat</sup> 

Recall that we assume _∥_ **_β_**<sup>_⋆_</sup> _∥_ 2 _≤ Bw/_ 4, we can then consider _Es_ := _{ε_ stat _< αBw/_ 4 _}_ . Once _Es_ holds, our � � � argument above yields _∥_ **w** _∥_ 2 _< Bw_ , which implies _∇L_<sup>�</sup> _N_ ( **w** ) = 0. Therefore, **w** = arg min **w** _∈_ R _d L_<sup>�</sup> _N_ ( **w** ). Further, by Theorem E.1, we can set 



so that as long as _N ≥ C_ 2 _d_ , the event _Es_ holds with probability at least 1 _−_ exp( _−N/C_ 2). This is the desired result. 

54 

### **E.6 Proof of Theorem E.1 (d) & (e)** 

We first prove Theorem E.1 (d). Notice that 



Therefore, _Lp_ is ( _µgµx_ )-strongly-convex over B2( _Bw_ ). Therefore, because **_β_**<sup>_⋆_</sup> _∈_ B2( _Bw_ ) is the global minimum of _Lp_ , it holds that for all **w** _∈_ B2( _Bw_ ), 



By the definition of _ε_ stat, _∥∇Lp_ ( **w** ) _∥_ 2 _≤ ε_ stat + _∥∇L_<sup>�</sup> _N_ ( **w** ) _∥_ 2, and hence the proof of Theorem E.1 (d) is completed. 

We next prove Theorem E.1 (e), where we assume that E[ _y|_ **x** ] = _g_ ( _⟨_ **x** _,_ **_β_** _⟩_ ) (which implies **_β_**<sup>_⋆_</sup> = **_β_** directly) and **w** GLM _∈_ B2( _Bw_ ). Notice that 



and hence 



On the other hand, by the ( _µgµx_ )-strong-convexity of _Lp_ over B2( _Bw_ ), it holds that 



Finally, using the definition of **w** GLM, we have _∇L_<sup>�</sup> _N_ ( **w** GLM) = 0, and hence _∥∇Lp_ ( **w** GLM) _∥_ 2 _≤ ε_ stat, which completes the proof of Theorem E.1 (e). 

## **F Proofs for Section 3.3** 

### **F.1 Proof of Theorem 10** 

Fix _λN ≥_ 0, _β >_ 0 and _Bw >_ 0, and consider any in-context data _D_ such that the precondition of Theorem 10 holds. Recall that 



denotes the lasso regression loss in (ICLasso), so that **w** lasso = arg min **w** _∈_ R _d L_ lasso( **w** ). We further write 



Note that _∇_<sup>2</sup> _L_<sup>�0</sup> _N_<sup>(</sup><sup>**w**) =</sup><sup>**X**</sup><sup>_⊤_</sup><sup>**X**</sup><sup>_/N_andthus</sup><sup>_L_�0</sup> _N_<sup>is</sup><sup>_β_-smoothoverR</sup><sup>_d_.</sup> 

Consider the proximal gradient descent algorithm on the ridge loss 



with initialization **w** PGD<sup>0:=</sup><sup>**0**</sup><sup>_d_,learningrate</sup><sup>_η_:=</sup><sup>_β−_1,andnumberofsteps</sup><sup>_T_tobespecifiedlater.</sup> Similar to the proof of Theorem 4, we can construct a transformer to approximate **w** GD<sup>_T_.Consider</sup><sup>_ℓ_(</sup><sup>_s, t_) =</sup> 

55 

<u>1</u> 2<sup>(</sup><sup>_s−t_)2 and</sup><sup>_R_(</sup><sup>**w**) =</sup><sup>_λN ∥_</sup><sup>**w**</sup><sup>_∥_</sup> 1<sup>, then</sup><sup>_∂sℓ_(</sup><sup>_s, t_) is (0</sup><sup>_,_+</sup><sup>_∞,_2</sup><sup>_,_4)-approximable by sum of relus (cf. Definition 12),</sup> and **prox** _ηR_ is (0 _,_ + _∞,_ 4 _d,_ 4 + 2 _ηλN_ )-approximable by sum of relus (Proposition C.1). Therefore, we can apply Theorem C.1 with the square loss _ℓ_ , regularizer _R_ , learning rate _η_ and accuracy parameter 0 to obtain that there exists a transformer TF **_θ_** with ( _T_ +1) layers, number of heads _M_<sup>(</sup><sup>_ℓ_)</sup> = 2 for all _ℓ ∈_ [ _L_ ], and hidden dimension _D_<sup>_′_</sup> = 2 _d_ , such that the final output **h**<sup>(</sup> _N_<sup>_L_</sup> +1<sup>)= [</sup><sup>**x**</sup><sup>_N_+1; �</sup><sup>_yN_+1;</sup><sup>**w**</sup> PGD<sup>_T_;</sup><sup>_∗_]with</sup><sup>_y_�</sup><sup>_N_+1=</sup> � **w** PGD<sup>_T,_</sup><sup>**x**</sup><sup>_N_+1</sup> �. Further, the weight matrices have norm bounds _|||_ **_θ_** _||| ≤_ 10 _R_ + (8 + 2 _λN_ ) _β_<sup>_−_1</sup> _._ 

By the standard convergence result for proximal gradient descent (Proposition A.3), we have for all _t ≥_ 1 that 



Plugging in _∥_ **w** lasso _∥_ 2 _≤ Bw/_ 2 and _T_ = _L −_ 1 = � _βBw_<sup>2</sup><sup>_/ε_</sup> � finishes the proof. 

### **F.2 Sharper convergence analysis of proximal gradient descent for Lasso** 

**Collection of parameters** Throughout the rest of this section, we consider fixed _N ≥_ 1, _λN_ = � _<u>ρν</u>_ lo _N_ <u>g</u> _d_ for _ρ ≥_ 0, _ν ≥_ 0 fixed (and to be determined), fixed 0 _< α ≤ β_ , and fixed _Bw_<sup>_⋆>_0.Wewrite</sup><sup>_κ_:=</sup><sup>_β/α, κs_:=</sup> _s_ log _d β_ ( _Bw_<sup>_⋆_)2</sup><sup>_/ν_2,and</sup><sup>_ωN_:=</sup> _α_<sup>_<u>ρ</u>_</sup> _N_ . Here we present a sharper convergence analysis on the proximal gradient descent algorithm for _L_ lasso under the following well-conditionedness assumption, which will be useful for proving Theorem 11 in the sequel. 

**Assumption C** (Well-conditioned property for Lasso) **.** _We say the (ICLasso) problem is well-conditioned with sparsity s if the following conditions hold:_ 

_1. The_ ( _α, ρ_ ) _-RSC condition holds:_ 



_Further, λ_ max( **X**<sup>_⊤_</sup> **X** _/N_ ) _≤ β._ 

_2. The data_ ( **X** _,_ **y** ) _is “approximately generated from a s-sparse linear model”: There_ exists _a_ **w** _⋆ ∈_ R<sup>_d_</sup> _such that ∥_ **w** _⋆∥_ 2 _≤ Bw_<sup>_⋆, ∥_</sup><sup>**w**</sup><sup>_⋆∥_</sup> 0<sup>_≤sandfortheresidue_</sup><sup>**_ε_**=</sup><sup>**y**</sup><sup>_−_</sup><sup>**Xw**</sup><sup>_⋆,_</sup> 



_3. It holds that N ≥_ 32 _α_<sup>_<u>ρ</u>· s_log</sup><sup>_d(i.e._32</sup><sup>_ωN≤_1</sup><sup>_)._</sup> 

Assumption C1 imposes the standard restricted strong convexity (RSC) condition for the feature matrix **X** _∈_ R<sup>_N×d_</sup> , and Assumption C2 asserts that the data is approximately generated from a sparse linear model, with a bound on the _L∞_ norm of the error vector **X**<sup>_⊤_</sup> **_ε_** . Assumption C is entirely deterministic in nature, and suffices to imply the following convergence result. In the proof of Theorem 11, we show that Assumption C is satisfied with high probability when data is generated from the standard sparse linear model considered therein. 

**Theorem F.1** (Sharper convergence guarantee for Lasso) **.** _Under Assumption C, for the PGD iterates {_ � **w**<sup>_t_</sup> _}t≥_ 0 _on loss function L_<sup>�</sup> lasso _with stepsize η_ = 1 _/β and starting point_ **w**<sup>0</sup> = **0** _, we have L_<sup>�</sup> lasso( **w**<sup>_T_</sup> ) _− L_ lasso( **w** lasso) _≤ ε for all_ 



_where C is a universal constant._ 

The proof can be found in Appendix F.4. Combining Theorem F.1 with the construction in Theorem 10, we directly obtain the following result as a corollary. 

56 

**Theorem F.2** (In-context Lasso with transformers with sharper convergence) **.** _For any N, d, s ≥_ 1 _,_ 0 _< α ≤ β, ν ≥_ 0 _, ρ ≥_ 0 _, there exists a L-layer transformer_ TF **_θ_** _with_ 



_such that the following holds. On any input data_ ( _D,_ **x** _N_ +1) _such that the (ICLasso) problem satisfies Assumption C (which implies ∥_ **w** lasso _∥_ 2 _≤ Bw/_ 2 _with Bw_ = 2 _Bw_<sup>_⋆_+</sup> � _ν/α),_ TF **_θ_** ( **H**<sup>(0)</sup> ) _approximately imple-_ � � _ments (ICLasso), in that it outputs yN_ +1 = ready(TF **_θ_** ( **H** )) = _⟨_ **x** _N_ +1 _,_ **w** _⟩ with_ 



### **F.3 Basic properties for Lasso** 

**Lemma F.1** (Relaxed basic inequality) **.** _Suppose that Assumption C2 holds. Then it holds that_ 



_As a corollary, ∥_ **w** lasso _−_ **w** _⋆∥_ 1 _≤_ 4<sup>_√_</sup> _<u>s ∥</u>_ **w** lasso _−_ **w** _⋆∥_ 2 _._ 

_Proof._ Let us first fix any **w** _∈_ R<sup>_d_</sup> . Denote **∆** = **w** _−_ **w** _⋆_ , and let _S_ = supp( **w** _⋆_ ) be the set of indexes of nonzero entries of **w** _⋆_ . Then by definition, **y** = **Xw** _⋆_ + **_ε_** and _|S| ≤ s_ , and hence 



Combining these inequalities, we obtain 





where the last inequality follows from _∥_ **∆** _S∥_ 1 _≤_<sup>_√_</sup> _<u>s ∥</u>_ **∆** _S∥_ 2 _≤_<sup>_√_</sup> _<u>s ∥</u>_ **∆** _∥_ 2. This completes the proof of our main inequality. As for the corollary, we only need to use the definition that _L_<sup>�</sup> lasso( **w** lasso) _≤ L_<sup>�</sup> lasso( **w** _⋆_ ). 

**Proposition F.1** (Gap to parameter estimation error) **.** _Suppose that Assumption C holds. Then for all_ **w** _∈_ R<sup>_d_</sup> _,_ 



> _where we write_ gap := _L_<sup>�</sup> lasso( **w** ) _− L_<sup>�</sup> lasso( **w** lasso) _, and C_ = 120 _is a universal constant. In particular, we have s_ log _d ∥_ **w** lasso _−_ **w** _⋆∥_<sup>2</sup> 2<sup>_≤_10</sup> _α_<sup>_<u>ρν</u>_2</sup> _N ._ 

57 

_Proof._ We follow the notation in the proof of Lemma F.1. By (33), we have 



and hence _∥_ **∆** _∥_ 1 _≤_ 4<sup>_√_</sup> _<u>s ∥</u>_ **∆** _∥_ 2 +<sup>2</sup> _λ_<sup><u>g</u></sup> _N_<sup>apdueto</sup><sup>_L_�lasso(</sup><sup>**w**)</sup><sup>_−L_�lasso(</sup><sup>**w**</sup><sup>_⋆_)</sup><sup>_≤_gap.Ontheotherhand,bytheRSC</sup> condition (32), it holds that 



Therefore, we have 



where the last inequality uses AM-GM inequality and Cauchy inequality. Notice that _ρ_<sup>20</sup><sup>_s_</sup> _N_<sup>log</sup><sup>_d_</sup> _≤_<sup><u>2</u></sup> 3<sup>_α_,we</sup> now derive that 



Plugging in _λN_ = � _<u>ρν</u>_ lo _N_ <u>g</u> _d_ completes the proof. The corollary follows immediately by letting **w** = **w** lasso in above proof (hence gap = 0). 

**Lemma F.2** (Growth) **.** _It holds that_ 



_Proof._ For simplicity we denote **w** lasso := **w** lasso. By the first order optimality condition, it holds that 



where we write _R_ ( **w** ) := _λN ∥_ **w** _∥_ 1. Then by the convexity of _R_ , we have 



Rearranging completes the proof. 

### **F.4 Proof of Theorem F.1** 

For the simplicity of presentation, we write **w** lasso = **w** lasso and we denote gap<sup>_t_</sup> := _L_<sup>�</sup> lasso( **w**<sup>_t_</sup> ) _− L_<sup>�</sup> lasso( **w** lasso). By Lemma F.1, we have _∥_ **w**<sup>_t_</sup> _−_ **w** _⋆∥_ 1 _≤_ 4<sup>_√_</sup> _<u>s ∥</u>_ **w**<sup>_t_</sup> _−_ **w** _⋆∥_ 2 +<sup>2</sup> _λ_<sup><u>ga</u></sup> _N_<sup><u>p</u></sup><sup>_t_,whichimplies</sup> 



58 

We denote _µN_ = _ρ_<sup>2 lo</sup> _N_<sup><u>g</u></sup><sup>_d_.Usingtheassumptionthat</sup><sup>**X**is(</sup><sup>_α, ρ_)-RSC,weobtainthat</sup> 



Thus, as long as _N ≥_<sup>30</sup><sup>_<u>ρ</u>_2</sup> _α_<sup>_s_log</sup><sup>_d_</sup> , we have 



where the last inequality follows from Lemma F.2 and the definition of _λN , µN_ . We define _ε_ stat := 640 _sµN ∥_ **w** lasso _−_ **w** _⋆∥_<sup>2</sup> 2<sup>,</sup><sup>_T_0:= 10</sup><sup>_βν−_1</sup><sup>_∥_</sup><sup>**w**lasso</sup><sup>_∥_2</sup> 2<sup>.ByPropositionA.3(3),itholdsthatfor</sup> _t ≥ T_ 0, 



Then for all _t ≥ T_ 0 _−_ 1, we have (the second _≤_ below uses Proposition A.3(2)) 

Therefore, for _t ≥ T_ 0 _−_ 1, 



where the last inequality follows from Proposition A.3(2). Further, by Proposition A.3(3), we have 



Hence, we can conclude that gap<sup>_T_</sup> _≤ ε_ for all _T_ such that 



Now, by Proposition F.1, it holds that _∥_ **w** lasso _−_ **w** _⋆∥_<sup>2</sup> 2<sup>_≤_10</sup> _α_<sup>_<u>ρν</u>_2</sup> _s_ lo _N_ <u>g</u> _d_ , and hence 



Plugging in our definition of 



completes the proof. 

59 

### **F.5 Proof of Theorem 11** 

In this section, we present the proof of Theorem 11 based on Theorem F.2. We begin by recalling the following RSC property of a Gaussian random matrix [84, Theorem 7.16], a classical result in the high-dimensional statistics literature. 

**Proposition F.2** (RSC for Gaussian random design) **.** _Suppose that_ **X** = [ **x** 1; _· · ·_ ; **x** _N_ ]<sup>_⊤_</sup> _∈_ R<sup>_N×d_</sup> _is a random matrix with each row_ **x** _i being i.i.d. samples from_ N(0 _,_ **Σ** ) _. Then there are universal constants c_ 1 = 8<sup><u>1</u></sup><sup>_, c_2= 50</sup> _<u>e</u>_<sup>_−N/_32</sup> _such that with probability at least_ 1 _−_ 1 _−e_<sup>_−N/_32</sup><sup>_,_</sup> 



_where ρ_ ( **Σ** ) = max _i∈_ [ _d_ ] Σ _ii is the maximum of diagonal entries of_ **Σ** _._ 

Fix a parameter _δ_ 1 _≤ δ_ (which we will specify in proof) and a large universal constant _C_ 0. Let us set 



Similar to the proof of Corollary 6 (Appendix D.4), we consider the following good events (where **_ε_** = **Xw** _⋆ −_ **y** ) 



and we define _E_ := _Ew ∩Er ∩Eb ∩Eb,N_ +1. 

Furthermore, we choose _ν >_ 0 that correspond to the choice _λN_ = 8 _σ_ ~~�~~ log( _N_ 4 _d/δ_ <u>)</u> , and we also assume _N ≥_<sup><u>32</u></sup> _c_ 1<sup>_<u>c</u>_</sup><sup><u>2</u></sup><sup>_· s_log</sup><sup>_d_.Then,AssumptionCholdsontheevent</sup><sup>_E_.</sup> 

Therefore, we can apply Theorem F.2 with _ε_ = _νωN_ , which implies that there exists a _L_ -layer trans� former **_θ_** such that its prediction _yN_ +1 := read<sup>�</sup> y(TF<sup>0</sup> **_θ_**<sup>(</sup><sup>**H**)),sothatunderthegoodevent</sup><sup>_E_wehave</sup> � � _yN_ +1 = clip _By_ ( _⟨_ **x** _N_ +1 _,_ **w** _⟩_ ), where 



In the following, we show that **_θ_** is indeed the desired transformer (similarly to the proof in Appendix D.4). Consider the conditional prediction error 



and we analyze these two parts separately under the good event _E_ 0 := _Ew ∩Er ∩Eb_ of _D_ . 

**Part I.** We first note that 



where the inequality is because _yN_ +1 _∈_ [ _−By, By_ ] under the good event _E_ . Notice that by our construction, � � under the good event _E_ , **w** = **w** ( _D_ ) depends only on the dataset _D_ (because it is the ( _L −_ 1)-th iterate of 

60 

PGD on (ICLasso) problem). Applying Proposition F.1 to **w** � ( _D_ ) and using the definition of _ωN_ and our choice of _λN_ , we obtain that (under _E_ 0) 



Therefore, under _E_ 0, 



**Part II.** Notice that under good event _E_ 0, the bad event _E_<sup>_c_</sup> holds if and only if _Eb,N_<sup>_c_</sup> +1<sup>holds,andhence</sup> 



With a large enough constant _C_ 0, we clearly have P( _Eb,N_<sup>_c_</sup> +1<sup>)</sup><sup>_≤_(</sup><sup>_δ_1</sup><sup>_/N_)10.Further, a simple calculation yields</sup> 

E( _y_ � _N_ +1 _− yN_ +1)<sup>4</sup> _≤_ 8E( _y_ � _N_<sup>4</sup> +1<sup>+</sup><sup>_y_</sup> _N_<sup>4</sup> +1<sup>)</sup><sup>_≤_8</sup><sup>_B_</sup> _y_<sup>4+ 8E</sup><sup>_y_</sup> _N_<sup>4</sup> +1<sup>_≤_16</sup><sup>_B_</sup> _y_<sup>4</sup><sup>_,_</sup> 

where the last inequality is because the marginal distribution of _yN_ +1 is simply N(0 _, σ_<sup>2</sup> + _∥_ **w** _⋆∥_ 2<sup>2).Combining</sup> these yields 



Therefore, choosing _δ_ 1 = min _{δ, Bσw_<sup>_⋆}_isenoughforourpurpose,andundersuchchoiceof</sup><sup>_δ_1,</sup> 



**Conclusion.** Combining the inequalities above, we can conclude that under _E_ 0, 



It remains to show that P( _E_ 0) _≥_ 1 _− δ_ . By Proposition F.2, Lemma A.2 and Lemma A.4, we have 



Therefore, as long as _N ≥_ 32 log(12 _/δ_ ), we have P( _E_ 0) _≥_ 1 _− δ_ . This completes the proof. We also remark that in the construction above, 



which would be useful for bounding _|||_ **_θ_** _|||_ . 

61 

## **G Gradient descent on two-layer neural networks** 

We now move beyond the convex setting by showing that transformers can implement gradient descent on two-layer neural networks in context. 

Suppose that the prediction function pred( **x** ; **w** ) :=<sup>�</sup><sup>_K_</sup> _k_ =1<sup>_ukr_(</sup><sup>**v**</sup> _k_<sup>_⊤_</sup><sup>**x**)isgivenbyatwo-layerneuralnetwork,</sup> parameterized by **w** = [ **v** _k_ ; _uk_ ] _k∈_ [ _K_ ] _∈_ R<sup>_K_(</sup><sup>_d_+1)</sup> . Consider the empirical risk minimization problem: 



where _W_ is a bounded domain. For the sake of simplicity, in the following discussion we assume that Proj _W_ can be _exactly_ implemented by a MLP layer (e.g. _W_ = B _∞_ ( _Rw_ ) for some _Rw >_ 0). 

**Theorem G.1** (Approximate ICGD on two-layer NNs) **.** _Fix any Bv, Bu >_ 0 _, L ≥_ 1 _, η >_ 0 _, and ε >_ 0 _. Suppose that_ 

_1. Both the activation function r and the loss function ℓ is C_<sup>4</sup> _-smooth;_ 

_2. W is a closed domain such that W ⊂{_ **w** = [ **v** _k_ ; _uk_ ] _k∈_ [ _K_ ] _∈_ R<sup>_K_(</sup><sup>_d_+1)</sup> : _∥_ **v** _k∥_ 2 _≤ Bv, |uk| ≤ Bu}, and_ Proj _W_ = MLP **_θ_** `mlp` _for some MLP layer_ **_θ_** `mlp` _with hidden dimension Dw and_ ������ **_θ_** `mlp` ������ _≤ Cw;_ 

_Then there exists a_ (2 _L_ ) _-layer transformer_ TF **_θ_** _with_ 



_where O_ ( _·_ ) _hides the constants that depend on K, the radius parameters Bx, By, Bu, Bv and the smoothness of r and ℓ, such that for_ any input data ( _D,_ **x** _N_ +1) _such that input sequence_ **H**<sup>(0)</sup> _∈_ R<sup>_D×_(</sup><sup>_N_+1)</sup> _takes form_ (3) _,_ TF **_θ_** ( **H**<sup>(0)</sup> ) _approximately implements in-context gradient descent on risk (35): For every ℓ ∈_ [ _L_ ] _, the_ 2 _ℓ-th layer’s output_ **h**<sup>(2</sup> _i_<sup>_ℓ_)</sup> = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>**w**�</sup><sup>_ℓ_;</sup><sup>**0**; 1;</sup><sup>_ti_]</sup><sup>_foreveryi ∈_[</sup><sup>_N_+ 1]</sup><sup>_,and_</sup> 



_where_ **_ε_** _ℓ−_ 1<sup>_isanerrorterm._</sup> �� ��2<sup>_≤ε_</sup> 

As a direct corollary, the transformer constructed above can approximate the true gradient descent trajectory _{_ **w** GD<sup>_ℓ}_</sup> _ℓ≥_ 0<sup>on(37),definedas</sup><sup>**w**</sup> GD<sup>0=</sup><sup>**0**and</sup><sup>**w**</sup> GD<sup>_ℓ_+1=</sup><sup>**w**</sup> GD<sup>_ℓ−η∇L_�</sup><sup>_N_(</sup><sup>**w**</sup> GD<sup>_ℓ_)forall</sup><sup>_ℓ≥_0.</sup> 

**Corollary G.1** (Approximating multi-step ICGD on two-layer NNs) **.** _For any L ≥_ 1 _, under the same setting as Theorem G.1, the_ (2 _L_ ) _-layer transformer_ TF **_θ_** _there approximates the true gradient descent trajectory {_ **w** GD<sup>_ℓ}_</sup> _ℓ≥_ 0<sup>_:Fortheintermediateiterates{_</sup><sup>**w**�</sup><sup>_ℓ}_</sup> _ℓ∈_ [ _L_ ]<sup>_consideredtherein,wehave_</sup> 





**Remark on error accumulation** Note that in Corollary G.1, the error accumulates _exponentially_ in _ℓ_ rather than linearly as in Theorem 13. This is as expected, since gradient descent on non-convex objectives is inherently unstable at a high level (a slight error added upon each step may result in a drastically different trajectories); technically, this happens as the stability-like property Lemma 14 no longer holds for the nonconvex case. 

Corollary G.1 is a simple implication of Theorem G.1 and Part (b) of the following convergence and trajectory closeness result for inexact gradient descent. For any closed convex set _W ⊂_ R<sup>_d_</sup> , any function _f_ : _W →_ R, and any initial point **w** _∈W_ , let 



62 

denote the gradient mapping at **w** with step size _η_ , a standard measure of stationarity in constrained optimization [60]. Note that G<sup>_f_</sup> _W,η_<sup>(</sup><sup>**w**)=</sup><sup>_∇f_(</sup><sup>**w**)when</sup><sup>**w**</sup><sup>_−η∇f_(</sup><sup>**w**)</sup><sup>_∈W_(sothattheprojectiondoesnot</sup> take effect). 

**Lemma G.1** (Convergence and trajectory closeness of inexact GD) **.** _Suppose f_ : _W →_ R _, where W ⊂_ R<sup>_d_</sup> � � _is a convex closed domain and ∇f is Lf -Lipschitz on W. Let sequence {_ **w**<sup>_ℓ_</sup> _}ℓ≥_ 0 _⊂_ R<sup>_d_</sup> _be given by_ **w**<sup>0</sup> = **w**<sup>0</sup> _,_ 



_for all ℓ ≥_ 0 _. Then the following holds._ 

- _(a) As long as η ≤_ 1 _/Lf , for all L ≥_ 1 _,_ 



- _(b) Let the sequences {_ **w** GD<sup>_ℓ}ℓ≥_0</sup><sup>_⊂_R</sup><sup>_dandbegivenby_</sup><sup>**w**</sup> GD<sup>0=</sup><sup>**w**0</sup><sup>_and_</sup><sup>**w**</sup> GD<sup>_ℓ_+1= Proj</sup> _W_<sup>(</sup><sup>**w**</sup> GD<sup>_ℓ−η∇f_(</sup><sup>**w**</sup> GD<sup>_ℓ_))</sup><sup>_._</sup> _Then it holds that_ 



### **G.1 Proof of Theorem G.1** 

We only need to prove the following single-step version of Theorem G.1. 

**Proposition G.1.** _Under the assumptions of Theorem G.1, there exists a 2-layer transformer_ TF **_θ_** _with the same bounds on the number of heads, hidden dimension and the norm, such that for_ any input data ( _D,_ **x** _N_ +1) _and_ any **w** _∈_ R<sup>_d_</sup> _,_ TF **_θ_** _maps_ 



_where_ 



Before we present the formal (and technical) proof of Proposition G.1, we first provide some intuitions. To begin with, we first note that 



where _∂_ 1 _ℓ_ is the partial derivative of _ℓ_ with respect to the first component, and 



Therefore, the basic idea is that we can use an attention layer to approximate ( **x** _i,_ **w** ) _�→_ pred( **x** _i_ ; **w** ), then use an MLP layer to implement (pred( **x** _i_ ; **w** ) _, yi_<sup>_′, ti_)</sup><sup>_�→_1</sup><sup>_{i<N_+ 1</sup><sup>_} · ∂_1</sup><sup>_ℓ_(pred(</sup><sup>**x**</sup><sup>_i_;</sup><sup>**w**)</sup><sup>_, yi_),andthenusean</sup> attention layer to compute the gradient descent step **w** _�→_ **w** _− η∇LN_ ( **w** ), and finally use an MLP layer to implement the projection into _W_ . 

Based on the observations above, we now present the proof of Proposition G.1. 

63 

_Proof of Proposition G.1._ We write _D_ 0 = _d_ + 1 + _K_ ( _d_ + 1) be the length of the vector [ **x** _i_ ; _yi_ ; **w** ]. We also define 



Let us fix _εr, εp, εℓ >_ 0 that will be specified later in proof (see (39)). By our assumption and Proposition A.1, the following facts hold. 

(1) The function _r_ ( _t_ ) is ( _εr, R_ 1 _, M_ 1 _, C_ 1) for _R_ 1 = max _{BxBu,_ 1 _}_ , _M_ 1 _≤ O_<sup>�</sup> � _C_ 1<sup>2</sup><sup>_ε−_</sup> _r_<sup>2</sup> �, where _C_ 1 depends only on _R_ 1 and the _C_<sup>2</sup> -smoothness of _r_ . Therefore, there exists 



such that sup _t∈_ [ _−R_ 1 _,R_ 1] _|r_ ( _t_ ) _−_ _<u>r</u>_ ( _t_ ) _| ≤ εr_ . 

(2) The function ( _t, y_ ) _�→ ∂_ 1 _ℓ_ ( _t, y_ ) is ( _εℓ, R_ 2 _, M_ 2 _, C_ 2) for _R_ 2 = max _{KBr, By,_ 1 _} M_ 2 _≤ O_<sup>�</sup> � _C_ 2<sup>2</sup><sup>_ε−_</sup> _ℓ_<sup>2</sup> �, where _C_ 2 depends only on _R_ 2 and the _C_<sup>3</sup> -smoothness of _∂_ 1 _ℓ_ . Therefore, there exists 



such that sup( _t,y_ ) _∈_ [ _−R_ 2 _,R_ 2]2 _|g_ ( _t, y_ ) _− ∂_ 1 _ℓ_ ( _t, y_ ) _| ≤ εℓ_ . 

(3) The function ( _s, t_ ) _�→ s · r_<sup>_′_</sup> ( _t_ ) is ( _εp, R_ 3 _, M_ 3 _, C_ 3) for _R_ 3 = max _{BxBu, BgBu,_ 1 _}_ , _M_ 3 _≤ O_<sup>�</sup> � _C_ 3<sup>2</sup><sup>_ε−_</sup> _p_<sup>2</sup> �, where _C_ 3 depends only on _R_ 3 and the _C_<sup>3</sup> -smoothness of _r_<sup>_′_</sup> . Therefore, there exists 



such that sup( _s,t_ ) _∈_ [ _−R_ 3 _,R_ 3]2 _|P_ ( _s, t_ ) _− s · r_<sup>_′_</sup> ( _t_ ) _| ≤ εp_ . 

In the following, we proceed to construct the desired transformer step by step. 

Step 1: construction of **_θ_** `attn`<sup>(1).</sup> We consider the matrices _{_ **Q**<sup>(1)</sup> _k,m_<sup>_,_</sup><sup>**K**(1)</sup> _k,m_<sup>_,_</sup><sup>**V**</sup> _k,m_<sup>(1)</sup><sup>_}k∈_[</sup><sup>_K_]</sup><sup>_,m∈_[</sup><sup>_M_</sup> 1<sup>]sothatforall</sup> _i, j ∈_ [ _N_ + 1], we have 



As the input has structure **h** _i_ = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>**w**;</sup><sup>**0**; 1;</sup><sup>_ti_], these matrices indeed exist, and further it is straightforward</sup> to check that they have norm bounds 



A simple calculation shows that 



For simplicity, we denote pred( **x** ; **w** ) :=<sup>�</sup><sup>_K_</sup> _k_ =1<sup>_uk_</sup> _<u>r</u>_ <u>(</u> _⟨_ **v** _k,_ **x** _⟩_ ) in the following analysis. Thus, letting the attention layer **_θ_** `attn`<sup>(1)=</sup><sup>_{_(</sup><sup>**V**</sup> _k,m_<sup>(1)</sup><sup>_,_</sup><sup>**Q**(1)</sup> _k,m_<sup>_,_</sup><sup>**K**(1)</sup> _k,m_<sup>)</sup><sup>_}_</sup> ( _k,m_ )<sup>,wehave</sup> 



64 

Step 2: construction of **_θ_** `mlp`<sup>(1).Wepickmatrices</sup><sup>**W**1</sup><sup>_,_</sup><sup>**W**2sothat</sup><sup>**W**1maps</sup> 



and **W** 2 _∈_ R<sup>_D×M_3</sup> with entries being ( **W** 2)( _j,m_ ) = _c_<sup>2</sup> _m_<sup>1</sup><sup>_{j_=</sup><sup>_D_0+ 2</sup><sup>_}_.Itisclearthat</sup><sup>_∥_</sup><sup>**W**1</sup><sup>_∥_</sup> op<sup>_≤R_2+ 1,</sup> _∥_ **W** 2 _∥_ op _≤ C_ 2. Then we have 



In the following, we abbreviate _gi_ = 1 _{tj_ = 1 _} · g_ (pred( **x** _i_ ; **w** ) _, yi_<sup>_′_).Hence,</sup><sup>**_θ_**</sup><sup>`mlp`maps</sup> 



By the definition of the function _g_ , for each _i ∈_ [ _N_ ], 



2 where _Lℓ_ := max _|t|≤KBr,|y|≤By_ �� _∂tt_<sup>_ℓ_(</sup><sup>_t, y_)</sup> �� is the smoothness of _∂_ 1 _ℓ_ . Also, _gN_ +1 = 0 by definition. Step 3: construction of **_θ_** `attn`<sup>(2).Weconsiderthematrices</sup><sup>_{_</sup><sup>**Q**(2)</sup> _k,_ 1 _,m_<sup>_,_</sup><sup>**K**(2)</sup> _k,_ 1 _,m_<sup>_,_</sup><sup>**V**</sup> _k,_<sup>(2)</sup> 1 _,m_<sup>_}k∈_[</sup><sup>_K_]</sup><sup>_,m∈_[</sup><sup>_M_</sup> 3<sup>]sothatforall</sup> _i, j ∈_ [ _N_ + 1], we have 



We further consider the matrices _{_ **Q**<sup>(2)</sup> _k,_ 2 _,m_<sup>_,_</sup><sup>**K**(2)</sup> _k,_ 2 _,m_<sup>_,_</sup><sup>**V**</sup> _k,_<sup>(2)</sup> 2 _,m_<sup>_}k∈_[</sup><sup>_K_]</sup><sup>_,m∈_[</sup><sup>_M_</sup> 1<sup>]sothatforall</sup><sup>_i, j∈_[</sup><sup>_N_+ 1],wehave</sup> 

By the structure of the input **h**<sup>(1)</sup> _i_<sup>,thesematricesindeedexist,andfurtheritisstraightforwardtocheck</sup> that they have norm bounds 



Furthermore, a simple calculation shows that 



where the summation is taken over all possibilities of the tuple ( _k, w, m_ ), i.e. over the union of [ _K_ ] _×{_ 1 _}×_ [ _M_ 3] and [ _K_ ] _× {_ 2 _} ×_ [ _M_ 1]. 

By our definition, we have _|P_ ( _s, t_ ) _− sr_<sup>_′_</sup> ( _t_ ) _| ≤ εp_ for all _s, t ∈_ [ _−R_ 3 _, R_ 3]. Therefore, for each _i ∈_ [ _N_ ], _k ∈_ [ _K_ ], 

_|P_ ( _ukgj, ⟨_ **v** _k,_ **x** _j⟩_ ) _− ∂_ 1 _ℓ_ (pred( **x** _j_ ; **w** ) _, yj_ ) _· uk · r_<sup>_′_</sup> ( _⟨_ **v** _k,_ **x** _j⟩_ ) _| ≤ εp_ + _|gj − ∂_ 1 _ℓ_ (pred( **x** _i_ ; **w** ) _, yi_ ) _| · |uk| · |r_<sup>_′_</sup> ( _⟨_ **v** _k,_ **x** _j⟩_ ) _|_ 

65 



where _Lr_ := max _|t|≤BxBu |r_<sup>_′_</sup> ( _t_ ) _|_ is the upper bound of _r_<sup>_′_</sup> . Similarly, for each _i ∈_ [ _N_ ], _k ∈_ [ _K_ ], we have 

_~~|~~_ _<u>r</u>_ <u>(</u> _⟨_ **v** _k,_ **x** _j⟩_ ) _· gj − r_ ( _⟨_ **v** _k,_ **x** _j⟩_ ) _· ∂_ 1 _ℓ_ (pred( **x** _j_ ; **w** ) _, yj_ ) _| ≤_ 2 _Bgεr_ + 2 _Br_ ( _εℓ_ + _BuL_<sup>2</sup> _ℓ_<sup>_εr_)</sup><sup>_._</sup> 

As for the case _i_ = _N_ +1, we have _gN_ +1 = 0 and _|P_ ( _ukgN_ +1 _, ⟨_ **v** _k,_ **x** _N_ +1 _⟩_ ) _| ≤ εp_ for each _k ∈_ [ _K_ ] by defintion. Combining these estimations and using (37) and (38), we can conclude that 



Thus, to ensure _η−_ 1 **g** ( **w** ) + _∇L_ � _N_ ( **w** )<sup>weonlyneedtochoose</sup><sup>_εp, εℓ, εr_as</sup> ��� ���2<sup>_≤ε_,</sup> 



Thus, letting the attention layer **_θ_**<sup>(2)</sup> 

Step 4: construction of **_θ_** `mlp`<sup>(2).Weonlyneedtopick</sup><sup>**_θ_**</sup> `mlp`<sup>(2)sothatitmaps</sup> 



### **G.2 Proof of Lemma G.1** 

For every _ℓ ≥_ 0, define the intermediate iterates (before projection) 



so that **w** �<sup>_ℓ_+1</sup> = Proj _W_ ( **w** �<sup>_ℓ_+</sup> 2<sup><u>1</u></sup> ) and **w** GD<sup>_ℓ_+1= Proj</sup> _W_<sup>(</sup><sup>**w**</sup> GD _ℓ_ + 2<sup><u>1</u>).</sup> 

We first prove part (a). We begin by deriving a relation between ��� **w** _ℓ_ +1 _−_ **w** � _ℓ_ ��22<sup>and</sup> ��� _η_ G _fW,η_<sup>(</sup><sup>**w**�</sup><sup>_ℓ_)</sup> ���22<sup>.Let</sup> **w** �<sup>_ℓ_+</sup><sup><u>1</u></sup> 2 := **w** �<sup>_ℓ_</sup> _− η∇f_ ( **w** �<sup>_ℓ_</sup> ) and **w** �<sup>_ℓ_+1</sup> := Proj _W_ ( **w** �<sup>_ℓ_+</sup> 2<sup><u>1</u></sup> ) denote the _exact_ projected gradient iterate starting � from **w**<sup>_ℓ_</sup> . We have 



Above, (i) uses the inequality _∥a − b∥_<sup>2</sup> 2<sup>_≥_</sup><sup><u>1</u></sup> 2<sup>_∥a∥_</sup> 2<sup>2</sup><sup>_−∥b∥_2</sup> 2<sup>;(ii)usesthefactthatprojectiontoaconvexsetis</sup> a non-expansion; (iii) uses the definition of the gradient mapping. 

By the _Lf_ -smoothness of _f_ within _W_ , we have 



66 



Above, (i) uses the property **w** �<sup>_ℓ_+1</sup> _−_ **w** �<sup>_ℓ_+</sup><sup><u>1</u></sup> 2 _,_ **w** �<sup>_ℓ_+1</sup> _−_ **w** �<sup>_ℓ_�</sup> _≤_ 0 of the projection **w** �<sup>_ℓ_+1</sup> = Proj _W_ ( **w** �<sup>_ℓ_+</sup><sup><u>1</u></sup> 2 ) (using � � **w**<sup>_ℓ_</sup> _∈W_ ); (ii) uses _Lf /_ 2 _≤_ 1 _/_ (2 _η_ ) by our choice of _η ≤_ 1 _/Lf_ ; (iii) uses (40). Rearranging and summing the above over _ℓ_ = 0 _, . . . , L −_ 1, we obtain 



Dividing both sides by _ηL/_ 8 yields part (a). 

Next, we prove part (b). Let _C_ := 1 + _ηLf_ . We prove by induction that 



� for all _ℓ ≥_ 0. The base case of _ℓ_ = 0 follows by definition that **w**<sup>0</sup> = **w** GD<sup>0=</sup><sup>**w**0.Supposetheresultholds</sup> for _ℓ_ . Then for _ℓ_ + 1, we have 



Above, (i) uses again the non-expansiveness of the convex projection Proj _W_ ; (ii) uses the fact that the operator **w** _�→_ **w** _− η∇f_ ( **w** ) is (1 + _ηLf_ ) = _C_ -Lipschitz; and (iii) uses the inductive hypothesis. This proves the case for _ℓ_ + 1 and thus finishes the induction. We can further relax (41) into 



This proves part (b). 

## **H Proofs for Section 4** 

### **H.1 Proof of Proposition 15** 

We begin by restating Proposition 15 into the following version, which contains additional size bounds on **_θ_** . **Theorem H.1** (Full statement of Proposition 15) **.** _Suppose that for_ 



67 

_ℓ_ ( _·, ·_ ) _is_ ( _γ/_ 3 _, R, M, C_ ) _-approximable by sum of relus (Definition 12). Then there exists a 3-layer transformer_ TF **_θ_** _with_ 



_that maps_ 



_where the predictor f_<sup>�</sup> : R<sup>_d_</sup> _→_ R _is a convex combination of {fk_ : _L_<sup>�</sup> val( _fk_ ) _≤_ min _k⋆∈_ [ _K_ ] _L_<sup>�</sup> val( _fk⋆_ ) + _γ}. As a corollary, for any convex risk L_ : (R<sup>_d_</sup> _→_ R) _→_ R _, f_<sup>�</sup> _satisfies_ 



To prove Theorem H.1, we first state and prove the following two propositions. 

**Proposition H.1** (Evaluation layer) **.** _There exists a 1-layer transformer_ TF **_θ_** _with MK heads and |||_ **_θ_** _||| ≤_ 3 _R_ + 2 _NKC/ |D_ val _| such that for all_ **H** _such that_ max _i{|yi_<sup>_′|} ≤R,_max</sup><sup>_i,k{|fk_(</sup><sup>**x**</sup><sup>_i_)</sup><sup>_|} ≤R,_TF</sup><sup>**_θ_**</sup><sup>_maps_</sup> 





_Proof of Proposition H.1._ As _ℓ_ is ( _ε, R, M, C_ )-approximable by sum of relus, there exists a function _g_ : R<sup>2</sup> _→_ R of form 



such that sup( _s,t_ ) _∈_ [ _−R,R_ ]2 _|g_ ( _s, t_ ) _− ℓ_ ( _s, t_ ) _| ≤ ε_ . We define 



Next, for every _m ∈_ [ _M_ ] and _k ∈_ [ _K_ ], we define matrices **Q** _m,k,_ **K** _m,k,_ **V** _m,k ∈_ R<sup>_D×D_</sup> such that for all _i, j ∈_ [ _N_ + 1], 



where **e** _s ∈_ R<sup>_D_</sup> is the vector with _s_ -th entry being 1 and others being 0. As the input has structure **h** _i_ = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>_∗_;</sup><sup>_f_1(</sup><sup>**x**</sup><sup>_i_);</sup><sup>_· · ·_;</sup><sup>_fK_(</sup><sup>**x**</sup><sup>_i_);</sup><sup>**0**</sup><sup>_K_+1; 1;</sup><sup>_ti_],thesematricesindeedexist,andfurtheritisstraightforward</sup> to check that they have norm bounds 



Now, for every _i, j ∈_ [ _N_ + 1], we have 



68 



where the last equality follows from the bound _|amfk_ ( **x** _j_ ) + _bmyj_ + _dm| ≤ R_ ( _|am|_ + _|bm|_ )+ _dm ≤_ 2 _R_ , so that the above relu equals 0 if _tj ≤_ 0. Therefore, for each _i ∈_ [ _N_ + 1] and _k ∈_ [ _K_ ], 



Thus letting the attention layer **_θ_** = _{_ ( **V** _m,k,_ **Q** _m,k,_ **K** _m,k_ ) _}_ ( _m,k_ ) _∈_ [ _M_ ] _×_ [ _K_ ], we have 



This is the desired result. 

**Proposition H.2** (Selection layer) **.** _There exists a 3-layer transformer_ TF **_θ_** _with_ 



_such that_ TF **_θ_** _maps_ 



_where f_<sup>�</sup> =<sup>�</sup><sup>_K_</sup> _k_ =1<sup>_λkfkisanaggregatedpredictor,wheretheweightsλ_1</sup><sup>_, · · ·, λK≥_0</sup><sup>_arefunctionsonlyon_</sup> L1 _, · · · ,_ L _k such that_ 



_Proof of Proposition H.2._ We construct a **_θ_** which is a composition of 2 MLP layers followed by an attention layer ( **_θ_** `mlp`<sup>(1)</sup><sup>_,_</sup><sup>**_θ_**</sup> `mlp`<sup>(2)</sup><sup>_,_</sup><sup>**_θ_**</sup> `attn`<sup>(3)).</sup> 

Step 1: construction of **_θ_** `mlp`<sup>(1).Weconsidermatrix</sup><sup>**W**</sup> 1<sup>(1)</sup> that maps 



69 



i.e. **W** 1<sup>(1)</sup><sup>**h**isa</sup><sup>_K_2 +</sup><sup>_K_dimensionalvectorsothatitsentrycontains</sup><sup>_{_L</sup><sup>_k −_L</sup><sup>_l}k,l∈_[</sup><sup>_K_]and</sup><sup>_{_L</sup><sup>_k, −_L</sup><sup>_k}k∈_[</sup><sup>_K_].</sup> (1) Clearly, such **W** 1<sup>(1)</sup> exists and can be chosen so that **W** 1<sup>Wethenconsideramatrix</sup><sup>**W**</sup> 2<sup>(1)</sup> that ��� ���op<sup>_≤_2</sup><sup>_K._</sup> maps 





(1) and hence such **W** 2<sup>(1)</sup> exists and can be chosen so that ��� **W** 2 ���op<sup>_≤K_+1.We set</sup><sup>**_θ_**</sup> `mlp`<sup>(1)= (</sup><sup>**W**</sup> 1<sup>(1)</sup><sup>_,_</sup><sup>**W**</sup> 2<sup>(1)), then</sup> MLP **_θ_** `mlp` (1)<sup>maps</sup><sup>**h**</sup><sup>_i_to</sup> 



The basic property of _{ck}k∈_ [ _K_ ] is that, if _ck ≤ γ_ , then L _k ≤_ min _k⋆∈_ [ _K_ ] L _k⋆_ + _γ_ . Step 2: construction of **_θ_** `mlp`<sup>(2).Weconsidermatrix</sup><sup>**W**</sup> 1<sup>(2)</sup> that maps 



(2) and **W** 1<sup>(2)</sup> can be chosen so that **W** 1<sup>Wethenconsideramatrix</sup><sup>**W**</sup> 2<sup>(2)</sup> that maps ��� ���op<sup>_≤K_+ 1 +</sup><sup>_γ−_1</sup><sup>_._</sup> 

_σ_ ( **W** 1<sup>(2)</sup><sup>**h**)</sup><sup>_�→_</sup><sup>**W**</sup> 2<sup>(2)</sup><sup>_σ_(</sup><sup>**W**</sup> 1<sup>(1)</sup><sup>**h**) = [</sup><sup>**0**</sup><sup>_D−K−_3;</sup><sup>_σ_(1</sup><sup>_−γ−_1</sup><sup>_c_1)</sup><sup>_−c_1;</sup><sup>_· · ·_;</sup><sup>_σ_(1</sup><sup>_−γ−_1</sup><sup>_cK_)</sup><sup>_−cK_;</sup><sup>**0**3]</sup><sup>_∈_R</sup><sup>_D,_</sup> (1) which exists and can be chosen so that ��� **W** 2 ���op<sup>_≤_2.We set</sup><sup>**_θ_**</sup> `mlp`<sup>(2)= (</sup><sup>**W**</sup> 1<sup>(2)</sup><sup>_,_</sup><sup>**W**</sup> 2<sup>(2)),then MLP</sup> **_θ_** `mlp`<sup>(2)maps</sup><sup>**h**</sup> _i_<sup>(1)</sup> to 



where _uk_ = _σ_ (1 _− γ_<sup>_−_1</sup> _ck_ ) _∀k ∈_ [ _K_ ]. Clearly, _uk ∈_ [0 _,_ 1], and _uk >_ 0 if and only if _ck ≤ γ_ . Step 3: construction of **_θ_** `attn`<sup>(3).Wedefine</sup> 

_λ_ 1 = 1 _− σ_ (1 _− u_ 1) _, λk_ = _σ_ (1 _− u_ 1 _−· · · − uk−_ 1) _− σ_ (1 _− u_ 1 _−· · · − uk_ ) _∀k ≥_ 2 _._ Clearly, _λk ≥_ 0, and<sup>�</sup> _k_<sup>_λk_= 1.Further,</sup> 

_λk >_ 0 _⇒ uk >_ 0 _⇒ ck ≤ γ ⇒_ L _k ≤_ min _k_<sup>_⋆_</sup> _∈_ [ _K_ ]<sup>L</sup><sup>_k⋆_+</sup><sup>_γ._</sup> 

Therefore, it remains to construct **_θ_** `attn`<sup>(3)thatimplements</sup><sup>_f_�= �</sup><sup>_K_</sup> _k_ =1<sup>_λkfk_basedon[</sup><sup>**h**(2)</sup> _i_<sup>]</sup><sup>_i_.Noticethat</sup> 



and hence we construct **_θ_** `attn`<sup>(3)asfollows:</sup> for every _k ∈_ [ _K_ + 1] and _w ∈{_ 0 _,_ 1 _}_ , we define matrices **Q** _k,w,_ **K** _k,w,_ **V** _k,w ∈_ R<sup>_D×D_</sup> such that for all _k ∈_ [ _K_ + 1] 



70 



for all _i, j ∈_ [ _N_ + 1], where we understand _f_ 0 = _fK_ +1 = 0 and **1** _k_ is the _k_ -dimensional vector with all entries being 1. By the structure of **h**<sup>(2)</sup> _i_<sup>,thesematricesindeedexist,andfurtheritisstraightforwardtocheckthat</sup> they have norm bounds 



Now, for every _i, j ∈_ [ _N_ + 1], _k ∈_ [ _K_ + 1] _, w ∈{_ 0 _,_ 1 _}_ , we have 



where the last equality follows from _fk_ ( **x** _i_ ) + _R ≥_ 0 _∀k ∈_ [ _K_ ]. Therefore, 



where the last equality is due to (42). Thus letting the attention layer **_θ_** `attn`<sup>(3)=</sup><sup>_{_(</sup><sup>**V**</sup> _k,w_<sup>_,_</sup><sup>**Q**</sup> _k,w_<sup>_,_</sup><sup>**K**</sup> _k,w_<sup>)</sup><sup>_}_</sup> ( _k,w_ ) _∈_ [ _K_ +1] _×{_ 0 _,_ 1 _}_<sup>,</sup> we have 



This is the desired result. 

Now, we are ready to prove Theorem H.1. 

**Proof of Theorem H.1** As _ℓ_ ( _·, ·_ ) is ( _γ/_ 3 _, R, M, C_ )-approximable by sum of relus, we can invoke Proposition H.1 to show that there exists a single attention layer **_θ_** `attn`<sup>(1)sothatAttn</sup> **_θ_** `attn`<sup>(1)maps</sup> 



for any input **H** = [ **h** _i_ ] _i_ of the form described in Theorem H.1, and _L_<sup>�</sup> val( _·_ ) is a functional such that � max _k L_ val( _fk_ ) _− L_ val( _fk_ ) _≤ γ/_ 3. ���� ��� 

Next, by the proof of Proposition H.2, there exists ( **_θ_** `mlp`<sup>(1)</sup><sup>_,_</sup><sup>**_θ_**</sup> `mlp`<sup>(2)</sup><sup>_,_</sup><sup>**_θ_**</sup> `attn`<sup>(3))thatmaps</sup> 



71 



This completes the proof. 

### **H.2 Proof of Theorem 16** 

We first restate Theorem 16 into the following version which provides additional size bounds for **_θ_** . For the simplicity of presentation, throughout this subsection and Appendix I, we denote _It_ = _{i_ : ( **x** _i, yi_ ) _∈D_ train _}_ , _Iv_ = _{i_ : ( **x** _i, yi_ ) _∈D_ val _}_ , **X** train = [ **x** _i_ ] _i∈It_ to be the input matrix corresponding to the training split only, and _N_ train = _|D_ train _|_ , _N_ val = _|D_ val _|_ . 

**Theorem H.2.** _For any sequence of regularizations {λk}k∈_ [ _K_ ] _,_ 0 _≤ α ≤ β with κ_ := max _k αβ_ <u>++</u> _λλkk_<sup>_,Bw>_0</sup><sup>_,_</sup> _γ >_ 0 _, and ε < Bw/_ 2 _, suppose in input format (3) we have D ≥_ Θ( _Kd_ ) _. Then there exists an L-layer transformer_ TF **_θ_** _with_ 



_such that the following holds. On any input data_ ( _D,_ **x** _N_ +1) _such that the problem (ICRidge) is wellconditioned and has a bounded solution:_ 



TF<sup>0</sup> **_θ_**<sup>_approximatelyimplementsridgeselection:itsprediction_</sup> 



_satisfies the following._ 





In particular, if we set _γ_<sup>_′_</sup> = 2( _BxBw_ + _By_ ) _Bxε_ + _γ_ , then it holds that<sup>6</sup> 



> 6This is because _L_ �val( **w** ) is ( _BxBw_ + _By_ ) _Bx_ -Lipschitz w.r.t. **w** _∈_ B2( _Bw_ ). 

72 

� where we denote **w** ridge<sup>_λk_</sup> _,_ train<sup>:=</sup><sup>**w**</sup> ridge<sup>_λk_(</sup><sup>_D_train).</sup> 

To prove Theorem H.2, we first show that, for the squared validation loss, there exists a 3-layer transformer that performs predictor selection based on the _exactly_ evaluated _L_<sup>�</sup> val( _fk_ ) for each _k ∈_ [ _K_ ]. (Proof in Appendix H.2.1.) 

**Theorem H.3** (Square-loss version of Theorem H.1) **.** _Consider the squared validation loss_ 



_Then there exists a 3-layer transformer_ TF **_θ_** _with_ 

_such that for any input_ **H** _that takes form_ 



_where_ TF **_θ_** _outputs_ **h** _N_ +1 = [ **x** _N_ +1; _f_<sup>�</sup> ( **x** _N_ +1); _∗_ ; 1; 0] _, where the predictor f_<sup>�</sup> : R<sup>_d_</sup> _→_ R _is a convex combination of {fk_ : _L_<sup>�</sup> val( _fk_ ) _≤_ min _k⋆∈_ [ _K_ ] _L_<sup>�</sup> val( _fk⋆_ ) + _γ}. As a corollary, for any convex risk L_ : (R<sup>_d_</sup> _→_ R) _→_ R _, f_<sup>�</sup> _satisfies_ 



**Proof of Theorem H.2** First, by the proof<sup>7</sup> of Theorem 4 and Proposition A.6, for each _k ∈_ [ _K_ ], there exists a _T_ = _L −_ 3 layer transformer **_θ_**<sup>(1:</sup><sup>_T_)</sup> such that TF **_θ_** (1: _T_ ) maps 



so that if (43) holds, we have ��� **w** _k −_ **w** ridge _λk_ ��2<sup>_≤ε_and</sup><sup>**w**�</sup><sup>_k∈_B2(</sup><sup>_Bw_).</sup> 

Next, by Theorem H.3, there exists a 3-layer transformer **_θ_**<sup>(</sup><sup>_T_+1:</sup><sup>_T_+3)</sup> that outputs 







This is the desired result. 

#### **H.2.1 Proof of Theorem H.3** 

Similar to the proof of Proposition 15, Theorem H.3 is a direct corollary by combining Proposition H.3 with Proposition H.2. 

**Proposition H.3** (Evaluation layer for the squared loss) **.** _There exists an attention layer_ TF **_θ_** _with_ 2 _K heads and |||_ **_θ_** _||| ≤_ 3 _R_ + 2 _NK/ |D_ val _| such that_ TF **_θ_** _maps_ 



> 7Technically, an adapted version where the underlying ICGD mechanism operates on the training split (with _ti_ = 1) with size _N_ train instead of on all _N_ training examples, which only changes _|||_ **_θ_** _|||_ by at most a constant factor, and does not change the number of layers and heads. 

73 

_Proof of Proposition H.3._ For every _k ∈_ [ _K_ ], we define matrices **Q** _m,k,_ **K** _m,k,_ **V** _m,k ∈_ R<sup>_D×D_</sup> such that for all _i, j ∈_ [ _N_ + 1], 



As the input has structure **h** _i_ = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>_∗_;</sup><sup>_f_1(</sup><sup>**x**</sup><sup>_i_);</sup><sup>_· · ·_;</sup><sup>_fK_(</sup><sup>**x**</sup><sup>_i_);</sup><sup>**0**</sup><sup>_K_+1; 1;</sup><sup>_ti_],thesematricesindeedexist,and</sup> further it is straightforward to check that they have norm bounds 



Now, for every _i, j ∈_ [ _N_ + 1], we have 



where the second equality follows from the bound _|fk_ ( **x** _j_ ) _− yj| ≤_ 2 _R_ , so that the relus equals 0 if _tj ≤_ 0. Thus letting the attention layer **_θ_** = _{_ ( **V** _k,w,_ **Q** _k,w,_ **K** _k,w_ ) _}_ ( _k,w_ ) _∈_ [ _K_ ] _×{_ 0 _,_ 1 _}_ , we have 



= [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>_∗_;</sup><sup>_f_1(</sup><sup>**x**</sup><sup>_i_);</sup><sup>_· · ·_;</sup><sup>_fK_(</sup><sup>**x**</sup><sup>_i_);</sup><sup>**0**</sup><sup>_K_+1; 1;</sup><sup>_ti_] + [</sup><sup>**0**</sup><sup>_D−K−_3; �</sup><sup>_L_val(</sup><sup>_f_1);</sup><sup>_· · ·_; �</sup><sup>_L_val(</sup><sup>_fK_); 0; 0; 0]</sup> = [ **x** _i_ ; _yi_<sup>_′_;</sup><sup>_∗_;</sup><sup>_f_1(</sup><sup>**x**</sup><sup>_i_);</sup><sup>_· · ·_;</sup><sup>_fK_(</sup><sup>**x**</sup><sup>_i_); �</sup><sup>_L_val(</sup><sup>_f_1);</sup><sup>_· · ·_; �</sup><sup>_L_val(</sup><sup>_fK_); 0; 1;</sup><sup>_ti_]</sup><sup>_,_</sup> _i ∈_ [ _N_ + 1] _._ This is the desired result. 

### **H.3 Proofs for Section 4.2** 

#### **H.3.1 Proof of Lemma 18** 

It is straightforward to check that the binary type check _ψ_ : R _→_ R can be expressed as a linear combination of 6 relu’s (recalling _σ_ ( _·_ ) = ReLU( _·_ )): 





with<sup>�</sup> _m_<sup>_|am|_= 8</sup><sup>_/ε_, max</sup><sup>_m_max</sup><sup>_{|bm|, |cm|} ≤_2.We can thus construct an attention layer</sup><sup>**_θ_**=</sup><sup>_{_(</sup><sup>**Q**</sup><sup>_m,_</sup><sup>**K**</sup><sup>_m,_</sup><sup>**V**</sup><sup>_m_)</sup><sup>_}_6</sup> _m_ =1 with 6 heads such that 



which gives that for every _i ∈_ [ _N_ + 1], 



Further, we have _|||_ **_θ_** _||| ≤_ 18 _/ε_ = _O_ (1 _/ε_ ). This is the desired result. 

By composing the above attention layer with one additional layer (with 2 heads) that implement the following function 



on the output Ψ<sup>binary</sup> ( _D_ ), we directly obtain the following corollary. 

**Corollary H.1** (Thresholded binary test) **.** _There exists a two-layer attention-only transformer with_ max _ℓ∈_ [2] _M_<sup>(</sup><sup>_ℓ_)</sup> _≤_ 6 _and |||_ **_θ_** _||| ≤O_ (1 _/ε_ ) _that exactly implements the thresholded binary test_ 



_at every token i ∈_ [ _N_ + 1] _, where we recall the definition of_ Ψ<sup>binary</sup> _in Lemma 18._ 

#### **H.3.2 Formal statement and proof of Proposition 19** 

We say a distribution P _y_ on R is ( _C, ε_ 0)-not-concentrated around _{_ 0 _,_ 1 _}_ if 



for all _ε ∈_ (0 _, ε_ 0]. A sufficient condition is that the density p _y_ is upper bounded by _C_ within [ _−ε_ 0 _, ε_ 0] _∪_ [1 _− ε_ 0 _,_ 1 + _ε_ 0]. 

Throughout this section, let _σ_ log( _t_ ) := (1 + _e_<sup>_−t_</sup> )<sup>_−_1</sup> denote the sigmoid activation, and let **w** � log denote the solution to the in-context logistic regression problem, i.e. (ICGLM) with _g_ ( _·_ ) = _σ_ log( _·_ ). 

**Proposition H.4** (Adaptive regression or classification; Formal version of Proposition 19) **.** _For any Bw >_ 0 _, ε ≤ BxBw/_ 10 _,_ 0 _< α ≤ β with κ_ := _β/α, and any_ ( _C, ε_ 0) _, there exists a L-layer attention-only transformer with_ 



_(with R_ := max _{BxBw, By,_ 1 _}, and ε depending only on_ ( _C, ε_ 0) _) such that the following holds. Suppose the input format is (3) with dimension D ≥_ 3 _d_ + 4 _._ 

75 

_ticOn regressionany_ classification _in the senseinstanceof (7)_ ( _D, ,it_ **x** _Noutputs_ +1) _(suchy_ � _N_ +1 _thatthat{yiε}-approximatesi∈_ [ _N_ ] _⊂{_ 0 _,_ 1 _})thethatpredictionis well-conditionedof in-contextforlogisticlogisregression:_ 



_On the contrary, for_ regression problems _, i.e. any in-context distribution_ P _whose marginal_ P _y is_ ( _C, ε_ 0) _- not-concentrated around {_ 0 _,_ 1 _}, with probability at least_ 1 _−_ exp( _−cN_ ) _over D (where c >_ 0 _depends only on_ ( _C, ε_ 0) _), y_ � _N_ +1 _ε-approximates the prediction of in-context least squares if the data is well-conditioned:_ 



� _where_ **w** LS _denotes the in-context least squares estimator, i.e. (ICRidge) with λ_ = 0 _._ 

_Proof._ The result follows by combining the binary test in Corollary H.1 with Theorem 4 and Theorem 7. By those results, there exists three attention-only transformers **_θ_** LS _,_ **_θ_** log _,_ **_θ_** bin, with (below _Lg, Cg_ = Θ(1) for _g_ = _σ_ log( _·_ )) 



that outputs prediction _y_ � _N_<sup>LS</sup> +1<sup>,</sup><sup>_y_�</sup> _N_<sup>log</sup> +1<sup>(atthe(</sup><sup>_N_+ 1)-thtoken)andΨbinary</sup> thres<sup>(</sup><sup>_D_)(ateverytoken)respectively,</sup> which satisfy 



� when the corresponding well-conditionednesses are satisfied. In particular, we can make **w** log well-defined on non-binary data, by multiplying Ψ<sup>binary</sup> thres<sup>(</sup><sup>_D_)ontothe</sup><sup>**x**</sup><sup>_i_’s(whichcanbeimplementedbyslightlymodifying</sup> � **_θ_** log without changing the order of the number of layers, heads, and norms) so that **w** log = **0** on any data where Ψ<sup>binary</sup> thres<sup>(</sup><sup>_D_) = 0.</sup> 

By joining **_θ_** LS and **_θ_** log using Proposition A.6, concatenating with **_θ_** bin before, and concatenating with one additional attention layer with 2 heads after to implement 



we obtain a single transformer **_θ_** with 



which outputs (45) as its prediction (at the location for _y_ � _N_ +1). 

It remains to show that (45) reduces to either one of _y_ � _N_<sup>log</sup> +1<sup>or</sup><sup>_y_�</sup> _N_<sup>LS</sup> +1<sup>.Whenthedataarebinary(</sup><sup>_yi∈{_0</sup><sup>_,_1</sup><sup>_}_),</sup> we have Ψ<sup>binary</sup> ( _D_ ) = 1 and Ψ<sup>binary</sup> thres<sup>(</sup><sup>_D_)=1,inwhichcase(45)becomesexactly</sup><sup>_y_�</sup> _N_<sup>log</sup> +1<sup>.Bycontrast,when</sup> data is sampled from a distribution that is ( _C, ε_ 0)-not-concentrated around _{_ 0 _,_ 1 _}_ , we have for any fixed _ε ≤ ε_ 0 _∧_ 41 _C_<sup>that,letting</sup><sup>_Bε_:= [</sup><sup>_−ε, ε_]</sup><sup>_∪_[1</sup><sup>_−ε,_1 +</sup><sup>_ε_]andp</sup><sup>_ε_:= P</sup><sup>_y_(</sup><sup>_Bε_)</sup><sup>_≤Cε ≤_</sup><sup><u>1</u></sup> 4<sup>,byHoeffding’sinequality,</sup> 



76 



where _c_<sup>_′_</sup> _>_ 0 is an absolute constant. On the event Ψ<sup>binary</sup> thres<sup>(</sup><sup>_D_) = 0(whichhappenswithprobabilityatleast</sup> � 1 _−_ exp( _−c_<sup>_′_</sup> _N_ )), (45) becomes exactly _yN_<sup>LS</sup> +1<sup>.Thisfinishestheproof.</sup> 

### **H.4 Linear correlation test and application** 

In this section, we give another instantiation of the pre-ICL testing mechanism by showing that the transformer can implement a _linear correlation test_ that tests whether the correlation vector E[ **x** _y_ ] has a large norm. We then use this test to construct a transformer to perform “confident linear regression”, i.e. output a prediction from linear regression only when the signal-to-noise ratio is high. 

For any fixed parameters _λ_ min _, Bw_<sup>_⋆>_0,considerthelinearcorrelationtestoverdata</sup><sup>_D_definedas</sup> 



Recall that _σ_ ( _·_ ) = ReLU( _·_ ) above denotes the relu activation. 

We show that Ψ<sup>lin</sup> can be exactly implemented by a 3-layer transformer. 

**Lemma H.1** (Expressing Ψ<sup>lin</sup> by transformer) **.** _There exists a 3-layer attention-only transformer_ TF **_θ_** _with at most_ 2 _heads per layer and |||_ **_θ_** _||| ≤O_ (1 + _λ_<sup>2</sup> min<sup>(</sup><sup>_B_</sup> _w_<sup>_⋆_)2)</sup><sup>_suchthatoninputsequence_</sup><sup>**H**</sup><sup>_oftheform(3)with_</sup> _D ≥_ 2 _d_ + 4 _, the transformer_ exactly _implements_ Ψ<sup>lin</sup> _: it outputs_ **H**<sup>�</sup> _such that_ **h**<sup>�</sup> _i_ = [ **x** _i_ ; _yiti_ ; _∗_ ; Ψ<sup>lin</sup> ( _D_ ); 1] _for all i ∈_ [ _N_ + 1] _._ 

- _Proof._ We begin by noting the following basic facts: 

- Identity function can be implemented exactly by two ReLUs: _t_ = _σ_ ( _t_ ) _− σ_ ( _−t_ ). 

- Squared _ℓ_ 2 norm can be implemented exactly by a single attention head (assuming every input **h** _i_ contains the same vector **g** ): _∥_ **g** _∥_<sup>2</sup> 2<sup>=</sup><sup>_σ_(</sup><sup>_⟨_</sup><sup>**g**</sup><sup>_,_</sup><sup>**g**</sup><sup>_⟩_).</sup> 

We construct the transformer **_θ_** as follows. 

Layer 1: Use 2 heads to implement<sup>�</sup> **t** = _N_<sup><u>1</u></sup> � _Ni_ =1<sup>**x**</sup><sup>_iyi_, where</sup><sup>**V**</sup> _{_<sup>(1)</sup> 1 _,_ 2 _}_<sup>**h**</sup><sup>_j_= [</sup><sup>_±_</sup><sup>**x**</sup><sup>_j_;</sup><sup>**0**</sup><sup>_D−d_],</sup><sup>**Q**(1)</sup> _{_ 1 _,_ 2 _}_<sup>**h**</sup><sup>_i_= [</sup><sup>_<u>N</u>_</sup> _N_<sup><u>+1</u>;</sup><sup>**0**</sup><sup>_D−_1],</sup> and **K**<sup>(1)</sup> _{_ 1 _,_ 2 _}_<sup>**h**</sup><sup>_j_= [</sup><sup>_±yjtj_;</sup><sup>**0**</sup><sup>_D−_1] = [</sup><sup>_±yj_1</sup><sup>_{j< N_+ 1</sup><sup>_}_;</sup><sup>**0**</sup><sup>_D−_1](wherewerecall</sup><sup>_tj_= 1</sup><sup>_{j< N_+ 1</sup><sup>_}_andnotethat</sup> _yjtj_ corresponds exactly to the location for _yj_ in **H** , cf. (3)). By manipulating the output dimension in **V**<sup>(1)</sup> , write the result<sup>�</sup> **t** into blank memory space with dimension _d_ at every token _i ∈_ [ _N_ + 1]. 

Layer 2: Use a single head to compute _∥_<sup>�</sup> **t** _∥_ 2<sup>2:</sup><sup>**Q**(2)</sup> 1<sup>**h**(1)</sup> _i_ = [<sup>�</sup> **t** ; **0** _D−d_ ], **K**<sup>(2)</sup> 1<sup>**h**(1)</sup> _j_ = [<sup>�</sup> **t** ; **0** _D−d_ ], and **V** 1<sup>(2)</sup><sup>**h**(1)</sup> _j_ = [1; **0** _D−_ 1]. By manipulating the output dimension in **V**<sup>(2)</sup> , write the result _∥_<sup>�</sup> **t** _∥_ 2<sup>2intoblankmemoryspace</sup> with dimension 1 at every token _i ∈_ [ _N_ + 1]. After layer 2, we have **h**<sup>(3)</sup> _i_ = [ **x** _i_ ; _yiti_ ; _∗_ ; _∥_<sup>�</sup> **t** _∥_ 2<sup>2;</sup><sup>_∗_; 1].</sup> Layer 3: Use 2 heads to implement two ReLU functions with bias: _∥_<sup>�</sup> **t** _∥_ 2<sup>2</sup><sup>_�→_</sup> _B−_ <u>1</u> _A_<sup>(</sup><sup>_σ_(</sup><sup>_∥_�</sup><sup>**t**</sup><sup>_∥_</sup> 2<sup>2</sup><sup>_−A_)</sup><sup>_−σ_(</sup><sup>_∥_�</sup><sup>**t**</sup><sup>_∥_2</sup> 2<sup>_−B_)).</sup> The two query (or key) matrices contain values _A_ and _B_ . In our problem we take 



so that the above ReLU function implements Ψ<sup>lin</sup> ( _D_ ) exactly. Write the result into a blank memory space with dimension 1. We finish the proof by noting that _|||_ **_θ_** _||| ≤O_ (1 + _λ_<sup>2</sup> min<sup>(</sup><sup>_B_</sup> _w_<sup>_⋆_)2).</sup> 

77 

**Statistical guarantee for** Ψ<sup>lin</sup> We consider the following well-posedness assumption for the linear correlation test Ψ<sup>lin</sup> . Note that, similar as Assumption A, the assumption does not require the data to be generated from any true linear model, but rather only requires some properties about the best linear fit **w** P<sup>_⋆_,aswellas</sup> sub-Gaussianity conditions. 

**Assumption D** (Well-posedness for linear correlation test) **.** _We say a distribution_ P _on_ R<sup>_d_</sup> _×_ R _is well-posed for linear independence tests, if_ ( **x** _, y_ ) _∼_ P _satisfies_ 

- _(1) ∥_ **x** _∥_ 2 _≤ Bx and |y| ≤ By almost surely;_ 

- _(2) The covariance_ **Σ** P := EP[ **xx**<sup>_⊤_</sup> ] _satisfies λ_ min **I** _d ⪯_ **Σ** P _⪯ λ_ max **I** _d, with_ 0 _< λ_ min _≤ λ_ max _, and κ_ := _λ_ max _/λ_ min _._ 

- _(3) The whitened vector_ **Σ**<sup>_−_</sup> P<sup>1</sup><sup>_/_2</sup> **x** _is K_<sup>2</sup> _-sub-Gaussian for some K ≥_ 1 _._ 

- _(4) The best linear predictor_ **w** P<sup>_⋆_:= EP[</sup><sup>**xx**</sup><sup>_⊤_]</sup><sup>_−_1EP[</sup><sup>**x**</sup><sup>_y_]</sup><sup>_satisfies∥_</sup><sup>**w**</sup> P<sup>_⋆∥_2</sup><sup>_≤_</sup> _Bw_<sup>_⋆_</sup> _._ 

- _(5) The label y is σ_<sup>2</sup> _-sub-Gaussian._ 

- _(6) The residual z_ := _y −⟨_ **x** _,_ **w** P<sup>_⋆⟩isσ_2</sup><sup>_-sub-Gaussianwithprobabilityone(over_</sup><sup>**x**</sup><sup>_)._</sup> 

The following results states that Ψ<sup>lin</sup> achieves high power as long as the sample size is high enough, and the signal _∥_ **w** P<sup>_⋆∥_2iseithersufficientlyhighorsufficientlylow.</sup> 

**Proposition H.5** (Power of linear correlation test) **.** _Suppose distribution_ P _satisfies Assumption D with parameters λ_ min _, λ_ max _, Bw_<sup>_⋆_</sup> _. Then, for the linear correlation test_ Ψ<sup>lin</sup> _with parameters_ ( _λ_ min _, Bw_<sup>_⋆_)</sup><sup>_withB_</sup> _w_<sup>_⋆≤_</sup> _Bw_<sup>_⋆_</sup> _and any N ≥ O_<sup>�</sup> �max _{K_<sup>4</sup> _,_ (<sup>_<u>λ</u>_</sup> _B_<sup><u>max</u></sup> _w_<sup>_⋆_)2</sup><sup>_<u>Kλ</u>_22</sup> min<sup>_<u>σ</u>_2</sup><sup>_} · d_</sup> � _, we have_ 

_1. If ∥_ **w** P<sup>_⋆∥_2</sup><sup>_≥B_</sup> _w_<sup>_⋆,thenwithprobabilityatleast_1</sup><sup>_−δoverD,wehave_Ψlin(</sup><sup>_D_) = 1</sup><sup>_._</sup> 

_2. If ∥_ **w** P<sup>_⋆∥_2</sup><sup>_≤_</sup> 10 _<u>λλ</u>_ <u>minmax</u><sup>_B_</sup> _w_<sup>_⋆,thenwithprobabilityatleast_1</sup><sup>_−δoverD,wehave_Ψlin(</sup><sup>_D_) = 0</sup><sup>_._</sup> 

_Proof._ For any P satisfying Assumption D, note that E[ **x** _z_ ] = E[ **x** ( _y −⟨_ **w** P<sup>_⋆,_</sup><sup>**x**</sup><sup>_⟩_)]=</sup><sup>**0**byconstruction.</sup> Therefore, by standard sub-Gaussian and sub-exponential concentration combined with union bound, the following events hold simultaneously with probability at least 1 _− δ_ : 



On the above event, we have 



Therefore, in case 1, we have 

In case 2, we have 

78 

The proof is finished by recalling the definition of Ψ<sup>lin</sup> in (46), so that Ψ<sup>lin</sup> ( _D_ ) = 1 if _∥_<sup>�</sup> **t** _∥_ 2 _≥_ 3 _λ_ min _Bw_<sup>_⋆/_4,and</sup> Ψ<sup>lin</sup> ( _D_ ) = 0 if _∥_<sup>�</sup> **t** _∥_ 2 _≤ λ_ min _Bw_<sup>_⋆/_4.</sup> 

**Application: Confident linear regression** By directly composing the linear correlation test in Lemma H.1 with the transformer construction in Corollary 5 (using an argument similar as the proof of Proposition H.4), and using the power of the linear correlation test Proposition H.5, we immediately obtain the following result, which outputs a prediction from (approximately) least squares if _ψ_<sup>�</sup> := Ψ<sup>lin</sup> ( _D_ ) = 1, and abstains from predicting if _ψ_<sup>�</sup> = 0. This can be viewed as a form of “confident linear regression”, where the model predicts only if it thinks the linear signal is strong enough. 

**Proposition H.6** (Confident linear regression) **.** _For any Bw >_ 0 _,_ 0 _< Bw_<sup>_⋆≤_</sup> _Bw_<sup>_⋆_</sup> _,_ 0 _≤ λ_ min _≤ λ_ max _, ε ≤ BxBw/_ 10 _,_ 0 _< α ≤ β with κ_ := _β/α, there exists a L-layer attention-only transformer with_ 



_(with R_ := max _{BxBw, By,_ 1 _}) such that the following holds. Let N ≥ O_<sup>�</sup> �max _{K_<sup>4</sup> _,_ (<sup>_<u>λ</u>_</sup> _B_<sup><u>max</u></sup> _w_<sup>_⋆_)2</sup><sup>_<u>Kλ</u>_22</sup> min<sup>_<u>σ</u>_2</sup><sup>_} · d_</sup> � _. Suppose the input format is (3) with dimension D ≥_ 2 _d_ +4 _. Let ICL instance_ ( _D,_ **x** _N_ +1) _be drawn from any distribution_ P **h** � _Nsatisfying_ +1 _) Assumption D. Then the transformer outputs a 2-dimensional prediction (within the test token_ 



_such that the following holds:_ 

_1. If_ � _∥_ **w** P<sup>_⋆∥_2</sup><sup>_≥B_</sup> _w_<sup>_⋆,thenwithprobabilityatleast_1</sup><sup>_−δoverD,wehave|y_�</sup><sup>_N_+1</sup><sup>_−⟨_</sup><sup>**w**�LS</sup><sup>_,_</sup><sup>**x**</sup><sup>_N_+1</sup><sup>_⟩|≤ε,and_</sup> _ψ_ = 1 _if D is in addition well-conditioned for least squares (in the sense of (5) with λ_ = 0 _)._ 

_2. If ∥_ **w** P<sup>_⋆∥_2</sup><sup>_≤_</sup> 10 _<u>λλ</u>_ <u>minmax</u><sup>_B_</sup> _w_<sup>_⋆,thenwithprobabilityatleast_1</sup><sup>_−δoverD,wehavey_�</sup><sup>_N_+1= 0</sup><sup>_andψ_�= 0</sup><sup>_._</sup> 

## **I Proof of Theorem 17: Noisy linear model with mixed noise levels** 

For each fixed _k ∈_ [ _K_ ], we consider the following data generating model P _k_ , where we first sample P = iid P **w** _⋆,σk ∼ π_ from **w** _⋆ ∼_ N( **0** _,_ **I** _d/d_ ), and then sample data _{_ ( **x** _i, yi_ ) _}i∈_ [ _N_ +1] _∼_ P **w** _⋆,σk_ as 



Also, recall that the Bayes optimal estimator on P _k_ is given by _y_ � _N_<sup>Bayes</sup> +1<sup>=</sup> � **w** ridge<sup>_λk_(</sup><sup>_D_)</sup><sup>_,_</sup><sup>**x**</sup><sup>_N_+1</sup> � with ridge _λk_ = _σk_<sup>2</sup><sup>_d/N_,andtheBayesriskonP</sup><sup>_k_isgivenby</sup> 



Recall that in Section 4.1.1, we consider a mixture law P _π_ that generates data from P _k_ with _k ∼_ Λ. It is clear that we have (pushing inf _A_ into E _k∼_ Λ does not increase the value) we have 



i.e., the Bayes risk can only be greater if we consider a mixture of models. In other words, if a transformer can achieve near-Bayes ICL on each meta-task P _k_ , then it can perform near-Bayes ICL on any meta-task _π_ which is a mixture of P _k_ with _k ∼_ Λ. Therefore, to prove Theorem 17, it suffices to show the following (strengthened) result. 

79 

**Theorem I.1** (Formal version of Theorem 17) **.** _Suppose that N ≥_ 0 _._ 1 _d and we write σ_ max = max _k{σk,_ 1 _}, σ_ min = min _k{σk,_ 1 _}. Suppose in input format (3) we have D ≥_ Θ( _Kd_ ) _. Then there exists a transformer_ **_θ_** _with_ 



_such that for any k ∈_ [ _K_ ] _, it holds that_ 



_if we choose N_ val := _|D_ val _| ≍ N_<sup>2</sup><sup>_/_3</sup> [log _K_ ]<sup>1</sup><sup>_/_3</sup> _._ 

The core of the proof of Theorem I.1 is to show that any estimator **w** � that achieves small validation loss _L_<sup>�</sup> val must achieve small population loss. 

Throughout the rest of this section, recall that we define _N_ train = _|D_ train _| , N_ val = _|D_ val _|_ , _I_ train = _{i_ : ( **x** _i, yi_ ) _∈ D_ train _}_ , _I_ val = _{i_ : ( **x** _i, yi_ ) _∈D_ val _}_ , and **X** train = [ **x** _i_ ] _i∈I_ train. 

### **I.1 Proof of Theorem I.1** 

Fix parameters _δ, ε, γ >_ 0 and a large universal constant _C_ 0. Let us set 





Then, we define good events similarly to the proof of Corollary 6 (Appendix D.4): 



For the good event _E_ := _Eπ ∩Ew ∩Eb,_ train _∩Eb,_ test _∩Eb,N_ +1, we can show that P( _E_<sup>_c_</sup> ) _≤O_ � _N_<sup>_−_10�</sup> . Further, by the proof of Lemma D.1 (see e.g. (31)), we know that max _k∈_ [ _K_ ] �� **w** ridge _λk_<sup>(</sup><sup>_D_train)</sup> ��2<sup>_≤Bw/_2holdsunderthe</sup> good event _E_ . 

For the ridge _λk_ = _Ndσ_ train _<u>k</u>_<sup>2andparameters(</sup><sup>_α, β, γ, ε_),weconsiderthetransformer</sup><sup>**_θ_**constructedinTheo-</sup> � rem H.2, with a clipped prediction _yN_ +1 = read<sup>�</sup> y(TF **_θ_** ( **H** )). 

� In the following, we upper bound the quantity E _k_ ( _yN_ +1 _− yN_ +1)<sup>2</sup> for any fixed _k_ . Similar to the proof of Corollary 6 (Appendix D.4), we decompose 



and we analyze these two parts separately. 

80 

� � ments **Part I.** of TheoremRecall thatH.2byholdourforconstruction, **w** � . Thus, wewhenhave _E_ holds, we have _yN_ +1 = clip _By_ ( _⟨_ **w** _,_ **x** _N_ +1 _⟩_ ) and the state- 



Let us consider the following risk functional 



Then, under the good event _E_ 0 := _Eπ ∩Ew ∩Eb,_ train _∩Eb,_ test of ( **w** _⋆, D_ ), 



By our construction, under the good event _E_ 0, we have 

where 





where we denote 2Risk _k,_ train = E _k_ �� **w** ridge _λk_<sup>(</sup><sup>_D_train)</sup><sup>_−_</sup><sup>**w**</sup><sup>_⋆_</sup> ��22<sup>+</sup><sup>_σ_</sup> _k_<sup>2,andwealsonotethatRisk</sup><sup>_k,_train</sup><sup>_≤_1 +</sup><sup>_σ_</sup> _k_<sup>2by</sup> definition. By Lemma I.1, we have 



We next deal with the term _ε_ val := max _l∈_ [ _K_ ] _L_ val( **w** � _l_ ( _D_ train)) _− L_ val _,_ **w** _⋆_ ( **w** � _l_ ( _D_ train)) . Note that for the good ���� ��� event _E_ train := _Eπ ∩Ew ∩Eb,_ train of ( **w** _⋆, D_ train), we have 

E _k_ [1 _{E_ 0 _}ε_ val] _≤_ E _k_ [1 _{E_ train _}ε_ val] _≤_ E **w** _⋆,D_ train _∼_ P _k_ [1 _{E_ train _} ·_ E _D_ val [ _ε_ val _|_ **w** _⋆, D_ train]] _._ 

Thus, Lemma I.2 yields 



Therefore, we can conclude that 



81 

Therefore, we can choose ( _ε, N_ val) so that _N_ val _≤ N/_ 2 as 



It is worth noting that such choice of _N_ val is feasible as long as _N_ ≳ _σB_ max<sup>4</sup> _<u>w</u>_<sup>4log</sup><sup>_K_.Undersuchchoice,we</sup> obtain 



**Part II.** Similar to the proof of Corollary 6, we have 



**Conclusion.** Combining the both cases, we obtain 



where we plug in our choice of _By_ . The bounds on _M_<sup>(</sup><sup>_ℓ_)</sup> _, D_<sup>(</sup><sup>_ℓ_)</sup> and _|||_ **_θ_** _|||_ follows immediately from Theorem H.2. This completes the proof. 

### **I.2 Derivation of the exact Bayes predictor** 

Let ( _D,_ **x** _N_ +1 _, yN_ +1) be ( _N_ + 1) observations from the data generating model _π_ considered in Section 4.1.1. On observing ( _D,_ **x** _N_ +1), the Bayes predictor of _yN_ +1 is given by its posterior mean: 



It thus remains to derive E _π_ [ **w** _⋆|D_ ]. Recall that our data generating model is given by _k ∼_ Λ, By Bayes’ rule, we have 



On _k_ = _k_<sup>_′_</sup> , the data is generated from the noisy linear model **w** _⋆ ∼_ N( **0** _,_ **I** _d/d_ ), and **y** = **Xw** _⋆_ + **_ε_** where iid _εi ∼_ N(0 _, σk_<sup>2</sup><sup>_′_).ItisastandardresultthatE</sup><sup>_π_[</sup><sup>**w**</sup><sup>_⋆|D, k_=</sup><sup>_k′_]isgivenbytheridgeestimator</sup> 



(Note that the sample covariance within **Σ**<sup>�</sup> _k′_ is not normalized by _N_ , which is not to be confused with remaining parts within the paper.) Therefore, the posterior mean (47) is exactly a weighted combination of _K_ ridge regression estimators, each with regularization _dσk_<sup>2</sup><sup>_/N_.</sup> 

82 

It remains to derive the mixing weights P _π_ ( _k_ = _k_<sup>_′_</sup> _|D_ ) for all _k_<sup>_′_</sup> _∈_ [ _K_ ]. By Bayes’ rule, we have 



Note that such mixing weights involve the determinant of the matrix **Σ**<sup>�</sup> _k′_ = **X**<sup>_⊤_</sup> **X** + _dσk_<sup>2</sup><sup>_′_</sup><sup>**I**</sup><sup>_d_,whichdepends</sup> on the data **X** in a non-trivial fashion; Any transformer has to approximate these weights if their mechanism is to directly approximate the exact Bayesian predictor (47). 

### **I.3 Useful lemmas** 



_Proof._ Recall that under P _k_ , we have 

**w** _⋆ ∼_ N(0 _,_ **I** _d/d_ ) _, yi_ = _⟨_ **x** _i,_ **w** _⋆⟩_ + _εi,_ **_ε_** _i ∼_ N(0 _, σ_<sup>2</sup> ) _._ We denote **y** _t_ = [ _yi_ ] _i∈I_ train, then by definition **w** ridge<sup>_λk_(</sup><sup>_D_train)=(</sup><sup>**X**</sup><sup>_⊤_</sup> train<sup>**X**train+</sup><sup>_dσ_</sup> _k_<sup>2)</sup><sup>_−_1</sup><sup>**X**train</sup><sup>**y**</sup><sup>_t_(with</sup><sup>_λk_=</sup> _dσk_<sup>2</sup><sup>_/N_train).Thus,asimplecalculationyields</sup> 





where in the above inequality we denote **Σ** := **X**<sup>_⊤_</sup> train<sup>**X**train +</sup><sup>_dσ_</sup> _k_<sup>2</sup><sup>**I**</sup><sup>_d_andusethefollowingfact:</sup> 



**Case 1.** We first suppose that _N_ train _≤_ 16 _d_ . Then by definition **Σ** _⪰ dσk_<sup>2</sup><sup>**I**</sup><sup>_d_,andhence</sup> 



83 

<u>1</u> **Case 2.** When _N_ train _≥_ 9 _d_ , then we consider the event _Et_ := _{λ_ min( **X**<sup>_⊤_</sup> train<sup>**X**train</sup><sup>_/N_train)</sup><sup>_≥_</sup> 16<sup>_}_.By Lemma A.2</sup> we have P( _Et_<sup>_c_)</sup><sup>_≤_exp(</sup><sup>_−N_train</sup><sup>_/_8).Therefore,</sup> 



Combining these two cases finishes the proof. 

**Lemma I.2.** _Condition on the event E_ train _, we have_ 



� � _where we denote_ **w** _l_ = **w** _l_ ( _D_ train) _._ 

_Proof._ We only need to work with a fixed pair of ( **w** _⋆, D_ train) such that _E_ train holds. Hence, in the following we only consider the randomness of _D_ val conditional on such a ( **w** _⋆, D_ train). 

Recall that for any **w** , 



and we have E _Dv_ [ _L_<sup>�</sup> val( **w** )] = _L_ val _,_ **w** _⋆_ ( **w** ). For each _i ∈I_ val, 



Note that under _E_ train, we have **w** � _l ∈_ B2( _Bw_ ) for all _l ∈_ [ _K_ ], and hence _σk_<sup>2+</sup><sup>_∥_</sup><sup>**w**</sup><sup>_⋆−_</sup><sup>**w**�</sup><sup>_l∥_2</sup><sup>_≤_5</sup><sup>_B_</sup> _w_<sup>2.Wethen</sup> have ( _yi −⟨_ **x** _i,_ **w** � _l⟩_ )<sup>2</sup> ’s are (conditional) i.i.d random variables in SE( _CBw_<sup>4).Then,byBernstein’sinequality,</sup> we have 



where _c_ is a universal constant. Applying the union bound, we obtain 

Taking integration completes the proof. 

### **I.4 Generalized linear models with adaptive link function selection** 

Suppose that ( _gk_ : R _→_ R) _k∈_ [ _K_ ] is a set of link functions such that _gk_ is non-decreasing and _C_<sup>2</sup> -smooth for each _k ∈_ [ _K_ ]. We consider the input format we introduce in Section 4.1 with _|D_ train _|_ = _⌈N/_ 2 _⌉ , |D_ val _|_ = _⌊N/_ 2 _⌋_ . 

**Theorem I.2** (GLMs with adaptive link function selection) **.** _For any fixed set of parameters defined in Assumption B, as long as N ≥O_ ( _d_ ) _, there exists a transformer_ **_θ_** _with L ≤O_ (log( _N_ )) _layers, input dimension D_ = Θ ( _dK_ ) _and_ max _ℓ∈_ [ _L_ ] _M_<sup>(</sup><sup>_ℓ_)</sup> _≤ O_<sup>�</sup> � _d_<sup>3</sup> _N_ � _, such that the following holds._ 

_For any k_<sup>_⋆_</sup> _∈_ [ _K_ ] _and any distribution_ P _that is a generalized linear model of the link function gk⋆ and some parameter_ **_β_** _, if Assumption B holds for each pair_ (P _, gk_ ) _, then_ 





84 

_Proof._ For each _k ∈_ [ _K_ ], we consider optimizing the following training loss: 



_t_ where _ℓk_ ( _t, y_ ) := _−yt_ + �0<sup>_gk_(</sup><sup>_s_)</sup><sup>_ds_istheconvex(integral)lossassociatedwith</sup><sup>_gk_(asinSection3.2).</sup> Also, for each predictor _f_ : R<sup>_d_</sup> _→_ R, we consider the squared validation loss _L_<sup>�</sup> val: 



Fix a large universal constant _C_ 0. Let us set 





Then, we define good events similarly to the proof of Corollary 6 (Appendix D.4): 



Similar to the proof of Theorem 8 (Appendix E.2), we know the good event _E_ := _Ew∩Er∩Eb,_ train _∩Eb,_ test _∩Eb,N_ +1 holds with high probability: P( _E_<sup>_c_</sup> ) _≤O_ � _N_<sup>_−_10�</sup> . 

� Similar� to the proof of Theorem H.2, we can show that there exists a transformer **_θ_** with prediction _yN_ +1 = ready(TF **_θ_** ( **H** )) (clipped by _By_ ), such that (for any P) the following holds under _E_ : 



- 

- (b) _yN_ +1 = clip _By_ ( _f_<sup>�</sup> ( **x** _N_ +1)), where _f_<sup>�</sup> = _A_ TF( _D_ ) is an aggregated predictor given by _f_<sup>�</sup> =<sup>�</sup> _k_<sup>_λkfk_,such</sup> that ( _λk_ ) is a distribution supported on _k ∈_ [ _K_ ] such that _L_<sup>�</sup> val( _fk_ ) _≤_ min _k′∈_ [ _K_ ] _L_<sup>�</sup> val( _fk′_ ) + _γ._ 

Similar to the proof of Theorem 8, for _E_ 0 := _Ew ∩Er ∩Eb,_ train _∩Eb,_ test, we have 



where we denote _L_ val( _f_ ) := E( **x** _,y_ ) _∼_ P�1 _{∥_ **x** _∥_ 2 _≤ Bx}_ ( _f_ ( **x** ) _− y_ )<sup>2�</sup> for each predictor _f_ . By the definition of _f_<sup>�</sup> , we then have (under _E_ 0) 



For the first term, repeating the argument in the proof of Theorem 8 directly yields that for _E_ train := _Ew ∩Er ∩Eb,_ train, 



85 

For the second term, similar to Lemma I.2, we can show that conditional on _D_ train such that _E_ train holds, it holds 



Combining these inequalities and suitably choosing _γ_ complete the proof. 

## **J Proofs for Section 5** 

### **J.1 Lipschitzness of transformers** 

For any _p ∈_ [1 _, ∞_ ], let _∥_ **H** _∥_ 2 _,p_ := (<sup>�</sup><sup>_N_</sup> _i_ =1<sup>_∥_</sup><sup>**h**</sup><sup>_i∥p_</sup> 2<sup>)1</sup><sup>_/p_denotethecolumn-wise(2</sup><sup>_, p_)-normof</sup><sup>**H**.Foranyradius</sup> R _>_ 0, we denote _H_ R := _{_ **H** : _∥_ **H** _∥_ 2 _,∞ ≤_ R _}_ be the ball of radius R under norm _∥·∥_ 2 _,∞_ . 

**Lemma J.1.** _For a single MLP layer_ **_θ_** `mlp` = ( **W** 1 _,_ **W** 2) _, we introduce its norm (as in_ (2) _)_ 



_For any fixed hidden dimension D_<sup>_′_</sup> _, we consider_ 



_Then for_ **H** _∈H_ R _,_ **_θ_** `mlp` _∈_ Θ `mlp` _,B, the function_ ( **_θ_** `mlp` _,_ **H** ) _�→_ MLP **_θ_** `mlp` ( **H** ) _is_ ( _B_ R) _-Lipschitz w.r.t._ **_θ_** `mlp` _and_ (1 + _B_<sup>2</sup> ) _-Lipschitz w.r.t._ **H** _._ 

_Proof._ Recall that by our definition, for the parameter **_θ_** `mlp` = ( **W** 1 _,_ **W** 2) _∈_ Θ `mlp` _,B_ and the input **H** = [ **h** _i_ ] _∈_ R<sup>_D×N_</sup> , the output MLP **_θ_** `mlp` ( **H** ) = **H** + **W** 2 _σ_ ( **W** 1 **H** ) = [ **h** _i_ + **W** 2 _σ_ ( **W** 1 **h** _i_ )] _i_ . Therefore, for _θ_ `mlp`<sup>_′_=</sup> ( **W** 1<sup>_′,_</sup><sup>**W**</sup> 2<sup>_′_)</sup><sup>_∈_Θ</sup><sup>`mlp`</sup><sup>_,B_,wehave</sup> 





**Lemma J.2.** _For a single attention layer_ **_θ_** `attn` = _{_ ( **V** _m,_ **Q** _m,_ **K** _m_ ) _}m∈_ [ _M_ ] _⊂_ R<sup>_D×D_</sup> _, we introduce its norm (as in_ (2) _)_ 



86 

_For any fixed dimension D, we consider_ 



_Then for_ **H** _∈H_ R _,_ **_θ_** `attn` _∈_ Θ `attn` _,B, the function_ ( **_θ_** `attn` _,_ **H** ) _�→_ Attn **_θ_** `attn` ( **H** ) _is_ ( _B_<sup>2</sup> R<sup>3</sup> ) _-Lipschitz w.r.t._ **_θ_** `attn` _and_ (1 + _B_<sup>3</sup> R<sup>2</sup> ) _-Lipschitz w.r.t._ **H** _._ 

_Proof._ Recall that by our definition, for the parameter **_θ_** `attn` = _{_ ( **V** _m,_ **Q** _m,_ **K** _m_ ) _}m∈_ [ _M_ ] _∈_ Θ `attn` _,B_ and the input **H** = [ **h** _i_ ] _∈_ R<sup>_D×N_</sup> , the output Attn **_θ_** `attn` ( **H** ) = [ **h**<sup>�</sup> _i_ ] is given by 



Now, for _θ_ `attn`<sup>_′_=</sup><sup>_{_(</sup><sup>**V**</sup> _m_<sup>_′,_</sup><sup>**Q**</sup><sup>_′_</sup> _m_<sup>_,_</sup><sup>**K**</sup><sup>_′_</sup> _m_<sup>)</sup><sup>_}_</sup> _m∈_ [ _M_ ]<sup>,weconsider</sup> 



� _′_ Clearly ��Attn **_θ_** `attn` ( **H** ) _−_ Attn _θ_ `attn` _′_<sup>(</sup><sup>**H**)</sup> ��2 _,∞_<sup>= max</sup><sup>_i_</sup> ���� **h** _i −_ **h** _i_ ���2<sup>.Forany</sup><sup>_i ∈_[</sup><sup>_N_],wehave</sup> 



where the second inequality uses the definition of operator norm, the third inequality follows from the triangle inequality, the forth inequality is because _∥_ **Q** _m_ **h** _i∥_ 2 _≤ B_ R _, ∥_ **K** _m_ **h** _j∥_ 2 _≤ B_ R, and _σ_ is 1-Lipschitz. This completes the proof the Lipschitzness w.r.t. **_θ_** `attn` . 

Similarly, we consider **H**<sup>_′_</sup> = [ **h**<sup>_′_</sup> _i_<sup>],and</sup> 



87 

By definition, we can similarly bound 



where the last inequality uses _|||_ **_θ_** `attn` _||| ≤ B_ and the AM-GM inequality. This completes the proof the Lipschitzness w.r.t. **H** . 

**Corollary J.1.** _For a fixed number of heads M and hidden dimension D_<sup>_′_</sup> _, we consider_ 



_Then for the function_ TF<sup>R</sup> _given by_ 



TF<sup>R</sup> _is B_ Θ _-Lipschitz w.r.t_ **_θ_** _and LH -Lipschitz w.r.t._ **H** _, where B_ Θ := _B_ R(1 + _B_ R<sup>2</sup> + _B_<sup>3</sup> R<sup>2</sup> ) _and BH_ := (1 + _B_<sup>2</sup> )(1 + _B_<sup>2</sup> R<sup>3</sup> ) _._ 



where the second inequality follows from Lemma J.2 and Lemma J.1 and the fact that _∥_ Attn **_θ_** `attn` ( **H** ) _∥_ 2 _,∞ ≤_ R := R + _B_<sup>3</sup> R<sup>3</sup> for all **H** _∈H_ R. 

Furthermore, for **H**<sup>_′_</sup> _∈H_ R, we have 



which also follows from Lemma J.2 and Lemma J.1. 

88 

**Proposition J.1** (Lipschitzness of transformers) **.** _For a fixed number of heads M and hidden dimension D_<sup>_′_</sup> _, we consider_ 



_Then the function_ TF<sup>R</sup> _is_ ( _LBH_<sup>_L−_1</sup> _B_ Θ) _-Lipschitz w.r.t_ **_θ_** _∈_ ΘTF _,L,B for any fixed_ **H** _._ 

_Proof._ For **_θ_** = **_θ_**<sup>(1:</sup><sup>_L_)</sup> _∈_ ΘTF _,L,B,_ **_θ_**<sup>�</sup> = **_θ_**<sup>�(1:</sup><sup>_L_)</sup> _∈_ ΘTF _,L,B_ , we have 



where the second inequality follows from Corollary J.1, and the last inequality is because _BH ≥_ 1. 

### **J.2 Proof of Theorem 20** 

In this section, we prove a slightly more general result by considering the general ICL loss 



We assume that the loss function _ℓ_ satisfies sup _|ℓ| ≤ Bℓ_<sup>0andsup</sup><sup>_|∂_1</sup><sup>_ℓ|≤B_</sup> _ℓ_<sup>1.Forthespecialcase</sup><sup>_ℓ_(</sup><sup>_s, t_)=</sup> <u>1</u> 2<sup>(</sup><sup>_s −t_)2,wecantake</sup><sup>_B_</sup> _ℓ_<sup>0= 4</sup><sup>_B_</sup> _y_<sup>2</sup><sup>_, B_</sup> _ℓ_<sup>1= 2</sup><sup>_By._</sup> 

We then consider 



where **Z**<sup>(1:</sup><sup>_n_)</sup> are i.i.d copies of **Z** _∼_ P _,_ P _∼ π_ . It remains to apply Proposition A.4 to the random process _{X_ **_θ_** _}_ . We verify the preconditions: 

(a) By [84, Example 5.8], it holds that log _N_ ( _δ_ ; B _|||·|||_ ( _r_ ) _, |||·|||_ ) _≤ L_ (3 _MD_<sup>2</sup> +2 _DD_<sup>_′_</sup> ) log(1+2 _r/δ_ ), where B _|||·|||_ ( _r_ ) is any ball of radius _r_ under norm _|||·|||_ . 

(b) _|ℓ_ icl( **_θ_** ; **Z** ) _| ≤ Bℓ_<sup>0andhence</sup><sup>_B_</sup> _ℓ_<sup>0-sub-Gaussian.</sup> 

� 1 � (c) _ℓ_ icl( **_θ_** ; **Z** ) _− ℓ_ icl( **_θ_** ; **Z** ) _≤ Bℓ_<sup>_·_(</sup><sup>_LB_</sup> _H_<sup>_L−_1</sup> _B_ Θ) _·_ **_θ_** _−_ **_θ_** , by Proposition J.1. ��� ��� ��������� ��������� 

Therefore, we can apply the uniform concentration result in Proposition A.4 to obtain that, with probability at least 1 _− ξ_ , 



where _ι_ = log(2 + _B · LBH_<sup>_L−_1</sup> _B_ Θ _Bℓ_<sup>1</sup><sup>_/B_</sup> _ℓ_<sup>0)</sup><sup>_≤_20</sup><sup>_L_log(2 + max</sup><sup>_{B,_R</sup><sup>_, B_</sup> _ℓ_<sup>1</sup><sup>_/B_</sup> _ℓ_<sup>0</sup><sup>_}_).Recallingthat</sup> 

completes the proof. 

89 

### **J.3 Proof of Theorem 21** 

By Corollary 5, there exists a transformer TF **_θ_** such that for every P satisfying Assumption A with canonical � parameters� (and thus in expectation over P _∼ π_ ) and every _N ≥ O_<sup>�</sup> ( _d_ ), it outputs prediction _yN_ +1 = ready(TF **_θ_** ( **H** )) such that 



where we recall that _L_ P( **w** P<sup>_⋆_) :=</sup><sup><u>1</u></sup> 2<sup>E(</sup><sup>**x**</sup><sup>_,y_)</sup><sup>_∼_P</sup> �( _y −⟨_ **w** P<sup>_⋆,_</sup><sup>**x**</sup><sup>_⟩_)2�</sup> . By inspecting the proof, the same result holds if we change TF **_θ_** to the clipped version TF<sup>R</sup> **_θ_**<sup>ifwechooseR2=</sup><sup>_O_(</sup><sup>_B_</sup> _x_<sup>2+</sup><sup>_B_</sup> _y_<sup>2+</sup><sup>_B_</sup> _w_<sup>2+ 1) =</sup><sup>_O_(</sup><sup>_d_+</sup><sup>_κ_),sothaton</sup> the good event _E_ cov _∩ Ew_ considered therein, all intermediate outputs within TF **_θ_** has _∥·∥_ 2 _,∞ ≤_ R and thus the clipping does not modify the transformer output on _E_ cov _∩ Ew_ . Further, recall by (30) that **_θ_** has size bounds 



We can thus apply Theorem 20 to obtain that the solution **_θ_**<sup>�</sup> to (TF-ERM) with the above choice of ( _L, M, B_ ) and _D_<sup>_′_</sup> = 0 (attention-only) satisfies the following with probability at least 1 _− ξ_ : 



Above, _ι_ = _O_ (log(1 + max _{By,_ R _, B}_ )) = _O_<sup>�</sup> (1). This finishes the proof. 

### **J.4 Proof of Theorem 22** 

We invoke Theorem 11 (using the construction in Theorem 10 with a different choice of _L_ ) with the following parameters: 



where _O_<sup>�</sup> ( _·_ ) hides polylogarithmic factors in _d, N, Bw_<sup>_⋆, κ_.</sup> 

Then, Theorem 11 shows that there exists a transformer **_θ_** with _L_ layers, max _ℓ∈_ [ _L_ ] _M_<sup>(</sup><sup>_ℓ_)</sup> _≤ M_ heads, _D_<sup>_′_</sup> hidden dimension for the MLP layers, and _|||_ **_θ_** _||| ≤ B_ such that, on almost surely every P _∼ π_ , it returns a prediction � _yN_ +1 such that, on the good event _E_ 0 considered therein (over _D ∼_ P) which satisfies P( _E_ 0) _≥_ 1 _−δ_ , 



By inspecting the proof, the same result holds if we change TF **_θ_** to the clipped version TF<sup>R</sup> **_θ_**<sup>ifwechoose</sup> R<sup>2</sup> = _O_ ( _Bx_<sup>2+</sup><sup>_B_</sup> _y_<sup>2+ (</sup><sup>_B_</sup> _w_<sup>_⋆_)2 + 1)=</sup><sup>_O_(</sup><sup>_d_+ (</sup><sup>_B_</sup> _w_<sup>_⋆_)2 +</sup><sup>_σ_2),sothatonthegoodevent</sup><sup>_E_0consideredtherein,all</sup> intermediate outputs within TF **_θ_** has _∥·∥_ 2 _,∞ ≤_ R and thus the clipping does not modify the transformer output on the good event. On the bad event _E_ 0<sup>_c_,usingthesameargumentasintheproofofTheorem11,</sup> we have 



90 

Combining the above two bounds and further taking expectation over P _∼ π_ gives 



We can thus apply Theorem 20 to obtain that the solution **_θ_**<sup>�</sup> to (TF-ERM) with the above choice of ( _L, M, B, D_<sup>_′_</sup> ) satisfies the following with probability at least 1 _− ξ_ : 



Above, _ι_ = _O_ (log(1 + max _{By,_ R _, B}_ )) = _O_<sup>�</sup> (1). This finishes the proof. 

### **J.5 Proof of Theorem 23** 

We invoke Theorem 17 and Theorem I.1, which shows that (recalling the input dimension _D_ = Θ( _Kd_ )) there exists a transformer **_θ_** with the following size bounds: 



such that it outputs _y_ � _N_ +1 that satisfies 



By inspecting the proof, the same result holds if we change TF **_θ_** to the clipped version TF<sup>R</sup> **_θ_**<sup>ifwechoose</sup> R<sup>2</sup> = _O_ ( _Bx_<sup>2+</sup><sup>_B_</sup> _y_<sup>2+ (</sup><sup>_B_</sup> _w_<sup>_⋆_)2 + 1) =</sup><sup>_O_(</sup><sup>_d_+</sup><sup>_σ_</sup> max<sup>2),sothatonthegoodeventconsideredtherein,allintermediate</sup> outputs within TF **_θ_** has _∥·∥_ 2 _,∞ ≤_ R and thus the clipping does not modify the transformer output on the good event. Using this clipping radius, we obtain 



We can thus apply Theorem 20 to obtain that the solution **_θ_**<sup>�</sup> to (TF-ERM) with the above choice of ( _L, M, B, D_<sup>_′_</sup> ) satisfies the following with probability at least 1 _− ξ_ : 



Above, _ι_ = _O_ (log(1 + max _{By,_ R _, B}_ )) = _O_<sup>�</sup> (1). This finishes the proof. 

91 

### **J.6 Proof of Theorem 24** 

The proof follows from similar arguments as of Theorem 22 and Theorem 23, where we plug in the size bounds (number of layers, heads, and weight norms) from Theorem 8 and Corollary 9. 

## **K Experimental details** 

### **K.1 Additional details for Section 6.1** 

**Architecture and optimization** We train a 12-layer encoder-only transformer, where each layer consists of an attention layer as in Definition 1 with _M_ = 8 heads, hidden dimension _D_ = 64, and ReLU activation (normalized by the sequence length), as well as an MLP layer as in Definition 2 hidden dimension _D_<sup>_′_</sup> = 64. We add Layer Normalization [3] after each attention and MLP layer to help optimization, as in standard implementations [81]. We append linear read-in layer and linear read-out layer before and after the transformer respectively, both applying a same affine transform to all tokens in the sequence and are trainable. The read-in layer maps any input vector to a _D_ -dimensional hidden state, and the read-out layer maps a _D_ -dimensional hidden state to a 1-dimensional scalar. 

Each training sequence corresponds to a single ICL instance with _N_ in-context training examples _{_ ( **x** _i, yi_ ) _}_<sup>_N_</sup> _i_ =1<sup>_⊂_</sup> R<sup>_d_</sup> _×_ R and test input **x** _N_ +1 _∈_ R<sup>_d_</sup> . The input to the transformer is formatted as in (3) where each token has dimension _d_ + 1 (no zero-paddings). The transformer is trained by minimizing the following loss with fresh mini-batches: 



where the loss function _ℓ_ P : R<sup>2</sup> _→_ R may depend on the training data distribution P in general; we use the square loss when P is regression data, and the logistic loss when P is classification data. We use the Adam optimizer with a fixed learning rate 10<sup>_−_4</sup> , which we find works well for all our experiments. Throughout all our experiments except for the sparse linear regression experiment in Figure 4a, we train the model for 300K steps, where each step consists of a (fresh) minibatch with batch size 64 in the base mode, and _K_ minibatches each with batch size 64 in the mixture mode. 

For the sparse linear regression experiment, we find that minimizing the training objective (48) alone was not enough, e.g. for the learned transformer to achieve better loss than the least squares algorithm (which achieves much higher test loss than the Lasso; cf. Figure 4a). To help optimization, we augment (48) with another loss that encourages the second-to-last hidden states to recover the true (sparse) coefficient **w** _⋆_ : 



Specifically, the above loss encourages the first _N_ 0 _≤ N_ tokens within the second-to-last layer to be close to **w**<sup>_⋆_</sup> . We choose _N_ 0 = 5 (recall that the total number of tokens is _N_ = 10 and sequence length is _N_ + 1 = 11 for this experiment). We minimize the loss _L_ ( **_θ_** ) + _λL_ fit-w( **_θ_** ) with _λ_ = 0 _._ 1 for 2M steps for this task. 

**Evaluation** All evaluations are done on the trained transformer with 6400 test instances. We use the square loss for regression tasks, and the classification error (1 _−_ accuracy) between the true label _yN_ +1 _∈{_ 0 _,_ 1 _}_ and � the predicted label 1 _{yN_ +1 _≥_ 1 _/_ 2 _}_ . We report the means in all experiments, as well as their standard deviations (using one-std error bars) in Figure 2a, 2b, 5a, 5b. In Figure 2c, 4b, 4c 5c, all standard deviations are sufficiently small (not significantly exceeding the width of the markers), thus we did not show error bars in those plots. 

**Baseline algorithms** We implement various baseline machine learning algorithms to compare with the learned transformers. A superset of the algorithms is shown in Figure 4a: 

> • `Least squares` , `Logistic regression` : Standard algorithms for linear regression and linear classification, respectively. Note that least squares is also a valid algorithm for classification. 

92 

- `Averaging` � �: The simple algorithm which computes the linear predictor **w** � = _N_ <u>1</u> � _Ni_ =1<sup>_yi_</sup><sup>**x**</sup><sup>_i_andpredicts</sup> _yN_ +1 = _⟨_ **w** _,_ **x** _N_ +1 _⟩_ ; 

- `3-NN` : 3-Nearest Neighbors. 

- `Ridge` : Standard ridge regression as in (ICRidge). We specifically consider two _λ_ ’s (denoted as `lam_1` and `lam_2` ): _λ_ 1 _, λ_ 2 = (0 _._ 005 _,_ 0 _._ 125). These are the Bayes-optimal regularization strengths for the noise levels ( _σ_ 1 _, σ_ 2) = (0 _._ 1 _,_ 0 _._ 5) respectively under the noisy linear model (cf. Corollary 6), using the formula _λ_<sup>_⋆_</sup> = _dσ_<sup>2</sup> _/N_ , with ( _d, N_ ) = (20 _,_ 40). 

- `Lasso` : Standard Lasso as in (ICLasso) with _λ ∈{_ 1 _,_ 0 _._ 1 _,_ 0 _._ 01 _,_ 0 _._ 001 _}_ . 

In Figure 2c, the `ridge_analytical` curve plots the expected risk of ridge regression under the noisy linear model over 20 geometrically spaced values of _λ_ ’s in between ( _λ_ 1 _, λ_ 2), using analytical formulae (with Monte Carlo simulations). The `Bayes_err_{1,2}` indicate the expected risks of _λ_ 1 on task 1 (with noise _σ_ 1) and _λ_ 2 on task 2 (with noise _σ_ 2), respectively. 

### **K.2 Computational resource** 

All our experiments are performed on 8 Nvidia Tesla A100 GPUs (40GB memory). The total GPU time is approximately 5 days (on 8 GPUs), with the largest individual training run taking about a single day on a single GPU. 

93 

