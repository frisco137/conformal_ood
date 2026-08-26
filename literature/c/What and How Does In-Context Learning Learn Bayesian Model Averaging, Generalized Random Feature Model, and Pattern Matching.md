# What and How does In-Context Learning Learn? Bayesian Model Averaging, Parameterization, and Generalization 

Yufeng Zhang<sup>∗†</sup> , Fengzhuo Zhang<sup>∗‡</sup> , Zhuoran Yang,<sup>§</sup> Zhaoran Wang<sup>¶</sup> 

##### Abstract 

In this paper, we conduct a comprehensive study of In-Context Learning (ICL) by addressing several open questions: (a) What type of ICL estimator is learned by large language models? (b) What is a proper performance metric for ICL and what is the error rate? (c) How does the transformer architecture enable ICL? To answer these questions, we adopt a Bayesian view and formulate ICL as a problem of predicting the response corresponding to the current covariate, given a number of examples drawn from a latent variable model. To answer (a), we show that, without updating the neural network parameters, ICL implicitly implements the Bayesian model averaging algorithm, which is proven to be approximately parameterized by the attention mechanism. For (b), we analyze the ICL performance from an online learning perspective and establish a O(1/T ) regret bound for perfectly pretrained ICL, where T is the number of examples in the prompt. To answer (c), we show that, in addition to encoding Bayesian model averaging via attention, the transformer architecture also enables a fine-grained statistical analysis of pretraining under realistic assumptions. In particular, we prove that the error of pretrained model is bounded by a sum of an approximation error and a generalization error, where the former decays to zero exponentially as the depth grows, and the latter decays to zero sublinearly with the number of tokens in the pretraining dataset. Our results provide a unified understanding of the transformer and its ICL ability with bounds on ICL regret, approximation, and generalization, which deepens our knowledge of these essential aspects of modern language models. 

## 1 Introduction 

With the ever-increasing sizes of model capacity and corpus, Large Language Models (LLM) have achieved tremendous successes across a wide range of tasks, including natural language understanding (Dong et al., 2019; Jiao et al., 2019), symbolic reasoning (Wei et al., 2022c; 



> ∗equal contribution 

> †Northwestern University; yufengzhang2023@u.northwestern.edu 

> ‡National University of Singapore; fzzhang@u.nus.edu 

> §Yale University; zhuoranyang.work@gmail.com 

> ¶Northwestern University; zhaoranwang@gmail.com 

1 

Kojima et al., 2022), and conversations (Brown et al., 2020; Ouyang et al., 2022). Recent studies have revealed that these LLMs possess immense potential, as their large capacity allows for a series of emergent abilities (Wei et al., 2022b; Liu et al., 2023). One such ability is In-Context Learning (ICL), which enables an LLM to learn from just a few examples, without changing the network parameters. That is, after seeing a few examples in the prompt, a pretrained language model seems to comprehend the underlying concept and is able to extrapolate the understanding to new data points. 

Despite the tremendous empirical successes, theoretical understanding of ICL remains limited. Specifically, existing works fail to explain why LLMs the ability for ICL, how the attention mechanism is related to the ICL ability, and how pretraining influences ICL. Although the optimality of ICL is investigated in Xie et al. (2021) and Wies et al. (2023), these works both make unrealistic assumptions on the pretrained models, and their results cannot demystify the particular role played by the attention mechanism in ICL. 

In this work, we focus on the scenario where a transformer is first pretrained on a large dataset and then prompted to perform ICL. Our goal is to rigorously understand why the practice of “pretraining + prompting” unleashes the power of ICL. To this end, we aim to answer the following three questions: (a) What type of ICL estimator is learned by LLMs? (b) What are suitable performance metrics to evaluate ICL accurately and what are the error rates? (c) What is the role played by the transformer architecture during the pretraining and prompting stages? The first and the third questions demand scrutinizing the transformer architecture to understand how ICL happens during transformer prompting. The second question then requires statistically analyzing the extracted ICL process. Moreover, the third question — necessitates a holistic understanding beyond prompting we also need to characterize the statistical error of pretraining and how this error affects prompting. 

To address these questions, we adopt a Bayesian view and assume that the examples fed into a pretrained LLM are sampled from a latent variable model parameterized by a hidden concept z∗ ∈ Z. Moreover, the pretrained dataset contains sequences of examples from the same latent variable model, but with the concept parameter z ∈ Z itself randomly distributed according to a prior distribution. We mathematically formulate ICL as the problem of predicting the response of the response corresponding to the current covariate, where the prompt contains t examples of covariate-response pairs and the current covariate. 

Under such a setting, to answer (a), we show that the perfectly pretrained LLMs perform ICL in the form of Bayesian Model Averaging (BMA). That is, LLM first computes a posterior distribution of z∗ ∈ Z given the first t examples, and then predicts the response of the (t+1)-th covariate by aggregating over the posterior (Proposition 4.1). 

In addition, to answer (b), we adopt the online learning framework and define a notion called ICL regret, which is the averaged prediction error of ICL on a sequence of covariateresponse examples. We prove that the ICL regret after prompting t examples is O(1/t) up to the statistical error of the pretrained model (Theorem 6.2). 

Finally, to answer (c), we elucidate the role played by the transformer architecture in prompting and pretraining respectively. In particular, we show that a variant of attention mechanism encodes BMA in its architecture, which enables the transformer to perform ICL via prompting. Such an attention mechanism can be viewed as an extension of linear attention and coincides with the standard softmax attention (Garnelo and Czarnecki, 2023) when the length of the prompt goes to infinity. And thus we show that softmax attention Vaswani et al. (2017) approximately encodes BMA (Proposition 4.3). Besides, the transformer architecture enables a 

2 

fine-grained analysis of the statistical error incurred by pretraining. In particular, applying the PAC-Bayes framework, we prove that the error of the pretrained language model, measured via total variation, is bounded by a sum of approximation error and generalization error (Theorem 5.3). The approximation error decays to zero exponentially fast as the depth of the transformer increases (Proposition 5.4), while the generalization error decays to zero sublinearly with the number of tokens in the pretraining dataset. This features the first pretraining analysis of transformers in total variation distance, which also takes the approximation error into account. Furthermore, as an interesting extension, we also study the misspecified case where the response variables of the examples fed into the LLM are perturbed. We provide sufficient conditions for ICL to be robust to the perturbations and establish the finite-sample statistical error (Proposition G.4). 

– In sum, by addressing questions (a) (c), we provide a unified understanding of the ICL ability of LLMs and the particular role played by the attention mechanism. Our theory provides a holistic theoretical understanding of the regret, approximation, and generalization errors of ICL. 

## 2 Related Work 

In-Context Learning. After Brown et al. (2020) showcased the in-context learning (ICL) capacity of GPT-3, there has been a notable surge in interest towards enhancing and comprehending this particular ability (Dong et al., 2022). The ICL ability has seen enhancements through the incorporation of extra training stages (Min et al., 2021; Wei et al., 2021; Iyer et al., 2022), carefully selecting and arranging informative demonstrations (Liu et al., 2021; Kim et al., 2022; Rubin et al., 2021; Lu et al., 2021), giving explicit instructions (Honovich et al., 2022; Zhou et al., 2022b; Wang et al., 2022), and prompting a chain of thoughts (Wei et al., 2022c; Zhang et al., 2022b; Zhou et al., 2022a). In efforts to comprehend the mechanisms of ICL ability, researchers have also conducted extensive work. Empirically, Chan et al. (2022) demonstrated that the distributional properties, including the long-tailedness, are important for ICL. Garg et al. (2022) investigated the function class that ICL can approximate. Min et al. (2022) showed that providing wrong mappings between the input-output pairs in examples does not degrade the ICL. Theoretically, Aky¨urek et al. (2022), von Oswald et al. (2022), Bai et al. (2023), and Dai et al. (2022) indicated that ICL implicitly implements the gradient descent or least-square algorithms from the function approximation perspective. However, the first three works only showed that transformers are able to approximate these two algorithms, which may not align with the pretrained model. The last work ignored the softmax module, which turns out to be important in practical implementation. Feng et al. (2023) derived the impossibility results of ICL and the advantage of chain-of-thought for the function approximation. Li et al. (2023) viewed ICL from the multi-task learning perspective and derived the generalization bound. Hahn and Goyal (2023) built the linguistic model for sentences and used the description length to bound the ICL error with this model. Xie et al. (2021) analyzed ICL within the Bayesian framework, assuming the access to the nominal language distribution and that the tokens are generated from Hiddn Markov Model (HMM)s. However, the first assumption hides the relationship between pretraining and ICL, and the second assumption is restrictive. Following this thread, Wies et al. (2023) relaxed the HMM assumption and assumed access to a pretrained model that is close to the nominal distribution conditioned on any token sequence, which is also unrealistic. Two recent works Wang et al. 

3 

(2023), and Jiang (2023) also provide the Bayesian analysis of ICL. Unfortunately, these Bayesian works cannot explain the importance of the attention mechanism for ICL and clarify how pretraining is related to ICL. In contrast, we prove that the attention mechanism enables BMA by encoding it in the network architecture and we relate the pretraining error of transformers to the ICL regret. 

## 3 Preliminary 

Notation. We denote {1, · · · , N} as [N]. For a Polish space S, we denote the collection of all the probability measures on it as ∆(S). The total variation distance between two distributions P, Q ∈ ∆(S) is TV(P, Q) = supA⊆S |P (A) − Q(A)|. The i<sup>th</sup> entry of a vector x is denoted as xi or [x]i. For a matrix X ∈ R<sup>T×d</sup> , we index its i<sup>th</sup> row and column as Xi,: and X:,i respectively. The ℓp,q norm of X is defined as ∥X∥p,q = (<sup>�d</sup> i=1<sup>∥X:,i∥q</sup> p<sup>)1/q,andtheFrobeniusnormofitis</sup> defined as ∥X∥F = ∥X∥2,2. 

Attention and Transformers. Attention mechanism has been the most powerful and popular neural network module in both Computer Vision (CV) and Natural Language Processing (NLP) communities, and it is the backbone of the LLMs (Devlin et al., 2018; Brown et al., 2020). Assume that we have a query vector q ∈ R<sup>dk</sup> . With T key vectors in K ∈ R<sup>T×dk</sup> and T value vectors in V ∈ R<sup>T×dv</sup> , the attention mechanism maps the query vector q to attn(q, K, V ) = V<sup>⊤</sup> softmax(Kq), where softmax normalizes a vector via the exponential function, i.e., for x ∈ R<sup>d</sup> , [softmax(x)]i = exp(xi)/<sup>�d</sup> j=1<sup>exp(xj)fori∈[d].</sup> The output is a weighted sum of V , and the weights reflect the closeness between W and q. For t query vectors, we stack them into Q ∈ R<sup>t×dk</sup> . Attention maps these queries using the function attn(Q, K, V ) = softmax(QK<sup>⊤</sup> )V ∈ R<sup>t×dv</sup> , where softmax is applied row-wisely. In the practical design of transformers, practitioners usually use Multi-Head Attention (MHA) instead of single attention to express sophisticated functions, which forwards the inputs through h attention modules in parallel and outputs the sum of these submodules. Here h ∈ N is a hyperparameter. Taking X ∈ R<sup>T×d</sup> as the input, MHA outputs mha(X, W ) =<sup>�h</sup> i=1<sup>attn(XW</sup> i<sup>Q, XW</sup> i<sup>K, XW</sup> i<sup>V),whereW=(W</sup> i<sup>Q, W</sup> i<sup>K, W</sup> i<sup>V)h</sup> i=1<sup>istheparam-</sup> eters set of h attention modules, Wi<sup>Q</sup> ∈ R<sup>d×dh</sup> , Wi<sup>K</sup> ∈ R<sup>d×dh</sup> , and Wi<sup>V</sup> ∈ R<sup>d×d</sup> for i ∈ [h] are weight matrices for queries, keys, and values, and dh is usually set to be d/h (Michel et al., 2019). The transformer is the concatenation of the attention modules and the fully-connected layers, which is widely adopted in LLMs (Devlin et al., 2018; Brown et al., 2020). 

Large Language Models and In-Context Learning. Many LLMs are autoregressive, such as GPT (Brown et al., 2020). It means that the model continuously predicts future tokens based on its own previous values. For example, starting from a token x1 ∈ X, where X is the alphabet of tokens, a LLM Pθ with parameter θ ∈ Θ continuously predicts the next token according to xt+1 ∼ Pθ(· | St) based on the past St = (x1, · · · , xt) for t ∈ N. Here, each token represents a word and the position of the word (Ke et al., 2020), and the token sequences St for t ∈ N live in the sequences space X<sup>∗</sup> . LLMs are first pretrained on a huge body of corpus, making the prediction xt+1 ∼ Pθ(· | St) accurate, and then prompted to perform downstream tasks. During the pretraining phase, we aim to maximize the conditional probability Pθ(x | S) over the nominal next token x (Brown et al., 2020). 

After pretraining, LLMs are prompted to perform downstream tasks without tuning parameters. Different from the finetuned models that learn the task explicitly (Liu et al., 2023), LLMs can implicitly learn from the examples in the prompt, which is known as ICL (Brown et al., 2020). Concretely, pretrained LLMs are provided with a prompt promptt = (�c1, r1, . . . , �ct, rt, �ct+1) 

4 

with t examples and a query as inputs, where each pair (�ci, ri) ∈ X<sup>∗</sup> × X is an example of the task, and �ct+1 is the query, as shown in Figure 1 in Appendix C. For example, the promptt with t = 2 can be “Cats are animals, pineapples are plants, mushrooms are”. Here �c1 ∈ X<sup>∗</sup> is a token sequence “Cats are”, while r1 is the response “animals”. The query �ct+1 is “mushrooms are”, and the desired response is “fungi”. The prompts are generated from a hidden concept z∗ ∈ Z, e.g., z∗ can be the classification of biological categories, where Z is the concept space. � The generation process is ci ∼ P(· | �c1, r1, · · · , �ci−1, ri−1, z∗) and ri ∼ P(· | prompti−1, z∗) for the nominal distribution P and i ∈ [t]. Thus, when performing ICL, LLMs aim to estimate the conditional distribution P(rt+1|promptt, z∗). It is widely conjectured and experimentally found that the pretrained LLMs can implicitly identify the hidden concept z∗ ∈ Z from the examples, and then perform ICL by outputting from P(rt+1|promptt, z∗). In the following, we will provide theoretical justifications for this claim. We note that delimiters are omitted in our work, and our results can be generalized to handle this case. Since LLMs are autoregressive, the definition of the notation P(· | S) with S ∈ X<sup>∗</sup> may be ambiguous because the length of the subsequent tokens is not specified. Unless explicitly specified, we let P(· | S) denote the distribution of the next single token conditioned on S. 

## 4 In-Context Learning via Bayesian Model Averaging 

In this section, we show that LLMs perform ICL implicitly via BMA. Given a sequence S = {(�ct, rt)}<sup>T</sup> t=1<sup>with Texamples generated from a hidden concept z∗∈Z, we use St= {(�ci, ri)}t</sup> i=1 to represent the first t ICL examples in the sequence. Here �ct and rt respectively denote the ICL covariate and response. During the ICL phase, a LLM is sequentially prompted with promptt = (St, �ct+1) for t ∈ [T − 1], i.e., the first t examples and the (t + 1)-th covariate. The prompted LLM aims to predict the response rt+1 based on promptt = (St, �ct+1) whose true distribution is rt+1 ∼ P(· | promptt, z∗). For the analysis of ICL, we focus on the following latent variable model 



where the hidden variable ht ∈H determines the relation between ct and rt, ξt ∈ Ξ for t ∈ [T ] are i.i.d. random noises, and f : X ×H× Ξ → X is a function that relates response rt to �ct, ht, and ξt. In the data generation process, a hidden concept z∗ ∈ Z is first generated from P(z). The hidden variables {ht}<sup>T</sup> t=1<sup>arethenastochasticprocesswhosedistributionisdetermined</sup> by the hidden concept z∗, that is 



for some function gz∗ parameterized by z<sup>∗</sup> , where {ζt}t<sup>T</sup> =1<sup>areexogenousnoises.Theresponse</sup> rt is then generated according to (4.1). The model in (4.1) essentially assumes that the hidden � concept z∗ implicitly determines the transition of the conditional distribution P(rt = · | ct) by affecting the evolution of the latent variables {ht}t∈[T ], and it does not impose any assumption on the distribution of �ct. This model is quite general, and it subsumes the models in previous works. When f is the emission function in HMM and ht = h for t ∈ [T ] is the values of hidden states that depend on z, model in (4.1) recovers the HMM assumption in Xie et al. (2021). When ht = z for t ∈ [T ] degenerate to the hidden concept, this recovers the casual graph model in Wang et al. (2023) and the ICL model in Jiang (2023). 

5 

Assuming that the tokens follow the statistical model given in (4.1), during pretraining, we collect Np independent trajectories by sampling from (4.1) with concept z randomly sampled from P(z). Intuitively, during pretraining, by training in an autoregressive manner, the LLM approximates the conditional distribution P(rt+1 | promptt) = Ez∼P(z)[P(rt+1 | promptt, z)], which is the conditional distribution of rt+1 given promptt, aggregated over the randomness of the concept z∗. 

Under the model in (4.1), we will show that pretrained LLMs are able to perform ICL because they secretly implement BMA (Wasserman, 2000) during prompting. For ease of presentation, we first consider the setting where the LLM is perfectly pretrained, i.e., the conditional distribution induced by the LLM is given by P(rt+1 | promptt). We relax this condition by analyzing the pretraining error in Section 5. 

Proposition 4.1 (LLMs Perform BMA). Under the model in (4.1), it holds that 

� P(rt+1 = · | promptt) = P(rt+1 = · | ct+1, St, z)P(z | St)dz. (4.2) � We note that the left-hand side of (4.2) is the prediction of the pretrained LLM given a prompt promptt. Meanwhile, the right-hand side is exactly the prediction given by the BMA algorithm that infers the posterior belief of the concept z∗ based on St and predicts rt+1 by aggregating the likelihood in (4.1) with respect to the posterior P(z∗ = · | St). Thus, this proposition shows that perfectly pretrained LLMs are able to perform ICL because they implement BMA during prompting. As mentioned, Proposition 4.1 is proved under a more general model than the previous works and thus serves as a generalized result of some claims in the previous works. We note that the claim of Proposition 4.1 is independent of the network structure. This partially explains why LSTMs demonstrate ICL ability in Xie et al. (2021). In the next section, we will demonstrate how the attention mechanism helps to implement BMA. The proof of Proposition 4.1 is in Appendix E.2. 

Next, we study the performance of ICL from an online learning perspective. Recall that LLMs are continuously prompted with St and aim to predict the (t + 1)-th covariate rt+1 for t ∈ [T − 1]. This can be viewed as an online learning problem. For any algorithm that generates a sequence of density estimators {P<sup>�</sup> (rt)}<sup>T</sup> t=1<sup>forpredicting{rt}t∈[T],weconsiderthe</sup> following ICL regret as its performance metric: 



This ICL regret measures the performance of the estimator P<sup>�</sup> compared with the best hidden concept in hindsight. For the perfectly trained LLMs, the estimator is exactly P<sup>�</sup> (rt) = P(rt+1 | promptt). By building the equivalence of pretrained LLM and BMA, we have the following corollary, which shows that predicting {rt}t∈[T ] by iteratively prompting the LLM incurs a O(1/T ) regret. 

Corollary 4.2 (ICL Regret of Perfectly Pretrained Model). Under the model in (4.1), we have for any t ∈ [T ] that 



Here PZ is the prior of the hidden concept z ∈Z. When the hidden concept space Z is finite and the prior PZ(z) is the uniform distribution on Z, we have that regrett ≤ log |Z|/t. When 

6 

t the nominal concept z∗ satisfies that supz �i=1<sup>P(ri | z, prompt</sup> i−1<sup>) = �t</sup> i=1<sup>P(ri | z∗, prompt</sup> i−1<sup>)</sup> for any t ∈ [T ], the regret is bounded as regrett ≤ log(1/PZ(z∗))/t. 

This theorem states that the ICL regret of the perfectly pretrained model is bounded by log(1/PZ(z∗))/t. This is intuitive since the regret is relatively large if the concept z∗ rarely appears according to the prior distribution. This corollary shows that, when given sufficiently many examples, predicting {rt}t∈[T ] via ICL is almost as good as the oracle method which knows true concept z∗ and the likelihood function P(ri | prompti−1, z∗). The practical relevance of this result is discussed in Appendix D. The proof of Corollary 4.2 is in Appendix E.3. In Section 5, we characterize the deviation between the learned model and the underlying true model. Next, we show how transformers parameterize BMA. 

### 4.1 Attention Parameterizes Bayesian Model Averaging 

In the following, we explore the role played by the attention mechanism in ICL. To simplify the presentation, we consider the case where the covariate �ct ∈ X<sup>∗</sup> is a single token ct ∈ X in this subsection. During the ICL phase, pretrained LLMs are prompted with promptt = (St, ct+1) and tasked with predicting the (t +1)-th response rt+1. The transformers first separately map the covariates �ci and responses ri for i ∈ [t] to the corresponding feature spaces, which are usually realized by the fully connected layers. We denote these two learnable mappings as k : R<sup>d</sup> → R<sup>dk</sup> and v : R<sup>d</sup> → R<sup>dv</sup> . Their nominal values are denoted as k∗ and v∗, respectively. The pretraining of the transformer essentially learns the nominal mappings v∗ and k∗ with sufficiently many data points. After these transformations, the attention module will take vi = v∗(ri) and ki = k∗(ci) for i ∈ [t] as the value and key vectors to predict the result for the query qt+1 = kt+1 = k∗(ct+1). To elucidate the role played by attention, we consider a Gaussian linear simplification of (4.1) 



where φ : R<sup>dk</sup> → R<sup>dφ</sup> refers to the feature mapping in some Reproducing Kernel Hilbert Space (RKHS), z∗ ∈ R<sup>dv×dφ</sup> corresponds to the hidden concept, and ξt ∼ N(0, σ<sup>2</sup> I), t ∈ [T ] are i.i.d. Gaussian noises with covariance σ<sup>2</sup> I. Besides, we assume the prior of z∗ is P(z) is a Gaussian distribution N(0, λI). Note that (4.4) can be written as 



which is a realization of (4.1) with ht = z, ξt = ǫt, and f (c, h, ξ) = v∗<sup>−1(hφ(k∗(c)) + ξ).In</sup> other words, (4.4), or equivalently (4.5), specifies a specialization of (4.1) where in the feature space, the hidden concept z∗ represents a transformation between the value v and the key k. Here, we simply take this as the transformation by a matrix, which can be easily generalized by building a bijection between concepts z and complex transformations. In the following, to simplify the notation, let K : R<sup>dk</sup> × R<sup>dk</sup> → R. denote the kernel function of the RKHS induced by φ. The stacks of the values and keys are denoted as Kt = (k1, . . . , kt)<sup>⊤</sup> ∈ R<sup>t×dk</sup> and Vt = (v1, . . . , vt)<sup>⊤</sup> ∈ R<sup>t×dv</sup> , respectively. Consequently, the model in (4.4) implies that 



7 

where we denote by Σt the covariance of vt+1 ∼ P(· | St, qt+1), and the mean concept z¯t is 



Combining (4.6) and (4.7), we can see that z¯tφ(qt+1) essentially measures the similarity between the query and keys, which is quite similar to the attention mechanism defined in Section 3. However, here the similarity is normalization according to (4.7), not by softmax. This motivates us to define a new structure of attention and explore the relationship between the newly defined attention and the original one. For any q ∈ R<sup>dk</sup> , K ∈ R<sup>t×dk</sup> , and V ∈ R<sup>t×dv</sup> , we define a variant of the attention mechanism as follows, 



From (4.6), (4.7), and (4.8), it holds that the response vt+1 for (t + 1)-th query is distributed as vt+1 ∼ N(attn†(qt+1, Kt, Vt), Σt). We note that attn† bakes the BMA algorithm for the Gaussian linear model in its architecture, by first estimating z¯t via (4.7) and deriving ¯ the final estimate from the inner product between zt and qt+1. Here attn†(·) is an instance of the intention mechanism studied in Garnelo and Czarnecki (2023) and can be viewed as a generalization of linear attention. Recall that we define the softmax attention (Vaswani et al., 2017) for any q ∈ R<sup>dk</sup> , K ∈ R<sup>t×dk</sup> , and V ∈ R<sup>t×dv</sup> as attn(q, K, V ) = V<sup>⊤</sup> softmax(Kq). In the following proposition, we show that the attention in (4.8) coincides with the softmax attention as the sequence length goes to infinity. 

Proposition 4.3. We assume that the key-value pairs {(kt, vt)}t<sup>T</sup> =1<sup>are independentand iden-</sup> tically distributed, and we adopt Gaussian RBF kernel KRBF. In addition, we assume that ∥kt∥2 = ∥vt∥ = 1. Then, it holds for an absolute constant C > 0 and any q ∈ R<sup>dk</sup> with ∥q∥ = 1 that limT →∞ attn†(q, KT , VT ) = C · limT →∞ attn(q, KT , VT ). 

The proof is in Appendix E.4. Combined with the conditional probability of vt+1 in (4.6), this proposition shows that softmax attention approximately encodes BMA in long token sequences (Wasserman, 2000), and thus is able to perform ICL when prompted after pretraining. 

## 5 Theoretical Analysis of Pretraining 5.1 Pretraining Algorithm 

In this section, we describe the pretraining setting. We largely follow the transformer structures in Brown et al. (2020). The whole network is a composition of D sub-modules, and each sub-module consists of a MHA and a Feed-Forward (FF) fully connected layer. Here, D > 0 is the depth of the network. The whole network takes X<sup>(0)</sup> = X ∈ R<sup>L×d</sup> as its input. In the t-th layer for t ∈ [D], it first takes the output X<sup>(t−1)</sup> of the (t − 1)-th layer as the input and forwards it through MHA with a residual link and a layer normalization Πnorm(·) to output Y<sup>(t)</sup> , which projects each row of the input into the unit ℓ2-ball. Here we take dh = d in MHA, and the generalization of our result to general cases is trivial. Then the intermediate output Y<sup>(t)</sup> is forwarded to the FF module. It maps each row of the input Y<sup>(t)</sup> ∈ R<sup>L×d</sup> through the same single-hidden layer neural network with dF neurons, that is ffn(Y<sup>(t)</sup> , A<sup>(t)</sup> ) = ReLU(Y<sup>(t)</sup> A<sup>(</sup> 1<sup>t))A(</sup> 2<sup>t),whereA(</sup> 1<sup>t)</sup> ∈ R<sup>d×dF</sup> , and A<sup>(</sup> 2<sup>t)</sup> ∈ R<sup>dF ×d</sup> are the weight 

8 

matrices. Combined with a residual link and layer normalization, it outputs the output of layer t as X<sup>(t)</sup> , that is 



Here we allocate weights γ1<sup>(t)</sup> and γ2<sup>(t)</sup> to residual links only for the convenience of theoretical analysis. In the last layer, the network outputs the probability of the next token via a softmax module, that is Y<sup>(D+1)</sup> = softmax(I<sup>⊤</sup> L<sup>X(D)A(D+1)/(Lτ))∈Rdy,whereIL∈RListhe</sup> vector with all ones, A<sup>(D+1)</sup> ∈ R<sup>d×dy</sup> is the weight matrix, τ ∈ (0, 1] is the fixed temperature parameter, and dy is the output dimension. The parameters of each layer are denoted as θ<sup>(t)</sup> = (γ1<sup>(t), γ</sup> 2<sup>(t), W (t), A(t))fort∈[D]andθ(D+1)=A(D+1),andtheparameterofthewhole</sup> network is the concatenation of these parameters, i.e., θ = (θ<sup>(1)</sup> , · · · , θ<sup>(D+1)</sup> ). We consider the transformers with bounded parameters. The set of parameters is 



where BA, BA,1, BA,2, BQ, BK, and BV are the bounds of parameter. Here we only consider the non-trivial case where these bounds are larger than 1, otherwise, the magnitude of the output in D<sup>th</sup> layer decreases exponentially with growing depth. The probability induced by the transformer with parameter θ is denoted as Pθ. 

The pretraining dataset consists of Np independent trajectories. For the n-th trajectory with n ∈ [Np], a hidden concept z<sup>n</sup> ∼ PZ(z) ∈ ∆(Z) is first sampled, which is the hidden variables of the token sequence to generate, e.g., the theme, the sentiment, and the style. Then the tokens are sequentially sampled from the Markov chain induced by z<sup>n</sup> as x<sup>n</sup> t+1<sup>∼P(· | S</sup> t<sup>n, zn)</sup> and St<sup>n</sup> +1<sup>=(S</sup> t<sup>n, xn</sup> t+1<sup>),wherexn</sup> t+1<sup>∈X,andS</sup> t<sup>n, S</sup> t<sup>n</sup> +1<sup>∈X∗.HeretheMarkovchainisdefined</sup> with respect to the state St<sup>n,whichobviouslysatisfiestheMarkovpropertysinceS</sup> i<sup>nfor</sup> i ∈ [t − 1] are contained in St<sup>n.ThepretrainingdatasetisDN</sup> p<sup>,T</sup> p<sup>={(S</sup> t<sup>n, xn</sup> t+1<sup>)}N</sup> n,t<sup>p</sup> =1<sup>,Tpwhere</sup> the concepts z<sup>n</sup> is hidden from the context and thus unobserved. Here each token sequence is divided into Tp pieces {(St<sup>n, xn</sup> t+1<sup>)}</sup> t<sup>T</sup> =1<sup>p.Wehighlightthatthispretrainingdatasetcollecting</sup> process subsumes those for GPT, and Masked AutoEncoders (MAE) (Radford et al., 2021). For GPT, each trajectory corresponds to a paragraph or an article in the pretraining dataset, and z<sup>n</sup> ∼ PZ(z) is realized by the selection process of these contexts from the Internet. For MAE, we take Tp = 1, and S1<sup>nandxn</sup> 2<sup>respectivelycorrespondtotheimageandthemasked</sup> token. 

To pretrain the transformer, we adopt the cross-entropy as the loss function, which is widely used in the training of BERT and GPT. The corresponding pretraining algorithm is 



We first analyze the population version of (5.2). In the training set, the conditional distribution of x<sup>n</sup> t+1<sup>conditioned on S</sup> t<sup>nis P(xn</sup> t+1<sup>| S</sup> t<sup>n) =</sup> �Z<sup>P(x</sup> t<sup>n</sup> +1<sup>| S</sup> t<sup>n, z)PZ(z | S</sup> t<sup>n)dz, where the unobserved</sup> hidden concept is weighed via its posterior distribution. Thus, the population risk of (5.2) is Et[ESt[KL(P(· | St)∥Pθ(· | St)) + H(P(· | St))]], where t ∼ Unif([Tp]), H(p) = −⟨p, log p⟩ is the entropy, and St is distributed as the pertaining distribution. Thus, we expect that Pθ will converge to P. For MAE, the network training adopts ℓ2-loss, and we defer the analysis of this case to Appendix F.4. 

9 

### 5.2 Performance Guarantee for Pretraining 

We first state the assumptions for the pretraining setting. 

Assumption 5.1. There exists a constant R > 0 such that for any z ∈ Z and St ∼ P(· | z), we have ∥St<sup>⊤∥2,∞≤Ralmostsurely.</sup> 

This assumption states that the ℓ2-norm of the magnitude of each token in the token sequence is upper bounded by R > 0. This assumption holds in most machine learning settings. For BERT and GPT, each token consists of word embedding and positional embedding. For MAE, each token consists of a patch of pixels. The ℓ2-norm of each token is bounded in these cases. 

Assumption 5.2. There exists a constant c0 > 0 such that for any z ∈ Z, x ∈ X and S ∈ X<sup>∗</sup> , we have P(x | S, z) ≥ c0. 

This assumption states that the conditional probability of x conditioned on S and z is lower bounded. This comes from the ambiguity of language, that is, a sentence can take lots of words as its next word. Similar regularity assumptions are also widely adopted in ICL literature (Xie et al., 2021; Wies et al., 2023). To state our result, we respectively use ES∼D and PD to denote the expectation and the distribution of the average distribution of St<sup>nin</sup> DNp,Tp, i.e., ES∼D[f (S)] =<sup>�T</sup> t=1<sup>pESt[f(St)]/Tpforanyfunctionf: X∗→R.</sup> 

Theorem 5.3. Let B<sup>¯</sup> = τ<sup>−1</sup> RhBABA,1BA,2BQBKBV and D<sup>¯</sup> = D<sup>2</sup> d(dF + dh + d) + d · dy. Under Assumptions 5.1 and 5.2, the pretrained model Pθ� by the algorithm in (5.2) satisfies 



with probability at least 1 − δ, where tmix is the mixing time of the Markov chains induced by P, formally defined in Appendix F.1. 

We define the right-hand side of the equation as ∆pre(Np, Tp, δ). The first and the second terms in the bound are the approximation error. It measures the distance between the nominal distribution P and the distributions induced by transformers with respect to KL divergence. If the nominal model P can be represented by transformers exactly, i.e., the realizable case, these two terms will vanish. The third term is the generalization error, and it does not increase with the growing sequence length Tp. This is proved via the PAC-Bayes framework. 

This pretraining analysis is missing in most existing theoretical works about ICL. Xie et al. (2021), Wies et al. (2023), and Jiang (2023) all assume access to an arbitrarily precise pretraining model. Although the generalization bound in Li et al. (2023) can be adapted to the pretraining analysis, the risk definition therein can not capture the approximation error in our result. Furthermore, their analysis cannot fit the maximum likelihood algorithm in (5.2). Concretely, their result can only show that the convergence rate of KL divergence is O((NpTp)<sup>−1/2</sup> ) with a realizable function class. Combined with Pinsker’s inequality, this gives the convergence rate for total variation as O((NpTp)<sup>−1/4</sup> ) even in the realizable case. 

10 

The deep neural networks are shown to be universal approximators for many function classes (Cybenko, 1989; Hornik, 1991; Yarotsky, 2017). Thus, the approximation error in Theorem 5.3 should vanish with the increasing size of the transformer. To achieve this, we slightly change the structure of the transformer by admitting a bias term in feed-forward modules, taking A<sup>(</sup> 2<sup>t)</sup> ∈ R<sup>dF ×dF</sup> , and admitting dF to vary across layers. This mildly affects the generalization error by replacing D · dF by the sum of dF of all the layers in Theorem 5.3. We derive the approximation error bound when the dimension of each word is equal to one, i.e., X ⊆ R. Our method can carry over the case d > 1. 

Proposition 5.4 (Informal). Under certain smoothness conditions, if dF ≥ 16dy, BA,1 ≥ 16Rdy, BA,2 ≥ dF BA ≥ �dy, and BV ≥ √d, then for some constant C > 0, we have 



The formal statement and proof are deferred to Appendix F.3. This proposition states that the approximation error decays exponentially with the increasing depth. Combined with this result, Theorem 5.3 provides the full description of the pretraining performance. 

## 6 ICL Regret under Practical Settings 

### 6.1 ICL Regret with an Imperfectly Pretrained Model 

xIn Section 4, we study the ICL regret with a perfect pretrained model. In what follows, we characterize the ICL regret when the pretrained model has an error. Note that the distribution DICL of the prompts of ICL tasks can be different from that of pretraining. We impose the following assumption on their relation. 

Assumption 6.1. We assume that there exists an absolute constant κ > 0 such that for any ICL prompt, it holds that PDICL(prompt) ≤ κ · PD(prompt). 

This assumption states that the prompt distribution is covered by the pretraining distribution. Intuitively, the pretrained model cannot precisely inference on the datapoint that is outside the support of the pretraining distribution. For example, if the pretraining data does not contain any mathematical symbols and numbers, it is difficult for the pretrained model to calculate 2 × 3 in ICL precisely. We then have the following theorem characterizing the ICL regret of the pretrained model. 

Theorem 6.2 (ICL Regret of Pretrained Model). We assume that the underlying hidden concept z∗ maximizes<sup>�t</sup> i=1<sup>log P(ri | prompt</sup> i−1<sup>, z)forany t ∈[T]andthere existsanabsolute</sup> constant β > 0 such that log(1/p0(z∗)) ≤ β. Under Assumptions 5.1, 5.2, and 6.1, we have with probability at least 1 − δ that 



Here we denote by ∆pre(Np, Tp, δ) the pretraining error in Theorem 5.3. 

11 

Theorem 6.2 shows that the expected ICL regret for the pretrained model is upper bounded by the sum of two terms: (a) the ICL regret for the underlying true model and (b) the pretraining error. These two terms are separately bounded in Sections 4 and 5. 

### 6.2 Prompting with Wrong Input-Output Mappings 

In the real-world implementations of ICL, the provided input-output examples may not conform to the nominal distribution induced by z∗, and the outputs in examples can be perturbed. We temporarily take concept space Z as a finite space, and our results can be generalized with a covering number argument. We denote the prompt considered in Sec� � tion 4 as promptt = (St, �ct+1), St = (c1, r1, · · · , �ct, rt) ∈ X<sup>∗</sup> , and (ci+1, ri+1) ∼ P(· | Si, z∗) � for i ∈ [t − 1]. Here, each input ci ∈ X<sup>l</sup> is a l-length token sequence, and each output ri ∈ X is a single token. The perturbed prompt is then denoted as prompt<sup>′</sup> = (St<sup>′, �ct+1),where</sup> St<sup>′= (�c1, r</sup> 1<sup>′, · · ·, �ct, r</sup> t<sup>′) ∈X∗, and r</sup> i<sup>′for i ∈[t] is the modified output.We denote the perturbed</sup> prompt distribution as P<sup>′</sup> . Then the performance of ICL with wrong input-output mappings can be stated as follows. 

Proposition 6.3 (Informal). Under certain assumptions, including the distinguishability assumption (minz=z∗ KLpair�P(· | z<sup>∗</sup> ) ∥ P(· | z)� > 2 log 1/c0), the pretrained model Pθ� in (5.2) predicts the outputs with the prompt containing wrong mappings as 



with probability at least 1 − δ. 

The first term is the pretraining error in Theorem 5.3, which is related to the size of the pretraining set and the capacity of the neural networks. The second term is the ICL error. Intuitively, this term represents the concept identification error. If the considered task z∗ is distinguishable, i.e., satisfying Assumption G.3, this term decays to 0 exponentially in √t. The required assumptions and formal statement are in Appendix G.2. 

12 

## References 

- Agarwal, A., Kakade, S., Krishnamurthy, A. and Sun, W. (2020). Flambe: Structural complexity and representation learning of low rank MDPs. Advances in Neural Information Processing Systems, 33 20095–20107. 

- Aky¨urek, E., Schuurmans, D., Andreas, J., Ma, T. and Zhou, D. (2022). What learning algorithm is in-context learning? investigations with linear models. arXiv preprint arXiv:2211.15661. 

- Anthony, M., Bartlett, P. L., Bartlett, P. L. et al. (1999). Neural network learning: Theoretical foundations, vol. 9. cambridge university press Cambridge. 

- Bai, Y., Chen, F., Wang, H., Xiong, C. and Mei, S. (2023). Transformers as statisticians: Provable in-context learning with in-context algorithm selection. arXiv preprint arXiv:2306.04637. 

- Bartlett, P. L., Foster, D. J. and Telgarsky, M. J. (2017). Spectrally-normalized margin bounds for neural networks. Neural Information Processing Systems. 

- Belghazi, M. I., Baratin, A., Rajeshwar, S., Ozair, S., Bengio, Y., Courville, A. and Hjelm, D. (2018). Mutual information neural estimation. In International Conference on Machine Learning. PMLR. 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A. et al. (2020). Language models are few-shot learners. Neural Information Processing Systems. 

- Caponnetto, A. and De Vito, E. (2007). Optimal rates for the regularized least-squares algorithm. Foundations of Computational Mathematics. 

- Chan, S. C., Santoro, A., Lampinen, A. K., Wang, J. X., Singh, A., Richemond, P. H., McClelland, J. and Hill, F. (2022). Data distributional properties drive emergent few-shot learning in transformers. arXiv preprint arXiv:2205.05055. 

- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 2 303–314. 

- Dai, D., Sun, Y., Dong, L., Hao, Y., Sui, Z. and Wei, F. (2022). Why can GPT learn InContext? Language models secretly perform gradient descent as meta optimizers. arXiv preprint arXiv:2212.10559. 

- Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805. 

- Dong, L., Yang, N., Wang, W., Wei, F., Liu, X., Wang, Y., Gao, J., Zhou, M. and Hon, H.-W. (2019). Unified language model pre-training for natural language understanding and generation. Advances in neural information processing systems, 32. 

- Dong, Q., Li, L., Dai, D., Zheng, C., Wu, Z., Chang, B., Sun, X., Xu, J. and Sui, Z. (2022). A survey for in-context learning. arXiv preprint arXiv:2301.00234. 

13 

- Duchi, J. C. (2019). Information theory and statistics. Lecture Notes for Statistics, 311 304. 

- Edelman, B. L., Goel, S., Kakade, S. and Zhang, C. (2021). Inductive biases and variable creation in self-attention mechanisms. arXiv preprint arXiv:2110.10090. 

- Elbr¨achter, D., Perekrestenko, D., Grohs, P. and B¨olcskei, H. (2021). Deep neural network approximation theory. IEEE Transactions on Information Theory, 67 2581–2623. 

- Feng, G., Gu, Y., Zhang, B., Ye, H., He, D. and Wang, L. (2023). Towards revealing the mystery behind chain of thought: a theoretical perspective. arXiv preprint arXiv:2305.15408. 

- Fukumizu, K. (2015). Nonparametric bayesian inference with kernel mean embedding. In Modern Methodology and Applications in Spatial-Temporal Modeling. Springer, 1–24. 

- Garg, S., Tsipras, D., Liang, P. and Valiant, G. (2022). What can transformers learn incontext? A case study of simple function classes. arXiv preprint arXiv:2208.01066. 

- Garnelo, M. and Czarnecki, W. M. (2023). Exploring the space of key-value-query models with intention. arXiv preprint arXiv:2305.10203. 

- Hahn, M. and Goyal, N. (2023). A theory of emergent in-context learning as implicit structure induction. arXiv preprint arXiv:2303.07971. 

- Honovich, O., Shaham, U., Bowman, S. R. and Levy, O. (2022). Instruction induction: From few examples to natural language task descriptions. arXiv preprint arXiv:2205.10782. 

- Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. Neural networks, 4 251–257. 

- Hron, J., Bahri, Y., Sohl-Dickstein, J. and Novak, R. (2020). Infinite attention: NNGP and NTK for deep attention networks. In International Conference on Machine Learning. 

- Iyer, S., Lin, X. V., Pasunuru, R., Mihaylov, T., Simig, D., Yu, P., Shuster, K., Wang, T., Liu, Q., Koura, P. S. et al. (2022). OPT-IML: Scaling language model instruction meta learning through the lens of generalization. arXiv preprint arXiv:2212.12017. 

- Jiang, H. (2023). A latent space theory for emergent abilities in large language models. arXiv preprint arXiv:2304.09960. 

- Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F. and Liu, Q. (2019). Tinybert: Distilling bert for natural language understanding. arXiv preprint arXiv:1909.10351. 

- Ke, G., He, D. and Liu, T.-Y. (2020). Rethinking positional encoding in language pre-training. arXiv preprint arXiv:2006.15595. 

- Kim, H. J., Cho, H., Kim, J., Kim, T., Yoo, K. M. and Lee, S.-g. (2022). Self-generated incontext learning: Leveraging auto-regressive language models as a demonstration generator. arXiv preprint arXiv:2206.08082. 

- Kojima, T., Gu, S. S., Reid, M., Matsuo, Y. and Iwasawa, Y. (2022). Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916. 

14 

- Ledent, A., Mustafa, W., Lei, Y. and Kloft, M. (2021). Norm-based generalisation bounds for deep multi-class convolutional neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35. 

- Li, Y., Ildiz, M. E., Papailiopoulos, D. and Oymak, S. (2023). Transformers as algorithms: Generalization and stability in in-context learning. arXiv preprint arXiv:2301.07067. 

- Liao, R., Urtasun, R. and Zemel, R. (2020). A pac-bayesian approach to generalization bounds for graph neural networks. arXiv preprint arXiv:2012.07690. 

- Lin, S. and Zhang, J. (2019). Generalization bounds for convolutional neural networks. arXiv preprint arXiv:1910.01487. 

- Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L. and Chen, W. (2021). What makes good in-context examples for gpt-3? arXiv preprint arXiv:2101.06804. 

- Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H. and Neubig, G. (2023). Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys, 55 1–35. 

- Lu, Y., Bartolo, M., Moore, A., Riedel, S. and Stenetorp, P. (2021). Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. arXiv preprint arXiv:2104.08786. 

- Malladi, S., Wettig, A., Yu, D., Chen, D. and Arora, S. (2022). A kernel-based view of language model fine-tuning. arXiv preprint arXiv:2210.05643. 

- Michel, P., Levy, O. and Neubig, G. (2019). Are sixteen heads really better than one? Advances in neural information processing systems, 32. 

- Min, S., Lewis, M., Zettlemoyer, L. and Hajishirzi, H. (2021). Metaicl: Learning to learn in context. arXiv preprint arXiv:2110.15943. 

- Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M., Hajishirzi, H. and Zettlemoyer, L. (2022). Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837. 

- Neyshabur, B., Bhojanapalli, S. and Srebro, N. (2017). A pac-bayesian approach to spectrallynormalized margin bounds for neural networks. arXiv preprint arXiv:1707.09564. 

- Noci, L., Anagnostidis, S., Biggio, L., Orvieto, A., Singh, S. P. and Lucchi, A. (2022). Signal propagation in transformers: Theoretical perspectives and the role of rank collapse. arXiv preprint arXiv:2206.03126. 

- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A. et al. (2022). Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35 27730– 27744. 

- Paulin, D. (2015). Concentration inequalities for markov chains by marton couplings and spectral methods. 

15 

- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J. et al. (2021). Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR. 

- Rubin, O., Herzig, J. and Berant, J. (2021). Learning to retrieve prompts for in-context learning. arXiv preprint arXiv:2112.08633. 

- Song, L., Huang, J., Smola, A. and Fukumizu, K. (2009). Hilbert space embeddings of conditional distributions with applications to dynamical systems. In International Conference on Machine Learning. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, �L. and Polosukhin, I. (2017). Attention is all you need. In Neural Information Processing Systems. 

- von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A. and Vladymyrov, M. (2022). Transformers learn in-context by gradient descent. arXiv preprint arXiv:2212.07677. 

- Vuckovic, J., Baratin, A. and Combes, R. T. d. (2020). A mathematical theory of attention. arXiv preprint arXiv:2007.02876. 

- Wang, X., Zhu, W. and Wang, W. Y. (2023). Large language models are implicitly topic models: Explaining and finding good demonstrations for in-context learning. arXiv preprint arXiv:2301.11916. 

- Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D. and Hajishirzi, H. (2022). Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560. 

- Wasserman, L. (2000). Bayesian model selection and model averaging. Journal of Mathematical Psychology, 44 92–107. 

- Wei, C., Chen, Y. and Ma, T. (2022a). Statistically meaningful approximation: a case study on approximating turing machines with transformers. Advances in Neural Information Processing Systems, 35 12071–12083. 

- Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M. and Le, Q. V. (2021). Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652. 

- Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D. et al. (2022b). Emergent abilities of large language models. arXiv preprint arXiv:2206.07682. 

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Chi, E., Le, Q. and Zhou, D. (2022c). Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903. 

- Wies, N., Levine, Y. and Shashua, A. (2023). The learnability of in-context learning. arXiv preprint arXiv:2303.07895. 

16 

- Xie, S. M., Raghunathan, A., Liang, P. and Ma, T. (2021). An explanation of in-context learning as implicit Bayesian inference. arXiv preprint arXiv:2111.02080. 

- Yang, G. (2020). Tensor programs II: Neural tangent kernel for any architecture. arXiv preprint arXiv:2006.14548. 

- Yarotsky, D. (2017). Error bounds for approximations with deep relu networks. Neural Networks, 94 103–114. 

- Yun, C., Bhojanapalli, S., Rawat, A. S., Reddi, S. J. and Kumar, S. (2019). Are transformers universal approximators of sequence-to-sequence functions? arXiv preprint arXiv:1912.10077. 

- Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R. and Smola, A. J. (2017). Deep sets. Neural Information Processing Systems. 

- Zhang, F., Liu, B., Wang, K., Tan, V. Y., Yang, Z. and Wang, Z. (2022a). Relational reasoning via set transformers: Provable efficiency and applications to MARL. arXiv preprint arXiv:2209.09845. 

- Zhang, Z., Zhang, A., Li, M. and Smola, A. (2022b). Automatic chain of thought prompting in large language models. arXiv preprint arXiv:2210.03493. 

- Zhou, D., Sch¨arli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D., Bousquet, O., Le, Q. and Chi, E. (2022a). Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625. 

- Zhou, Y., Muresanu, A. I., Han, Z., Paster, K., Pitis, S., Chan, H. and Ba, J. (2022b). Large language models are human-level prompt engineers. arXiv preprint arXiv:2211.01910. 

17 

## Appendix for “What and How does In-Context Learning Learn? Bayesian Model Averaging, Parameterization, and Generalization” 

## A Conclusion 

In this paper, we investigated the theoretical foundations of ICL for the pretrained language models. We proved that the perfectly pretrained LLMs implicitly implements BMA with regret O(1/t) over a general response generation modeling, which subsumes the models in previous works. Based on this, we showed that the attention mechanism parameterizes the BMA algorithm. Analyzing the pretraining process, we demonstrated that the total variation between the pretrained model and the nominal distribution consists of the approximation error and the generalization error. The combination of the ICL regret and the pretraining performance gives the full description of ICL ability of pretrained LLMs. We mainly focus on the prompts that comprise several examples in this work and leave the analysis of instructionbased prompts for future works. 

## B More Related Works 

Transformers. Our work is also related to the works that theoretically analyze the performance of transformers. For the analytic properties of transformers, Vuckovic et al. (2020) proved that attention is Lipschitz-continuous via the view of interacting particles. Noci et al. (2022) provided the theoretical justification of the rank collapse phenomenon in transformers. Yun et al. (2019) demonstrated that transformers are universal approximators. For the statistical properties of transformers, Malladi et al. (2022), Hron et al. (2020), and Yang (2020) analyzed the training of transformers within the neural tangent kernel framework. Wei et al. (2022a) presented the approximation and generalization bounds for learning boolean circuits and Turing machines with transformers. Edelman et al. (2021) and Li et al. (2023) derived the generalization error bound of transformers. In our work, we analyze transformers from both the analytic and statistical sides. We show that attention essentially implements the BMA algorithm in the ICL setting. Furthermore, we derive the approximation and generalization bounds for transformers in the pretraining phase. 

Generalization. Our analysis of the pretraining is also related to the generalization analysis of the neural networks. This topic has attracted a lot of interests for a long time. Anthony et al. (1999) derived the uniform generalization bound for fully-connected neural networks with the help pf VC dimension. Bartlett et al. (2017) sharpened this generalization bound for classification problem by adopting the Dudley’s integral and calculating of the covering number of neural network class. At the same time, Neyshabur et al. (2017) derived a similar as Bartlett et al. (2017) from PAC-Bayes framework. Following this line, Liao et al. (2020) , Ledent et al. (2021) and Lin and Zhang (2019) built the generalization bound for graph neural networks and convolutional neural network. These results respected the underlying graph structure and the translation-invariance in the networks. Edelman et al. (2021) established the generalization bound for transformer, but this result did not reflect the permutation-invariance, still depending on the channel number. Our work focuses on the 

18 

analysis of Maximum Likelihood Estimate (MLE) with transformer function class, which is not covered by previous works. Our bounds are sharper than that of Edelman et al. (2021) on the channel number dependency. 

## C Figure for Pretraining and ICL 



Figure 1: To form the pretraining dataset, a hidden concept z is first sampled according to PZ, and a document is generated from the concept. Taking the token sequence St up to position t ∈ [T ] as the input, the LLM is pretrained to maximize the next token xt+1. During the ICL phase, the pretrained LLM is prompted with several examples to predict the response of the query. 

## D Discussion About The Experimental Results in Existing Works 

We note that Aky¨urek et al. (2022) trains the transformer from scratch with the ICL distribution. This is a special case of the pretraining setting in our work, where they set the pretraining distribution as the ICL distribution. Figure 2 in Aky¨urek et al. (2022) indicates that the ICL behavior of the transformer matches the behavior of the Bayesian predictor. This is exactly the result we prove in Theorem 4.1. 

We note that that dependency on the size of the hidden variable space is also verified in the experiments in Garg et al. (2022). Figure 2 in Garg et al. (2022) indicates that the ICL of LLMs only has a significant error when T ≤ d, where d is the dimension of the hidden variable. This implies that the regret of the ICL by LLMs is at most linear in O(d/T ). From the view of our theoretical result, discretizing the set {z ∈ R<sup>d</sup> | ∥z∥2 ≤ d} with approximation error δ > 0 will result in a set with (C/δ)<sup>d</sup> elements, where C > 0 is an absolute constant. Corollary 4.2 implies that the regret is log |Z|/T = d log(C/δ)/T , which matches Figure 2 in Garg et al. (2022). 

19 

## E Proofs for Section 4.1 

### E.1 Introduction of Conditional Mean Embedding 

Let Hk and Hv be the two RKHSs over the spaces Q and V with the kernels K and L, respectively. We denote by φ : Q → ℓ2 and ϕ : V → ℓ2 the feature mappings associated with Hk and Hv, respectively. Here l2 is the space of the square-integrable function class. Then it holds for any k, k<sup>′</sup> ∈ Q and v, v<sup>′</sup> ∈ V that 



Let PK,V be the joint distribution of the two random variables K and V taking values in Q and V, respectively. Then the conditional mean embedding CME(q, PK,V) ∈Hv of the conditional distribution PV | K is defined as 



The conditional mean embedding operator CV | K : Hk →Hv is a linear operator such that 



for any q ∈ Q. We define the (uncentered) covariance operator CKK : Hk →Hk and the (uncentered) cross-covariance operator CVK : Hk →Hv as follows, 

CKK = E�K(K, ·) ⊗ K(K, ·)�, CVK = E�L(V, ·) ⊗ K(K, ·)�. Here ⊗ is the tensor product. Song et al. (2009) shows that CV | K = CVKCKK<sup>−1.Thus,wehave</sup> that 



For i.i.d. samples {(k<sup>ℓ</sup> , v<sup>ℓ</sup> )}ℓ∈[L] of PK,V, ∥·∥HS denotes the Hilbert-Schmidt norm, we write φ(K) = (φ(k<sup>1</sup> ), . . . , φ(k<sup>L</sup> ))<sup>⊤</sup> ∈ R<sup>L×dφ</sup> and ϕ(V ) = (φ(v<sup>1</sup> ), . . . , φ(v<sup>L</sup> ))<sup>⊤</sup> ∈ R<sup>L×dϕ</sup> . Then the empirical covariance operator C<sup>�</sup> KK and empirical cross-covariance operator C<sup>�</sup> VK are defined as 



The empirical version of the conditional operator is 

C�V | K<sup>λ= ϕ(Y )⊤φ(X)</sup> �φ(X)<sup>⊤</sup> φ(X) + λI�−1 = C�VK( �CKK + L−1λI)−1 ∈ Rdϕ×dφ. 

### E.2 Proof of Proposition 4.1 

Proof. By (4.1), we have that 



20 

where the first and the last equalities results from model (4.2), and the second equality results from Bayes’ theorem. 

### E.3 Proof of Corollary 4.2 

Proof. Note that 



Then, by Bayesian model averaging, we have the following density estimation, 



Thus, it holds that 



We consider q to be in the class of all Dirac measures. Then, we have that 



Thus, the statistical convergence rate of the Bayesian posterior averaging is O(1/T ). 



### E.4 Proof of Proposition 4.3 

Proof. The proof of Proposition 4.3 mainly involves two steps 

- Build the relationship between attn† and conditional mean embedding. 

- Build the relationship between the attn and conditional mean embedding. 

#### Step 1: Build the relationship between attn† and conditional mean embedding. 

In the following, we adopt Hk and Hv to denote the RKHSs for the key and the value with the kernel functions K and L, respectively. Also, we use ∥· ∥ to denote the norm of RKHS for an element in the corresponding RKHS and the operator norm of the operators that transform elements between RKHSs. For the value space, we adopt the Euclidean kernel L(v, v<sup>′</sup> ) = v<sup>⊤</sup> v<sup>′</sup> , and the feature mapping ϕ is the identity mapping. Recall the definition of the empirical 

21 

covariance operator and the empirical cross-covariance operator in Appendix E.1. For keys and values, we correspondingly define them as 



where φ(K) = (φ(k<sup>1</sup> ), . . . , φ(k<sup>L</sup> ))<sup>⊤</sup> ∈ R<sup>L×dφ</sup> and ϕ(V ) = (ϕ(v<sup>1</sup> ), . . . , ϕ(v<sup>L</sup> ))<sup>⊤</sup> ∈ R<sup>L×dϕ</sup> By the definition of the newly defined attention in Section 4.1, we have that 



which implies that attn† recovers the empirical conditional mean embedding. By (E.2), it holds that 



Upper bounding term (i) of (E.5). Following the proof from Song et al. (2009), we only need to upper bound ∥C<sup>�</sup> VK(C<sup>�</sup> KK + L<sup>−1</sup> λI)<sup>−1</sup> − CVK(CKK + L<sup>−1</sup> λI)<sup>−1</sup> ∥. It holds that 



where the last inequality follows from the fact that 



Combining (E.7) and (E.6), we have 



In the following, we will upper bound the second term on the right-hand side of (E.8) with Lemma I.1. For this purpose, we define ξ : R<sup>dp</sup> × R<sup>d</sup> →Hk ⊗Hv as follows, 



22 

Since the operator norm of (CKK + L<sup>−1</sup> λI)<sup>−1</sup> is upper bounded by (L<sup>−1</sup> λ)<sup>−1</sup> , we have that 

��ξ(k, v)�� = ��(CKK + L−1λI)−1�� · ��ϕ(v)�� · ��φ(k)�� ≤ C · (L−1λ)−1, 

where C > 0 is an absolute constant. Additionally, we can bound the expectation of the squared norm of ξ(k, v) as 



Using the definition of the trace operator, we have 



Here Γ(L<sup>−1</sup> λ) is the effective dimension of CKK in Caponnetto and De Vito (2007), which is defined as follows, 



We apply Lemma I.1 with B = C(L<sup>−1</sup> λ)<sup>−1</sup> and σ<sup>2</sup> = (L<sup>−1</sup> λ)<sup>−1</sup> · Γ(L<sup>−1</sup> λ), then we have that with probability at least 1 − δ, the following holds 



where C > 0 is an absolute constant. Similarly, we can prove that with probability at least 1 − δ, the following holds 



Here C<sup>′</sup> > 0 is an absolute constant. Combining (E.8), (E.9), and (E.10), we have with probability at least 1 − δ that 



Upper bounding term (ii) of (E.5). We follow the procedures in the proof from Fukumizu (2015). For any g ∈Hk, we have that 



23 

Similarly, for any q ∈ R<sup>dp</sup> and any g ∈Hk, we have that 



Taking g = (CKK + L<sup>−1</sup> λI)<sup>−1</sup> K(q, ·), we have that 



We note that E[L(v, ¯v) | k = ·, k<sup>¯</sup> = ‡] ∈Hk ⊗Hk is in the range spanned by CKK ⊗ CKK. Thus, we can define C<sup>�</sup> ∈Hk × Hk such that (CKK ⊗ CKK)C<sup>�</sup> = E[L(v, ¯v) | k = ·, k<sup>¯</sup> = ‡]. Let {λi}<sup>∞</sup> i=1 and {ϕi}<sup>∞</sup> i=1<sup>betheeigenvaluesandeigenvectorsofCKK,respectively.Wethenhavethat</sup> 



Thus, we have 



where C > 0 is an absolute constant. 

Combining (E.5), (E.11), and (E.12), we have with probability at least 1 − δ, the following holds 



Since K is Gaussian RBF kernel, we have that Γ(L<sup>−1</sup> λ) = O(L/λ). 

Step 2: Build the relationship between the attn and conditional mean embedding. 

We achieve our goal in two sub-steps. In the first step, we prove that there exists a constant C > 0 such that 



24 

where S<sup>d−1</sup> is the (d − 1)-dimensional unit sphere. Here P<sup>�K</sup> V | K<sup>is the kernelconditionaldensity</sup> estimation of PV | K defined as follows, 



where ι = 1/ �S<sup>d−1 K(k, q)dqisanormalizationconstant.Notethatιdoesnotdependonthe</sup> value of k by symmetry. We transform the right-hand side of this equality as 



Thus, it suffices to calculate the integration term�S<sup>d−1 v · K(vℓ, v)dv.Tothisend,wehavethe</sup> following lemma. 

Proposition E.1. Let K(a, b) = exp(a<sup>⊤</sup> b/γ) be the exponential kernel with a fixed γ > 0. It holds for any b ∈ S<sup>d−1</sup> that 



where C1 > 0 is an absolute constant. 

Proof. See Section H.1 for a detailed proof. 



Thus, it holds for the right-hand side of (E.15) that 



where the first equality follows from the definition of the softmax function and the second equality follows from the definition of the softmax attention. 

The second step is to relate the right-hand side of (E.14) to conditional mean embedding. In fact, under the condition that P<sup>�K∈Sdp−1as L →∞,</sup> V | K<sup>(v | q) →P(v | q) uniformly for any q</sup> we have 



Thus, we have that 



for some constant C > 0. Combining (E.16) and (E.13) and choosing λ = L<sup>3/4</sup> , we complete the proof of Proposition 4.3. 



25 

## F Appendix for Section 5 

### F.1 Supplemental Definitions for Markov Chains 

We follow the notations in Paulin (2015). Let Ωbe a Polish space. The transition kernel for a time-homogeneous Markov chain {Xi}<sup>∞</sup> i=1<sup>supported on Ωis a probability distribution P(x, dy)</sup> for every x ∈ Ω. Given X1 = x1, · · · , Xt−1 = xt−1, the conditional distribution of Xt equals P(xt−1, dy). A distribution π is said to be a stationary distribution of this Markov chain if �x∈Ω<sup>P(x, dy)π(dx)=π(dy).WeadoptPt(x, ·)todenotethedistributionofXtconditioned</sup> on X1 = x. The mixing time of the chain is defined by 



### F.2 Proof of Theorem 5.3 

Proof of Theorem 5.3. Our proof mainly involves three steps. 

- Error decomposition with the PAC-Bayes framework. 

- Control each term in the error decomposition. 

- Conclude the proof. 

Step 1: Error decomposition with the PAC-Bayes framework. 

For ease of notation, we temporarily write Tp and Np as T and N, respectively. Recall that the pretraining dataset is D = {(St<sup>n, xn</sup> t+1<sup>)}N,T</sup> n,t=1<sup>,whichconsistsofNtrajectories(es-</sup> says), and each essay have T + 1 words. Given St<sup>n,thenextwordisgeneratedasxn</sup> t+1<sup>∼</sup> P�(· | St<sup>n),andS</sup> t<sup>n</sup> +1<sup>=(S</sup> t<sup>n, xn</sup> t+1<sup>).Here,weconstructaghostsampleD�={(S�</sup> t<sup>n, �xn</sup> t+1<sup>)}N,T</sup> n,t=1<sup>as</sup> St<sup>n=S</sup> t<sup>nandx�n</sup> t+1<sup>∼P(· | �S</sup> t<sup>n)independentlyfromD.Wedefinefunctiong(θ)=L(θ, D) −</sup> log ED �<sup>[exp(L(θ,D�)) | D],where</sup> 



For distributions Q, P ∈ ∆(Θ), where P can potentially depends on D, Lemma I.3 shows that 



Substituting the definition of g(θ) and taking expectation with respect to the distribution of D on the both sides of the inequality, we can derive that 



With Chernoff inequality, we can show that with probability at least 1 − δ, the following holds 



26 

We first cope with the left-hand side of (F.1). 



where the first inequality results from the definition of L(θ, D) and Cauchy-Schwarz inequality, the equality results from that the transitions of x�<sup>n</sup> t+1<sup>areindependentgivenD,andthelast</sup> inequality results from Lemma I.5. The second term in the right-hand side of (F.2) can be controlled if the distribution P is chosen to concentrate around θ<sup>�</sup> . This will be done in Step 2. Now we consider the right-hand side of (F.1). For any θ<sup>∗</sup> ∈ Θ, we can decompose it as 



where the inequality results from the fact that θ<sup>�</sup> maximizes the likelihood. We will choose θ<sup>∗</sup> as the projection of P onto {Pθ | θ ∈ Θ}, i.e., P<sup>∗</sup> θ<sup>is the bestapproximation of P withrespectto</sup> the KL divergence. Thus, the first term in the right-hand side of (F.3) is the approximation error. The second term in the right-hand side of (F.3) can be controlled in the same way as the second term in the right-hand side of (F.2). Combining inequalities (F.1), (F.2), and 

27 

(F.3), we have that 



where term (I) is the fluctuation error induced by θ ∼ P , term (II) is the approximation error, and term (III) is the KL divergence between P and Q. 

Step 2: Control each term in the error decomposition. 

We first consider term (I). We need to quantify the fluctuation of Pθ when θ is changing. 

Proposition F.1. For any input X ∈ R<sup>L×d</sup> and θ, θ<sup>�</sup> ∈ Θ, we have that 



where 



for all t ∈ [D]. 

Proof of Proposition F.1 . See Appendix H.3. 



28 

With the help of Proposition F.1, we set the distribution P as 



for t ∈ [D], where Unif denotes the uniform distribution on the set, B(a, r, ∥·∥) = {x | ∥x−a∥≤ r} denotes the ball centered in a with radius r, the radius is set as 



Under this assignment, we now bound | log Pθ�(x | S)/Pθ(x | S)| for any S ∈ R<sup>L×d</sup> and x ∈ R<sup>dy</sup> . We note that 



for any S and x. This results from the fact that 

� If TV(Pθ(· | S), Pθ(· | S)) = ε ≤ by/2, some basic calculations show that 





Thus, we have 



Based on this, we conclude that 



29 

Next, we control term (III) in (F.4). We take Q as 



Then the KL divergence between P and Q is 



Finally, we control term (II) in (F.4). This term can be controlled as 



The first two terms in the right-hand side of the equality is the generalization error, which can be bounded with Lemma I.4. With Assumption 5.2, we note that 



so the function satisfies the condition in Lemma I.4 with ci = 2b<sup>∗</sup> . Using the moment generating function bound in Lemma I.4 and Chernoff bound, we have that 



with probability at least 1 − δ. 

Step 3: Conclude the proof. 

30 

Combining inequalities (F.4), (F.7), (F.9), and (F.11), we have that 



where we take θ<sup>∗</sup> as the best approximation parameters. Finally, we will change the left-hand side of this inequality to the expectation of it. In fact, we have that 

Proposition F.2. Let F be the collection of functions of f : R<sup>n</sup> → R, and we assume that |f | ≤ b for any function f ∈F . For a Markov chain X = (X1, ·, XN ), we define f (X) =<sup>�N</sup> i=1<sup>f(Xi)/N.ThemixingtimeofthisMarkovchainisdenotedastmix(ε).Givena</sup> distribution Q on F , with probability at least 1 − δ, we have 



for any distribution P on F simultaneously with probability at least 1 − δ, where 



#### Proof of Proposition F.2. See Appendix H.2. 



We note that Proposition F.2 is indeed an uniform convergence bound, since it holds simultaneously for all P . Thus, we can set P and Q as those in equalities (F.5) and (F.8), then we have that 



Thus, we have that 



31 

We conclude the proof of Theorem 5.3. 



### F.3 Formal Statement and Proof of Proposition 5.4 

Denote the alphabet of the language as X ⊆ R (d = 1), then the conditional distribution P<sup>∗</sup> can be viewed as a function g<sup>∗</sup> : X<sup>L</sup> → R<sup>dy</sup> , where L is the maximal length of a sentence, and the output is the distribution of the next word. Since A is finite, Theorem 2 in Zaheer et al. (2017) shows that there exist ρ<sup>∗</sup> : R → R<sup>dy</sup> and φ<sup>∗</sup> : X → R such that 



where X = [x1, · · · , xL]. The i<sup>th</sup> component of ρ<sup>∗</sup> is denoted as ρ<sup>∗</sup> i<sup>fori ∈[dy].Forafunction</sup> f defined on Ω, the L<sup>∞</sup> norm of it is defined as ∥f ∥∞ = supx∈Ω |f (x)|. The set of the realvalued smooth functions on it is denoted as S<sup>∞</sup> (Ω, R), Then we denote the set of the smooth functions with bounded derivatives as 



where f<sup>(n)</sup> is the n<sup>th</sup> -order derivative of f . 

Assumption F.3. There exists B > 0 such that φ<sup>∗</sup> , τ log ρ<sup>∗</sup> i<sup>∈SBfori ∈[dy].</sup> 

This assumption states that the function g<sup>∗</sup> is smooth enough for transformers to approximate. 

Proposition F.4. Under Assumptions 5.2 and F.3, if dF ≥ 16dy, BA,1 ≥ 16Rdy, BA,2 ≥ dF BA ≥ �dy, and BV ≥ √d, then 



for some constant C > 0. 

Proof of Proposition F.4. Our proof mainly involves three steps. 

- Build the high-level transformer approximator for g<sup>∗</sup> . 

- Build the approximators in the transformer for φ<sup>∗</sup> and ρ<sup>∗</sup> i<sup>separately.</sup> 

- Conclude the proof. 

#### Step 1: Build the high-level transformer approximator for g<sup>∗</sup> 

Without loss of generality, we assume that B > 1 in Assumption F.7. To approximate φ<sup>∗</sup> , we ignore the attention module in the transformer by setting Wi<sup>V,(t)</sup> = 0, γ1<sup>(t)</sup> = 1, γ2<sup>(t)</sup> = 0 for all i ∈ [h]. We further set A<sup>(</sup> 2<sup>t)</sup> = IdF ∈ R<sup>dF ×dF</sup> , which is the identity matrix. The network structure now is 



32 

where b<sup>(t+1)</sup> ∈ R is the bias term. In Step 2, we will use this fully-connected network to L approximate φ<sup>∗</sup> . To approximate the average L<sup>1</sup> �i=1<sup>φ∗(xi),wetakeW</sup> i<sup>Q,(t)</sup> = 0, Wi<sup>K,(t)</sup> = 0, and Wi<sup>V,(t)</sup> = Id, γ1<sup>(t)</sup> = 0, γ2<sup>(t)</sup> = 1, A<sup>(</sup> 2<sup>t)</sup> = 0. After this average aggregation, we still take Wi<sup>V,(t)</sup> = 0, γ1<sup>(t)</sup> = 1, γ2<sup>(t)</sup> = 0 for all i ∈ [h] and A<sup>(</sup> 2<sup>t)</sup> = IdF ∈ R<sup>dF ×dF</sup> to approximate ρ<sup>∗</sup> i<sup>for</sup> i ∈ [dy]. We stack the approximators for ρ<sup>∗</sup> i<sup>toapproximateρ∗,multiplyingthewidthofthe</sup> networks by dF . 

Step 2: Build the approximators in the transformer for φ<sup>∗</sup> and ρ<sup>∗</sup> i<sup>separately.</sup> In the first and the D<sup>th</sup> layer, we take A<sup>(1)</sup> 1<sup>,′</sup> = A<sup>(1)</sup> 1<sup>/RandA</sup> 1<sup>(D),′</sup> = A<sup>(</sup> 1<sup>D)</sup> · R to normalize and retrieve the magnitudes of inputs, where R is the range of the inputs. This will keep the magnitudes of the intermediate outputs small. Next, we will use Lemma I.9 to construct the networks. In the proof of Lemma I.9, the norm of the outputs of the intermediate layers do not excess the range of the inputs, so the layer normalization in our networks will not influence the constructed approximators. In this case, we can respectively approximate φ<sup>∗</sup> and ρ<sup>∗</sup> i<sup>with</sup> fully-connected networks Ψφ∗ and Ψρ<sup>∗</sup> i<sup>fori ∈[dy]as</sup> 



where the depth D(·), the width W (·), and the maximal weight B(·) of the networks satisfy that 



for some constant C > 0. The bounds for width and maximal weight require that dF ≥ 16dy and BA,1 ≥<sup>√</sup> dF · dF ≥ 16dy. Then we have that for any X = (x1, · · · , xL) 



where the first inequality results from the triangle inequality, (BA,1)<sup>D′′</sup> in the second inequality results from the error propagation through a depth-D<sup>′′</sup> network. For the whole network, we have that 



We take that D<sup>′</sup> = D/2 + D<sup>3/4</sup> and D<sup>′′</sup> = √D/(√C · B log BA,1) for the constant C in Lemma I.9. Then for D > 3, we have that 



Step 3: Conclude the proof. 

33 

We denote Ψρ∗(<sup>�L</sup> i=1<sup>Ψφ∗(xi)/L)asPθ∗.ThenifTV(P(· | X), Pθ∗(· | X))=ε≤c0/2,some</sup> basic calculations show that 





Thus, we have 





### F.4 Pretraining Results for ℓ2 Loss 

#### F.4.1 Pretraining Algorithm with ℓ2 Loss 

Training with ℓ2 loss is common in the CV community, e.g. Radford et al. (2021). The network structure is largely similar to those in Brown et al. (2020) and Devlin et al. (2018). Here, we modify the network structure of the last layer. The network derives the final output as Y<sup>(D+1)</sup> = L<sup>1I</sup> L<sup>⊤X(D)A(D+1),whereIL∈RListhevectorwithallones,A(D+1)∈Rd×dy.The</sup> parameters in each layer are θ<sup>(t)</sup> = (γ1<sup>(t), γ</sup> 2<sup>(t), W (t), A(t))fort∈[D],andθ(D+1)=A(D+1),</sup> and the parameters of the whole network is θ = (θ<sup>(1)</sup> , · · · , θ<sup>(D+1)</sup> ). Similar to Section 5.1, we consider the transformer with bounded weights. The set of parameters is 



where BA, BA,1, BA,2, BQ, BK, and BV are the bounds of parameter. We only consider the non-trivial case where these bounds are larger than 1, otherwise the magnitude of the output in D<sup>th</sup> layer decades exponentially with growing depth. We denote the transformer with parameter θ as fθ. 

In such case, we focus on the pretraining setting in CV tasks, i.e., the pretraining set D = {(S<sup>i</sup> , x<sup>i</sup> )}<sup>N</sup> i=1<sup>consistsofi.i.d.pairs.Theunderlyingdistributionisdenotedas(S, x)∼</sup> µ ∈ ∆(X<sup>∗</sup> × X). In such case, d = dy, i.e., the transformer directly predicts the musked token. The training algorithm is 



From the population version of (F.12), it is easy to see that the function f<sup>∗</sup> (S) = E[x | S] achieves the minimal population error, where the conditional expectation is defined from µ. In the following, we will quantify the error between fθ� and f<sup>∗</sup> . 

#### F.4.2 Performance Guarantee for Pretraining with ℓ2 Loss 

We first state the assumptions for the pretraining setting. 

34 

Assumption F.5. There exists a constant R > 0 such that for (S, x) ∼ µ, we have ∥S<sup>⊤</sup> ∥2,∞ ≤ R and ∥x∥2 ≤ Bx almost surely. 

Then the performance guarantee for the pretraining result θ<sup>�</sup> can be derived as following. 

Theorem F.6. Let B<sup>¯</sup> = BxRhBABA,1BA,2BQBKBV and D<sup>¯</sup> = D<sup>2</sup> d(dF + dh + d) + d · dy. If Assumption F.5 holds, the pretrained model fθ� by the algorithm in (F.12) satisfies 



with probability at least 1 − δ. 

The first term is the approximation error. It measures the proximity between the nominal function f<sup>∗</sup> and the functions induced by the parameter set Θ. The second term is the generalization error. Similar as Theorem 5.3, the generalization error is independent of the token sequence length. 

Since the neural networks are universal approximators, we will explicitly approximate f<sup>∗</sup> from the transformer function class. Theorem 2 in Zaheer et al. (2017) shows that there exist ρ<sup>∗</sup> : R → R<sup>dy</sup> and φ<sup>∗</sup> : R → R such that 



where X = [x1, · · · , xL]. The i<sup>th</sup> component of ρ<sup>∗</sup> is denoted as ρ<sup>∗</sup> i<sup>fori ∈[dy].Forafunction</sup> f defined on Ω, the L<sup>∞</sup> norm of it is defined as ∥f ∥∞ = supx∈Ω |f (x)|. The set of the realvalued smooth functions on it is denoted as S<sup>∞</sup> (Ω, R), Then we denote the set of the smooth functions with bounded derivatives as 



where f<sup>(n)</sup> is the n<sup>th</sup> -order derivative of f . 

Assumption F.7. There exists B > 0 such that φ<sup>∗</sup> , ρ<sup>∗</sup> i<sup>∈SBfori ∈[dy].</sup> 

This assumption states that the function f<sup>∗</sup> is smooth enough. Then we have that 

Proposition F.8. Under F.7, if dF ≥ 16dy, BA,1 ≥ 16Rdy, BA,2 ≥ dF BA ≥ �dy, and BV ≥ √d, then 



for some constant C > 0. 

35 

#### F.4.3 Proof of Theorem F.6 

Proof of Theorem F.6. For ease of notation, we respectively define the empirical risk and the population risk as 



The our proof mainly involves three steps. 

- Error decomposition for the excess population risk. 

- Control each term in the error decomposition. 

- Conclude the proof. 

Step 1: Error decomposition for the excess population risk. The excess population risk for the estimate θ<sup>�</sup> can be decomposed to the sum of the generalization error and the approximation error as 



where θ<sup>∗</sup> = argminθ∈Θ L(fθ), and the inequality results from that θ<sup>�</sup> achieves the minimal empirical risk. 

Step 2: Control each term in the error decomposition. 

We first consider the generalization error and will adapt Lemma I.2 to bound it. Define the function 



To verify the conditions in Lemma I.2, we notice that |g(S, x, θ)| ≤ (Bx + Bf )<sup>2</sup> and that 



where the second equality results from the definition of f<sup>∗</sup> , the second inequality results from Cauchy–Schwarz inequality, and the last inequality result from the boundedness of x, f<sup>∗</sup> , and 

36 

fθ. Then Lemma I.2 shows that for a distribution Q ∈ ∆(Θ) and 0 < λ ≤ 1/(2(Bx + Bf )<sup>2</sup> ), the following holds with probability at least 1 − δ simultaneously for all P ∈ ∆(Θ) 



Taking λ = 1/(2(3Bx + Bf )<sup>2</sup> ), we have 



Next, we will take proper P and Q to relate this equation and the generalization error. For this purpose, we quantify how the perturbation of network parameters influence the output of the network. 

Proposition F.9. For any input X ∈ R<sup>L×d</sup> and θ, θ<sup>�</sup> ∈ Θ, we have that 



where 



for all t ∈ [D]. 

Proof of Proposition F.9 . See Appendix H.4. 



37 

With the help of Proposition F.9, we set the distribution P as 



for t ∈ [D], where Unif denotes the uniform distribution on the set, B(a, r, ∥·∥) = {x | ∥x−a∥≤ r} denotes the ball centered in a with radius r, the radius is set as 



Under this assignment, we now bound Eθ∼P [∥x − fθ(S)∥<sup>2</sup> 2<sup>−∥x −f</sup> θ<sup>�(S)∥</sup> 2<sup>2]as</sup> 



where the inequality results from Cauchy-Schwarz inequality, and the equality results from Proposition F.9. Thus, we have that 



To access to the value of KL(P ∥ Q), we take Q as the distribution in (F.8) except that 



Then the KL divergence between P and Q is 



Combining this equality with (F.15), we have that with probability at least 1 − δ, the generalization error can be bounded as 



38 

Next we control the approximation error in (F.13). 



where the second equality results from the definition of f<sup>∗</sup> . To bound the first two terms in the right-hand side of (F.18), we use Lemma I.2 and take P and Q as (F.14) and (F.16), replacing θ<sup>�</sup> by θ<sup>∗</sup> . Then we have that 



#### Step 3: Conclude the proof. 

Combining inequalities (F.13), (F.17), (F.18), and (F.19), we have that 



Thus, we conclude the proof of Theorem F.6. 



#### F.4.4 Proof of Proposition F.8 

Proof of Proposition F.8. Our proof mainly involves three steps. 

- Build the high-level transformer approximator for f<sup>∗</sup> . 

- Build the approximators in the transformer for φ<sup>∗</sup> and ρ<sup>∗</sup> i<sup>separately.</sup> 

- Conclude the proof. 

The first two steps follow the procedures of the proof of Proposition F.4 exactly. Now we present the final step. 

#### Step 3: Conclude the proof. 

In the final layer, we just take A<sup>(D+1)</sup> = Idy as the identity matrix. Denoting the derived parameters as θ<sup>∗</sup> we have that 



Thus, we conclude the proof of Proposition F.8. 



39 

## G Proofs and Formal Statements for **§** 6 

### G.1 Proof of Theorem 6.2 

Proof. By Corollary 4.2 and the fact that log(1/p0(z∗)) ≤ β, we have that 



In addition, we have that 



Similar to (F.10), we have that 



By Lemma I.10, we have that 



By Assumption 6.1, we have that PDICL(prompt) ≤ κPD(prompt). Thus, by Theorem 5.3, we have with probability at least 1 − δ that 



Combining (G.4), (G.1), and (G.2), we have with probability at least 1 − δ that 



which completes the proof of Theorem 6.2. 



### G.2 Assumptions and Formal Statement for Prompting With Wrong Input-Output Mappings 

We first state assumptions for this setting. 

Assumption G.1. Conditioned on any z ∈ Z, the input-output pairs are independent, i.e., for any two input-output pair sequences St, St<sup>′′∈X∗,wehaveP((St, S</sup> t<sup>′′) | z) = P(St | z) · P(S</sup> t<sup>′′ | z).</sup> 

40 

This assumption states that for any task z ∈ Z, the input-output pairs are independently generated. This largely holds in realistic applications since the examples usually are independently produced. It can be relaxed when there are more structures in the token generation process, e.g. the hidden Markov model in Xie et al. (2021). 

Assumption G.2. There exists a constant c1 > 0 such that PZ(z∗) ≥ c1. 

This assumption states that the prior distribution of the hidden concept z∗ is strictly larger than 0, otherwise this concept can never be deduced. For two concepts z, z<sup>′</sup> ∈ Z, we define the KL divergence between the conditional distributions of input-output pair on them as KLpair(P(· | z)∥P(· | z<sup>′</sup> )) = EX,y∼P(· | z)[log(P(X, y | z)/P(X, y | z<sup>′</sup> ))]. This divergence measures the distance between distributions of input-output pairs conditioned on different tasks z and z<sup>′</sup> . 

Assumption G.3. The concept z∗ satisfies that minz=z∗ KLpair(P(· | z∗) ∥ P(· | z)) > 2 log 1/c0, where c0 is the constant in Assumption 5.2. 

This distinguishability assumption requires that the divergence between z∗ and other concepts z is large enough to infer the concept z∗ from the prompt. We denote the pretraining error in Theorem 5.3 as ∆pre(Np, Tp, δ), then we have the following result. 

Proposition G.4. Under Assumptions 5.2, G.1, G.2, and G.3, the pretrained model Pθ� in (5.2) predicts the outputs with the prompt containing wrong mappings as 



with probability at least 1 − δ. 

### G.3 Proof of Proposition G.4 

Proof of Proposition G.4. From Bayesian model averaging, the output distribution is 



where the first equality results from Bayesian model averaging, the last equality results from Bayes’ theorem. Next, we upperbound the ratio P(St<sup>′| z)/P(S</sup> t<sup>′| z∗)intheright-handsideof</sup> Eqn. (G.6). We have that 



41 

where the equality results from Assumption G.1, and the inequality results from Assump� � tion 5.2. Assumption 5.2 also implies that | log P((ci, ri) | z)/P((ci, ri) | z<sup>∗</sup> )| ≤ (1 + l) log 1/c0. Hoeffding inequality shows that with probability at least 1 − δ, we have 



Thus, we have that with probability at least 1 − δ, the following holds for all z = z<sup>∗</sup> 



Combining this inequality with Eqn. (G.6), we have that 

Taking expectations with respect to the distribution of St<sup>′, �ct+1onthebothsidesin(G.7),we</sup> have that 



We set δ = |Z exp(−a√t/2b)|, where a = minz=z∗ KLpair�P(· | z<sup>∗</sup> ) ∥ P(· | z)� + 2 log c0, b = −(1 + l) log c0. Then the right-hand side of (G.8) can be upper bounded as 



Let Eprompt′[TV(P(· | St<sup>′, �ct+1), P</sup> θ<sup>�(· | S</sup> t<sup>′, �ct+1))]≤∆pre(Np, Tp, δ),where∆pre(Np, Tp, δ)isthe</sup> bound in Theorem 5.3. Then we have that 



where the first equality results from Assumption 5.2. Thus, we conclude the proof of Proposition G.4. 

42 

## H Proof of Supporting Propositions 

### H.1 Proof of Proposition E.1 

Proof. Let a, b be two vectors in the (d − 1)-dimensional unit sphere S<sup>d−1</sup> . We first define the following vector, 



By direct calculation, we have the following property of c defined in (H.1), 



By (H.1) and (H.2), we have that 



We now calculate the desired integration. Note that 



For the second term on the right-hand side of (H.4), it follows from (H.1) and (H.2) and (H.3) that 



where the equality follows from the fact that dc = 2∥b∥<sup>2</sup> 2<sup>da −da = da.Byreplacingcbyaon</sup> the right-hand side of (H.5), we have 



Finally, by plugging (H.6) into (H.4), we obtain that 



Thus, by setting 



we complete the proof of Proposition E.1. Note that here C1 is an absolute constant that does not depend on b due to the symmetry on the unit sphere. 

43 

### H.2 Proof of Proposition F.2 

Proof of Proposition F.2. We note that f (X) satisfies the condition in Lemma I.4 with ci = 2b/N for i ∈ [N]. Then Lemma I.4 shows that 



Take λ = �2N log 2/(b<sup>2</sup> tmin). The Markov inequality shows that 



for any 0 < δ < 1. We note that this probability inequality does not involve P . Take the function g in Lemma I.3 as g(f ) = λ(f (X) − Ef (X)), then it shows that 



for any P simultaneously. Combining these inequalities, we have 



for any distribution P on F simultaneously with probability at least 1 − δ. Thus, we conclude the proof of Proposition F.2. 

### H.3 Proof of Proposition F.1 

Proof of Proposition F.1 . We analyze the error layer by layer in the neural network. Denote the outputs of each layer in the networks parameterized by θ and θ<sup>�</sup> as X<sup>(t)</sup> and X<sup>�(t)</sup> , respectively. In the final layer, we have that 



where the first inequality results from Lemma I.6, and the second inequality results from Lemma I.7 and that ∥X<sup>(D),⊤</sup> ∥2,∞ ≤ 1 due to the layer normalization. In the following, we build the recursion relationship between ∥X<sup>(t),⊤</sup> − X<sup>�(t),⊤</sup> ∥2,∞ for t ∈ [D]. 





where the first inequality results from the triangle inequality and that Πnorm is not expansive, the second inequality results from the following proposition 

44 



Proof of Proposition H.1. See Appendix H.5. 



Next, we build the relationship between ∥Y<sup>(t+1),⊤</sup> − Y<sup>�(t+1),⊤</sup> ∥2,∞ in the right-hand side of inequality (H.7) and ∥X<sup>(t),⊤</sup> − X<sup>�(t),⊤</sup> ∥2,∞. 



where the first inequality results from the triangle inequality, and the second inequality results from Lemma I.8. Combining inequalities (H.7) and (H.8), we derive that 



This concludes the proof of Proposition F.1. 



### H.4 Proof of Proposition F.9 

Proof of Proposition F.9 . We analyze the error layer by layer in the neural network. Denote the outputs of each layer in the networks parameterized by θ and θ<sup>�</sup> as X<sup>(t)</sup> and X<sup>�(t)</sup> , respectively. In the final layer, we have that 



where the inequality results from Lemma I.7 and that ∥X<sup>(D),⊤</sup> ∥2,∞ ≤ 1 due to the layer normalization. The remaining proof just follows the procedures in the proof of Proposition F.1, and we have that 



Thus, we conclude the proof of Proposition F.9. 



45 

### H.5 Proof of Proposition H.1 

Proof of Proposition H.1. We have that 



where the first inequality results from the triangle inequality, the second and the last inequalities result from Lemma I.7 and that ReLU is not expansive. Thus, we conclude the proof of Proposition H.1. 

## I Technical Lemmas 

Lemma I.1 (Caponnetto and De Vito (2007)). Let (Ω, ν) be a probability space and ξ be a random variable on Ωtaking value in a real separable Hilbert space H. We assume that there exists constants B, σ > 0 such that 



Then, it holds with probability at least 1 − δ that 



Lemma I.2 (Proposition 4.5 in Duchi (2019)). Let F be the collection of functions of f : R<sup>n</sup> → R. For any f ∈F , we define 



where the expectation is taken with respect to a random variable X ∼ ν on (R<sup>n</sup> , B(R<sup>n</sup> )). Assume that |f (X) − µ(f )| ≤ b a.s. for some constant b ∈ R for all f ∈F . Then for any 0 < λ ≤ 1/(2b), given a distribution P0 on F , with probability at least 1 − δ, we have 



for any distribution Q on F , where Xi are i.i.d. samples of ν. If the function class F further satisfies σ<sup>2</sup> (f ) ≤ cµ(f ) for some constant c ∈ R for all f ∈F , we have 



with probability at least 1 − δ. 

46 

Lemma I.3 (Donsker–Varadhan representation in Belghazi et al. (2018)). Let P and Q be distributions on a common space X . Then 



where G = {g : X → R | EQ[exp(g(X))] < ∞}. 

Lemma I.4 (Corollary 2.11 in Paulin (2015)). Let X = (X1, · · · , XN ) be a Markov chain, taking values in Λ =<sup>�N</sup> i=1<sup>Λiwithmixingtimetmix(ε)forε ∈[0, 1].Let</sup> 



If function f : Λ → R is such that f (x) − f (y) ≤<sup>�N</sup> i=1<sup>ciIx</sup> i<sup>=y</sup> i<sup>for everyx, y∈Λ,thenforany</sup> λ ∈ R, 



For any t ≥ 0, we have 



Lemma I.5 (Lemma 25 in Agarwal et al. (2020)). For any two conditional probability densities P (· | X), P<sup>′</sup> (· | X) and any distribution ν ∈ ∆(X ),we have 



Lemma I.6 (Corollary A.7 in Edelman et al. (2021) ). For any x, y ∈ R<sup>d</sup> , we have 

∥softmax(x) − softmax(y)∥1 ≤ 2∥x − y∥∞. 

Lemma I.7 (Lemma 17 in Zhang et al. (2022a) ). Given any two conjugate numbers u, v ∈ [1, ∞], i.e., u<sup>1+</sup> v<sup>1= 1,and1 ≤p ≤∞,foranyA ∈Rr×candx ∈Rc,wehave</sup> 



Lemma I.8 (Propositions 20 and 21 in Zhang et al. (2022a)). For any X, X<sup>�</sup> ∈ R<sup>L×d</sup> , and any Wi<sup>Q, �</sup> Wi<sup>Q, W</sup> i<sup>K, �</sup> Wi<sup>K</sup> ∈ R<sup>d×dh</sup> , Wi<sup>V, �</sup> Wi<sup>V</sup> ∈ R<sup>d×d</sup> for i ∈ [h] , if ∥X<sup>⊤</sup> ∥p,∞, ∥X<sup>�⊤</sup> ∥2,∞ ≤ BX, ∥Wi<sup>Q∥F, ∥</sup> W<sup>�</sup> i<sup>Q∥F≤BQ,∥W</sup> i<sup>K∥F, ∥</sup> W<sup>�</sup> i<sup>K∥F≤BK,∥W</sup> i<sup>V∥F, ∥</sup> W<sup>�</sup> i<sup>V∥F≤BVfori∈[h],thenwe</sup> have 



Lemma I.9 (Lemma A.6 in Elbr¨achter et al. (2021)). For a, b ∈ R with a < b, let 



There exists a constant C > 0 such that for all a, b ∈ R with a < b, f ∈S[a,b], and ε ∈ (0, 1/2), there is a fully connect network Ψf such that 



with the depth of the network as D(Ψf ) ≤ C max{2, b − a}(log ε<sup>−1</sup> )<sup>2</sup> + log(⌈max{|a|, |b|}⌉) + log(⌈1/(b − a)⌉), the width of the network as W (Ψf ) ≤ 16, and the maximal weight in the network as B(Ψf ) ≤ 1. 

Lemma I.10. Let b = supx log(p(x)/q(x)). We have that 



Proof. We let f (t) = log t and g(t) = |1/t − 1|. Then, for 0 ≤ t ≤ exp(b), we have that 



Note that KL(p ∥ q) = Ep[f (p(x)/q(x))] and TV(p, q) = Ep[g(p(x)/q(x))], which concludes the proof. 

48 

