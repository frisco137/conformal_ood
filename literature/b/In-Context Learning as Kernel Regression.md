Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

# **Understanding Emergent In-Context Learning from a Kernel Regression Perspective** 

**Chi Han Ziqi Wang Han Zhao Heng Ji** _Siebel School of Computing and Data Science University of Illinois Urbana-Champaign_ 

_chihan3@illinois.edu ziqiw9@illinois.edu hanzhao@illinois.edu hengji@illinois.edu_ 

**Reviewed on OpenReview:** _`https: // openreview. net/ forum? id= 6rD50Q6yYz`_ 

## **Abstract** 

Large language models (LLMs) have initiated a paradigm shift in transfer learning. In contrast to the classic pretraining-then-finetuning procedure, in order to use LLMs for downstream prediction tasks, one only needs to provide a few demonstrations, known as in-context examples, without adding more or updating existing model parameters. This in-context learning (ICL) capability of LLMs is intriguing, and it is not yet fully understood how pretrained LLMs acquire such capabilities. In this paper, we investigate the reason why a transformer-based language model can accomplish in-context learning after pre-training on a general language corpus by proposing a kernel-regression perspective of understanding LLMs’ ICL bahaviors when faced with in-context examples. More concretely, we first prove that Bayesian inference on in-context prompts can be asymptotically understood as kerˆ nel regression _y_ =<sup>�</sup> _i_<sup>_yiK_(</sup><sup>_x, xi_)</sup><sup>_/_�</sup> _i_<sup>_K_(</sup><sup>_x, xi_)asthenumberofin-contextdemonstrations</sup> grows. Then, we empirically investigate the in-context behaviors of language models. We find that during ICL, the attention and hidden features in LLMs match the behaviors of a kernel regression. Finally, our theory provides insights into multiple phenomena observed in the ICL field: why retrieving demonstrative samples similar to test samples can help, why ICL performance is sensitive to the output formats, and why ICL accuracy benefits from selecting in-distribution and representative samples. Code and resources are publicly available at `https://github.com/Glaciohound/Explain-ICL-As-Kernel-Regression` . 

## **1 Introduction** 

Pre-trained large language models (LLMs) have emerged as powerful tools in the field of natural language processing, demonstrating remarkable performance across a broad range of applications (Wei et al., 2022a; Kojima et al., 2023; Wei et al., 2022b; Brown et al., 2020; Li et al., 2023). They have been used to tackle diverse tasks such as text summarization, sentiment analysis, schema induction and translation, among others (Brown et al., 2020; Radford et al., 2023; Li et al., 2023). One of the most fascinating capabilities of LLMs is their ability to perform in-context learning (ICL), a process in which a language model can make predictions on a test sample based on a few demonstrative examples provided in the input context (Logan IV et al., 2022). This feature makes LLMs particularly versatile and adaptive to different tasks. Studies have found ICL to emerge especially when the size of LLM is large enough and pre-trained over a massive corpus (Wei et al., 2023). 

Although intuitive for human learners, ICL poses a mystery for optimization theories because of the significant format shift between ICL prompts and pre-training corpus. There have been lots of efforts to provide a theoretical understanding of how LLMs implement ICL. Some work (Xie et al., 2022; Wang et al., 2023) approaches this problem from a data perspective: they claim that ICL is possible if a model masters Bayesian 

1 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 



<!-- Start of picture text -->
x y<br>demonstrative  Input: moving and important.   Output: Positive.<br>samples Input: excruciatingly unfunny and pitifully unromantic.  Output: Negative.<br>Input: the plot is nothing but boilerplate clichés from start to finish.  Output: Negative.<br>…<br>test input Input: intelligent and moving Output: ________<br>70%: “Positive”<br>K ( x i ,  x test ) y  = ∑ i K ( x i ,  x test ) yi<br>(similarity kernel) ∑ i K ( x i ,  x test )<br><!-- End of picture text -->

Figure 1: Our results suggests that LLMs might be conducting kernel regression on ICL prompts. 

inference on pre-training distribution. Our paper also follows the setting in Xie et al. (2022) that models the natural language as a mixture of sub-distributions and that LLMs conduct Bayesian inference on it while focusing on providing a tractable approximation of ICL inference via kernel regression along with insights into ICL behaviors. Another stream of work conjectures that under a simple linear setting: [ _x, y_ ] where _y_ = _w_<sup>_⊤_</sup> _x_ and the input sequence _x_ only has length 1, they can construct a Transformer (Vaswani et al., 2017) to implement gradient descent (GD) algorithm over ICL prompt (Akyürek et al., 2023; von Oswald et al., 2022; Garg et al., 2023). However, this constrained setting diverges from the most interesting part of ICL, as state-of-the-art LLMs work with linguistic tasks where the sequential textual inputs have complex semantic structures, and ICL emerges from pre-training on general-purpose corpus instead of explicit ICL training. 

In this work, we delve deeper into the question of _how to understand the mechanism that enables Transformerbased pre-trained LLMs to accomplish in-context learning on sequential data_ . We specifically explore the hypothesis that LLMs employ a kernel regression algorithm when confronted with in-context prompts. Kernel regression adopts a non-parametric form 



when making predictions, where _K_ ( _x, xi_ ) is a kernel that measures the similarity between inputs _x_ and _xi_ . In plain words, it estimates the output _y_ ˆ on _x_ by drawing information from similar other data points _xi_ and taking a weighted sum on their _yi_ . 

We first provide a theoretical analysis demonstrating that Bayesian inference predictions on in-context prompts converge to a kernel regression in Section 4. In Section 4.2, our results also shed light on numerous phenomena observed in previous empirical studies, such as the advantage of retrieving in-context examples that are similar to the test sample, the sensitivity of ICL performance to the output formats, and why using a group of in-distribution and representative samples improves ICL accuracy. 

Following our theoretical investigation, in Section 5 we conduct empirical studies to verify our explanation of in-context learning of LLMs in more detail. Our results reveal that during LLM ICL, the attention map used by the last token to predict the next token is allocated in accordance with our explanation. By plugging attention values into our equation, we are also able to reconstruct the model’s output with over 80% accuracy. Moreover, we are able to reveal how information necessary to kernel regression is computed in intermediate LLM layers. 

## **2 Related Work** 

### **2.1 In-Context Learning** 

As an intriguing property of large language models, in-context learning has attracted high attention in the research community. There have been numerous studies on empirical analysis of in-context learning, 

2 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

including format and effects of in-context samples (Min et al., 2022b;a; Zhao et al., 2021), selection of incontext samples (Liu et al., 2022; Lu et al., 2022), characteristics of in-context learning behaviors (Zhao et al., 2021; Lu et al., 2022; Khashabi et al., 2022; Wei et al., 2023), and the relation between ICL and pre-training dataset (Wu et al., 2022; Chan et al., 2022). 

Theoretical analysis of ICL is another active research direction. One branch of studies investigates ICL from a data perspective: under some general assumptions of the data, (Xie, 2021; Wang et al., 2023) demonstrate that with sufficient pre-training data and a good enough generative model of the data, ICL could be understood as performing Bayesian inference. However, they do not fully explain if such Bayesian inference is computationally feasible in practical language models since such Bayesian inference involves unbounded depth computational graphs as the number of samples increases. Our study builds on top of some similar assumptions but goes further to explain how ICL can be accomplished with the attention mechanism in Transformers Vaswani et al. (2017). Recently, there is controversy regarding whether LLMs always conduct Bayesian inference in the ICL setting, and Falck et al. (2024); Raventós et al. (2023); Park et al. (2024) show that there exist scenarios where LLMs’ predictions deviate from the predictions of Bayesian inference. This distinction can be quantified as an error term in Equation 7 related to _ϵθ_ which we use to measure the deviation between ICL task and pre-training data. Raventós et al. (2023); Lu et al. (2024) additionally show that under certain conditions, this divergence enables Transformers to achieve stronger performance than Bayesian inference. Both Raventós et al. (2023); Park et al. (2024) discover that there is a threshold needed where ICL behaviors become less similar to Bayesian inference. 

Another perspective to explain ICL is by analyzing what algorithms might be implemented in LLMs under ICL settings, which is closely related to our study. Representative studies include (Akyürek et al., 2023; Garg et al., 2023; von Oswald et al., 2022; Dai et al., 2022; Mahankali et al.; Ren & Liu, 2023), with the majority of them proposing gradient descent (GD) algorithm as a promising candidate answer. More recently, Fu et al. (2024) show that transformers can learn to approximate second-order optimization methods for ICL, such as the Iterative Newton’s method, which is exponentially faster than GD. These studies are mostly constructive, in the sense that there exist parameters of a transformer that can be used to approximate certain algorithms. On the other hand, theoretical explanations of whether or why such parameters might be learned during pre-training on language data are less explored. Furthermore, such settings usually focus on a specific type of question, such as linear regression, to construct the algorithm. 

Going beyond the linear setting, Guo et al. constructs solutions of ridge regression assuming non-linear representations on synthetic datasets. On top of individual algorithms, Bai et al. (2024) explores the possibility of Transformers selecting from a pool of algorithms. Some following-up work analyzed how training dynamics converge Transformer to ICL solutions on linear functions Zhang et al. (2024); Huang et al.; Ahn et al. (2023), and when linear-attention Transformers is trained on linear tasks Lu et al. (2024). Kim & Suzuki analyzed the training landscape of Transformers on ICL inputs, including the avoidance of saddle points and proximity to global optima. These attempts in explicit constructions of algorithms in Transformers mostly assume a relatively simplified setting with input lengths equal to 1, and evaluate Transformers after training on a synthetic dataset (including (Akyürek et al., 2023; Garg et al., 2023)). This is different from ICL’s main advantage as an emergent ability on language pre-training, and LLMs can work on textual data which involves sentences with lengths longer than 1. 

### **2.2 Emergent Ability of LLMs** 

This is a larger topic that in-context learning is also highly related to. (Wei et al., 2022a; Brown et al., 2020; Kojima et al., 2023; Wei et al., 2022b) showed that abilities including reasoning, in-context learning, few-shot learning, instruction understanding, and multilingualism emerge in large language models after pre-training on massive language data. These impressive and mysterious capacities have boosted significant progress in natural language processing as well as artificial intelligence, but still baffle theoretical analysis. In this work, we make a preliminary step towards understanding ICL as a special case of LM capacity emergence. 

3 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

### **2.3 Associating Attention with Kernel Regression** 

Chen & Li (2023) presents a related view of associating self-attention with kernel ridge regressions. They focus on a different problem and propose to substitute attention mechanisms with a kernel to tackle the uncertainty calibration problem. Another blog, Olsson et al. (2022), also relates attention to kernel regression but defines ICL as “decreasing loss at increasing token indices”, rather than the more widely adopted definition of learning from in-context demonstrations as in ours. Another related stream of work is the topic of associative memory in deep learning networks. Ramsauer et al. revisits Hopfield networks which is capable of reusing past patterns for inference on new data. Arora et al.; Bietti et al. (2023) explore components in Transformers which might be responsible for associative recalling. 

## **3 Formulation** 

### **3.1 Preliminaries: Hidden Markov Models** 

Following the setting of (Xie et al., 2022), we assume that the pre-training corpus can be modeled by a mixture of HMMs. Each HMM corresponds to a certain task _θ ∈_ Θ. Assuming a large finite number of tasks, one can include all task-specific HMMs into one single HMM. In this unified HMM, let _S_ be the set of states, and _O_ be the set of observations where _|O|_ = _m_ . At each time step, state _st_ randomly emits one observation _ot_ and then transits to the next state _st_ +1. Quantities _p_ pre-train, _P_ ( _st_ +1 = _s_<sup>_′_</sup> _|st_ = _s_ ) and _P_ ( _ot_ = _o|st_ = _s_ ) denote the pre-training initial distribution, transition distribution, and emission distribution respectively. Under an arbitrary ordering of _S_ and _O_ , we can define the transition matrix _T_ : _T_ ( _s, s_<sup>_′_</sup> ) = _P_ ( _s_<sup>_′_</sup> _|s_ ), and emission matrix _B_ : _B_ ( _s, o_ ) = _P_ ( _o|s_ ), respectively. We also let **o** = ( _o_ 0 _, · · ·_ ) be the full observation sequence, and **o** [0: _l_ ] denote its first _l_ tokens. 

### **3.2 In-Context Learning** 

In this work, we consider the following formulation of in-context learning (ICL). Let Θ be the set of tasks. We follow the ICL prompt formulation, the HMM mixture formulation, and assumptions in (Xie et al., 2022) as follows. The distribution of sequences generated by each individual task in the HMM together composes the pre-training distribution. Specifically, each task _θ ∈_ Θ is associated with a distinct initial state _sθ ∈S_ with transition rate lower bound _ϵd_ (the Assumption 5 “Regularity” in Xie et al. (2022)), and the set of all such initial states _S_ start = _{sθ|θ ∈_ Θ _}_ forms the support of _p_ pre-train. For a test task _θ_<sup>_⋆_</sup> , the in-context learning prompt follows the format: 



where the input-output pairs [ **x** _i, yi_ ] are i.i.d. demonstrate samples sampled from _θ_<sup>_⋆_</sup> , and _o_<sup>delim</sup> is the delimiter token with emission rate lower bound _ϵr_ (the Assumption 5 “Regularity” in Xie et al. (2022)). We further make some connections between in-context learning and the HMM model. Note that the probability of generating a sequence from the initial distribution _p_ 0 can be expressed as follows(Jaeger, 2000): 



where **p** _o_ is vector of emission probabilities _P_ ( _o|s ∈S_ ) for _o_ . We denote the intermediate matrices in Equation 3 as one operator for a (sub)sequence **o** : 



We use a matrix Σ _p,l_ to denote the covariance between all of its _d_<sup>2</sup> elements of vec( _T_ **o** [0: _l−_ 1]) when **o** [0: _l−_ 1] is generated from initial distribution _p_ . For each individual task, we denote _ϵθ_ = inf _l ρ_ (Σ<sup>_−_</sup> _p_ pre-train<sup>1</sup> _,l_<sup>_−_Σ</sup><sup>_−_</sup> _sθ_<sup>1</sup> _,l_<sup>)to</sup> quantify the difference between sequences generated by _sθ_ and those from pre-training distribution, where 

4 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

_ρ_ denotes the spectral radius of a matrix. Let _η_ = sup **o** [0: _l−_ 1] _∥T_ **o** [0: _l_ ] _∥F_ be the upper bound of _T_ **o** [0: _l_ ]’s Frobenius-norm. 

### **3.3 Assumptions** 

We go on and present the assumptions we borrow from Xie et al. (2022). 

**Assumption 1.** _(Delimiter Tokens) The delimiter token indicates the start of sampling of new sequences:_ 



**Remark** : this means that the delimiter tokens are indicative enough, such as the start of a new line before the beginning of a paragraph. 

**Assumption 2.** _(Task Distinguishability) The Kullback–Leibler divergence (KL divergence) between the first l tokens between two distinct tasks θ_ = _θ_<sup>_′_</sup> _is lower-bounded by:_ 



**Remark** : this requires that tasks are distinguishable from each other. As KL-divergence is non-decreasing with length _l_ , it suffices to increase the length _l_ to provide sufficient task information. 

## **4 Theoretical Analysis** 

### **4.1 Explaining ICL as Kernel Regression** 

Within the framework presented in Section 3, we pose the following result. The basic idea is that, as the number of samples _n_ increases, inference on the in-context learning prompt converges to a kernel-regression form. 

**Theorem 1.** _Let us denote a kernel_ 



_T_ **x** _is defined in Equation 4. Let_ **e** ( _y_ ) _be the one-hot vector for index y. Then the difference between the following logit vector in the form of kernel regression:_ 



_and it converges polynomially to the conditional likelihood of Y conditioned on ICL prompt_ **o** _ICL, P_ ( _Y |_ **o** _ICL, ppre-train_ ) _, with probability_ 1 _− δ:_ 



Equation 6 can be interpreted as follows: it calculates the semantic similarity between the test input **x** test and each sample **x** _i_ and aggregates their outputs to compute a most likely prediction for the test sample. This is natural to the motivation of ICL: we encourage the LLM to leverage the pattern provided in demonstrative samples and mimic the pattern to predict the test input. Equation 6 is also similar to the form of attention mechanism used in Transformer decoder models: 



where _q_ is the query vector corresponding to the last token, _k, K_ are the key vectors and matrix, and _v, V_ are the value vectors and matrix used in the Transformer, respectively. The only difference is that _e_<sup>_<q,ki>_</sup> 

5 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

is replaced with a dot product in Equation 6, which can be regarded as a kernel trick. We assume that previously hidden layers are responsible for learning the semantic vectors of sample inputs vec( _T_ **x** ). We can then make the following loose analogy between our kernel regression explanation (Equation 6) and the attention mechanism (Equation 8): 

- Label information **e** ( _yi_ ) corresponds with the _value_ vector _vi_ 

- The similarity kernel vec( _T_ **x** )<sup>_⊤_</sup> Σ<sup>_−_</sup> _p_ pre-train<sup>1</sup> _,_<sup>vec(</sup><sup>_T_</sup><sup>**x**</sup><sup>_′_)looselycorrespondstotheattentionvalue</sup><sup>_e⟨q,ki⟩_,</sup> where: 

- the semantic information vec( _T_ **x** _i_ ) corresponds to the _key_ vectors _ki_ and _query_ vectors _qi_ for samples [ **x** _, y_ ]. 

Here we further explain how the kernel is effected by the LLM. Equation 6 measures similarity between **x** and **x**<sup>_′_</sup> on the space of vec( _T_ **x** ) and vec( _T_ **x** _′_ ). This flattened vector of _T_ **x** defines the “belief state” in HMMs, which determines the conditional probability _P_ ( _·|_ **x** ) after prefix **x** . This is the pre-training objective functionality of LLMs. Therefore, if _xi_ and _xtest_ have similar follow-up conditional probabilities under the LLM, their similarity value will be larger, and vice versa. 

One might argue that it is also theoretically possible to directly compute the next token likelihood in Equation 3. However, this form involves 2 _n_ consecutive matrix multiplications. When _n_ increases, this is infeasible for a practical Transformer architecture which is composed of a fixed number of layers. In comparison, Equation 6 only requires semantic information for each sample **x** to be provided beforehand and then applies kernel regression (which can be done by one attention layer) to get the answer. Learning to represent _T_ **x** is probable for preceding layers, as it is also used for ordinary inference _P_ ( _y |_ **x** ) = _p_<sup>_⊤_</sup> pre-train<sup>_T_</sup><sup>**x**.</sup> In experiments in Section 5 we demonstrate that this analogy can explain the ICL behaviors of LLMs to an extent. 

### **4.2 Insights Provided by the Explanation** 

Theorem 1 can provide insights into multiple phenomena in the ICL field observed by previous studies. This is helpful for understanding and predicting the behaviors of ICL, and providing heuristics for future development. 

**Retrieving Similar Samples** It is empirically observed (Rubin et al., 2022; Liu et al., 2022) that retrieving demonstrative samples **x** _i_ that is similar to the test input **x** test can benefit ICL performance. This phenomenon is understandable from our explanation. Encouraging the selection of similar samples can be understood as limiting the cosine distance between demonstrative samples **x** _i_ and test sample **x** test in sentence embedding space. This is similar to selecting a smaller “bandwidth” in kernel regression and sampling only from a local window, which reduces the bias in kernel regression. Therefore, devising better retrieval techniques for selecting samples, especially those with similar representations as LLMs, is a promising direction for further boosting ICL scores. 

**Sensitivity to Label Format** (Min et al., 2022b) also observes that the ICL performance relies on the label format. Replacing the label set with another random set will reduce the performance of ordinary auto-regressive LLMs. This can be explained in our theory that the model’s output comes from a weighted voting of demonstrative sample labels _{yi}_ . If the label space is changed, the next token will also be misled to a different output space. So it is generally beneficial for ICL to ensure the demonstrative samples and test samples share an aligned label space and output format. 

Table 1: Effect of OOD inputs on ICL on SST2 dataset. 

|OOD-type|None|rare word<br>complex|typo|
|---|---|---|---|
|accuracy|0.805|0.677<br>0.788|0.534|



**Sample Selection** In Equation 7, the existence of the _η_<sup>2</sup> _ϵθ_ term implies that the demonstrations [ **x** _i, yi_ ] should be sampled in a way close to _p_ pre-train. To verify this, we conduct an experiment by converting test inputs to semantically similar but rarer sentences. Specifically, on the SST2 dataset, we prompt GPT3.5-turbo to generate semantically similar while rarer 

6 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 



<!-- Start of picture text -->
head 1<br>layer 1  head 2<br>layer 2  head 3<br>layer 3  head 4<br>layer 4  head 5<br>layer 5  head 6<br>layer 6  …<br>layer 7  head 16<br>layer 8<br>layer 9<br>layer 10<br>layer 11<br>layer 12<br>layer 13<br>layer 14<br>layer 15<br>layer 16<br>layer 17<br>layer 18<br>layer 19<br>layer 20<br>layer 21<br>layer 22<br>layer 23<br>layer 24<br>layer 25<br>layer 26<br>layer 27<br>layer 28<br>x 1 y 1 x 2 y 2 x 3 y 3 x 4 y 4 x 5 y 5 x 6 y 6 x n yn x test<br>…<br>sample 1 sample 2 sample 3 sample 4 sample 5 sample 6 sample  n<br><!-- End of picture text -->

Figure 2: Averaged attention map over GLUE-sst2 test set. A portion of attention on demonstrative samples is generally focused on label positions _yi_ . This conforms to the intuition in Theorem 1 that the inference on in-context learning prompts is a weighted average over sample labels. 

(i.e. OOD) expressions of inputs and use them to substitute the original dataset inputs. We consider three types of OOD types. “Rare word“ is where words are substituted with rare synonyms. “Complex“ is to express the original sentence in a more complex structure. “Typo“ is where we require adding typos to the original input. The results are listed below. We see that compared with original scores (“None“ ), these OOD types more or less decrease the ICL accuracy, with “typo“ having the largest effect, dropping the accuracy to near random. 

**Bias from Pre-training Data** Theorem 1 implies that the final prediction depends both on in-context examples and prior knowledge from pre-training. This can be seen from the _η_<sup>2</sup> _ϵθ_ term in Equation 7, which comes from the pre-training information in Equation 17 in Appendix A. Specifically, _ϵθ_ measures the similarity between _θ_ ’s distribution and pre-training distribution, and _η_<sup>2</sup> _ϵθ_ describes how the pre-training distribution’s bias affects the ICL prediction. When the task _θ_ emits a sequence distribution similar to the pretraining distribution, _ϵθ_ will be small, and the ICL prediction _y_ ˆ will converge to _P_ ( _Y |_ **o** _ICL, pp_ pre-train) with a smaller error _η_<sup>2</sup> _ϵθ_ , and vise versa. This partially elucidates Kossen et al. (2023)’s finding that “ICL cannot overcome prediction preferences from pre-training.” 

**Remaining Challenges** However, we need to point out that there are still phenomena not explainable by our framework, as well as most previous explanations. One most mysterious one is the sensitivity to sample ordering (Lu et al., 2022) as a kernel regression should be order-ignorant, which no existing explanations (including (Xie et al., 2022; Akyürek et al., 2023)) take into account. Another intriguing question is that some work finds LLMs robust to perturbed or random labels(Kossen et al., 2023) while others find the 

7 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 



<!-- Start of picture text -->
Layer Index<br>Head Index<br><!-- End of picture text -->



<!-- Start of picture text -->
Layer Index<br>Head Index<br><!-- End of picture text -->

ˆ (a) Accuracy compared with model output _y_ . (b) Accuracy compared with ground truth label _y_ test. 

Figure 3: We use each head’s attention weights on demonstrative samples to manually average sample labels _yi_ . These figures show that “reconstructed” outputs in some heads from layers 16 _∼_ 21 matches LLM prediction with as high as 89.2% accuracy, and matches ground truth with 86.4% accuracy. 

opposite (Min et al., 2022b). We attribute such phenomena to the fact that LLMs also rely on a large portion of implicit reasoning in text generation and might benefit from linguistic cues in text prompts. Our theory provides a partial explanation, which needs to be combined with this implicit ability of LLMs to form a more comprehensive understanding of ICL. 

## **5 Empirical Analysis** 

In this section, we conduct empirical analysis on LLMs in order to verify our hypothesis. Because Equation 6 is only one special solution among infinitely many of its equivalent forms, and it also relies on the unknown HMM structure, it is infeasible to directly evaluate it on data. However, we can verify if it can predict observable behaviors on LLMs in experiments. Limited by computation resources in an academic lab, we analyze the GPT-J 6B model(Wang & Komatsuzaki, 2021) on one Tesla V100. It employs a decoder-only Transformer architecture. In this section, we use the validation set of the sst2 dataset as a case study, while results on more tasks can be found in Appendix B. We investigate the ICL behavior of LLMs from shallow to deep levels, and sequentially ask the following 4 questions: Do the attention heads collect label information **e** ( _yi_ ) as predicted? Does the attention-kernel analogy explain LLM’s prediction? Can we actually explain the attention values as a kind of similarity? Can we find where algorithmic features **e** ( _yi_ ) _, T_ **x** _i_ are stored? The following sections answer these questions one by one. 

### **5.1 Where Are Attentions Distributed During ICL?** 

First, we notice that Equation 2 implies that the LLM takes a weighted average over sample labels _yi_ in ICL. Figure 2 shows how attention weights are distributed on in-context learning inputs [ _Sn,_ **x** test]. On each test point, we sample one ICL prompt and collect the attention map over previous tokens for predicting the next token. After getting the attention maps, as ICL samples **x** _i_ may have varied lengths, we re-scale the attentions on each **x** from _|_ **x** _|_ to a fixed 30-token length with linear interpolation. After aligning the attention lengths, we average all attention maps. The horizontal axis is the aligned positions on prompt [ _Sn,_ **x** test]. Each bar corresponds to one of 28 Transformer layers. Within each bar, each thin line is 1 out of 16 attention heads. Darker (blue) areas mean smaller averaged attention, while brighter areas indicate high attention. 

In Figure 2, there are three major locations of attention masses. First, a majority of attention is focused on the final few tokens in **x** test, especially in the first 3 layers. This accords with previous observations that Transformer attentions tend to locate in a local window to construct local semantic feature for **x** test. Secondly, as also observed in previous studies, LLMs tend to allocate much attention on the first few tokens in a sequence to collect starter information. Finally and most intriguingly, we observe concentrated attention on each sample label tokens _{yi}_ . This phenomenon confirms an aggregation of label information in LLM ICL, in line with the prediction by Equation 6. Note that our explanation does not specify or limit the 

8 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

model from implementing kernel regression in a particular layer. In fact, an equivalent mechanism can occur in one or more layers as long as these (possibly redundant) results can be passed with skip connections to the final layer and aggregated. 

### **5.2 Can Attentions Be Interpreted as Kernel Functions?** 

Now that we observe expected locations of attention weights on labels, we go on to verify if the LLM really predicts by averaging on labels as suggested by Theorem 1. We iterate over 16 heads and 28 layers, and insert their attention weights into Equation 6 to manually average the label distribution. This is similar in concept to a mind-reading experiment to predict one’s next word using brain waves only (Wang & Ji, 2022). Specifically, for each attention head, we use the maximal attention value _ai_ within the range of [ **x** _i, yi_ ] as the kernel weight. Then on the ICL samples, we reconstruct the prediction as follows: 



The resulting “reconstructed output” _y_ ˜ is compared for both LLM’s actual prediction _y_ ˆ and ground truth label _y_ test to calculate its accuracy. Figure 3a and 3b plot the accuracy between _y_ ˜ and _y_ ˆ and between _y_ ˜ and _y_ test respectively. Interestingly, we spot the existence of several heads in layers 18 _∼_ 21 which demonstrate high accuracy in reconstruction. The highest of them (layer 17, head 10) achieves 89.2% accuracy on _y_ ˆ and 86.4% accuracy on _y_ test. This result validates our hypothesis that some components in Transformer-based LLMs implement kernel regression. Note that this phenomenon happens within a few adjacent layers in the middle of the model. This is similar to our prediction in Section 4.1: not many attention layers are needed for kernel regression, as long as the required features have been computed by preceding layers. It is enough for the higher layers to only pass on the computed results. 

### **5.3 Which Samples Receive High Attention?** 



Figure 4: Pearson correlation between sample’s attentions and _prediction similarity_ simpred( **x** test _,_ **x** _i_ ) (Equation 9). _x_ -axis are layers and _y_ -axis are heads in each layer. Note the resemblence between this heatmap and Figure 3. 

We go on and ask the question: if the LLMs use attention to implement kernel regression, _what kind of similarity does this kernel function evaluate?_ From Equation 6, we see that the dot product is measuring similarity between _T_ **x** , which encodes information necessary for HMM inference: _p_ ( _o|_ **x** ) = _p_<sup>_⊤_</sup> pre-train<sup>_T_</sup><sup>**x**</sup><sup>_B_.</sup> Therefore, we conjecture that the attention value _ai_ between **x** test and **x** _i_ correlates with their prediction similarity. Specifically, we define the _prediction similarity_ as follows: 

sim( **x** 1 _,_ **x** 2) = _P_ ( _o|_ **x** 1)<sup>_⊤_</sup> _P_ ( _o|_ **x** 2) _,_ (9) 

which is measured by applying LLMs on these texts _alone_ , rather than in ICL prompt. Finally, we compute the Pear- 

son correlation coefficient between sim( **x** tes _,_ **x** _i_ ) and each attention values on samples for each attention head. The results are shown in Figure 4. The absolute value of correlation is not high, as _P_ ( _o|_ **x** ) is a dimension reduction to _T_ **x** and can lose and mix information. However, we can still note a striking similarity between it and Figure 3. This means that the heads responsible for ICL mostly attend to _prediction-similar_ samples. 

### **5.4 Do Intermediate Features Store Information Useful for Kernel Regression?** 

In this section, we go into a more detailed level, and investigate the question: _where do Transformer-based LMs store the algorithmic information needed by kernel regression?_ To this end, we take out the intermediate 

9 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 



<!-- Start of picture text -->
Accuracy Accuracy<br>(a) Predicting arg max o P ( o| x i ) with key vectors. (b) Predicting yi with value vectors.<br>Layer Number Relative Position Layer Number Relative Position<br><!-- End of picture text -->

Figure 5: Key and value vectors encode label and LLM prediction information at high-attention position. Here y-axis denotes the relative position to the high-attention position in each sample. Each sphere is an attention head. The curve shows average accuracy within each layer. 

_key_ and _value_ features in all layer heads and see if the correct information is stored in the correct locations. Note in Section 5.1, we observe that a major part of attention weights are located at the label position, so we focus on positions within [ _−_ 1 _,_ 3] relative to this position. Noticing the analogy we made at Section 4.1 that _ki ∼_ vec( _T_ **x** _j_ ) and _vj ∼ yj_ , we study two sub-questions: (1) whether _value_ vectors encode label information _yi_ ; and (2) whether _key_ vectors encode LLM prediction information _P_ ( _o|_ **x** _i_ ). For each head, we conduct Ridge regression with _λ_ = 0 _._ 01 to fit the task in these 2 questions. Results are presented in Figure 5. We can observe that, generally the high-attention position (y-axis = 0) indeed achieves the best accuracy. Figure 5b is intuitive, as tokens at a position later than the label token _yi_ can easily access the information of _yi_ by self-attention. The slight drop at position +3 means that a longer distance introduces more noise to this information flow. Results in Figure 5a tell us that, although sentence **x** _i_ ’s starting position in ICL prompt is shifted and different from 0, _ki_ is still strongly correlated with _P_ ( _o|_ **x** _i_ ), which indicates a sense of translation invariance. Overall, the results mean that, with the attention map distributed in Figure 2, LLM is able to use the attention mechanism to extract information regarding _T_ **x** _i_ and _yi_ from _key_ and _value_ vectors effectively just as we described. 

### **5.5 How Well Can Our Explanation Reconstruct ICL Prediction?** 

Finally, we numerically evaluate our explanation on its ability to reconstruct the ICL predictions and tasks. We uniformly use 700 data in each validation set to represent tasks in a balanced way. We select the attention head that has the highest correlation with model predictions. This head is then evaluated on ICL prediction reconstruction and task performance on a held-out set of 300 data per task. In Table 2 we see that the ICL output reconstruction has an accuracy from 68% to 80% except for the harder task of MNLI. The task performance of reconstructed outputs matches the model’s performance level. It also achieves similar or superior performance than kernel regression on sentence encoders such as all-MiniLM-L6-v2 and bert-base-nli-mean-tokens (Reimers & Gurevych, 2019). 

## **6 Conclusions and Future Work** 

In conclusion, our work provides a novel theoretical view to understand the intriguing in-context learning (ICL) capabilities of Transformer-based large language models (LLMs). We propose that LLMs ICL can be understood as kernel regression. Our empirical investigations into the in-context behaviors of LLMs reveal that the model’s attention and hidden features during ICL are congruent with the behaviors of kernel regression. Furthermore, our theory also explains several observable phenomena in the field of ICL: why the retrieval of demonstrations similar to the test sample can enhance performance, the sensitivity of ICL to output formats, and the benefit from selecting in-distribution and representative samples. There are still 

10 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

Table 2: Performance of explicit kernel regression (KR) and LLM ICL on downstream tasks. 

|**Method**|**sst2**|**mnli**|**rotten-**<br>**tomatoes**|**tweet_eval**<br>**(hate)**|**tweet_eval**<br>**(irony)**|**tweet_eval**<br>**(ofensive)**|
|---|---|---|---|---|---|---|
|**GPT-J-6B ICL**|0.805|0.383|0.671|0.539|0.519|0.542|
|**ICL output reconstruction**|0.797|0.597|0.683|0.713|0.722|0.753|
|**Reconstructed performance**|0.750|0.360|0.527|0.587|0.522|0.513|
|**all-MiniLM-L6-v2**|0.503|0.321|0.478|0.548|0.491|0.588|
|**bert-base-nli-mean-tokens KR**|0.523|0.325|0.502|0.545|0.479|0.597|



remaining challenges in this topic, such as understanding the effect of sample orderings and the robustness to perturbed labels. These questions, along with understanding other perspectives of LLMs, are exciting questions for future research. 

## **7 Limitations** 

This work is based on several assumptions, such as the HMM mixture setting described in Section 3, and more specifically the assumptions in Section 3.3. Though theoretically, the HMM can be exhaustively constructed to express any finite-length distribution (e.g., by using the sequences of prefix tokens as hidden states _S_ = _{_ [ _o_ 0 _, o_ 1 _, . . . , oi_ ] _|i ∈_ N<sup>_⋆_</sup> _, oi ∈O}_ ), it might not be an elegant model of natural language. It does not fully capture the full complexity and generality of natural language with unbounded lengths, and might not be the most efficient model considering the large number of hidden states needed. 

There is also controversy that challenges the Bayesian inference view of ICL behaviors. Some work such as Falck et al. (2024) is discussed in Section 2. Besides, not all ICL phenomena can be explained as Bayesian inference, such as those discussed in Section 4.2. The divergence from kernel-regression behaviors can be partly explained by the first error term in Equation 7, as detailed in Section 2. 

## **Acknowledgments** 

This work was supported in part by US DARPA KAIROS Program No. FA8750-19-2-1004 and AIDA Program No. FA8750-18-2-0014. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation here on. 

## **References** 

- Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to implement preconditioned gradient descent for in-context learning. _Advances in Neural Information Processing Systems_ , 36: 45614–45650, 2023. 

- Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. In _The Eleventh International Conference on Learning Representations_ , 2023. URL `https://openreview.net/forum?id=0g0X4H8yN4I` . 

- Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Re. Zoology: Measuring and improving recall in efficient language models. In _The Twelfth International Conference on Learning Representations_ . 

- Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians: Provable incontext learning with in-context algorithm selection. _Advances in neural information processing systems_ , 36, 2024. 

11 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

- Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herve Jegou, and Leon Bottou. Birth of a transformer: A memory viewpoint. _Advances in Neural Information Processing Systems_ , 36:1560–1588, 2023. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- Stephanie Chan, Adam Santoro, Andrew Lampinen, Jane Wang, Aaditya Singh, Pierre Richemond, James McClelland, and Felix Hill. Data distributional properties drive emergent in-context learning in transformers. _Advances in Neural Information Processing Systems_ , 35:18878–18891, 2022. 

- Wenlong Chen and Yingzhen Li. Calibrating transformers via sparse gaussian processes. _arXiv preprint arXiv:2303.02444_ , 2023. 

- Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can gpt learn incontext? language models implicitly perform gradient descent as meta-optimizers. In _ICLR 2023 Workshop on Mathematical and Empirical Understanding of Foundation Models_ , 2022. 

- Fabian Falck, Ziyu Wang, and Christopher C Holmes. Is in-context learning in large language models bayesian? a martingale perspective. In _International Conference on Machine Learning_ , pp. 12784–12805. PMLR, 2024. 

- Deqing Fu, Tian-Qi Chen, Robin Jia, and Vatsal Sharan. Transformers learn to achieve second-order convergence rates for in-context linear regression. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_ , 2024. 

- Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. In _Advances in Neural Information Processing Systems_ , 2023. 

- Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

- Tianyu Guo, Wei Hu, Song Mei, Huan Wang, Caiming Xiong, Silvio Savarese, and Yu Bai. How do transformers learn in-context beyond simple functions? a case study on learning with representations. In _The Twelfth International Conference on Learning Representations_ . 

- Yu Huang, Yuan Cheng, and Yingbin Liang. In-context convergence of transformers. In _Forty-first International Conference on Machine Learning_ . 

- Herbert Jaeger. Observable operator models for discrete stochastic time series. _Neural computation_ , 12(6): 1371–1398, 2000. 

- Daniel Khashabi, Chitta Baral, Yejin Choi, and Hannaneh Hajishirzi. Reframing instructional prompts to gptk’s language. In _Findings of the Association for Computational Linguistics: ACL 2022_ , pp. 589–612, 2022. 

- Juno Kim and Taiji Suzuki. Transformers learn nonlinear features in context: Nonconvex mean-field dynamics on the attention landscape. In _Forty-first International Conference on Machine Learning_ . 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In _Advances in Neural Information Processing Systems_ , 2023. 

- Jannik Kossen, Tom Rainforth, and Yarin Gal. In-context learning in large language models learns label relationships but is not conventional learning. _arXiv preprint arXiv:2307.12375_ , 2023. 

- Sha Li, Ruining Zhao, Manling Li, Heng Ji, Chris Callison-Burch, and Jiawei Han. Open-domain hierarchical event schema induction by incremental prompting and verification. In _Proc. The 61st Annual Meeting of the Association for Computational Linguistics (ACL2023)_ , 2023. 

12 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

- Jiachang Liu, Dinghan Shen, Yizhe Zhang, William B Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for gpt-3? In _Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures_ , pp. 100–114, 2022. 

- Robert Logan IV, Ivana Balažević, Eric Wallace, Fabio Petroni, Sameer Singh, and Sebastian Riedel. Cutting down on prompts and parameters: Simple few-shot learning with language models. In _Findings of the Association for Computational Linguistics: ACL 2022_ , pp. 2824–2835, 2022. 

- Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 8086–8098, 2022. 

- Yue M Lu, Mary I Letey, Jacob A Zavatone-Veth, Anindita Maiti, and Cengiz Pehlevan. Asymptotic theory of in-context learning by linear attention. _arXiv preprint arXiv:2405.11751_ , 2024. 

- Arvind V Mahankali, Tatsunori Hashimoto, and Tengyu Ma. One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. In _The Twelfth International Conference on Learning Representations_ . 

- Sewon Min, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Noisy channel language model prompting for few-shot text classification. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 5316–5330, 2022a. 

- Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? _arXiv preprint arXiv:2202.12837_ , 2022b. 

- Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. _arXiv preprint arXiv:2209.11895_ , 2022. 

- Core Francisco Park, Ekdeep Singh Lubana, Itamar Pres, and Hidenori Tanaka. Competition dynamics shape algorithmic phases of in-context learning. _arXiv preprint arXiv:2412.01003_ , 2024. 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. 2023. 

- Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Lukas Gruber, Markus Holzleitner, Thomas Adler, David Kreil, Michael K Kopp, et al. Hopfield networks is all you need. In _International Conference on Learning Representations_ . 

- Allan Raventós, Mansheej Paul, Feng Chen, and Surya Ganguli. Pretraining task diversity and the emergence of non-bayesian in-context learning for regression. _Advances in neural information processing systems_ , 36: 14228–14246, 2023. 

- Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing_ . Association for Computational Linguistics, 11 2019. URL `http://arxiv.org/abs/1908.10084` . 

- Ruifeng Ren and Yong Liu. Towards understanding how transformers learn in-context through a representation learning lens. _arXiv preprint arXiv:2310.13220_ , 2023. 

- Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. In _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 2655–2671, 2022. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

13 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. _arXiv preprint arXiv:2212.07677_ , 2022. 

- Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. `https://github.com/kingoflolz/mesh-transformer-jax` , May 2021. 

- Xinyi Wang, Wanrong Zhu, and William Yang Wang. Large language models are implicitly topic models: Explaining and finding good demonstrations for in-context learning. _arXiv preprint arXiv:2301.11916_ , 2023. 

- Zhenhailong Wang and Heng Ji. Open vocabulary electroencephalography-to-text decoding and zero-shot sentiment classification. In _Proc. Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI2022)_ , 2022. 

- Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. _Transactions on Machine Learning Research_ , 2022a. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed H Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In _Advances in Neural Information Processing Systems_ , 2022b. 

- Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, et al. Larger language models do in-context learning differently. _arXiv preprint arXiv:2303.03846_ , 2023. 

- Yuhuai Wu, Felix Li, and Percy S Liang. Insights into pre-training via simpler synthetic tasks. _Advances in Neural Information Processing Systems_ , 35:21844–21857, 2022. 

- Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. In _International Conference on Learning Representations_ , 2022. 

- Wanying Xie. Gx at semeval-2021 task 2: Bert with lemma information for mcl-wic task. In _Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021)_ , pp. 706–712, 2021. 

- Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear models in-context. _Journal of Machine Learning Research_ , 25(49):1–55, 2024. 

- Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models. In _International Conference on Machine Learning_ , pp. 12697–12706. PMLR, 2021. 

## **A Proofs** 

_Proof._ First, we denote the kernel regression function 



In expectation, 

14 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 



As vec( _·_ ) is a linear function, if we replace vec( _T_ **x** _i_ )<sup>_⊤_</sup> with _T_ **x** _i_ : 



and then 



As ( **x** _i,_ **e** ( _yi_ )) can be seen as independent samples from _P_ ( _Y |_ **x** _i, pθ⋆_ ), we can use Hoeffding’s inequality and bound that, with 1 _−_ 2<sup>_<u>δ</u>_probability,</sup> 



Considering the difference between Σ _pθ⋆_ and Σ _p_ 0, we see that 



Therefore, 



Next we bridge _P_ ( _Y |_ **x** test _, θ_<sup>_⋆_</sup> ) with _P_ ( _Y |_ **o** _ICL, p_ pre-train). Let _s_ test be the hidden state corresponding to first token of **x** test, i.e., **x** test _,_ 0. We see that, the likelihood of _s_ test = _sθ⋆_ is lower bounded by: 







15 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

For another task _θ_<sup>_′_</sup> , _s_ test is unlikely to be _sθ′_ because: 



Therefore, the Bayesian inference over _s_ test, is: 



So that 



16 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

## **B Results on more tasks** 

Besides the case study on SST2 dataset in Section 5, in this section we also provide experiment results on other tasks. In specific, we experiment on Rotten Tomatoes<sup>1</sup> , Tweet Eval<sup>2</sup> ’s ( `hate` , `irony` and `offensive` subtasks) and MNLI<sup>3</sup> . The results are as follows. 

### **B.1 Rotten Tomatoes** 



Figure 6: Averaged attention map over Rotten Tomatoes test set. 

> 1 `https://huggingface.co/datasets/rotten_tomatoes/` 

> 2 `https://huggingface.co/datasets/tweet_eval/` 

> 3 `https://huggingface.co/datasets/glue/viewer/mnli_matched/test` 

17 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 





Figure 7: Interpreting attention values from kernerl regression perspective on Rotten Tomatoes dataset. 





Figure 8: Investigating information in key and value vectors on Rotten Tomatoes dataset. 

18 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

### **B.2 Tweet Eval (Hate)** 



Figure 9: Averaged attention map over Tweet Eval (Hate) test set. 

19 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 





(a) Accuracy on reconstruction of _y_ ˆ by interpreting attention as kernel weights. 

Figure 10: Interpreting attention values from kernerl regression perspective on Tweet Eval (Hate) dataset. 





Figure 11: Investigating information in key and value vectors on Tweet Eval (Hate) dataset. 

20 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

### **B.3 Tweet Eval (Irony)** 



Figure 12: Averaged attention map over Tweet Eval (Irony) test set. 

21 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 





Figure 13: Interpreting attention values from kernerl regression perspective on Tweet Eval (Irony) dataset. 





Figure 14: Investigating information in key and value vectors on Tweet Eval (Irony) dataset. 

22 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

### **B.4 Tweet Eval (Offensive)** 



Figure 15: Averaged attention map over Tweet Eval (Offensive) test set. 

23 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 





(a) Accuracy on reconstruction of _y_ ˆ by interpreting attention as kernel weights. 

Figure 16: Interpreting attention values from kernerl regression perspective on Tweet Eval (Offensive) dataset. 





Figure 17: Investigating information in key and value vectors on Tweet Eval (Offensive) dataset. 

24 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

### **B.5 MNLI** 



Figure 18: Averaged attention map over MNLI test set. 

25 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 





Figure 19: Interpreting attention values from kernerl regression perspective on MNLI dataset. 





Figure 20: Investigating information in key and value vectors on MNLI dataset. 

26 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

## **C Synthetic Verification** 

To verify the theoretical results, we conduct experiments on a synthetic HMM. We experimented on randomly parameterized synthetic HMMs with 8 tasks, 80 states, and 100 observations. We vary the number of samples and evaluate the proposed kernel regression on fitting the Bayesian posterior, which is the intuition in Theorem 1. Results are listed in the Table 3. We indeed see a decreasing loss and increasing accuracy with more demonstrative examples. The loss also converges to a non-zero value according to Equation 7. 

Table 3: Accuracy and distance of predicting the Bayesian posterior on a synthetic HMM with Equation 6. 

|#samples|1|2|4|8|16|32|64|128|
|---|---|---|---|---|---|---|---|---|
|distance|1.322|0.942|0.519|0.218|0.163|0.085|0.093|0.083|
|accuracy|0.337|0.532|0.745|0.901|0.926|0.968|0.961|0.964|



27 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 

## **D Results on Llama-2 and Llama-3** 

To demonstrate the generality of our conclusions across model architectures, we also conduct analysis on Llama-2 model (Touvron et al., 2023) and Llama-3 8B model (Grattafiori et al., 2024). The experiment setting is similar to Section 5.1 and 5.2. Visualizations are as follows. They generally show similar trends with Section 5. 



Figure 21: Averaged attention map over SST2 dataset on Llama-2 model. 

28 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 





Figure 22: Interpreting attention values from kernerl regression perspective on Llama-2 model. 



Figure 23: Averaged attention map over SST2 dataset on Llama-3 8B model. 

## **E Single Datum Visualization** 

We also plot the visualization on a single datum as follows. It shows a similar but slightly sparser pattern to Section 5.1. 

29 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 



<!-- Start of picture text -->
1.0 0.70<br>0 0<br>0.9 0.65<br>2 2<br>0.60<br>4 0.8 4<br>0.55<br>6 0.7 6<br>8 8 0.50<br>0.6<br>10 10 0.45<br>0.5<br>12 12 0.40<br>14 0.4 14 0.35<br>0 5 10 15 20 25 0.3 0 5 10 15 20 25 0.30<br><!-- End of picture text -->



<!-- Start of picture text -->
(a) Accuracy on reconstruction of y ˆ by interpreting at-<br>tention as kernel weights.<br><!-- End of picture text -->

(b) Accuracy of predicting the ground-truth label by interpreting attention as kernel weights. 

Figure 24: Interpreting attention values from kernerl regression perspective on Llama-3 8B model. 

30 

Published in Transactions on Machine Learning Research <u>(09/2025)</u> 



Figure 25: The attention map on a single datum. 

31 

