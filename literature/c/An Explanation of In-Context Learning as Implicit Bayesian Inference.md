# An Explanation of In-context Learning as Implicit Bayesian Inference 

Sang Michael Xie Stanford University xie@cs.stanford.edu 

Aditi Raghunathan Stanford University aditir@stanford.edu 

Percy Liang Stanford University pliang@cs.stanford.edu 

Tengyu Ma Stanford University tengyuma@cs.stanford.edu 

##### **Abstract** 

Large language models (LMs) such as GPT-3 have the surprising ability to do in-context learning, where the model learns to do a downstream task simply by conditioning on a prompt consisting of input-output examples. The LM learns from these examples _without being explicitly pretrained to learn_ . Thus, it is unclear what enables in-context learning. In this paper, we study how in-context learning can emerge when pretraining documents have long-range coherence. Here, the LM must infer a latent document-level concept to generate coherent next tokens during pretraining. At test time, in-context learning occurs when the LM also infers a shared latent concept between examples in a prompt. We prove when this occurs despite a distribution mismatch between prompts and pretraining data in a setting where the pretraining distribution is a mixture of HMMs. In contrast to messy large-scale datasets used to train LMs capable of in-context learning, we generate a small-scale synthetic dataset (GINC) where Transformers and LSTMs both exhibit in-context learning<sup>1</sup> . Beyond the theory, experiments on GINC exhibit large-scale real-world phenomena including improved in-context performance with model scaling (despite the same pretraining loss), sensitivity to example order, and instances where zero-shot is better than few-shot in-context learning. 

## **1 Introduction** 

Large language models (LMs) such as GPT-3 (Brown et al., 2020, Lieber et al., 2021, Radford et al., 2019, Wang and Komatsuzaki, 2021) are pretrained on massive text corpora to predict the next word given previous words. They demonstrate the surprising ability to do _in-context learning_ , where an LM “learns” to do a task simply by conditioning on a prompt containing input-output pairs, achieving SOTA results on LAMBADA (Paperno et al., 2016) and TriviaQA (Joshi et al., 2017) tasks (18% and 3% over previous SOTA (Brown et al., 2020)). For example, consider the task of predicting nationalities from names. A prompt (Figure 1) is constructed by concatenating independent “training” examples (e.g., “Albert Einstein was German”) followed by a “test example” (“Marie Curie was”). Conditioning on this prompt, GPT-3 places the largest probability on the correct output 

_p_ (“Polish” _|_ “Albert Einstein was German _\_ n Mahatma Gandhi was Indian _\_ n Marie Curie was”) 

> 1 The code, data, and experiments are located on GitHub and CodaLab. 

1 



Figure 1: In-context learning can emerge from modeling long-range coherence in the pretraining data. During pretraining, the language model (LM) implicitly learns to infer a latent concept (e.g., wiki bios, which typically transition between name (Albert Einstein) _→_ nationality (German) _→_ occupation (physicist) _→_ ...) shared across sentences in a document. Although prompts are unnatural sequences that concatenate independent examples, in-context learning occurs if the LM can still infer the shared concept across examples to do the task (name _→_ nationality, which is part of wiki bios). 

by inferring the task from examples. Intruigingly, GPT-3 was not explicitly pretrained to learn from examples, and the distribution of prompts (which concatenate independent examples) is quite different from natural language. Our understanding of in-context learning is limited since (i) real pretraining data is messy and (ii) in-context learning has so far required large-scale datasets and models. 

In this paper, we introduce a simple pretraining distribution where in-context learning emerges. To generate a document, we first draw a latent concept _θ_ , which parameterizes the transitions of a Hidden Markov Model (HMM) (Baum and Petrie, 1966), then sample a sequence of tokens from the HMM (Figure 9). This latent variable structure is common in topic models such as LDA (Blei et al., 2003, Gruber et al., 2007). During pretraining, the LM must infer the latent concept across multiple sentences to generate coherent continuations. When conditioning on a prompt, in-context learning occurs when the LM also infers a shared _prompt concept_ across examples to make a prediction. We assume the LM fits the pretraining distribution _p_ exactly with enough data and expressivity, so that the question of in-context learning becomes characterizing the conditional distribution of completions given prompts _p_ (output _|_ prompt) under the pretraining distribution, where the prompt is generated from a different distribution _p_ prompt. This conditional distribution, which is the _posterior predictive distribution_ , marginalizes out the latent concepts: 



2 

If _p_ (concept _|_ prompt) concentrates on the prompt concept with more examples, then the LM learns via marginalization by “selecting” the prompt concept. Thus, in-context learning can be viewed as the LM implicitly performing Bayesian inference. 

The main challenge is that prompts are sampled from a different distribution than the pretraining distribution. The canonical Bayesian asymptotic tool is the Bernstein-von Mises theorem (Gunst and Shcherbakova, 2008, Kleijn and van der Vaart, 2012, van der Vaart, 1998), which asserts (under regularity conditions) that the posterior distribution of a latent variable concentrates on the maximum likelihood estimate. However, Bernstein-von Mises typically assumes observations are independent and/or drawn from the same distribution as the model, both of which are not satisfied. We prove that despite the distribution mismatch, the asymptotic prediction error of in-context learning is optimal when the signal about the latent concept in each prompt example is larger than the error due to the distribution mismatch. Additionally, we prove that the in-context learning error decreases with the length of each example—thus, information in the inputs, not just the input-output mapping, can be useful for in-context learning. 

As a companion to this theory, we created the **G** enerative **IN** - **C** ontext learning dataset (GINC), which is a small-scale synthetic dataset for studying in-context learning. We find that both Transformers (Vaswani et al., 2017) and LSTMs (Hochreiter and Schmidhuber, 1997) trained on GINC exhibit in-context learning. We verify intuitions from the theory, showing that the accuracy of incontext learning improves with the number of examples and example length. Ablations of the GINC dataset show that the latent concept structure in the pretraining distribution is crucial to the emergence of in-context learning. 

The experiments also bring up open questions which go beyond our theory, which only studies the pretraining distribution. We find that scaling up the number of model parameters steadily improves the in-context accuracy despite achieving the same pretraining loss, showing that larger models may improve in-context learning beyond increasing the capacity for memorizing the training data better. Previously observed in-context learning phenomena such as sensitivity to example ordering (Zhao et al., 2021) and the existence of settings where zero-shot is better than one/fewshot learning (Brown et al., 2020) are also mirrored in GINC. 

## **2 In-context learning setting** 

**Pretraining distribution.** In our framework, a latent concept _θ_ from a family of concepts Θ defines a distribution over observed tokens _o_ from a vocabulary _O_ . To generate a document, we first sample a concept from a prior _p_ ( _θ_ ) and then sample the document given the concept. Each pretraining document is a length _T_ sequence: 



We assume _p_ ( _o_ 1 _, . . . , oT |θ_ ) is defined by a Hidden Markov Model (HMM). The concept _θ_ determines the transition probability matrix of the HMM hidden states _h_ 1 _, . . . , hT_ from a hidden state set _H_ . 

**Prompt distribution.** The prompt distribution _p_ prompt generates prompts for in-context learning. The prompt is a concatenation of _n_ independent training examples and 1 test input _x_ test, which are all conditioned on a shared prompt concept _θ_<sup>_∗_</sup> . The goal is to predict the test output _y_ test by predicting the next token. 

A prompt example is composed of an input token sequence _x_ (e.g., Albert Einstein was) followed by an output token _y_ (e.g., German). In particular, the _i_ -th training example _Oi_ consists of an input 

3 

_xi_ = _Oi_ [1: _k −_ 1] (the first _k −_ 1 tokens) followed by an output token _yi_ = _Oi_ [ _k_ ] at the end<sup>2</sup> . The _i_ -th training example is independently generated as follows: 

1. Generate a start hidden state _h_<sup>start</sup> _i_ from a _prompt start distribution p_ prompt. 

2. Given _h_<sup>start</sup> _i_ , generate the example sequence _Oi_ = [ _xi, yi_ ] from _p_ ( _Oi|h_<sup>start</sup> _i , θ_<sup>_∗_</sup> ), the _pretraining distribution_ conditioned on a prompt concept _θ_<sup>_∗_</sup> . 

The test input _x_ test = _xn_ +1 is sampled similarly. Between each example, there is a special delimiter token _o_<sup>delim</sup> . The prompt consists of a sequence of training examples ( _Sn_ ) followed by the test example _x_ test: 



**Mismatch between prompt and pretraining distributions.** Since transitions between independent examples can be unnatural, the prompts are low probability sequences under the pretraining distribution. We provide a simple illustration using the names to nationalities example. Suppose that wiki bio documents in the pretraining data typically transition between name _→_ nationality _→_ occupation _→ . . ._ . In the prompt, the examples transition between name _→_ nationality _→_ name _→_ nationality _→ . . ._ , which contains low-probability transitions such as “German” _→_ “Mahatma Gandhi”. The prompt formatting (e.g., choice of delimiter) can also be a source of mismatch. We aim to show that despite this mismatch, large LMs can infer the prompt concept from examples. 

**In-context predictor and task.** For in-context learning, the output target _y_ for each example _x_ is sampled according to _p_ prompt( _y|x_ ): 



where _h_<sup>start</sup> test<sup>denotes the hidden state corresponding to the first token of</sup><sup>_x_test.</sup> We analyze the in-context predictor _fn_ ( _x_ test) = arg max _y p_ ( _y|Sn, x_ test), which outputs the most likely prediction over the _pretraining_ distribution conditioned on the prompt from the _prompt_ distribution<sup>3</sup> . We study the in-context predictor and its expected 0-1 error with _n_ examples _L_ 0-1( _fn_ ) = E _x_ test _,y_ test _∼p_ prompt[ **1** [ _fn_ ( _x_ test) = _y_ test]]. 

### **2.1 Assumptions** 

We detail the assumptions in our framework, including the structure of delimiters and regularity assumptions. We first assume that there exists a subset of _delimiter hidden states D_ which generates the special delimiter token _o_<sup>delim</sup> deterministically. 

**Assumption 1** (Delimiter hidden states) **.** _Let the delimiter hidden states D be a subset of H. For any h_<sup>_delim_</sup> _∈D and θ ∈_ Θ _, p_ ( _o_<sup>_delim_</sup> _|h_<sup>_delim_</sup> _, θ_ ) = 1 _and for any h ∈D/ , p_ ( _o_<sup>_delim_</sup> _|h, θ_ ) = 0 _._ 

Thus, observing the delimiter _o_<sup>delim</sup> reveals that the corresponding hidden state is in _D_ , but does not reveal which element of _D_ it is. The delimiter is usually a token that can appear in a broad range of contexts (e.g., newline). The delimiter ideally does not distract from the examples — for example, an adversarial delimiter could look like part of the input _x_ . To mitigate these scenarios, we assume that no delimiter (e.g., newline) is significantly more likely under one concept rather than another. 

> 2The example length _k_ is fixed for simplicity — we leave extending our analysis to variable _k_ as future work. 

> 3In practice, greedy decoding or nucleus sampling (Holtzman et al., 2020) are used for likely completions. 

4 

**Assumption 2** (Bound on delimiter transitions) **.** _For any delimiter state h_<sup>_delim_</sup> _∈D and any hidden state h ∈H, the probability of transitioning to a delimiter hidden state under θ is upper bounded p_ ( _h_<sup>_delim_</sup> _|h, θ_ ) _< c_ 2 _for any θ ∈_ Θ _\ {θ_<sup>_∗_</sup> _}, and is lower bounded p_ ( _h_<sup>_delim_</sup> _|h, θ_<sup>_∗_</sup> ) _> c_ 1 _>_ 0 _for θ_<sup>_∗_</sup> _. Additionally, the start hidden state distribution for delimiter hidden states is bounded as p_ ( _h_<sup>_delim_</sup> _|θ_ ) _∈_ [ _c_ 3 _, c_ 4] _._ 

The choice of prompt start distribution can be a source of distribution shift which is separate from the distribution shift from concatenating independent examples. We make an assumption that limits how much distribution shift is introduced by the prompt start distribution. 

**Assumption 3** (Distribution shift from prompt start distribution) **.** _We assume that the prompt start distribution pprompt is close in TV distance to all hidden transition distributions (under θ_<sup>_∗_</sup> _) starting from a delimiter hidden state:_ max _hdelim∈D TV_ ( _pprompt_ ( _h_ ) _∥p_ ( _h|h_<sup>_delim_</sup> _, θ_<sup>_∗_</sup> )) _<_ ∆ _/_ 4 _. Here,_ ∆= _pprompt_ ( _ymax|xtest_ ) _−_ max _y_ = _ymax pprompt_ ( _y|xtest_ ) _is the margin between the most likely label ymax_ = arg max _y pprompt_ ( _y|xtest_ ) _and the second most likely label._ 

Note that even when the maximum TV distance is 0, there is still distribution shift from concatenating independent examples. 

We also assume the prompt concept _θ_<sup>_∗_</sup> is in the family Θ, which is a broad set of concepts. 

**Assumption 4** (Well-specification) **.** _The prompt concept θ_<sup>_∗_</sup> _is in_ Θ _._ 

Even though the pretraining distribution is broad, the prompt is still low probability under the pretraining distribution since it concatenates independent examples. 

Finally, if the prompt has zero probability under the prompt concept _θ_<sup>_∗_</sup> , then Bayesian inference will not be able to infer the prompt concept as in Section 3.1. The following are regularity assumptions which mainly ensure that the prompt is not zero probability under _θ_<sup>_∗_</sup> . 

**Assumption 5** (Regularity) **.** _The pretraining distribution p satisfies: 1) Lower bound on transition probability for the prompt concept θ_<sup>_∗_</sup> _: for any pair of hidden states h, h_<sup>_′_</sup> _∈H, p_ ( _h|h_<sup>_′_</sup> _, θ_<sup>_∗_</sup> ) _> c_ 5 _>_ 0 _. 2) Start hidden state is lower bounded: for any h ∈H, p_ ( _h|θ_<sup>_∗_</sup> ) _≥ c_ 8 _>_ 0 _. 3) All tokens can be emitted: for every symbol o, there is some hidden state h ∈H such that p_ ( _o|h, θ_<sup>_∗_</sup> ) _> c_ 6 _>_ 0 _, 4) The prior p_ ( _θ_ ) _has support over the entire concept family_ Θ _and is bounded above everywhere._ 

## **3 Theoretical analysis** 

We prove that in the limit of infinite examples, the error of the in-context predictor is optimal if a _distinguishability_ condition holds — the prompt concept _θ_<sup>_∗_</sup> is distinct enough from the other concepts in Θ (e.g., when Θ is a discrete set). When distinguishability does not hold (e.g, Θ is continuousvalued), we show that the expected error still decreases with the length of each example, showing that information in both the inputs and the input-output mapping contribute to in-context learning. 

### **3.1 High-level approach** 

Our goal is to show that arg max _y p_ ( _y|Sn, x_ test) _→_ arg max _y p_ prompt( _y|x_ test) as the number of examples _n_ grows. In the following, assume that the prompt has non-zero probability under the pretraining distribution _p_ given _θ_<sup>_∗_</sup> , meaning that _p_ ( _Sn, x_ test _|θ_<sup>_∗_</sup> ) _>_ 0. We expand _p_ ( _y|Sn, x_ test) to analyze its 

5 

limit: 



where _rn_ ( _θ_ ) = _n_<sup><u>1</u>log</sup> _p_<sup>_<u>p</u>_</sup> (<sup><u>(</u></sup> _S_<sup>_S_</sup> _n_<sup>_n_</sup> _,x_<sup>_<u>,x</u>_</sup> test<sup>test</sup> _|_<sup>_<u>|</u>_</sup> _θ_<sup>_θ∗_</sup><sup><u>)</u></sup> )<sup>.In Theorem 1, we prove that under a distinguishability condition,</sup> exp( _n · rn_ ( _θ_ )) _→_ 0 for all concepts _θ_ except the prompt concept _θ_<sup>_∗_</sup> , where exp( _n · rn_ ( _θ_<sup>_∗_</sup> )) = 1. The only nonzero term in the integral is when _θ_ = _θ_<sup>_∗_</sup> , and thus the prompt concept is “selected” as a consequence of Bayesian inference<sup>4</sup> . Lemma 1 shows that the argmax after restricting to _θ_<sup>_∗_</sup> is the same as the most likely label under _p_ prompt( _y|x_ test) (using Assumption 3). Putting these together with Equation 6, the in-context predictor infers the prompt concept _θ_<sup>_∗_</sup> : 



Thus, the in-context predictor is optimal as the number of in-context examples increases. 

### **3.2 Heuristic derivation** 

Recall from Section 3.1 that if exp( _n · rn_ ( _θ_ )) _→_ 0 for all _θ_ = _θ_<sup>_∗_</sup> , then Bayesian inference “selects” the prompt concept through marginalization. To do this, we focus on showing that _rn_ ( _θ_ ), the average log-likelihood ratio between _θ_ and _θ_<sup>_∗_</sup> , converges to a negative constant, and thus _nrn_ goes to _−∞_ . 

The main technical challenge is to handle the sequence-of-examples structure of the prompt, which makes all the examples dependent with respect to the pretraining distribution. Our approach uses properties of delimiter tokens to approximately factorize the examples, with constant error per example. We let _Oi_<sup>ex= [</sup><sup>_o_delim</sup> _i−_ 1<sup>_, Oi_] be the</sup><sup>_i_-th input-output pair and the previous delimiter together</sup> for _i >_ 1 and define _O_ 1<sup>ex=</sup><sup>_O_1.Expanding the likelihood term inside</sup><sup>_rn_(</sup><sup>_θ_), our goal is to show</sup> 



To show this, we expand _p_ ( _Sn|θ_ ) with the chain rule, and with Assumption 5 (to bound _p_ ( _x_ test _|Sn, θ_ ) by _O_ (1)) it can be shown that 



We then marginalize _p_ ( _Oi_<sup>ex</sup><sup>_|O_</sup> 1:<sup>ex</sup> _i−_ 1<sup>_, θ_) over the hidden state</sup><sup>_h_delim</sup> _i−_ 1 corresponding to the delimiter in _Oi_<sup>ex= [</sup><sup>_o_delim</sup> _i−_ 1<sup>_, Oi_]:</sup> 



4We can exchange limits and integrals since the probabilities are bounded (dominated convergence). 

6 

While summing over _H_ above would be a trivial equality, we can replace _H_ with the set of delimiter hidden states _D_ since _p_ ( _h|O_ 1:<sup>ex</sup> _i−_ 1<sup>_, θ_)=0 for non-delimiter hidden states</sup><sup>_h∈D/_(Assumption 1).We</sup> used in the first equality that _O_ 1:<sup>ex</sup> _i−_ 1<sup>_→h_delim</sup> _i−_ 1 _→ Oi_<sup>exforms a Markov chain and</sup><sup>_p_(</sup><sup>_o_delim</sup> _i−_ 1<sup>_|h_delim</sup> _i−_ 1<sup>)=1</sup> (Assumption 1) to change _Oi_<sup>ex</sup> to _Oi_ . Finally, we can show using properties of delimiter hidden states (Assumption 2) that _p_ ( _h_<sup>delim</sup> _i−_ 1<sup>_|O_</sup> 1:<sup>ex</sup> _i−_ 1<sup>_, θ_)=</sup><sup>_O_(1) and �</sup> _h_<sup>delim</sup> _i−_ 1<sup>_∈D p_(</sup><sup>_Oi|h_</sup> _i_<sup>delim</sup> _−_ 1<sup>_, θ_)</sup><sup>_≈O_(1)</sup><sup>_p_(</sup><sup>_Oi|θ_) in</sup> the second step. Therefore, we can upper bound _rn_ ( _θ_ ) as 



The expectation term can be written as the difference of two KL divergences, _KL_ ( _p_ prompt( _O_ ) _∥p_ ( _O|θ_<sup>_∗_</sup> )) _− KL_ ( _p_ prompt( _O_ ) _∥p_ ( _O|θ_ )). We bound the first KL term by a constant using Assumption 5 — intuitively for one example, _p_ prompt and _p_ ( _·|θ_<sup>_∗_</sup> ) are close. We break the second term into a sum of negative KL divergences over _k_ tokens. There are _O_ ( _k_ ) KL terms and only _O_ (1) other error terms, which come from the distribution mismatch between the prompt and pretraining distributions. If the KL terms are larger than the error terms, then _rn_ ( _θ_ ) has a negative limit. If this holds for all _θ_ = _θ_<sup>_∗_</sup> , then we have exp( _n · rn_ ( _θ_ )) _→_ 0 for all _θ_ = _θ_<sup>_∗_</sup> , enabling in-context learning. 

### **3.3 Formal results** 

#### **3.3.1 In-context learning under distinguishability** 

We define a distinguishability condition which formalizes when in-context learning occurs. Letting _p_<sup>_j_</sup> _θ_<sup>(</sup><sup>_o_) :=</sup><sup>_p_(</sup><sup>_O_[</sup><sup>_j_] =</sup><sup>_o|O_[1 :</sup><sup>_j −_1]</sup><sup>_, θ_) be the output distribution of the</sup><sup>_j_-th token given the previous to-</sup> kens and _p_<sup>_j_</sup> prompt<sup>(</sup><sup>_o_) :=</sup><sup>_p_prompt(</sup><sup>_O_[</sup><sup>_j_] =</sup><sup>_o|O_[1 :</sup><sup>_j −_1]) be the analogous distribution under the prompt</sup> distribution, the distinguishability condition depends on the KL divergence between _p_<sup>_j_</sup> prompt<sup>(which</sup> represents _θ_<sup>_∗_</sup> ) and _p_<sup>_j_</sup> _θ_<sup>as well as error terms</sup><sup>_ϵ_</sup> start<sup>_θ_and</sup><sup>_ϵθ_</sup> delim<sup>coming from the distribution mismatch</sup> between the prompt and pretraining distributions at the start and delimiter token for each example: 



**Condition 1** (Distinguishability) **.** _We define θ_<sup>_∗_</sup> _to be distinguishable if for all θ ∈_ Θ _, θ_ = _θ_<sup>_∗_</sup> _,_ 



When the signal from KL divergence (LHS) is larger than the error terms, Equation 14 is satisfied (Figure 2). For larger example lengths _k_ , the LHS increases, improving distinguishability. Intuitively, larger example lengths increase the proportion of the prompt sampled from the pretraining distribution by providing more evidence for Bayesian inference. Under Condition 1, the in-context predictor asymptotically achieves the optimal expected error. 

**Theorem 1.** _Assume the assumptions in Section 2.1 hold. If Condition 1 holds, then as n →∞ the prediction according to the pretraining distribution is_ 



_Thus, the in-context predictor fn achieves the optimal 0-1 risk:_ lim _n→∞ L0-1_ ( _fn_ ) = inf _f L0-1_ ( _f_ ) _._ 

7 



Figure 2: When the signal about the prompt concept within each example (green) is greater than the error from low-probability transitions between examples, in-context learning succeeds in our latent concept setting (Theorem 1). Increasing the example length _k_ increases the signal. The signal for in-context learning comes from tokens in both the inputs and the input-output mapping. 

#### **3.3.2 Non-distinguishable case** 

The distinguishability condition (Condition 1) fails when there is some _θ_ = _θ_<sup>_∗_</sup> for which the KL divergence between _θ_ and _θ_<sup>_∗_</sup> is less than the error terms. However, this also means that the output distributions of _θ_ and _θ_<sup>_∗_</sup> are close in KL. We leverage this to prove that the expected 0-1 error decreases with the example length _k_ under two different settings where distinguishability does not hold. 

**Continuity.** Our first result relies on a continuity assumption between the concept parameter and its corresponding output distribution. Our assumption is based on prior works (Kleijn and van der Vaart, 2012), where the KL divergence is assumed to have a 2nd-order Taylor expansion. 

**Theorem 2.** _Let the set of θ which does not satisfy Equation 14 in Condition 1 to be B. Assume that KL divergences have a 2nd-order Taylor expansion around θ_<sup>_∗_</sup> _:_ 



_where Ij,θ∗ is the Fisher information matrix of the j-th token distribution with respect to θ_<sup>_∗_</sup> _. Let γθ∗_ = maxmin _jλj λminmax_ (( _IIj,θj,θ∗∗_ ))<sup>_where λmax, λminreturn the largest and smallest eigenvalues.Then for k≥_2</sup><sup>_and as n→∞,_</sup> _the 0-1 risk of the in-context learning predictor fn is bounded as_ 



_where g_ ( _δ_ ) = 2<sup><u>1</u>((1</sup><sup>_−δ_) log(1</sup><sup>_−δ_) + (1 +</sup><sup>_δ_) log(1 +</sup><sup>_δ_))</sup><sup>_is a calibration function (Steinwart, 2007, Ávila_</sup> _Pires and Szepesvári, 2016) for the multiclass logistic loss for δ ∈_ [0 _,_ 1) _, assuming that the minimizers of the 0-1 risk and multiclass logistic risk are the same._ 

Since the inverse calibration function _g_<sup>_−_1</sup> is roughly linear in _ϵ_ for _ϵ ≤_ 0 _._ 7, the excess risk roughly decreases as _O_ (1 _/k_ ). When the “worst-case condition number” _γθ∗_ of the Fisher information matrices is smaller (well-conditioned), the error decreases. Intuitively, this means that there is no direction to vary _θ_<sup>_∗_</sup> in which the output distribution will sharply change. As a consequence, the concepts _θ_ that are not distinguishable from the prompt concept _θ_<sup>_∗_</sup> parameterize distributions that produce similar outputs to the prompt concept and thus achieve a small error. 

8 



<!-- Start of picture text -->
90 100<br>k=3 k=3<br>80 k=5 k=5<br>k=8 80 k=8<br>70<br>k=10 k=10<br>60<br>60<br>50<br>40 40<br>0 20 40 60 0 20 40 60<br>Num examples Num examples<br>Acc Acc<br><!-- End of picture text -->

Figure 3: In-context accuracy (95% intervals) of Transformers (left) and LSTMs (right) on the GINC dataset. Accuracy increases with number of examples _n_ and length of each example _k_ . 



<!-- Start of picture text -->
26 3<br>k=3 k=3 13 k=3<br>k=5 k=5 k=5<br>24 k=8 2 k=8 12 k=8<br>k=10 k=10 k=10<br>11<br>22<br>1 10<br>20 9<br>0 20 40 60 0 20 40 60 0 20 40 60<br>Num examples Num examples Num examples<br>Acc Acc Acc<br><!-- End of picture text -->

Figure 4: Ablation studies for 4 layer Transformers on the GINC dataset with vocab size 50. **(Left)** When pretrained with only one concept, in-context learning fails. **(Middle)** When the pretraining data has random transitions, the model sees all token transitions but in-context learning fails. **(Right)** When prompts are from random unseen concepts, in-context learning fails to extrapolate. 

**Varying-length test examples.** In the setting where the length of _x_ test is random (uniformly from 2 to _k_ ), we can give a similar error guarantee without continuity. 

**Theorem 3.** _Let the set of θ which does not satisfy Equation 14 in Condition 1 to be B. Let the length of the test example xtest be uniformly distributed between 2 and k, for k ≥_ 2 _. Then for k ≥_ 2 _and as n →∞, the 0-1 risk of the in-context learning predictor fn is bounded as_ 



_assuming that the minimizers of the 0-1 risk and multiclass logistic risk are the same._ 

Instead of measuring only the error at the _k_ -th token, we average the prediction error on the 2nd to _k_ -th tokens. However, we leave bridging the mismatch between training examples, which are consistently length _k_ , and test examples, which have random length, to future work. 

## **4 Simulations** 

We generate the GINC dataset and show that Transformers (Vaswani et al., 2017) and LSTMs (Hochreiter and Schmidhuber, 1997) trained on GINC exhibit in-context learning. In the theory, we assumed that the pretrained LM fits the pretraining distribution exactly. Here, we pretrain LMs to approximate the pretraining distribution, showing that the in-context learning properties of the pretraining distribution transfer to the LM. 

9 

**GINC dataset.** We construct the GINC dataset according to our theory (see Appendix F.1). For pretraining, we define a uniform mixture of HMMs over a family Θ of 5 concepts to generate 1000 pretraining documents with _∼_ 10 million tokens total. For prompting, we generate prompts with 0 to 64 training examples and example lengths _k ∈{_ 3 _,_ 5 _,_ 8 _,_ 10 _}_ (2500 prompts for each setting). The target token _y_ test is taken to be the most likely output arg max _y p_ prompt( _y|x_ test) instead of sampling so that the intrinsic error is 0. 

**Main result.** We train GPT-2-based Transformers (Radford et al., 2019) and LSTMs on three versions of the GINC dataset with vocabulary sizes 50, 100, and 150, then evaluate the in-context accuracy (see Appendix F.2, F.3). We average all results over 5 pretraining runs. Figure 3 shows that for both Transformer and LSTMs, in-context accuracy improves as the number of prompt examples _n_ and the example length _k_ increase, verifying our theory. 

**Ablations on the latent concept structure.** We ablate the role of the mixture-of-concepts structure in GINC. In Figure 4 (left), we pretrain a 4 layer Transformer on data with only one concept (removing the prior) from Θ, resulting in flat in-context learning curves. Figure 4 (middle) shows that pretraining on random pretraining data, which contains all possible token transitions, in-context learning also fails. Therefore, the mixture-of-concepts structure is important and simply seeing diverse token transitions does not enable in-context learning. 

**Extrapolation to unseen concepts.** Full generative control of GINC allows for experimentation with latent variables in the pretraining distribution. For example, in large-scale datasets, it is difficult to test whether a concept or task is in the pretraining data. We test this in GINC by testing the in-context accuracy of a 4 layer Transformer on prompts generated from 5 random concepts that are not in the pretraining family of concepts. Figure 4 (right) shows that in-context learning also fails for these novel concepts. 

**Effect of model size and architecture.** Figure 5 shows that increasing the size of the Transformer (4, 12, 16 layers) steadily increases the in-context accuracy, corroborating the results of Brown et al. (2020). Table 6 shows that even though larger Transformers may have the same pretraining loss (e.g., 12 and 16 layer Transformers both get 1.33 validation loss for vocab size 50), the in-context accuracy still improves (81% to 85% from 12 to 16 layers), suggesting that larger models can improve in-context learning beyond improving pretraining perplexity. This may be related to phenomena from overparameterization and overtraining (Power et al., 2021, Zhang et al., 2017). Finally, the model architecture also plays a role — LSTMs consistently outperform Transformers on GINC despite having fewer parameters, perhaps due to the similarity between HMMs and LSTMs. We leave analysis of the effect of model scaling and model architecture as open questions. 

**Sensitivity to example ordering.** In Figure 7 (left), we test the sensitivity of in-context accuracy on GINC to the ordering of the prompt examples, following Zhao et al. (2021). For this experiment, we consider prompts generated from a single concept and prompt start distribution. We sample 10 different sets (leading to 10 training set IDs) of 4 examples and generate all 24 possible permutations for each example set. We consider the in-context accuracy of the 4 layer Transformer trained on GINC with vocabulary size 50. Similarly to the behavior of GPT-3 (Zhao et al., 2021), there is a significant variation (10–40% difference) between permutations of the same set of examples. 

**Zero-shot is sometimes better than few-shot.** In some settings in GINC, we find that zero-shot performance can be better than few-shot performance. This mirrors GPT-3 on some datasets (e.g., LAMBADA, HellaSwag, PhysicalQA, RACE-m, CoQA/SAT analogies for smaller models (Brown 

10 



<!-- Start of picture text -->
100<br>90<br>80<br>70<br>Vocab size=50<br>Vocab size=100<br>60<br>Vocab size=150<br>0.4 0.6 0.8 1.0<br>Num Parameters 1e8<br>Acc<br><!-- End of picture text -->

Figure 5: In-context accuracy (95% intervals) of Transformers improves as model size increases on the GINC dataset for vocabulary sizes 50, 100, and 150. 

|Model|# Params|Train loss<br>(pretraining)|Val loss<br>(pretraining)|In-context Acc|
|---|---|---|---|---|
|Vocab size 50,_k_ = 10_, n_= 64|||||
|Transformer (4 layer)|29M|1.49|1.50|60.2_±_5.7|
|Transformer (12 layer)|85M|1.31|1.33|81.2_±_7.1|
|Transformer (16 layer)|115M|1.31|1.33|84.7_±_3.4|
|LSTM|28M|1.31|1.35|95.8_±_1.11|
|Vocab size 100,_k_ = 10_, n_= 64|||||
|Transformer (4 layer)|29M|1.58|1.59|67.4_±_4.7|
|Transformer (12 layer)|85M|1.40|1.42|84.6_±_3.0|
|Transformer (16 layer)|115M|1.41|1.43|88.7_±_1.6|
|LSTM|28M|1.43|1.44|95.8_±_1.54|
|Vocab size 150,_k_ = 10_, n_= 64|||||
|Transformer (4 layer)|29M|1.44|1.45|92.8_±_1.9|
|Transformer (12 layer)|85M|1.27|1.28|98.4_±_0.4|
|Transformer (16 layer)|115M|1.27|1.28|98.1_±_0.5|
|LSTM|28M|1.26|1.31|99.2_±_1.06|



Figure 6: In-context accuracies (95% intervals) on GINC with vocab sizes (50, 100, 150) for Transformers and LSTMs. Accuracy improves with scale even though the pretraining loss may be the same. 



<!-- Start of picture text -->
50 k=3<br>80 k=5<br>k=8<br>70 40 k=10<br>60<br>30<br>50<br>40 20<br>30<br>0 1 2 3 4 5 6 7 8 9 0 20 40 60<br>Training set ID Num examples<br>Acc<br>Accuracy<br><!-- End of picture text -->

Figure 7: **(Left)** In-context accuracy varies widely with example ordering. Each training ID refers to a set of training examples. Each dot refers to the in-context learning accuracy of one permutation of the training examples for that particular training ID. **(Right)** Zero-shot performance can be higher than one/few-shot performance in some settings in GINC, mirroring the behavior of GPT3 on some datasets such as LAMBADA (Brown et al., 2020). The few-shot setting introduces the distracting prompt structure, which can initially lower accuracy. 

et al., 2020)). This occurs especially when the transition probabilities in GINC are lower entropy (controlled via a temperature parameter). For this experiment, we consider GINC with transition matrix temperature parameter 0.01 (instead of 0.1), 12 concepts, and vocabulary size 100. Figure 7 (right) shows that here, few-shot accuracy is initially worse than zero-shot accuracy, but can recover with more examples. We hypothesize that the distracting prompt structure initially decreases the accuracy in this setting. 

## **5 Discussion and related work** 

**Learning via Bayesian inference and extrapolation.** The canonical Bernstein-von Mises theorem (van der Vaart, 1998) does not apply for in-context learning since the prompt examples are not independent under the pretraining distribution. Gunst and Shcherbakova (2008) show a Bernsteinvon Mises-type result for observations from an HMM, but do not handle observations from a dif- 

11 

ferent distribution. Future directions include more precise asymptotic results about the posterior distribution and results under misspecification/extrapolation (Kleijn and van der Vaart, 2012). A possible avenue for extrapolation to some types of unseen concepts is to factorize the latent concept into semantics and syntax. While the pretraining data may contain only some semantics-syntax pairs, the language model could generalize to unseen pairs if it learns generalizable syntactical operations such as copying or reordering. 

**Topic models and HMMs.** Topic models such as LDA (Blei et al., 2003) also have document-level latent variables, but learning is typically relies on algorithms such as EM (Dempster et al., 1977), variational inference (Jordan et al., 1999), or MCMC (Hastings, 1970, Metropolis et al., 1953). We focus on learning as a natural result of Bayesian inference without an explicit inference algorithm. Wei et al. (2021a) also use an HMM model in their pretraining analysis. However, they analyze how pre-trained representations learned with masked LMs (Clark et al., 2020, Devlin et al., 2019, Lewis et al., 2020, Liu et al., 2019) can improve optimization-based downstream learning (Lester et al., 2021, Li and Liang, 2021) rather than in-context learning. 

**Bridging the mismatch between pretraining and prompting.** Prior works support our theoretical intuitions that reducing the prompt distribution mismatch would improve in-context learning. Finetuning LMs on text with a prompting format improves its zero-shot performance (Sanh et al., 2021, Wei et al., 2021b) and optimizing prompt templates improves few-shot finetuning (Gao et al., 2021, Jiang et al., 2020, Schick and Schütze, 2021, Shin et al., 2020). Holtzman et al. (2021), Zhao et al. (2021) improve in-context accuracy via calibration or renormalization, a form of adaptation to the prompt distribution. 

**Meta-learning.** Meta-learning methods can also train a sequence model to learn from examples (Ravi and Larochelle, 2017). However, meta-learning models are trained to learn, while incontext learning emerges from LM pretraining. 

**Studying large-scale phenomena at a small scale.** We can study in-context learning, a large scale phenomenon, at a small scale in GINC because the complexity of the pretraining distribution (HMM hidden state size, number of latent concepts) is small, such that the data and models are relatively larger. Since GINC is synthetic, we can also control the latent data properties (e.g., unseen concepts) to make predictions about large LMs while working at a small scale. 

## **6 Conclusion** 

We cast in-context learning as implicit Bayesian inference, where the pretrained LM implicitly infers a concept when making a prediction. We show that in-context learning occurs when the pretraining distribution is a mixture of HMMs. Our work provides a first step towards understanding in-context learning, which we hope will provide insight for improving pretraining and prompting. 

## **Acknowledgements** 

We thank Tianyi Zhang, Frieda Rong, Lisa Li, Colin Wei, Shibani Santurkar, Tri Dao, Ananya Kumar, and Shivam Garg for helpful discussions and feedback. SMX is supported by an NDSEG Fellowship. The work is partially supported by an Open Philanthropy Project Award, SDSI, and SAIL at Stanford University. TM acknowledges support of Google Faculty Award, NSF IIS 2045685, the Sloan Fellowship, and JD.com. Toyota Research Institute provided funds to support this work. 

12 

## **References** 

- Leonard E Baum and Ted Petrie. Statistical inference for probabilistic functions of finite state markov chains. _The annals of mathematical statistics_ , 37(6):1554–1563, 1966. 

- D. Blei, Andrew Ng, and M. I. Jordan. Latent Dirichlet allocation. _Journal of Machine Learning Research (JMLR)_ , 3:993–1022, 2003. 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. _arXiv preprint arXiv:2005.14165_ , 2020. 

- Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. Electra: Pre-training text encoders as discriminators rather than generators. In _International Conference on Learning Representations (ICLR)_ , 2020. 

- A. P. Dempster, Laird N. M., and Rubin D. B. Maximum likelihood from incomplete data via the EM algorithm. _Journal of the Royal Statistical Society: Series B_ , 39(1):1–38, 1977. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In _Association for Computational Linguistics (ACL)_ , pages 4171–4186, 2019. 

- Tianyu Gao, Adam Fisch, and Danqi Chen. Making pre-trained language models better few-shot learners. _arXiv_ , 2021. 

- Zoubin Ghahramani and Michael Jordan. Factorial hidden Markov models. _Machine Learning_ , 29: 245–273, 1997. 

- Amit Gruber, Yair Weiss, and Michal Rosen-Zvi. Hidden topic Markov models. In _Artificial Intelligence and Statistics (AISTATS)_ , 2007. 

- M. Gunst and O. Shcherbakova. Asymptotic behavior of Bayes estimators for hidden Markov models with application to ion channels. _Mathematical Methods of Statistics_ , 17, 2008. 

- Keith W. Hastings. Monte Carlo sampling methods using Markov chains and their applications. _Biometrika_ , 57(1):97–109, 1970. 

- Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. _Neural Computation_ , 9(8): 1735–1780, 1997. 

- Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In _International Conference on Learning Representations (ICLR)_ , 2020. 

- Ari Holtzman, Peter West, Vered Shwartz, Yejin Choi, and Luke Zettlemoyer. Surface form competition: Why the highest probability answer isn’t always right, 2021. 

- Zhengbao Jiang, Frank F Xu, Jun Araki, and Graham Neubig. How can we know what language models know? In _Association for Computational Linguistics (ACL)_ , 2020. 

13 

- Michael I. Jordan, Zoubin Ghahramani, Tommi S. Jaakkola, and Lawrence K. Saul. An introduction to variational methods for graphical models. _Machine Learning_ , 37:183–233, 1999. 

- Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In _Association for Computational Linguistics (ACL)_ , 2017. 

- Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _International Conference on Learning Representations (ICLR)_ , 2015. 

- B.J.K. Kleijn and A.W. van der Vaart. The Bernstein-von mises theorem under misspecification. _Electronic Journal of Statistics_ , 6, 2012. 

- Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. _arXiv preprint arXiv:2104.08691_ , 2021. 

- Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In _Association for Computational Linguistics (ACL)_ , 2020. 

- Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In _Association for Computational Linguistics (ACL)_ , 2021. 

- Opher Lieber, Or Sharir, Barak Lenz, and Yoav Shoham. Jurassic-1: Technical details and evaluation. Technical report, AI21 Labs, August 2021. 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A robustly optimized BERT pretraining approach. _arXiv preprint arXiv:1907.11692_ , 2019. 

- Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _International Conference on Learning Representations (ICLR)_ , 2019. 

- Nicholas Metropolis, Arianna W. Rosenbluth, Marshall N. Rosenbluth, Augusta H. Teller, and Edward Teller. Equation of state calculations by fast computing machines. _The journal of chemical physics_ , 21(6):1087–1092, 1953. 

- Denis Paperno, German Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The LAMBADA dataset: Word prediction requiring a broad discourse context. In _Association for Computational Linguistics (ACL)_ , 2016. 

- Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets. In _ICLR MATH AI Workshop_ , 2021. 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. _OpenAI Blog_ , 1(8), 2019. 

- Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In _International Conference on Learning Representations (ICLR)_ , 2017. 

14 

- Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Stella Biderman, Leo Gao, Tali Bers, Thomas Wolf, and Alexander M. Rush. Multitask prompted training enables zero-shot task generalization, 2021. 

- Timo Schick and Hinrich Schütze. Exploiting cloze questions for few shot text classification and natural language inference. In _European Association for Computational Linguistics (EACL)_ , 2021. 

- Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh. Eliciting knowledge from language models using automatically generated prompts. In _Empirical Methods in Natural Language Processing (EMNLP)_ , 2020. 

- Ingo Steinwart. How to compare different loss functions and their risks. _Constructive Approximation_ , 26, 2007. 

A. W. van der Vaart. _Asymptotic statistics_ . Cambridge University Press, 1998. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. _arXiv preprint arXiv:1706.03762_ , 2017. 

- Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax, May 2021. 

- Colin Wei, Sang Michael Xie, and Tengyu Ma. Why do pretrained language models help in downstream tasks? an analysis of head and prompt tuning. _arXiv_ , 2021a. 

- Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. _arXiv_ , 2021b. 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R’emi Louf, Morgan Funtowicz, and Jamie Brew. HuggingFace’s transformers: State-of-the-art natural language processing. _arXiv preprint arXiv:1910.03771_ , 2019. 

- Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In _International Conference on Learning Representations (ICLR)_ , 2017. 

- Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models. In _International Conference on Machine Learning (ICML)_ , 2021. 

Bernardo Ávila Pires and Csaba Szepesvári. Multiclass classification calibration functions. _arXiv_ , 2016. 

15 

## **A Framework details** 

**Prompt distribution details.** For in-context learning, we sample a prompt from a new distribution _p_ prompt, which consists of _n_ independent training examples and 1 test example. We first sample _n_ hidden segments _H_ of length _k_ by sampling the first element _h_<sup>start</sup> = _H_ [1] from a prompt start distribution _p_ prompt. Then, we sample the rest of the segment _H_<sup>seg</sup> = _H_ [2 : _k_ ] from the hidden transition distribution of the pretraining distribution _p_ corresponding to a particular concept _θ_<sup>_∗_</sup> : 





Conditioned on hidden variables _Hi_ and _h_<sup>delim</sup> _i_ , we sample the observed tokens _Oi_ = [ _oi,_ 1 _, . . . , oi,k_ ] and _o_<sup>delim</sup> _i_ respectively from the pre-training distribution: 



The “input” for each example is _xi_ = _Oi_ [1 : _k −_ 1] and the “output” is _yi_ = _Oi_ [ _k_ ]. Taking _S_ to be the sequence of training examples (without the test example), the resulting prompt sequence is [ _Sn, x_ test] = [ _O_ 1 _, o_<sup>delim</sup> 1 _, . . . , On, o_<sup>delim</sup> _n , x_ test] = [ _x_ 1 _, y_ 1 _, o_<sup>delim</sup> 1 _, x_ 2 _, y_ 2 _, o_<sup>delim</sup> 2 _, . . . , xn, yn, o_<sup>delim</sup> _n , x_ test] _∼ p_ prompt (24) 

where _x_ test = _xn_ +1 = _On_ +1[1 : _k −_ 1] is sampled via the same process but with _k −_ 1 elements. 

## **B Propositions for Theorem 1** 

The following propositions, which lower bound the probability of a delimiter token and probability of an example under _θ_<sup>_∗_</sup> , are direct corollaries of the assumptions. 

**Proposition 1.** _For all i, we have p_ ( _h_<sup>_delim_</sup> _i |O_ 1 _, o_<sup>_delim_</sup> 1 _, . . . , Oi, θ_<sup>_∗_</sup> ) _> c_ 1 _and p_ ( _h_<sup>_delim_</sup> _i |O_ 1 _, o_<sup>_delim_</sup> 1 _, . . . , Oi, θ_ ) _< c_ 2 _._ 

_Proof._ By Assumption 2, 



Similarly, 



16 

**Proposition 2.** _The probability of an example is lower bounded for θ_<sup>_∗_</sup> _: there is some c_ 7 _>_ 0 _such that p_ ( _Oi|h_<sup>_start_</sup> _i , hj,l, θ_<sup>_∗_</sup> ) _> c_ 7 _for all i and future hidden states hj,l, for any l and j > i._ 

_Proof._ By Assumption 5, we have 



for some _Hi_ . We have 



which lower bounds the terms in the numerator by _c_ 5 (marginalizing over previous hidden states), and upper bounding the denominator by 1. Setting _c_ 7 = ( _c_ 6)<sup>_k_</sup> _c_<sup>2</sup> 5<sup>finishes the proof.</sup> 

## **C Convergence of the in-context predictor** 

Under Assumption 3, we show that the in-context predictor _fn_ ( _x_ test) = arg max _y p_ ( _y|Sn, x_ test) converges when abstracting away the Bayesian inference component (the selection of _θ_<sup>_∗_</sup> from Θ) of the in-context predictor. We will complete the argument for the convergence of the in-context predictor 1. in the proof of Theorem 

**Lemma 1.** _Suppose the prompt Sn and the test input xtest are given. Under Assumption 3, we show that the argmax of the averaged predictive distribution conditioned on θ_<sup>_∗_</sup> _and a prompt Sn is the same as the argmax of the prompt predictive distribution:_ 



_Proof._ First, we note by definition that 



Expanding the last term, we have 



which is proportional to a constant in _x_ test. 

On the other hand, analyzing one term inside the LHS of the lemma statement, we have 



which is proportional to a constant in _x_ test and _Sn_ . The quantities differ in the last term, which we expand below and put in matrix form. Let _T ∈_ R<sup>_|H|×|D|_</sup> be the matrix that represents the transition probabilities starting from a delimiter state: _p_ ( _h_<sup>start</sup> test<sup>_|h_delim) for</sup><sup>_h_start</sup> test<sup>_∈H_and</sup><sup>_h_delim</sup><sup>_∈D_.As a result,</sup> 



where _h_<sup>delim</sup> _n_ is the delimiter hidden state before _h_<sup>start</sup> test<sup>.</sup> 

17 

Let _W ∈_ R<sup>_|Y|×|H|_</sup> be the matrix that represents the probabilities _p_ ( _y|x_ test _, h_<sup>start</sup> test<sup>_, θ∗_)</sup><sup>_p_(</sup><sup>_x_test</sup><sup>_|h_start</sup> test<sup>_, θ∗_) for</sup> all the possible _y ∈Y_ and _h_<sup>start</sup> test<sup>_∈H_.Overall, we can write</sup> 





where _u ∈_ R<sup>_|H|_</sup> is the vector of probabilities that corresponds to the prompt start distribution _p_ prompt. Bounding the difference between the two predictive distributions, 





Using Assumption 3, we can further bound this by ∆ _/_ 2: 



Since the probability of any output does not change by more than ∆ _/_ 2 and the margin between the most likely label and the second most likely label is ∆, the argmax’s are the same, showing the result. 

18 

## **D Proof of Theorem 1** 

_Proof._ We analyze the most likely prediction over the pretraining distribution conditioned on the prompt arg max _y p_ ( _y|Sn, x_ test). 



Defining the following quantity, 



we will show that under distinguishability for all _θ_ = _θ_<sup>_∗_</sup> , _rn_ ( _θ_ ) converges to a negative constant such that 



for _θ_ = _θ_<sup>_∗_</sup> , whereas this ratio is always 1 for _θ_ = _θ_<sup>_∗_</sup> . This will then “select” the desired prompt concept through marginalization. 

Supposing that Equation 53 holds, we show that the theorem statement holds. Let 



and let _ϵ <_ (∆ _/_ 2 _−_ ∆<sup>_′_</sup> ) _p_ ( _θ_<sup>_∗_</sup> ). Then for _n_ large enough (due to Equation 53), 



where _ϵθ_ ( _y_ ) _≤ ϵ/_ 2 for all _y ∈Y_ . 

By Lemma 1, the argmax of the first term of Equation 57 is the same as arg max _y p_ prompt( _y|x_ test), where the margin between the most likely label and the second most likely is at least ∆ _/_ 2 _−_ ∆<sup>_′_</sup> . Since 



for all _y ∈Y_ , the argmax of Equation 57 is also the same as arg max _p_ prompt( _y|x_ test). 

19 

Now it remains to show that _rn_ ( _θ_ ) converges to a negative constant for _θ_ = _θ_<sup>_∗_</sup> . Let _Oi_<sup>ex= [</sup><sup>_o_delim</sup> _i−_ 1<sup>_, Oi_]</sup> be the _i_ -th observation segment and the previous delimiter together for _i >_ 1 and define _O_ 1<sup>ex=</sup><sup>_O_1.</sup> Expanding the numerator of the ratio in _rn_ ( _θ_ ), we have 





Note that in the last line, the inner sum is over the set of delimiter states _D_ by using the assumption that observing a delimiter _o_<sup>delim</sup> implies that the corresponding hidden state _h_<sup>delim</sup> must be in _D_ . We also see that<sup>�</sup> _h_<sup>delim</sup> _n p_ ( _h_<sup>delim</sup> _n |O_ 1:<sup>ex</sup> _n_<sup>_, θ_) = 1.</sup> 

We restrict our attention to _θ_ where _p_ ( _Sn, x_ test _|θ_ ) _>_ 0, since otherwise _θ_ does not affect the prediction. Expanding _rn_ ( _θ_ ), we have the following upper bound: 



In the above steps, we used both Propositions 1 and 2 in the terms involving _c_ 2 _, c_ 1 (bounding the probability of _h_<sup>delim</sup> hidden states) and _c_ 7 (bounding the probability of _x_ test). Note that in the second line, the sum can must be over the set of delimiter states _D_ by using the assumption that observing a delimiter _o_<sup>delim</sup> implies that the corresponding hidden state _h_<sup>delim</sup> must be in _D_ . 

20 

Focusing on the numerator of the ratio term and summing over the start hidden state for the _i_ -th example, 



where the last step applies Bayes’ rule. We can lower and upper bound the following quantity for any _θ_ using Assumption 2: 



This implies that 



Plugging in these bounds, we have 



where we set 

Next, we convert the expectation in the bound into a KL divergence. We have 



We will upper bound the first KL term: 



21 

Expanding the numerator and denominator of the ratio inside, we have 



which differ in only the hidden start distribution. Using Assumption 5, we have that _p_ ( _h|θ_<sup>_∗_</sup> ) _≥ c_ 8 for any _h ∈H_ , which implies that 



Finally, this implies that the KL term is bounded as 



This term is non-negative since _c_ 8 _≤_ 1. 

Aiming to decompose the second KL term into a sum over the _k_ tokens, we write _p_<sup>_j_</sup> _θ_<sup>(</sup><sup>_o_) =</sup><sup>_p_(</sup><sup>_O_[</sup><sup>_j_] =</sup> _o|O_ [1 : _j −_ 1] _, θ_ ) and _p_<sup>_j_</sup> prompt<sup>(</sup><sup>_o_) =</sup><sup>_p_prompt(</sup><sup>_O_[</sup><sup>_j_] =</sup><sup>_o|O_[1 :</sup><sup>_j −_1]).We have</sup> 



Then we have that 



The second term (set _ϵ_<sup>_θ_</sup> start<sup>=log(</sup> _c_<sup><u>1</u></sup> 8<sup>))isanerrortermthatdependsonhowdifferentthestarting</sup> prompt distribution _p_ prompt (which is part of _p_ prompt) is to the pretraining distribution. The third term is an error term that comes from the delimiter transitions. The bound is negative when the sum of KL terms is larger in magnitude than the error terms. Note that as _k_ becomes larger, the number of observations of _θ_<sup>_∗_</sup> “overpowers” the distracting transitions in the prompt distribution. This condition is equivalent to the disinguishability condition (Condition 1). 

By assumption, for _θ_ = _θ_<sup>_∗_</sup> the Condition 1 holds, and thus 



since _rn_ ( _θ_ ) has a negative, constant limit. Note that exp( _n · rn_ ( _θ_<sup>_∗_</sup> )) = 1 for _θ_<sup>_∗_</sup> . 

22 

## **E Non-distinguishable case** 

When Condition 1 is unsatisfied, Equation 14), gives an upper bound on the sum of KL divergences for the next token distributions given different-length histories. In contrast, the in-context task only measures the accuracy of the last ( _k_ -th) token. The main challenge is to relate the different-length histories to each other to give a more precise bound for the error on the in-context task (last token). Before addressing this challenge, we give the following lemma, which leverages the result of Steinwart (2007), Ávila Pires and Szepesvári (2016) to relate a bound on the KL divergence to 0-1 loss. 

**Lemma 2.** _Let the set of θ which does not satisfy Condition 1 to be B. Assume that KL_ ( _pprompt_ ( _ytest|xtest_ ) _∥p_ ( _ytest|xtest, θ_ ) _is bounded above for all θ and that θ_<sup>_∗_</sup> _minimizes the multiclass logistic risk LCE_ ( _θ_ ) = _−_ E _xtest∼pprompt_ [ _pprompt_ ( _ytest|xtest_ ) log _p_ ( _ytest|xtest, θ_ )] _. If_ 



_then_ 



_where_ 



_is a calibration function for the multiclass logistic loss for δ ∈_ [0 _,_ 1] _._ 

_Proof._ First, we note that we can study the 0-1 risk of the limiting predictor: 



where in the last step we use that since the output space of _fn_ is discrete and the probabilities that the in-context predictor takes an argmax over converges, then for _N_ large enough, _fN_ ( _x_ test) = lim _n→∞ fn_ ( _x_ test). 

Note that for every input _x_ test, the limiting in-context learning predictor outputs the argmax of a predictive distribution which can be a mixture of predictive distributions over _B_ : 



for some distribution _q_ over _B_ . The KL divergence between this mixture and the prompt concept is bounded by the KL divergence of any one _θ ∈B_ , due to the convexity of KL: 



where we can exchange the order of expectations since the KL is bounded (dominated convergence). 

23 

From the KL bound _KL_ ( _p_ prompt( _y_ test _|x_ test) _∥p_ ( _y_ test _|x_ test _, θ_ ), we thus have 



where _L_ CE( _θ_ ) = _−_ E _x_ test _∼p_ prompt[ _p_ prompt( _y_ test _|x_ test) log _p_ ( _y_ test _|x_ test _, θ_ )] is the multiclass logistic risk, and _L_ CE( _θ_<sup>_∗_</sup> ) is the optimal risk over _θ ∈_ Θ by assumption. Applying Theorem 2.2 and 5.11 of Ávila Pires and Szepesvári (2016), _g_ is a calibration function for the multiclass logistic loss, and allows us to convert the surrogate risk bound to a bound on the 0-1 loss, giving the result. Note that we have zero approximation error here, since _θ_<sup>_∗_</sup> _∈_ Θ. 

Note that _g_<sup>_−_1</sup> is roughly linear in _ϵ_ for _ϵ_ smaller than 0.7, where the bound is non-vacuous. 

### **E.1 Proof of Theorem 2** 

_Proof._ By the continuity assumption, we have for any _θ_ in _B_ that 





We use this to bound the last KL term by plugging it in below: 



Rearranging and noting that _KLk_ ( _θ_<sup>_∗_</sup> _∥θ_ ) = E _x_ test _∼p_ prompt[ _KL_ ( _p_ prompt( _y_ test _|x_ test) _∥p_ ( _y_ test _|x_ test _, θ_ ))], we have 



Plugging into Lemma 2 gives the result. 

### **E.2 Proof of Theorem 3** 

Note that Condition 1 ensures that the sum of KL divergences between positions within a _k_ -length input is bounded. This means that we have a bound over not only the last-position KL divergence, but also for all the intermediate tokens. Intuitively, the random length test example allows the incontext predictor to “take credit” for fitting the intermediate tokens. The proof is immediate given the KL bound and Lemma 2, given that the length of _x_ test is uniformly random between 2 to _k_ . 

24 



Figure 8: Example pretraining document snippet ( **Left** ) and example prompt with 3 training examples, 1 test example, and example length 3 ( **Right** ). The delimiter token is the backslash. 

_Proof._ Let the set of _θ_ that does not satisfy Condition 1 to be _B_ . We have for any _θ_ in _B_ that 





by Theorem 1 and Condition 1. Plugging this into Lemma 2 gives the result. 

## **F Experimental details** 

### **F.1 GINC dataset** 

**Pretraining distribution.** We consider a pretraining distribution from a mixture of HMMs with an interpretable hidden state structure and emission distribution. The HMM hidden state _ht_ = [ _st, vt_ ] at time _t_ is composed of an _entity vt ∈{_ 1 _, . . . , |V|}_ (e.g., Einstein) and a _property st ∈{_ 1 _, . . . , |S|}_ (e.g., nationality, first name, last name, other grammatical tokens). We model the entities and properties as independent Markov chains (i.e., a factorial HMM (Ghahramani and Jordan, 1997)), while the emissions depend on both. In pretraining documents, we expect that the entities (e.g., Einstein) change slowly over time while and the properties of the entity (e.g., their nationality) change quickly with some pattern to generate natural sentences. We implement this by ensuring that the probability of transitioning to the same entity index in the next step is at least 0.9. The emission distribution depends on a memory matrix _M_ with _|V|_ rows and _|S|_ columns (Figure 9). At step _t_ , we use the entity _vt_ and property _st_ to index into the memory matrix. In particular, the observed tokens are deterministic with _p_ ( _ot|ht_ ) = 1 if _ot_ = _M_ [ _vt, st_ ]. This construction satisfies the structure on delimiter states (Assumption 1). We ensure that all the transitions have nonzero probability and use a uniform prior over concepts, satisfying Assumptions 2 and 5. 

25 



Figure 9: The GINC dataset generates sequences from a mixture of HMMs. The HMM hidden states consist of entities ( _v_ ) and properties ( _s_ ), which index into a memory matrix to produce the observed token. The entity and property sequences are sampled from independent Markov chains. The concept parameter _θ_ is the transition matrix for properties, which defines relations between properties. In this example, the sequence of properties [2,3,5,4] relates names to nationalities, defining the in-context task. The blue color represents hidden states/observations sampled from the prompt distribution, and the purple color represents hidden states/observations sampled from the pretraining distribution. 

**Concept parameter.** The concept parameter is the property transition matrix, while the entity transition matrix is fixed for all concepts. The prompt start distribution and the concept together determine the in-context task. We define a uniform mixture of HMMs over a family Θ of 5 concepts to generate 1000 documents with _∼_ 10 million tokens total. 

**Vocabulary.** The GINC dataset is generated from a mixture of HMMs. These HMMs output tokens from a vocabulary of size in _{_ 50 _,_ 100 _,_ 150 _}_ . The vocabulary contains a special delimiter token (backslash – see Figure 8, designated to be index 1. The vocabulary is generated as combinations of letters starting from a to z, then aa to az, and so on. All sequences are tokenized by splitting on whitespaces. 

**Memory matrix.** The shared memory matrix has 10 entities and 10 properties, totaling 100 entries (corresponding to 100 hidden states). The first column of the memory matrix is fixed to be the delimiter token, while each remaining entry of the shared memory matrix is populated with a token sampled uniformly from the vocabulary. 

**Transition matrix for properties.** We generate 5 property transition matrices, one for each component of the HMM mixture. We generate each transition matrix via a convex combination of 100 random permutation matrices. The weights of the convex combination are randomly generated as 



where _u ∈_ R<sup>100</sup> has uniform random entries in [0 _,_ 1] and _t_ is a temperature parameter, set to 0.1. 

26 



<!-- Start of picture text -->
35 k=3 k=3 90 k=3<br>k=5 60 k=5 80 k=5<br>30 k=8 k=8 k=8<br>k=10 50 k=10 70 k=10<br>25<br>60<br>40<br>20 50<br>30 40<br>0 20 40 60 0 20 40 60 0 20 40 60<br>Num examples Num examples Num examples<br>Acc Acc Acc<br><!-- End of picture text -->

Figure 10: In-context accuracy curve of the 4 layer Transformer on the GINC dataset when the entity transition matrix does not have an additional identity component, for vocabulary sizes 50 (left), 100 (middle), and 150 (right). In-context learning is still generally successful. 

**Transition matrix for entities.** The entity transition matrix is shared between all the HMMs that consistute the mixture. The entity transition matrix is generated in the same way as the property transition matrices, except with one additional step. Letting _T_ be a transition matrix sampled in the same way as a property transition matrix, 

In pretraining documents, we expect that the entities (e.g., Einstein) change slowly over time while and the properties of the entity (e.g., their occupation) change quickly with some pattern to generate natural sentences. We implement this by ensuring that the probability of transitioning to the same entity index in the next step is at least 0.9. The final entity transition matrix is then 0 _._ 1 _T_ +0 _._ 9 _I_ where _I_ is the identity matrix. Although we add the diagonal component for added realism, we also consider not adding this component. Figure 10 shows in-context learning curves for a small (4 layer) Transformer trained on data that does not add the diagonal component (we check this for vocabulary sizes 50, 100, and 150). In-context learning still works in this case, although not as well for the 50 vocab size case. 

**Start distribution.** The starting distribution for the hidden states in all HMMs in the mixture are close to uniform. We generate the start distribution as softmax(( _u −_ 0 _._ 5) _/t_ ) for random vector _u_ with entries uniformly from [0 _,_ 1] and temperature _t_ = 10. In the pretraining documents, we only sample from the start distribution in the beginning of the document. 

**Prompt distribution.** We generate prompts with 0 to 64 training examples and example lengths _k ∈{_ 3 _,_ 5 _,_ 8 _,_ 10 _}_ (2500 prompts for each setting). The target token _y_ test is taken to be the most likely output arg max _y p_ prompt( _y|x_ test) instead of sampling so that the intrinsic error is 0. 

**Prompt distribution.** To generate the prompts, we first sample a concept _θ_ uniformly at random from Θ (well-specification, Assumption 4), then use it to generate all the prompt examples. The prompt start distribution is chosen to be uniform over entities but with a fixed starting property that is chosen randomly for each prompt, for consistency in the task. This may not satisfy Assumption 3, but we found this to still work empirically and is simpler. Given the starting property, we sample _k_ tokens from the HMM defined by the concept _θ_ . Finally, we append the delimiter token for the example. We repeat this process for each example in the prompt, concatenating all examples. The label is generated as 



under the prompt concept _θ_<sup>_∗_</sup> . This differs from the theory, which samples _y_ test instead of taking it to be the most likely token. However, there can be a large amount of intrinsic error that sampling 

27 

introduces. We define the label this way in the simulations to remove the intrinsic error from sampling. 

**Example of prompt generation.** In the example in Figure 8 (right), the starting property is fixed to be 5 (for example). The first token (l) is generated by sampling a random entity index (3), and indexing into the memory matrix returns l. Running the hidden state chain of the HMM forward gives the next pair of property and entity. Since the entity Markov chain changes slowly, the entity is still 3 in the next step – however, the property has changed to 4, and indexing into the memory matrix outputs the next token (aw). Following this same process to generate the third token (the output for the first example), we finish generating one example. To end the example, we append a delimiter (backslash). We repeat this example generation process for all the examples, except for the test example at the end, where we do not generate the last token. We condition the HMM on the generated prompt to compute the posterior distribution over the next token _p_ prompt( _y|x_ test). We take the argmax of this distribution to be the ground truth label. 

**Dataset details.** The dataset contains 1000 training documents and 100 validation documents, where training documents have 10240 tokens and validation documents have 1024 tokens. Each document is generated by first selecting one of the HMMs from the mixture uniformly at random, then generating 10240 tokens from the HMM. 

We also generate 2500 in-context prompts for each (example length,number of examples) pair, for example lengths _k_ = [3 _,_ 5 _,_ 8 _,_ 10] and number of examples _n_ = [0 _,_ 1 _,_ 2 _,_ 4 _,_ 8 _,_ 16 _,_ 32 _,_ 64]. Each prompt is generated using a random HMM in the mixture. 

### **F.2 Transformer details** 

Our Transformer models are based on the GPT-2 architectures with 4, 12, and 16 layers respectively, with 12 attention heads, 768 dimensional embeddings, residual/embedding/attention dropout set to 0.1, and a context window of 1024. Other than the number of layers, the other parameters are the default settings from the HuggingFace library (Wolf et al., 2019). We train for 5 epochs using the AdamW optimizer (Kingma and Ba, 2015, Loshchilov and Hutter, 2019) with a batch size of 8 and a linear learning rate schedule (with 1000 step warmup) up to a learning rate of 8e-4 for the 4 layer and 12 layer model, while for the 16 layer model we start with a constant learning rate of 8e-4 and reduce by a factor of 0.25 whenever the best validation loss does not improve. We tried both learning rate strategies for all models and take the most consistent. We tuned these models so that the training loss curves between seeds have smaller variability between the runs in terms of the curve shape and when the loss decreases – we found that this is an important indication of stable results. The models took 50 minutes, 2 hours, 3 hours to train respectively. The hardware was mainly Titan Xp GPUs, trained and evaluated using 16-bit precision. All the results are reported with 5 pretraining runs (5 different seeds). 

### **F.3 LSTM details** 

We train an LSTM language model with embedding size 768, hidden layer size 768, and 6 layers. We use dropout 0.2 and weight decay 1e-5. The optimizer is AdamW starting with a learning rate of 1e-3, then reducing by a factor of 0.25 whenever the best validation loss does not go down. We train for a total of 10 epochs, with gradient clipping at norm 1.0. We use a batch size of 8 and backpropagate through time for 1024 steps (each pretraining data segment is also 1024 tokens). Each model takes roughly 2 hours to train on Titan Xp GPUs. 

28 

### **F.4 Varying the vocabulary size** 

To do well on the in-context learning task, the model must both infer the prompt concept and the last HMM hidden state. In general, increasing the number of observable symbols makes the incontext task easier by making the inference of the HMM hidden state easier. With more symbols, each hidden state is more likely to output a different symbol, making the inference problem easier. This improvement comes despite the number of output classes in the problem (same as the vocabulary size) increasing. Figures 11, 12, 13, 14 show in-context learning curves for vocabulary sizes 50, 100, and 150, keeping other hyperparmeters of the dataset the same. 



<!-- Start of picture text -->
k=3 70 k=3 90 k=3<br>60 k=5 k=5 k=5<br>k=8 k=8 k=8<br>k=10 60 k=10 80 k=10<br>50<br>50 70<br>40<br>60<br>40<br>30<br>0 20 40 60 0 20 40 60 0 20 40 60<br>Num examples Num examples Num examples<br>Acc Acc Acc<br><!-- End of picture text -->

Figure 11: In-context accuracy of the 4 layer Transformer on the GINC dataset for vocabulary sizes 50 (left), 100 (middle) and 150 (right). Accuracies generally improve as the vocabulary size increases. 



<!-- Start of picture text -->
90<br>100<br>k=3 k=3 k=3<br>80 k=5 80 k=5 k=5<br>k=8 k=8 90 k=8<br>k=10 70 k=10 k=10<br>60 80<br>60<br>50 70<br>40<br>40 60<br>0 20 40 60 0 20 40 60 0 20 40 60<br>Num examples Num examples Num examples<br>Acc Acc Acc<br><!-- End of picture text -->

Figure 12: In-context accuracy of the 12 layer Transformer on the GINC dataset for vocabulary sizes 50 (left), 100 (middle) and 150 (right). Accuracies generally improve as the vocabulary size increases. 



<!-- Start of picture text -->
90 k=3 90 k=3 100 k=3<br>80 k=5 80 k=5 k=5<br>70 k=8k=10 70 k=8k=10 90 k=8k=10<br>60 60 80<br>50 50 70<br>40 40<br>60<br>0 20 40 60 0 20 40 60 0 20 40 60<br>Num examples Num examples Num examples<br>Acc Acc Acc<br><!-- End of picture text -->

Figure 13: In-context accuracy of the 16 layer Transformer on the GINC dataset for vocabulary sizes 50 (left), 100 (middle) and 150 (right). Accuracies generally improve as the vocabulary size increases. 

29 



<!-- Start of picture text -->
100 100<br>100<br>k=3 k=3 k=3<br>k=5 k=5 k=5<br>80 k=8 80 k=8 80 k=8<br>k=10 k=10 k=10<br>60 60 60<br>40<br>40 40<br>0 20 40 60 0 20 40 60 0 20 40 60<br>Num examples Num examples Num examples<br>Acc Acc Acc<br><!-- End of picture text -->

Figure 14: In-context accuracy of the LSTM on the GINC dataset for vocabulary sizes 50 (left), 100 (middle) and 150 (right). Accuracies generally improve as the vocabulary size increases. 

|Prompt example length|Test Acc (200–300 chars)|
|---|---|
|5 examples||
|Short (200–300 chars)|69.8|
|Long (500–600 chars)|70.7|
|10 examples||
|Short, duplicated examples|69.6|
|Short, independent examples|71.4|



Table 1: Accuracies for 5-shot in-context learning of GPT-3 on a filtered LAMBADA test set with short examples (200–300 characters). Even though there is distribution mismatch with the test set, having longer examples improves the accuracy, supporting theoretical intuitions. The first two rows use 5 training examples in the prompt, while the last two rows use 10 training examples to equalize the total length. 

### **F.5 Experiment on GPT-3** 

We conduct an additional experiment which shows that longer examples improve in-context learning in GPT-3 on the LAMBADA (Paperno et al., 2016) completion task. 

**Data.** In this experiment, we define a short version of the LAMBADA test dataset (LAMBADA test-short) which contains only test examples with up to 200–300 characters in length. We also define two “training” datasets from which to sample examples for the in-context prompts from. The short training dataset (LAMBADA train-short) contains examples from the training set that are 200–300 characters in length, which matches the distribution of test-short. The long training dataset (LAMBADA train-long) contains training examples that are 500–600 characters long. We cut the number of examples in the larger of the two training datasets so that the two training datasets are equally sized (47 examples). For each test example, we sample 5 random training examples (5-shot learning). 

We also consider equalizing the total length of the prompts in two ways. First, we consider duplicating the 5 short examples (if the examples are [1,2,3,4,5], duplicating refers to [1,2,3,4,5,1,2,3,4,5]). This allows for equalizing the total length without increasing the number of examples. As a skyline comparison, we also consider sampling 10 independent short examples, which contains more input-output pairs for the task. 

**Result.** Table 1 shows that when evaluating only on LAMBADA test-short, 5-shot in-context learning using LAMBADA train-long improves the test accuracy by almost 1% compared to LAMBADA 

30 

train-short, despite the long/short distribution mismatch between train and test. This supports intuitions from our theory. 

In comparison, simply increasing the total prompt length by duplicating the short examples does not improve the accuracy. Intuitively, the longer examples have additional information that is not directly related to mapping between the input and output, but can be leveraged to improve incontext learning by helping the model infer the latent concept. Using 5 long examples (as opposed to 5 short examples) closes about 56% of the gap between using 5 short examples and 10 independent short examples despite not adding additional examples or task-related information. 

31 

