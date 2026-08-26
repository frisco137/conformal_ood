# **TabPFN Through The Looking Glass: An interpretability study of TabPFN and its internal representations** 

### **Aviral Gupta**<sup>1</sup> **Armaan Sethi**<sup>1</sup> **Dhruv Kumar**<sup>1</sup> 

## **Abstract** 

Tabular foundational models are pre-trained models designed for a wide range of tabular data tasks. They have shown strong performance across domains, yet their internal representations and learned concepts remain poorly understood. This lack of interpretability makes it important to study how these models process and transform input features. In this work, we analyze the information encoded inside the model’s hidden representations and examine how these representations evolve across layers. We run a set of probing experiments that test for the presence of linear regression coefficients, intermediate values from complex expressions, and the final answer in early layers. These experiments allow us to reason about the computations the model performs internally. Our results provide evidence that meaningful and structured information is stored inside the representations of tabular foundational models. We observe clear signals that correspond to both intermediate and final quantities involved in the model’s prediction process. This gives insight into how the model refines its inputs and how the final output emerges. Our findings contribute to a deeper understanding of the internal mechanics of tabular foundational models. They show that these models encode concrete and interpretable information, which moves us closer to making their decision processes more transparent and trustworthy. 

## **1. Introduction** 

Tabular Foundational Models are becoming increasingly popular as a tool to replace traditional machine learning techniques in real world prediction tasks. These models are trained at scale on large collections of synthetic datasets generated from structural causal models (Peters et al., 2017; 

> 1Birla Institute of Technology and Science, Pilani. Correspondence to: Aviral Gupta _<_ f20220097@pilani.bits-pilani.ac.in _>_ . 

Pearl, 2009) and related generative processes. This broad pretraining gives them strong generalization across many tabular distributions and has led to performance that often surpasses classical methods such as gradient boosted trees or shallow neural networks. As a result they are now seen as potential drop in replacements for a wide range of supervised pipelines in domains such as finance (Brauer, 2024; Chu et al., 2024; Nguyen), healthcare (Noda et al., 2024; Dyikanov et al., 2024; Alzakari et al., 2024; Karabacak et al., 2024), manufacturing and industrial engineering (Magadan´ et al., 2023; Bellarmino et al., 2025; He et al., 2025) and scientific data analysis (Chen et al., 2025; Sharma et al., 2025; Sharma, 2025). 

However, the growing use of such models introduces a significant challenge. The black box transformer based nature of their architecture makes their processing opaque. Their predictions are produced through stacked attention layers and MLP blocks that obscure the intermediate steps of the computation. This opacity stands in contrast to traditional tabular models, which usually provide clearer reasoning paths or explicit feature contributions. When TabFMs begin replacing these more transparent pipelines, the lack of interpretability becomes a central concern for trust, debugging, and compliance with regulatory requirements. Understanding what information TabFMs store internally and how they compute their outputs becomes increasingly important as these models move toward deployment. 

Although TabPFN and related models achieve strong accuracy across a wide range of tasks, there is still very limited understanding of how these architectures represent tabular structure or carry out the computations required for prediction. Existing interpretability work on tabular models mostly focuses on post hoc feature importance, which does not reveal the internal mechanisms that produce the final decision (Rundel et al., 2024). Meanwhile, the interpretability community studying large language models (Zhao et al., 2023) has shown that transformer representations often contain linearly readable information about tasks (Todd et al., 2024), intermediate computations, and concepts (Park et al., 2024). These results suggest that transformers learn structured internal algorithms rather than acting as unstructured black box approximators. Since TabPFN relies heavily on 

1 

**TabPFN Through The Looking Glass** 

in context learning to approximate the target function implied by the provided training data, it becomes natural to ask whether similar algorithmic structure appears inside its hidden representations. 

In this work we investigate this question through a set of targeted probing experiments. We construct synthetic datasets where the true functional relationships are known, and we examine how TabPFN v2 (Ye et al., 2025) encodes these relationships inside it’s activations. Our probes target three categories of information. First, we test whether the coefficients of simple linear relationships are recoverable from the hidden states. Second, we probe for intermediate quantities that arise inside multi step arithmetic expressions, which allows us to study whether the model represents internal computational steps. Third, we analyze how the final predicted value forms throughout the network using both linear probes (Belinkov, 2022) and the logit lens (nostalgebraist, 2020) to track the evolution of the output across layers. 

Across the experiments we find interpretable structure in the residual stream. Linear coefficients and intermediate arithmetic terms are often linearly decodable in the middle layers, indicating that these quantities are represented explicitly during computation. Although the information encoded in the representations might not be consistent across different instances of the model with separate datasets as contexts, we can still see success in extracting different quantities from a single large context with all the datasets combined. We also observe that the correct answer emerges earlier in the forward pass than the final prediction head, showing that the model converges on the correct value before the last layer projects it into the output space. These findings suggest that TabFMs perform multi step structured computation internally, rather than relying on an opaque end to end mapping. Our contributions are as follows. 

1. We present the first mechanistic and representational analysis of TabPFN models, to the best of our knowledge. 

2. We show that both linear coefficients and intermediate arithmetic quantities are embedded in hidden states in a linearly decodable manner. 

3. We trace how the model forms the final answer and characterize how this signal evolves through the depth of the transformer. 

4. We provide empirical evidence that TabFMs implement structured internal computation, pointing toward the possibility of more interpretable and trustworthy tabular foundation models. 

## **2. Related Work** 

While tree-based methods like XGBoost have historically dominated tabular data tasks (Grinsztajn et al., 2022), recent work has shifted toward Deep Learning architectures designed for tabular modalities (Gorishniy et al., 2021). A significant paradigm shift was the introduction of Prior-Data Fitted Networks (PFNs), most notably TabPFN (Hollmann et al., 2023). Unlike traditional transfer learning, TabPFN is a Transformer pre-trained on a vast corpus of synthetic datasets generated from Structural Causal Models (SCMs) (Peters et al., 2017; Pearl, 2009). It acts as a proxy for Bayesian inference, solving new classification tasks in a single forward pass via In-Context Learning (ICL) (Ye et al., 2025). Subsequent iterations, such as TabPFN-v2, have addressed scalability limitations regarding context length and feature counts, with a new improved architecture (Hollmann et al., 2025).However, interpretability for these models has largely remained restricted to post-hoc feature importance methods like Shapley Value’s or Representation Analysis and other classical techniques (Ye et al., 2025; Hollmann et al., 2025; Rundel et al., 2024) rather than mechanistic investigation. 

As TabPFN relies on ICL to approximate classification functions, understanding the mechanics of ICL is crucial. In Large Language Models (LLMs), ICL is hypothesized to emerge from specific attention circuits. Olsson et al. (2022) identified induction heads which are circuits that implement a ”copying” mechanism by attending to previous tokens that appeared in similar contexts as a primary driver of ICL. Furthermore, Hendel et al. (2023) and Todd et al. (2024) introduced the concept of Task Vectors (or Function Vectors), showing that the ”task” specified by the context (e.g., a classification rule) is compressed into a distinct, manipulable direction within the model’s residual stream. 

Our work is grounded in the methodology of linear probing (Alain & Bengio, 2018) seeking to decode information stored in intermediate layers. While this is well-established in NLP (Belinkov, 2021), recent work has extended these techniques to continuous-value domains closer to tabular data. Wilinski et al.´ (2025) analyzed representations in Time Series Foundation Models (TSFMs), employing Centered Kernel Alignment (CKA) to map layer-wise similarity and activation steering to identify steerable concepts (e.g., trend and seasonality) within the latent space. They demonstrated that, much like LLMs, continuous-domain Transformers learn structured, disentangled representations of high-level concepts. We extend this line of inquiry to Tabular FMs, investigating whether TabPFN similarly encodes decision boundaries and feature interactions in its residual stream. To the best of our knowledge, this represents the first mechanistic interpretability study of TabPFN models. 

2 

**TabPFN Through The Looking Glass** 

## **3. Experiments and Results** 

### **3.1. Preliminaries** 

Transformer models have been used extensively in both NLP and computer vision to capture relationships within sequential inputs. This paradigm has been extended to structured tabular data, where transformer-based approaches have become competitive with classical models. A tabular dataset is defined consisting of data in which each row _xi_ consists of _d_ features or attributes, where _d_ typically varies across datasets with a label _yi_ belongs to [ _C_ ] = _{_ 1 _, . . . , C}_ for a classification task or is a continuous numerical value for a regression task. 

Among transformer models for tabular learning, TabPFN currently represents the state of the art, with widespread adoption in both industry and academia. TabPFN performs in-context learning to predict the test set given the train set, without requiring further parameter updates. To enable this it is pretrained on a large collection of synthetic datasets and learns to approximate Bayesian inference over this prior distribution. 

The primary model used in this paper is TabPFN v2, thus necessitating a deeper dive into its architecture. 

**TabPFN v2.** The first step of the pre-processing pipeline includes numerical encoding of categorical features and normalisation of all the features. The model then embeds the _d_ features into a _k_ -dimensional space, adding perturbations to differentiate the features. Alongside the label embedding _y_ ˜ _i ∈_ R<sup>_k_</sup> , each training instance _xi_ is represented by ( _d_ + 1) tokens with dimension _k_ . For a test instance _x_<sup>_∗_</sup> , a dummy label is used to generate the label embedding _y_ ˜<sup>_∗_</sup> . The full shape of the input is thus represented as a tensor of shape ( _N_ + 1) _×_ ( _d_ + 1) _× k_ . TabPFN v2 uses two types of self-attention, one after the other, the first being over samples known as sample attention and the other over attributes known as feature attention, which combine to enable in-context learning across either axis. At the end of the transformer layers the output token corresponding to the dummy label _y_ ˜<sup>_∗_</sup> is extracted and mapped to a 10-class logit for classification or is mapped to a single number in case of regression. 

### **3.2. Probing for coefficients** 

As TabPFN is an In-Context Learning based model, previous work has shown that coefficients of linear relationships can be extracted from transformers when they are given train dataset as context and asked to predict the answer for test samples. Building on this we investigate whether TabPFN’s internal representations encode the coefficients of the modeled linear function 





_Figure 1._ **Coefficient probe across layers** . Probing _R_<sup>2</sup> and MSE for coefficient of the linear relationship plotted across different layer activations. The probe _R_<sup>2</sup> sees a sharp increase at layer 6 and drops off at the last layer accompanied by the inverse behaviour in the MSE. High _R_<sup>2</sup> and low MSE values indicate a better performing probe. 



_Figure 2._ **Coefficient probe across complexities** . Probing _R_<sup>2</sup> and MSE for coefficient of the linear relationship plotted across increasing probe complexities, adding hidden layers to an MLP. The probe _R_<sup>2</sup> decreases and the MSE increases with increasing probe complexity, giving evidence of a linear encoding of the coefficients within the activations. 

. To recover these coefficients from the activations, we use two distinct probing setups. A probe is a function 



which maps an activation tensor _A ∈_ R<sup>_d_</sup> to a target property, in this case _α ∈Y._ Probes are typically chosen to be simple, lightweight functions, such as linear maps or a shallow neural network. 

**First configuration** . We vary _α_ and _β_ across multiple datasets and then fit separate instances of TabPFN for each of these datasets. We can write dataset _n_ thus as: 



We then extract forat a single layer for all datapoints activation _A ∈_ R<sup>_n×d×k_</sup> and flatten it to get a activation _A ∈_ R<sup>_ndk_</sup> 

3 

**TabPFN Through The Looking Glass** 

vector, trying to learn the probe. We collate these activations for all the different datasets to construct the probing dataset. 

The probing dataset is a mapping from this activation vector to the coefficient _α_ or _β_ . 



This approach yields poor training accuracy. The low probe performance indicates that each TabPFN fit encodes the coefficients differently, so when activations from many fits are pooled, the coefficients no longer align in any common representation that a probe can read out. 

**Second configuration** . In this setup we combine the multiple datasets with varying _α_ and _β_ into one large dataset with another added input variable which is unique and maps directly to the different coefficient combinations. So, this added variable acts like a switch input that switches between the coefficient pairs. 



With this dataset we fit the TabPFN model with dataset containing different coefficient pairs and then extract activations from a holdout test set. From these extracted activations we construct the probing dataset which is a mapping from the activations of all the tokens but the switch token to the coefficients. The individual activations which are of shape _A ∈_ R<sup>_d×k_</sup> are flattened to _A ∈_ R<sup>_dk_</sup> . The probing dataset thus is a mapping from the activations of a single data point to the coefficient _α_ or _β_ . 



The results for this setup, as show in Figure 1, indicate high train and test performance for the probes, showing that we are able to extract the coefficients from the internal activations from a single fit. 

This performance disparity between the first and the second configuration indicates that the successful probes in the second configuration setting likely rely on distinguishing between dataset-specific signatures in the coefficients of the linear expressions rather than decoding the coefficient themselves. We posit from this evidence that the coefficients are not encoded as a universal quantity in the representation as shown by the low training accuracy on different fits, but it can be extracted from the representations to a degree as shown by the same fit probing results. 

We also run the above probing experiments with increasingly complex probes. The complexity of the probes is controlled by adding more hidden layers into the MLP. The results in Figure 2 show that with increasing complexity, the _R_<sup>2</sup> scores decrease. This gives evidence for the linear encoding of the coefficients within the activations of the model. 



_Figure 3._ **Intermediary probe across layers** . Probing _R_<sup>2</sup> and MSE for the intermediary of the expression plotted across different layer activations. Two different expressions and intermediaries displayed. The probe _R_<sup>2</sup> sees a sharp increase at layer 6 and drops off at the last layer accompanied by the inverse behaviour in the MSE. High _R_<sup>2</sup> values and low MSE indicate a better performing probe. 



_Figure 4._ **Intermediary probe across complexities** . Probing _R_<sup>2</sup> and MSE for the intermediary of the expression plotted across increasing probe complexities, adding hidden layers to an MLP. The probe _R_<sup>2</sup> decreases with increasing probe complexity, giving evidence of linear encoding of the intermediary within the activations. 

### **3.3. Probing for Intermediaries** 

To further investigate the model’s internal computation, we formulate an experiment to probe for intermediate expressions required to compute the final answer. This might reveal if the model is working through the expression in a hierarchical manner, computing simpler products and then combining to get the final answer. 

So, to elucidate the compositional nature of TabPFN’s internal reasoning, we constructed a dataset of an arithmetic relationship 



In this setting, the product _a · b_ acts as a necessary intermedi- 

4 

**TabPFN Through The Looking Glass** 

ate variable that must be computed before the final addition with _c_ . We aim to probe for this intermediate product 

We fit a TabPFN instance on this dataset and then extract the internal activations of the test set during the forward pass. We then construct the probing dataset by mapping the internal activation to the intermediate product for that sample. 



We train a linear probe on this dataset which shows that the representation of _a · b_ is highly recoverable in the middle layers of the model, before fading in later layers as the final output _z_ is constructed. This localization of the intermediate term provides evidence that TabPFN performs structured, hierarchical computation, effectively mirroring the mathematical order of operations within its internal activations. 

Again, as explained in section 3.3 we repeat the probing experiments for increasingly complex probes. The results in Figure 2 show that with increasing complexity, the _R_<sup>2</sup> scores decrease. This gives evidence for the linear encoding of the intermediaries in the internal representations of the model. 

### **3.4. Probing for the answer and Logit Lens** 

We now moved to experiments aimed at localizing where the answer is formed withing the model’s computation. Taking from previous work on LLM interpretability (Zhao et al., 2023), to investigate how the solution emerges across the network depth, we employed the logit lens technique (nostalgebraist, 2020). In this technique we apply the model’s final unembedding matrix directly to the intermediate layer activations aiming to find how the answer is refined through the layers. 



We use the same setup as Section 3.2 to fit TabPFN models on linear relationships. We observed that the internal representations strictly align with the valid output space only in the later stages of the network, with Layer 7 proving critical for this alignment in the linear regression task. 

However, extracting the output using trained linear probes reveals a contrasting dynamic. For relatively simple arithmetic relationships both linear regression and the compound _z_ = _a_ + _b · c_ task, we found that the final answer _z_ can be linearly decoded with high effectiveness as early as Layer 5. The discrepancy between the probe-accessible solution (Layer 5) and the native output alignment (Layer 8) suggests a degree of computational inefficiency for simple tasks. TabPFN appears to compute the solution early but continues to process the representation through several subsequent layers before projecting it onto the final output manifold 



_Figure 5._ **Answer probe across layers** . Probing _R_<sup>2</sup> and MSE for the answer of the expression in the answer token activations plotted across different layer activations. The probe reaches a very high _R_<sup>2</sup> at layer 5 and stays consistent, till an expected drop at the last layer accompanied by the inverse behaviour in the MSE. This indicates the probes effectiveness at extracting the answer earlier in the model through its activations. 



_Figure 6._ **Logit Lens results** . This shows the results of the logit lens plotted against the model layer where it is applied. Here, we can see the the result converges to the true answer around layer 8. 

### much later. 

### **3.5. Probing for inputs in answer token** 

Building on earlier work in large language model interpretability, which identified copy and induction heads that move information between tokens for later computation, we design an experiment to probe for input information directly in the answer token. Similiar to 3.4, we create a dataset with the compound expression 



Our goal is to test whether the input values and their linear combinations appear in the activations of the answer token. We extract the activations of the TabPFN test set only at the last index, which corresponds to the asnwer token, _y ∈_ R<sup>_d_</sup> .Using these activations, we form a probing dataset by mapping them to the different input values and 

5 

#### **TabPFN Through The Looking Glass** 



_Figure 7._ **Input probe across layers** . Probing _R_<sup>2</sup> for the inputs and their linear combination in the answer token activations plotted across different layer activations. The probe reaches a very high _R_<sup>2</sup> at layer 5 and stays consistent, till an expected drop at the last layer. This indicates the probes effectiveness at extracting the inputs purely from the answer tokens indicating some kind of copying behavior in the earlier parts of the model. 

encoding of the coefficients in its internal representations. 

The results in the probing and logit lens section again indicate that although the semantic content of the solution is computed early, the subsequent layers are dedicated to transforming this representation into the specific geometric alignment required by the output head with certain layers, such as layer 7, being important for this shift in representation. 

Our analysis is currently limited to simple arithmetic (additive/multiplicative) toy expressions compared to the highly complex noisy functions found in real world datasets which can be improved upon in future work. Future work could also entail finding the mechanism/mechanisms through which the answer is computed using mechanistic interpretability methods like activation patching, or trying to manipulate the function through steering vectors or vector ablation. 

## **References** 

their combinations. 



Figure 7 shows that these quantities can be recovered early in the model. This suggests that the model copies input information forward and encodes it in different directions within the answer token subspace. 

## **4. Discussion and Future Work** 

In this work, we presented a mechanistic analysis of TabPFN, moving beyond post-hoc feature importance to investigate the internal algorithmic structure of Tabular Foundational Models. Our experiments shine light on the TabPFN black-box and reveal that TabPFN constructs its answers systematically across several layers with varying functions and importance, with intermediate results linearly represented in the residual stream. 

Our results regarding intermediate value probing offer strong evidence that TabPFN implements systematic reasoning. By successfully recovering the intermediate term _a · b_ in the calculation of _z_ = _a · b_ + _c_ , particularly in the middle layers, we show that the model systematically constructs the answer, respecting the mathematical order of operations within its residual stream. Probing for coefficients revealed that TabPFN may not generalize across fits but does generalize within them, i.e similar functions may be represented differently across fits, but have same or similar representation within the same fit, this is demonstrated by the failure of probes to generalize across different fits in section 3.2 (first configuration), contrasted with their high success within a single fit in section 3.2 (second configuration). These results imply that TabPFN does not maintain fixed, universal 

- Alain, G. and Bengio, Y. Understanding intermediate layers using linear classifier probes, 2018. URL https:// arxiv.org/abs/1610.01644. 

- Alzakari, S. A., Aldrees, A., Umer, M. F., Cascone, L., Innab, N., and Ashraf, I. Artificial intelligence-driven predictive framework for early detection of still birth. _SLAS Technology_ , 29(6): 100203, 2024. doi: 10.1016/j.slast.2024.100203. URL https://www.sciencedirect.com/ science/article/pii/S2472630324000852. 

- Belinkov, Y. Probing classifiers: Promises, shortcomings, and advances, 2021. URL https://arxiv.org/ abs/2102.12452. 

- Belinkov, Y. Probing classifiers: Promises, shortcomings, and advances. _Computational Linguistics_ , 48(1):207– 219, 04 2022. ISSN 0891-2017. doi: 10.1162/coli ~~a~~ 00422. URL https://doi.org/10.1162/coli_ a_00422. 

- Bellarmino, N., Cantoro, R., Huch, M., and Kilian, T. Minimal supervision, maximum accuracy: Tabpfn for microcontroller performance prediction. In _Proceedings of the International Test Conference (ITC)_ , 2025. doi: 10.1109/ITC58126.2025.00067. URL https://iris. polito.it/handle/11583/3002056. Applies TabPFN for MCU performance screening with minimal supervision. 

Brauer, A. Enhancing actuarial non-life pricing models via transformers. _European Actuarial Journal_ , 14: 991–1012, 2024. doi: 10.1007/s13385-024-00388-2. 

6 

**TabPFN Through The Looking Glass** 

- URL https://link.springer.com/article/ 10.1007/s13385-024-00388-2. 

- Chen, B., Xiong, Z., Zhao, Y., and Zhang, J. Multi-view machine learning model of ash chemical composition– minerals: Improving ash fusibility prediction and interpretability of high-alkali coal. SSRN preprint 5406504, 2025. URL https://ssrn.com/abstract= 5406504. 

- Chu, J. Z. K., Than, J. C. M., and Jo, H. S. Deep learning for cross-selling health insurance classification. In _Proceedings of the 2024 International Conference on Green Energy, Computing and Sustainable Technology (GECOST)_ , Miri, Sarawak, Malaysia, 2024. IEEE. URL https://ieeexplore.ieee.org/ abstract/document/10475046. 

- Dyikanov, D., Zaitsev, A., Vasileva, T., Wang, I., Sokolov, A. A., Bolshakov, E. S., and et al. Comprehensive peripheral blood immunoprofiling reveals five immunotypes with immunotherapy response characteristics in patients with cancer. _Cancer Cell_ , 42(5): 759–779.e12, 2024. doi: 10.1016/j.ccell.2024.04.008. URL https://www.cell.com/cancer-cell/ fulltext/S1535-6108(24)00132-6. 

- Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. Revisiting deep learning models for tabular data. In _Advances in Neural Information Processing Systems_ , volume 34, pp. 18932–18943, 2021. 

- Grinsztajn, L., Oyallon, E., and Varoquaux, G. Why do tree-based models still outperform deep learning on tabular data? _Advances in Neural Information Processing Systems_ , 2022. 

- He, P., Cao, Z., Di, H., Shen, G., and Zhou, S. Application of machine learning in caisson inclination prediction: Model performance comparison and interpretability analysis. _Underground Space_ , 2025. URL https: //www.sciencedirect.com/science/ article/abs/pii/S2214391225001734. Includes TabPFN-based models among compared approaches. 

- Hendel, R., Geva, M., and Globerson, A. In-context learning creates task vectors, 2023. URL https://arxiv. org/abs/2310.15916. 

- Hollmann, N., Muller, S., Eggensperger, K., and Hutter, F.¨ Tabpfn: A transformer that solves small tabular classification problems in a second. In _International Conference on Learning Representations_ , 2023. 

- Hollmann, N., Muller, S., Purucker, L., Krishnakumar, A.,¨ Korfer,¨ M., Hoo, S. B., Schirrmeister, R. T., and Hutter, F. Accurate predictions on small data with a tabular foundation model. _Nature_ , 01 2025. doi: 10.1038/ s41586-024-08328-6. URL https://www.nature. com/articles/s41586-024-08328-6. 

- Karabacak, M., Schupper, A., Carr, M., and Margetis, K. A machine learning-based approach for individualized prediction of short-term outcomes after anterior cervical corpectomy. _Asian Spine Journal_ , 18(4):541–549, 2024. doi: 10.31616/asj.2024.0048. URL https://pmc.ncbi. nlm.nih.gov/articles/PMC11366553/. 

- Magadan, L., Rold´ an-G´ omez, J., Granda, J. C., and Su´ arez,´ F. J. Early fault classification in rotating machinery with limited data using TabPFN. _IEEE Sensors Journal_ , 23(24):30960–30970, 2023. doi: 10.1109/JSEN. 2023.3331100. URL https://ieeexplore.ieee. org/document/10318062. 

- Nguyen, H. Recovery rate prediction for corporate bonds — experiments. https://github.com/ hoanguyen94/Recovery-rate-prediction. GitHub repository; accessed 7 Nov 2025. 

- Noda, R., Ichikawa, D., and Shibagaki, Y. Machine learning-based diagnostic prediction of minimal change disease: Model development study. _Scientific Reports_ , 14:23460, 2024. doi: 10.1038/ s41598-024-73898-4. URL https://www.nature. com/articles/s41598-024-73898-4. 

- nostalgebraist. Interpreting GPT: the logit lens. https://www.lesswrong. com/posts/AcKRB8wDpdaN6v6ru/ interpreting-gpt-the-logit-lens, August 2020. Accessed: 2025-02-22. 

- Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., and Olah, C. In-context learning and induction heads, 2022. URL https:// arxiv.org/abs/2209.11895. 

- Park, K., Choe, Y. J., and Veitch, V. The linear representation hypothesis and the geometry of large language models, 2024. URL https://arxiv.org/abs/2311. 03658. 

- Pearl, J. _Causality_ . Cambridge University Press, 2 edition, 2009. 

7 

**TabPFN Through The Looking Glass** 

- Peters, J., Janzing, D., and Scholkopf, B.¨ _Elements of causal inference: foundations and learning algorithms_ . The MIT Press, 2017. 

- Rundel, D., Kobialka, J., von Crailsheim, C., Feurer, M., Nagler, T., and Rugamer,¨ D. _Interpretable Machine Learning for TabPFN_ , pp. 465–476. Springer Nature Switzerland, 2024. ISBN 9783031637971. doi: 10.1007/978-3-031-63797-1 ~~2~~ 3. URL http://dx. doi.org/10.1007/978-3-031-63797-1_23. 

- Sharma, S. Data and models for shape-selective adsorption in zeolites for long-chain alkane hydroisomerization. https://doi.org/10.4233/uuid: f36da034-5cb3-42ca-a53d-d351f68a9ffa, 2025. Repository associated with shape-selectivity modeling in zeolites; includes TabPFN-based components. 

- Sharma, S. et al. Machine learning-based predictions of henry coefficients for long-chain alkanes in onedimensional zeolites: Application to hydroisomerization. _The Journal of Physical Chemistry C_ , 2025. doi: 10.1021/ acs.jpcc.5c03868. URL https://pubs.acs.org/ doi/10.1021/acs.jpcc.5c03868. In press / early access; uses ML including TabPFN-style approaches for Henry coefficient prediction. 

- Todd, E., Li, M. L., Sharma, A. S., Mueller, A., Wallace, B. C., and Bau, D. Function vectors in large language models, 2024. URL https://arxiv.org/abs/ 2310.15213. 

- Wilinski, M., Goswami, M., Potosnak, W.,´ Zukowska, N.,<sup>˙</sup> and Dubrawski, A. Exploring representations and interventions in time series foundation models, 2025. URL https://arxiv.org/abs/2409.12915. 

- Ye, H.-J., Liu, S.-Y., and Chao, W.-L. A closer look at tabpfn v2: Understanding its strengths and extending its capabilities, 2025. URL https://arxiv.org/ abs/2502.17361. 

- Zhao, H., Chen, H., Yang, F., Liu, N., Deng, H., Cai, H., Wang, S., Yin, D., and Du, M. Explainability for large language models: A survey, 2023. URL https:// arxiv.org/abs/2309.01029. 

8 

