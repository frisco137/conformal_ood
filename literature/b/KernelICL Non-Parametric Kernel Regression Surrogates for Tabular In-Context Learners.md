**Interpretable Tabular Foundation Models via In-Context Kernel Regression** 

**Ratmir Miftachov**<sup>1 *</sup> **Bruno Charron**<sup>2</sup> **Simon Valentin**<sup>3</sup> 

# **Abstract** 

Tabular foundation models like TabPFN and TabICL achieve state-of-the-art performance through in-context learning, yet their architectures remain fundamentally opaque. We introduce KernelICL, a framework to enhance tabular foundation models with quantifiable samplebased interpretability. Building on the insight that in-context learning is akin to kernel regression, we make this mechanism explicit by replacing the final prediction layer with kernel functions (Gaussian, dot-product, kNN) so that every prediction is a transparent weighted average of training labels. We introduce a two-dimensional taxonomy that formally unifies standard kernel methods, modern neighbor-based approaches, and attention mechanisms under a single framework, and quantify inspectability via the perplexity of the weight distribution over training samples. On 55 TALENT benchmark datasets, KernelICL achieves performance on par with existing tabular foundation models, demonstrating that explicit kernel constraints on the final layer enable inspectable predictions without sacrificing performance. 

# **1. Introduction** 

Tabular data remains the backbone of decision-making across industries, from healthcare diagnostics to financial risk assessment. While foundation models have revolutionized natural language processing and computer vision through in-context learning (ICL), their application to tabular data has only recently gained traction. Models like TabPFN (Hollmann et al., 2022), TabPFNv2 (Hollmann et al., 2025), and TabICL (Qu et al., 2025) demonstrate that transformers can achieve state-of-the-art performance on tabular classification through ICL, predicting in a single forward pass without dataset-specific training or hyperparameter tuning. TabPFN-2.5 (Grinsztajn et al., 2025) matches 

> *Work completed during an internship at Amazon. 1HumboldtUniversitat zu Berlin¨<sup>2</sup> Amazon<sup>3</sup> AWS AI Labs. Correspondence to: Ratmir Miftachov _<_ contact@miftachov.com _>_ , Bruno Charron _<_ bcharro@amazon.fr _>_ . 

complex ensembles, establishing ICL as a strong paradigm for tabular learning. Recently, MITRA (Zhang et al., 2025) has used diverse mixtures of synthetic priors to increase generalization and sample efficiency. 

However, these models function as black boxes, limiting adoption in domains where decisions must be explainable. In healthcare, clinicians need to validate model reasoning against medical knowledge before acting on predictions (Rudin, 2019). In finance, regulatory frameworks increasingly demand transparency in automated decisionmaking (Bracke et al., 2019). Practitioners across domains require the ability to inspect which training cases inform a prediction and verify alignment with domain expertise. 

Recent work has begun addressing interpretability in tabular foundation models. GAMformer (Mueller et al., 2024) achieves _feature-level_ interpretability by learning additive decompositions where each feature’s contribution is represented as a univariate shape function. ModernNCA (Ye et al., 2025) achieves _sample-level_ interpretability through weighted averaging of training labels with weights determined by learned embeddings, though it requires datasetspecific training rather than leveraging pre-trained foundation models. Similarly, methods like ProtoAttend (Arik & Pfister, 2020) have utilized attention mechanisms to identify influential samples (or prototypes). SoftKNN-ICL (Koshil et al., 2025) similarly produces predictions as weighted averages of training labels using an attention mechanism in an in-context learning setting, though without a clear connection to classical kernel methods or interpretability measures. 

We introduce **KernelICL** , a framework to enhance tabular foundation models with quantifiable sample-based interpretability. Our key contributions are: (1) a two-dimensional taxonomy characterizing kernel regression methods by interpretability and dataset dependence; (2) efficient symmetric in-context embeddings; (3) systematic kernel function exploration, where symmetric embeddings enable distancebased kernels (Gaussian, kNN) previously unavailable in in-context learning; and (4) quantifiable inspectability control through perplexity measurement and kernel scale tuning, enabling practitioners to navigate an accuracy-inspectability tradeoff. Figure 1 illustrates our approach. 

_Preprint. February 3, 2026._ 

1 

**KernelICL** 



<!-- Start of picture text -->
Input Data: D, x<br>Standard<br>Kernel Reg.<br>Foundation Model<br>TabICL<br>Embeddings: hD ( x ) MLP<br>KernelICL<br>Kernel  KD<br>Gauss Dot kNN<br>Weights: wi<br>Predictions: y ˆ<br><!-- End of picture text -->

_Figure 1._ Three approaches to tabular prediction (see Section 3 for notation). **Standard Kernel Regression (dotted):** Kernel functions applied to inputs yield transparent predictions but lack learned representations. **TabICL or similar (dashed):** Foundation model learns powerful embeddings but uses an opaque MLP head. **KernelICL (ours, solid):** Combines both by fine-tuning the foundation model with explicit kernel form, producing transparent weighted averages with inspectable coefficients. 

# **2. Related Work** 

## **2.1. Tabular Foundation Models** 

Foundation models for tabular data have evolved rapidly, achieving remarkable performance through in-context learning. TabPFN (Hollmann et al., 2022) pioneered this direction by training a transformer on millions of synthetic datasets, enabling predictions on real datasets with up to 1,000 samples in under a second without hyperparameter tuning. TabPFNv2 (Hollmann et al., 2025) extended this to 10,000 samples through alternating column-wise and rowwise attention. TabPFN-2.5 (Grinsztajn et al., 2025) has extended scalability to 50,000 samples, achieving parity with complex four-hour tuned ensembles. 

While these models excel on small to medium datasets, TabICL (Qu et al., 2025) addresses scalability to large data through a two-stage architecture: distribution-aware columnwise embeddings followed by context-aware row-wise interaction produce fixed-dimensional representations, which are then processed by a transformer for efficient ICL. This design enables handling up to 500,000 samples while maintaining competitive performance. 

## **2.2. Interpretable Tabular Learning** 

The interpretability literature typically distinguishes between inherently interpretable models and post-hoc explanations of black-box predictions (Rudin, 2019). Post-hoc methods like SHAP (Lundberg & Lee, 2017) and LIME (Ribeiro 

et al., 2016) compute feature attributions, and while recent work has developed kernel-smoothing perspectives for such measures (Miftachov et al., 2025), these approaches may still be unfaithful to actual model reasoning (Rudin, 2019; Lipton, 2018). For high-stakes tabular decisions, models which are inherently interpretable by design are thus often preferable. 

Among other inherently interpretable approaches, generalized additive models (GAMs) decompose predictions as sums of univariate functions (Hastie & Tibshirani, 1986; Fan et al., 1998). GAMformer (Mueller et al., 2024) introduced the first foundation model approach to GAMs, using ICL to estimate shape functions in a single forward pass. A complementary approach provides example-based explanations through explicit dependence on training samples, where classical kNN (Cover & Hart, 1967) is the canonical example. This aligns with case-based reasoning common in domains like medicine and law, where practitioners validate decisions by comparing to past cases (Molnar, 2020). 

ModernNCA (Ye et al., 2025) combines kNN with deep learning, learning embeddings via a Neighbourhood Components Analysis objective with stochastic neighborhood sampling. It achieved state-of-the-art results but requires perdataset model training. TabR (Gorishniy et al., 2023) combines learned embeddings with kNN retrieval. SoftKNNICL (Koshil et al., 2025) operates in the ICL setting, using attention to produce weighted averages of training labels, but employs only a dot-product attention-like kernel. 

## **2.3. Kernel View of In-Context Learning** 

Recent theoretical work has established connections between attention mechanisms and kernel methods. Han et al. (2025) show that in-context learning predictions can be asymptotically approximated by kernel regression as the number of context examples increases, and empirically validate this by reconstructing large language model predictions from attention weights with high accuracy. Their framework explains several ICL phenomena, including why retrieving similar demonstrations improves performance. However, this kernel structure emerges implicitly rather than being enforced by design. 

The connection between attention and kernel regression is well-established. Standard scaled dot-product attention (Vaswani et al., 2017) can be viewed as Nadaraya-Watson estimation (Nadaraya, 1964; Watson, 1964) with an exponential kernel: _K_ ( _q, k_ ) = exp( _q_<sup>_⊤_</sup> _k/_<sup>_√_</sup> _dk_ ), where _q_ is the query, _k_ is the key, and _dk_ is the key dimension. This equivalence reveals that the exponential dot-product kernel underlying scaled dot-product attention is one choice among many possible kernels (Genton, 2001), each encoding different notions of similarity and locality. 

2 

**KernelICL** 

# **3. Methodology** 

Given a training dataset _D_ = _{_ ( _xi, yi_ ) _}_<sup>_n_</sup> _i_ =1<sup>where training</sup> samples have features _xi ∈X_ and labels _yi ∈Y_ , we aim to predict the label for a test sample _x ∈X_ . Without loss of generality, we focus on binary classification ( _Y_ = _{_ 0 _,_ 1 _}_ ) for clarity, though the approach extends naturally to multiclass and regression settings. 

## **3.1. Taxonomies for Kernel Regression** 

We characterize kernel regression methods along two dimensions that together determine their interpretability and adaptability. The first dimension (Section 3.1.1) captures how transparently predictions can be interpreted through training samples. The second (Section 3.1.2) characterizes how model components adapt to each dataset. Together, this taxonomy allows us to position existing methods and identify gaps. Table 1 applies this to position existing methods alongside the KernelICL variants introduced in this work. 

## 3.1.1. INTERPRETABILITY TAXONOMY 

Sample-based interpretability allows predictions to be understood through individual training examples, supporting case-based reasoning where practitioners validate decisions by examining similar past cases. This contrasts with featurebased interpretability (e.g., GAMformer (Mueller et al., 2024)), which decomposes predictions into feature contributions. We characterize the transparency of the prediction mechanism through progressively constrained forms. 

**Level 0: Opaque.** In the most general form, predictions are computed by an arbitrary function: 



While _fD_ may achieve strong performance, it provides no inherent structure for mechanistic understanding. Interpretability, if desired, must rely on post-hoc explanation methods. 

**Level 1: Mechanistic.** A first constraint, following Nadaraya (1964); Watson (1964), requires predictions to take the form of weighted averages: 



where _κD_ : _X × X →_ R<sup>+</sup> is a weighting function<sup>1</sup> that can depend on the dataset _D_ . The prediction is now an explicit linear combination of training labels with inspectable coefficients _wi_ . 

> 1 _κD_ need not satisfy standard kernel assumptions from nonparametric statistics (Hardle¨ , 1990), as we make no asymptotic consistency claims. 

To further characterize interpretability, we decompose the weighting function. Without loss of generality: 



where _qD_ : _X →Q_ embeds the test sample into a query space, and _kD_ : _X →K_ embeds training samples into a key space (each key retrieves an associated value _yi_ ). The kernel _KD_ : _Q×K →_ R<sup>+</sup> then computes a scalar similarity measure. This decomposition separates learned geometric transformations (the embeddings) from a scalar summary (the kernel). When the query and key spaces differ, geometric interpretation is precluded. 

**Level 2: Geometric.** A further constraint requires shared embedding functions: query and key spaces become unified. We denote this shared function by _hD_ , where _qD_ = _kD_ = _hD_ , mapping _X →H_ : 



Now both test and training samples inhabit a common embedding space _H_ , where kernel weights _wi ∝ KD_ ( _hD_ ( _x_ ) _, hD_ ( _xi_ )) have geometric meaning: they measure similarity between _x_ and _xi_ in the learned space. The weight distribution becomes a geometric snapshot from the test point’s perspective. 

**Level 3: Distance-Based.** The most constrained form uses an isotropic kernel (Genton, 2001), depending solely on Euclidean distance in embedding space: 



for some monotone decreasing function _FD_ : R<sup>+</sup> _→_ R<sup>+</sup> . Distance-based kernels provide a concrete spatial mental model for the abstract notion of similarity: samples are near or far, offering an intuitive handle for understanding the weight distribution. 

## 3.1.2. DATASET DEPENDENCE TAXONOMY 

Having established interpretability levels, we now characterize how model components adapt to each dataset _D_ , progressing from fixed methods to increasingly adaptive approaches. 

**Level 0: Fixed.** The simplest case uses fixed embeddings and a fixed kernel: 



where both _q_ and _k_ are fixed functions (e.g., identity or polynomial features) and the kernel _K_ has no parameters that depend on the data. This is the standard NadarayaWatson estimator. 

**Level 1: Scale Adaptation.** A first generalization allows a _scale_ (or _bandwidth_ ) parameter _γD_ to adapt to each dataset: 



3 

**KernelICL** 

|**Method**|**Interpretability Level**|**Dataset Dependence Level**|**Kernel**|**Key Characteristics**|
|---|---|---|---|---|
|Standard kNN|3|0|kNN|Identity_h_, fxed k|
|Gaussian Kernel Regression|3|0|Gaussian|Identity_h_, fxed_γ_|
|Adaptive kNN|3|1|kNN|Identity_h_, k via CV|
|Adaptive Gaussian Ker. Reg.|3|1|Gaussian|Identity_h_,_γD_ via CV|
|ModernNCA (Ye et al.,2025)|3|2|Gaussian|Per-dataset training,_hD_|
|SoftKNN-ICL (Koshil et al.,2025)|1|2|Dot-product|ICL,_qD_ =_kD_|
|**KernelICL-Dot**|2|2|Dot-product|ICL,_hD_ (symmetric)|
|**KernelICL-Gaussian**|3|2|Gaussian|ICL,_hD_ (symmetric)|
|**KernelICL-kNN**|3|2|kNN|ICL,_hD_ (symmetric)|



_Table 1._ Taxonomy of kernel regression methods showing progression from classical (fixed embeddings) to modern (learned embeddings) approaches. KernelICL (bold) achieves interpretable in-context learning and covers diverse kernel types. 

The embeddings _q_ and _k_ remain fixed, but a scale parameter _γD_ is tuned per dataset (e.g., via cross-validation). Different datasets have different intrinsic densities and scales; adapting _γD_ accounts for this without changing the underlying representation. 

**Level 2: Geometric Adaptation.** The next generalization uses embeddings that depend on the dataset: 



Embeddings may depend arbitrarily on _D_ , whether learned via in-context learning (foundation models) or per-dataset training (e.g., ModernNCA). 

**Level 3: Kernel Selection.** The most general case selects kernel structure from a rich function space: 



where _KD_<sup>_∗_is chosen per dataset from a parameterized fam-</sup> ily. Examples include kernel mixtures of different types (nikov, _KD_ =etc.)<sup>�</sup> _j_<sup>_α_</sup> (G<sup>_j_(</sup> onen¨<sup>_D_)</sup><sup>_Kj_</sup> &<sup>where</sup> Alpaydın<sup>_Kj_can be Gaussian, Epanech-</sup> , 2011), or learning multiple kernel shape parameters beyond a single scale (e.g. Silverman (1986)). Selection could be performed via crossvalidation. This flexibility comes at the cost of interpretability: the kernel function itself becomes complex rather than remaining a transparent similarity measure. 

## **3.2. KernelICL: Symmetric In-Context Kernel Regression** 

Symmetric embeddings ( _qD_ = _kD_ = _hD_ , Equation (4)) enable geometric interpretation where kernel weights measure similarity in a shared learned space. Classical methods (Table 1) and modern data-adaptive approaches like ModernNCA (Ye et al., 2025) employ symmetric embeddings, though the latter requires per-dataset training. In-context learning has been constrained to asymmetric embeddings: SoftKNN-ICL (Koshil et al., 2025) uses _qD_ = _kD_ due to attention’s inherent role distinction between context and 

query. We introduce **KernelICL** , which enables symmetric embeddings for interpretable in-context kernel regression with limited computational overhead while maintaining the benefits of pre-trained representations. 

Schematically, in-context embedding _E_ for a dataset with training samples _X_ train = _{xi}_<sup>_n_</sup> _i_ =1<sup>, training labels</sup><sup>_y_train=</sup> _{yi}_<sup>_n_</sup> _i_ =1<sup>and test samples</sup><sup>_X_test=</sup><sup>_{xj}m_</sup> _j_ =1<sup>operates as:</sup> 



where TF is a transformer with attention masking that prevents training samples from attending to test samples. Since training and test samples have distinct roles (context vs query), their embeddings differ even for identical inputs. To obtain symmetric embeddings, one would need to pass all training samples as both context and queries, effectively doubling the computational cost. 

Our approach leverages TabICL’s (Qu et al., 2025) threestage embedding architecture, which separates column-wise feature processing, row-wise sample interaction, and labelconditioned in-context learning: 



where _g_ encodes training labels. Crucially, TFcol processes each feature column via Set Transformers (Lee et al., 2019) with learnable inducing vectors that attend only to training samples, computing distributional statistics broadcast to all positions. This makes TFcol apply identical operations to all samples (train and test) for a given training set. Similarly, TFrow performs feature-wise self-attention within each row, again position-agnostic. Only TFicl introduces asymmetry via distinct attention masks for context versus query. 

To achieve symmetric embeddings, we reprocess training samples as additional queries through TFicl: 



4 

**KernelICL** 

where _∥_ denotes concatenation along the sample axis and the underscore indicates the discarded first output (context embeddings). Since TFcol and TFrow already apply identical operations, only TFicl requires reprocessing and incurs an overhead compared to typical asymmetric ICL embeddings. 

While TabICL passes its 512-dimensional test embeddings _E_ test to an MLP for predictions, we instead apply a learnable projection _W ∈_ R<sup>512</sup><sup>_×dk_</sup> to define embedding functions: 



for each training sample _xi_ and test sample _xj_ . These embeddings are passed to a kernel function _Kγ_ (explored in Section 3.3) to compute kernel weights. By construction, identical inputs produce identical projected embeddings regardless of whether they appeared in context or query positions, achieving _qD_ = _kD_ = _hD_ as required by Equation (4) for geometric interpretability. 

based kernels for in-context learning. The Gaussian kernel 



controls locality via _γ_ : large _γ_ concentrates weight on nearby samples, small _<u>γ</u>_ distributes weight uniformly. We use default _γ_ = 1 _/_ (2<sup>_√_</sup> _dk_ ); under unit-norm embeddings, this makes the Gaussian kernel equivalent to the dot-product kernel as _∥q − k∥_<sup>2</sup> = 2 _−_ 2 _q_<sup>_T_</sup> _k_ . Unlike the dot-product kernel, the Gaussian kernel is isotropic, depending only on Euclidean distance, offering intuitive spatial interpretation of similarity. 

**kNN Kernel.** With symmetric embeddings providing a shared geometric space, kNN becomes applicable to incontext learning. The kNN kernel is defined as: 



## **3.3. Kernel Function Exploration** 

The decomposition _κD_ ( _x, xi_ ) = _KD_ ( _hD_ ( _x_ ) _, hD_ ( _xi_ )) from Section 3.1.1 separates learned geometry (embedding _hD_ ) from similarity measurement (kernel _KD_ ). Using simple, single-parameter kernels for interpretable predictions concentrates representational complexity in the embedding function, while keeping the kernel operation transparent and inspectable. Among kernel families, distance-based kernels offer particularly intuitive notions of similarity: samples are near or far in the learned space. However, distance-based kernels require symmetric embeddings to have geometric meaning. KernelICL’s symmetric mode (Section 3.2) enables systematic exploration of distance-based kernels in the ICL setting. We examine three specifications of the kernel function _Kγ_ ( _q, k_ ) for _q ∈Q, k ∈K_ spanning the interpretability spectrum. 

**Dot-Product Kernel.** Standard transformer attention (Vaswani et al., 2017) uses the exponential dot-product kernel: 



where _γ_ is a scale parameter controlling the sharpness of the distribution. The default scale is _γ_ = 1 _/_<sup>_√_</sup> _dk_ , a standard choice in attention mechanisms to stabilize the variance (Vaswani et al., 2017). The dot-product kernel works in both asymmetric and symmetric modes, though only symmetric embeddings enable geometric interpretation where weights reflect similarity in a shared space. SoftKNN-ICL (Koshil et al., 2025) used this kernel asymmetrically for ICL. The dot-product kernel measures similarity through alignment rather than distance. 

**Gaussian Kernel.** Symmetric embeddings enable distance- 

where **_k_** = ( _k_ 1 _, . . . , kn_ ) denotes all training keys and _σγ_ ( _q,_ **_k_** ) is the _γ_ -th smallest distance among _{∥q − ki∥}_<sup>_n_</sup> _i_ =1<sup>.</sup> The scale _γ_ (or _k_ when there is no ambiguity with the key) represents the number of neighbors. Our kernel regression approach naturally extends to such vectorial kernels, though we use pairwise kernels _Kγ_ ( _q, ki_ ) in the exposition for simplicity. The kNN kernel’s binary weights provide maximum inspectability: predictions use exactly _γ_ samples, though at the cost of differentiability due to the sorting operation. 

## **3.4. Quantifying Inspectability** 

Figure 2 illustrates the KernelICL approach using a distancebased kernel on a synthetic dataset. Kernel weights concentrate on samples nearby in the learned embedding space (middle panel), with concentration controlled by the scale parameter _γ_ . This control matters because mechanistic interpretability alone does not ensure practical inspectability: with datasets containing thousands of samples, diffuse weight distributions become impractical for human examination. Practitioners can adjust _γ_ after training to achieve desired sparsity levels. To explore the resulting tradeoff between inspectability and performance, we quantify inspectability through the perplexity (PPL) of the weight vector: 



Lower perplexity indicates sparser weights, enabling easier inspection. For example, PPL( _w_ ) = 5 indicates a sparsity level similar to using 5 nearest neighbors. To enable comparison across datasets of different sizes, we use the _relative perplexity_ PPL( _w_ ) _/n_ for a given test point, and its geometric average over all test points at the dataset level. 

5 

**KernelICL** 



_Figure 2._ Illustration of the KernelICL approach with Gaussian kernel on a 2D synthetic dataset. **Left:** Input space _x_ showing concentric circles of different classes. **Middle:** 2D UMAP projection of 512D ICL embedding _hD_ ( _x_ ) showing class separation. **Right:** 1D “sample space”, x-axis representing the training samples sorted by first UMAP dimension. **Top Row:** Training samples colored by class with decision boundary in input space. **Other Rows:** Weight _wi_ (circle size for input and embedding space, height for sample space) of each training sample _i_ for 3 example test points (red crosses). Relative perplexity quantifies weight inspectability. 

# **4. Experiments** 

## **4.1. Training and Synthetic Validation** 

We fine-tune the TabICL embedding module and the projection matrix _W_ end-to-end with cross-entropy loss on the kernel predictions using 5,000 batches of synthetic data from TabICL’s prior distribution (64 datasets per batch). For the kNN kernel, which is non-differentiable due to neighbor selection, we use embeddings trained with the Gaussian kernel as they share the same distance-based structure. Note that there are also differentiable relaxations of top-k ranking operations (Swezey et al., 2021), which would enable end-to-end kNN training. Training details in Appendix A. 

To validate the benefits of learned embeddings, we test on controlled synthetic problems: Moons, Circles, and Linear datasets from Scikit-learn (Pedregosa et al., 2011) (200 

samples, 60/40 split). We append 18 Gaussian noise features to 2 signal features, yielding 20D inputs. Each kernel’s scale is calibrated via 5-fold cross-validation on training data. Standard kernels in input space degrade substantially with noise features, while KernelICL maintains clean boundaries by filtering noise in the learned embedding space (Figure 3). 

## **4.2. Benchmark Evaluation** 

We evaluate KernelICL’s accuracy and inspectability on the 55 binary classification datasets in the TALENT benchmark (Liu et al., 2024; Ye et al., 2024). These datasets span various domains and range from 3 to 970 features and from 645 to 109,099 samples. We use _dk_ = 512 as embedding dimension and perform 5-fold cross-validation on each dataset to select the scale _γD_ . Further details on the experimental setup, including the full list of datasets and baselines, are provided in Appendix C. 

|**Method**|**Mean**|**Rank**|**Mean Accuracy (%)**|
|---|---|---|---|
|TabICL (ensemble)||4.95|83.33|
|TabICL (single)||5.52|83.05|
|KernelICL-Gaussian||6.25|82.87|
|TabICL-MLP||6.39|82.91|
|KernelICL-Dot||6.49|82.88|
|KernelICL-kNN||6.75|82.79|



_Table 2._ Subset of benchmark results for methods using the TabICL embedding architecture. KernelICL variants achieve similar accuracy to TabICL-MLP while providing interpretable weights. 

Figure 4 presents statistical comparison across 14 methods on 55 TALENT datasets. The critical difference diagram indicates no statistically significant performance difference between KernelICL variants and TabICL or TabPFN variants. While TabICL (ensemble) achieves superior mean rank (4.95), all KernelICL variants (ranks 6.25-6.75) fall within the critical difference threshold (2.68), showing that explicit kernel constraints preserve competitive performance. Table 8 (Appendix) provides complete rankings. 

Table 2 examines methods sharing the TabICL embedding architecture. TabICL-MLP provides a controlled comparison: a similar architecture to TabICL (embedding + MLP) but fine-tuned with the KernelICL procedure. All three KernelICL variants match TabICL-MLP within 0.12 accuracy points, demonstrating that interpretability through explicit kernel form comes at negligible accuracy cost. 

## **4.3. Ablation Studies** 

KernelICL’s results in Section 4.2 use symmetric embeddings with _dk_ = 512 and calibrated kernel scales. We examine how those design choices impact accuracy, relative perplexity (inspectability), and runtime. Complete results with ablations are in Table 10 and Figure 7 (Appendix). 

6 



_Figure 3._ Decision boundaries on synthetic datasets with 18 added noise features. Standard kernels operate in input space; KernelICL kernels operate in learned embedding space and show noise robustness with test accuracies (bottom right) on par with TabICL. 



_Figure 4._ Critical difference diagram comparing 14 methods on 55 TALENT binary classification datasets. Methods connected by horizontal bars show no statistically significant difference in accuracy. KernelICL variants form a tight cluster with TabICL and TabPFN, demonstrating that explicit kernel constraints preserve competitive performance. 

**Effect of Calibration.** Table 3 compares cross-validated scale calibration against default values. Calibration improves accuracy for all kernels while reducing perplexity for Gaussian and Dot-product variants. For those soft kernels, the accuracy gain is not statistically significant and does not justify the 21-24x runtime overhead of calibration for most practical purposes. For the kNN variant, the default _k_ = 5 (following Scikit-learn) achieves high inspectability with 0.28% relative perplexity, while calibration significantly increases accuracy at the cost of an increase in perplexity to 11.89%, though sparser than soft kernels. The 50x runtime cost highly depends on the hyperparameter grid and an alternate view on scale calibration is presented in Section 4.4. 

**Effect of Symmetric Embeddings.** Section 3.2 introduced symmetric embeddings where test and training samples are projected identically ( _qD_ = _kD_ = _hD_ ). We compare against a non-symmetric variant with separate projections for queries and keys ( _qD_ = _kD_ ) and without duplication of training samples, following SoftKNN-ICL (Koshil et al., 2025). Table 4 shows symmetric embeddings consistently 

|**Method**|**Acc.** (%)|**Perp.** (%)|**Time**(s)|
|---|---|---|---|
|KernelICL-Gaussian|**82.87**|**28.63**|42.9|
|KernelICL-Gaussian (non-calibrated)|82.81|37.35|**2.0**|
|KernelICL-Dot|**82.88**|**28.81**|45.3|
|KernelICL-Dot (non-calibrated)|82.79|38.64|**1.9**|
|KernelICL-kNN|**82.79**|11.89|69.4|
|KernelICL-kNN (non-calibrated)|81.28|**0.28**|**1.4**|



_Table 3._ Effect of scale calibration on accuracy, relative perplexity, and runtime. Calibration uses 5-fold cross-validation, incurring computational overhead but improving both accuracy and perplexity. Best values per section and metric shown in bold. 

achieve higher accuracy and lower perplexity across all kernels, with only 9% to 16% runtime overhead. The dual performance and inspectability gains justify the computational cost. 

|**Method**|**Acc.** (%)|**Perp.** (%)|**Time**(s)|
|---|---|---|---|
|KernelICL-Gaussian|**82.87**|**28.63**|42.9|
|KernelICL-Gaussian (non-symmetric)|82.44|40.59|**39.3**|
|KernelICL-Dot|**82.88**|**28.81**|45.3|
|KernelICL-Dot (non-symmetric)|82.79|47.35|**39.2**|
|KernelICL-kNN|**82.79**|**11.89**|69.4|
|KernelICL-kNN (non-symmetric)|82.52|16.57|**60.4**|



_Table 4._ Effect of symmetric embeddings. Symmetric embeddings improve both accuracy and perplexity at minimal runtime overhead. Best values per section and metric shown in bold. 

To better understand the overhead from symmetric embeddings, we measure embedding time on synthetic datasets with varying number of samples and features. Figure 5 shows the overhead reaches 100% in the large training set limit, with slower convergence at large number of features. The limit corresponds to TFicl dominating compared to the column-wise and row-wise embedding stages which do not need sample duplication. The methodology for this analysis is detailed in Appendix E. 

7 

**KernelICL** 



_Figure 5._ Overhead of symmetric embeddings on the embedding time measured on synthetic datasets, approaching a 2x factor in the large sample limit due to duplication of the training samples as both context and queries. Setups running out of memory are skipped. 

**Effect of Projection Dimension.** Table 9 (Appendix) varies embedding dimension _dk_ from 16 to 512. We use _dk_ = 512 for the main results as it provides the best combination of high accuracy and low perplexity. 

## **4.4. Inspectability-Accuracy Trade-off** 

While cross-validated scale selection (Section 4.3) maximizes accuracy within the mechanistic framework, the resulting 11-29% relative perplexity may not provide adequate inspectability. Depending on the dataset size, 10% relative perplexity still corresponds to many samples being examined. KernelICL enables practitioners to adjust the kernel scale to achieve their desired sparsity level when inspecting predictions, accepting an accuracy cost for interpretability. 

Figure 6 shows the resulting trade-off. For a hyperparameter grid (bandwidth _γ_ or neighborhood size _k_ ), we measure accuracy and relative perplexity on the test set. The x-axis shows target relative perplexity levels; for each target, we select the hyperparameter with perplexity closest to (but not exceeding) the target, and report its accuracy averaged across the 55 datasets. At 100% perplexity, weights are uniform and methods converge to the baseline ( _∼_ 70%). 

At comparable sparsity, KernelICL-kNN achieves _∼_ 5 percentage points higher accuracy than standard kNN, confirming that learned embeddings are essential. To examine whether this improvement reflects meaningful similarity, we analyze neighbor selection on the Pima Indians Diabetes dataset. Measuring neighborhood compactness per feature, KernelICL-kNN shows tight neighborhoods on glucose (+61% vs standard kNN) and BMI (+35%), the primary risk factors in medical literature (DeFronzo et al., 2015), while standard kNN treats all features roughly equally. This suggests learned embeddings concentrate similarity along clinically relevant dimensions. Full analysis in Appendix B. 



_Figure 6._ Accuracy-sparsity trade-off for KernelICL variants and standard kNN. TabICL-MLP shown for reference. 

Below 10% perplexity, KernelICL-kNN achieves highest accuracy among KernelICL variants. Comparing soft kernels, Gaussian and Dot-product achieve similar peak accuracies (close to TabICL-MLP at 20 to 60% perplexity), but at lower perplexities (below 20%), Gaussian consistently outperforms Dot-product, indicating distance-based similarity suits sparse regimes better than alignment-based measures. 

# **5. Conclusion** 

We introduced KernelICL, a framework that replaces the prediction head of tabular foundation models with explicit kernel functions. Every prediction becomes a weighted average of training labels with inspectable coefficients. Our twodimensional taxonomy classifies kernel regression methods by interpretability and dataset dependence, enabling systematic comparison of their trade-offs. Enabled by symmetric embeddings that place test and training samples in a shared space, exploration of multiple kernel functions reveals that distance-based kernels suit sparse regimes while achieving competitive peak performance. Perplexity metrics quantify inspectability, enabling practitioners to explicitly control the accuracy-sparsity trade-off. 

On 55 TALENT datasets, KernelICL achieves 82.88% accuracy, within 0.2% of TabICL, while providing explicit sample weights. This demonstrates that inspectability through kernel structure comes at negligible accuracy cost. We hope this work encourages further research on interpretable prediction mechanisms for tabular foundation models. 

# **References** 

Arik, S. O. and Pfister, T. Protoattend: Attention-based prototypical learning. _Journal of Machine Learning Research_ , 21(210):1–35, 2020. 

Bracke, P., Datta, A., Jung, C., and Sen, S. Machine learning explainability in finance: an application to default risk 

8 

**KernelICL** 

analysis. 2019. 

- Cover, T. and Hart, P. Nearest neighbor pattern classification. _IEEE Transactions on Information Theory_ , 13(1):21–27, 1967. 

- DeFronzo, R. A., Ferrannini, E., Groop, L., Henry, R. R., Herman, W. H., Holst, J. J., Hu, F. B., Kahn, C. R., Raz, I., Shulman, G. I., et al. Type 2 diabetes mellitus. _Nature reviews Disease primers_ , 1(1):1–22, 2015. 

- Demsar, J.ˇ Statistical comparisons of classifiers over multiple data sets. _Journal of Machine learning research_ , 7 (Jan):1–30, 2006. 

- Fan, J., Hardle, W., and Mammen, E.¨ Direct estimation of low-dimensional components in additive models. _The Annals of Statistics_ , 26(3):943–971, 1998. 

- Garg, A., Ali, M., Hollmann, N., Purucker, L., Muller, S.,¨ and Hutter, F. Real-tabpfn: Improving tabular foundation models via continued pre-training with real-world data. _arXiv preprint arXiv:2507.03971_ , 2025. 

- Genton, M. G. Classes of kernels for machine learning: a statistics perspective. _Journal of Machine Learning Research_ , 2(Dec):299–312, 2001. 

- Gonen, M. and Alpaydın, E.¨ Multiple kernel learning algorithms. _The Journal of Machine Learning Research_ , 12: 2211–2268, 2011. 

- Gorishniy, Y., Rubachev, I., Kartashev, N., Shlenskii, D., Kotelnikov, A., and Babenko, A. Tabr: Tabular deep learning meets nearest neighbors in 2023. _arXiv preprint arXiv:2307.14338_ , 2023. 

- Grinsztajn, L., Floge, K., Key, O., Birkel, F., Jund, P., Roof,¨ B., Jager,¨ B., Safaric, D., Alessi, S., Hayler, A., et al. Tabpfn-2.5: Advancing the state of the art in tabular foundation models. _arXiv preprint arXiv:2511.08667_ , 2025. 

- Han, C., Wang, Z., Zhao, H., and Ji, H. Understanding emergent in-context learning from a kernel regression perspective. _Transactions on Machine Learning Research_ , 2025. 

- Hardle, W.¨ _Applied nonparametric regression_ . Number 19. Cambridge university press, 1990. 

- Hastie, T. and Tibshirani, R. Generalized additive models. _Statistical Science_ , 1(3):297–310, 1986. 

- Hollmann, N., Muller,¨ S., Eggensperger, K., and Hutter, F. Tabpfn: A transformer that solves small tabular classification problems in a second. _arXiv preprint arXiv:2207.01848_ , 2022. 

- Hollmann, N., Muller, S., Purucker, L., Krishnakumar, A.,¨ Korfer, M., Hoo, S. B., Schirrmeister, R. T., and Hutter,¨ F. Accurate predictions on small data with a tabular foundation model. _Nature_ , 637(8045):319–326, 2025. 

- Kahn, S. E., Hull, R. L., and Utzschneider, K. M. Mechanisms linking obesity to insulin resistance and type 2 diabetes. _Nature_ , 444(7121):840–846, 2006. 

- Koshil, M., Feurer, M., and Eggensperger, K. In-context learning of soft nearest neighbor classifiers for intelligible tabular machine learning. In _Proceedings of the 4th Table Representation Learning Workshop_ , pp. 182–191, 2025. 

- Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., and Teh, Y. W. Set transformer: A framework for attention-based permutation-invariant neural networks. In _International conference on machine learning_ , pp. 3744–3753. PMLR, 2019. 

- Lipton, Z. C. The mythos of model interpretability: In machine learning, the concept of interpretability is both important and slippery. _Queue_ , 16(3):31–57, 2018. 

- Liu, S.-Y., Cai, H.-R., Zhou, Q.-L., and Ye, H.-J. Talent: A tabular analytics and learning toolbox. _arXiv preprint arXiv:2407.04057_ , 2024. 

- Lundberg, S. M. and Lee, S.-I. A unified approach to interpreting model predictions. _Advances in Neural Information Processing Systems_ , 30, 2017. 

- Miftachov, R., Keilbar, G., and Hardle,¨ W. K. Shapley curves: A smoothing perspective. _Journal of Business & Economic Statistics_ , 43(2):312–323, 2025. 

- Molnar, C. _Interpretable machine learning_ . Lulu.com, 2020. 

- Mueller, A., Siems, J., Nori, H., Salinas, D., Zela, A., Caruana, R., and Hutter, F. Gamformer: In-context learning for generalized additive models. _arXiv preprint arXiv:2410.04560_ , 2024. 

- Nadaraya, E. A. On estimating regression. _Theory of Probability & Its Applications_ , 9(1):141–142, 1964. 

- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., et al. Scikit-learn: Machine learning in python. _Journal of Machine Learning Research_ , 12:2825–2830, 2011. 

- Qu, J., Holzmuller, D., Varoquaux, G., and Morvan, M. L.¨ Tabicl: A tabular foundation model for in-context learning on large data. _arXiv preprint arXiv:2502.05564_ , 2025. 

9 

**KernelICL** 

- Ribeiro, M. T., Singh, S., and Guestrin, C. ”why should i trust you?” explaining the predictions of any classifier. In _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ , pp. 1135–1144, 2016. 

- Rudin, C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. _Nature Machine Intelligence_ , 1(5):206– 215, 2019. 

- Silverman, B. W. _Density estimation for statistics and data analysis_ , volume 26. CRC press, 1986. 

- Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., and Johannes, R. S. Using the adap learning algorithm to forecast the onset of diabetes mellitus. In _Proceedings of the annual symposium on computer application in medical care_ , pp. 261, 1988. 

- Swezey, R., Grover, A., Charron, B., and Ermon, S. Pirank: Scalable learning to rank via differentiable sorting. _Advances in Neural Information Processing Systems_ , 34: 21644–21654, 2021. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. _Advances in Neural Information Processing Systems_ , 30, 2017. 

- Watson, G. S. Smooth regression analysis. _Sankhya:¯ The Indian Journal of Statistics, Series A_ , pp. 359–372, 1964. 

- Ye, H.-J., Liu, S.-Y., Cai, H.-R., Zhou, Q.-L., and Zhan, D.C. A closer look at deep learning on tabular data. _arXiv preprint arXiv:2407.00956_ , 2024. 

- Ye, H.-J., Yin, H.-H., Zhan, D.-C., and Chao, W.-L. Revisiting nearest neighbor for tabular data: A deep tabular baseline two decades later. In _The Thirteenth International Conference on Learning Representations_ , 2025. 

- Zhang, X., Maddix, D. C., Yin, J., Erickson, N., Ansari, A. F., Han, B., Zhang, S., Akoglu, L., Faloutsos, C., Mahoney, M. W., et al. Mitra: Mixed synthetic priors for enhancing tabular foundation models. _arXiv preprint arXiv:2510.21204_ , 2025. 

# **A. Training Details** 

We sample 5000 batches from TabICL’s synthetic data prior, where each batch contains 64 datasets generated from random MLP functions. The number of features ranges from 5 to 100, the sequence length contains up to 1024 observations in total, with the training data being between 60% and 80%. This requires approximately 40GB of GPU memory. For evaluation of the training process, we sample additional 32 batches from the same distribution and evaluate the loss function regularly on the validation data. Thus, we are using 32 _×_ 64 = 2048 datasets for validation. We choose the parameters corresponding to the smallest validation loss as the final model parameters. 

# **B. Case Study Details: Pima Indians Diabetes** 

Section 4.4 summarizes findings on the Pima Indians Diabetes dataset (Smith et al., 1988). Here we provide methodological details and complete results. 

We compare standard kNN (Euclidean distance in input space) with KernelICL-kNN (distance in learned embedding space) using _k_ = 5 neighbors on 154 test samples. KernelICL achieves 75.3% accuracy versus 68.8% for standard kNN. 

For each test point, we compute the mean distance to its _k_ neighbors along each feature in standardized space. Table 5 reports values normalized by method mean, enabling comparison of relative feature emphasis. Medical literature establishes glucose and BMI as primary causal drivers of diabetes, with age as a secondary contributor (DeFronzo et al., 2015; Kahn et al., 2006). 

_Table 5._ Neighborhood compactness on Pima Indians Diabetes (154 test samples, _k_ = 5). Values normalized by method mean; positive relative difference indicates KernelICL has tighter neighborhoods. 

|**Feature**|**Standard**|**KernelICL**|**Rel. Diff.**|
|---|---|---|---|
|Glucose|1.21|0.59|+61%|
|BMI|1.17|0.81|+35%|
|Age|0.98|0.82|+17%|
|BloodPressure|1.07|1.02|+6%|
|Pregnancies|1.00|1.08|_−_8%|
|DiabetesPedigree|1.16|1.39|_−_23%|
|Insulin|0.71|1.11|_−_41%|
|SkinThickness|0.71|1.18|_−_47%|



Standard kNN shows roughly isotropic behavior, while KernelICL concentrates on clinically established predictors. Whether this alignment reflects learned causal structure requires further clinical validation. 

10 

**KernelICL** 

# **C. Benchmark Details** 

**Experimental Setup.** We evaluate on 55 binary classification datasets from TALENT (Liu et al., 2024; Ye et al., 2024) using the standard 76%/24% train/test split. Table 6 lists the 55 datasets, which range from 645 to 109,099 samples and 3 to 970 features, spanning diverse domains. For KernelICL, we use symmetric mode with projection dimension _dk_ = 512 and perform 5-fold cross-validation on training data to select kernel scale _γD_ (calibration grids in Table 7). 

**Metrics.** We report three metrics averaged over the 55 datasets: (1) mean accuracy (arithmetic mean), (2) mean rank (Wilcoxon signed-rank) by accuracy, and (3) mean time (average inference time per dataset in seconds). For KernelICL variants, we also report relative perplexity (arithmetic mean over the datasets of the geometric mean of PPL( _w_ ) _/n_ across test samples within each dataset; defined in Section 3.4). 

**Baseline Methods.** We compare against a number of baselines provided by the TALENT benchmark library<sup>2</sup> : foundation models (TabICL (Qu et al., 2025), Real-TabPFN (Garg et al., 2025), TabPFN v2 (Hollmann et al., 2025), Mitra (Zhang et al., 2025)), ModernNCA (Ye et al., 2025), and traditional machine learning methods (CatBoost, XGBoost, RandomForest, LightGBM). All baselines use the defaults in the library. For ModernNCA, we use 20 epochs of training (no default). SoftKNN-ICL (Koshil et al., 2025) is not included as no public implementation is available; our KernelICL-Dot (non-symmetric, non-calibrated) configuration represents a similar approach with dot-product attention in asymmetric mode. For TabICL, the default is an ensemble method averaging predictions across 32 normalizations and feature/class shuffles. We add a non-ensemble (single) version for closer comparison with KernelICL which does not use ensembling for interpretability purposes. We also introduce a TabICL-MLP method using the same training procedure and architecture as KernelICL but replacing the projection _W_ and kernel with an MLP. TabICL-MLP is therefore very close to TabICL (single) in architecture but differs in its prior distribution due the effect of fine-tuning, thereby allowing to isolate the effects of the kernel regression head. 

# **D. Ablation Studies** 

Section 4.3 examined three design choices (scale calibration, symmetric embeddings, projection dimension) measuring their impact on accuracy, perplexity, and runtime. Table 9 provides the projection dimension sweep, Table 10 extends analysis to all 23 ablation configurations including baselines, and Figure 7 shows statistical significance. Together, these results demonstrate that symmetric embeddings and calibrated scales consistently improve performance across all kernels. 

# **E. Embedding Overhead** 

To isolate the computational overhead of symmetric embeddings from other inference components (I/O, preprocessing, cross-validation), we measure embedding time on synthetic datasets with controlled sizes. We generate binary classification datasets with _n ∈{_ 10<sup>3</sup> _,_ 2 _×_ 10<sup>3</sup> _,_ 5 _×_ 10<sup>3</sup> _,_ 10<sup>4</sup> _,_ 2 _×_ 10<sup>4</sup> _,_ 5 _×_ 10<sup>4</sup> _,_ 10<sup>5</sup> _,_ 2 _×_ 10<sup>5</sup> _}_ training samples, _m_ = 50 (fixed) test samples and _d ∈{_ 1 _,_ 5 _,_ 20 _,_ 100 _}_ features, measuring the ratio of embedding times (symmetric / non-symmetric) on GPU (H100, 80GB memory). We only consider the TabICL embedding module, not including the projection to the _dk_ - dimension final embedding space since that projection has no overhead in symmetric mode. 

Figure 5 shows the overhead ratio approaches 2× in the large sample limit, consistent with theory: symmetric mode processes training samples twice (once as context, once as query through TFicl). Convergence is slower with more features because column-wise and row-wise stages (TFcol, TFrow) do not require sample duplication and represent larger fractions of total embedding time. The 2× overhead applies only to TFicl, which becomes dominant for large _n_ . 

**Statistical Methodology.** We assess statistical significance using Friedman omnibus test followed by Nemenyi post-hoc test for pairwise comparisons (Demsarˇ , 2006). Mean ranks are computed via Wilcoxon signed-rank test across the 55 datasets. The critical difference (CD) threshold determines when two methods are statistically indistinguishable; methods within CD are connected by horizontal bars in the CD diagrams. 

> 2https://github.com/LAMDA-Tabular/TALENT 

11 

**KernelICL** 

_Table 6._ The 55 binary classification datasets from the TALENT benchmark (Ye et al., 2024) used for benchmarking. _N_ denotes the total sample size and _d_ denotes the number of features. 

|**Dataset**|_N_|_d_|**Train**|**Test**|**Dataset**|_N_|_d_|**Train**|**Test**|
|---|---|---|---|---|---|---|---|---|---|
|Pima Indians Diabetes|645|8|491|154|Ada Agnostic|3,832|48|2,919|913|
|Sports Articles (Obj.)|840|59|640|200|Employee|3,908|8|2,977|931|
|Statlog|840|20|640|200|Wilt|4,049|5|3,084|965|
|QSAR Biodegradation|885|41|674|211|Company Bankruptcy|5,728|95|4,364|1,364|
|Golf Play (Extended)|919|9|700|219|Taiwanese Bankruptcy|5,728|95|4,364|1,364|
|PC1|931|21|709|222|Water Quality|6,716|20|5,116|1,600|
|Diabetic Retinopathy|967|19|736|231|Bank Customer Churn|8,400|10|6,400|2,000|
|Basketball|1,125|11|857|268|JM1|9,143|21|6,966|2,177|
|Banknote Auth.|1,152|4|877|275|E-Commerce Shipping|9,239|10|7,039|2,200|
|PC4|1,224|37|932|292|<br>Online Shoppers|10,357|14|7,891|2,466|
|IBM HR Analytics|1,234|31|940|294|Coupon Recommend.|10,654|21|8,117|2,537|
|<br>Fitness Club|1,260|6|960|300|<br>HTRU2|15,034|8|11,454|3,580|
|PC3|1,313|37|1,000|313|HR Analytics|16,092|13|12,260|3,832|
|Forex (AUD/JPY Day)|1,539|10|1,172|367|California Housing|17,337|8|13,209|4,128|
|Forex (AUD/CHF Day)|1,539|10|1,172|367|Android Permissions|24,639|86|18,772|5,867|
|Forex (AUD/CAD Day)|1,540|10|1,173|367|Default Credit Card|25,200|23|19,200|6,000|
|<br>Forex (CAD/JPY Day)|1,540|10|1,173|367|INN Hotels Group|30,471|17|23,216|7,255|
|KC1|1,771|21|1,349|422|Click Prediction (S)|33,556|3|25,566|7,990|
|Customer Personality|1,881|24|1,433|448|Forex (AUD/CAD Hour)|36,813|10|28,048|8,765|
|Marketing Campaign|1,881|27|1,433|448|Forex (AUD/JPY Hour)|36,813|10|28,048|8,765|
|NHANES|1,913|7|1,457|456|Forex (AUD/SGD Hour)|36,813|10|28,048|8,765|
|Pumpkin Seeds|2,100|12|1,600|500|Forex (AUD/USD Hour)|36,813|10|28,048|8,765|
|Wine|2,145|4|1,634|511|Forex (CAD/JPY Hour)|36,813|10|28,048|8,765|
|Seismic Bumps|2,170|18|1,653|517|Bank Marketing|37,977|16|28,934|9,043|
|<br>Water Quality (Pot.)|2,752|8|2,096|656|<br>Mobile C36|43,478|6|33,126|10,352|
|Telecom Churn|2,799|17|2,132|667|Diabetes (130-US)|85,483|20|65,129|20,354|
|Gina Agnostic|2,913|970|2,219|694|Airline Satisfaction|109,099|21|83,123|25,976|
|Rice Cammeo|3,200|7|2,438|762||||||





_Figure 7._ Critical difference diagram including KernelICL ablation configurations. Methods sharing horizontal bars show no statistically significant difference. 

12 

**KernelICL** 

|**Method**|**Hyperparameter Grid**|
|---|---|
|KernelICL-Gaussian|_γ ∈{_0_._01_,_0_._05_,_0_._1_,_0_._3_,_0_._5_,_0_._8_,_1_,_3_/_2_}_|
|KernelICL-Dot|_γ ∈{_1_/_<br>_√_<br>4_,_1_/_<br>_√_<br>8_,_1_/_<br>_√_<br>16_,_1_/_<br>_√_<br>32_,_<br><br><br><br>|
||1_/_<br>_√_<br>64_,_1_/_<br>_√_<br>128_,_1_/_<br>_√_<br>256_,_1_/_<br>_√_<br>512_}_|
|KernelICL-kNN|_k ∈{_1_,_4_,_5_,_16_,_32_,_64_,_128_,_256_,_<br>512_,_1024_,_2048_,_4096_,_8192_}_|



_Table 7._ Scale calibration grids used for each KernelICL variant. 

|**Method**|**Rank**|**Accuracy (%)**|**Time (s)**|
|---|---|---|---|
|TabICL (ensemble)|4.95|83.33|2.9|
|Real-TabPFN|5.45|83.28|3.0|
|TabICL (single)|5.52|83.05|0.6|
|TabPFN v2|6.10|83.16|2.9|
|KernelICL-Gaussian|6.25|82.87|42.9|
|TabICL-MLP|6.39|82.91|1.3|
|KernelICL-Dot|6.49|82.88|45.3|
|KernelICL-kNN|6.75|82.79|69.4|
|CatBoost|8.46|81.47|26.2|
|Mitra|8.61|82.31|5.3|
|XGBoost|8.95|80.99|0.2|
|RandomForest|9.57|80.70|32.6|
|ModernNCA|10.65|81.48|25.1|
|LightGBM|10.86|80.54|1.2|



_Table 8._ Complete benchmark comparison on 55 TALENT datasets including foundation models, KernelICL variants, and traditional baselines. Metrics are arithmetic means over the datasets. Critical difference threshold on mean ranks: 2.68. 

|**Method**|**Accuracy (%)**|**Rel. Perp. (%)**|
|---|---|---|
|KernelICL-Gaussian (_dk_ = 512)|82.87|**28.63**|
|KernelICL-Gaussian (_dk_ = 256)|82.88|32.00|
|KernelICL-Gaussian (_dk_ = 128)|**82.91**|35.20|
|KernelICL-Gaussian (_dk_ = 64)|82.89|40.83|
|KernelICL-Gaussian (_dk_ = 32)|82.68|37.18|
|KernelICL-Gaussian (_dk_ = 16)|82.87|37.50|
|KernelICL-Dot (_dk_ = 512)|**82.88**|**28.81**|
|KernelICL-Dot (_dk_ = 256)|82.78|35.39|
|KernelICL-Dot (_dk_ = 128)|82.70|38.86|
|KernelICL-Dot (_dk_ = 64)|82.79|40.63|
|KernelICL-Dot (_dk_ = 32)|82.63|57.61|
|KernelICL-Dot (_dk_ = 16)|82.79|60.12|



_Table 9._ Effect of projection dimension _dk_ on symmetric KernelICL with calibrated scale. Highest accuracy and lowest perplexity per section are shown in bold. 

13 

**KernelICL** 

|**Method**|**Mean Rank**|**Mean Accuracy (%)**|**Mean Time (s)**|**Rel. Perp. (%)**|
|---|---|---|---|---|
|TabICL (ensemble)|7.25|83.33|2.9|-|
|TabICL (single)|8.03|83.05|0.6|-|
|Real-TabPFN|8.56|83.28|3.0|-|
|TabICL-MLP|9.27|82.91|1.3|-|
|KernelICL-Gaussian|9.27|82.87|42.9|28.63|
|TabPFN v2|9.44|83.16|2.9|-|
|KernelICL-Dot|9.75|82.88|45.3|28.81|
|KernelICL-kNN|10.11|82.79|69.4|11.89|
|KernelICL-Dot (non-symmetric)|10.29|82.79|39.2|47.35|
|KernelICL-Dot (non-calibrated)|10.41|82.79|1.9|38.64|
|KernelICL-Gaussian (non-calibrated)|10.65|82.81|2.0|37.35|
|KernelICL-Dot (non-symmetric, non-calibrated)|11.06|82.69|1.7|61.09|
|KernelICL-kNN (non-symmetric)|11.15|82.52|60.4|16.57|
|KernelICL-Gaussian (non-symmetric)|11.44|82.44|39.3|40.59|
|KernelICL-Gaussian (non-symmetric, non-calibrated)|12.05|82.50|1.8|51.43|
|Mitra|13.24|82.31|5.3|-|
|CatBoost|13.57|81.47|26.2|-|
|XGBoost|14.15|80.99|0.2|-|
|RandomForest|15.48|80.70|32.6|-|
|KernelICL-kNN (non-calibrated)|16.76|81.28|1.4|0.28|
|ModernNCA|16.92|81.48|25.1|-|
|LightGBM|17.61|80.54|1.2|-|
|KernelICL-kNN (non-symmetric, non-calibrated)|19.54|78.96|1.0|0.28|



_Table 10._ Comprehensive ablation analysis on 55 TALENT datasets comparing 23 configurations: KernelICL variants (symmetric/nonsymmetric embeddings, calibrated/non-calibrated scales), foundation models, and traditional baselines. Symmetric+calibrated KernelICL variants (ranks 9-10) cluster near top. Critical difference threshold on mean ranks: 4.68. Relative perplexity shown for KernelICL variants. 

14 

