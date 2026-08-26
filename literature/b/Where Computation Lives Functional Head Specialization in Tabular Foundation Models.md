# **Where Computation Lives Inside TabPFN: Causal Localisation of Attention Head Function** 

**Atharva Gupta**<sup>1</sup> **Dhruv Kumar**<sup>1</sup> **Murari Mandal**<sup>2</sup> **Saurabh Deshpande**<sup>3</sup> 

## **Abstract** 

We present the first causal mechanistic analysis of a tabular foundation model, investigating how TabPFN-2.5’s feature-wise attention heads distribute computation across layers. Using activation patching, ablation, and attention entropy across two synthetic regression datasets, we find clear temporal specialisation: one head’s causal necessity dominates that of the others by 2–5× at peak layer, with its dominant layer shifting across tasks of different complexity, while the remaining heads exhibit symmetric late-layer profiles. Attention entropy and patching provide convergent evidence for the computationally active layers of the dominant head. We additionally investigate inference-time steerability via contrastive activation steering, which fails to transfer across samples. We attribute this result to TabPFN’s in-context learning mechanism, which encodes task structure through context-dependent attention rather than the stable parametric directions that make steering tractable in language models. 

## **1. Introduction** 

Trained on synthetic structural causal models (Peters et al., 2017; Pearl, 2009), TabPFN-2.5 (Grinsztajn et al., 2025) acquires strong predictive capabilities that transfer to new tabular tasks through in-context learning. Despite continued scaling of context size (Hollmann et al., 2025) and sustained competition from tree-based methods (Grinsztajn et al., 2022; Gorishniy et al., 2021), mechanistic understanding of how TabPFN produces its predictions has lagged 

Code available at https://github.com/atharva7-g/ tabfm-interp.<sup>1</sup> Department of Computer Science, Birla Institute of Technology and Science, Pilani, India<sup>2</sup> School of Computer Engineering, Kalinga Institute of Industrial Technology, Bhubaneswar, India<sup>3</sup> Birla AI Labs, Office of Ananya Birla, Aditya Birla Group, India. Correspondence to: Atharva Gupta _<_ f20240519@pilani.bits-pilani.ac.in _>_ . 

_Proceedings of the 2_<sup>_nd_</sup> _ICML Workshop on Foundation Models for Structured Data_ , Seoul, South Korea. 2026. Copyright 2026 by the author(s). 

behind: existing interpretability work is confined to posthoc attribution (Grinsztajn et al., 2025; Rundel et al., 2024). A recent exception is Knauer & Rodner (2026), who find correlational evidence that individual neurons exhibit selective responses to high-level concepts, motivating a causal investigation of where and how. 

Mechanistic interpretability has made progress in language models (Olsson et al., 2022; Elhage et al., 2021; Todd et al., 2024) and time-series transformers (Wilinski et al.´ , 2025), but no comparable causal analysis exists for tabular architectures. 

We address this gap with a causal mechanistic study grounded in a concrete question: **Which components of TabPFN-2.5 are causally responsible for specific computations, and at which layers do they emerge?** Using causal activation patching (Meng et al., 2023; Heimersheim & Nanda, 2024) and ablation across two synthetic datasets, we identify two functional head classes in TabPFN-2.5’s feature-wise self-attention: one head whose ablation effect dominates the others on both datasets (with the peak layer differing across tasks), and two heads exhibiting symmetric patching and ablation profiles at late layers on the simpler task. We additionally investigate inference-time steerability via contrastive activation steering (Turner et al., 2023; Panickssery et al., 2023); directions do not transfer to heldout samples, a result we attribute to TabPFN’s relational ICL mechanism (Appendix G). To the best of our knowledge, this is the first causal mechanistic analysis of a tabular foundation model. 

## **2. Experiments and Results** 

### **2.1. Preliminaries** 

TabPFN-2.5 takes a labelled training set and a test instance as input and predicts the test label in a single forward pass via in-context learning (Hollmann et al., 2025; Grinsztajn et al., 2025). Full experimental hyperparameters are listed in Appendix A. The model applies two sequential self-attention mechanisms per layer: self ~~a~~ ttn ~~b~~ etween ~~i~~ tems across the sample dimension and self ~~a~~ ttn ~~b~~ etween ~~f~~ eatures across the feature-block dimension within each sample. We fo- 

1 

**Where Computation Lives Inside TabPFN** 



<!-- Start of picture text -->
Activation Patching<br>Layer Patching Component Patching<br>Residual Stream Feature-Block Level Attention Head Level<br>at Layer  ℓ (module output) (pre-output projection)<br>Token-Level Blocks  a, b, c Individual<br>(sequence position) + Label Heads  h ∗ ∈{ 0 ,  1 ,  2 }<br><!-- End of picture text -->

_Figure 1._ Activation patching hierarchy. Component patching targets self ~~a~~ ttn ~~b~~ etween ~~f~~ eatures at two granularities: feature-block level (post-projection output) and attention head level (per-head outputs before _WO_ ; see Appendix C). Token-level patching results are in Appendix F. 

cus on self ~~a~~ ttn ~~b~~ etween ~~f~~ eatures: it is the only module that operates across feature representations, making it the natural locus for cross-feature computation in regression tasks. We use the regression model (TabPFNRegressor) (Grinsztajn et al., 2025; Prior Labs, 2025a), which has 18 transformer layers, _H_ =3 attention heads with _dh_ =64, and _d_ model=192 (Prior Labs, 2025b). Per-dataset feature block counts and their derivation are in Appendix D (Table 5). The label _yi_ is embedded as the final token. 

### **2.2. Datasets** 

**Multiplication Dataset.** Each sample has three features _a, b, c ∼N_ (0 _,_ 1) with target _y_ = _a · b_ + _c_ , combining a nonlinear interaction ( _a · b_ ) with an additive term ( _c_ ). The low dimensionality ( _d_ = 3) allows direct inspection of individual feature contributions. 

**Pairwise-50 Dataset.** Each sample has _d_ = 50 features _x_ 1 _, . . . , x_ 50 _∼N_ (0 _,_ 1) with target 



The 50 _×_ 50 = 2 _,_ 500 terms cover all ordered pairs and self-products. 

### **2.3. Causal Mechanistic Analysis** 

### 2.3.1. ACTIVATION PATCHING 

Activation patching (Meng et al., 2023; Heimersheim & Nanda, 2024) tests causal responsibility by replacing a hidden activation in a corrupted forward pass with the corresponding value from a clean run, then measuring output recovery (formal setup and metrics in Appendix B.1). We sweep over layers and intervention sites as summarized in Figure 1. 

Layer-level patching — replacing the full residual stream at depth _ℓ_ with the clean-run value — achieves _≈_ 100% pre- 

diction recovery at every layer (Appendix B.1), consistent with a highly distributed representation in which information sufficient for correct prediction is preserved throughout the network. 

### 2.3.2. COMPONENT-LEVEL PATCHING 

The feature-wise self-attention module applies standard multi-head attention (see Appendix C) across the _⌈_ (2 _d_ + 1) _/_ 3 _⌉_ + 1 feature blocks per sample (see Appendix D). We focus on attention head level interventions; feature-block patching and ablation results are deferred to Appendix B.3 and B.4. 

**Attention head level.** We patch individual heads before _WO_ (the output projection; defined in Appendix C), at the point where theˆ _H_ per-head outputs are concatenated. Let _Vℓ ∈_ R<sup>_B·N×F ×H×d_</sup> _h_ denote the pre-projection per-head outputs, where _F_ = _⌈d_<sup>_′_</sup> _/_ 3 _⌉_ + 1. 

Patching head _h_<sup>_∗_</sup> replaces _V_<sup>ˆ</sup> _ℓ_ [: _,_ : _, h_<sup>_∗_</sup> _,_ :] = _V_<sup>ˆ</sup> _ℓ_<sup>clean</sup> [: _,_ : _, h_<sup>_∗_</sup> _,_ :]. The patched tensor is then passed through _WO_ , isolating each head’s individual causal contribution. 

### **2.4. Attention Head Patching and Ablation** 

**Multiplication Dataset.** Under mean ~~s~~ hift corruption, all three heads achieve comparable patching restoration: Head 0 peaks at layer 13 (0.228 _σ_ ), Head 1 at layer 12 (0.196 _σ_ ), and Head 2 at layer 6 (0.228 _σ_ ); corruption parameters are in Appendix B.3. Ablation reveals a sharp asymmetry (Figure 2): Head 2 peaks at layer 0 at 0.076 _σ_ , roughly 5 _×_ larger than any other head-layer combination, while Heads 0 and 1 show small effects of 0.015 _σ_ and 0.016 _σ_ at layers 12–13. 



_Figure 2._ MHA attention head ablation, Multiplication Dataset ( _n_ = 512). Head 2 ablation is largest at layer 0; Heads 0 and 1 peak at layers 12–13. 

**Head 2’s distinctive ablation profile.** Head 2’s ablation peaks at L0 while its patching peaks at L6: the layer of 

2 

**Where Computation Lives Inside TabPFN** 

greatest _necessity_ differs from the layer of greatest _restorability_ . Its attention at L0 is concentrated (entropy 0.22/0.24 on Mult./Pair-50), matching Head 0 (0.22 on both) — yet only Head 2 shows a large ablation effect there, confirming that selective attention is not sufficient for causal necessity. 

**Attention entropy confirms targeted computation.** Figure 3 shows normalised attention entropy across both datasets (Appendix E); Head 2 has the lowest entropy at L6 (0.21 on both datasets) and at L13 on Pairwise-50 (0.31), coinciding with its largest patching deviations. Heads 0 and 1 are broadly distributed (entropy _>_ 0.6) at most layers; Head 0 is co-selective with Head 2 at L0 (0.22 on both datasets) yet has near-zero ablation effect there. 



_Figure 3._ Normalised attention entropy per head at five key layers, computed per sample then averaged (see Appendix E). Lower entropy indicates more concentrated attention. Head 2 has the lowest entropy at layer 6 in both datasets (0.21 on both) and additionally at layer 13 on Pairwise-50 (0.31). Heads 0 and 1 maintain higher entropy ( _>_ 0.6) at most layers. Head 0 at layer 0 is co-selective with Head 2 (entropy 0.22 on both datasets) but has near-zero ablation effect there, demonstrating that attentional selectivity does not imply causal necessity. 

**Heads 0 and 1 as late computation heads.** Heads 0 and 1 exhibit symmetric patching and ablation profiles peaking at layers 12–13 on Multiplication, indicating that their contributions are both necessary and restorable at those depths; a mechanistic interpretation is in Appendix B.3.<sup>1</sup> 

**Results: Pairwise-50 Dataset.** Table 1 summarises ablation and entropy results; we use mean ~~s~~ hift corruption throughout (details and corruption-mode analysis in Appendix B.3). Figure 4 shows patching results: Head 0 achieves 13.2% gap recovery at layer 5 (positive signed); Heads 1 and 2 reach their largest _unsigned_ deviations at layer 17 (25.5%) and layer 13 (18.5%), both negative signed. 

**Head 2’s peak ablation magnitude is consistent across tasks; the peak layer is not.** Head 2 peaks at 0.076 _σ_ (L0, Multiplication) and 0.074 _σ_ (L16, Pairwise-50): consistent magnitude but substantially different depth, tentatively attributed to task complexity; two datasets cannot establish this as a general property. Head 1 at layer 17 and Head 2 

> 1Head 2’s layer-0 ablation effect is stable across sample sizes (0.078 _σ_ at _n_ =64 versus 0.076 _σ_ at _n_ =512), confirming this peak is not a single-batch artefact. 

_Table 1._ Head ablation effects ( _σ_ ) and minimum attention entropy across both datasets. Subscripts denote layer. Patching results (which depend on corruption mode and gap magnitude) are reported in the body text and Figures 4 and 5. 

||Ablati|on (_σ_)|Entro|py min|
|---|---|---|---|---|
|Head|Mult.|Pair-50|Mult.|Pair-50|
|H0|0.01513|0.02315|0.220|0.220|
|H1|0.01612|0.03517|0.610|0.636|
|**H2**|**0.076**0|**0.074**16|**0.21**6|**0.21**6|





_Figure 4._ MHA head patching recovery ratio (signed), Pairwise50, mean ~~s~~ hift (features 0–9, strength 3 _._ 0, _n_ = 512). Head 0 achieves 13.2% positive recovery at layer 5. Heads 1 and 2 reach their largest absolute deviations at layer 17 ( _|−_ 25 _._ 5% _|_ , unsigned) and layer 13 ( _|−_ 18 _._ 5% _|_ , unsigned); these are negative-signed recoveries — see body text. Left panel (absolute restoration) omitted as a constant rescaling of the ratio. 

at layer 13 show negative signed recovery despite large unsigned deviations: substituting clean activations disrupts rather than restores the corrupted computation, consistent with forward-pass divergence by those depths; Head 2’s L13 entropy minimum still identifies L13 as computationally active. The same interference pattern is observed for the label block on Multiplication. 

## **3. Discussion** 

**Causal localization.** Head 2’s peak ablation magnitude is consistent across tasks (0.074–0.076 _σ_ ) but its peak layer shifts from L0 to L16; entropy minima and patching deviations converge on the same computationally active layers on each task. Heads 0 and 1 show symmetric late-layer profiles on Multiplication; on Pairwise-50, Head 1 retains this symmetry but Head 0 does not (patching peaks at L5, ablation at L15) — whether these asymmetries scale with task complexity cannot be established from two datasets. 

**Steerability.** Activation steering experiments (Appendix G) show that contrastive directions do not generalise 

3 

**Where Computation Lives Inside TabPFN** 



_Figure 5._ MHA head ablation effect ( _σ_ ), Pairwise-50 ( _n_ = 512). Head 2’s peak is at layer 16 (0.074 _σ_ ), with a secondary peak at layer 0 (0.066 _σ_ ). Heads 0 and 1 show moderate late-layer effects. 

across samples: a direction computed on a held-out train split produces near-zero MSE improvement on a test split across all hook sites tested. We attribute this to a structural property of pure ICL architectures: unlike LLMs, where few-shot ICL is mediated by function vector heads that produce stable, transferable task representations (Todd et al., 2024; Yin & Steinhardt, 2025; Hendel et al., 2023), TabPFN encodes task relationships entirely through context-dependent attention, leaving no injectable task direction. 

**Limitations and future work.** The primary limitation is scope: two synthetic datasets are insufficient to establish the observed head classes as general properties of TabPFN-2.5. The closest cross-dataset evidence is Head 2’s peak ablation magnitude (0.074–0.076 _σ_ on both datasets), but the peak layer differs (L0 vs L16); whether this reflects a genuine task-complexity-dependent depth shift or an artefact of these particular tasks requires extending to a broader family of synthetic functions — sinusoidal, polynomial, exponential. Direct attention weight visualisation at L6 and L13 would reveal _which_ feature-block pairs Head 2 attends to at its entropy minima, sharpening the mechanistic account beyond the per-layer selectivity evidence we report. Extending to real-world datasets and classification tasks is an important longer-term direction. 

## **4. Conclusion** 

We presented a causal mechanistic analysis of TabPFN-2.5’s feature-wise attention module across two synthetic datasets. Activation ablation at the attention head level identifies two functional head classes: Head 2, whose ablation effect is the largest of any head on both datasets (0.076 _σ_ at L0 on Multiplication, 0.074 _σ_ at L16 on Pairwise-50), and Heads 0 and 1, which show symmetric patching and ablation pro- 

files at late layers on Multiplication and a partial version of this pattern on Pairwise-50. Head 2’s peak ablation _magnitude_ is comparable across tasks differing eight-fold in dimensionality, but the peak _layer_ differs substantially (L0 to L16); we tentatively attribute this to task complexity, but two datasets cannot establish the depth shift as a general property. Patching under mean ~~s~~ hift corruption and attention entropy minima provide converging evidence that L6 (Multiplication) and L13 (Pairwise-50) are layers where Head 2 performs targeted, selective operations. 

On steerability, contrastive activation steering fails to transfer to held-out samples — a result we attribute to a structural property of pure ICL architectures: TabPFN encodes task relationships through context-dependent attention compositions rather than the fixed, extractable task vectors that make steering tractable in LLMs (Todd et al., 2024; Yin & Steinhardt, 2025). This is a preliminary finding on a single task, but it identifies a meaningful architectural boundary for steering-based interpretability methods. 

## **References** 

- Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., and Olah, C. A mathematical framework for transformer circuits. _Transformer Circuits Thread_ , 2021. URL https://transformer-circuits. pub/2021/framework/index.html. 

- Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. Revisiting deep learning models for tabular data. In _Advances in Neural Information Processing Systems_ , volume 34, pp. 18932–18943, 2021. 

- Grinsztajn, L., Oyallon, E., and Varoquaux, G. Why do tree-based models still outperform deep learning on tabular data? _Advances in Neural Information Processing Systems_ , 2022. 

- Grinsztajn, L., Floge, K., Key, O., Birkel, F., Jund, P., Roof,¨ B., Jager, B., Safaric, D., Alessi, S., Hayler, A., Manium,¨ M., Yu, R., Jablonski, F., Hoo, S. B., Garg, A., Robertson, J., Buhler,¨ M., Moroshan, V., Purucker, L., Cornu, C., Wehrhahn, L. C., Bonetto, A., Scholkopf, B., Gambhir,¨ S., Hollmann, N., and Hutter, F. TabPFN-2.5: Advancing the state of the art in tabular foundation models, 2025. URL https://arxiv.org/abs/2511.08667. 

- Heimersheim, S. and Nanda, N. How to use and interpret activation patching, 2024. URL https://arxiv.org/ abs/2404.15255. 

4 

**Where Computation Lives Inside TabPFN** 

- Hendel, R., Geva, M., and Globerson, A. In-context learning creates task vectors, 2023. URL https://arxiv. org/abs/2310.15916. 

- Hollmann, N., Muller, S., Purucker, L., Krishnakumar, A.,¨ Korfer,¨ M., Hoo, S. B., Schirrmeister, R. T., and Hutter, F. Accurate predictions on small data with a tabular foundation model. _Nature_ , 01 2025. doi: 10.1038/ s41586-024-08328-6. URL https://www.nature. com/articles/s41586-024-08328-6. 

- Knauer, R. and Rodner, E. In search of grandmother cells: Tracing interpretable neurons in tabular representations, 2026. URL https://arxiv.org/abs/ 2601.03657. 

- Meng, K., Bau, D., Andonian, A., and Belinkov, Y. Locating and editing factual associations in gpt, 2023. URL https://arxiv.org/abs/2202.05262. 

- Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., and Olah, C. In-context learning and induction heads, 2022. URL https:// arxiv.org/abs/2209.11895. 

- Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., and Turner, A. M. Steering Llama 2 via contrastive activation addition, 2023. URL https: //arxiv.org/abs/2312.06681. 

   - Rundel, D., Kobialka, J., von Crailsheim, C., Feurer, M., Nagler, T., and Rugamer,¨ D. _Interpretable Machine Learning for TabPFN_ , pp. 465–476. Springer Nature Switzerland, 2024. ISBN 9783031637971. doi: 10.1007/978-3-031-63797-1 ~~2~~ 3. URL http://dx. doi.org/10.1007/978-3-031-63797-1_23. 

   - Todd, E., Li, M. L., Sharma, A. S., Mueller, A., Wallace, B. C., and Bau, D. Function vectors in large language models, 2024. URL https://arxiv.org/abs/ 2310.15213. 

   - Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., and MacDiarmid, M. Steering language models with activation engineering, 2023. URL https: //arxiv.org/abs/2308.10248. 

   - Wilinski, M., Goswami, M., Potosnak, W.,´ Zukowska, N.,<sup>˙</sup> and Dubrawski, A. Exploring representations and interventions in time series foundation models, 2025. URL https://arxiv.org/abs/2409.12915. 

   - Ye, H.-J., Liu, S.-Y., and Chao, W.-L. A closer look at tabpfn v2: Understanding its strengths and extending its capabilities, 2025. URL https://arxiv.org/ abs/2502.17361. 

   - Yin, K. and Steinhardt, J. Which attention heads matter for in-context learning? In _Proceedings of the 42nd International Conference on Machine Learning_ , volume 267 of _Proceedings of Machine Learning Research_ . PMLR, 2025. 

- Pearl, J. _Causality_ . Cambridge University Press, 2 edition, 2009. 

- Peters, J., Janzing, D., and Scholkopf, B.¨ _Elements of causal inference: foundations and learning algorithms_ . The MIT Press, 2017. 

- Prior Labs. TabPFN-2.5 model card. https:// huggingface.co/Prior-Labs/tabpfn_2_5, 2025a. Documents the 18-layer (regression) and 24-layer (classification) architecture; accessed 2026. 

- Prior Labs. TabPFN source code: transformer.py (base architecture). https://github.com/ PriorLabs/TabPFN/blob/6.3.0/src/ tabpfn/architectures/base/transformer. py, 2025b. Tag 6.3.0; experiments run on tabpfn==6.3.1; accessed 2026. 

- Prior Labs. TabPFN source code: ensemble.py (preprocessing pipeline). https://github.com/ PriorLabs/TabPFN/blob/68ea9eb/src/ tabpfn/preprocessing/ensemble.py#L716, 2025c. Commit 68ea9eb; experiments run on tabpfn==6.3.1; accessed 2026. 

5 

**Where Computation Lives Inside TabPFN** 

## **A. Experimental Hyperparameters** 

All experiments use a fixed random seed of 42 unless otherwise noted. Tables 2 and 3 list the key experimental hyperparameters for patching and steering experiments respectively. 

_Table 2._ Patching experimental hyperparameters. 

|**Parameter (default)**|**Description**|
|---|---|
|noise<br>~~s~~td(1.0)|Noise standard deviation<br>(_≥_0).|
|corruption<br>~~s~~trength(1.0)|Corruption<br>strength<br>scalar (_≥_0).|
|seed(42)|Random seed.|
|n<br>~~s~~amples(1000)|Dataset size (_≥_10).|
|test<br>~~s~~ize(0.5)|Test<br>split<br>proportion,<br>(0_,_ 1).|



_Table 3._ Activation steering hyperparameters. 

|**Parameter (default)**|**Description**|
|---|---|
|seed(42)|Random seed|
|n<br>~~s~~amples(1000)|Dataset size|
|test<br>~~s~~ize(0.5)|Test split proportion|
|steering strength (1.0)|Scaling factor_α_applied to the direc-<br>tion vector|



## **B. Patching and Ablation: Supplementary Results** 

### **B.1. Patching Setup** 

In activation patching, a clean run _x_<sup>clean</sup> and a corrupted run _x_<sup>corr</sup> differ in a controlled way; the clean activation replaces the corrupted one at site _S_ and the remaining forward pass runs with this patched state: 



ˆ For a scalar regression output _y_ ( _·_ ) we report a normalised restoration score: 



### **B.2. Layer-Level Patching** 

For layer-level patching the intervention site is the full residual-stream representation after layer _ℓ_ : 



This aggregates all heads and MLP computations and measures the total information carried at each depth. Figure 6 shows _≈_ 100% recovery at every layer on the Multiplication Dataset, consistent with a highly distributed representation. 



_Figure 6._ Full-layer patching recovery ratio, Multiplication Dataset. _≈_ 100% recovery at every layer. 

### **B.3. Feature-Block Patching** 

Feature-block patching replaces one feature-block position _b_<sup>_∗_</sup> in the post- _WO_ output tensor _Aℓ ∈_ R<sup>_B×N×F ×k_</sup> (Equation (1)): 



This operates on the post- _WO_ output of self ~~a~~ ttn ~~b~~ etween ~~f~~ eatures and is coarser than head-level patching: it captures the aggregate contribution of all three heads to that block after recombination. The label block ablation (108% effect ratio at layer 5, see below) confirms self ~~a~~ ttn ~~b~~ etween ~~f~~ eatures is on the critical computational path; feature-block patching examines which block positions carry that computation. 

**Multiplication Dataset (** _n_ = 512 **).** For _y_ = _a · b_ + _c_ ( _d_ = 3), the four blocks are _a_ (index 0), _b_ (index 1, corrupted), _c_ (index 2), and the label (index 3). 

Block _b_ (index 1, corrupted) carries the dominant causal signal in early layers (0–4), exceeding 100% recovery: directly patching the corrupted input is the strongest single-block intervention. Block _a_ contributes at _∼_ 55%, consistent with its role as the non-corrupted partner in _a · b_ . Block _c_ is negligible throughout. The label block shows near-zero recovery early, then rises to _≈_ +100% at layer 17: by the final layers, the label token’s representation alone carries sufficient information to recover the prediction. 



_Figure 7._ Feature-block patching on the Multiplication Dataset ( _n_ = 512). _Left:_ absolute restoration per block. _Right:_ recovery ratio (%). Block 1 (green) dominates early layers before handing off to Block 3 (purple) by layer 13. 

**Pairwise-50 Dataset (** _n_ = 512 **).** All feature blocks produce near-zero recovery across all 18 layers, consistent with 

6 

**Where Computation Lives Inside TabPFN** 

_y_ = _S_<sup>2</sup> requiring global aggregation across all 50 features before squaring. Single-block patching can localise computation only when the interaction is token-local ( _a · b_ ); it cannot for an inherently global computation. 

**Corruption parameters and the failure of gaussian** **~~r~~ eplace on Pairwise-50.** For Pairwise-50 patching, mean ~~s~~ hift shifts features 0–9 by +3 _._ 0, producing a signed gap of _−_ 1 _._ 81 — a clean directional signal that supports per-head patching analysis. We initially also tested gaussian ~~r~~ eplace (corrupting 25 features with strength 1 _._ 0 or 3 _._ 0), but found this corruption mode produces a near-zero signed mean gap for _y_ = _S_<sup>2</sup> due to sign cancellation in _S_ =<sup>�</sup> _i_<sup>_xi_(</sup><sup>_≈−_0</sup><sup>_._05 across both strengths).</sup> At _n_ = 64 all heads are at the noise floor ( _≈_ 0 _._ 0003 _σ_ ); at _n_ = 512, gap-normalised recovery percentages can appear large, but per-sample validation reveals these “peaks” are unstable across random seeds and show no consistent directional restoration on the samples that contribute to them. We therefore report only mean ~~s~~ hift results in the main body and treat gaussian ~~r~~ eplace on Pairwise-50 as uninformative for per-head patching. On Multiplication, gaussian ~~r~~ eplace (feature _b_ , strength 1 _._ 0) does produce a measurable directional gap, but to maintain a single corruption regime across the head-level analysis we report mean ~~s~~ hift numbers throughout the main body. 

from all feature blocks; its essentiality confirms self ~~a~~ ttn ~~b~~ etween ~~f~~ eatures is on the critical computational path rather than computation routing through sample-wise attention. 

**Blocks** _a_ **and** _b_ contribute moderately ( _≈_ 40%) and are most influential in early layers (2–3). Block _c_ is dispensable (1% effect), consistent with its purely additive role. 

## **C. MHA Formulation** 

The feature-wise self-attention module computes: 



where _WQ_<sup>(</sup><sup>_h_)</sup><sup>_, W_</sup> _K_<sup>(</sup><sup>_h_)</sup><sup>_, W_</sup> _V_<sup>(</sup><sup>_h_)</sup> _∈_ R<sup>_k×dh_</sup> are per-head projection matrices and _WO ∈_ R<sup>_Hdh×k_</sup> is the output projection. For TabPFN-2.5: _H_ =3, _dh_ =64, _d_ model=192. This decomposition exposes the two intervention granularities described in Section 2.4: feature-block level (patching the post-projection output _Aℓ_ ) and attention head level (patching per-head outputs _V_<sup>ˆ</sup> _ℓ_ before _WO_ ). 

### **B.4. Feature-Block Causal Ablation** 

## **D. Feature Block Construction** 

Ablation tests necessity: for each feature block _b_ at layer _l_ : 



Table 4 summarizes results on the Multiplication Dataset. 

|Block|Feature|_y_normal|_y_ablated|Max Effect|Effect Ratio|Best Layer|
|---|---|---|---|---|---|---|
|0|_a_|4.117|2.591|1.53|37%|3|
|1|_b_(corrupted)|4.117|2.508|1.61|39%|2|
|2|_c_|4.117|4.074|0.04|1%|5|
|3|label|4.117|_−_0_._32<sup>_†_</sup>|4.44|108%|5|



_Table 4._ Ablation effects per feature block, _y_ = _a · b_ + _c_ . Max Effect = _y_ normal _− y_ ablated; Effect Ratio = Max Effect _/|y_ normal _|_ . _†_ The 108% effect ratio is measured at layer 5, where the ablated output changes sign, while the 97% reduction is measured at the layer with minimal ablated output; these quantities therefore come from different layers. 

The label block (index 3) is the model’s most important component. Removing it reduces the prediction by up to 97%, showing that the model relies heavily on it. At layer 5, the measured effect exceeds 100% because the ablated prediction changes sign, making the ratio unstable rather than indicating a stronger effect. 

All experiments use TabPFNRegressor from the tabpfn==6.3.1 package; TabPFN-2.5 model weights are released under a non-commercial license, while the surrounding package code is Apache 2.0 (Grinsztajn et al., 2025). 

The TabPFNRegressor default pipeline expands _d_ raw features to _d_<sup>_′_</sup> = 2 _d_ + 1 preprocessed features: a quantile transform with append ~~o~~ riginal=True doubles the count to 2 _d_ , and one fingerprint feature is appended (Prior Labs, 2025c). The transformer then groups every three consecutive features into a single token and appends one label token (Prior Labs, 2025b), giving _⌈d_<sup>_′_</sup> _/_ 3 _⌉_ + 1 feature blocks per sample. Note that features ~~p~~ er ~~g~~ roup= 3 is specific to the TabPFN-2.5 checkpoint; the TabPFNv2 checkpoint uses a group size of 2 (Ye et al., 2025). 

|Dataset|_d_|_d_<sup>_′_ </sup>= 2_d_+1|_⌈d_<sup>_′_</sup>_/_3_⌉_|+label=blocks|
|---|---|---|---|---|
|Multiplication|3|7|3|**4**|
|Pairwise-50|50|101|34|**35**|



_Table 5._ Feature block counts per dataset. Column 4 is the number of feature groups _⌈d_<sup>_′_</sup> _/_ 3 _⌉_ ; adding one label token gives the total block count. 

The label block is the output-read position whose feature-attention representation aggregates information 

7 

**Where Computation Lives Inside TabPFN** 

## **E. Attention Entropy Computation** 

For a given head _h_ at layer _ℓ_ , the feature-wise self-attention module produces an attention tensor of shape [batch _× N, F, F_ ] after the softmax, where _F_ = _⌈d_<sup>_′_</sup> _/_ 3 _⌉_ + 1 is the number of feature blocks and _N_ is the number of samples in context. We extract this tensor across _B_ evaluation samples, giving _A ∈_ R<sup>_B×F ×F_</sup> (one matrix per sample), where each row _A_ [ _b, q,_ :] is a probability distribution over keys summing to 1. 

**Entropy per query per sample.** For each sample _b_ and query position _q_ , we compute the Shannon entropy: 



**Averaging.** We average _Hb,q_ across all query positions and batch samples: 



Thisˆ quantity equals the empirical conditional entropy _H_ ( _K | Q_ ) — the average uncertainty about which key a randomly drawn query attends to. 

**Normalisation.** We normalise by the maximum entropy log _F_ (achieved by a uniform distribution over keys): 



The normalised entropy _H_<sup>˜</sup> _∈_ [0 _,_ 1], where 0 indicates fully concentrated attention (a single key receives all weight) and 1 indicates uniform attention across all keys. Normalising by log _F_ is necessary for comparing entropy values across the two datasets, which have _F_ = 4 and _F_ = 35 feature blocks respectively — the maximum possible raw entropy differs by a factor of log(35) _/_ log(4) _≈_ 2 _._ 6. 

**Correctness note.** A naive implementation would average the attention matrices across the batch before computing entropy. This is incorrect: since entropy is a concave function, Jensen’s inequality gives _H_ (E[ _A_ ]) _≥_ E[ _H_ ( _A_ )], so averaging first systematically overestimates entropy. The values reported in Figure 3 and Table 1 are computed by the correct order: entropy per sample, then average. 

**Selectivity versus causal necessity.** On both datasets, Head 0 is co-selective with Head 2 at layer 0 (entropy 0.22 on both datasets for Head 0; 0.22 and 0.24 for Head 2 on Multiplication and Pairwise-50 respectively), yet Head 0 

has near-zero ablation effect there while Head 2’s L0 ablation effect (0.076 _σ_ ) is the largest in the experiment. This demonstrates that attentional selectivity at a given layer is necessary but not sufficient for causal necessity: a head can attend selectively without that selective attention being loadbearing for the prediction. We use the alignment between Head 2’s entropy minima and its patching deviation peaks as one of several converging signals, not as a standalone identifier of computational sites. 

## **F. Token-Level Patching** 

Token-level patching replaces the full activation vector at a single sequence position in the feature-attention module (self ~~a~~ ttn ~~b~~ etween ~~f~~ eatures) with the corresponding clean-run activation. The patched tensor has shape [batch _,_ tokens _,_ heads _, dh_ ], so patching token index _t_ replaces one slice along the sequence dimension while leaving all other positions unchanged. 

The sequence length depends on the number of eval samples: seq ~~l~~ en = 264 + _n_ eval, with the test-label token always occupying the final position (index seq ~~l~~ en _−_ 1). 

**Multiplication Dataset,** _n_ **eval** = 1 **.** With a single eval sample, seq ~~l~~ en= 265 and the test-label token is at index 264. Sweeping all 265 token positions, token 264 achieves 100% signed fractional recovery — the only token to do so. All other tokens produce negligible restoration. This is a clean sanity check: causal information funnels entirely through the test-label token position when the computation is isolated to a single sample. 

**Multiplication Dataset,** _n_ **eval** = 64 **.** With 64 eval samples, seq ~~l~~ en= 328 and the test-label token is at index 327. Sweeping all 328 positions, the best token is index 324 (not the label position), and restoration is small throughout. The valid signed fractional recovery count is zero across all layers, confirming the low-gap regime. This null result is consistent with the head-level finding: when the eval batch is larger, single-token interventions cannot recover the distributed computation. 

**Interpretation.** The contrast between the two runs reflects the structure of the task rather than a failure of the method. At _n_ eval = 1 the output is determined by a single test sample and causal leverage is concentrated at the label token; at _n_ eval = 64 the signal is averaged across samples and no single token position carries sufficient causal weight for recovery. Together, the token-level results confirm that the computation is not localizable by sequence position under averaged eval conditions, consistent with the feature-block null results on the Pairwise-50 Dataset (Appendix B.4). 

8 

**Where Computation Lives Inside TabPFN** 

## **G. Activation Steering Experiments** 

Activation steering (Turner et al., 2023; Panickssery et al., 2023) tests whether injecting a contrastive direction vector _α · δ_ into the residual stream can shift model outputs toward a target concept. We test whether the multiplicative relationship ( _y_ = _a · b_ + _c_ ) is encoded as a steerable linear direction anywhere in TabPFN-2.5’s activation space. 

**Setup.** We use the full residual stream at Layer 0 as the hook site — the most complete representation of the model’s state at the earliest layer, and the site where Head 2’s causal necessity is largest. The contrastive direction is _δ_ = mean( _X_ mult) _−_ mean( _X_ add), where the additive batch sets _b_ =0 (reducing _y_ = _a · b_ + _c_ to _y_ = _c_ ) while keeping ( _a, c_ ) fixed. To prevent same-sample leakage, _δ_ is computed on a held-out train split ( _N_ =512), _α_ is selected on a val split ( _N_ =256), and results are reported on a separate test split ( _N_ =256). The generalizable delta has shape [ _F, d_ model] and norm 0.84 after averaging across samples. 

computation is purely relational (Olsson et al., 2022), the task is read entirely from context, and no head produces a context-independent task vector. Contrastive steering, which requires a stable transferable direction, is therefore architecturally mismatched to this setting. 

These are preliminary findings on a single task (Multiplication) and a single direction estimator (mean-diff). Whether the null result holds for Pairwise-50 and more expressive estimators (e.g. linear probes, PCA on contrastive pairs) remains an open question and the primary direction for future work on Q2. 

**Result.** Injecting _α · δ_ at Layer 0 produces near-zero effect on the test split across all _α ∈_ [0 _,_ 10]: MSE improvement is +0 _._ 0% and recovery is _−_ 0 _._ 8% at the val-selected _α_ . Random and shuffled direction controls show identical flat responses, confirming that the null result is not directionspecific. The direction itself is geometrically stable: cosine similarity between _δ_ computed on the train split and independently on the val split is 0.71, ruling out the explanation that the direction is simply noisy. 

**Interpretation.** The direction is consistent but impotent. The per-sample activation differences between multiplicative and additive contexts are large and structured, but when averaged across samples to extract a generalizable _δ_ , the norm collapses _≈_ 300 _×_ . The information that distinguishes multiplicative from additive contexts is sample-specific: it exists in the attention-weighted composition over the particular context set, not as a shared additive component of the activations. 

We argue this is a structural property of pure ICL architectures rather than a feature of the task. TabPFN’s weights encode only how to learn from context; the task relationship is computed fresh on every forward pass through attention over the in-context training set. This contrasts with LLMs, where few-shot ICL is driven primarily by function vector (FV) heads — specific attention heads that produce compact, transferable task vectors which can be extracted and re-injected to recover ICL behaviour without any in-context demonstrations (Todd et al., 2024; Hendel et al., 2023; Yin & Steinhardt, 2025). It is precisely this FV mechanism that makes contrastive steering transferable in LLMs: the task representation is stable across inputs because it lives in parametric head weights. TabPFN has no equivalent: its ICL 

9 

