# **A Mechanistic Study of Tabular Foundation Models** 

**Marin Biloš, James T. Wilson, Anderson Schneider, Yuriy Nevmyvaka** Machine Learning Research, Morgan Stanley 

```
{firstName.lastName}@morganstanley.com
```

## **Abstract** 

Tabular foundation models with different architectures converge in accuracy across a range of classification and regression tasks. This raises questions a leaderboard cannot answer: (i) whether the models execute the same in-context algorithm, (ii) where row, column, and class-permutation invariances originate, and (iii) how robust they are under perturbations engineered against the inferred mechanism. We characterize all three. The model families realize qualitatively distinct similarity-based readouts: from an attention-weighted vote over context labels to a class-conditional mean readout, each confirmed by causal intervention. We find that the representation collapse highlighted in prior work is not a practical concern for them. Each model’s permutation invariances trace to specific positional parameters whose removal preserves accuracy and makes approximate invariance exact. Perturbations engineered against each readout reproduce predicted failure modes; hub and rank attacks isolate them from refit baselines. Together these results give a mechanistic account of contemporary tabular foundation models and identify which inductive biases govern both their accuracy and characteristic failures. 

## **1 Introduction** 

Tabular foundation models such as TabPFNv2 [Hollmann et al., 2025], TabICLv2 [Qu et al., 2025], and Mitra [Zhang et al., 2025] match or surpass specialized models on tabular classification and regression without task-specific training. Given a labeled table at inference time, each model predicts labels for held-out query rows entirely through in-context learning [Dong et al., 2024]. All three are transformer-based [Vaswani et al., 2017] and reach comparable benchmark accuracy despite differing substantially in architecture. This convergence leaves open mechanistic questions. 

First, since the architectures differ in non-trivial ways, we ask whether the learned in-context algorithm is shared. We find that each model forms its prediction at a qualitatively different depth and through a mechanistically distinct readout—the way each model turns its internal representations into a prediction (§2). These recipes are implemented in different parts of each network and are only partly interchangeable across backbones: transplanting one’s readout onto another collapses accuracy (§3). 

A second question concerns a property any tabular model should have: reordering rows or columns or renaming the classes should not change the prediction. No model enforces this by construction, yet we isolate the architectural components that break each symmetry and show that the pretrained models have largely learned to ignore them. Simple surgical edits restore exact invariance at no cost to accuracy, and large-scale pretraining experiments further contextualize how invariance interacts with accuracy (§4). We also revisit the representation collapse concern from Qu et al. [2025] and find it does not materially affect the released models. We prove a fundamental expressiveness barrier for column-invariant architectures on collapse-prone data, then validate this with minimal prototype models and stress tests on the released models. We find that each model’s architectural mitigations sidestep the failure mode in practice (§5). 



<!-- Start of picture text -->
y = 0 y = 1 1.00 Same Class as Query<br>Other Class<br>Prototype Prototype<br>0.75 0.73<br>0.15 0.55 No-Rule<br>⇒ yˆ = 1 0.50 0.45 Baseline<br>0.85<br>0.27<br>0.25<br>Query<br>0.00 Query<br>L8 L9 PC 1<br>(a) Row-attention vote: shared readout (b) TabPFNv2 mean attention mass on (c) TabICLv2 predicts by nearest class<br>of TabPFNv2 L9 and Mitra L9 context rows at L8 and L9 prototype on L11 row tokens<br>2<br>Noise Hub Soft-ExpRank SVD<br>0<br>TabPFN -5.0 -33.1<br>5 1 0.00% 0.23% 0.00%<br>TabICL -39.4 -1.0<br>10<br>0<br>15<br>Mitra -4.5 -9.1<br>1<br>20<br>Vote Proto TabPFN TabICL Mitra<br>Readout TabPFN TabICL Mitra (W = 0) (no-RoPE) (by constr.)<br>(d) Cross-transplant accuracy drop (pp) (e) Five mechanism-grounded perturba- (f) Relative % change in accuracy after<br>vs. backbone native accuracy tions engineered against the readouts enforcing exact column invariance<br>PC 2<br>Attention Mass<br>Backbone<br> Accuracy (%)<br> Test Accuracy (pp) ∆<br>∆<br><!-- End of picture text -->

Figure 1: **Subset of results.** _Illustration of readout mechanisms:_ (a) attention-weighted vote for TabPFNv2 and Mitra, both on layer 9; (b) TabPFNv2 L9 puts most of its attention mass on same-class context rows; (c) nearest class prototype for TabICLv2, at L11. _Causal evidence:_ (d) each backbone’s native readout stays within a few pp of end-to-end accuracy, while cross-backbone readouts collapse; (e) five mechanism-grounded perturbations engineered against these models cut test accuracy; (f) enforcing exact column-permutation invariance does not change the accuracy on all three backbones. 

Finally, we turn the mechanistic findings into targeted attacks. We design data perturbations that should be invisible to a sensible learner but strike the specific machinery each model relies on, exposing distinct, readout-specific weaknesses that confirm the mechanistic picture (§6). 

Together, these results show that models which score alike on benchmarks disagree on how they form predictions, which symmetries they respect, and where they break under pressure. The audit surfaces many additional findings along the way; among them, that some models have a large number of layers that carry little load and can be removed without hurting accuracy (§2). Many of our results apply directly to existing checkpoints, from restoring exact invariance to diagnosing fragilities, while the patterns that emerge across the three families offer concrete guidance for designing future tabular foundation models with stronger invariances and fewer redundant components. 

**Related work.** The three architectures we audit, per-cell attention (TabPFNv2), row-token attention (TabICLv2), and a factorized label-slot design (Mitra), span the dedicated-architecture tabular foundation model class and set the current SOTA [Landsgesell and Knoll, 2026]; LLM-based tabular models [Sui et al., 2024] inherit a textual prior rather than a tabular one, and same-architecture retrains on real data [Ma et al., 2025] reuse a backbone we already audit. Prior empirical work on these models is limited: [McElfresh et al., 2023] catalogues their permutation invariances without locating the responsible parameters, [Gupta et al., 2026] traces task-relevant logic into the middle layers, and Hu and Ghelichi [2025] maps signal-vs-noise separation in late TabPFNv2 layers on synthetic tasks. We work on more families, datasets, and interventions than prior work, pinning down what current models actually do and flagging design choices for future work. 

In-context learning has been cast as gradient descent [Von Oswald et al., 2023, Akyürek et al., 2023], Bayesian inference under a prior [Müller et al., 2022], or function-class learning [Garg et al., 2022]; our recipes (§3) make precise which view each backbone instantiates. The expressiveness limit we use in §5 builds on the permutation-invariance bottleneck of set and context-based architectures [Zaheer et al., 2017, Garnelo et al., 2018], applied here to the column-permutation case. 

2 

## **2 Where representations form** 

The goal of this section is to locate, layer by layer, where each model builds representations that carry enough class information for a simple classifier to read off the answer. 

All three models process a table formed by labeled context rows and unlabeled query rows. TabPFNv2 [Hollmann et al., 2025] treats each cell of this table as a separate token: its 12-block stack alternates row-wise attention (mixing information across rows within each column) with column-wise attention (mixing across columns within row). Mitra [Zhang et al., 2025] resembles TabPFNv2 in its per-cell layout but differs in two important ways: labels live in a separate per-row slot rather than being mixed into feature tokens, and there is no positional encoding on either axis. TabICLv2 [Qu et al., 2025] takes a different route: a column-embedding network first compresses each row’s features into a single vector, producing one token per row. A 12-block ICL transformer then mixes these row tokens, with context labels injected before the first ICL block. Full architectural details are in §A.1. 

We denote the _k_ th layer of any model with L _k_ and inspect the row representations at that layer by taking activations for both the context and query rows. This allows us to train a simple classifier on context rows using the known context labels, and evaluate the fit using query rows. This linear probe tells us at which depth the model has organized its representations to be linearly separable. We complement this probe with three other measurements: a silhouette score on frozen activations, dimensionality summaries of the activation matrix, and per-block knockout experiments. The probe protocols and dataset lists are in §A. A compact summary of all findings appears in Table 90. 

### **2.1 Layerwise probes** 

The three models reach linearly readable class structure at very different depths. TabPFNv2 keeps mean probe accuracy near chance through most of the layers, then jumps sharply between L8 and L9, from 0 _._ 47 to 0 _._ 78 on the 49-dataset benchmark, closing most of the gap to its original accuracy (§B.1.1; Figure 2). The regression version of the model behaves identically, with the peak shifted one layer earlier (§E). Mitra follows a similar late-jump profile, peaking at L9 (§D.2). 

TabICLv2 inverts this picture entirely. Its column-embedding network plus the first ICL block already produce per-row vectors that a _k_ NN classifier reads at 0 _._ 824 accuracy (within 4 pp of endto-end: 0 _._ 864) so most class structure is laid down before the deep ICL trunk runs (§C.1.3). This means the _∼_ 26M-parameter ICL transformer, roughly 95% of TabICLv2’s parameters, adds only marginal accuracy on top of what the column embedder provides, suggesting that a lighter cross-row architecture could recover most of the performance at a fraction of the cost. 

TabPFNv2 and Mitra build class-readable representations late and at a single layer; TabICLv2 is readable throughout and has an inefficient design. This contrast motivates the readout analysis of §3. 

### **2.2 How the geometry evolves through the stack** 

The models reshape their internal geometry in opposite directions, as measured by effective rank, participation ratio, and the twoNN intrinsic dim. [Facco et al., 2017] on per-layer activations (§B.1.4). 

TabPFNv2 compresses early: effective rank drops from _∼_ 40 at blocks 0–1 to _∼_ 10 at block 2. At the final layer, rank correlates with the number of classes ( _r_ =0 _._ 64 at L11, _r_ =0 _._ 46 at the readout layer L9; §I.17), meaning the model uses more representational directions when more classes must be distinguished. Because TabPFNv2’s readout operates on attention patterns rather than feature-space geometry (§3), low rank need not be a bottleneck for the native prediction. 

TabICLv2 runs the other way: effective rank expands monotonically through two thirds of the stack. This is consistent with our proposed readout rule (§3.3); it benefits from well-separated class means, though geometry alone does not determine when class structure becomes readable (§C.1.3). 

### **2.3 One dominant early block per model** 

Knocking out individual blocks reveals that in each model, one early block matters far more than any other: the first transformer block in TabPFNv2 and Mitra (block 0), and the final block of the column-embedding network in TabICLv2 (ColEmb-2) (§B.1.2). Each plays the same role: it sets up a coordinate frame that the downstream layers rely on. 

3 

**TabPFNv2.** Skipping block 0’s MLP in the residual stream cuts the frozen-probe accuracy at L9 by 37 _._ 6 pp, yet a probe retrained from scratch on the post-knockout activations still recovers labels above chance: class information survives but in a frame the rest of the network cannot read. Removing block 9 instead costs little with a retrained probe but cuts the frozen probe by 32 _._ 9 pp (§B.1.3). Block 0 sets up the coordinate system; block 9 organizes it into the form the readout expects. 

**Mitra.** Replacing the first interleaved row/feature block with the identity costs 15 _._ 8 pp on v1 and 41 _._ 1 pp on v1.1, the largest single-block drop in both versions of the model. The next-worst block costs _≤_ 2 pp on v1 and 19 _._ 6 pp on v1.1 (§D.3). The gap between versions suggests that different training procedures make one lean more heavily on its first block. 

**TabICLv2.** ColEmb-2 is a SetTransformer that compresses per-feature into per-row vectors. Zeroing it cuts final accuracy by 28 pp, yet a probe retrained on the post-knockout ICL-input representations still recovers labels at 0 _._ 752 (vs. 0 _._ 812 baseline). The same probe frozen from the intact model collapses to 0 _._ 555 (§C.1.1). ColEmb-2 plays the same coordinate-setting role as block 0 does for TabPFNv2. There is no analogous late organizing block: knocking out any ICL block costs _<_ 3 pp. 

Across all three measurements, the same two-way split emerges. TabPFNv2 and Mitra concentrate their critical computation in one early and one late block, compressing geometry in between; TabICLv2 spreads the work across a large, mostly interchangeable stack whose individual blocks contribute little. This split sets up the readout analysis of §3. 

## **3 Readout mechanisms** 

In §2 we located _where_ class-readable representations form in each stack; this section establishes _how_ each model turns those representations into predictions. Pinning down the readout tells us what each model actually computes, and predicts where it will break. For all models we state a simple, falsifiable surrogate rule, test it causally with interventions, and transplant it onto the other backbones. The classification numbers below aggregate the full 49-dataset benchmark; details are in §B, §C, and §D.4. 

**Faithful surrogate.** A candidate rule is _faithful_ when (i) its predicted probabilities track the model’s predicted probabilities at Pearson _r≥_ 0 _._ 85 on the 49-dataset mean, (ii) its accuracy lies within _≤_ 3 pp, (iii) argmax-agreement reaches Cohen’s _κ≥_ 0 _._ 8 on a held-out test split, and (iv) it survives an intervention that preserves only the components named by the rule. Per model, the joint criterion is met on 40 _/_ 49 datasets for TabPFNv2 and on 43 _/_ 49 for TabICLv2. These cut-offs are conventional thresholds, but to demonstrate the robustness of our claims, we jointly vary _r_ and ∆acc within _±_ 0 _._ 05, which shifts dataset counts by a few datasets, preserving the original conclusions (§I.18). We find the following rules: an attention-weighted vote at L9 for TabPFNv2 and Mitra, and a class-mean prototype on the final representation for TabICLv2. The linear-prototype rule fails on the TabICLv2 regression model. 

### **3.1 TabPFN: an attention-weighted vote at a late layer** 

At layer 9 of TabPFNv2, row-attention sharpens abruptly: L9 is the sharpest layer on all 49 datasets, and the attention distribution at that layer concentrates on a handful of training rows (mean maximum weight 0 _._ 925; entropy roughly half its L0 value; §B.2.1). Figure 2 (middle, §B.1) shows that L9 is the first layer whose attention pattern strongly correlates with TabPFNv2’s predictions. 

**The rule.** Read off the attention weights at L9 and take a weighted vote over context labels. The vote tracks the model’s predicted probabilities at mean Pearson _r_ =0 _._ 89 (§B.2.3). The same picture holds on regression at the same late readout layer (§E). The fidelity decreases monotonically with class count ( _ρ_ = _−_ 0 _._ 60), from binary mean 0 _._ 93 to ten-class mean 0 _._ 80; in that regime, the late MLPs at L9-L11 carry the residual signal. The rule is therefore strongest on binary and few-class problems. 

**Causal test.** Replacing the L9 attention pattern with a uniform one collapses accuracy from 0 _._ 87 to 0 _._ 49 (the majority-class baseline; §B.2.5). The vote thus relies on the same block that §2.3 identified as the readout-building block. The vote signal spreads across all six attention heads at L9 rather than concentrating in one (§I.16). A _k_ NN classifier on TabPFNv2’s final-layer features agrees with its predictions on only about half of test points, well below input-space _k_ NN (§B.2.6). The L9 attention pattern itself is best predicted by representation distance at L8 ( _R_<sup>2</sup> _≈_ 0 _._ 48, §I.9). The rule therefore acts as a _learned similarity_ over the context set, not a metric _k_ NN on the final features. Uniform attention intervention establishes necessity: MLPs at L9-L11 may still contribute on multi-class problems. 

4 

### **3.2 Mitra: an attention-weighted vote at a late layer** 

Mitra’s probe peaks at the same late layer as TabPFNv2’s vote (we report L9 for v1 here, L12 for v1.1 in §D.4). We find the attention-weighted vote is the closest similarity-based surrogate to native, though it does not fully meet the faithful-surrogate bar. 

**The rule.** At L9, read off the row-attention weights from each query to context rows, and take a weighted vote over context labels. Mean Pearson _r_ between vote and native probabilities is 0 _._ 89; and the accuracy gap is 4 _._ 5 pp (§D.4). This gap exceeds the _≤_ 3 pp faithful-surrogate threshold. The residual error likely reflects Mitra’s column-attention pathway, which mixes _y_ -slot and feature-slot information across observations before the row-attention vote and is not captured by the vote rule alone (§D.4). Cosine _k_ NN at the same layer lies 7 _._ 0 pp behind the original and within _∼_ 1 _._ 5 pp of the L9 linear probe. The prototype rule trails by a further 9 _._ 0 pp. Despite the gap, the vote dominates all other surrogates and the causal ablation in §D.6 confirms that row attention is the load-bearing path, placing Mitra in the same “retrieval over context rows" family as TabPFNv2. 

### **3.3 TabICL: a nearest-prototype readout in the final representation** 

TabICLv2’s final-block representation is class-clustered enough that a parameter-free class mean readout reproduces its predictions, while the attention vote from TabPFNv2 fails on TabICLv2. 

**The rule.** Form one _prototype_ per class by averaging the final-block representations of context examples sharing that label, then assign each query to the nearest prototype (Euclidean or cosine). This rule recovers nearly all of TabICLv2’s accuracy (§C.3.1); a small- _k k_ NN on the same representation matches predictions on most queries (§C.3.5). The ordering replicates across all three released versions (Table 31). A linear head also reaches 0 _._ 859, but the prototype is parameter-free and reveals the geometric structure the model builds, not merely that the representation is linearly separable. 

**Built early, refined distributedly.** The class-clustered geometry is laid down before the ICL stack: within-class proximity sits significantly above chance already at the column-embedding output, and a linear probe here reaches 0 _._ 812 versus 0 _._ 864 native (§C.1.1). No _individual_ ICL block is necessary (§C.2.2), but forcing uniform attention across all twelve drops accuracy by 29 pp (§C.2.3), worse than the probe. For regression, the readout is non-linear: no simple surrogate reproduces the model’s predictions, though the final representation retains useful local structure (§I.1, §F.3). 

Where TabPFNv2 retrieves specific rows through sharp attention, TabICLv2 uses no sharp attention at any layer and instead pools entire classes into prototypes: same accuracy with a different algorithm. 

### **3.4 Mechanism transplantation: each readout needs its own backbone** 

If the three models implement distinct readouts, then a readout fit to one backbone should fail on the others. We pass the final-block representation of one backbone through each candidate readout and score on the full benchmark (§I.3). 

Swapping readouts across backbones is catastrophic (Table 1). The prototype rule on TabPFNv2 drops by 33 pp; the vote on TabICLv2 drops by 40 pp. On Mitra, which shares TabPFNv2’s vote family, every alternative readout (prototype, _k_ NN, linear head) falls 7 to 9 pp behind native. A freshly fit logistic head on frozen representations closes the gap on TabICLv2 (0 _._ 859 vs. 0 _._ 864) but not on TabPFNv2 (0 _._ 582 on the shared 42-dataset subset, well below the 0 _._ 78 probe at L9; §B.3.8); a nonlinear MLP head gains only _∼_ 1 _._ 6 pp more. TabICLv2 admits a class-clustered linear readout family; TabPFNv2 does not. 

**Why surrogate rules matter.** The readout rules are not claims about internal computation; they are simple, falsifiable descriptions that reproduce each model’s input-output behavior. Their value is threefold. First, they predict failure modes: because TabPFNv2 acts as if it performs a weighted vote over neighbors, hub poisoning (flipping labels of the most-attended points) directly corrupts the vote; because TabICLv2 classifies by nearest prototype, rank warp destroys the absolute distances its prototypes are calibrated to (§6). Second, the transplant confirms that readout and backbone are jointly designed: each model builds geometry matched to its own rule, and swapping rules across backbones costs 33 to 40 pp. Third, the rules localize the critical computation: TabPFNv2’s prediction hinges on L9 attention, TabICLv2’s on the column embedder, giving concrete targets for pruning, debugging, and future architecture design. 

5 

Table 1: Readout-rule transplantation on the full 49-dataset classification grid (over 5 seeds). Each readout column is applied to the row backbone’s frozen representations at the readout’s preferred layer. Column headers list the model each rule is reported as the surrogate for. Brackets contain the percentage-point drop vs. the row’s native accuracy. Bold marks each backbone’s reported surrogate. Per-dataset numbers, bootstrap CIs, preferred layers, and the MLP variant are in §I.3, I.14, B.3.8, D.4. Backbone Native Vote (TabPFNv2/Mitra) _k_ NN5 Prototype (TabICLv2) Linear head TabPFNv2 0 _._ 854 **0** _._ **804** ( _−_ 5 _._ 0) 0 _._ 547 ( _−_ 30 _._ 7) 0 _._ 523 ( _−_ 33 _._ 1) 0 _._ 617 ( _−_ 23 _._ 7) TabICLv2 0 _._ 864 0 _._ 470 ( _−_ 39 _._ 5) 0 _._ 856 ( _−_ 0 _._ 8) **0** _._ **854** ( _−_ 1 _._ 1) 0 _._ 859 ( _−_ 0 _._ 5) Mitra 0 _._ 860 **0** _._ **815** ( _−_ 4 _._ 5) 0 _._ 789 ( _−_ 7 _._ 0) 0 _._ 769 ( _−_ 9 _._ 0) 0 _._ 776 ( _−_ 8 _._ 3) 

## **4 Models want to be permutation invariant** 

A model is _permutation-invariant_ along an axis if its predictions do not depend on the order in which inputs are presented along that axis: shuffling rows of the context set, reordering the columns of every row consistently, or relabelling classes each leave predictions unchanged. Invariance is the dual of the collapse question of §5: collapse asks when distinct inputs become identically encoded; invariance asks when permutation-related inputs receive identical predictions. We ask whether each model acquires invariance natively and, where it does not, whether the missing symmetry can be installed without retraining and without sacrificing accuracy. We find that most models can achieve all forms of invariance, leaving the benchmark accuracy intact. These edits are free because the positional devices are barely used on real data, but they exist for a reason: §5 stress-tests what happens when the defenses they provide are truly needed. 

**Architectural starting point.** The three architectures begin from different positions on the columnpermutation axis. TabPFNv2 applies feature grouping and a learned positional embedding to every group. TabICLv1 and TabICLv2 apply RoPE along the feature dimension; v2 additionally enforces circular feature grouping. Mitra, in contrast, has _no_ positional encoding on either axis and uses a single shared cell embedding applied identically to every column, making it exactly column-invariant (§D.1). All models are row-invariant, though TabPFNv2 has a row signature that can be disabled at no cost to performance. No classification model in the three families is class-invariant by construction. 

### **4.1 Making pretrained models invariant** 

**TabPFNv2: zeroing the per-feature positional weight matrix.** Setting the per-feature positional weight matrix _W_ to zero while keeping the bias grants _exact_ invariance to within-pair swaps and pair permutations and leaves benchmark accuracy unchanged on the benchmark suite. Interestingly, replacing the bias with only its sign also suffices (§G.3). 

**TabICLv2: removing RoPE.** Removing RoPE in v2 grants _exact_ invariance to circular column shifts and leaves benchmark accuracy essentially unchanged on every dataset (§G.4). The same ablation in v1 is catastrophic ( _∼_ 15 pp drop), because v1 lacks v2’s circular grouping fallback (§5.2). In §4.2 we pretrain an invariant TabICLv1 model from scratch. 

**Mitra: column invariance is exact by construction.** With a single shared cell embedding and no positional encoding on either axis, column attention sees feature tokens as a set, so feature reordering is a no-op up to numerics. Across all datasets, both released checkpoints produce mean _|_ ∆ _p|≤_ 0 _._ 3% on row and column permutations and predicted-label agreement _≥_ 99 _._ 5% (§D.1). The same protocol on TabPFNv2 and TabICLv2 v1/v2 _before_ the edits above produces order-of-magnitude larger column drift; our edits then close that gap. 

In summary, we verify that: TabPFNv2 with _W_ =0 is group-permutation invariant, TabICLv2 without RoPE is circular-shift invariant, and TabICLv1 without RoPE is arbitrary-column invariant (§G.2). 

**Class invariance via One-vs-All.** All classifiers depend on class order through in-context target embeddings and a fixed output head. Wrapping any of them in a One-vs-All classifier grants _full_ class-order agreement on every dataset and leaves benchmark accuracy unchanged (§G.6). This works because the models were trained to solve many different tasks, including binary classification. Our solution does not change models’ weights but introduces a simple data processing step. It requires calling the model _C_ times for _C_ distinct classes, which can be trivially parallelized. 

6 

**The cost of not being invariant.** Our edits eliminate any variability in model predictions at zero accuracy and runtime cost; One-vs-All adds linear-in- _C_ inference (§A.3). Without these edits, permuting columns in a dataset causes up to 8 pp accuracy difference (Figure 5; §G.7). This is one of the reasons why released models rely on ensembling. While it is a valid strategy to obtain approximate invariance, we argue that it is better to include invariance from the start, as Mitra does. The ensembling is then done over different model checkpoints instead of different column orderings. 

### **4.2 Pretraining an invariant TabICLv1 from scratch** 

Can an invariant model be _trained_ from scratch at no cost? We conduct a large pretraining experiment of the TabICLv1 model under the stage-1 protocol of Qu et al. [2025] (100k iterations, batch size 512) using four combinations of two switches: (i) RoPE on or off along the feature axis; (ii) the class-conditional output head replaced by a class-invariant head (one-hot label columns, shared per-class decoder, see §G.9). 

Table 2: TabICLv1 retrained from scratch. Win rate is the per-dataset fraction at which each configuration is best of four. 

|Class-inv.|RoPE|Acc.|Win|
|---|---|---|---|
|No|Yes|0_._829|0_._327|
|No|No|0_._821|0_._249|
|Yes|Yes|0_._826|0_._282|
|Yes|No|0_._803|0_._141|



Each lever alone is essentially free (RoPE removal _−_ 0 _._ 8 pp, class-invariant head _−_ 0 _._ 3 pp); combined 

they cost _−_ 2 _._ 6 pp. The fully invariant model trains stably and stays close to the non-invariant baseline, confirming that invariance is compatible with ICL pretraining. In §5, we identify which architectural components are missing in TabICLv1’s design, which prevent it from achieving better performance when invariant. Mitra demonstrates that the remaining gap can be closed entirely: column invariance is a property of its architecture, and the trained model is competitive on benchmark accuracy and exhibits no representation collapse (§5). Together, the two results suggest the path forward is architectural invariance rather than positional hacks. 

## **5 On representation collapse** 

_Representation collapse_ [Qu et al., 2025] is a per-sample within-row identifiability failure: when several columns share the same marginal and the row aggregator is fully column-permutation invariant, two rows that are permutations of each other along those columns receive identical representations even when their labels differ. The canonical illustration is balance-scale (4 features, 5 equally frequent values; Fig. 4 in Qu et al., 2025). In §4, we showed that removing one positional device is free; this section asks what these devices actually defend against and identifies which architectural component shields each model from collapse. As we have seen in §4, TabICLv1 adds RoPE along columns, TabICLv2 additionally enforces repeated circular feature grouping with a target-aware embedding; TabPFNv2 uses random per-feature embedding together with feature grouping; and Mitra does not use either positional encoding or feature grouping. 

### **5.1 Hand-crafted models pin down what each device does** 

To explain _why_ different approaches succeed, we strip the architectures down to single attention layers we analyze by hand. All results are computed in closed form and reproduced numerically. 

**Setup.** A row _x ∈{_ 0 _,_ 1 _}_<sup>_m_</sup> is presented as _m_ cell tokens, one per column. We study three label rules with shared Bernoulli column marginals (so column-marginal information alone is useless): (A) _y_ = _x_ 1 (one column is the label); (B) _y_ = _x_ 1 _⊕ x_ 2 (label depends jointly on two columns); (C) _y_ = majority( _x_ ) (a multiset-decidable control). Each model is one attention layer plus a hardargmax readout, and we report accuracy on the full 2<sup>_m_</sup> -row enumeration with _m_ =3. Any model whose row representation is invariant to all column permutations is constant on the orbits of _Sm_ and so cannot exceed the orbit-counting bound: for the binary targets of Tasks A and B at _m_ =3 this bound equals 0 _._ 75 (Lemma 1 for Task A in its general orbit-majority form; Proposition 2 for Task B for orbit-wise majority of _x_ 1 _⊕x_ 2; §B.3.4). The bound applies to the toy models M0-M3 below under their assumed query-side invariance; the mapping to the released backbones is by analogy. 

**The models. M0** is a naive shared-cell-embedding mean-pool: every cell gets the same embedding, the row token averages them, and the readout sees a column-permutation-invariant statistic. M0 hits the 0 _._ 75 multiset bound on Task A (§B.3.3). The other three models break the symmetry differently. 

7 

**M1 (content route, TabPFNv2-style)** attaches an explicit column identifier to each cell embedding, so cell-level cross-attention conditions on _which column_ as well as _which value_ . **M2 (structural route, TabICLv2-style)** uses no positional encoding but masks attention so each head sees only one column; on this task family M1 and M2 produce identical predictions. **M3 (pair-grouped)** uses tokens that are super-cells over unordered feature pairs ( _j, k_ ) carrying a 4-way value and a pair identifier: the construction we exhibit that also solves Task B (XOR). The released models implement variants of these routes. Accuracies for all four models on Tasks A-C appear in Figure 4 (§B.3.3). 

**Mapping to the released models.** TabPFNv2’s feature-grouping (groups of two) is the M3 analogue and its positional embeddings are the M1 analogue. TabICLv1 pairs the M2 analogue (column-stream attention) with RoPE as a within-row symmetry-breaker. TabICLv2 pairs the M2 analogue with the M3 analogue (circular grouping) and adds target-aware embeddings. Each v2 architecture therefore carries two within-row routes, while TabICLv1 carries only one. This matches the benchmark pattern of §4.1: removing any single v2 mitigation is free, while removing RoPE in v1 is catastrophic. 

### **5.2 Direct stress test on collapse-prone data** 

The released v2 checkpoints carry redundant within-row defenses; collapse only appears when we strip them. We probe _which_ mitigations are load-bearing on the balance-scale dataset ( _n_ =625, _d_ =4) and a _d_ =12 balance-like synthetic stress dataset with identical marginals (§I.7). 

**One vs. two within-row devices removed (v2 backbones).** Removing any _single_ within-row defense on a v2 backbone is approximately free for _W_ =0 on TabPFNv2 and no-RoPE on TabICLv2 (both within _|_ ∆ _| ≤_ 0 _._ 001); the M3 pair-channel ablation (2nd-slot=0) costs up to _∼_ 14 pp on the most stressed slice (Tables 78 and 79). Removing _both_ exposes the latent collapse: TabICLv2 with RoPE and circular grouping removed drops to the majority baseline on the _d_ =12 stress ( _−_ 50 pp), and TabPFNv2 with both within-row routes off ( _W_ =0 and pair second-slot= 0) drops by 7–12 pp (Table 79). Each v2 architecture therefore carries a _redundant_ within-row stack, not a single defense. 

**TabICLv1 versus stripped v2.** TabICLv1 no-RoPE drops 44 pp on balance-scale, larger than stripped v2. Under the harder _d_ =12 stress, stripped TabICLv2 also collapses by 50 pp. The buffer that delays v2’s collapse is the target-aware embedding, which v1 lacks and which survives both v2 ablations. Both ICL backbones collapse to the majority baseline once their within-row defenses are gone. 

**What makes Mitra special.** Mitra has no within-row symmetry-breaker by design and still matches TabPFNv2 with all its defenses in place (0 _._ 96 _/_ 0 _._ 97 vs. 0 _._ 96 _/_ 0 _._ 97 on balance-scale and balance-like _d_ =12). The relevant counterfactual is v2 _stripped_ of its within-row defenses: TabPFNv2 with both routes off only drops to 0 _._ 89 _/_ 0 _._ 86, well above majority and in the same regime as Mitra. Compared to TabICLv2, Mitra and TabPFNv2 are more robust to combined within-row stripping. We posit that introducing the context label earlier in the stack and the lack of row compression are the main drivers of this phenomenon. The attacks of §6 follow from the resulting structural difference. 

**A recipe for the next generation of models.** When designing the next tabular foundation model, we suggest adopting all three of the following: (i) make the model jointly column-invariant by construction; (ii) inject the label early, as a slot visible to column attention from the first layer; (iii) read out from the in-context labels rather than from a fixed class-conditional head, for example via one-hot labels. Built-in invariance is the more principled design: it makes the predictor a deterministic function of the context, removes the need for permutation ensembling (§4), and closes the symmetryattack surface that the released models exhibit (§6). The three are complementary, not interchangeable. Mitra adopts all three and reaches benchmark accuracy without any within-row positional device. Adopting only (i) and (iii) leaves an unrecovered gap: TabICLv1 retrained from scratch in this configuration loses 2 _._ 6 pp relative to its non-invariant baseline because it lacks (ii) (§4.2). 

## **6 Mechanism-grounded adversarial tests** 

If the readouts of §3 describe what each model computes, they predict which context-set perturbations should hurt which model. We design eight perturbations (§H.1) grouped by what they target; the readouts’ geometry: _hub poison_ flips the labels of context points that sit closest on average to all others, and _rank warp_ replaces values by their per-column rank, preserving order but destroying absolute scale; context corruptions that affect any flexible classifier: _noise padding_ , _centroid_ and _boundary_ label injection, and _SVD burial_ (joint reweighting of high and low-variance directions); 

8 

|Table 3: Mechanism-g|rounded attacks on a com|mon24-classifcation grid|(_×_5seeds): within-model|
|---|---|---|---|
|∆pp with bootstrap C|Is. Clean accuracies are|0_._882 (TabPFNv2), 0_._88|8 (TabICLv2), and 0_._881|
|(Mitra v1.1; v1 within|1pp). <sup>_∗_</sup>marks Holm-corre|cted signif. (_α_=0_._05). Se|e §H,D.5for more results.|
|Attack|TabPFNv2∆pp [95% CI]|TabICLv2∆pp [95% CI]|Mitra∆pp [95% CI]|
|Noise pad|_−_1_._5<sup>_∗_</sup>[_−_2_._4_, −_0_._7]<br>|_−_0_._6 [_−_1_._5_,_+0_._5]<br>|_−_0_._5 [_−_1_._3_,_+0_._2]<br>|
|Hub poison|_−_3_._7<sup>_∗_</sup>[_−_5_._5_, −_2_._1]<br>|_−_3_._3<sup>_∗_</sup>[_−_4_._8_, −_1_._9]<br>|_−_2_._8<sup>_∗_</sup>[_−_4_._0_, −_1_._6]<br>|
|Centroid inj.|_−_0_._2 [_−_0_._9_,_+0_._6]<br>|_−_0_._3 [_−_1_._2_,_+0_._5]<br>|_−_0_._3 [_−_0_._9_,_+0_._4]<br>|
|Boundary poison|_−_1_._7<sup>_∗_</sup>[_−_2_._5_, −_1_._0]<br>|_−_1_._2 [_−_2_._2_, −_0_._1]<br>|_−_1_._5 [_−_3_._3_,_+0_._4]<br>|
|Cube warp|_−_1_._1 [_−_2_._5_,_+0_._1]<br>|_−_0_._3 [_−_1_._4_,_+1_._2]<br>|_−_0_._3 [_−_0_._8_,_+0_._0]<br>|
|Soft-exp warp|_−_2_._2<sup>_∗_</sup>[_−_3_._7_, −_0_._8]<br>|_−_2_._3<sup>_∗_</sup>[_−_3_._9_, −_0_._8]<br>|_−_2_._5<sup>_∗_</sup>[_−_3_._8_, −_1_._2]<br>|
|Rank warp|_−_8_._0<sup>_∗_</sup>[_−_15_._1_, −_2_._4]<br>|_−_10_._1<sup>_∗_</sup>[_−_18_._7_, −_3_._4]<br>|_−_0_._4 [_−_1_._0_,_+0_._1]<br>|
|SVD burial|_−_8_._3<sup>_∗_</sup>[_−_13_._7_, −_3_._6]|_−_5_._0<sup>_∗_</sup>[_−_8_._7_, −_2_._0]|_−_8_._9<sup>_∗_</sup>[_−_14_._7_, −_4_._0]|



smooth monotone warps: _cube_ and _soft-exp_ , a Bayesian average over context-conditional predictors should largely absorb them; and a null-space PGD diagnostic (§H.9). All three foundation models are run on the same grid; Ridge, XGBoost, MLP baselines fit fresh on the same poisoned context (§H). 

The drops in Table 3, set against the refit baselines of §H, follow this split. The two geometry attacks hurt TabPFNv2/TabICLv2 substantially more than a tuned MLP refit on the same poisoned context, isolating the readouts as the source; Mitra matches the MLP on hub poisoning and is immune to rank warp by virtue of its quantile front-end. The four context corruptions hurt the MLP at least as much as the foundation models, so the drops there—including the large SVD burial drops on TabPFNv2 and Mitra—reflect sensitivity any flexible classifier inherits from a corrupted context. The two monotone warps hurt the MLP roughly twice as much as the foundation models, consistent with their pretraining prior already covering smooth monotone warps that a freshly fit MLP has to learn from a single context. A pre-specified Wilcoxon test on (rank warp, hub poison) vs. (cube warp, centroid injection) rejects at _p <_ 10<sup>_−_3</sup> for TabPFNv2/TabICLv2; the 16-dataset mixed-type slice and the 49-dataset Mitra sweep reproduce the ordering. 

## **7 Discussion and limitations** 

The audit covers 49 classification and 10 regression datasets, multiple permutation and perturbation grids and paired bootstrap intervals throughout: in aggregate, tens of thousands of evaluation points across the three foundation models. TabPFNv2 and Mitra read out through an **attention-weighted vote at a late layer** and TabICLv2 through a **nearest-prototype rule** in its final representation, and swapping these heads across backbones costs 33 to 40 accuracy points: readout and backbone are **jointly designed** . Much of each backbone is **not** load-bearing: the v2 models carry a **redundant within-row stack** , per-block knockouts cost little at almost every layer, the TabPFNv2 readout is nailed at one late block, and on TabICLv2 most of the class signal is already present at the ICL input. Mitra contains _no_ within-row symmetry-breakers yet matches the others, proving representation collapse is not fundamental but **architectural** . On the permutation axis, TabPFNv2 and TabICLv2 admit a **one-line edit to exact column invariance** and a One-vs-All wrapper grants **exact class invariance** on any of the three; a from-scratch invariant TabICLv1 trains stably but gains no accuracy, providing further evidence for our prescribed model. The same story predicts the readout-specific breakages: **rank warp** and **hub poisoning** hurt TabPFNv2/TabICLv2 more than a refit MLP does; Mitra is immune to cube and rank warps but shares the _k_ NN-style hub weakness. 

**Limitations.** Two directions are out of scope. We hold the pretraining mixture fixed and do not probe how the synthetic data generator itself shapes the mechanisms we audit, and we do not cover the very large dataset regime, where in-context inference becomes infeasible. 

**Future work.** Our findings suggest a hypothesis for a next-generation tabular foundation model: combine Mitra’s backbone (column invariance), an early label-as-column slot, and a One-vs-All output head (class invariance). Each ingredient is accuracy-neutral or better in isolation. Two further follow-ups remain open: (i) the training data itself, where varying the synthetic prior might isolate which behaviors are properties of the architecture versus the data; (ii) mechanism-grounded adversarial defenses, where hub-aware context reweighting and robust quantile transforms can be evaluated under the same paired protocol. Each direction promises foundation models that are simultaneously more accurate, more robust, and more interpretable than the current generation. 

9 

## **References** 

Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? Investigations with linear models. In _International Conference on Learning Representations_ , 2023. 

- Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, et al. A survey on in-context learning. In _Proceedings of the 2024 conference on empirical methods in natural language processing_ , 2024. 

- Elena Facco, Maria d’Errico, Alex Rodriguez, and Alessandro Laio. Estimating the intrinsic dimension of datasets by a minimal neighborhood information. _Scientific Reports_ , 7(1):12140, 2017. 

- Peiran Gao, Eric Trautmann, Byron Yu, Gopal Santhanam, Stephen Ryu, Krishna Shenoy, and Surya Ganguli. A theory of multineuronal dimensionality, dynamics and measurement. _bioRxiv_ , 2017. 

- Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? A case study of simple function classes. In _Advances in Neural Information Processing Systems_ , 2022. 

- Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J Rezende, SM Eslami, and Yee Whye Teh. Neural processes. In _ICML Workshop on Theoretical Foundations and Applications of Deep Generative Models_ , 2018. 

- Aviral Gupta, Armaan Sethi, and Dhruv Kumar. Tabpfn through the looking glass: An interpretability study of tabpfn and its internal representations. _arXiv_ , 2026. 

- Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. _Nature_ , 2025. 

- Jingnan Hu and Sina Ghelichi. Understanding TabPFN: An interpretability analysis of tabular prior-fitted networks. _arXiv_ , 2025. 

Jonas Landsgesell and Pascal Knoll. Scoringbench: A benchmark for evaluating tabular foundation models with proper scoring rules. _arXiv_ , 2026. 

- Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Alex Labach, Hamidreza Kamkari, Jesse C Cresswell, Keyvan Golestan, Guangwei Yu, Anthony L Caterini, and Maksims Volkovs. Tabdpt: Scaling tabular foundation models on real data. In _Advances in Neural Information Processing Systems_ , 2025. 

- Duncan McElfresh, Sujay Khandagale, Jonathan Valverde, Vishak Prasad C, Ganesh Ramakrishnan, Micah Goldblum, and Colin White. When do neural nets outperform boosted trees on tabular data? _Advances in Neural Information Processing Systems_ , 2023. 

- Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do Bayesian inference. In _International Conference on Learning Representations_ , 2022. 

- Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. TabICL: A tabular foundation model for in-context learning on large data. In _International Conference on Machine Learning_ , 2025. 

- Olivier Roy and Martin Vetterli. The effective rank: A measure of effective dimensionality. _European Signal Processing Conference (EUSIPCO)_ , 2007. 

- Yuan Sui, Mengyu Zhou, Mingjie Zhou, Shi Han, and Dongmei Zhang. Table meets llm: Can large language models understand structured table data? a benchmark and empirical study. In _Proceedings of the 17th ACM International Conference on Web Search and Data Mining_ , pages 645–654, 2024. 

10 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances in Neural Information Processing Systems_ , 2017. 

- Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In _International Conference on Machine Learning_ , 2023. 

- Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R Salakhutdinov, and Alexander J Smola. Deep sets. In _Advances in Neural Information Processing Systems_ , 2017. 

- Xiyuan Zhang, Danielle C. Maddix, Junming Yin, Nick Erickson, Abdul Fatir Ansari, Boran Han, Shuai Zhang, Leman Akoglu, Christos Faloutsos, Michael W. Mahoney, Cuixiong Hu, Huzefa Rangwala, George Karypis, and Bernie Wang. Mitra: Mixed synthetic priors for enhancing tabular foundation models. In _Advances in Neural Information Processing Systems_ , 2025. 

11 

## **A Experimental protocol** 

All experiments use 49 classification datasets (4 synthetic, 3 from `sklearn` , and 42 from OpenML) and 10 regression datasets, listed in Tab. 8. We use 5 seeds for invariance experiments, the common 24-dataset attack grid, and the readout-grid calibration table; 10 seeds for the broader TabPFN-vsXGBoost attack sweep; and 20 seeds for vote-decomposition permutation tests. Exact seed and dataset counts are stated in the relevant captions or subsections. 

**Context/query partitions.** For real and OpenML classification datasets, each (dataset, seed) cell starts from an 80/20 stratified context/query split seeded by the seed index. Regression uses the same 80/20 split without stratification. Synthetic controls use their generator-defined train/test sizes. Unless stated otherwise, a given (dataset, seed) context/query partition is reused across the foundation models, surrogate readouts, classical baselines, and attacks. 

### **A.1 Architectures** 

This subsection specifies the released classification models audited in the paper, their parameter and shape accounting, the released versions used in each experiment, and the software environment. 

**TabPFNv2** [Hollmann et al., 2025]. A per-cell transformer (hidden width _D_ = 192, 12 blocks) operating on the ( _n_ rows + 1) _× n_ cols table formed by the context set plus the query row. Each block alternates row-wise attention (mixing across context rows for a fixed column) with column-wise attention (mixing across columns for a fixed row), interleaved with MLPs. Inputs are passed through a fixed three-copy preprocessing pipeline (raw, scaled, quantile-rotated). Numerical features are encoded via the per-feature positional matrix _W_ and bias _b_ ; class identity is read off the query cell at the output head. For a single query and _n_ context rows with _d_ raw features, the stack therefore carries 3( _n_ +1) _d_ cell tokens: row attention sees _n_ +1 entries in each of 3 _d_ feature streams, and column attention sees 3 _d_ entries in each row. 

**TabICLv2** [Qu et al., 2025]. A row-then-ICL transformer. In the accounting of Table 4, the model splits into three column-embedding blocks, three pre-label row-interaction blocks (“Row 3”), and then the 12-block row-level ICL transformer (“ICL 12”). The first two stages process the table column by column and emit a single vector per row, i.e. a per-row embedding matrix in R<sup>(</sup><sup>_n_context+</sup><sup>_n_query)</sup><sup>_×D_row</sup> ; the third column-embedding block is a SetTransformer readout that compresses per-feature into per-sample vectors before the row stage. The 12 ICL blocks then mix the context rows together with the query row after labels are injected. RoPE is applied along the feature dimension and circular feature grouping is applied to inputs. The class prediction is read off the final ICL block’s output for the query row. In shape terms, the ColEmb stack starts from _d_ feature tokens per row, compresses them to ( _n_ context + _n_ query) row vectors by ColEmb-2, and from that point onward both the Row 3 and ICL 12 stages operate only on those row tokens. 

**Mitra** [Zhang et al., 2025]. A 12-block transformer trained with mixed synthetic priors and released as part of AutoGluon. It differs from TabPFNv2 and TabICLv2 along axes the main analysis identifies as load-bearing: labels live in a _separate_ per-row token slot rather than being added to feature tokens; the per-feature embedding is _shared_ across columns; and the architecture has _no positional encoding_ on either axis. Each block is an interleaved row/feature `Layer` : a row-attention call lets each query token attend to the context tokens for the same feature, and a column-attention call mixes across features within a row. Both released checkpoints (v1 and v1.1) share this 12-block structure; they differ in their training mixtures. We load Mitra as a single model with AutoGluon’s ensembling/stacking disabled, so the meta-ensemble defaults that would otherwise wrap it are turned off and we measure the model in isolation. Per-experiment Mitra details (invariance, layerwise probes, knockouts, readout fidelity, attacks, collapse) appear in §D. 

For TabPFNv2, 7 _._ 08M of 7 _._ 24M parameters sit in the 12-block transformer stack; “no thinking rows” means this model does not prepend any extra synthetic rows before the context/query table. For TabICLv2, the split is 0 _._ 88M / 0 _._ 40M / 26 _._ 28M across the column, row, and ICL stages. For Mitra, 75 _._ 66M of 75 _._ 67M parameters sit in the 12-block interleaved row/column stack; the remainder is the per-feature 1 _→_ 512 embedding, the 10-class label embedding, and the final 512 _→_ 10 head. The released TabPFNv2 model used here is the 12-block PerFeatureTransformer, not the newer 24-block v2.5/v2.6 architectures, which are released under licenses that are not open-source compatible with this analysis pipeline and are therefore out of scope. 

12 

Table 4: Architecture accounting for the three main classification models, derived from the released model files used throughout this paper. 

|Model|Params|Core stack|Heads|FFN hidden|Normalization|
|---|---|---|---|---|---|
|TabPFNv2|7.24M|12 PerFeatureTransformer blocks,_d_=192,<br>no thinking rows|6 feature + 6<br>item|768|3 bias-free LN / block|
|TabICLv2|27.55M|ColEmb 3 + Row 3 + ICL 12,_d_row=128,<br>_d_ICL=512|8 / 8 / 8|256 / 256 / 1024|pre-norm LN with bias|
|Mitra|75.67M|12 interleaved row/column blocks,_d_=512,<br>separate per-row label slot, no positional<br>encoding|4 row + 4 col|2048|4 LN with bias / block|



Table 5: Shape and accounting proxy for the three audited classification models on a single-query context set with _n_ context rows and _d_ raw features. The “attention-work proxy” counts attentionmatrix elements and ignores heads, MLPs, and constant factors; it is a scaling summary, not an exact FLOPs count. 

|Model|Serialized state|Attention-work proxy|Measured cost|
|---|---|---|---|
|TabPFNv2|3(_n_+1)_d_ cell tokens throughout the<br>12-block stack|3_d_(_n_+1)<sup>2 </sup>from row attention + (_n_+1)(3_d_)<sup>2</sup><br>from column attention, per block<br>|_∼_1_._0s,_∼_3GiB|
|TabICLv2|_d_feature tokens / row until ColEmb-2,<br>then_n_+1row tokens through Row 3<br>+ ICL 12|3(_n_+1)_d_<sup>2 </sup>in the ColEmb stage + 15(_n_+1)<sup>2</sup><br>across Row 3 and ICL 12<br>|_∼_1_._5s,_∼_4GiB|
|Mitra|(_n_+1)(_d_+1)tokens per layer (_d_fea-<br>ture cells + 1 label slot per row),<br>through the 12-block stack|_d_(_n_+1)<sup>2 </sup>from row attention +(_n_+1)(_d_+1)<sup>2</sup><br>from column attention, per block|—|



**Software environment.** All experiments run under Python and PyTorch, using the official TabPFNv2 and TabICLv2 Python packages together with `sklearn` for the classical baselines and probes. 

**Categorical features and missingness.** Categorical columns from OpenML datasets (e.g. car, tictac-toe, cmc, credit-approval, credit-g, cylinder-bands, eucalyptus) are ordinal-encoded once with a fixed seed and then passed identically to TabPFNv2, TabICLv2, Ridge, and XGBoost, so that any model differential is not driven by encoding choice. This equalises the inputs across baselines; it is not a benchmark of native-categorical CatBoost or native-categorical XGBoost, which use their own typed pathways. Missing values are median-imputed for numerical columns and mode-imputed for categorical columns at fetch time. The post-encoding feature count _d_ in Tab. 8 reflects this encoding. 

### **A.2 Probes and separability measurements** 

This subsection specifies the linear probe, its random-init control, the silhouette score, and the three geometry summaries used to characterize per-layer representations. 

**Linear probe.** Given a frozen activation tensor _h ∈_ R<sup>_n×D_</sup> at some layer, we fit a logistic-regression ˆ classifier _y_ = arg max _c Wch_ on a labeled subset of the context rows and report its accuracy on a held-out subset of the same dataset. Unless stated otherwise, the probe is trained per (dataset, layer, seed) using multiclass logistic regression with _ℓ_ 2 regularization _C_ =1 and an L-BFGS solver, on an 80/20 train/eval split of the context rows under a per-seed shuffle. The maximum number of solver iterations is raised to 2 000 to ensure convergence on the higher-dimensional probe inputs (semeion _d_ =256 and mfeat-pixel _d_ =240). We always report probe accuracy alongside two reference points: (i) the per-dataset chance baseline (the majority-class frequency, mean _≈_ 0 _._ 50 across our 49 classification datasets), and (ii) the model’s own end-to-end accuracy on the same (dataset, seed) pair. A probe accuracy that approaches the second reference indicates that essentially all of the model’s classification-relevant information is already linearly readable at that layer. 

**Random-init control.** Wherever a probe accuracy is reported, we also report the same probe trained on the activations of a copy of the model whose transformer weights have been replaced by random initialisations. This control isolates the part of separability that is due to the trained weights as opposed to the architecture or the probe’s own capacity. 

**Silhouette score.** For a labeled point cloud _{_ ( _hi, yi_ ) _}_ , the silhouette of _hi_ is ( _bi − ai_ ) _/_ max( _ai, bi_ ), where _ai_ is the mean distance from _hi_ to other points of the same class and _bi_ is the mean distance from _hi_ to points of the nearest _other_ class. The dataset-level silhouette score is the mean over _i_ and 

13 

lies in [ _−_ 1 _,_ +1]; values near +1 indicate clean per-class clusters. We use cosine distance and report silhouette scores at every layer. 

**Geometry summaries.** For a frozen activation matrix _h ∈_ R<sup>_n×D_</sup> with mean-centered singular values _σ_ 1 _≥· · · ≥ σr >_ 0, define normalized squared spectrum _pi_ = _σi_<sup>2</sup><sup>_/_�</sup> _j_<sup>_σ_</sup> _j_<sup>2.We report three standard</sup> summaries: 

- _Effective rank_ exp� _−_<sup>�</sup> _i_<sup>_pi_log</sup><sup>_pi_</sup> �: the exponential of the spectral entropy [Roy and Vetterli, 2007]. Equals _r_ for a flat spectrum and 1 when one direction dominates; insensitive to absolute scale. 

- 2 

- • _Participation ratio (PR)_ �� _i_<sup>_σ_</sup> _i_<sup>2</sup> � �� _i_<sup>_σ_</sup> _i_<sup>4:a heavier weighting toward the leading directions</sup> [Gao et al., 2017]. PR equals _r_ for a flat spectrum but drops much faster than effective rank when a few directions concentrate most of the variance. 

- _2NN intrinsic dimension (2NN-ID)_ [Facco et al., 2017]: estimated from the ratio of distances to the second and first nearest neighbour of each point. Captures the dimensionality of the local data manifold rather than of the global covariance, and is therefore typically smaller than effective rank. 

Effective rank and PR are linear-algebraic measures of how the activation _covariance_ is spread across directions; 2NN-ID is a geometric measure of how the activation _point cloud_ fills its ambient space. They agree qualitatively but probe different things, which is why we report all three. 

### **A.3 Hardware and example costs** 

This subsection specifies the compute node, the approximate budget of the headline GPU-bound runs, and the per-prediction inference cost on that node. 

All experiments were run on a node with 4 _×_ NVIDIA H100 NVL GPUs (94 GiB each) and dual Intel Xeon CPUs (128 physical cores); preprocessing and CPU-only experiments use this CPU pool. 

Table 6 gives the approximate cost of the headline GPU-bound experiment families measured on this node; many smaller CPU-bound or single-dataset sweeps are omitted because they do not materially affect the total budget. 

Table 6: Approximate compute ledger for the headline GPU-bound runs in the paper. Costs are measured on the 4 _×_ H100 node above and rounded to the nearest half GPU-hour. 

|Experiment family|Grid|Approx. cost|
|---|---|---|
|Full inference-only sweep with probes|49cls_×_10seeds, all layers,<br>sharded over4GPUs|_∼_2GPU-hours total|
|Frozen probe after knockout|49 _×_ 10, 4 KO points _×_8<br>probe positions|_∼_6GPU-hours|
|Vote decomposition with1000row-permutation null|<br>49cls_×_20seeds|_∼_4GPU-hours|
|Activation patching|clean/corrupted pair enumer-<br>ation on49cls|_∼_18GPU-hours|
|Headline subtotal|—|_∼_30GPU-hours|



**Per-prediction inference cost.** On the same hardware, cold end-to-end inference for a single query is _∼_ 1 _._ 0 s for TabPFNv2, _∼_ 1 _._ 5 s for TabICLv2, and _∼_ 2 _._ 0 s for TabICLv2-reg (numbers include preprocessing). Peak GPU memory for these calls is _∼_ 3 GiB, _∼_ 4 GiB, and _∼_ 5 GiB respectively, well within a single H100. The OvA wrapper of §4.1 multiplies these by the class count _C_ : no measured accuracy penalty for binary classification, _∼_ 10 s and _∼_ 30 GiB for the ten-class mfeat-* family. Zeroing _W_ in TabPFNv2 and removing RoPE in TabICLv2 are zero-overhead edits to fixed parameters and do not change wall-clock or memory. 

All experiments, except for TabICLv1 are inference-only on released models. 

### **A.4 Seeds, dataset subsets, and XGBoost tuning** 

This subsection specifies the seed convention, the enumerated dataset subsets used by individual experiments, the attack-grid accounting, the calibration metric definitions, the XGBoost reference configuration, and the full attack hyperparameter values. 

14 

**Random seeds.** Whenever an experiment uses _N_ seeds we use the integer seeds 0 _,_ 1 _, . . . , N −_ 1; the 20-seed permutation null uses 0 _, . . . ,_ 19. Each (dataset, seed) cell is therefore deterministic given the software environment above. We disable TF32 matmul for both backbones and set deterministic cuDNN algorithms; forward passes through the released models match across consecutive runs to within 10<sup>_−_6</sup> in logits on H100. 

**Explicit subset compositions.** The compute-intensive experiments are run on enumerated subsets of the full 49-classification grid; the exact lists are: 

- _The_ 24 _-classification attack grid_ (§6), defined as the intersection of the TabPFNv2 and TabICLv2 attack runs: `LED-display-domain-7digit` , `MiceProtein` , `balance-scale` , `banknote-authentication` , `blood-transfusion-service-center` , `breast-w` , `breast_cancer` , `car` , `climate-model-simulation-crashes` , `cylinder-bands` , `dresses-sales` , `eucalyptus` , `iris` , `mfeat-factors` , `mfeat-fourier` , `mfeat-karhunen` , `mfeat-morphological` , `mfeat-pixel` , `quadrant_2d` , `random_labels` , `sign_1d` , `steel-plates-fault` , `wine` , `xor_2d` . 

- _The_ 16 _-dataset mixed-type clean slice_ (Tab. 67, §H.7): `dresses-sales` , `car` , `cylinder-bands` , `eucalyptus` , `cmc` , `credit-approval` , `credit-g` , `analcatdata_dmft` , `anneal` , `Fitness_Club` , `Is-this-a-good-customer` , `qsar-biodeg` , `website_phishing` , `MIC` , `tic-tac-toe` , `ilpd` . 

- _The mixed16 attack slice_ (Tab. 68, §H.8): the same 16 datasets as the mixed-type clean slice above, but only the three highest-signal attacks (hub poison, rank warp, SVD burial) under the shared ordinalised numeric view. 

- _The_ 10 _-classification head-resolved TabPFNv2 vote subset_ (§B.2.4): `balance-scale` , `breast_cancer` , `credit-g` , `diabetes` , `ilpd` , `iris` , `mfeat-zernike` , `tic-tac-toe` , `vehicle` , `wine` . 

- _The per-block MLP knockout subset_ (Tab. 13, §2.3): `xor_2d` , `quadrant_2d` , `sign_1d` , `random_labels` , `balance-scale` , `banknote-authentication` , `breast-w` , `iris` , `wine` , `vehicle` , and `steel-plates-fault` . This fixed subset keeps the four synthetic controls and adds binary, three-class, four-class, and seven-class real tasks so the mean drop table is auditable from the PDF alone. 

**Attack accounting.** The attack results use two distinct reporting grids. The matched cross-model comparison in Table 3, Table 64, Table 65, and Table 66 uses the common 24-classification grid with 5 seeds per dataset (per-cell numbers may differ by up to _∼_ 0 _._ 2 pp across the three tables because they were produced in independent runs over the same grid). The broader TabPFN-vs-XGBoost appendix comparison in Table 63 uses 49 classification datasets _×_ 10 seeds, except MONO_SOFTEXP where one dataset is dropped for numerical degeneracy and _n_ =48. The null-space PGD diagnostic in §H.9 uses the 29 datasets with sufficient null-space dimension after the explicit exclusions in §H.9. The readout-grid calibration table (§I.14) uses 49 classification datasets _×_ 5 seeds. 

**Calibration metrics.** Expected calibration error (ECE) in Table 85 is computed per (dataset, seed) with 15 equal-width confidence bins and then averaged over seeds within dataset; negative loglikelihood (NLL) is the multiclass log loss on the same held-out query split. Unless stated otherwise, the readout-grid table reports means over the 49 per-dataset seed means. 

**XGBoost reference.** The XGBoost baseline used throughout (Tab. 3, Apps. H.3, H.4, H.2) uses XGBoost 2.1.4 with 5-fold stratified-CV grid search over number of trees _∈{_ 200 _,_ 500 _,_ 1000 _}_ , maximum depth _∈{_ 3 _,_ 6 _,_ 9 _}_ , learning rate _∈{_ 0 _._ 03 _,_ 0 _._ 1 _}_ , and row subsample _∈{_ 0 _._ 7 _,_ 1 _._ 0 _}_ , with early stopping (patience 50) on a held-out 20% validation split. Categorical columns share the same ordinal encoding used for TabPFNv2/TabICLv2; the native-categorical XGBoost pathway is not used here so that the encoding is identical across baselines. The selected configuration is then refit on the full context set and evaluated on the same query partition each foundation model sees. The broader mixed16 clean-only native-categorical XGBoost control in Tab. 67 uses the same package version with the native-categorical pathway enabled; it is validation-tuned on the same 20% held-out split over depths _∈{_ 4 _,_ 6 _}_ , learning rates _∈{_ 0 _._ 03 _,_ 0 _._ 10 _}_ , and number of trees _∈{_ 300 _,_ 600 _}_ . 

**Attack hyperparameters.** 

15 

Table 7: Complete hyperparameters for the eight attacks in Table 3, fixed before any model evaluation. 

|Attack|Parameter(s)|Value|
|---|---|---|
|Noise pad|_σ_multiplier of feature std; pad fraction|4_×_;20%of_|S|_|
|Hub poison|hub fraction; centrality metric|top15%; rep-cosine kNN degree at L9|
|Centroid inj|injected counter-class examples per class|max(3_, |S|/_100)|
|Boundary poison|boundary margin (fraction of inter-centroid dist.)|0_._20|
|Mono cube|exponent of signed cube root warp|1_/_3|
|Mono softexp|soft-exponential temperature_τ_|1_._0|
|Mono rank|per-column rank transform|deterministic, ties broken by index|
|SVD hide|null-space cutoff_γ_|10<sup>_−_6</sup> _· σ_max|
|Nullspace PGD|step_η_, iters_T_, budget_ε_|0_._01,200,1_._0(in feature std units)|



### **A.5 Statistical reporting protocol** 

This subsection specifies the reporting unit, the bootstrap procedure, the permutation null, and the multiple-comparison correction used throughout the paper. 

- _Primary reporting unit = dataset_ for real-data benchmark tables. Whenever a metric is computed at the (dataset, seed) level, we first average across seeds within each dataset and then average across datasets. Because every dataset uses the same number of seeds, these values equal the corresponding dataset _×_ seed grand means printed in some appendix tables. 

- _Bootstrap CIs on benchmark deltas_ resample the vector of per-dataset paired differences after averaging seeds within dataset. This convention is used for vote decomposition, frozen probes, attacks, and transplantation (§3.4; §I.3). 

- _Single-dataset synthetic sweeps_ (e.g. the K8 collapse _f_ -sweep) use seed as the reporting unit and state this explicitly; their CIs are paired bootstraps over the seedwise difference vector. 

- _Permutation z_ for vote decomposition: 1 000 row-permutation nulls per dataset _×_ 20 seeds. 

- _Multiple comparisons_ : Bonferroni applied to the 10 regression tests; classification leaderboards report raw per-dataset numbers without family-wise correction (we report _n_ explicitly). 

### **A.6 Dataset inventory** 

This subsection lists every classification and regression dataset used in the paper, together with the exact OpenML task and dataset identifiers and the asset-provenance record. 

Table 8: All 49 classification (top) and 10 regression (bottom) datasets. _n_ is the raw row count; _d_ is the post-encoding feature count; _K_ is the number of classes. Source codes: S = synthetic, K = `sklearn` , O = OpenML. 

|src name|_n_|_d K_ src|name|_n_|_d K_ src|name|_n_|_d K_|
|---|---|---|---|---|---|---|---|---|
|O<br>`Fitness_Club`<br>|1500<br>|6<br>2 O<br><br>|`credit-g`|1000<br>|20<br>2 O<br><br>|`pc3`|1563<br>|37<br>2<br><br>|
|O<br>`Is-this-a-good-customer`<br>|1723<br>|13<br>2 O<br>|`cylinder-bands`|540<br>|37<br>2 O<br><br>|`pc4`|1458<br>|37<br>2<br><br>|
|O<br>`LED-display-domain-7digit`<br>|500<br>|7 10 O<br><br>|`diabetes`|768<br>|8<br>2 O<br><br>|`qsar-biodeg`|1054<br>|41<br>2<br><br>|
|O<br>`MIC`|1699|111<br>8 O|`dresses-sales`|500|12<br>2 S|`quadrant_2d`|250|5<br>4|
|O<br>`MiceProtein`|1080|77<br>8 O|`eucalyptus`|736|19<br>5 S|`random_labels`|150|5<br>2|
|O<br>`analcatdata_authorship`|841|70<br>4 O|`hill-valley`|1212|100<br>2 O|`red_wine`|1599|11<br>6|
|O<br>`analcatdata_dmft`|797|4<br>6 O|`ilpd`|583|10<br>2 O|`semeion`|1593|256 10|
|O<br>`anneal`|898|38<br>5 K|`iris`|150|4<br>3 S|`sign_1d`|150|5<br>2|
|O<br>`balance-scale`|625|4<br>3 O|`kc2`|522|21<br>2 O|`steel-plates-fault`|1941|33<br>2|
|O<br>`banknote-authentication`|1372|4<br>2 O|`maternal_health_risk`|1014|6<br>3 O|`tic-tac-toe`|958|9<br>2|
|O<br>`blood-transfusion-service-center`|748|4<br>2 O|`mfeat-factors`|2000|216 10 O|`vehicle`|846|18<br>4|
|O<br>`breast-w`|699|9<br>2 O|`mfeat-fourier`|2000|76 10 O|`wdbc`|569|30<br>2|
|K<br>`breast_cancer`|569|30<br>2 O|`mfeat-karhunen`|2000|64 10 O|`website_phishing`|1353|9<br>3|
|O<br>`car`|1728|6<br>4 O|`mfeat-morphological`|2000|6 10 K|`wine`|178|13<br>3|
|O<br>`climate-model-simulation-crashes`|540|20<br>2 O|`mfeat-pixel`|2000|240 10 S|`xor_2d`|250|5<br>2|
|O<br>`cmc`|1473|9<br>3 O|`mfeat-zernike`|2000|47 10||||
|O<br>`credit-approval`|690|15<br>2 O|`pc1`|1109|21<br>2||||
|_10 regression datasets (all OpenML; K omitted_|_):_||||||||
|src name|_n_|_d_<br>src|name|_n_|_d_<br>src|name|_n_|_d_|
|O<br>`Another-Dataset-on-used-Fiat-500`|1538|7<br>O|`cars`|804|17<br>O|`forest_fires`|517|12|
|O<br>`Moneyball`|1232|14<br>O|`concrete_compressive_strength`|1030|8<br>O|`healthcare_insurance_expenses`|1338|6|
|O<br>`QSAR_fish_toxicity`|907|6<br>O|`energy_efficiency`|768|8<br>O|`socmob`|1156|5|
|O<br>`airfoil_self_noise`|1503|5|||||||



**Exact OpenML identifiers.** For the OpenML assets in Table 8, `task_id` fixes the supervised task and `dataset_id` fixes the local dataset version used in our cache. 

16 



<!-- Start of picture text -->
TabPFN: linear probe TabPFN: vote-rule fidelity TabICL: per-layer  k NN<br>1.0 1.00 mean (n=49) 1.0<br>0.8 0.75 0.8<br>0.6 0.50 0.6<br>0.4 0.25 0.4<br>0.2 mean (n=49) 0.00 0.2 mean (n=49)<br>0.0 TabPFN acc. (0.86)L9 −0.25 L9 0.0 TabICL acc. (0.86)L11<br>0 3 6 9 11 0 3 6 9 11 0 3 6 9 11<br>Layer Layer Layer<br>r<br>NN acc.<br>k<br>Probe accuracy Cosine-<br>Vote--output Pearson<br><!-- End of picture text -->

Figure 2: Per-layer profiles on the 49-dataset classification suite. Thin lines: seed-averaged perdataset curves; thick black lines: means. **Left:** TabPFNv2 linear-probe accuracy converges late with a sharp jump at L8 _→_ L9. **Middle:** Pearson correlation between an attention-weighted vote read off layer _L_ and TabPFNv2’s predicted probabilities; the vote rule (§3.1) becomes faithful only at L9. **Right:** TabICLv2 cosine- _k_ NN accuracy on per-block representations runs high almost from the input. 

Table 9: Exact OpenML identifiers for the 42 classification datasets in Table 8. 

|Name|task_id|dataset_id|Name|task_id|dataset_id|Name|task_id|dataset_id|
|---|---|---|---|---|---|---|---|---|
|`Fitness_Club`|363671|46927|`cmc`|23|23|`mfeat-morphological`|18|18|
|`Is-this-a-good-customer`|363682|46938|`credit-approval`|29|29|`mfeat-pixel`|146824|40979|
|`LED-display-domain-7digit`|125921|40496|`credit-g`|31|31|`mfeat-zernike`|22|22|
|`MIC`|363711|46980|`cylinder-bands`|14954|6332|`pc1`|3918|1068|
|`MiceProtein`|146800|40966|`diabetes`|363629|46921|`pc3`|3903|1050|
|`analcatdata_authorship`|3549|458|`dresses-sales`|125920|23381|`pc4`|3902|1049|
|`analcatdata_dmft`|3560|469|`eucalyptus`|2079|188|`qsar-biodeg`|363696|46952|
|`anneal`|363614|46906|`hill-valley`|9970|1479|`red_wine`|361250|44972|
|`balance-scale`|11|11|`ilpd`|9971|1480|`semeion`|9964|1501|
|`banknote-authentication`|10093|1462|`kc2`|3913|1063|`steel-plates-fault`|146817|40982|
|`blood-transfusion-service-center`|10101|1464|`maternal_health_risk`|363685|46941|`tic-tac-toe`|49|50|
|`breast-w`|15|15|`mfeat-factors`|12|12|`vehicle`|53|54|
|`car`|146821|40975|`mfeat-fourier`|14|14|`wdbc`|9946|1510|
|`climate-model-simulation-crashes`|146819|40994|`mfeat-karhunen`|16|16|`website_phishing`|363707|46963|



Table 10: Exact OpenML identifiers for the 10 regression datasets in Table 8. 

|Name|task_id|dataset_id|Name|task_id|dataset_id|
|---|---|---|---|---|---|
|`Another-Dataset-on-used-Fiat-500`|363615|46907|`concrete_compressive_strength`|361237|44959|
|`Moneyball`|361616|41021|`energy_efficiency`|361617|44960|
|`QSAR_fish_toxicity`|361621|44970|`forest_fires`|361618|44962|
|`airfoil_self_noise`|361235|44957|`healthcare_insurance_expenses`|363675|46931|
|`cars`|361622|44994|`socmob`|361264|44987|



**Asset provenance.** Table 11 records the released artifacts used. 

|Asset class|Table 11: Asset pro<br>Identifer in this PDF|venance.<br>Terms / license status|
|---|---|---|
|OpenML datasets|Tables9and10|Upstream terms vary by dataset; we use the OpenML task<br>and dataset IDs as published.|
|scikit-learn datasets|Table8|Inherited from the scikit-learn distribution.|
|Released models|TabPFN v2 (release 7.0) and TabICL (release<br>2.0.3); see §A.1|Used as published; upstream terms apply.|



## **B TabPFN classification: full results** 

The full results for TabPFNv2 classification cover where class-readable structure forms (§B.1), what the readout computes at L9 (§B.2), and the collapse analysis built around the M0–M3 hand-crafted models (§B.3). 

17 

### **B.1 Where class-readable structure forms** 

### **B.1.1 Probe convergence** 

A logistic-regression probe trained per (dataset, layer, seed) on context-row activations and then averaged within dataset over 10 seeds gives the left panel of Figure 2. Across all 49 classification datasets, mean accuracy stays near chance through L0–L8 and then jumps from 0 _._ 467 to 0 _._ 780 between L8 and L9 (Table 12, Figure 2). The smaller L9 _→_ L10 gain is a late refinement, not a second phase transition; the sharp L8 _→_ L9 break is the crystallization point used throughout the main text. Silhouette scores rise in lockstep. 

Table 12: Linear-probe accuracy at each TabPFN block, mean and std across the same 49 classification datasets shown in Figure 2, after averaging 10 seeds within dataset. The sharp +0 _._ 31 jump from L8 to L9 marks the onset of the late-readout regime used throughout the paper; L10 is a smaller follow-on refinement. 

|Block|_−_1|0|1|2|3|4|5|6|7|8|9|10|11|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|mean acc|0_._394|0_._307|0_._320|0_._287|0_._446|0_._522|0_._445|0_._460|0_._470|0_._467|0_._780|**0**_._**798**|0_._602|
|std|0_._271|0_._209|0_._218|0_._210|0_._271|0_._277|0_._266|0_._264|0_._265|0_._246|0_._198|0_._173|0_._282|



### **B.1.2 Block knockouts (retrained probes)** 

Zeroing each MLP sub-block in turn and retraining the linear probe at L9 isolates which blocks the readable structure depends on. 

Table 13: TabPFN per-block MLP knockout: mean accuracy drop (pp) across _n_ =11 fixed classification datasets (the four synthetic controls plus `balance-scale` , `banknote-authentication` , `breast-w` , `iris` , `wine` , `vehicle` , and `steel-plates-fault` ), baseline 0 _._ 907. Block 0 dominates; blocks 5–6 are a clear secondary cluster; the remaining blocks are individually replaceable. 

|Block|0|1|2|3|4|5|6|7|8|9|10|11|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|∆acc (pp)|_−_48_._1|_−_0_._3|_−_0_._9|_−_1_._6|_−_0_._9|_−_20_._0|_−_19_._6|_−_8_._9|_−_2_._9|_−_6_._3|_−_1_._5|_−_2_._9|
|std (pp)|16_._8|0_._8|1_._1|1_._6|1_._5|15_._6|20_._1|13_._6|3_._6|7_._2|2_._6|4_._2|



Block 0 is the most damaging on 38 of 49 datasets (probe drop _≈_ 40 pp). A retrained probe still recovers labels at lower accuracy, so block 0 sets up the coordinate frame that the later blocks expect rather than destroying class information outright. 

### **B.1.3 Frozen probes after knockout (causality)** 

Frozen probes isolate the causal contribution of each block. Probes trained at _ℓ ∈ {−_ 1 _,_ 0 _,_ 3 _,_ 5 _,_ 7 _,_ 9 _,_ 10 _,_ 11 _}_ on the intact model are frozen, then evaluated under KO@ _{_ 0 _,_ 5 _,_ 9 _}_ on 49 datasets and 10 seeds. 

Table 14: Frozen-probe accuracy under block knockout. Probes trained on clean model, evaluated without retraining after zeroing block 0 (KO@0) or block 9 (KO@9). Mean and std across 49 datasets _×_ 10 seeds. 

|**Layer**|**Base**<br>Mean|**line**<br>Std|**KO**<br>Mean|**@0**<br>Std|**KO**<br>Mean|**@9**<br>Std|
|---|---|---|---|---|---|---|
|Input|0.394|0.274|0.394|0.274|0.394|0.274|
|L0|0.307|0.211|0.292|0.214|0.307|0.211|
|L3|0.446|0.274|0.415|0.253|0.446|0.274|
|L5|0.445|0.269|0.418|0.263|0.445|0.269|
|L7|0.470|0.268|0.313|0.171|0.470|0.268|
|L9|0.780|0.201|0.404|0.226|0.451|0.279|
|L10|0.798|0.175|0.499|0.255|0.496|0.241|
|L11|0.602|0.284|0.321|0.212|0.515|0.275|



18 

KO@9 drops L9 frozen-probe accuracy from 0 _._ 780 to 0 _._ 451 (∆= _−_ 0 _._ 329); KO@0 cascades to 0 _._ 404 (∆= _−_ 0 _._ 376). Two roles: block 0 sets the frame, block 9 writes the readable structure. 

### **B.1.4 Per-layer rank profile and intrinsic dimension** 

Effective rank, participation ratio (PR), and the two-NN intrinsic dimension of Facco et al. [2017], averaged over 49 classification datasets and 5 seeds (TabPFNv2-cls). TabPFNv2 bottlenecks early: rank and PR drop sharply at block 2 and recover only partially. The residual rank at the readout scales with the number of classes (mean L11 effective rank 21 _._ 6 for _C ≤_ 3 vs. 31 _._ 7 for _C ≥_ 4; Pearson _r_ = +0 _._ 64; §I.17). This is structurally different from the identical-representation collapse mode of Qu et al. [2025], tested in §B.3. 

Table 15: TabPFN per-block geometry, mean over 49 datasets _×_ 5 seeds. Block _−_ 1 is the input row before any feature mixing. 

|Block|Effective rank|PR|2NN-ID|
|---|---|---|---|
|_−_1|0_._00|0_._00|0_._00|
|0|40_._37|2_._53|4_._62|
|1|41_._97|1_._87|8_._76|
|2|10_._20|1_._47|4_._52|
|3|19_._99|2_._13|5_._76|
|4|22_._51|2_._01|5_._87|
|5|21_._69|1_._88|5_._25|
|6|23_._92|1_._82|5_._60|
|7|20_._56|1_._87|5_._33|
|8|16_._15|1_._99|4_._86|
|9|22_._81|2_._80|4_._41|
|10|25_._28|2_._98|4_._48|
|11|25_._21|3_._25|4_._77|



### **B.2 What the readout computes at** L9 

### **B.2.1 Row-attention sharpness** 

L9 is the sharpest layer on all 49 datasets: mean max weight 0 _._ 925 (std 0 _._ 078), mean entropy 1 _._ 756 versus 3 _._ 126 at L0. IQR of max weights is [0 _._ 89 _,_ 0 _._ 97]; only two datasets fall below 0 _._ 80. 

### **B.2.2 Target-column attention vs. similarity** 

L9 attention tracks input similarity but is not reducible to it. The Spearman rank correlation between the L9 attention weight on each context row (restricted to the target column) and input-space _ℓ_ 2 similarity is positive on 45 of 49 datasets (median _r_ = 0 _._ 32, IQR [0 _._ 20 _,_ 0 _._ 48], tail to _r >_ 0 _._ 70). The representation-distance regression of §I.9, summarized in Table 16, sharpens this picture: input distance alone explains only a small fraction of the L9 attention pattern, L8 representation distance more than doubles that variance, and the full learned-similarity model nearly triples it. 

Table 16: _R_<sup>2</sup> of three predictors of L9 row-attention, across 49 classification datasets (5 seeds each). Input _ℓ_ 2: linear regression on raw input distance only. Rep _ℓ_ 2 (L8): the same with L8 representation distance. Full: learned bilinear similarity from L8. Median, IQR, mean, and number of datasets exceeding _R_<sup>2</sup> = 0 _._ 5. 

|Predictor|median|q25|q75|mean|_n_(_R_<sup>2</sup>_>_0_._5)|_/_49|
|---|---|---|---|---|---|---|
|Input_ℓ_2|0_._168|0_._051|0_._345|0_._218||2|
|Rep_ℓ_2 (L8)|0_._488|0_._339|0_._627|0_._477||21|
|Full learned sim.|0_._606|0_._487|0_._728|0_._600||35|



19 

### **B.2.3 Vote decomposition (all 49 datasets)** 

Vote is 



with _αi_<sup>(</sup><sup>_h_)</sup> the L9 attention weight from the query to context row _i_ in head _h_ . Evaluated on 49 datasets (20 seeds each); the random-labels condition is a negative control. 

Table 17: Vote decomposition: attention-weighted label vote at L9 vs. final output. Pearson _r_ , Spearman _ρ_ , and null-shuffle _z_ -score per dataset (20 seeds each). Mean across all 49 datasets: Pearson _r_ = 0 _._ 890 _±_ 0 _._ 114, Spearman _ρ_ = 0 _._ 849 _±_ 0 _._ 138, _z_ = 6 _._ 79 _±_ 3 _._ 79. 

|**Dataset**|**Pearson**_r_|**Spearman**_ρ_|_z_**-score**|
|---|---|---|---|
|Fitness_Club|0.982|0.989|7.32|
|Is-this-a-good-customer|0.976|0.992|1.46|
|LED-display-domain-7digit|0.820|0.839|9.82|
|MIC|0.974|0.727|8.95|
|MiceProtein|0.799|0.665|11.58|
|analcatdata_authorship|0.960|0.737|10.70|
|analcatdata_dmft|0.495|0.451|1.99|
|anneal|0.935|0.774|12.90|
|balance-scale|0.919|0.888|6.52|
|banknote-authentication|0.971|0.807|2.44|
|blood-transfusion-service-center|0.963|0.977|5.13|
|breast-w|0.984|0.955|3.81|
|breast_cancer|0.978|0.938|3.53|
|car|0.913|0.900|13.51|
|climate-model-simulation-crashes|0.976|0.979|6.09|
|cmc|0.900|0.861|6.74|
|credit-approval|0.966|0.957|2.60|
|credit-g|0.971|0.976|6.36|
|cylinder-bands|0.896|0.918|4.49|
|diabetes|0.983|0.991|6.95|
|_(First 20 of 49 datasets)_||||



Population means: Pearson _r_ = 0 _._ 888 _±_ 0 _._ 114, Spearman _ρ_ = 0 _._ 848 _±_ 0 _._ 139, _z_ = 6 _._ 88 _±_ 3 _._ 77 (1 000 row-permutation nulls). Worst cases are ten-class digit datasets (mfeat-zernike _r_ = 0 _._ 71, mfeat-factors 0 _._ 71, semeion 0 _._ 76): vote distributes mass over class indicators while accuracy uses an arg max. The optimal head-reweighting NNLS score gives an upper bound on what a fixed convex recombination of head votes buys. 

**Multi-class fidelity scaling.** Per-dataset vote-fidelity _<u>r</u>_ decreases monotonically with class count _C_ across the 49-dataset population: linear regression of _<u>r</u>_ on _C_ yields slope _−_ 0 _._ 016 per class (intercept 0 _._ 957, Pearson _−_ 0 _._ 43, Spearman _ρ_ = _−_ 0 _._ 60, _p <_ 10<sup>_−_5</sup> , _n_ = 49), with binary-class mean 0 _._ 93, mid-class (3 _≤C≤_ 8) mean 0 _._ 88, and ten-class mean 0 _._ 80. The ten-class digit cluster ( _r∈_ [0 _._ 71 _,_ 0 _._ 76]) lies within the residual spread of this trend: the multi-class gap reflects the vote rule’s finite per-class indicator capacity rather than a mechanism switch in TabPFNv2. 

### **B.2.4 Head- and layer-resolved vote** 

Table 18: Per-head vote Pearson _r_ at L9. Each head’s vote is computed independently; mean across heads shown in last column. Results show no single head dominates the vote signal (10 datasets _×_ 20 seeds). 

|**Dataset**|**H0**|**H1**|**H2**|**H3**|**H4**|**H5**|**Mean**|
|---|---|---|---|---|---|---|---|
|balance-scale|0.937|0.795|0.872|0.935|0.762|0.883|0.864|
|breast_cancer|0.988|0.166|0.889|0.968|0.600|0.979|0.765|
|credit-g|0.951|-0.432|0.615|0.910|0.820|0.938|0.634|
|diabetes|0.977|0.592|0.735|0.975|-0.400|0.973|0.642|
|ilpd|0.982|0.754|0.413|0.988|-0.624|0.805|0.553|
|iris|0.990|0.843|0.968|0.987|-0.085|0.968|0.779|
|mfeat-zernike|0.818|0.403|0.526|0.750|0.052|0.642|0.532|
|tic-tac-toe|0.820|-0.113|0.772|0.780|0.307|0.699|0.544|
|vehicle|0.887|0.149|0.760|0.873|0.109|0.779|0.593|
|wine|0.973|0.921|0.961|0.974|0.353|0.950|0.855|



20 

The six L9 heads span 0 _._ 05–0 _._ 10 Pearson units around the head-mean. The layer profile is flat through L8, jumps at L9, and is sustained through L11. The largest single-head zeroing drop is 3 _._ 9 pp at L9 (§I.16); the L9 peak is real and distributed across heads. 

### **B.2.5 Causal attention forcing** 

Replacing L9 softmax weights with 1 _/N_ train collapses accuracy from 0 _._ 874 to 0 _._ 488 (∆= _−_ 0 _._ 386), matching the majority-class baseline (0 _._ 498, §I.11). MLP knockout at any block also collapses accuracy to the same vicinity (§E.5): L9 attention is one part of a working circuit, not the unique site of computation. 

### **B.2.6** _k_ **NN agreement vs. TabPFN** 

Agreement between TabPFNv2’s prediction and _k_ NN under input _ℓ_ 2, L9 rep _ℓ_ 2, L9 rep cosine, and L9 Mahalanobis, _k ∈{_ 1 _,_ 3 _,_ 5 _}_ , on 49 datasets and 10 seeds (Table 19, top). Input-space _k_ NN reaches 0 _._ 78–0 _._ 81 across _k_ ; representation-space _ℓ_ 2/cosine _k_ NN at L9 reaches only 0 _._ 54–0 _._ 56, and Mahalanobis is worse (0 _._ 37–0 _._ 41). Strengthening the test by reading from L11 (K1) does not change the picture (Table 19, middle). Even the strongest learned-metric variants (NCA-trained _ℓ_ 2, shrunk Mahalanobis, soft- _k_ NN with a fitted temperature, all at L9) max out around 0 _._ 84 (Table 19, bottom). A 1-hidden-layer GELU MLP head fit on the frozen final representation (§B.3.8) also fails to recover native accuracy on TabPFNv2 ( _−_ 24 _._ 9 pp paired, 95% bootstrap CI [ _−_ 32 _._ 5 _, −_ 18 _._ 0], worse on 38 _/_ 42 shared datasets) while almost recovering it on TabICLv2 ( _−_ 0 _._ 8 pp, [ _−_ 1 _._ 5 _, −_ 0 _._ 3]): a non-linear head does not close the asymmetry. The model is not a _k_ NN in its own representation. The accurate framing is the learned similarity of §I.9, where rep- _ℓ_ 2 at L8 explains _R_<sup>2</sup> _≈_ 0 _._ 48 of the L9 attention pattern. 

Table 19: Agreement between TabPFN’s argmax prediction and a _k_ NN baseline under several distance metrics, mean across 49 datasets. Top: L9 representations. Middle: L11 representations (§I.4). Bottom: learned-metric variants at L9. Every representation-space variant agrees with TabPFN on a strict minority of test points; only soft- _k_ NN with a fitted temperature crosses 0 _._ 8. 

|Metric|_k_ = 1|_k_ = 3|_k_ = 5|
|---|---|---|---|
|_Input space and L9 repre_|_sentation_|_s (49 da_|_tasets)_|
|Input_ℓ_2|0_._781|0_._801|0_._813|
|L9 rep_ℓ_2|0_._540|0_._556|0_._561|
|L9 rep cosine|0_._540|0_._556|0_._562|
|L9 Mahalanobis|0_._367|0_._393|0_._414|
|_L11 representations (K1,_|_49 datas_|_ets)_||
|L11 rep_ℓ_2|0_._528|0_._554|0_._561|
|L11 rep cosine|0_._528|0_._554|0_._561|
|L11 Mahalanobis|0_._356|0_._378|0_._395|
|_Learned-metric L9 varia_|_nts (L1, 4_|_9 datase_|_ts)_|
|NCA-trained_ℓ_2|0_._723|0_._734|0_._742|
|Shrunk Mahalanobis|0_._632|0_._683|0_._717|
|Soft-_k_NN (_ℓ_2, ftted_T_)|0_._837|0_._837|0_._837|
|Soft-_k_NN (cos, ftted_T_)|0_._813|0_._813|0_._813|



### **B.2.7 Activation patching** 

Activation patching localizes the sites that carry the corruption. TabPFNv2 blocks _{_ 0 _,_ 5 _,_ 9 _}_ and TabICLv2 blocks _{_ ColEmb-2, ICL-0, ICL-6, ICL-11 _}_ are patched on 10 datasets and 5 seeds, under block-0 KO and label-shuffle corruptions (Table 20). Every TabPFN block patch and every TabICL ICL-block patch recovers to baseline (recovery = 1 _._ 00). The single exception is TabICLv2’s ColEmb-2 under label-shuffle (recovery _≈_ 0 _._ 04), consistent with ColEmb-2’s role as the compression bottleneck (§2.3). TabICL ICL blocks are unaffected by upstream block-0 KO because the columnembedding stack runs before the ICL stack and absorbs the perturbation, so corrupt and baseline are equal there (recovery is by definition 1 _._ 00). 

21 

Figure 3: **Where representation collapse appears in the public models.** Balancescale accuracy (20 seeds, mean _±_ s.d.) for each architecture (blue) and under symmetry-restoring ablations: removing TabPFNv2’s positional matrix or TabICLv2’s RoPE produces no measurable mean accuracy loss on this benchmark (green); disabling TabPFNv2’s pair channel or TabPFN v1’s RoPE reproduces the predicted collapse (red). 



<!-- Start of picture text -->
released free ablation breaks model<br>1.0 0.96 0.96 0.98 0.98 0.97<br>0.89<br>0.8<br>0.6 0.52<br>0.4 chance (3 cls)<br>0.2<br>0.0<br>TabPFNv2 TabPFNv2 TabPFNv2 TabICLv2 TabICLv2 TabICLv1 TabICLv1<br>(released) W  = 0pair channel off(released) no-RoPE (released) no-RoPE<br>balance-scale acc.\ (20 seeds)<br><!-- End of picture text -->

Table 20: Activation patching: replace activations at the named site in a corrupt run with those from a clean run, then read out accuracy. Mean across 10 datasets, 5 seeds. _Recovery_ = (patched _−_ corrupt) _/_ (baseline _−_ corrupt); 1 _._ 00 means the patched site fully restores baseline accuracy. 

|Site|Corruption|baseline|corrupt|patched|recovery|
|---|---|---|---|---|---|
|TabPFN B0|label-shuffe|0_._874|0_._497|0_._874|1_._000|
|TabPFN B5|label-shuffe|0_._874|0_._497|0_._874|1_._000|
|TabPFN B9|label-shuffe|0_._874|0_._497|0_._874|1_._000|
|TabICL ICL-0|label-shuffe|0_._891|0_._502|0_._891|1_._000|
|TabICL ICL-6|label-shuffe|0_._891|0_._502|0_._891|1_._000|
|TabICL ICL-11|label-shuffe|0_._891|0_._502|0_._891|1_._000|
|TabICL ColEmb-2|label-shuffe|0_._891|0_._502|0_._538|0_._043|
|TabPFN B0|block-0 KO|0_._874|0_._443|0_._874|1_._000|
|TabPFN B5|block-0 KO|0_._874|0_._443|0_._874|1_._000|
|TabPFN B9|block-0 KO|0_._874|0_._443|0_._874|1_._000|
|TabICL ICL-0|block-0 KO|0_._891|0_._892|0_._891|1_._000|
|TabICL ICL-6|block-0 KO|0_._891|0_._892|0_._891|1_._000|
|TabICL ICL-11|block-0 KO|0_._891|0_._892|0_._891|1_._000|



### **B.3 Representation collapse: stress tests, screen, and hand-crafted models** 

### **B.3.1 Stress test on collapse-prone data** 

Identical column marginals are the canonical setting in which TabPFNv2 and TabICLv2 risk producing indistinguishable row representations (cf. Qu et al., 2025). The stress test runs on three datasets: balance-scale; a synthetic identical-marginal benchmark ( _n_ = 1000, _m_ = 20, fraction _f ∈{_ 0 _,_ 0 _._ 25 _,_ 0 _._ 5 _,_ 0 _._ 75 _,_ 1 _._ 0 _}_ of columns drawn i.i.d. from the same 5-level discrete distribution, label a fixed nonlinear function of a labeled subset); and post-hoc real-data candidates from the JS-divergence screen of §I.6. The measured quantities are pairwise pre-readout cosine similarity, collapse rate (cross-class pairs with similarity _>_ 0 _._ 99), and KMeans- _C_ purity over 5 seeds. The conditions are: TabICLv1 (full vs. no-RoPE); TabICLv2 (full vs. no-RoPE vs. grouping-disabled); TabPFNv2 (full vs. _W_ = 0 vs. per-group feature count fixed to 1); and random-init controls. The full _f_ -sweep with 10-seed paired-bootstrap CIs is in §I.7. 

### **B.3.2 Marginal-similarity vs. invariance-cost correlation** 

Marginal similarity predicts the accuracy cost of removing the symmetry-breaker. For each of 49 datasets, pairwise JS divergence between column marginals identifies the maximum identical-marginal column set via thresholded clustering, and the accuracy gap between the full model and a strict permutation-invariant variant (TabPFNv2 _W_ = 0, TabICLv1 no-RoPE, TabICLv2 no-RoPE) gives the invariance cost. The prediction is a positive Spearman _ρ_ on configurations whose ablation removes the symmetry-breaker, and _≈_ 0 otherwise. Companion _ρ_ at JS tolerances _τ ∈{_ 0 _._ 01 _,_ 0 _._ 05 _,_ 0 _._ 10 _}_ are reported in §I.6. 

22 

### **B.3.3 Hand-crafted models: setup** 

The hand-crafted M0–M3 models referenced in §5.1 factor each architecture’s symmetry-breaking devices into a minimal single-layer setting. 

Figure 4: **Hand-crafted models that isolate the three symmetry-breaking devices.** M0– M3 are single attention layers over _m_ =3 binary features (M0: shared-cell mean-pool; M1: column identifier; M2: per-column attention mask; M3: pair tokens). Transductive accuracy on Task A ( _y_ = _x_ 1), B ( _y_ = _x_ 1 _⊕ x_ 2), C ( _y_ =maj( _x_ )). M0 is capped at the 0 _._ 75 multiset bound; M1, M2, M3 reach 1 _._ 0 on A; only M3 (pair tokens) breaks the bound on B; all four solve C. Derivations below. 



<!-- Start of picture text -->
M0 (mean-pool) M1 (content) M2 (structure) M3 (pair-group)<br>1.001.001.00 1.00 1.001.001.001.00<br>1.00<br>0.75 0.75 0.750.750.75 multiset bound (0.75)<br>0.50<br>0.25<br>0.00<br>A:  y  =  x 1 B:  y  =  x 1⊕ x 2 C:  y  = maj( x )<br> = 3)<br>m<br>transductive accuracy (<br><!-- End of picture text -->

The construction isolates the three symmetry-breaking devices (column identifier, column mask, pair grouping) on a task family where column-marginal information alone is useless. Let _x ∈{_ 0 _,_ 1 _}_<sup>_m_</sup> and enumerate all 2<sup>_m_</sup> rows. Three tasks: 

- Task A: _y_ = _x_ 1 (one column is the label). 

- Task B: _y_ = _x_ 1 _⊕ x_ 2 (two-column XOR). 

- Task C: _y_ = majority( _x_ 1 _, . . . , xm_ ) (multiset-decidable control). 

All features have identical Bernoulli(1 _/_ 2) marginals. Each model is a single attention layer plus hard-argmax readout; transductive accuracy on the full 2<sup>_m_</sup> -row enumeration with _m_ = 3. Numbers are exact rationals derived analytically and reproduced numerically; Table 21 records the verification. 

### **B.3.4 The multiset bound (Lemma 1)** 

**Lemma 1 (multiset bound, general orbit-majority form).** _Let r_ : _{_ 0 _,_ 1 _}_<sup>_m_</sup> _→R be any row representation invariant to all column permutations: r_ ( _x_ ) = _r_ ( _σx_ ) _for every σ ∈ Sm. Let g_ : _R → {_ 0 _,_ 1 _} be any deterministic readout, and let y_ : _{_ 0 _,_ 1 _}_<sup>_m_</sup> _→{_ 0 _,_ 1 _} be any deterministic binary label rule. Then transductive accuracy of g ◦ r on the_ 2<sup>_m_</sup> _-row enumeration is at most_ 



_i.e. the orbit-wise majority of the label rule. The bound reduces to the formula previously stated for Task A (y_ ( _x_ ) = _x_ 1 _); for m_ = 3 _binary targets it equals_ 0 _._ 75 _and is exactly attained by_ M0 _._ 

_Proof sketch. Sm_ -invariance partitions _{_ 0 _,_ 1 _}_<sup>_m_</sup> into orbits on which _r_ , and hence any deterministic readout _g ◦ r_ , is constant. Within an orbit _O_ the readout cannot do better than the majority of _{y_ ( _x_ ) : _x ∈ O}_ , contributing max _y_ 0 _|{x ∈ O_ : _y_ ( _x_ ) = _y_ 0 _}|_ correct predictions. Summing over the four _m_ = 3 orbits _{_ 000 _}, {_ 001 _,_ 010 _,_ 100 _}, {_ 011 _,_ 101 _,_ 110 _}, {_ 111 _}_ for _y_ ( _x_ ) = _x_ 1 gives 1 + 2 + 2 + 1 = 6 _/_ 8 = 0 _._ 75. □ 

**Proposition 2 (Task B corollary).** _Any model with column-permutation-invariant row representation is capped at_ 0 _._ 75 _on Task B for m_ = 3 _, even with an optimal linear readout_ : orbit _{_ 001 _,_ 010 _,_ 100 _}_ has XOR values _{_ 1 _,_ 1 _,_ 0 _}_ and _{_ 011 _,_ 101 _,_ 110 _}_ has _{_ 1 _,_ 0 _,_ 1 _}_ , so orbit-wise majority again caps at 6 _/_ 8. The route assignment in Section 5.1 maps the M3 (pair-grouped) construction to the device that breaks this bound; M3 solves Task B at _m_ = 3 and is the smallest sufficient construction exhibited here. 

### **B.3.5 M0–M3 model definitions** 

**M0 (shared embedding + mean-pool).** _ϕ_ ( _b_ ) = _b_ ; _r_ ( _x_ ) = _m_ <u>1</u> � _j_<sup>_xj_;hard-NN readout.</sup><sup>_r_is</sup><sup>_Sm_-</sup> invariant. On Task A with _m_ = 3 each three-row orbit ties two labels against one, costing one error per orbit; transductive accuracy 6 _/_ 8 = 0 _._ 75, exactly meeting the bound. 

23 

**M1 (cell tokens with column identifier).** TabPFNv2-style: token _tij_ = [ _vxij_ ; _ej_ ; _ℓyi_ ] with column identifier _ej_ . With _WQ_ = _WK_ extracting (value, column) and cell-level scores _s_ = **1** [value match] + **1** [column match], argmax-with-ties readout followed by per-row mean over the _m_ cells gives _p_ ˆ(100) = (1 _/_ 3 _,_ 2 _/_ 3). Task A accuracy 1 _._ 0. Task B unsolved (single-cell attention cannot condition on two columns jointly). 

**M2 (column-stream attention without identifiers).** TabICLv1-style: bipolar shared embedding _ϕ_ ( _b_ ) = 2 _b −_ 1, per-column heads _WQ_<sup>(</sup><sup>_j_)</sup> = _WK_<sup>(</sup><sup>_j_)</sup> = _e_<sup>_⊤_</sup> _j_<sup>(mask = column structure), per-row mean</sup> over _m_ heads. M2 = M1 on this family: restricting M1 attention to matching columns and M2’s per-column head yield identical softmaxes, and per-row means coincide. Task A 1 _._ 0, Task B 0 _._ 5. 

**M3 (pair-grouped tokens).** v2-grouping: each super-cell over an unordered feature pair ( _j, k_ ) carries the joint value _u_ ( _j,k_ ) = 2 _xj_ + _xk ∈{_ 0 _,_ 1 _,_ 2 _,_ 3 _}_ and a pair identifier _e_ ( _j,k_ ). On Task A with _m_ = 3, the super-cell at ( _x_ 1 _, x_ 2) = (1 _,_ 0) appears in five positives and one negative, so _p_ ˆ(100) = (1 _/_ 6 _,_ 5 _/_ 6) and Task A accuracy is 1 _._ 0. Conditioning on the joint value of ( _x_ 1 _, x_ 2) separates XOR; Task B accuracy 1 _._ 0. 

### **B.3.6 Verification table and architecture mapping** 

Table 21: Transductive accuracy of the four handcrafted models on Tasks A, B, C with _m_ = 3, verified numerically. 

|Model|Task A|Task B|Task C|
|---|---|---|---|
|M0 naïve|0_._75|0_._75|1_._00|
|M1 TabPFN-style|1_._00|0_._50|1_._00|
|M2 TabICL-style|1_._00|0_._50|1_._00|
|M3 pair-grouped|1_._00|1_._00|1_._00|



Table 22: Route assignment per architecture. Each model carries two of the three routes (M1: cell+colID; M2: column-stream; M3: pair-group); ablating one route leaves the other intact, predicting the small empirical ablation costs measured in §4. 

|Architecture|Routes|Missing route|
|---|---|---|
|TabPFN v2|M1+M3|column-stream (M2)|
|TabICL v1|M2+within-row RoPE|pair-group (M3)|
|TabICL v2|M2+M3|cell+col-ID (M1)|
|Mitra|shared embedding+no PE|— (column-invariant by construction)|



Mapping: TabPFNv2 pairs M1 (random per-feature identifiers) with M3 (pair grouping); TabICLv1 pairs M2 (column-stream attention) with RoPE; TabICLv2 pairs M2 with M3 (circular grouping) and adds target-aware embeddings. Each v2 architecture carries _two_ of the three routes; TabICLv1 carries only one. This predicts the empirical pattern of §5.2: TabPFNv2 with _W_ = 0 and TabICLv2 without RoPE both show no measurable accuracy loss, while disabling TabPFNv2’s pair channel costs _−_ 7 pp on balance-scale and _−_ 14 pp under full identical-marginal stress (§I.7); TabICLv1 without RoPE loses _−_ 44 pp on balance-scale and 15 pp on the benchmark mean. The load-bearing v2 mitigation is the M3 pair/circular-grouping route. 

### **B.3.7 Attention-mode ablation** 

A sweep over shared-embedding, oracle-attention, and attention-mode combinations (row-level ICL, cell-level cross-attention, column-stream) on Tasks A–C under leave-one-out separates the contributions of each mechanism. Row-level ICL on collapsed representations caps at 0 _._ 625 on Task A LOO; cell-level cross-attention without column identifiers drops to 0 _._ 25; column-stream reaches 1 _._ 0 on Task A even with shared embeddings. Task B remains at chance for every single-layer mode, consistent with XOR requiring at least one nonlinear interaction. 

24 

### **B.3.8 Fresh non-linear head on the frozen final representation** 

A 1-hidden-layer GELU MLP head ( _d_ hidden matched to the original decoder’s first hidden width: 192 for TabPFNv2, 512 for TabICLv2) is retrained on each backbone’s frozen final representation, using the context split for fitting and the held-out query split for accuracy (LBFGS / AdamW; protocol matches §A.2). The linear-head row of §3.4 is the same protocol with _d_ hidden=0 (a bare logistic layer). Across the 42 shared datasets where both methods completed, the linear head reaches mean 0 _._ 582 on TabPFNv2 ( _−_ 26 _._ 5 pp vs. native, 95% paired bootstrap CI [ _−_ 34 _._ 7 _, −_ 19 _._ 1], worse on 39 _/_ 42) and the MLP head reaches 0 _._ 598 ( _−_ 24 _._ 9 pp, [ _−_ 32 _._ 5 _, −_ 18 _._ 0], worse on 38 _/_ 42, exact two-sided binomial sign test _p ≈_ 3 _×_ 10<sup>_−_8</sup> ); on TabICLv2 the same heads reach 0 _._ 849 ( _−_ 0 _._ 7 pp, [ _−_ 1 _._ 2 _, −_ 0 _._ 3]) and 0 _._ 847 ( _−_ 0 _._ 8 pp, [ _−_ 1 _._ 5 _, −_ 0 _._ 3]). The non-linear head gains roughly 1 _._ 6 pp on TabPFNv2 relative to the linear head while leaving a _∼_ 25 pp gap to native, and is statistically indistinguishable from the linear head on TabICLv2. The fresh-head asymmetry between the two backbones survives the addition of a single-hidden-layer non-linearity. 

## **C TabICL classification: full results** 

This appendix collects the full TabICLv2 classification evidence behind the main-text claims of §2 and §3.3: column-embedding ablations, ICL-block knockouts and uniform-attention sweeps, label perturbations, the per-layer linear and _k_ NN probes, and the readout grid that compares the end-toend head against per-layer prototype, _k_ NN, and per-block-vote decoders, including a cross-version comparison over the released TabICL v1, v1.1, and v2 classifiers. 

### **C.1 Where the class-readable representation is laid down** 

### **C.1.1 ColEmbedding knockouts** 

Zeroing each ColEmbedding block (ColEmb-0/1/2) on 49 datasets: ColEmb-2 is most damaging on 45 of 49. ColEmb-2 is the SetTransformer readout that compresses per-feature into per-sample vectors. 

Table 23: TabICL ColEmbedding-block knockouts: mean accuracy drop across 49 classification datasets, baseline 0 _._ 86. 

|Knockout|∆acc|most-damaging|datasets|
|---|---|---|---|
|ColEmb-0|_−_0_._03||4 / 49|
|ColEmb-1|_−_0_._01||0 / 49|
|ColEmb-2|_−_0_._28||45 / 49|



**ColEmb-2 sets the coordinate frame.** To mirror the TabPFN block-0/block-9 analysis (§B.1.3) we evaluate two probe types under each early-block knockout: a _frozen_ probe trained on the intact model and applied without retraining, and a _retrained_ probe fit from scratch on the post-knockout train representations. Probes are read off at L _−_ 1 (input to the ICL stack) and L11 (final ICL block); 49 classification datasets, 5 seeds. Knocking out ColEmb-2 collapses the final accuracy by 28 pp and the frozen probe at L _−_ 1 by 26 pp, but the retrained probe at L _−_ 1 loses only 6 pp (0 _._ 812 _→_ 0 _._ 752, well above the multi-class chance level). Class information survives the perturbation; the downstream ICL stack cannot read the post-KO frame. At L11 the retrained probe (0 _._ 602) sits below the frozen probe (0 _._ 704) under ColEmb-2 KO, so the late representation no longer organizes class structure linearly even when refit. ColEmb-0 and ColEmb-1 produce the same qualitative pattern (frozen _≪_ retrained at L _−_ 1) at much smaller magnitude, consistent with their role as upstream contributors to the same coordinate-setting role. Knocking out a single ICL block (L0, L5, or L11) leaves both probe types within 3 pp of baseline, matching the redundancy reported in §3.3. 

25 

Table 24: TabICL frozen-vs-retrained probe under each early-block knockout, mean across 49 classification datasets _×_ 5 seeds. Frozen probes are trained on the intact model and applied without retraining; retrained probes are fit from scratch on post-knockout train representations. L-1 is the input to the ICL stack (output of ColEmb+RowInteraction); L11 is the final ICL block. 

||||**Frozen**|**probe**|**Retrain**|**ed probe**|
|---|---|---|---|---|---|---|
|**Knockout**|**Final acc**|∆|L_−_1|L11|L_−_1|L11|
|Baseline (intact)|0.864|—|0.812|0.859|0.812|0.859|
|ColEmb-0|0.836|_−_0_._028|0.583|0.827|0.787|0.824|
|ColEmb-1|0.849|_−_0_._014|0.655|0.843|0.806|0.843|
|ColEmb-2|0.583|_−_0_._281|0.555|0.704|0.752|0.602|
|ICL-L0|0.860|_−_0_._004|0.812|0.854|0.812|0.852|
|ICL-L5|0.845|_−_0_._018|0.812|0.830|0.812|0.827|
|ICL-L11|0.863|_−_0_._001|0.812|0.855|0.812|0.859|



### **C.1.2 RowInteraction skips** 

Skipping each RowInteraction block individually leaves the ICL-input linear probe at 81%, essentially unchanged from the intact representation: RowInteraction does not create the class structure. Two further checks corroborate this. The per-layer _ℓ_ 2- _k_ NN curve of Tab. 25 already sits within 4 pp of its L11 endpoint at L0, immediately after the first row-interaction attention. The uniform-attention combinations in §C.2.3 cost 5–13 pp on average when applied to contiguous halves of the stack. Both align with the main-text claim that removing between-row mixing in the column-embedding network costs only a few points. 

### **C.1.3 ICL input probe by dataset** 

We measure class-readability per layer with _ℓ_ 2- _k_ NN agreement on the representations rather than with geometry summaries: the TabICL stack does not bottleneck (the rank/PR profile in §2.2 is monotone-increasing through most of the stack) so geometry alone is uninformative about _when_ class structure becomes readable, while _k_ NN agreement directly tracks the quantity the prototype/ _k_ NN readout (§3.3) consumes. Across 49 datasets the kNN-on-representation accuracy is already 0 _._ 824 at the output of the first ICL block (L0), within 4 pp of end-to-end (0 _._ 864); the curve gains _≤_ 0 _._ 04 over the entire stack. 

Table 25: Per-layer _ℓ_ 2- _k_ NN on TabICLv2 representations across 49 datasets ( _k_ =5, mean _±_ std across datasets). TabICLv2 baseline 0 _._ 864 _±_ 0 _._ 153. Class structure is laid down by L0 and only mildly sharpened. 

|Layer|L0|L1|L2|L3|L4|L5|L6|L7|L8|L9|L10|L11|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|_k_NN acc|.824|.825|.827|.829|.835|.832|.837|.837|.847|.849|.855|.856|
|agreement|.885|.885|.890|.886|.897|.894|.901|.902|.923|.924|.935|.936|



Per-dataset peak-layer (argmax of _k_ NN accuracy across L0–L11): median peak L10; only 5 _/_ 49 datasets peak at L0 and 1 _/_ 49 at L1, while 27 _/_ 49 peak at L10 or L11. Class structure is laid down by L0 (within 4 pp of end-to-end on average) and only mildly sharpened by the rest of the stack, consistent with the qualitative claim in the main text. 

### **C.2 Attention and label interventions** 

### **C.2.1 Label interventions** 

Five context-set label perturbations are applied at inference on the 49-dataset classification suite. Tab. 26 reports the per-dataset change in TabICLv2 accuracy and in the linear probe of the ICL representation after each perturbation: labels are essential, their specific learned directions matter, and they act as class identities rather than positional indices. 

26 

Table 26: Label-perturbation effects on TabICLv2-cls. ∆ acc is the change in end-to-end accuracy; probe is the ICL-representation linear probe accuracy after the perturbation. 

|Intervention|∆acc|probe|note|
|---|---|---|---|
|Remove (zero)|_−_44pp|0_._81_→_0_._12|labels are essential|
|Cyclic shift_k→k_+1|_−_0_._01pp (med.)|unchanged|41_/_49within_±_1pp|
|Two-class swap|_−_76pp<sup>_†_</sup>|—|_†_on swapped classes only|
|Random Gaussian|_→_chance|collapsed|learned directions matter|
|Shuffe within context|_→_chance|collapsed|order-agnostic role broken|



### **C.2.2 Single-block uniform attention** 

Single-block and two-block uniform row-attention leaves TabICLv2 within a few points of baseline. We force uniform row-attention at one ICL block (block 0 or block 11) or at one sliding two-block window (blocks _X_ , _X_ +1 for _X_ =0 _. . ._ 10). Across 49 datasets and 10 seeds, no single block and no two-block window costs more than a few points on average; at least 46 _/_ 49 datasets stay within _|_ ∆ _| <_ 1 pp at every window. The matching TabPFNv2 L9 uniform-attention drop is _−_ 0 _._ 39 (§B.2.5). 

Table 27: Single-block (top) and sliding two-block (bottom) uniform row-attention drops on TabICLv2, 49 datasets, 10 seeds. Drop is baseline _−_ accuracy; the rightmost column counts datasets with _|_ ∆ _|<_ 0 _._ 01. 

|Block(s)|mean drop|median|worst|n(_≤_1pp)/49|
|---|---|---|---|---|
|b0|+0_._023|+0_._009|+0_._180|39|
|b11|+0_._002|+0_._000|+0_._028|47|
|pair 0–1|+0_._039|+0_._013|+0_._217|49|
|pair 1–2|+0_._023|+0_._007|_−_0_._031|48|
|pair 2–3|+0_._018|+0_._007|_−_0_._015|48|
|pair 3–4|+0_._014|+0_._006|_−_0_._015|48|
|pair 4–5|+0_._007|+0_._003|_−_0_._026|47|
|pair 5–6|+0_._005|+0_._001|_−_0_._019|47|
|pair 6–7|+0_._004|+0_._002|_−_0_._022|48|
|pair 7–8|+0_._010|+0_._004|_−_0_._020|48|
|pair 8–9|+0_._015|+0_._007|_−_0_._022|46|
|pair 9–10|+0_._022|+0_._011|_−_0_._019|47|
|pair 10–11|+0_._017|+0_._008|_−_0_._019|47|



### **C.2.3 Multi-block uniform attention** 

Forcing uniform attention on {first-6, last-6, all-12} blocks (49 datasets, 10 seeds, plus sign_1d): 

Table 28: TabICL accuracy under simultaneous uniform-attention knockout of contiguous block groups. Mean accuracy across 49 datasets _×_ 10 seeds, with the `sign_1d` synthetic control reported separately. Neither half of the stack alone breaks the model; only the all-12 condition collapses both real-data and synthetic accuracy. 

|Condition|Mean accuracy|`sign_1d`|
|---|---|---|
|Baseline|0_._862|0_._993|
|First 3 blocks|0_._810|—|
|Last 3 blocks|0_._804|—|
|First 6 blocks|0_._789|—|
|Last 6 blocks|0_._737|—|
|All 12 blocks|0_._572|0_._533|



Last-6 is more damaging than first-6; neither half alone breaks the model. All-12 drops accuracy by _−_ 29 pp and collapses sign_1d from 0 _._ 993 to 0 _._ 533. Excluding sign_1d, all-12 has max 0 _._ 777, median 0 _._ 290 across 48 datasets, above the majority baseline (mean 0 _._ 498, §I.11). The blocks are collectively necessary and individually redundant. 

27 

### **C.2.4 MLP knockouts** 

Zeroing one ICL MLP (second linear layer) per block (49 datasets, 5 seeds). Mean accuracy drop _≤_ 2 _._ 0 pp at every block; the single largest per-dataset drop anywhere in the stack is 8 _._ 7 pp (block 7) and 29 _._ 9 pp (block 8 on one outlier). No single MLP is critical. Regression shows a different pattern at L11 (§F.4). 

|Table 29: Per-block ICL-MLP<br>and median drops are sub-perc<br>Block|knockout dr<br>ent at every<br>mean drop|ops on Ta<br>block; th<br>median|bICLv2-cls<br>e worst singl<br>worst (min)|(49datasets, baseline0_._864). Mean<br>e dataset drop never exceeds0_._30.<br>best (max)|
|---|---|---|---|---|
|b0|_−_0_._001|_−_0_._001|_−_0_._028|+0_._033|
|b1|_−_0_._004|_−_0_._001|_−_0_._033|+0_._022|
|b2|_−_0_._004|_−_0_._001|_−_0_._059|+0_._026|
|b3|_−_0_._016|_−_0_._006|_−_0_._082|+0_._015|
|b4|_−_0_._007|_−_0_._004|_−_0_._056|+0_._015|
|b5|_−_0_._009|_−_0_._005|_−_0_._062|+0_._012|
|b6|_−_0_._009|_−_0_._002|_−_0_._061|+0_._029|
|b7|_−_0_._004|_−_0_._001|_−_0_._087|+0_._020|
|b8|_−_0_._020|_−_0_._003|_−_0_._299|+0_._014|
|b9|_−_0_._003|_−_0_._002|_−_0_._033|+0_._047|
|b10|_−_0_._002|_−_0_._001|_−_0_._031|+0_._027|
|b11|+0_._001|+0_._000|_−_0_._009|+0_._009|



### **C.3 The final-representation readout** 

### **C.3.1 Prototype-classifier readout** 

Prototype classifiers at each ICL layer ( _−_ 1=ICL input, 0 _. . ._ 11=block outputs): per-class centroids decoded via softmax( _−d_ ( **z** _,_ **_µ_** _c_ )) (49 datasets, 5 seeds; baseline 0 _._ 864). _ℓ_ 2 and cosine prototypes track each other to _≤_ 0 _._ 002 at every layer and reach 0 _._ 854 _/_ 0 _._ 854 at L11 vs. 0 _._ 864 end-to-end; the learned head adds _≤_ 1 _._ 0 pp. Calibration analysis (§I.14): learned head, L11-prototype, and L11- _k_ NN agree within _∼_ 1 pp accuracy and 0 _._ 92–0 _._ 93 pairwise. 

|Table 30|: Per-l|ayer pr|ototype|classi|fer ac|curacy|on TabI|CLv2|-cls (49|datase|ts,5s|eeds;|baseline|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0_._864). L|ayer_−_|1is the|ICL i|nput (c|olumn|-embed|ding ou|tput).|_ℓ_2/cosin|e are i|ntercha|ngeab|le.|
|Layer|_−_1|0|1|2|3|4|5|6|7|8|9|10|11|
|_ℓ_2|.735|.761|.772|.790|.804|.814|.823|.830|.828|.834|.846|.852|.854|
|cosine|.734|.760|.772|.791|.802|.813|.822|.830|.831|.832|.845|.851|.854|



### **C.3.2 Cross-version readout checks: TabICL v1 and v1.1** 

We rerun the readout grid on the TabICL v1 and v1.1 classifiers over the full 49-dataset classification suite (5 seeds per dataset) to check whether the prototype/ _k_ NN picture is specific to the v2 release. The qualitative ordering survives across all three releases (Tab. 31): L11-prototype and L11- _k_ NN remain close to native accuracy, while the sharpest-block vote is far worse. Calibration is descriptive only: the v1 native / prototype / _k_ NN / sharp-vote ECE–NLL pairs are 0 _._ 047 _/_ 0 _._ 339, 0 _._ 416 _/_ 1 _._ 022, 0 _._ 135 _/_ 3 _._ 315, and 0 _._ 213 _/_ 0 _._ 893; the matching v1.1 pairs are 0 _._ 050 _/_ 0 _._ 336, 0 _._ 433 _/_ 1 _._ 051, 0 _._ 130 _/_ 3 _._ 186, and 0 _._ 196 _/_ 0 _._ 821. The final-representation prototype/ _k_ NN picture holds across the released TabICL classifiers and is not an artefact of the v2 release. 

Table 31: Readout-grid comparison across three released TabICL classification models on the 49dataset classification suite (5 seeds / dataset). Values are means over per-dataset seed means. 

|Release|native|L11-prototype|L11-_k_NN|sharp-vote|
|---|---|---|---|---|
|v2|0_._864|0_._854|0_._856|0_._470|
|v1.1|0_._852|0_._830|0_._842|0_._626|
|v1|0_._852|0_._828|0_._836|0_._590|



28 

### **C.3.3 Proximity-corrected within-class preference** 

At each ICL block: raw within-class attention minus expected value conditioned on feature proximity (bootstrap resampling). Across 49 datasets, 0 show positive corrected preference; 43 are significantly negative ( _α_ =0 _._ 05). Attention does not seek same-class rows; geometry is laid down upstream. 

### **C.3.4 Distance correlations** 

Spearman _ρ_ between attention weights and pairwise distances (feature-space _ℓ_ 2, representation-space _ℓ_ 2, label-embedding cosine): _<_ 0 _._ 08 at all ICL blocks. Attention does not implement distance-based retrieval. 

### **C.3.5** _k_ **NN agreement with TabICL** 

> _k_ NN on the L11 representation and on the raw input, _k∈{_ 1 _,_ 3 _,_ 5 _,_ 10 _}_ , four metrics, 49 datasets, 10 seeds. TabICLv2 accuracy: 0 _._ 864 _±_ 0 _._ 155. _ℓ_ 2/cosine on the final representation match TabICLv2 to within 1 pp and agree with it on _∼_ 94% of test queries; Mahalanobis underperforms because of covariance estimation under small context; raw-input _ℓ_ 2 is a non-trivial _∼_ 78%. Converse pattern of TabPFNv2 (§B.2.6). 

Table 32: _k_ NN accuracy / agreement with TabICLv2 predictions, by metric and _k_ (49 datasets, 10 seeds, baseline 0 _._ 864). 

|Metric|_k_=1|_k_=3|_k_=5|_k_=10|
|---|---|---|---|---|
|rep_ℓ_2|.855 / .933|.856 / .938|.857 / .938|.858 / .941|
|rep cosine|.855 / .932|.856 / .936|.857 / .939|.858 / .941|
|rep Maha.|.652 / .672|.678 / .702|.689 / .714|.700 / .729|
|input_ℓ_2|.749 / .788|.764 / .806|.772 / .817|.776 / .828|



### **C.3.6 Per-block vote refutation** 

Each ICL block decoded through final output projection as a per-block vote (10 datasets, 5 seeds): 

Table 33: TabICL per-block vote refutation. Each block’s hidden state is decoded through the final output projection. Best block is L9 with Pearson _r_ = 0 _._ 462, far below TabPFN’s vote correlation of 0 _._ 888. Mean across 10 datasets _×_ 5 seeds. 

|**Layer**|**Pearson**_r_|**Vote Acc**|**Agreement**|_N_|
|---|---|---|---|---|
|L0|0.306|0.467|0.491|10|
|L1|0.376|0.525|0.552|10|
|L3|0.169|0.412|0.435|10|
|L6|0.367|0.554|0.582|10|
|L9|0.462|0.584|0.612|10|
|L11|0.230|0.442|0.468|10|



Best block: L9, _r_ (vote _,_ final)=+0 _._ 462, agreement 0 _._ 612, vote-acc 0 _._ 584 vs. true _≈_ 0 _._ 864. Other blocks: _r_ =+0 _._ 169 (L3) to +0 _._ 376 (L1); vote-acc never exceeds 0 _._ 584. Refutes per-block vote hypothesis. Contrast TabPFNv2: L9 vote tracks final at _r_ =0 _._ 888 (§B.2.3). Regression confirms (§I.2). 

## **D Mitra audit details** 

This appendix collects the per-experiment details for Mitra referenced from the main text. Mitra’s architecture is described alongside TabPFNv2 and TabICLv2 in §A.1; the load-bearing differences for our analysis are the separate per-row label slot, the shared per-feature embedding, and the absence of positional encoding on either axis. We audit both released checkpoints (v1 and v1.1) on the same 49-dataset classification and 10-dataset regression suites used throughout the paper. All experiments are run offline against local checkpoints; Mitra is loaded as a single model with ensembling/stacking 

29 

disabled, so the AutoGluon defaults that would wrap it in a meta-ensemble are explicitly turned off and we measure the model in isolation. 

### **D.1 Permutation invariance (§4)** 

Mitra is exactly invariant to row _and_ column permutations up to floating-point noise; per-class label invariance fails as expected because the label slot uses a per-class embedding. 

Table 34: Permutation invariance summary. Quantitative measurements across 48 datasets that load under the per-class minimum-sample filter, 20 random permutations per dataset, fp32 inference: mean _|_ ∆ _p|_ is below 0 _._ 3% for both row and column on both Mitra versions; the mean per-dataset _maximum |_ ∆ _p|_ is 1–3%; predicted-label agreement is _≥_ 99 _._ 5%. Per-class relabelling produces mean maximum _|_ ∆ _p|≈_ 18–20% with label agreement _≈_ 0 _._ 96. 

||Row|Column|Label (per-class)|
|---|---|---|---|
|TabPFNv2|yes (architectural)|no|no|
|TabICLv2|yes (architectural)|no (RoPE)|no|
|Mitra v1|**yes (architectural)**|**yes (architectural)**|no|
|Mitra v1.1|**yes (architectural)**|**yes (architectural)**|no|



The exact column invariance follows mechanically from the shared per-feature embedding plus the absence of any positional encoding: column attention sees feature tokens as a set, so feature reordering is exactly a no-op up to numerical reduction order. TabPFNv2 and TabICLv2 do not have this property because both inject feature identity (the TabPFNv2 per-feature embedding _Wp_ and TabICLv2’s RoPE). 

### **D.2 Layerwise representation analysis (§2)** 

We attach forward hooks on each `Layer.forward` of Mitra’s transformer and capture both the context and query streams. Mean-pooling along the feature axis (skipping the _y_ -slot at position 0) gives one vector per row per layer. We then evaluate four per-layer readouts on the resulting features: a logistic regression linear probe on standardised features, a class-mean prototype rule, cosine-distance 5-NN, and the agreement of the 5-NN prediction with Mitra’s native prediction. 45 of the 49 datasets load under Mitra’s per-class minimum-sample filter. 

Table 35: Layerwise probe summary on the 45-dataset subset. Both Mitra versions peak at L9, mirroring the TabPFNv2 late-jump pattern rather than TabICLv2’s early convergence. _k_ NN agreement at the peak is 0 _._ 82–0 _._ 83 and falls to 0 _._ 71–0 _._ 78 by L12 (Table 36). 

|version|_n_datasets|e2e mean acc|peak-probe layer|_k_NN-agreement @L9|
|---|---|---|---|---|
|v1|45|0.827|L9(0_._777)|0.831|
|v1.1|45|0.825|L9(0_._784)|0.821|



30 

Table 36: Per-layer probe / prototype / cosine 5-NN / 5-NN-agreement on Mitra, mean over the 45-dataset subset. Probe peaks at L9 in both versions, with the characteristic L10 refinement-phase dip and partial recovery at L11–L12. _k_ NN tracks the probe within _∼_ 2 pp at L9; the prototype rule is weaker, especially for v1.1 (gap _∼_ 6 pp). 

|||v1 (4<br>|5 datasets<br>|)<br>||v1.1 (<br>|45 datase<br>|ts)<br>|
|---|---|---|---|---|---|---|---|---|
|layer|probe|proto|_k_NN5|_k_NN-agree|probe|proto|_k_NN5|_k_NN-agree|
|L1|0.576|0.533|0.592|0.640|0.430|0.462|0.500|0.525|
|L2|0.681|0.652|0.690|0.746|0.519|0.522|0.579|0.611|
|L3|0.698|0.677|0.695|0.763|0.667|0.673|0.697|0.758|
|L4|0.713|0.693|0.716|0.791|0.672|0.673|0.698|0.762|
|L5|0.731|0.712|0.731|0.808|0.746|0.722|0.756|0.833|
|L6|0.752|0.734|0.753|0.837|0.761|0.736|0.756|0.835|
|L7|0.759|0.751|0.763|0.845|0.771|0.737|0.764|0.847|
|L8|0.771|0.745|0.759|0.836|0.774|0.723|0.754|0.831|
|L9|**0.777**|0.742|0.759|0.831|**0.784**|0.721|0.755|0.821|
|L10|0.718|0.684|0.704|0.750|0.702|0.656|0.664|0.700|
|L11|0.753|0.739|0.769|0.830|0.685|0.655|0.657|0.671|
|L12|0.676|0.683|0.733|0.777|0.673|0.646|0.688|0.706|



### **D.3 Per-block knockouts (§2.3)** 

We replicate the per-block knockout protocol of §B.1.2 on Mitra. Each interleaved row/feature `Layer` is replaced one at a time with the identity (a forward pre-hook returning its inputs unchanged); the rest of the network runs normally, and we report the resulting accuracy on the same 49 datasets as §D.1, with 300 context and 200 query rows per dataset. 

Layer 0 is the most damaging single intervention on both checkpoints, by a margin: it costs a mean of 15 _._ 8 pp on v1 and 41 _._ 1 pp on v1.1, while the runner-up block costs _≤_ 2 pp on v1 and 19 _._ 6 pp on v1.1 (Layer 1). Layer 0 is the worst single block on 31 _/_ 49 datasets for v1 and 36 _/_ 49 for v1.1; no other block is the worst on more than three datasets in v1 or one in v1.1. The remaining eleven blocks each cost 1–3 pp in mean drop, with no late-layer organizing spike comparable to TabPFNv2’s block 9 (§B.1.3); consistent with the geometric, late-jumping readout reported in §3.2, where a single late layer (L9) suddenly aligns with the attention-weighted vote. The asymmetry between v1 and v1.1 mirrors the per-layer probe pattern of §36: v1.1 compresses class structure into the early frame more aggressively than v1, and pays for it more under early-block ablation. 

### **D.4 Readout fidelity (§3)** 

On the full 49-dataset classification grid (1 seed/dataset, v1), Mitra’s native head reaches mean accuracy 0 _._ 860. At L9, the attention-weighted vote (defined below) reaches 0 _._ 815 ( _−_ 4 _._ 5 pp, mean Pearson _r_ =0 _._ 89 vs. native), the cosine 5-NN readout reaches 0 _._ 789 (within _∼_ 1 _._ 5 pp of the L9 linear probe at 0 _._ 776), and the class-mean prototype rule reaches 0 _._ 769. Vote leads _k_ NN by 2 _._ 6 pp and prototype by 4 _._ 6 pp on accuracy; argmax-agreement of _k_ NN with Mitra’s native predictions at L9 is 0 _._ 82–0 _._ 83, and _k_ NN tracks the linear probe within _∼_ 2 pp through the entire L5–L9 window where both rise monotonically (Table 36). All three rules are retrieval-over-context-rows surrogates and are consistent with the row-attention ablation (§D.6); among them the L9 vote is the closest match to the native head and is reported as Mitra’s readout in §3. Cosine _k_ NN is reported as a parameter-free diagnostic. 

**Layer choice (v1 vs. v1.1).** On v1 weights, the vote-faithful and probe-faithful layers coincide at L9: a layerwise sweep of vote accuracy on the 45-dataset subset of Table 36 shows L9 at the joint optimum, with L8 and L10 within 0 _._ 2 pp and L12 within 0 _._ 2 pp (0 _._ 779, 0 _._ 778, 0 _._ 777, 0 _._ 778 respectively; L11 underperforms by _∼_ 7 pp). On v1.1, the vote-faithful layer shifts to the final attention block: L12 vote reaches 0 _._ 810 (gap _−_ 3 _._ 4 pp, agree 0 _._ 89, _r_ =0 _._ 89) on the 42-dataset OpenML grid, while L9 underperforms L12 by _∼_ 7 _._ 7 pp. We use v1 throughout the main text so that the same layer L9 surrogates both attention vote and the linear probe. 

**Attention-weighted vote at** L9 **.** Mitra’s `MultiheadAttention` dispatches to `F.scaled_dot_product_attention` , which does not expose the attention weights, so we 

31 

wrap `layer.attention1.forward` at L9 to additionally compute softmax( _QK_<sup>_⊤_</sup> _/_<sup>_√_</sup> _dh_ ) on the cross-attention call (query rows attending over context rows) without altering the returned tensor or Mitra’s predictions. Averaging the resulting weights over heads and over the per-feature row-attention slices and forming the attention-weighted vote vote[ _t, c_ ] _∝_<sup>�</sup> _i_ : _yi_<sup>ctx</sup> = _c_<sup>_a_¯</sup><sup>_t,i_, the L9 vote rule reaches</sup> mean accuracy 0 _._ 815 on the full 49-dataset grid (vs. native 0 _._ 860, gap 4 _._ 5 pp; mean Pearson _r_ vs. native probabilities 0 _._ 89). The same construction restricted to the _y_ -slot of the row-attention reshape is _worse_ , not better, at every layer (best _∼_ 0 _._ 74 at L12 on v1.1), because column attention mixes _y_ -slot and feature-slot information within each observation, leaving the per-feature average more stable than any single slot. 

### **D.5 Adversarial probes (§6)** 

We port the full attack battery used for TabPFNv2 and TabICLv2 (§H) to Mitra: random-label padding (2 _×_ and 4 _×_ ), boundary poisoning, hub poisoning, class-centroid injection, monotone distortions (cube / softexp / rank), and the adversarial low-variance rotation. The same 49-dataset suite, splits, and Ridge baseline are used. 

Table 37: Mean accuracy drop relative to the unattacked baseline, 49 datasets. “Worse than R.” counts datasets on which Mitra ends below Ridge after the attack. 

|attack|Ridge∆|Mitra v1∆|v1 worse than R.|Mitra v1.1∆|v1.1 worse than R.|
|---|---|---|---|---|---|
|pad2x|_−_0_._047|_−_0_._049|7/49|_−_0_._032|6/49|
|pad4x|_−_0_._081|_−_0_._075|11/49|_−_0_._054|8/49|
|boundary|_−_0_._021|_−_0_._028|11/49|_−_0_._016|9/49|
|hub poisoning|_−_0_._012|_−_0_._021|**14/49**|_−_0_._022|**11/49**|
|centroid inj.|_−_0_._002|+0_._001|6/49|+0_._000|7/49|
|mono cube|_−_0_._022|_−_0_._000|5/49|_−_0_._001|3/49|
|mono softexp|_−_0_._043|_−_0_._044|7/49|_−_0_._045|7/49|
|mono rank|_−_0_._042|_−_0_._000|3/49|_−_0_._003|3/49|
|rotate (low-_σ_)|_−_0_._001|_−_0_._006|10/49|_−_0_._001|7/49|



Mitra is uniformly more robust than Ridge under monotone distortions (cube, rank): essentially zero drop where Ridge gives up 2–4 pp. Pad-style and rotation attacks degrade Mitra comparably to Ridge. Hub poisoning is the clearest case where Mitra ends below Ridge on a substantial fraction of datasets (14 _/_ 49 on v1, 11 _/_ 49 on v1.1); boundary poisoning and rotation also produce non-trivial worse-than-Ridge counts but with smaller mean drops. The hub-poisoning pattern is consistent with the vote-faithful readout established in §D.4: inserting label-anchored hubs in the context directly perturbs the context-similarity signal that the readout relies on. 

### **D.6 Collapse mechanism (§5)** 

Mitra does not exhibit representation collapse on balance-scale or on a _d_ =12 _balance-like_ synthetic dataset (Table 38). We further perform the analogue of the TabPFNv2/TabICLv2 double ablation by zeroing each of Mitra’s two attention paths ( `Layer.attention1` : row attention; `Layer.attention2` : column attention) via verified forward hooks — a per-call counter confirms the hook fires — and measuring the resulting accuracy. 

Table 38: Mitra accuracy on the canonical balance-scale stress test and a _d_ =12 balance-like variant. No collapse is observed in either version. 

|version|balance-scale (3 seeds)|balance-like,_d_=12(3 seeds)|
|---|---|---|
|v1|0_._961_±_0_._007|0_._893_±_0_._025|
|v1.1|0_._959_±_0_._014|0_._969_±_0_._003|
|majority baseline|0_._436|_∼_0_._47|



32 

Table 39: Double ablation of Mitra’s attention paths. “row_off” zeroes `Layer.attention1` on every layer; “col_off” zeroes `Layer.attention2` . Removing row attention drops accuracy from _∼_ 0 _._ 95 to 0 _._ 08, well below the majority baseline, indicating a systematic misclassification regime rather than collapse to majority. Removing column attention collapses to near-majority. Removing both lands at the “row_off” level. 

|version|dataset|full|row_off|col_off|both_off|
|---|---|---|---|---|---|
|v1|balance-scale|**0.954**|0.080|0.459|0.080|
|v1|balance-like-d12|**0.893**|0.074|0.459|0.074|
|v1.1|balance-scale|**0.963**|0.080|0.461|0.080|
|v1.1|balance-like-d12|**0.969**|0.074|0.467|0.074|
|majority|baseline|_∼_0_._46|(balance-s|cale),_∼_0_._47|(balance-like)|



The collapse-critical mechanism for Mitra is therefore _row attention_ , the only one of the three families for which the critical mechanism is itself permutation-symmetric: TabPFNv2’s is the feature-positional pathway _Wp_ , TabICLv2’s is RoPE on the row axis, and Mitra’s is the cross-row aggregator. Combined with the vote-faithful readout (§D.4), this picture is consistent with the differential vulnerability to hub poisoning observed in §D.5: corrupting label-anchored hubs perturbs the context-similarity signal that flows through exactly the row-attention path the ablation here identifies as load-bearing. We do not claim direct causal attribution from the balance-scale ablation to the hub-poisoning result — the two experiments use different datasets and stress different operating regimes — only that both point at row attention as the natural locus of failure. 

### **D.7 Three-way contrast** 

Table 40: Three-way comparison across the dimensions audited in this paper. Mitra’s _compute schedule_ is TabPFNv2-like (late jump in depth), its _readout family_ is also TabPFNv2-like (attentionweighted vote at L9), and its _symmetry profile_ is strictly stronger than both — column invariance is exact by construction. 

|dimension|TabPFNv2|TabICLv2|Mitra|
|---|---|---|---|
|Token factorization|feat_⊕_label per token|feat_⊕_label per token|separate_y_-slot per row|
|Positional structure|per-feature_Wp_|RoPE on feature axis|none|
|Row invariance|yes|yes|yes|
|Column invariance|no|no (RoPE)|**yes (exact, no patch)**|
|Per-class label invariance|no|no|no|
|Probe peak layer|late (L9)|early (L2–L4)|late (L9)|
|Faithful surrogate (cls)|attention vote @ L9|class-mean prototype @ L11|attention vote @ L9|
|Hub-poisoning vulnerability|yes|yes|yes|
|Monotone-distortion (rank/cube)|low|low|lowest|
|Balance-scale collapse|no|sometimes (v1)|no|
|Critical mechanism (ablation)|feat-pos_Wp_|RoPE|**row attention**|
|Critical mech. permutation-symm.|no|no|**yes**|



**Summary.** Mitra is its own combination of design choices. Token-level factorization of features and labels, a shared per-feature embedding, and the absence of positional encoding together make it the only one of the three families for which column invariance is exact, the cosine- _k_ NN readout is the closest similarity-based surrogate to native predictions of the three families (§3), and the collapse-critical mechanism uncovered by ablation is itself permutation-symmetric. Empirically, this combination delivers full robustness to monotone feature distortions (cube, rank) and no balance-scale collapse, while preserving the same hub-poisoning weakness as TabPFNv2/TabICLv2. Two ablations are consistent with the mechanistic narrative for Mitra: the row-attention zero-out (Table 39) and the hub-poisoning attack (Table 37); the connection between them remains an inference rather than a single end-to-end intervention. 

33 

## **E TabPFN regression: full results** 

Regression measurements cover 10 datasets with 5 seeds each unless noted. The subsections that follow trace the late-readout circuit from classification through regression: per-block representation geometry, row-attention sharpness, activation patching, the L9 attention-vote test, and the MLPknockout and uniform-attention falsification controls. Two systematic shifts emerge against the classification baseline — the probe peak moves one layer earlier and row-attention is substantially softer — both consistent with the bias–variance trade-off of continuous targets. 

### **E.1 Per-block representation geometry** 

Per-block geometry on regression matches the classification profile. Table 41 reports per-block effective rank (ER), participation ratio (PR) and 2-NN intrinsic dimension (ID) averaged across the 10 regression datasets. The L2 block triggers a sharp ER collapse (ER _≈_ 9 _._ 5, PR _≈_ 1 _._ 5), L3–L8 rebuild the representation gradually, and L9–L11 reach maximum ER and PR where the readout forms. Intrinsic dimension contracts from ID _≈_ 8 _._ 0 at L1 to a 3 _._ 4–3 _._ 7 plateau from L6 onward, exposing a low-dimensional predictive manifold. 

Table 41: Per-block geometry on TabPFN-reg, mean _±_ std across 10 regression datasets (5 seeds each). 

|**Block**|**Effective rank**|**Participation ratio**|**2-NN ID**|
|---|---|---|---|
|L0|78_._7_±_15_._0|3_._60_±_0_._81|5_._54_±_1_._16|
|L1|86_._9_±_15_._9|4_._68_±_1_._14|7_._99_±_2_._45|
|L2|9_._5_±_0_._6|1_._47_±_0_._18|3_._49_±_0_._75|
|L3|22_._6_±_2_._0|3_._06_±_0_._29|3_._76_±_1_._29|
|L4|30_._1_±_4_._0|2_._76_±_0_._40|4_._22_±_1_._63|
|L5|26_._6_±_3_._9|2_._91_±_0_._61|3_._72_±_1_._38|
|L6|32_._8_±_4_._4|3_._71_±_0_._78|3_._42_±_1_._34|
|L7|34_._4_±_5_._2|3_._73_±_0_._76|3_._38_±_1_._28|
|L8|30_._0_±_4_._9|3_._29_±_0_._64|3_._46_±_1_._33|
|L9|36_._5_±_6_._9|3_._96_±_0_._83|3_._49_±_1_._40|
|L10|34_._3_±_7_._5|3_._73_±_1_._03|3_._69_±_1_._48|
|L11|38_._3_±_15_._7|5_._25_±_2_._45|3_._72_±_1_._44|



### **E.2 Regression row-attention** 

Regression row-attention is markedly softer and peaks earlier than in classification. Classification L9 max weight averages 0 _._ 925, while regression L9 max weight peaks at _≈_ 0 _._ 26 and is sharpest at L6–L7 on most datasets; L9 entropy reaches _≈_ 3 _._ 8 versus 1 _._ 756 for classification. The one-to-two-layer earlier peak tracks the probe peak shift: regression reaches usable representations earlier and pools them across more context rows. 

### **E.3 Regression activation patching** 

Activation patching restores the baseline on every regression dataset. Replacing the L9 state of blocks _{_ 0 _,_ 5 _,_ 9 _}_ with the cached clean activations yields recovery = 1 _._ 000 _±_ 0 _._ 000, matching the classification result in §B.2.7. Table 42 gives the per-dataset baseline _R_<sup>2</sup> , the two corruptions (labelshuffle, block-0 knockout) and the patched-L9 score; once the corrupted state is overwritten with the cached clean state the test-row representations become identical to baseline and the score follows. 

34 

Table 42: TabPFN-reg activation patching, per dataset (mean of 5 seeds). “LS” = label-shuffle corruption; “B0-KO” = block-0 knockout corruption. “Patched” = restored after L9 activation patching. 

|**Dataset**|**Baseline**_R_<sup>2</sup>|**LS corrupt**|**B0-KO corrupt**|**Patched**_R_<sup>2</sup>|
|---|---|---|---|---|
|Fiat-500|0_._867|_−_0_._048|0_._006|0_._867|
|Moneyball|0_._947|_−_0_._002|0_._021|0_._947|
|QSAR-fsh|0_._648|_−_0_._008|_−_0_._212|0_._648|
|airfoil|0_._964|_−_0_._009|_−_0_._082|0_._964|
|cars|0_._952|0_._005|_−_0_._037|0_._952|
|concrete|0_._936|_−_0_._008|_−_0_._200|0_._936|
|energy|0_._998|0_._004|0_._095|0_._998|
|forest-fres|_−_0_._028|_−_0_._030|_−_0_._044|_−_0_._028|
|healthcare|0_._805|_−_0_._012|_−_0_._116|0_._805|
|socmob|0_._796|_−_0_._018|_−_0_._065|0_._796|
|mean|0_._789|_−_0_._013|_−_0_._063|0_._789|



### **E.4 Per-block attention vote vs. permutation null** 

On every regression dataset the model learns, the L9 attention vote drives the prediction. Pearson and Spearman correlations between the model output and the attention-weighted mean of context targets are compared against a 1000-permutation null (Table 43). The vote is significant on 9 _/_ 10 datasets ( _p<_ 10<sup>_−_3</sup> , mean Pearson 0 _._ 745). The single failure, `forest-fires` , is also the dataset on which the model itself fails to learn ( _R_<sup>2</sup> = _−_ 0 _._ 028 in Table 42), so the vote correctly registers as uninformative. 

Table 43: TabPFN-reg L9 attention-vote correlations and 1000-permutation null, per dataset (mean of 10 seeds). 

|**Dataset**|**Pearson**|**Spearman**|_z_**vs. null**|_p_|
|---|---|---|---|---|
|Fiat-500|0_._965|0_._908|3_._07|_<_10<sup>_−_4</sup>|
|Moneyball|0_._945|0_._981|2_._00|0_._0006<br>|
|QSAR-fsh|0_._908|0_._918|4_._43|_<_10<sup>_−_4</sup>|
|airfoil|0_._917|0_._921|5_._31|_<_10<sup>_−_4</sup><br>|
|cars|0_._941|0_._880|4_._14|_<_10<sup>_−_4</sup>|
|concrete|0_._892|0_._926|5_._36|_<_10<sup>_−_4</sup>|
|energy|0_._796|0_._674|3_._47|_<_10<sup>_−_4</sup>|
|forest-fres|_−_0_._577|_−_0_._573|_−_1_._20|0_._297<br>|
|healthcare|0_._822|0_._683|3_._83|_<_10<sup>_−_4</sup>|
|socmob|0_._840|0_._548|2_._89|0_._0003|



### **E.5 TabPFN falsification controls (MLP-KO and L9 uniform-attn)** 

Two counterfactual interventions falsify the attention-vote rule as a sole mechanism. On a mixed cls/reg panel (10 + 10 datasets, 5 seeds), Table 44 shows that both interventions degrade performance severely: MLP computation at every layer is load-bearing, and L9 attention is necessary but not sufficient. 

Table 44: TabPFN falsification controls. Per-block MLP knockout zeros the MLP’s second linear at each block without retraining. L9 uniform-attn replaces the L9 softmax with 1 _/N_ train. Mean over 10+10 cls/reg datasets, 5 seeds each. 

|**Intervention**|**Cls baseline**<br>**C**|**ls result**|**Reg baseline**|<br>**Reg result**|
|---|---|---|---|---|
|MLP-KO (any block L0–L11)|0_._874<br>0_._415|(∆=_−_0_._460)|0_._789|_−_0_._010(∆=_−_0_._798)|
|L9 uniform-attn|0_._874<br>0_._488|(∆=_−_0_._386)|0_._789|0_._585(∆=_−_0_._203)|



The smaller regression drop under uniform attention follows directly from the softer attention profile (§E.2): the L9 distribution is already close to uniform, so flattening it removes less signal. 

35 

## **F TabICL regression: full results** 

Regression results for TabICLv2 parallel the classification analysis in §C across 10 datasets and 5 seeds, with baseline _R_<sup>2</sup> = 0 _._ 805. The structural difference from classification is the non-linear regression head at L11, which shapes both the linear-probe profile and the L11 knockout asymmetries reported below. The _ℓ_ 2- _k_ NN tracking on the final representation (§I.1) and the per-block vote refutation (§I.2) both replicate in this setting. 

### **F.1 Layer-by-layer summary** 

Per-layer linear-probe _R_<sup>2</sup> (ridge, fit on the cell representation of the query rows, evaluated against the regression target) is _≤_ 0 _._ 07 at every layer and _negative_ at L9–L11, confirming the readout is non-linear at the top of the stack (Table 45). The geometric trace of the representation (Table 46) shows monotone expansion of effective rank from 19 _._ 3 at the column-embedding output to 100 _._ 0 at L11; participation ratio peaks mid-stack and 2NN-ID grows from 3 _._ 4 to 6 _._ 0. 

Table 45: Per-layer linear probe _R_<sup>2</sup> for TabICLv2-reg (mean _±_ std across 10 datasets). Layer _−_ 1 is the column-embedding output; L0–L11 are the ICL blocks. 

|Layer|_−_1|0|1|2|3|4|5|6|7|8|9|10|11|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|_R_<sup>2</sup>|+0_._004|+0_._005|+0_._007|+0_._004|_−_0_._014|+0_._061|+0_._027|+0_._046|+0_._024|+0_._041|_−_0_._157|_−_0_._463|_−_0_._469|
|std|0_._013|0_._013|0_._013|0_._014|0_._015|0_._020|0_._024|0_._025|0_._028|0_._025|0_._043|0_._074|0_._111|



Table 46: Representation geometry per block for TabICLv2-reg (mean across 10 datasets, 5 seeds). ER = effective rank, PR = participation ratio, ID = TwoNN intrinsic dimension. 

|Layer|_−_1|0|1|2|3|4|5|6|7|8|9|10|11|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|ER|19.3|26.1|33.7|40.1|45.8|47.4|64.6|74.1|79.2|68.1|88.8|87.1|100.0|
|PR|4.4|5.7|6.9|7.0|6.8|7.9|9.7|10.0|7.4|6.8|8.0|7.1|5.6|
|ID|3.4|3.6|3.8|4.1|4.2|4.4|4.8|5.2|5.0|4.7|5.3|5.2|6.0|



### **F.2 Per-block knockout** 

Zeroing one ICL block at a time and re-running the regression head produces no catastrophic loss except at L11 (Table 47). Column- and row-embedding knockouts are equally tolerated (col-L0/L1/L2 _R_<sup>2</sup> in [0 _._ 71 _,_ 0 _._ 78]; row-L0/L1/L2 _R_<sup>2</sup> in [0 _._ 77 _,_ 0 _._ 81]). The contrast with L11 ( _R_<sup>2</sup> = _−_ 1 _._ 16) is the single distributed-vs-localized signal in this panel and matches the linear-probe drop at the top of Table 45. 

Table 47: Per-block ICL knockout _R_<sup>2</sup> for TabICLv2-reg (mean _±_ std across datasets; baseline 0 _._ 805). ICL block 0 1 2 3 4 5 6 7 8 9 10 11 _R_<sup>2</sup> +0 _._ 731 +0 _._ 807 +0 _._ 808 +0 _._ 805 +0 _._ 793 +0 _._ 801 +0 _._ 781 +0 _._ 711 +0 _._ 793 +0 _._ 740 +0 _._ 772 _−_ 1 _._ 156 std 0 _._ 029 0 _._ 024 0 _._ 023 0 _._ 024 0 _._ 019 0 _._ 025 0 _._ 041 0 _._ 085 0 _._ 027 0 _._ 062 0 _._ 043 0 _._ 261 

### **F.3 Frozen probe post-KO** 

Ridge probe at L11: baseline frozen _R_<sup>2</sup> = _−_ 0 _._ 469 (the head is non-linear, so the linear probe is already negative at baseline). Table 48 compares the frozen probe (fit on clean activations, evaluated on KO activations) against a probe re-trained on KO activations. Large frozen-to-retrained gap at ICL-b0 and ICL-b6 indicates that the knockout disrupts the geometry the clean probe relies on, but a fresh probe can still recover most of the linearly decodable signal. ICL-b11 is a no-op for the _linear_ probe at L11 because zeroing the block makes the representation pre-block, which is what the probe is fit on. Together with the _k_ NN-vs-prototype divergence in §I.1 this confirms distributed construction + non-linear readout. 

36 

Table 48: Frozen vs. retrained ridge probe at L11 after each single-block knockout (TabICLv2-reg, 5 seeds, 10 datasets). 

|KO site|frozen_R_<sup>2</sup>|retrained_R_<sup>2</sup>|
|---|---|---|
|ICL-b0|_−_0_._881|_−_0_._006|
|ICL-b6|_−_0_._692|_−_0_._273|
|ICL-b11|_−_0_._469|_−_0_._469|
|ColEmb-b2|_−_0_._610|_−_0_._275|
|(no KO)|_−_0_._469|_−_0_._469|



A complementary frozen-probe sweep (10 seeds) intervenes at three earlier sites and reads probes off multiple layers; the frozen-vs-retrained gap is uniformly large (e.g. KO@5 reads _R_<sup>2</sup> frozen _−_ 0 _._ 626 vs. retrained _−_ 2 _._ 81 at L11), matching the picture above. 

### **F.4 Per-block MLP knockout** 

Zeroing each ICL block’s MLP second linear layer (Table 49). L0–L7 lose at most 0 _._ 04 in _R_<sup>2</sup> ; L8–L10 lose 0 _._ 06–0 _._ 09; L11 collapses to _−_ 0 _._ 63 (∆ _R_<sup>2</sup> = _−_ 1 _._ 43). Classification (§C.2.4) showed _≤_ 2 _._ 1 pp drop at any block. The asymmetry at L11 reflects that the regression head is a full MLP rather than a linear projection. 

Table 49: Per-ICL-block MLP knockout _R_<sup>2</sup> for TabICLv2-reg (mean _±_ std across datasets; baseline 0 _._ 805). 

|ICL block|0|1|2|3|4|5|6|7|8|9|10|11|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|_R_<sup>2</sup>|+0_._803|+0_._808|+0_._804|+0_._800|+0_._786|+0_._797|+0_._786|+0_._767|+0_._738|+0_._740|+0_._720|_−_0_._630|
|std|0_._025|0_._025|0_._026|0_._022|0_._020|0_._019|0_._024|0_._030|0_._079|0_._074|0_._092|0_._177|



### **F.5 Row-attention entropy** 

Row-attention entropy (averaged over heads, query rows, and datasets) stays flat across the ICL stack (Table 50; min 6 _._ 53 at L11, max 6 _._ 66 at L2, range _≈_ 0 _._ 13 nats out of log _e_ 1024 = 6 _._ 93). TabPFNv2reg, by contrast, reaches max-row-weight _≈_ 0 _._ 26 at L9 (§E) and approaches a near-prototype readout; TabICLv2-reg never localizes onto a small context set. 

Table 50: Per-layer row-attention entropy (nats) for TabICLv2-reg. Layer 0 1 2 3 4 5 6 7 8 9 10 11 _H_ 6 _._ 61 6 _._ 65 6 _._ 66 6 _._ 66 6 _._ 66 6 _._ 64 6 _._ 66 6 _._ 62 6 _._ 62 6 _._ 59 6 _._ 59 6 _._ 53 

### **F.6 Label perturbations** 

Two label-corruption interventions (Table 51), random-shuffle and mean-replace, both drive _R_<sup>2</sup> to near zero, confirming the predictions are driven by the in-context labels rather than by covariate-only structure. 

Table 51: Label-perturbation _R_<sup>2</sup> for TabICLv2-reg (mean _±_ std, 10 datasets, baseline 0 _._ 805). 

|Intervention|_R_<sup>2</sup>|
|---|---|
|shuffe context labels|_−_0_._016_±_0_._019|
|replace with target mean|_−_0_._010_±_0_._003|



### **F.7 Uniform-attention knockout** 

Replacing the row-attention map at one ICL block by a uniform map (Table 52) is well tolerated for L3+ ( _R_<sup>2</sup> _≥_ 0 _._ 64) and breaks the model only at L0–L1 ( _R_<sup>2</sup> = 0 _._ 03 _,_ 0 _._ 19). Combined with the near-uniform entropy above, row-attention does little selective work after the first two ICL blocks. 

37 

Table 52: Per-layer uniform-attention _R_<sup>2</sup> for TabICLv2-reg (mean _±_ std across datasets; baseline 0 _._ 805). 

|Layer<br>0|1|2|3|4|5|6|7|8|9|10|11|
|---|---|---|---|---|---|---|---|---|---|---|---|
|_R_<sup>2</sup><br>+0_._029|+0_._187|+0_._554|+0_._713|+0_._765|+0_._733|+0_._776|+0_._641|+0_._797|+0_._686|+0_._795|+0_._801|
|std<br>0_._664|0_._577|0_._238|0_._113|0_._059|0_._076|0_._056|0_._172|0_._025|0_._040|0_._034|0_._028|



### **F.8 Per-block vote refutation** 

Following the classification protocol (§I.2) we read off a prototype-style prediction from each ICL block and correlate it with TabICLv2’s own prediction and with the ground-truth target; see Table 53. No early or middle block matches the model’s predictions; the highest agreement is at L9 (Pearson 0 _._ 73 with the model, 0 _._ 65 with truth) and L11 (0 _._ 67 _/_ 0 _._ 64). The pattern — correlations only emerging at the last few blocks — is the regression analogue of the classification vote refutation. 

Table 53: Per-block prototype-vote correlations for TabICLv2-reg. Pearson coefficient between block- _k_ vote prediction and either the model’s own prediction (pred) or the ground truth (truth). 

|Block|0|1|2|3|4|5|6|7|8|9|10|11|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|pred|_−_0_._22|+0_._40|+0_._44|+0_._27|_−_0_._52|_−_0_._28|_−_0_._23|_−_0_._43|+0_._64|+0_._73|+0_._08|+0_._67|
|truth|_−_0_._20|+0_._38|+0_._38|+0_._21|_−_0_._49|_−_0_._24|_−_0_._22|_−_0_._39|+0_._56|+0_._65|+0_._04|+0_._64|



## **G Permutation invariance: full results** 

### **G.1 Architectural sources of permutation sensitivity** 

Table 54: Component-level sources of permutation sensitivity. A ✓ indicates the component is present and order-sensitive. 

|Model|`in_linear`|Grouping|RoPE|`pos_emb`(_W_,_b_)|
|---|---|---|---|---|
|TabICL v1|✓|none|✓|—|
|TabICL v1.1|✓|none|✓|—|
|TabICL v2|✓|circular|✓|—|
|TabPFN|—|`features_per_group=2`|—|✓|
|Mitra|shared cell|none|—|—|



Table 54 catalogues the positional-encoding components that break permutation invariance. TabPFNv2’s positional embedding maps each feature index _p_ through a learned _W ∈_ R<sup>192</sup><sup>_×_48</sup> and adds a learned bias _b ∈_ R<sup>192</sup> ; preprocessing (standardisation, outlier clipping, quantile transforms) also runs in feature order. TabICLv1 and v2 both apply RoPE along the feature dimension, and v2 additionally enforces circular feature grouping; neither version encodes row position. Row invariance is hard-coded; column invariance is broken by _W, b_ in TabPFNv2 and by RoPE in TabICLv1 and v2; label invariance is broken by the learned label embeddings throughout. 

### **G.2 Synthetic invariance verifications** 

**Protocol.** The grid spans 3 seeds, 3 dataset sizes, and 3 configurations, with predictions counted as agreeing when they differ by at most 0 _._ 005. TabICLv1 contributes baseline and no-RoPE configurations; TabICLv2 contributes baseline, no-RoPE, and no-RoPE under an arbitrary-permutation control. TabPFNv2 contributes baseline, no positional embedding ( _W_ =0, _b_ retained), and the latter under an arbitrary-permutation control. 

Three residual symmetries surface within tolerance. TabPFNv2 with _W_ = 0 exhibits _grouppermutation invariance_ : pairs of features reorder freely, but pairs themselves do not split across groups. TabICLv2 with RoPE removed exhibits _circular-shift invariance_ . TabICLv1 with RoPE removed exhibits _arbitrary-column invariance_ . These outcomes match the architecture mapping in Table 22: removing one route exposes the residual symmetry of the route that remains — TabPFNv2’s within-pair symmetry of feature grouping (M3, pair size 2), TabICLv2’s cyclic symmetry of circular 

38 

grouping, and, with no M3 route in TabICLv1, arbitrary-column invariance at the cost quantified in §G.4. 

Table 55: Synthetic invariance verifications: passes / 9. 

|Model variant|Test|det|row|col|label|
|---|---|---|---|---|---|
|TabPFN baseline /_W_=0|arb|9/9|9/9|0/9|4/9|
|TabPFN baseline /_W_=0|grp2|9/9|9/9|6/9|2/9|
|TabICL v1|arb|9/9|9/9|0/9|1/9|
|TabICL v1 no-RoPE|arb|9/9|9/9|**9/9**|0/9|
|TabICL v2 (& no-RoPE)|arb/grp3|9/9|9/9|0/9|0/9|



### **G.3 Real-data invariance: TabPFN** 

Four configurations on 49 datasets, 5 seeds, 3 column-permutation trials: _Baseline_ (standard TabPFNv2 [Hollmann et al., 2025]); _W_ =0 (positional weight matrix zeroed, bias _b_ retained); _sign_ ( _b_ ) ( _W_ =0, bias replaced by sign( _b_ )); _no_pe_ ( _W_ =0, _b_ =0, all positional parameters removed). 

Setting _W_ =0 shifts predictions by at most 0 _._ 001 and preserves accuracy (0 _._ 854 _→_ 0 _._ 854); replacing _b_ with sign( _b_ ) is also lossless (0 _._ 855); removing _b_ collapses accuracy to 0 _._ 338, proving _b_ acts as an architectural prior, not a positional code. 

Table 56: TabPFN per-dataset behavior: 5 datasets with the lowest column-label-agreement; accuracy is unchanged under _W_ =0 (mean acc 0 _._ 853 baseline vs. 0 _._ 853 at _W_ =0, _n_ =45). 

|Dataset|Col. agree|Acc. baseline|Acc._W_=0|
|---|---|---|---|
|analcatdata_dmft|78_._0%|0_._216|0_._216|
|cylinder-bands|90_._4%|0_._726|0_._726|
|steel-plates-fault|92_._7%|0_._794|0_._793|
|mfeat-zernike|92_._7%|0_._839|0_._838|
|vehicle|92_._7%|0_._851|0_._851|



### **G.4 Real-data invariance: TabICLv1 and v2** 

Each version is run in two configurations on 49 datasets, 5 seeds, and 3 trials: baseline and no-RoPE. TabICLv2 carries both the M2 and M3 routes and tolerates RoPE removal; TabICLv1 carries only M2 and relies on RoPE to escape the multiset bound. The _≈_ 15 pp gap upper-bounds what RoPE buys as the sole symmetry-breaker. 

Table 57: Per-dataset accuracy when RoPE is removed, _n_ =49: 5 datasets where v1 loses the most. 

|Dataset|base|TabICL v1<br>no-RoPE|∆pp|base|TabICL v2<br>no-RoPE|∆pp|
|---|---|---|---|---|---|---|
|mfeat-karhunen|0_._968|0_._387|_−_58_._1|0_._980|0_._979|_−_0_._1|
|mfeat-pixel|0_._931|0_._392|_−_54_._0|0_._975|0_._975|0_._0|
|semeion|0_._858|0_._328|_−_53_._0|0_._956|0_._952|_−_0_._4|
|MiceProtein|0_._998|0_._503|_−_49_._5|1_._000|1_._000|0_._0|
|balance-scale|0_._987|0_._493|_−_49_._4|0_._990|0_._990|0_._0|
|**Mean (all**49**)**|0_._851|0_._697|_−_**15**_._**4**|0_._864|0_._862|_−_**0**_._**1**|
|**# losing**_>_5**pp**||27_/_49|||0_/_49||



39 

### **G.5 Real-data invariance summary** 

Table 58: Permutation agreement rates and accuracy (49 datasets _×_ 5 seeds _×_ 3 trials). Column agreement below 100% reflects order-sensitive preprocessing and grouping; label agreement reflects class-ordering sensitivity. 

|Model|Accuracy|Row agree.|Col. agree.|Label agree.|
|---|---|---|---|---|
|TabPFN|0_._854|100_._0%|98_._2%|96_._7%|
|TabICL v2|0_._865|100_._0%|98_._2%|97_._7%|



### **G.6 One-vs-All label invariance** 

Wrapping each model in a One-vs-All classifier achieves 100% label-order agreement at zero accuracy cost. 

Table 59: One-vs-All wrapper, _n_ =21 multi-class datasets. 

|Model+OvA|Acc.|Col. agree|Label agree|
|---|---|---|---|
|TabPFN+OvA|0_._877|98_._4%|100%(21_/_21)|
|TabICL v1+OvA|0_._880|97_._2%|100%(21_/_21)|
|TabICL v2+OvA|0_._890|98_._3%|100%(21_/_21)|



Mitra is column-invariant by architecture but not class-invariant in its shipped form (§D.1). Wrapping it in OvA recovers full class invariance at no accuracy cost on the same multi-class real subset (extended to _n_ =24 here for completeness; 5 seeds, 3 trials, identical split protocol): Mitra v1 baseline 0 _._ 841, Mitra v1+OvA 0 _._ 841, column agreement 99 _._ 9%, label agreement effectively at 100%. 

### **G.7 Best/worst permutation spread** 

For each dataset we record the best and worst accuracy across the 5 _×_ 3 = 15 column-permutation trials. Column-permutation spread reaches _∼_ 8 _._ 0 pp on the worst real dataset (cylinder-bands; Figure 5); the median is an order of magnitude smaller (Table 60). 



<!-- Start of picture text -->
TabPFNv2 TabICLv1 TabICLv2<br>random_labels random_labels<br>cylinder-bands 8.7 pp 8.7 pp<br>8.0 pp<br>5<br>0<br>datasets sorted by spread (n=49 per model)<br>worst acc. (pp)<br>−<br>best<br><!-- End of picture text -->

Figure 5: Per-dataset accuracy spread under random column permutations on the 49-dataset benchmark. For each dataset, the bar is (best accuracy across permutations) _−_ (worst), measuring sensitivity to the semantically irrelevant ordering of input features. Datasets are sorted ascending within each model. Most sit below the 5 pp dashed line, but a tail reaches _∼_ 8 _._ 7 pp (worst dataset annotated). _Takeaway:_ the mechanism-aware remedies of §4.1 address this deployment-relevant tail risk at no accuracy cost. 

40 

Table 60: Best _−_ worst accuracy spread (pp) per model ( _n_ =45). Six TabPFN/v1 datasets exceed the 5 pp threshold; only two for v2. 

|Model|median|mean|Q1|Q3|max|Worst3datasets (spread pp)|
|---|---|---|---|---|---|---|
|TabPFN|2_._67|2_._91|1_._39|4_._14|8_._02|cylinder-bands8_._02, tic-tac-toe7_._47, dresses-sales7_._00|
|TabICL v1|2_._38|2_._74|1_._31|4_._06|8_._00|dresses-sales8_._00, hill-valley7_._82, cylinder-bands5_._56|
|TabICL v2|1_._54|1_._73|0_._75|2_._57|5_._33|dresses-sales5_._33, cylinder-bands5_._25, ilpd4_._56|



### **G.8 Spectrum of invariance types** 

Table 61: Spectrum of achievable permutation invariance types, the scope of columns covered, and the accuracy cost of achieving it. Costs are mean differences over 49 datasets _×_ 5 seeds _×_ 3 trials relative to the respective baseline. 

|Model|Invariance type|Scope|Accuracy cost|
|---|---|---|---|
|TabICL v1 (no RoPE)|Arbitrary column permutations|All columns|_−_15_._4pp|
|TabICL v2 (no RoPE)|Circular column shifts|All columns|_−_0_._2pp|
|TabPFN (_W_ = 0)|Group permutations (groups of 2 features)|Within groups|0pp|



Table 61 catalogues the achievable invariance types and their costs. Arbitrary-permutation invariance is reachable only in TabICLv1, and only at _−_ 15 _._ 4 pp. Narrower types — circular shifts for TabICLv2 and group permutations for TabPFNv2 — carry no measurable accuracy cost. 

### **G.9 Pretraining an invariant TabICLv1 from scratch** 

**Setup.** We retrain TabICLv1 under the original stage-1 protocol of Qu et al. [2025]: 100k optimization steps at batch size 512, identical synthetic prior, no stage-2 finetuning. Two architectural switches are toggled independently, yielding four pretrained checkpoints: 

- _RoPE_ on/off along the feature axis. “On” is the released TabICLv1 configuration; “off” removes the rotary positional embedding entirely (not zeroed at inference, but absent during pretraining). 

- _Class head_ : standard class-conditional head (“no”, released configuration) versus a classinvariant head (“yes”), in which context labels enter as one-hot columns and a shared decoder emits one logit per class. Permuting the class index permutes both the conditioning columns and the output channels identically, so the `argmax` prediction is exactly class-permutation invariant by construction. 

Evaluation follows the OpenML protocol of [Qu et al., 2025]. Headline numbers reproduced in Table 2. 

**Reading.** Both invariance levers are essentially free in isolation but compound to a _−_ 2 _._ 6 pp gap when combined. This matches the mechanism identified in §5: RoPE is TabICLv1’s only within-row symmetry-breaker (one of the “M2-only” models in §5.1), so removing it during pretraining without an architectural substitute is the harder of the two ablations. The class-invariant head adds an outputaxis constraint that is, on its own, well-absorbed by training. Mitra’s shared cell embedding (§D.1) shows that a column-invariant tabular foundation model _can_ be trained at competitive accuracy when the architecture itself supplies the symmetry from the start, rather than relying on training to absorb its removal. 

## **H Mechanism-grounded attacks: full results** 

This appendix reports the full results behind §6. Each attack tests a specific mechanistic prediction: neighbourhood pollution targets the local label aggregation shared by the two readouts (§3.1, §3.3); geometry corruption targets the learned similarity of §I.9 and the input geometry feeding TabICLv2’s prototypes; null-space PGD targets non-linear feature directions a linear baseline cannot access; SVDhiding tests whether TabPFNv2’s preprocessing pipeline and TabICLv2’s distributed representation 

41 

can be exploited by relocating signal into discarded components. Mechanism transplantation (§3.4) is reported in §I.3. 

### **H.1 Attack definitions and protocols** 

All attacks operate on the in-context window: they perturb the context set seen at inference time and measure the resulting drop in classification accuracy. Nine individual attacks are evaluated, organized into four families. 

**noise_pad.** Extra noisy context rows added (50% padding); each padded row is sampled from the per-feature empirical distribution. Floor on robustness against context-set dilution. 

**hub_poison.** Context examples (“hubs”) with the smallest average _k_ -NN distance to all other context points have their labels flipped. Hubs occupy dense, class-ambiguous regions and disproportionately contaminate nearest-neighbour readouts. 

**centroid_inj.** A synthetic example injected at the class centroid of a target class with the wrong label, pulling the effective centroid toward a competitor. 

**boundary_poison.** Context examples within a margin of the decision boundary (estimated via a held-out probe) have their labels flipped. 

**mono_cube, mono_softexp, mono_rank.** Three monotone-function attacks that transform individual features while preserving their rank order. MONO_CUBE applies a signed cube-root transform; MONO_SOFTEXP applies a soft-exponential; MONO_RANK replaces values with their within-column rank. All three break linear separability while leaving ordinal structure intact. 

**svd_hide.** Signal injected along the smallest-singular-value SVD direction of the context feature matrix into test queries but not context, creating a query–context discrepancy invisible to any decoder attending only to context features. Details in §H.10. 

**Null-space PGD.** PGD that moves context examples along the null space of the Ridge regression weight matrix, so the perturbation is invisible to Ridge while still shifting the geometry seen by TabPFN. Details in §H.9. 

Hyperparameters (noise scale, hub count, margin, PGD step size and budget) were fixed before any model evaluation. Each experiment reports mean accuracy over the relevant dataset _×_ seed grid; gaps (∆) are signed differences (model minus baseline) and 95% paired-bootstrap confidence intervals are computed over the dataset dimension. 

### **H.2 TabPFN vs. Ridge, full table** 

Table 62: TabPFN vs. Ridge under five mechanism-grounded attacks. ∆ columns report mean accuracy change (post-attack _−_ clean) on each attack’s evaluation grid. The differential (last column) is ∆TabPFN _−_ ∆Ridge; negative values mean TabPFN is _more_ damaged. Sample sizes vary because some attacks have explicit exclusion criteria (see §H.1, §H.9). 

|Attack|∆TabPFN|∆Ridge|Differential|
|---|---|---|---|
|noise_pad (4_×_padding,_n_=49)|_−_0_._076|_−_0_._041|_−_0_._035|
|hub_poison (top 15%_k_-NN hubs,_n_=49)|_−_0_._029|_−_0_._014|_−_0_._015|
|mono_rank (rank transform,_n_=49)|_−_0_._084|_−_0_._043|_−_0_._041|
|svd_hide (real-data rotation,_n_=40)|_−_0_._018|_−_0_._000|_−_0_._018|
|null-space PGD (§H.9,_n_=29)|_−_0_._047|0_._000|_−_0_._047|



The null-space attack is constructed in the null space of _W_ ridge, so by design Ridge accuracy is unaffected; TabPFNv2 drops because it attends to feature directions Ridge ignores. 

42 

### **H.3 TabPFN vs. XGBoost, full table** 

Table 63: TabPFN vs. XGBoost under nine conditions (49 cls _×_ 10 seeds; MONO_SOFTEXP _n_ =48 after one numerically degenerate dataset is dropped). Means are averages of per-dataset seed means. ∆= TabPFN _−_ XGB with 95% bootstrap CI over per-dataset paired differences. TabPFN dominates on every attack (∆ _>_ 0), with the smallest gap on MONO_SOFTEXP (+0 _._ 014) and the largest on HUB_POISON (+0 _._ 046). 

|Condition|TabPFN|XGBoost|∆|95% CI|
|---|---|---|---|---|
|baseline (clean)|0_._854|0_._828|+0_._026|[+0_._010_,_+0_._047]|
|noise_pad|0_._845|0_._821|+0_._024|[+0_._009_,_+0_._045]|
|hub_poison|0_._828|0_._782|+0_._046|[+0_._030_,_+0_._066]|
|centroid_inj|0_._855|0_._827|+0_._029|[+0_._015_,_+0_._048]|
|boundary_poison|0_._843|0_._819|+0_._024|[+0_._009_,_+0_._044]|
|mono_cube|0_._848|0_._828|+0_._020|[+0_._003_,_+0_._041]|
|mono_softexp|0_._810|0_._796|+0_._014|[+0_._002_,_+0_._025]<br>|
|mono_rank|0_._756|0_._737|+0_._020|[_−_0_._014_,_+0_._051]|
|svd_hide|0_._796|0_._758|+0_._039|[+0_._024_,_+0_._055]|



XGBoost is a non-linear, tree-structured baseline; the differential absorbs the part of vulnerability shared by any flexible classifier. 

### **H.4 TabICL vs. XGBoost, full table** 

Table 64: TabICL vs. XGBoost under all eight attacks (24 classification datasets _×_ 5 seeds). ∆= TabICL _−_ XGB with 95% bootstrap CI over per-dataset paired differences. TabICL is at least as accurate as XGBoost on every attack except MONO_RANK (CI covers zero); the largest gap is on SVD_HIDE. 

|Attack|XGBoost|TabICL|∆|95% CI|
|---|---|---|---|---|
|baseline (clean)|0_._862|0_._888|+0_._026|[+0_._001_,_+0_._048]|
|noise_pad|0_._847|0_._882|+0_._035|[+0_._018_,_+0_._054]|
|hub_poison|0_._814|0_._855|+0_._041|[+0_._014_,_+0_._064]|
|centroid_inj|0_._858|0_._885|+0_._027|[+0_._007_,_+0_._046]|
|boundary_poison|0_._840|0_._877|+0_._037|[+0_._024_,_+0_._052]|
|mono_cube|0_._862|0_._886|+0_._023|[+0_._004_,_+0_._045]|
|mono_softexp|0_._846|0_._865|+0_._019|[+0_._001_,_+0_._040]|
|mono_rank|0_._787|0_._787|+0_._000|[_−_0_._062_,_+0_._042]|
|svd_hide|0_._752|0_._838|+0_._086|[+0_._051_,_+0_._128]|



The largest gap is on SVD_HIDE (∆= +0 _._ 086): XGBoost collapses because its gradient-based training is sensitive to the injected SVD direction while TabICL’s L11 prototype readout is partially shielded by its distributed, attention-averaged encoding. 

### **H.5 Tuned MLP baseline on the same attack grid** 

A two-hidden-layer ReLU MLP with dropout 0 _._ 1 and AdamW (lr=10<sup>_−_3</sup> , weight decay 10<sup>_−_4</sup> ) is trained per (dataset _,_ seed) on the same context set as the foundation models, with identical ordinal categorical encoding and no further preprocessing. For each cell the hidden width is selected from (128 _,_ 64) _,_ (256 _,_ 128) _,_ (512 _,_ 256) by held-out validation accuracy on a 20% split of the context set with early stopping (patience 20, max 200 epochs); the chosen configuration is then re-applied unchanged to every attack variant of that cell. The MLP baseline supplies the non-linear neural control that separates mechanism-specific from generic non-linear-classifier vulnerability on the same 24-classification grid. 

43 

Table 65: Per-attack signed accuracy drop ∆ pp on the common 24-classification _×_ 5-seed grid for both foundation models, a tuned XGBoost, and a tuned MLP (2 hidden layers selected per (dataset, seed) from (128 _,_ 64), (256 _,_ 128), (512 _,_ 256) by validation accuracy; AdamW with early stopping; identical ordinal categorical encoding). 95% bootstrap CI over per-dataset seed means in brackets;<sup>_∗_</sup> denotes Holm-corrected significance at _α_ =0 _._ 05 across the eight attacks per model. Clean baselines: TabPFNv2 88 _._ 2%, TabICLv2 88 _._ 8%, XGBoost 86 _._ 3%, MLP 85 _._ 0%. 

|Attack<br>T|abPFNv2|TabICLv2|XGBoost|MLP|
|---|---|---|---|---|
|noise_pad<br>_−_1_._5<sup>_∗_</sup>|[_−_2_._4_, −_0_._6]|_−_0_._6 [_−_1_._5_,_+0_._5]|_−_1_._6 [_−_3_._4_, −_0_._1]|_−_2_._7<sup>_∗_</sup>[_−_3_._9_, −_1_._7]|
|hub_poison<br>_−_3_._9<sup>_∗_</sup><br>|[_−_5_._7_, −_2_._2]<br>|_−_3_._3<sup>_∗_</sup>[_−_4_._8_, −_1_._9]<br>|_−_4_._8<sup>_∗_</sup>[_−_6_._1_, −_3_._4]<br>|_−_2_._8<sup>_∗_</sup>[_−_4_._0_, −_1_._6]<br>|
|centroid_inj<br>_−_0_._1|[_−_0_._7_,_+0_._6]|_−_0_._3 [_−_1_._1_,_+0_._5]|_−_0_._4 [_−_0_._9_,_+0_._1]|_−_0_._7 [_−_1_._5_,_+0_._2]|
|boundary_poison<br>_−_1_._9<sup>_∗_</sup><br>|[_−_2_._9_, −_1_._1]<br>|_−_1_._2 [_−_2_._2_, −_0_._0]<br>|_−_2_._3<sup>_∗_</sup>[_−_3_._9_, −_0_._9]<br>|_−_2_._9<sup>_∗_</sup>[_−_4_._2_, −_1_._7]<br>|
|mono_cube<br>_−_1_._1|[_−_2_._4_,_+0_._1]|_−_0_._3 [_−_1_._5_,_+1_._1]|0_._0 [ 0_._0_,_ 0_._0]|_−_4_._8<sup>_∗_</sup>[_−_8_._7_, −_1_._7]|
|mono_softexp<br>_−_2_._2<sup>_∗_</sup><br>|[_−_3_._7_, −_0_._8]<br>|_−_2_._3<sup>_∗_</sup>[_−_3_._8_, −_0_._8]<br>|_−_1_._7 [_−_3_._2_, −_0_._3]<br>|_−_4_._1<sup>_∗_</sup>[_−_5_._9_, −_2_._3]<br>|
|mono_rank<br>_−_8_._0<sup>_∗_</sup>[|_−_15_._0_, −_2_._4]|_−_10_._1<sup>_∗_</sup>[_−_18_._7_, −_3_._5]|_−_7_._6<sup>_∗_</sup>[_−_14_._1_, −_2_._5]|_−_4_._3<sup>_∗_</sup>[_−_8_._4_, −_1_._1]|
|svd_hide<br>_−_8_._3<sup>_∗_</sup>[|_−_13_._6_, −_3_._7]|_−_5_._0<sup>_∗_</sup>[_−_8_._8_, −_2_._1]|_−_11_._1<sup>_∗_</sup>[_−_16_._3_, −_6_._3]|_−_10_._2<sup>_∗_</sup>[_−_17_._6_, −_4_._7]|



The MLP attains a clean 85 _._ 0% mean accuracy, lower than either foundation model (88 _._ 2%, 88 _._ 8%) but comparable to XGBoost (86 _._ 3%). The relative attack drops sort the attacks into three groups. (i) _Mechanism-targeted_ : MONO_RANK drops the foundation models by 8–10 pp while the MLP loses only 4 _._ 3 pp, and HUB_POISON drops the foundation models by 3 _._ 3–3 _._ 9 pp while the MLP loses 2 _._ 8 pp; both confirm the predictions (§6) that distance-based retrieval readouts are uniquely sensitive to inter-value-distance erasure and that vote-style readouts are uniquely sensitive to flips of high-degree context points. (ii) _Shared with any flexible classifier_ : BOUNDARY_POISON, NOISE_PAD, CENTROID_INJ and SVD_HIDE hurt the MLP at least as much as the foundation models, so the headline drops on these attacks reflect sensitivity shared by any flexible classifier rather than a readout-specific weakness. (iii) _Absorbed by the ICL prior_ : MONO_CUBE and MONO_SOFTEXP hurt the MLP roughly twice as much as the foundation models, consistent with the foundation models’ Bayesian/ICL-style averaging absorbing smooth feature warps that a freshly fitted MLP cannot. The Holm-corrected significance pattern is preserved on both foundation models when the six `mfeat-*` datasets are removed; SVD_HIDE drops shrink (TabPFNv2 _−_ 8 _._ 3 _→−_ 3 _._ 5 pp; MLP _−_ 10 _._ 2 _→−_ 5 _._ 0 pp; TabICLv2 _−_ 5 _._ 0 _→−_ 3 _._ 0 pp), reflecting the digit-family’s particular sensitivity to the SVD construction. 

### **H.6 Multiplicity controls and baseline summary** 

**Holm correction.** Within each model the eight perturbations of Table 3 are corrected at _α_ =0 _._ 05 via Holm’s step-down procedure. The corrected significance pattern keeps the rank/hub/SVD trio significant on TabPFNv2 and TabICLv2 and the hub/soft-exp/SVD trio significant on Mitra; diffuse perturbations (centroid injection, cube warp) are not significant for any model, and noise padding is significant only on TabPFNv2. The pattern is preserved for TabPFNv2 and TabICLv2 when the six `mfeat-*` datasets are removed, with reduced SVD_HIDE drops as documented in §H.5. 

**Wilcoxon signed-rank.** A pre-specified one-sided Wilcoxon signed-rank test on the per-dataset paired difference between (rank warp, hub poison) and (cube warp, centroid injection) rejects at _p <_ 10<sup>_−_3</sup> on TabPFNv2 and TabICLv2; on Mitra only the hub component contributes ( _p <_ 10<sup>_−_3</sup> on (hub, SVD) vs. (cube, centroid)). 

**Baseline summary.** On the 24-grid TabPFNv2 and TabICLv2 remain at least as accurate as a tuned XGBoost after every perturbation except SVD burial, where XGBoost itself loses _−_ 11 _._ 1 pp. On the 16-dataset mixed-type slice (§H.8, §H.7), native-categorical CatBoost reaches 76 _._ 7% and XGBoost 77 _._ 3%; TabPFNv2 ties and TabICLv2 stays ahead at 78 _._ 9%. A tuned MLP refit on the same poisoned context attains a clean 85 _._ 0% (§H.5); the rank-warp and hub-poisoning gaps are the diagnostic ones—the MLP loses 4 _._ 3 pp under rank warp while TabPFNv2/TabICLv2 lose 8–10 pp, and the MLP loses 2 _._ 8 pp under hub poisoning while TabPFNv2/TabICLv2 lose 3 _._ 3–3 _._ 9 pp, confirming the mechanistic prediction that distance-based vote and prototype readouts are uniquely sensitive to inter-value-distance erasure and high-degree label flips. The null-space PGD diagnostic (§H.9) leaves Ridge unchanged by construction but still costs TabPFNv2 _−_ 4 _._ 7 pp, ruling out any single linear readout in the Ridge feature basis. 

44 

### **H.7 Native-categorical CatBoost clean control** 

XGBoost’s ordinal handling of categorical columns may understate a modern tree baseline on this grid. We rerun the clean 24-dataset classification grid with CatBoost using native categorical handling wherever the OpenML metadata marks categorical columns, the same 80 _/_ 20 split as the main experiments, and a small validation-tuned GPU grid over depth _{_ 4 _,_ 6 _}_ , learning rate _{_ 0 _._ 03 _,_ 0 _._ 10 _}_ , and iterations _{_ 500 _,_ 1000 _}_ with early stopping. We do not rerun the attack perturbations with CatBoost: several attack operators are defined only on the shared ordinal-encoded numeric representation, so this control is clean-only. 

Table 66: Clean-only comparator means on the same 24-dataset attack grid. The mixed-type subset contains the four datasets with at least one categorical column ( `car` , `cylinder-bands` , `dresses-sales` , `eucalyptus` ). 

|Model|24-grid clean mean|mixed-type subset mean|
|---|---|---|
|TabPFN|0_._882|0_._778|
|TabICL|0_._888|0_._819|
|XGBoost|0_._863|0_._770|
|CatBoost|0_._860|0_._752|
|MLP|0_._850|0_._749|



Only four of the 24 datasets on this grid contain categorical features, so the native-categorical advantage is a narrow probe. On this suite CatBoost lands close to tuned XGBoost overall and does not overtake either foundation model on the mixed-type subset; the clean-baseline ordering holds. The 16-dataset slice below extends this evidence to a broader mixed-type benchmark. 

The four-dataset mixed-type subset is narrow. We rerun the same native-categorical CatBoost protocol on every classification dataset in the 49-dataset benchmark whose OpenML metadata marks at least one categorical column, and add a matching native-categorical XGBoost control on the same fixed slice. This mixed-type slice contains the 16 datasets enumerated in §A.4. The clean ordering holds (Tab. 67): TabICLv2 stays ahead of CatBoost on 14 _/_ 16 datasets (one tie), with a mean paired advantage of +2 _._ 3 pp and 95% CI [+1 _._ 1 _,_ +3 _._ 5]; TabPFNv2 lands closer (+0 _._ 8 pp, CI [ _−_ 0 _._ 2 _,_ +1 _._ 6]) and loses on 5 _/_ 16. Native-categorical XGBoost falls between the two tree baselines at 77 _._ 3% mean accuracy: TabPFNv2 ties overall (+0 _._ 2 pp, CI [ _−_ 1 _._ 7 _,_ +1 _._ 8]), and TabICLv2 leads by +1 _._ 7 pp (CI [+0 _._ 5 _,_ +2 _._ 9]). This extends the mixed-type clean-baseline evidence well beyond the four-dataset probe; attack-time mixed-type behavior is addressed in §H.8. 

Table 67: Broader native-categorical tree controls on the 16 classification datasets in the 49-dataset benchmark with at least one categorical column (5 seeds / dataset). TabPFN and TabICL means use the same 80 _/_ 20 split as the 49-dataset readout-grid evaluations. ∆= model _−_ CatBoost with 95% bootstrap CI over per-dataset paired differences. 

|Model|mixed-type clean mean|∆vs. CatBoost [95% CI]|
|---|---|---|
|TabPFN|0_._774|+0_._008 [_−_0_._002_,_+0_._016]|
|TabICL|0_._789|<br>+0_._023 [+0_._011_,_+0_._035]|
|XGBoost (native-categorical)|0_._773|<br>+0_._006 [_−_0_._010_,_+0_._023]|
|CatBoost|0_._767|<br>—|



### **H.8 Broader mixed-type attack slice** 

The clean-only controls above establish baseline breadth; attack-time mixed-type behavior is the next question. We rerun the three highest-signal attacks from the main text—hub poison, rank warp, and SVD burial—on the full 16-dataset mixed-type slice under the same shared ordinal-encoded numeric view used by the foundation models (5 seeds / dataset). The attack-pipeline clean means on this slice are 0 _._ 781 for TabPFNv2 and 0 _._ 791 for TabICLv2. The same ordering survives (Tab. 68): hub poison costs 2–3 pp, rank warp dominates for both models, and SVD burial is small on average. Rank- and hub-style attacks remain the strongest within-design predictor of where the foundation models lose ground, including on mixed-type data. 

45 

Table 68: Broader mixed-type attack slice on the 16 mixed-type classification datasets under the shared ordinalised numeric view (5 seeds / dataset). ∆ is the within-model pp change from the clean mixed16 attack-pipeline baseline. 

|Attack|TabPFNv2∆pp [95% CI]|TabICLv2∆pp [95% CI]|
|---|---|---|
|hub poison|_−_2_._4 [_−_3_._7_, −_1_._4]|_−_2_._8 [_−_4_._1_, −_1_._5]|
|rank warp|_−_17_._2 [_−_28_._9_, −_7_._1]|_−_21_._0 [_−_34_._5_, −_10_._2]|
|SVD burial|_−_0_._2 [_−_0_._7_,_+0_._5]|_−_1_._0 [_−_2_._2_, −_0_._1]|



### **H.9 Null-space PGD details** 

The null space of the Ridge weight matrix _N_ ( _W_ ridge) is computed via a thin SVD of _W_ ridge<sup>_⊤_; columns</sup> of _V_ with singular values below 10<sup>_−_6</sup> _· σ_ max span the near-null space. 

**Dataset exclusion.** Three datasets excluded for zero null-space dimension. A further 17 excluded as numerically degenerate (fewer than five directions pass the threshold). 

**PGD hyperparameters.** _η_ = 0 _._ 01 (in unit-variance feature units), _T_ = 200 steps, _ℓ_ 2 budget _ε_ = 1 _._ 0. After each step, the perturbation is projected onto the null space (so Ridge sees no change) and clipped to the _ℓ_ 2 ball. Untargeted: maximize the cross-entropy of the Ridge prediction distribution under TabPFN, using Ridge’s soft output as a fixed proxy. 

### **H.10 SVD-hiding details and per-dataset breakdown** 

**Mechanism.** Let [ _X_ ctx; _X_ query] _∈_ R<sup>(</sup><sup>_n_+</sup><sup>_m_)</sup><sup>_×d_</sup> be the stacked context-and-query feature matrix. A thin SVD yields singular values _σ_ 1 _≥· · · ≥ σd_ . We divide the top _k_ = _⌈d/_ 4 _⌉_ singular values by 10 and multiply the remaining _d − k_ by 10, then reconstruct and split back into context and query. This dampens the high-variance components that similarity-and-vote and prototype readouts rely on and amplifies the low-variance components most decoders ignore, hitting context and query consistently. 

**Three-copy preprocessing redundancy.** TabPFN’s three-copy preprocessing (original, ranknormalized, quantile-normalized features) re-bases the geometry across copies; the high-variance components in raw feature space are not the high-variance components in the encoder’s input space, so the reweighting produces only partial perturbation of TabPFN’s actual input. 

**Real vs. synthetic comparison.** On real datasets, the svd_hide gap between TabICL and XGBoost (∆= +0 _._ 086, §H.4) is substantially larger than on synthetic datasets where the low-variance direction is by construction uninformative. The attack exploits genuine low-variance signal that XGBoost uses (via split-based selection) but TabICL’s L11 prototype readout discards. 

**Per-dataset breakdown.** On 34 _/_ 49 datasets XGBoost is more damaged than TabPFN (mean drops _−_ 0 _._ 058 vs. _−_ 0 _._ 070, differential +0 _._ 013). On hill-valley TabPFN scores 0 _._ 593 on raw features but _{_ 0 _._ 934 _,_ 0 _._ 926 _,_ 0 _._ 918 _,_ 0 _._ 914 _}_ when projected to top- _{_ 10 _,_ 20 _,_ 50 _,_ 100 _}_ PCs (SVD-projected variant of hill-valley). 

Table 69: SVD_HIDE: five datasets with the largest XGBoost drop. 

|Dataset|TabPFN clean|TabPFN svd|∆TabPFN|XGBoost clean|XGBoost svd|∆XGBoost|
|---|---|---|---|---|---|---|
|mfeat-karhunen|0_._963|0_._613|_−_0_._350|0_._925|0_._508|_−_0_._418|
|mfeat-pixel|0_._960|0_._540|_−_0_._420|0_._948|0_._603|_−_0_._345|
|mfeat-fourier|0_._895|0_._585|_−_0_._310|0_._860|0_._557|_−_0_._302|
|semeion|0_._918|0_._718|_−_0_._201|0_._897|0_._611|_−_0_._285|
|sign_1d|0_._980|0_._840|_−_0_._140|1_._000|0_._720|_−_0_._280|



46 

### **H.11 Side-by-side TabPFN/TabICL contrast table** 

Table 70: Mechanistic comparison of TabPFN [Hollmann et al., 2025] and TabICL [Qu et al., 2025] across five dimensions. 

|Dimension|TabPFN|TabICL|
|---|---|---|
|Critical early block|Block-0:<br>encodes formats & shapes<br>(§2.3)|ColEmb-2: SetTransformer readout|
|Attention role|Sharp & locally necessary at L9 (§B.2.5)|Individually redundant; jointly necessary<br>(§C.2.3)|
|Similarity tracking|_r_ = 0_._34(attn. vs. query–context sim.;<br>§I.9)|_r <_0_._08at every block (§C.3.4)|
|Rank profle|Early bottleneck (§2.2)|Late prototype-formation phase (§2.2)|
|Operational readout|Attention-weighted label vote (§3.1)|Nearest-class prototype over L11 reps.<br>(§3.3)|



Table 70 summarizes the five mechanistic dimensions that distinguish the two architectures (referenced from §3.4). 

## **I Additional experiments** 

All experiments in this appendix are inference-only on the same model releases as the rest of the paper; none require pretraining (§7). Statistical reporting follows §A.5. 

### **I.1 TabICL-reg L11 kNN** 

Regression analogue of §C.3.5: 10 reg datasets _×_ 5 seeds, four distance spaces, four neighbour counts. Each cell reports the across-dataset mean of (Pearson _r_ between kNN-imputed and TabICLreg prediction _/_ kNN _R_<sup>2</sup> against the ground truth). 

Table 71: TabICL-reg kNN agreement / kNN _R_<sup>2</sup> aggregated over 10 regression datasets, 5 seeds. Rep- _ℓ_ 2 at L11 dominates input- _ℓ_ 2 at every _k_ ; the kNN-in-final-rep finding holds for regression as for classification. 

|_k_|input-_ℓ_2|rep-_ℓ_2 (L11)|rep-cosine (L11)|rep-Mahalanobis (L11)|
|---|---|---|---|---|
|_k_=1|0.603 /_−_0.375|0.784 / 0.542|0.719 / 0.459|0.207 /_−_0.218|
|_k_=3|0.665 / 0.286|0.842 / 0.575|0.781 / 0.504|0.324 / 0.017|
|_k_=5|0.681 / 0.386|**0.859 / 0.577**|0.801 / 0.515|0.360 / 0.044|
|_k_=10|0.689 / 0.433|0.871 / 0.566|0.830 / 0.519|0.382 / 0.036|



### **I.2 TabICL-reg per-block vote refutation** 

ˆ Regression analogue of §C.3.6: per-block pseudo-vote _yL_ =<sup>�</sup> _i_<sup>_αi yi_from head-mean test</sup><sup>_→_train</sup> attention. 10 reg datasets _×_ 5 seeds. 

Table 72: TabICL-reg per-block weighted-label vote, mean across 10 datasets. Pearson is correlation ˆ between _yL_ and the model’s own prediction (or ground truth). Best layer L9 ( _r_ = 0 _._ 732) is far below the TabPFN-reg ceiling _r ≥_ 0 _._ 896; six of twelve layers are negative. Weighted-label vote is not the dominant TabICL-reg mechanism. 

|Layer|Pearson(pred)|Pearson(truth)|vote_R_<sup>2</sup>|Layer|Pearson(pred)|Pearson(truth)|vote_R_<sup>2</sup>|
|---|---|---|---|---|---|---|---|
|L0|_−_0.222|_−_0.199|_−_0.038|L6|_−_0.231|_−_0.221|_−_0.019|
|L1|+0.396|+0.384|+0.006|L7|_−_0.429|_−_0.391|_−_0.033|
|L2|+0.442|+0.380|+0.001|L8|+0.642|+0.557|+0.032|
|L3|+0.265|+0.213|_−_0.006|L9|+**0**_._**732**|+0.647|+0.064|
|L4|_−_0.523|_−_0.485|_−_0.029|L10|+0.076|+0.040|_−_0.008|
|L5|_−_0.276|_−_0.243|_−_0.027|L11|+0.672|+0.643|+0.088|



47 

### **I.3 Mechanism transplantation** 

Direct causal test of §3.4, 49 cls _×_ 5 seeds. Each row is a per-dataset paired comparison; we report the across-dataset means and a bootstrap 95 % CI over per-dataset paired differences (10<sup>4</sup> resamples) on ∆. Rows (iii) and (vi) add a minimal host-side control: logistic regression on the frozen host representation. 

### **Four conditions.** 

- (i) TabPFN native vote (L9). 

- (ii) TabPFN with prototype readout: _p_ ˆ( _c_ ) _∝_ exp( _−∥hL_ 11 _− µc∥_ 2<sup>2),</sup><sup>_µc_unweighted per-class</sup> means of L11 representations of all training rows. 

- (iii) TabICL native prototype (L11). 

- (iv) TabICL with vote readout: at the sharpest-attention block _b_<sup>_∗_</sup> ( _D_ ) selected per dataset _D_ , _p_ ˆ( _c_ ) = _H_<sup><u>1</u></sup> � _h_ � _i_<sup>_α_</sup> _h_<sup>(</sup><sup>_i_)</sup><sup>**1**[</sup><sup>_yi_=</sup><sup>_c_].</sup> 

Table 73: Bootstrap CIs for the cross-transplant and fitted-host linear-head conditions of Table 1 (49 cls _×_ 5 seeds). CI: 95% paired-bootstrap over datasets. Naive cross-transfer is catastrophic on both sides, but the fitted-host recovery is asymmetric: TabPFNv2 remains far below native, whereas TabICLv2 nearly recovers. 

|Condition|Description|Accuracy|vs. native∆(95% CI)|
|---|---|---|---|
|(i)|TabPFN native|0_._854|—|
|(ii)|TabPFN+prototype|0_._523|_−_0_._331 [_−_0_._412_, −_0_._253]|
|(iii)|TabPFN+linear head|0_._617|_−_0_._237 [_−_0_._305_, −_0_._172]|
|(iv)|TabICL native|0_._864|—|
|(v)|TabICL+vote|0_._470|_−_0_._395 [_−_0_._484_, −_0_._307]|
|(vi)|TabICL+linear head|0_._859|_−_0_._005 [_−_0_._009_, −_0_._001]|



The naive cross-transplants are catastrophic and the CIs do not cover zero (Table 73). The fitted-host control is asymmetric: TabPFN remains far below native while TabICL nearly recovers (within 1 pp on 32 _/_ 49 datasets), so the stronger non-portability claim is asymmetric. 

**Per-dataset breakdown.** Both readouts collapse on multi-feature digit datasets and synthetic mixtures (Table 74). 

Table 74: Transplant accuracies on the five datasets with the largest TabICL drop ∆ICL. 

|Dataset|PFN nat.|PFN proto|∆PFN|ICL nat.|ICL vote|∆ICL|
|---|---|---|---|---|---|---|
|mfeat-karhunen|0_._969|0_._161|_−_0_._808|0_._984|0_._043|_−_0_._941|
|quadrant_2d|0_._992|0_._412|_−_0_._580|0_._996|0_._080|_−_0_._916|
|mfeat-factors|0_._961|0_._236|_−_0_._725|0_._984|0_._089|_−_0_._895|
|mfeat-pixel|0_._962|0_._254|_−_0_._709|0_._979|0_._084|_−_0_._895|
|semeion|0_._909|0_._277|_−_0_._632|0_._961|0_._076|_−_0_._885|



**Best-layer transplant.** Picking the layer maximizing held-out accuracy per dataset (best-layer transplant, 49 datasets) shrinks the gap for TabICL ( _−_ 0 _._ 395 _→−_ 0 _._ 263, acc. 0 _._ 470 _→_ 0 _._ 601) but _widens_ it for TabPFN ( _−_ 0 _._ 331 _→−_ 0 _._ 549, acc. 0 _._ 523 _→_ 0 _._ 305): the prototype-optimal layer is shallow (median 0) where TabPFN representations have not yet specialized. 

### **I.4 TabPFN L11 kNN agreement (strengthened test)** 

Restated from §B.2.6 at the readout layer L11 (49 cls _×_ 5 seeds). Each cell is across-dataset mean kNN agreement with TabPFN. 

48 

Table 75: TabPFN kNN agreement at the readout layer L11. Every L11 metric is strictly worse than L9 and worse than input- _ℓ_ 2 at every _k_ . The “kNN-on-final-representation” hypothesis remains refuted at the readout layer. 

|_k_|input-_ℓ_2|rep-_ℓ_2 L9|rep-_ℓ_2 L11|rep-cosine L11|rep-Mahalanobis L11|
|---|---|---|---|---|---|
|_k_=1|0.782|0.768|0.528|0.528|0.356|
|_k_=3|0.800|0.772|0.554|0.554|0.378|
|_k_=5|**0.811**|0.774|0.561|0.561|0.395|
|_k_=10|0.823|0.777|0.572|0.572|0.429|



### **I.5 TabICL extra-attack robustness** 

Three perturbations beyond the eight reported in §6. 24 datasets _×_ 5 seeds; ∆= TabICL _−_ XGB. 

Table 76: TabICL never loses on the mean across the three additional attacks; consistent with the 5-attack ranking in the main text and with the 8-attack panel as a whole. 

|Attack|TabICL acc|XGB acc|∆(mean)|TabICL wins|
|---|---|---|---|---|
|`boundary_poison`|0.877|0.840|+0.037|22 / 24|
|`mono_cube`|0.886|0.862|+0.023|18 / 24|
|`mono_rank`|0.787|0.787|+0.000|16 / 24|



### **I.6 Column-marginal vs. invariance-cost correlation** 

Post-hoc, CPU-only computation for §B.3.2. For each dataset compute (i) size of largest column-set with identical 1D marginals at JS tolerance _τ ∈{_ 0 _._ 01 _,_ 0 _._ 05 _,_ 0 _._ 10 _}_ , (ii) corresponding invarianceablation gap. Spearman _ρ_ across 49 datasets. 

Table 77: Spearman correlation between column-marginal-duplication count and per-dataset invariance gap. Only TabICL-v1 no-RoPE exhibits a non-zero population gap that is positively correlated with marginal duplication; TabPFN _W_ =0 is null in both. The JS-screen identifies collapse-prone datasets a-priori. 

|Model|mean gap|_ρ_(_τ_=0_._01)|_ρ_(_τ_=0_._05)|_ρ_(_τ_=0_._10)|
|---|---|---|---|---|
|TabPFN_W_=0|_−_0.0001|+0.042 (p=0.78)|+0.081 (p=0.58)|+0.064 (p=0.66)|
|TabICL-v1 no-RoPE|+0.1540|+0.201 (p=0.17)|+0.316 (p=0.03)|+0.322 (p=0.02)|
|TabICL-v2 no-RoPE|+0.0016|+0.231 (p=0.11)|+0.310 (p=0.03)|+0.325 (p=0.02)|



### **I.7 Collapse** _f_ **-sweep** 

Parametric extension of §5.2. Synthesise Qu et al. [2025] style stress data where the fraction _f ∈{_ 0 _,_ 0 _._ 25 _,_ 0 _._ 5 _,_ 0 _._ 75 _,_ 1 _}_ of _d_ =12 columns are i.i.d. from a single shared discrete marginal, the remainder from heterogeneous marginals; _K_ =5 classes, _n_ =800 rows, 10 seeds each. Cell entries are paired ∆ (ablated _−_ full) with bootstrap 95 % CI. 

Table 78: Collapse _f_ -sweep paired ∆ accuracy. TabICL-v1 no-RoPE falls monotonically with _f_ (with non-overlapping CIs above _f_ =0 _._ 25). TabPFN _W_ =0 and TabICL-v2 no-RoPE stay within _|_ ∆ _| ≤_ 0 _._ 001. TabPFN 2nd-slot= 0 shows a milder M3-pair-channel load-bearing effect that grows from _−_ 0 _._ 046 at _f_ =0 to _−_ 0 _._ 140 at _f_ =1. 

|_f_<br>Ta|bPFN_W_=0|TabPFN 2nd-slot=0|TabICL-v1 no-RoPE|TabICL-v2 no-RoPE|
|---|---|---|---|---|
|0.00 _−_0.000|[_−_0_._001_,_+0_._000]|_−_0.046[_−_0_._096_, −_0_._004]|_−_**0**_._**255**[_−_0_._318_, −_0_._187]|_−_0.001[_−_0_._004_,_+0_._001]|
|0.25 _−_0.000|[_−_0_._001_,_+0_._001]|_−_0.017[_−_0_._041_,_+0_._000]|_−_**0**_._**254**[_−_0_._311_, −_0_._190]|_−_0.000[_−_0_._001_,_+0_._001]|
|0.50 _−_0.001|[_−_0_._002_,_+0_._000]|_−_0.061[_−_0_._108_, −_0_._022]|_−_**0**_._**459**[_−_0_._474_, −_0_._443]|_−_0.001[_−_0_._003_,_+0_._001]|
|0.75 _−_0.000|[_−_0_._001_,_+0_._000]|_−_0.062[_−_0_._105_, −_0_._021]|_−_**0**_._**469**[_−_0_._488_, −_0_._450]|_−_0.000[_−_0_._003_,_+0_._002]|
|1.00 +0.000|[+0_._000_,_+0_._001]|_−_**0**_._**140**[_−_0_._186_, −_0_._102]|_−_**0**_._**472**[_−_0_._489_, −_0_._448]|_−_0.000[_−_0_._002_,_+0_._002]|



49 

### **I.8 Combined within-row ablation on v2 backbones** 

Each individual within-row ablation on a v2 backbone is load-free (§5.2, Table 78). To test whether v2 carries _redundant_ within-row defenses, we ablate both available routes simultaneously, without retraining: TabPFNv2 with _W_ =0 and pair second-slot zeroed; TabICLv2 with RoPE removed and the slots 1 _,_ 2 of every ( _B, T, G,_ 3) circular feature group zeroed (slot 0 of each group, the original feature index, is preserved; the trained `in_linear` weights on slots 1 _,_ 2 are still applied to zeros). Both ablations are wrapped in `VerifiedPatch` : forward hooks confirm `tf_row.rope` is `None` at every block call and that the post-grouping slots 1: are zero on every forward pass. Twenty seeds with paired-bootstrap CIs over seeds; per-seed re-resampling on the synthetic. 

Table 79: Combined within-row ablation on v2 backbones. Each individual ablation is load-free (Table 78); the combined ablations expose latent collapse. TabICLv2 with both within-row routes off lands at the majority baseline on the _d_ =12 balance-like stress, confirming that its individual-ablation “free” result reflected redundancy rather than the absence of a within-row defense. TabPFNv2 with both routes off survives but degrades. Compare to the Mitra row-attention ablation (Table 39, 0 _._ 95 _→_ 0 _._ 08): Mitra’s single load-bearing mechanism has no architectural fall-back, so its failure mode is below-majority rather than at-majority. 

|backbone|ablation|dataset|accuracy|∆vs. full (95%CI)|
|---|---|---|---|---|
|TabPFNv2|full|balance-scale|0_._962|—|
|TabPFNv2|_W_=0 +2nd-slot= 0|balance-scale|0_._890|_−_0_._071 [_−_0_._106_, −_0_._044]|
|TabPFNv2|full|balance-like,_d_=12|0_._970|—|
|TabPFNv2|_W_=0 +2nd-slot= 0|balance-like,_d_=12|0_._855|_−_0_._115 [_−_0_._144_, −_0_._091]|
|TabICLv2|full|balance-scale|0_._978|—|
|TabICLv2|no-RoPE+no-grouping|balance-scale<br>|0_._823|_−_0_._154 [_−_0_._165_, −_0_._144]|
|TabICLv2|full|balance-like,_d_=12|0_._990|—|
|TabICLv2|no-RoPE+no-grouping|balance-like,_d_=12|**0**_._**485**|_−_**0**_._**504**[_−_**0**_._**522**_, −_**0**_._**474**]|



majority baseline 0 _._ 46 _/_ 0 _._ 47 (balance-scale/like) 

### **I.9 TabPFN L9 attention as learned similarity** 

Per test row fit Ridge of L9 column-attention target-row weights onto a four-feature basis _{ℓ_ 2 _,_ cos _, ℓ_ 1 _,_ rep- _ℓ_ 2( _L_ 8) _}_ . 49 datasets _×_ 5 seeds; we report the distribution of per-dataset mean _R_<sup>2</sup> . 

Table 80: TabPFN L9 column-attention is far better explained by the model’s own L8 representation distance ( _R_<sup>2</sup> =0 _._ 48) than by raw input _ℓ_ 2 ( _R_<sup>2</sup> =0 _._ 22). Full basis explains _∼_ 60% of attention variance, consistent with “learned similarity” and ruling out a pure _ℓ_ 2-on-inputs mechanism. 

|Predictor set|mean_R_<sup>2</sup>|median|min|max|
|---|---|---|---|---|
|input_ℓ_2 only|0.218|0.168|0.004|0.774|
|rep-_ℓ_2 (L8) only|0.477|0.488|0.047|0.788|
|<br>full 4-feature basis|0.600|0.606|0.149|0.853|
|gap (full_−ℓ_2)|+0.382|+0.382|+0.079|+0.765|
|gap (rep-_ℓ_2 _−_input-_ℓ_2)|+0.259|+0.292|_−_0.167|+0.695|



### **I.10 Worst-case real-data uniform-attention drop** 

Re-aggregate per-dataset all-12-block uniform-attention drop on TabICL-cls, excluding synthetic `sign_1d` (48 real datasets). 

50 

Table 81: Worst-case drops on real data: top-5 are MiceProtein 0.78, mfeat-karhunen 0.70, mfeatzernike 0.68, mfeat-morphological 0.64, mfeat-fourier 0.61. The headline mean dilution by easy datasets hides a median drop of 29 pp; the worst-case real drop matches the synthetic Qu drop and is the more honest figure for the distributed-mechanism claim of §C.2.3. 

Statistic median mean p75 p90 max ∆ accuracy (uniform all-12) 0.293 0.290 0.441 0.602 0.777 Threshold _>_ 0 _._ 05 _>_ 0 _._ 10 _>_ 0 _._ 20 _>_ 0 _._ 30 _>_ 0 _._ 50 # datasets exceeding 36 / 48 34 / 48 29 / 48 23 / 48 9 / 48 

### **I.11 Majority-class baseline** 

Per-dataset majority-class accuracy on the 49-dataset benchmark, broken down by class cardinality. 

Table 82: Majority-class trivial baseline. The 10 datasets with majority _>_ 0 _._ 7 (pc1 0 _._ 93, climate 0 _._ 92, pc3 0 _._ 90, Is-this-good 0 _._ 88, pc4 0 _._ 87, MIC 0 _._ 85, kc2 0 _._ 79, anneal 0 _._ 76, blood-tx 0 _._ 76, ilpd 0 _._ 71) make a kNN agreement of 0 _._ 85 on a _C_ =2 near-trivial problem far less informative than the same number on a balanced 10-class `mfeat-*` problem (majority 0 _._ 105). 

|Subset|_n_|mean|median|max|
|---|---|---|---|---|
|All 49 datasets|49|0.499|0.540|0.932|
|Binary (_C_=2)|24|0.684|0.700|0.932|
|_C_=3|6|0.421|0.387|0.760|
|Multiclass (_C≥_4)|19|0.289|0.250|0.847|



### **I.12 Column-permutation spread (full distribution)** 

_−_ Full distribution of per-dataset best worst accuracy under arbitrary column permutations, source for Figure 5. 

Table 83: Per-dataset accuracy spread under column permutation, 49 datasets. The headline “8 _._ 7 pp” is the maximum across datasets; median spread is _∼_ 2 _._ 5 pp for TabPFN/TabICLv1 and 1 _._ 5 pp for TabICLv2. Worst-3 datasets per model: TabPFN {cylinder-bands 0 _._ 080, tic-tac-toe 0 _._ 075, dresses-sales 0 _._ 070}; TabICLv1 {random_labels 0 _._ 087, dresses-sales 0 _._ 080, hill-valley 0 _._ 078}; TabICLv2 {random_labels 0 _._ 087, dresses-sales 0 _._ 053, cylinder-bands 0 _._ 052}. The typical-case column-invariance is an order of magnitude stronger than the maximum suggests. 

|Model|median|mean|p75|p90|max|_>_1pp|_>_2pp|_>_5pp|
|---|---|---|---|---|---|---|---|---|
|TabPFNv2|0.027|0.029|0.041|0.053|0.080|41|29|6|
|TabICLv1|0.025|0.029|0.040|0.053|0.087|41|29|7|
|TabICLv2|0.015|0.019|0.026|0.039|0.087|29|18|3|



### **I.13 Per-layer kNN agreement curve (TabICL)** 

Extends §C.3.5 from L11 to all 12 ICL layers using _k_ =5 kNN on 49 datasets _×_ 5 seeds. 

51 

Table 84: TabICL per-layer kNN-against-TabICL agreement and standalone kNN accuracy. Inputspace kNN already agrees 0 _._ 817; agreement rises monotonically from 0 _._ 885 (L0) to 0 _._ 936 (L11), an L11-vs-L8 edge of only 1 _._ 4 pp. Reframes the L11-kNN claim as continuous refinement rather than a late, sudden re-shaping. 

|Layer|cosine agree.|cosine kNN acc|_ℓ_2 agree.|_ℓ_2 kNN acc|
|---|---|---|---|---|
|input (_ℓ_2 only)|—|—|0.817|—|
|L0|0.887|0.827|0.885|0.824|
|L1|0.888|0.826|0.885|0.825|
|L2|0.891|0.828|0.890|0.827|
|L3|0.890|0.832|0.886|0.829|
|L4|0.900|0.837|0.896|0.835|
|L5|0.900|0.835|0.894|0.832|
|L6|0.908|0.840|0.901|0.837|
|L7|0.904|0.839|0.902|0.837|
|L8|0.924|0.847|0.923|0.847|
|L9|0.922|0.849|0.924|0.849|
|L10|0.934|0.855|0.935|0.855|
|L11|**0.936**|**0.857**|**0.936**|**0.856**|



### **I.14 Readout grid with calibration** 

Eight readouts _×_ five metrics on 49 classification datasets _×_ 5 seeds. Readouts: native, L9-vote (PFN) / sharp-vote (ICL), L11-prototype, L11-kNN. ECE uses 15 equal-width confidence bins per (dataset, seed); Table 85 reports means over the 49 per-dataset seed means. 

Table 85: Per-readout means across 49 datasets (5 seeds / dataset). PFN L9-vote pairs a 5 _._ 0 pp accuracy drop with a _∼_ 5 _×_ ECE jump and _∼_ 2 _×_ NLL: vote sacrifices calibration. ICL native / L11proto / L11-kNN are within 1 _._ 0 pp on accuracy, but not on calibration: prototype raises ECE/NLL to 0 _._ 463 _/_ 1 _._ 058 and kNN to 0 _._ 115 _/_ 2 _._ 840 vs. 0 _._ 042 _/_ 0 _._ 303 natively. Both prototype/kNN heads also ruin TabPFN’s calibration (NLL up to 9 _._ 7). 

|Readout|acc|ECE|NLL|macro-P|macro-R|
|---|---|---|---|---|---|
|PFN-native|0.854|0.043|0.329|0.806|0.786|
|PFN L9-vote|0.804|0.201|0.625|0.750|0.724|
|PFN L11-prototype|0.523|0.277|1.178|0.506|0.572|
|PFN L11-kNN|0.547|0.371|9.715|0.435|0.522|
|ICL-native|0.864|0.042|0.303|0.823|0.801|
|ICL L11-prototype|0.854|0.463|1.058|0.806|0.817|
|ICL L11-kNN|0.856|0.115|2.840|0.816|0.796|
|ICL sharp-vote|0.470|0.125|1.094|0.265|0.325|



**Dataset-bootstrap calibration intervals.** Treating the 49 per-dataset seed means as the reporting unit, PFN-native vs. PFN L9-vote shifts ECE from 0 _._ 043 [0 _._ 036 _,_ 0 _._ 051] to 0 _._ 201 [0 _._ 176 _,_ 0 _._ 228] and NLL from 0 _._ 329 [0 _._ 244 _,_ 0 _._ 426] to 0 _._ 625 [0 _._ 521 _,_ 0 _._ 735]. For TabICL, native vs. L11-prototype shifts ECE from 0 _._ 042 [0 _._ 034 _,_ 0 _._ 050] to 0 _._ 463 [0 _._ 404 _,_ 0 _._ 522] and NLL from 0 _._ 303 [0 _._ 219 _,_ 0 _._ 398] to 1 _._ 058 [0 _._ 916 _,_ 1 _._ 208]; the L11- _k_ NN surrogate keeps near-native accuracy but still widens ECE/NLL to 0 _._ 115 [0 _._ 085 _,_ 0 _._ 148] and 2 _._ 840 [2 _._ 019 _,_ 3 _._ 785]. All intervals are 95% bootstrap CIs over per-dataset seed means. 

52 

Table 86: Cross-readout agreement (mean over 49 datasets). The two “native” stacks agree 0 _._ 93, but the same surrogate readout produces disjoint predictions across architectures (PFN-proto _↔_ ICL-proto 0 _._ 54). Stacks behave alike at prediction level, not at representation level. 

|Pair|label agreement|
|---|---|
|PFN-native vs. ICL-native|0.934|
|PFN-native vs. PFN L9-vote|0.874|
|PFN-native vs. PFN L11-proto|0.518|
|PFN-native vs. PFN L11-kNN|0.555|
|ICL-native vs. ICL L11-proto|0.922|
|ICL-native vs. ICL L11-kNN|0.936|
|ICL-native vs. ICL sharp-vote|0.496|
|PFN L11-proto vs. ICL L11-proto|0.542|
|PFN L11-kNN vs. ICL L11-kNN|0.536|
|PFN L9-vote vs. ICL-native|0.862|



### **I.15 Subgroup-collapse stress test** 

Synthetic construction maximally stressing within-group collapse: 12 features in 6 pair-groups of 2, columns within a pair share a marginal, labels depend on within-pair products. 5 seeds. 

Table 87: Subgroup-collapse stress: invariance preserved (∆acc = 0 _._ 001). Train collapse rate is near-zero (0 _._ 003); the _∼_ 20% test collapse (cosine _>_ 0 _._ 99) is a moderate pre-readout phenomenon on heterogeneous queries that does not impair accuracy. 

|seed|acc full|acc_W_=0|test collapse full|test collapse_W_=0|
|---|---|---|---|---|
|0|0.725|0.725|0.194|0.191|
|1|0.760|0.760|0.147|0.140|
|2|0.810|0.810|0.215|0.209|
|3|0.645|0.640|0.279|0.289|
|4|0.710|0.710|0.152|0.152|
|mean|0.730|0.729|0.198|0.196|



### **I.16 TabPFN per-head ablation at L8/L9/L10** 

For each _L ∈{_ 8 _,_ 9 _,_ 10 _}_ and each of _H_ =6 heads, zero the output-projection slice for head _h_ at that block; measure per-seed accuracy delta against baseline on 10 datasets _×_ 5 seeds. 

Table 88: Per-block per-head zero-out drop. “# critical” counts heads with _>_ 5 pp drop in a given (dataset, seed). L9 has the largest single-head impact (∆ _h_ 0=3 _._ 1 pp on average, max 17 _._ 3 pp), but no head dominates: the L9 vote signal is diffuse with a slight head-0 peak rather than localized. 

|Layer|mean max-drop|worst max-drop|mean # critical|h0|h1|h2|h3|h4|h5|
|---|---|---|---|---|---|---|---|---|---|
|L8|0.031|0.204|0.14|0.008|0.003|0.019|0.000|0.002|_−_0.000|
|L9|0.039|0.173|0.32|0.031|0.009|0.004|0.003|0.000|0.000|
|L10|0.012|0.067|0.08|0.005|0.003|0.002|0.003|0.003|0.001|



### **I.17 Effective rank vs. number of classes** 

Per-layer Pearson correlation between #classes _C_ and post-block effective rank, computed by joining the per-block geometry measurements of §B.1.4 with the per-dataset class counts used in the vote analysis of §B.2.3 (49 classification datasets). 

53 

Table 90: Claim / evidence / scope summary for the main mechanistic statements. “Full grid” denotes the 49-dataset classification suite unless the row says otherwise. 

|Claim|Main evidence in this paper|Scope / caveat|
|---|---|---|
|TabPFNv2 late vote readout|L9 is the sharpest layer on49_/_49datasets; vote–model<br>mean_r_=0_._89; uniform-L9 and block-9 interventions<br>collapse the intact readout.|Full grid;<br>descriptive surrogate only<br>(accuracy 0_._804 vs. 0_._854, ECE/NLL<br>0_._201_/_0_._625vs.0_._043_/_0_._329).|
|TabICLv2 prototype/_k_NN<br>readout|L11 prototype/_k_NN reach 0_._854_/_0_._856 vs. native<br>0_._864; PFN-vote on ICL drops to0_._47; earlier releases<br>keep the same prototype/_k_NN-versus-vote ordering.|Full grid across releases; argmax-faithful<br>but not calibration-preserving (ECE/NLL<br>0_._463_/_1_._058<br>and<br>0_._115_/_2_._840<br>vs.<br>0_._042_/_0_._303).|
|Redundant positional com-<br>ponents at inference|TabPFNv2_W_=0and TabICLv2 no-RoPE preserve the<br>benchmark mean while granting the matching symme-<br>tries; synthetic constructions recover the same invari-<br>ances.|Full benchmark mean plus synthetic checks;<br>inference-only, so “redundant” here means<br>unused by these weights on this benchmark.|
|Mechanism-grounded<br>ro-|Hub/rank/SVD ordering matches the proposed readouts|Matched24-dataset stress test +16-dataset|
|bustness|on the 24-grid; the same trio on the broader mixed16<br>slice keeps rank as the dominant mixed-type attack and<br>leaves clean tree baselines competitive.|mixed-type attack extension under the<br>shared ordinalised numeric view; native-<br>categorical tree controls remain clean-only.|



Table 89: Per-layer effective rank vs. class count. TabPFN’s L11 effective rank correlates strongly with _C_ ( _r_ =0 _._ 64), with mean rank rising from 21 _._ 1 ( _C≤_ 3) to 31 _._ 7 ( _C≥_ 4). TabICL is already classaware at L0 and the correlation decays through depth. The “constant low rank” framing in §2.2 is too strong for TabPFN-L11; the L2 anti-correlation in TabPFN is a notable input-projection artefact. 

|Layer|_r_|T<br>_p_|abPFN-cls<br>mean_C≤_3|mean_C≥_4|_r_|_p_|TabICL-cls<br>mean_C≤_3|mean_C≥_4|
|---|---|---|---|---|---|---|---|---|
|L0|_−_0.29|0.079|55.6|44.3|+0.50|0.000|31.6|49.2|
|L1|+0.50|0.000|34.9|53.2|+0.51|0.000|38.3|59.5|
|L2|_−_0.36|0.012|10.9|9.1|+0.52|0.000|42.8|65.5|
|L3|_−_0.26|0.072|20.8|18.6|+0.50|0.000|50.2|73.4|
|L4|+0.33|0.023|21.0|24.9|+0.48|0.000|54.1|77.7|
|L5|+0.40|0.004|20.1|24.2|+0.48|0.001|65.2|92.1|
|L6|+0.48|0.001|21.6|27.7|+0.47|0.001|69.4|97.4|
|L7|+0.44|0.002|18.7|23.5|+0.46|0.001|77.6|109.0|
|L8|+0.33|0.020|15.3|17.5|+0.46|0.001|67.0|94.5|
|L9|+0.46|0.001|20.5|26.5|+0.42|0.002|68.5|94.5|
|L10|+0.51|0.000|22.3|30.1|+0.36|0.012|61.1|79.7|
|L11|+**0**_._**64**|0.000|21.1|31.7|+0.21|0.152|63.0|73.3|



### **I.18 Faithful-surrogate coverage by regime** 

The four criteria of §3 are: **(i) Pearson** _r≥_ 0 _._ 85 between the rule’s predicted probabilities and the model’s, evaluated on the held-out test split of each dataset and aggregated as the 49-dataset mean. This tests whether the rule reproduces the model’s _soft_ decision shape, not just the argmax. **(ii) accuracy gap** _≤_ 3 **pp** between the rule’s accuracy and native model accuracy on the same split, averaged across the 49-dataset suite. This tests whether the rule reproduces the model’s _performance level_ , ruling out rules that match distributions on average but lose on hard examples. **(iii) Cohen’s** _κ≥_ 0 _._ 8 between the rule’s argmax and the model’s argmax on the held-out test split, computed per dataset and aggregated as the mean across the 49-dataset grid. This is a strict _point-wise_ agreement test that penalizes chance-level agreement and rules out a candidate that reaches the right accuracy by hitting different examples than the model does. **(iv) intervention-survival** : the rule must continue to match the model after a causal intervention that preserves only the components it names (e.g. uniformising attention outside L9 for the vote rule, replacing all final-layer features with class means for the prototype rule). This rules out rules that happen to track the model in the unperturbed network but rely on machinery they do not explicitly name. 

Table 91 summarizes which readout passes each criterion, broken out by model and regime. Failing criteria are flagged at point of use in the main text rather than collapsed into a universal claim. 

54 

Table 91: Where each readout passes the four faithful-surrogate criteria (§3). ✓ pass at the stated cut-off, _∼_ borderline, _×_ fail (failing statistic in parentheses); _n/a_ where the criterion does not apply (e.g. _κ_ on regression). The intervention column gives the observed causal signature. 

|Model / regime|Rule|(i)_r≥_0_._85|(ii)_≤_3pp|(iii)_κ≥_0_._8|(iv) intervention|
|---|---|---|---|---|---|
|TabPFNv2 cls (binary–mid)|vote@L9|✓(0_._93)|✓(_−_0_._02)|✓|✓uniform-attn_→_maj.|
|TabPFNv2 cls (10-class)|vote@L9|_×_(0_._71–0_._76)|_∼_|_∼_|✓|
|TabPFNv2 reg|vote (reg)|✓|✓|n/a (reg)|✓|
|TabICL v1/v1.1/v2 cls|proto@L11|✓|✓(_−_0_._01)|✓|✓cross-rule collapse|
|TabICLv2 reg|lin. proto|_×_|_×_|n/a (reg)|✓(_ℓ_2-_k_NN tracks)|
|Mitra cls|vote@L9|✓(0_._89)|_∼_(_−_4_._5)|✓|✓row-attn KO|



**Threshold sensitivity.** We stress the descriptive thresholds in §3 by varying the soft-fidelity (i) and accuracy (ii) cut-offs and recounting populated-rule passes (Table 92); the _κ_ and intervention criteria are binary and not swept. Tightening or loosening either threshold shifts populated-rule counts by a few datasets, and the readout-family verdict (§3.4) does not flip in any of these cells. 

Table 92: Faithful-surrogate pass counts as a function of thresholds ( _r,_ ∆acc). Cells are _populatedrule_ passes (criterion (i) and (ii) only; intervention always passes for these rules). TabPFNv2 vote counts use the 49-dataset population of Table 17 (per-dataset Pearson _r_ has mean 0 _._ 890, sd 0 _._ 114); TabICLv2 prototype counts use the 49-dataset population of §C.3.1. 

||TabP|FNv2 vote|(L9)|<br>TabICLv|2 prototy|pe (L11)|
|---|---|---|---|---|---|---|
|Threshold_r /_|∆<br>_≤_5pp|_≤_3pp|_≤_1|pp<br>_≤_5pp|_≤_3pp|_≤_1pp|
|_r≥_0_._80|43_/_49|42_/_49|38_/_|49<br>46_/_49|45_/_49|41_/_49|
|_r≥_0_._85|41_/_49|40_/_49|36_/_|49<br>44_/_49|43_/_49|40_/_49|
|_r≥_0_._90|37_/_49|36_/_49|32_/_|49<br>41_/_49|41_/_49|38_/_49|



### **I.19 Global BH-FDR summary across headline rejections** 

This subsection delivers a joint Benjamini–Hochberg FDR check at _q_ =0 _._ 05 across the 30 headline rejections of §2–§6, alongside the within-family multiplicity controls already used in the body (Holm across the eight attacks per model in §6; Bonferroni on the 10 regression tests; TOST equivalence margins pre-specified for the invariance claim). The list comprises: the L8 _→_ L9 probe-accuracy jump, the per-block knockout _p_ -values for blocks _{_ 0 _,_ 5 _,_ 9 _}_ in TabPFNv2 and _{_ ColEmb-2, ICL-0, ICL-6, ICL-11 _}_ in TabICLv2 (7 tests), the cross-rule sign tests (PFN _→_ ICL, ICL _→_ PFN; 2), the fresh-head asymmetry on each backbone (2), the per-version prototype/vote ordering tests across v1, v1.1, v2 (3), the _W_ =0 and no-RoPE TOST equivalences (2), and the 16 per-attack ∆ tests (eight attacks _×_ two FMs). 

All 30 rejections survive BH-FDR at _q_ =0 _._ 05 (largest _p<_ 10<sup>_−_3</sup> in the list passing BH; the marginal cases are the TabICLv2 `cube_warp` and `centroid_inj.` entries which are reported as _non_ -rejections in Table 3 and are not in the headline list). The within-family Holm decisions in Table 3 are therefore consistent with the global BH picture; no additional headline finding flips sign or loses significance under joint control. We note that BH-FDR on 30 tests with this many small _p_ -values is a weak sieve in practice; the more informative discipline remains the local Holm/Bonferroni choices already in the body. 

### **I.20 Serving-cost ledger** 

Per-query latency and peak GPU memory are reported in Table 5; the corresponding throughput on a single H100-NVL is _≈_ 1 query/s for TabPFNv2 and _≈_ 0 _._ 7 query/s for TabICLv2 when each query is served as its own forward pass. Both backbones allow batched query forwards over a fixed context: amortizing 32 queries per context pass on H100 raises throughput to _≈_ 18 queries/s for TabPFNv2 and _≈_ 12 queries/s for TabICLv2 at peak memory _≈_ 6 GiB (TabPFNv2) and _≈_ 8 GiB (TabICLv2); above this batch the attention-cost term dominates and throughput plateaus. The OvA class-invariance wrapper of §4 multiplies wall-clock by _C_ and peak memory by _C_ when the _C_ binary re-encodings are run independently; on the _C_ =10 `mfeat-*` tasks this is _≈_ 10 s and _≈_ 30 GiB per query batch. Memory rather than wall-clock is the limiting factor; on a single 24 GiB consumer 

55 

card the OvA wrapper does not fit at _C≥_ 8 without query-batch reduction. The _W_ =0 and no-RoPE patches change neither wall-clock nor peak memory at any batch size we measured (§4). 

## **J Robustness to context size** 

Across the context sizes we evaluated ( _n ∈{_ 1000 _,_ 1500 _}_ on six OpenML datasets with native size _>_ 1500 rows), the readout and invariance findings of §3 and §4 are unchanged: vote and prototype fidelities and column-permutation spread are statistically indistinguishable across the two context sizes (Table 93). We re-run the attention-weighted L9 vote for TabPFNv2, the nearest-prototype rule in the final representation for TabICLv2, and the column-permutation invariance probe at both context sizes. 

**Setup.** Six OpenML datasets have native size above 1500 rows: `car` (1728), `Is-this-a-good-customer` (1723), `steel-plates-fault` (1941), `mfeat-morphological` (2000), `mfeat-fourier` (2000), `red_wine` (1599). For each (dataset, _n_ train, seed) cell we hold out 200 stratified test rows and subsample the remaining pool to _n_ train without replacement; `red_wine` caps at _n_ = 1399. Each cell is run with five seeds. Vote and prototype fidelities are top-1 agreement with the native model prediction. Column-permutation spread is the max _−_ min accuracy across eight independently sampled random column orderings. 

Table 93: Scale audit on six OpenML datasets with native size _>_ 1500 rows. Vote fidelity (TabPFNv2) and prototype fidelity (TabICLv2) are top-1 agreement with the native model prediction (mean _±_ std over 5 seeds). Col-perm spread is max _−_ min accuracy over 8 random column permutations. 

|Model|Dataset|_n_train|Fidelity (mean_±_std)|Col-perm spread|
|---|---|---|---|---|
|TabPFNv2|`Is-this-a-good-customer`|1000|0_._996_±_0_._004|0_._000|
|TabPFNv2|`Is-this-a-good-customer`|1500|0_._986_±_0_._009|0_._000|
|TabPFNv2|`car`|1000|0_._953_±_0_._018|0_._012|
|TabPFNv2|`car`|1500|0_._970_±_0_._016|0_._011|
|TabPFNv2|`mfeat-fourier`|1000|0_._789_±_0_._048|0_._022|
|TabPFNv2|`mfeat-fourier`|1500|0_._788_±_0_._023|0_._022|
|TabPFNv2|`mfeat-morphological`|1000|0_._753_±_0_._048|0_._020|
|TabPFNv2|`mfeat-morphological`|1500|0_._816_±_0_._044|0_._011|
|TabPFNv2|`red_wine`|1000|0_._803_±_0_._030|0_._025|
|TabPFNv2|`red_wine`|1399|0_._798_±_0_._027|0_._027|
|TabPFNv2|`steel-plates-fault`|1000|0_._728_±_0_._038|0_._022|
|TabPFNv2|`steel-plates-fault`|1500|0_._745_±_0_._037|0_._022|
|TabICLv2|`Is-this-a-good-customer`|1000|0_._963_±_0_._019|0_._000|
|TabICLv2|`Is-this-a-good-customer`|1500|0_._947_±_0_._044|0_._003|
|TabICLv2|`car`|1000|0_._989_±_0_._004|0_._006|
|TabICLv2|`car`|1500|0_._996_±_0_._004|0_._004|
|TabICLv2|`mfeat-fourier`|1000|0_._946_±_0_._012|0_._010|
|TabICLv2|`mfeat-fourier`|1500|0_._958_±_0_._012|0_._014|
|TabICLv2|`mfeat-morphological`|1000|0_._869_±_0_._032|0_._007|
|TabICLv2|`mfeat-morphological`|1500|0_._862_±_0_._039|0_._009|
|TabICLv2|`red_wine`|1000|0_._836_±_0_._043|0_._012|
|TabICLv2|`red_wine`|1399|0_._858_±_0_._020|0_._011|
|TabICLv2|`steel-plates-fault`|1000|0_._932_±_0_._024|0_._023|
|TabICLv2|`steel-plates-fault`|1500|0_._923_±_0_._020|0_._020|
|_Marginals o_|_ver (model, n):_||||
|TabPFNv2|all six datasets|1000|0_._837_±_0_._107|0_._017|
|TabPFNv2|all six datasets|1500|0_._860_±_0_._103|0_._015|
|TabICLv2|all six datasets|1000|0_._923_±_0_._059|0_._010|
|TabICLv2|all six datasets|1500|0_._924_±_0_._054|0_._010|



Vote and prototype fidelities are stable across _n_ , and the column-permutation spread does not grow with _n_ . The readout characterizations of §3 and the invariance results of §4 extend to the larger context size. 

56 

## **K Probe** _ℓ_ 2 **-regularization sensitivity (C3)** 

The main-text per-layer linear probe in §2.1 is reported at _ℓ_ 2 regularization _C_ =1. We check whether the qualitative findings — in particular the sharp L8 _→_ L9 crystallization in TabPFNv2 — depend on this choice. 

We re-fit the per-layer logistic probe with _C ∈{_ 0 _._ 01 _,_ 0 _._ 1 _,_ 1 _,_ 10 _,_ 100 _}_ on the 45-dataset suite, with 5 seeds per (dataset, layer, _C_ ) cell. Frozen activations are identical across _C_ values, so the only varying quantity is the _ℓ_ 2 penalty applied during the probe fit. Aggregate per-layer accuracies are in Table 94. 

**Aggregate result.** Across the 45 _×_ 13 _×_ 5 _×_ 5 = 14 _,_ 625 (dataset, layer, seed, _C_ ) cells, the mean per-layer accuracy at _C_ =1 is within 0 _._ 012 of the mean accuracy at _C ∈{_ 0 _._ 1 _,_ 10 _}_ on every layer, and the L8 _→_ L9 jump is preserved at every _C ∈{_ 0 _._ 01 _,_ 0 _._ 1 _,_ 1 _,_ 10 _,_ 100 _}_ (jump magnitude 0 _._ 291–0 _._ 321 across _C_ , with the smallest gap at _C_ =1). Even the most strongly regularized _C_ =0 _._ 01 probe recovers the same step (input 0 _._ 44, L8 0 _._ 48, L9 0 _._ 80), so the late-stack crystallization is a property of the representation rather than of the probe fit. We therefore report _C_ =1 in the main text without loss of generality. 

Table 94: C3: per-layer linear-probe accuracy on TabPFNv2, averaged over 45 datasets and 5 seeds, for each _ℓ_ 2 regularization strength _C_ . 

|layer|_C_=0_._01|_C_=0_._1|_C_=1_._0|_C_=10_._0|_C_=100_._0|
|---|---|---|---|---|---|
|input|0.437|0.396|0.388|0.396|0.397|
|L0|0.326|0.304|0.296|0.296|0.295|
|L1|0.340|0.322|0.310|0.310|0.310|
|L2|0.306|0.275|0.275|0.277|0.277|
|L3|0.490|0.448|0.448|0.447|0.447|
|L4|0.539|0.531|0.527|0.518|0.500|
|L5|0.458|0.450|0.446|0.446|0.448|
|L6|0.492|0.455|0.455|0.450|0.450|
|L7|0.484|0.476|0.465|0.455|0.453|
|L8|0.483|0.481|0.480|0.477|0.460|
|L9|0.804|0.778|0.771|0.773|0.772|
|L10|0.802|0.795|0.790|0.788|0.784|
|L11|0.621|0.615|0.623|0.623|0.609|



## **L Head-capacity saturation curve on TabPFNv2 L12 (C2)** 

We measure the held-out test accuracy of fresh-fit heads attached to the frozen TabPFNv2 stack at the top-of-stack representation (L12, target column), as the head capacity is varied. Heads: 

- **Linear** — _ℓ_ 2-regularized logistic regression. 

- **MLP-1** — one hidden layer (64 units, GELU). 

- **MLP-2** — two hidden layers (128 _,_ 64). 

- **MLP-3** — three hidden layers (256 _,_ 128 _,_ 64). 

- **MLP-4** — four hidden layers (512 _,_ 256 _,_ 128 _,_ 64). 

- **Kernel SVM** — RBF SVM on a 32-dim PCA projection. 

We report mean test accuracy across one context-shuffling seed on a 10-dataset subset of the suite. The protocol mirrors `h6_probe_ladder` from `tab_inv.hardening_experiments` . Per-dataset numbers are in Table 95. 

**Result.** The linear head attains the highest mean accuracy (0 _._ 719); MLP capacity does not help (MLP-1 0 _._ 695, MLP-2 0 _._ 591, MLP-3 0 _._ 601, MLP-4 0 _._ 587), and the RBF kernel SVM is the worst fresh-head (0 _._ 406). The non-monotonicity across MLP depth is consistent with overfitting on a representation in which the relevant class signal is already linearly readable; the kernel-SVM gap is consistent with a representation whose useful directions are not preserved under PCA-32 + RBF. In 

57 

other words, the L12 representation of TabPFNv2 does not contain additional non-linearly extractable class structure beyond what a linear head recovers. 

Table 95: C2: held-out test accuracy of fresh heads at TabPFNv2 L12 (top-of-stack) across head capacities, on 10 datasets (1 seed). 

|dataset|chance|Linear|MLP-1|MLP-2|MLP-3|MLP-4|Kernel SVM|
|---|---|---|---|---|---|---|---|
|LED-display-domain-7di|0.10|0.250|0.160|0.110|0.180|0.150|0.110|
|balance-scale|0.33|0.664|0.768|0.728|0.640|0.368|0.464|
|banknote-authenticatio|0.50|0.996|1.000|0.924|0.713|1.000|0.556|
|blood-transfusion-serv|0.50|0.760|0.760|0.760|0.240|0.240|0.760|
|breast_cancer|0.50|0.965|0.956|0.368|0.965|0.965|0.368|
|dresses-sales|0.50|0.580|0.580|0.580|0.460|0.580|0.420|
|iris|0.33|0.667|0.667|0.667|0.967|0.667|0.333|
|mfeat-factors|0.10|0.670|0.757|0.287|0.530|0.515|0.205|
|mfeat-fourier|0.10|0.690|0.665|0.655|0.730|0.662|0.453|
|wine|0.33|0.944|0.639|0.833|0.583|0.722|0.389|
|mean|0.33|0.719|0.695|0.591|0.601|0.587|0.406|



## **M Native-categorical GBDT under the eight perturbations (C6)** 

The main-text attack tables (§H.3, §H.4, §H.2) compare against ordinal-encoded GBDT/XGBoost baselines; here we additionally report CATBOOST, XGBOOST, and LIGHTGBM with native categorical handling on the mixed-type subset of the perturbation grid (16 OpenML datasets with at least one categorical column). 

The eight perturbations are the same as in the main text: pad-2x, pad-4x, boundary, hub-poison, centroid-injection, mono-cube, mono-softexp, and mono-rank. 

Table 96: C6: mean test accuracy across the mixed-type OpenML subset ( _N_ =16 datasets) for native-categorical GBDTs under the eight perturbations. CatBoost’s `cat_features` interface rejects float-typed arrays produced by the non-monotone perturbations, so its row only contains values for the three numeric mono-* perturbations. 

|backend|orig|pad-2x|pad-4x|bound|hub|centr|m-cube|m-sexp|m-rank|
|---|---|---|---|---|---|---|---|---|---|
|CatBoost|–|–|–|–|–|–|0.774|0.760|0.662|
|XGBoost|0.759|0.742|0.742|0.741|0.710|0.759|0.762|0.762|0.623|
|LightGBM|0.761|0.748|0.742|0.749|0.709|0.763|0.763|0.756|0.654|



**Result.** The qualitative pattern matches the main-text ordinal-encoded baseline: the smooth monotone-feature attacks (MONO-CUBE, MONO-SOFTEXP) and the geometric perturbations (pad-2x, pad-4x, centroid-injection) leave native-cat GBDTs essentially at _orig_ accuracy, while HUB-POISON ( _−_ 0 _._ 05 for both backends able to score it) and MONO-RANK ( _−_ 0 _._ 14 for XGBoost, _−_ 0 _._ 11 for LightGBM) cause the largest drops. CATBOOST rejects pure-numeric arrays under `cat_features` , so its row only contains values for the three purely numeric mono-* perturbations; for those, CATBOOST behaves comparably to the other two backends. 

58 

