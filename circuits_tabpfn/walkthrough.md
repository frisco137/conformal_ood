# TabPFN Mechanistic Interpretability Walkthrough

This document outlines the mechanistic interpretability experiments conducted to understand how TabPFN performs in-context classification on prior-conforming data (In-Distribution / ID) and how it detects and reacts to data that violates the prior (Out-Of-Distribution / OOD). 

## Experiment 1: Attention Divergence & Component Ablation

### Theory and Motivation

The goal of Experiment 1 is to locate the internal components responsible for core classification logic and those that detect out-of-distribution shifts. Transformer models distribute their reasoning across multiple layers, attention heads, and MLPs. To isolate these mechanisms, the experiment relies on two complementary analyses:

1.  **Part A: Attention Divergence (Jensen-Shannon Divergence):**
    When a model processes OOD data, parts of its internal mechanism must "notice" the anomaly and change behavior. The theory here is that specific attention heads are responsible for recognizing shift. By calculating the Jensen-Shannon Divergence (JSD) between the average attention maps on ID data versus OOD data, we can spot which heads alter their routing of information. Because TabPFN evaluates attention twice in each layer — once for context attending to context (`ctoc`) and once for query attending to context (`qtoc`) — JSD is evaluated separately for both calls.
    *   **Expected Outcome:** A sparse subset of heads will show high JSD, flagging them as strong OOD respondents. The majority will show near-zero JSD.

2.  **Part B: Zero-Ablation Study (Impact on Prediction Quality):**
    To analyze the functional importance of each component for ID data, we iteratively zero out (ablate) the output of each individual attention head and MLP layer. We then measure the drop in performance, calculated as the increase in Cross-Entropy ($\Delta$CE).
    *   **Expected Outcome:** A few critical MLPs and heads ("workhorses") will cause huge spikes in $\Delta$CE when ablated, indicating they are vital for standard ID predictions. Meanwhile, eliminating non-essential or pure "OOD-detector" heads should barely impact the ID performance.

---

### Analysis of the Experiment Outputs

#### Output 1: Impact of Component Ablation (Ranked Bar Chart)

![Impact of Component Ablation](circuits_tabpfn/results/ablation_delta_ce_bar.png)

*   **Explanation & Expectation Match:** The bar chart plots the $\Delta$CE for all 48 heads and 12 MLPs in descending order. As expected, power is heavily concentrated. `L0_MLP` handles massive foundational processing (highest $\Delta$CE > 2.5), followed by a few specific heads (`L7H1`, `L1H2`, `L0H1`) and earlier MLPs. The long tail trailing off shows that most components are functionally redundant or unnecessary for ID predictions.
*   **Next Steps for Circuit Discovery:** To find the precise circuit, we should closely investigate `L0_MLP` and the top attention heads like `L7H1` and `L1H2`. For the MLPs we could use sparse autoencoders or analyze their projections to see what raw input features are being encoded early on. For `L7H1`, we can perform causal interventions (like Activation Patching) to see what exact information it builds and passes forward. 

#### Output 2: Head Ablation Impact Heatmap 

![Head Ablation Impact Map](circuits_tabpfn/results/ablation_ce_heatmap.png)

*   **Explanation & Expectation Match:** This heatmap displays the $\Delta$CE for all 48 attention heads mapped by layer and head index. Similar to the bar chart, we can clearly see the sparseness. Layer 1 Head 2 (1.401 $\Delta$CE) and Layer 7 Head 1 (1.698 $\Delta$CE) are absolute bottlenecks for performance. Most late-layer heads (Layers 8-11) show nearly 0 $\Delta$CE impact. This validates the expectation that early-to-mid heads do the heavy classification lifting.
*   **Next Steps for Circuit Discovery:** We should analyze the exact attention patterns of `L7H1` and `L1H2` on ID data. Do they attend to specific matching labels in the context? Because they are so important, we can trace what keys and queries consistently activate them.

#### Output 3: Attention JSD (context$\rightarrow$context)

![JSD ctoc](circuits_tabpfn/results/attn_jsd_heatmap_ctoc.png)

*   **Explanation & Expectation Match:** This heatmap reveals the heads that shift their context-to-context information routing when encountering OOD data. The signal is hyper-localized: `L10H1` shows extreme divergence (JSD of 0.046), and `L7H0` also shows a distinct reaction (0.023). This fits the expected outcome smoothly; OOD detection is heavily specialized in late-layer heads rather than broadly distributed.
*   **Next Steps for Circuit Discovery:** `L10H1` acts as a shift sensor. We can ablate `L10H1` entirely on OOD data. If the model incorrectly tries to make confident predictions on OOD data instead of behaving conservatively, we know `L10H1` essentially functions as a regularization trigger or a "caution" signal.

#### Output 4: Attention JSD (query$\rightarrow$context)

![JSD qtoc](circuits_tabpfn/results/attn_jsd_heatmap_qtoc.png)

*   **Explanation & Expectation Match:** The query-to-context attention maps how inference handles pulling answers from the context. We again see strong divergence in `L7H0`, `L0H1`, `L11H2`, and `L7H3`. When processing OOD queries, these heads change where they look for contextual evidence. Again, this fits our expectation of finding specialized shift-respondents.
*   **Next Steps for Circuit Discovery:** For these heads (especially `L7H0` and `L0H1`), we should visualize their exact raw attention weights side-by-side (ID vs OOD). They very likely stop attending to class-matching context points and instead disperse their attention uniformly when the data is messy/OOD. 

#### Output 5: Cross-Reference (JSD vs Ablation) 

![Cross-Reference](circuits_tabpfn/results/cross_reference_jsd_vs_ablation.png)

*   **Explanation & Expectation Match:** These scatter plots correlate a component's importance to ID tasks (Ablation $\Delta$CE) against its sensitivity to OOD shift (Attention JSD). The correlation is remarkably low (Pearson r ~0.1). High-$\Delta$CE heads like `L7H1` fall on the far left (low JSD), while high-JSD heads like `L10H1` fall on the bottom (near zero $\Delta$CE). As theorized, this demonstrates a clean "division of labor" — the mechanism that structurally solves the task is decoupled from the mechanism that notices an out-of-prior shift. 
*   **Next Steps for Circuit Discovery:** We can isolate these two disjoint circuits for Experiment 2 (Activation Patching). For the workhorse circuit (high $\Delta$CE, low JSD), we execute path-patching to trace the core flow of prediction. For the OOD circuit (high JSD, low $\Delta$CE), we can do reverse-patching, injecting OOD activations of these heads into an ID context to see if we can trigger "artificial hesitation" or uncertainty in an otherwise confident ID prediction.

## Experiment 2: Causal Tracing via Activation Patching

### Theory and Motivation

While Experiment 1 identified *which* components are important through zero-ablation, zeroing a component is a destructive sledgehammer that takes the model out of its normal operating distribution. **Activation Patching** (or causal tracing) is a much sharper scalpel. 

The theory relies on generating two parallel runs:
1.  **Clean Run:** The model operates normally on In-Distribution (ID) data, achieving low Cross-Entropy (good performance).
2.  **Corrupted Run:** We intentionally break the model's ability to solve the task.

By actively transplanting (patching) specific hidden activations from the Clean Run into the Corrupted Run (**Forward Patching**), we causally prove that the patched component contains the precise "missing information" needed to restore accuracy. The metric here is **Recovery Fraction** ($1.0$ means the model regained full clean accuracy).

Conversely, patching corrupted activations into the clean run (**Reverse Patching**) determines if degrading a component collapses performance, measured as **Degradation Fraction**. 

We analyze two major corruption subexperiments:
*   `shuffle_y`: The model receives correct features ($X$), but the context labels ($y[:p]$) are randomly shuffled. The model is forced to make bad predictions based on bad context. By patching, we track exactly where the model "reads" context labels to infer query labels.
*   `noise`: The model receives ID features corrupted with heavy Gaussian noise. By patching, we track which layers/heads are responsible for recognizing and filtering raw features.

---

### Analysis of the Subexperiment Outputs

#### 1. Context Label Shuffling (`shuffle_y`)

Because TabPFN exclusively relies on the $y$ labels in the in-context subset, scrambling them completely destroys performance. We want to find the exact component that bridges the context labels to the queries.

**Output 6: Forward Recovery and Reverse Degradation Curves (`shuffle_y`)**

![Forward Patching](circuits_tabpfn/results/patching_shuffle_y/forward_recovery_curves.png)

![Reverse Patching](circuits_tabpfn/results/patching_shuffle_y/reverse_degradation_curves.png)

*   **Explanation & Expectation Match:** The line plots show the Recovery/Degradation fraction for layers, attention blocks, and MLPs. As expected, restoring the early `MLP 0` causes massive recovery (over 35x gap closure), implying it does heavy foundational embedding of the inputs. But the absolute standout is the green Attention line severely spiking at **Layer 7** (nearly 38x recovery!). The Reverse Degradation mirrors this: corrupting Layer 7's Attention plunges clean performance by roughly -32x to -36x.
*   **Next Steps:** Layer 7's Attention block is crucial, but it contains 4 heads. We need to drill down into the per-head heatmaps to identify the specific "Hero Head" doing the label-routing.

**Output 7: Per-Head Forward Recovery and Reverse Degradation (`shuffle_y`)**

![Head Forward](circuits_tabpfn/results/patching_shuffle_y/head_recovery_heatmap.png)

![Head Reverse](circuits_tabpfn/results/patching_shuffle_y/head_degradation_heatmap.png)

*   **Explanation & Expectation Match:** Looking at the granular head heatmaps beautifully validates the ablation findings from Experiment 1. **Layer 7 Head 1 (L7H1)** yields a stunning **47.32 Recovery Fraction** and a **-39.17 Degradation Fraction**. 
    This means if we take a TabPFN that is completely failing because of shuffled context labels, and we magically restore *just the activation of L7H1*, the model suddenly makes perfect predictions. `L7H1` is definitively the "Label Fetcher" or "In-Context Router". It reads the labels from the context data and passes them to the queries.
*   **Next Steps for Circuit Discovery:** Since `L7H1` is the undisputed Label Fetcher, next we should do *Path Patching* backward from `L7H1` to find out *how* it decides which context labels to fetch. Does it match raw features calculated by earlier layers? 

---

#### 2. Feature Noise Degradation (`noise`)

In this ablation, the context labels are correct, but the feature data $X$ is overwhelmed with noise. The model's baseline performance drops.

**Output 8: Forward Recovery and Reverse Degradation Curves (`noise`)**

![Noise Forward](circuits_tabpfn/results/patching_noise/forward_recovery_curves.png)

![Noise Reverse](circuits_tabpfn/results/patching_noise/reverse_degradation_curves.png)

*   **Explanation & Expectation Match:** Unlike the `shuffle_y` experiment where a single head recovered 40x the performance gap, fixing the `noise` corruption is much harder. The plots show maximum recovery fractions of only ~0.6 (60% recovery) located in late MLPs (e.g., L10) and scattered early Attention blocks. The reverse degradation is similarly spread out (fraction peaking at 0.45 in L7). This confirms our expectation that robustness against feature noise is not a localized "trick" performed by a single head, but a distributed embedding-cleanup effort processed iteratively across the network.
*   **Next Steps for Circuit Discovery:** Because noise robustness is distributed, component-wise patching yields diminishing returns here. A better next step would be analyzing the residual stream directly (e.g., using SVD or PCA) to track how the signal-to-noise ratio in the feature representations improves layer-by-layer.

---

#### 3. Out-of-Distribution Shift (`out_prior`)

In this experiment, the Corrupted run is the model processing independent Out-Of-Distribution (OOD/anti-prior) data. Unlike previous corruptions, the underlying structure and labels of the ID and OOD splits are unaligned independent draws. 

**Output 9: Forward Recovery and Reverse Degradation Curves (`out_prior`)**

![Out-Prior Forward](circuits_tabpfn/results/patching_out_prior/forward_recovery_curves.png)

![Out-Prior Reverse](circuits_tabpfn/results/patching_out_prior/reverse_degradation_curves.png)

*   **Explanation & Expectation Match:** The Forward Recovery fractions (e.g., substituting entire MLPs or Layers) plunge drastically negative (the line charts show recovery well below zero). This is physically expected: overwriting valid OOD features with ID features meant for completely different independent labels destroys whatever tracking the OOD inference had. The more meaningful metric is the *Reverse Degradation*: injecting OOD activations into a clean ID run to see if we can trigger catastrophic "OOD-like" hesitant behavior. The reverse curves show massive degradation (up to 2.5x the baseline gap) when substituting entire layers, but we need to look at the heads to see what fails exactly.

**Output 10: Per-Head Reverse Degradation Heatmap (`out_prior`)**

![Out-Prior Head Reverse](circuits_tabpfn/results/patching_out_prior/head_degradation_heatmap.png)

*   **Explanation & Expectation Match:** The heatmap reveals deep red zones—heads which, when replaced with OOD processing, collapse the ID predictions. The strongest offenders are `L8H2` (0.65), `L6H2` (0.62), and `L7H1` (0.57). 
    We already know from the `shuffle_y` experiment that `L7H1` is a powerful routing head (the Label Fetcher). Because the OOD and ID datasets are totally unaligned, patching `L7H1` simply transplants "wrong labels" meant for the OOD drawing into the queries for the ID drawing. Hence, prediction fails because the route carries incorrect data, rather than because an "OOD alarm" fired.
*   **Next Steps for Circuit Discovery:** Patching causally between unaligned independent datasets reveals "what routing breaks" rather than "what triggers an OOD response". For the true OOD-sensor heads found in Experiment 1 (like the `L10H1` shift detector), we see only minor targeted degradation (~0.05). To prove `L10H1` forces the model into high-entropy (low confidence) predictions, we must inspect the projection matrix to see its directional pull on logit entropy rather than relying on direct independent-patching.