# Supervised learning pays attention 

### Erin Craig<sup>∗</sup> Robert Tibshirani<sup>†</sup> 

December 11, 2025 

###### **Abstract** 

In-context learning with attention enables large neural networks to make context-specific predictions by selectively focusing on relevant examples. Here, we adapt this idea to supervised learning procedures such as lasso regression and gradient boosting, for _tabular data_ . Our goals are to (1) flexibly fit _personalized_ models for each prediction point and (2) retain model simplicity and interpretability. 

Our method fits a local model for each test observation by weighting the training data according to _attention_ , a supervised similarity measure that emphasizes features and interactions that are predictive of the outcome. Attention weighting allows the method to adapt to heterogeneous data in a data-driven way, without requiring cluster or similarity pre-specification. Further, our approach is uniquely interpretable: for _each test observation_ , we identify which features are most predictive and which training observations are most relevant. We then show how to use attention weighting for time series and spatial data, and we present a method for adapting pretrained tree-based models to distributional shift using attention-weighted residual corrections. Across real and simulated datasets, attention weighting improves predictive performance while preserving interpretability, and theory shows that attention-weighting linear models attain lower mean squared error than the standard linear model under mixture-of-models data-generating processes with known subgroup structure. 

## **1 Introduction** 

Consider a dataset consisting of features **_x_** 1 and **_x_** 2 and response **_y_** . Our goal is to predict **_y_** given **_x_** 1 and **_x_** 2. Further suppose that, unknown to us, the data are heterogeneous: observations fall into two subgroups with different covariate-response relationships, as illustrated in Figure 1. 



Figure 1: _Toy example of a dataset with two features, x_ 1 _and x_ 2 _, where the true model depends on the values of the features._ 

> ∗Department of Biostatistics, University of Michigan; ercr@umich.edu 

> †Departments of Biomedical Data Science and Statistics, Stanford University; tibs@stanford.edu 

1 

Our setup here is a common statistical framework: our data is composed of a feature matrix **_X_** and a vector response **_y_** . This has become known as _tabular data_ in the machine learning community, in contrast to sequence data (e.g. text) that is the focus of large language models and has led to the development of attention and in-context learning. 

If we use a standard linear model (fit by say least squares or lasso), we will fit a single set of coefficients to all observations, thereby averaging effects across the two groups. If we suspected heterogeneity in our data, we might cluster the data and then fit separate models to each resulting group. This might be successful if the clusters are well separated and we choose the correct number of clusters. Another option is to fit a more complicated model like gradient boosting or a neural network, which adapt to data heterogeneity but sacrifices interpretability. 

In this paper we propose a method to fit personalized, locally weighted models for each test point, anchored by a global model fit to the full training set. At a high level, for a test observation **_x_**<sup>_∗_</sup> and training data ( **_X_** _,_ **_y_** ), our method (1) derives a set of observation weights **_w_**<sup>_∗_</sup> ( _attention weights_ ) that describe similarity between **_x_**<sup>_∗_</sup> and the rows of **_X_** as they relate to the response **_y_** ; and (2) uses them to fit a “bespoke” weighted model to make a prediction at **_x_**<sup>_∗_</sup> . 

Importantly, we define similarity in a _supervised_ way: two observations are considered similar only if they share features and interactions useful for predicting **_y_** , which underlies the improvement in predictive performance of our method. This supervised similarity distinguishes our approach from local and kernel regression [Cleveland and Devlin, 1988], and is inspired by the success of row-wise attention in foundation models for tabular data, particularly TabPFN [Hollmann et al., 2025]. 

The idea of _supervised attention weights_ is quite general, and we show how to apply it to standard supervised learning models for tabular data, time series, spatial data, and longitudinal data subject to drift. Our intuition is that “one-size-fits-all” models do not always provide the best prediction for each data point. Rather, data can be heterogeneous in ways that are difficult to enumerate or anticipate<sup>1</sup> . Attention weighting naturally accommodates a potentially continuous spectrum of latent subgroups in the data. By assigning attention weights rather than discrete labels, it adapts to both hard clustering (distinct groups) and soft clustering (overlapping or blended subgroup memberships). 

The contributions of this work are as follows: 

1. We present a simple overview of attention and in-context learning (Section 4), and its connection to local linear regression and kernel methods (Section 5). 

2. We develop a method to use attention to fit interpretable, personalized models for tabular data (Sections 2 and 7) whose predictive performance typically matches or exceeds that of its nonattention counterparts. 

3. We extend our method to spatial and time-series data (Section 6). 

4. We introduce a novel idea for model interpretability (Section 3.1). Rather than identify a single set of predictive features for a dataset, we identify _for each test point_ which features are most useful as well as which training points are most relevant. 

5. We present a method that uses attention to adapt pretrained tree-based models at prediction time without refitting, for settings where the data distribution may drift between training and deployment and refitting the model is expensive or otherwise prohibitive (Section 8). 

We hope that contribution (1) will benefit those seeking to understand and apply the powerful AI/ML idea of attention to statistical and machine learning for tabular data, and also reveal interesting connections to local regression and kernel methods. Contributions (2-5) should be useful for researchers interested in prediction and characterizing heterogeneity. 

The rest of this paper is as follows. Section 2 shows our method in detail as applied to lasso linear regression ( _attention lasso_ ), and is followed by real and simulated examples in Section 3. Section 4 explains attention, self-attention and in-context learning as they relate to our work, and Section 5 discusses other related work, including kernel regression and local linear regression. We extend our method to spatial and time series data in Section 6, to other base learners in Section 7, and to longitudinal data in Section 8. We close with a discussion in Section 9. Appendix A compares lasso 

> 1In his popular children’s show, Mr. Rogers says “You are the only one like you.” We extend this idea from ourselves to our data: each data point is unique and may be best fit by its own model. 

2 

and attention lasso in a mixture-of-models setting and finds that attention lasso reduces both prediction error and bias in the fitted coefficients, and Appendix B shows the relationship between attention and Gaussian kernel regression. 

## **2 Supervised learning with attention for tabular data** 

### **2.1 General procedure** 

We first describe _attention for supervised learning_ as applied to tabular data. While we focus on continuous outcomes, our approach applies to any response type, including binary, survival, and multiclass. The special case of the lasso is described in the next section. 

Figure 2 and Algorithm 1 describe our method in detail. Given training data **_X_** with response **_y_** and prediction point **_x_**<sup>_∗_</sup> , we fit a weighted model where the weights reflect similarity to **_x_**<sup>_∗_</sup> . One natural similarity measure, inspired by neural network attention (Section 4), is 



where **_W_**<sup>**ˆ**</sup> is a diagonal matrix whose entries are the absolute values of coefficients from ridge regression fit to ( **_X_** _,_ **_y_** ). This ensures that two points are considered similar only with respect to features predictive of **_y_** . This formulation is much simpler than even a single linear attention head, which is typically a dense, non-symmetric matrix, and its simplicity makes it amenable to theoretical analysis (Appendix A). However, the diagonal structure of **_W_**<sup>**ˆ**</sup> cannot represent feature interactions. 

As a critical enhancement, we instead use random forest proximity as our similarity measure: we fit a random forest to ( **_X_** _,_ **_y_** ) and define the proximity between two points as the proportion of trees in which they land in the same terminal node. In our experiments, proximity-based weights greatly outperform the ridge-based weights of Equation (1). 

Because we fit a separate model for each test observation, we employ a regularization strategy to control complexity: the final prediction is a convex combination of a _baseline model_ (fit without weights to the full dataset) and an _attention model_ (the weighted model): 



where _m_ is a mixing hyperparameter selected via cross-validation. When _m_ = 0, we recover a single global model; when _m_ = 1, predictions rely entirely on the personalized model. 



Figure 2: _An outline of supervised learning with attention. First, we fit a random forest to estimate similarity between observations; second, a baseline model (for example, lasso or boosting) is fit to the training data. Then for each test observation_ **_x_**<sup>_∗_</sup> _, we estimate attention weights using the random forest similarities between_ **_x_**<sup>_∗_</sup> _and each of the training observations; we use these weights to fit an attentionweighted model specific to_ **_x_**<sup>_∗_</sup> _. Finally, we blend the weighted model with the baseline model to make our prediction at_ **_x_**<sup>_∗_</sup> _. More details are given in Algorithm 1_ 

3 

**Algorithm 1** Supervised Learning with Attention 

**Input:** Train set **_X_** _,_ **_y_** with _n_ observations, test set **_X_**<sup>_∗_</sup> , mixing parameter _m ∈_ [0 _,_ 1]. **<u>Output:</u>** <u>Predictions</u> **_<u>y</u>_ ˆ**<sup>_∗_</sup> <u>for each row of</u> **_<u>X</u>_**<sup>_∗_</sup> <u>.</u> 

1. **Compute attention weights** **_A_**<sup>**ˆ**</sup><sup>_∗_</sup> **:** Fit a random forest to ( **_X_** _,_ **_y_** ). For each observation **_x_**<sup>_∗_</sup> _i_<sup>of</sup> **_X_**<sup>_∗_</sup> , define _n_ similarity scores **_S_**<sup>**ˆ**</sup><sup>_∗_</sup> equal to the proportion of times an observation **_x_**<sup>_∗_</sup> _i_<sup>fallsinto</sup> the same terminal node as each of the rows of **_X_** . 

Define attention weights 



where softmax is applied row-wise to **_S_**<sup>**ˆ**</sup> . The _i_<sup>th</sup> row of the attention matrix **_A_**<sup>**ˆ**</sup><sup>_∗_</sup> corresponds to the attention scores from test observation _i_ to each of the train observations. 

2. **Fit baseline model:** Fit a supervised learning model to **_X_** _,_ **_y_** . Predict for **_X_**<sup>_∗_</sup> to obtain **_y_ ˆ** base<sup>_∗_.</sup> 

3. **Fit attention-weighted models:** For **_x_**<sup>_∗_</sup> _i_<sup>, the</sup><sup>_i_throw of</sup><sup>**_X_**</sup><sup>_∗_, fit a weighted supervised learning</sup> model to **_X_** _,_ **_y_** using weights **_A_**<sup>**ˆ**</sup><sup>_∗_</sup> _i_<sup>.Predictat</sup><sup>**_x_**</sup><sup>_∗_</sup> _i_<sup>toobtain</sup><sup>**_y_ˆ**</sup> _i,_<sup>_∗_</sup> attn<sup>.</sup> 

4. **Combine predictions:** Return the weighted average 



Note: the mixing <u>parameter</u> _m_ is selected through cross-validation. 

### **2.2 The attention lasso** 

In much of this paper, we focus on the application of the above idea to the special case of the lasso, detailed in this section. We explore its application to other models— especially gradient boosting— in Section 7. 

Algorithm 2 shows the details of our proposal when the lasso is used for both the baseline and attention-weighted models. 

**Algorithm 2** Attention Lasso 

**Input:** Train set **_X_** _,_ **_y_** with _n_ observations, test set **_X_**<sup>_∗_</sup> , mixing parameter _m ∈_ [0 _,_ 1]. 

**<u>Output:</u>** <u>Predictions</u> **_<u>y</u>_ ˆ**<sup>_∗_</sup> <u>for</u> **_<u>X</u>_**<sup>_∗_</sup> <u>, for each row of</u> **_<u>X</u>_**<sup>_∗_</sup> <u>.</u> 

1. **Compute attention weights** **_A_**<sup>**ˆ**</sup><sup>_∗_</sup> **:** Fit a random forest to ( **_X_** _,_ **_y_** ). For each observation **_x_**<sup>_∗_</sup> _i_<sup>of</sup> **_X_**<sup>_∗_</sup> , define _n_ similarity scores **_S_**<sup>**ˆ**</sup><sup>_∗_</sup> equal to the proportion of times an observation **_x_**<sup>_∗_</sup> _i_<sup>fallsinto</sup> the same terminal node as each of the rows of **_X_** . 

Define attention weights 



where softmax is applied row-wise to **_S_**<sup>**ˆ**</sup> . The _i_<sup>th</sup> row of the attention matrix **_A_**<sup>**ˆ**</sup><sup>_∗_</sup> corresponds to the attention scores from test observation _i_ to each of the train observations. 

2. **Fit baseline model:** Fit a lasso regression to ( **_X_** _,_ **_y_** ) and select regularization parameter _λ_<sup>ˆ</sup> . Predict for **_X_**<sup>_∗_</sup> to obtain **_y_ ˆ** base<sup>_∗_.</sup> 

3. **Fit attention-weighted lasso models:** For **_x_**<sup>_∗_</sup> _i_<sup>,the</sup><sup>_i_throwof</sup><sup>**_X_**</sup><sup>_∗_,fitaweightedlassore-</sup> gression to **_X_** _,_ **_y_** using weights **_A_**<sup>**ˆ**</sup><sup>_∗_</sup> _i_<sup>andregularizationparameter</sup><sup>_λ_ˆ.</sup> Predict at **_x_**<sup>_∗_</sup> _i_<sup>toobtain</sup> **_y_ ˆ** _i,_<sup>_∗_</sup> attn<sup>.</sup> 

4. **Combine predictions:** Return the weighted average 



To control complexity, all of the attention models share a common lasso regularization parameter 

4 

_λ_ ˆ, selected through cross-validation when fitting the baseline model. 

We interpret fitted models by clustering coefficient vectors using `protoclust` , a hierarchical clustering method where each cluster is represented by a prototypical element Bien and Tibshirani [2011, 2022]. This groups test data according to their fitted models and enables interpretation. Importantly, because the lasso produces sparse coefficients, clustering occurs in a lower-dimensional space than the original features, which is typically much simpler computationally and easier to interpret. Example clusterings and visualizations appear in Section 3. 

_Remark_ 2.1 _. Parallelization of weighted fitting._ Attention lasso fits a separate model for each test point, which may seem prohibitive. However, model fitting is embarrassingly parallel across test points. Additionally, the computational cost of attention lasso is not without precedent; it is similar to leave-one-out CV. For application to other models— for example gradient boosting (Section 7)— other computational tricks are needed. 

_Remark_ 2.2 _. Cost of cross-validation._ Selecting _m_ via cross-validation is efficient because it does not require refitting models for each candidate value. For each fold _k_ , apply the full training algorithm using ( **_X_** _−k,_ **_y_** _−k_ ) as the training set and **_X_** _k_ as the prediction set. Store the baseline lasso predictions **_y_ ˆ** base _,k_ and the attention model predictions **_y_ ˆ** attn _,k_ , and use these pre-computed predictions to compare each candidate mixing value (e.g. _m ∈{_ 0 _,_ 0 _._ 1 _,_ 0 _._ 2 _, . . . ,_ 1 _}_ ): **_y_ ˆ** _k_ ( _m_ ) = (1 _− m_ ) **_y_ ˆ** base _,k_ + _m_ **_y_ ˆ** attn _,k_ . Finally, estimate the cross-validation error for each candidate _m_ and select the value _m_ ˆ that minimizes the mean error across folds. 

_Remark_ 2.3 _. Adaptive selection of a mixing parameter for each test point._ For each prediction point, attention lasso blends the standard lasso model with a weighted model. So far, we have selected a single mixing parameter _m_ for all data. But perhaps some data points are well-represented by the global model and therefore should have _m_ = 0, while others are from a smaller subset within the data and would benefit from _m_ closer to 1. To compute _adaptive_ mixing values for a given prediction point **_x_**<sup>**_∗_**</sup> , compute a _weighted_ error metric using the attention weights for **_x_**<sup>**_∗_**</sup> when doing cross-validation. Because cross-validation predictions are already stored, this can be done in a vectorized fashion across all test points at once. By optimizing this weighted metric, we ˆ are selecting the mixing parameter _m_<sup>_∗_</sup> that gives the best performance for data similar to **_x_**<sup>**_∗_**</sup> . 

_Remark_ 2.4 _. Softmax temperature._ If desired we may use a “temperature” hyperparameter to augment softmax: before applying softmax, simply divide the scores by the temperature. For temperature _<_ 1, this will cause scores to concentrate more heavily on a small subset of features; temperature _>_ 1 will make scores more spread out and diffuse. 

## **3 Examples** 

To evaluate attention lasso, we compare its predictive performance to the lasso, XGBoost, LightGBM, random forest, and K-nearest neighbors (KNN) across real and simulated datasets. All methods are implemented using standard R packages: `glmnet` Friedman et al. [2010], `lightgbm` Shi et al. [2025], `xgboost` Chen et al. [2025], `ranger` Wright and Ziegler [2017], and `FNN` Beygelzimer et al. [2024], with default settings except where mentioned. The random forest was fit with 500 trees; XGBoost used _η_ = 0 _._ 1, maximum depth 6, with the number of rounds selected by 10-fold cross-validation with early stopping; LightGBM used 500 maximum rounds and no limit to tree depth; KNN used cross-validation to select the number of neighbors among _K_ = 3, 5, 10, or 15. Attention lasso used 500 trees for random forest proximity and 10-fold cross-validation to select the mixing hyperparameter and lasso shrinkage parameter. The lasso likewise used 10-fold cross-validation to select the shrinkage parameter. In all cases, we selected the parameters that minimized the cross-validated error. Methods were compared using _relative improvement_ in prediction squared error to the lasso: the percent improvement for model<sup><u>PSE</u></sup><sup>_−_</sup><sup><u>model</u></sup><sup>_<u>i</u>_</sup><sup><u>PSE</u></sup> _i_ is defined as 100 _×_<sup><u>lasso</u></sup> lasso PSE . 

Across our examples, we find that attention lasso usually matched or outperformed the lasso and was competitive with more complex models like LightGBM, XGBoost, KNN, and random forests. Importantly, it retains the advantage of the lasso’s sparsity and interpretability, and it offers a lens into data heterogeneity. 

### **3.1 Real data examples** 

##### **3.1.1 UC Irvine Machine Learning Repository** 

We consider 12 datasets from the UCI Machine Learning Repository Kelly et al. [2025]. We selected datasets that are regression tasks with fewer than _n_ = 5000 observations. For each dataset, we ran 50 iterations: each iteration split data into 50/50 train/test sets and fit 6 models (lasso, attention lasso, 

5 

LightGBM, XGBoost, random forest, KNN) with train data. In each iteration, we impute missing values using the column means of the non-missing values in the train set. A few datasets required preprocessing: we log-transformed the response (log( _y_ + 1)) in the Facebook Metrics data, and we averaged responses in multi-response datasets. Our results are summarized in Table 1 and Figure 3. In all but one dataset, attention lasso outperformed the lasso. Looking across all models, attention lasso had the best performance in four of the 12 datasets, XGBoost and LightGBM each had best performance for two, random forest and lasso each for one. In the dataset where lasso outperformed attention lasso, the difference was small (0 _._ 4%), which we attribute to the mixing parameter allowing the model to revert toward the baseline lasso when local adaptation is not useful. 

|Dataset|n|p|Attention|LightGBM|XGBoost|RF||KNN|
|---|---|---|---|---|---|---|---|---|
|Airfoil Self-Noise|1503|5|75.0 (1.1)|**84.3 (0.3)**|84.1 (0.3)|69.3 (0.3)|48.2|(0.8)|
|Auto MPG|398|7|**31.8 (0.9)**|26.1 (1.0)|17.2 (1.1)|26.4 (1.1)|10.1|(1.2)|
|Automobile|205|25|**37.1 (1.3)**|27.4 (1.6)|12.3 (3.1)|33.0 (1.2)|-25.6|(2.5)|
|Communities & Crime|1994|127|**3.1 (0.4)**|-1.5 (0.4)|-13.5 (0.7)|1.1 (0.3)|-19.0|(0.7)|
|Concrete Comp. Strength|1030|8|62.8 (0.7)|**77.7 (0.3)**|74.4 (0.4)|64.1 (0.4)|17.6|(1.0)|
|Facebook Metrics|500|18|93.6 (0.9)|90.5 (0.4)|**94.1 (0.5)**|93.4 (0.3)|56.8|(2.0)|
|Forest Fires|517|12|-0.4 (0.3)|-0.1 (0.8)|-9.0 (4.1)|-21.1 (6.3)|-24.5|(5.6)|
|Infrared Therm. Temp.|1020|33|3.4 (0.5)|-1.8 (1.0)|-4.3 (1.1)|**6.0 (0.7)**|-11.2|(0.9)|
|Liver Disorders|345|5|0.2 (0.7)|1.7 (1.0)|-12.5 (1.7)|**3.7 (1.0)**|3.6|(1.2)|
|Real Estate Valuation|414|6|18.2 (1.3)|24.2 (0.8)|14.0 (2.0)|**29.9 (0.9)**|8.6|(0.8)|
|Servo|167|4|63.8 (1.7)|23.3 (1.9)|**74.0 (1.6)**|42.2 (1.5)|-21.9|(4.5)|
|Stock Portfolio Perf.|315|12|**60.4 (0.8)**|32.1 (1.9)|23.9 (2.4)|-15.6 (2.8)|-200.7|(6.1)|



Table 1: _Mean (SE) of relative improvement (%) over lasso across 12 datasets in the UCI repository (higher is better). Best method per dataset in bold. Attention lasso has best performance on 4 of the 12, random forest on 3, LightGBM and XGBoost each on 2, and lasso on 1. Attention lasso outperforms lasso on 11 of the 12, and nearly matches on the last (0.4% worse)._ 

6 

#### Relative improvement over Lasso across UCI ML datasets 

(comparing attention methods to baseline ML methods) 



<!-- Start of picture text -->
Airfoil Self−Noise Auto MPG<br>n = 1503, p = 5 n = 398, p = 7<br>Attention Lasso<br>LightGBM<br>XGBoost<br>Random Forest<br>K−Nearest Neighbors<br>0.00 0.25 0.50 0.75 0.0 0.2 0.4<br>Automobile Communities and Crime<br>n = 205, p = 25 n = 1994, p = 127<br>Attention Lasso<br>LightGBM<br>XGBoost<br>Random Forest<br>K−Nearest Neighbors<br>−0.50 −0.25 0.00 0.25 0.50 −0.2 −0.1 0.0 0.1<br>Concrete Compressive Facebook Metrics<br>Strength<br>n = 500, p = 18<br>n = 1030, p = 8<br>Attention Lasso<br>LightGBM<br>XGBoost<br>Random Forest<br>K−Nearest Neighbors<br>0.0 0.2 0.4 0.6 0.8 0.00 0.25 0.50 0.75 1.00<br>Forest Fires Infrared Thermography<br>Temperature<br>n = 517, p = 12<br>n = 1020, p = 33<br>Attention Lasso<br>LightGBM<br>XGBoost<br>Random Forest<br>K−Nearest Neighbors<br>−0.75 −0.50 −0.25 0.00 −0.2 −0.1 0.0 0.1<br>Liver Disorders Real Estate Valuation<br>n = 345, p = 5 n = 414, p = 6<br>Attention Lasso<br>LightGBM<br>XGBoost<br>Random Forest<br>K−Nearest Neighbors<br>−0.2 0.0 0.2 0.0 0.1 0.2 0.3 0.4<br>Stock Portfolio<br>Servo<br>Performance<br>n = 167, p = 4<br>n = 315, p = 12<br>Attention Lasso<br>LightGBM<br>XGBoost<br>Random Forest<br>K−Nearest Neighbors<br>−0.5 0.0 0.5 −3 −2 −1 0<br>Relative Improvement over Lasso<br>Each box represents 50 train/test split simulations. Lasso baseline shown at 0.<br><!-- End of picture text -->

Figure 3: _Results described in Section 3.1 and summarized in Table 1. Across 50 train/test splits for each dataset, attention lasso has strong performance relative to lasso, XGBoost, LightGBM, random forest and KNN. In each plot, the vertical line at x_ = 0% _indicates no change relative to the lasso, and larger values indicate better performance (lower PSE) than the lasso._ 

7 

##### **3.1.2 Interpretation of the final model** 

The lasso is easy to interpret: a lasso model consists of a single, sparse vector of coefficients for the entire dataset. Although attention lasso produces a coefficient vector for each test point, these can be summarized by clustering. The shared regularization and blending with the baseline model ensure that individual models are similar enough for clustering to be meaningful. 

To interpret the fitted models from attention lasso, first compute the blended model coefficients for each data point: (1 _− m_ ) **_β_**<sup>ˆ</sup> base + _m_ **_β_**<sup>ˆ</sup> attn. Then cluster these blended coefficients with `protoclust` Bien and Tibshirani [2022] and visualize them with a heatmap to reveal within-cluster patterns. Figure 4 illustrates this pipeline. The rightmost plot in each row compares the PSE from attention lasso and baseline lasso within each cluster. 



<!-- Start of picture text -->
Auto MPG<br>Clustering Coefficients across clusters Performance Cluster size<br>3 16<br>40<br>80<br>2 12 120<br>160<br>1 8<br>Cluster ID<br>1<br>0 Features 4 2<br>0 50 100 150 200 3<br>2 3 4 5 6 7 4<br>Horizontal line shows 4 clusters −1 0 1 Attention PSE<br>Stock Portfolio Performance<br>Clustering Coefficients across clusters Performance Cluster size<br>20<br>2.0 40<br>3 60<br>1.5<br>1.0 2 Cluster ID<br>1<br>0.5<br>2<br>0.0 Features 1 3<br>4<br>0 50 100 150<br>5<br>0.5 1.0 1.5 2.0 6<br>Horizontal line shows 6 clusters −2 −1 0 1 2 Attention PSE<br>Facebook Metrics<br>Clustering Coefficients across clusters Performance Cluster size<br>16<br>40<br>0.75 80<br>120<br>12<br>160<br>0.50<br>8<br>0.25 Cluster ID<br>1<br>0.00 Features 4 2<br>0 50 100 150 200 250 3<br>2 3 4 5 6 7 4<br>Horizontal line shows 5 clusters −0.2 −0.1 0.0 0.1 0.2 Attention PSE<br>(a) Coefficient clustering. (b) Coefficient heatmap. (c) Performance comparison.<br>Height<br>Lasso PSE<br>Height<br>Lasso PSE<br>Height<br>Lasso PSE<br><!-- End of picture text -->

Figure 4: _Clustered coefficients and performance for the Auto MPG (top row) Stock Portfolio Performance (middle row) and Facebook Metrics datasets (bottom row). Models were trained using a random 50% of data, and performance is reported using the remainder. Attention lasso coefficient clustering reveals patterns in the data that may be useful for characterizing data heterogeneity._ 

8 

### **3.2 Simulations** 

We evaluate attention lasso on four simulated settings designed to test different forms of data heterogeneity. Each setting uses training and test datasets with _n_ = 300 observations and _p_ = 30 features (except Setting 2, which uses _p_ = 100). Features are generated as **_X_** _∼ N_ ( **0** _,_ **_I_** _p_ ) with modifications described below, and responses are generated as _yi_ = **_x_**<sup>_⊤_</sup> _i_<sup>**_β_**</sup><sup>_i_+</sup><sup>_ϵi_where</sup><sup>_ϵi∼N_(0</sup><sup>_, σ_2)with</sup><sup>_σ_</sup> chosen to achieve an average signal-to-noise ratio of approximately 2.5. Table 2 summarizes the key characteristics of each setting. 

|Setting|Type of Heterogeneity|Challenge|
|---|---|---|
|1|Continuous coefcient variation|Smooth gradient in coefcients and covariates|
|2|High-dimensional continuous variation|High noise (_p_= 100)|
|3|Discrete subgroups with confounders|Minority subgroup (20%) with spurious co-|
|||variate shifts|
|4|Overlapping soft clusters|Three groups with blended membership|



Table 2: _Overview of simulation settings._ 

**Setting 1: Continuous coefficient variation.** We generate observation-specific coefficients that vary smoothly along a latent gradient. Each observation _i_ has a latent position _zi ∼_ Uniform( _−_ 1 _,_ 1), and coefficients are defined as a convex combination: 



Here, **_β_** 0 = (3 _,_ 3 _,_ 3 _,_ 3 _,_ 0 _, . . . ,_ 0)<sup>_⊤_</sup> and **_β_** 1 = ( _−_ 2 _, −_ 2 _, −_ 2 _, −_ 2 _,_ 0 _, . . . ,_ 0)<sup>_⊤_</sup> . Thus, observations with _zi_ = _−_ 1 have coefficients **_β_** 0, those with _zi_ = 1 have coefficients **_β_** 1, and intermediate values smoothly interpolate between them. To create covariate distribution shifts correlated with the coefficient variation, we add _zi_ to features 1–4: **_x_** _i,j ←_ **_x_** _i,j_ + _zi_ for _j ∈{_ 1 _,_ 2 _,_ 3 _,_ 4 _}_ . This setting tests whether attention lasso can adapt to a continuous spectrum of coefficient variation rather than discrete clusters. 

**Setting 2: High-dimensional continuous variation.** This extends Setting 1 to higher dimensions ( _p_ = 100) with different coefficient patterns. We again use _zi ∼_ Uniform(0 _,_ 1) to define smoothly varying coefficients, but now with **_β_** 0 = (3 _,_ 2 _,_ 1 _,_ 0 _,_ 0 _,_ 0 _,_ 0 _, . . . ,_ 0)<sup>_⊤_</sup> and **_β_** 1 = ( _−_ 1 _,_ 0 _,_ 1 _,_ 2 _,_ 3 _,_ 2 _,_ 0 _, . . . ,_ 0)<sup>_⊤_</sup> . Features 1–6 are shifted by _zi_ to create a smooth gradient in the covariate distribution. This setting evaluates performance when the signal must be identified among many noise features. 

**Setting 3: Discrete subgroups with spurious heterogeneity.** We partition observations into a majority group (80%) and a minority group (20%). The majority has coefficients **_β_** 1 with four non-zero entries in positions 1–4; the minority has coefficients **_β_** 2 with four non-zero entries in positions 5–8. All non-zero coefficient values are drawn independently from _N_ (0 _,_ 1). To ensure the minority group is distinguishable in covariate space, we add a constant shift of 2 to features 1–8 for minority observations. Additionally, we introduce spurious covariate heterogeneity to make subgroup identification more challenging: for a random 50% of all observations (cutting across both groups), we add random shifts drawn from _N_ (0 _,_ 1) to 10 randomly selected noise features. This creates covariate differences that are uncorrelated with the response and do not align with the true subgroup structure, testing whether attention lasso can focus on relevant heterogeneity. 

**Setting 4: Overlapping soft clusters.** We generate data with three overlapping subgroups having distinct coefficient vectors: **_β_** 1 = (3 _,_ 3 _,_ 2 _,_ 1 _,_ 0 _, . . . ,_ 0)<sup>_⊤_</sup> , **_β_** 2 = ( _−_ 2 _,_ 1 _,_ 3 _,_ 2 _,_ 0 _, . . . ,_ 0)<sup>_⊤_</sup> , and **_β_** 3 = (1 _, −_ 2 _, −_ 1 _,_ 3 _,_ 0 _, . . . ,_ 0)<sup>_⊤_</sup> . Rather than hard cluster assignments, each observation has soft membership in all three groups. We draw a latent position _zi ∼_ Uniform(0 _,_ 1) and compute membership weights by evaluating three Gaussian density functions centered at 0.2, 0.5, and 0.8 (each with standard deviation 0.15) at _zi_ , then normalizing these densities to sum to 1. Each observation’s coefficients are then a weighted blend: **_β_** _i_ = _wi,_ 1 **_β_** 1 + _wi,_ 2 **_β_** 2 + _wi,_ 3 **_β_** 3. We shift features 1–4 by _zi_ to create covariate patterns correlated with the blended coefficient structure. This setting evaluates whether attention lasso can handle graded, overlapping group memberships rather than discrete clusters. 

For each setting, we generate 100 independent train sets with test sets of equal size. Our results are summarized in Table 3 and Figure 5. When data are heterogeneous, attention lasso matches or 

9 

outperforms the lasso as expected. We also find that attention lasso has performance close to that of random forest and XGBoost, but it is much more interpretable. Further, because attention lasso blends a baseline model (fit to all data, equally weighted) with the individual weighted model, it does not typically perform _worse_ than the lasso, whereas random forest and XGBoost have no such protection. 

|Setting|Attention|Light|GBM|XGBoost||RF||KNN|
|---|---|---|---|---|---|---|---|---|
|1|53.0 (0.7)|49.1|(0.7)|49.0 (0.7)|52.2|(0.5)|**56.9 **|**(0.5)**|
|2|**1.9 (0.4)**|-15.5|(1.3)|-36.4 (2.0)|-18.1|(1.3)|-22.2|(1.6)|
|3|**5.8 (0.8)**|0.2|(1.2)|-8.5 (1.8)|1.2|(1.3)|-15.9|(2.0)|
|4|11.0 (0.9)|6.2|(0.7)|2.3 (1.5)|**11.5 **|**(0.6)**|11.5|(0.8)|



Table 3: _Mean (SE) of relative improvement (%) over the lasso for attention lasso, LightGBM, XGBoost, random forest, and KNN across four data-generating settings (100 simulations each). Simulation details are described in Section 3.2._ 

Relative improvement over lasso across simulation setups (comparing attention lasso to random forest, XGBoost, and KNN) 



<!-- Start of picture text -->
Setting 1 Setting 2<br>Attention lasso Attention lasso<br>LightGBM LightGBM<br>XGBoost XGBoost<br>Random forest Random forest<br>K−nearest neighbors K−nearest neighbors<br>0.0 0.2 0.4 0.6 −0.75 −0.50 −0.25 0.00<br>Setting 3 Setting 4<br>Attention lasso Attention lasso<br>LightGBM LightGBM<br>XGBoost XGBoost<br>Random forest Random forest<br>K−nearest neighbors K−nearest neighbors<br>−0.6 −0.4 −0.2 0.0 0.2 −0.4 −0.2 0.0 0.2 0.4<br>Relative improvement in PSE<br>Each box represents 100 simulations.<br><!-- End of picture text -->

Figure 5: _Simulation results described in Section 3.2 and summarized in Table 3. Across a variety of simulations, attention lasso typically (1) matches or improves on the lasso and (2) is competitive with more complex models when they perform well. The vertical line at x_ = 0% _indicates no performance difference from the lasso, and values to the right indicate better performance._ 

## **4 Background: attention and in-context learning** 

The work in this paper was inspired by self-attention and in-context learning, which in many settings offer impressively accurate predictions, but are not easy to interpret. We aimed to develop a method that retains interpretability and improves predictive performance relative to other common supervised learning methods. Here, we offer background on self-attention and in-context learning before summarizing their relationship to attention for supervised learning. 

10 

### **4.1 Self-attention** 

We describe _self-attention_ Vaswani et al. [2017] in the context of natural language processing. Selfattention assigns weights to reveal relationships between words within a sentence. For example, in the sentence “Harvey played with Jonny on the swings before he went home”, the self-attention mechanism should assign a large weight from the word “he” to “Harvey” to resolve the reference. 

Concretely, suppose our 11-word example sentence is written as a matrix **_X_** _∈_ R<sup>11</sup><sup>_×p_</sup> , where each row is a word embedding in _p_ dimensions. The _i_<sup>th</sup> row of **_X_** corresponds to the _i_<sup>th</sup> word in the sentence; in our example, the first row is the embedding for “Harvey”, the second “played” and so on. Then the attention mechanism is written as: 



where 



hyperparameters _dk, dv ∈_ N _._ 

The matrices _Q, K_ and _V_ are called “query”, “key” and “value” matrices: they represent three different encodings of the data. This allows for non-symmetrical relationships. That is, the attention from word _i_ to word _j_ can differ from the attention from word _j_ to word _i_ .<sup>2</sup> 

This concise notation obscures the fact that attention uses a weighted similarity matrix **_XW X_**<sup>_T_</sup> : 



where **_W_** = **_W_** _q_ **_W_** _k_<sup>_T_</sup> for simplicity. Element ( _i, j_ ) of **_XW X_**<sup>_T_</sup> is a weighted dot product between the projected word embeddings for word _i_ and word _j_ . Now it is clearer to see that the attention mechanism performs two important tasks. First, it identifies _weights_ relating each word (row of **_X_** ) to each other word: 



The expression **_XW X_**<sup>_T_</sup> is a matrix where the first row encodes the similarity of “Harvey” to every word in the sentence, the second row encodes similarity from “played” to each other word, and so on. Applying row-wise softmax makes each row sum to 1, and therefore each row can be thought of as a set of weights. 

The second step of attention multiplies our weights by **_XW_** _v_ : 



thereby rewriting each row of **_X_** — each word — as a weighted sum of the words in the sentence, and multiplying this by weight matrix **_W_** _v_ . 

Importantly, there is no global, “one-size-fits-all” representation of each word. Rather, self-attention enables each word’s representation to be dynamically informed by the entire context of the sentence. 

> 2In the sentence “Harvey played with Jonny on the swings before he went home,” “he” may attend strongly to “Harvey”, while “Harvey” may attend more strongly to “played”. 

11 

### **4.2 In-context learning: attention from the test set to the train set** 

In-context learning (ICL) Dong et al. [2024] is a method whereby models adapt their predictions based on an input query which includes both train and test data without updating their parameters. For example, an ICL query could be: 

Harvey is sliding on the slide. Martha is swinging on the swing. Harvey and Martha are playing in a ? 

We have given our model example sentences and their completing words (“demonstration pairs”), and we expect our model to reply with “playground”. 

More generally, suppose we have _k_ demonstration pairs ( **_S_** 1 _,_ **_y_** 1) _, . . . ,_ ( **_S_** _k,_ **_y_** _k_ ), where **_S_** _i_ is a sentence and **_y_** _i_ is its label or completion, and a new test sentence **_S_**<sup>_∗_</sup> . The prompt is: 



and the goal is to predict **_y_**<sup>_∗_</sup> . We will again use attention; now, we use attention to relate **_S_**<sup>_∗_</sup> to the demonstration examples **_S_** 1 _, . . . ,_ **_S_** _k_ . We begin with an embedding that maps each **sentence** **_S_** _i_ to a **vector** **_x_** _i ∈_ R<sup>_p_</sup> . We define **_X_** _∈_ R<sup>_k×p_</sup> as the matrix where each row is one of our demonstration examples and **_x_**<sup>_∗_</sup> _∈_ R<sup>_p_</sup> , our test sentence. Then we use attention as in Section 4.1, however now we also consider attention from **_x_**<sup>_∗_</sup> to **_X_** : 



where 



We have simplified **W** = **W** _q_ **W** _k_<sup>_T_asinEquation8.</sup> As before, attention performs two steps: 

1. constructing a weight matrix, softmax � **_X_** ICL _~~√~~_ **_W X_** _dk_ <u>ICL</u><sup>_T_</sup> �, the _k_ + 1<sup>th</sup> row of which relates test observation **_x_**<sup>_∗_</sup> to each of the demonstration observations, and 

2. multiplying these weights by **_X_** ICL **_W_** _v_ , where the final row is a weighted combination of the demonstration observations and itself, multiplied by **_W_** _v_ . 

The final vector corresponding to the test query **_x_**<sup>_∗_</sup> is our prediction, and can be written as: 



**_x_**<sup>_∗_</sup> **0** <u>�</u> **_W X_** ICL<sup>_T_</sup> The attention weights, softmax _~~√~~ dk_ , closely mirror our attention weights, softmax � **_x_**<sup>_∗_</sup> **_W X_**<sup>_T_�</sup> _._ �<sup><u>�</u></sup> � 

Note that in-context learning starts with a pretrained model and _does not update its weights at test time_ . Rather, it uses the attention mechanism to represent the test observation in context of the demonstration examples. 

12 

### **4.3 Attention, in-context learning, and supervised learning** 

The first component of attention takes the form 



and defines a similarity matrix across rows of **_X_** according to **_W_** . Typically, attention appears in complex models, involving multi-head attention (multiple attention mechanisms in the same layer of a neural network, each with different weights) and sequential attention (many layers of a neural network will use attention heads), making **_W_** difficult to interpret directly. 

Here we use just a _single_ attention head. Initially, we tried form (13) with a diagonal **_W_** for this head: the similarity weights provide an intuitive soft matching, relative to the feature importances on the diagonal of **_W_** . It is therefore natural to consider **_w_**<sup>_∗_</sup> = softmax � **_x_**<sup>_∗_</sup> **_W X_**<sup>_T_�</sup> as interpretable weights for localized, weighted model fitting for **_x_**<sup>_∗_</sup> . 

But to make our procedure more powerful, we instead use random forest proximity. Though this appears very different from the neural network attention mechanism, it can approximate its behavior more closely by exploiting nonlinearities. 

At first glance, fitting a model for **_x_**<sup>_∗_</sup> distinguishes our approach from ICL, which does not update model weights. This connects to recent work showing that ICL with transformers can emulate gradient descent Von Oswald et al. [2023], Ren and Liu [2024], Deutch et al. [2023], and in some cases, corresponds exactly to fitting a linear model to the context. Thus, our weighted model fitting step can be viewed as an explicit analog of the optimization that is done implicitly by ICL. 

## **5 Other related work** 

### **5.1 Kernel methods and local regression** 

Our attention approach is closely related to kernel-based methods that weight training observations by their similarity to a test point. We compare the softmax weighting scheme used in the attention mechanism with the model fitting approaches of kernel regression and locally weighted regression. 

##### **5.1.1 Nadaraya–Watson kernel regression and attention** 

The Nadaraya–Watson (N–W) kernel estimator Nadaraya [1964], Watson [1964] with a Gaussian kernel defines the weight between a query point **_x_**<sup>_∗_</sup> and training point **_x_** _i_ as: 



where _σ_ is a bandwidth parameter. Attention weights take an analogous softmax form: 



where **_W_** is a learned weight matrix. Importantly, the two approaches define similarity quite differently: the attention mechanism learns **_W_** using the response **_y_** , while N-W regression defines similarity based only on geometric distance in covariate space. Additionally, N-W regression uses these weights to ˆ compute a weighted average of responses, _y_<sup>_∗_</sup> =<sup>�</sup> _i_<sup>_wiyi_,ratherthanperformingcontinuedmodel</sup> fitting (like neural networks and attention lasso). 

##### **5.1.2 Locally weighted regression** 

Locally weighted regression (LWR), including methods like LOESS Cleveland and Devlin [1988], is structurally similar to attention lasso. LWR fits a separate weighted linear (or polynomial) model at each prediction point: 



13 

where the weights _wi_ typically decrease with Euclidean distance from **_x_**<sup>_∗_</sup> . 

Drawing on the principle of supervised similarity from neural attention, our method extends locally weighted regression by defining similarity in a _supervised_ manner, ensuring that two observations are considered similar when they are close with respect to features and feature interactions that relate **_X_** to **_y_** . Our ridge-based attention uses feature importance to weight distances, while random forest proximity finds nonlinear relationships that Euclidean distance cannot represent. 

Additionally, attention lasso also incorporates sparsity through _L_ 1 regularization, fitting models of the form: 



using a shared penalty parameter _λ_ across all test points to encourage consistent model complexity. Finally, our approach blends the global (baseline) model with the local (attention) model using a mixing parameter _m_ chosen through cross-validation, allowing the data to reveal the extent to which local adaptation improves over the single global model. 

### **5.2 Customized training** 

Customized training Powers et al. [2016] addresses data heterogeneity by first partitioning test observations into clusters and then fitting separate models within each cluster using nearby training points. This approach is most appropriate when subgroups are clearly defined and the chosen clustering aligns with the underlying data structure. However, customized training has limitations. First, it requires pre-specifying the number of clusters and making hard assignments of observations to groups. Second, because clustering is performed on test data, it cannot leverage information from the response **_y_** to define clusters meaningful for predictin **_y_** . In contrast, our method adaptively determines the appropriate training set for each prediction point through a soft weighting mechanism that is explicitly informed by the relationship between features and the response. 

## **6 Attention for time series and spatial data** 

Thus far, we have defined similarity between a test and train point using random forest proximity on their covariates and the response. Now, we consider data where there are relationships between observations; for example, time series data consists of measurements of the same variables at different points in time, and likewise for spatial data across 2- or 3-dimensional coordinates. This is somewhat similar to _retrieval_ for neural networks. For simplicity, we first discuss the time series case. 

With time series data, we define similarity not only with regards to the current covariates, but also to their context. For example, a feature value may be interpreted differently depending on whether its recent history has been increasing or decreasing. To address this, we fit a random forest using the current features as well as their lagged values. Our definition of “similarity” therefore prioritizes observations with similar values _and similar contexts_ . During model fitting, we then have the choice to include only current values or current and lagged values, whichever are more appropriate for the task. The same principle applies to spatial data. To define similarity across pixels in images, we fit our random forest using features from each individual pixel and its nearest neighbors. 

There are many possible modifications to the similarity definition, appropriate for different datasets: 

- **Spatial symmetry:** In image data, the relative position of neighboring pixels may not matter—only that two pixels are adjacent. A simple approach is to use the mean and standard deviation of neighbors rather than their raw feature values. 

- **Categorical context:** In single-cell spatial data, we may define similarity using a cell’s measured covariates together with neighboring cell _types_ rather than their raw features. 

- **Distance weighting:** For time series, we choose a lag; for spatial data, we choose a number of neighbors. These could be extended to a kernel weighting that downweights observations further away in time or space. 

14 

### **6.1 Example: time series regression** 

We consider a time series dataset **_X_** _,_ **_y_** where each row of **_X_** and the corresponding entry of **_y_** represent the same variables recorded across time. For instance, in finance, **_y_** might denote the return at time _t_ and **_X_** includes variables representing market indicators from previous periods. 

Time series forecasting typically faces the challenge of limited observations similar to the current moment. We leverage the attention mechanism to identify historically relevant training examples, both in terms of current features (measured at time _t_ ) and recent history (lagged indicators from time _< t_ included as columns of **_X_** ). 

We applied attention lasso to the US economic change dataset ( `us` ~~`c`~~ `hange` ) from the fpp3 Hyndman [2025] package in R, which contains quarterly percentage changes in personal consumption expenditure, disposable income, production, savings, and unemployment. This dataset spans 1972-2019 with quarterly measurements, totaling _n_ = 188 rows. For baseline lasso, random forest, XGBoost and K-nearest neighbors, we included 10 lagged values of each variable, resulting in _p_ = 54 features. For attention lasso, we use the lagged features only to compute random forest similarity; the baseline and attention models are fitted with the original 4 features only, and the random forest uses the most recent 1 year (lag = 4) to define similarity. The response variable is the percentage change in consumption. 

We vary the training set size from 50% to 90% of the data and test on the remaining observations, and as before, we compare baseline lasso to attention lasso, random forest, XGBoost and K-nearest neighbors. To choose hyperparameters, we use 5-fold cross-validation split by time: for each fold _k ≥_ 2, we train on all observations from folds 1 through _k −_ 1 and test on fold _k_ . 

In this example, attention lasso performed much better than its competitors when trained with at least 70% of the data, but with smaller datasets, random forest with lagged features had the best performance. 

|Train Fra|ction|Attention|RF|XGBoost|KNN|Mixing<br>(0: base, 1: attn)|
|---|---|---|---|---|---|---|
||0.5|3.0|**64.4**|63.1|61.0|1.0|
||0.6|-16.4|**48.5**|38.5|34.4|1.0|
||0.7|**11.4**|-72.2|-54.6|-130.5|1.0|
||0.8|**32.8**|-31.5|-40.2|-153.9|1.0|
||0.9|**54.2**|-44.5|-50.4|-144.2|1.0|
|Mean|(SE)|**17 (12.2)**|-7.1 (26.9)|-8.7 (24.7)|-66.6 (47.0)|1 (0)|



Table 4: Relative improvement (%) over lasso baseline with the time series dataset `us` ~~`c`~~ `hange` from the fpp3 Hyndman [2025] package. Positive values indicate better performance than lasso, negative indicate worse. We find that attention lasso performs best with longer history ( _>_ 70% of data), and the CV-selected mixing values at 1 suggest the best performance is from the attention-weighted models. 

### **6.2 Example: spatial data** 

Now we turn to a dataset of 45 DESI mass spectrometry images of prostate tissues, taken from [Banerjee et al., 2017]. Each pixel in a mass spectrometry image represents a location at which 1,600 molecular abundances were measured. Our goal is to assign labels to the pixels in each image using these 1,600 features. Of the 45 images, 17 correspond to tumor tissue (pixel labels _y_ = 1), and 28 to normal tissue (pixel labels _y_ = 0). We have 17,735 pixels total. 

We compared lasso and attention lasso across ten train/test splits, with entire images assigned to either train or test. For attention weights, we defined pixel similarity by training the random forest on each pixel’s features together with those of its 8 spatial neighbors. This provides local context analogous to the temporal context in time series. 

Across the ten splits, attention lasso outperformed lasso: it had an average AUC of 0.65 relative to lasso’s average of 0.59. In 3 splits, lasso only fit the null model, presumably due to pixel heterogeneity masking global signal. In 2 of these cases, attention lasso was still able to find signal in local models. Model performance is shown in Table 5, and an example is in Figure 6. 

15 



<!-- Start of picture text -->
Tissue sample Coefficient clusters<br>(true y = 0) Total features: 1600, selected features: 9<br>Feature<br>Probability value<br>−0.6 −0.3 0.0 0.3 0.6<br>0.00 0.25 0.50 0.75 1.00<br><!-- End of picture text -->

Figure 6: _Example mass spectrometry image from split 1 with attention lasso mixing = 1 (chosen by cross-validation). Left: normal tissue sample with pixels colored by attention lasso model probability. Right: fitted model coefficients for the sample image._ 

|Split|Lasso|Attention lasso|Mixing|
|---|---|---|---|
|1|0.500|**0.718**|1.0|
|2|0.731|**0.746**|0.2|
|3|0.425|**0.475**|1.0|
|4|0.669|**0.675**|0.2|
|5|**0.680**|0.678|1.0|
|6|0.500|**0.750**|0.2|
|7|**0.543**|0.539|0.0|
|8|**0.500**|**0.500**|0.0|
|9|0.684|**0.685**|0.0|
|10|**0.698**|0.696|0.3|
|Mean|0.593|**0.646**|0.39|
|Std. err|0.035|0.032|0.137|



Table 5: _AUC comparison across train/test splits for DESI mass spectrometry data, distinguishing prostate cancer tissue from normal._ 

## **7 Generalizations** 

### **7.1 Attention for other machine learning methods** 

So far we have focussed on the special case of the lasso, but the algorithm we have described applies to supervised learning methods beyond the lasso. The steps are: (1) fit a random forest to estimate attention weights, (2) fit a base learner, (3) fit a _weighted_ base learner for each test point and (4) blend the predictions of the models from (2) and (3). The base learner could be, for example, XGBoost or LightGBM. This is described in Algorithm 1 given earlier. 

Nonlinear models like LightGBM typically adapt well to data heterogeneity: they can fit localized models in a way that linear models cannot. However, attention can still be useful. The attention weights focus each test point’s model on the most relevant training examples; as a result, the localized models may not need to be as complex. In the case of gradient boosted trees, for example, the local models may each be shallower and composed of fewer trees. This results in a more interpretable ensemble of models: first, the attention weights are useful for understanding similarity between test and train samples, and second, clustered feature importances across trees may identify a clustering of 

16 

the data (Figure 7 shows an example). The importance heatmap reveals distinct feature importance patterns across clusters, suggesting meaningful subgroup structure. 



<!-- Start of picture text -->
Automobile<br>Clustering Importance across clusters<br>4<br>3<br>Importance<br>1.5<br>2<br>1.0<br>0.5<br>0.0<br>1<br>0<br>Feature<br><!-- End of picture text -->

Figure 7: _Clustered feature importances for the Automobile dataset modeled with attention LightGBM. Models were trained using a random 50% of data, and performance is reported using the remainder._ 

### **7.2 Approximate weighted attention** 

Repeated model fitting for complex models may be computationally prohibitive. We therefore present an approximate version of our attention algorithm that avoids refitting by keeping the original tree structure but updating the predictions within each leaf. Specifically, we replace the original leaf predictions with attention-weighted averages of the train data responses. The tree ensemble provides a partition of the space; the attention weights refine predictions within each partition. See Algorithm 3 for details. 

**Algorithm 3** _Approximate attention_ _<u>for</u> tree methods_ 

**Input:** Train set **_X_** _,_ **_y_** , test set **_X_**<sup>_∗_</sup> , mixing parameter _m ∈_ [0 _,_ 1], base model class (e.g., XGBoost). **<u>Output:</u>** <u>Predictions</u> **_<u>y</u>_ ˆ**<sup>_∗_</sup> <u>for</u> **_<u>X</u>_**<sup>_∗_</sup> <u>, and ftted models for each row of</u> **_<u>X</u>_**<sup>_∗_</sup> <u>.</u> 

1. **Compute attention weights** **_A_**<sup>**ˆ**</sup> 

   - **_A_**<sup>_∗_</sup> as in Algorithm 2. 

2. **Fit baseline model:** Fit a baseline model to **_X_** _,_ **_y_** using the base model class. Predict for **_X_**<sup>_∗_</sup> to obtain **_y_ ˆ** base<sup>_∗_.</sup> 

3. **Estimate attention-weighted predictions:** For **_x_**<sup>_∗_</sup> _i_<sup>,the</sup><sup>_i_throwof</sup><sup>**_X_**</sup><sup>_∗_,traverseeachof</sup> the trees to their terminal nodes. In each terminal node, compute a weighted average of the responses **_y_** for the training points in that node using attention weights. Then take an average of these predictions across all the trees to obtain **_y_ ˆ** attn<sup>_∗_.</sup> 

4. **Combine predictions:** Return the weighted average 



Note: the mixing <u>parameter</u> _m_ is selected through cross-validation. 

> _Remark_ 7.1 _. Weighting the trees_ . Another possibility to step 3 above would be to fix the trees, and use their predictions as features. Then fit a ridge or lasso model to these features, with using attention weights as the observation weights. 

17 

### **7.3 Examples: machine learning model attention** 

We return to the UCI datasets (Section 3.1) to evaluate attention with LightGBM as the base learner. We compared LightGBM with max number of rounds 500 and no limit on max terminal leaves to three models trained with 100 rounds and 8 leaves max: (1) “shallow” LightGBM, (2) attention LightGBM and (3) approximate attention LightGBM. Our goal is to study whether attention weighting permits us to fit smaller, more interpretable models without sacrificing performance. All models used crossvalidation with early stopping to determine the number of rounds. Results are displayed over 50 random train/test splits per dataset. We find that attention LightGBM with fewer, shallower trees typically has performance close to or above that of LightGBM and shallow LightGBM (Figure 8), and attention LightGBM enables us to study feature importances across test points (Figure 7). 

Relative improvement over LightGBM across UCI ML datasets 



<!-- Start of picture text -->
Airfoil Self−Noise Auto MPG<br>n = 1503, p = 5 n = 398, p = 7<br>Approx. Attention LightGBM<br>Attention LightGBM<br>LightGBM (Small)<br>−1.0 −0.5 0.0 −0.05 0.00 0.05 0.10 0.15<br>Automobile Communities and Crime<br>n = 205, p = 25 n = 1994, p = 127<br>Approx. Attention LightGBM<br>Attention LightGBM<br>LightGBM (Small)<br>−0.1 0.0 0.1 0.2 0.3 0.000 0.025 0.050 0.075<br>Concrete Compressive Facebook Metrics<br>Strength<br>n = 500, p = 18<br>n = 1030, p = 8<br>Approx. Attention LightGBM<br>Attention LightGBM<br>LightGBM (Small)<br>−0.2 −0.1 0.0 −0.25 0.00 0.25 0.50 0.75<br>Forest Fires Infrared Thermography<br>Temperature<br>n = 517, p = 12<br>n = 1020, p = 33<br>Approx. Attention LightGBM<br>Attention LightGBM<br>LightGBM (Small)<br>−2e−04 0e+00 2e−04 4e−04 6e−04−0.05 0.00 0.05 0.10<br>Liver Disorders Real Estate Valuation<br>n = 345, p = 5 n = 414, p = 6<br>Approx. Attention LightGBM<br>Attention LightGBM<br>LightGBM (Small)<br>−0.050 −0.025 0.000 0.025 −0.05 0.00 0.05 0.10<br>Stock Portfolio<br>Servo<br>Performance<br>n = 167, p = 4<br>n = 315, p = 12<br>Approx. Attention LightGBM<br>Attention LightGBM<br>LightGBM (Small)<br>0.0 0.2 0.4 0.6 0.8 −0.1 0.0 0.1 0.2 0.3<br>Relative Improvement over LightGBM<br><!-- End of picture text -->

Figure 8: _Experiment comparing attention LightGBM to LightGBM. As our baseline we trained LightGBM using its default parameters: 500 rounds and no limit on tree depth. This model corresponds to x_ = 0 _in all plot facets. Then, using 100 rounds and a maximum of 8 leaves, we fit (1) LightGBM (small), (2) attention LightGBM, which fits a separate LightGBM model for each test point, and (3) approximate attention LightGBM, which uses a weighted average of training y in each node. LightGBM (small) shows the effect on performance of training fewer, shallower trees. The other two boxes additionally show the effect of using attention. Values to the right of 0 indicate improvement over LightGBM, values to the left indicate worse performance._ 

18 

## **8 Attention for longitudinal data drift** 

We designed an algorithm to address a scenario common in industry and medicine, where a model is fitted at some initial time and then applied days, weeks, or months later. In these cases, refitting a complex model may be difficult for computational or administrative reasons, even though applying a stale model ignores distribution shift and hurts performance. We propose a middle ground where we use the original model, but adapt predictions using the more recent labeled data. 

Suppose we fit a gradient boosting model to data ( **_X_** 1 _,_ **_y_** 1) at time 1. At time 2, we observe new training data ( **_X_** 2 _,_ **_y_** 2) drawn from a shifted distribution and wish to predict on test points **_X_** 3 from (or closer to) this new distribution. Specifically, we suppose the distribution of **_X_** has shifted, but the relationship **_y_** _|_ **_X_** has not changed much. We propose a method which does not require refitting a model using **_X_** 2 _,_ **_y_** 2, described in Algorithm 4. 

**Algorithm 4** _Attention_ _<u>for</u> data drift: attention-weighted residual correction_ 

**Input:** Boosted tree model _f_<sup>ˆ</sup> fitted at initial training time (with **_X_** 1 _,_ **_y_** 1) More recent data **_X_** 2 _,_ **_y_** 2 Prediction data **_X_** 3 

**<u>Output:</u>** <u>Predictions</u> **_<u>y</u>_ ˆ3** <u>for</u> **_<u>X</u>_** <u>3, and ftted models for each row of</u> **_<u>X</u>_** <u>3.</u> 

1. **Compute attention weights** **_A_**<sup>**ˆ**</sup><sup>_∗_</sup> from **_X_** 3 to **_X_** 2 using “boosted tree similarity”: the fraction of trees in _f_<sup>ˆ</sup> in which two points land in the same terminal node. For each observation in **_X_** 3, apply softmax to the similarity scores over **_X_** 2 so that weights sum to 1. 

- ˆ 

- 2. **Estimate residuals:** Compute residuals **_r_** 2 = **_y_** 2 _− f_<sup>ˆ</sup> ( **_X_** 2). For each test point, define **_r_** 3 as the attention-weighted average of **_r_** 2. 

- ˆ ˆ 

- 3. **Adjust predictions:** Compute final predictions as **_y_** 3 = _f_<sup>ˆ</sup> ( **_X_** 3) + **_r_** 3. 

_Remark_ 8.1 _. Intuition for Algorithm 4_ . Our approach blends gradient boosting with kernel-weighted local averaging: the tree provides both the base prediction and the similarity weights, while the local averaging corrects the predictions using recent data. 

_Remark_ 8.2 _. Extension for non tree-based learning algorithms:_ This method is designed for tree-based algorithms, but could in principle be applied to other learning methods. We need only a fitted model and a similarity measure. Tree-based algorithms conveniently provide a natural similarity measure via terminal node co-occurrence; for other methods, one could use a separate similarity measure such as Euclidean distance or a learned embedding. 

We studied the performance of our method under a scenario with covariate shift. We simulated covariate shift using a mixture of two distributions: in distribution A, all features are standard normal; in distribution B, features 6-10 have mean 2. Data are drawn from mixtures that shift over time: 10% B at time 1 (model training), 90% B at time 2 (adaptation data), and 95% B at time 3 (prediction). The response _y_ is generated as _y_ = _Xβ_ + _X_ 1<sup>2</sup><sup>_−X_</sup> 3<sup>2+ (</sup><sup>_X_4+</sup><sup>_X_5)2 +</sup><sup>_ϵ_,where</sup><sup>_β_has20nonzeroentries</sup> out of 50 total (each _±_ 2) and _ϵ ∼ N_ (0 _, σ_<sup>2</sup> ) with _σ_ = 36. We have _n_ = 300 observations at training and 200 at testing. We compared four modeling approaches: 

1. **Baseline** : Model trained and tested on time 1 data. 

2. **Refit** : Model trained on time 2 data and tested at time 3. 

3. **No adaptation** : Model trained on time 1 and tested at time 3. 

4. **Attention** : Model trained at time 1 with attention-weighted residual correction applied using time 2 data, tested at time 3. 

In all cases, we ran LightGBM using the `lightgbm` library in R, and using default hyperparameters. Across 50 simulations, we found that our approach recovered much of the performance lost due to data drift (Figure 9). 

19 



<!-- Start of picture text -->
Attention weighted residual−correction under data drift<br>n = 50 simulations<br>110<br>90<br>70<br>50<br>30<br>Baseline Refit No adaptation Attention<br>Train and test data<br>2) −σ<br>MSE<br>(<br>MSE<br>Excess<br><!-- End of picture text -->

Figure 9: _Model performance under data drift. “Baseline” is trained and tested at time 1; “Refit” is trained at time 2 and tested at time 3. “No adaptation” applies the time 1 model directly to time 3 data. “Attention” uses the time 1 model with attention-weighted residual correction from time 2 data to predict at time 3. Horizontal line shows median performance of Refit, which should be the best possible performance._ 

## **9 Discussion** 

This paper adapts the attention mechanism from neural networks to a general method for supervised statistical and machine learning methods for tabular data. By weighting training observations according to their _supervised_ similarity to each test point, our method fits models for heterogeneous data without requiring pre-specification of cluster structure. For the specific case of linear models, we found that attention lasso achieves lower prediction error than standard lasso under mixture-of-models settings, and empirical examples show that clustering the fitted coefficients reveals interpretable patterns that characterize data heterogeneity. 

In self-attention, the attention weights **_W_** are learned end-to-end: they are optimized jointly with all other model parameters via backpropagation. In contrast, our approach first computes the weights through random forest proximity, and then separately fits a weightedˆ model. We could instead have set up an optimization problem to jointly learn a weight matrix **_W_** _and_ fit the prediction model. However, this non-convex objective is challenging to optimize in practice, and the dependence between the attention weights and the downstream model introduces additional complexity and the risk of poor local minima. For this reason, we instead prefer a two-step approach which can be implemented easily with standard, well-tested software. 

We use random forest proximity to define the training weights, but note that any supervised weighting scheme that relates test points **_x_**<sup>_∗_</sup> to the training set **_X_** may be used. Random forest proximity is appealing because it naturally captures complex, nonlinear relationships that are useful for predicting _y_ , and random forests themselves are usually fast and straightforward to fit. 

We expect there are other interesting applications or extensions of our approach. For example, to estimate conditional average treatment effects, we could design a method based on the _R-learner_ (and R-lasso) Nie and Wager [2021]. The R-learner orthogonalizes treatment effect estimation by first regressing out nuisance functions for the average treatment effect and propensity scores, then fitting a penalized regression for the heterogeneous treatment effect. We could use the attention mechanism to model the heterogeneous treatment effect to localize or personalize the estimation. 

Our approach has limitations. For large datasets, fitting a separate model for every test point may seem computationally prohibitive. However, each local model is simply a weighted version of the base learner and is typically fast to fit. Moreover, because the test models are independent, they can be fit in parallel across CPU nodes. Further, the computational cost is on par with leave-one-out crossvalidation, where a separate model is fit for each _train_ observation. We also note that interpretability can become challenging when many subgroups are discovered, and we have not developed formal 

20 

inference procedures (confidence intervals or hypothesis tests for the fitted coefficients), which is an important direction for future work. 

**Acknowledgements** . We would like to thank Ryan Tibshirani, Aaron Niskin, Isaac Mao, Caroline Kimmel, Trevor Hastie, Louis Abraham and Samet Oymak for helpful comments. R.T. was supported by the NIH (5R01EB001988-16) and the NSF (19DMS1208164). 

## **References** 

- S Banerjee, RN Zare, RJ Tibshirani, CA Kunder, R. Nolley R, Fan, JD Brooks, and GA Sonn. Diagnosis of prostate cancer by desorption electrospray ionization mass spectrometric imaging of small metabolites and lipids. _Proc Natl Acad Sci USA_ , 114:3334–3339, 2017. 

- Alina Beygelzimer, Sham Kakadet, John Langford, Sunil Arya, David Mount, and Shengqiao Li. _FNN: Fast Nearest Neighbor Search Algorithms and Applications_ , 2024. URL `https://CRAN.R-project. org/package=FNN` . R package version 1.1.4.1. 

- Jacob Bien and Rob Tibshirani. _protoclust: Hierarchical Clustering with Prototypes_ , 2022. URL `https://CRAN.R-project.org/package=protoclust` . R package version 1.6.4. 

- Jacob Bien and Robert Tibshirani. Hierarchical clustering with prototypes via minimax linkage. _Journal of the American Statistical Association_ , 106(495):1075–1084, 2011. 

- Tianqi Chen, Tong He, Michael Benesty, Vadim Khotilovich, Yuan Tang, Hyunsu Cho, Kailong Chen, Rory Mitchell, Ignacio Cano, Tianyi Zhou, Mu Li, Junyuan Xie, Min Lin, Yifeng Geng, Yutian Li, Jiaming Yuan, and David Cortes. _xgboost: Extreme Gradient Boosting_ , 2025. URL `https: //github.com/dmlc/xgboost` . R package version 3.1.1.1. 

- William S Cleveland and Susan J Devlin. Locally weighted regression: an approach to regression analysis by local fitting. _Journal of the American statistical association_ , 83(403):596–610, 1988. 

- Gilad Deutch, Nadav Magar, Tomer Bar Natan, and Guy Dar. In-context learning and gradient descent revisited. _arXiv preprint arXiv:2311.07772_ , 2023. 

- Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. A survey on in-context learning. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , pages 1107–1128, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.64. URL `https://aclanthology.org/2024.emnlp-main.64/` . 

- Jerome Friedman, Trevor Hastie, and Robert Tibshirani. Regularization paths for generalized linear models via coordinate descent. _Journal of Statistical Software_ , 33(1):1–22, 2010. doi: 10.18637/jss. v033.i01. 

- Noah Hollmann, Samuel M¨uller, Lennart Purucker, Arjun Krishnakumar, Max K¨orfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. _Nature_ , 637(8045):319–326, 2025. 

- Rob Hyndman. _fpp3: Data for ”Forecasting: Principles and Practice” (3rd Edition)_ , 2025. URL `https://CRAN.R-project.org/package=fpp3` . R package version 1.0.2. 

- Markelle Kelly, Rachel Longjohn, and Kolby Nottingham. The uci machine learning repository. `https: //archive.ics.uci.edu` , 2025. 

- Elizbar A Nadaraya. On estimating regression. _Theory of Probability & Its Applications_ , 9(1):141–142, 1964. 

- Xinkun Nie and Stefan Wager. Quasi-oracle estimation of heterogeneous treatment effects. _Biometrika_ , 108(2):299–319, 2021. 

21 

- Scott Powers, Trevor Hastie, and Robert Tibshirani. Customized training with an application to mass spectrometric imaging of cancer tissue. _The annals of applied statistics_ , 9(4):1709, 2016. 

- Ruifeng Ren and Yong Liu. Towards understanding how transformers learn in-context through a representation learning lens. _Advances in Neural Information Processing Systems_ , 37:892–933, 2024. 

- Yu Shi, Guolin Ke, Damien Soukhavong, James Lamb, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu, Nikita Titov, and David Cortes. _lightgbm: Light Gradient Boosting Machine_ , 2025. URL `https://CRAN.R-project.org/package=lightgbm` . R package version 4.6.0. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, �Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, Jo˜ao Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In _International Conference on Machine Learning_ , pages 35151–35174. PMLR, 2023. 

- Geoffrey S Watson. Smooth regression analysis. _Sankhy¯a: The Indian Journal of Statistics, Series A_ , pages 359–372, 1964. 

- Marvin N. Wright and Andreas Ziegler. ranger: A fast implementation of random forests for high dimensional data in C++ and R. _Journal of Statistical Software_ , 77(1):1–17, 2017. doi: 10.18637/ jss.v077.i01. 

- Peng Zhao and Bin Yu. On model selection consistency of lasso. _The Journal of Machine Learning Research_ , 7:2541–2563, 2006. 

22 

## **A Theoretical comparison: lasso and attention lasso** 

### **Problem setup** 

We consider a mixture-of-models setting where data are generated from two distinct linear models, and cluster membership is unobserved. 

We assume that the data are generated as _yi_ = **_x_**<sup>_⊤_</sup> _i_<sup>**_β_**</sup><sup>_Z_</sup> _i_<sup>+</sup><sup>_εi_where:</sup> 

- _Zi ∈{_ 1 _,_ 2 _}_ is the (unobserved) cluster membership with _P_ ( _Zi_ = _k_ ) = _πk_ for _k ∈{_ 1 _,_ 2 _}_ and _π_ 1 + _π_ 2 = 1. 

- _εi ∼ N_ (0 _, σ_<sup>2</sup> ) are i.i.d. noise terms independent of **_x_** _i_ and _Zi_ . 

- **_β_** 1 _,_ **_β_** 2 _∈_ R<sup>_p_</sup> are the true coefficient vectors with _∥_ **_β_** 1 _−_ **_β_** 2 _∥_ 2 _≥ δ >_ 0 for some constant _δ_ . 

- Both **_β_** 1 and **_β_** 2 are _s_ -sparse (at most _s_ non-zero entries). 

We also assume that the design matrix **_X_** _∈_ R<sup>_n×p_</sup> has rows **_x_** _i_ that are i.i.d. draws. Let **Σ** _k_ = _E_ [ **_x_** _i_ **_x_**<sup>_⊤_</sup> _i_<sup>_|Zi_=</sup><sup>_k_]for</sup><sup>_k∈{_1</sup><sup>_,_2</sup><sup>_}_,andassumeboth</sup><sup>**Σ**1and</sup><sup>**Σ**2arepositivedefinite.</sup> 

### **Lasso MSE in this setting** 

The lasso solves: 



We first characterize the mean squared error over the mixture distribution. 

**Lemma A.1** (Minimizer of mean squared error over the full population) **.** _Define the mean squared error (MSE) over the mixture distribution as_ 



_where_ ( **_x_** _, Z_ ) _∼ P_ ( _X, Z_ ) _with y_ = **_x_**<sup>_⊤_</sup> **_β_** _Z_ + _ε. Then:_ 



_The unpenalized minimizer is given by:_ 



Equation 16 is a weighted combination of **_β_** 1 and **_β_** 2, where the weights depend on both the cluster proportions _πk_ and the covariance structures **Σ** _k_ . Importantly, **_β_**<sup>_∗_</sup> = **_β_** 1 or **_β_** 2 (unless _πk_ = 0 or **_β_** 1 = **_β_** 2). 

_Proof._ Expanding the MSE: 



For each cluster, we compute: 



23 

Therefore: 



The minimizer satisfies: 



Solving for **_β_**<sup>_∗_</sup> yields the stated result. 

**Theorem A.2** (Irreducible Bias of Standard Lasso) **.** _Under our assumed model, as n →∞ with λ ∼_ �log _p/n, the standard lasso estimator satisfies:_ 



_where_ **_β_**<sup>_∗_</sup> _is given by Lemma A.1. Furthermore, for test points from cluster 1:_ 



_Assuming_ **Σ** 1 = **Σ** 2 _, we have ∥_ **_β_**<sup>_∗_</sup> _−_ **_β_** 1 _∥_ 2 _≥ c_ 1 _δ for some constant c_ 1 _>_ 0 _, and the expected squared prediction error satisfies:_ 



_for some constant c_ 2 _>_ 0 _._ 

_Proof._ Standard results on lasso consistency (e.g., Zhao and Yu [2006]) show that for _λ ∼_ �log _p/n_ , the penalty term vanishes relative to the empirical risk, and **_β_**<sup>ˆ</sup> lasso _→_ **_β_**<sup>_∗_</sup> in probability. To compute the bias, we have: 



Since _∥_ **_β_** 1 _−_ **_β_** 2 _∥_ 2 _≥ δ >_ 0 and both **Σ** 1 _,_ **Σ** 2 are positive definite, under standard conditions we have _∥_ **_β_**<sup>_∗_</sup> _−_ **_β_** 1 _∥_ 2 _≥ c_ 1 _δ_ for some _c_ 1 _>_ 0. 

For the prediction error, with **_x_**<sup>_∗_</sup> _∼ P_ ( _X|Z_ = 1) and assuming **Σ** 1 = **Σ** 2 = **Σ** for simplicity: 



for some constant _c_ 2 _>_ 0. 

_Remark_ A.3 _._ Theorem A.2 establishes that lasso cannot eliminate bias regardless of sample size. The estimator converges to a weighted combination of **_β_** 1 and **_β_** 2 rather than to the cluster-specific **_β_** 1. 

24 

### **Results for attention lasso in this setting** 

We consider the attention lasso with ridge weights. For a test point **_x_**<sup>_∗_</sup> _∈_ R<sup>_p_</sup> , the attention weight between **_x_**<sup>_∗_</sup> and training observation **_x_** _i_ is defined as: 



and **_D_** ( **_β_**<sup>ˆ</sup> ridge) = Diag( _|_ **_β_**<sup>ˆ</sup> ridge _|_ ) is a diagonal matrix with entries given by the absolute values of ridge regression coefficients fit to the full training data. 

We require three additional assumptions for successful cluster separation: 

**Assumption A.4** (Ridge-weighted cluster separability) **.** The clusters are separable with respect to the ridge-weighted metric. Specifically, for a point from cluster _k_ , the ridge-weighted similarity to its own cluster mean exceeds the similarity to the other cluster mean: 



for _k, ℓ ∈{_ 1 _,_ 2 _}_ with _k_ = _ℓ_ , where **_µ_** _k_ = _E_ [ **_x_** _|Z_ = _k_ ]. Additionally, the clusters have equal covariance: **Σ** 1 = **Σ** 2. 

**Proposition A.5** (Cluster separation via attention) **.** _Under Assumption A.4, for a test point_ **_x_**<sup>_∗_</sup> _∼ P_ ( _X|Z_ = 1) _:_ 



_Proof._ We condition on (or treat as fixed) **_β_**<sup>ˆ</sup> ridge throughout. For a test point **_x_**<sup>_∗_</sup> and training point **_x_** _i_ , the similarity score is _si_ = **_x_**<sup>_∗⊤_</sup> **_D_** ( **_β_**<sup>ˆ</sup> ridge) **_x_** _i_ . 

The expected similarity score to a training point from cluster _k_ is: 



where **_µ_** _k_ = _E_ [ **_x_** _|Z_ = _k_ ] is the mean vector for cluster _k_ . 

For a test point from cluster 1, taking expectation over **_x_**<sup>_∗_</sup> _∼ P_ ( _X|Z_ = 1) where _E_ [ **_x_**<sup>_∗_</sup> ] = **_µ_** 1: 



By Assumption A.4, **_µ_**<sup>_⊤_</sup> 1<sup>**_D_**( ˆ</sup><sup>**_β_**ridge)</sup><sup>**_µ_**1</sup><sup>_>_</sup><sup>**_µ_**</sup><sup>_⊤_</sup> 1<sup>**_D_**( ˆ</sup><sup>**_β_**ridge)</sup><sup>**_µ_**2.Therefore:</sup> 



This shows that on average, training points from cluster 1 have higher similarity scores than cluster 2 training points, for test points drawn from cluster 1. 

Next we show that, under appropriate assumptions, attention lasso coefficients more closely match the individual cluster coefficients **_β_** 1 and **_β_** 2 than the single lasso model fit to the full dataset. 

**Definition A.6** (Attention lasso) **.** For a test point **_x_**<sup>_∗_</sup> from cluster 1, the attention lasso estimator is: 



where _wi_ ( **_x_**<sup>_∗_</sup> ) are the attention weights from Equation 19. 

**Lemma A.7** (Weighted MSE minimizer) **.** _Let Wk_ =<sup>�</sup> _i∈Ik_<sup>_wi_(</sup><sup>**_x_**</sup><sup>_∗_)</sup><sup>_denotethetotalweightoncluster_</sup> _k training points, where Ik_ = _{i_ : _Zi_ = _k}. The minimizer of the weighted MSE is:_ 



25 

_Proof._ Taking the expectation of the weighted attention lasso objective and splitting by cluster: 





_Consequently,_ **_β_** _att_<sup>_∗iscloserto_</sup><sup>**_β_**1</sup><sup>_than_</sup><sup>**_β_**</sup><sup>_∗fromLemmaA.1._</sup> 

_Proof._ From Proposition A.5, the expected similarity scores satisfy _E_ [ _si|Zi_ = 1 _,_ **_x_**<sup>_∗_</sup> ] _> E_ [ _si|Zi_ = 2 _,_ **_x_**<sup>_∗_</sup> ]. Under the equal covariance condition **Σ** 1 = **Σ** 2 from Assumption A.4, the variance of _si_ given **_x_**<sup>_∗_</sup> is the same for both clusters: 



where **Σ** = **Σ** 1 = **Σ** 2. Since the similarity scores have equal variance but different means, we have _E_ [exp( _si_ ) _|Zi_ = 1 _,_ **_x_**<sup>_∗_</sup> ] _> E_ [exp( _si_ ) _|Zi_ = 2 _,_ **_x_**<sup>_∗_</sup> ], which implies _E_ [ _wi_ ( **_x_**<sup>_∗_</sup> ) _|Zi_ = 1] _> E_ [ _wi_ ( **_x_**<sup>_∗_</sup> ) _|Zi_ = 2]. 

Since without attention each point receives uniform weight 1 _/n_ , and with attention same-cluster points receive higher weights: 



Similarly, _w_ ¯2 _<_ 1 _/n_ = _⇒ W_ 2 _< π_ 2. Since weights sum to 1, we have _W_ 1 + _W_ 2 = 1. The weighted combination in **_β_** att<sup>_∗_placesweight</sup><sup>_W_1</sup><sup>_>π_1on</sup><sup>**_β_**1comparedtoweight</sup><sup>_π_1in</sup><sup>**_β_**</sup><sup>_∗_,</sup> bringing the estimator closer to the true cluster-1 parameter. 

_Remark_ A.9 _._ In the ideal case where attention perfectly separates clusters ( _wi ≈_ 1 _/n_ 1 for _i ∈ I_ 1 and _wi ≈_ 0 for _i ∈ I_ 2), we would have **_β_** att<sup>_∗≈_</sup><sup>**_β_**1,eliminatingthebiasentirely.</sup> 

### **Main result: prediction error comparison** 

**Theorem A.10** (Improved prediction via attention) **.** _Under Assumption A.4, assume_ **Σ** 1 = **Σ** 2 = **Σ** _and consider test points_ **_x_**<sup>_∗_</sup> _∼ P_ ( _X|Z_ = 1) _. With λ ∼_ �log _p/n:_ 

##### **_(i) Lasso:_** 



_The first term is the squared bias from model misspecification, which satisfies π_ 2<sup>2(</sup><sup>**_β_**2</sup><sup>_−_</sup><sup>**_β_**1)</sup><sup>_⊤_</sup><sup>**Σ**(</sup><sup>**_β_**2</sup><sup>_−_</sup><sup>**_β_**1)</sup><sup>_≥cδ_2</sup> _for some constant c >_ 0 _by assumptions made in our data generation. The second term is the variance from estimation error._ 

**_(ii) Attention lasso:_** _Under the conditions of Proposition A.5:_ 



26 

_where W_ 2 _< π_ 2 _is the total weight on cluster 2 training points._ **_(iii) Comparison:_** _As n →∞:_ 



_That is, attention lasso reduces the asymptotic prediction error by a factor of_ ( _W_ 2 _/π_ 2)<sup>2</sup> _._ 

_Proof._ **Part (i):** From Theorem A.2, **_β_**<sup>ˆ</sup> lasso = **_β_**<sup>_∗_</sup> + _Op s_ lo _n_ <u>g</u> _<u>p</u>_ where **_β_**<sup>_∗_</sup> _−_ **_β_** 1 = _π_ 2( **_β_** 2 _−_ **_β_** 1) under �� � equal covariances. For **_x_**<sup>_∗_</sup> _∼ P_ ( _X|Z_ = 1): 



Taking expectations: 

**Part (ii):** From Lemma A.7 with **Σ** 1 = **Σ** 2 = **Σ** : 

**_β_** att<sup>_∗_= (</sup><sup>_W_1</sup><sup>**Σ**+</sup><sup>_W_2</sup><sup>**Σ**)</sup><sup>_−_1(</sup><sup>_W_1</sup><sup>**Σ**</sup><sup>**_β_**1+</sup><sup>_W_2</sup><sup>**Σ**</sup><sup>**_β_**2) =</sup><sup>_W_1</sup><sup>**_β_**1+</sup><sup>_W_2</sup><sup>**_β_**2</sup> 

since _W_ 1 + _W_ 2 = 1. The bias for cluster 1 test points is: 



By the same argument as Part (i), replacing _π_ 2 with _W_ 2: 



**Part (iii):** From Proposition A.8, _W_ 2 _< π_ 2. As _n →∞_ , the variance terms vanish and: 



_Remark_ A.11 _._ Theorem A.10 demonstrates that attention lasso reduces the irreducible bias of standard lasso by a factor of ( _W_ 2 _/π_ 2)<sup>2</sup> _<_ 1. The improvement comes from upweighting training examples from the same cluster as the test point. The bias does not vanish entirely because attention weights are soft: even with perfect cluster separation in the ridge-weighted feature space, the softmax still assigns positive weight to the wrong cluster. This can be adjusted using a temperature parameter that pushes from a soft clustering to a hard clustering. 

27 

## **B Comparison of simplified softmax attention and Gaussian kernel regression weights** 

Here, we discuss similarities and differences between softmax attention and Gaussian kernel regression weighting. Let **_X_** _∈_ R<sup>_n×p_</sup> have rows **_x_** 1 _, . . . ,_ **_x_** _n_ . 

Importantly we discuss here a very simplified form of attention weights, namely softmax � **_XX_**<sup>_T_�</sup> . This is simpler than even our simplest approach to attention, which weights the inner product by a diagonal matrix using the absolute value of the ridge coefficients and therefore incorporates information from the relationship **_y_** _|_ **_X_** . 

### **1. Simplified attention weights** 

We first form the score matrix 



and apply a row-wise softmax (optionally with temperature parameter _τ_ ) to **_S_** to obtain weights **_w_** such that: 



The weight vector **_w_** _i_ defines a probability distribution over all _j_ . 

### **2. Gaussian kernel regression weights** 

In Nadaraya–Watson or Gaussian kernel regression with bandwidth _σ_ , we use 

and define normalized weights 



Expanding the squared distance: 



Substitute this into the Gaussian kernel: 



For a fixed _i_ , the factor exp � _−_<sup>_<u>∥</u>_</sup> (2<sup>**_x_**</sup> _σ_<sup>_i∥_22</sup> ) � is constant in _j_ and cancels during normalization, giving (up to normalization) 



### **Summary** 

Gaussian kernel weights and softmax attention weights are the same when (1) the temperature and bandwidth satisfy _τ_ = _σ_<sup>2</sup> , and (2) all data points have equal norm, _∥_ **x** _j∥_ = _c_ . In this case the factor 



28 

is constant in _j_ and cancels in the normalization, so that 



This situation arises, for example, when the rows are _ℓ_ 2-normalized so that _∥_ **x** _j∥_ = 1 for all _j_ . 

However, in general, the two constructions are slightly different. Gaussian weights use squared Euclidean distance, 



and are invariant under rotations and translations of the input space. Softmax attention depends on inner products, 



and is therefore not translation invariant and more sensitive to the individual norms _∥_ **x** _i∥_ and _∥_ **x** _j∥_ , allowing large-norm vectors to dominate the weights. Additionally, Gaussian weights include an additional global factor 



which penalizes high-norm points uniformly across all queries _i_ , whereas attention weights lack such a term and only reweight via relative inner products. Finally, the bandwidth _σ_ and temperature _τ_ play analogous roles—both control how concentrated the weights are—but they do so in different geometries: _σ_ sets the scale of decay in distance space, while _τ_ controls the sharpness of the distribution in inner-product space. 

In summary, when all rows are normalized and _τ_ = _σ_<sup>2</sup> , the two weighting schemes are the same. Away from this regime, Gaussian kernel regression remains strictly radial in distance, whereas softmax attention depends on vector norms and loses translation invariance. 

29 

