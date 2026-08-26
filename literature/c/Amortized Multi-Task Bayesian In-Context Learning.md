**Amortized In-Context Bayesian Posterior Estimation** 

**Sarthak Mittal**<sup>1 2</sup> **Niels Leif Bracher**<sup>3</sup> **Guillaume Lajoie**<sup>1 2</sup> **Priyank Jaini**<sup>4</sup> **Marcus Brubaker**<sup>4 5 6</sup> 

## **Abstract** 

Bayesian inference provides a natural way of incorporating prior beliefs and assigning a probability measure to the space of hypotheses. Current solutions rely on iterative routines like Markov Chain Monte Carlo (MCMC) sampling and Variational Inference (VI), which need to be re-run whenever new observations are available. Amortization, through conditional estimation, is a viable strategy to alleviate such difficulties and has been the guiding principle behind simulation-based inference, neural processes and in-context methods using pre-trained models. In this work, we conduct a thorough comparative analysis of amortized in-context Bayesian posterior estimation methods from the lens of different optimization objectives and architectural choices. Such methods train an amortized estimator to perform posterior parameter inference by conditioning on a set of data examples passed as context to a sequence model such as a transformer. In contrast to language models, we leverage permutation invariant architectures as the true posterior is invariant to the ordering of context examples. Our empirical study includes generalization to outof-distribution tasks, cases where the assumed underlying model is misspecified, and transfer from simulated to real problems. Subsequently, it highlights the superiority of the reverse KL estimator for predictive problems, especially when combined with the transformer architecture and normalizing flows. 

## **1. Introduction** 

Bayesian analysis of data has become increasingly popular and is widely used in numerous scientific disciplines. In politics, predictive models based on public polling and other factors play a crucial role in the discourse around the state of a campaign. Throughout the COVID-19 pandemic, models that estimate the infectiousness of the virus, the effi- 

> 1Universite de Montreal´ 2Mila 3Rensselaer Polytechnic Institute 4Google DeepMind 5York University 6Vector Institute. Correspondence to: Sarthak Mittal _<_ sarthmit@gmail.com _>_ . 

Preprint. 

cacy of public health measures, and the future course of the pandemic became critical to government planning and the public’s understanding of the pandemic (Cooper et al., 2020). In cryogenic electron microscopy (cryo-EM), the posterior over an unknown 3D atomic-resolution molecular structure is explored given image observations (Glaeser et al., 2021). 

While recent years have made such methods more accessible (Bingham et al., 2019; Carpenter et al., 2017; Strumbelj<sup>ˇ</sup> et al., 2023), they still remain computationally burdensome. Further, in practical contexts where new observations are continuously available, the analysis must be re-run every time new data becomes available, e.g., when new case counts become available, previous measurements are corrected, or when applied to different geographic regions. As a result practitioners adopt approximations (Welling & Teh, 2011; Gelfand, 2000; Brooks, 1998), simplify their models (Hoffman et al., 2013; Blei et al., 2017) or reduce the frequency with which they perform their analyses. 

A common thread is that the probabilistic model defining the relationship between its parameters and the observations is fixed. Poll aggregation models use hierarchical time series models (Athanasopoulos et al., 2023; Chen et al., 2023), infectious diseases are studied using variants on compartment models (Tang et al., 2020), and cryo-EM uses a linear image formation model (Glaeser et al., 2021). This makes these applications ideal candidates for amortized inference (Morris, 2013; Paige & Wood, 2016; Kingma & Welling, 2013; Rezende et al., 2014; Stuhlm¨uller et al., 2013). 

Multiple approaches leverage neural networks to learn functions that map an observed _dataset_ directly to a posterior distribution (Garnelo et al., 2018b; Cranmer et al., 2020) or model the posterior predictive directly (Garnelo et al., 2018a; Muller¨ et al., 2021; Garg et al., 2022; Hollmann et al., 2022). They sidestep the need for iterative procedures, e.g., Markov chain Monte Carlo (MCMC) sampling (Gelfand, 2000; Hoffman et al., 2014) or standard variational inference (VI) and efficiently handle permutation invariance stemming from _iid_ observations using Transformers and DeepSets (Zaheer et al., 2017; Vaswani et al., 2017; Lee et al., 2019). If learned properly, this mapping allows generalization to new datasets passed in context in zero-shot. 

However, analysis into evaluating different in-context posterior estimation objectives is currently lacking. The goal 

1 

**Amortized In-Context Bayesian Posterior Estimation** 

of such estimators is to model the posterior distribution by leveraging observations in context as opposed to invoking iterative estimation procedures again from scratch. We provide a rigorous analysis into different training objectives, i.e. forward and reverse KL objectives, where the former is equivalent to neural posterior estimation in simulation-based inference (Cranmer et al., 2020) and the latter has connections to neural processes (Garnelo et al., 2018b). However, NP only model the posterior over some unstructured latent variable (akin to Kingma & Welling (2013); Rezende et al. (2014)) with the objective being a proxy to maximum likelihood while we are interested in a fully Bayesian treatment of all parameters defining the likelihood. 

Our benchmark considers a wide variety of probabilistic models and evaluates different design choices in inferring the posterior over their parameters. We look at different permutation invariant architectures, parametrizations for the approximate density, as well as training objectives. Our evaluation criteria tests for both in-distribution (ID) and out-of-distribution (OoD) generalization, and relies on a simple masking procedure to amortize posterior estimation over datasets with a variable number of features, inching closer towards a generalist in-context Bayesian learner _−_ as evidenced by its generalization capabilities on real-world tasks zero-shot through only pre-training on synthetic data. 

Generally, real-world datasets do not exactly follow standard models, e.g., while practitioners often rely on linear models, data rarely follows them exactly. We further evaluate the estimators on tasks where the assumed probabilistic model is incorrect (misspecification), or where we only have access to samples but not underlying parameters, which is a common paradigm in most machine learning tasks. Our detailed experiments provide clear insights into the architectural choices that lead to better amortized posterior estimation, through the lens of both predictive and sample-based metrics. Our contributions include 

- Providing a general framework for in-context Bayesian posterior estimation with different training objectives. 

- Benchmarking various design choices like architectural backbones, parametrizations of approximate density and training objectives through extensive ablations. 

- Evaluating the ability of estimators to generalize OoD when the modeling assumption is different from the underlying true model class (misspecification), especially to real-world tasks when trained only on synthetic data. 

## **2. Background** 

We first cover some of the important preliminaries below. 

**Bayesian Inference** . Let **_x_** _∈_ R<sup>_d_</sup> denote the outcome of an experiment observed through a set of independent and identically distributed ( _iid_ ) samples _D_ := _{_ **_x_** 1 _, ...,_ **_x_** _N } ⊆_ R<sup>_d_</sup> . Given these observations, we are interested in either 

quantifying the certainty of or generating potential future observations **_x_** _∗_ . Bayesian Inference provides a natural methodology of quantifying _p_ ( **_x_** _∗|D_ ) by prescribing a space of hypotheses **_θ_** _∈_ R<sup>_k_</sup> and a _prior_ belief _p_ ( **_θ_** ) over it. These hypotheses define the _likelihood_ of observing an outcome, i.e., _p_ ( **_x_** _|_ **_θ_** ). The likelihood and prior are then combined through Bayes rule to infer the _posterior p_ ( **_θ_** _|D_ ), through which the quantity of interest can then be easily expressed as 



This poses two challenges: (a) the _posterior_ , often a quantity of interest in itself, is not known, and (b) the integration can be intractable which is often resolved through Monte Carlo estimation 



where **_θ_**<sup>(</sup><sup>_m_)</sup> _∼ p_ ( **_θ_** _|D_ ). The quantity _p_ ( **_θ_** _|D_ ) can be obtained through an application of Bayes rule 



Given the form of the _likelihood_ and _prior_ , the above distribution is often difficult to sample from, especially with the added complexity of the marginal _p_ ( _D_ ) = � **_θ_**<sup>_p_(</sup><sup>_D|_</sup><sup>**_θ_**)</sup><sup>_p_(</sup><sup>**_θ_**)</sup> being intractable. Additionally, the posterior itself is often of interest on its own, especially in cases where **_θ_** is interpretable, _e.g._ if we model the bias of a coin based on multiple tosses. We refer the readers to Bishop & Nasrabadi (2006) for additional applications of Bayesian Inference. 

**Approximate Bayesian Inference** . To bypass the intractability of the posterior distribution, or at least the difficulty to sample from it, approximate methods are used. 

Sampling based methods provide ways of sampling from the true posterior distribution based on easy access to an unnormalized density function, e.g. rejection sampling. More advanced methods like MCMC construct a chain of updates **_θ_** 1 _,_ **_θ_** 2 _, . . ._ such that asymptotically the samples converge to samples from the true posterior. Such sampling methods rely on transition kernels _T_ ( **_θ_** _t_ +1 _|_ **_θ_** _t_ ) and often some acceptance criteria _A_ ( **_θ_** _t_ +1 _,_ **_θ_** _t_ ), a key example of which is the Metropolis-Hastings algorithm. We refer the readers to (Hoffman et al., 2014; Welling & Teh, 2011) for a detailed analysis into different MCMC methods like Langevin and Hamiltonian Monte Carlo which rely on gradient of the log density as additional signal for better convergence. 

In contrast, another class of methods approximate the true posterior with a parametric family _qφ_ ( **_θ_** ) and convert the estimation problem into the following optimization problem 



2 

**Amortized In-Context Bayesian Posterior Estimation** 





<!-- Start of picture text -->
Mean of Gaussian Linear Regression Nonlinear Regression Gaussian Mixture Linear Classification Nonlinear Classification<br><!-- End of picture text -->

_Figure 1._ **Amortized Bayesian Posterior Estimation:** Illustration of predictions from the reverse KL in-context estimator. Model predictions, true predictions and sample points are shown in red, black and blue respectively. Additionally for classification, we label sample points with their ground-truth class, and draw the decision boundary according to the model. 

where D is a notion of divergence between two distributions. Once the optimal parameters _φ_<sup>_∗_</sup> are obtained, _qφ∗_ can be used to substitute the true posterior wherever needed. The above optimization procedure finds a member in the family of variational distributions _{qφ}φ_ that is closest to the true posterior under D. An example of this is Variational Inference, where the reverse KL divergence is used 



which is equivalent to optimizing the well known Evidence Lower-Bound (ELBO) (Gelman et al., 2013) 



Another example is Expected Propagation (EP; Minka, 2013) which relies on the forward KL divergence 



Once the optimal parameters _φ_<sup>_∗_</sup> are obtained, the _posterior predictive_ distribution _p_ ( **_x_** _∗|D_ ) can be approximated as 



The family of distributions _qφ_ is chosen such that it is easy to sample from. Typical choices include independent multivariate Gaussian distribution (mean-field approximation) or normalizing flows (Rezende & Mohamed, 2015; Papamakarios et al., 2021; Ardizzone et al., 2018-2022). 

**Estimators and Amortization** . A core benefit of deep learning is its ability to generalize well. Amortized approaches leverage this ability by training conditional models to solve a family of problems in an efficient and scalable manner, as opposed to independently solving each problem. For _e.g._ , the encoder in Variational Autoencoders (VAEs; Kingma & Welling, 2013; Rezende et al., 2014) is tasked with estimating the posterior distribution _p_ ( **_z_** _|_ **_x_** _i_ ) for each **_x_** _i ∈D_ . Here, _p_ ( **_z_** ) is the standard normal prior and _p_ ( **_x_** _i|_ **_z_** ) is the decoder defining the trainable likelihood. Instead of separate optimization problems _qφ_<sup>_∗_</sup> _i_<sup>(</sup><sup>**_z_**) for each posterior</sup> _p_ ( **_z_** _|_ **_x_** _i_ ), VAEs rely on amortization to train a shared network _qφ_ ( **_z_** _|_ **_x_** ), where _φ_ now represents the parameters of a neural network and takes **_x_** explicitly as input, allowing zero-shot generalization to new **_x_** _∗_ at inference. 

Amortization plays a key role in multiple domains of machine learning, beyond VAEs. For _e.g._ , score-based diffusion models (Song et al., 2020) amortize training of a timeconditioned score model, while Neural Processes (NPs) (Garnelo et al., 2018a;b) and neural posterior estimation in Simulation-Based Inference (SBI) (Cranmer et al., 2020) amortize dataset-conditioned VAE-styled encoders under DR-KL and DF-KL respectively. Even further, in-context learning (ICL) (Von Oswald et al., 2023; Muller et al.¨ , 2021) can also be seen as amortizing the posterior predictive distribution _p_ ( _y∗|_ **_x_** _∗, D_ ) based on context examples _D_ as an emergent phenomena owing to the shared modality of language tying different tasks, where _y∗_ defines the label. 

We refer to Appendix A for details about related work. 

## **3. Posterior Estimation from Data in Context** 

As described earlier, standard in-context approaches are predominantly concerned with prediction and model the posterior predictive directly, taking _D_ as input. However, they can be tweaked to perform posterior estimation instead. We discuss ways to train such an in-context estimator and showcase its connections to existing amortization methods. 

Given any modeling assumption defined via a probabilistic model _p_ ( _·|_ **_θ_** ) with parameters **_θ_** , we are interested in estimating the full Bayesian posterior over the parameters _p_ ( **_θ_** _|D_ ) after obtaining some observations _D_ , in a manner that allows fast and scalable approximation. Equations (5) and (7) showcase two different methodologies of performing posterior estimation, however, both the methods train a new _qφ_ every time new observations _D_ are obtained. However, such methods can be easily amortized by leveraging in-context learning with a goal towards posterior estimation instead of prediction, i.e. training a model to approximate the posterior distribution based on in-context examples. Mathematically, this is obtained by considering an approximate density _qφ_ ( _·|D_ )<sup>1</sup> which is explicitly conditioned on the set of observations _D_ and trained over multiple such sets 



where _χ_ denotes some distribution over observations _D_ . 

If the measure of divergence is the forward KL D _F_ -KL, it leads to the neural posterior estimation methodology of 

> 1We term this conditional model _in-context posterior estimator_ . 

3 

**Amortized In-Context Bayesian Posterior Estimation** 

||||_L_|2 _Loss_(_↓_)|||_A_|_ccuracy_(_↑_|)|
|---|---|---|---|---|---|---|---|---|---|
|**Objective**|_qφ_<br>**Model**|**Gaussian**|**GMM**|**LR**|**NL**|**R**|**LC**|**N**|**LC**|
|||_100D_|_5D 2 cl_|_100D_|_1D_|_25D_|_100D_|_2D_|_25D_|
||-<br>Random|301_._06_±_0.35|5_._00_±_0.04|202_._6_±_0.3|65_._94_±_0.91|831_._6_±_8.7|50_._0_±_0.0|50_._3_±_0.6|50_._0_±_0.3|
|Baseline|-<br>Optimization <br>-<br>Langevin|101_._24_±_0.00<br> 102_._35_±_0.03|0_._42_±_0.00<br>0_._45_±_0.01|25_._1_±_0.0<br>23_._3_±_0.7|0_._36_±_0.00<br>0_._31_±_0.00|104_._0_±_0.1<br>132_._4_±_1.0|70_._3_±_0.0<br>65_._1_±_0.4|96_._9_±_0.0<br>96_._0_±_0.3|77_._9_±_0.0<br>73_._2_±_0.3|
||-<br>HMC|102_._41_±_0.03|0_._48_±_0.01|18_._7_±_0.2|0_._37_±_0.00|98_._1_±_0.7|62_._1_±_0.2|91_._8_±_0.2|70_._4_±_0.1|
||GRU|102_._64_±_0.01|2_._43_±_0.03|124_._8_±_0.1|49_._33_±_0.95|671_._6_±_10.5|59_._7_±_0.1|59_._5_±_0.4|56_._9_±_0.3|
|Fwd-KL|n<br>DeepSets|103_._22_±_0.05|2_._44_±_0.04|123_._1_±_1.1|49_._86_±_0.98|684_._9_±_2.6|50_._0_±_0.1|59_._4_±_0.2|56_._8_±_0.2|
||ssia<br>Transformer|102_._78_±_0.00|2_._50_±_0.03|45_._9_±_1.3|49_._68_±_0.94|680_._9_±_5.8|63_._0_±_0.1|59_._6_±_0.4|57_._1_±_0.4|
||Gau<br>GRU|102_._51_±_0.01|0_._47_±_0.01|60_._2_±_0.9|0_._43_±_0.00|106_._0_±_0.6|63_._5_±_0.3|92_._4_±_0.2|72_._5_±_0.0|
|Fwd-KL|DeepSets|102_._60_±_0.04|0_._50_±_0.02|62_._8_±_0.6|0_._43_±_0.00|125_._9_±_0.8|60_._9_±_0.3|92_._5_±_0.1|59_._8_±_0.3|
||Transformer|102_._54_±_0.03|0_._49_±_0.02|28_._7_±_0.3|0_._42_±_0.01|102_._3_±_1.8|68_._2_±_0.0|92_._6_±_0.4|75_._2_±_0.1|
||GRU|102_._66_±_0.02|0_._67_±_0.09|119_._1_±_0.2|15_._78_±_0.21|539_._0_±_4.3|59_._9_±_0.2|76_._9_±_0.3|58_._3_±_0.0|
|Fwd-KL|ow<br>DeepSets <br>Transformer|103_._34_±_0.03<br> 102_._77_±_0.02|0_._65_±_0.08<br>0_._62_±_0.07|125_._7_±_3.7 <br>43_._3_±_2.7|15_._05_±_0.12<br>16_._11_±_0.31|548_._5_±_3.3<br>539_._3_±_4.3|50_._1_±_0.0<br>64_._3_±_0.1|72_._3_±_1.8<br>77_._3_±_0.2|58_._1_±_0.1<br>58_._3_±_0.1|
||Fl<br>GRU|102_._49_±_0.01|0_._47_±_0.00|61_._3_±_1.0|0_._41_±_0.01|106_._0_±_0.4|64_._7_±_0.2|93_._4_±_0.1|72_._0_±_0.5|
|Rev-KL|DeepSets|102_._67_±_0.05|0_._52_±_0.01|76_._4_±_2.0|0_._40_±_0.00|128_._2_±_1.5|58_._4_±_0.8|93_._3_±_0.2|60_._9_±_0.2|
||Transformer|102_._53_±_0.05|0_._47_±_0.01|29_._4_±_1.6|0_._39_±_0.00|102_._6_±_0.9|68_._7_±_0.1|93_._6_±_0.1|75_._0_±_0.5|



_Table 1._ **Fixed-dimensional In-Context Posterior Estimation** for estimating the mean of a Gaussian (Gaussian), means of a Gaussian mixture (GMM), parameters of (non-)linear regression (NLR/LR) and (non-)linear binary classification (NLC/LC). We ablate over different architectures and density parametrizations and use the expected predictive _L_ 2 loss and accuracy as the metrics. 

simulation-based inference (SBI-NPE) as long as an additional constraint is satisfied, i.e. _χ_ defines sampling from the assumed underlying model _p_ , _i.e._ 



The importance of this constraint is that it leads to a simpler gradient-based optimization procedure as opposed to EP 





where we focus our attention to the change in expectations which is only possible when _D_ is sampled according to _p_ . This removes the requirement of sampling or evaluating the true posterior, a known hurdle with EP methods. 

In contrast, one could also take motivation from VAEs and NP which predominantly work under the reverse KL divergence D _R_ -KL minimization. An amortized in-context learner in this setting can be mathematically formalized as 



It is important to note that unlike forward KL, reverse KL provides the freedom of choosing any arbitrary _χ_ while still maintaining ease in training, i.e. the in-context estimator can be trained on datasets that come from a different distribution than _p_ . Such flexibility is important because we 

often want to generalize well to data whose underlying true probabilistic model is unknown, and thus we often prescribe a likelihood based on our best belief. For _e.g._ , practitioners often model regression problems as linear even when the data might come from a nonlinear process like a Gaussian Process. Reverse KL allows us to train the in-context learner on a steady stream of data even when the likelihood is misspecified, while the forward KL objective in this case will have to be trained on simulated linear data. 

The estimators defined in Equations (12) and (14) rely on a parameterization of _qφ_ ( _·|D_ ), which involves two components: a flexible parametric form of density and an architecture that can be conditioned on entire datasets. In this work, we provide an in-depth analysis into the use of a diagonal Gaussian and discrete-time normalizing flows for the former, and Gated Recurrent Units (GRU), DeepSets and Transformers for the latter. We note that the conditioning architecture should respect permutation invariance of the approximate posterior to the ordering in the observations, which is respected in DeepSets and Transformers but not in GRUs. See Appendix B for architectural choice details. 

Finally, these training procedures naturally introduce a dependency on the dataset generating distribution _χ_ . Since we are working with a known probabilistic model, an obvious choice of _χ_ is to treat this probabilistic model as a black-box simulator and generate samples using ancestral sampling. 

In the following sections, we provide a comparative analysis between the two approaches to in-context posterior estimation, as well as the different architectural choices and 

4 

**Amortized In-Context Bayesian Posterior Estimation** 

||||_L_|2 _Loss_(_↓_)|||_A_|_ccuracy_(_↑_|)|
|---|---|---|---|---|---|---|---|---|---|
|**Objective**|_qφ_<br>**Model**|**Gaussian**|**GMM**|**LR**|**N**|**LR**|**LC**|**N**|**LC**|
|||_100D_|_5D 2 cl_|_100D_|_1D_|_50D_|_100D_|_2D_|_50D_|
||-<br>Random|298_._24_±_0.23|4_._66_±_0.03|200_._8_±_0.6|73_._01_±_0.17|1704_._3_±_9.3|50_._0_±_0.1|50_._0_±_0.3|49_._9_±_0.3|
|Baseline|-<br>Optimization <br>-<br>Langevin <br>-<br>HMC|100_._88_±_0.00<br> 101_._92_±_0.04<br> 102_._01_±_0.01|0_._43_±_0.00<br>0_._44_±_0.00<br>0_._46_±_0.01|20_._1_±_0.0<br>21_._8_±_1.0<br>17_._8_±_0.1|0_._36_±_0.00<br>0_._31_±_0.00<br>0_._38_±_0.01|309_._2_±_0.2<br>N/A<br>303_._9_±_2.5|71_._2_±_0.0<br>65_._5_±_0.5<br>62_._6_±_0.2|96_._8_±_0.0<br>96_._1_±_0.0<br>91_._7_±_0.2|76_._1_±_0.0<br>70_._1_±_0.2<br>68_._0_±_0.4|
||GRU|133_._22_±_0.58|2_._36_±_0.02|139_._4_±_1.0|51_._45_±_0.03|1346_._5_±_6.8|57_._9_±_0.2|59_._6_±_0.2|58_._6_±_0.2|
|Fwd-KL|ssian<br>DeepSets <br>Transformer|129_._69_±_0.74<br> 108_._98_±_0.10|2_._35_±_0.02<br>2_._40_±_0.02|149_._8_±_0.8 <br>64_._3_±_3.7|51_._90_±_1.54 <br>50_._81_±_0.53|1357_._5_±_5.3<br> 1319_._9_±_12.2|50_._8_±_0.1<br> 62_._4_±_0.0|49_._9_±_0.3<br>59_._9_±_0.2|49_._9_±_0.3<br>58_._8_±_0.2|
||Gau<br>GRU|105_._14_±_0.10|0_._46_±_0.01|62_._6_±_0.1|2_._31_±_0.13|316_._3_±_6.2|63_._4_±_0.2|88_._8_±_0.5|68_._4_±_0.3|
|Rev-KL|DeepSets|105_._06_±_0.21|0_._48_±_0.02|64_._1_±_0.2|0_._98_±_0.12|451_._9_±_2.8|61_._3_±_0.1|91_._0_±_0.5|61_._7_±_0.1|
||Transformer|104_._71_±_0.12|0_._47_±_0.01|32_._0_±_0.5|0_._81_±_0.02|278_._3_±_1.1|67_._7_±_0.1|90_._0_±_0.2|73_._7_±_0.3|
||GRU|125_._84_±_1.98|0_._60_±_0.07|138_._3_±_1.0|38_._41_±_0.36|1097_._4_±_9.5|58_._0_±_0.1|61_._2_±_0.8|60_._2_±_0.1|
|Fwd-KL|low<br>DeepSets <br>Transformer|133_._23_±_1.93<br> 108_._48_±_0.16|0_._58_±_0.03<br>0_._59_±_0.08|153_._2_±_0.8 <br>63_._1_±_2.0|43_._31_±_2.06 <br>39_._70_±_0.52|1120_._0_±_5.5<br> 1073_._3_±_1.5|50_._5_±_0.1<br>63_._6_±_0.1|49_._6_±_0.2<br>60_._9_±_0.3|50_._1_±_0.1<br>60_._3_±_0.1|
||F<br>GRU|105_._19_±_0.03|0_._47_±_0.01|71_._3_±_1.3|2_._31_±_0.41|302_._9_±_5.6|63_._4_±_0.1|90_._4_±_0.7|66_._2_±_0.1|
|Rev-KL|DeepSets|105_._09_±_0.06|0_._49_±_0.01|76_._8_±_1.8|0_._83_±_0.02|454_._1_±_10.2|59_._1_±_0.5|89_._1_±_0.3|62_._9_±_0.1|
||Transformer|104_._91_±_0.11|0_._46_±_0.00|33_._1_±_0.3|0_._99_±_0.07|274_._0_±_1.3|68_._1_±_0.2|91_._1_±_0.2|72_._6_±_0.1|



_Table 2._ **Variable-Dimensional In-Context Posterior Estimation** for estimating the mean of a Gaussian (Gaussian), means of a Gaussian mixture model (GMM), (non-)linear regression (NLR/LR) and (non-)linear binary classification (NLC/LC). For each task, a single model is trained to estimate the posterior for a variable number of features; e.g. the same model estimates both _1D_ and _50D_ NLR parameters. 

parametrizations of the density _qφ_ . While this has been independently studied in the SBI and NP framework, a rigorous comparative study between them through the lens of predictive and sample-based metrics has been lacking. Further, we aim to understand the impact of misspecification in such in-context learners, i.e. when real data may not come from the _p_ defined by the likelihood and the prior. To study this rigorously, we further evaluate the in-context estimators on observations coming from different underlying known and unknown processes in Sections 4.3 and 4.4. 

## **4. Experiments** 

To provide a fair and comprehensive evaluation of the different estimators and modeling choices, we consider a variety of well-known probabilistic models encompassing supervised and unsupervised scenarios. In particular, we look at the problem of estimating the Bayesian posterior over the (a) mean of a Gaussian distribution (GM), (b) means of a Gaussian mixture model (GMM), (c) parameters of a (non)linear regression model (NLR/LR), and (d) parameters of a (non-)linear classification model (NLC/LC). We refer the readers to Appendix C for particulars about the probabilistic models, including their likelihoods and priors considered. 

**Baselines** . We consider dataset-specific baselines to compare different amortized in-context posterior estimators with. In particular, we use the prior (Random), perform maximum likelihood estimation using gradient-based optimization (Optimization) as well as an approximate Bayesian inference procedure through Langevin and Hamiltonian based MCMC sampling. Such baselines rely on iterative proce- 

dures and must be run independently for different datasets. 

**Metrics** . We consider two different types of metrics: predictive and sample-based. For the former, we consider _L_ 2 loss and accuracy as applicable, in the following manner 



where _y_ ˆ is the mode of the distribution _p_ ( _·|_ **_x_** _∗,_ **_θ_** ) and METRIC is _L_ 2 loss or accuracy for regression and classification respectively. For unsupervised learning settings, we consider a similar _L_ 2 based metric defining distance from the mean of the Gaussian or the closest mean in the GMM. For sample-based metrics, we leverage the 



where _π_ denotes a joint distribution over ( **_θ_** _q,_ **_θ_** _p_ ) with marginals _qφ_ ( _·|D_ ) and _p_ ( _·|D_ ) respectively. This is called the 2-Wasserstein metric which can be computed with finite samples from each, where we use samples from MCMC as reference for _p_ . We also leverage the symmetric KL divergence as a metric whenever the true posterior is available. 

We refer to Appendices D, F and H for details about the experiments, metrics and additional results respectively. 

### **4.1. Zero-Shot Posterior Approximation** 

We first test the in-context estimators’ ability to succeed at novel tasks solely at inference over _qφ_ ( _·|D_ ). To do so, we train the estimators on datasets being generated as _D_ train _∼ p_ , and are then evaluated on new _D_ test _∼ p_ . Mathematically this is equivalent to setting _χ_ according to 

5 

**Amortized In-Context Bayesian Posterior Estimation** 

||||_L_2 _L_|_oss_(_↓_)|_Accur_|_acy_(_↑_)|
|---|---|---|---|---|---|---|
|**Objective**|_qφ_|**Model**|LR|NLR|LC|NLC|
|Baseline|-|Random|23_._52_±_0.42|209_._35_±_9.92|50_._10_±_0.17|50_._84_±_1.00|
|||GRU|8_._95_±_0.47|84_._63_±_3.96|76_._34_±_1.58|60_._00_±_0.98|
|Fwd-KL|an|DeepSets|10_._81_±_0.08|97_._51_±_2.96|68_._26_±_0.31|50_._84_±_0.99|
||ssi|Transformer|9_._35_±_0.99|111_._07_±_6.55|63_._97_±_3.21|60_._39_±_0.53|
||Gau|GRU|9_._78_±_2.75|17_._01_±_5.16|79_._71_±_1.12|77_._19_±_0.21|
|Rev-KL||DeepSets|9_._26_±_0.10|8_._03_±_0.23|77_._40_±_0.14|72_._63_±_0.14|
|||Transformer|7_._30_±_0.21|8_._23_±_1.26|76_._96_±_1.58|71_._33_±_5.44|
|||GRU|8_._28_±_0.33|52_._04_±_3.39|76_._45_±_1.43|61_._10_±_0.61|
|Fwd-KL||DeepSets|12_._94_±_0.41|71_._69_±_2.38|67_._99_±_2.15|49_._74_±_0.76|
||low|Transformer|9_._64_±_0.50|84_._45_±_7.88|68_._26_±_3.23|61_._65_±_1.22|
||F|GRU|8_._17_±_0.25|17_._35_±_8.02|66_._30_±_2.81|78_._55_±_1.27|
|Rev-KL||DeepSets|11_._05_±_0.35|8_._54_±_0.49|78_._18_±_0.13|71_._69_±_0.18|
|||Transformer|7_._48_±_0.26|8_._80_±_1.80|72_._75_±_5.73|78_._23_±_0.89|



_Table 3._ **Tabular Experiments** : Zero-shot performance of the amortized variable-dimensional models across (non-)linear regression and classification real-world tabular tasks. All the models are trained solely on simulated data, and evaluated zero-shot on real-world data with varying number of both features and training (observations that are fed as in-context amortization) observations. 

Equation (10) where the number of observations, _|D|_ , is varied in some range both during training and evaluation. 

Figure 1 visualizes the amortized estimators in lowdimensional problems, showing that they learn meaningful distributions over the parameters zero-shot on new tasks. Next, we turn our attention to quantitative assessment of the different estimators on more complex, high-dimensional counterparts of the same probabilistic models. Table 1 shows that the in-context estimators are often comparable to optimization and MCMC baselines, with reverse KL objective combined with normalizing flows and the transformer architecture outperforming other design choices. In particular, we see that for high-dimensional multi-modal problems like NLR/NLC, forward KL approach does not fare well potentially due to its mode averaging property. Surprisingly, we also see non permutation invariant architectures like GRUs perform well, and often better than DeepSets. 

dure of masking unnecessary dimensions, similar to (Hollmann et al., 2022). This simple but strong insight allows us to amortize _qφ_ over datasets with varying dimensionalities. 

We embed all low-dimensional problems in a 100dimensional space by masking the unnecessary dimensions. Our experiments in Table 2 indicate that the same in-context learner can generalize to novel datasets _with a variable number of feature dimensions_ zero-shot. 

### **4.3. Model Misspecification** 

The true likelihood model underlying a data-generating process is often unknown. Practitioners address this by assuming a likelihood model and fitting its parameters to best explain the data. For example, while the true model for classifying emails as spam or not is unknown, one can assume a linear model to approximate the problem. This introduces model misspecification—a mismatch between the assumed and true model. 

### **4.2. Generalizing to Variable Feature Dimensions** 

So far, we only considered amortization over datasets for the same underlying likelihood model, which fixes the dimensionality of the problem. For example, a different incontext estimator has to be trained for a 2-dimensional and 5-dimensional Bayesian linear regression model since the dimensionality of **_θ_** changes. It is important to note that a deep learning-based approach leaves hopes of generalizing to new datasets of different dimensionalities since the underlying functional form of the solution remains constant across different datasets, irrespective of the number of features, and is given by the solution obtained from Equation 3. 

Alternatively, we can see that a low-dimensional problem can just be embedded into a high-dimensional space, with the extra features and parameters set to 0, akin to the proce- 

As discussed in Section 3, forward KL methods train only on simulated data from the assumed model, whereas reverse KL methods can use real-world data. Consequently, forward KL approaches struggle with sim-to-real transfer because they cannot incorporate real data during training. In contrast, reverse KL methods leverage real data, leading to more robust predictions in practical settings. 

Mathematically, let _χsim_ from Equation (10) denote simulated data and _χreal_ the actual target data. Table 4 shows that reverse KL methods outperform forward KL when trained on _χsim_ but tested on _χreal_ . Moreover, reverse KL methods trained directly on _χreal_ generalize even better (see rows labeled “+ switched data”). For experimental details and additional results on model misspecification, see Appendices F.3 and H.3. 

6 

**Amortized In-Context Bayesian Posterior Estimation** 



<!-- Start of picture text -->
25.0 Linear Regression 200 Nonlinear Regression 80 Linear Classification 80 Nonlinear Classification<br>22.5 175 75 75<br>20.0 150<br>70 70<br>125<br>17.5<br>100 65 65<br>15.0<br>75 60 60<br>12.5<br>50<br>55 55<br>10.0 25<br>7.5 0 50 50<br>5.0 25 45 45<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>Iterations Iterations Iterations Iterations<br>Random Initialization Fwd-KL Initialization Rev-KL Initialization<br>Loss<br>Accuracy<br><!-- End of picture text -->

_Figure 2._ **Tabular Experiments:** Initializing parameters from the proposed amortized model leads to good zero-shot performance and often optimal initialization across (non-)linear regression and classification tasks. 

|_χreal_ (_→_)<br>|_qφ _|<sup>_Data_</sup><br>Linear|MLP<br>Nonlinear|GP<br>Nonlinear|
|---|---|---|---|---|
|_χsim_ (_→_)||_Model_<br>NLR|LR|NLR|
|Fwd-KL|ian|15_._454_±_0.246|2_._216_±_0.097|14_._733_±_0.513|
|Rev-KL|uss|0_._382_±_0.003|1_._892_±_0.113|0_._155_±_0.006|
|_+ switched data_|Ga|0_._367_±_0.006|1_._226_±_0.001|0_._066_±_0.004|
|Fwd-KL|w|7_._949_±_0.419|1_._632_±_0.070|8_._557_±_0.561|
|Rev-KL|Flo|0_._347_±_0.001|1_._471_±_0.016|0_._120_±_0.005|
|_+ switched data_||0_._346_±_0.002|1_._226_±_0.004|0_._055_±_0.002|



_Table 4._ **Misspecification** . OoD evaluation under predictive _L_ 2 metric when the true data generating process, _χreal_ , is not known, while _χsim_ denotes the simulated one according to the (wrongly) assumed model. Estimators are trained on _χsim_ , with switched data denoting training on _χreal_ , and evaluation is solely on _χreal_ . 

### **4.4. Application to Tabular Benchmarks** 

To evaluate the efficacy of the trained in-context estimators and their ability to generalize out-of-distribution, we test the models trained in Section 4.2 on a suite of regression and classification problems chosen from the OpenML platform. These tasks have varying number of feature dimensions and inherently have different data statistics than the ones obtained from _χ_ during training. Table 3 shows the zero-shot performance of the parameters inferred from the in-context estimators, and highlights that they perform considerably better than chance, with transformer models and reverse KL methods outperforming other modeling choices. 

We also look at finetuning the inferred parameters from the in-context estimators with a maximum-a-posteriori ( _MAP_ ) objective and compare its performance with a corresponding model initialized from the prior. Our results in Figure 2 highlights that in-context estimators lead to much faster convergence, with reverse KL methods being superior in complex, multi-modal and nonlinear tasks. 

Finally, we look at a suite of problems that are extremely outof-distribution from the _χ_ used during training. In particular, we look at a suite of regression and classification tasks from the OpenML platform which consist of tasks with varying number of features. We refer the readers to Appendix H.4 for results on individual datasets with different _qφ_ , as well 

|||_Symm_<br>|_etric KL D_<br>|_ivergence_ <br>|(_↓_)<br>|
|---|---|---|---|---|---|
||**Model**|**Gaussian**|**Mean**|**L**|**R**|
|||_2D_|_100D_|_1D_|_100D_|
|Baseline|Random|44_._32|46_._78|179_._2|186_._8|
||GRU|0_._018_±_0.007|0_._07_±_0.00|0_._04_±_0.01|81_._8_±_0.2|
|Fwd-KL|DeepSets|0_._037_±_0.015|0_._22_±_0.01|0_._06_±_0.00|82_._0_±_0.4|
||Transformer|0_._030_±_0.008|0_._06_±_0.00|0_._04_±_0.01|20_._6_±_1.0|
||GRU|0_._017_±_0.005|0_._08_±_0.00|0_._03_±_0.00|78_._7_±_1.5|
|Rev-KL|DeepSets|0_._029_±_0.002|0_._19_±_0.01|0_._05_±_0.00|104_._5_±_8.8|
||Transformer|0_._035_±_0.013|0_._05_±_0.00|0_._03_±_0.00|32_._7_±_1.0|



_Table 5._ **Normalized Symmetric KL Divergence** . Amortized models with Gaussian _qφ_ approximate the true posterior well in tasks with tractable posteriors, when compared to the prior. 

as Appendix F.4 for implementation details. 

### **4.5. Evaluating Posterior Quality** 

While comparing with the true posterior is hard due to its intractability, it is still available when estimating the mean of a Gaussian distribution or performing Bayesian Linear Regression. Figure 3 ( **Right** ) shows the kernel density estimate of the samples from the true posterior, amortized forward, and reverse KL model, showing that both estimators efficiently capture the true posterior. We further quantify it through the symmetric KL divergence in Table 5. 

For more complex problems, we compute the squared Wasserstein metric _W_ 2<sup>2betweensamplesfromtheamor-</sup> tized posterior and multiple chains of Langevin MCMC in Table 6. Our results indicate that reverse KL approaches do slightly better in high-dimensional setups, while for lowdimensional multi-modal scenarios (eg. GMM), forward KL approaches fare better. Importantly, we note that this metric only provides a crude proxy to the quality of the posterior, since MCMC methods only provide asymptotic guarantees. 

## **5. Discussion and Conclusion** 

We show that Bayesian posterior inference can be amortized for a broad class of probabilistic models and explore a variety of design decisions associated with it. Some key conclusions from our analysis are described below. 

7 

**Amortized In-Context Bayesian Posterior Estimation** 



<!-- Start of picture text -->
1.3<br>0.6 True Posterior<br>1.2 Forward-KL<br>0.8 Reverse-KL<br>1.1<br>1.0 1.0<br>1.2 0.9<br>1.4 0.8<br>0.75 1.00 1.25 2.00 1.75 1.50<br>x1 w1<br>(a) Forward-KL (b) Reverse-KL (c) Mean of Gaussian (d) Linear Regression<br>x2 w2<br><!-- End of picture text -->

_Figure 3._ **Left** : Estimation of the means of a GMM, where red and green samples denote the first and second mean vectors. Unlike in reverse KL, the cluster labels switch in forward KL, highlighting its ability to capture underlying multi-modality. **Right** : Kernel density estimation of the true posterior, overlaid with estimates from forward and reverse KL systems, for different probabilistic models. 

||||||_W_|<sup>2</sup><br>2 <sup>(</sup><sup>_↓_)</sup>||||
|---|---|---|---|---|---|---|---|---|---|
|**Objective**|_qφ_<br>**Model **|**Gaussian**|**GMM**|**LR**|**N**|**LR**|**LC**|**N**|**LC**|
|||_100D_|_5D 2 cl_|_100D_|_1D_|_25D_|_100D_|_2D_|_25D_|
|Baseline|-<br>Random|13_._96_±_0.00|4_._01_±_0.00|13_._53_±_0.00|11_._21_±_0.00|36_._46_±_0.00|16_._72_±_0.00|14_._71_±_0.00|36_._67_±_0.00|
||GRU|1_._37_±_0.00|2_._35_±_0.01|10_._42_±_0.02|11_._10_±_0.00|36_._33_±_0.01|15_._25_±_0.01|14_._67_±_0.00|36_._66_±_0.00|
|Fwd-KL|an<br>DeepSets|1_._55_±_0.01|2_._35_±_0.01|10_._39_±_0.03|11_._11_±_0.01|36_._41_±_0.01|16_._72_±_0.00|14_._68_±_0.00|36_._66_±_0.00|
||ussi<br>Transformer|1_._41_±_0.00|2_._40_±_0.01|5_._83_±_0.09|11_._09_±_0.01|36_._32_±_0.01|14_._70_±_0.01|14_._67_±_0.00|36_._66_±_0.00|
||Ga<br>GRU|1_._34_±_0.00|2_._98_±_0.01|7_._39_±_0.03|11_._31_±_0.01|35_._92_±_0.32|12_._31_±_0.01|14_._14_±_0.01|35_._16_±_0.00|
|Rev-KL|DeepSets|1_._38_±_0.02|2_._98_±_0.02|7_._58_±_0.05|11_._31_±_0.02|35_._58_±_0.20|12_._93_±_0.06|14_._10_±_0.01|35_._05_±_0.00|
||Transformer|1_._34_±_0.01|2_._98_±_0.03|4_._84_±_0.03|11_._38_±_0.01|35_._86_±_0.08|12_._84_±_0.04|14_._17_±_0.02|35_._36_±_0.00|
||GRU|1_._37_±_0.00|1_._71_±_0.16|10_._18_±_0.04|11_._09_±_0.02|36_._35_±_0.01|15_._29_±_0.02|14_._66_±_0.01|36_._66_±_0.00|
|Fwd-KL|DeepSets|1_._58_±_0.01|1_._81_±_0.08|10_._48_±_0.14|11_._08_±_0.01|36_._41_±_0.00|16_._72_±_0.00|14_._67_±_0.01|36_._66_±_0.00|
||Flow<br>Transformer|1_._40_±_0.01|1_._20_±_0.26|5_._64_±_0.23|11_._08_±_0.00|36_._32_±_0.02|14_._66_±_0.01|14_._65_±_0.01|36_._66_±_0.00|
||GRU|1_._33_±_0.00|2_._99_±_0.02|7_._48_±_0.06|11_._15_±_0.04|35_._97_±_0.04|13_._51_±_0.02|14_._33_±_0.01|35_._79_±_0.01|
|Rev-KL|DeepSets|1_._41_±_0.02|2_._96_±_0.01|8_._40_±_0.09|11_._15_±_0.04|36_._02_±_0.09|13_._61_±_0.05|14_._32_±_0.01|35_._69_±_0.01|
||Transformer|1_._33_±_0.01|3_._00_±_0.04|4_._90_±_0.15|11_._21_±_0.02|36_._00_±_0.17|13_._63_±_0.02|14_._37_±_0.01|35_._88_±_0.02|



_Table 6._ **Sample Based Metrics** . We compute the 2-Wasserstein metric between samples from the approximate posterior and MCMC. 

**Forward vs Reverse KL** . Our GMM experiments (Figure 3; **Left** ) indicate that forward KL is more amenable to learning multimodal solutions compared to reverse KL in lowdimensional problems. However, the latter outperforms the former in both predictive and sample-based metrics when the parameter space is high-dimensional. Further, reverse KL methods do not require access to ( **_θ_** _, D_ ) samples during training and thus show improvements in misspecification and simulation to real transfer. 

**Architectural Choices** . We compare permutation invariant architectures like DeepSets and Transformers with non invariant architecture like GRU and see that GRU outperforms DeepSets even though the latter is permutation invariant. We hypothesize that this could be due to limited expressivity of DeepSets and their reliance on a fixed pooling operator. In contrast, GRUs can learn to be approximately permutation invariant through training. We further see that Transformers outperform both DeepSets and GRUs as they do not rely on fixed aggregation schemes but still respect the invariant structure of the posterior. 

**Capacity of** _qφ_ . Increasing the capacity of _qφ_ using normalizing flows substantially helps for forward KL but only marginally for the reverse KL objective. We hypothesize that because of the mode-seeking tendency of reverse KL, even with the capacity to model different modes, the algorithm latches to a single one. However, in forward KL setup without additional capacity the model overestimates the variance a lot. 

We provide a rigorous comparison of different in-context posterior estimators, especially in the presence of misspecification and generalization to real-world problems. It provides an exciting direction of research which could reduce the load of real-world, complex, and iterative approximations through quick and cheap inference over a trained amortized network – providing a direction into learning a generalist in-context Bayesian estimator. We believe that scaling our approach to more complex probabilistic models, leveraging better modeling choices for high-dimensional problems (Bengio et al., 2021; Zhang & Chen, 2021; Vargas et al., 2023), and training a single model for multiple probabilistic models are important directions of future work. 

8 

**Amortized In-Context Bayesian Posterior Estimation** 

## **Acknowledgements** 

The authors would like to acknowledge the computing resources provided by the Mila cluster to enable the experiments outlined in this work. SM acknowledges the support of UNIQUE’s scholarship. GL acknowledges the support of the Canada CIFAR AI Chair program, NSERC Discovery Grant RGPIN-2018-04821, and a Canada Research Chair in Neural Computations and Interfacing. MAB acknowledges the support of the Canada First Research Excellence Fund (CFREF) for the Vision: Science to Applications (VISTA) program, NSERC Discovery Grant RGPIN-2017-05638 and Google. The authors also thank NVIDIA for computing resources. 

## **Impact Statement** 

We provide a comprehensive evaluation of different approaches and design choices in performing Bayesian posterior estimation for a wide variety of probabilistic models. We believe that analysis into such amortized estimators could lead to more efficient and scalable Bayesian methods that can lead to robust predictions and better OoD generalization. Thus, we believe that our work generally advances the field of machine learning through careful and thorough benchmarking. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. 

## **References** 

- Ardizzone, L., Bungert, T., Draxler, F., Kothe, U., Kruse,¨ J., Schmier, R., and Sorrenson, P. FrEIA: Framework for easily invertible architectures, 2018. URL https: //github.com/vislearn/FrEIA. 

- Ardizzone, L., Bungert, T., Draxler, F., Kothe, U., Kruse,¨ J., Schmier, R., and Sorrenson, P. Framework for Easily Invertible Architectures (FrEIA), 2018-2022. URL https://github.com/vislearn/FrEIA. 

- Arenz, O., Dahlinger, P., Ye, Z., Volpp, M., and Neumann, G. A unified perspective on natural gradient variational inference with gaussian mixture models. _arXiv preprint arXiv:2209.11533_ , 2022. 

- Athanasopoulos, G., Hyndman, R. J., Kourentzes, N., and Panagiotelis, A. Forecast reconciliation: A review. _International Journal of Forecasting_ , 2023. ISSN 01692070. doi: https://doi.org/10.1016/j.ijforecast.2023.10. 010. URL https://www.sciencedirect.com/ science/article/pii/S0169207023001097. 

- Bengio, E., Jain, M., Korablyov, M., Precup, D., and Bengio, Y. Flow network based generative models for noniterative diverse candidate generation. _Advances in Neu-_ 

_ral Information Processing Systems_ , 34:27381–27394, 2021. 

- Bingham, E., Chen, J. P., Jankowiak, M., Obermeyer, F., Pradhan, N., Karaletsos, T., Singh, R., Szerlip, P., Horsfall, P., and Goodman, N. D. Pyro: Deep universal probabilistic programming. _The Journal of Machine Learning Research_ , 20(1):973–978, 2019. 

- Bischl, B., Casalicchio, G., Feurer, M., Hutter, F., Lang, M., Mantovani, R. G., van Rijn, J. N., and Vanschoren, J. Openml benchmarking suites. _arXiv:1708.03731v2 [stat.ML]_ , 2019. 

- Bishop, C. M. and Nasrabadi, N. M. _Pattern recognition and machine learning_ , volume 4. Springer, 2006. 

- Bitzer, M., Meister, M., and Zimmer, C. Amortized inference for gaussian process hyperparameters of structured kernels. _arXiv preprint arXiv:2306.09819_ , 2023. 

- Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. Variational inference: A review for statisticians. _Journal of the American statistical Association_ , 112(518):859–877, 2017. 

- Brooks, S. Markov chain monte carlo method and its application. _Journal of the royal statistical society: series D (the Statistician)_ , 47(1):69–100, 1998. 

- Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M., Brubaker, M. A., Guo, J., Li, P., and Riddell, A. Stan: A probabilistic programming language. _Journal of statistical software_ , 76, 2017. 

- Chauhan, V. K., Zhou, J., Lu, P., Molaei, S., and Clifton, D. A. A brief review of hypernetworks in deep learning. _arXiv preprint arxiv:2306.06955_ , 2023. 

- Chen, Y., Garnett, R., and Montgomery, J. M. Polls, context, and time: A dynamic hierarchical bayesian forecasting model for us senate elections. _Political Analysis_ , 31(1): 113–133, 2023. doi: 10.1017/pan.2021.42. 

- Cooper, I., Mondal, A., and Antonopoulos, C. G. A sir model assumption for the spread of covid19 in different communities. _Chaos, Solitons & Fractals_ , 139:110057, 2020. ISSN 0960-0779. doi: https://doi.org/10.1016/j.chaos.2020.110057. URL https://www.sciencedirect.com/ science/article/pii/S0960077920304549. 

- Cranmer, K., Brehmer, J., and Louppe, G. The frontier of simulation-based inference. _Proceedings of the National Academy of Sciences_ , 117(48):30055–30062, May 2020. ISSN 1091-6490. doi: 10.1073/ pnas.1912789117. URL http://dx.doi.org/10. 1073/pnas.1912789117. 

9 

**Amortized In-Context Bayesian Posterior Estimation** 

- Dinh, L., Sohl-Dickstein, J., and Bengio, S. Density estimation using real NVP. _5th International Conference on Learning Representations, ICLR 2017 - Conference Track Proceedings_ , 2017. URL http://arxiv.org/ abs/1605.08803. 

- Finn, C., Abbeel, P., and Levine, S. Model-agnostic metalearning for fast adaptation of deep networks. In _International conference on machine learning_ , pp. 1126–1135. PMLR, 2017. 

- Fischer, S. F., Feurer, M., and Bischl, B. OpenML-CTR23 – a curated tabular regression benchmarking suite. In _AutoML Conference 2023 (Workshop)_ , 2023. URL https: //openreview.net/forum?id=HebAOoMm94. 

- Gardner, J., Pleiss, G., Weinberger, K. Q., Bindel, D., and Wilson, A. G. Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration. _Advances in neural information processing systems_ , 31, 2018. 

- Garg, S., Tsipras, D., Liang, P. S., and Valiant, G. What can transformers learn in-context? a case study of simple function classes. _Advances in Neural Information Processing Systems_ , 35:30583–30598, 2022. 

- Garnelo, M., Rosenbaum, D., Maddison, C., Ramalho, T., Saxton, D., Shanahan, M., Teh, Y. W., Rezende, D., and Eslami, S. A. Conditional neural processes. In _International conference on machine learning_ , pp. 1704–1713. PMLR, 2018a. 

- Garnelo, M., Schwarz, J., Rosenbaum, D., Viola, F., Rezende, D. J., Eslami, S., and Teh, Y. W. Neural processes. _arXiv preprint arXiv:1807.01622_ , 2018b. 

- Geffner, T., Papamakarios, G., and Mnih, A. Compositional score modeling for simulation-based inference. 2023. 

- Gelfand, A. E. Gibbs sampling. _Journal of the American statistical Association_ , 95(452):1300–1304, 2000. 

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and Rubin, D. B. _Bayesian Data Analysis, Third Edition_ . CRC Press, November 2013. ISBN 9781439840955. URL https://play.google.com/store/books/ details?id=ZXL6AQAAQBAJ. 

- Glaeser, R. M., Nogales, E., and Chiu, W. _Single-particle Cryo-EM of Biological Macromolecules_ . 2053-2563. IOP Publishing, 2021. ISBN 978-0-7503-3039-8. doi: 10. 1088/978-0-7503-3039-8. URL https://dx.doi. org/10.1088/978-0-7503-3039-8. 

- Gordon, J., Bruinsma, W. P., Foong, A. Y., Requeima, J., Dubois, Y., and Turner, R. E. Convolutional conditional neural processes. _arXiv preprint arXiv:1910.13556_ , 2019. 

- Grant, E., Finn, C., Levine, S., Darrell, T., and Griffiths, T. Recasting gradient-based meta-learning as hierarchical bayes. _arXiv preprint arXiv:1801.08930_ , 2018. 

- Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., and Lerchner, A. beta-VAE: Learning basic visual concepts with a constrained variational framework. In _International Conference on Learning Representations_ , 2017. URL https: //openreview.net/forum?id=Sy2fzU9gl. 

- Hoffman, M. D., Blei, D. M., Wang, C., and Paisley, J. Stochastic variational inference. _Journal of Machine Learning Research_ , 2013. 

- Hoffman, M. D., Gelman, A., et al. The no-u-turn sampler: adaptively setting path lengths in hamiltonian monte carlo. _J. Mach. Learn. Res._ , 15(1):1593–1623, 2014. 

- Hollmann, N., Muller,¨ S., Eggensperger, K., and Hutter, F. Tabpfn: A transformer that solves small tabular classification problems in a second. _arXiv preprint arXiv:2207.01848_ , 2022. 

- Hospedales, T., Antoniou, A., Micaelli, P., and Storkey, A. Meta-learning in neural networks: A survey. _IEEE Transactions on Pattern Analysis & Machine Intelligence_ , 44(09):5149–5169, sep 2022. ISSN 1939-3539. doi: 10.1109/TPAMI.2021.3079209. 

- Kim, H., Mnih, A., Schwarz, J., Garnelo, M., Eslami, A., Rosenbaum, D., Vinyals, O., and Teh, Y. W. Attentive neural processes. _arXiv preprint arXiv:1901.05761_ , 2019. 

- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- Kingma, D. P. and Dhariwal, P. Glow: Generative flow with invertible 1x1 convolutions. _Advances in neural information processing systems_ , 31, 2018. 

- Kingma, D. P. and Welling, M. Auto-encoding variational bayes. _arXiv preprint arXiv:1312.6114_ , 2013. 

- Kingma, D. P., Welling, M., et al. An introduction to variational autoencoders. _Foundations and Trends® in Machine Learning_ , 12(4):307–392, 2019. 

- Kobyzev, I., Prince, S. J., and Brubaker, M. A. Normalizing flows: An introduction and review of current methods. _IEEE transactions on pattern analysis and machine intelligence_ , 43(11):3964–3979, 2020. 

- Koch, G., Zemel, R., Salakhutdinov, R., et al. Siamese neural networks for one-shot image recognition. In _ICML deep learning workshop_ , volume 2. Lille, 2015. 

10 

**Amortized In-Context Bayesian Posterior Estimation** 

- Krueger, D., Huang, C.-W., Islam, R., Turner, R., Lacoste, A., and Courville, A. Bayesian hypernetworks. _arXiv preprint arxiv:1710.04759_ , 2017. 

- Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., and Teh, Y. W. Set transformer: A framework for attention-based permutation-invariant neural networks. In _International conference on machine learning_ , pp. 3744–3753. PMLR, 2019. 

- Lin, W., Schmidt, M., and Khan, M. E. Handling the positive-definite constraint in the bayesian learning rule. In _International conference on machine learning_ , pp. 6116–6126. PMLR, 2020. 

- Liu, S., Sun, X., Ramadge, P. J., and Adams, R. P. Taskagnostic amortized inference of gaussian process hyperparameters. _Advances in Neural Information Processing Systems_ , 33:21440–21452, 2020. 

- Lorch, L., Sussex, S., Rothfuss, J., Krause, A., and Scholkopf, B.¨ Amortized inference for causal structure learning. _Advances in Neural Information Processing Systems_ , 35:13104–13118, 2022. 

- Minka, T. P. Expectation propagation for approximate bayesian inference. _arXiv preprint arXiv:1301.2294_ , 2013. 

- Morris, Q. Recognition networks for approximate inference in bn20 networks. _arXiv preprint arXiv:1301.2295_ , 2013. 

- Muller, S., Hollmann, N., Arango, S. P., Grabocka, J., and¨ Hutter, F. Transformers can do bayesian inference. _arXiv preprint arXiv:2112.10510_ , 2021. 

- Paige, B. and Wood, F. Inference networks for sequential monte carlo in graphical models. In _International Conference on Machine Learning_ , pp. 3040–3049. PMLR, 2016. 

- Pakman, A., Wang, Y., Mitelut, C., Lee, J., and Paninski, L. Neural clustering processes. In _International Conference on Machine Learning_ , pp. 7455–7465. PMLR, 2020. 

- Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., and Lakshminarayanan, B. Normalizing flows for probabilistic modeling and inference. _The Journal of Machine Learning Research_ , 22(1):2617–2680, 2021. 

- Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., and Kothe, U.¨ Bayesflow: Learning complex stochastic models with invertible neural networks. _IEEE transactions on neural networks and learning systems_ , 33(4):1452–1466, 2020. 

- Rezende, D. and Mohamed, S. Variational inference with normalizing flows. In _International conference on machine learning_ , pp. 1530–1538. PMLR, 2015. 

- Rezende, D. J., Mohamed, S., and Wierstra, D. Stochastic backpropagation and approximate inference in deep generative models. In _International conference on machine learning_ , pp. 1278–1286. PMLR, 2014. 

- Simpson, F., Davies, I., Lalchand, V., Vullo, A., Durrande, N., and Rasmussen, C. E. Kernel identification through transformers. _Advances in Neural Information Processing Systems_ , 34:10483–10495, 2021. 

- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. _arXiv preprint arXiv:2011.13456_ , 2020. 

- Stuhlmuller,¨ A., Taylor, J., and Goodman, N. Learning stochastic inverses. _Advances in neural information processing systems_ , 26, 2013. 

- Sun, Z., Ozay, M., and Okatani, T. Hypernetworks with statistical filtering for defending adversarial examples. _arXiv preprint arxiv:1711.01791_ , 2017. 

- Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P. H., and Hospedales, T. M. Learning to compare: Relation network for few-shot learning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 1199–1208, 2018. 

Tang, L., Zhou, Y., Wang, L., Purkayastha, S., Zhang, L., He, J., Wang, F., and Song, P. X.K. A review of multi-compartment infectious disease models. _International Statistical Review_ , 88 (2):462–513, 2020. doi: https://doi.org/10.1111/insr. 12402. URL https://onlinelibrary.wiley. com/doi/abs/10.1111/insr.12402. 

- Vargas, F., Grathwohl, W., and Doucet, A. Denoising diffusion samplers. _arXiv preprint arXiv:2302.13834_ , 2023. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Vinyals, O., Blundell, C., Lillicrap, T., Wierstra, D., et al. Matching networks for one shot learning. _Advances in neural information processing systems_ , 29, 2016. 

- Von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., and Vladymyrov, M. Transformers learn in-context by gradient descent. In _International Conference on Machine Learning_ , pp. 35151–35174. PMLR, 2023. 

- von Oswald, J., Niklasson, E., Schlegel, M., Kobayashi, S., Zucchet, N., Scherrer, N., Miller, N., Sandler, M., 

11 

**Amortized In-Context Bayesian Posterior Estimation** 

Vladymyrov, M., Pascanu, R., et al. Uncovering mesaoptimization algorithms in transformers. _arXiv preprint arXiv:2309.05858_ , 2023. 

- Welling, M. and Teh, Y. W. Bayesian learning via stochastic gradient langevin dynamics. In _Proceedings of the 28th international conference on machine learning (ICML-11)_ , pp. 681–688, 2011. 

- Zaheer, M., Kottur, S., Ravanbhakhsh, S., Poczos,´ B., Salakhutdinov, R., and Smola, A. J. Deep sets. In _Advances in Neural Information Processing Systems_ , volume 2017-December, 2017. 

- Zhang, Q. and Chen, Y. Path integral sampler: a stochastic control approach for sampling. _arXiv preprint arXiv:2111.15141_ , 2021. 

- Strumbelj, E., Bouchard-Cˇ otˆ e, A., Corander, J., Gelman, A.,´ Rue, H., Murray, L., Pesonen, H., Plummer, M., and Vehtari, A. Past, present, and future of software for bayesian inference, 2023. URL http://hdl.handle.net/ 10754/694575. 

12 

**Amortized In-Context Bayesian Posterior Estimation** 

# **Appendix** 

## **A. Related Work** 

In this section, we draw parallels of our work to various approaches that have been proposed to tackle the problem of either providing a good initialization for different tasks, performing implicit optimization to model predictive distributions for new tasks, or estimating the posterior through a different objective. 

### **A.1. Variational Autoencoders** 

VAEs (Kingma & Welling, 2013; Rezende et al., 2014; Rezende & Mohamed, 2015; Kingma et al., 2019) are latent variable models which model observations **_x_** conditioned on latent variables **_z_** through the joint distribution _pθ_ ( **_x_** _,_ **_z_** ) = _pθ_ ( **_x_** _|_ **_z_** ) _p_ ( **_z_** ) where _p_ ( **_z_** ) is generally chosen as _N_ ( **0** _,_ **I** ). Training the model is done through VI where _qφ_ ( **_z_** ) is obtained by explicit amortization over the data point, that is, _qφ_ ( **_z_** _|_ **_x_** ) = _N_ ( **_µ_** _φ_ ( **_x_** ) _,_ **Σ** _φ_ ( **_x_** )). Training this system on a dataset _D_ is done by similarly optimizing the Evidence Lower-Bound, which boils down to the following optimization problem 



This objective can easily be optimized using gradient-based learning and the reparameterization trick. While typically, a diagonal Gaussian distribution is considered for _qφ_ , more complex distributions utilizing normalizing flows can also be used. 

### **A.2. Hypernetworks** 

Hypernetworks are neural networks that generate weights for another neural network, used in tasks such as uncertainty quantification, zero-shot learning, etc. We refer for a comprehensive overview to (Chauhan et al., 2023). Based on experiments on predicting the weights of a compact MLP (section 4), our work shows similarities with studies in this area but also has significant differences. Regarding uncertainty quantification, hypernetworks are instrumental in creating an ensemble of models by generating multiple weight vectors for the primary network. Each model within this ensemble possesses distinct parameter configurations, enabling robust estimation of uncertainty in model predictions. This feature is precious in safety-critical domains like healthcare, where confidence in predictions is essential. Multiple weight sets can be generated through techniques like dropout within hypernetworks or sampling from a noise distribution. The latter (Krueger et al., 2017) is based on a Bayesian framework where weights can be sampled using invertible network architecture, such as normalizing flows. However, while we amortize posterior inference, the weights sampled from the hypernetwork are not conditioned on information from the currently observed input data during inference time but indirectly solely on the dataset available during training, and retraining would need to be done given a new dataset. Departing from the Bayesian framework, (Sun et al., 2017) have shown data-specific discriminative weight prediction, which aligns well with their specific objective of defending a convolutional neural network against adversarial attacks. Combining the ability to sample a new set of weights dataset-specifically but also handling dataset exchangeability, even in the more realistic case of missing information, our work has a distinctly different focus but also can be seen as an extension to hypernetwork research. 

### **A.3. In-Context Learning** 

Amortized inference has close links to in-context learning (ICL), which has been gaining popularity, especially in natural language modeling. Various works show how in-context learning can be seen as performing implicit optimization based on the context examples, with some constructions showing exact equivalence with gradient descent in linear regression (Von Oswald et al., 2023; von Oswald et al., 2023). Other works have shown how such systems can be seen as implicitly modeling the Bayesian posterior predictive distribution (Muller et al.¨ , 2021). In a similar vein, there have been additional works aimed at directly modeling the posterior predictive distribution by providing the training data as “context” to a Transformer model and training it based on the maximum log-likelihood principle (Hollmann et al., 2022). While such approaches have been seeing tremendous success, they cannot be directly applied to cases where we care about and want to analyze the solution space as the solution space is only modeled implicitly, and thus, recovering it is not possible. For example, if our goal is to learn a linear regression model, an ICL model could end up learning a nonlinear model and would provide no information about the actual parameters used for prediction. As opposed to this, we obtain parameters explicitly. We thus can answer questions like the relevance of a particular feature (which corresponds to its weight in the output, and we know the weight vector explicitly). Even further, many systems grounded in physics and economics only admit a constrained solution space; for example, the movement of a human arm lies on a particular manifold, or the configuration of molecules and proteins 

13 

**Amortized In-Context Bayesian Posterior Estimation** 

|||||_L_2 _Los_|_s_(_↓_)|||
|---|---|---|---|---|---|---|---|
|**Objective**|_qφ_<br>**Model**|**Gau**|**ssian**||**GM**|**M**||
|||_2D_|_100D_|_2D-2cl_|_2D-5cl_|_5D-2cl_|_5D-5cl_|
||-<br>Random|5_._839_±_0.015|301_._065_±_0.346|1_._887_±_0.031|0_._730_±_0.004|5_._001_±_0.037|1_._670_±_0.008|
|Baseline|-<br>Optimization|1_._989_±_0.000|101_._243_±_0.000|0_._169_±_0.000|0_._119_±_0.001|0_._425_±_0.000|0_._308_±_0.000|
||-<br>Langevin|2_._013_±_0.004|102_._346_±_0.031|0_._173_±_0.001|0_._125_±_0.001|0_._448_±_0.009|0_._352_±_0.005|
||-<br>HMC|2_._018_±_0.008|102_._413_±_0.028|0_._174_±_0.001|0_._135_±_0.001|0_._479_±_0.007|0_._449_±_0.002|
||GRU|2_._014_±_0.001|102_._641_±_0.011|0_._921_±_0.013|0_._522_±_0.001|2_._430_±_0.034|1_._235_±_0.011|
|Fwd-KL|n<br>DeepSets|2_._012_±_0.002|103_._215_±_0.054|0_._920_±_0.019|0_._522_±_0.001|2_._436_±_0.037|1_._238_±_0.009|
||ssia<br>Transformer|2_._013_±_0.002|102_._783_±_0.005|0_._931_±_0.017|0_._522_±_0.001|2_._498_±_0.026|1_._230_±_0.009|
||Gau<br>GRU|2_._012_±_0.001|102_._509_±_0.008|0_._183_±_0.002|0_._132_±_0.002|0_._471_±_0.010|0_._413_±_0.019|
|Rev-KL|DeepSets|2_._011_±_0.001|102_._599_±_0.042|0_._186_±_0.001|0_._127_±_0.002|0_._495_±_0.018|0_._409_±_0.005|
||Transformer|2_._013_±_0.002|102_._540_±_0.025|0_._185_±_0.004|0_._122_±_0.001|0_._489_±_0.019|0_._328_±_0.002|
||GRU|2_._014_±_0.001|102_._656_±_0.019|0_._186_±_0.006|0_._242_±_0.005|0_._670_±_0.094|0_._563_±_0.018|
|Fwd-KL|DeepSets|2_._014_±_0.001|103_._340_±_0.029|0_._185_±_0.006|0_._237_±_0.008|0_._648_±_0.082|0_._583_±_0.028|
||ow<br>Transformer|2_._016_±_0.002|102_._774_±_0.024|0_._188_±_0.012|0_._252_±_0.001|0_._621_±_0.070|0_._592_±_0.019|
||Fl<br>GRU|2_._013_±_0.001|102_._490_±_0.012|0_._184_±_0.006|0_._130_±_0.002|0_._467_±_0.003|0_._384_±_0.005|
|Rev-KL|DeepSets|2_._011_±_0.001|102_._674_±_0.046|0_._188_±_0.005|0_._131_±_0.002|0_._519_±_0.008|0_._405_±_0.005|
||Transformer|2_._013_±_0.001|102_._525_±_0.050|0_._187_±_0.004|0_._123_±_0.001|0_._468_±_0.007|0_._326_±_0.008|



_Table 7._ **Fixed-Dimensional** . Results for estimating the mean of a Gaussian (Gaussian) and means of a Gaussian mixture model (GMM) with the expected _L_ 2 loss according to the posterior predictive as metric. 

cannot be arbitrary. Thus, performing predictions through an implicit solution space, which may violate several constraints, is not ideal. Furthermore, explicitly modeling the solution space and encoding the constraints present can be done through the prior and the parametric distribution used for modeling. 

### **A.4. Meta Learning** 

Meta-learning (Hospedales et al., 2022) aims to equip models with the ability to quickly learn from different tasks or data sets to generalize to new tasks in resource-constrained domains. This attribute is precious in practical scenarios where obtaining large amounts of task-specific data is impractical or costly. A simple way of obtaining this is through nonparametric or similarity-based models like k-Nearest Neighbours, where no training is involved. Thus, new tasks can be solved quickly based on a few examples by computing a similarity metric with these examples (Koch et al., 2015; Vinyals et al., 2016; Sung et al., 2018). Another way of achieving this is through optimization-based setups, which use a nested optimization procedure. An inner step learns individual tasks from a shared initialization, whereas the outer loop computes the gradient of the whole inner process and moves the initialization in a way that allows for better generalization. Here, by relying on only a few iterations in the inner loop, the outer loop has the incentive to move the initialization to a point from which solutions to multiple tasks are reachable (Finn et al., 2017). Given the similarities between meta-learning and hierarchical Bayesian inference (Grant et al., 2018), our approach can be considered as a kind of meta-learning framework; however, the line between meta-learning and Bayesian posterior inference is quite blurry as any amortized approach for the latter can be seen as a case of the former. 

### **A.5. Neural Processes** 

A notable approach in meta-learning related to our research is neural processes (NP), which excel in learning scenarios with few examples. NPs (Garnelo et al., 2018a;b; Kim et al., 2019; Pakman et al., 2020; Gordon et al., 2019) can be seen as a more flexible and powerful extension of Gaussian processes that leverage a neural network-based encoder-decoder architecture for learning to model a distribution over functions that approximate a stochastic process. However, while we are interested in approximating the posterior distribution over the parameters, NPs are used to approximate the posterior predictive distribution to make predictions based on observed data. Similar to our setup, NPs rely on amortized VI for 

14 

**Amortized In-Context Bayesian Posterior Estimation** 

|||_L_2 _L_|_oss_(_↓_)|_Accura_|_cy_(_↑_)|||
|---|---|---|---|---|---|---|---|
|**Objective**|_qφ_<br>**Model**|**Linear R**|**egression**||**Linear Cla**|**ssifcation**||
|||_2D_|_100D_|_2D-2cl_|_2D-5cl_|_100D-2cl_|_100D-5cl_|
||-<br>Random|4_._178_±_0.018|202_._601_±_0.321|50_._498_±_0.357|19_._891_±_0.028|50_._046_±_0.047|20_._054_±_0.053|
|Baseline|-<br>Optimization|0_._257_±_0.000|25_._083_±_0.006|96_._982_±_0.000|93_._449_±_0.002|70_._258_±_0.012|41_._338_±_0.012|
||-<br>Langevin|0_._263_±_0.002|23_._340_±_0.689|95_._034_±_0.412|88_._277_±_0.290|65_._123_±_0.370|32_._544_±_0.422|
||-<br>HMC|0_._263_±_0.001|18_._659_±_0.189|92_._659_±_0.344|82_._169_±_0.518|62_._145_±_0.245|29_._582_±_0.371|
||GRU|0_._264_±_0.001|124_._823_±_0.135|81_._170_±_0.389|71_._170_±_0.275|59_._740_±_0.102|23_._042_±_0.246|
|Fwd-KL|n<br>DeepSets|0_._264_±_0.000|123_._133_±_1.080|81_._281_±_0.278|70_._993_±_0.191|50_._047_±_0.051|20_._053_±_0.045|
||ssia<br>Transformer|0_._264_±_0.000|45_._856_±_1.331|80_._960_±_0.285|71_._484_±_0.437|62_._954_±_0.062|26_._789_±_0.110|
||Gau<br>GRU|0_._263_±_0.000|60_._215_±_0.866|94_._258_±_0.034|87_._339_±_0.023|63_._465_±_0.307|28_._270_±_0.462|
|Rev-KL|DeepSets|0_._263_±_0.000|62_._837_±_0.617|94_._285_±_0.116|87_._342_±_0.021|60_._867_±_0.265|21_._339_±_0.085|
||Transformer|0_._264_±_0.001|28_._735_±_0.252|94_._302_±_0.054|87_._540_±_0.117|68_._185_±_0.007|32_._950_±_0.284|
||GRU|0_._264_±_0.001|119_._119_±_0.233|96_._305_±_0.008|88_._927_±_0.200|59_._920_±_0.221|23_._025_±_0.077|
|Fwd-KL|DeepSets|0_._264_±_0.001|125_._677_±_3.731|96_._191_±_0.021|88_._643_±_0.102|50_._061_±_0.021|20_._021_±_0.094|
||ow<br>Transformer|0_._264_±_0.000|43_._272_±_2.700|96_._344_±_0.059|89_._624_±_0.215|64_._349_±_0.147|26_._952_±_0.203|
||Fl<br>GRU|0_._263_±_0.000|61_._295_±_1.008|95_._241_±_0.012|88_._429_±_0.024|64_._669_±_0.207|28_._409_±_1.167|
|Rev-KL|DeepSets|0_._263_±_0.001|76_._412_±_2.038|95_._296_±_0.021|88_._464_±_0.061|58_._384_±_0.812|21_._569_±_0.117|
||Transformer|0_._263_±_0.000|29_._358_±_1.569|95_._339_±_0.063|88_._644_±_0.047|68_._721_±_0.121|33_._107_±_0.333|



_Table 8._ **Fixed-Dimensional** . Results for estimating the parameters of linear regression (LR) and classification (LC) models with the expected _L_ 2 loss and accuracy according to the posterior predictive as metrics. 

obtaining the predictive posterior. Still, instead of working with a known probabilistic model, they train the probabilistic model primarily for prediction-based tasks through approaches analogous to variational expectation maximization. Thus, they cannot provide an explicit posterior over the parameters, but they are suitable for tasks where only predictive posteriors are essential, such as those in supervised learning. NPs, in their most basic form, accomplish this by training for the objective: 



where **_z_** _∈_ R<sup>_p_</sup> is an arbitrary latent variable often uninterpretable, and the parameters of the probabilistic model **_θ_** do not get a Bayesian treatment. In particular, NPs are more suited to modeling datasets of the form _D_ = _{_ **_x_** _i,_ **_y_** _i}_<sup>_n_</sup> _i_ =1<sup>, where all</sup> probabilities in Equation 18 are conditioned on the input **_x_** ’s, and only the predictive over **_y_** ’s is modeled, and _p_ **_θ_** is modeled as a Neural Network. 

These approaches can be seen as quite related to ICL, where the exchangeable architecture backbone is switched from DeepSets to Transformers. Similar to ICL, they do not provide control over the solution space as they aim to model either the posterior predictive or an arbitrary latent space. While this leads to good predictive performance on various tasks, they cannot be freely applied to problems that pose certain constraints on the underlying probabilistic model. In such cases, estimating the actual parameters is important to enforce constraints in the parameter space as well as for interpretability, which we already discussed in the ICL section. 

### **A.6. Simulation-Based Inference** 

In the case of simulation-based inference (Cranmer et al., 2020), when the likelihood _p_ ( **_x_** _|_ **_θ_** ) is intractable, BayesFlow (Radev et al., 2020) and similar methods (Lorch et al., 2022) provide a solution framework to amortize Bayesian inference of parameters in complex models. Starting from the forward KL divergence between the true and approximate posteriors, the resulting objective is to optimize for parameters of the approximate posterior distribution that maximize the posterior probability of data-generating parameters **_θ_** given observed data _D_ for all **_θ_** and _D_ . Density estimation of the approximate posterior can then be done using the change-of-variables formula and a conditional invertible neural network that 

15 

**Amortized In-Context Bayesian Posterior Estimation** 

|||||_L_2|_Loss_(_↓_)||
|---|---|---|---|---|---|---|
|**Objective**|_qφ_|**Model**||**Nonlinear R**|**egression**_|_**ReLU**||
||||_1-l_|_ayer_|_2-l_|_ayers_|
||||_1D_|_25D_|_1D_|_25D_|
||-|Random|65_._936_±_0.913|831_._595_±_8.696|1029_._407_±_11.542|12067_._691_±_183.598|
|Baseline|-|Optimization|0_._360_±_0.001|103_._967_±_0.110|2_._370_±_0.015|1894_._574_±_4.266|
||-|Langevin|0_._308_±_0.000|132_._391_±_0.992|N/A|N/A|
||-|HMC|0_._374_±_0.002|98_._061_±_0.730|22_._314_±_0.814|3903_._510_±_5.377|
|||GRU|49_._332_±_0.946|671_._639_±_10.494|774_._045_±_7.521|9905_._246_±_214.545|
|Fwd-KL|n|DeepSets|49_._864_±_0.979|684_._853_±_2.581|768_._921_±_8.278|9946_._090_±_109.933|
||ssia|Transformer|49_._678_±_0.940|680_._853_±_5.838|747_._221_±_12.189|9982_._609_±_85.596|
||Gau|GRU|0_._426_±_0.004|105_._976_±_0.586|1_._066_±_0.069|1796_._512_±_5.805|
|Rev-KL||DeepSets|0_._426_±_0.004|125_._853_±_0.791|1_._394_±_0.108|1892_._402_±_2.793|
|||Transformer|0_._417_±_0.005|102_._295_±_1.825|2_._075_±_0.147|1811_._440_±_115.435|
|||GRU|15_._781_±_0.210|538_._962_±_4.269|614_._925_±_15.494|7564_._076_±_67.160|
|Fwd-KL||DeepSets|15_._051_±_0.120|548_._535_±_3.288|622_._461_±_7.043|7618_._364_±_115.946|
||ow|Transformer|16_._109_±_0.307|539_._338_±_4.336|597_._718_±_8.358|7635_._052_±_109.037|
||Fl|GRU|0_._405_±_0.010|106_._001_±_0.420|0_._988_±_0.045|1814_._649_±_8.327|
|Rev-KL||DeepSets|0_._395_±_0.004|128_._169_±_1.451|1_._215_±_0.028|1886_._698_±_7.294|
|||Transformer|0_._387_±_0.004|102_._610_±_0.863|2_._549_±_0.058|1791_._741_±_49.585|



_Table 9._ **Fixed-Dimensional** . Results for estimating the parameters of nonlinear regression models with ReLU activation function, with the expected _L_ 2 loss according to the posterior predictive as metric. 

parameterizes the approximate posterior distribution. 

arg min KL[ _p_ ( **_θ_** _|D_ ) _||qφ_ ( **_θ_** _|D_ )] = arg min E( **_θ_** _,D_ ) _∼p_ ( **_θ_** _,D_ ) [ _−_ log _p_ **_z_** ( _fν_ ( **_θ_** ; _hψ_ ( _D_ ))) _−_ log _|_ det _Jfν |_ ] (19) _φ φ_ = _{ν,ψ}_ 

Since their goal is to learn a global estimator for the probabilistic mapping from _D_ to data generating **_θ_** , the information about the observed dataset is encoded in the output of a summary network _hψ_ . It is used as conditional input to the normalizing flow _fν_ . Although the likelihood function does not need to be known, the method requires access to paired observations ( **_x_** _,_ **_θ_** ) for training, which is sometimes unavailable. This approach is equivalent to the _Forward KL_ setup in our experiments when trained with DeepSets and Normalizing Flows. Current research has also leveraged score-based generative models for SBI which can condition on a dataset by learning a score model conditional only on single observations (Geffner et al., 2023). 

### **A.7. Amortization in Gaussian Processes** 

Gaussian Processes (GPs) define a class of probabilistic models that do enjoy tractable likelihood. However, inference in such systems is slow and sensitive to the choice of kernel function that defines the covariance matrix. Similar to meta learning and neural processes, current research also focuses on estimating the kernel function in GPs by leveraging permutation invariant architectures like transformers (Liu et al., 2020; Simpson et al., 2021; Bitzer et al., 2023). Additionally, often these approaches amortize based on point estimates and are leveraged when considering GPs for regression problems, and it is not straightforward to extend them to classification or unsupervised learning. In contrast, our approach is more general and can work for all problems that define a differentiable likelihood function. Additionally, our approach also approximates the Bayesian posterior distribution over the parameters of interest, as opposed to point estimates. 

### **A.8. Mode Collapse in Variational Inference** 

Reverse KL based methods have been widely known to suffer from mode collapse due to the nature of the optimization objective (Bishop & Nasrabadi, 2006), which implies that even if the approximate distribution possesses the ability to represent multiple modes, optimization is often sub-optimal and the distribution ends up covering only a small handful of 

16 

**Amortized In-Context Bayesian Posterior Estimation** 

|||||_L_2 _Lo_<br>|_ss_(_↓_)<br>||
|---|---|---|---|---|---|---|
|**Objective**|_qφ_|**Model**||**Nonlinear Reg**|**ression**_|_**TanH**||
||||_1-l_|_ayer_|_2-la_|_yers_|
||||_1D_|_25D_|_1D_|_25D_|
||-|Random|31_._448_±_0.186|52_._644_±_0.173|52_._735_±_1.122|52_._583_±_0.132|
|Baseline|-|Optimization|0_._366_±_0.001|13_._352_±_0.005|0_._651_±_0.002|30_._176_±_0.056|
||-|Langevin|0_._296_±_0.003|17_._221_±_0.130|0_._363_±_0.003|28_._528_±_0.115|
||-|HMC|0_._398_±_0.003|12_._607_±_0.192|0_._733_±_0.021|23_._571_±_0.346|
|||GRU|31_._391_±_0.161|52_._008_±_0.282|52_._725_±_1.149|51_._989_±_0.139|
|Fwd-KL|n|DeepSets|31_._421_±_0.074|52_._137_±_0.215|52_._850_±_1.192|51_._904_±_0.334|
||ssia|Transformer|31_._350_±_0.219|52_._945_±_0.430|52_._693_±_1.188|52_._364_±_0.164|
||Gau|GRU|0_._415_±_0.003|15_._874_±_6.958|0_._951_±_0.047|25_._907_±_0.012|
|Rev-KL||DeepSets|0_._405_±_0.004|25_._333_±_0.010|0_._912_±_0.013|25_._877_±_0.002|
|||Transformer|0_._412_±_0.013|11_._784_±_0.949|0_._847_±_0.010|20_._405_±_3.874|
|||GRU|12_._415_±_0.800|52_._039_±_0.065|52_._695_±_0.611|52_._576_±_0.225|
|Fwd-KL||DeepSets|31_._790_±_0.163|51_._933_±_0.115|52_._903_±_0.625|52_._643_±_0.239|
||ow|Transformer|10_._392_±_0.195|52_._470_±_0.364|52_._385_±_0.689|52_._646_±_0.622|
||Fl|GRU|0_._386_±_0.005|11_._401_±_0.041|0_._736_±_0.009|25_._892_±_0.010|
|Rev-KL||DeepSets|0_._374_±_0.005|25_._685_±_0.004|0_._686_±_0.019|25_._885_±_0.007|
|||Transformer|0_._376_±_0.002|10_._486_±_0.040|0_._724_±_0.026|25_._885_±_0.011|



_Table 10._ **Fixed-Dimensional** . Results for estimating the parameters of nonlinear regression models with TanH activation function, with the expected _L_ 2 loss according to the posterior predictive as metric. 

them. Improving normalizing flow based methods with repulsive terms or through the lens of natural gradient optimization procedure for a mixture approximate distribution (Arenz et al., 2022; Lin et al., 2020) is an important topic of research, and we believe it would be quite an important future work to experimentally validate if they help in learning multi-modality in amortized posterior inference problems that are studied in this work. 

## **B. Architectures respecting Exchangeability** 

In this section, we highlight how DeepSets and Transformer models satisfy the dataset exchangeability criteria, which is essential in modeling the posterior distribution over the parameters of any probabilistic model relying on _iid_ data. 

### **B.1. DeepSets** 

DeepSets (Zaheer et al., 2017) operate on arbitrary sets _X_ = _{x_ 1 _, ...xN } ⊂_ R<sup>_d_</sup> of fixed dimensionality _d_ by first mapping each individual element **_x_** _i ∈X_ to some high-dimensional space using a nonlinear transform, which is parameterized as a multi-layered neural network with parameters _φ_ 1 



After having obtained this high-dimensional embedding of each element of the set, it applies an aggregation function _a_ ( _·_ ), which is a permutation invariant function that maps a set of elements _Z_ = _{_ **_z_** 1 _, ...,_ **_z_** _N } ∈_ R<sup>_z_</sup> to an element **_h_** _∈_ R<sup>_z_</sup> , 



Thus, the outcome does not change under permutations of _Z_ . Finally, another nonlinear transform, parameterized by a multi-layered neural network with parameters _φ_ 2, is applied to the outcome **_h_** to provide the final output. 



For our experiments, we then use the vector **_o_** to predict the parameters of a parametric family of distributions (e.g., Gaussian or Flows) using an additional nonlinear neural network. As an example, for the Gaussian case, we consider the distribution 

17 

**Amortized In-Context Bayesian Posterior Estimation** 

|||||_Accur_|_acy_(_↑_)||
|---|---|---|---|---|---|---|
|**Objective**|_qφ_|**Model**|**Non**|**linear Classifca**|**tion**_|_**ReLU - 2**|**class**|
||||_1-la_|_yer_|_2-la_|_yers_|
||||_2D_|_25D_|_2D_|_25D_|
||-|Random|50_._306_±_0.590|50_._008_±_0.326|50_._394_±_0.190|49_._846_±_0.635|
|Bli|-|Optimization|96_._879_±_0.028|77_._896_±_0.023|96_._770_±_0.062|82_._073_±_0.200|
|asene|-|Langevin|95_._971_±_0.313|73_._165_±_0.282|96_._645_±_0.101|76_._541_±_0.139|
||-|HMC|91_._763_±_0.163|70_._395_±_0.110|91_._797_±_0.048|76_._445_±_0.372|
|||GRU|59_._518_±_0.355|56_._858_±_0.319|60_._962_±_0.599|60_._063_±_0.695|
|Fwd-KL|n|DeepSets|59_._383_±_0.244|56_._806_±_0.204|61_._090_±_0.690|59_._933_±_0.599|
||ssia|Transformer|59_._588_±_0.389|57_._089_±_0.376|61_._151_±_0.560|60_._041_±_0.680|
||Gau|GRU|92_._384_±_0.195|72_._455_±_0.032|86_._157_±_0.066|69_._966_±_0.333|
|Rev-KL||DeepSets|92_._488_±_0.133|59_._806_±_0.315|86_._275_±_0.733|69_._550_±_0.371|
|||Transformer|92_._627_±_0.377|75_._178_±_0.142|86_._351_±_0.217|69_._812_±_0.527|
|||GRU|76_._931_±_0.266|58_._338_±_0.026|62_._981_±_0.452|63_._199_±_0.202|
|Fwd-KL||DeepSets|72_._313_±_1.829|58_._113_±_0.126|62_._438_±_0.277|62_._884_±_0.201|
||ow|Transformer|77_._296_±_0.201|58_._344_±_0.148|63_._753_±_0.155|63_._590_±_0.390|
||Fl|GRU|93_._392_±_0.073|72_._002_±_0.506|85_._712_±_0.665|71_._419_±_0.200|
|Rev-KL||DeepSets|93_._312_±_0.197|60_._943_±_0.184|85_._960_±_0.896|71_._288_±_0.211|
|||Transformer|93_._578_±_0.093|74_._956_±_0.546|87_._138_±_0.438|71_._525_±_0.159|



_Table 11._ **Fixed-Dimensional** . Results for estimating the parameters of nonlinear classification models with ReLU activation function and two classes, with the expected accuracy according to the posterior predictive as metric. 





which makes **_µ_** implicitly a function of the original input set _X_ . To understand why the posterior distribution modeled in this fashion does not change when the inputs are permuted, let us assume that Π is a permutation over the elements of _X_ . If we look at one of the parameters of the posterior distribution, e.g., **_µ_** , we can see that 



which simply follows from the fact that _a_ ( _·_ ) is a permutation invariant operation, e.g., sum or mean. We can also provide similar reasoning for the other parameters (e.g., **Σ** ). This shows that DeepSets can be used to model the posterior distribution over parameters of interest as it respects the exchangeability criteria ( _iid_ observations) assumptions in the data through its permutation invariant structure. 

### **B.2. Transformers** 

Similarly, we can look at Transformers (Vaswani et al., 2017) as candidates for respecting the exchangeability conditions in the data. In particular, we consider transformer systems without positional encodings and consider an additional [CLS] token, denoted by **_c_** _∈_ R<sup>_d_</sup> , to drive the prediction. If we look at the application of a layer of transformer model, it can be broken down into two components. 

**Multi-Head Attention** . Given a query vector obtained from **_c_** and keys and values coming from our input set _X ⊂_ R<sup>_d_</sup> , we can model the update of the context **_c_** as 



18 

**Amortized In-Context Bayesian Posterior Estimation** 

|||||_Accur_|_acy_(_↑_)||
|---|---|---|---|---|---|---|
|**Objective**|_qφ_|**Model**|**Non**|**linear Classifca**|**tion**_|_**ReLU - 5**|**class**|
||||_1-la_|_yer_|_2-la_|_yers_|
||||_2D_|_25D_|_2D_|_25D_|
||-|Random|19_._952_±_0.119|20_._026_±_0.062|19_._832_±_0.202|19_._985_±_0.091|
|Bli|-|Optimization|94_._369_±_0.022|60_._589_±_0.064|93_._664_±_0.025|60_._824_±_0.023|
|asene|-|Langevin|91_._286_±_0.149|50_._845_±_0.366|92_._449_±_0.068|52_._216_±_0.170|
||-|HMC|81_._387_±_0.742|47_._408_±_0.548|81_._098_±_0.246|52_._854_±_0.190|
|||GRU|32_._597_±_0.169|29_._932_±_0.142|31_._061_±_0.089|30_._187_±_0.070|
|Fwd-KL|n|DeepSets|32_._481_±_0.090|29_._909_±_0.218|30_._923_±_0.205|19_._983_±_0.089|
||ssia|Transformer|32_._977_±_0.114|30_._064_±_0.134|31_._478_±_0.079|30_._307_±_0.121|
||Gau|GRU|83_._460_±_0.269|36_._688_±_0.189|71_._710_±_0.375|28_._234_±_0.148|
|Rev-KL||DeepSets|83_._734_±_0.313|34_._078_±_0.066|67_._519_±_1.735|47_._585_±_0.143|
|||Transformer|84_._645_±_0.515|36_._850_±_0.138|74_._390_±_0.280|27_._808_±_0.232|
|||GRU|43_._850_±_0.412|31_._926_±_0.096|33_._771_±_0.400|31_._871_±_0.228|
|Fwd-KL||DeepSets|43_._620_±_0.250|31_._829_±_0.126|32_._919_±_0.126|19_._950_±_0.192|
||ow|Transformer|44_._057_±_0.122|31_._789_±_0.094|34_._079_±_0.224|32_._392_±_0.212|
||Fl|GRU|84_._459_±_0.194|37_._343_±_0.134|64_._309_±_0.273|48_._459_±_0.542|
|Rev-KL||DeepSets|84_._775_±_0.129|34_._917_±_0.069|67_._937_±_4.341|48_._563_±_0.199|
|||Transformer|85_._857_±_0.096|46_._968_±_0.094|75_._005_±_0.605|48_._889_±_0.380|



_Table 12._ **Fixed-Dimensional** . Results for estimating the parameters of nonlinear classification models with ReLU activation function and five classes, with the expected accuracy according to the posterior predictive as metric. 

where **_W_** _Q ∈_ R<sup>_d×k_</sup> _,_ **_W_** _K ∈_ R<sup>_d×k_</sup> _,_ **_W_** _V ∈_ R<sup>_d×k_</sup> and **_X_** _∈_ R<sup>_N×d_</sup> denotes a certain ordering of the elements in _X_ . Further, ˆ **_c_** is the updated vector after attention, and Softmax is over the rows of **_X_** . Here, we see that if we were to apply a permutation to the elements in **_X_** , the outcome would remain the same. In particular 



which follows because Softmax is an equivariant function, i.e., applying Softmax on a permutation of columns is equivalent to applying Softmax first and then permuting the columns correspondingly. Thus, we see that the update to the [CLS] token **_c_** is permutation invariant. This output is then used independently as input to a multi-layered neural network with residual connections, and the entire process is repeated multiple times without weight sharing to simulate multiple layers. Since all the individual parts are permutation invariant w.r.t permutations on _X_ , the entire setup ends up being permutation invariant. Obtaining the parameters of a parametric family of distribution for posterior estimation then follows the same recipe as DeepSets, with **_o_** replaced by **_c_** . 

## **C. Probabilistic Models** 

This section details the various candidate probabilistic models used in our experiments for amortized computation of Bayesian posteriors over the parameters. Here, we explain the parameters associated with the probabilistic model over which we want to estimate the posterior and the likelihood and prior that we use for experimentation. 

**Mean of Gaussian (GM):** As a proof of concept, we consider the simple setup of estimating the posterior distribution over the mean of a Gaussian distribution _p_ ( **_µ_** _|D_ ) given some observed data. In this case, prior and likelihood defining the 

19 

**Amortized In-Context Bayesian Posterior Estimation** 

|||||_Accura_|_cy_(_↑_)||
|---|---|---|---|---|---|---|
|**Objective**|_qφ_|**Model**|**Nonli**|**near Classifcat**|**ion**_|_**TanH - 2**|**class**|
||||_1-la_|_yer_|_2-la_|_yers_|
||||_2D_|_25D_|_2D_|_25D_|
||-|Random|50_._147_±_0.603|49_._963_±_0.083|49_._942_±_0.452|49_._978_±_0.113|
|Baseline|-|Optimization|96_._552_±_0.005|75_._228_±_0.029|94_._130_±_0.018|69_._052_±_0.029|
||-|Langevin|94_._778_±_0.210|68_._787_±_0.258|92_._417_±_0.149|62_._722_±_0.180|
||-|HMC|91_._674_±_0.200|67_._415_±_0.667|88_._218_±_0.202|62_._323_±_0.262|
|||GRU|50_._151_±_0.616|49_._959_±_0.087|49_._942_±_0.448|49_._987_±_0.112|
|Fwd-KL|n|DeepSets|50_._147_±_0.601|49_._961_±_0.083|49_._939_±_0.451|49_._977_±_0.110|
||ssia|Transformer|50_._146_±_0.622|49_._959_±_0.082|49_._948_±_0.448|49_._970_±_0.104|
||Gau|GRU|89_._813_±_0.181|49_._969_±_0.085|49_._956_±_0.442|49_._974_±_0.088|
|Rev-KL||DeepSets|89_._558_±_0.264|49_._970_±_0.100|49_._961_±_0.436|49_._986_±_0.094|
|||Transformer|89_._879_±_0.387|49_._978_±_0.075|79_._102_±_0.155|49_._987_±_0.089|
|||GRU|50_._066_±_0.488|49_._835_±_0.240|49_._836_±_0.328|49_._911_±_0.206|
|Fwd-KL||DeepSets|50_._078_±_0.468|49_._848_±_0.242|49_._834_±_0.333|49_._910_±_0.205|
||ow|Transformer|50_._067_±_0.483|49_._827_±_0.235|49_._813_±_0.331|49_._905_±_0.210|
||Fl|GRU|90_._396_±_0.126|49_._910_±_0.169|50_._018_±_0.171|50_._001_±_0.076|
|Rev-KL||DeepSets|90_._247_±_0.067|49_._901_±_0.124|50_._132_±_0.162|49_._942_±_0.110|
|||Transformer|90_._416_±_0.311|49_._904_±_0.133|81_._729_±_0.070|49_._945_±_0.203|



_Table 13._ **Fixed-Dimensional** . Results for estimating the parameters of nonlinear classification models with TanH activation function and two classes, with the expected accuracy according to the posterior predictive as metric. 

probabilistic model _p_ ( **_x_** _,_ **_θ_** ) (with **_θ_** being the mean **_µ_** ) are given by: 



and **Σ** is known beforehand and defined as a unit variance matrix. 

**Linear Regression (LR):** We then look at the problem of estimating the posterior over the weight vector for Bayesian linear regression given a dataset _p_ ( **_w_** _, b|D_ ), where the underlying model _p_ ( _D,_ **_θ_** ) is given by: 



and with _σ_<sup>2</sup> = 0 _._ 25 known beforehand. Inputs **_x_** are generated from _p_ ( **_x_** ) = _N_ ( **0** _, I_ ). 

**Linear Classification (LC):** We now consider a setting where the true posterior cannot be obtained analytically as the likelihood and prior are not conjugate. In this case, we consider the underlying probabilistic model by: 





where _τ_ is the known temperature term which is kept as 0 _._ 1 to ensure peaky distributions, and **_x_** is being generated from _p_ ( **_x_** ) = _N_ ( **0** _, I_ ). 

**Nonlinear Regression (NLR):** Next, we tackle the more complex problem where the posterior distribution is multi-modal and obtaining multiple modes or even a single good one is challenging. For this, we consider the model as a Bayesian Neural Network (BNN) for regression with fixed hyper-parameters like the number of layers, dimensionality of the hidden 

20 

**Amortized In-Context Bayesian Posterior Estimation** 

|||||_Accura_|_cy_(_↑_)||
|---|---|---|---|---|---|---|
|**Objective**|_qφ_|**Model**|**Nonli**|**near Classifcat**|**ion**_|_**TanH - 5**|**class**|
||||_1-la_|_yer_|_2-la_|_yers_|
||||_2D_|_25D_|_2D_|_25D_|
||-|Random|19_._745_±_0.269|20_._037_±_0.075|20_._214_±_0.043|19_._864_±_0.089|
|Baseline|-|Optimization|92_._919_±_0.011|49_._972_±_0.070|88_._156_±_0.014|39_._410_±_0.043|
||-|Langevin|88_._635_±_0.374|39_._758_±_0.212|84_._050_±_0.138|31_._472_±_0.010|
||-|HMC|81_._057_±_0.277|35_._378_±_0.130|75_._305_±_0.134|29_._730_±_0.670|
|||GRU|19_._960_±_0.271|20_._235_±_0.083|20_._452_±_0.059|20_._028_±_0.077|
|Fwd-KL|n|DeepSets|19_._807_±_0.209|20_._040_±_0.072|20_._210_±_0.043|19_._861_±_0.094|
||ssia|Transformer|19_._977_±_0.273|20_._241_±_0.068|20_._453_±_0.062|20_._029_±_0.082|
||Gau|GRU|77_._711_±_0.014|20_._026_±_0.079|20_._213_±_0.043|19_._877_±_0.092|
|Rev-KL||DeepSets|76_._414_±_0.378|20_._038_±_0.060|20_._216_±_0.027|19_._887_±_0.089|
|||Transformer|79_._163_±_0.183|20_._026_±_0.093|51_._408_±_0.484|19_._872_±_0.083|
|||GRU|32_._900_±_0.115|20_._209_±_0.048|20_._105_±_0.385|20_._040_±_0.037|
|Fwd-KL||DeepSets|20_._137_±_0.070|20_._000_±_0.048|19_._887_±_0.392|19_._895_±_0.025|
||ow|Transformer|30_._135_±_2.459|20_._224_±_0.048|20_._104_±_0.399|20_._030_±_0.039|
||Fl|GRU|79_._329_±_0.320|20_._074_±_0.033|19_._904_±_0.180|19_._864_±_0.079|
|Rev-KL||DeepSets|20_._071_±_0.302|19_._999_±_0.045|19_._872_±_0.259|19_._906_±_0.051|
|||Transformer|80_._064_±_0.159|20_._002_±_0.032|19_._777_±_0.210|19_._911_±_0.082|



_Table 14._ **Fixed-Dimensional** . Results for estimating the parameters of nonlinear classification models with TanH activation function and five classes, with the expected accuracy according to the posterior predictive as metric. 

layer, etc. Let the BNN denote the function _f_ **_θ_** where **_θ_** are the network parameters such that the estimation problem is to approximate _p_ ( **_θ_** _|D_ ). Then, for regression, we specify the probabilistic model using: 





where _σ_<sup>2</sup> = 0 _._ 25 is a known quantity and **_x_** being generated from _p_ ( **_x_** ) = _N_ ( **0** _, I_ ). 

**Nonlinear Classification (NLC):** Like in Nonlinear Regression, we consider BNNs with fixed hyper-parameters for classification problems with the same estimation task of approximating _p_ ( **_θ_** _|D_ ). In this formulation, we consider the probabilistic model as: 





where _τ_ is the known temperature term which is kept as 0 _._ 1 to ensure peaky distributions, and **_x_** is being generated from _p_ ( **_x_** ) = _N_ ( **0** _, I_ ). 

**Gaussian Mixture Model (GMM):** While we have mostly looked at predictive problems, where the task is to model some predictive variable _y_ conditioned on some input **_x_** , we now look at a well-known probabilistic model for unsupervised learning, Gaussian Mixture Model (GMM), primarily used to cluster data. Consider a _K_ -cluster GMM with: 





We assume **Σ** _k_ and _πk_ to be known and set **Σ** _k_ to be an identity matrix and the mixing coefficients to be equal, _πk_ = 1 _/K_ , for all clusters _k_ in our experiments. 

21 

**Amortized In-Context Bayesian Posterior Estimation** 

|||||_L_2 _Los_|_s_(_↓_)|||
|---|---|---|---|---|---|---|---|
|**Objective**|_qφ_<br>**Model**|**Gau**|**ssian**||**GM**|**M**||
|||_2D_|_100D_|_2D-2cl_|_2D-5cl_|_5D-2cl_|_5D-5cl_|
||-<br>Random|6_._297_±_0.017|298_._238_±_0.228|2_._078_±_0.134|0_._626_±_0.037|4_._659_±_0.034|1_._632_±_0.004|
|Baseline|-<br>Optimization|2_._020_±_0.000|100_._885_±_0.000|0_._175_±_0.002|0_._121_±_0.002|0_._427_±_0.000|0_._323_±_0.002|
||-<br>Langevin|2_._036_±_0.004|101_._917_±_0.042|0_._178_±_0.002|0_._123_±_0.002|0_._440_±_0.002|0_._340_±_0.006|
||-<br>HMC|2_._044_±_0.008|102_._015_±_0.009|0_._189_±_0.013|0_._132_±_0.004|0_._462_±_0.013|0_._423_±_0.009|
||GRU|2_._300_±_0.105|133_._224_±_0.579|1_._119_±_0.150|0_._473_±_0.012|2_._360_±_0.017|1_._208_±_0.002|
|Fwd-KL|n<br>DeepSets|2_._216_±_0.017|129_._695_±_0.737|1_._113_±_0.150|0_._477_±_0.013|2_._352_±_0.018|1_._210_±_0.003|
||ssia<br>Transformer|2_._352_±_0.013|108_._977_±_0.100|1_._134_±_0.153|0_._476_±_0.013|2_._399_±_0.019|1_._208_±_0.004|
||Gau<br>GRU|2_._047_±_0.002|105_._141_±_0.102|0_._187_±_0.005|0_._147_±_0.006|0_._462_±_0.011|0_._398_±_0.019|
|Rev-KL|DeepSets|2_._049_±_0.003|105_._062_±_0.212|0_._194_±_0.004|0_._145_±_0.003|0_._481_±_0.021|0_._387_±_0.002|
||Transformer|2_._057_±_0.004|104_._709_±_0.122|0_._195_±_0.004|0_._140_±_0.003|0_._468_±_0.006|0_._335_±_0.016|
||GRU|2_._358_±_0.005|125_._835_±_1.983|0_._287_±_0.032|0_._212_±_0.002|0_._596_±_0.068|0_._513_±_0.015|
|Fwd-KL|DeepSets|2_._053_±_0.003|133_._229_±_1.933|0_._271_±_0.050|0_._202_±_0.006|0_._584_±_0.030|0_._517_±_0.016|
||ow<br>Transformer|2_._060_±_0.005|108_._484_±_0.164|0_._344_±_0.054|0_._221_±_0.006|0_._591_±_0.080|0_._533_±_0.010|
||Fl<br>GRU|2_._050_±_0.004|105_._187_±_0.030|0_._199_±_0.014|0_._142_±_0.004|0_._466_±_0.007|0_._373_±_0.004|
|Rev-KL|DeepSets|2_._054_±_0.003|105_._095_±_0.064|0_._202_±_0.007|0_._146_±_0.003|0_._494_±_0.013|0_._379_±_0.003|
||Transformer|2_._049_±_0.003|104_._914_±_0.113|0_._193_±_0.004|0_._138_±_0.002|0_._460_±_0.003|0_._327_±_0.006|



_Table 15._ **Variable-Dimensional** . Results for estimating the mean of a Gaussian (Gaussian) and means of a Gaussian mixture model (GMM) with the expected _L_ 2 loss according to the posterior predictive as metric. 

## **D. Metrics** 

In this section, we provide details about the metrics considered for the different tasks. We generally look at two main metrics for benchmarking performance: _L_ 2 loss and Accuracy. For estimating the mean of a Gaussian distribution, the _L_ 2 loss is defined as 



where _D_ = _{_ **_x_** _i}_<sup>_N_</sup> _i_ =1<sup>_D_.Intuitively, this captures the quality of the estimation of the mean parameter by measuring how far the</sup> observations are from it. Lower value implies better estimation of the mean parameter. Similarly, for estimating the means of a Gaussian Mixture Model, we rely on a similar metric but we also find the cluster closest to the observation, which can be defined as 





which intuitively captures the distance of observations from the cluster closest to them. Next, we define the metric for evaluating (non-)linear regression models as 



Finally, for the (non-)linear classification setups, we define the accuracy metric as 



22 

**Amortized In-Context Bayesian Posterior Estimation** 

|||_L_2 _L_|_oss_(_↓_)|_Accura_|_cy_(_↑_)|||
|---|---|---|---|---|---|---|---|
|**Objective**|_qφ_<br>**Model**|**Linear R**|**egression**||**Linear Cla**|**ssifcation**||
|||_2D_|_100D_|_2D-2cl_|_2D-5cl_|_100D-2cl_|_100D-5cl_|
||-<br>Random|4_._272_±_0.068|200_._836_±_0.609|50_._125_±_0.264|20_._078_±_0.065|50_._005_±_0.061|20_._033_±_0.082|
|Baseline|-<br>Optimization|0_._258_±_0.000|20_._127_±_0.003|97_._301_±_0.000|91_._752_±_0.000|71_._231_±_0.010|42_._345_±_0.001|
||-<br>Langevin|0_._263_±_0.002|21_._781_±_0.953|95_._441_±_0.209|86_._445_±_0.496|65_._469_±_0.513|32_._668_±_0.145|
||-<br>HMC|0_._263_±_0.000|17_._774_±_0.120|92_._961_±_0.228|78_._793_±_0.314|62_._602_±_0.171|30_._055_±_0.506|
||GRU|0_._271_±_0.004|139_._396_±_1.012|79_._467_±_0.711|65_._124_±_0.861|57_._872_±_0.157|22_._677_±_0.081|
|Fwd-KL|n<br>DeepSets|0_._269_±_0.001|149_._784_±_0.766|80_._323_±_0.429|20_._078_±_0.059|50_._767_±_0.058|20_._035_±_0.081|
||ssia<br>Transformer|0_._279_±_0.001|64_._282_±_3.711|79_._901_±_0.271|60_._984_±_1.590|62_._382_±_0.029|26_._997_±_0.098|
||Gau<br>GRU|0_._291_±_0.013|62_._624_±_0.123|93_._367_±_0.289|82_._020_±_0.127|63_._411_±_0.248|28_._655_±_0.149|
|Rev-KL|DeepSets|0_._279_±_0.004|64_._064_±_0.221|93_._977_±_0.093|83_._832_±_0.106|61_._305_±_0.114|27_._877_±_0.265|
||Transformer|0_._271_±_0.007|31_._984_±_0.482|94_._336_±_0.210|82_._976_±_0.074|67_._676_±_0.078|33_._125_±_0.051|
||GRU|0_._273_±_0.001|138_._284_±_1.030|92_._078_±_0.190|75_._151_±_0.274|57_._982_±_0.138|22_._430_±_0.208|
|Fwd-KL|DeepSets|0_._270_±_0.002|153_._207_±_0.814|85_._950_±_6.332|20_._059_±_0.233|50_._494_±_0.055|19_._976_±_0.095|
||ow<br>Transformer|0_._276_±_0.004|63_._102_±_1.963|94_._494_±_0.368|74_._876_±_0.857|63_._559_±_0.072|27_._098_±_0.147|
||Fl<br>GRU|0_._276_±_0.008|71_._260_±_1.265|94_._292_±_0.175|83_._622_±_0.057|63_._391_±_0.133|27_._340_±_0.243|
|Rev-KL|DeepSets|0_._274_±_0.003|76_._772_±_1.836|94_._570_±_0.178|85_._059_±_0.127|59_._116_±_0.491|22_._810_±_0.245|
||Transformer|0_._279_±_0.013|33_._056_±_0.321|94_._793_±_0.135|84_._929_±_0.027|68_._124_±_0.214|33_._251_±_0.130|



_Table 16._ **Variable-Dimensional** . Results for estimating the parameters of linear regression (LR) and classification (LC) models with the expected _L_ 2 loss and accuracy according to the posterior predictive as metrics. 

where _δ_ ( _a, b_ ) = 1 if and only if _a_ = _b_ . Thus this metric captures the accuracy of the posterior predictive distribution. Another metric that we use to test the quality of the posterior is the symmetric KL divergence, defined as 



## **E. Architecture Details** 

In this section, we outline the two candidate architectures that we consider for the backbone of our amortized variational inference model. We discuss the specifics of the architectures and the hyperparameters used for our experiments. 

### **E.1. Transformer** 

We use a transformer model (Vaswani et al., 2017) as a permutation invariant architecture by removing positional encodings from the setup and using multiple layers of the encoder model. We append the set of observations with a [CLS] token before passing it to the model and use its output embedding to predict the parameters of the variational distribution. Since no positional encodings or causal masking is used in the whole setup, the final embedding of the [CLS] token becomes invariant to permutations in the set of observations, thereby leading to permutation invariance in the parameters of _qφ_ . 

We use 4 encoder layers with a 256 dimensional attention block and 1024 feed-forward dimensions, with 4 heads in each attention block for our Transformer models to make the number of parameters comparative to the one of the DeepSets model. 

### **E.2. DeepSets** 

Another framework that can process set-based input is Deep Sets (Zaheer et al., 2017). In our experiments, we used an embedding network that encodes the input into representation space, a mean aggregation operation, which ensures that the representation learned is invariant concerning the set ordering, and a regression network. The latter’s output is either used to directly parameterize a diagonal Gaussian or as conditional input to a normalizing flow, representing a summary statistics of the set input. 

23 

**Amortized In-Context Bayesian Posterior Estimation** 

||||_1-l_<br>|_L_2 <br>_ayer_<br>|_Loss_(_↓_)<br>_2-l_<br>|_ayers_<br>|
|---|---|---|---|---|---|---|
|**Objective**|_qφ_|**Model**|_1D_|_50D_|_1D_|_50D_|
||-|Random|73_._006_±_0.171|1704_._335_±_9.330|998_._890_±_16.317|27799_._887_±_165.653|
|Baseline|-|Optimization|0_._359_±_0.002|309_._162_±_0.204|3_._078_±_0.057|4894_._141_±_4.290|
||-|Langevin|0_._308_±_0.002|N/A|N/A|N/A|
||-|HMC|0_._381_±_0.005|303_._857_±_2.487|7_._999_±_0.595|12905_._846_±_9.903|
|||GRU|51_._448_±_0.029|1346_._462_±_6.833|754_._388_±_32.650|21121_._735_±_112.458|
|Fwd-KL|n|DeepSets|51_._901_±_1.542|1357_._462_±_5.348|767_._781_±_18.262|21110_._264_±_72.613|
||ssia|Transformer|50_._813_±_0.532|1319_._868_±_12.165|753_._177_±_23.023|21037_._666_±_116.929|
||Gau|GRU|2_._308_±_0.126|316_._344_±_6.189|23_._520_±_3.816|4673_._633_±_98.443|
|Rev-KL||DeepSets|0_._977_±_0.118|451_._942_±_2.785|8_._034_±_0.375|6127_._317_±_190.224|
|||Transformer|0_._815_±_0.024|278_._282_±_1.073|8_._300_±_1.673|4744_._375_±_24.609|
|||GRU|38_._407_±_0.364|1097_._410_±_9.498|664_._409_±_9.590|18372_._905_±_62.019|
|Fwd-KL||DeepSets|43_._305_±_2.063|1119_._990_±_5.461|745_._412_±_27.599|20719_._466_±_627.997|
||ow|Transformer|39_._701_±_0.519|1073_._256_±_1.504|619_._631_±_22.299|16700_._463_±_350.596|
||Fl|GRU|2_._305_±_0.406|302_._918_±_5.644|13_._729_±_1.445|4832_._392_±_50.014|
|Rev-KL||DeepSets|0_._827_±_0.024|454_._141_±_10.203|5_._889_±_0.135|7589_._795_±_373.293|
|||Transformer|0_._985_±_0.075|274_._021_±_1.333|6_._364_±_0.201|4801_._964_±_59.175|



_Table 17._ **Variable-Dimensional** . Results for estimating the parameters of nonlinear regression models with ReLU activation function, with the expected _L_ 2 loss according to the posterior predictive as metric. 

For DeepSets, we use 4 layers each in the embedding network and the regression network, with a mean aggregation function, ReLU activation functions, and 627 hidden dimensions to make the number of parameters comparable to those in the Transformer model. 

### **E.3. RNN** 

For the recurrent neural network setup, we use the Gated Recurrent Unit (GRU). Similar to the above setups, we use a 4-layered GRU model with 256 hidden dimensions. While such an architecture is not permutation invariant, by training on tasks that require such invariance could encourage learning of solution structure that respects this invariance. 

### **E.4. Normalizing Flows** 

Assuming a Gaussian posterior distribution as the approximate often leads to poor results as the true posterior distribution can be far from the Gaussian shape. To allow for more flexible posterior distributions, we use normalizing flows (Kingma & Dhariwal, 2018; Kobyzev et al., 2020; Papamakarios et al., 2021; Rezende & Mohamed, 2015) for approximating _qφ_ ( **_θ_** _|D_ ) conditioned on the output of the summary network _hψ_ . Specifically, let _gν_ : **_z_** _�→_ **_θ_** be a diffeomorphism parameterized by a conditional invertible neural network (cINN) with network parameters _ν_ such that **_θ_** = _gν_ ( **_z_** ; _hψ_ ( _D_ )). With the change-of-variables formula it follows that _p_ ( **_θ_** ) = _p_ ( **_z_** ) ��det _∂∂_ **_z_**<sup>_gν_(</sup><sup>**_z_**;</sup><sup>_hψ_(</sup><sup>_D_))</sup> �� _−_ 1 = _p_ ( **_z_** ) _|_ det _Jν_ ( **_z_** ; _hψ_ ( _D_ )) _|−_ 1, where _Jν_ is the Jacobian matrix of _gν_ . Further, integration by substitution gives us _d_ **_θ_** = _|_ det _Jν_ ( **_z_** ; _hψ_ ( _D_ ) _|d_ **_z_** to rewrite the objective from eq. 14 as: 



As shown in BayesFlow (Radev et al., 2020), the normalizing flow _gν_ and the summary network _hψ_ can be trained simultaneously. The AllInOneBlock coupling block architecture of the FrEIA Python package (Ardizzone et al., 2018), 

24 

**Amortized In-Context Bayesian Posterior Estimation** 

|||||_L_2 _L_|_oss_(_↓_)||
|---|---|---|---|---|---|---|
||||_1-la_|_yer_|_2-la_|_yers_|
|**Objective**|_qφ_|**Model**|_1D_|_50D_|_1D_|_50D_|
||-|Random|33_._972_±_0.273|56_._891_±_0.213|50_._843_±_0.270|55_._383_±_0.124|
|Baseline|-|Optimization|0_._330_±_0.000|23_._257_±_0.029|0_._672_±_0.007|34_._900_±_0.120|
||-|Langevin|0_._296_±_0.003|26_._292_±_0.266|0_._356_±_0.006|36_._598_±_0.354|
||-|HMC|0_._404_±_0.005|19_._937_±_0.259|0_._654_±_0.014|32_._884_±_0.448|
|||GRU|34_._235_±_0.240|56_._947_±_0.339|50_._493_±_0.676|55_._633_±_0.396|
|Fwd-KL|n|DeepSets|33_._979_±_0.273|56_._900_±_0.213|50_._844_±_0.284|55_._390_±_0.132|
||ssia|Transformer|33_._998_±_0.499|56_._365_±_0.129|49_._940_±_0.317|55_._708_±_0.149|
||Gau|GRU|0_._813_±_0.036|18_._042_±_0.084|17_._506_±_10.364|26_._729_±_2.167|
|Rev-KL||DeepSets|0_._604_±_0.015|22_._582_±_2.064|24_._819_±_0.009|28_._247_±_0.006|
|||Transformer|0_._896_±_0.097|16_._726_±_0.724|2_._249_±_0.651|24_._442_±_0.643|
|||GRU|34_._989_±_0.475|56_._847_±_0.289|49_._915_±_0.980|55_._505_±_0.479|
|Fwd-KL||DeepSets|34_._857_±_0.278|56_._736_±_0.395|49_._594_±_0.622|55_._862_±_0.498|
||ow|Transformer|34_._878_±_0.829|55_._751_±_0.409|49_._309_±_0.651|55_._475_±_0.090|
||Fl|GRU|0_._969_±_0.038|17_._454_±_0.041|24_._796_±_0.028|28_._258_±_0.008|
|Rev-KL||DeepSets|0_._729_±_0.013|22_._888_±_2.610|24_._794_±_0.021|28_._253_±_0.003|
|||Transformer|0_._624_±_0.051|15_._971_±_0.075|3_._095_±_0.010|23_._740_±_0.343|



_Table 18._ **Variable-Dimensional** . Results for estimating the parameters of nonlinear regression models with TanH activation function, with the expected _L_ 2 loss according to the posterior predictive as metric. 

which is very similar to the RNVP style coupling block (Dinh et al., 2017), is used as the basis for the cINN. AllInOneBlock combines the most common architectural components, such as ActNorm, permutation, and affine coupling operations. 

For our experiments, 6 coupling blocks define the normalizing flow network, each with a 1 hidden-layered non-linear feed-forward subnetwork with ReLU non-linearity and 128 hidden dimensions. 

## **F. Experimental Details** 

Unless specified, we obtain a stream of datasets for all our experiments by simply sampling from the assumed probabilistic model, where the number of observations _n_ is sampled uniformly in the range [64 _,_ 128]. For efficient mini-batching over datasets with different cardinalities, we sample datasets with maximum cardinality (128) and implement different cardinalities by masking out different numbers of observations for different datasets whenever required. 

To evaluate both our proposed approach and the baselines, we compute an average of the predictive performances across 25 different posterior samples for each of the 100 fixed test datasets for all our experiments. That means for our proposed approach, we sample 25 different parameter vectors from the approximate posterior that we obtain. For MCMC, we rely on 25 MCMC samples, and for optimization, we train 25 different parameter vectors where the randomness comes from initialization. For the optimization baseline, we perform a quick hyperparameter search over the space _{_ 0 _._ 01 _,_ 003 _,_ 0 _._ 001 _,_ 0 _._ 0003 _,_ 0 _._ 0001 _,_ 0 _._ 00003 _}_ to pick the best learning rate that works for all of the test datasets and then use it to train for 1000 iterations using the Adam optimizer (Kingma & Ba, 2014). For the MCMC baseline, we use the open-sourced implementation of Langevin-based MCMC sampling<sup>2</sup> where we leave a chunk of the starting samples as burn-in and then start accepting samples after a regular interval (to not make them correlated). The details about the burn-in time and the regular interval for acceptance are provided in the corresponding experiments’ sections below. 

For our proposed approach of amortized inference, we do not consider explicit hyperparameter optimization and simply use a learning rate of 1e-4 with the Adam optimizer. For all experiments, we used linear scaling of the KL term in the training objectives as described in (Higgins et al., 2017), which we refer to as warmup. Furthermore, training details for 

> 2https://github.com/alisiahkoohi/Langevin-dynamics 

25 

**Amortized In-Context Bayesian Posterior Estimation** 

|||||_Accur_|_acy_(_↑_)||
|---|---|---|---|---|---|---|
||||_1-la_|_yer_|_2-la_|_yers_|
|**Objective**|_qφ_|**Model**|_2D_|_50D_|_2D_|_50D_|
||-|Random|49_._951_±_0.287|49_._904_±_0.281|50_._040_±_0.467|50_._044_±_0.239|
|Baseline|-|Optimization|96_._762_±_0.034|76_._139_±_0.033|96_._810_±_0.009|78_._225_±_0.129|
||-|Langevin|96_._077_±_0.027|70_._142_±_0.229|96_._564_±_0.113|71_._328_±_0.410|
||-|HMC|91_._734_±_0.152|67_._986_±_0.372|91_._336_±_0.591|71_._825_±_0.507|
|||GRU|59_._551_±_0.199|58_._637_±_0.250|60_._247_±_0.645|58_._862_±_0.065|
|Fwd-KL|n|DeepSets|49_._946_±_0.285|49_._910_±_0.282|50_._032_±_0.466|50_._040_±_0.242|
||ssia|Transformer|59_._887_±_0.235|58_._826_±_0.237|60_._552_±_0.398|58_._953_±_0.110|
||Gau|GRU|88_._822_±_0.471|68_._368_±_0.342|81_._884_±_1.450|67_._264_±_0.138|
|Rev-KL||DeepSets|91_._019_±_0.454|61_._732_±_0.111|82_._396_±_0.471|67_._320_±_0.143|
|||Transformer|89_._988_±_0.197|73_._744_±_0.319|83_._399_±_0.841|67_._167_±_0.028|
|||GRU|61_._179_±_0.833|60_._225_±_0.115|60_._400_±_1.019|59_._027_±_0.212|
|Fwd-KL||DeepSets|49_._568_±_0.230|50_._130_±_0.101|50_._356_±_0.773|49_._806_±_0.331|
||ow|Transformer|60_._886_±_0.252|60_._253_±_0.082|61_._694_±_0.314|60_._426_±_0.203|
||Fl|GRU|90_._363_±_0.709|66_._197_±_0.118|83_._443_±_0.619|69_._053_±_0.256|
|Rev-KL||DeepSets|89_._150_±_0.338|62_._939_±_0.112|79_._889_±_0.567|69_._015_±_0.147|
|||Transformer|91_._065_±_0.156|72_._581_±_0.117|83_._533_±_0.677|68_._933_±_0.120|



_Table 19._ **Variable-Dimensional** . Results for estimating the parameters of nonlinear classification models with ReLU activation function and two classes, with the expected accuracy according to the posterior predictive as metric. 

each experiment can be found below. 

### **F.1. Fixed-Dim** 

In this section, we provide the experimental details relevant to reproducing the results of Section 4. All the models are trained with streaming data from the underlying probabilistic model, such that every iteration of training sees a new set of datasets. Training is done with a batch size of 128, representing the number of datasets seen during one optimization step. Evaluations are done with 25 samples and we ensure that the test datasets used for each probabilistic model are the same across all the compared methods, i.e., baselines, forward KL, and reverse KL. We train the amortized inference model and the forward KL baselines for the following different probabilistic models: 

**Mean of Gaussian (GM):** We train the amortization models over 20 _,_ 000 iterations for both the 2-dimensional as well as the 100-dimensional setup. We use a linear warmup with 5000 iterations over which the weight of the KL term in our proposed approach scales linearly from 0 to 1. We use an identity covariance matrix for the data-generating process, but it can be easily extended to the case of correlated or diagonal covariance-based Gaussian distributions. 

**Gaussian Mixture Model (GMM):** We train the mixture model setup for 200 _,_ 000 iterations with 50 _,_ 000 iterations of warmup. We mainly experiment with 2-dimensional and 5-dimensional mixture models, with 2 and 5 mixture components for each setup. While we do use an identity covariance matrix for the data-generating process, again, it can be easily extended to other cases. 

**Linear Regression (LR):** The amortization models for this setup are trained for 50 _,_ 000 iterations with 12 _,_ 500 iterations of warmup. The feature dimensions considered for this task are 1 and 100 dimensions, and the predictive variance _σ_<sup>2</sup> is assumed to be known and set as 0 _._ 25. 

**Nonlinear Regression (NLR):** We train the setup for 100 _,_ 000 iterations with 25 _,_ 000 iterations consisting of warmup. The feature dimensionalities considered are 1-dimensional and 25-dimensional, and training is done with a known predictive variance similar to the LR setup. For the probabilistic model, we consider both a 1-layered and a 2-layered multi-layer perceptron (MLP) network with 32 hidden units in each, and either a RELU or TANH activation function. 

26 

**Amortized In-Context Bayesian Posterior Estimation** 

|||||_Accur_|_acy_(_↑_)||
|---|---|---|---|---|---|---|
||||_1-la_|_yer_|_2-la_|_yers_|
|**Objective**|_qφ_|**Model**|_2D_|_50D_|_2D_|_50D_|
||-|Random|19_._847_±_0.352|20_._052_±_0.066|19_._874_±_0.222|20_._028_±_0.106|
|Baseline|-|Optimization|94_._607_±_0.011|56_._091_±_0.158|93_._873_±_0.028|60_._253_±_0.053|
||-|Langevin|90_._815_±_0.341|46_._072_±_0.225|91_._849_±_0.088|49_._808_±_0.337|
||-|HMC|81_._145_±_0.303|44_._561_±_0.309|79_._559_±_0.483|50_._967_±_0.436|
|||GRU|30_._960_±_0.638|32_._164_±_0.151|31_._017_±_0.397|32_._224_±_0.108|
|Fwd-KL|n|DeepSets|19_._846_±_0.348|20_._053_±_0.063|19_._871_±_0.226|20_._032_±_0.104|
||ssia|Transformer|30_._652_±_0.401|32_._208_±_0.178|31_._148_±_0.429|32_._342_±_0.217|
||Gau|GRU|72_._874_±_0.113|37_._987_±_0.119|56_._999_±_0.599|29_._971_±_0.248|
|Rev-KL||DeepSets|69_._456_±_0.370|36_._712_±_0.249|55_._193_±_0.538|36_._417_±_9.779|
|||Transformer|73_._531_±_0.391|44_._702_±_0.165|57_._724_±_0.332|30_._175_±_0.177|
|||GRU|33_._232_±_0.607|33_._704_±_0.026|31_._937_±_0.483|32_._370_±_0.284|
|Fwd-KL||DeepSets|19_._898_±_0.156|19_._950_±_0.256|20_._062_±_0.347|20_._064_±_0.207|
||ow|Transformer|32_._916_±_0.194|33_._766_±_0.137|32_._374_±_0.301|33_._846_±_0.486|
||Fl|GRU|77_._997_±_0.663|38_._715_±_0.153|61_._947_±_0.294|51_._962_±_0.812|
|Rev-KL||DeepSets|68_._957_±_0.551|37_._123_±_0.108|51_._145_±_14.201|42_._707_±_12.882|
|||Transformer|77_._867_±_2.241|44_._156_±_0.485|57_._410_±_0.088|52_._077_±_0.077|



_Table 20._ **Fixed-Dimensional** . Results for estimating the parameters of nonlinear classification models with ReLU activation function and five classes, with the expected accuracy according to the posterior predictive as metric. 

**Linear Classification (LC):** We experiment with 2-dimensional and 100-dimensional setups with training done for 50 _,_ 000 iterations, out of which 12 _,_ 500 are used for warmup. Further, we train for both binary classification as well as a 5-class classification setup. 

**Nonlinear Classification (NLC):** We experiment with 2-dimensional and 25-dimensional setups with training done for 100 _,_ 000 iterations, out of which 2 _,_ 5000 are used for warmup. Further, we train for both binary classification as well as a 5-class classification setup. For the probabilistic model, we consider both a 1-layered and a 2-layered multi-layer perceptron (MLP) network with 32 hidden units in each, and either a RELU or TANH activation function. 

### **F.2. Variable-Dim** 

In this section, we provide the experimental details relevant to reproducing the results of Section 4. All the models are trained with streaming data from the underlying probabilistic model, such that every iteration of training sees a new set of datasets. Training is done with a batch size of 128, representing the number of datasets seen during one optimization step. Further, we ensure that the datasets sampled resemble a uniform distribution over the feature dimensions, ranging from 1-dimensional to the maximal dimensional setup. Evaluations are done with 25 samples and we ensure that the test datasets used for each probabilistic model are the same across all the compared methods, i.e., baselines, forward KL, and reverse KL. We train the amortized inference model and the forward KL baselines for the following different probabilistic models: 

**Mean of Gaussian (GM):** We train the amortization models over 50 _,_ 000 iterations using a linear warmup with 12 _,_ 5000 iterations over which the weight of the KL term in our proposed approach scales linearly from 0 to 1. We use an identity covariance matrix for the data-generating process, but it can be easily extended to the case of correlated or diagonal covariance-based Gaussian distributions. In this setup, we consider a maximum of 100 feature dimensions. 

**Gaussian Mixture Model (GMM):** We train the mixture model setup for 500 _,_ 000 iterations with 125 _,_ 000 iterations of warmup. We set the maximal feature dimensions as 5 and experiment with 2 and 5 mixture components. While we do use an identity covariance matrix for the data-generating process, again, it can be easily extended to other cases. 

**Linear Regression (LR):** The amortization models for this setup are trained for 100 _,_ 000 iterations with 25 _,_ 000 iterations of warmup. The maximal feature dimension considered for this task is 100-dimensional, and the predictive variance _σ_<sup>2</sup> is 

27 

**Amortized In-Context Bayesian Posterior Estimation** 

|||||_Accura_|_cy_(_↑_)||
|---|---|---|---|---|---|---|
||||_1-la_|_yer_|_2-la_|_yers_|
|**Objective**|_qφ_|**Model**|_2D_|_50D_|_2D_|_50D_|
||-|Random|50_._278_±_0.337|50_._028_±_0.064|50_._188_±_0.479|49_._982_±_0.084|
|Bli|-|Optimization|96_._943_±_0.012|68_._086_±_0.016|94_._444_±_0.024|63_._950_±_0.022|
|asene|-|Langevin|95_._143_±_0.094|61_._694_±_0.340|92_._719_±_0.016|57_._447_±_0.437|
||-|HMC|92_._489_±_0.338|59_._963_±_0.202|87_._548_±_0.094|56_._319_±_0.882|
|||GRU|50_._274_±_0.337|50_._023_±_0.060|50_._187_±_0.477|49_._992_±_0.072|
|Fwd-KL|n|DeepSets|50_._271_±_0.334|50_._024_±_0.061|50_._188_±_0.472|49_._984_±_0.077|
||ssia|Transformer|50_._273_±_0.336|50_._031_±_0.066|50_._191_±_0.471|49_._994_±_0.078|
||Gau|GRU|89_._270_±_0.272|50_._014_±_0.048|50_._191_±_0.463|49_._995_±_0.080|
|Rev-KL||DeepSets|89_._788_±_0.213|50_._018_±_0.061|50_._191_±_0.478|49_._977_±_0.075|
|||Transformer|89_._366_±_0.108|64_._926_±_0.260|50_._182_±_0.469|49_._986_±_0.071|
|||GRU|49_._651_±_0.040|50_._119_±_0.068|49_._987_±_0.015|49_._904_±_0.018|
|Fwd-KL||DeepSets|49_._639_±_0.031|50_._113_±_0.065|49_._988_±_0.022|49_._910_±_0.043|
||ow|Transformer|49_._636_±_0.040|50_._115_±_0.063|49_._989_±_0.017|49_._909_±_0.042|
||Fl|GRU|49_._769_±_0.141|50_._082_±_0.091|49_._915_±_0.070|50_._004_±_0.084|
|Rev-KL||DeepSets|49_._782_±_0.073|50_._080_±_0.087|49_._831_±_0.152|49_._994_±_0.078|
|||Transformer|63_._233_±_19.243|50_._026_±_0.047|49_._869_±_0.207|50_._036_±_0.056|



_Table 21._ **Variable-Dimensional** . Results for estimating the parameters of nonlinear classification models with TanH activation function and two classes, with the expected accuracy according to the posterior predictive as metric. 

### assumed to be known and set as 0 _._ 25. 

**Nonlinear Regression (NLR):** We train the setup for 250 _,_ 000 iterations with 62 _,_ 500 iterations consisting of warmup. The maximal feature dimension considered is 100-dimensional, and training is done with a known predictive variance similar to the LR setup. For the probabilistic model, we consider both a 1-layered and a 2-layered multi-layer perceptron (MLP) network with 32 hidden units in each, and either a RELU or TANH activation function. 

**Linear Classification (LC):** We experiment with a maximal 100-dimensional setup with training done for 100 _,_ 000 iterations, out of which 25 _,_ 000 are used for warmup. Further, we train for both binary classification as well as a 5-class classification setup. 

**Nonlinear Classification (NLC):** We experiment with a maximal 100-dimensional setup with training done for 250 _,_ 000 iterations, out of which 62 _,_ 500 are used for warmup. Further, we train for both binary classification as well as a 5-class classification setup. For the probabilistic model, we consider both a 1-layered and a 2-layered multi-layer perceptron (MLP) network with 32 hidden units in each, and either a RELU or TANH activation function. 

### **F.3. Model Misspecification** 

In this section, we provide the experimental details relevant to reproducing the results of Section 4. All models during this experiment are trained with streaming data from the currently used dataset-generating function _χ_ , such that every iteration of training sees a new batch of datasets. Training is done with a batch size of 128, representing the number of datasets seen during one optimization step. Evaluation for all models is done with 10 samples from each dataset-generator used in the respective experimental subsection and we ensure that the test datasets are the same across all compared methods, i.e., baselines, forward KL, and reverse KL. 

**Linear Regression Model:** The linear regression amortization models are trained following the training setting for linear regression fixed dimensionality, that is, 50 _,_ 000 training iterations with 12 _,_ 500 iterations of warmup. The feature dimension considered for this task is 1-dimension. The model is trained separately on datasets from three different generators _χ_ : linear regression, nonlinear regression, and Gaussian processes, and evaluated after training on test datasets from all of them. For training with datasets from the linear regression probabilistic model, the predictive variance _σ_<sup>2</sup> is assumed to be known and 

28 

**Amortized In-Context Bayesian Posterior Estimation** 



<!-- Start of picture text -->
Linear Regression Linear Classification<br>120 100<br>95<br>100<br>90<br>80 85<br>80<br>60<br>75<br>40<br>70<br>20 65<br>60<br>0<br>55<br>20 50<br>0 20 40 60 80 100 0 20 40 60 80 100<br>Dimensionality Dimensionality<br>Forward-KL Reverse-KL<br>Loss<br>Accuracy<br><!-- End of picture text -->

_Figure 4._ **Trends of Performance over different Dimensions in Variable Dimensionality Setup:** We see that our proposed reverse KL methodology outperforms the forward KL one. 



<!-- Start of picture text -->
Linear Regression Linear Classification<br>140 95<br>120 90<br>85<br>100<br>80<br>80<br>75<br>60<br>70<br>40<br>65<br>20<br>60<br>0 55<br>20 50<br>0 20 40 60 80 100 0 20 40 60 80 100<br>Dimensionality Dimensionality<br>DeepSets Transformer<br>Loss<br>Accuracy<br><!-- End of picture text -->

_Figure 5._ **Trends of Performance over different Dimensions in Variable Dimensionality Setup:** We see that transformer models generalize better to different dimensional inputs than DeepSets. 

29 

**Amortized In-Context Bayesian Posterior Estimation** 

|||||_Accura_|_cy_(_↑_)||
|---|---|---|---|---|---|---|
||||_1-la_|_yer_|_2-la_|_yers_|
|**Objective**|_qφ_|**Model**|_2D_|_50D_|_2D_|_50D_|
||-|Random|20_._041_±_0.136|20_._002_±_0.079|19_._914_±_0.111|19_._954_±_0.005|
|Baseline|-|Optimization|92_._059_±_0.012|40_._722_±_0.027|88_._848_±_0.005|34_._136_±_0.041|
||-|Langevin|88_._357_±_0.309|30_._941_±_0.097|83_._788_±_0.170|26_._538_±_0.206|
||-|HMC|79_._161_±_0.292|27_._508_±_0.379|74_._987_±_0.130|25_._377_±_0.246|
|||GRU|20_._264_±_0.139|20_._158_±_0.056|20_._215_±_0.138|20_._093_±_0.028|
|Fwd-KL|n|DeepSets|20_._042_±_0.133|20_._000_±_0.088|19_._916_±_0.114|19_._955_±_0.007|
||ssia|Transformer|20_._240_±_0.126|20_._153_±_0.070|20_._118_±_0.131|20_._089_±_0.022|
||Gau|GRU|66_._565_±_7.725|20_._011_±_0.099|19_._913_±_0.125|19_._954_±_0.022|
|Rev-KL||DeepSets|57_._294_±_0.362|20_._011_±_0.091|19_._915_±_0.115|19_._959_±_0.020|
|||Transformer|72_._865_±_1.340|22_._185_±_1.872|19_._911_±_0.124|19_._953_±_0.014|
|||GRU|19_._963_±_0.239|20_._176_±_0.056|19_._952_±_0.189|20_._156_±_0.079|
|Fwd-KL||DeepSets|19_._757_±_0.259|20_._045_±_0.071|19_._692_±_0.184|20_._019_±_0.069|
||ow|Transformer|19_._925_±_0.262|20_._185_±_0.059|19_._882_±_0.175|20_._159_±_0.070|
||Fl|GRU|67_._042_±_2.230|20_._065_±_0.060|19_._707_±_0.245|19_._989_±_0.101|
|Rev-KL||DeepSets|35_._220_±_10.870|20_._000_±_0.054|19_._739_±_0.216|19_._966_±_0.019|
|||Transformer|72_._798_±_1.049|20_._032_±_0.040|19_._752_±_0.276|20_._017_±_0.037|



_Table 22._ **Variable-Dimensional** . Results for estimating the parameters of nonlinear classification models with TanH activation function and five classes, with the expected accuracy according to the posterior predictive as metric. 

set as 0 _._ 25. The same variance is used for generating datasets from the nonlinear regression dataset generator with 1 layer, 32 hidden units, and TANH activation function. Lastly, datasets from the Gaussian process-based generator are sampled similarly, using the GPytorch library (Gardner et al., 2018), where datasets are sampled of varying cardinality, ranging from 64 to 128. We use a zero-mean Gaussian Process (GP) with a unit lengthscale radial-basis function (RBF) kernel serving as the covariance matrix. Further, we use a very small noise of _σ_<sup>2</sup> = 1e<sup>_−_6</sup> in the likelihood term of the GP. Forward KL training in this experiment can only be done when the amortization model and the dataset-generating function are the same: when we train on datasets from the linear regression-based _χ_ . Table 23 provides a detailed overview of the results. 

**Nonlinear Regression Models:** The nonlinear regression amortization models are trained following the training setting for nonlinear regression fixed dimensionality, that is, 100 _,_ 000 training iterations with 25 _,_ 000 iterations of warmup. Here, we consider two single-layer perceptions with 32 hidden units with a TANH activation function. The feature dimensionality considered is 1 dimension. We consider the same dataset-generating functions as in the misspecification experiment for a linear regression model above. However, the activation function used in the nonlinear regression dataset generator matches the activation function of the currently trained amortization model. In this case, forward KL training is possible in the two instances when trained on datasets from the corresponding nonlinear regression probabilistic model. A more detailed overview of the results can be found in Table 23 and 24. 

### **F.4. Tabular Experiments** 

For the tabular experiments, we train the amortized inference models for (non-)linear regression (NLR/LR) as well as (non-)linear classification (NLC/LC) with **_x_** _∼N_ ( **0** _,_ **I** ) as opposed to **_x_** _∼U_ ( _−_ **1** _,_ **1** ) in the dataset generating process _χ_ , with the rest of the settings the same as MAXIMUM-DIM experiments. For the nonlinear setups, we only consider the RELU case as it has seen predominant success in deep learning. Further, we only consider a 1-hidden layer neural network with 32 hidden dimensions in the probabilistic model. 

After having trained the amortized inference models, both for forward and reverse KL setups, we evaluate them on real-world tabular datasets. We first collect a subset of tabular datasets from the OpenML platform as outlined in Appendix G. Then, for each dataset, we perform a 5-fold cross-validation evaluation where the dataset is chunked into 5 bins, of which, at any time, 4 are used for training and one for evaluation. This procedure is repeated five times so that every chunk is used for 

30 

**Amortized In-Context Bayesian Posterior Estimation** 

||||||_L_2 _Loss_|(_↓_)||||
|---|---|---|---|---|---|---|---|---|---|
|**Objective**|_qφ_|**Model**|**Linear**|**Model**_|_**MLP**|**-TanH Data**|**MLP-Tan**|**H Model**_|_**Lin**|**ear Data**|_←χreal_|
||||_LR_|_NLR_|_GP_|_LR_|_NLR_|_GP_|_←χsim_|
||-|Random|-|17_._761_±_0.074|-|17_._847_±_0.355|-|-||
|Baseline|-|Optimization|-|1_._213_±_0.000|-|0_._360_±_0.001|-|-||
||-|Langevin|-|1_._218_±_0.002|-|0_._288_±_0.001|-|-||
||-|HMC|-|1_._216_±_0.002|-|0_._275_±_0.001|-|-||
|||GRU|2_._415_±_0.269|-|-|-|15_._632_±_0.283|-||
|Fwd-KL|n|DeepSets|1_._402_±_0.017|-|-|-|16_._046_±_0.393|-||
||ssia|Transformer|2_._216_±_0.097|-|-|-|15_._454_±_0.246|-||
||Gau|GRU|1_._766_±_0.044|1_._216_±_0.001|4_._566_±_0.199|0_._375_±_0.001|0_._386_±_0.002|0_._524_±_0.019||
|Rev-KL||DeepSets|1_._237_±_0.006|1_._216_±_0.001|3963_._694_±_5602.411|0_._365_±_0.000|0_._377_±_0.003|0_._385_±_0.011||
|||Transformer|1_._892_±_0.113|1_._226_±_0.001|4_._313_±_0.707|0_._367_±_0.006|0_._382_±_0.003|0_._458_±_0.048||
|||GRU|2_._180_±_0.024|-|-|-|9_._800_±_0.473|-||
|Fwd-KL||DeepSets|1_._713_±_0.244|-|-|-|15_._253_±_0.403|-||
||ow|Transformer|1_._632_±_0.070|-|-|-|7_._949_±_0.419|-||
||Fl|GRU|1_._830_±_0.081|1_._214_±_0.001|5_._690_±_0.196|0_._346_±_0.004|0_._349_±_0.001|0_._520_±_0.015||
|Rev-KL||DeepSets|1_._282_±_0.036|1_._218_±_0.001|11_._690_±_10.602|0_._339_±_0.003|0_._344_±_0.002|0_._397_±_0.026||
|||Transformer|1_._471_±_0.016|1_._226_±_0.004|5_._194_±_0.320|0_._346_±_0.002|0_._347_±_0.001|0_._480_±_0.030||



_Table 23._ **Model Misspecification** . Results for model misspecification under different training data _χsim_ , when evaluated under MLPTanH and Linear Data ( _χreal_ ), with the underlying model as a linear and MLP-TanH model respectively. 

### evaluation once. 

For each dataset, we normalize the observations and the targets so that they have zero mean and unit standard deviation. For the classification setups, we only normalize the inputs as the targets are categorical. For both forward KL and reverse KL amortization models, we initialize the probabilistic model from samples from the amortized model and then further finetune it via dataset-specific maximum a posteriori optimization. We repeat this setup over 25 different samples from the inference model. In contrast, for the optimization baseline, we initialize the probabilistic models’ parameters from _N_ (0 _, I_ ), which is the prior that we consider, and then train 25 such models with maximum a posteriori objective using Adam optimizer. 

While we see that the amortization models, particularly the reverse KL model, lead to much better initialization and convergence, it is important to note that the benefits vanish if we initialize using the Xavier-init initialization scheme. However, we believe that this is not a fair comparison as it means that we are considering a different prior now, while the amortized models were trained with _N_ (0 _, I_ ) prior. We defer the readers to the section below for additional discussion and experimental results. 

## **G. OpenML Datasets** 

For the tabular regression problems, we consider the suite of tasks outlined in _OpenML-CTR23 - A curated tabular regression benchmarking suite_ (Fischer et al., 2023), from which we further filter out datasets that have more than 2000 examples and 100 features. We also remove datasets with missing information and NaNs. Similarly, we consider the _OpenML-CC18 Curated Classification benchmark_ (Bischl et al., 2019) suite of tasks for classification and perform a similar filtering algorithm. We remove datasets with missing information and NaNs, as well as datasets with more than 2000 examples and 100 features. In addition, we also exclude datasets that are not made for binary classification. At the end of this filtering mechanism, we end up with 9 regression and 13 classification problems, and our dataset filtration pipeline is heavily inspired by (Hollmann et al., 2022). We provide the datasets considered for both regression and classification below: 

**Regression** : AIRFOIL SELF NOISE, CONCRETE COMPRESSIVE STRENGTH, ENERGY EFFICIENCY, SOLAR FLARE, STUDENT PERFORMANCE POR, QSAR FISH TOXICITY, RED WINE, SOCMOB and CARS. 

31 

**Amortized In-Context Bayesian Posterior Estimation** 

||||||_L_2 _L_|_oss_(_↓_)||||
|---|---|---|---|---|---|---|---|---|---|
|**Objective**|_qφ_|**Model**|**Line**|**ar Model**_|_**GP**|**Data**|**MLP-**|**TanH Model**_|_|**GP Data**|_←χreal_|
||||_LR_|_NLR_|_GP_|_LR_|_NLR_|_GP_|_←χsim_|
||-|Random|-|-|2_._681_±_0.089|-|-|16_._236_±_0.381||
|Baseline|-|Optimization|-|-|0_._263_±_0.000|-|-|0_._007_±_0.000||
||-|Langevin|-|-|0_._266_±_0.001|-|-|0_._022_±_0.001||
||-|HMC|-|-|0_._266_±_0.000|-|-|0_._090_±_0.002||
|||GRU|0_._268_±_0.000|-|-|-|14_._077_±_0.368|-||
|Fwd-KL|n|DeepSets|0_._269_±_0.001|-|-|-|14_._756_±_0.280|-||
||ssia|Transformer|0_._270_±_0.001|-|-|-|14_._733_±_0.513|-||
||Gau|GRU|0_._268_±_0.000|0_._269_±_0.000|0_._266_±_0.000|0_._334_±_0.005|0_._157_±_0.003|0_._080_±_0.003||
|Rev-KL||DeepSets|0_._269_±_0.000|0_._269_±_0.000|0_._265_±_0.000|0_._331_±_0.003|0_._146_±_0.002|0_._063_±_0.000||
|||Transformer|0_._269_±_0.000|0_._269_±_0.000|0_._267_±_0.000|0_._310_±_0.013|0_._155_±_0.006|0_._066_±_0.004||
|||GRU|0_._268_±_0.000|-|-|-|9_._756_±_0.192|-||
|Fwd-KL||DeepSets|0_._269_±_0.001|-|-|-|14_._345_±_0.628|-||
||ow|Transformer|0_._269_±_0.000|-|-|-|8_._557_±_0.561|-||
||Fl|GRU|0_._268_±_0.000|0_._270_±_0.001|0_._266_±_0.000|0_._289_±_0.011|0_._120_±_0.004|0_._059_±_0.003||
|Rev-KL||DeepSets|0_._269_±_0.000|0_._269_±_0.001|0_._266_±_0.000|0_._270_±_0.008|0_._115_±_0.002|0_._059_±_0.002||
|||Transformer|0_._269_±_0.001|0_._270_±_0.000|0_._267_±_0.000|0_._293_±_0.008|0_._120_±_0.005|0_._055_±_0.002||



_Table 24._ **Model Misspecification** . Results for model misspecification under different training data _χsim_ , when evaluated under GP Data ( _χreal_ ), with the underlying model as a linear and MLP-TanH model respectively. 

**Classification** : CREDIT-G, DIABETES, TIC-TAC-TOE, PC4, PC3, KC2, PC1, BANKNOTE-AUTHENTICATION, BLOODTRANSFUSION-SERVICE-CENTER, ILPD, QSAR-BIODEG, WDBC and CLIMATE-MODEL-SIMULATION-CRASHES. 

## **H. Additional Experiments** 

In this section, we outline the additional experiments we conducted in obtaining Bayesian posteriors for the different probabilistic models for different hyperparameters and their downstream uses. We provide a comprehensive account of the results in the relevant sections below. 

### **H.1. Fixed-Dim** 

While we highlighted the results with the Gaussian mixture model and classification settings with only 2 clusters/classes, we also conducted experiments with an increased number of clusters and classes, making the problem even more challenging. Tables 7-14 shows that both forward and reverse KL methods perform reasonably, with forward KL struggling more in challenging scenarios. 

Next, we also consider harder tasks based on the Bayesian Neural Network (BNN) paradigm, where we consider nonlinear regression and classification setups with different activation functions: TANH and RELU for a 1-layered and 2-layered BNN. We provide the results of our experiments in Tables 7-14. The results indicate that forward KL approaches struggle a lot in such scenarios, often achieving performance comparable to random chance. On the contrary, we see that reverse KL-based amortization leads to performances often similar to dataset-specific optimization, thereby showing the superiority of our proposed method. 

### **H.2. Variable-Dim** 

Our experiments on variable dimensional datasets can be evaluated for arbitrary feature cardinality, of which we show a few examples in Section 4. In this section, we provide results for additional dimensionality setups. In particular, we refer the readers to Tables 15-22, which contain experimental results w.r.t different dimensionalities (e.g. 50D setup), 

32 

**Amortized In-Context Bayesian Posterior Estimation** 



<!-- Start of picture text -->
Linear Regression Linear Classification<br>100 100<br>95<br>80<br>90<br>60 85<br>80<br>40<br>75<br>20 70<br>65<br>0<br>60<br>20 55<br>0 20 40 60 80 100 0 20 40 60 80 100<br>Dimensionality Dimensionality<br>Gaussian Normalizing Flow<br>Loss<br>Accuracy<br><!-- End of picture text -->

_Figure 6._ **Trends of Performance over different Dimensions in Variable Dimensionality Setup:** We see that normalizing flows leads to similar performances than Gaussian based variational approximation. 

as well as different number of clusters and classes, respectively, for the GMM and LC setup. Throughout, we see that amortization leads to reasonable performance, and in particular, we see forward KL-based amortization starting to struggle in high-dimensional setups. 

Again, to make the setup more challenging, we consider the Bayesian Neural Network (BNN) setup where we consider nonlinear regression and classification with different activation functions: TANH and RELU for a 1-layered and 2-layered BNN, but which can now be tested for an arbitrary number of input features. Our experiments are highlighted in Tables 15-22, for 1- and 2-layered BNN, among others. In such complex multi-modal and complicated setups, forward KL often performs comparable to random chance and thus does not lead to any good approximation of the true posterior distribution. On the other hand, our proposed method indeed leads to good predictive performance, often comparable to dataset-specific optimization routines. 

### **H.3. Model Misspecification** 

As a representative of the results on model misspecification (Section 4), we highlighted training and evaluation of the amortization models with Transformer backbone on a subset of in-distribution and OoD data-generating functions (Table 4) to show superiority in generalization of reverse KL trained system vs. forward KL based ones on OoD data but also to highlight that training a misspecified amortization model on OoD datasets directly with our approach results in even better posterior predictive performance. 

### **H.4. Tabular Experiments** 

As a case of extreme OoD generalization, we test our amortized models trained to handle variable feature dimensions on the suite of regression and classification problems that we filtered out from the OpenML platform, as outlined in Appendix G. We consider both linear and nonlinear probabilistic models to tackle the regression and binary classification setups, which lead to predicting the parameters of a linear regression/classification model and a small nonlinear neural network based on RELU activation function. Further, we also perform the analysis with a diagonal Gaussian assumption and a normalizing flow-based amortization model trained with both a forward and reverse KL objective. We provide the results on the regression problems in (a) linear model with diagonal Gaussian assumption (Figure 7), (b) linear model with normalizing flow (Figure 8), (c) nonlinear model with diagonal Gaussian assumption (Figure 9), and (d) nonlinear model with normalizing flow (Figure 10). The results of the classification problems are shown in (a) linear model with diagonal Gaussian assumption (Figure 11), (b) 

33 

**Amortized In-Context Bayesian Posterior Estimation** 



<!-- Start of picture text -->
airfoil_self_noise concrete_compressive_strength energy_efficiency solar_flare<br>20 22<br>12 18 20<br>20<br>16<br>10 14 15 18<br>8 12 16<br>10 10 14<br>6<br>8<br>12<br>4 6 5<br>10<br>4<br>0<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>student_performance_por QSAR_fish_toxicity red_wine socmob<br>16 14<br>25.0<br>60 14 22.5 12<br>50 1210 20.017.5 10 Experiment NamePrior Initialization<br>8 Xavier Initialization<br>40 8 15.0 Fwd-KL Initialization<br>6 Rev-KL Initialization<br>6 12.5<br>30 10.0 4<br>4<br>7.5 2<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>Iteration Iteration Iteration<br>cars<br>40<br>35<br>30<br>25<br>20<br>15<br>10<br>5<br>0<br>0 500 1000 1500 2000 2500<br>Iteration<br>Performance<br>Performance<br>Performance<br><!-- End of picture text -->

_Figure 7._ **Tabular Experiments** _|_ **Linear Regression with Diagonal Gaussian** : For every regression dataset from the OpenML platform considered, we initialize the parameters of a linear regression-based probabilistic model with the amortized inference models which were trained with a diagonal Gaussian assumption. The parameters are then further trained with maximum-a-posteriori (MAP) estimate with gradient descent. Reverse and Forward KL denote initialization with the correspondingly trained amortized model. Prior refers to a MAP-based optimization baseline initialized from the prior _N_ (0 _, I_ ), whereas Xavier refers to initialization from the Xavier initialization scheme. 

34 

**Amortized In-Context Bayesian Posterior Estimation** 



<!-- Start of picture text -->
airfoil_self_noise concrete_compressive_strength energy_efficiency solar_flare<br>20 22<br>12 18 20<br>20<br>16<br>10<br>14 15 18<br>8 12 16<br>10 10 14<br>6<br>8<br>4 6 5 12<br>4 10<br>2 0<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>student_performance_por QSAR_fish_toxicity red_wine socmob<br>16 14<br>25.0<br>60 14 22.5 12<br>50 1210 20.017.5 10 Experiment NamePrior Initialization<br>8 Xavier Initialization<br>40 8 15.0 Fwd-KL Initialization<br>6 12.5 6 Rev-KL Initialization<br>30 10.0 4<br>4<br>7.5<br>2<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>Iteration Iteration Iteration<br>cars<br>40<br>35<br>30<br>25<br>20<br>15<br>10<br>5<br>0 500 1000 1500 2000 2500<br>Iteration<br>Performance<br>Performance<br>Performance<br><!-- End of picture text -->

_Figure 8._ **Tabular Experiments** _|_ **Linear Regression with Normalizing Flow** : For every regression dataset from the OpenML platform considered, we initialize the parameters of a linear regression-based probabilistic model with the amortized inference models which were trained with a normalizing flow-based model. The parameters are then further trained with maximum-a-posteriori (MAP) estimate with gradient descent. Reverse and Forward KL denote initialization with the correspondingly trained amortized model. Prior refers to a MAP-based optimization baseline initialized from the prior _N_ (0 _, I_ ), whereas Xavier refers to initialization from the Xavier initialization scheme. 

35 

**Amortized In-Context Bayesian Posterior Estimation** 



<!-- Start of picture text -->
airfoil_self_noise concrete_compressive_strength energy_efficiency solar_flare<br>175 175<br>100 150 150 175<br>150<br>80 125 125<br>125<br>60 100 100 100<br>75 75<br>40 75<br>50 50<br>50<br>20 25 25<br>25<br>0 0 0<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>student_performance_por QSAR_fish_toxicity red_wine socmob<br>600<br>120 200 100<br>500<br>100<br>80<br>400 150<br>80 Experiment Name<br>60 Prior Initialization<br>300 60 100 Xavier InitializationFwd-KL Initialization<br>200 40 40 Rev-KL Initialization<br>100 20 50 20<br>0 0 0 0<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>Iteration Iteration Iteration<br>cars<br>350<br>300<br>250<br>200<br>150<br>100<br>50<br>0<br>0 500 1000 1500 2000 2500<br>Iteration<br>Performance<br>Performance<br>Performance<br><!-- End of picture text -->

_Figure 9._ **Tabular Experiments** _|_ **Nonlinear Regression with Diagonal Gaussian** : For every regression dataset from the OpenML platform considered, we initialize the parameters of a nonlinear regression-based probabilistic model with the amortized inference models which were trained with a diagonal Gaussian assumption. The parameters are then further trained with maximum-a-posteriori (MAP) estimate with gradient descent. Reverse and Forward KL denote initialization with the correspondingly trained amortized model. Prior refers to a MAP-based optimization baseline initialized from the prior _N_ (0 _, I_ ), whereas Xavier refers to initialization from the Xavier initialization scheme. 

36 

**Amortized In-Context Bayesian Posterior Estimation** 



<!-- Start of picture text -->
airfoil_self_noise concrete_compressive_strength energy_efficiency solar_flare<br>175 175<br>100 150 150 175<br>150<br>80 125 125<br>125<br>60 100 100 100<br>75 75<br>40 75<br>50 50<br>50<br>20 25 25<br>25<br>0 0 0<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>student_performance_por QSAR_fish_toxicity red_wine socmob<br>600<br>120 200 100<br>500<br>100<br>80<br>400 150<br>80 Experiment Name<br>60 Prior Initialization<br>300 60 100 Xavier InitializationFwd-KL Initialization<br>200 40 40 Rev-KL Initialization<br>100 20 50 20<br>0 0 0 0<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>Iteration Iteration Iteration<br>cars<br>350<br>300<br>250<br>200<br>150<br>100<br>50<br>0<br>0 500 1000 1500 2000 2500<br>Iteration<br>Performance<br>Performance<br>Performance<br><!-- End of picture text -->

_Figure 10._ **Tabular Experiments** _|_ **Nonlinear Regression with Normalizing Flow** : For every regression dataset from the OpenML platform considered, we initialize the parameters of a nonlinear regression-based probabilistic model with the amortized inference models which were trained with a normalizing flow-based model. The parameters are then further trained with maximum-a-posteriori (MAP) estimate with gradient descent. Reverse and Forward KL denote initialization with the correspondingly trained amortized model. Prior refers to a MAP-based optimization baseline initialized from the prior _N_ (0 _, I_ ), whereas Xavier refers to initialization from the Xavier initialization scheme. 

37 

**Amortized In-Context Bayesian Posterior Estimation** 

linear model with normalizing flow (Figure 12), (c) nonlinear model with diagonal Gaussian assumption (Figure 13), and (d) nonlinear model with normalizing flow (Figure 14). Our experiments indicate that initializing with amortized models leads to better performance and training than models trained via maximum a-posteriori approach and initialized with the prior, i.e., _N_ (0 _, I_ ). 

We do provide an additional baseline of initializing with XAVIER-INIT initialization, which often leads to faster convergence; however, as we consider the prior to be a unit normal, this is an unfair baseline as we assume the weights to be initialized from a different prior. We leave the work of computing Bayesian posteriors with different priors and testing an amortized Bayesian model with XAVIER-INIT prior for the future. 

In addition to those experiments, we also conducted a broader range of experiments utilizing DeepSets as the backbone, various OoD data-generating functions for training and evaluation of the reverse KL system, and an additional nonlinear regression model with RELU activation function. For a comprehensive description of these experiments and the complete setup, please refer to Section F.3. We considered two probabilistic models, including a linear regression model and a nonlinear regression models utilizing the TANH activation function. The detailed results for each model can be found in Tables 23 and 24. 

In all experiments, reverse KL outperforms forward KL trained amortization models in in-distribution performance and excels in posterior prediction on OoD datasets. Although the significant difference in posterior prediction performance of forward vs. reverse KL in cases where the underlying model is nonlinear was already mentioned in previous experiments, here, reverse KL-trained models also excel in evaluations of posterior prediction for the linear regression model. Although only by a margin, in the case of approximating the posterior of the simpler linear regression model, a diagonal Gaussianshaped posterior shows the best posterior prediction results when evaluated on OoD datasets from the nonlinear regression dataset generating function. In almost all other experiments, the posterior prediction performance could be enhanced when we used the normalizing flow based posterior. A definitive conclusion cannot be drawn regarding the superiority of one backbone over the other, i.e. between DeepSets or Transformer. However, amortization models with DeepSets as the backbone tend towards better generalization regarding OoD datasets. 

38 

**Amortized In-Context Bayesian Posterior Estimation** 



<!-- Start of picture text -->
75 credit-g 75 diabetes tic-tac-toe 90 pc4<br>65.0<br>70 70<br>62.5 80<br>65 65 60.0<br>70<br>60 57.5<br>60<br>55 55.0 60<br>55 52.5<br>50<br>50 50.0 50<br>45<br>47.5<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>pc3 kc2 pc1 100 banknote-authentication<br>90<br>80<br>90 90<br>80<br>70 80 80<br>70 70<br>60 70<br>60 60 60<br>50<br>50<br>50 50<br>40 40<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 40 0 500 1000 1500 2000 2500 Experiment NamePrior Initialization<br>blood-transfusion-service-center ilpd qsar-biodeg wdbc Xavier InitializationFwd-KL Initialization<br>Rev-KL Initialization<br>75 70 80 90<br>70 75<br>65 80<br>65 70<br>60 60 65 70<br>55 55 60 60<br>55<br>50 50<br>50 50<br>45<br>40<br>45 45<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>Iteration Iteration Iteration<br>climate-model-simulation-crashes<br>90<br>80<br>70<br>60<br>50<br>0 500 1000 1500 2000 2500<br>Iteration<br>Performance<br>Performance<br>Performance<br>Performance<br><!-- End of picture text -->

_Figure 11._ **Tabular Experiments** _|_ **Linear Classification with Diagonal Gaussian** : For every classification dataset from the OpenML platform considered, we initialize the parameters of a linear classification-based probabilistic model with the amortized inference models which were trained with a diagonal Gaussian assumption. The parameters are then further trained with maximum-a-posteriori (MAP) estimate with gradient descent. Reverse and Forward KL denote initialization with the correspondingly trained amortized model. Prior refers to a MAP-based optimization baseline initialized from the prior _N_ (0 _, I_ ), whereas Xavier refers to initialization from the Xavier initialization scheme. 

39 

**Amortized In-Context Bayesian Posterior Estimation** 



<!-- Start of picture text -->
credit-g diabetes tic-tac-toe pc4<br>90<br>75 75 65<br>70 70 80<br>60<br>65 65<br>70<br>60 60 55<br>55 60<br>55 50<br>50<br>50 50<br>45 45<br>45<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>pc3 kc2 pc1 banknote-authentication<br>100<br>90 80<br>90<br>90<br>80 70 80<br>80<br>60<br>70 70<br>70<br>50 60<br>60 60<br>40 50<br>50 30 40 50<br>0 500 1000 1500 2000 2500 20 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 40 0 500 1000 1500 2000 2500 Experiment NamePrior Initialization<br>blood-transfusion-service-center ilpd qsar-biodeg wdbc Xavier InitializationFwd-KL Initialization<br>75 70 80 90 Rev-KL Initialization<br>75<br>70<br>65 80<br>70<br>65<br>60 65 70<br>60<br>55 55 60 60<br>55<br>50 50<br>50 50<br>45<br>40<br>45 45<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>Iteration Iteration Iteration<br>climate-model-simulation-crashes<br>90<br>80<br>70<br>60<br>50<br>0 500 1000 1500 2000 2500<br>Iteration<br>Performance<br>Performance<br>Performance<br>Performance<br><!-- End of picture text -->

_Figure 12._ **Tabular Experiments** _|_ **Linear Classification with Normalizing Flow** : For every classification dataset from the OpenML platform considered, we initialize the parameters of a linear classification-based probabilistic model with the amortized inference models which were trained with a normalizing flow-based model. The parameters are then further trained with maximum-a-posteriori (MAP) estimate with gradient descent. Reverse and Forward KL denote initialization with the correspondingly trained amortized model. Prior refers to a MAP-based optimization baseline initialized from the prior _N_ (0 _, I_ ), whereas Xavier refers to initialization from the Xavier initialization scheme. 

40 

**Amortized In-Context Bayesian Posterior Estimation** 



<!-- Start of picture text -->
credit-g diabetes 67.5 tic-tac-toe pc4<br>90<br>70 70 65.0<br>62.5 80<br>65 65<br>60.0<br>60 60 57.5 70<br>55.0<br>55 55 60<br>52.5<br>50<br>50 50.0 50<br>45 47.5<br>45 40<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>pc3 kc2 pc1 banknote-authentication<br>90 80 90<br>90<br>80 70 80 80<br>70 70 70<br>60<br>60 60 60<br>50<br>50<br>50 50<br>40 40<br>40 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 40 0 500 1000 1500 2000 2500 Experiment NamePrior Initialization<br>80 blood-transfusion-service-center 75 ilpd 80 qsar-biodeg 90 wdbc Xavier InitializationFwd-KL InitializationRev-KL Initialization<br>75 70 75<br>80<br>70<br>65 70<br>65 60 65 70<br>60<br>60<br>55 55 60<br>55<br>50 50<br>50 50<br>45<br>45 45<br>40<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>Iteration Iteration Iteration<br>climate-model-simulation-crashes<br>90<br>80<br>70<br>60<br>50<br>40<br>0 500 1000 1500 2000 2500<br>Iteration<br>Performance<br>Performance<br>Performance<br>Performance<br><!-- End of picture text -->

_Figure 13._ **Tabular Experiments** _|_ **Nonlinear Classification with Diagonal Gaussian** : For every classification dataset from the OpenML platform considered, we initialize the parameters of a nonlinear classification-based probabilistic model with the amortized inference models which were trained with a diagonal Gaussian assumption. The parameters are then further trained with maximum-a-posteriori (MAP) estimate with gradient descent. Reverse and Forward KL denote initialization with the correspondingly trained amortized model. Prior refers to a MAP-based optimization baseline initialized from the prior _N_ (0 _, I_ ), whereas Xavier refers to initialization from the Xavier initialization scheme. 

41 

**Amortized In-Context Bayesian Posterior Estimation** 



<!-- Start of picture text -->
credit-g diabetes 67.5 tic-tac-toe pc4<br>90<br>70 70 65.0<br>65 65 62.5 80<br>60.0<br>60 60 57.5 70<br>55.0<br>55 55 60<br>52.5<br>50 50 50.0 50<br>45 47.5<br>45 40<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>pc3 kc2 pc1 banknote-authentication<br>90 80 90<br>90<br>80 70 80 80<br>70 70 70<br>60<br>60 60 60<br>50<br>50<br>50 50<br>40 40<br>40 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 40 0 500 1000 1500 2000 2500 Experiment NamePrior Initialization<br>80 blood-transfusion-service-center 75 ilpd qsar-biodeg wdbc Xavier InitializationFwd-KL Initialization<br>80 Rev-KL Initialization<br>75 70 90<br>75<br>70<br>65 70 80<br>65<br>60 65 70<br>60<br>60<br>55 55 60<br>55<br>50 50 50<br>50<br>45<br>45 45 40<br>40<br>0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500 0 500 1000 1500 2000 2500<br>Iteration Iteration Iteration<br>climate-model-simulation-crashes<br>90<br>80<br>70<br>60<br>50<br>40<br>0 500 1000 1500 2000 2500<br>Iteration<br>Performance<br>Performance<br>Performance<br>Performance<br><!-- End of picture text -->

_Figure 14._ **Tabular Experiments** _|_ **Nonlinear Classification with Normalizing Flow** : For every classification dataset from the OpenML platform considered, we initialize the parameters of a linear classification-based probabilistic model with the amortized inference models which were trained with a normalizing flow-based model. The parameters are then further trained with maximum-a-posteriori (MAP) estimate with gradient descent. Reverse and Forward KL denote initialization with the correspondingly trained amortized model. Prior refers to a MAP-based optimization baseline initialized from the prior _N_ (0 _, I_ ), whereas Xavier refers to initialization from the Xavier initialization scheme. 

42 

