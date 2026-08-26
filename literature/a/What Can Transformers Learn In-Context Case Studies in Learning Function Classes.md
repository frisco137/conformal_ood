# What Can Transformers Learn In-Context? A Case Study of Simple Function Classes 

Shivam Garg<sup>*</sup> Dimitris Tsipras<sup>*</sup> Stanford University Stanford University `shivamg@cs.stanford.edu tsipras@stanford.edu` 

Percy Liang Gregory Valiant Stanford University Stanford University `pliang@cs.stanford.edu valiant@stanford.edu` 

#### **Abstract** 

In-context learning refers to the ability of a model to condition on a prompt sequence consisting of in-context examples (input-output pairs corresponding to some task) along with a new query input, and generate the corresponding output. Crucially, in-context learning happens only at inference time without any parameter updates to the model. While large language models such as GPT-3 exhibit some ability to perform in-context learning, it is unclear what the relationship is between tasks on which this succeeds and what is present in the training data. To make progress towards understanding in-context learning, we consider the well-defined problem of training a model to in-context learn a function class (e.g., linear functions): that is, given data derived from some functions in the class, can we train a model to in-context learn “most” functions from this class? We show empirically that standard Transformers can be trained from scratch to perform in-context learning of linear functions—that is, the trained model is able to learn unseen linear functions from in-context examples with performance comparable to the optimal least squares estimator. In fact, in-context learning is possible even under two forms of distribution shift: (i) between the training data of the model and inference-time prompts, and (ii) between the in-context examples and the query input during inference. We also show that we can train Transformers to in-context learn more complex function classes—namely sparse linear functions, two-layer neural networks, and decision trees—with performance that matches or exceeds task-specific learning algorithms.<sup>1</sup> 

## **1 Introduction** 

Large language models such as GPT-3 [Brown et al., 2020] are able to perform _in-context learning_ : given a prompt containing examples from a task (input-output pairs) and a new query input, the language model can generate the corresponding output. For example, these models are able to produce English translations of French words after being prompted on a few such translations, e.g.: 



This capability is quite intriguing as it allows models to adapt to a wide range of downstream tasks on-thefly—i.e., without the need to perform any parameter updates after the model is trained [Brown et al., 2020, 

> *Equal contribution. 

> 1Our code and models are available at `https://github.com/dtsip/in-context-learning` . 

1 

Lieber et al., 2021, Rae et al., 2021, Black et al., 2022]. However, it is unclear to what extent these models have developed the ability to learn _new tasks_ from in-context examples alone as opposed to simply indexing into a vast set of known tasks from the training data (e.g., see Min et al. [2022]).<sup>2</sup> 

To make progress towards understanding in-context learning, we consider the well-defined problem of learning a _function class_ from in-context examples. That is, we say that a model can in-context learn a function class _F_ if, for “most” functions _f ∈F_ , the model can approximate _f_ ( _x_ query) for a new query input _x_ query by conditioning on a prompt sequence ( _x_ 1, _f_ ( _x_ 1), . . . , _xk_ , _f_ ( _xk_ ), _x_ query) containing in-context examples and the query input. 

Formally, let _DX_ be a distribution over inputs and _DF_ be a distribution over functions in _F_ . A prompt _P_ is a sequence ( _x_ 1, _f_ ( _x_ 1), . . . , _xk_ , _f_ ( _xk_ ), _x_ query) where inputs (i.e., _xi_ and _x_ query) are drawn i.i.d. from _DX_ and _f_ is drawn from _DF_ . We say that a model _M_ can in-context learn the function class _F_ up to _ϵ_ , with respect to ( _DF_ , _DX_ ), if it can predict _f_ ( _x_ query) with an average error 



where _ℓ_ ( _·_ , _·_ ) is some appropriate loss function, such as the squared error. Within this framework, we can now concretely ask: 

_Can we train a model to in-context learn a certain function class?_ 

Note that, here, being able to in-context learn a function class is a property of model _M_ alone, independent of how it was trained. _Training_ such a model can be viewed as an instance of _meta-learning_ [Schmidhuber, 1987, Naik and Mammone, 1992, Thrun and Pratt, 2012], a general paradigm for learning a model or method that can learn from data. 

We empirically study this question, focusing on Transformer models [Vaswani et al., 2017, Radford et al., 2018]—the architecture behind recent large language models—trained from scratch to in-context learn a range of simple, well-defined function classes (e.g. linear functions). Specifically, we sample prompts containing in-context examples (input-output pairs) generated using functions in a given class and train models to predict the function value at the corresponding query inputs. (see illustration in Figure 1). Our findings are as follows. 



<!-- Start of picture text -->
Training data Inference<br>w 1, …,  wn i .∼ i . d . N (0, Id ) wtest  ∼ N (0, Id )<br>y ( 1 i ) y ( 2 i ) … y ( k i )<br>wtest ⊤ x<br>wi ⊤ x ( x 2 ( i ), … y 2 ( i )) auto-regressive model ( x 2, y 2)<br>…<br>( xk ( i ), y k ( i ))<br>… ( xj , ?)<br>( x 1 ( i ), y 1 ( i )) x 1 ( i ) y 1 ( i ) x 2 ( i ) y 2 ( i ) xk ( i ) yk ( i ) ( x 1, y 1)<br>i  = 1,…,  n<br><!-- End of picture text -->

Figure 1: _Can we train a model that in-context learns a function class (here linear functions)?_ We train Transformers by repeatedly sampling a random function _f_ from that class, as well as random inputs _x_ 1, . . . , _xk_ and training the model to predict each _f_ ( _xi_ ) given the prompt _x_ 1, _f_ ( _x_ 1), . . . , _xi−_ 1, _f_ ( _xi−_ 1), _xi_ (wrt squared loss). Then, during inference, we evaluate the model’s ability to predict accurately on new, _unseen_ functions. 

> 2The term “in-context learning” has also been used to refer to a more general notion of learning from a prompt [Olsson et al., 2022]. In this work, we focus on the standard notion which refers to learning a task/function given in-context examples [Brown et al., 2020]. 

2 

**Transformers can in-context learn linear functions.** We show empirically that we can train a standard Transformer from scratch to in-context learn the class of linear functions, with respect to the input distribution _DX_ being an isotropic Gaussian in 20 dimensions, and _DF_ being the distribution over linear functions with weight vectors drawn from an isotropic Gaussian (the model was trained on prompts generated from the same distributions _DX_ and _DF_ ). Specifically, the trained model achieves error comparable to the optimal least squares estimator, suggesting that it encodes an effective learning algorithm, at least for the distribution used to generate the training prompts. 

**Generalization to out-of-distribution prompts.** To understand the extent to which the trained model encodes an algorithm that works beyond the training distribution, we consider in-context learning under two types of distribution shifts: (a) a shift between the prompts encountered during training and inference (e.g., training on prompts without any noise in the in-context example outputs but testing with noisy outputs), (b) a shift between the in-context examples and the query input during inference (e.g., in-context examples lie in one orthant and the query input lies in another). We find that the performance of our model is quite robust to such shifts, indicating that it has learned to perform linear regression with some generality. 

**More complex function classes.** We also consider the function classes of 3-sparse linear functions, two-layer ReLU neural networks with 100 hidden units, and decision trees of depth 4, all with 20 dimensional inputs. We show that we can again train Transformer models that can in-context learn these classes (with respect to isotropic Gaussian inputs and appropriately defined distributions over functions). For sparse linear functions, the trained model is able to exploit sparsity, obtaining performance better than least squares and comparable to Lasso. For neural networks, the corresponding model is able to obtain performance comparable to neural networks of the same architecture trained using gradient descent on in-context examples. Moreover, it is also able to in-context learn linear functions. For decision trees, the trained model can learn unseen trees with as few as 100 in-context examples, whereas greedy learning and tree boosting algorithms are unable to achieve competitive performance (for the distribution of prompts studies here). Note that learning these function classes requires involved algorithms (e.g., gradient descent with the Lasso objective), and our results show that Transformers can encode algorithms with similar performance in a single forward pass. 

**Role of model capacity and problem dimension.** Finally, we explore how the ability of Transformers to in-context learn linear functions scales with model capacity and problem dimensionality. We find that increasing the capacity of the model improves performance significantly, and also allows the model to incontext learn higher-dimensional functions. Moreover, increasing the capacity often significantly improves performance with distribution shifts, even when the absolute improvement in the standard error is small. 

## **2 Training models for in-context learning** 

We now describe a general methodology for training a model that can in-context learn a function class _F_ with respect to a distribution _DF_ over functions, and _DX_ over inputs. To do so, we start by constructing random training prompts as follows. For each prompt, we first sample a random function _f_ from the class according to _DF_ , then create a set of random inputs _x_ 1, . . . , _xk_ +1 drawn independently from _DX_ , and finally evaluate _f_ on these inputs to produce the prompt _P_ = ( _x_ 1, _f_ ( _x_ 1), . . . , _xk_ +1, _f_ ( _xk_ +1)). For example, in the case of linear functions, inputs could be drawn from the isotropic Gaussian distribution _N_ (0, _Id_ ), and a random function chosen by sampling weight vector _w_ from _N_ (0, _Id_ ) and setting _f_ ( _x_ ) = _w_<sup>_⊤_</sup> _x_ . 

Now, given such prompts, we train a model to predict _f_ ( _xi_ ) for a given _xi_ based on a set of preceding in-context examples. Concretely, let _P_<sup>_i_</sup> denote the prompt prefix containing _i_ in-context examples (the first _i_ input-output pairs) and the ( _i_ + 1)<sup>th</sup> input: _P_<sup>_i_</sup> = ( _x_ 1, _f_ ( _x_ 1), _x_ 2, _f_ ( _x_ 2), . . . , _xi_ , _f_ ( _xi_ ), _xi_ +1). Then, we train a 

3 

model _Mθ_ parameterized by _θ_ aiming to minimize the expected loss over all the prompt prefixes: 



where _ℓ_ ( _·_ , _·_ ) is an appropriately chosen loss function. Below, we describe how this general methodology can be implemented for a concrete model family (see Appendix A for additional details). 

**Model structure.** We use a decoder-only Transformer architecture [Vaswani et al., 2017] from the GPT-2 family [Radford et al., 2019]. Our model consists of 12 layers, 8 attention heads, and a 256-dimensional embedding space (9.5M parameters). This architecture takes as input a sequence of vectors in its embedding space and predicts the next vector in the sequence within the same space (in language modeling, these vectors correspond to input tokens). We apply this architecture to our prompt format of ( _x_ 1, _f_ ( _x_ 1), . . . , _xk_ +1, _f_ ( _xk_ +1)) as follows. We map each prompt output _f_ ( _xi_ ) to the same dimension as prompt inputs _xi_ by appending zeros, and map the prompt inputs and outputs into the latent embedding space of the Transformer through a (learnable) linear transformation. We then use another (learnable) linear transformation to map the vector predicted by the model to a scalar. Note that the Transformer architecture allows us to compute the prediction ( _Mθ_ ( _P_<sup>_i_</sup> )) for all prompt prefixes in a single forward pass. 

**Training.** We train the model according to the training objective in (2) using squared error as the loss function. We do so by sampling a batch of random prompts at each training step and then updating the model through a gradient update (we use a batch size of 64 and train for 500k total steps). This training is done from scratch, that is, we do _not_ fine-tune a pre-trained language model, nor do we train on actual text. 

**Curriculum learning.** Many natural function classes contain functions of varying complexity. We exploit this by training our model using a curriculum [Bengio et al., 2009, Elman, 1993, Sanger, 1994, Wu et al., 2020], where we train on a simpler distribution of functions in the beginning (e.g., linear functions with weight vectors restricted to a low-dimensional subspace) and gradually increase the function complexity. This speeds up training drastically, often allowing us to train models that would be significantly more expensive to train without a curriculum (see Section 6 for details). 

## **3 In-context learning of linear functions** 

In the previous section, we describe a general methodology for training Transformer models to in-context learn a class of functions. Here, we focus on a simple function class—namely linear functions—and study how well models trained using our methodology can in-context learn this class. 

**Prompt distribution.** We consider the class of linear functions _F_ = _f | f_ ( _x_ ) = _w_<sup>_⊤_</sup> _x_ , _w ∈_ **R**<sup>_d_�</sup> , in � _d_ dimensions where _d_ = 20. We sample _x_ 1, . . . , _xk_ , _x_ query, and _w_ independently from the isotropic Gaussian distribution _N_ (0, _Id_ ). We then compute each _yi_ = _w_<sup>_⊤_</sup> _xi_ and construct the prompt as _P_ = ( _x_ 1, _y_ 1, _x_ 2, _y_ 2, . . . , _xk_ , _yk_ , _x_ query). 

**Baselines.** To contextualize the performance of our trained model, we compare it to other learning algorithms: (a) the least squares estimator, computing the minimum-norm linear fit to the in-context examples ( _xi_ , _yi_ ), (b) _n_ -Nearest Neighbors, averaging the _yi_ values for the _n_ nearest neighbors of _x_ query, (c) averaging the values _yixi_ to estimate _w_ and compute the inner product of this estimate with _x_ query. Least squares is the optimal estimator for this problem and thus serves as a lower bound to the best error one can achieve. The other two baselines are consistent (but sub-optimal) estimators that are easier to compute and thus provide an estimate of the performance achieved by simple approaches. See Appendix A.3 for more details. 

4 

### **3.1 Transformers can in-context learn linear functions** 

We show the in-context learning ability of the resulting model along with the relevant baselines in Figure 2. The trained Transformer is able to in-context learn the class of linear functions with respect to the prompt distribution specified above, performing comparably to the optimal least squares estimator for any number of in-context examples considered. While the simpler baselines achieve non-trivial error, they are far worse, indicating that the trained model encodes a more complex algorithm. 



<!-- Start of picture text -->
1.2<br>Transformer<br>1.0 Least Squares<br>3-Nearest Neighbors<br>0.8<br>Averaging<br>0.6<br>0.4<br>0.2<br>0.0<br>0 10 20 30 40<br>in-context examples<br>squared error<br><!-- End of picture text -->

Figure 2: _Evaluating the trained Transformer on in-context learning linear functions._ We plot the normalized squared error of the Transformer (( _M_ ( _P_ ) _− w_<sup>_⊤_</sup> _x_ query)<sup>2</sup> / _d_ ), along with the relevant baselines, as a function of the number of in-context examples. Transformer’s error decreases at a rate comparable to least squares. When the number of in-context examples reaches the problem dimension _d_ (here 20), least squares achieves 0 error while the Transformer achieves an error of 0.02, improving to 0.0006 at 2 _d_ in-context examples. While the simple baselines obtain better-than-trivial error (zero estimator, dashed line), their performance is relatively poor. (Error averaged over 1280 prompts. 90% confidence intervals over 1000 bootstrap trials.) 

**Can memorization of training prompts explain model performance?** Note that the probability of the model encountering a training prompt similar to the one used for testing is astronomically low—the prompt inputs alone lie in a 800-dimensional space when predicting with 2 _d_ in-context examples ( _d_ = 20). Moreover, even considering the possibility that the model encountered a similar _weight vector_ during training cannot explain its performance. That is, the model encounters 32 million random weight vectors during training and even using the best of these vectors would lead to an expected error of around 0.2 (computed empirically, see Appendix B.7 for details). However, the model is able to achieve an error of less than 0.001 for a prompt with 2 _d_ in-context examples. Further, in Section 6, we show that the model is able to obtain a similar error even when trained on prompts generated using only 10, 000 distinct weight vectors, in which case the best weight vector seen during training would yield an even worse error of around 0.5. Thus, the model cannot be relying on memorization of training prompts or weight vectors, and instead encodes an algorithm capable of in-context learning linear functions that are very different from those seen during training. 

### **3.2 What functions is the model learning in-context?** 

Recall that the goal of our model is: given the prompt _P_ = ( _x_ 1, _w_<sup>_⊤_</sup> _x_ 1, . . . , _xk_ , _w_<sup>_⊤_</sup> _xk_ , _x_ query), output _w_<sup>_⊤_</sup> _x_ query. Thus, if we fix the prefix given by theˆ _k_ in-context examples, we can view the output of the model as a function _fw_ , _x_ 1: _k_ ( _x_ query), that approximates _w_<sup>_⊤_</sup> _x_ query. When _k < d_ (fewer in-context examples than dimensions), the ground truth cannot be recovered perfectly and the ideal model should approximate (proj _x_ 1: _k_ ( _w_ ))<sup>_⊤_</sup> _x_ query, where proj _x_ 1: _k_ ( _w_ ) is the projection of _w_ onto the subspace spanned by _x_ 1, . . . , _xk_ . Here, we will evaluate how accurately the model approximates this. 

5 



<!-- Start of picture text -->
ground truth #dims / 2 in-context examples #dims * 2 in-context examples<br>ground truth projected #dims in-context examples<br>20 20 20 1.0<br>0.8<br>10 10 10<br>0.6<br>0 0 0 0.4<br>10 10 10 0.2 gradient and true w<br>0.0 gradient and projected w<br>20 20 20<br>10 0 10 10 0 10 10 0 10 0 10 20 30 40<br>distance from origin distance from origin distance from origin in-context examples<br>(a) function visualizations (b) gradients<br>function value<br>average inner product<br><!-- End of picture text -->

Figure 3: _Understanding the prefix-conditioned function_ . (a) We plot the model prediction as we fix the in-context examples and vary the query input along a random direction (for three random prompts). The shaded regions denote the intervals in which the norm of a randomly training input lies with probability 0.99. When the scale of the query input is close to this range, the model prediction is close to the ground truth linear function (or its projection to the space of in-context inputs when _k < d_ ). (b) We compute the gradient of the model prediction with respect to the query input, and plot its (normalized) inner product with the true _w_ and projected _w_ , averaged over 1280 random prompts. The gradient aligns almost perfectly with _w_ when _k ≥ d_ , and with projected _w_ for all _k_ , indicating that the model locally aligns with the ground truth. 

**Visualizing along a random direction.** For a randomly sampled fixed prefix, we visualize _f_<sup>ˆ</sup> _w_ , _x_ 1: _k_ ( _x_ query) as we vary the query input along a random direction _x_ (Figure 3a). That is, we pick a random unit vector _x_ , and evaluate _f_<sup>ˆ</sup> _w_ , _x_ 1: _k_ ( _λx_ ) as we vary _λ_ , the distance of the query input from origin. We observe that _f_<sup>ˆ</sup> _w_ , _x_ 1: _d_ ( _λx_ ) and _f_<sup>ˆ</sup> _w_ , _x_ 1:2 _d_ ( _λx_ ) closely match the ground truth and _f_<sup>ˆ</sup> _w_ , _x_ 1: _d_ /2 ( _λx_ ) matches the projected ground truth, when the distance from origin is not too large compared to the norm of a typical randomly sampled input. In fact, in Appendix B.1, we show that the model is quite robust to scaling the query input: the error doesn’t increase much as we scale up the query input by a factor of up to 2, or scale down by a factor of up to 16, and degrades slowly after that. 

**Local correctness.** So far, we have seen that the model is able to make predictions close to the ground truth for randomly drawn query inputs and in-context examples. We will now turn our attention to the local change of _f_<sup>ˆ</sup> around _x_ query by considering the gradient of the function _f_<sup>ˆ</sup> _w_ , _x_ 1: _k_ ( _x_ query) with respect to _x_ query (our model is fully differentiable so we can compute the gradient directly). Since _f_<sup>ˆ</sup> computed by the model should ideally approximate proj _x_ 1: _k_ ( _w_ )<sup>_⊤_</sup> _x_ , this gradient should lie in the direction of the projected ground truth proj _x_ 1: _k_ ( _w_ ). In Figure 3b, we show the inner product between the gradient and proj _x_ 1: _k_ ( _w_ ) (both normalized), averaged over 1280 random prompts, and observe that they align almost perfectly. Since proj _x_ 1: _k_ ( _w_ ) = _w_ almost surely when _k ≥ d_ , we observe that the gradient also aligns with _w_ perfectly in this regime. Thus the model is locally correct with respect to changes in the query input. 

## **4 Extrapolating beyond the training distribution** 

In the previous section, we demonstrated that we can train a model to in-context learn linear functions with respect to the distribution of prompts encountered during training. That is, we evaluate the in-context learning ability of the model with respect to distributions _DX_ and _DF_ that were also used to train the model. 

Here, we evaluate the in-context learning performance of our model on prompt distributions different from the one used for training. Our overarching goal here is to better understand the learning algorithm encoded by our model by analysing how it responds to different prompt distributions. 

6 

Formally, we will refer to the distribution of functions used during training as _DF_<sup>train</sup> and the corresponding distribution of prompt inputs as _DX_<sup>train</sup> . Then, during inference, functions are sampled from a (potentially different) distribution _DF_<sup>test, while prompt inputs from a distribution</sup><sup>_D_</sup> _X_<sup>test.Moreover, deviating again from</sup> our analysis so far, we also consider a separate distribution _D_ query<sup>test, from which the query input is sampled,</sup> potentially dependent on the rest of the in-context inputs _x_ 1, . . . , _xk_ (which are still sampled from _DX_<sup>test).</sup> 

Within this framework, we consider the same model as last section, and evaluate its performance on prompts that deviate from those encountered during training, either by 

1. sampling prompt inputs or functions from a different distribution, that is _DX_<sup>train</sup> / _F_<sup>=</sup><sup>_D_</sup> _X_<sup>test</sup> / _F_<sup>or</sup> 

2. introducing a mismatch between in-context examples and the query input, that is _D_ query<sup>test=</sup><sup>_D_</sup> _X_<sup>test.</sup> 

We describe each such prompt structure below and present a subset of the results in Figure 4 (see Appendix B.2 for additional details and full results). Overall, the model performs reasonably accurate in-context learning with respect to these prompt distributions, indicating that it has indeed learnt to perform linear regression to some generality. 

Recall that we generate a training prompt _P_ = ( _x_ 1, _w_<sup>_T_</sup> _x_ 1, . . . , _xk_ , _w_<sup>_T_</sup> _xk_ , _x_ query) by drawing the prompt inputs ( _xi_ and _x_ query), and the weight vector ( _w_ ) i.i.d. from _N_ (0, _Id_ ), with _d_ = 20. For all the settings below, except prompt scaling, we normalize the inputs so that their expected squared norm is equal to that of inputs encountered during training. 



<!-- Start of picture text -->
1.50 1.50 1.50<br>1.25 1.25 1.25<br>1.00 1.00 1.00 Transformer<br>0.75 0.75 0.75 Least Squares<br>3-Nearest Neighbors<br>0.50 0.50 0.50 Averaging<br>0.25 0.25 0.25<br>0.00 0.00 0.00<br>0 10 20 30 40 0 10 20 30 40 0 10 20 30 40<br>in-context examples in-context examples in-context examples<br>(a) Skewed covariance (b) Noisy linear regression (c) Different orthants<br>squared error<br><!-- End of picture text -->

Figure 4: _In-context learning on out-of-distribution prompts._ We evaluate the trained model on prompts that deviate from those seen during training by: (a) sampling prompt inputs from a non-isotropic Gaussian, (b) adding label noise to in-context examples, (c) restricting in-context examples to a single (random) orthant. In all cases, the model error degrades gracefully and remains close to that of the least squares estimator, indicating that its in-context learning ability extrapolates beyond the training distribution. 

**Skewed covariance.** We sample prompt inputs from _N_ (0, Σ) where Σ is a skewed covariance matrix with eigenbasis chosen uniformly at random and _i_<sup>th</sup> eigenvalue proportional to<sup>1</sup> / _i_<sup>2</sup> . The model matches the performance of least squares until _k_ = 10, mimicking the sharp drop in the error in this regime, but its error plateaus afterwards (see Figure 4a). Thus, it is not perfectly robust to this distribution mismatch but still does relatively well, achieving less than half the error of the nearest neighbor baseline in most cases. 

**Low-dimensional subspace.** We sample prompt inputs from a random 10 dimensional subspace. In this case, the model achieves low error after 10 in-context examples, closely matching the behavior of the optimal least squares estimator (the model achieves an error of 0.036, 0.0014, and 0.00057 at 10, 20, and 40 in-context examples respectively)—see Appendix Figure 8b. Crucially, unlike the training prompts, when _k_ is between 10 and 20, the prompt inputs are linearly dependent, and a model achieving low error in this regime indicates that it encodes a valid orthogonalization procedure for these inputs. 

7 

**Noisy linear regression.** We add noise to each prompt output, that is, the _i_<sup>th</sup> output is equal to _w_<sup>_T_</sup> _xi_ + _ϵi_ where _ϵi ∼ N_ (0, 1). The trained model closely tracks the performance of least squares when the number of in-context examples is not close to the input dimension 20 (see Figure 4b). Interestingly, the model also exhibits the double descent error curve [Belkin et al., 2019] that is known to manifest for the least squares estimator [Nakkiran, 2019]. Note that in this noisy setting, the optimal estimator corresponds to solving least squares with appropriate _ℓ_ 2-regularization. However, since the model was trained on noiseless data, we cannot expect it to learn this. 

**Prompt scale.** We consider the setting where the prompt scale between training and inference is different. We either scale the prompt inputs or the weight vectors, by a factor _{_<sup>1</sup> /3,<sup>1</sup> /2, 2, 3 _}_ . The model is relatively robust when scaling the weight vector, but not as robust when scaling the prompt inputs, especially for the more extreme scales<sup>1</sup> /3 and 3. Specifically, for 40 in-context examples, the model achieves errors 0.0012, 0.0008, 0.0016, 0.0278 when scaling the weights, and errors 0.30, 0.013, 0.043, 0.58 while scaling the inputs, by factors<sup>1</sup> /3,<sup>1</sup> /2, 2 and 3 respectively (Appendix Figure 9). For context, recall that with 40 in-context examples, the least squares estimator achieves an error of 0 whereas the model achieves an error of 0.0006 at the original scale. 

**Different orthants for in-context and query inputs.** We fix the sign of each coordinate to be positive or negative for all in-context inputs _xi_ (at random). As a result, all in-context inputs lie in the same orthant, while the query input lies in another orthant with high probability. The model is not affected by the mismatch between in-context and query inputs and closely match the performance of least squares. In this case, the model achieves errors 0.062 and 0.004 for 20 and 40 in-context examples respectively (see Figure 4c), whereas recall that it achieves errors 0.02 and 0.0006 on standard prompts. This indicates that the model is not relying on some variant of nearest neighbor search as in that case, its error would have been significantly larger (see the 3-nearest neighbor baseline). 

**Query input orthogonal to in-context inputs.** We sample the query input from the subspace orthogonal to the subspace spanned by in-context example inputs. Here, there is no information relevant to the query input in the in-context examples and thus the model would ideally predict something close to 0 to minimize the error. Indeed, the model outputs such a prediction, achieving an error close to 1 (Appendix Figure 8d). 

**Query input matches an in-context example.** We choose the query input to match one of the in-context examples inputs chosen uniformly at random. In this case, the model achieves errors 0.001, 0.001, 0.0005 for 10, 20, 40 examples respectively thus making close to the correct prediction, without being affected by the additional in-context examples present (Appendix Figure 8e). 

## **5 More complex function classes** 

We now turn our attention to in-context learning for more complex function classes, namely sparse linear functions, decision trees, and two-layer ReLU neural networks. Here, we are back in the setting where the distribution of prompts during inference is same as that during training (except the setting of neural networks where we evaluate on linear functions as well). The overall methodology remains the same: we sample random functions from these families and train a Transformer from scratch to approximate these functions given in-context examples. (See Appendix A.3 for more details and baselines.) 

**Sparse linear functions.** First, we consider functions of the form _f_ ( _x_ ) = _w_<sup>_⊤_</sup> _x_ where _w ∈_ **R**<sup>_d_</sup> and has exactly _s_ non-zero coordinates. To sample a prompt _P_ = ( _x_ 1, _f_ ( _x_ 1), . . . , _xk_ , _f_ ( _xk_ ), _x_ query), we draw prompt inputs _xi_ and _x_ query, and a weight vector _w_ from _N_ (0, _Id_ ), and then zero out all but _s_ coordinates of _w_ uniformly at random. We choose _d_ = 20 and _s_ = 3. In this setting, the least squares estimator is no longer 

8 

optimal—one can perform better by leveraging the weight vector sparsity. One estimator that leverages sparsity is Lasso [Tibshirani, 1996], which involves solving the least squares objective with an _ℓ_ 1-norm regularizer for the weight vector. We plot the performance of our model in Figure 5a, and observe that it is also able to leverage sparsity, nearly matching the performance of Lasso. Our model achieves errors 0.58 and 0.09 while Lasso achieves errors 0.62 and 0.08 for _k_ = 5 and 10 respectively. Note that, unlike least squares, Lasso does not have a closed form expression and involves iterative minimization of the regularized objective, yet the Transformer is able to achieve comparable performance in a single forward pass. 



<!-- Start of picture text -->
1.2 Transformer 1.50 Transformer<br>1.0 Least Squares 1.25 3-Nearest Neighbors<br>Averaging Greedy Tree Learning<br>0.8 Lasso 1.00 XGBoost<br>0.6 0.75 Greedy Tree Learning<br>(w/ sign preproc.)<br>0.4 0.50 XGBoost<br>0.2 0.25 (w/ sign preproc.)<br>0.0 0.00<br>0 10 20 30 40 0 20 40 60 80 100<br>in-context examples in-context examples<br>(a) Sparse linear functions (b) Decision trees<br>1.2 1.2<br>Transformer Transformer<br>1.0 Least Squares 1.0 Least Squares<br>3-Nearest Neighbors 3-Nearest Neighbors<br>0.8 0.8<br>2-layer NN, GD 2-layer NN, GD<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>0.0 0.0<br>0 20 40 60 80 100 0 20 40 60 80 100<br>in-context examples in-context examples<br>(c) 2-layer NN (d) 2-layer NN, eval on linear functions<br>squared error<br>squared error<br><!-- End of picture text -->

Figure 5: _Training a Transformer to in-context learn more complex function classes._ (a) A Transformer trained on prompts generated using sparse linear functions can in-context learn this class, with error decreasing at a rate similar to Lasso, and significantly better than minimum norm least squares. (b) A Transformer trained on prompts generated using random decision trees can in-context learn this class, with much better performance than greedy tree learning or tree boosting. (c) A Transformer trained on prompts generated using random 2-layer ReLU neural networks can in-context learn this class. The error decreases at a rate similar to the baseline which involves training a neural network using a variant of gradient descent with in-context examples as the training data. (d) The same model (from (c)) can in-context learn the class of linear functions. The error decreases at a rate slower than least squares, but comparable to a neural network trained using a variant of gradient descent. In all cases, the errors are normalized so that the trivial zero estimator achieves an error of 1 (dashed line). 

**Decision trees.** Next, we consider the class of depth 4 decision trees with 20 dimensional inputs. A function _f_ in this class is represented by a full binary tree (with 16 leaf nodes) where each non-leaf node is associated with a coordinate, and each leaf node is associated with a target value. To evaluate _f_ on an input _x_ , we traverse the tree starting from the root node, and go to the right child if the coordinate associated with the current node is positive and go to the left child otherwise (that is, the threshold at each node is 0). _f_ ( _x_ ) is given by the value associated with the leaf node reached at the end. To sample a random prompt _P_ = ( _x_ 1, _f_ ( _x_ 1), . . . , _xk_ , _f_ ( _xk_ ), _x_ query), we draw prompt inputs _xi_ s and _x_ query from _N_ (0, _Id_ ), and _f_ corresponds to a tree where the coordinates associated with the non-leaf nodes are drawn uniformly at random from 

9 

_{_ 1, 2, . . . , _d}_ and the values associated with the leaf nodes are drawn from _N_ (0, 1). In Figure 5b, we show that Transformers can be trained to in-context learn this class, with performance much better than greedy tree learning and boosting (via XGBoost [Chen and Guestrin, 2016]). With _k_ = 100 in-context examples, the Transformer achieves an error of 0.12 whereas greedy learning achieves an error of 0.80 and XGBoost achieves an error of 0.62. 

Since the decision trees in our function class predict solely based on the sign of each coordinate of _xi_ , we also consider a baseline where we provide the greedy learning and XGBoost algorithms with the signs of each _xi_ instead. This significantly improves their performance—at 100 in-context examples, greedy achieves an error of 0.50 and XGBoost an error if 0.31—but they still perform much worse than the trained Transformer. 

Note that, in general, we do not have a good understanding of the space of efficient algorithms for learning decision trees, and the conditions under which known heuristics work [Blanc et al., 2021, Brutzkus et al., 2020]. At the same time, we found that Transformers can be trained to directly discover such an algorithm for the prompt distribution we considered. This suggests an intriguing possibility where we might be able to reverse engineer the algorithm encoded by a Transformer to obtain new sample efficient algorithms for existing learning problems. 

**Two-layer ReLU neural networks.** Finally, we consider the class of two layer ReLU neural networks containing functions of the form _f_ ( _x_ ) = ∑<sup>_r_</sup> _i_ =1<sup>_αiσ_(</sup><sup>_w_</sup> _i_<sup>_⊤x_), where</sup><sup>_αi∈_</sup><sup>**R**,</sup><sup>_wi∈_</sup><sup>**R**</sup><sup>_d_and</sup><sup>_σ_(</sup><sup>_·_)=max(0,</sup><sup>_·_)is</sup> the ReLU activation function. To draw a random prompt _P_ = ( _x_ 1, _f_ ( _x_ 1), . . . , _xk_ , _f_ ( _xk_ ), _x_ query), we sample prompt inputs _xi_ s and _x_ query from _N_ (0, _Id_ ), along with network parameters _ai_ s and _wi_ s from _N_ (0, 2/ _r_ ) and _N_ (0, _Id_ ) respectively. We set the input dimension _d_ to 20 and the number of the hidden nodes _r_ to 100. In Figure 5c, we show that Transformers can be trained to in-context learn this class of functions. In fact, the Transformer performs comparably to the baseline which involves training a two-layer neural network of the same architecture on in-context examples using Adam [Kingma and Ba, 2014], a variant of gradient descent (see Appendix A.3 for details). Specifically, for _k_ = 100 in-context examples, both the Transformer and the neural network trained on in-context examples achieve an error of 0.17. 

Moreover, the model trained to in-context learn two-layer neural networks is also able to in-context learn linear functions (for which it is not explicitly trained), albeit with a rate slower than least squares, but comparable to a neural network trained on in-context examples generated using a linear function (Figure 5d). For _k_ = 20, 50, and 100 in-context examples respectively, the Transformer achieves error 0.34, 0.05, and 0.01, and the two-layer network achieves error 0.37, 0.04, and 0.003 (the least squares estimator achieves error 0 for _k ≥_ 20). 

## **6 Investigating what matters for in-context learning** 

We now return to the setting of training models to in-context learn linear functions and explore different factors that lead to successful in-context learning. 

**Problem Dimension and Capacity.** In Section 3 and 4, we saw that Transformer models can be trained to in-context learn 20-dimensional linear functions accurately and relatively robustly. To explore the interplay between problem dimensionality and capacity, we also consider models with fewer parameters (see Appendix A.1) and train each architecture on {10, 30, 40, 50}-dimensional problems. In Figure 6, we plot the model error with 2 _d_ in-context examples as we vary the problem dimension _d_ and the model capacity. In the standard setting, i.e., when the training and inference time prompt distributions are the same, we observe that the error decreases as we increase the capacity or reduce the problem dimensionality (see Figure 6a). Thus, model capacity helps perform accurate in-context learning. For out-of-distribution prompts, we observe that the settings where the input covariance is skewed or where in-context example inputs and query inputs lie in different orthants are particularly challenging, especially for higher dimensional problems. However, the error decreases considerably (in most cases) as we increase the model capacity, even when absolute decrease in the standard error is small (see Figure 6b and 6c). See Appendix B.3 for additional plots. 

10 



<!-- Start of picture text -->
10 1 10 1 10 1<br>dimensions<br>10<br>10 0 10 0 10 0<br>20<br>10 1 10 1 10 1 30<br>40<br>10 2 10 2 10 2 50<br>10 3 10 3 10 3<br>10 4 10 4 10 4<br>0.2M 1.2M  9.5M 0.2M 1.2M  9.5M 0.2M 1.2M  9.5M<br>number of parameters number of parameters number of parameters<br>(a) Standard (b) Different orthants (c) Skewed covariance<br>squared error<br><!-- End of picture text -->

Figure 6: _Understanding the effect of model capacity and problem dimension on in-context learning performance for in-distribution (a) and out-of-distribution (b,c) prompts._ We train Transformers to in-context learn linear functions and plot the error with 2 _d_ in-context examples as we vary problem dimension _d_ and model capacity. Capacity helps with in-context learning in most cases, especially on out-of-distribution prompts (even when the absolute gains in the in-distribution setting are small). We train 3 models in each case with different random seeds, and show the median error (solid lines), and the minimum and maximum errors (shaded region). (See Appendix B.4 for training variance analysis.) 

**Curriculum.** We train our models using curriculum learning. That is, we initially draw the prompt inputs from a fixed 5 dimensional subspace (by setting some of the coordinates to 0) with prompt length 11 (number of input-output pairs), and increase the subspace dimension by 1 and prompt length by 2 every 2, 000 training steps, until the subspace dimension reaches the ambient dimension _d_ and prompt length reaches 2 _d_ + 1 (see Appendix A.2 for details). This process can also be viewed as gradually increasing the complexity of the function class. This speeds up training drastically, especially for higher dimensional problems: for dimension 50, the loss barely decreases through the 500k training steps without curriculum but reaches close to the optimum with curriculum. For the 20 dimensional problem where we were able to train the model without curriculum within the training (step count) budget, we did not observe any qualitative difference in accuracy or robustness compared to the model trained with curriculum. We include plots comparing the speed and accuracy of training with and without curriculum in Appendix B.5. 

Notably, when training Transformers without curriculum, there is an initial—relatively long—period in training where the loss does not decrease, followed by a period of sharp decrease. The length of this period varies with training randomness and seems to increase on average with problem dimension. Understanding the model just before and after this transition moment is a promising future direction, which can give insights into the emergence of in-context learning. Interestingly, Olsson et al. [2022] observe a similar jump in the in-context learning ability of a language model which they attribute to the formation of “induction heads”. 

**Number of distinct prompts or functions seen during training.** To estimate the amount of training data required for in-context learning, we perform two ablation studies. In the first study, we limit the number of distinct prompts seen during training. That is, we create a set of _np_ randomly generated prompts (as described in Section 2), and sample prompts from this set during training (here, we train without curriculum, as it would introduce additional prompts during the warmup phase). In the second study, we only limit the number of distinct functions used for training. That is we create a set of _nw_ randomly chosen vectors (corresponding to _nw_ linear functions) and sample weight vectors uniformly from that set to generate the training prompts (the inputs are still sampled from _N_ (0, _Id_ ) for each training prompt). We find that the amount of training data required is relatively small: non-trivial in-context learning is possible with _np_ = 100k or _nw_ = 1k, and the error drops close to that of the unrestricted model (discussed in Section 3) with _np_ = 1M or _nw_ = 10k (details in Appendix B.6). For context, in Section 3, the model is trained on fresh prompts each step, thus encountering 32M distinct functions and prompts (500k training steps with 64 prompts/batch). 

11 

## **7 Related work** 

**In-context learning.** Since Brown et al. [2020] demonstrated the in-context learning ability of GPT-3, there has been a significant interest in improving and understanding this capability [Liu et al., 2021, Min et al., 2021a, Zhao et al., 2021, Lu et al., 2021b, Rubin et al., 2021, Min et al., 2021b, Chen et al., 2021, Mishra et al., 2021, Lampinen et al., 2022]. The works most relevant to ours are as follows. Xie et al. [2022] propose a Bayesian inference framework explaining how in-context learning works despite formatting differences between training and inference distributions. Razeghi et al. [2022] show that in-context learning for numerical reasoning tasks is better for instances whose terms are more prevalent in training data. Min et al. [2021a] demonstrate tasks where in-context learning works even when the prompt outputs are chosen randomly, questioning to what extent these models are truly learning new tasks on-the-fly, while Rong [2021] gives examples of novel tasks on which these models demonstrate on-the-fly learning ability. Chan et al. [2022] demonstrate that distributional properties such as long-tailedness are crucial for in-context learning on an image-based few-shot dataset. Olsson et al. [2022] and Elhage et al. [2021] consider a different framing of in-context learning, referring to any model behavior that utilizes information in a prompt to make predictions that improve with prompt size. They hypothesize the existence of special circuits inside Transformer models responsible for in-context learning, that can complete prompts by copying previous similar patterns in the prompt sequence. Pesut [2022] and Dinh et al. [2022, Table 16] consider in-context learning for small tabular datasets and learning problems in one and two dimensions, and show that GPT-3 can obtain non-trivial accuracy. Our work contributes to and complements this line of work, by posing in-context learning as a well-defined problem of learning function classes at inference time, and empirically investigating training models that in-context learn simple function classes. 

**Transformers.** There is a long line of work investigating the capabilities [Vaswani et al., 2017, Dehghani et al., 2018, Yun et al., 2019, Pérez et al., 2019, Yao et al., 2021, Bhattamishra et al., 2020b, Zhang et al., 2022], limitations [Hahn, 2020, Bhattamishra et al., 2020a], applications [Lu et al., 2021a, Dosovitskiy et al., 2020, Parmar et al., 2018], and internal workings [Elhage et al., 2021, Snell et al., 2021, Weiss et al., 2021, Edelman et al., 2022, Olsson et al., 2022] of Transformer models. Most similar to our work, Müller et al. [2021] and Nguyen and Grover [2022] demonstrate the ability of Transformer models to solve prediction tasks using the input context, albeit in different settings. Müller et al. [2021] introduce a “Prior-data fitted transformer network” that is trained to approximate Bayesian inference with priors such as Gaussian processes and Bayesian neural networks, and use it to perform downstream tasks such as tabular dataset classification and few-shot image classification. Nguyen and Grover [2022] introduce Transformer neural processes, building on prior work on neural processes [Garnelo et al., 2018b,a, Kim et al., 2019], and show that they achieve state-of-the art performance on tasks such as image completion and contextual multi-armed bandits. Our work complements these works, focusing on understanding the in-context learning ability of Transformers for various simple function classes and the extent to which this ability extrapolates beyond the training distribution. 

**Meta learning.** Training a model to perform in-context learning can be viewed as an instance of the more general learning-to-learn or meta-learning paradigm [Schmidhuber, 1987, Naik and Mammone, 1992, Thrun and Pratt, 2012]. Typical approaches from this extensive line of work (see [Hospedales et al., 2020] for a survey) include: training a meta-learner on how to update the parameters of a downstream learner [Bengio et al., 1995, Li and Malik, 2016], learning parameter initializations from which one can quickly train for many downstream tasks [Finn et al., 2017, Ravi and Larochelle, 2017], learning latent embeddings that allow for effective similarity search [Snell et al., 2017]. Most relevant to our setting are approaches that directly take as input examples from a downstream task and a query input and produce the corresponding output [Hochreiter et al., 2001, Mishra et al., 2018, Santoro et al., 2016, Garnelo et al., 2018b,a, Kirsch and Schmidhuber, 2021]. Our work contributes to this line of work, by investigating the learning-to-learn abilities of Transformer models in a well-defined setting. 

12 

**Data-driven algorithm design.** Another line of work aims to discover algorithms that perform well on a distribution of inputs [Horvitz et al., 2001, Xu et al., 2008, Vinyals et al., 2015, Bello et al., 2016, Khalil et al., 2017, Selsam et al., 2018, Schwarzschild et al., 2021] (as opposed to algorithms with guarantees on their worst-case performance). See Balcan [2020] for a survey on advancements on the theoretical foundations of such algorithms. Our work can be viewed as part of this line of work, as we train Transformer models to discover algorithms for different learning problems. 

## **8 Discussion** 

In this work, we formalize and study the question: can we train models that learn different classes of functions in-context? We show that Transformer models trained from scratch can in-context learn the class of linear functions, with performance comparable to the optimal least squares estimator, even under distribution shifts. Moreover, we show that in-context learning is also possible for sparse linear functions, decision trees, and two-layer neural networks; learning problems which are solved in practice with involved iterative algorithms such as gradient descent. 

At the same time, understanding the implications of our results for language models requires further investigation. A pertinent question regarding the in-context learning capabilities of language models is how they leverage in-context examples [Min et al., 2022]. Our results demonstrate that Transformers can encode complex learning algorithms that utilize in-context examples in a far-from-trivial manner. In fact, this is the case for standard Transformer architectures trained with standard optimization procedures. The extent to which such non-trivial in-context learning behavior exists in large language models is still open, but we believe that our work takes a step towards formalizing and investigating this question. 

Our work lays the groundwork for several future directions. 

**Complexity of in-context learning.** We empirically show that model capacity helps in performing incontext learning accurately and robustly. This raises the question: How does the in-context learning loss (1) depend on the complexity of the function class _F_ , the capacity of model _M_ , and the number of prompts used to train _M_ . Even the right notion of complexity of _F_ is unclear and may depend on the model family. Understanding this question for models explicitly trained to perform in-context learning may suggest an upper bound for the in-context learning performance of models such as GPT-3 that have not been explicitly trained for this purpose. 

**Curriculum learning.** Within our framework, there is natural notion of curriculum learning where during training, we gradually increase the complexity of the function class learned in-context. This leads to drastic speed-ups in training. What is the reason behind such a speedup? Are similar speedups also possible for training large language models? Understanding these questions can have implications for training of models on large real-world datasets, potentially reducing the time and energy used for training. 

**Inductive bias of model families.** Our framework presents an opportunity to understand and compare the inductive biases of different model families (e.g., Transformers vs. LSTMs) in a well-defined setting. For instance, a concrete question is: Are there function classes that are easier to in-context learn using Transformers but harder for LSTMs and vice-versa? 

**Understanding the learning algorithms encoded in Transformers.** The models we train are able to perform in-context learning, and are thus themselves encoding learning algorithms. A worthwhile research direction would be to investigate the internal workings of these models and better understand the exact learning algorithms that they encode. Moreover, for settings such as decision trees, we do not have a good understanding of what the optimal learning algorithms are. Nevertheless, in Section 5 we found that Transformers are able to discover sample efficient algorithms when being trained to perform in-context 

13 

learning. This suggests an intriguing possibility where we might be able to reverse engineer the Transformer to obtain better learning algorithms for such problems. 

## **Acknowledgements** 

We thank Niladri Chatterji, Micah Goldblum, Rohith Kuditipudi, Shibani Santurkar, Carmen Strassle, Mirac Sugzun, and Li-Yang Tan for helpful conversations, and anonymous reviewers for helpful comments. 

SG was funded by a Stanford Interdisciplinary Graduate Fellowship. DT was funded by Open Philanthropy, and partially supported by NSF Award CCF-1813049. GV was supported by NSF Awards CCF-1704417, CCF-1813049, Frontier Award 1804222 and DOE award DE-SC0019205. We performed our experiments on the Stanford NLP cluster. 

## **References** 

- Maria-Florina Balcan. Data-driven algorithm design. In Tim Roughgarden, editor, _Beyond Worst Case Analysis of Algorithms_ . Cambridge University Press, 2020. 

- Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal. Reconciling modern machine-learning practice and the classical bias–variance trade-off. _Proceedings of the National Academy of Sciences_ , 2019. 

- Irwan Bello, Hieu Pham, Quoc V Le, Mohammad Norouzi, and Samy Bengio. Neural combinatorial optimization with reinforcement learning. _arXiv preprint arXiv:1611.09940_ , 2016. 

- Samy Bengio, Yoshua Bengio, Jocelyn Cloutier, and Jan Gecsei. On the optimization of a synaptic learning rule. In _Preprints Conf. Optimality in Artificial and Biological Neural Networks_ , 1995. 

- Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In _International Conference on Machine Learning (ICML)_ , pages 41–48, 2009. 

- Satwik Bhattamishra, Kabir Ahuja, and Navin Goyal. On the ability and limitations of transformers to recognize formal languages. _arXiv preprint arXiv:2009.11264_ , 2020a. 

- Satwik Bhattamishra, Arkil Patel, and Navin Goyal. On the computational power of transformers and its implications in sequence modeling. _arXiv preprint arXiv:2006.09286_ , 2020b. 

- Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. Gpt-neox-20b: An open-source autoregressive language model. _arXiv preprint arXiv:2204.06745_ , 2022. 

- Guy Blanc, Jane Lange, Mingda Qiao, and Li-Yang Tan. Decision tree heuristics can fail, even in the smoothed setting. _arXiv preprint arXiv:2107.00819_ , 2021. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Neural Information Processing Systems (NeurIPS)_ , 2020. 

- Alon Brutzkus, Amit Daniely, and Eran Malach. Id3 learns juntas for smoothed product distributions. In _Conference on Learning Theory_ , pages 902–915. PMLR, 2020. 

- Stephanie CY Chan, Adam Santoro, Andrew K Lampinen, Jane X Wang, Aaditya Singh, Pierre H Richemond, Jay McClelland, and Felix Hill. Data distributional properties drive emergent few-shot learning in transformers. _arXiv preprint arXiv:2205.05055_ , 2022. 

- Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In _conference on knowledge discovery and data mining (KDD)_ , 2016. 

14 

- Yanda Chen, Ruiqi Zhong, Sheng Zha, George Karypis, and He He. Meta-learning via language model in-context tuning. _arXiv preprint arXiv:2110.07814_ , 2021. 

- Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. _arXiv preprint arXiv:1807.03819_ , 2018. 

- Tuan Dinh, Yuchen Zeng, Ruisu Zhang, Ziqian Lin, Shashank Rajput, Michael Gira, Jy-yong Sohn, Dimitris Papailiopoulos, and Kangwook Lee. Lift: Language-interfaced fine-tuning for non-language machine learning tasks. _arXiv preprint arXiv:2206.06565_ , 2022. 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_ , 2020. 

- Benjamin L Edelman, Surbhi Goel, Sham Kakade, and Cyril Zhang. Inductive biases and variable creation in self-attention mechanisms. In _International Conference on Machine Learning (ICML)_ , 2022. 

- Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. _Transformer Circuits Thread_ , 2021. https://transformercircuits.pub/2021/framework/index.html. 

- Jeffrey L Elman. Learning and development in neural networks: The importance of starting small. _Cognition_ , 1993. 

- Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In _International conference on machine learning (ICML)_ , 2017. 

- Jerome H Friedman. Greedy function approximation: a gradient boosting machine. _Annals of statistics_ , 2001. 

- Marta Garnelo, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo Rezende, and SM Ali Eslami. Conditional neural processes. In _International Conference on Machine Learning_ , pages 1704–1713. PMLR, 2018a. 

- Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J Rezende, SM Eslami, and Yee Whye Teh. Neural processes. _arXiv preprint arXiv:1807.01622_ , 2018b. 

- Michael Hahn. Theoretical limitations of self-attention in neural sequence models. _Transactions of the Association for Computational Linguistics_ , 2020. 

- Sepp Hochreiter, A Steven Younger, and Peter R Conwell. Learning to learn using gradient descent. In _International conference on artificial neural networks (ICANN)_ , 2001. 

- Eric Horvitz, Yongshao Ruan, Carla Gomes, Henry Kautz, Bart Selman, and Max Chickering. A bayesian approach to tackling hard computational problems (preliminary report). _Electronic Notes in Discrete Mathematics_ , 2001. 

- Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Meta-learning in neural networks: A survey. _arXiv preprint arXiv:2004.05439_ , 2020. 

- Elias Khalil, Hanjun Dai, Yuyu Zhang, Bistra Dilkina, and Le Song. Learning combinatorial optimization algorithms over graphs. _Neural Information Processing Systems (NeurIPS)_ , 2017. 

- Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, Marta Garnelo, Ali Eslami, Dan Rosenbaum, Oriol Vinyals, and Yee Whye Teh. Attentive neural processes. _arXiv preprint arXiv:1901.05761_ , 2019. 

15 

- Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- Louis Kirsch and Jürgen Schmidhuber. Meta learning backpropagation and improving it. _Neural Information Processing Systems (NeurIPS)_ , 2021. 

- Andrew K Lampinen, Ishita Dasgupta, Stephanie CY Chan, Kory Matthewson, Michael Henry Tessler, Antonia Creswell, James L McClelland, Jane X Wang, and Felix Hill. Can language models learn from explanations in context? _arXiv preprint arXiv:2204.02329_ , 2022. 

- Ke Li and Jitendra Malik. Learning to optimize. _arXiv preprint arXiv:1606.01885_ , 2016. 

- Opher Lieber, Or Sharir, Barak Lenz, and Yoav Shoham. Jurassic-1: Technical details and evaluation. _White Paper. AI21 Labs_ , 2021. 

- Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for gpt-3? _arXiv preprint arXiv:2101.06804_ , 2021. 

- Kevin Lu, Aditya Grover, Pieter Abbeel, and Igor Mordatch. Pretrained transformers as universal computation engines. _arXiv preprint arXiv:2103.05247_ , 2021a. 

- Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. _arXiv preprint arXiv:2104.08786_ , 2021b. 

- Sewon Min, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Noisy channel language model prompting for few-shot text classification. _arXiv preprint arXiv:2108.04106_ , 2021a. 

- Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. Metaicl: Learning to learn in context. _arXiv preprint arXiv:2110.15943_ , 2021b. 

- Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? _arXiv preprint arXiv:2202.12837_ , 2022. 

- Nikhil Mishra, Mostafa Rohaninejad, Xi Chen, and Pieter Abbeel. A simple neural attentive meta-learner. In _International Conference on Learning Representations (ICLR)_ , 2018. 

- Swaroop Mishra, Daniel Khashabi, Chitta Baral, Yejin Choi, and Hannaneh Hajishirzi. Reframing instructional prompts to gptk’s language. _arXiv preprint arXiv:2109.07830_ , 2021. 

- Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do bayesian inference. _arXiv preprint arXiv:2112.10510_ , 2021. 

- Devang K Naik and Richard J Mammone. Meta-neural networks that learn by learning. In _International Joint Conference on Neural Networks (IJCNN)_ , 1992. 

- Preetum Nakkiran. More data can hurt for linear regression: Sample-wise double descent. _arXiv preprint arXiv:1912.07242_ , 2019. 

- Tung Nguyen and Aditya Grover. Transformer neural processes: Uncertainty-aware meta learning via sequence modeling. _arXiv preprint arXiv:2207.04179_ , 2022. 

- Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads. _Transformer Circuits Thread_ , 2022. https://transformer-circuits.pub/2022/in-contextlearning-and-induction-heads/index.html. 

16 

- Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In _International Conference on Machine Learning (ICML)_ , 2018. 

- F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. _Journal of Machine Learning Research_ , 2011. 

- Jorge Pérez, Javier Marinkovi´c, and Pablo Barceló. On the turing completeness of modern neural network architectures. _arXiv preprint arXiv:1901.03429_ , 2019. 

- Lovre Pesut. Who models the models that model models? an exploration of gpt-3’s in-context model fitting ability, 2022. URL `https://www.alignmentforum.org/posts/c2RzFadrxkzyRAFXa/ who-models-the-models-that-model-models-an-exploration-of` . 

- Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. _OpenAI blog_ , 2018. 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 2019. 

- Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. _arXiv preprint arXiv:2112.11446_ , 2021. 

- Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. _International Conference for Learning Representations (ICLR)_ , 2017. 

- Yasaman Razeghi, Robert L Logan IV, Matt Gardner, and Sameer Singh. Impact of pretraining term frequencies on few-shot reasoning. _arXiv preprint arXiv:2202.07206_ , 2022. 

- Frieda Rong. Extrapolating to unnatural language processing with gpt-3’s in-context learning: The good, the bad, and the mysterious), 2021. URL `http://ai.stanford.edu/blog/in-context-learning/` . 

- Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. _arXiv preprint arXiv:2112.08633_ , 2021. 

- Terence D Sanger. Neural network learning control of robot manipulators using gradually increasing task difficulty. _IEEE transactions on Robotics and Automation_ , 1994. 

- Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. Meta-learning with memory-augmented neural networks. In _International conference on machine learning (ICML)_ , 2016. 

- Jürgen Schmidhuber. _Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta-... hook_ . PhD thesis, Technische Universität München, 1987. 

- Avi Schwarzschild, Eitan Borgnia, Arjun Gupta, Furong Huang, Uzi Vishkin, Micah Goldblum, and Tom Goldstein. Can you learn an algorithm? generalizing from easy to hard problems with recurrent networks. _Neural Information Processing Systems (NeurIPS)_ , 2021. 

- Daniel Selsam, Matthew Lamm, B Benedikt, Percy Liang, Leonardo de Moura, David L Dill, et al. Learning a sat solver from single-bit supervision. In _International Conference on Learning Representations (ICLR)_ , 2018. 

- Charlie Snell, Ruiqi Zhong, Dan Klein, and Jacob Steinhardt. Approximating how single head attention learns. _arXiv preprint arXiv:2103.07601_ , 2021. 

- Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. _Neural Information Processing Systems (NeurIPS)_ , 2017. 

17 

- Sebastian Thrun and Lorien Pratt. _Learning to learn_ . Springer Science & Business Media, 2012. 

- Robert Tibshirani. Regression shrinkage and selection via the lasso. _Journal of the Royal Statistical Society: Series B (Methodological)_ , 1996. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Neural Information Processing Systems (NeurIPS)_ , 2017. 

- Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer networks. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, _Neural Information Processing Systems (NeurIPS)_ , 2015. 

- Gail Weiss, Yoav Goldberg, and Eran Yahav. Thinking like transformers. In _International Conference on Machine Learning_ , 2021. 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_ . Association for Computational Linguistics (ACL), 2020. 

- Xiaoxia Wu, Ethan Dyer, and Behnam Neyshabur. When do curricula work? _arXiv preprint arXiv:2012.03107_ , 2020. 

- Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. In _International Conference on Learning Representations (ICLR)_ , 2022. 

- Lin Xu, Frank Hutter, Holger H Hoos, and Kevin Leyton-Brown. Satzilla: portfolio-based algorithm selection for sat. _Journal of artificial intelligence research_ , 2008. 

- Shunyu Yao, Binghui Peng, Christos Papadimitriou, and Karthik Narasimhan. Self-attention networks can process bounded hierarchical languages. _arXiv preprint arXiv:2105.11115_ , 2021. 

- Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J Reddi, and Sanjiv Kumar. Are transformers universal approximators of sequence-to-sequence functions? _arXiv preprint arXiv:1912.10077_ , 2019. 

- Yi Zhang, Arturs Backurs, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, and Tal Wagner. Unveiling transformers with lego: a synthetic reasoning task. _arXiv preprint arXiv:2206.04301_ , 2022. 

- Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models. In _International Conference on Machine Learning (ICML)_ , 2021. 

18 

## **A Experimental setup** 

Here, we provide additional details on our experimental setup. 

### **A.1 Model architecture** 

We use architectures from the GPT-2 family [Radford et al., 2018] as implemented by HuggingFace [Wolf et al., 2020]<sup>3</sup> . Specifically, we consider the following set of configurations. 

|Model|Embedding size|#Layers|#Heads|(Total parameters)|
|---|---|---|---|---|
|Tiny|64|3|2|0.2M|
|Small|128|6|4|1.2M|
|**Standard**|256|12|8|9.5M|



We use the Standard model for the bulk of our experiments and only consider the smaller models for the capacity explorations in Section 6 and Appendix B.3. Since we train on each input once (we sample new inputs at each training step), overfitting to the training data is not an issue. Therefore, we set the Dropout probability to 0. 

Out of the box, these models take as input a sequence of vectors in embedding space and output a sequence of vectors in the same space. However, the tasks we study are functions from a lower dimensional vector space (e.g., 10-50 dimensions) to a scalar value. Thus, in order to use a prompt such as _x_ 1, _f_ ( _x_ 1), _x_ 2, _f_ ( _x_ 2) . . ., we need to map _xi_ s and _f_ ( _xi_ )s to vectors in embedding space. We do so by first turning the scalars _f_ ( _xi_ ) into vectors of the same dimension as _xi_ by appending 0s and then applying a learnable linear transformation to map all these vectors into the embedding space. Finally, we map the model output into a scalar value through a dot product with a learnable vector. 

We treat the prediction of the model at the position corresponding to _xi_ (that is absolute position 2 _i −_ 1) as the prediction of _f_ ( _xi_ ). Due to the structure of these models, this prediction only depends on ( _xj_ , _f_ ( _xj_ )) for _j < i_ and _xi_ . We ignore the model predictions at positions corresponding to _f_ ( _xi_ ). 

### **A.2 Training** 

Each training prompt is produced by sampling a random function _f_ from the function class we are training on, then sampling inputs _xi_ from the isotropic Gaussian distribution _N_ (0, _Id_ ) and constructing a prompt as ˆ ( _x_ 1, _f_ ( _x_ 1), . . . , _xk_ , _f_ ( _xk_ )). Given a prompt, we obtain model predictions _yi_ (meant to approximate _f_ ( _xi_ )) for each input, and compute the loss 



At each training step, we average the loss over a batch of randomly generated prompts (with different functions and prompt inputs), and perform an update step. We use the Adam optimizer [Kingma and Ba, 2014], and train for 500,000 total steps with a batch size of 64. We use a learning rate of 10<sup>_−_4</sup> for all function classes and models. 

**Curriculum learning.** To accelerate training, we start by training on prompt inputs _xi_ lying in a smaller dimensional subspace, and with fewer inputs per prompt, and gradually increase the subspace dimension and number of prompt inputs. Specifically, we zero out all but the first _d_ cur coordinates of _xi_ , sample prompts of size _k_ cur and leave the rest of the training process the same. We use the same schedule for all training runs for the function classes of linear functions and sparse linear functions, starting with _d_ cur = 5, _k_ cur = 11, and increasing _d_ cur and _k_ cur by 1 and 2 respectively, every 2000 steps, until _d_ cur = _d_ , _k_ cur = 2 _d_ + 1. We use a slightly different schedule for 2 layer neural networks and decision trees as we want prompts with more 

> 3 `https://huggingface.co/docs/transformers/model_doc/gpt2` 

19 

inputs for these function classes. For these classes, we start with _d_ cur = 5, _k_ cur = 26, and increase _d_ cur and _k_ cur by 1 and 5 respectively, every 2000 steps, until _d_ cur = _d_ , _k_ cur = 5 _d_ + 1. 

Overall, with curriculum, a training prompt ( _x_ 1, _f_ ( _x_ 1), . . . , _xk_ cur, _f_ ( _xk_ cur ) is generated by sampling a random function _f_ from the function class, drawing inputs _xi_ by sampling i.i.d. from _N_ (0, _Id_ ) and zeroing out all but the first _d_ cur coordinates. Given model predictions _y_ ˆ _i_ , the loss is given by 



**Sampling random functions.** For the class of linear functions, we sample random function _f_ ( _x_ ) = _w_<sup>_⊤_</sup> _x_ by drawing _w ∼ N_ (0, _Id_ ). For our main setting (Section 3 and 4), we set _d_ = 20. 

For the class of two-layer neural networks, we sample _f_ ( _x_ ) = ∑<sup>_r_</sup> _i_ =1<sup>_αiσ_(</sup><sup>_w_</sup> _i_<sup>_⊤x_),where</sup><sup>_αi_s and</sup><sup>_wi_s are</sup> drawn i.i.d. from _N_ (0, 2/ _r_ ) and _N_ (0, _Id_ ) respectively. We set _d_ = 20 and _r_ = 100. 

For the class of _k_ -sparse linear functions, we sample _f_ ( _x_ ) = _w_<sup>_⊤_</sup> _x_ by drawing _w ∼ N_ (0, _Id_ ) and zeroing out all but _k_ coordinates of _w_ chosen uniformly at random from the first _d_ cur coordinates (as defined in the curriculum learning description above). We set _d_ = 20 and _k_ = 3. 

For the class of decision trees, the random function _f_ is represented by a decision tree of depth 4 (with 16 leaf nodes), with 20 dimensional inputs. Each non-leaf node of the tree is associated with a coordinate selected uniformly at random from _{_ 1, 2, . . . , _d}_ , and each leaf node is associated with a value drawn randomly from _N_ (0, 1). To evaluate _f_ on an input _x_ , we traverse the tree starting from the root node, and go to the right child if the coordinate associated with the current node is positive and go to the left child otherwise. _f_ ( _x_ ) is given by the value associated with the leaf node reached at the end. 

**Computational resources.** We train using a single NVIDIA GeForce RTX 3090 GPU and most training runs take 5-20 hours depending on model size and context length. For instance, for the class of linear functions, training the standard model takes 17 hours for _d_ = 50, 7 hours for _d_ = 20 and 5.5 hours for _d_ = 10. For decision trees, training the standard model takes 17 hours. The time it takes for decision trees and 50 dimensional linear functions is higher due to larger context lengths (we train for _d_ dimensional linear functions with 2 _d_ + 1 input-output pairs per prompt). 

### **A.3 Baselines** 

**Least squares.** Minimum norm least squares is the optimal estimator for the linear regression problem. Given a prompt _P_ = ( _x_ 1, _y_ 1, . . . , _xk_ , _yk_ , _x_ query), let _X_ be a _k × d_ matrix with row _i_ given by _xi_ , and let _y_ be a _k_ ˆ dimensional vector with the _i_<sup>th</sup> entry _yi_ . Set _w_<sup>_T_</sup> = _X_<sup>+</sup> _y_ , where _X_<sup>+</sup> denotes the Moore-Penrose pseudoinverse of _X_ . The estimator predicts _M_ ( _P_ ) = _w_ ˆ<sup>_T_</sup> _x_ query. 

ˆ ˆ **Averaging estimator.** This corresponds to _M_ ( _P_ ) = _w_<sup>_T_</sup> _x_ query where _w_ =<sup><u>1</u></sup> _k_<sup>∑</sup> _i_<sup>_k_</sup> =1<sup>_xiyi_.Thisestimatoris</sup> consistent (yet sub-optimal) when _xi_ s are drawn from _N_ (0, _Id_ ). Unlike least squares, this estimator does not involve an inverse computation, and might be easier for a model to encode. 

**Nearest neighbors.** This corresponds to setting _M_ ( _P_ ) = _n_<sup><u>1</u>∑</sup><sup>_i∈S yi_.Here,</sup><sup>_S_is the set of indices of the</sup><sup>_n_</sup> nearest neighbors of _x_ query among _x_ 1 to _xk_ . For _k < n_ , we average over all the _yi_ s from 1 to _k_ , and for _k_ = 0, we set _M_ ( _P_ ) = 0. We consider the nearest neighbors baselines as it might be easier for a Transformer model to encode using self-attention compared to least squares. 

**Lasso.** We use this baseline for sparse linear functions (Section 5). This corresponds to _M_ ( _P_ ) = _w_ ˆ<sup>_T_</sup> _x_ query, where _w_ ˆ minimizes the _ℓ_ 1-norm regularized least squares objective: 



20 

We try different values of _α ∈{_ 1, 10<sup>_−_1</sup> , 10<sup>_−_2</sup> , 10<sup>_−_3</sup> , 10<sup>_−_4</sup> _}_ , and report the best solution (achieving the smallest error with 10 in-context examples) corresponding to _α_ = 10<sup>_−_2</sup> . To solve the optimization problem, we use the Lasso implementation from Scikit-learn [Pedregosa et al., 2011]<sup>4</sup> . 

**Greedy Tree Learning.** We use this baseline for the class of decision trees. This corresponds to greedily learning a decision tree using the in-context examples, and using it to classify the query input. To construct the tree, at each node (starting from a root node), we choose a coordinate for partitioning the examples into two sets, so as to minimize the variance of _yi_ s in each set, averaged across the two sets. The value associated with a leaf node is the average _yi_ value of the examples belonging to it. We use Scikit-learn’s decision tree regressor [Pedregosa et al., 2011]<sup>5</sup> implementation for this, with all the arguments set to their default value except the max_depth argument which is set to 2. We considered values _{_ 1, 2, 3, 4, 5, 6, unbounded _}_ for the maximum depth and chose the value that performs best at 100 in-context examples which was 2 (which differs from the decision trees sampled from the function class which have depth 4). We also considered a baseline where we learn this tree using only the signs of each _xi_ coordinate—after all, the decision tree we are trying to learn depends only on the signs of _xi_ . In this case, we found the optimal depth to be 4. 

**Tree boosting.** For the class of decision trees, we also consider a tree boosting baseline that corresponds to learning an ensemble of decision trees (see Friedman [2001] for a description of the general framework). Specifically, we use the XGBoost library [Chen and Guestrin, 2016]<sup>6</sup> , an implementation commonly used for a wide range of real-world machine learning tasks. 

We performed a hyperpameter search by considering {1, 2, 5, 10, 50, 100, 200, 400} estimators in the ensemble (equivalent to number of boosting rounds), a learning rate of {0.001, 0.01, 0.1, 0.3, 0.6, 1, 3}, and a maximum depth of {1, 2, 3, 4, 6, 10, 16}. In general, we found the performance of the learning algorithm to be quite robust. We chose the hyperparameters obtaining the best performance with 100 training examples, corresponding to 50 estimators, a maximum depth of 4, and a learning rate f 0.1. We found these hyperparameters to also be optimal when learning based on the signs of each _xi_ . 

**Learning neural networks with gradient descent.** We use this baseline for the class of two-layer neural networks (Section 5). This corresponds to training a two-layer neural network on the in-context examples, and outputting its prediction on the query point. That is, _M_ ( _P_ ) = _f_<sup>ˆ</sup> ( _x_ query), where 



ˆ ˆ Here, _σ_ ( _·_ ) is the ReLU activation. We find parameters _αi_ , _wi_ by minimizing the squared error of the prediction for the in-context examples 



using the Adam optimizer. We use a batch size of 10 (we use full batch when the number of in-context examples is less than 10) with 5000 optimization steps, and set _r_ = 100. We use a learning rate of 5 _·_ 10<sup>_−_3</sup> in the case when the data is generated using a neural network, and a learning rate of 5 _·_ 10<sup>_−_2</sup> when the data is generated using a linear function. We consider the setting with 100 in-context examples, and do a hyperparameter grid search over learning rate _∈{_ 5 _·_ 10<sup>_−_4</sup> , 5 _·_ 10<sup>_−_3</sup> , 5 _·_ 10<sup>_−_2</sup> , 5 _·_ 10<sup>_−_1</sup> , 5 _}_ , _r ∈{_ 100, 400 _}_ , batch size _∈{_ 10, 100 _}_ , optimization algorithm _∈{_ adam, sgd _}_ . All the hyperparameter settings in this grid led to a similar or worse performance compared to the hyperparameter setting we choose. 

> 4 `https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html` 

> 5 `https://scikit-learn.org/stable/modules/tree.html#regression` 

> 6 `https://github.com/dmlc/xgboost` 

21 

## **B Additional experimental results** 

### **B.1 Robustness to query scale** 

In Figure 7, we show that the trained model is quite robust to scaling the query input (while keeping the in-context examples fixed): the error does not increase much as we scale up the query input by a factor of up to 2, or scale down by a factor of up to 16, and degrades slowly after that. 



<!-- Start of picture text -->
1.0 in-context examples<br>10<br>0.8<br>20<br>0.6 40<br>0.4<br>0.2<br>0.0<br>10 1 10 0 10 1<br>scale<br>squared error<br><!-- End of picture text -->

Figure 7: _Robustness to the scale of query input._ For a fixed set of in-context examples, we measure the model’s error as we scale the query input by a scalar. 

22 

### **B.2 Out-of-distribution prompts** 

Here, we describe the structure of our out-of-distribution prompts (cf. Section 4), and show the corresponding plots (Figure 8). To avoid conflating factors, we normalize the prompt inputs such that their expected norm is equal to the expected norm of inputs during training and investigate the role of scaling these inputs separately. We summarize how these prompts deviate from those seen during training in the table below. 

|Prompting strategy|_D_<sup>train</sup><br>_X_|= _D_<sup>test</sup><br>_X_|_D_<sup>train</sup><br>_F_|= _D_<sup>test</sup><br>_F_|_D_<sup>test</sup><br>query|<sup>=</sup> <sup>_D_test</sup><br>_X_|
|---|---|---|---|---|---|---|
|Skewed covariance||✓|||||
|_d_/2-dimensional subspace||✓|||||
|Scale inputs||✓|||||
|Noisy output||||✓|||
|Scale weights||||✓|||
|Different Orthants||✓||||✓|
|Orthogonal query||||||✓|
|Query matches example||||||✓|



**Skewed covariance.** (Figure 8a) We sample inputs from _N_ (0, Σ) where Σ is a skewed covariance matrix with eigenbasis chosen uniformly at random and _i_<sup>th</sup> eigenvalue proportional to<sup>1</sup> / _i_<sup>2</sup> . 

**Low-dimensional subspace.** (Figure 8b) We sample prompt inputs from a random _d_ /2 dimensional subspace. That is, we pick a random _d_ /2 dimensional subspace, and draw the prompt inputs from an isotropic Gaussian distribution restricted to this subspace. As a result, it is possible to achieve zero error after _d_ /2 in-context examples. 

**Prompt scale.** (Figure 9) We consider the setting where the prompt scale between training and inference is different. We either scale the prompt inputs or the weight vectors, by a factor _{_ 1/3, 1/2, 2, 3 _}_ . 

**Noisy linear regression.** (Figure 8c) We add noise to each prompt output, that is, the _i_<sup>th</sup> output is equal to _w_<sup>_T_</sup> _xi_ + _ϵi_ where _ϵi ∼ N_ (0, _d_ /20). 

**Different orthants for in-context and query inputs.** (Figure 8f) We fix the sign of each coordinate to be positive or negative for all in-context inputs _xi_ (at random), and draw _x_ query (as before) i.i.d. from _N_ (0, _Id_ ). As a result, all in-context inputs lie in the same orthant, while the query input lies in another orthant with high probability. 

**Query input orthogonal to in-context inputs.** (Figure 8d) We choose the query input randomly in the space orthogonal to the space spanned by in-context example inputs. That is, we draw the query input from an isotropic Gaussian distribution restricted to the subspace orthogonal to the space spanned by the in-context examples. Thus, the optimal normalized error is 1 for any number of in-context examples (there can be at most _d −_ 1 in-context examples for an orthogonal query to exist). 

**Query input matches an in-context example.** (Figure 8e) We set the query input equal to one of the in-context examples chosen uniformly at random. Thus it’s possible to achieve zero error since the in-context examples include the correct prediction for the query input already. 

23 



<!-- Start of picture text -->
1.50 1.50 1.50<br>1.25 1.25 1.25<br>1.00 1.00 1.00<br>0.75 0.75 0.75<br>0.50 0.50 0.50<br>0.25 0.25 0.25<br>0.00 0.00 0.00<br>0 10 20 30 40 0 10 20 30 40 0 10 20 30 40<br>in-context examples in-context examples in-context examples<br>(a) skewed covariance (b)  d /2-dimensional subspace (c) noisy output<br>1.50 1.50 1.50<br>1.25 1.25 1.25<br>1.00 1.00 1.00 Transformer<br>0.75 0.75 0.75 Least Squares<br>3-Nearest Neighbors<br>0.50 0.50 0.50 Averaging<br>0.25 0.25 0.25<br>0.00 0.00 0.00<br>0 5 10 15 0 10 20 30 40 0 10 20 30 40<br>in-context examples in-context examples in-context examples<br>(d) orthogonal query (e) query matches in-context example (f) different orthants<br>squared error<br>squared error<br><!-- End of picture text -->

Figure 8: _In-context learning on out-of-distribution prompts._ We evaluate the model trained to in-context learn linear functions on prompt distribution that deviates from the training prompt distribution. In general, the model error degrades gracefully and closely tracks that of the least squares estimator. 



<!-- Start of picture text -->
1.0 1.0 scale<br>1/3<br>0.8 0.8<br>1/2<br>0.6 0.6 1<br>2<br>0.4 0.4 3<br>Least Squares<br>0.2 0.2<br>0.0 0.0<br>0 10 20 30 40 0 10 20 30 40<br>in-context examples in-context examples<br>(a) scaled  x , Transformer (b) scaled  w , Transfomer<br>squared error<br><!-- End of picture text -->

Figure 9: _In-context learning robustness to prompt scaling._ We evaluate the model trained to in-context learn linear functions when we scaled the prompt inputs _x_ or the weight of the function class _w_ . The model appear to be quite robust to scaling _w_ but their performance degrades when scaling the inputs up or down by a factor of 3. 

24 

### **B.3 Effect of problem dimension and model capacity** 

We plot the model error for additional out-of-distribution prompts in Figure 10 for 2 _d_ in-context examples (with the exception of orthogonal queries where we use _d −_ 1 in-context examples). 

Similar to the settings in Section 6 (skewed covariance and different orthants), accuracy improves with capacity in most cases. One exception is scaling _x_ (Figure 10e), in which case we do not see any clear trend. In the case of noisy output (Figure 10b), the accuracy almost saturates at 1.2M parameters, close to the error of the least squares estimator. In the case of orthogonal query input (Figure 10c), the model achieves the optimum error of 1 even with the tiny model with 0.2M parameters. 



<!-- Start of picture text -->
dimensions<br>10 20 30 40 50<br>10 1 10 1 10 1<br>10 0 10 0 10 0<br>10 1 10 1 10 1<br>10 2 10 2 10 2<br>10 3 10 3 10 3<br>10 4 10 4 10 4<br>0.2M 1.2M  9.5M 0.2M 1.2M  9.5M 0.2M 1.2M  9.5M<br>number of parameters number of parameters number of parameters<br>(a)  d /2-dimensional subspace (b) noisy output (c) orthogonal query<br>10 1 10 1 10 1<br>10 0 10 0 10 0<br>10 1 10 1 10 1<br>10 2 10 2 10 2<br>10 3 10 3 10 3<br>10 4 10 4 10 4<br>0.2M 1.2M  9.5M 0.2M 1.2M  9.5M 0.2M 1.2M  9.5M<br>number of parameters number of parameters number of parameters<br>(d) query matches in-context examples (e) scale  x  by a factor of 2 (f) scale  w  by a factor of 2<br>squared error<br>squared error<br><!-- End of picture text -->

Figure 10: _The effect of model capacity and problem dimension for in-context learning performance on out-ofdistribution prompts._ We train Transformer models of varying capacity to in-context learn linear function in varying dimensions _d_ . We plot the error with 2 _d_ in-context examples (or _d −_ 1 for orthogonal queries). We find that capacity helps in most cases, with the exception of scaling _x_ where we find no clear trend. For each setting, we train 3 models with different random seeds, and show the median error (solid lines), and the minimum and maximum errors (shaded region). (See Figures 6b, 6c in the main text for the corresponding plots on different-orthants and skewed-covariance.) 

25 

### **B.4 Training variance** 

In Figure 11, we show the variance in error across training runs for the standard Transformer model (9.5M parameters). We plot the squared error for 3 models (with different random seeds) each for _d ∈ {_ 10, 20, 30, 40, 50 _}_ , trained to in-context learn linear functions. The error is quite concentrated in the standard setting as well as for most out-of-distribution prompts. In the different-orthants and skewed-covariance settings, we observe a high variance for higher dimensional problems ( _d ≥_ 30). However, in Section 6, we saw that the error in these settings usually decreases as we increase the model size. In the setting where we scale _x_ , there is high variance even when _d_ = 10. 



<!-- Start of picture text -->
Transformer, trial 1 Transformer, trial 2 Transformer, trial 3 Least Squares<br>10 dimensions 20 dimensions 30 dimensions 40 dimensions 50 dimensions<br>1.0 1.0 1.0 1.0 1.0<br>0.5 0.5 0.5 0.5 0.5<br>0.0 0.0 0.0 0.0 0.0<br>0 10 20 0 20 40 0 30 60 0 40 80 0 50 100<br>in-context examples in-context examples in-context examples in-context examples in-context examples<br>(a) standard<br>10 dimensions 20 dimensions 30 dimensions 40 dimensions 50 dimensions<br>1.0 1.0 1.0 1.0 1.0<br>0.5 0.5 0.5 0.5 0.5<br>0.0 0.0 0.0 0.0 0.0<br>0 10 20 0 20 40 0 30 60 0 40 80 0 50 100<br>in-context examples in-context examples in-context examples in-context examples in-context examples<br>(b) different orthants<br>10 dimensions 20 dimensions 30 dimensions 40 dimensions 50 dimensions<br>1.0 1.0 1.0 1.0 1.0<br>0.5 0.5 0.5 0.5 0.5<br>0.0 0.0 0.0 0.0 0.0<br>0 10 20 0 20 40 0 30 60 0 40 80 0 50 100<br>in-context examples in-context examples in-context examples in-context examples in-context examples<br>(c) skewed covariance<br>10 dimensions 20 dimensions 30 dimensions 40 dimensions 50 dimensions<br>1.0 1.0 1.0 1.0 1.0<br>0.5 0.5 0.5 0.5 0.5<br>0.0 0.0 0.0 0.0 0.0<br>0 10 20 0 20 40 0 30 60 0 40 80 0 50 100<br>in-context examples in-context examples in-context examples in-context examples in-context examples<br>(d)  d /2-dimensional subspace<br>squared error<br>squared error<br>squared error<br>squared error<br><!-- End of picture text -->

Figure 11: _Errors for models trained with different random seeds._ For each dimension, we train three models with different random seeds and show the corresponding error curves. 

26 



<!-- Start of picture text -->
Transformer, trial 1 Transformer, trial 2 Transformer, trial 3 Least Squares<br>10 dimensions 20 dimensions 30 dimensions 40 dimensions 50 dimensions<br>1.0 1.0 1.0 1.0 1.0<br>0.5 0.5 0.5 0.5 0.5<br>0.0 0.0 0.0 0.0 0.0<br>0 10 20 0 20 40 0 30 60 0 40 80 0 50 100<br>in-context examples in-context examples in-context examples in-context examples in-context examples<br>(e) noisy output<br>10 dimensions 20 dimensions 30 dimensions 40 dimensions 50 dimensions<br>1.0 1.0 1.0 1.0 1.0<br>0.5 0.5 0.5 0.5 0.5<br>0.0 0.0 0.0 0.0 0.0<br>0 5 0 10 0 10 20 0 20 0 20 40<br>in-context examples in-context examples in-context examples in-context examples in-context examples<br>(f) orthogonal query<br>10 dimensions 20 dimensions 30 dimensions 40 dimensions 50 dimensions<br>1.0 1.0 1.0 1.0 1.0<br>0.5 0.5 0.5 0.5 0.5<br>0.0 0.0 0.0 0.0 0.0<br>0 10 20 0 20 40 0 30 60 0 40 80 0 50 100<br>in-context examples in-context examples in-context examples in-context examples in-context examples<br>(g) query matches in-context example<br>10 dimensions 20 dimensions 30 dimensions 40 dimensions 50 dimensions<br>1.0 1.0 1.0 1.0 1.0<br>0.5 0.5 0.5 0.5 0.5<br>0.0 0.0 0.0 0.0 0.0<br>0 10 20 0 20 40 0 30 60 0 40 80 0 50 100<br>in-context examples in-context examples in-context examples in-context examples in-context examples<br>(h) scaled  x  by a factor of 2<br>10 dimensions 20 dimensions 30 dimensions 40 dimensions 50 dimensions<br>1.0 1.0 1.0 1.0 1.0<br>0.5 0.5 0.5 0.5 0.5<br>0.0 0.0 0.0 0.0 0.0<br>0 10 20 0 20 40 0 30 60 0 40 80 0 50 100<br>in-context examples in-context examples in-context examples in-context examples in-context examples<br>(i) scaled  w  by a factor of 2<br>squared error<br>squared error<br>squared error<br>squared error<br>squared error<br><!-- End of picture text -->

Figure 11: (continued) _Errors for models trained with different random seeds._ For each dimension, we train three models with different random seeds and show the corresponding error curves. 

27 

### **B.5 Curriculum** 

In Figure 12, we show the training loss of the Transformer model trained to in-context learn linear functions, with and without a curriculum. Specifically, given a random training prompt sequence ( _x_ 1, _f_ ( _x_ 1), _x_ 2, _f_ ( _x_ 2), ˆ . . ., _xk_ cur, _f_ ( _xk_ cur )), let _yi_ be the model’s prediction for the _i_<sup>th</sup> input (meant to approximate _f_ ( _xi_ )). For each such prompt, we consider the loss given by the normalized squared error averaged over all prompt prefixes 



At each training step, we plot the loss averaged over a batch of 64 random prompts. For training with curriculum, _k_ cur is gradually increased to 2 _d_ + 1 as described in Section A.2. For training without curriculum _k_ cur = 2 _d_ + 1 at all times. 

Note that the loss often increases in the beginning as we train the model with curriculum. This is due to a sharp increase in the loss at steps where we increase the effective dimensionality ( _d_ cur) of prompt inputs ( _xi_ ). There are two reasons for this increase: (i) variance of the target output ( _f_ ( _xi_ ) = _w_<sup>_⊤_</sup> _xi_ ) increases, so even the optimum loss is larger, (ii) the model performance is worse for the prompt inputs with increased effective dimension. After each such step where we increment _d_ cur, the loss starts to decrease again until the next increment. The overall trend in the loss looks upward when the sharp increase dominates the decrease that follows. Some observations worth highlighting are as follows. 

**Curriculum drastically speed-ups training.** For functions in 20 or more dimensions, curriculum allows us to train a low-error model often 4 times faster. Moreover, training without curriculum does not always succeed within our training budget (500k steps), e.g., for one run with _d_ = 30 and _all_ runs with _d_ = 50, the loss does not decrease at all without curriculum. 

**Initial lull without curriculum.** For training without curriculum, we observe that the loss does not decrease for relatively a long period in the beginning, and starts to decrease sharply thereafter. There is a large variance in the length of this period for any fixed dimension, and the average length seems to increase with dimension. This period is almost non-existent for smaller dimensions (e.g., see the plot for _d_ = 10), and therefore we do not observe such a period while training with curriculum where we start training with inputs lying in a 5 dimensional subspace. 

**Curriculum does not affect final performance significantly.** For our core setting ( _d_ = 20), where we are able to train the model to low error even without curriculum, we do not observe any qualitative differences in the error in most cases (both with and without distribution shifts). One exception is the case with skewed covariance, where the model trained without curriculum seems to do slightly better. We plot the error curves for the standard, different orthants and skewed covariance cases in Figure 13. 

28 



<!-- Start of picture text -->
1.0 1.0 1.0<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0.0 0.0 0.0<br>0 100k 0 100k 200k 300k 400k 500k 0 100k 200k 300k 400k 500k<br>training steps training steps training steps<br>(a) 10 dimensions (b) 20 dimensions (c) 30 dimensions<br>1.0 1.0<br>0.8 0.8<br>0.6 0.6<br>with curriculum<br>0.4 0.4<br>without curriculum<br>0.2 0.2<br>0.0 0.0<br>0 100k 200k 300k 400k 500k 0 100k 200k 300k 400k 500k<br>training steps training steps<br>(d) 40 dimensions (e) 50 dimensions<br>loss loss loss<br>loss loss<br><!-- End of picture text -->

Figure 12: _Loss progression during training with and without curriculum._ For each dimension, we show the loss progression with 3 random seeds each for training with and without curriculum. The vertical dashed line shows the point at which the effective dimension of prompt inputs _d_ cur reaches the actual dimension _d_ , after which training with and without curriculum have the same prompt distribution. The horizontal dashed line shows the optimum expected loss. There is a drastic speedup in training with curriculum. Without curriculum, there is an initial relatively long period where the loss does not decrease. For each dimension, there is a large variance in the length of this period, and the average length seems to increase with dimension. 



<!-- Start of picture text -->
1.2 1.2 1.2<br>Transformer<br>1.0 1.0 1.0 Transformer, no curr iculum<br>Least Squares<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0.0 0.0 0.0<br>0 10 20 30 40 0 10 20 30 40 0 10 20 30 40<br>in-context examples in-context examples in-context examples<br>(a) standard (b) different orthants (c) skewed<br>squared error<br><!-- End of picture text -->

Figure 13: _In-context learning performance for models trained with and without curriculum_ . We show the performance for models trained with and without curriculum for in-context learning linear functions ( _d_ = 20). We did not observe any major qualitative difference in performance between the two settings in most cases. One exception is the case with skewed covariance where the model trained without curriculum does better. 

29 

### **B.6 Effect of number of distinct prompts/functions seen during training** 

Here, we investigate the effect of amount of training data required for in-context learning linear functions. First, we consider the effect of number of distinct prompts encountered during training. For this, we create a set _Sp_ of _np_ randomly generated prompts, where each prompt in _Sp_ is generated by sampling a weight vector and prompt inputs from _N_ (0, _Id_ ). We generate random prompts during training by sampling uniformly from this set. As before, we train the model for 500 _k_ steps with a batch size of 64. We observe (see Figure 14) that a model trained with _np_ = 100 _k_ is able to achieve non-trivial error and a model trained with _np_ = 1 _M_ achieves error close to that of the unrestricted model (trained with 32 _M_ distinct prompts). Recall that with curriculum learning, we zero out some of the coordinates of prompt inputs in the beginning of training, which will increase the total number of prompts the model sees during training. Therefore we do not use curriculum learning for this study to avoid inflating the number of distinct prompts seen during training. 

Second, we consider the effect of number of distinct weight vectors (equivalently, distinct functions) encountered during training. For this, we create a set _Sw_ of _nw_ weight vectors where each weight _w_ is drawn i.i.d. from _N_ (0, _Id_ ). To generate a training prompt, ( _x_ 1, _w_<sup>_⊤_</sup> _x_ 1, . . . , _xk_ , _w_<sup>_⊤_</sup> _xk_ ), we draw prompt inputs ( _xi_ s) i.i.d. from _N_ (0, _Id_ ) as in the unrestricted setting, and sample _w_ uniformly at random from _Sw_ . Thus while we sample from a finite set of weight vectors, we sample fresh inputs at each step. As before, we train the model for 500 _k_ steps with a batch size of 64. Here, we observe (see Figure 14) that the model trained with as few as 10 _k_ distinct weight vectors achieves error close to the unrestricted model (trained with 32 _M_ distinct functions). We use curriculum learning for this study as in our standard setting. Recall that with curriculum learning, we only zero out some coordinates of prompt inputs in the beginning, so this does not change the number of distinct weight vectors seen by the model during training. 



<!-- Start of picture text -->
1.0 1.0 #in-context examples<br>10<br>0.8 0.8 20<br>0.6 0.6 30<br>40<br>0.4 0.4<br>0.2<br>0.2<br>0.0<br>0.0<br>1K 10K 100K 1M 10M32M 100 1K 10K 100K 32M<br>#prompts used for training #weight vectors used for training<br>squared error squared error<br><!-- End of picture text -->

Figure 14: _Effect of number of distinct prompts/functions seen during training._ We plot the squared error for models trained to in-context linear functions, as we increase the number of distinct prompts and distinct weight vectors (equivalently, distinct functions) seen during training. (Note that 32M corresponds to the unrestricted model where we sample fresh prompts at each training step.) The models are able to achieve error close to that of the unrestricted model with 1 _M_ distinct prompts or 10 _k_ distinct weight vectors. 

30 

### **B.7 Can memorization explain model performance?** 

In Section 3.1, we discussed that memorization of prompts seen during training cannot explain model performance. This is because the probability of the model encountering a training prompt similar to the one used for testing is astronomically low—the prompt inputs alone lie in a 800-dimensional space when predicting with 2 _d_ in-context examples ( _d_ = 20). 

Moreover, even considering the possibility that the model encountered a similar _weight vector_ during training cannot explain its performance. Let _Sw_ be the set of weight vectors used to generate training prompts. At inference time, given a prompt with in-context examples generated using a weight vector _w⋆_ , suppose the model is somehow able to find the best weight vector _w_ ˆ in _Sw_ minimizing the normalized squared error on query inputs: 



Taking expectation over the weight vector _w⋆_ , we get the expected normalized squared error of the model (with respect to randomly drawn in-context examples and query inputs): 



To empirically estimate this quantity, we sample _nw_ weight vectors from _N_ (0, _Id_ ) (with _d_ = 20) that form the set _Sw_ , and 500 weight vectors from _N_ (0, _Id_ ) to estimate the outer expectation. We do this 20 times, freshly sampling the 500 weight vectors and the vectors comprising _Sw_ each time, and compute the mean of the 20 estimates obtained. When _nw_ = 32 _M_ (number of weight vectors encountered in our standard training setup), we get a mean of 0.216 (standard deviation 0.004). However, our model is able to achieve an expected error of less than 0.001 for prompts with 2 _d_ in-context examples. Similarly, when _nw_ = 10, 000, we get a mean of 0.505 (standard deviation 0.006), while a model trained on prompts generated using 10, 000 distinct weight vectors is able to achieve a much smaller error (see Figure 14). 

Thus we can conclude that the model cannot be relying on memorization of the training prompts or weight vectors, and is encoding a more sophisticated algorithm capable of in-context learning linear functions that are very different from those seen during training. 

31 

