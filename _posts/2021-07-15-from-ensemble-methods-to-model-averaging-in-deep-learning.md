---
layout: post
title: "From Ensemble Methods to Model Averaging in Deep Learning"
date: "2021-7-15"
author: "Hawren Fang"
usemathjax: true
tags: boosting snapshot-ensembles fast-geometric-ensembles stochastic-weight-averaging lookahead-optimizer
---

In machine learning,
ensemble methods create multiple models \\(M_1,M_2,\dots,M_n\\)
and fuse the results from them to achieve the test accuracy
better than any of \\(M_1,M_2,\dots,M_n\\).
In a number of machine learning competitions,
the winning solutions involved ensemble learning.
For example, the [Netflix prize winner][netflix_prize_winner]
is an ensemble of 3 solutions to deliver
excellent collaborative filtering results.

[netflix_prize_winner]: https://www.wired.com/2009/09/bellkors-pragmatic-chaos-wins-1-million-netflix-prize/

In deep learning, neural networks can be relatively expensive,
in both training and inference.
It restricts the applicability of ensemble methods
in the productization of deep learning techniques.

In this blog post, we review the ensemble methods used in deep learning, and
discuss the model averaging methods implicitly supported or inspired by ensemble learning.
These model averaging methods improve the predictive performance 
at little or limited extra computational cost.

- [Ensembles by Averaging the Predictions](#ensembles-by-averaging-the-predictions)
  - [Simple Averaging](#simple-averaging)
  - [Boosting](#boosting)
  - [Snapshot Ensembles](#snapshot-ensembles)
  - [Fast Geometric Ensembles](#fast-geometric-ensembles)
- [Model Averaging](#model-averaging)
  - [Implicit Ensembles](#implicit-ensembles)
  - [Stochastic Weight Averaging](#stochastic-weight-averaging)
  - [Lookahead](#lookahead-optimizer)

# Ensembles by Averaging the Predictions

An approach to combine multiple models in an ensemble is
to average the predictions, weighted or unweighted, of these models.
We begin with simple averaging.


## Simple Averaging

In machine learning, ensemble methods are popular to improve the predictive
performance, by combining the results of multiple models.

In deep learning, training a neural network is a process commonly involved
with random factors, such as randomized initialization and
mini-batch sampling.
We may train several models by the same procedure, yielding
different models due to these random factors.
In the inference phase,
the prediction is by simple averaging.
For example, in the ImageNet classification,
by fusing the results of 7 GoogLeNet models in this way,
the top-5 error rate was 6.67%,
as compared to 7.89% by a single GoogLeNet model [(Szegedy et al., 2015)][googlenet_paper].

In image classification, a common trick to improve accuracy is
averaging the prediction results of multiple crops of the input image
by a single network.
It is different from the ensemble of models discussed here.
Both can be applied jointly, and the reasoning in the following discussion
remains valid.


### Statistical support

The statistical support is briefed as follows.
Consider regression problems.
Given a test case, the predictions by \\(n\\) multiple models can be regarded as
instances of *independent and identically distributed* (i.i.d.) random variables
\\(F_1,F_2,\dots,F_n\\),
since these models are independently trained by the same procedure
with the same randomization.
Hence, simply averaging the predictions is
the outcome of the mean of these i.i.d. random variables,
denoted by \\(G_n=\frac{1}{n}\sum_{i=1}^n F_i\\).
Note that

$$
\begin{eqnarray}
Var(G_n) & = & \frac{1}{n^2} Var(F_1+F_2+\cdots+F_n) \\
       & = & \frac{1}{n^2} (Var(F_1)+Var(F_2)+\cdots+Var(F_n)) \\
       & = & \frac{1}{n} Var(F),
\end{eqnarray}
$$

where we have denoted \\(Var(F)=Var(F_1)=\cdots=Var(F_2)\\) since they are all the same.
Therefore, the prediction accuracy is improved by variance reduction,
which can be seen from \\(Var(G_n)=\frac{1}{n} Var(F)\\).
By the strong law of large numbers, we obtain
\\[
\lim_{n\rightarrow\infty}G_n = E(F).
\\]
That is, \\(G_n\\) converges to the expected value of \\(F\\) as \\(n\rightarrow\infty\\) for sure,
despite the random factors in the training process.

A few points are worth noting.

- Simple averaging is applicable to classification problems,
  where each \\(F_i\\) is a probability distribution, an array of predicted
  class probabilities, for \\(i=1,\dots,n\\).
- In a similar vein, the asymptotic certainty can be derived for
  an ensemble by voting for classification problems.
- This statistical guarantee is also valid for other ensemble
  methods by simple averaging or voting, such as random forests [(Breiman, 2001)][random_forests_paper].


### Remark

> In deep learning, the disadvantage of simple averaging is clear.
> For an ensemble of \\(n\\) networks,
> the cost is \\(n\\) times, in both training and inference.


## Boosting

Boosting refers to a generic sequential ensemble algorithm to improve the accuracy of
*any* given learning method, called the *base learner*.
A specific example may help the clarity of the discussion.
Here we consider AdaBoost [(Freund & Schapire, 1997)][adaboost_paper],
short for adaptive boosting, for binary classification.
A Python-style pseudo-code of AdaBoost is provided as follows.

```python
def AdaBoost(X, Y, base_learner, n):
    '''
    The goal is to trained an ensemble f such that f(x) approximates y.
    We are given supervised training data X, Y, and a base_learner.
    (X[i], Y[i]) is a paired sample for i = 1,...,m; each Y[i] is 1 or -1.
    The base_learner takes training data X, Y with distribution D, and
    outputs a trained model, where D[i] is the probability of sample i.
    The trained model is a binary classifier. More precisely, given x,
    the trained model predicts the class label 1 or -1.
    n is the number of models to be trained by base_learner in the ensemble.
    '''

    m = len(Y)  # number of training samples
    Initialize D[i] = 1/m for i = 0,...,m-1  # uniform distribution
    for t =  0,...,n-1:
        model[t] = base_learner(X, Y, D)
        e = sum(model[t](X) != Y) / m  # training error rate of model[t]
        w[t] = 0.5 * ln((1-e)/e)  # weight of model[t], positive for e<0.5
        # update sample weights D
        for i = 0,...,m-1:
            if model[t](X[i]) != Y[i]:
                # increase importance of those misclassified
                D[i] = D[i] * exp(w[t])
            else:
                # decrease importance of those correctly classified
                D[i] = D[i] * exp(-w[t])
        D = D / sum(D)  # normalize D such that D is a distribution

    ensemble = sum{w[i]*model[i] : i=0,...,n-1}
    # prediction for input x is 1 if ensemble(x)>0 else -1
    return ensemble
```

Algorithm 1. AdaBoost.

As shown in the pseudo-code, right before training a model,
AdaBoost tunes the frequency of each training sample according
to the training error of the previous model, to increase the importance
of those misclassified, and decrease the importance of those correctly classified.
The flow is illustrated in the following picture.

![Boosting]({{ '/assets/images/1280px-Ensemble_Boosting.svg.png' | relative_url }})

Figure 1. The flow of training an ensemble of models by boosting.
Image source: [Wikipedia][wiki_boosting]

AdaBoost is commonly regarded as the first practical boosting algorithm.
It can be generalized for multiclass classification problems [(Zhu et al., 2009)][multiclass_adaboost_paper].
In addition, there are various boosting methods, which are beyond the scope of this blog post.
The readers are referred to [Wikipedia][wiki_boosting] for more information.


### Properties

AdaBoost for binary classification has nice theoretical properties to support that
it is indeed a boosting algorithm.

- The training error of the ensemble decreases exponentially fast to zero
  as \\(n\rightarrow\infty\\), with \\(n\\) the number of combined binary
  classifiers, under the modest assumption that each binary classifier
  performs better than random guessing (i.e. error rate < 0.5) on training
  data.
- It has a provable generalization error bound.

See [Freund & Schapire (1997)][adaboost_paper] for the theoretical details.

In practice, AdaBoost is typically applied with a base learner as a weak
learner, such as decision trees, or even just decision stumps (i.e., decision
trees with a single split).
In theory, AdaBoost is not limited to weak learners;
it is applicable with neural networks [(Schwenk & Bengio, 1997)][adaboost_nn_paper].

The boosting methods share some common properties, such as
every model is trained to improve the ensemble of the former models, and
they can work with virtually any base learner, at least in theory.


### Result

Here is an example to combine the merits of boosting and neural networks.
[Moghimi et al. (2016)][boost_cnn_paper] applied GD-MCBoost [(Saberian & Vasconcelos, 2011)][mcboost_paper]
with convolutional neural networks, called BoostCNN, for multiclass classification.
They reported the improved accuracy by BoostCNN on multiple datasets.
The following picture shows the result on [CIFAR-10][cifar_datasets].

[cifar_datasets]: https://www.cs.toronto.edu/~kriz/cifar.html

![BoostCNN CIFAR-10 result]({{ '/assets/images/mcboost_cnn_cfar10.png' | relative_url }} )

Figure 2. CIFAR-10 classification error rate with single CNN, Bagging (i.e. simple averaging), and BoostCNN.
Image source: [Moghimi et al. (2016)][boost_cnn_paper]

[Moghimi et al. (2016)][boost_cnn_paper] trained 9
convolutional neural networks for each ensemble:
Bagging (i.e. simple averaging), BoostCNNs with and without reset.
A practical trick for BoostCNN is hinted in Figure 2.
The recommendation is to initialize each base learner with the last trained base
learner (without reset). The result is better than that with randomized
weight initialization for all the base learners (with reset).


### Comparison

A comparison between simple averaging and AdaBoost is as follows.

- From the machine learning perspective,
  simple averaging or voting reduces variance, whereas AdaBoost reduces the bias.
- Each model trained by AdaBoost has the purpose to improve the ensemble of
  the former models. In this respect, AdaBoost may require fewer models than
  simple averaging or voting for the comparable accuracy improvement;
  see e.g. the example in Figure 2.
  Hence the computational cost may be reduced.
- The models in AdaBoost are trained sequentially, whereas in the ensemble
  by simple averaging or voting, the models can be independently trained,
  good for parallel computing.
- In inference, the models can be applied separately, and
  the results are fused to form the final prediction.
  This property remains valid in AdaBoost.

The above comments are applicable to other boosting methods, not limited to
AdaBoost.


### Remark

> Boosting still involves multiple models, in both training and inference.
> For an ensemble of \\(n\\) models,
> the cost is \\(n\\) times, in both training and inference.
> The computational expense may still be a concern in some deep learning
> applications.


## Snapshot Ensembles

[Huang et al. (2017a)][snapshot_ensembles_paper] proposed a training scheme
to obtain \\(n\\) models from training 1 model.
The method is called the *snapshot ensemble*.
In one word,
when we train a neural network with a cyclic learning rate schedule,
the iterates are converging to and escaping from multiple local minima,
*snapshotted* to form an ensemble for test-time inference.
This idea is illustrated in the following picture on the right hand side.

![Snapshot ensemble \label{fig:3}]({{ '/assets/images/snapshot_ensemble_illustration.png' | relative_url }})

Figure 3.
**Left:** Optimization with a typical (non-increasing) learning rate schedule.
**Right:** Snapshot ensembling with a cyclic learning rate schedule.
Image source: [Huang et al. (2017a)][snapshot_ensembles_paper]

The cyclic training procedure parallels the idea of simulated annealing
toward global optimization, by allowing the opportunities
to escape from unsatisfactory local minima.


### Cyclic learning rate schedule

A key point in snapshot ensembling (SSE) is the cyclic learning rate schedule,
where the sudden jumps of learning rate allow escaping from local minima,
in each cycle the decreasing learning rate helps the convergence to,
hopefully, another better local minimum.
We take a snapshot of the model weights before raising the learning rate.
During the inference phase, we average the predictions of the model snapshots
to form the final prediction for each test case.

An embodiment of the cyclic learning rate schedule is by *cosine annealing*
[(Loshchilov & Hutter, 2017)][sgdr_paper].
The formula is given as follows.
Within the \\(t\\)-th cycle, the learning rate at \\(j\\)-th iterations is

\\[
\alpha^t_j=\alpha_{min}^t+\frac{1}{2}(\alpha_{max}^t-\alpha_{min}^t) (1+\cos(\frac{j}{C_t})\pi),
\text{ for }
j=1,\dots,C_t-1,
\\]

where \\(C_t\\) is the number of iterations in the \\(t\\)-th cycle, and
\\(\alpha_{max}^t\\) and \\(\alpha_{min}^t\\) are upper and lower bounds of the
learning rate, in the \\(t\\)-th cycle.

In practice, it is reasonable expect the starting iterate is improved from one cycle
to another. Hence we may apply a discount on the learning rate or the
number of iterations for every new cycle.
For example, \\(\alpha_{max}^t=0.8\alpha_{max}^{t-1}\\) and
\\(\alpha_{min}^t=0.8\alpha_{min}^{t-1}\\), or \\(C_t=0.6C_{t-1}\\).

On the other hand, we can simply set \\(\alpha_{max}^t=\alpha_0\\), \\(\alpha_{min}^t=0\\),
and \\(C_t=C\\) for some constant starter learning rate \\(\alpha_0>0\\) and
fixed cycle length \\(C\\) for all cycles \\(t=0,\dots,n-1\\).

The following picture by [Huang et al. (2017a)][snapshot_ensembles_paper] shows
the result of CIFAR-10 training loss of 100-layer DenseNet
[(Huang et al., 2017b)][densenet_paper],
using a cyclic learning rate schedule by cosine annealing,
with \\(\alpha_{max}^t=0.1\\), \\(\alpha_{min}^t=0\\), and 50 epochs,
for all \\(t=0,\dots,n-1\\) (\\(n=6\\) cycles).
The result of a standard learning rate schedule is also shown
in the picture for comparison.

![Training loss vs. learning rate schedules]({{ '/assets/images/cyclic_lr_loss.png' | relative_url }})

Figure 4. CIFAR-10 training loss of 100-layer DenseNet,
with a standard learning rate schedule (blue)
and cyclic cosine annealing (red).
Image source: [Huang et al. (2017a)][snapshot_ensembles_paper]


### Result

With \\(n=6\\) cycles, we may have a snapshot ensemble with up to 6 models.
The final prediction for each test case is from averaging the predictions of
the models in the ensemble.
The following picture displays the CIFAR-10 and CIFAR-100 results,
with various numbers of models in the ensemble, and
two different start learning rates \\(\alpha_0=0.1\\) and \\(\alpha_0=0.2\\).
The test accuracy improvement by snapshot ensembling is clear.

![CIFAR result of snapshot ensembles]({{ '/assets/images/snapshot_ensembles_cifar_result.png' | relative_url }})

Figure 5. CIFAR-10 training loss of 100-layer DenseNet,
with a standard learning rate schedule (blue)
and cyclic cosine annealing (red).
Image source: [Huang et al. (2017a)][snapshot_ensembles_paper]


## Fast Geometric Ensembling

Intuitively, we may conjecture or even think that,
the models found by a training procedure with a cyclic learning rate
are *strict* local minima of the training loss function.
A novel finding by [Garipov et al. (2018)][fge_paper] is that,
for two comparable local minima, there may be a simple path connecting them,
such that along this path the training loss value is stably low,
i.e. no substantial increase.
It is illustrated in the following pictures.

![Illustration of mode connectivity.]({{ '/assets/images/mode_connectivity_illustration.png' | relative_url }})
![Boosting]({{ '/assets/images/1280px-Ensemble_Boosting.svg.png' | relative_url }})

Figure 6. \\(L_2\\)-regularized cross-entropy training loss surface of ResNet-164 on CIFAR-10.
**Left:** Three optima of independently trained models.
The lowered two optima are used in the next two pictures.
**Middle:** A quadratic Bezier curve connecting the two optima, optimized
for the low training loss along the curve.
**Right:** A polygonal chain with one bend, optimized
for the low training loss along the path.
Image source: [Garipov et al. (2018)][fge_paper]

To visualize the training loss as a function of model weights,
we perform an affine transformation from
the high-dimensional weight space to a 2D plane,
which can be uniquely determined by 3 linearly independent models,
as 3 points in the weight space.
That's how the left picture of Figure 6 is obtained,
with 3 independently trained models.

Consider the right picture in Figure 6.
We use a polygonal chain with one bend to connect two models.
We need to determine the bend point, which
uniquely decides the polygonal chain and therefore the 2D plane for visualization.
The bend point is chosen to minimize the integral of the training loss
along the polygonal chain.
Likewise, as shown in the middle picture in Figure 6,
we can obtain a quadratic Bezier curve connecting the two minima
on which curve the training loss stays low.

Now the finding, called *mode connectivity*, is clear.
Given two comparable local minima of the training loss in deep learning,
there may be a simple connecting path, such as a quadratic Bezier curve
or a polygonal chain with one bend, on which the training loss stays
close to the local minimum value.
[Garipov et al. (2018)][fge_paper] reported that
the mode connectivity holds for a wide range of modern deep neural networks,
not limited to the image classification network ResNet-164 on CIFAR-10,
shown in Figure 6.

Inspired by this finding, [Garipov et al. (2018)][fge_paper] proposed
fast geometric ensembling (FGE), which uses the following
piecewise linear learning rate schedule.
At mini-batch iteration \\(i=0,1,\dots\\), the learning rate is:

$$
\alpha(i)= 
  \begin{cases}
    (1-2t(i))\alpha_1 + 2t(i)\alpha_2, & 0 < t(i) \leq \frac{1}{2} \\
    (2-2t(i))\alpha_2 + (2t(i)-1)\alpha_1, & \frac{1}{2} < t(i) \leq 1
  \end{cases},
$$

where \\(t(i)=\frac{1}{c}(\text{mod}(i,c)+1)\\),
with \\(c\\) the cycle length as an even number.
The result of ResNet-164 on CIFAR-10 is as follows.

![Result of ResNet-164 on CIFAR-10.]({{ '/assets/images/fge_result_resnet164_cifar10.png' | relative_url }})

Figure 7. Result of ResNet-164 on CIFAR-10.
**Left:** Plots of learning rate (top), test error (middle), and
distance from the initial weights (bottom),
as a function of iterations of FGE.
The circles indicate the models saved for ensembling.
**Right:** Ensemble performance of FGE and SSE (snapshot ensembles)
as a function of training time.
Image source: [Garipov et al. (2018)][fge_paper]

The key difference between snapshot ensembles (SSE) and
fast geometric ensembles (FGE) is the cyclic learning rate schedule.

- SSE adopts the cyclic cosine annealing,
  with the cycle length on the scale of 20 to 40 epochs.
- FGE employs a piece-wise linear cyclic learning rate schedule,
  with the cycle length on the scale of 2 to 4 epochs.

[Garipov et al. (2018)][fge_paper] decided to use much shorter
learning cycles in FGE,
as their analysis indicated that relatively small steps
in the weight space are sufficient for diverse models.
As shown in the right picture of Figure 7,
FGE outperformed SSE on CIFAR-10 image classification by ResNet-164
ensembles.


### Remark

> In both snapshot ensembling (SSE) and fast geometric ensembling (FGE),
> we train a network with a cyclic learning rate schedule, and
> obtained multiple models to form an ensemble.
> The extra training cost, if any, is negligible.
> However, the inference cost remains a concern, since
> all models in the ensemble are applied for the final predictions.


# Model Averaging

A prediction by an ensemble of neural networks requires
predictions by the neural networks in the ensemble and then merge them.
Here, we consider ensembling in model space, by averaging the models,
which ends up with only *one* model at test time.
Therefore, it addresses or reduces the concern of the inference cost in practice.


## Implicit Ensembles


### Dropout

We start with the *dropout* technique [(Srivastava et al., 2014)][dropout_paper]
for neural network training.
Consider only dense (i.e. fully connected) layers for simplicity.
During the training phase, in every mini-batch iteration,
each node is *dropped* with probability \\(p\\), called dropout rate.
It results in a *thinned* network.
An example is illustrated in the following picture.

![Illustration of dropout]({{ '/assets/images/dropout_illustration.png' | relative_url }})

Figure 8. Illustration of dropout.
**Left:** A standard network.
**Right:** The thinned network after dropout.
Image source: [Srivastava et al. (2014)][dropout_paper]

For a network with \\(N\\) internal nodes,
there are \\(2^N\\) such thinned networks by dropout,
including those with broken layers.
In every mini-batch iteration, we train one of the thinned network.
From this point of view, we are implicitly
training exponentially many networks with extensively shared weights.

After the training, we form a model with weights
as the weighted average the weights of the \\(2^N\\) thinned networks,
weighted according to the frequency in the mini-batch training iterations.
Equivalently, we multiply the weights by \\(1-p\\), with \\(p\\) the dropout rate,
to get the model used in test time *without* dropout.
Therefore, dropout implicitly forms an ensemble of many networks,
while there is only one network explicitly used for prediction!


### Effects of dropout

From the optimization point of view, dropout injects *noise* into the
training, and it has a regularization effect
which improves the generalization.
On the other hand, since the noise *puzzles* the training,
it takes longer to converge.
[Srivastava et al. (2014)][dropout_paper] reported that
a dropout network typically takes 2-3 times longer to train
than its counterpart without dropout.
That's the price to pay for the generalization.

The following picture gives an example of improved test accuracy by dropout
in the classification experiments by [Srivastava et al. (2014)][dropout_paper].

![Test error with and without dropout]({{ '/assets/images/test_error_dropout.png' | relative_url }})

Figure 9. Test error with various network architectures (2 to 4 hidden
layers), with and without dropout.
Image source: [Srivastava et al. (2014)][dropout_paper]


### Inverted dropout

In deep learning software packages,
dropout is commonly realized by *inverted* dropout, which means that
instead of scaling the weights by \\(1-p\\) for the inference network,
we scale the weights by \\(1/(1-p)\\) during the training, and
the weights are intact in the inference phase.
Here \\(p\\) is the dropout rate.

While the inverted dropout delivers the equivalent training behaviors as the
dropout, it provides extra practical convenience, such as:

- For the weight initialization schemes involving variance control, such as
  the Xavier initialization [(Xavier & Bengio, 2010)][xavier_init_paper],
  the statistical support remains valid with the inverted dropout.
  Therefore, modification of the weight initialization scheme is not required.
- With the inverted dropout, the dropout rate can be changed dynamically
  from one iteration to another during the training.


### Other random dropping techniques

As listed below, there are other schemes with *random dropping* during the training.
These methods also lead to implicit ensembles as single networks used for prediction.

- For convolutional layers, spatial dropout [(Tompson et al., 2015)][spatial_dropout_paper] and
  DropBlock [(Ghiasi et al., 2018)][dropblock_paper] are alternatives to dropout.
- DropConnect [(Wan et al., 2013)][dropconnect_paper] drops the connections instead of nodes.
- The stochastic depth technique [(Huang et al., 2016)][stochastic_depth_paper]
  randomly skips *layers* instead of dropping nodes or connections.
- Swapout [(Singh et al., 2016)][swapout_paper] is a more general framework of random dropping,
  in which dropout and the stochastic depth technique are two particular instances.

> The implicit ensembles, with a random dropping scheme during the training,
> are single networks. It means that the prediction is by one network
> instead of many in an explicit ensemble.
> Therefore, the concern of inference cost is addressed to a certain extent.
> On the other hand, the random dropping creates noise, and that may result in
> the longer training for convergence.


## Stochastic Weight Averaging

Recall that both snapshot ensembling [(Huang et al., 2017a)][snapshot_ensembles_paper] and
fast geometric ensembling [(Garipov et al., 2018)][fge_paper] obtain
multiple models from training one neural network with a cyclic learning rate schedule.

Based on fast geometric ensembling (FGE),
[Izmailov et al. (2018)][swa_paper] proposed
*stochastic weight averaging* (SWA), that
uniformly averages these model weights to obtain the final model used for prediction.
As a result, SWA achieves the low test-time inference cost, since there is
only one single model, rather than multiple ones in an ensemble. 

SWA can be interpreted as an approximation to FGE.
[Izmailov et al. (2018)][swa_paper] reported that
SWA performed comparably or close to FGE in test error,
in the experiments on CIFAR-100 classification by various networks.
Both SWA and FGE outperformed the standard training
with the stochastic gradient descent (SGD) optimizer.


### Wider local minima by SWA

A conjecture is that SWA ends up with a wider local minimum.
There are empirical evidence to support that a wider local minimum tends to
lead to better generalization [(Li et al., 2018)][loss_landscape_paper].
To verify this conjecture,
[Izmailov et al. (2018)][swa_paper] use the function for visualization of
training loss:

$$
  \begin{cases}
    w_{\text{SWA}}(t,d)  =  w_{\text{SWA}} + t \cdot d \\
    w_{\text{SGD}}(t,d)  =  w_{\text{SGD}} + t \cdot d
  \end{cases},
$$

where \\(w_{\text{SWA}}\\) and \\(w_{\text{SGD}}\\) are model weights from
SWA training and SGD training, respectively.
\\(d\\) is a random direction, drawn from a uniform distribution on the unit sphere,
and \\(t\\) is the distance from \\(w_{\text{SWA}}\\) or \\(w_{\text{SGD}}\\).

Now we can plot training loss and test error of
\\(w_{\text{SWA}}(t,d)\\) and \\(w_{\text{SGD}}(t,d)\\) in terms of distance \\(t\\),
as shown in the following pictures.

![Loss sensitivity: SWA vs. SGD]({{ '/assets/images/swa_vs_sgd_1d_loss_landscape.png' | relative_url }})

Figure 10. Plots of test error (**left**) and
\\(L_2\\)-regularized cross-entropy training loss (**right**),
as functions of distance \\(t\\).
Each line (10 lines in each setting) corresponds to
a random direction \\(d\\) as a unit vector.
The testbed was ResNet-164 on CIFAR-10.
Image source: [Izmailov et al. (2018)][swa_paper]

In Figure 10, from the right picture, we see that SWA leads to a wider local
minimum than SGD, while SGD results in a lower local minimum than SWA.
As shown in the left picture, the wider local minimum by SWA achieves
lower test error, as compared to SGD. 
It supports that SWA helps find wider local minima,
which may improve the generalization and that eventually translates to
better test performance.


### Practical consideration

To average \\(n\\) model weights from \\(n\\) learning cycles,
we do not really need a copy for each of the \\(n\\) models.
During the training, we need keep only two models,
the current iterate \\(w\\), and the averaged model \\(w_{\text{SWA}}\\) so far.
At the end of each learning cycle, we use the update formula:

\\[
w_{\text{SWA}} \leftarrow \frac{w_{\text{SWA}}\cdot n_{\text{model}}+w}{n_{\text{model}}+1},
\\]

where \\(n_{\text{model}}\\) is the number of models averaged for \\(w_{\text{SWA}}\\)
before the update.


### Remark

> Stochastic weight ensembling (SWA) trains one network with a cyclic
> learning rate schedule. Each learning cycle yields one model.
> The model from averaging the weights of these models is used for
> test-time prediction. Empirically, SWA may lead to wider local minima,
> which eventually translate to improved test accuracy.
>
> During the training, SWA requires an extra space to store the averaged
> weights, updated incrementally from one learning cycle to another.
> The extra computation for the training,
> in terms of the number of floating-point operations, is negligible.
> There is no impact on the inference cost, since only one model
> is used in test time.


## Lookahead Optimizer

Stochastic weight ensembling [(Izmailov et al., 2018)][swa_paper]
averages model weights of local minima to obtain a model,
practically at a wider local minimum that improves the predictive performance
at test time.
On the other hand, it does not alter the optimization procedure.
Hence there is no acceleration of training.


### An overview of Lookahead

[Zhang et al. (2019)][lookahead_paper] introduces a simple but effective
scheme, called *Lookahead*, that can accelerate the training.
Lookahead frequently averages the weights of two models
during the training process.
To be precise, for every \\(k\\) mini-batch iterations,
we average the current iterate and the iterate \\(k\\) mini-batch iterations
earlier, and then starting from the averaged iterate for the next mini-batch
iterations.
An illustration with \\(k=5\\) is given in the following picture.

![Illustration of Lookahead Optimization]({{ '/assets/images/lookahead_optimizer_illustration.png' | relative_url }})

Figure 11. Illustration of Lookahead Optimization.
Image source: [Zhang's talk][lookahead_slides]

[lookahead_slides]: https://michaelrzhang.github.io/assets/lookahead_slides.pdf

In Figure 11, the iterates are in sequence:

\\[
\underline{\theta_{1,0}},\theta_{1,1},\dots,\theta_{1,5},
\underline{\theta_{2,0}},\theta_{2,1},\dots,\theta_{2,5},
\underline{\theta_{3,0}},\theta_{3,1},\dots,
\\]

where we have underlined the initial iterate \\(\underline{\theta_{1,0}}\\) and the averaged
iterates \\(\underline{\theta_{2,0}}\\) and \\(\underline{\theta_{3,0}}\\).
They are called *slow weights*, whereas the others are called *fast weights*.

We may allow weighted average, as:

\\[
\theta_{t+1,0}
=
(1-\alpha)\theta_{t,0}+\alpha\theta_{t,k}
\equiv
\theta_{t,0}+\alpha(\theta_{t,k}-\theta_{t,0}),
\\]

where \\(t\\) is the index of slow weights,
\\(k\\) is the synchronization period, and
\\(\alpha\\) is the step length.
In the illustration in Figure 11, \\(k=5\\) and \\(\alpha=0.5\\).

Note that slow weights can be written as an exponential moving average (EMA) of
the fast weights at the end of each inner-loop.

$$
\begin{eqnarray}
\theta_{t+1,0}
& = &
\alpha\theta_{t,k} + (1-\alpha)\theta_{t,0} \\
& = &
\alpha [ \theta_{t,k} + (1-\alpha)\theta_{t-1,k} + \cdots + (1-\alpha)^{t}\theta_{0,k} ] + (1-\alpha)^{t+1} \theta_{0,0}.
\end{eqnarray}
$$

It underscores the claim that "an exponentially-decayed moving average typically works much better in practice"
[(Martens, 2020)][natural_gradient_insights_paper].


### Details

A Python-style pseudo-code of Lookahead is provided as follows.
For simplicity, we set the number of mini-batch iterations, but no
convergence criterion, and no model selection scheme.

```python
def Lookahead_optimizer(theta, L, k, alpha, base_optimizer, num_iter):
    '''
    theta is the initial weights (parameters) of the neural network;
    L is the training loss function;
    k is the synchronization period;
    alpha is the step length of Lookahead;
    base_optimizer is the optimizer (e.g. SGD) to train the neural network.
    num_iter is the number of mini-batch iterations.
    '''

    T = num_iter // k  # number of updates by model averaging
    for t =  0,...,T:
        theta0 = theta  # make a copy of slow weights
        # now mini-batch iterations in the inner loop
        for i = 0,...,k-1:
            sample mini-batch of training data D
            # update fast weights
            theta = base_learner.update(L, theta, D)
        # update slow weights
        theta = theta0 + alpha*(theta-theta0)

    # return the last iterate/model
    return theta
```

Algorithm 2. Lookahead optimizer.


A few points are worth noting:
- Lookahead builds upon the optimizer for neural network training.
  There is virtually no restriction to the optimizer, which can be SGD
  (stochastic gradient descent), or Adam [(Kingma & Ba, 2015)][adam_paper], etc.
- The synchronization period \\(k\\) is supposed to be small, such as 5 or 10.
- We should have the step length \\(0.5\leq\alpha<1\\), explained as follows.
  - The latter iterate \\(\theta_{t,k}\\) is expected to be more reliable than \\(\theta_{t,0}\\);
    hence \\(\alpha\geq0.5\\).
  - \\(\alpha=1\\) means turning off Lookahead, and \\(\alpha\\) should
    be restricted within 1 for positive weighting; hence \\(\alpha<1\\).


### Result

In Figure 12 is the result of experiments on CIFAR-100 image classification
with synchronization period \\(k=5,10\\) and step length \\(\alpha=0.5,0.8\\)
[(Zhang et al., 2019)][lookahead_paper].
Note that \\(k\\) and \\(\alpha\\) are the only two hyper-parameters of Lookahead.

![Effects of hyper-parameters of Lookahead, on CIFAR-100]({{ '/assets/images/lookahead_hyperparameters_cifar100.png' | relative_url }})

Figure 12. Effects of hyper-parameters of Lookahead, in the experiments on CIFAR-100.
**Left:** training loss vs. number of epochs.
**Right:** Validation accuracy table (mean ± stdev).
Source: [Zhang et al. (2019)][lookahead_paper]

We can see from Figure 12 that, in the experiments on CIFAR-100 image classification:
- The synchronization period \\(k=5\\) works better than \\(k=10\\), in terms of
  not only better accuracy and also lower variation of accuracy in multiple repeats.
  It perhaps implies the value of more frequent model averaging.
- The step length \\(\alpha=0.5\\) gives better result than \\(\alpha=0.8\\).
  With a handful of mini-batch iterations, such as \\(k=5\\) or \\(k=10\\),
  the model quality change may not be that significant.
  Hence averaging the two models evenly, i.e. \\(\alpha=0.5\\), may work
  better than the aggressive step length \\(\alpha=0.8\\) of Lookahead.
- In all settings, Lookahead outperforms the baseline SGD (i.e. turning off Lookahead).


Another evaluation is the sensitivity to the learning rate of the base optimizer.
The result on CIFAR-10 image classification in shown in Figure 13 [(Zhang et al., 2019)][lookahead_paper],
where the shadows reflect the variations in multiple repeats.
We see that Lookahead not only stabilizes the training,
but help the result less sensitive to the change of learning rate.

![Sensitivity of learning rate, Lookahead vs SGD, on CIFAR-10]({{ '/assets/images/lookahead_vs_sgd_cifar10.png' | relative_url }})

Figure 13. Sensitivity of learning rate, Lookahead vs. the baseline SGD, on
CIFAR-10 image classification. Source: [Zhang et al. (2019)][lookahead_paper]

Consider the observation that averaging model weights
leads to wider minimum and better generalization [(Izmailov et al., 2018)][swa_paper].
It can be empirically verified in the context of Lookahead, which applies
frequent model averaging.

Figure 14 [(Zhang et al., 2019)][lookahead_paper] gives the plot of test
accuracy vs. mini-batch iterations on epoch 65,
where fast weights are marked in blue solid lines (inner iterations)
and slow weights are marked in the green dashed line (outer iterations).

![Test accuracy, slow weights vs. fast weights of Lookahead]({{ '/assets/images/slow_vs_fast_weights_test_error.png' | relative_url }})

Figure 14. Test accuracy, slow weights vs. fast weights of Lookahead.
Source: [Zhang et al. (2019)][lookahead_paper]

Mini-batch iterations, which update fast weights, are expectedly reducing
the training loss, as the purpose of optimization in the training.
On the other hand, as shown in Figure 14, when it is close to the convergence,
updating the fast weights may more likely hurt the test accuracy,
while slow weights help regain the predictive performance.

Note that Lookahead is not only useful for convolution neural networks for image
classification, but also helpful for sequence models.
See [Zhang et al. (2019)][lookahead_paper] for more information.


### Merits

Lookahead has the following merits.

- Lookahead can be regarded as a simplified version of Anderson acceleration
  [(Anderson, 1965)][anderson_mixing_paper], which is a quasi-Newton method
  [(Fang & Saad, 2009)][nonlinear_acceleration_paper], and
  therefore accelerated convergence may be expected.
- Lookahead performs frequent model averaging,
  which may lead to wider local minima and better generalization [(Izmailov et al., 2018)][swa_paper] .

These merits are observed in the experiments [(Zhang et al., 2019)][lookahead_paper].


### Remark

> Lookahead wraps around an optimizer, such as SGD or Adam, for neural network training.
> It can not only accelerate the training (in terms of number of epochs or mini-batch iterations) 
> but also improve the generalization and predictive performance.
>
> During the training, Lookahead requires an extra space to store the incrementally
> updated slow weights, as  an exponential moving average (EMA) of the fast weights
> at the end of each inner-loop.
> There is no extra inference cost with Lookahead, as it does not change the
> picture that only one model is used in test time.


### Afternote

Since Lookahead can wrap around virtually any optimizer for deep learning,
what if we pick the most effective optimizer off the shelf?
Consider RAdam, short for *rectified Adam* [(Liu et al., 2020)][radam_paper],
which is commonly regarded as a significant improvement over
the already successful Adam [(Kingma & Ba, 2015)][adam_paper].

In short, RAdam improves the training stability
by using un-adapted momentum (i.e. with momentum but without adaptive learning rate)
at early iterations, where the exponential moving average (EMA) of each squared
gradient component is not that reliable due to insufficient gradient samples.
It also rectifies the adaptive learning to improve the reliability.

The combination of Lookahead and RAdam, called *Ranger* optimizer
[(Wright, 2019)][ranger_blog_post], is synergistic,
since we can take advantage of all nice properties from both methods.
The ranger optimizer helped established 12 leader-board records
on the FastAI global leader-boards.
See [Wright, (2019)][ranger_blog_post] and the links therein for more
information.

Cited as:
```
@misc{hawren2021ensemble,
      title        = "From Ensemble Methods to Model Averaging in Deep Learning",
      author       = "Fang, Hawren",
      year         = 2021,
      howpublished = "http://lilianweng.github.io/hawren-log/2021/7/15/from-ensemble-methods-to-model-averaging-in-deep-learning.html"
}
```


# References

[1] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich,
["Going deeper with convolutions,"][googlenet_paper]
CVPR, 2015.

[googlenet_paper]: https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html

[2] L. Breiman,
["Random forests,"][random_forests_paper]
Machine Learning, Vol. 45, pp. 5-32, 2001.

[random_forests_paper]: https://link.springer.com/article/10.1023/A:1010933404324

[3] Y. Freund and R. E. Schapire
["A decision-theoretic generalization of on-line learning and an application to boosting,"][adaboost_paper]
Journal of Computer and System Sciences, Vol. 55, pp. 119-139, 1997.

[adaboost_paper]: https://www.sciencedirect.com/science/article/pii/S002200009791504X

[4] J. Zhu, H. Zou, S. Rosset, and T. Hastie,
["Multi-class AdaBoost"][multiclass_adaboost_paper]
Statistics and Its Interface, Vol. 2, pp 349-360, 2009.

[multiclass_adaboost_paper]: https://www.intlpress.com/site/pub/pages/journals/items/sii/content/vols/0002/0003/a008/

[5]
[“Boosting (machine learning),”][wiki_boosting]
Wikipedia.

[wiki_boosting]: https://en.wikipedia.org/wiki/Boosting_(machine_learning)

[6] H. Schwenk and Y. Bengio,
["Training methods for adaptive boosting of neural networks,"][adaboost_nn_paper]
NeurIPS 1997.

[adaboost_nn_paper]: https://papers.nips.cc/paper/1997/hash/9cb67ffb59554ab1dabb65bcb370ddd9-Abstract.html

[7] M. Moghimi, M. Saberian, J. Yang, L.-J. Li, N. Vasconcelos, and S. Belongie,
["Boosted convolutional neural networks,][boost_cnn_paper]
BMVC, 2016.

[boost_cnn_paper]: http://www.bmva.org/bmvc/2016/papers/paper024/index.html

[8] M. J. Saberian and N. Vasconcelos,
["Multiclass boosting: theory and algorithms,"][mcboost_paper]
NeurIPS, 2011.

[mcboost_paper]: https://papers.nips.cc/paper/2011/hash/2ac2406e835bd49c70469acae337d292-Abstract.html

[9] G. Huang, Y. Li, G. Pleiss, Z. Liu, J. E. Hopcroft, and K. Q. Weinberger
["Snapshot ensembles: train 1, get \\(M\\) for free,"][snapshot_ensembles_paper]
ICLR 2017.

[snapshot_ensembles_paper]: https://openreview.net/forum?id=BJYwwY9ll

[10] I. Loshchilov and F. Hutter,
["SGDR: stochastic gradient descent with restarts,"][sgdr_paper]
ICLR, 2017.

[sgdr_paper]: https://arxiv.org/abs/1608.03983

[11] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger,
["Densely connected convolutional networks,"][densenet_paper]
CVPR, pp. 4700-4708, 2017.

[densenet_paper]: https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html

[12] T. Garipov, P. Izmailov, D. Podoprikhin, D. P. Vetrov, and A. G. Wilson,
["Loss surfaces, mode connectivity, and fast ensembling of DNNs,"][fge_paper]
NeurIPS, 2018.

[fge_paper]: https://proceedings.neurips.cc/paper/2018/hash/be3087e74e9100d4bc4c6268cdbe8456-Abstract.html

[13] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov,
["Dropout: A Simple Way to Prevent Neural Networks from Overfitting,"][dropout_paper]
Journal of Machine Learning Research, Vol. 15, No. 56, pp. 1929-1958, 2014.

[dropout_paper]: https://jmlr.org/papers/v15/srivastava14a.html

[14] X. Glorot and Y. Bengio,
["Understanding the difficulty of training deep feedforward neural networks,"][xavier_init_paper]
AISTATS, Vol. 9, pp. 249-256, 2010.

[xavier_init_paper]: http://proceedings.mlr.press/v9/glorot10a.html

[15] J. Tompson, R. Goroshin, A. Jain, Y. LeCun, and C. Bregler,
["Efficient object localization using convolutional networks,"][spatial_dropout_paper]
CVPR, pp. 648-656, 2015.

[spatial_dropout_paper]: https://openaccess.thecvf.com/content_cvpr_2015/html/Tompson_Efficient_Object_Localization_2015_CVPR_paper.html

[16] G. Ghiasi, T.-Y. Lin, Q. V. Le
["DropBlock: A regularization method for convolutional networks,"][dropblock_paper]
NeurIPS, 2018.

[dropblock_paper]: https://papers.nips.cc/paper/2018/hash/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Abstract.html

[17] L. Wan, M. Zeiler, S. Zhang, Y. LeCun, and R. Fergus,
["Regularization of neural networks using DropConnect,"][dropconnect_paper]
ICML, Vol. 28, No. 3, pp. 1058-1066, 2013.

[dropconnect_paper]: http://proceedings.mlr.press/v28/wan13.html

[18] G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger,
["Deep networks with stochastic depth,"][stochastic_depth_paper]
ECCV, pp. 646-661, 2016.

[stochastic_depth_paper]: https://link.springer.com/chapter/10.1007/978-3-319-46493-0_39

[19] S. Singh, D. Hoiem, and D. Forsyth,
[Swapout: Learning an ensemble of deep architectures,"][swapout_paper]
NeurIPS, 2016.

[swapout_paper]: https://proceedings.neurips.cc/paper/2016/hash/c51ce410c124a10e0db5e4b97fc2af39-Abstract.html

[20] P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, and A. G. Wilson,
["Averaging weights leads to wider optima and better generalization,"][swa_paper]
UAI, 2018.

[swa_paper]: http://auai.org/uai2018/proceedings/papers/313.pdf

[21] H. Li, Z. Xu, G. Taylor, C. Studer, and T. Goldstein,
["Visualizing the loss landscape of neural nets,"][loss_landscape_paper]
NeurIPS, 2018.

[loss_landscape_paper]: https://papers.nips.cc/paper/2018/hash/a41b3bb3e6b050b6c9067c67f663b915-Abstract.html

[22] M. Zhang, J. Lucas, J. Ba, and G. E. Hinton,
["Lookahead Optimizer: \\(k\\) steps forward, 1 step back,"][lookahead_paper]
NeurIPS, 2019.

[lookahead_paper]: https://papers.nips.cc/paper/2019/hash/90fd4f88f588ae64038134f1eeaa023f-Abstract.html

[23] J. Martens,
["New Insights and Perspectives on the Natural Gradient Method,"][natural_gradient_insights_paper]
Journal of Machine Learning Research, Vol. 21, No. 146, pp. 1-76, 2020.

[natural_gradient_insights_paper]: https://jmlr.org/papers/v21/17-678.html

[24] D. P. Kingma and J. Ba,
["Adam: A Method for Stochastic Optimization,"][adam_paper]
ICLR, 2015.

[adam_paper]: https://arxiv.org/abs/1412.6980

[25] D. G. Anderson,
["Iterative procedures for nonlinear integral equations,"][anderson_mixing_paper]
Journal of the ACM, Vol. 12, No. 4, pp. 547-560, 1965.

[anderson_mixing_paper]: https://dl.acm.org/doi/abs/10.1145/321296.321305

[26] H.-r. Fang and Y. Saad,
["Two classes of multisecant methods for nonlinear acceleration,"][nonlinear_acceleration_paper]
Numerical Linear Algebra and Its Applications, Vol. 16, No. 3, pp. 197-221, 2009.
 
[nonlinear_acceleration_paper]: https://onlinelibrary.wiley.com/doi/abs/10.1002/nla.617

[27] L. Liu, H. Jiang, P. He, W. Chen, X. Liu, J. Gao, and J. Han
["On the variance of the adaptive learning rate and beyond,"][radam_paper]
ICLR, 2020.

[radam_paper]: https://openreview.net/forum?id=rkgz2aEKDr

[28] L. Wright:
["New deep learning optimizer, Ranger: synergistic combination of RAdam + LookAhead for the best of both,"][ranger_blog_post]
Medium blog post, 2019.

[ranger_blog_post]: https://lessw.medium.com/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d
