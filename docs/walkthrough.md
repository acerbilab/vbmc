# A gentle introduction to Variational Bayesian Monte Carlo (work in progress)

Variational Bayesian Monte Carlo (VBMC, from now on) is a fairly complex algorithm that combines several machine learning ideas.
The goal of this walkthrough is to build the necessary background knowledge to understand the principles behind VBMC and how the algorithm works.

#### Table of contents

1. [Bayesian inference](#1-bayesian-inference)    
1.1 [Sample-efficient approximate Bayesian inference](#11-sample-efficient-approximate-bayesian-inference)
2. [Gaussian processes](#2-gaussian-processes)    
2.1 [Details of GPs in VBMC](#21-details-of-gps-in-vbmc)
3. [Active sampling and Bayesian optimization](#3-active-sampling-and-bayesian-optimization)

## 1. Bayesian inference

The goal of VBMC is to perform Bayesian inference for a given model and dataset, that is to compute:
- the *posterior distribution* p(θ|*D*) for a given model, model parameters θ and dataset *D*;
- the *marginal likelihood* (also known as *model evidence*) p(*D*) = ∫p(*D*|θ)p(θ)dθ.

The posterior distribution encodes our uncertainty over model parameters, while the marginal likelihood is a principled metric for model selection that automatically corrects for model complexity ("Bayesian Occam's razor").

Related probabilistic objects and concepts that are referred often are: the *prior* p(θ), the *likelihood* p(*D*|θ), and the *joint* density p(*D*,θ) = p(*D*|θ)p(θ) (or their logarithms). Also note that in the paper and in the documentation, the terms probability density and probability distribution are used interchangeably (somewhat improperly).

### 1.1 Sample-efficient approximate Bayesian inference

VBMC performs *approximate* Bayesian inference, in the sense that it computes an approximation of the posterior q(θ) ≈ p(θ|*D*), and an approximation of the marginal likelihood, as we will explain later. VBMC has two layers of approximation: first, it approximates the log joint with a [Gaussian process](#gaussian-processes) surrogate model. Second, it fits a [variational posterior](#variational-inference) to the Gaussian process surrogate. Both these steps are explained in the sections below.

The key feature of VBMC is that it is *sample-efficient*, i.e. it works with only a relatively small number of evaluations of the log-joint distribution. This is in contrast to most other approaches to approximate inference, which can require a very large of number of evaluations (easily 10-100x more than VBMC). Thus, VBMC shines when a solution is needed with only a relatively small number of evaluations (e.g., when the model is somewhat expensive to evaluate — although it does not necessarily have to be *very* expensive). Even when the researcher could afford more time-consuming methods, VBMC can still be a useful tool of the [Bayesian workflow](https://arxiv.org/abs/2011.01808) to quickly get solutions when prototyping new models.

Another useful property of VBMC is that it can easily deal with *noisy* evaluations of the log-joint. Some other approximate inference methods can deal with noisy evaluations as well, but it can get tricky and very expensive in terms of evaluations.

#### Resources:

- Book: If you are not familiar with Bayesian inference, you might want to start with the first chapter(s) of the *Bayesian Data Analysis* (BDA) book, available for free [here](https://users.aalto.fi/~ave/BDA3.pdf).
- Book: A crash-course in Bayesian inference consists of Chapters 2 and 7 of *Probabilistic Machine Learning: An Introduction*, available for free [here](https://github.com/probml/pml-book/releases/latest/download/pml1.pdf). (To fully understand Chapter 7 you may have to read parts of previous Chapters.)
- Book: A great discussion of the marginal likelihood (and "Bayesian Occam's razor") can be found in Chapter 28 of MacKay's book, available [here](http://www.inference.org.uk/itprnn/book.pdf).
- Video: An introduction to Bayesian inference in machine learning is given in [this lecture](https://www.youtube.com/watch?v=mgBrXnjF8R4) by Zoubin Ghahramani (the first 30 minutes or so).


#### References:
- Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin, DB (2013). Bayesian data analysis (Third edition). CRC press ([PDF](https://users.aalto.fi/~ave/BDA3.pdf)).
- MacKay DJ (2003). Information theory, inference and learning algorithms. Cambridge university press ([PDF](http://www.inference.org.uk/itprnn/book.pdf)).
- Murphy, KP (2021). Probabilistic Machine Learning: An introduction. MIT Press ([webpage](https://probml.github.io/pml-book/book1.html), [PDF](https://github.com/probml/pml-book/releases/latest/download/pml1.pdf)).

## 2. Gaussian processes

VBMC first approximates the log joint distribution *f*(θ) = log p(*D*|θ)p(θ) with a *Gaussian process*. Let's see what that means.

Gaussian processes (GPs) are flexible distributions over functions with many nice mathematical properties — for example, we can often perform calculations involving GPs analytically. In VBMC, we perform GP regression — that is, we observe a few evaluations of *f*(θ) at some points, and infer the posterior GP (a posterior distribution over functions) compatible with those observations. The GP model built this way is also known as a *surrogate* model of *f*, which we can use in place of the original (unknown) *f*. 
Crucially, the GP is a probabilistic model that gives us a posterior mean and posterior variance prediction for the function at each point.

One crucial aspect of the GP model is the *kernel* or covariance function *k*(θ,θ') defined between two points of the input space. 
In loose terms, a kernel or covariance function *k*(θ,θ') specifies the statistical relationship between two points θ, θ' in the input space; that is, how markedly a change in the value of the GP at θ correlates with a change in the GP at θ'. In some sense, you can think of *k*(⋅,⋅) as defining a similarity between inputs. 

To familiarize yourself with GPs and covariance functions, have a look at this Distill article, [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/), and at the [kernel cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/).
Plenty of more material about GPs can be found at the [GP model zoo](https://jejjohnson.github.io/gp_model_zoo/intro/).

The GP bible is the Gaussian Processes for Machine Learning book, available [online](http://www.gaussianprocess.org/gpml/chapters/RW.pdf). For VBMC, the most relevant parts are Chapter 2 (all), Chapter 4 (sections 4.1 and 4.2) and Chapter 5 (sections 5.1, 5.2, 5.4.1).

### 2.1 Details of GPs in VBMC

VBMC uses the standard *squared exponential* (or rescaled Gaussian) kernel, and a standard Gaussian likelihood (observation noise for the function values). The observation noise takes a small value when the log-joint is deterministic, or otherwise is specified by the user when VBMC is applied to a noisy/stochatic function.
VBMC also uses a *negative quadratic* mean function, which is akin to a prior assumption for the posterior to be Gaussian (note that this can be overridden by data — the posterior in VBMC is *not* restricted to be Gaussian). We tried a few other mean functions, with little success ([Acerbi, 2019](http://proceedings.mlr.press/v96/acerbi19a.html)).

In VBMC, the GP hyperparameters (e.g., GP kernel length scales, location and scale of the GP mean function, etc.) are approximately marginalized by sampling from the hyperparameter posterior via Markov Chain Monte Carlo (see below).
Since hyperparameter marginalization becomes expensive with larger GPs, in later iterations VBMC switches to *maximizing* the GP marginal likelihood, which yields a single point estimate for the GP hyperparameters (see Section 5.4.1 of Rasmussen & Williams).

#### References:
- Rasmussen CE, Williams CK (2006). Gaussian processes for machine learning. MIT press ([PDF](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)).
- Acerbi L (2019). An Exploration of Acquisition and Mean Functions in Variational Bayesian Monte Carlo. PMLR ([link](http://proceedings.mlr.press/v96/acerbi19a.html)).

## 3. Active sampling and Bayesian optimization



## 4. Bayesian quadrature

## 5. Variational inference

## 6. Markov Chain Monte Carlo (extra)
