# A gentle introduction to Variational Bayesian Monte Carlo (work in progress)

Variational Bayesian Monte Carlo (VBMC, from now on) is a fairly complex algorithm that combines several machine learning ideas.
If you are not fully familiar with these key concepts, understanding how VBMC works can be challenging. 
The goal of this walkthrough is to provide useful material and references to build the necessary background knowledge to understand the principles behind VBMC.

## Bayesian inference

The goal of VBMC is to perform Bayesian inference, that is to compute:
- the *posterior distribution* p(θ|*D*) for a given model, model parameters θ and dataset *D*;
- the *marginal likelihood* (also known as *model evidence*) p(*D*) = ∫p(*D*|θ)p(θ)dθ.

Other probabilistic objects that are referred often are: the *prior* p(θ), the *likelihood* p(*D*|θ), and the *joint* density p(*D*,θ) = p(*D*|θ)p(θ) (or their logarithms). Also note that in the paper and in the documentation, the terms probability density and probability distribution are used interchangeably (somewhat improperly).

If you are not familiar with Bayesian inference, you might want to start with the first chapter(s) of the *Bayesian Data Analysis* (BDA) book, available for free [here](https://users.aalto.fi/~ave/BDA3.pdf). 

A great introduction to the marginal likelihood, and its usage as a principled metric for model selection that automatically corrects for model complexity ("Bayesian Occam's razor"), can be found in Chapter 28 of MacKay's book, available [here](http://www.inference.org.uk/itprnn/book.pdf).

### Bayesian inference in VBMC

VBMC performs *approximate* Bayesian inference, in the sense that it computes an approximation of the posterior q(θ) ≈ p(θ|*D*), and an approximation of the marginal likelihood. VBMC has two layers of approximation: first, it approximates the log joint with a [Gaussian process](#gaussian-processes) surrogate model. Second, it fits a [variational posterior](#variational-inference) to the Gaussian process surrogate. Both these steps are explained below.

The key feature of VBMC is that it is sample-efficient, i.e. it tries to perform only a small number of evaluations of the log-joint distribution (as opposed to most other approaches to approximate inference). Another useful property is that it can deal with *noisy* evaluations of the log-joint.

#### References:
- Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin, DB (2013). Bayesian data analysis (Third edition). CRC press ([PDF](https://users.aalto.fi/~ave/BDA3.pdf)).
- MacKay DJ (2003). Information theory, inference and learning algorithms. Cambridge university press ([PDF](http://www.inference.org.uk/itprnn/book.pdf)).

## Gaussian processes

VBMC first approximates the log joint distribution *f*(θ) = log p(*D*|θ)p(θ) with a *Gaussian process*. 

Gaussian processes (GPs) are flexible distributions over functions with many nice mathematical properties — for example, we can often perform calculations involving GPs analytically. In VBMC, we perform GP regression — that is, we observe a few evaluations of *f*(θ) at some points, and infer the posterior GP compatible with those observations. The GP model built this way is also known as a *surrogate* model of *f*, which we can use in place of the original (unknown) *f*. 
Crucially, the GP is a probabilistic model that gives us a posterior mean and posterior variance prediction at each point.

One crucial aspect of the GP model is the *kernel* or covariance function *k*(θ,θ') defined between two points of the input space. 
In loose terms, a kernel or covariance function *k*(θ,θ') specifies the statistical relationship between two points θ, θ' in the input space; that is, how markedly a change in the value of the GP at θ correlates with a change in the GP at θ'. In some sense, you can think of *k*(⋅,⋅) as defining a similarity between inputs. 

To familiarize yourself with GPs and covariance functions, have a look at this Distill article, [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/), and at the [kernel cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/).
Plenty of more material about GPs can be found at the [GP model zoo](https://jejjohnson.github.io/gp_model_zoo/intro/).

The GP bible is the Gaussian Processes for Machine Learning book, available [online](http://www.gaussianprocess.org/gpml/chapters/RW.pdf). For VBMC, the most relevant parts are Chapter 2 (all), Chapter 4 (sections 4.1 and 4.2) and Chapter 5 (sections 5.1, 5.2, 5.4.1).

### Gaussian processes in VBMC


VBMC uses the standard *squared exponential* (or rescaled Gaussian) kernel.


#### References:
- Rasmussen CE, Williams CK (2006). Gaussian processes for machine learning. MIT press ([PDF](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)).

### Training the Gaussian Process

## Active sampling and Bayesian optimization

## Bayesian quadrature

## Variational inference

## Markov Chain Monte Carlo (extra)
