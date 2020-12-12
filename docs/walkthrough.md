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

**References:**
- Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin, DB (2013). Bayesian data analysis (Third edition). CRC press.
- MacKay DJ (2003). Information theory, inference and learning algorithms. Cambridge university press.

## Gaussian processes

VBMC first approximates the log joint distribution *f*(θ) = log p(*D*|θ)p(θ) with a *Gaussian process*. 

Gaussian processes (GPs) are flexible distributions over functions with many nice mathematical properties — for example, we can often perform calculations involving GPs analytically. In VBMC, we perform GP regression — that is, we observe a few evaluations of *f*(θ) at some points, and infer the posterior GP compatible with those observations. The GP model built this way is also known as a *surrogate* model of *f*, which we can use in place of the original (unknown) *f*. 
Crucially, the GP is a probabilistic model that gives us a posterior mean and posterior variance prediction at each point.

One crucial aspect of the GP model is the *kernel* or covariance function K(θ,θ') defined between two points of the input space. See [here](https://stats.stackexchange.com/questions/228552/covariance-functions-or-kernels-what-exactly-are-they/).

To familiarize yourself with Gaussian processes, have a look at this Distill article, [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/).

### Training the Gaussian Process

## Active sampling and Bayesian optimization

## Bayesian quadrature

## Variational inference

## Markov Chain Monte Carlo (extra)
