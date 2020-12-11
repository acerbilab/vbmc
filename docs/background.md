# Background to understand Variational Bayesian Monte Carlo (work in progress)

Variational Bayesian Monte Carlo (VBMC from now on) is a fairly complex algorithm that combines several machine learning ideas.
If you are not fully familiar with these key concepts, understanding how VBMC works can be quite challenging. 
The goal of this document is to provide useful material and links to references to build the necessary background knowledge to understand the principles behind VBMC.

## Bayesian inference

The goal of VBMC is to perform Bayesian inference, that is to compute:
- the *posterior distribution* p(θ|*D*) for a given model, model parameters θ and dataset *D*;
- the *marginal likelihood* (also known as *model evidence*) p(*D*) = ∫p(*D*|θ)p(θ)dθ.

If you are not familiar with Bayesian inference, you might want to start with the first chapter(s) of the *Bayesian Data Analysis* (BDA) book, available for free [here](https://users.aalto.fi/~ave/BDA3.pdf). 

A great introduction to the marginal likelihood, and its usage as a principled metric for model selection that automatically corrects for model complexity ("Bayesian Occam's razor"), can be found in Chapter 28 of MacKay's book, available [here](http://www.inference.org.uk/itprnn/book.pdf).

**References:**
- Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin, DB (2013). Bayesian data analysis (Third edition). CRC press.
- MacKay DJ, Mac Kay DJ (2003). Information theory, inference and learning algorithms. Cambridge university press.

## Gaussian processes

VBMC approximates the log joint distribution *f*(θ) = log p(*D*|θ)p(θ) with a *Gaussian process*. 

Gaussian processes (GPs) are flexible distributions over functions with many nice mathematical properties — for example, we can often peform calculations involving GPs analytically. In VBMC, we perform GP regression — that is, we observe a few evaluations of *f*(θ) at some points, and infer the posterior GP compatible with those observations. The GP model built this way is also known as a *surrogate* model of *f*, which we can use in place of the original (unknown) *f*. 
Crucially, the GP is a probabilistic model that gives us a posterior mean and posterior variance prediction at each point.

To familiarize yourself with Gaussian processes, you could have a look at this Distill article, [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/).

