# Variational Bayesian Monte Carlo (VBMC) - v0.9 (beta)

**News:** 
- The VBMC paper [[1](#reference)] has been accepted for a poster presentation at [NIPS 2018](https://nips.cc/Conferences/2018/Schedule?showEvent=11786)! (20.8% acceptance rate this year, for a total of 4856 submissions)

## What is it

> *What if there was a model-fitting method similar to Bayesian optimization (e.g., [BADS](https://github.com/lacerbi/bads)), which, instead of returning just the optimal parameter vector, would also return its uncertainty (even better, the full posterior distribution of the parameters), and maybe even a metric than can be used for Bayesian model comparison?*

VBMC is a novel approximate inference method designed to fit and evaluate computational models with a limited budget of likelihood evaluations (e.g., for computationally expensive models). Specifically, VBMC simultaneously computes:
- an approximate posterior distribution of the model parameters; 
- an approximation — technically, an approximate lower bound — of the log model evidence (also known as log marginal likelihood or log Bayes factor), a metric used for [Bayesian model selection](https://en.wikipedia.org/wiki/Bayes_factor).

Our first benchmark using an array of meaningful artificial test problems and a real neuronal model shows that, with the exception of the easiest cases, VBMC vastly outperforms existing inference methods for expensive models [[1](#reference)]. We are currently validating the algorithm on several other real model-fitting problems, as we previously did for our model-fitting algorithm based on Bayesian optimization, [BADS](https://github.com/lacerbi/bads).

VBMC runs with virtually no tuning and it is very easy to set up for your problem (especially if you are already familiar with `bads`).

### Should I use VBMC?

VBMC is effective when:

- the model log-likelihood function is a black-box (e.g., the gradient is unavailable);
- the likelihood is at least moderately expensive to compute (say, half a second or more per evaluation);
- the model has up to `D = 10` parameters (maybe a few more, but no more than `D = 20`).

Conversely, if your model can be written analytically, you should exploit the powerful machinery of probablistic programming frameworks such as [Stan](http://mc-stan.org/) or [PyMC3](https://docs.pymc.io/).

## Installation

[**Download the latest version of VBMC as a ZIP file**](https://github.com/lacerbi/vbmc/archive/master.zip).
- To install VBMC, clone or unpack the zipped repository where you want it and run the script `install.m`.
   - This will add the VBMC base folder to the MATLAB search path.
- To see if everything works, run `vbmc('test')`.

## Quick start

The VBMC interface is similar to that of MATLAB optimizers. The basic usage is:

```matlab
[VP,ELBO,ELBO_SD] = vbmc(FUN,X0,LB,UB,PLB,PUB);
```
with input parameters:
- `FUN`, a function handle to the log posterior distribution of your model (that is, log prior plus log likelihood of a dataset and model, for a given input parameter vector);
- `X0`, the starting point of the inference (a row vector);
- `LB` and `UB`, hard lower and upper bounds for the parameters;
- `PLB` and `PUB`, *plausible* lower and upper bounds, that is a box where you would expect to find most of the posterior probability mass.

The output parameters are:
- `VP`, a struct with the variational posterior approximating the true posterior;
- `ELBO`, the (estimated) lower bound on the log model evidence;
- `ELBO_SD`, the standard deviation of the estimate of the `ELBO` (*not* the error between the `ELBO` and the true log model evidence, which is generally unknown).

The variational posterior `vp` can be manipulated with functions such as `vbmc_moments` (compute posterior mean and covariance), `vbmc_pdf` (evaluates the posterior density), `vbmc_rnd` (draw random samples), `vbmc_kldiv` (Kullback-Leibler divergence between two posteriors).

For a tutorial with many extensive usage examples, see [**vbmc_examples.m**](https://github.com/lacerbi/vbmc/blob/master/vbmc_examples.m). You can also type `help vbmc` to display the documentation.

For practical recommendations, such as how to set `LB` and `UB`, and any other question, check out the FAQ on the [VBMC wiki](https://github.com/lacerbi/vbmc/wiki).


## How does it work

BADS follows a [mesh adaptive direct search](http://epubs.siam.org/doi/abs/10.1137/040603371) (MADS) procedure for function minimization that alternates **poll** steps and **search** steps (see **Fig 1**). 

- In the **poll** stage, points are evaluated on a mesh by taking steps in one direction at a time, until an improvement is found or all directions have been tried. The step size is doubled in case of success, halved otherwise. 
- In the **search** stage, a [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) (GP) is fit to a (local) subset of the points evaluated so far. Then, we iteratively choose points to evaluate according to a *lower confidence bound* strategy that trades off between exploration of uncertain regions (high GP uncertainty) and exploitation of promising solutions (low GP mean).

**Fig 1: BADS procedure** ![BADS procedure](https://github.com/lacerbi/bads/blob/master/docs/bads-cartoon.png "Fig 1: BADS procedure")

See [here](https://github.com/lacerbi/optimviz) for a visualization of several optimizers at work, including BADS.

See our paper for more details [[1](#reference)].

## Troubleshooting

The VBMC toolbox is under active development and currently in its beta version (close to final).

It is still in beta since we are validating the algorithm on an additional batch of model-fitting problems, and we want to include in the toolbox some semi-automated diagnostics tools for robustness. The toolbox interface (that is, details of input and output arguments of some functions) may change from the beta to the final version.
As of now, the toolbox is usable, but you should double-check your results (as you would do in any case, of course).

If you have trouble doing something with VBMC, spot bugs or strange behavior, or you simply have some questions, please contact me at <luigi.acerbi@unige.ch>, putting 'VBMC' in the subject of the email.

## VBMC for other programming languages

VBMC is currently available only for MATLAB. A Python version is being planned.

If you are interested in porting VBMC to Python or another language (R, [Julia](https://julialang.org/)), please get in touch at <luigi.acerbi@unige.ch> (putting  'VBMC' in the subject of the email); I'd be willing to help.
However, before contacting me for this reason, please have a good look at the codebase here on GitHub, and at the paper [[1](#reference)]. VBMC is a fairly complex piece of software, so be aware that porting it will require considerable effort and programming/computing skills.

## Reference

1. Acerbi, L. (2018). Variational Bayesian Monte Carlo. To appear in *Advances in Neural Information Processing Systems 31*. [arXiv preprint](https://arxiv.org/abs/XXXX.YYYY) arXiv:XXXX.YYYY

You can cite VBMC in your work with something along the lines of

> We estimated approximate posterior distibutions and approximate lower bounds to the model evidence of our models using Variational Bayesian Monte Carlo (VBMC; Acerbi, 2018). VBMC combines variational inference and Bayesian quadrature to perform approximate Bayesian inference in a sample-efficient manner.

Besides formal citations, you can demonstrate your appreciation for VBMC in the following ways:

- *Star* the VBMC repository on GitHub;
- [Follow me on Twitter](https://twitter.com/AcerbiLuigi) for updates about VBMC and other projects I am involved with;
- Tell me about your model-fitting problem and your experience with VBMC (positive or negative) at <luigi.acerbi@unige.ch> (putting  'VBMC' in the subject of the email).

You may also want to check out [Bayesian Adaptive Direct Search](https://github.com/lacerbi/bads) (BADS), our method for fast Bayesian optimization.

### License

VBMC is released under the terms of the [GNU General Public License v3.0](https://github.com/lacerbi/vbmc/blob/master/LICENSE.txt).
