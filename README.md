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

The function `vbmc_rnd` draws random samples from the obtained variational posterior, whereas `vbmc_pdf` evaluates the variational posterior density.

For more usage examples, see [**vbmc_examples.m**](https://github.com/lacerbi/bads/blob/master/vbmc_examples.m). You can also type `help vbmc` to display the documentation.

For practical recommendations, such as how to set `LB` and `UB`, and any other question, we will prepare a FAQ soon. For now, you can find useful information on the [BADS wiki](https://github.com/lacerbi/bads/wiki).

## Reference

1. Acerbi, L. (2018). Variational Bayesian Monte Carlo. To appear in *Advances in Neural Information Processing Systems 31*. [arXiv preprint](https://arxiv.org/abs/XXXX.YYYY) arXiv:XXXX.YYYY
