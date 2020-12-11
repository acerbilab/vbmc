# Background to understand Variational Bayesian Monte Carlo (work in progress)

Variational Bayesian Monte Carlo (VBMC from now on) is a fairly complex algorithm that combines several machine learning ideas.
If you are not fully familiar with these key concepts, understanding how VBMC works can be quite challenging. 
The goal of this document is to provide useful material to build the necessariy background knowledge to understand the principles behind VBMC.

## Bayesian inference

The goal of VBMC is to perform Bayesian inference, that is to compute:
- the *posterior distribution* p(θ|*D*) for a given model, model parameters θ and dataset *D*;
- the *marginal likelihood* (also known as *model evidence*) p(*D*) = ∫p(D|θ)p(θ)dθ.

If you are not familiar with Bayesian inference, you might want to start with the first chapter(s) of the *Bayesian Data Analysis* (BDA) book, available for free [here](https://users.aalto.fi/~ave/BDA3.pdf).

> Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian data analysis (Third edition). CRC press.
