function [lz, dlz] = gp_fixedlik(hyp, inf, mean, cov, lik, x, y)
% A wrapper for gp that sets some derivatives to zero,
% so that they won't be changed during hyperparameter learning.
%
% David Duvenaud
% November 2011

[lz, dlz] = penalized_gp(hyp, inf, mean, cov, lik, x, y);

dlz.lik = 0;  % Don't update the noise variance.
%dlz.cov(end) = 0;  % Don't update the signal variance, it cancels out anwyays.