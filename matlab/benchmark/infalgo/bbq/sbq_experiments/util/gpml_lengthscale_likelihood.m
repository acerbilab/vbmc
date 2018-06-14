function l = gpml_lengthscale_likelihood( log_in_scale, gp_hypers, inference, meanfunc, covfunc, likfunc, X, y)
% Just a wrapper function to evaluate the likelihood of a certain set of
% lengthscales.
%
% David Duvenaud
% February 2012

gp_hypers.cov(1:end-1) = log_in_scale;
l = exp(-gp_fixedlik(gp_hypers, inference, meanfunc, covfunc, likfunc, X, y));

