% example of sbq use to provide the mean and variance for the integral 
% evidence = Z = int l(x) p(x)
% where l(x) is the likelihood function and p(x) is the prior, specified
% below

% dimensionality (note input x is a ROW vector)
N = 2;

% set our prior
prior.mean = zeros(1, N);
% the prior must have diagonal covariance for the current codebase
prior.covariance = eye(N);

% make up a log-likelihood function
likelihood.mean = rand(1, N)-0.5;
likelihood.covariance = rand(N)-0.5;
likelihood.covariance = likelihood.covariance' * likelihood.covariance;
log_likelihood_fn = @(x) logmvnpdf(x, likelihood.mean, likelihood.covariance);

% find true answer
log_Z = log_volume_between_two_gaussians(prior.mean, ...
                                     prior.covariance, ...
                                     likelihood.mean, likelihood.covariance);

% set sbq options
opt.num_samples = 100;

% run sbq
[log_mean_Z, log_var_Z, samples] = ...
    sbq(log_likelihood_fn, prior, opt);

fprintf(' true Z:\t%g\n mean Z:\t%g\n var Z: \t%g\n', exp(log_Z), exp(log_mean_Z(end)), exp(log_var_Z(end)))