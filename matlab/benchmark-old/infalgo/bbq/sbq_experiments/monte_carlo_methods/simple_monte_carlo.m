function [log_mean_evidence, log_var_evidence, samples, logliks] = ...
    simple_monte_carlo(loglik_fn, prior, opt)
% Simple Monte Carlo
% 
% Inputs:
% - loglik_fn: a function that takes a single argument, a 1*d vector,
%             and returns the log of the likelihood
% - prior: requires fields
%          * mean
%          * covariance
% - opt: takes fields:
%        * num_samples: the number of samples to draw.
%
%
% Outputs:
%   mean_log_evidence: the mean of our poterior over the log of the evidence.
%   var_log_evidence: the variance of our posterior over the log of the
%                     evidence.
% - samples : n*d matrix of samples
%
%
%
% David Duvenaud
% January 2012

if nargin < 3
    opt.num_samples = 1000;
end

% Draw samples.
samples = mvnrnd( prior.mean, prior.covariance, opt.num_samples );
logliks = nan(opt.num_samples, 1);
parfor i = 1:opt.num_samples
    logliks(i) = loglik_fn( samples(i, :) );
end

% Remove any bad likelihoods
good_ix = ~isinf(logliks);
num_good = sum(good_ix);


% Compute empirical mean.
log_mean_evidence = logsumexp(logliks(good_ix)) - log(num_good);

% Compute standard error in a numerically stable way.
log_second_moment = logsumexp(2.*(logliks(good_ix))) - log(num_good);
log_first_moment_sq = 2*log_mean_evidence;
b = min([ log_second_moment log_first_moment_sq]);
log_var_evidence = log( exp(log_second_moment - b) -exp(log_first_moment_sq - b) ) + b - log(num_good);
end
