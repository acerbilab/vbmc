function [log_mean_evidence, log_var_evidence, sample_locs, diagnostics, sample_vals] = ...
    bmc(loglik_fn, prior, opt)
% Naive Bayesian Monte Carlo.  Chooses samples based on AIS.
%
% Based on: http://mlg.eng.cam.ac.uk/zoubin/papers/RasGha03.pdf
%
% Inputs:
%   loglik_fn: a function that takes a single argument, a 1*d vector, and
%              returns the log of the likelihood.
%   prior: requires fields
%          * mean
%          * covariance
%   opt: takes fields:
%        * num_samples: the number of samples to draw.
%        * proposal_covariance: for the AIS.
% 
% Outputs:
%   mean_log_evidence: the mean of our poterior over the log of the evidence.
%   var_log_evidence: the variance of our posterior over the log of the
%                     evidence.
%   samples: n*d matrix of the locations of the samples.
%   weights: n*1 list of weights.
%
%
% David Duvenaud
% January 2012


% Define default options.
if nargin < 3
    opt.num_samples = 100;
end

% Get sample locations from a run of AIS.
[ais_mean_log_evidence, ais_var_log_evidence, ais_sample_locs, ais_sample_vals, stats] = ...
    ais_mh(loglik_fn, prior, opt);

%[sample_locs, sample_vals] = ...
%    remove_duplicate_samples(sample_locs, sample_vals);

sample_locs = stats.all_samples.locations;
sample_vals = stats.all_samples.logliks;

opt.num_samples = length(sample_vals);

% Now call BMC using the exp of those samples.
[mean_evidence, var_evidence, hypers] = ...
    bmc_integrate(sample_locs, exp(sample_vals - max(sample_vals)), prior);

log_mean_evidence = log(mean_evidence) + max(sample_vals);
log_var_evidence = log(var_evidence) + 2*max(sample_vals);

diagnostics.hypers = hypers;

if var_evidence < 0
    warning('variance of evidence negative');
    fprintf('variance of evidence: %g\n', var_evidence);
    log_var_evidence = log(eps);
end
end
