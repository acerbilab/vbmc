function [log_mean_evidence, log_var_evidence, sample_locs, sample_vals] = ...
    log_bmc(loglik_fn, prior, opt)
% Log-Naive Bayesian Monte Carlo.  Chooses samples based on AIS.
%
% This variant on BMC also learns a GP modef of the log-likelihood surface,
% then learns a GP modeling the difference between the mean predictions
% of the GP on the likelihood versus the exp of the mean predictions of the
% GP on the log-likelihood.  The final estimate is the integral under the
% GP on the likeihood surface plus the integral under the 'correction' GP.
%
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
% February 2012


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

normalized_likelihoods = exp(sample_vals - max(sample_vals));
normalized_log_likelihoods = sample_vals - max(sample_vals);


init_hypers.mean = [];
init_hypers.lik = log(0.01);  % Values go between 0 and 1, so no need to scale.
init_lengthscales = mean(sqrt(diag(prior.covariance)))/2;
init_output_variance = .1;
init_hypers.cov = log( [init_lengthscales init_output_variance] );

covfunc = @covSEiso;

% Now call BMC using the exp of those samples.
[mean_evidence, var_evidence, like_hypers] = ...
    bmc_integrate(sample_locs, normalized_likelihoods, prior, covfunc, init_hypers);

% Learn a model over the log-evidences.
% Todo: separate the GP-learning part.
init_hypers.lik = log(std(normalized_log_likelihoods)/100);
[ignore1, ignore2, loglike_hypers] = ...
    bmc_integrate(sample_locs, normalized_log_likelihoods, prior, covfunc, init_hypers);

% Compute the difference at a set of pseudo-points, and include that the
% difference is zero at the evaluated points.
mahal_scales = exp(like_hypers.cov(1:end-1));

% candidate locations will be constrained to a box defined by the prior
%lower_bound = prior.mean - 3*sqrt(diag(prior.covariance))';
%upper_bound = prior.mean + 3*sqrt(diag(prior.covariance))';

% Find the candidate locations, far removed from existing samples, with the
% use of a Voronoi diagram
%candidate_locs = find_farthest(sample_locs, ...
%                    [lower_bound; upper_bound], max(200,length(normalized_likelihoods)), ...
%                     mahal_scales);

candidate_locs = find_candidates(sample_locs, ...
                     max(200,length(normalized_likelihoods)), mahal_scales);

%loglike_hypers.cov(1) = 0;
%loglike_hypers.cov(2) = 0;

% Combine existing and candidate points.
delta_locs = [sample_locs; candidate_locs];

[mean_gp_loglik] = ...
    gp_mean2( loglike_hypers, delta_locs, sample_locs, normalized_log_likelihoods);

%plot(candidate_locs, mean_gp_loglik, 'b.'); hold on;
%plot(sample_locs, normalized_log_likelihoods, 'g.'); hold on;

mean_gp_lik = gp_mean2( like_hypers, delta_locs, sample_locs, normalized_likelihoods);
delta_vals = exp(mean_gp_loglik) - mean_gp_lik;  % Add variance as well?

% Compute the expected correction factor.
init_hypers.lik = log(std(delta_vals)/100);
[mean_correction, var_correction, delta_hypers] = ...
    bmc_integrate(delta_locs, delta_vals, prior, covfunc, init_hypers);

delta_mean = gp_mean2( delta_hypers, delta_locs, delta_locs, delta_vals);


if 0;%numel(prior.mean) == 1
    plot(delta_locs, delta_vals, 'b.'); hold on;
    plot(delta_locs, mean_gp_lik, 'k.');
    plot(delta_locs, exp(mean_gp_loglik), 'g.');
    plot(delta_locs, delta_mean + mean_gp_lik, 'r.');
end

% Combine the correction factor with the original estimate, assuming that
% the covariance between the two is zero.
mean_evidence = mean_evidence + mean_correction;
var_evidence = var_evidence + var_correction;

log_mean_evidence = log(mean_evidence) + max(sample_vals);
log_var_evidence = log(var_evidence) + 2*max(sample_vals);

if var_evidence < 0
    warning('variance of evidence negative');
    fprintf('variance of evidence: %g\n', var_evidence);
    log_var_evidence = log(eps);
end
end


function [mu, var] = gp_mean2(hypers, xstar, X, y)
    inference = @infExact;
    likfunc = @likGauss;
    meanfunc = {'meanZero'};
    covfunc = @covSEiso;
    
    [mu, var] = gp( hypers, inference, meanfunc, covfunc, likfunc, X, y, xstar );
end

