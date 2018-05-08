function [mean_log_evidence, var_log_evidence, sample_locs, logliks, times, stats] = ...
    ais_mh(loglik_fn, prior, opt)
% Annealed Importance Sampling w.r.t. a Gaussian prior
% using a Metropolis-Hastings sampler with a Gaussian proposal distribution.
%
% Runs Mentropolis-Hastings over tempered versions of the posterior
% so as to best estimate the evidence, an integral over input space
% of p(x)exp(log_r_fn(x)).
%
% More info at:
% http://www.cs.toronto.edu/~radford/ftp/ais-rev.pdf
%
% Inputs:
%   loglik_fn: a function that takes a single argument, a 1*d vector, and
%              returns the log of the likelihood.
%   prior: requires fields
%          * mean
%          * covariance
%   opt: takes fields:
%        * num_samples: the number of samples to draw.
%        * proposal_covariance
%        * learn_proposal: if true, calls ais_mh to tune the acceptance ratio
%                          to approximately
% 
% Outputs:
%   mean_log_evidence: the mean of our poterior over the log of the evidence.
%   var_log_evidence: the variance of our posterior over the log of the
%                     evidence.
%   samples
%          * locations: n*d matrix of the locations of the samples.
%          * logliks
%
%   sample_vals:
%   stats: a struct of stats about the run, containing:
%          * weights: n*1 list of weights.
%          * acceptance ratio.
%          * all_samples
%                locations: n*d matrix of the locations of the samples.
%                logliks
%
%
% David Duvenaud
% January 2012


% Define default options.
if nargin < 3
    opt.num_samples = 1000;
end

% Todo: set this adaptively with a burn-in?
opt.proposal_covariance = prior.covariance;

% Define annealing schedule.  This can be anything, as long as it starts
% from zero and doesn't go above one.
temps = linspace( 0, 1, opt.num_samples + 1);
log_prior_fn = @(x) logmvnpdf(x, prior.mean, prior.covariance);

% Allocate memory.
weights = nan(opt.num_samples, 1);
times = zeros(opt.num_samples, 1);
sample_locs = nan(opt.num_samples - 1, numel(prior.mean));
logliks = nan(opt.num_samples - 1, 1);
stats.all_samples.locations = nan(opt.num_samples - 1, numel(prior.mean));
stats.all_samples.logliks = nan(opt.num_samples - 1, 1);

% Start with a sample from the prior.
tmpT = cputime;
cur_pt = mvnrnd( prior.mean, prior.covariance );
cur_ll = loglik_fn(cur_pt);
times(1) = tmpT - cputime;

stats.all_samples.locations(1, :) = cur_pt;
stats.all_samples.logliks(1) = cur_ll;

num_accepts = 1;
for t = 2:length(temps)
    tmpT = cputime; 
    % Compute MH proposal.
    proposal = mvnrnd( cur_pt, opt.proposal_covariance );
    proposal_ll = loglik_fn(proposal);
    
    % Possibly take a MH step.
    annealed_proposal_ll = temps(t)*proposal_ll + log_prior_fn(proposal);
    annealed_cur_ll = temps(t)*cur_ll + log_prior_fn(cur_pt);
    ratio = exp(annealed_proposal_ll - annealed_cur_ll);
    if ratio > rand
        num_accepts = num_accepts + 1;
        cur_pt = proposal;
        cur_ll = proposal_ll;
    end

    % Compute weights.
    weights(t - 1) = cur_ll*(temps(t) - temps(t - 1));

    % Record locations.
    sample_locs(t - 1, :) = cur_pt;
    logliks(t - 1) = cur_ll;
    stats.all_samples.locations(t, :) = proposal;
    stats.all_samples.logliks(t) = proposal_ll;
    times(t) = cputime - tmpT;
end

mean_log_evidence = cumsum(weights);

times = times(2:end);

stats.acceptance_ratio = num_accepts / length(temps);
fprintf('\nAcceptance ratio: %f \n', stats.acceptance_ratio);

% Try to estimate variance, in a not so great way.
rho = auto_correlation(weights);
effective_sample_size = opt.num_samples * ( 1 - rho ) / (1 + rho );
var_log_evidence = var(weights)*opt.num_samples/effective_sample_size^2;
end


