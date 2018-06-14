function [mean_log_evidences, var_log_evidences, samples, diagnostics] = ...
    make_online_slow(sampler, loglik_fn, prior, opt)
% Makes a fixed-length sampler into an online sampler, the slow way.
% Simply re-fixes the random seed and calls the sampler again and again with
% more samples each time.
% 
% Inputs:
% - sampler: the function to be called at every iteration.
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
%   mean_log_evidences: the means of our poterior over the log of the evidence.
%   var_log_evidences: the variances of our posterior over the log of the
%                     evidence.
% - samples : n*d matrix of samples
%
%
% David Duvenaud
% February 2012

if nargin < 3
    opt.num_samples = 100;
end

% Get the random seed.
stream = RandStream.getDefaultStream;
savedState = stream.State;


mean_log_evidences = NaN(opt.num_samples, 1);
var_log_evidences = NaN(opt.num_samples, 1);

cur_opt = opt;  % The options that will be used by the sampler.

for num_s = 1:opt.num_samples
    stream.State = savedState;    % Set the random seed.
    
    cur_opt.num_samples = num_s;
    [mean_log_evidences(num_s), var_log_evidences(num_s), samples, cur_diagnostics] = ...
        sampler(loglik_fn, prior, cur_opt);
    diagnostics(num_s).field = cur_diagnostics;
end
end
