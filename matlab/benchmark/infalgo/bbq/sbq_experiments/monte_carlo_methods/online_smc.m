function [mean_log_evidences, var_log_evidences, samples, diagnostics] = ...
    online_smc(loglik_fn, prior, opt)
% Makes a fixed-length sampler into an online sampler, the slow way.
% Simply re-fixes the random seed and calls the sampler again.
%
% David Duvenaud
% February 2012

[mean_log_evidences, var_log_evidences, samples, diagnostics] = ...
    make_online_slow(@simple_monte_carlo, loglik_fn, prior, opt);

% Now convert to a distribution over Z instead of LogZ.
%[log_mean_evidences, log_var_evidences] = ...
%    log_of_normal_to_log_normal( mean_log_evidences, var_log_evidences );

end
