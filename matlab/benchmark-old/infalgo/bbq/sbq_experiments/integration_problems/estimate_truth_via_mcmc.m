function [log_mean, log_var] = estimate_truth_via_smc(p_ix, num_samples)
% A script to estimate the ground truth for some problems.

problems = define_integration_problems();
p = problems{p_ix};
p.name

opt.num_samples = num_samples

[log_mean, log_var] = simple_monte_carlo(p.log_likelihood_fn, p.prior, opt)

save([p.name '_truth_' int2str(num_samples) ]);

fprintf('\n\n mean = %20.15f\n\n', log_mean);

