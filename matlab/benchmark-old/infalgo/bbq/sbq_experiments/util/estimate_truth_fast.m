% A script to estimate the ground truth for some problems.

problems = define_integration_problems();
f3 = problems{9};
f3.name

opt.num_samples = 1000000;

%[mean3, var3] = simple_monte_carlo(f3.log_likelihood_fn, f3.prior, opt)


%fprintf('mean3: %20.15f\n', mean3);


problems = define_integration_problems();
f7 = problems{end};
f7.name

[mean7, var7] = simple_monte_carlo(f7.log_likelihood_fn, f7.prior, opt)

save 'long_run'



fprintf('mean7: %20.15f\n', mean7);
