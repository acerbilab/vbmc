function hs_weights = hypersample_weights(hypersamples, hypersamples_logLs, ...
    hyperparam_means, hyperparam_sds)
% exaple arguments:
% hypersamples = rand(100,5);
% hypersamples_logLs = rand(100,1);
% 
% hyperparam_means = rand(1,5);
% hyperparam_sds = rand(1,5);


[quad_noise_sd, quad_input_scales, quad_output_scale] = ...
hp_heuristics(hypersamples, hypersamples_logLs, 100);

quad_gp.quad_noise_sd = quad_noise_sd;
quad_gp.quad_input_scales = quad_input_scales;
quad_gp.quad_output_scale = quad_output_scale;

[num_hypersamples, num_hyperparams] = size(hypersamples);

for i = 1:num_hypersamples
    gp.hypersamples(i).hyperparameters = hypersamples(i,:);
end
for i = 1:num_hyperparams
    gp.hyperparams(i).priorMean = hyperparam_means(i);
    gp.hyperparams(i).priorSD = hyperparam_sds(i);
end

weights_mat = bq_params(gp,quad_gp);

for i = 1:num_hypersamples
    gp.hypersamples(i).logL = hypersamples_logLs(i);
end

hs_weights = weights(gp, weights_mat);