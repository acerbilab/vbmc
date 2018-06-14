clear;
%cd ~/Code/GP/BQR
% 


problem_bbq_predict_bq;

opt.num_samples = 10;
opt.num_retrains = 3;
[log_mean_evidences, log_var_evidences, samples, ...
    diagnostics] = ...
    sbq(log_l_fn, prior, opt);

for i = 1:size(samples.locations, 1)
    samples.qd(i,1) = q_fn(samples.locations(i,:));
end

% GP training options.
gp_train_opt.optim_time = 60;
gp_train_opt.noiseless = true;
% print to screen diagnostic information about gp training
gp_train_opt.print = 0;
% plot diagnostic information about gp training
gp_train_opt.plots = false;
gp_train_opt.parallel = true;
gp_train_opt.num_hypersamples = 10;



qd_gp = train_gp('sqdexp', 'constant', [], ...
                             samples.locations, samples.qd, ...
                             gp_train_opt);

% Put the values of the best hyperparameters into dedicated structures.
qd_gp_hypers_SE = best_hyperparams(qd_gp);
qdd_gp_hypers_SE = qd_gp_hypers_SE;

[mean_out, sd_out, unadj_mean_out, unadj_sd_out] = ...
    predict_bq(samples, prior, ...
    l_gp_hypers_SE, tl_gp_hypers_SE, del_gp_hypers_SE, ...
    qd_gp_hypers_SE, qdd_gp_hypers_SE, ev_params, opt);