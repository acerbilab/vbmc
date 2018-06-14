function log_l = log_gp_lik(sample, X_data, y_data, gp, active_hp_inds)

if nargin<5
    active_hp_inds = gp.active_hp_inds;
end

if size(sample, 1) ~= 1
    sample = sample';
end
gp.grad_hyperparams = false;
gp.hypersamples(1).hyperparameters = horzcat(gp.hyperparams(:).priorMean);
gp.hypersamples(1).hyperparameters(active_hp_inds) = sample;
gp = revise_gp(X_data, y_data, gp, 'overwrite', [], 1);
log_l = gp.hypersamples(1).logL;
%fprintf('%g,',log_l)