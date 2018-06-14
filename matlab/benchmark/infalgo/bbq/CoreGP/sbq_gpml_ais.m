function [log_mean_evidence, log_var_evidence, samples, gp_hypers] = ...
    sbq_gpml_ais(log_likelihood_fn, prior, opt)
% Take samples samples_mat so as to best estimate the
% evidence, an integral over exp(log_l_fn) against the prior in prior_struct.
% 
% This version uses GPML to set hyperparams, and AIS to choose points.
%
% OUTPUTS
% - mean_log_evidences: our mean estimate for the log of the evidence
% - var_log_evidences: the variance for the log of the evidence
% - sample_locations: n*d matrix of hyperparameter samples
% - gp_hypers
% 
% INPUTS
% - log_likelihood_fn: a function that takes a single argument, a 1*n vector of
%                      hyperparameters, and returns the log of the likelihood.
% - prior: requires fields
%                 * means
%                 * sds
% - opt: takes fields:
%        * num_samples: the number of samples to draw. If opt is a number rather
%          than a structure, it's assumed opt = num_samples.
%        * plots: Whether to plot the expected variance surface (only works in 1d)
%        * set_ls_var_method:  How to estimate the variance of lengthscale
%                              parameters.  Can be one of:
%            + 'laplace': Compute the Hessian of the log-likelihood surface.
%            + 'none': Assume zero variance in the lengthscales.


% Initialize options.
% ===========================
if nargin<3
    opt = struct();
elseif ~isstruct(opt)
    num_samples = opt;
    opt = struct();
    opt.num_samples = num_samples;
end

D = numel(prior.mean);

% Set unspecified fields to default values.
default_opt = struct('num_samples', 100, ...
                     'plots', false, ...
                     'gamma', 1/100, ...
                     'set_ls_var_method', 'laplace');
opt = set_defaults( opt, default_opt );

% Get sample locations from a run of AIS.
[ais_mean_log_evidence, ais_var_log_evidence, ais_sample_locs, ais_sample_vals, stats] = ...
    ais_mh(log_likelihood_fn, prior, opt);

%[sample_locs, sample_vals] = ...
%    remove_duplicate_samples(sample_locs, sample_vals);
sample_locs = stats.all_samples.locations;
sample_vals = stats.all_samples.logliks;

opt.num_samples = length(sample_vals);

% Update sample struct.
% ==================================
samples.locations = sample_locs;
for i = 1:opt.num_samples
    samples.log_l(i,:) = log_likelihood_fn(samples.locations(i,:));
end
samples.max_log_l = max(samples.log_l); % all log-likelihoods have max_log_l subtracted off
samples.scaled_l = exp(samples.log_l - samples.max_log_l);
samples.tl = log_transform(samples.scaled_l, opt.gamma);


% Train GPs
% ===========================   
inference = @infExact;
likfunc = @likGauss;
meanfunc = {'meanZero'};
max_iters = 1000;
covfunc = @covSEiso;

% Init GP Hypers.
init_hypers.mean = [];
init_hypers.lik = log(0.01);  % Values go between 0 and 1, so no need to scale.
init_lengthscales = mean(sqrt(diag(prior.covariance)))/10;
init_output_variance = .1;
init_hypers.cov = log( [init_lengthscales init_output_variance] ); 

% Fit the model, but not the likelihood hyperparam (which stays fixed).
fprintf('Fitting GP to observations...\n');
gp_hypers = init_hypers;
gp_hypers = minimize(gp_hypers, @gp_fixedlik, -max_iters, ...
                     inference, meanfunc, covfunc, likfunc, ...
                     samples.locations, samples.scaled_l);
if any(isnan(gp_hypers.cov))
    gp_hypers = init_hypers;
    warning('Optimizing hypers failed');
end
l_gp_hypers.log_output_scale = gp_hypers.cov(end);
l_gp_hypers.log_input_scales(1:D) = gp_hypers.cov(1:end - 1);
fprintf('Output variance: '); disp(exp(l_gp_hypers.log_output_scale));
fprintf('Lengthscales: '); disp(exp(l_gp_hypers.log_input_scales));

fprintf('Fitting GP to log-observations...\n');
gp_hypers_log = init_hypers;
gp_hypers_log = minimize(gp_hypers_log, @gp_fixedlik, -max_iters, ...
                         inference, meanfunc, covfunc, likfunc, ...
                         samples.locations, samples.tl);        
if any(isnan(gp_hypers_log.cov))
    gp_hypers_log = init_hypers;
    warning('Optimizing hypers on log failed');
end
tl_gp_hypers.log_output_scale = gp_hypers_log.cov(end);
tl_gp_hypers.log_input_scales(1:D) = gp_hypers_log.cov(1:end - 1);
fprintf('Output variance of logL: '); disp(exp(tl_gp_hypers.log_output_scale));
fprintf('Lengthscales on logL: '); disp(exp(tl_gp_hypers.log_input_scales));

if opt.plots && D == 1;
    gpml_plot( gp_hypers, samples.locations, samples.scaled_l);
    title('GP on untransformed values');
    gpml_plot( gp_hypers_log, samples.locations, samples.tl);
    title('GP on log( exp(scaled) + 1) values');
end

[log_mean_evidence, log_var_evidence] = ...
    log_evidence(samples, prior, l_gp_hypers, tl_gp_hypers, [], opt);
end
