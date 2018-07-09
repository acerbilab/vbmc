function untransformed_vs_transformed()

col_width = 8.25381;  % ICML double column width in cm.

clf;
randn('state', 0);
rand('twister', 0);  

prior.mean = 0;
prior.covariance = 1;

log_likelihood_fn = @(x)logmvnpdf( x, -.5, .5 ); 

D = 1;

% Set unspecified fields to default values.
opt = struct('num_samples', 20, ...
                     'gamma', .1, ...
                     'set_ls_var_method', 'laplace');

% Get sample locations from a run of AIS.
[ais_mean_log_evidence, ais_var_log_evidence, sample_locs, sample_vals] = ...
    ais_mh(log_likelihood_fn, prior, opt);

[sample_locs, sample_vals] = ...
    remove_duplicate_samples(sample_locs, sample_vals);
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

    %subplot( 1, 2, 1);
    subaxis( 1, 2, 1,'SpacingHorizontal',0.1, 'MarginLeft', .1,'MarginRight',.02);
    custom_gpml_plot( gp_hypers, samples.locations, samples.scaled_l, '$\ell(x)$', [-2 2]);
    %title('GP on $\ell(x)$', 'Fontsize', 8, 'Interpreter','latex');
    
    %subplot( 1, 2, 2);
    subaxis( 1, 2, 2,'SpacingHorizontal',0.1, 'MarginLeft', .1,'MarginRight', .02);
    handles = custom_gpml_plot( gp_hypers_log, samples.locations, samples.tl, '$\log( \ell(x))$', [-2 2]);
    %title('GP on $\log( \gamma \ell(x) + 1)$', 'Fontsize', 8, 'Interpreter','latex');
    
    %subplot(1, 3, 3);
    %legend( handles, {'GP Posterior Mean', 'GP Posterior Uncertainty', 'Data'}, 'Location', 'SouthEast');
    %legend boxoff
    

set_fig_units_cm( col_width, 4 );
matlabfrag('~/Dropbox/papers/sbq-paper/figures/log_transform2');    


end
