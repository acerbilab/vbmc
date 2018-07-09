function [gp, quad_gp] = train_gpgo(gp, X_data, y_data, opt)
% train gp for gpgo

fprintf('\n Beginning retraining of GP\n');


warning off

% switch derivatives on now.
if isfield(gp, 'grad_hyperparams')
    grad_hyperparams = gp.grad_hyperparams;
else
    grad_hyperparams = true;
end
gp.grad_hyperparams = true;

r_X_data = vertcat(gp.hypersamples.hyperparameters);
r_y_data = vertcat(gp.hypersamples.logL);

[quad_noise_sd, quad_input_scales] = ...
    hp_heuristics(r_X_data, r_y_data, 100);

[max_logL,max_ind] = max(r_y_data);

if opt.verbose
    fprintf('Initial best log-likelihood: \t%g,\t for sample: [',max_logL);
    fprintf('%g ',r_X_data(max_ind,:));
    fprintf(']\n');
end


gp = rmfield(gp,{'hyperparams','hypersamples'});
% now we completely overwrite gp

if opt.derivative_observations
    % set_gp assumes a standard homogenous covariance, we don't want to tell
    % it about derivative observations.
    plain_obs = X_data(:,end) == 0;

    set_X_data = X_data(plain_obs,1:end-1);
    set_y_data = y_data(plain_obs,:);


    gp = set_gp(opt.cov_fn, opt.mean_fn, gp, set_X_data, set_y_data, ...
        opt.num_hypersamples);

    hps_struct = set_hps_struct(gp);
    % need to define this handle or else infinite recursion results
    if ~isfield(gp, 'non_deriv_cov_fn')
        gp.non_deriv_cov_fn = gp.covfn;
    end
    gp.covfn = @(flag) derivativise(gp.non_deriv_cov_fn,flag);
    gp.meanfn = @(flag) wderiv_mean_fn(hps_struct,flag);

    gp.X_data = X_data;
    gp.y_data = y_data;


else
    gp = set_gp(opt.cov_fn, opt.mean_fn, [], X_data, y_data, ...
        opt.num_hypersamples);
end



hps_struct = set_hps_struct(gp);
input_scale_inds = hps_struct.logInputScales;
noise_sd_ind = hps_struct.logNoiseSD;
output_scale_ind = hps_struct.logOutputScale;
num_dims = size(X_data,2);
actual_log_noise_sd = log(1e-9) + gp.hyperparams(output_scale_ind).priorMean;
%gp.hyperparams(noise_sd_ind).priorMean;
big_log_noise_sd = log(0.1) + gp.hyperparams(output_scale_ind).priorMean;

gp.hyperparams(noise_sd_ind)=orderfields(...
    struct('name','logNoiseSD',...
        'priorMean',actual_log_noise_sd,...
        'priorSD',eps,...
        'NSamples',1,...
        'type','inactive'),gp.hyperparams);

sampled_gp = hyperparams(gp);
full_active_inds = sampled_gp.active_hp_inds;
hypersamples = sampled_gp.hypersamples;
num_hypersamples = numel(hypersamples);
names = {'logL', 'glogL', 'datahalf', 'datatwothirds', 'cholK', 'K', 'jitters'};
for i = 1:length(names)
    hypersamples(1).(names{i}) = nan;
end

if nargin<4
    total_num_evals = num_dims*10;
else
    total_num_evals = opt.train_evals;
end
num_input_scale_passes = max(2,ceil(total_num_evals/(num_hypersamples*2)));
num_split_evals = ...
    max(1,floor(total_num_evals/(num_input_scale_passes*(num_dims+1)+1)));
opt.maxevals = num_split_evals;

tic;

%par
for hypersample_ind = 1:num_hypersamples

    if opt.verbose
        fprintf('Hyperparameter sample %g\n',hypersample_ind)
    end

    warning off

    hypersample = hypersamples(hypersample_ind);
    sample_quad_input_scales = quad_input_scales;

    for num_pass = 1:num_input_scale_passes

        % optimise input scales

        hypersample.hyperparameters(noise_sd_ind) = big_log_noise_sd;

        for d = 1:length(input_scale_inds)
            active_hp_inds = [input_scale_inds(d), noise_sd_ind];

            [inputscale_hypersample] = ...
                move_hypersample(...
                hypersample, gp, sample_quad_input_scales, ...
                active_hp_inds, ...
                X_data, y_data, opt);

            hypersample.hyperparameters(input_scale_inds(d)) = ...
                inputscale_hypersample.hyperparameters(input_scale_inds(d));
        end

        hypersample.hyperparameters(noise_sd_ind) = actual_log_noise_sd;

         % optimise output scale

        active_hp_inds = [output_scale_ind];

        [outputscale_hypersample] = ...
            move_hypersample(...
                hypersample, gp, sample_quad_input_scales, ...
                active_hp_inds, ...
                X_data, y_data, opt);
        hypersample.hyperparameters(active_hp_inds) = ...
                outputscale_hypersample.hyperparameters(active_hp_inds);


    end



    % now do a joint optimisation to finish off

    [hypersamples(hypersample_ind)] = ...
        move_hypersample(...
                hypersample, gp, sample_quad_input_scales, ...
                full_active_inds, ...
                X_data, y_data, opt);

    if opt.verbose
        fprintf('Final log-likelihood: \t%g\n',hypersamples(hypersample_ind).logL)
    end
end


gp.hypersamples = hypersamples;
gp.X_data = X_data;
gp.y_data = y_data;
gp.active_hp_inds = full_active_inds;
gp.input_scale_inds = input_scale_inds;



gp.grad_hyperparams = grad_hyperparams;

r_X_data = vertcat(gp.hypersamples.hyperparameters);
r_y_data = vertcat(gp.hypersamples.logL);

[max_logL,max_ind] = max(r_y_data);

if opt.verbose
    fprintf('Final best log-likelihood: \t%g,\t for sample: [',max_logL);
    fprintf('%g ',r_X_data(max_ind,:));
    fprintf(']\n');
end

[quad_noise_sd, quad_input_scales, quad_output_scale] = ...
    hp_heuristics(r_X_data, r_y_data, 100);

quad_gp.quad_noise_sd = quad_noise_sd;
quad_gp.quad_input_scales = quad_input_scales;
quad_gp.quad_output_scale = quad_output_scale;

warning on
fprintf('Completed retraining of GP\n')
toc;
fprintf('\n');

function [hypersample] = move_hypersample(...
    hypersample, gp, quad_input_scales, active_hp_inds, X_data, y_data, opt)


gp.active_hp_inds = active_hp_inds;
gp.hypersamples = hypersample;
a_quad_input_scales = quad_input_scales(active_hp_inds);

flag = false;
i = 0;
a_hs_mat = nan(opt.maxevals+1,length(active_hp_inds));
logL_mat = nan(opt.maxevals+1,1);

while ~flag && i < opt.maxevals
    i = i+1;

    gp = revise_gp(X_data, y_data, gp, 'new_hps');

    logL = gp.hypersamples.logL;
    if opt.verbose
        fprintf('%g,',logL)
    end
    a_hs = gp.hypersamples.hyperparameters(active_hp_inds);

    a_hs_mat(i,:) = a_hs;
    logL_mat(i) = logL;

    if i>1 && logL_mat(i) < logL_mat(i-1)
        a_quad_input_scales = 0.1*a_quad_input_scales;

        a_hs = a_hs_mat(i-1,:);
    else
        a_grad_logL = [gp.hypersamples.glogL{:}];
        a_grad_logL = a_grad_logL(active_hp_inds);
    end


    [a_hs, flag] = simple_zoom_pt(a_hs, a_grad_logL, ...
                            a_quad_input_scales, 'maximise');
    gp.hypersamples.hyperparameters(active_hp_inds) = a_hs;
end

gp = revise_gp(X_data, y_data, gp, 'new_hps');
hypersample = gp.hypersamples;
logL = gp.hypersamples.logL;

a_hs_mat(i+1,:) = a_hs;
logL_mat(i+1) = logL;

[max_logL,max_ind] = max(logL_mat);
gp.hypersamples.hyperparameters(active_hp_inds) = a_hs_mat(max_ind,:);
gp = revise_gp(X_data, y_data, gp, 'new_hps');
hypersample = gp.hypersamples;

% not_nan = all(~isnan([a_hs_mat,logL_mat]),2);
%
% [quad_noise_sd, a_quad_input_scales] = ...
%     hp_heuristics(a_hs_mat(not_nan,:), logL_mat(not_nan), 10);
% quad_input_scales(active_hp_inds) = a_quad_input_scales;

if opt.verbose
fprintf('%g\n',logL)
end
