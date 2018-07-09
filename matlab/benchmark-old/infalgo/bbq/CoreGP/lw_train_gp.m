function gp = lw_train_gp(cov_fn, mean_fn, gp, X_data, y_data, opt)
% gp = lw_train_gp(gp, X_data, y_data, opt)
% gp = lw_train_gp(cov_fn, mean_fn, gp, X_data, y_data, opt);
%
% 'light-weight' training of a GP by locally optimising the likelihood
% around each of a number of hyperparameter samples. Each input scale is
% optimised in parallel, allowing high dimensional datasets to be fitted.
%
% INPUTS
% cov_fn: a string describing the covariance function to be used e.g.
%   'sqdexp', 'matern', 'ratquad', 'poly' or 'prodcompact' (see fcov.m).
% meanfn: a string describing the mean function to be used e.g.
%   'constant', 'planar' or 'quadratic'.
% gp: a structure describing the gp (leave as gp = [] if this is the
%   initial training, otherwise existing hyperparameters in gp will be used
%   to initialise).
% X_data: evaluated inputs to condition gp on, an N by D matrix
% y_data: evaluated outputs to condition gp on, an N by 1 vector
% opt: options, with defaults:
%                     'cov_fn', 'sqdexp', ...
%                     'mean_fn', 'constant', ...
%                     'num_hypersamples', min(500, 100 * num_dims), ...
%                     'derivative_observations', false, ...
%                     'optim_time', 60, ...
%   if print is false, print no output whatsoever
%                     'print', true, ... 
%                     'verbose', false, ...
%                     'maxevals_hs', 10, ...
%                     'plots', true, ...
%                     'num_passes', 6, ...
%                     'force_training', true, ...
%                     'parallel', true, ...
%   set noiseless to true if the function is known to be noiseless
%                     'noiseless', false, ...
%   if prior_mean is set to a number, that is taken as the constant prior
%   mean, if set to 'train', the prior mean constant is trained from data,
%   if 'default', it is simply set to the mean of y_data
%                     'prior_mean', 'default'); 
%
% OUTPUTS
% gp: trained gp structure

% Read inputs
% ========================================================================

if ischar('cov_fn')
    % lw_train_gp called as
    % [gp, logl_gp] = lw_train_gp(cov_fn, mean_fn, gp, X_data, y_data, opt);

    if nargin<6
        opt = struct();
    end
    
    opt.cov_fn = cov_fn;
    opt.mean_fn = mean_fn;
else
    % lw_train_gp called as
    % [gp, logl_gp] = lw_train_gp(gp, X_data, y_data, opt);
    
    gp = cov_fn;
    X_data = mean_fn;
    y_data = gp;
    if nargin<4
        opt = struct();
    else
        opt = X_data;
    end

end

if ~isstruct(opt)
    optim_time = opt;
    opt = struct();
    opt.optim_time = optim_time;
end

[num_data, num_dims] = size(X_data);

% Initialisation of gp structure
% ========================================================================

% only train those hyperparameters that are explicitly represented as
% active
if isfield(opt, 'active_hp_inds')
    gp.active_hp_inds = opt.active_hp_inds;
end

% don't want to use likelihood gradients in constructing Bayesian
% quadrature estimate
gp.grad_hyperparams = false;

% call set_gp just to create all appropriate hyperparameter structures in
% gp.hyperparams; we do not yet initialise their values
gp = set_gp(opt.cov_fn, opt.mean_fn, gp, X_data, [], ...
    opt.num_hypersamples);
hps_struct = set_hps_struct(gp);

    
% Set options
% ========================================================================

default_opt = struct(...
                    'cov_fn', 'sqdexp', ...
                    'mean_fn', 'constant', ...
                    'num_hypersamples', min(500, 100 * num_dims), ...
                    'derivative_observations', false, ...
                    'optim_time', 60, ...
                    'print', true, ... %if false, print no output whatsoever
                    'verbose', false, ...
                    'maxevals_hs', 10, ...
                    'plots', true, ...
                    'num_passes', 6, ...
                    'force_training', true, ...
                    'noiseless', false, ...
                    'prior_mean', 'default', ...  % if set to a number, that is taken as the prior mean
                    'hp_prior_mean', nan(hps_struct.num_hps, 1), ...
                    'hp_prior_sds', 10*ones(hps_struct.num_hps, 1));

opt = set_defaults( opt, default_opt );
opt.verbose = opt.verbose && opt.print;

if opt.print
fprintf('Beginning training of GP, budgeting for %g seconds\n', ...
    opt.optim_time);
end
start_time = cputime;

% Decide what to do about noise sd hyperparameter
% ========================================================================

output_scale_ind = hps_struct.logOutputScale;
noise_ind = hps_struct.logNoiseSD;
gp.noise_ind = noise_ind;
if opt.noiseless
    % we know the function is noiseless, no need to learn a noise sd
    % hyperparameter
    
    gp.active_hp_inds = setdiff(gp.active_hp_inds, noise_ind);
    gp.hyperparams(noise_ind).type = 'inactive';
    
    % include just a teensy little bit of noise (jitter) anyway to improve
    % conditioning of covariance matrix
    noiseless_const = -14;
    gp.hyperparams(noise_ind).priorMean = noiseless_const; 
   
    % we optimise each input scale independently by also allowing the noise
    % to vary when performing each optimisation; such that the noise can
    % soak up variation due to incorrect input scales in other dimensions.
    % The log noise sd we use for this purpose is equal to 
    %     big_noise_const + ...
    %         hypersamples(hypersample_ind).hyperparameters(big_noise_ind)
    big_noise_const = 6;
    big_noise_ind = noise_ind;
else
    big_noise_const = 0;
    big_noise_ind = noise_ind;
end

% Decide what to do about prior mean constant
% ========================================================================
mean_inds = hps_struct.mean_inds;
if any(strcmpi(opt.prior_mean, {'optimise','optimize','train'}))
    gp.active_hp_inds = union(gp.active_hp_inds, mean_inds);
    for i = 1:length(mean_inds)
        mean_ind = mean_inds(i);
        gp.hyperparams(mean_ind).type = 'real';
    end
elseif isnumeric(opt.prior_mean)
    gp.active_hp_inds = setdiff(gp.active_hp_inds, mean_inds);
    for i = 1:length(mean_inds)
        mean_ind = mean_inds(i);
        gp.hyperparams(mean_ind).type = 'inactive';
        gp.hyperparams(mean_ind).priorMean = opt.prior_mean(i);
    end
end


% Call set_gp again to actually appropriately initialise the values of
% hyperparameters
% ========================================================================

if opt.derivative_observations
    % we have derivative observations of our function
    
    % set_gp assumes a standard homogenous covariance, we don't want to tell
    % it about derivative observations.
    plain_obs = X_data(:,end) == 0;
    set_gp_X_data = X_data(plain_obs,1:end-1);
    set_gp_y_data = y_data(plain_obs,:);
    gp = set_gp(opt.cov_fn,opt.mean_fn, gp, set_gp_X_data, set_gp_y_data, ...
        opt.num_hypersamples);
    
    % now create a covariance function appropriate for derivative
    % observations
    
    hps_struct = set_hps_struct(gp);
    gp.covfn = @(flag) derivativise(@gp.covfn,flag);
    gp.meanfn = @(flag) wderiv_mean_fn(hps_struct,flag);
    
    gp.X_data = X_data;
    gp.y_data = y_data;

else
    gp = set_gp(opt.cov_fn, opt.mean_fn, gp, X_data, y_data, ...
        opt.num_hypersamples);
    

end

% Allocate the indices of hyperparameters we're going to train
% ========================================================================

% the indices of all hyperparameters to be trained
full_active_inds = gp.active_hp_inds;

% input_inds is a cell, input_inds{i} is a vector containing the indices of
% all hyperparameters related to the i'th input (this might include, for
% example, both the period of a periodic term and the input scale of a
% non-periodic term).
input_inds = hps_struct.input_inds;
gp.input_scale_inds = input_inds;
input_ind_vector = horzcat(input_inds{:});
active_input_inds = cellfun(@(x) intersect(x, full_active_inds), ...
    input_inds, ...
    'UniformOutput', false);
active_dims = find(~cellfun(@(x) isempty(x),active_input_inds));
num_active_dims = length(active_dims);


% the indices of non-input related hyperparameters
other_active_inds = ...
    setdiff(full_active_inds, input_ind_vector); 

% Work out how much time we can allow per hypersample
% ========================================================================

if isempty(full_active_inds)
    warning('train_gp:all_inactive', 'no hyperparameters active, no training performed')
    gp = revise_gp(X_data, y_data, gp, 'overwrite', []);  
    return
end
if opt.optim_time <= 0
    warning('train_gp:insuff_time', 'no time allowed for training, no training performed')
    
    % if we have little data, fill in prior means for hyperparameters where 
    % supplied
    if length(y_data) < 5;
        have_prior_mean = ~isnan(opt.hp_prior_mean);
        
        for i = 1:numel(gp.hypersamples)
            gp.hypersamples(i).hyperparameters(have_prior_mean) = ...
                opt.hp_prior_mean(have_prior_mean);
        end
    end
    
    gp = revise_gp(X_data, y_data, gp, 'overwrite', []);
    return
end

num_hypersamples = numel(gp.hypersamples);
tic
gp = ...
    revise_gp(X_data, y_data, gp, 'overwrite',[], 'all', ...
    [input_inds{1}, noise_ind]);
hs_eval_time = toc/num_hypersamples;

% for the purposes of using parfor, we need to strip off hypersamples from
% gp; we'll return it later
hypersamples = gp.hypersamples;
gp = rmfield(gp, 'hypersamples');

ideal_time = num_hypersamples * (...
                opt.maxevals_hs * (opt.num_passes * (num_dims + 1)) ...
                * hs_eval_time...
                );
% could potentially increase this thanks to expected speed-up from
% parallelisation
scale_factor = opt.optim_time / ideal_time;

% set the allowed number of likelihood evaluations
opt.maxevals_hs = ceil(opt.maxevals_hs * scale_factor);

if opt.maxevals_hs == 1
    warning('train_gp:insuff_time','insufficient time allowed to train GP, consider decreasing opt.num_hypersamples or increasing opt.optim_time');
    if opt.force_training
        warning('train_gp:insuff_time','proceeding with minimum possible likelihood evaluations');
        opt.maxevals_hs = 2;
    else
        gp.hypersamples = hypersamples;
        return
    end
elseif opt.verbose
    fprintf('Using %g likelihood evals per pass, per input\n', opt.maxevals_hs)
end

% Begin training proper
% ========================================================================

[max_logL, max_ind] = max(vertcat(hypersamples.logL));

if opt.print
fprintf('Initial best log-likelihood: \t%g',max_logL);
end
if opt.verbose
    fprintf(', for ')
    disp_gp_hps(hypersamples, max_ind,'no_logL',...
        noise_ind, input_inds, output_scale_ind, mean_inds);
end
if opt.print
fprintf('\n');
end

% We'll perform a number of passes, optimising input scales in parallel
% within each pass
for num_pass = 1:opt.num_passes
    
    if opt.verbose
        fprintf('Pass %g\n', num_pass)
    end
    
    % Use parallel toolbox over number of hyperparameter samples (rather
    % than over input scales), which we typically expect to exceed the
    % number of active input dimensions
    for hypersample_ind = 1:num_hypersamples
        warning('off', 'MATLAB:nearlySingularMatrix');
        warning('off', 'MATLAB:singularMatrix');
        warning('off','revise_gp:small_num_data');
        
        
        if opt.verbose
            fprintf('Hyperparameter sample %g\n',hypersample_ind)
        end
        
        % we optimise each input scale independently by also allowing the
        % noise to vary when performing each optimisation; such that the
        % noise can soak up variation due to incorrect input scales in
        % other dimensions.
        big_log_noise_sd = big_noise_const + ...
            hypersamples(hypersample_ind).hyperparameters(big_noise_ind);
        actual_log_noise_sd = ...
            hypersamples(hypersample_ind).hyperparameters(noise_ind);
        hypersamples(hypersample_ind).hyperparameters(noise_ind) = ...
            big_log_noise_sd;
        
        current_hypersample = hypersamples(hypersample_ind);
        current_log_input_scales = nan(num_active_dims, 1);

        % optimise input scales in parallel
        %par
        for d_ind = 1:num_active_dims 
            d = active_dims(d_ind);
            active_hp_inds = [input_inds{d}, noise_ind];
            
            try
                inputscale_hypersample = ...
                    move_hypersample(...
                    current_hypersample, gp, ...
                    active_hp_inds, ...
                    X_data, y_data, opt);
            catch err
                if opt.verbose
                    warning('error experienced in training input scales, reverting to initial input scale');
                    disp(err);
                end
                inputscale_hypersample = current_hypersample;
            end
            
            current_log_input_scales(d_ind) = ...
                inputscale_hypersample.hyperparameters(input_inds{d});
                        
            if opt.verbose
                fprintf(', \t for input_scale(%g) = %g\n', ...
                    d, exp(current_log_input_scales(d_ind)));
            end
        end
        
        hypersamples(hypersample_ind).hyperparameters...
            ([input_inds{1:num_active_dims}]) = ...
                current_log_input_scales;
        
        hypersamples(hypersample_ind).hyperparameters(noise_ind) = ...
            actual_log_noise_sd;

         % optimise other hyperparameters

        active_hp_inds = other_active_inds;

        other_hypersample = ...
            move_hypersample(...
                hypersamples(hypersample_ind), gp, ...
                active_hp_inds, ...
                X_data, y_data, opt);
            
       hypersamples(hypersample_ind) = other_hypersample;
       
       if opt.noiseless
           % we're not training noise, but at least try to keep it relevant
           % to trained output scale.
            hypersamples(hypersample_ind).hyperparameters(noise_ind) = ...
            noiseless_const + ...
            hypersamples(hypersample_ind).hyperparameters(output_scale_ind);
       end
                
        if opt.verbose
            fprintf(', \t for ');
            disp_gp_hps(hypersamples(hypersample_ind), [], 'no_logL', ...
                noise_ind, input_inds, output_scale_ind, mean_inds);
            fprintf('\n');
        end

    end
    
    if (cputime-start_time > opt.optim_time) && num_pass > 0 ...
        || num_pass == opt.num_passes
        % need to end here so that we have gp that has been trained on the
        % whole dataset, rather than on a `close' subset
        break
    end  
  
    if opt.print
    fprintf('\n');
    end
    
end

gp.hypersamples = hypersamples;
gp.X_data = X_data;
gp.y_data = y_data;
gp.active_hp_inds = full_active_inds;
gp.output_scale_ind = output_scale_ind;

[max_logL, max_ind] = max([gp.hypersamples.logL]);

if opt.print
fprintf('Final best log-likelihood: \t%g',max_logL);
end
if opt.verbose
    fprintf(', for ')
    disp_gp_hps(gp, max_ind, 'no_logL');
elseif opt.print
    fprintf('\n');
end

if opt.print
    fprintf('Completed retraining of GP in %g seconds\n', cputime-start_time)
    fprintf('\n');
end

end
       

function hypersample = move_hypersample(...
    hypersample, gp, active_hp_inds, X_data, y_data, opt)

gp.hypersamples = hypersample;

fn = @(a_hs) gp_neg_log_posterior_fn...
    (a_hs, X_data, y_data, gp, active_hp_inds, opt);

minFunc_opt.MaxFunEvals = opt.maxevals_hs;
minFunc_opt.Display = 'off';
a_hs_init = hypersample.hyperparameters(active_hp_inds)';
logL_init = hypersample.logL;
[a_hs_final, neg_logL_final] = minFunc(fn, a_hs_init, minFunc_opt);

hypersample.hyperparameters(active_hp_inds) = a_hs_final';
gp.hypersamples = hypersample;
gp = revise_gp(X_data, y_data, gp, 'overwrite', [], 'all', active_hp_inds);
hypersample = gp.hypersamples;

if opt.verbose
    fprintf('LogL: %g -> %g',logL_init, -neg_logL_final);
end

end

function [neg_log_post, neg_g_log_post] = ...
    gp_neg_log_posterior_fn(a_hs, X_data, y_data, gp, active_hp_inds, opt)

gp.hypersamples(1).hyperparameters(active_hp_inds) = a_hs';

gp = revise_gp(X_data, y_data, gp, 'overwrite', [], 'all', active_hp_inds);

neg_logL = -gp.hypersamples(1).logL;
neg_glogL = -vertcat(gp.hypersamples(1).glogL{active_hp_inds});

a_prior_mean = opt.hp_prior_mean(active_hp_inds);
a_prior_sds = opt.hp_prior_sds(active_hp_inds);
% nd stands for non-diffuse: for these hyperparameters, we do MAP rather 
% than ML. 
a_nd_priors = ~isnan(a_prior_mean);

neg_log_prior = 0.5 * ((a_hs - a_prior_mean)./a_prior_sds).^2;
neg_log_prior = sum(neg_log_prior(a_nd_priors));
neg_g_log_prior = (a_hs - a_prior_mean) .* a_prior_sds.^-2;
neg_g_log_prior(~a_nd_priors) = 0;

neg_log_post = neg_logL + neg_log_prior;
neg_g_log_post = neg_glogL + neg_g_log_prior;

end
    