function [gp, logl_gp] = train_gp(cov_fn, mean_fn, gp, X_data, y_data, opt)
% [gp, logl_gp] = train_gp(gp, X_data, y_data, opt)
% [gp, logl_gp] = train_gp(cov_fn, mean_fn, gp, X_data, y_data, opt);
%
% trains a GP by locally optimising the likelihood around each of a number
% of hyperparameter samples. Each input scale is optimised in parallel,
% allowing high dimensional datasets to be fitted.
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
% logl_gp: contains fields noise_sd, input_scales and
%   output_scale; it describes the (quadrature) hyperparameters of the
%   gp fitted to the log-likelihood as a function of the hyperparameters.


% Read inputs
% ========================================================================

if ischar('cov_fn')
    % train_gp called as
    % [gp, logl_gp] = train_gp(cov_fn, mean_fn, gp, X_data, y_data, opt);

    if nargin<6
        opt = struct();
    end
    
    opt.cov_fn = cov_fn;
    opt.mean_fn = mean_fn;
else
    % train_gp called as
    % [gp, logl_gp] = train_gp(gp, X_data, y_data, opt);
    
    gp = cov_fn;
    X_data = mean_fn;
    y_data = gp;
    if nargin<4
        opt = struct();
    else
        opt = X_data;
    end

end

[num_data, num_dims] = size(X_data);

if ~isstruct(opt)
    optim_time = opt;
    opt = struct();
    opt.optim_time = optim_time;
end
    
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
                    'prior_mean', 'default'); % if set to a number, that is taken as the prior mean

opt = set_defaults( opt, default_opt );
opt.verbose = opt.verbose && opt.print;

if opt.print
fprintf('Beginning training of GP, budgeting for %g seconds\n', ...
    opt.optim_time);
end
start_time = cputime;

% only train those hyperparameters that are explicitly represented as
% active
if isfield(opt, 'active_hp_inds')
    gp.active_hp_inds = opt.active_hp_inds;
end

% Initialisation of gp structure
% ========================================================================

% don't want to use likelihood gradients in constructing Bayesian
% quadrature estimate
gp.grad_hyperparams = false;

% call set_gp just to create all appropriate hyperparameter structures in
% gp.hyperparams; we do not yet initialise their values
gp = set_gp(opt.cov_fn, opt.mean_fn, gp, X_data, [], ...
    opt.num_hypersamples);
hps_struct = set_hps_struct(gp);

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
    gp.hyperparams(noise_ind).priorMean = ...
       gp.hyperparams(output_scale_ind).priorMean - 14; 
   
    % we optimise each input scale independently by also allowing the noise to
    % vary when performing each optimisation; such that the noise can soak
    % up variation due to incorrect input scales in other dimensions.
    big_noise_const = +9;
    big_noise_ind = noise_ind;
else
    big_noise_const = 0;
    big_noise_ind = noise_ind;
end

% deal with the prior mean constant appropriately depending on the value of
% opt.prior_mean
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


% call set_gp again to actually appropriately initialise the values of
% hyperparameters
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

full_active_and_noise_inds = union(full_active_inds, noise_ind);

% Work out how much time we can allow per hypersample
% ========================================================================

if isempty(full_active_inds)
    warning('train_gp:all_inactive', 'no hyperparameters active, no training performed')
    gp = revise_gp(X_data, y_data, gp, 'overwrite', []);  
    return
end
if opt.optim_time <= 0
    warning('train_gp:insuff_time', 'no time allowed for training, no training performed')
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

% Fit GP to log-likelihood surface
% ========================================================================

logl_X_data = vertcat(hypersamples.hyperparameters);
logl_y_data = vertcat(hypersamples.logL);

num_hps = size(logl_X_data,2);
logl_input_scales = nan(1,num_hps);
[logl_noise_sd, ...
    logl_input_scales(full_active_and_noise_inds), ...
    logl_output_scale] = ...
    hp_heuristics(logl_X_data(:,full_active_and_noise_inds), logl_y_data, 100);

% only specified in case of early return
logl_gp.noise_sd = logl_noise_sd;
logl_gp.input_scales = logl_input_scales;
logl_gp.output_scale = logl_output_scale;

[max_logL, max_ind] = max(logl_y_data);

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

% Begin training proper
% ========================================================================

% We'll perform a number of passes, optimising input scales in parallel
% within each pass
for num_pass = 1:opt.num_passes
    
    if opt.verbose
        fprintf('Pass %g\n', num_pass)
    end
    
    % log_input_scale_cell{i}{j}(k, :) stores the kth log input scale
    % vector evaluated in the process of optimising the jth input scale for
    % the ith hyperparameter sample
    log_input_scale_cell = cell(num_hypersamples, 1);
    input_scale_logL_cell = cell(num_hypersamples, 1);
    
    other_cell = cell(num_hypersamples,1);
    other_logL_cell = cell(num_hypersamples,1);
    
    parfor hypersample_ind = 1:num_hypersamples
        warning('off','revise_gp:small_num_data');
        
        if opt.verbose
            fprintf('Hyperparameter sample %g\n',hypersample_ind)
        end
        
        big_log_noise_sd = big_noise_const + ...
            hypersamples(hypersample_ind).hyperparameters(big_noise_ind);
        actual_log_noise_sd = ...
            hypersamples(hypersample_ind).hyperparameters(noise_ind);

        hypersamples(hypersample_ind).hyperparameters(noise_ind) = ...
            big_log_noise_sd;

        % optimise input scale

        log_input_scale_cell{hypersample_ind} = cell(1, num_active_dims);
        input_scale_logL_cell{hypersample_ind} = cell(1, num_active_dims);
        for d_ind = 1:num_active_dims 
            d = active_dims(d_ind);
            active_hp_inds = [input_inds{d}, noise_ind];
            

            [inputscale_hypersample, log_input_scale_mat, input_scale_logL_mat] = ...
                move_hypersample(...
                hypersamples(hypersample_ind), gp, logl_input_scales, ...
                active_hp_inds, ...
                X_data, y_data, opt);
            
            log_input_scale_d = ...
                inputscale_hypersample.hyperparameters(input_inds{d});

            hypersamples(hypersample_ind).hyperparameters(input_inds{d}) = ...
                log_input_scale_d;
            
            log_input_scale_cell{hypersample_ind}{d_ind} = log_input_scale_mat;
            input_scale_logL_cell{hypersample_ind}{d_ind} = input_scale_logL_mat;
            
            
            if opt.verbose
                fprintf(', \t for input_scale(%g) = %g\n', ...
                    d, exp(log_input_scale_d));
            end
        end
        
        hypersamples(hypersample_ind).hyperparameters(noise_ind) = ...
            actual_log_noise_sd;

         % optimise other hyperparameters

        active_hp_inds = other_active_inds;

        [other_hypersample, other_mat, other_logL_mat] = ...
            move_hypersample(...
                hypersamples(hypersample_ind), gp, logl_input_scales, ...
                active_hp_inds, ...
                X_data, y_data, opt);
            
        other_cell{hypersample_ind} = other_mat;
        other_logL_cell{hypersample_ind} = other_logL_mat;
            

        hypersamples(hypersample_ind) = other_hypersample;
                
        if opt.verbose
            fprintf(', \t for ');
            disp_gp_hps(hypersamples(hypersample_ind), [], 'no_logL', ...
                noise_ind, input_inds, output_scale_ind, mean_inds);
            fprintf('\n');
        end

    end

    % now we estimate the scale of variation of the likelihood wrt
    % log input_scale.
    
    log_input_scale_compcell = cat(1,log_input_scale_cell{:});
    input_scale_logL_compcell = cat(1,input_scale_logL_cell{:});
    logl_noise_sds = nan(num_dims+1,1);
    logl_output_scales = nan(num_dims+1,1);
        
    for d_ind = 1:num_active_dims
        a_hps_mat = cat(1, log_input_scale_compcell{:,d_ind});
        logL_mat = cat(1, input_scale_logL_compcell{:,d_ind});
        
        sorted_logL_mat = sort(logL_mat);
              
        top_inds = logL_mat >= sorted_logL_mat(round(0.9*end));
        a_hps_mat = a_hps_mat(top_inds,:);
        logL_mat = logL_mat(top_inds,:);

        [logl_noise_sds(d_ind), a_logl_input_scales, logl_output_scales(d_ind)] = ...
            hp_heuristics(a_hps_mat,logL_mat,10);

        logl_input_scales(input_ind_vector(d_ind)) = a_logl_input_scales(1);
    end
    
    % now we estimate the scale of variation of the likelihood wrt
    % log other and log noise sd.
    a_hps_mat = cat(1,other_cell{:});
    logL_mat = cat(1,other_logL_cell{:});
    
    a_hps_mat = max(a_hps_mat, -100);
    logL_mat = max(logL_mat,-1e100);
    sorted_logL_mat = sort(logL_mat);

    top_inds = logL_mat >= sorted_logL_mat(round(0.9*end));
    a_hps_mat = a_hps_mat(top_inds,:);
    logL_mat = logL_mat(top_inds,:);

    [logl_noise_sds(end), a_logl_input_scales, logl_output_scales(end)] = ...
        hp_heuristics(a_hps_mat,logL_mat,10);

    logl_input_scales(other_active_inds) = a_logl_input_scales;
    
    logl_noise_sd = min(logl_noise_sds);
    logl_output_scale = max(logl_output_scales);
    
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


logl_gp.noise_sd = (logl_noise_sd);
logl_gp.input_scales = (logl_input_scales); 
logl_gp.output_scale = (logl_output_scale);

if opt.print
    fprintf('Completed retraining of GP in %g seconds\n', cputime-start_time)
    fprintf('\n');
end
       

function [hypersample, a_hps_mat, logL_mat] = move_hypersample(...
    hypersample, gp, logl_input_scales, active_hp_inds, X_data, y_data, opt)

gp.hypersamples = hypersample;
a_logl_input_scales = logl_input_scales(active_hp_inds);

flag = false;
i = 0;
a_hps_mat = nan(opt.maxevals_hs,length(active_hp_inds));
logL_mat = nan(opt.maxevals_hs,1);

if opt.verbose && opt.plots
    for a = 1:length(active_hp_inds)
        figure(a);clf; hold on;
        title(['Optimising hyperperparameter ',num2str(a)])
    end
end

broken = false;

while (~flag || ceil(opt.maxevals_hs/5) > i) && i < opt.maxevals_hs-1
    i = i+1;
    
    try
        gp = ...
            revise_gp(X_data, y_data, gp, 'overwrite', [], 'all', active_hp_inds);
    catch
        broken = true;
        i = i - 1;
        break;
    end
    
    logL = gp.hypersamples.logL;
    a_hs=gp.hypersamples.hyperparameters(active_hp_inds);
    
    a_hps_mat(i,:) = a_hs;
    logL_mat(i) = logL;
    
    if opt.verbose && opt.plots
        for a = 1:length(active_hp_inds)
            figure(a)
            x = a_hs(a);
            plot(x, logL, '.');

            g = [gp.hypersamples.glogL{a}];
            scale = a_logl_input_scales(a);

            line([x-scale,x+scale],...
                [logL-g*scale,logL+g*scale],...
                'Color',[0 0 0],'LineWidth',1.5);
        end
    end
    
    if i>1 && logL_mat(i) < backup_logL
        % the input scale which predicted the largest increase in logL is
        % likely wrong
        
        dist_moved = (a_hs - backup_a_hs).*a_grad_logL;
        [dummy,max_ind] = max(dist_moved);

        a_logl_input_scales(max_ind) = 0.5*a_logl_input_scales(max_ind);
        
%         [~,a_logl_input_scales] = ...
%             hp_heuristics(a_hps_mat(1:i,:),logL_mat(1:i,:),10);
%         
        a_hs = backup_a_hs;
    else
        backup_logL = logL;
        backup_a_hs = a_hs;
        a_grad_logL = [gp.hypersamples.glogL{active_hp_inds}];
    end
    

    [a_hs, flag] = simple_zoom_pt(a_hs, a_grad_logL, ...
                            a_logl_input_scales, 'maximise');
    gp.hypersamples.hyperparameters(active_hp_inds) = a_hs;
    
end

if ~broken
    try

    
    gp = revise_gp(X_data, y_data, gp, 'overwrite');
    logL = gp.hypersamples.logL;
    a_hs = gp.hypersamples.hyperparameters(active_hp_inds);

    i = i+1;
    
    a_hps_mat(i,:) = a_hs;
    logL_mat(i) = logL;
    catch
    end
end

a_hps_mat = a_hps_mat(1:i,:);
logL_mat = logL_mat(1:i,:);

[max_logL,max_ind] = max(logL_mat);
gp.hypersamples.hyperparameters(active_hp_inds) = a_hps_mat(max_ind,:);
gp = revise_gp(X_data, y_data, gp, 'overwrite');
hypersample = gp.hypersamples;

% not_nan = all(~isnan([a_hps_mat,logL_mat]),2);
% 
% [logl_noise_sd, a_logl_input_scales] = ...
%     hp_heuristics(a_hps_mat(not_nan,:), logL_mat(not_nan), 10);
% logl_input_scales(active_hp_inds) = a_logl_input_scales;

if opt.verbose
fprintf('LogL: %g -> %g',logL_mat(1), max_logL)
end
if opt.verbose && opt.plots
    %keyboard;
end

% hp = 4;
% log_ins = linspace(-5,10, 1000);
% logLs = nan(1000,1);
% gp.hypersamples = hypersample;
% for i =1:1000;
%     gp.hypersamples(1).hyperparameters(hp) = log_ins(i);
%     gp = revise_gp(X_data, y_data, gp, 'overwrite', [], 'all', hp);
%     logLs(i) = gp.hypersamples(1).logL;
%     dlogLs(i) = gp.hypersamples(1).glogL{hp};
% end
% clf
% hold on
% plot(log_ins, (logLs),'r')
% plot(log_ins, (dlogLs),'k')

    