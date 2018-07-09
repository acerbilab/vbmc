function [minimum, minimum_location, X_data, f_data, gp] = ...
  min_zoom(fn, x_0, opt)


% Initialize options.
% ===========================
if nargin<3
    opt = struct();
elseif ~isstruct(opt)
    num_samples = opt;
    opt = struct();
    opt.num_samples = num_samples;
end

D = numel(x_0);

% Set unspecified fields to default values.
default_opt = struct('print', 2, ...
                        'num_samples', 100, ...
                     'train_gp_time', 120, ...
                     'train_gp_num_samples', 5*D, ...
                     'num_retrains', 5, ...
                     'train_gp_print', false ...
                     );
opt = set_defaults( opt, default_opt );

% GP training options.
gp_train_opt.optim_time = opt.train_gp_time;
gp_train_opt.noiseless = true;
gp_train_opt.prior_mean = 0;
gp_train_opt.hp_prior_mean = nan(D+3, 1);
gp_train_opt.hp_prior_sds = nan(D+3, 1);
% we will perform MAP for our input scales, rather than ML, to prevent
% stupidly lagre and small input scales creeping in. The mean taken for our
% prior over these hps errs on the side of very small input scales.
gp_train_opt.hp_prior_mean(2:D+1) = -5*ones(D,1);
gp_train_opt.hp_prior_sds(2:D+1) = 2*ones(D,1);
% print to screen diagnostic information about gp training
gp_train_opt.print = opt.train_gp_print;
gp_train_opt.verbose = opt.train_gp_print;
% plot diagnostic information about gp training
gp_train_opt.plots = false;
gp_train_opt.num_hypersamples = opt.train_gp_num_samples;


% Specify iterations when we will retrain the GP on r.
retrain_inds = intlogspace(ceil(opt.num_samples/10), ...
                                opt.num_samples, ...
                                opt.num_retrains+1);
retrain_inds(end) = inf;

X_data = zeros(0, D);
f_data = zeros(0, 1);


% Start of actual minimisation algorithm
% =======================================

warning('off','revise_gp:small_num_data');

x_next = x_0;

if opt.print == 2
    fprintf('\n_n_\t_x_\t_f_')
end

for i = 1:opt.num_samples

    X_data = [X_data; x_next];
    [f, g] = fn(x_next);
    f_data = [f_data; f];
    
    switch opt.print
        case 0
        case 1
            if rem(i, 20) == 0
                fprintf('\n%g',i);
            else
                fprintf('.');
            end
        case 2
            fprintf('\n%g.\t%g\t%g', i, x_next, f)
    end
    
    if i == opt.num_samples
        break
    end
    
    % Retrain GP
    % ===========================   
    retrain_now = i >= retrain_inds(1);  % If we've passed the next retraining index.
    if i==1  % First iteration.

        % Set up GP without training it, because there's not enough data.
        gp_train_opt.optim_time = 0;
        gp = lw_train_gp('sqdexp', 'constant', [], ...
                                     X_data, f_data, ...
                                     gp_train_opt);     
                                
                                 
        gp_train_opt.optim_time = opt.train_gp_time;
        
    elseif retrain_now
        % Retrain gp.
        gp = lw_train_gp('sqdexp', 'constant', gp, ...
                                     X_data, f_data, ...
                                     gp_train_opt);         
                               
        retrain_inds(1) = [];   % Move to next retraining index. 
    else
        % for hypersamples that haven't been moved, update
        gp = revise_gp(X_data, f_data, ...
                         gp, 'update', i);
                     
    end
    
    % Put the values of the best hyperparameters into dedicated structures.
    gp_hypers = best_hyperparams(gp);
    input_scales = exp(gp_hypers.log_input_scales);
    
    [x_next, local_optimum_flag] = simple_zoom_pt(x_next, g, input_scales, ...
        'minimise');
    
    if local_optimum_flag
        break
    end
    
end

[minimum, min_ind] = min(f_data);
minimum_location = X_data(min_ind, :);