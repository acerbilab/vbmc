function [mean_out, sd_out] = predict_ML(X_star, gp, opt)
% function [mean, sd] = predict_ML(X_star, gp, r_gp, opt)
% return the posterior mean and sd by marginalising hyperparameters using
% maximum likelihood. 
% - X_star (n by d) is a matrix of the n (d-dimensional) points at which
% predictions are to be made
% - gp requires fields:
% * hyperparams(i).priorMean
% * hyperparams(i).priorSD
% * hypersamples.logL
% * hypersamples (if opt.prediction_model is gp or spgp)
% * hypersamples.hyperparameters (if using a handle for
% opt.prediction_model)
% - (optional) r_gp requires fields
% * quad_output_scale
% * quad_noise_sd
% * quad_input_scales
% alternatively: 
% [mean, sd] = predict(sample_struct, prior_struct, r_gp, opt)
% - sample_struct requires fields
% * samples
% * log_r
% and
% * mean_y
% * var_y
% or
% * qd
% * qdd
% or
% * q (if a posterior is required; returned in mean_out)
% - prior_struct requires fields
% * means
% * sds

if nargin<3
    opt.prediction_model = 'gp';
end

if isstruct(X_star)
    sample_struct = X_star;
    prior_struct = gp;
    
    hs_s = sample_struct.samples;
    log_r_s = sample_struct.log_r;
    
    [num_s, num_hps] = size(hs_s);
    if isfield(sample_struct, 'mean_y')
        
        mean_y = sample_struct.mean_y;
        var_y = sample_struct.var_y;
        
        % these quantities need to be num_s by num_star matrices
        if size(mean_y, 1) ~= num_s
            mean_y = mean_y';
        end
        if size(var_y, 1) ~= num_s
            var_y = var_y';
        end

        qd_s = mean_y;
        qdd_s = var_y + mean_y.^2;
        
    elseif isfield(sample_struct, 'qd') 
        
        qd_s = sample_struct.qd;
        if isfield(sample_struct, 'qdd')
            qdd_s = sample_struct.qdd;
        else
            qdd_s = sample_struct.qd;
        end
        
    elseif isfield(sample_struct, 'q')
        % output argument will be 
        want_sds = true;
        want_posterior = true;
        
        qd_s = sample_struct.q;
        qdd_s = sample_struct.q;
        
    end
        
    num_star = size(qd_s, 1);
    
    prior_means = prior_struct.means;
    prior_sds = prior_struct.sds;
    
    opt.prediction_model = 'arbitrary';
    
else
    [num_star] = size(X_star, 1);
    
    hs_s = vertcat(gp.hypersamples.hyperparameters);
    if isfield(gp.hypersamples,'logL')
        log_r_s = vertcat(gp.hypersamples.logL);
    else
        log_r_s = zeros(numel(gp.hypersamples),1);
    end

    
    [num_s, num_hps] = size(hs_s);
    
    prior_means = vertcat(gp.hyperparams.priorMean);
    prior_sds = vertcat(gp.hyperparams.priorSD);
    
    mean_y = nan(num_star, num_s);
    var_y = nan(num_star, num_s);

    if ischar(opt.prediction_model)
        switch opt.prediction_model
            case 'spgp'
                for hs = 1:num_s
                    [mean_y(:, hs), var_y(:, hs)] = ...
                        posterior_spgp(X_star,gp,hs,'var_not_cov');
                end
            case 'gp'
                for hs = 1:num_s
                    [mean_y(:, hs), var_y(:, hs)] = ...
                        posterior_gp(X_star,gp,hs,'var_not_cov');
                end
        end
    elseif isa(opt.prediction_model, 'function_handle')
        for hs = 1:num_s
            sample = gp.hypersamples(hs).hyperparameters;
            [mean_y(:, hs), var_y(:, hs)] = ...
                opt.prediction_model(X_star,sample);
        end
    end
    
    mean_y = mean_y';
    var_y = var_y';

    qd_s = mean_y;
    qdd_s = var_y + mean_y.^2;
    
end

[max_log_r, max_ind] = max(log_r_s);

mean_out = qd_s(max_ind, :);
sd_out = sqrt(qdd_s(max_ind, :) - mean_out.^2);

