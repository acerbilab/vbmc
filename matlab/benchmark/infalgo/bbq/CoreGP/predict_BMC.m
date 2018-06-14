function [mean_out, sd_out] = ...
    predict_BMC(X_star, gp, r_gp, qd_gp, qdd_gp, opt)
% function [mean, sd] = predict_BMC(X_star, gp, r_gp, qd_gp, qdd_gp, opt)
% return the posterior mean and sd by marginalising hyperparameters using
% BMC as per e.g. Rasmussen & Ghahramani `Bayesian Monte Carlo'.
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

allowed_cond_error = 10^-14;

if nargin<6
    opt = struct();
end
% not fully optimised, further operations could be avoided if only the mean
% is required
want_sds = nargout > 1; 

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
        % output argument will be posterior
       
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
    log_r_s = vertcat(gp.hypersamples.logL);
    
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

prior_sds_stack = reshape(prior_sds, 1, 1, num_hps);
prior_var_stack = prior_sds_stack.^2;

  
log_r_s = log_r_s - max(log_r_s);
r_s = exp(log_r_s);

qdr_s = bsxfun(@times, qd_s, r_s);
qddr_s = bsxfun(@times, qdd_s, r_s);


if nargin<3 || isempty(r_gp)
    [r_noise_sd, r_input_scales, r_output_scale] = ...
        hp_heuristics(hs_s, r_s, 10);

    sqd_output_scale = r_output_scale^2;
    r_input_scales = 10*r_input_scales;
    r_mean = mean(r_s,1);
else
    sqd_output_scale = r_gp.quad_output_scale^2;
    r_noise_sd =  r_gp.quad_noise_sd;
    r_input_scales = r_gp.quad_input_scales;
    r_mean = r_gp.quad_mean;
end
if nargin<4 || isempty(qd_gp)
    qd_mean = mean(qd_s,1);
else
    qd_mean = qd_gp.quad_mean;
end
if nargin<5 || isempty(qdd_gp)
    qdd_mean = mean(qdd_s,1);
else
    qdd_mean = qdd_gp.quad_mean;
end

% we force GPs for r, qd, qdd, tr, and tqdd to share the same input scales.
% eps_rr, eps_qdr, eps_rqdd, eps_qddr are assumed to all have input scales
% equal to half of those for r.

input_scales = r_input_scales;

sqd_lambda = sqd_output_scale* ...
    prod(2*pi*input_scales.^2)^(-0.5);
r_noise_sd = r_noise_sd / sqrt(sqd_lambda);

sqd_dist_stack_s = bsxfun(@minus,...
                    reshape(hs_s,num_s,1,num_hps),...
                    reshape(hs_s,1,num_s,num_hps))...
                    .^2;  

sqd_input_scales_stack = reshape(input_scales.^2,1,1,num_hps);

mu_r = r_mean;
rmm_s = r_s - mu_r;

mu_qdr = r_mean.*qd_mean;
qdrmm_s = bsxfun(@minus, qdr_s, mu_qdr);

mu_qddr = r_mean.*qdd_mean;
qddrmm_s = bsxfun(@minus, qddr_s, mu_qddr); 
                
K_s = sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s, sqd_input_scales_stack), 3)); 
[K_rs,jitters_r_s] = improve_covariance_conditioning(K_s, rmm_s, allowed_cond_error);
R_rs = chol(K_rs);
[K_qdrs,jitters_qdr_s] = improve_covariance_conditioning(K_s, ...
    mean(abs(qdrmm_s),2), allowed_cond_error);
R_qdrs = chol(K_qdrs);
K_qddrs = improve_covariance_conditioning(K_s, ...
    mean(abs(qddrmm_s),2), allowed_cond_error);
R_qddrs = chol(K_qddrs);

% R_rs = cholproj(K_s);
% R_qdrs = R_rs;
% R_qddrs = R_rs;
   
sum_prior_var_sqd_input_scales_stack = ...
    prior_var_stack + sqd_input_scales_stack;
    
hs_s_minus_mean_stack = reshape(bsxfun(@minus, hs_s, prior_means'),...
                    num_s, 1, num_hps);

yot_s = sqd_output_scale * ...
    prod(2*pi*sum_prior_var_sqd_input_scales_stack)^(-0.5) * ...
    exp(-0.5 * ...
    sum(bsxfun(@rdivide, hs_s_minus_mean_stack.^2, ...
    sum_prior_var_sqd_input_scales_stack),3));

yot_inv_K_qdrs = solve_chol(R_qdrs, yot_s)';
yot_inv_K_qddrs = solve_chol(R_qddrs, yot_s)';
yot_inv_K_rs = solve_chol(R_rs, yot_s)';



mean_out = (yot_inv_K_qdrs * qdrmm_s + mu_qdr)/ (yot_inv_K_rs * rmm_s + mu_r);
second_moment = (yot_inv_K_qddrs * qddrmm_s + mu_qddr) / (yot_inv_K_rs * rmm_s + mu_r);
sd_out = sqrt(second_moment - mean_out.^2);