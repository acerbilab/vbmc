function [mean_out, sd_out, unadj_mean_out, unadj_sd_out] = ...
    predict(X_star, gp, r_gp, qd_gp, qdd_gp, opt)
% function [mean, sd] = predict(X_star, gp, r_gp, qd_gp, qdd_gp, opt)
% return the posterior mean and sd by marginalising hyperparameters.
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



if nargin<6
    opt = struct();
end

default_opt = struct('num_c', 400,...
                    'gamma_const', 1, ...
                    'num_box_scales', 5, ...
                    'prediction_model', 'spgp', ...
                    'no_adjustment', false, ...
                    'allowed_r_cond_error',10^-14,...
                    'allowed_q_cond_error',10^-16,...
                    'print', true);
                
names = fieldnames(default_opt);
for i = 1:length(names);
    name = names{i};
    if (~isfield(opt, name))
      opt.(name) = default_opt.(name);
    end
end

% not fully optimised, further operations could be avoided if only the mean
% is required
want_sds = nargout > 1; 
 want_posterior = false;

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
        % output argument will be a posterior, not a posterior mean
        want_sds = true;
        want_posterior = true;
        
        qd_s = sample_struct.q;
        qdd_s = sample_struct.q;
        
    end
        
    num_star = size(qd_s, 2);
    
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


opt.num_c = min(opt.num_c, num_s);
num_c = opt.num_c;

prior_sds_stack = reshape(prior_sds, 1, 1, num_hps);
prior_var_stack = prior_sds_stack.^2;

[max_log_r_s, max_ind] = max(log_r_s);
log_r_s = log_r_s - max_log_r_s;
r_s = exp(log_r_s);




% predict(X_star, gp, r_gp, qd_gp, qdd_gp, opt)

% r is assumed to have zero mean
if nargin<3 || isempty(r_gp)
    [r_noise_sd, r_input_scales, r_output_scale] = ...
        hp_heuristics(hs_s, r_s, 10);

    r_sqd_output_scale = r_output_scale^2;
else
    r_sqd_output_scale = r_gp.quad_output_scale^2;
    r_input_scales = r_gp.quad_input_scales;
end


if nargin<4 || isempty(qd_gp)
    [qd_noise_sd, qd_input_scales, qd_output_scale] = ...
        hp_heuristics(hs_s, qd_s, 10);

    qd_sqd_output_scale = qd_output_scale^2;
    mu_qd = qd_s(max_ind,:);
else
    qd_sqd_output_scale = qd_gp.quad_output_scale^2;
    qd_input_scales = qd_gp.quad_input_scales;
    if isfield(qd_gp, 'quad_mean')
        mu_qd = qd_gp.quad_mean;
    else
        mu_qd = qd_s(max_ind,:);
    end
end

if want_posterior
    qdd_gp = qd_gp;
end
if ~want_posterior && (nargin<5 || isempty(qdd_gp))
    [qdd_noise_sd, qdd_input_scales, qdd_output_scale] = ...
        hp_heuristics(hs_s, qdd_s, 10);

    qdd_sqd_output_scale = qdd_output_scale^2;
    mu_qdd = qdd_s(max_ind,:);
else
    qdd_sqd_output_scale = qdd_gp.quad_output_scale^2;
    qdd_input_scales = qdd_gp.quad_input_scales;
    if isfield(qd_gp, 'quad_mean')
        mu_qdd = qdd_gp.quad_mean;
    else
        mu_qdd = qdd_s(max_ind,:);
    end
end





qdmm_s = bsxfun(@minus, qd_s, mu_qd);
qddmm_s = bsxfun(@minus, qdd_s, mu_qdd);

% we force GPs for r and tr to share hyperparameters; we also assume the
% gps for qdd and tqdd share hyperparameters. eps_rr, eps_qdr, eps_rqdd,
% eps_qddr are assumed to all have input scales equal to half of those for
% r.

min_input_scales = min([r_input_scales;qd_input_scales;qdd_input_scales]);

r_sqd_lambda = r_sqd_output_scale* ...
    prod(2*pi*r_input_scales.^2)^(-0.5);
qd_sqd_lambda = qd_sqd_output_scale* ...
    prod(2*pi*qd_input_scales.^2)^(-0.5);
qdd_sqd_lambda = qdd_sqd_output_scale* ...
    prod(2*pi*qdd_input_scales.^2)^(-0.5);

eps_input_scales = 0.5 * r_input_scales;
eps_sqd_output_scale = r_sqd_output_scale;
eps_sqd_lambda = eps_sqd_output_scale* ...
    prod(2*pi*eps_input_scales.^2)^(-0.5);

lower_bound = min(hs_s) - opt.num_box_scales*min_input_scales;
upper_bound = max(hs_s) + opt.num_box_scales*min_input_scales;

% find the candidate points, far removed from existing samples
try
hs_c = find_farthest(hs_s, [lower_bound; upper_bound], num_c, ...
                            min_input_scales);
catch
    warning('find_farthest failed')
    hs_c = far_pts(hs_s, [lower_bound; upper_bound], num_c);
end
    

hs_sc = [hs_s; hs_c];
num_sc = size(hs_sc, 1);
num_c = num_sc - num_s;

sqd_dist_stack_sc = bsxfun(@minus,...
                    reshape(hs_sc,num_sc,1,num_hps),...
                    reshape(hs_sc,1,num_sc,num_hps))...
                    .^2;  
sqd_dist_stack_s = sqd_dist_stack_sc(1:num_s, 1:num_s, :);

sqd_r_input_scales_stack = reshape(r_input_scales.^2,1,1,num_hps);
sqd_qd_input_scales_stack = reshape(qd_input_scales.^2,1,1,num_hps);
sqd_qdd_input_scales_stack = reshape(qdd_input_scales.^2,1,1,num_hps);
sqd_eps_input_scales_stack = reshape(eps_input_scales.^2,1,1,num_hps);
                
K_r_s = r_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s, sqd_r_input_scales_stack), 3)); 
[K_r_s, jitters_r_s] = improve_covariance_conditioning(K_r_s, ...
    r_s.*mean(abs(qdmm_s),2), ...
    opt.allowed_r_cond_error);
R_r_s = chol(K_r_s);

K_qd_s = qd_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s, sqd_qd_input_scales_stack), 3)); 
[K_qd_s, jitters_qd_s] = improve_covariance_conditioning(K_qd_s, ...
    r_s.*mean(abs(qdmm_s),2), ...
    opt.allowed_q_cond_error);
R_qd_s = chol(K_qd_s);

K_qdd_s = qdd_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s, sqd_qdd_input_scales_stack), 3)); 
K_qdd_s = improve_covariance_conditioning(K_qdd_s, ...
    r_s.*mean(abs(qddmm_s),2), ...
    opt.allowed_q_cond_error);
R_qdd_s = chol(K_qdd_s);

K_eps = eps_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_sc, sqd_eps_input_scales_stack), 3)); 
importance_sc = ones(num_sc,1);
importance_sc(num_s + 1 : end) = 2;
K_eps = improve_covariance_conditioning(K_eps, importance_sc, ...
    opt.allowed_r_cond_error);
R_eps = chol(K_eps);     

sqd_dist_stack_s_sc = sqd_dist_stack_sc(1:num_s, :, :);

K_r_s_sc = r_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s_sc, sqd_r_input_scales_stack), 3));  
K_qd_s_sc = qd_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s_sc, sqd_qd_input_scales_stack), 3)); 
K_qdd_s_sc = qdd_sqd_lambda * exp(-0.5*sum(bsxfun(@rdivide, ...
                    sqd_dist_stack_s_sc, sqd_qdd_input_scales_stack), 3)); 
       
sum_prior_var_sqd_input_scales_stack_r = ...
    prior_var_stack + sqd_r_input_scales_stack;
sum_prior_var_sqd_input_scales_stack_eps = ...
    prior_var_stack + sqd_eps_input_scales_stack;

opposite_eps = sqd_eps_input_scales_stack;
opposite_r = sqd_r_input_scales_stack;
opposite_qd = sqd_qd_input_scales_stack;
opposite_qdd = sqd_qdd_input_scales_stack;
    
hs_sc_minus_mean_stack = reshape(bsxfun(@minus, hs_sc, prior_means'),...
                    num_sc, 1, num_hps);
sqd_hs_sc_minus_mean_stack = ...
    repmat(hs_sc_minus_mean_stack.^2, 1, num_sc, 1);
tr_sqd_hs_sc_minus_mean_stack = tr(sqd_hs_sc_minus_mean_stack);

yot_r = r_sqd_output_scale * ...
    prod(2*pi*sum_prior_var_sqd_input_scales_stack_r)^(-0.5) * ...
    exp(-0.5 * ...
    sum(bsxfun(@rdivide, hs_sc_minus_mean_stack(1:num_s, :, :).^2, ...
    sum_prior_var_sqd_input_scales_stack_r),3));
yot_eps = eps_sqd_output_scale * ...
    prod(2*pi*sum_prior_var_sqd_input_scales_stack_eps)^(-0.5) * ...
    exp(-0.5 * ...
    sum(bsxfun(@rdivide, hs_sc_minus_mean_stack.^2, ...
    sum_prior_var_sqd_input_scales_stack_eps),3));

yot_inv_K_r = solve_chol(R_r_s, yot_r)';
yot_inv_K_eps = solve_chol(R_eps, yot_eps)';

prior_var_times_sqd_dist_stack_sc = bsxfun(@times, prior_var_stack, ...
                    sqd_dist_stack_sc);
                
% 2 pi is outside of sqrt because each element of determ is actually the
% determinant of a 2 x 2 matrix

inv_determ_qd_r = (prior_var_stack.*(...
        sqd_r_input_scales_stack + sqd_qd_input_scales_stack) + ...
        sqd_r_input_scales_stack.*sqd_qd_input_scales_stack).^(-1);
Yot_qd_r = qd_sqd_output_scale * r_sqd_output_scale * ...
    prod(1/(2*pi) * sqrt(inv_determ_qd_r)) .* ...
    exp(-0.5 * sum(bsxfun(@times,inv_determ_qd_r,...
                bsxfun(@times, opposite_r, ...
                    sqd_hs_sc_minus_mean_stack(1:num_s, 1:num_s, :)) ...
                + bsxfun(@times, opposite_qd, ...
                    tr_sqd_hs_sc_minus_mean_stack(1:num_s, 1:num_s, :)) ...
                + prior_var_times_sqd_dist_stack_sc(1:num_s, 1:num_s, :)...
                ),3));
            
% some code to test that this construction works          
% Lambda = diag(prior_sds.^2);
% W_qd = diag(qd_input_scales.^2);
% W_r = diag(r_input_scales.^2);
% mat = kron(ones(2),Lambda)+blkdiag(W_qd,W_r);
% 
% Yot_qd_r_test = @(i,j) qd_sqd_output_scale * r_sqd_output_scale *...
%     mvnpdf([hs_s(i,:)';hs_s(j,:)'],[prior_means';prior_means'],mat);
            
inv_determ_qdd_r = (prior_var_stack.*(...
        sqd_r_input_scales_stack + sqd_qdd_input_scales_stack) + ...
        sqd_r_input_scales_stack.*sqd_qdd_input_scales_stack).^(-1);
Yot_qdd_r = qdd_sqd_output_scale * r_sqd_output_scale * ...
    prod(1/(2*pi) * sqrt(inv_determ_qdd_r)) .* ...
    exp(-0.5 * sum(bsxfun(@times,inv_determ_qdd_r,...
                bsxfun(@times, opposite_r, ...
                    sqd_hs_sc_minus_mean_stack(1:num_s, 1:num_s, :)) ...
                + bsxfun(@times, opposite_qdd, ...
                    tr_sqd_hs_sc_minus_mean_stack(1:num_s, 1:num_s, :)) ...
                + prior_var_times_sqd_dist_stack_sc(1:num_s, 1:num_s, :)...
                ),3));
            
inv_determ_qd_eps = (prior_var_stack.*(...
        sqd_eps_input_scales_stack + sqd_qd_input_scales_stack) + ...
        sqd_eps_input_scales_stack.*sqd_qd_input_scales_stack).^(-1);
Yot_qd_eps = qd_sqd_output_scale * eps_sqd_output_scale * ...
    prod(1/(2*pi) * sqrt(inv_determ_qd_eps)) .* ...
    exp(-0.5 * sum(bsxfun(@times,inv_determ_qd_eps,...
                bsxfun(@times, opposite_eps, ...
                    sqd_hs_sc_minus_mean_stack(1:num_s, :, :)) ...
                + bsxfun(@times, opposite_qd, ...
                    tr_sqd_hs_sc_minus_mean_stack(1:num_s, :, :)) ...
                + prior_var_times_sqd_dist_stack_sc(1:num_s, :, :)...
                ),3));

inv_determ_qdd_eps = (prior_var_stack.*(...
        sqd_eps_input_scales_stack + sqd_qdd_input_scales_stack) + ...
        sqd_eps_input_scales_stack.*sqd_qdd_input_scales_stack).^(-1);
Yot_qdd_eps = qdd_sqd_output_scale * eps_sqd_output_scale * ...
    prod(1/(2*pi) * sqrt(inv_determ_qdd_eps)) .* ...
    exp(-0.5 * sum(bsxfun(@times,inv_determ_qdd_eps,...
                bsxfun(@times, opposite_eps, ...
                    sqd_hs_sc_minus_mean_stack(1:num_s, :, :)) ...
                + bsxfun(@times, opposite_qdd, ...
                    tr_sqd_hs_sc_minus_mean_stack(1:num_s, :, :)) ...
                + prior_var_times_sqd_dist_stack_sc(1:num_s, :, :)...
                ),3));
          

% As = [hs_sc_minus_mean_stack(3,:);hs_sc_minus_mean_stack(4,:)];
% Bs = 0*[prior_means';prior_means'];
% scalesss = [input_scales;eps_input_scales].^2;
% covmat = kron2d(diag(prior_sds.^2), ones(2)) + diag(scalesss(:));
% sqd_output_scale.^2 * mvnpdf(As(:),Bs(:),covmat);

inv_K_Yot_inv_K_qd_r = solve_chol(R_qd_s, ...
    solve_chol(R_r_s, Yot_qd_r')');
inv_K_Yot_inv_K_qdd_r = solve_chol(R_qdd_s, ...
    solve_chol(R_r_s, Yot_qdd_r')');
inv_K_Yot_inv_K_qd_eps = solve_chol(R_qd_s, ...
    solve_chol(R_eps, Yot_qd_eps')');
inv_K_Yot_inv_K_qdd_eps = solve_chol(R_qdd_s, ...
    solve_chol(R_eps, Yot_qdd_eps')');
          

tilde = @(x, gamma_x) log(bsxfun(@rdivide, x, gamma_x) + 1);
%inv_tilda = @(tx, gamma_x) exp(bsxfun(@plus, tx, log(gamma_x))) - gamma_x;

gamma_r = opt.gamma_const;
tr_s = tilde(r_s, gamma_r);

gamma_qdd = opt.gamma_const*max(eps,max(qdd_s));
tqdd_s = tilde(qdd_s, gamma_qdd);

[dummy,max_ind] = max(r_s);

% IMPORTANT NOTE: THIS NEEDS TO BE CHANGED TO MATCH WHATEVER MEAN IS USED
% FOR qdd
mu_tqdd = tqdd_s(max_ind,:);
tqddmm_s = bsxfun(@minus, tqdd_s, mu_tqdd);


lowr.UT = true;
lowr.TRANSA = true;
uppr.UT = true;

two_thirds_r = linsolve(R_r_s,linsolve(R_r_s, K_r_s_sc, lowr), uppr)';
two_thirds_qd = linsolve(R_qd_s,linsolve(R_qd_s, K_qd_s_sc, lowr), uppr)';
two_thirds_qdd = linsolve(R_qdd_s,linsolve(R_qdd_s, K_qdd_s_sc, lowr), uppr)';

mean_r_sc =  two_thirds_r * r_s;
mean_qd_sc = bsxfun(@plus, mu_qd, two_thirds_qd * qdmm_s);
mean_qdd_sc = bsxfun(@plus, mu_qdd, two_thirds_qdd * qddmm_s);

mean_tr_sc = two_thirds_r * tr_s;
mean_tqdd_sc = bsxfun(@plus, mu_tqdd, two_thirds_qdd * tqddmm_s);

%c_inds = num_sc - (num_c-1:-1:0);

% Delta_tr = zeros(num_sc, 1);
% eps_rr_sc = zeros(num_sc, 1);
% Delta_tqdd = zeros(num_sc, num_star);
% eps_qdr_sc = zeros(num_sc, num_star);
% eps_rqdd_sc = zeros(num_sc, num_star);
% eps_qddr_sc = zeros(num_sc, num_star);

% use a crude thresholding here as our tilde transformation will fail if
% the mean goes below zero
mean_r_sc = max(eps, mean_r_sc);
mean_qdd_sc = max(eps, mean_qdd_sc);

Delta_tr = mean_tr_sc - tilde(mean_r_sc, gamma_r);
Delta_tqdd = mean_tqdd_sc - tilde(mean_qdd_sc, gamma_qdd);

eps_rr_sc = mean_r_sc .* Delta_tr;
eps_qdr_sc = bsxfun(@times, mean_qd_sc, Delta_tr);
eps_rqdd_sc = bsxfun(@times, mean_r_sc, Delta_tqdd);
eps_qddr_sc = bsxfun(@times, mean_qdd_sc, Delta_tr);

minty_r = yot_inv_K_r * r_s;
minty_Delta_tr = yot_inv_K_eps * Delta_tr;
minty_eps_rr = yot_inv_K_eps * eps_rr_sc;
minty_eps_qdr = (yot_inv_K_eps * eps_qdr_sc)';
minty_eps_rqdd = (yot_inv_K_eps * eps_rqdd_sc)';
minty_eps_qddr = (yot_inv_K_eps * eps_qddr_sc)';

% all the quantities below need to be adjusted to account for the non-zero
% prior means of qd and qdd
minty_qd_r = qdmm_s' * inv_K_Yot_inv_K_qd_r * r_s;
rhod = minty_qd_r / minty_r + mu_qd';
minty_qd_eps_rr = qdmm_s' * inv_K_Yot_inv_K_qd_eps * eps_rr_sc + ...
                mu_qd' * minty_eps_rr;

if want_sds               
minty_qdd_r = qddmm_s' * inv_K_Yot_inv_K_qdd_r * r_s;
rhodd = minty_qdd_r / minty_r + mu_qdd';
% only need the diagonals of this quantity, the full covariance is not
% required
minty_qdd_eps_rqdd = ...
    sum((qddmm_s' * inv_K_Yot_inv_K_qdd_eps) .* eps_rqdd_sc', 2) + ...
                mu_qdd' .* minty_eps_rqdd;
minty_qdd_eps_rr = qddmm_s' * inv_K_Yot_inv_K_qdd_eps * eps_rr_sc + ...
                mu_qdd' * minty_eps_rr;
end




adj_rhod_tr = (minty_qd_eps_rr + gamma_r * minty_eps_qdr ...
                -(minty_eps_rr + gamma_r * minty_Delta_tr) * rhod) / minty_r;
if want_sds            
adj_rhodd_tq = (minty_qdd_eps_rqdd + gamma_qdd * minty_eps_rqdd) / minty_r;
adj_rhodd_tr = (minty_qdd_eps_rr + gamma_r * minty_eps_qddr ...
    -(minty_eps_rr + gamma_r * minty_Delta_tr) * rhodd) / minty_r;
end
if opt.no_adjustment
    adj_rhod_tr = 0;
    adj_rhodd_tq = 0;
    adj_rhodd_tr = 0;
end


mean_out = rhod + adj_rhod_tr;
unadj_mean_out = mean_out;
if want_sds
second_moment = rhodd + adj_rhodd_tq + adj_rhodd_tr;
if want_posterior
    mean_out = second_moment;
    sd_out = nan;
else
    var_out = second_moment - mean_out.^2;
    problems = var_out<0;
    var_out(problems) = qdd_s(max_ind,problems) - qd_s(max_ind,problems).^2;
    
    sd_out = sqrt(var_out);
    
    var_out = rhodd - mean_out.^2;
    problems = var_out<0;
    var_out(problems) = qdd_s(max_ind,problems) - qd_s(max_ind,problems).^2;
    
    unadj_sd_out = sqrt(var_out);
end
end

% for i = 1:num_hps
%     figure(i);clf;
%     hold on
%     plot(phi(:,i), qd, '.k');
%     plot(phi(:,i), qdd, '.r');
%     xlabel(['x_',num2str(i)]);
% end
% 
% [qd_noise_sd, qd_input_scales, qd_output_scale] = ...
%         hp_heuristics(phi, qd, 10);
%     
% [qdd_noise_sd, qdd_input_scales, qdd_output_scale] = ...
%         hp_heuristics(phi, qdd, 10);