function [mean_out, sd_out, unadj_mean_out, unadj_sd_out] = ...
    predict_bq(samples, prior, ...
    l_gp_hypers_SE, tl_gp_hypers_SE, del_gp_hypers_SE, ...
    qd_gp_hypers_SE, qdd_gp_hypers_SE, ev_params, opt)
% [mean_out, sd_out, unadj_mean_out, unadj_sd_out] = ...
%    predict_bq(samples, priol_struct, ...
%    l_gp_hypers_SE, tl_gp_hypers_SE, ...
%    qd_gp_hypers_SE, qdd_gp_hypers_SE, opt)
% 
% return the posterior mean and sd by marginalising hyperparameters.
%
% OUTPUTS
% - mean_out
% - sd_out
% - unadj_mean_out: mean without correction for delta
% - unadj_sd_out: sd without correction for delta
%
% INPUTS
% - samples: requires fields
%   - locations
%   - scaled_l: likelihoods divided by maximum likelihood
%   - tl: log-transformed scaled likelihoods
%   - max_log_l: max log likelihood
%   - qd: (alternatively: mean_y, such that mean_y = qd) predictive mean
%   - qdd: (alternatively: var_y, such that var_y = qdd - qd^2) predictive
%       second moment
% - prior requires fields
%   - mean
%   - covariance
% - l_gp_hypers_SE: hypers for sqd exp covariance over l, with fields
%   - log_output_scale
%   - log_input_scales
% - tl_gp_hypers_SE: hypers for sqd exp covariance over tl, with fields
%   - log_output_scale
%   - log_input_scales
% - del_gp_hypers_SE: hypers for sqd exp covariance over del, with fields
%   - log_output_scale
%   - log_input_scales
% - qd_gp_hypers_SE: hypers for sqd exp covariance over l, with fields
%   - log_output_scale
%   - log_input_scales
%   - prior_mean
% - qdd_gp_hypers_SE: hypers for sqd exp covariance over tl, with fields
%   - log_output_scale
%   - log_input_scales
%   - prior_mean
% - ev_params: (see log_evidence.m) has fields
%   - x_c
%   - sqd_dist_stack_s
%   - R_tl_s
%   - K_tl_s
%   - inv_K_tl_s
%   - jitters_l
%   - sqd_dist_stack_s
%   - R_del
%   - K_del
%   - ups_l
%   - ups_del
%   - Ups_sc_s
%   - del_inv_K_del
%   - delta_tl_sc
%   - minty_del
%   - log_mean_second_moment

% Load options, set to default if not available
% ======================================================

if nargin<6
    opt = struct();
end

default_opt = struct(...
                    'gamma', 1, ...
                    'no_adjustment', false, ...
                    'allowed_cond_error',10^-14, ... % allowed conditioning error
                    'print', true);
                
opt = set_defaults( opt, default_opt );


% Load data
% ======================================================
    
x_s = samples.locations;
num_s = size(x_s, 1);

% rescale all log-likelihood values for numerical accuracy; we'll correct
% for this at the end of the function
l_s = samples.scaled_l;


if isfield(samples, 'mean_y')

    mean_y = samples.mean_y;
    var_y = samples.var_y;

    % these quantities need to be num_s by num_star matrices
    if size(mean_y, 1) ~= num_s
        mean_y = mean_y';
    end
    if size(var_y, 1) ~= num_s
        var_y = var_y';
    end

    qd_s = mean_y;
    qdd_s = var_y + mean_y.^2;

elseif isfield(samples, 'qd') 

    qd_s = samples.qd;
    if isfield(samples, 'qdd')
        qdd_s = samples.qdd;
    else
        qdd_s = samples.qd;
    end

end
    
gamma_l = opt.gamma;
gamma_qdd = opt.gamma * max(eps,max(qdd_s));

tqdd_s = log_transform(qd_s, gamma_qdd);

% IMPORTANT NOTE: THIS NEEDS TO BE CHANGED TO MATCH WHATEVER MEAN IS USED
% FOR qdd
[max_log_l, max_ind] = max(samples.log_l);

mu_qd = qd_s(max_ind, :);
mu_qdd = qdd_s(max_ind, :);
mu_tqdd = tqdd_s(max_ind,:);

% rescale by subtracting appropriate prior means
qdmm_s = bsxfun(@minus, qd_s, mu_qd);
qddmm_s = bsxfun(@minus, qdd_s, mu_qdd);
tqddmm_s = bsxfun(@minus, tqdd_s, mu_tqdd);


% Compute our covariance matrices and their cholesky factors
% ======================================================

% input hyperparameters are for a sqd exp covariance, whereas in all that
% follows we use a gaussian covariance. We correct the output scales
% appropriately.
l_gp_hypers = sqdexp2gaussian(l_gp_hypers_SE);
qd_gp_hypers = sqdexp2gaussian(qd_gp_hypers_SE);
qdd_gp_hypers = sqdexp2gaussian(qdd_gp_hypers_SE);
eps_gp_hypers = sqdexp2gaussian(del_gp_hypers_SE);

% we assume the gps for qd and tqd share hyperparameters, as do qdd and
% tqdd. eps_ll, eps_qdl, eps_lqdd, eps_qddl are assumed to all have the
% same hypers as del (the output scales are probably wildly wrong, but are
% not that important as they should cancel out)
tqd_gp_hypers = qd_gp_hypers;
tqdd_gp_hypers = qdd_gp_hypers;

sqd_dist_stack_s = ev_params.sqd_dist_stack_s;
sqd_dist_stack_s_sc = ev_params.sqd_dist_stack_s_sc;

importance_s = samples.scaled_l.*mean(abs(qdmm_s),2);

% The cholesky factor of the Gram matrix over the likelihood
R_l = ev_params.R_l_s;

% The gram matrix over the predictive mean qd and its cholesky factor
K_qd = gaussian_mat(sqd_dist_stack_s, qd_gp_hypers);
K_qd = improve_covariance_conditioning(K_qd, ...
    importance_s, ...
    opt.allowed_cond_error);
R_qd = chol(K_qd);
inv_K_qd_qdmm = solve_chol(R_qd, qdmm_s);
% The covariance over the transformed qdd between x_sc and x_s
K_qd_sc = gaussian_mat(sqd_dist_stack_s_sc, tqd_gp_hypers);

% The gram matrix over the predictive second moment qdd and its cholesky 
% factor
K_qdd = gaussian_mat(sqd_dist_stack_s, qdd_gp_hypers);
K_qdd = improve_covariance_conditioning(K_qdd, ...
    importance_s, ...
    opt.allowed_cond_error);
R_qdd = chol(K_qdd);
inv_K_qdd_qddmm = solve_chol(R_qdd, qddmm_s);
% The covariance over the transformed qdd between x_sc and x_s
K_qdd_sc = gaussian_mat(sqd_dist_stack_s_sc, tqdd_gp_hypers);

% The gram matrix over the transformed predictive second moment qdd and its
% cholesky factor
K_tqdd = gaussian_mat(sqd_dist_stack_s, tqdd_gp_hypers);
K_tqdd = improve_covariance_conditioning(K_tqdd, ...
    importance_s, ...
    opt.allowed_cond_error);
R_tqdd = chol(K_tqdd);
inv_K_tqdd_tqddmm = solve_chol(R_tqdd, tqddmm_s);
% The covariance over the transformed qdd between x_sc and x_s
K_tqdd_sc = gaussian_mat(sqd_dist_stack_s_sc, tqdd_gp_hypers);

% The cholesky factor of the Gram matrix over delta; assumed identical to
% that over various eps quantities
R_eps = ev_params.R_del_sc;
 
% Compute eps quantities
% ======================================================

% the mean of qd at x_sc
mean_qd_sc =  bsxfun(@plus, mu_qd, K_qd_sc' * inv_K_qd_qdmm);

% the mean of qdd at x_sc
mean_qdd_sc =  bsxfun(@plus, mu_qdd, K_qdd_sc' * inv_K_qdd_qddmm);
% use a crude thresholding here as our tilde transformation will fail if
% the mean goes below zero
mean_qdd_sc = max(mean_qdd_sc, eps);

% the mean of the transformed (log) likelihood at x_sc
mean_tqdd_sc = bsxfun(@plus, mu_tqdd, K_tqdd_sc' * inv_K_tqdd_tqddmm);

% the difference between the mean of the transformed (log) likelihood and
% the transform of the mean likelihood
delta_tqdd_sc = mean_tqdd_sc - log_transform(mean_qdd_sc, opt.gamma);

mean_l_sc = ev_params.mean_tl_sc;
delta_tl_sc = ev_params.delta_tl_sc;

% Compute eps quantities, the scaled difference between the mean of the
% transformed (log) quantities and the transform of the mean quantities
eps_ll_sc = mean_l_sc .* delta_tl_sc;
eps_qdl_sc = bsxfun(@times, mean_qd_sc, delta_tl_sc);
eps_lqdd_sc = bsxfun(@times, mean_l_sc, delta_tqdd_sc);
eps_qddl_sc = bsxfun(@times, mean_qdd_sc, delta_tl_sc);

% Compute various Gaussian-derived quantities required to evaluate the mean
% integrals over qd and qdd
% ======================================================

sqd_x_sub_mu_stack_sc = ev_params.sqd_x_sub_mu_stack_sc;
sqd_x_sub_mu_stack_s = sqd_x_sub_mu_stack_sc(1:num_s, :, :);

% calculate ups for eps, where ups is defined as
% ups_s = int K(x, x_s)  prior(x) dx
ups_eps = small_ups_vec(sqd_x_sub_mu_stack_sc, eps_gp_hypers, prior);

% calculate Ups for qd & the likelihood, where Ups is defined as
% Ups_s_s' = int K(x_s, x) K(x, x_s') prior(x) dx
Ups_qd_l = big_ups_mat...
    (sqd_x_sub_mu_stack_s, sqd_x_sub_mu_stack_s, ...
    sqd_dist_stack_s, ...
    qd_gp_hypers, l_gp_hypers, prior);

% calculate Ups for qd & eps, where Ups is defined as
% Ups_s_s' = int K(x_s, x) K(x, x_s') prior(x) dx
Ups_qd_eps = big_ups_mat...
    (sqd_x_sub_mu_stack_s, sqd_x_sub_mu_stack_sc, ...
    sqd_dist_stack_s_sc, ...
    qd_gp_hypers, eps_gp_hypers, prior);

% calculate Ups for qdd & the likelihood, where Ups is defined as
% Ups_s_s' = int K(x_s, x) K(x, x_s') prior(x) dx
Ups_qdd_l = big_ups_mat...
    (sqd_x_sub_mu_stack_s, sqd_x_sub_mu_stack_s, ...
    sqd_dist_stack_s, ...
    qdd_gp_hypers, l_gp_hypers, prior);

% calculate Ups for qdd & eps, where Ups is defined as
% Ups_s_s' = int K(x_s, x) K(x, x_s') prior(x) dx
Ups_qdd_eps = big_ups_mat...
    (sqd_x_sub_mu_stack_s, sqd_x_sub_mu_stack_sc, ...
    sqd_dist_stack_s_sc, ...
    qdd_gp_hypers, eps_gp_hypers, prior);


ups_inv_K_eps = solve_chol(R_eps, ups_eps)';

inv_K_Ups_inv_K_qd_l = solve_chol(R_qd, solve_chol(R_l, Ups_qd_l')');
inv_K_Ups_inv_K_qd_eps = solve_chol(R_qd, solve_chol(R_eps, Ups_qd_eps')');
inv_K_Ups_inv_K_qdd_l = solve_chol(R_qdd, solve_chol(R_l, Ups_qdd_l')');
inv_K_Ups_inv_K_qdd_eps = solve_chol(R_qdd, solve_chol(R_eps, Ups_qdd_eps')');
    
% Compute various integrals over qd and qdd
% ======================================================

% integrals required to compute uncorrected estimates
minty_l = ev_params.minty_l;
minty_qd_l = qdmm_s' * inv_K_Ups_inv_K_qd_l * l_s;
minty_qdd_l = qddmm_s' * inv_K_Ups_inv_K_qdd_l * l_s;

% all the quantities below have been adjusted to account for the non-zero
% prior means of qd and qdd

% uncorrected estimates for the integrals over qd & qdd
rhod = minty_qd_l / minty_l + mu_qd';
rhodd = minty_qdd_l / minty_l + mu_qdd';

% integrals required for adjustment factors
minty_delta_tl = ups_inv_K_eps * delta_tl_sc;
minty_eps_ll = ups_inv_K_eps * eps_ll_sc;
minty_eps_qdl = (ups_inv_K_eps * eps_qdl_sc)';
minty_eps_lqdd = (ups_inv_K_eps * eps_lqdd_sc)';
minty_eps_qddl = (ups_inv_K_eps * eps_qddl_sc)';

minty_qd_eps_ll = qdmm_s' * inv_K_Ups_inv_K_qd_eps * eps_ll_sc + ...
                mu_qd' * minty_eps_ll;
minty_qdd_eps_ll = qddmm_s' * inv_K_Ups_inv_K_qdd_eps * eps_ll_sc + ...
                mu_qdd' * minty_eps_ll;
% only need the diagonals of this quantity, the full covariance is not
% required
minty_qdd_eps_lqdd = ...
    sum((qddmm_s' * inv_K_Ups_inv_K_qdd_eps) .* eps_lqdd_sc', 2) + ...
                mu_qdd' .* minty_eps_lqdd;

% adjustment factors
adj_rhod_tl = (minty_qd_eps_ll + gamma_l * minty_eps_qdl ...
                -(minty_eps_ll + gamma_l * minty_delta_tl) * rhod) / minty_l;         

            adj_rhodd_tq = (minty_qdd_eps_lqdd + gamma_qdd * minty_eps_lqdd) / minty_l;
adj_rhodd_tl = (minty_qdd_eps_ll + gamma_l * minty_eps_qddl ...
    -(minty_eps_ll + gamma_l * minty_delta_tl) * rhodd) / minty_l;

if opt.no_adjustment
    adj_rhod_tl = 0;
    adj_rhodd_tq = 0;
    adj_rhodd_tl = 0;
end

% final adjusted quantities
mean_out = rhod + adj_rhod_tl;
unadj_mean_out = mean_out;

second_moment = rhodd + adj_rhodd_tq + adj_rhodd_tl;

var_out = second_moment - mean_out.^2;
problems = var_out<0;
var_out(problems) = qdd_s(max_ind,problems) - qd_s(max_ind,problems).^2;

sd_out = sqrt(var_out);

var_out = rhodd - mean_out.^2;
problems = var_out<0;
var_out(problems) = qdd_s(max_ind,problems) - qd_s(max_ind,problems).^2;

unadj_sd_out = sqrt(var_out);
