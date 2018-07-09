function [xpc_unc, tm_a, tv_a] = expected_uncertainty_evidence...
      (new_sample_location, samples, prior, ...
      l_gp_hypers_SE, tl_gp_hypers_SE, del_gp_hypers_SE, ...
      ev_params, opt)
%   [xpc_unc, tm_a, tv_a] = expected_uncertainty_evidence...
%       (new_sample_location, samples, prior, ...
%       l_gp_hypers, tl_gp_hypers, del_gp_hypers, ...
%       ev_params, opt)
%
% returns the expected variance in the evidence after adding a new sample.
%
% OUTPUTS
% - xpc_unc: the expected variance in the evidence
% - tm_a: the mean for the transformed likelihood at the new sample
%       location
% - tv_a: the variance for the transformed likelihood at the new sample
%       location
%
% INPUTS
% - new_sample_location: (1 by the number of dimensions) is a row vector
%       expressing the new sample location.
% - samples: requires fields
%   * locations
%   * log_l
% - prior requires fields
%   * mean
%   * covariance
% - l_gp_hypers_SE: hypers for sqd exp covariance over l, with fields
%   * log_output_scale
%   * log_input_scales
% - tl_gp_hypers_SE: hypers for sqd exp covariance over tl, with fields
%   * log_output_scale
%   * log_input_scales
% - del_gp_hypers_SE: hypers for sqd exp covariance over del, with fields
%   * log_output_scale
%   * log_input_scales
% - ev_params: (see log_evidence.m) has fields
%   * candidate_locations
%   * sqd_dist_stack_s
%   * R_l_s
%   * K_l_s
%   * R_tl_s
%   * K_tl_s
%   * inv_K_tl_s
%   * jitters_tl_s
%   * R_del_sc
%   * K_del_sc
%   * ups_l_s
%   * ups_del_sc
%   * Ups_sc_s
%   * del_inv_K
%   * delta_tl_sc
%   * minty_del
%   * log_mean_second_moment
%   * Dtheta_K_tl_s (if opt.marginalise_scales)
%   * V_theta (if opt.marginalise_scales)


                
% Load options: for now this has been disabled for speed
% ======================================================

if nargin<5
    opt = struct('sds_tl_log_input_scales', false);
end

opt = struct(...
                    'allowed_cond_error',10^-14, ... % allowed conditioning error
                    'delta_update', false, ... % update for the influence of the new observation at x_a on delta.
                    'marginalise_scales', opt.marginalise_scales, ...
                    'gamma', opt.gamma ... % log_transform scaling factor.
                    );           
%opt = set_defaults( opt, default_opt );


% Load likelihood samples and their locations
% ======================================================

l_s = samples.scaled_l;

x_s = samples.locations;
x_c = ev_params.candidate_locations;
x_a = new_sample_location;
x_sc = [x_s; x_c];
x_sca = [x_sc; x_a];
x_sa = [x_s; x_a];

[num_s, D] = size(x_s);
num_sc = size(x_sc, 1);
num_sca = num_sc + 1;
num_sa = num_s + 1;
range_sa = [1:num_s,num_sca];

% Perform prediction for the transformed likelihood at the new sample
% location
% ======================================================

% input hyperparameters are for a sqd exp covariance, whereas in all that
% follows we use a gaussian covariance. We correct the output scales
% appropriately.
l_gp_hypers = sqdexp2gaussian(l_gp_hypers_SE);
tl_gp_hypers = sqdexp2gaussian(tl_gp_hypers_SE);
del_gp_hypers = sqdexp2gaussian(del_gp_hypers_SE);

% load existing covariance matrix and its cholesky factor
R_tl_s = ev_params.R_tl_s;
K_tl_s = ev_params.K_tl_s;
inv_K_tl_s = ev_params.inv_K_tl_s;

% compute covariance between existing samples and the new location
sqd_dist_stack_s_a = sqd_dist_stack(x_s, x_a);  
K_tl_s_a = gaussian_mat(sqd_dist_stack_s_a, tl_gp_hypers); 

% remove the jitter associated with the closest datum to to x_a -- only
% required if we start using large amounts of jitter
if any(ev_params.jitters_tl_s./diag(K_tl_s) > 1e-4)
    [K_tl_s, R_tl_s] = ...
        jitter_correction(ev_params.jitters_tl_s, K_tl_s_a', K_tl_s, R_tl_s);
end

% compute predictive mean for transformed likelihood, given zero prior mean
tm_a = K_tl_s_a' * inv_K_tl_s;

% options for linsolve
lowr.UT = true;
lowr.TRANSA = true;

% compute predictive variance for transformed likelihood, given zero prior
% mean
inv_R_K_tl_s_a = linsolve(R_tl_s, K_tl_s_a, lowr);    
tv_a = gaussian_mat(0, tl_gp_hypers) - sum(inv_R_K_tl_s_a.^2);
if tv_a < 0
    tv_a = eps;
end

if opt.marginalise_scales
    % we correct for the impact of learning this new hyperparameter sample,
    % r_a, on our belief about the log input scales
    
    % Dtheta_K_tl_a_s is the gradient of the tl Gaussian covariance over
    % the transformed likelihood between x_a and x_s: each plate in the
    % stack is the derivative with respect to a different log input scale
    Dtheta_K_tl_a_s = tr(d_log_scales_gaussian...
                        (K_tl_s_a, sqd_dist_stack_s_a, tl_gp_hypers));
    
    uppr.UT = true;
    K_inv_K_tl_a_s = linsolve(R_tl_s, inv_R_K_tl_s_a, uppr)'; 
    
    % gradient of the mean of the log-likelihood at the added point wrt the
    % log-input scales
    Dtheta_tm_a = prod3(Dtheta_K_tl_a_s, inv_K_tl_s) ...
                    - prod3(K_inv_K_tl_a_s, ...
                        prod3(ev_params.Dtheta_K_tl_s, inv_K_tl_s));
        
    % Now perform the correction to our predictive variance
    tv_a = tv_a + sum(reshape(Dtheta_tm_a.^2, D, 1 , 1) .* ...
        ev_params.V_theta);
end


% Compute the new row & column of our covariance matrices, and their
% cholesky factors
% ======================================================

% squared distances are expensive to compute, so we store them for use in
% the functions below, rather than having each funciton compute them
% afresh.
sqd_dist_stack_sca_a = sqd_dist_stack(x_sca, new_sample_location);
sqd_dist_stack_sa_a = sqd_dist_stack(x_sa, new_sample_location);

% we update the covariance matrix over the likelihood. 
K_l_sa = update_gaussian_mat(ev_params.K_l_s, sqd_dist_stack_sa_a, ...
    l_gp_hypers);
% this importances vector is to force the jitter to be applied solely to
% the added point x_a. improve_covariance_conditioning will automatically
% do this so long as K_l_sa has nans in the appropriate off-diagonal
% elements, but not if K_l_sa is 2x2, so that there are no off-diagonal
% elements.
importances = [inf(num_s,1);0];
K_l_sa = improve_covariance_conditioning(K_l_sa, ...
    importances, opt.allowed_cond_error);
R_l_sa = updatechol(K_l_sa, ev_params.R_l_s, num_sa);


if opt.delta_update
    
    % we update the covariance matrix over delta. 
    K_del_sca = update_gaussian_mat(ev_params.K_del_sc, ...
        sqd_dist_stack_sca_a, del_gp_hypers);
    importances = [inf(num_sc,1);0];
    K_del_sca = improve_covariance_conditioning(K_del_sca, ...
        importances, opt.allowed_cond_error);
    R_del_sca = updatechol(K_del_sca, ev_params.R_del_sc, num_sca); 

    % we add noise to delta to account for the fact that it will change
    % following the addition of a new observation. delta will be unchanged
    % at zero at x_s, and will change at x_c more for locations close to
    % x_a. Of course, we also know with certainty that delta at x_a will
    % be zero.
    del_noise = K_del_sca(:,num_sca);
    del_noise(1:num_s) = 0;
    del_noise(num_sca) = 0;
    R_del_sca = perturbchol(R_del_sca, del_noise);
end

% Now we compute the new rows and columns of Ups and ups matrices required
% to evaluate the sqd mean evidence after adding this new trial sample
% ======================================================

% squared distances are expensive to compute, so we store them for use in
% the functions below, rather than having each funciton compute them
% afresh.
sqd_x_sub_mu_stack_sca = sqd_dist_stack(x_sca, prior.mean);
sqd_x_sub_mu_stack_a = sqd_dist_stack(x_a, prior.mean);

% update ups for the likelihood, where ups is defined as
% ups_s = int K(x, x_s)  prior(x) dx

ups_l_a = small_ups_vec(sqd_x_sub_mu_stack_a, l_gp_hypers, prior);
ups_l_sa = [ev_params.ups_l_s; ups_l_a];
ups_inv_K_l_sa = solve_chol(R_l_sa, ups_l_sa)';

% update Ups for delta & the likelihood, where Ups is defined as
% Ups_s_s' = int K(x_s, x) K(x, x_s') prior(x) dx

Ups_sca_a = big_ups_mat...
    (sqd_x_sub_mu_stack_sca, sqd_x_sub_mu_stack_a, ...
    sqd_dist_stack_sca_a, ...
    del_gp_hypers, l_gp_hypers, prior);

% ... continued below, depending on what we want to do with delta

% do some further updating if we wish to consider updates to delta, and
% then update the mean of our integral over delta, minty_del

if opt.delta_update
    % update for the influence of the new observation at x_a on delta.

    % update ups for delta, where ups is defined as
    % ups_s = int K(x, x_s)  prior(x) dx
    
    ups_del_a = small_ups_vec(sqd_x_sub_mu_stack_a, del_gp_hypers, prior);
    ups_del_sca = [ev_params.ups_del_sc; ups_del_a];
    ups_inv_K_del_sca = solve_chol(R_del_sca, ups_del_sca)';  
    
    % update Ups for delta & the likelihood
    
    Ups_sca_sa = [ev_params.Ups_sc_s, Ups_sca_a(1:num_sc,:);
                    Ups_sca_a(range_sa,:)'];
                
    % delta will certainly be zero at x_a
    delta_tl_sca = [ev_params.delta_tl_sc;0];
    
    del_inv_K = solve_chol(R_del_sca, delta_tl_sca)';
    del_inv_K_Ups_inv_K_l_sa = del_inv_K * solve_chol(R_l_sa, Ups_sca_sa')';

    % mean of int delta(x) p(x) dx given delta_tl_sca 
    minty_del = ups_inv_K_del_sca * delta_tl_sca;
else     
    % update Ups for delta & the likelihood
    
    Ups_sc_sa = [ev_params.Ups_sc_s,Ups_sca_a(1:num_sc,:)];
    del_inv_K_Ups_inv_K_l_sa = ...
        ev_params.del_inv_K * solve_chol(R_l_sa, Ups_sc_sa')';
    
    % mean of int delta(x) p(x) dx given delta_tl_sca 
    minty_del = ev_params.minty_del;
end

% Now we finish up by subtracting the expected squared mean from the
% previously computed second moment
% ======================================================

% n_l_sa is the vector of weights, which, when multipled with the
% likelihoods, gives us our mean estimate for the evidence
n_l_sa = del_inv_K_Ups_inv_K_l_sa + ups_inv_K_l_sa;
n_l_a = n_l_sa(num_sa);
n_l_s = n_l_sa(1:num_s) * l_s + opt.gamma * minty_del;

unscaled_xpc_sqd_mean = n_l_s^2 ...
    + 2 * n_l_s * n_l_a * (opt.gamma * exp(tm_a + 0.5*tv_a) - opt.gamma) ...
    + n_l_a^2 * opt.gamma^2 * ...
        (exp(2*tm_a + 2*tv_a) - 2 * exp(tm_a + 0.5*tv_a) + 1);
    
xpc_unc = - unscaled_xpc_sqd_mean;

% the code below is only required if it is desirable to compute the actual
% expected variance, as opposed to a scaled and translated version of it.
% Note that it may introduce numerical issues if exp(2*samples.max_log_l) 
% is numerically zero.

% % expected squared mean evidence: here we have to correct for our scaling
% % of l_s earlier.
% xpc_sqd_mean = exp(2*samples.max_log_l) * unscaled_xpc_sqd_mean;
%     
% % compare against 
% % exp(2*samples.max_log_l) * mean([l_s;opt.gamma*exp(tm_a)- opt.gamma].^2)
% 
% xpc_unc =  exp(ev_params.log_mean_second_moment) - xpc_sqd_mean;
    
