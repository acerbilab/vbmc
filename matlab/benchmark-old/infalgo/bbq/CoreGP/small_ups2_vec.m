function vec = small_ups2_vec(sqd_dist_stack_Bmu, gp_A_hypers, gp_B_hypers, prior)
% Returns the column vector
% ups2_s = int int K_A(x, x') K_B(x', x_s) prior(x) prior(x') dx dx'
% = N(0, 0, (W_A + 2*L)) * ...
%   N(x_s, mu, W_B + L - L*(W_A + 2*L)^(-1)*L);
% where K_A has sqd input scales W_A,
% and where x_s is an element of B (forming the rows), which is modelled by a
% GP with sqd input scales W_B. 
% The prior is Gaussian with mean mu and variance L.

L = diag(prior.covariance)';

W_A = exp(2*gp_A_hypers.log_input_scales);
W_B = exp(2*gp_B_hypers.log_input_scales);

const_ups2_log_input_scales = ...
    log(sqrt(W_A + 2*L));
ups2_log_input_scales = log(sqrt( W_B + L - L.^2 ./ (W_A + 2*L) ));

% A_sqd_output_scale appears twice, as there were two covariances in the
% original integral
vec =   gaussian_mat(0, ...
            gp_A_hypers.log_output_scale, const_ups2_log_input_scales) * ...
        gaussian_mat(sqd_dist_stack_Bmu, ...
            gp_B_hypers.log_output_scale, ups2_log_input_scales);
