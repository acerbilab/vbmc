function const = small_chi_const(gp_A_hypers, prior)
% Returns the constant
% chi = int int K(x, x') prior(x) prior(x') dx dx'
% = N(0, 0, (W_A + 2*L)) 
% where A is modelled by a GP with sqd input scales W_A. 
% the prior is Gaussian with mean mu and variance L.

prior_var = diag(prior.covariance)';

A_sqd_input_scales = exp(2*gp_A_hypers.log_input_scales);
chi_log_input_scales = ...
    log(sqrt(2 * prior_var + A_sqd_input_scales));

% A_sqd_output_scale appears twice, as there were two covariances in the
% original integral
const =   gaussian_mat(0, ...
            gp_A_hypers.log_output_scale, chi_log_input_scales);
