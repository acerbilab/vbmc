function [H, DK] = hess_log_scales_lik_gaussian...
        (K, R, inv_K_y, sqd_dist_stack, gp_hypers)
% H = hess_log_scales_lik_gaussian(K, R, inv_K_y, sqd_dist_stack, gp_hypers) 
%
% returns the diagonal of the hessian of the likelihood of a GP with
% gaussian covariance with respect to its log input scales. 
%
% OUTPUTS
% - H: a column vector representing the diagonal of the hessian of the
%       likelihood of a GP with gaussian covariance with respect to its log
%       input scales
% - DK: a stack of first derivative matrices, each plate in the stack
%       is the derivative with respect to a different log input scale.
%
% INPUTS
% - K: the covariance matrix
% - R: its cholesky factor
% - inv_K_y: inv(K) times the data y
% - sqd_dist_stack: N by N by D stack of squared
%       distances, where N is the number of data, and D is the number of
%       dimensions.
% - gp_hypers: hypers for gaussian covariance, with fields
%   * log_output_scale
%   * log_input_scales

D = size(sqd_dist_stack, 3);

[DK, DDK] = d_log_scales_gaussian(K, sqd_dist_stack, gp_hypers);

lowr.UT = true;
lowr.TRANSA = true;

inv_K_DK = solve_chol3(R, DK);

H = + 0.5 * prod3(inv_K_y', prod3(DDK, inv_K_y)) ...
    - 0.5 * trace3(solve_chol3(R, DDK)) ...
    - prod3(prod3(inv_K_y', DK), prod3(inv_K_DK, inv_K_y)) ...
    + 0.5 * trace3(prod3(inv_K_DK, inv_K_DK));

H = reshape(H, D, 1, 1);