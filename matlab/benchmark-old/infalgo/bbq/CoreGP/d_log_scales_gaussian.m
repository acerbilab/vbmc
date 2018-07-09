function [DK, DDK] = d_log_scales_gaussian(K, sqd_dist_stack, gp_hypers)
% function [DK, DDK] = d_log_scales_gaussian(K, sqd_dist_stack, gp_hypers)
%
% returns the matrix of derivatives and the matrix of second derivatives of
% a squared exponential covariance matrix with respect to the log input
% scales.
%
% OUTPUTS
% - DK: a stack of first derivative matrices, each plate in the stack
%       is the derivative with respect to a different log input scale.
% - DDK: a stack of second derivative matrices, each plate in the stack
%       is the derivative with respect to a different log input scale.
%
% INPUTS
% - K: the covariance matrix
% - sqd_dist_stack: N by N by D stack of squared
%       distances, where N is the number of data, and D is the number of
%       dimensions.
% - gp_hypers: hypers for gaussian covariance, with fields
%   * log_output_scale
%   * log_input_scales


% hyperparameters for gp over the transformed likelihood, tl, assumed
% to have zero mean
input_scales = exp(gp_hypers.log_input_scales);
D = length(input_scales);
inv_sqd_input_scales_stack = ...
    reshape(input_scales.^-2, 1, 1, D);

% DK is the gradient of the Gaussian covariance: each plate in the stack
% is the derivative with respect to a different log input scale
DK_const = -1 + bsxfun(@times, ...
    sqd_dist_stack, ...
    inv_sqd_input_scales_stack);
DK = bsxfun(@times, K, DK_const);

if nargout > 1
% DK is the gradient of the Gaussian covariance: each plate in the stack
% is the derivative with respect to a different log input scale
DDK_const = 1 ...
    -4 * bsxfun(@times, ...
    sqd_dist_stack, ...
    inv_sqd_input_scales_stack) ...
    + bsxfun(@times, ...
    sqd_dist_stack.^2, ...
    inv_sqd_input_scales_stack.^2);
DDK = bsxfun(@times, K, DDK_const);
end
