function out = ...
    log_gaussian_mat(sqd_dist_stack, gp_hypers, log_input_scales, log_det_input_scales)
% out = gaussian_mat(sqd_dist_stack, gp_hypers, log_input_scales,
%     log_det_input_scales)
%
% returns the log of a gram matrix of a Gaussian covariance. 
%
% OUTPUTS
% - out: the log - covariance matrix.
%
% INPUTS 
% - sqd_dist_stack: N by N by D stack of squared
%       distances, where N is the number of data, and D is the number of
%       dimensions.
% - gp_hypers: a structure containing all hyperparameters; alternatively,
%       just the log output scale
% - log_input_scales
% - (optional) log_det_input_scales: the input scales used to define the
%       multiplicative factor for the Gaussian, useful for non-normalised
%       Gaussians.

switch nargin 
    case 2
        log_output_scale = gp_hypers.log_output_scale;
        log_input_scales = gp_hypers.log_input_scales;
        log_det_input_scales = log_input_scales;
    case 3
        log_output_scale = gp_hypers;
        log_det_input_scales = log_input_scales;
    case 4      
        log_output_scale = gp_hypers;
end
    
D = length(log_input_scales);
sqd_input_scales_stack = reshape(exp(2*log_input_scales),1,1,D);
log_sqd_lambda = 2*log_output_scale ...
    - sum(log_input_scales) - D/2 * log(2*pi);

out = log_sqd_lambda ...
        -0.5 * sum(bsxfun(@rdivide, ...
                    sqd_dist_stack, sqd_input_scales_stack), 3);