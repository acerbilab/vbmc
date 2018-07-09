function [ gaussian ] = sqdexp2gaussian( sqd_exp )
% convert a structure containing the hyperparameters of a sqd exp
% covariance into another containing the hyperparameters of a gaussian
% covariance

gaussian = sqd_exp;
% Note that gaussian_mat(0, s) returns exp(2*s.log_output_scale) times the
% normalisation constant of a Gaussian with covariance
% diag(exp(2*s.log_input_scales))
gaussian.log_output_scale = ...
    2 * sqd_exp.log_output_scale - 0.5 * log_gaussian_mat(0, sqd_exp);
end
