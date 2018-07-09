function log_Z = log_volume_between_two_gaussians2( mu_a, sigma_a, mu_b, sigma_b)
% Find the volume of the integral of the product of two multivariate Gaussians.
%
% Inputs:
%   mu_a, sigma_a are the mean and covariance of the first gaussian.
%   mu_b, sigma_b are the mean and covariance of the second gaussian.
%
% David Duvenaud
% January 2012

log_Z = logmvnpdf(mu_a, mu_b, sigma_a + sigma_b)

end

