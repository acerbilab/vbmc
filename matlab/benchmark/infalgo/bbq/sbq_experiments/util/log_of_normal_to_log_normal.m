function [log_mean_Z, log_var_Z] = ...
    log_of_normal_to_log_normal( mean_log_Z, var_log_Z )
%
% Converts the log of the mean and variance of a Gaussian into the
% mean and variance of a log-normal distribution with those parameters.
%
% inputs:  log of the mean and variance of a random variable Z.
% outputs: the mean and variance of a log-normal distribution, with params given
%          by the exp of the input params.
%
% David Duvenaud
% February 2012

log_mean_Z = mean_log_Z + var_log_Z ./ 2;
log_var_Z = (2.*mean_log_Z + 2 * var_log_Z ) ./ ( 2.*mean_log_Z - var_log_Z );
