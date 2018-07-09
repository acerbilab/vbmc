function K_new = update_gaussian_mat(K_old, ...
       sqd_dist_stack_col, varargin)
% return the gram matrix of a Gaussian covariance, updated with a new stack
% of squared distances. It's assumed that the new elements correspond to
% columns, and old and new together form the rows.
%
% OUTPUTS
% - K_new: the new covariance matrix; with nans on the old off-diagonals.
%
% INPUTS 
% - sqd_dist_stack_col: (N_old+N) by N by D stack of squared
%       distances, where N_old is the old number of data, N is the number
%       additional elements and D is the number of dimensions.
% - varargin: passed directly to gaussian_mat.m
    
N_new = size(sqd_dist_stack_col, 1);
N_old = size(K_old, 1);

K_new = nan(N_new);
diag_K = diag_inds(K_new);
K_new(diag_K(1:N_old)) = diag(K_old);
K_col = gaussian_mat(sqd_dist_stack_col, varargin{:});
    
range_new = N_old+1:N_new;
K_new(range_new, :) = K_col;
K_new(:, range_new) = K_col';