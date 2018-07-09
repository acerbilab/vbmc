function inds = diag_inds(A)
% inds = diag_inds(A)
% returns the indices of square matrix A corresponding to its diagonal

N = length(A);
inds = 1:N+1:N^2;