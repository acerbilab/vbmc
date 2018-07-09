function inds = triu_inds(A,k)
% inds = diag_inds(A)
% returns the indices of square matrix A corresponding to its upper
% triangle. k is as per matlab function triu.

if nargin<2
    k=0;
end

N = length(A);
inds = triu(true(N),k);

% A = rand(1000);
% N = length(A);
% tic;for i=1:100;A(triu_inds(A))=ones(0.5*N*(N+1),1);end;toc
% tic;for i=1:100;A=A-triu(A)+triu(ones(N));end;toc
% the triu_inds approach is faster