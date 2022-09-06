function r = munifboxrnd(a,b,n)
%MUNIFBOXRND Random arrays from the multivariate uniform box distribution.
%   R = MUNIFBOXRND(A,B) returns an N-by-D matrix R of random vectors
%   chosen from the multivariate uniform box distribution with bounds A and
%   B. A and B are N-by-D matrices, and MUNIFBOXRND generates each row of R 
%   using the corresponding row of A and B.
%
%   R = MUNIFBOXRND(A,B,N) returns a N-by-D matrix R of random vectors
%   chosen from the multivariate uniform box distribution with 1-by-D bound
%   vectors A and B.
%
%   See also MUNIFBOXPDF.

% Luigi Acerbi 2022

[N,D] = size(a);
[Nb,Db] = size(b);

if nargin < 3 || isempty(n)
    n = N;
else
    if (N ~= 1 && N ~= n) || (Nb ~= 1 && Nb ~= n)
        error('munifboxrnd:SizeError', ...
            'A and B should be 1-by-D or N-by-D arrays.');
    end    
end
if N ~= Nb || D ~= Db 
    error('munifboxrnd:SizeError', ...
        'A and B should be arrays of the same size.');
end

if any(a(:) >= b(:))
    error('munifboxpdf:OrderError', ...
        'For all elements of A and B, the order A < B should hold.');
end


r = bsxfun(@plus, a, bsxfun(@times, rand(n,D), b - a));  