function y = munifboxpdf(x,a,b)
%MUNIFBOXPDF Multivariate uniform box probability density function.
%   Y = MUNIFBOXPDF(X,A,B) returns the pdf of the multivariate uniform-box
%   distribution with bounds A and B, evaluated at the values in X. The 
%   multivariate uniform box pdf is the product of univariate uniform
%   pdfs in each dimension. 
%
%   For each dimension i, the univariate uniform-box pdf is defined as:
%
%                 |   
%                 |   ______________
%         p(X(i)) |   |            |
%                 |   |            |  
%                 |___|____________|_____
%                    A(i)          B(i)
%                           X(i)
%                          
%   X can be a matrix, where each row is a separate point and each column
%   is a different dimension. Similarly, A and B can also be matrices of 
%   the same size as X.
%
%   See also MUNIFBOXRND.

% Luigi Acerbi 2022

[N,D] = size(x);

if D > 1
    if isscalar(a); a = a*ones(1,D); end
    if isscalar(b); b = b*ones(1,D); end
end

if size(a,2) ~= D || size(b,2) ~= D
    error('munifboxpdf:SizeError', ...
        'A, B should be scalars or have the same number of columns as X.');
end

if size(a,1) == 1; a = repmat(a,[N,1]); end
if size(b,1) == 1; b = repmat(b,[N,1]); end

if any(a(:) >= b(:))
    error('munifboxpdf:OrderError', ...
        'For all elements of A and B, the order A < B should hold.');
end

nf = prod(b - a,2);
y = 1 ./ nf .* ones(N,1);
idx = any(bsxfun(@lt, x, a),2) | any(bsxfun(@gt, x, b),2);
y(idx) = 0;