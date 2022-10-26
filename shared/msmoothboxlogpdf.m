function y = msmoothboxlogpdf(x,a,b,sigma)
%MSMOOTHBOXLOGPDF Multivariate smooth-box log probability density function.
%   Y = MSMOOTHBOXLOGPDF(X,A,B,SIGMA) returns the logarithm of the pdf of 
%   the multivariate smooth-box distribution with pivots A and B and scale 
%   SIGMA, evaluated at the values in X. The multivariate smooth-box pdf is
%   the product of univariate smooth-box pdfs in each dimension. 
%
%   For each dimension i, the univariate smooth-box pdf is defined as a
%   uniform distribution between pivots A(i), B(i) and Gaussian tails that
%   fall starting from p(A(i)) to the left (resp., p(B(i)) to the right) 
%   with standard deviation SIGMA(i).
%                          
%   X can be a matrix, where each row is a separate point and each column
%   is a different dimension. Similarly, A, B, and SIGMA can also be
%   matrices of the same size as X.
%
%   The log pdf is typically preferred in numerical computations involving
%   probabilities, as it is more stable.
%
%   See also MSMOOTHBOXPDF, MSMOOTHBOXRND.

% Luigi Acerbi 2022

[N,D] = size(x);

if any(sigma(:) <= 0)
    error('msmoothboxpdf:NonPositiveSigma', ...
        'All elements of SIGMA should be positive.');    
end

if D > 1
    if isscalar(a); a = a*ones(1,D); end
    if isscalar(b); b = b*ones(1,D); end
    if isscalar(sigma); sigma = sigma*ones(1,D); end
end

if size(a,2) ~= D || size(b,2) ~= D || size(sigma,2) ~= D
    error('msmoothboxpdf:SizeError', ...
        'A, B, SIGMA should be scalars or have the same number of columns as X.');
end

if size(a,1) == 1; a = repmat(a,[N,1]); end
if size(b,1) == 1; b = repmat(b,[N,1]); end
if size(sigma,1) == 1; sigma = repmat(sigma,[N,1]); end

if any(a(:) >= b(:))
    error('msmoothboxpdf:OrderError', ...
        'For all elements of A and B, the order A < B should hold.');
end

y = -inf(size(x));
lnf = log(1/sqrt(2*pi)./sigma) - log1p(1/sqrt(2*pi)./sigma.*(b - a));

for ii = 1:D
    idx = x(:,ii) < a(:,ii);
    y(idx,ii) = lnf(idx,ii) - 0.5*((x(idx,ii) - a(idx,ii))./sigma(idx,ii)).^2;

    idx = x(:,ii) >= a(:,ii) & x(:,ii) <= b(:,ii);
    y(idx,ii) = lnf(idx,ii);
    
    idx = x(:,ii) > b(:,ii);
    y(idx,ii) = lnf(idx,ii) - 0.5*((x(idx,ii) - b(idx,ii))./sigma(idx,ii)).^2; 
end

y = sum(y,2);