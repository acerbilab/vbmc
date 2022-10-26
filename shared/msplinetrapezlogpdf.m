function y = msplinetrapezlogpdf(x,a,b,c,d)
%MSPLINETRAPEZLOGPDF Multivariate spline-trapezoidal log pdf.
%   Y = MSPLINETRAPEZLOGPDF(X,A,B,C,D) returns the logarithm of the pdf of 
%   the multivariate spline-trapezoidal distribution with external bounds 
%   A and D and internal points B and C, evaluated at the values in X. The 
%   multivariate pdf is the product of univariate spline-trapezoidal pdfs 
%   in each dimension. 
%
%   For each dimension i, the univariate spline-trapezoidal pdf is defined 
%   as a trapezoidal pdf whose points A, B and C, D are connected by cubic
%   splines such that the pdf is continuous and its derivatives at A, B, C,
%   and D are zero (so the derivatives are also continuous):
%
%                 |       __________
%                 |      /|        |\
%         p(X(i)) |     / |        | \ 
%                 |    /  |        |  \
%                 |___/___|________|___\____
%                    A(i) B(i)     C(i) D(i)
%                             X(i)
%                          
%   X can be a matrix, where each row is a separate point and each column
%   is a different dimension. Similarly, A, B, C, and D can also be
%   matrices of the same size as X.
%
%   The log pdf is typically preferred in numerical computations involving
%   probabilities, as it is more stable.
%
%   See also MSPLINETRAPEZPDF, MSPLINETRAPEZRND.

% Luigi Acerbi 2022

[N,D] = size(x);

if D > 1
    if isscalar(a); a = a*ones(1,D); end
    if isscalar(b); b = b*ones(1,D); end
    if isscalar(c); c = c*ones(1,D); end
    if isscalar(d); d = d*ones(1,D); end
end

if size(a,2) ~= D || size(b,2) ~= D || size(c,2) ~= D || size(d,2) ~= D
    error('msplinetrapezlogpdf:SizeError', ...
        'A, B, C, D should be scalars or have the same number of columns as X.');
end

if size(a,1) == 1; a = repmat(a,[N,1]); end
if size(b,1) == 1; b = repmat(b,[N,1]); end
if size(c,1) == 1; c = repmat(c,[N,1]); end
if size(d,1) == 1; d = repmat(d,[N,1]); end

y = -inf(size(x));
% Normalization factor
% nf = c - b + 0.5*(d - c + b - a);
lnf = log(0.5*(c - b + d - a));

for ii = 1:D    
    idx = x(:,ii) >= a(:,ii) & x(:,ii) < b(:,ii);
    z = (x(idx,ii) - a(idx,ii))./(b(idx,ii) - a(idx,ii));    
    y(idx,ii) = log(-2*z.^3 + 3*z.^2) - lnf(idx,ii);
    
    idx = x(:,ii) >= b(:,ii) & x(:,ii) < c(:,ii);
    y(idx,ii) = -lnf(idx,ii);
    
    idx = x(:,ii) >= c(:,ii) & x(:,ii) < d(:,ii);
    z = 1 - (x(idx,ii) - c(idx,ii)) ./ (d(idx,ii) - c(idx,ii));    
    y(idx,ii) = log(-2*z.^3 + 3*z.^2) - lnf(idx,ii);
end

y = sum(y,2);

