function y = mtrapezlogpdf(x,a,u,v,b)
%MTRAPEZLOGPDF Multivariate trapezoidal probability log pdf.
%   Y = MTRAPEZLOGPDF(X,A,U,V,B) returns the logarithm of the pdf of the 
%   multivariate trapezoidal distribution with external bounds A and B and 
%   internal points U and V, evaluated at the values in X. The multivariate
%   trapezoidal pdf is the product of univariate trapezoidal pdfs in each
%   dimension. 
%
%   For each dimension i, the univariate trapezoidal pdf is defined as:
%
%                 |       __________
%                 |      /|        |\
%         p(X(i)) |     / |        | \ 
%                 |    /  |        |  \
%                 |___/___|________|___\____
%                    A(i) U(i)     V(i) B(i)
%                             X(i)
%                          
%   X can be a matrix, where each row is a separate point and each column
%   is a different dimension. Similarly, A, B, C, and D can also be
%   matrices of the same size as X.
%
%   The log pdf is typically preferred in numerical computations involving
%   probabilities, as it is more stable.
%
%   See also MTRAPEZPDF, MTRAPEZRND.

% Luigi Acerbi 2022

[N,D] = size(x);

if D > 1
    if isscalar(a); a = a*ones(1,D); end
    if isscalar(u); u = u*ones(1,D); end
    if isscalar(v); v = v*ones(1,D); end
    if isscalar(b); b = b*ones(1,D); end
end

if size(a,2) ~= D || size(u,2) ~= D || size(v,2) ~= D || size(b,2) ~= D
    error('mtrapezpdf:SizeError', ...
        'A, B, C, D should be scalars or have the same number of columns as X.');
end

if size(a,1) == 1; a = repmat(a,[N,1]); end
if size(u,1) == 1; u = repmat(u,[N,1]); end
if size(v,1) == 1; v = repmat(v,[N,1]); end
if size(b,1) == 1; b = repmat(b,[N,1]); end

y = -inf(size(x));
lnf = log(0.5) + log(b - a + v - u) + log(u - a);

for ii = 1:D
    idx = x(:,ii) >= a(:,ii) & x(:,ii) < u(:,ii);
    y(idx,ii) = log(x(idx,ii) - a(idx,ii)) - lnf(idx,ii);
    
    idx = x(:,ii) >= u(:,ii) & x(:,ii) < v(:,ii);
    y(idx,ii) = log(u(idx,ii)-a(idx,ii)) - lnf(idx,ii);
    
    idx = x(:,ii) >= v(:,ii) & x(:,ii) < b(:,ii);
    y(idx,ii) = log(b(idx,ii) - x(idx,ii)) - log(b(idx,ii) - v(idx,ii)) + log(u(idx,ii)-a(idx,ii)) - lnf(idx,ii);
end

y = sum(y,2);