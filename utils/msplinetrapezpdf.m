function y = msplinetrapezpdf(x,a,b,c,d)
%MSPLINETRAPEZPDF Multivariate cubic-spline trapezoidal probability density function.

% a < b < c < d

y = zeros(size(x));
% Normalization factor
% nf = c - b + 0.5*(d - c + b - a);
nf = 0.5*(c - b + d - a);

for ii = 1:size(x,2)
    
    idx = x(:,ii) >= a(ii) & x(:,ii) < b(ii);
    z = (x(idx,ii) - a(ii))/(b(ii) - a(ii));    
    y(idx,ii) = (-2*z.^3 + 3*z.^2) / nf(ii);
    
    idx = x(:,ii) >= b(ii) & x(:,ii) < c(ii);
    y(idx,ii) = 1 / nf(ii);
    
    idx = x(:,ii) >= c(ii) & x(:,ii) < d(ii);
    z = 1 - (x(idx,ii) - c(ii))/(d(ii) - c(ii));    
    y(idx,ii) = (-2*z.^3 + 3*z.^2) / nf(ii);
end

y = prod(y,2);

