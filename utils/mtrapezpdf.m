function y = mtrapezpdf(x,a,b,c,d)
%MTRAPEZPDF Multivariate trapezoidal probability density function.

% a < b < c < d

y = zeros(size(x));
nf = 0.5 * (d - a + c - b) .* (b - a);

for ii = 1:size(x,2)
    
    idx = x(:,ii) >= a(ii) & x(:,ii) < b(ii);
    y(idx,ii) = (x(idx,ii) - a(ii)) / nf(ii);
    
    idx = x(:,ii) >= b(ii) & x(:,ii) < c(ii);
    y(idx,ii) = (b(ii)-a(ii)) / nf(ii);
    
    idx = x(:,ii) >= c(ii) & x(:,ii) < d(ii);
    y(idx,ii) = (d(ii) - x(idx,ii))/(d(ii) - c(ii)) *  (b(ii)-a(ii)) / nf(ii);    
end

y = prod(y,2);

