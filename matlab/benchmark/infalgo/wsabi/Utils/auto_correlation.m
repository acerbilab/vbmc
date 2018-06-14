function rho = auto_correlation(y,k)
% ACF_K - Autocorrelation at Lag k
%
% Inputs:
% y - series to compute acf for
% k - which lag to compute acf
% 

if nargin < 2
    k = 1;
end
cross_sum = zeros(length(y)-k,1) ;

ybar = mean(y);

% Numerator, unscaled covariance
for i = (k+1):length(y)
    cross_sum(i) = (y(i)-ybar)*(y(i-k)-ybar) ;
end

% Denominator, unscaled variance
yvar = (y-ybar)'*(y-ybar) ;

rho = sum(cross_sum) / yvar ;
