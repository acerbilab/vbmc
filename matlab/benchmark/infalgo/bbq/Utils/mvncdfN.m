function [ y ] = mvncdfN( xl, xu, mu, sigma )
% Y = mvncdfN(XL,XU,MU,SIGMA) returns the multivariate normal cumulative
%     probability evaluated over the rectangle (hyper-rectangle for D>2)
%     with lower and upper limits defined by XL and XU, respectively. We
%     assume the multivariate normal distribution with mean MU and
%     covariance SIGMA.  SIGMA is a D-by-D-by-N array; the cumulative
%     probability is evaluated for each row of X with the corresponding
%     page of SIGMA, i.e., mvnpdf computes Y(I) using X(I,:) and
%     SIGMA(:,:,I).
%
%     see: mvncdf and mvnpdf. 

N = size(sigma, 3);
y = nan(N, 1);

for i = 1:N
    y(i) = mvncdf( xl(:, i), xu(:, i), mu(:, i), sigma(:, :, i) );
end


end

