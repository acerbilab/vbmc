function y = logmvgamma(x,d)
%LOGMVGAMMA Logarithm of multivariate Gamma function.
%
%   Y = LOGMVGAMMA(X,D) returns the logarithm of the multivariate Gamma 
%   function in D dimensions, which is used in the probability density 
%   function of the Wishart and inverse Wishart distributions.
%      
%     Gamma_d(x) = pi^{d(d-1)/4} \prod_{j=1}^d Gamma(x+(1-j)/2)
%     log(Gamma_d(x)) = d(d-1)/4 log(pi) + \sum_{j=1}^d log(Gamma(x+(1-j)/2))

s = size(x);
x = reshape(x,1,prod(s));
x = bsxfun(@plus,x,(1-(1:d)')/2);
y = d*(d-1)/4*log(pi)+sum(gammaln(x),1);
y = reshape(y,s);