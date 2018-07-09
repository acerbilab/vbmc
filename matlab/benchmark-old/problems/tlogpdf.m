function y = tlogpdf(x,mu,sigma,nu)
%TLOGPDF  Log probability density function (pdf) for Student's T distribution
%   Y = TLOGPDF(X,MU,SIGMA,V) returns the pdf of Student's T distribution 
%   with mean MU, scale parameter SIGMA and V degrees of freedom, at the 
%   values in X.

z = bsxfun(@rdivide,bsxfun(@minus,x,mu),sigma);
nf = bsxfun(@minus,gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*log(pi*nu),log(sigma));
y = bsxfun(@plus,nf,bsxfun(@times,-(nu+1)/2,log1p(bsxfun(@rdivide,z.^2,nu))));

end