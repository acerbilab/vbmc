function [mubar,Sigma] = vbmc_moments(vp,origflag,Ns)
%VBMC_MOMENTS Compute moments of variational posterior.

if nargin < 2 || isempty(origflag); origflag = true; end
if nargin < 3 || isempty(Ns); Ns = 1e5; end

K = vp.K;

if origflag
    X = vbmc_rnd(Ns,vp,1,1);
    mubar = mean(X,1)';
    Sigma = cov(X);    
else
    w(1,:) = vp.w;                       % Mixture weights
    mu(:,:) = vp.mu;
    sigma(1,:) = vp.sigma;
    lambda(:,1) = vp.lambda(:);

    mubar = sum(bsxfun(@times,w,mu),2);

    Sigma = sum(w.*sigma.^2)*diag(lambda.^2);
    for k = 1:K; Sigma = Sigma + w(k)*(mu(:,k)-mubar)*(mu(:,k)-mubar)'; end
end