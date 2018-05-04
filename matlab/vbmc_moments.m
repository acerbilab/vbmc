function [mubar,Sigma] = vbmc_moments(vp,origflag,Ns)
%VBMC_MOMENTS Compute moments of variational posterior.

if nargin < 2 || isempty(origflag); origflag = true; end
if nargin < 3 || isempty(Ns); Ns = 1e6; end

K = vp.K;

if origflag
    X = vbmc_rnd(Ns,vp,1,1);
    mubar = mean(X,1)';
    Sigma = cov(X);    
else
    mu(:,:) = vp.mu;
    sigma(1,:) = vp.sigma;
    lambda(:,1) = vp.lambda(:);

    mubar = mean(mu,2);

    Sigma = mean(sigma.^2)*diag(lambda.^2);
    for k = 1:K; Sigma = Sigma + (mu(:,k)-mubar)*(mu(:,k)-mubar)'/K; end
end