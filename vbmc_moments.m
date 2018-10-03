function [mubar,Sigma] = vbmc_moments(vp,origflag,Ns)
%VBMC_MOMENTS Compute moments of variational posterior.
%   [MU,SIGMA] = VBMC_MOMENTS(VP) computes the mean MU and covariance
%   matrix SIGMA of the variational posterior VP via Monte Carlo sampling.
%
%   [...] = VBMC_MOMENTS(VP,ORIGFLAG) computes the moments of the
%   variational posterior VP in the original problem space if ORIGFLAG=1
%   (default), or in the transformed VBMC space if ORIGFLAG=0. In the
%   transformed space, the moments are computed analytically.
%
%   [...] = VBMC_MOMENTS(VP,1,NS) uses NS samples to evaluate the moments
%   of the variational posterior in the original space (default NS=1e6).
%
%   See also VBMC, VBMC_MODE, VBMC_PDF, VBMC_RND.

if nargin < 2 || isempty(origflag); origflag = true; end
if nargin < 3 || isempty(Ns); Ns = 1e6; end

covflag = nargout > 1;      % Compute covariance?

K = vp.K;

if origflag
    X = vbmc_rnd(vp,Ns,1,1);
    mubar = mean(X,1);
    if covflag
        Sigma = cov(X);
    end
else
    w(1,:) = vp.w;                       % Mixture weights
    mu(:,:) = vp.mu;

    mubar = sum(bsxfun(@times,w,mu),2);
    
    if covflag
        sigma(1,:) = vp.sigma;
        lambda(:,1) = vp.lambda(:);

        Sigma = sum(w.*sigma.^2)*diag(lambda.^2);
        for k = 1:K; Sigma = Sigma + w(k)*(mu(:,k)-mubar)*(mu(:,k)-mubar)'; end
    end
    
    mubar = mubar(:)';              % Return row vector
end