function [X,I] = vbmc_rnd(N,vp,origflag,permflag,df)
%VBMC_RND Draw random samples from VBMC posterior approximation.

if nargin < 3 || isempty(origflag); origflag = true; end
if nargin < 4 || isempty(permflag); permflag = false; end
if nargin < 5 || isempty(df); df = Inf; end

D = vp.D;   % Number of dimensions
K = vp.K;   % Number of components

if N < 1
    X = zeros(0,D);
    I = zeros(0,1);
else
    mu_t(:,:) = vp.mu';             % MU transposed
    sigma(1,:) = vp.sigma;
    lambda_t(1,:) = vp.lambda(:)';

    if permflag
        I = randperm(ceil(N/K)*K)';
        I = mod(I,K)+1;
    else
        I = randi(K,[N,1]);
    end

    if ~isfinite(df) || df == 0
        % Sample from variational posterior
        X = mu_t(I(1:N),:) + bsxfun(@times, lambda_t, bsxfun(@times,randn(N,D),sigma(I(1:N))'));
    else
        % Sample from heavy-tailed variant of variational posterior
        t = df/2./sqrt(gamrnd(df/2,df/2,[N,1]));
        X = mu_t(I(1:N),:) + bsxfun(@times, lambda_t, bsxfun(@times,bsxfun(@times,t,randn(N,D)),sigma(I(1:N))'));
    end

    if origflag && ~isempty(vp.trinfo)
        % Convert generated points back to original space
        X = warpvars(X,'inv',vp.trinfo);
    end
end
