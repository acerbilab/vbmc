function [X,I] = vbmc_rnd(N,vp,origflag,exactflag,df)
%VBMC_RND Draw random samples from VBMC posterior approximation.

if nargin < 3 || isempty(origflag); origflag = true; end
if nargin < 4 || isempty(exactflag); exactflag = false; end
if nargin < 5 || isempty(df); df = Inf; end

D = vp.D;   % Number of dimensions
K = vp.K;   % Number of components

if N < 1
    X = zeros(0,D);
    I = zeros(0,1);
else
    w = vp.w;                       % Mixture weights
    mu_t(:,:) = vp.mu';             % MU transposed
    sigma(1,:) = vp.sigma;
    lambda_t(1,:) = vp.lambda(:)';

    if vp.K > 1        
        if exactflag
            % Exact split of samples according to mixture weights
            N_floor = floor(w*N);
            cumN = [0,cumsum(N_floor)];
            I = zeros(cumN(end),1);
            for k = 1:K
                I((1:N_floor(k))+cumN(k)) = k;
            end
            
            % Compute remainder samples (with correct weights) if needed
            if N > cumN(end)
                w_extra = w*N - N_floor;
                N_extra = ceil(sum(w_extra));
                delta_extra = N_extra - sum(w_extra);
                w_extra = w_extra + w*delta_extra;
                I_extra = catrnd(w_extra,N_extra);
                I = [I; I_extra];
            end
            
            % Randomly permute indices and take only N
            I = I(randperm(numel(I),N));
        else
            I = catrnd(w,N);
        end
        
        if ~isfinite(df) || df == 0
            % Sample from variational posterior
            X = mu_t(I,:) + bsxfun(@times, lambda_t, bsxfun(@times,randn(N,D),sigma(I(1:N))'));
        else
            % Sample from heavy-tailed variant of variational posterior
            t = df/2./sqrt(gamrnd(df/2,df/2,[N,1]));
            X = mu_t(I,:) + bsxfun(@times, lambda_t, bsxfun(@times,bsxfun(@times,t,randn(N,D)),sigma(I(1:N))'));
        end
    else
        if ~isfinite(df) || df == 0
            % Sample from variational posterior
            X = mu_t + bsxfun(@times, lambda_t, randn(N,D)*sigma);
        else
            % Sample from heavy-tailed variant of variational posterior
            t = df/2./sqrt(gamrnd(df/2,df/2,[N,1]));
            X = mu_t + bsxfun(@times, lambda_t, sigma*bsxfun(@times,t,randn(N,D)));
        end
        if nargout > 1; I = ones(N,1); end        
    end

    if origflag && ~isempty(vp.trinfo)
        % Convert generated points back to original space
        X = warpvars(X,'inv',vp.trinfo);
    end
end

end

%--------------------------------------------------------------------------
function x = catrnd(p,n)
%CATRND Sample from categorical distribution.

cdf(1,:) = cumsum(p);
u = rand(n,1)*cdf(end);
x = sum(bsxfun(@lt, cdf, u),2) + 1;

end