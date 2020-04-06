function [X,I] = vbmc_rnd(vp,N,origflag,balanceflag,df)
%VBMC_RND Random samples from VBMC posterior approximation.
%   X = VBMC_RND(VP,N) returns an N-by-D matrix X of random vectors chosen 
%   from the variational posterior VP.
%
%   X = VBMC_RND(VP,N,ORIGFLAG) returns the random vectors in the original
%   parameter space if ORIGFLAG=1 (default), or in the transformed VBMC
%   space if ORIGFLAG=0.
%
%   X = VBMC_RND(VP,N,ORIGFLAG,BALANCEFLAG) for BALANCEFLAG=1 balances the 
%   generating process such that the random samples in X come from each 
%   mixture component exactly proportionally (or as close as possible) to 
%   the variational mixture weights. If BALANCEFLAG=0 (default), the 
%   generating mixture for each sample is determined randomly according to 
%   the mixture weights.
%
%   X = VBMC_RND(VP,N,ORIGFLAG,BALANCEFLAG,DF) returns random samples
%   generated from an heavy-tailed version of the variational posterior,
%   in which the multivariate normal components have been replaced by
%   multivariate t-distributions with DF degrees of freedom. The default is
%   DF=Inf, limit in which the t-distribution becomes a multivariate normal.
%
%   [X,I] = VBMC_RND(...) also returns an N-by-1 array such that the n-th
%   element of I indicates the index of the variational mixture component 
%   from which the n-th row of X has been generated.
%
%   X = VBMC_RND(VP,N,ORIGFLAG,'gp') generates random samples using the 
%   Gaussian process the variational posterior has been fit to. This 
%   approach is much slower than sampling from the variational posterior, 
%   as samples are generated via Markov Chain Monte Carlo, but it can 
%   occasionally be more precise.
%
%   See also VBMC, VBMC_MOMENTS, VBMC_PDF.

if nargin < 3 || isempty(origflag); origflag = true; end
if nargin < 4 || isempty(balanceflag); balanceflag = false; end
if nargin < 5 || isempty(df); df = Inf; end

D = vp.D;   % Number of dimensions
K = vp.K;   % Number of components

if N < 1
    X = zeros(0,D);
    I = zeros(0,1);
elseif ischar(balanceflag) && strcmpi(balanceflag,'gp')
    if ~isfield(vp,'gp') || isempty(vp.gp)
        error('Cannot sample from GP, the GP associated to the variational posterior is empty.');
    end    
    X = gpsample_vbmc(vp,vp.gp,N,origflag);    
else
    w = vp.w;                       % Mixture weights
    mu_t(:,:) = vp.mu';             % MU transposed
    sigma(1,:) = vp.sigma;
    lambda_t(1,:) = vp.lambda(:)';

    if vp.K > 1        
        if balanceflag
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
        X = warpvars_vbmc(X,'inv',vp.trinfo);
    end
end

end

%--------------------------------------------------------------------------
function x = catrnd(p,n)
%CATRND Sample from categorical distribution.

maxel = 1e6;
Nel = n*numel(p);
stride = ceil(maxel/numel(p));

cdf(1,:) = cumsum(p);
u = rand(n,1)*cdf(end);

% Split for memory reasons
if Nel <= maxel
    x = sum(bsxfun(@lt, cdf, u),2) + 1;
else
    x = zeros(n,1);
    idx_min = 1;
    while idx_min <= n
        idx_max = min(idx_min+stride-1,n);
        idx = idx_min:idx_max;
        x(idx) = sum(bsxfun(@lt, cdf, u(idx)),2) + 1;
        idx_min = idx_max+1;
    end
end

end