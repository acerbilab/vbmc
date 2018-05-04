function [nlZ,dnlZ,post] = gplite_inf(hyp,X,y,varargin)
%GPLITE_INF Inference for lite Gaussian Processes regression.
%   [NLZ,DNLZ,POST] = GPLITE_INF(HYP,X,Y) computes the log marginal likelihood 
%   NLZ and its gradient DNLZ for a GP defined by HYP. HYP is a column vector 
%   of hyperparameters (see below). X is a N-by-D matrix of training inputs 
%   and Y a N-by-1 vector of training targets (function values at X).
%   Also returns the GP posterior struct POST, useful to speed up subsequent 
%   GPLITE computations.
%   
%   [NLZ,DNLZ,POST] = GPLITE_INF(HYP,X,Y,HPRIOR) uses prior over hyperparameters
%   defined by the struct HPRIOR. HPRIOR has fields HPRIOR.mu, HPRIOR.sigma
%   and HPRIOR.nu which contain vectors representing, respectively, the mean, 
%   standard deviation and degrees of freedom of the prior for each 
%   hyperparameter. Priors are generally represented by Student's t distributions.
%   Set HPRIOR.nu(i) = Inf to have instead a Gaussian prior for the i-th
%   hyperparameter. Set HPRIOR.sigma(i) = Inf to have a (non-normalized)
%   flat prior over the i-th hyperparameter. Priors are defined in
%   transformed hyperparameter space (i.e., log space for positive-only
%   hyperparameters).
%
%   POST = GPLITE_INF(HYP,X,Y,1) only computes the GP posterior struct.
%
%   POST_NEW = GPLITE_INF(HYP,X,POST,XSTAR,YSTAR) performs a fast rank-1 
%   update for a GPLITE posterior structure, given a single new observation 
%   at XSTAR with observed value YSTAR.

post = [];
post_only = false;
rank1_update = false;
hprior = [];

if isstruct(y); post = y; end

if nargin > 3
    if isstruct(varargin{1})
        hprior = varargin{1};
    elseif ~isempty(post) && nargin > 4
        rank1_update = true;
        xstar = varargin{1};
        ystar = varargin{2};
        post_only = true;
        if size(xstar,1) > 1 || numel(ystar) > 1
            error('gplite_inf:NotRankOne', ...
              'GPLITE_INF with this input format only supports rank-one updates.');
        end
    elseif varargin{1}
        post_only = true;
    end
end

[N,D] = size(X);            % Number of training points and dimension
[Nhyp,Ns] = size(hyp);      % Hyperparameters and samples
compute_grad = nargout > 1; % Compute gradient if required

if all(Nhyp ~= [D+3,3*D+3])
    error('gplite_inf:dimmismatch','Number of hyperparameters mismatched with dimension of training inputs.');
end
if compute_grad && Ns > 1
    error('gplite_inf:NoSampling', ...
        'GP inference with log marginal likelihood is available only for one-sample hyperparameter inputs.');
end
if compute_grad && post_only
    error('gplite_inf:TooManyOutputs', ...
        'Too many output arguments with current input. Expected to return only the GP posterior.');            
end


% Loop over hyperparameter samples
if isempty(post)
    for s = 1:Ns

        % Extract GP hyperparameters from HYP
        ell = exp(hyp(1:D,s));
        sf2 = exp(2*hyp(D+1,s));
        sn2 = exp(2*hyp(D+2,s));
        sn2_mult = 1;  % Effective noise variance multiplier

        nf = 1 / (2*pi)^(D/2) / prod(ell);  % Kernel normalization factor
        
        % Evaluate mean function (and gradient if needed)
        if compute_grad
            [m,dm] = gplite_meanfun(hyp(:,s),X);    
        else
            m = gplite_meanfun(hyp(:,s),X);
        end
        
        % Compute kernel matrix K_mat
        K_mat = sq_dist(diag(1./ell)*X');
        K_mat = sf2 * nf * exp(-K_mat/2);

        % Cholesky decomposition until it works
        while 1
            [L,p] = chol(K_mat/(sn2*sn2_mult)+eye(N));
            if p > 0; sn2_mult = sn2_mult*2; else; break; end
        end
        alpha = L\(L'\(y-m)) / (sn2*sn2_mult);     % alpha = inv(K_mat + sn2.*eye(N)) * (y - m)  I
        
        % GP posterior parameters
        post(s).alpha = alpha;
        post(s).sW = ones(N,1)/sqrt(sn2*sn2_mult);   % sqrt of noise precision vector
        post(s).L = L;
        post(s).sn2_mult = sn2_mult;
    end
    
elseif rank1_update    
    post_new = post;

    for s = 1:Ns

        % Extract GP hyperparameters from HYP
        ell = exp(hyp(1:D,s));
        sf2 = exp(2*hyp(D+1,s));
        sn2 = exp(2*hyp(D+2,s));
        sn2_eff = sn2*post(s).sn2_mult;            
        nf = 1 / (2*pi)^(D/2) / prod(ell);  % Kernel normalization factor

        K = sf2 * nf;
        Ks_mat = sq_dist(diag(1./ell)*X',diag(1./ell)*xstar');
        Ks_mat = sf2 * nf * exp(-Ks_mat/2);    

        [mstar, vstar] = gplite_pred(hyp(:,s),X,post(s),xstar);
        
        L = post(s).L;  % Cholesky representation

        new_L_column = linsolve(L, Ks_mat, ...
                            struct('UT', true, 'TRANSA', true)) / sn2_eff;
        post_new(s).L = [L, new_L_column; ...
                           zeros(1, size(L, 1)), ...
                           sqrt(1 + K / sn2_eff - new_L_column' * new_L_column)];

        % alpha_update now contains (K + \sigma^2 I) \ k*
        alpha_update = (L\(L'\Ks_mat)) / sn2_eff;
        post_new(s).alpha = ...
            [post(s).alpha; 0] + ...
            (mstar - ystar) / vstar * [alpha_update; -1];

        post_new(s).sW = [post(s).sW; 1/sqrt(sn2_eff)];

    end
    
    
else
    alpha = post(1).alpha;
    sW = post(1).sW;
    L = post(1).L;
    sn2_mult = post(1).sn2_mult;
end

if post_only
    % Only return GP posterior
    if rank1_update; nlZ = post_new; else; nlZ = post; end
    
else    
    % Compute negative log marginal likelihood
    nlZ = (y-m)'*alpha/2 + sum(log(diag(L))) + N*log(2*pi*sn2*sn2_mult)/2;
    
    if compute_grad
        % Compute gradient of negative log marginal likelihood
        
        dnlZ = zeros(Nhyp,1);    % allocate space for derivatives
        Q = L\(L'\eye(N))/(sn2*sn2_mult) - alpha*alpha';    % precomputed
        
        for i = 1:D                             % Grad of cov length scales
            K_temp = K_mat .* (sq_dist(X(:,i)'/ell(i)) - 1);
            dnlZ(i) = sum(sum(Q.*K_temp))/2;
        end        
        dnlZ(D+1) = sum(sum(Q.*(2*K_mat)))/2;  % Grad of cov variability
        dnlZ(D+2) = sn2*sn2_mult*trace(Q);               % Grad of GP obs noise
        
        % Gradient of mean function        
        dnlZ(D+2+(1:size(dm,2))) = -dm'*alpha;
        
    end
    
    % Compute hyperparameter prior if specified
    if ~isempty(hprior)
        
        if compute_grad
            [P,dP] = gplite_hypprior(hyp,hprior);
            nlZ = nlZ - P;
            dnlZ = dnlZ - dP;
        else
            P = gplite_hypprior(hyp,hprior);
            nlZ = nlZ - P;
        end
        
    end
end