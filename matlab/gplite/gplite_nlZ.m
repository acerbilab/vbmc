function [nlZ,dnlZ,post,K_mat,Q] = gplite_nlZ(hyp,gp,hprior)
%GPLITE_NLZ Negative log marginal likelihood for lite GP regression.
%   [NLZ,DNLZ] = GPLITE_INF(HYP,GP) computes the log marginal likelihood 
%   NLZ and its gradient DNLZ for hyperparameter vector HYP. HYP is a column 
%   vector (see below). GP is a GPLITE struct.
%   
%   [NLZ,DNLZ] = GPLITE_INF(HYP,GP,HPRIOR) uses prior over hyperparameters
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
%   [NLZ,DNLZ,POST] = GPLITE_INF(...) also returns a POST structure
%   associated with the provided hyperparameters.
%
%   [NLZ,DNLZ,POST,K_MAT] = GPLITE_INF(...) also returns the computed
%   kernel matrix K_MAT.
%
%   [NLZ,DNLZ,POST,K_MAT,Q] = GPLITE_INF(...) also returns the computed
%   auxiliary matrix Q used for computing derivatives.

if nargin < 3; hprior = []; end

[N,D] = size(gp.X);         % Number of training points and dimension
[Nhyp,Ns] = size(hyp);      % Hyperparameters and samples
compute_grad = nargout > 1; % Compute gradient if required

Ncov = gp.Ncov;
Nmean = gp.Nmean;

if Nhyp ~= (Ncov+Nmean+1)
    error('gplite_nlZ:dimmismatch','Number of hyperparameters mismatched with dimension of training inputs.');
end
if compute_grad && Ns > 1
    error('gplite_nlZ:NoSampling', ...
        'Computation of the log marginal likelihood is available only for one-sample hyperparameter inputs.');
end

% Extract GP hyperparameters from HYP
ell = exp(hyp(1:D));
sf2 = exp(2*hyp(D+1));
sn2 = exp(2*hyp(Ncov+1));
sn2_mult = 1;  % Effective noise variance multiplier

nf = 1 / (2*pi)^(D/2) / prod(ell);  % Kernel normalization factor

% Evaluate mean function on training inputs
hyp_mean = hyp(Ncov+2:Ncov+1+Nmean); % Get mean function hyperparameters
if compute_grad
    [m,dm] = gplite_meanfun(hyp_mean,gp.X,gp.meanfun);
else
    m = gplite_meanfun(hyp_mean,gp.X,gp.meanfun);
end

% Compute kernel matrix K_mat
K_mat = sq_dist(diag(1./ell)*gp.X');
K_mat = sf2 * nf * exp(-K_mat/2);

if sn2 < 1e-6   % Different representation depending on noise size
    for iter = 1:10     % Cholesky decomposition until it works
        [L,p] = chol(K_mat+sn2*sn2_mult*eye(N));
        if p > 0; sn2_mult = sn2_mult*10; else; break; end
    end
    sl = 1;
    if nargout > 2
        pL = -L\(L'\eye(N));    % L = -inv(K+inv(sW^2))
        Lchol = 1;         % Tiny noise representation
    end
else
    for iter = 1:10
        [L,p] = chol(K_mat/(sn2*sn2_mult)+eye(N));
        if p > 0; sn2_mult = sn2_mult*10; else; break; end
    end
    sl = sn2*sn2_mult;
    if nargout > 2
        pL = L;
        Lchol = 0;
    end
end
alpha = L\(L'\(gp.y-m)) / sl;     % alpha = inv(K_mat + sn2.*eye(N)) * (y - m)  I

% Compute negative log marginal likelihood
nlZ = (gp.y-m)'*alpha/2 + sum(log(diag(L))) + N*log(2*pi*sl)/2;

if compute_grad
    % Compute gradient of negative log marginal likelihood

    dnlZ = zeros(Nhyp,1);    % allocate space for derivatives
    Q = L\(L'\eye(N))/sl - alpha*alpha';    % precomputed

    for i = 1:D                             % Grad of cov length scales
        K_temp = K_mat .* (sq_dist(gp.X(:,i)'/ell(i)) - 1);
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

% Output POST struct if requested
if nargout > 2
    post.hyp = hyp;
    post.alpha = alpha;
    post.sW = ones(N,1)/sqrt(sn2*sn2_mult);   % sqrt of noise precision vector
    post.L = pL;
    post.sn2_mult = sn2_mult;
    post.Lchol = Lchol;
end
