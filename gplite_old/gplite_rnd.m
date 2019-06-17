function [Fstar,Ystar] = gplite_rnd(gp,Xstar)
%GPLITE_RND Draw a random function from Gaussian process.
%   FSTAR = GPLITE_RND(GP,XSTAR) draws a random function from GP, evaluated 
%   at XSTAR.
%
%   [FSTAR,YSTAR] = GPLITE_RND(GP,XSTAR) adds observation noise to the
%   drawn function.
%
%   See also GPLITE_POST, GPLITE_PRED.

[N,D] = size(gp.X);            % Number of training points and dimension
Ns = numel(gp.post);           % Hyperparameter samples
Nstar = size(Xstar,1);         % Number of test inputs

Ncov = gp.Ncov;
Nmean = gp.Nmean;

% Draw from hyperparameter samples
s = randi(Ns);

hyp = gp.post(s).hyp;

alpha = gp.post(s).alpha;
L = gp.post(s).L;
Lchol = gp.post(s).Lchol;
sW = gp.post(s).sW;
sn2_mult = gp.post(s).sn2_mult;

% Extract GP hyperparameters from HYP
ell = exp(hyp(1:D));
sf2 = exp(2*hyp(D+1));
sn2 = exp(2*hyp(D+2));

% Compute GP mean function
hyp_mean = hyp(Ncov+2:Ncov+1+Nmean);                % Get mean function hyperparameters
mstar = gplite_meanfun(hyp_mean,Xstar,gp.meanfun);  % GP mean evaluated at test points

% Compute kernel matrix K_mat
Kstar_mat = sq_dist(diag(1./ell)*Xstar');
Kstar_mat = sf2 * exp(-Kstar_mat/2);

if N > 0    
    % Compute cross-kernel matrix Ks_mat
    Ks_mat = sq_dist(diag(1./ell)*gp.X',diag(1./ell)*Xstar');
    Ks_mat = sf2 * exp(-Ks_mat/2);
    
    fmu = mstar + Ks_mat'*alpha;            % Conditional mean

    if Lchol
        V = L'\(repmat(sW,[1,Nstar]).*Ks_mat);
        C = Kstar_mat - V'*V;       % predictive variances
    else
        LKs = L*Ks_mat;
        C = Kstar_mat + Ks_mat'*LKs;
    end
else    
    fmu = mstar;                            % No data, draw from prior
    C = Kstar_mat + eps*eye(Nstar);
end 

C = (C + C')/2;   % Enforce symmetry if lost due to numerical errors

% Draw random function
T = robustchol(C); % CHOL usually crashes, this is more stable
Fstar = T' * randn(size(T,1),1) + fmu;

% Add observation noise
if nargout > 1
    Ystar = Fstar + sqrt(sn2*sn2_mult).*randn(size(fmu));
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [T,p] = robustchol(Sigma)
%ROBUSTCHOL  Cholesky-like decomposition for covariance matrix.

[n,m] = size(Sigma);    % Should be square
[T,p] = chol(Sigma);

if p > 0
    [U,D] = eig((Sigma+Sigma')/2);

    [~,maxidx] = max(abs(U),[],1);
    negidx = (U(maxidx + (0:n:(m-1)*n)) < 0);
    U(:,negidx) = -U(:,negidx);

    D = diag(D);
    tol = eps(max(D)) * length(D);
    t = (abs(D) > tol);
    D = D(t);
    p = sum(D<0); % negative eigenvalues

    if p == 0
        T = diag(sqrt(D)) * U(:,t)';
    else
        T = zeros(0,'like',Sigma);
    end
end


end
    