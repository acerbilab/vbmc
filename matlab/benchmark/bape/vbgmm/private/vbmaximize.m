function vbmodel = vbmaximize(X,vbmodel)
%VBMAXIMIZE Variational maximization step.

% Partially based on code written by Mo Chen:
% http://www.mathworks.com/matlabcentral/fileexchange/35362-variational-bayesian-inference-for-gaussian-mixture-model

prior = vbmodel.prior;
alpha0 = prior.alpha;
beta0 = prior.beta;
m0 = prior.m;
nu0 = prior.nu;
M0 = prior.M;
R = vbmodel.R;

nk = sum(R,1);                      % N points per component (10.51)
alpha = alpha0 + nk;                % Dirichlet posterior    (10.58)
beta = beta0 + nk;                  % Pseudo-observations    (10.60)
nu = nu0 + nk;                      % Degrees of freedom     (10.63)

m = bsxfun(@plus,beta0*m0,X*R);
m = bsxfun(@rdivide,m,beta);        % Component means        (10.61)

[d,k] = size(m);
U = zeros(d,d,k); 
logW = zeros(1,k);
r = sqrt(R');
for i = 1:k
    Xm = bsxfun(@minus,X,m(:,i));
    Xm = bsxfun(@times,Xm,r(i,:));
    m0m = m0-m(:,i);
    M = M0+Xm*Xm'+beta0*(m0m*m0m');     % equivalent to 10.62
    [temp,p] = chol(M);
    nugget = 1e-6;
    % If Cholesky decomposition fails, try again with nugget
    while p
        M = M + diag(nugget*max(M(:))*ones(1,d));
        [temp,p] = chol(M);
        nugget = nugget*2;
    end
    U(:,:,i) = temp;
    logW(i) = -2*sum(log(diag(U(:,:,i))));      
end

vbmodel.alpha = alpha;
vbmodel.beta = beta;
vbmodel.m = m;
vbmodel.nu = nu;
vbmodel.U = U;
vbmodel.logW = logW;
    
end