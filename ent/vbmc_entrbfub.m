function [H,dH] = vbmc_entrbfub(vp,grad_flags,jacobian_flag)
%VBMC_ENTRBFUB Entropy of variational posterior via radial basis functions

% Check if gradient computation is required
if nargout < 2                              % No 2nd output, no gradients
    grad_flags = 0;
elseif nargin < 2 || isempty(grad_flags)    % By default compute all gradients
    grad_flags = 1;
end
if isscalar(grad_flags); grad_flags = ones(1,3)*grad_flags; end

% By default assume variational parameters were transformed (before the call)
if nargin < 3 || isempty(jacobian_flag); jacobian_flag = true; end

D = vp.D;           % Number of dimensions
K = vp.K;           % Number of components
mu(:,:) = vp.mu;
sigma(1,:) = vp.sigma;
lambda(:,1) = vp.lambda(:);

eta = 1;          % Step size for building RBF approximation

% Check which gradients are computed
if grad_flags(1); mu_grad = zeros(D,K); else, mu_grad = []; end
if grad_flags(2); sigma_grad = zeros(K,1); else, sigma_grad = []; end
if grad_flags(3); lambda_grad = zeros(D,1); else, lambda_grad = []; end

% Reshape in 4-D to allow massive vectorization
mu_4 = zeros(D,1,1,K);
mu_4(:,1,1,:) = reshape(mu,[D,1,1,K]);
sigma_4(1,1,1,:) = sigma;

% sigmalambda = bsxfun(@times, sigma_4, lambda);

lambda_t = vp.lambda(:)';       % LAMBDA is a row vector
mu_t(:,:) = vp.mu';             % MU transposed
nf = 1/(2*pi)^(D/2)/prod(lambda);  % Common normalization factor

% Entropy of non-interacting mixture
H = log(K) + 0.5*D*(1 + log(2*pi)) + D/K*sum(log(sigma)) + sum(log(lambda));

if grad_flags(2)
    sigma_grad(:) = D./(K*sigma(:));
end

if grad_flags(3)
    % Should be dividing by LAMBDA, see below
    lambda_grad(:) = ones(D,1); % 1./lambda(:);
end

mu_rescaled = bsxfun(@rdivide,vp.mu,vp.lambda(:));

if K > 1
    
    vp_w = ones(1,K)/K;
    gridmat = eta*[zeros(D,1),eye(D),-eye(D)];
    
    Xtrain_base = [];
    for k = 1:K
        Xtrain_base = [Xtrain_base,bsxfun(@plus,mu_rescaled(:,k),gridmat*vp.sigma(k))];
    end
    
    % Loop over mixture components to compute interaction terms
    for k = 1:K

        for j = 1:K
            mu_star(:,j) = (mu_rescaled(:,k).*sigma(j)^2 + mu_rescaled(:,j).*sigma(k)^2)./(sigma(k)^2 + sigma(j).^2);
        end
        sigma_star = (sigma .* sigma(k)./sqrt(sigma(k)^2 + sigma.^2));

        Xtrain_star = [];
        for j = 1:K
            Xtrain_star = [Xtrain_star,bsxfun(@plus,mu_star(:,j),gridmat*sigma_star(j))];
        end
        
        Xtrain = [Xtrain_base,Xtrain_star];        
        Ytrain = normphi(Xtrain,vp_w,mu_rescaled,vp.sigma,k);

        mu_rbf = mu_rescaled;
        sigma_rbf = sigma;

        mu_rbf = [mu_rbf, mu_star];
        sigma_rbf = [sigma_rbf, sigma_star];

        w_rbf = rbfn_train(Xtrain,Ytrain,mu_rbf,sigma_rbf);
        H = H - sum(w_rbf.*sigma_rbf.^D)*(2*pi)^(D/2);
        
        if any(grad_flags)

            if grad_flags(1)
            end

            if grad_flags(2)
            end

            if grad_flags(3)
            end
        end
    end
end

if nargout > 1
    % Correct for standard log reparameterization of SIGMA
    if jacobian_flag && grad_flags(2)
        sigma_grad = bsxfun(@times,sigma_grad, sigma(:));        
    end
    % Correct if NOT using standard log reparameterization of LAMBDA
    if ~jacobian_flag && grad_flags(3)
        lambda_grad = bsxfun(@rdivide,lambda_grad, lambda(:));        
    end
        
    dH = [mu_grad(:); sigma_grad(:); lambda_grad(:)];
end

end

%--------------------------------------------------------------------------
function [F,rho] = rbfn_eval(X,w,Mu,Sigma)
%RBFNEVAL Evaluate radial basis function network.

[D,N] = size(X);

D2 = sq_dist(Mu, X);
if isscalar(Sigma)
    D2 = D2/Sigma^2;
else
    D2 = bsxfun(@rdivide,D2,Sigma(:).^2);
end

rho = exp(-0.5*D2);

if isempty(w); F = []; else; F = w*rho; end


end
%--------------------------------------------------------------------------

function [w,Phi] = rbfn_train(Xtrain,Ytrain,Mu,Sigma)
    [~,Phi] = rbfn_eval(Xtrain,[],Mu,Sigma);
    w = ((Phi'+ 1e-6*eye(size(Phi'))) \ Ytrain(:))';
end
%--------------------------------------------------------------------------
function y = normphi(xx,w,mu,sigma,k)

[D,K] = size(mu);

logphi_k = NaN(K,size(xx,2));
d2_k = sq_dist(xx./sigma(k), mu(:,k)./sigma(k))';
logq_k = -0.5*D*log(2*pi)-D*log(sigma(k))-0.5*d2_k;

for j = 1:K
    d2_j = sq_dist(xx./sigma(j), mu(:,j)./sigma(j))';
    logq_j = -0.5*D*log(2*pi)-D*log(sigma(j))-0.5*d2_j;    
    logphi_k(j,:) = log(w(j)) - log(w(k)) + logq_j-logq_k;
end
lnZ = max(logphi_k,[],1);
logphi_k = bsxfun(@minus, logphi_k, lnZ);
phi_k = log(nansum(exp(logphi_k),1)) + lnZ;

y = w(k)*exp(logq_k).*phi_k;

end


%--------------------------------------------------------------------------
function y = normpdf(x,mu,sigma)
%NORMPDF Normal probability density function (pdf)
    y = exp(-0.5 * ((x - mu)./sigma).^2) ./ (sqrt(2*pi) .* sigma);
end