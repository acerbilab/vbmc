function [H,dH] = vbmc_entmcalt(vp,Ns,grad_flags,jacobian_flag)
%VBMC_ENTMCALT Alternatie Monte Carlo estimate of entropy of variational posterior and gradient

if nargin < 2 || isempty(Ns); Ns = 10; end
% Check if gradient computation is required
if nargout < 2                              % No 2nd output, no gradients
    grad_flags = 0;
elseif nargin < 3 || isempty(grad_flags)    % By default compute all gradients
    grad_flags = 1;
end
if isscalar(grad_flags); grad_flags = ones(1,3)*grad_flags; end

% By default assume variational parameters were transformed (before the call)
if nargin < 4 || isempty(jacobian_flag); jacobian_flag = true; end

D = vp.D;           % Number of dimensions
K = vp.K;           % Number of components
mu(:,:) = vp.mu;
sigma(1,:) = vp.sigma;
lambda(:,1) = vp.lambda(:);

% Check which gradients are computed
if grad_flags(1); mu_grad = zeros(D,K); else, mu_grad = []; end
if grad_flags(2); sigma_grad = zeros(K,1); else, sigma_grad = []; end
if grad_flags(3); lambda_grad = zeros(D,1); else, lambda_grad = []; end

% Reshape in 4-D to allow massive vectorization
mu_4 = zeros(D,1,1,K);
mu_4(:,1,1,:) = reshape(mu,[D,1,1,K]);
sigma_4(1,1,1,:) = sigma;

sigmalambda = bsxfun(@times, sigma_4, lambda);

lambda_t = vp.lambda(:)';       % LAMBDA is a row vector
mu_t(:,:) = vp.mu';             % MU transposed
nf = 1/(2*pi)^(D/2)/prod(lambda);  % Common normalization factor

% Entropy of non-interacting mixture
H = log(K) + 0.5*D*(1 + log(2*pi)) + D/K*sum(log(sigma)) + sum(log(lambda));

if grad_flags(2)
    sigma_grad(:) = D./(K*sigma(:));
end

if grad_flags(3)
    lambda_grad(:) = 1./lambda(:);
end

% Loop over mixture components for generating samples
for k = 1:K

    % Draw Monte Carlo samples from the k-th component
    epsilon = randn(D,1,Ns);
    xi = bsxfun(@plus, bsxfun(@times, bsxfun(@times, epsilon, lambda), sigma(k)), mu_4(:,1,1,k));
    
    Xs = reshape(xi,[D,Ns])'; 

    % Compute sum inside Phi_k
    ys = zeros(Ns,1);
    for l = 1:K
        if l == k; continue; end
        d2 = sum(bsxfun(@rdivide,bsxfun(@minus,Xs,mu_t(l,:)),sigma(l)*lambda_t).^2,2);
        nn_l = nf/sigma(l)^D*exp(-0.5*d2);
        ys = ys + nn_l;
    end
    
    % Compute N_k (denominator)
    d2 = sum(bsxfun(@rdivide,bsxfun(@minus,Xs,mu_t(k,:)),sigma(k)*lambda_t).^2,2);
    nn_k = nf/sigma(k)^D*exp(-0.5*d2);
    
    Phi_k = 1 + bsxfun(@rdivide,ys,nn_k);
    H = H - sum(log(Phi_k))/Ns/K;
    
    if any(grad_flags)
        norm_jl = bsxfun(@times, nf./(sigma_4.^D), exp(-0.5*sum(bsxfun(@rdivide, bsxfun(@minus, xi, mu_4), sigmalambda).^2,1)));
        norm_jl(:,:,:,k) = 0;
        kden(1,1,:) = 1./(Phi_k .* nn_k);
        
        zeta_l = bsxfun(@rdivide,bsxfun(@minus,xi,mu_4),sigmalambda.^2);        
        zeta2_l = bsxfun(@rdivide,bsxfun(@minus,xi,mu_4),sigmalambda).^2;        
        
        if grad_flags(1)
            for j = 1:K
                if j == k
                    mu_grad(:,j) = mu_grad(:,j) - 1/K*sum(bsxfun(@times, kden, sum(bsxfun(@times, norm_jl, -zeta_l),4)),3)/Ns;
                else
                    mu_grad(:,j) = mu_grad(:,j) - 1/K*sum(bsxfun(@times, kden .* norm_jl(1,1,:,j), zeta_l(:,:,:,j)),3)/Ns;
                end
            end
        end
        
        if grad_flags(2)
            for j = 1:K
                if j == k
                    sigma_grad(j) = sigma_grad(j) - 1/K*sum(bsxfun(@times, kden, sum(bsxfun(@times, norm_jl, ...
                        bsxfun(@minus, sum(bsxfun(@times,bsxfun(@times,lambda,epsilon),bsxfun(@minus,zeta_l(:,:,:,k),zeta_l)),1), ...
                        1/sigma(k)*(sum(zeta2_l(:,:,:,k),1)-1))),4)),3)/Ns;
                else
                    sigma_grad(j) = sigma_grad(j) - 1/K*sum(bsxfun(@times, kden, bsxfun(@times, norm_jl(1,1,:,j), ...
                        1/sigma(j)*(sum(zeta2_l(:,:,:,j),1)-1))),3)/Ns;
                end                
            end
        end
        
        if grad_flags(3)
            lambda_grad = lambda_grad - 1/K * sum(bsxfun(@times, kden, sum(bsxfun(@times, norm_jl, bsxfun(@plus, ...
                bsxfun(@times, sigma(k) * epsilon, bsxfun(@minus,zeta_l(:,:,:,k), zeta_l)), ...
                bsxfun(@times, 1./lambda, bsxfun(@minus, zeta2_l, zeta2_l(:,:,:,k))) ...
                )),4)),3)/Ns;
        end
    end
end

if grad_flags(3)
    lambda_grad = bsxfun(@times,lambda_grad,lambda);    % Reparameterization
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