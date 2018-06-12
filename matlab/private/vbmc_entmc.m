function [H,dH] = vbmc_entmc(vp,Ns,grad_flags,jacobian_flag)
%VBMC_ENTMC Monte Carlo estimate of entropy of variational posterior and gradient

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
nconst = 1/(2*pi)^(D/2)/prod(lambda);

lambda_t = vp.lambda(:)';       % LAMBDA is a row vector
mu_t(:,:) = vp.mu';             % MU transposed
nf = 1/(2*pi)^(D/2)/prod(lambda)/K;  % Common normalization factor

H = 0;

% Loop over mixture components for generating samples
for j = 1:K

    % Draw Monte Carlo samples from the j-th component
    epsilon = randn(D,1,Ns);
    xi = bsxfun(@plus, bsxfun(@times, bsxfun(@times, epsilon, lambda), sigma(j)), mu_4(:,1,1,j));
    
    Xs = reshape(xi,[D,Ns])'; 

    % Compute pdf -- this block is equivalent to: ys = vbmc_pdf(Xs,vp,0);
    ys = zeros(Ns,1);
    for k = 1:K
        d2 = sum(bsxfun(@rdivide,bsxfun(@minus,Xs,mu_t(k,:)),sigma(k)*lambda_t).^2,2);
        nn = nf/sigma(k)^D*exp(-0.5*d2);
        ys = ys + nn;
    end
        
    H = H - sum(log(ys))/Ns/K;
    
    if any(grad_flags)    
        % Full mixture (for sample from the j-th component)
        norm_jl = bsxfun(@times, nconst./(sigma_4.^D), exp(-0.5*sum(bsxfun(@rdivide, bsxfun(@minus, xi, mu_4), sigmalambda).^2,1)));
        q_j = 1/K * sum(norm_jl,4);
        
        % Compute sum for gradient wrt mu
        lsum = sum(bsxfun(@times,bsxfun(@rdivide, bsxfun(@minus, xi, mu_4), sigmalambda.^2), norm_jl),4);

        if grad_flags(1)
            mu_grad(:,j) = sum(bsxfun(@rdivide, lsum, q_j),3) / K^2 / Ns;
        end
        
        if grad_flags(2)
            % Compute sum for gradient wrt sigma
            isum = sum(bsxfun(@times,lsum,bsxfun(@times, epsilon, lambda)),1);
            sigma_grad(j) = sum(bsxfun(@rdivide, isum, q_j),3) / K^2 / Ns;
        end
        
        if grad_flags(3)
            % Should be dividing by LAMBDA, see below
            lambda_grad = lambda_grad + sum(bsxfun(@times, lsum, bsxfun(@rdivide, sigma(j)*epsilon,q_j)),3) / K^2 / Ns;
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