function [H,dH] = entub_vbmc(vp,grad_flags,jacobian_flag)
%ENTUB_VBMC Entropy upper bound for variational posterior

% Uses entropy upper bound of multivariate normal approximation

% Check if gradient computation is required
if nargout < 2                              % No 2nd output, no gradients
    grad_flags = false;
elseif nargin < 2 || isempty(grad_flags)    % By default compute all gradients
    grad_flags = true;
end
if isscalar(grad_flags); grad_flags = ones(1,4)*grad_flags; end

% By default assume variational parameters were transformed (before the call)
if nargin < 3 || isempty(jacobian_flag); jacobian_flag = true; end

D = vp.D;           % Number of dimensions
K = vp.K;           % Number of components
mu(:,:) = vp.mu;
sigma(1,:) = vp.sigma;
lambda(:,1) = vp.lambda(:);
w(1,:) = vp.w;

% Check which gradients are computed
if grad_flags(1); mu_grad = zeros(D,K); dS_mu = zeros(D,D,K); else, mu_grad = []; end
if grad_flags(2); sigma_grad = zeros(K,1); else, sigma_grad = []; end
if grad_flags(3); lambda_grad = zeros(D,1); else, lambda_grad = []; end
if grad_flags(4); w_grad = zeros(K,1); dS_w = zeros(D,D,K); else, w_grad = []; end

if K == 1
    % Entropy of single component, uses exact expression
    H = 0.5*D*(1 + log(2*pi)) + D*sum(log(sigma)) + sum(log(lambda));

    if grad_flags(2)
        sigma_grad(:) = D./sigma(:);
    end

    if grad_flags(3)
        % Should be dividing by LAMBDA, see below
        lambda_grad(:) = ones(D,1); % 1./lambda(:);
    end
    
    if grad_flags(4)
        w_grad = 0;
    end
else
    
    Mu = sum(bsxfun(@times,vp.w,vp.mu),2);
    Sigma = zeros(D,D);
    delta_mu = bsxfun(@minus,mu,Mu);    
    for k = 1:K
        S_k = diag((lambda*sigma(k)).^2) + delta_mu(:,k)*delta_mu(:,k)';
        Sigma = Sigma + w(k)*S_k;
        if grad_flags(4); dS_w(:,:,k) = S_k; end        
    end
    L = chol(Sigma);
    
    H = 0.5*D*(log(2*pi) + 1) + sum(log(diag(L)));
        
     if any(grad_flags)
         invK = L\(L'\eye(D));
         
         if grad_flags(1)
             for k = 1:K
                 mu_grad((1:D)+(k-1)*D) = 0.5*w(k).*(sum(bsxfun(@times,invK,delta_mu(:,k)'),2) + sum(bsxfun(@times,invK,delta_mu(:,k)),1)');
             end
         end
         
         if grad_flags(2)
             Q = sum(sum(invK.*diag(lambda.^2)));
             sigma_grad(:) = Q*(w.*sigma); 
         end
         
         if grad_flags(3)
             lambda_grad(:) = diag(invK).*lambda.^2*sum(w.*(sigma.^2));
         end
         
         if grad_flags(4)
             for k = 1:K
                w_grad(k) = 0.5*sum(sum(invK.*dS_w(:,:,k)));
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
    % Correct for standard softmax reparameterization of W
    if jacobian_flag && grad_flags(4)
        eta_sum = sum(exp(vp.eta));
        J_w = bsxfun(@times,-exp(vp.eta)',exp(vp.eta)/eta_sum^2) + diag(exp(vp.eta)/eta_sum);
        w_grad = J_w*w_grad;
    end
    dH = [mu_grad(:); sigma_grad(:); lambda_grad(:); w_grad(:)];
end

end