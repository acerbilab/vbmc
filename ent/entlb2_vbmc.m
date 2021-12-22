function [H,dH,gammasum] = entlb2_vbmc(vp,grad_flags,jacobian_flag,alpha)
%ENTLB2_VBMC Alternative entropy lower bound for variational posterior

% Kolchinsky and Brendan D. Tracey (2017). Entropy.
BigK = Inf; % Large number of components

% Uses entropy lower bound of Gershman et al. (2012)

% Check if gradient computation is required
if nargout < 2                              % No 2nd output, no gradients
    grad_flags = false;
elseif nargin < 2 || isempty(grad_flags)    % By default compute all gradients
    grad_flags = true;
end
if isscalar(grad_flags); grad_flags = ones(1,4)*grad_flags; end

% By default assume variational parameters were transformed (before the call)
if nargin < 3 || isempty(jacobian_flag); jacobian_flag = true; end

% By default use Bhattacharyya distance
if nargin < 4 || isempty(alpha); alpha = 0.5; end


D = vp.D;           % Number of dimensions
K = vp.K;           % Number of components
mu(:,:) = vp.mu;
sigma(1,:) = vp.sigma;
lambda(:,1) = vp.lambda(:);
w(1,:) = vp.w;

% Check which gradients are computed
if grad_flags(1); mu_grad = zeros(D,K); else, mu_grad = []; end
if grad_flags(2); sigma_grad = zeros(K,1); else, sigma_grad = []; end
if grad_flags(3); lambda_grad = zeros(D,1); else, lambda_grad = []; end
if grad_flags(4); w_grad = zeros(K,1); else, w_grad = []; end

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
    % Multiple components

    % Reshape in 3-D to allow vectorization
    mu_3(:,1,:) = mu;
    sigma_3(1,1,:) = sigma;
    w_3(1,1,:) = w;    
    sum_loglambda = sum(log(lambda),1);
    
    % Weighted sum of individual components entropy
    H = D*sum(w.*log(sigma)) + sum_loglambda + 0.5*D*(log(2*pi) + 1);
    
    % Bhattacharyya distance
    if alpha == 0.5    
        sigma2_tilde = bsxfun(@times,bsxfun(@plus,0.5*sigma.^2,0.5*sigma_3.^2),lambda.^2);    
        C = bsxfun(@minus, 0.125*sum(bsxfun(@minus,mu,mu_3).^2 ./ sigma2_tilde,1) ...
            + 0.5*sum(log(sigma2_tilde),1), ...
            bsxfun(@plus, sum_loglambda + 0.5*D*log(sigma), 0.5*D*log(sigma_3)));
    else
        sigma2_tilde = bsxfun(@times,bsxfun(@plus,(1-alpha)*sigma.^2,alpha*sigma_3.^2),lambda.^2);    
        C = bsxfun(@minus, 0.5*(1-alpha)*alpha*sum(bsxfun(@minus,mu,mu_3).^2 ./ sigma2_tilde,1) ...
            + 0.5*sum(log(sigma2_tilde),1), ...
            bsxfun(@plus, sum_loglambda + (1-alpha)*D*log(sigma), alpha*D*log(sigma_3)));        
    end
    
    % No need to use logsumexp here, we know that max(-C(i,:)) = 0 for any i
    H = H - sum(bsxfun(@times, w, log(sum(bsxfun(@times, w_3, exp(-C)),3))),2);
        
    % Compute gradient if requested
    if any(grad_flags)

        error('Entropy gradient not supported yet for large number of components.');
        
%         gammafrac = bsxfun(@rdivide, gamma, gammasum);
%         wgammafrac = bsxfun(@times,w_3,gammafrac);
%         
%         if grad_flags(1)
%             dmu = bsxfun(@rdivide, bsxfun(@minus, mu_3, mu), bsxfun(@times, sumsigma2, lambda.^2));
%         end
%         if grad_flags(2)
%             dsigma = -D./sumsigma2 + 1./sumsigma2.^2 .* sum(bsxfun(@rdivide, bsxfun(@minus, mu, mu_3), lambda).^2,1);
%         end
% 
%         % Loop over mixture components
%         for j = 1:K
%             if grad_flags(1)
%                 % Compute terms of gradient with respect to mu_j
%                 m1 = sum(bsxfun(@times, wgammafrac(:,j,:), dmu(:,j,:)),3);
%                 m2 = sum( bsxfun(@times, dmu(:,j,:), gamma(1,j,:).*w_3),3) ./ gammasum(j);
%                 mu_grad(:,j) = -w(j) * (m1 + m2);
%             end
% 
%             if grad_flags(2)
%                 % Compute terms of gradient with respect to sigma_j
%                 s1 = sum(bsxfun(@times, wgammafrac(:,j,:), dsigma(:,j,:)),3);
%                 s2 = sum( bsxfun(@times, dsigma(:,j,:), gamma(1,j,:).*w_3),3) ./ gammasum(j);
%                 sigma_grad(j) = -w(j) * sigma(j) * (s1 + s2);
%             end
%         end
% 
%         if grad_flags(3)
%             dmu2 = bsxfun(@rdivide, bsxfun(@minus, mu_3, mu).^2, bsxfun(@times, sumsigma2, lambda.^2));
%             lambda_grad(:,1) = -sum(bsxfun(@rdivide, ...
%                 bsxfun(@times,w_3,sum(bsxfun(@times,dmu2-1,bsxfun(@times,gamma,w)),2)),...
%                 gammasum),3); 
%             % Should be dividing by LAMBDA, see below
%         end
%         
%         if grad_flags(4)
%             w_grad(:) = -log(gammasum(:)) - sum(wgammafrac,3)';
%         end
        
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