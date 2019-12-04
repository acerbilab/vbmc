function [H,dH,gammasum] = entlb_vbmc(vp,grad_flags,jacobian_flag)
%ENTLB_VBMC Entropy lower bound for variational posterior

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
elseif K > BigK
    % Large number of components, avoid vectorization

    nconst = 1/(2*pi)^(D/2)/prod(lambda);
    H = 0;
    
    for n = 1:K
        sumsigma = sqrt(sigma(n)^2 + sigma.^2);
        d2 = sum(bsxfun(@rdivide, bsxfun(@minus, mu, mu(:,n)), bsxfun(@times, sumsigma, lambda)).^2,1);
        gamma = nconst./sumsigma.^D .* exp(-0.5*d2);
        gammasum = sum(w.*gamma,2);
        H = H - w(n)*log(gammasum);
    end
    
    if any(grad_flags)
        error('Entropy gradient not supported yet for large number of components.');
    end
    
else
    % Multiple components

    % Reshape in 3-D to allow vectorization
    mu_3(:,1,:) = mu;
    sigma_3(1,1,:) = sigma;
    w_3(1,1,:) = w;

    sumsigma2 = bsxfun(@plus, sigma.^2, sigma_3.^2);
    sumsigma = sqrt(sumsigma2);

    nconst = 1/(2*pi)^(D/2)/prod(lambda);

    d2 = sum(bsxfun(@rdivide, bsxfun(@minus, mu, mu_3), bsxfun(@times, sumsigma, lambda)).^2,1);
    gamma(1,:,:) = bsxfun(@times, nconst./(sumsigma.^D), exp(-0.5*d2));
    gammasum = sum(bsxfun(@times,w,gamma(1,:,:)),2);

    H = -sum(w_3 .* log(gammasum),3);

    % Compute gradient if requested
    if any(grad_flags)

        gammafrac = bsxfun(@rdivide, gamma, gammasum);
        wgammafrac = bsxfun(@times,w_3,gammafrac);
        
        if grad_flags(1)
            dmu = bsxfun(@rdivide, bsxfun(@minus, mu_3, mu), bsxfun(@times, sumsigma2, lambda.^2));
        end
        if grad_flags(2)
            dsigma = -D./sumsigma2 + 1./sumsigma2.^2 .* sum(bsxfun(@rdivide, bsxfun(@minus, mu, mu_3), lambda).^2,1);
        end

        % Loop over mixture components
        for j = 1:K
            if grad_flags(1)
                % Compute terms of gradient with respect to mu_j
                m1 = sum(bsxfun(@times, wgammafrac(:,j,:), dmu(:,j,:)),3);
                m2 = sum( bsxfun(@times, dmu(:,j,:), gamma(1,j,:).*w_3),3) ./ gammasum(j);
                mu_grad(:,j) = -w(j) * (m1 + m2);
            end

            if grad_flags(2)
                % Compute terms of gradient with respect to sigma_j
                s1 = sum(bsxfun(@times, wgammafrac(:,j,:), dsigma(:,j,:)),3);
                s2 = sum( bsxfun(@times, dsigma(:,j,:), gamma(1,j,:).*w_3),3) ./ gammasum(j);
                sigma_grad(j) = -w(j) * sigma(j) * (s1 + s2);
            end
        end

        if grad_flags(3)
            dmu2 = bsxfun(@rdivide, bsxfun(@minus, mu_3, mu).^2, bsxfun(@times, sumsigma2, lambda.^2));
            lambda_grad(:,1) = -sum(bsxfun(@rdivide, ...
                bsxfun(@times,w_3,sum(bsxfun(@times,dmu2-1,bsxfun(@times,gamma,w)),2)),...
                gammasum),3); 
            % Should be dividing by LAMBDA, see below
        end
        
        if grad_flags(4)
            w_grad(:) = -log(gammasum(:)) - sum(wgammafrac,3)';
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