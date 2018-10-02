function [H,dH] = vbmc_entrbf(vp,grad_flags,jacobian_flag)
%VBMC_ENTRBF Entropy of variational posterior via radial basis functions

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
tol = 1e-8;

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

else
    % Multiple components

    H = 0;
    mu_rescaled = bsxfun(@rdivide,vp.mu,vp.lambda(:));
    
    vp_w = ones(1,K)/K;
    gridmat = eta*[zeros(D,1),eye(D),-eye(D)];
    shiftmat = [-2.57235211094289 -2.16610675289233 -1.95566143558817 -1.80735419679911 -1.6906216295849 -1.59321881802305 -1.50894385503804 -1.43420015968638 -1.36670697180796 -1.30492263775272 -1.24775385535132 -1.19439566356816 -1.14423726510021 -1.09680356209351 -1.05171725299848 -1.0086733576468 -0.967421566101701 -0.927753685357425 -0.889494507530634 -0.852495034274694 -0.816627360848605 -0.781780752765072 -0.747858594763302 -0.714775988103151 -0.68245783666933 -0.650837306444477 -0.619854573565494 -0.589455797849778 -0.559592274227433 -0.530219725824228 -0.501297710767729 -0.472789120992267 -0.444659755988672 -0.416877957995407 -0.389414297852144 -0.362241302844737 -0.335333219514398 -0.308665805694934 -0.282216147062508 -0.255962494294065 -0.229884117579232 -0.203961175751314 -0.17817459772241 -0.152505974246244 -0.126937458305643 -0.101451672641948 -0.0760316231203884 -0.0506606167658766 -0.0253221834133462 0.0253221834133464 0.0506606167658766 0.0760316231203884 0.101451672641948 0.126937458305643 0.152505974246244 0.17817459772241 0.203961175751314 0.229884117579232 0.255962494294065 0.282216147062508 0.308665805694934 0.335333219514398 0.362241302844737 0.389414297852145 0.416877957995407 0.444659755988672 0.472789120992268 0.501297710767729 0.530219725824228 0.559592274227433 0.589455797849778 0.619854573565494 0.650837306444477 0.68245783666933 0.71477598810315 0.747858594763302 0.781780752765073 0.816627360848605 0.852495034274694 0.889494507530634 0.927753685357426 0.967421566101701 1.0086733576468 1.05171725299848 1.09680356209351 1.14423726510021 1.19439566356816 1.24775385535132 1.30492263775272 1.36670697180796 1.43420015968638 1.50894385503804 1.59321881802305 1.6906216295849 1.80735419679911 1.95566143558817 2.16610675289233 2.5723521109429];    
    
    Xtrain_base = [];
    for k = 1:K
        Xtrain_base = [Xtrain_base,bsxfun(@plus,mu_rescaled(:,k),gridmat*vp.sigma(k))];
    end
    
    % Loop over mixture components
    for k = 1:K

        Xtrain_k = [];
        for d = 1:D
            v = zeros(D,1);
            v(d) = sigma(k)*eta;
            Xtrain_k = [Xtrain_k, bsxfun(@plus, mu_rescaled(:,k), bsxfun(@times,v,shiftmat))];
        end
        
        for j = 1:K
            mu_star(:,j) = (mu_rescaled(:,k).*sigma(j)^2 + mu_rescaled(:,j).*sigma(k)^2)./(sigma(k)^2 + sigma(j).^2);
        end
        sigma_star = (sigma .* sigma(k)./sqrt(sigma(k)^2 + sigma.^2));
        
        Xtrain_star = mu_star;
        %Xtrain_star = [];
        %for j = 1:K
        %    Xtrain_star = [Xtrain_star,bsxfun(@plus,mu_star(:,j),gridmat*sigma_star(j))];
        %end
        
        Xtrain = [Xtrain_base,Xtrain_k,Xtrain_star];       
        idx = cleank(Xtrain,mu_rescaled(:,k),5*sigma(k));
        Xtrain = Xtrain(:,idx);        
        
        Ytrain = logp(Xtrain,vp_w,mu_rescaled,vp.sigma,tol);

        mu_rbf = mu_rescaled;
        sigma_rbf = sigma;

        mu_rbf = [mu_rbf, mu_star];
        sigma_rbf = [sigma_rbf, sigma_star]*sqrt(2*pi);

        idx = cleank(mu_rbf,mu_rescaled(:,k),5*sigma(k));
        mu_rbf = mu_rbf(:,idx);
        sigma_rbf = sigma_rbf(idx);
        
        d2 = sum(bsxfun(@minus,mu_rbf,mu(:,k)).^2,1);
        sigmatilde2 = sigma(k)^2 + sigma_rbf.^2;
        npdf = 1./(2*pi*sigmatilde2).^(D/2).*exp(-0.5*d2./sigmatilde2);        
        w_rbf = rbfn_train(Xtrain,Ytrain,mu_rbf,sigma_rbf);
        H = H - vp_w(k)*sum(w_rbf.*sigma_rbf.^D.*npdf)*(2*pi)^(D/2);
                
        if any(grad_flags)

            if grad_flags(1)
            end

            if grad_flags(2)
            end

            if grad_flags(3)
            end
        end
    end
    
    H = H - log(tol) + sum(log(lambda));
    
    
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
function y = logp(xx,w,mu,sigma,tol)

[D,K] = size(mu);
y = zeros(1,size(xx,2));
for j = 1:K
    d2_j = sq_dist(xx./sigma(j), mu(:,j)./sigma(j))';
    y = y + w(j)/sigma(j)^D*exp(-0.5*d2_j);
end
%lnZ = max(logq_j,[],1);
%logq_j = bsxfun(@minus, logq_j, lnZ);
%y = log(nansum(exp(logq_j) + tol,1)) + lnZ - log(tol);
y = log(y/(2*pi)^(D/2) + tol) - log(tol);

end


%--------------------------------------------------------------------------
function idx = cleank(mu,mu0,delta0)

idx = all(bsxfun(@le, mu, mu0 + delta0),1) ...
    & all(bsxfun(@ge, mu, mu0 - delta0),1);

end
%--------------------------------------------------------------------------
function y = normpdf(x,mu,sigma)
%NORMPDF Normal probability density function (pdf)
    y = exp(-0.5 * ((x - mu)./sigma).^2) ./ (sqrt(2*pi) .* sigma);
end