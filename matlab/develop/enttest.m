function [bias,H,H_alt] = enttest

plotflag = nargout == 0;

K = 6;

w = exp(randn(1,K));
w = w ./ sum(w);
mu = 200*rand(1,K);
sigma = exp(0.2*randn(1,K));

% w, mu, sigma


xx = linspace(min(mu)-max(sigma)*5, max(mu)+max(sigma)*5, 1e3);
dx = xx(2)-xx(1);

p = zeros(size(xx));
for k = 1:K
    p = p + w(k)*normpdf(xx,mu(k),sigma(k));
end

if plotflag
    hold off;
    plot(xx, -p.*log(p), 'k'); hold on;
end

H = qtrapz(-p.*log(p))*dx;

y = normphi(xx,w,mu,sigma);
if plotflag
    plot(xx, y, 'k--'); hold on;
end

H_p = 0;

tic
y_rbf = zeros(size(xx));
for k = 1:K
    eta = 1.0;
    Xtrain = [mu(:); mu(:) + eta*sigma(:); mu(:) - eta*sigma(:)]';

    mu_star = (mu(k).*sigma.^2 + mu.*sigma(k)^2)./(sigma(k)^2 + sigma.^2);
    sigma_star = sqrt(sigma .* sigma(k)./sqrt(sigma(k)^2 + sigma.^2));
    
    Xtrain = [Xtrain, mu_star, mu_star + eta*sigma_star, mu - eta*sigma_star];
    Ytrain = normphi(Xtrain,w,mu,sigma,k)';
        
    mu_rbf = mu;
    sigma_rbf = repmat(sigma,[1,1]);
    
    mu_rbf = [mu_rbf, mu_star];
    sigma_rbf = [sigma_rbf, sigma_star];
    
    w_rbf = rbfn_train(Xtrain,Ytrain,mu_rbf,sigma_rbf);
    y_rbf = y_rbf + rbfn_eval(xx,w_rbf,mu_rbf,sigma_rbf);
    
    H_p = H_p + sum(w_rbf.*sigma_rbf)*sqrt(2*pi);
end
toc

if plotflag
    plot(xx, y_rbf, 'r:'); hold on;
end

%H
H_alt = -sum(w.*log(w)) + sum(w.*log(sigma.*sqrt(2*pi*exp(1)))) - H_p;

bias = H_alt - H;

end

function y = normphi(xx,w,mu,sigma,k_range)

K = numel(w);
if nargin < 5 || isempty(k_range); k_range = 1:K; end

y = zeros(size(xx));
for k = k_range
    logphi_k = NaN(K,numel(xx));
    for j = 1:K
        logphi_k(j,:) = log(w(j)) - log(w(k)) + normlogpdf(xx,mu(j),sigma(j))-normlogpdf(xx,mu(k),sigma(k));
    end
    lnZ = max(logphi_k,[],1);
    logphi_k = bsxfun(@minus, logphi_k, lnZ);
    phi_k = log(nansum(exp(logphi_k),1)) + lnZ;
    
    y = y + w(k)*normpdf(xx,mu(k),sigma(k)).*phi_k;
end

end

function [F,rho] = rbfn_eval(X,w,Mu,Sigma)
%RBFNEVAL Evaluate radial basis function network.

if nargin < 4 || isempty(Sigma); Sigma = 1; end

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

function [w,Phi] = rbfn_train(Xtrain,Ytrain,Mu,Sigma)
    [~,Phi] = rbfn_eval(Xtrain,[],Mu,Sigma);
    w = ((Phi'+ 1e-6*eye(size(Phi'))) \ Ytrain(:))';
end

