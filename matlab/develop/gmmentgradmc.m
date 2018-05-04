function g = gmmentgradmc(mu,sigma,ell,Ns)
%GMMENTGRADMC Monte Carlo estimate of gradient of entropy of Gaussian mixture

if nargin < 4 || isempty(Ns); Ns = 10; end

[D,K] = size(mu);   % Number of dimensions and mixture components

mu_grad = zeros(D,K);
sigma_grad = zeros(1,K);

% Reshape in 4-D to allow massive vectorization
mu_4(:,1,1,:) = reshape(mu, [D,1,1,K]);
sigma_4(1,1,1,:) = sigma;

sigmarho = bsxfun(@times, sigma_4, ell);
nconst = 1/(2*pi)^(D/2)/prod(ell);

% Loop over mixture components for generating samples
for j = 1:K

    % Draw Monte Carlo samples from the j-th component
    epsilon = randn(D,1,Ns);
    xi = bsxfun(@plus, bsxfun(@times, bsxfun(@times, epsilon, ell), sigma(j)), mu_4(:,1,1,j));
    
    % Compute sum for gradient wrt mu
    norm_jl = bsxfun(@times, nconst./(sigma_4.^D), exp(-0.5*sum(bsxfun(@rdivide, bsxfun(@minus, xi, mu_4), sigmarho).^2,1)));        
    lsum = sum(bsxfun(@times,bsxfun(@rdivide, bsxfun(@minus, xi, mu_4), sigmarho.^2), norm_jl),4);
    
    % Full mixture (for sample from the j-th component)
    q_j = 1/K * sum(norm_jl,4);
            
    % Compute sum for gradient wrt sigma
    isum = sum(bsxfun(@times,lsum,bsxfun(@times, epsilon, ell)),1);
    
    mu_grad(:,j) = sum(bsxfun(@rdivide, lsum, q_j),3) / K^2 / Ns;
    sigma_grad(j) = sum(bsxfun(@rdivide, isum, q_j),3) / K^2 / Ns;
end

g = [mu_grad(:); sigma_grad(:)];

end