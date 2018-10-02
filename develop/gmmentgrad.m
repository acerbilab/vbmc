function g = gmmentgrad(mu,sigma,ell)
%GMMENTGRAD Estimate of gradient of approximate entropy of Gaussian mixture

[D,K] = size(mu);   % Number of dimensions and mixture components

mu_grad = zeros(D,K);
sigma_grad = zeros(1,K);

% Reshape in 3-D to allow vectorization
mu_2(:,:) = reshape(mu, [D,1,1,K]);
sigma_2(1,:) = sigma;
mu_3(:,1,:) = mu_2;
sigma_3(1,1,:) = sigma;

sumsigma2 = bsxfun(@plus, sigma_2.^2, sigma_3.^2);
sumsigma = sqrt(sumsigma2);

nconst = 1/(2*pi)^(D/2)/prod(ell);

gamma(1,:,:) = bsxfun(@times, nconst./(sumsigma.^D), exp(-0.5*sum(bsxfun(@rdivide, bsxfun(@minus, mu_2, mu_3), bsxfun(@times, sumsigma, ell)).^2,1)));
gammasum = sum(gamma(1,:,:),2);
gammafrac = bsxfun(@rdivide, gamma, gammasum);

dmu = bsxfun(@rdivide, bsxfun(@minus, mu_3, mu_2), bsxfun(@times, sumsigma2, ell.^2));
dsigma = -D./sumsigma2 + 1./sumsigma2.^2 .* sum(bsxfun(@rdivide, bsxfun(@minus, mu_2, mu_3), ell).^2,1);

% Loop over mixture components
for j = 1:K
    % Compute terms of gradient with respect to mu_j
    m1 = sum(bsxfun(@times, gammafrac(:,j,:), dmu(:,j,:)),3);
    m2 = sum( bsxfun(@times, dmu(:,j,:), gamma(1,j,:)),3) ./ gammasum(j);
    mu_grad(:,j) = -1/K * (m1 + m2);
    
    % Compute terms of gradient with respect to sigma_j
    s1 = sum(bsxfun(@times, gammafrac(:,j,:), dsigma(:,j,:)),3);
    s2 = sum( bsxfun(@times, dsigma(:,j,:), gamma(1,j,:)),3) ./ gammasum(j);
    sigma_grad(j) = -sigma(j)/K * (s1 + s2);
end

g = [mu_grad(:); sigma_grad(:)];

end