function [Sigma,x0] = covcma(X,y,x0,d,frac)
%WCMA Weighted covariance matrix (inspired by CMA-ES).

if nargin < 3; x0 = []; end
if nargin < 4 || isempty(d); d = 'descend'; end
if nargin < 5 || isempty(frac); frac = 0.5; end

[N,D] = size(X);

% Compute vector weights
mu = frac*N;
weights = zeros(1,1,floor(mu));
weights(1,1,:) = log(mu+1/2)-log(1:floor(mu));
weights = weights./sum(weights);

% Compute top vectors
[~,index] = sort(y,d);

if isempty(x0)
    x0 = sum(bsxfun(@times,weights(:),X(index(1:floor(mu)),:)),1);
end

% Compute weighted covariance matrix wrt X0
topx = bsxfun(@minus,X(index(1:floor(mu)),:),x0);
Sigma = sum(bsxfun(@times,weights,topx'*topx),3);

% % Rescale covariance matrix according to mean vector length
% [E,lambda] = eig(C);
% % [sqrt(diag(lambda))',jit]
% lambda = diag(lambda) + jit.^2;
% lambda = lambda/sum(lambda);
% 
% % Square root of covariance matrix
% sigma = diag(sqrt(lambda))*E';
% 
% % Rescale by current scale (reduced)
% sigma = MeshSize*SearchFactor*sigma;
% 
% % Random draw from multivariate normal
% xs = bsxfun(@plus, x, randn(options.Nsearch,D)*sigma);

end