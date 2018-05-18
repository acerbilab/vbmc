function [gsKL,Mean,Cov,Mode] = ComputeAlgoStats(X,y,probstruct,Ns_moments)
%COMPUTEALGOSTATS Compute GP model-based statistics from given training set.
    
if nargin < 4 || isempty(Ns_moments); Ns_moments = 2e4; end

% Add prior to y if not previously added
if ~probstruct.AddLogPrior
    lnp = infbench_lnprior(X,probstruct);
    y = y + lnp;
end

% Compute Gaussianized symmetric KL-divergence with ground truth
gp.X = X;
gp.y = y;
gp.meanfun = 4; % Negative quadratic mean fcn
    
xx = gplite_sample(gp,Ns_moments);
Mean = mean(xx,1);
Cov = cov(xx);
[kl1,kl2] = mvnkl(Mean,Cov,probstruct.Post.Mean,probstruct.Post.Cov);
gsKL = 0.5*(kl1 + kl2);

if nargout > 3
    Mode = gplite_fmin(gp,[],1);    % Max flag - finds maximum
end

end