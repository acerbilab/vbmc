function [lp,dlp] = gplite_hypprior(hyp,hprior)
%GPLITE_HYPPRIOR Log priors for hyperparameters of lite GP regression.

compute_grad = nargout > 1; % Compute gradient if required

[Nhyp,Ns] = size(hyp);      % Hyperparameters and samples
if Ns > 1
    error('gplite_hypprior:nosampling', ...
        'Hyperparameter log priors are available only for one-sample hyperparameter inputs.');
end

lp = 0;
if compute_grad; dlp = zeros(Nhyp,1); end

mu = hprior.mu(:);
sigma = abs(hprior.sigma(:));
if ~isfield(hprior,'df') || isempty(hprior.df)  % Degrees of freedom
    df = 7*ones(Nhyp,1); % ~ from Gelman et al. (2009)
else
    df = hprior.df(:);
end

uidx = ~isfinite(mu) | ~isfinite(sigma);                        % Uniform
gidx = ~uidx & (df == 0 | ~isfinite(df)) & isfinite(sigma);     % Gaussian
tidx = ~uidx & df > 0 & isfinite(df);                           % Student's t

% Quadratic form
z2 = zeros(Nhyp,1);
z2(gidx | tidx) = ((hyp(gidx | tidx) - mu(gidx | tidx))./sigma(gidx | tidx)).^2;

% Gaussian prior
if any(gidx)
    lp = lp -0.5*sum(log(2*pi*sigma(gidx).^2) + z2(gidx));
    if compute_grad
        dlp(gidx) = -(hyp(gidx) - mu(gidx))./sigma(gidx).^2;
    end
end    

% Student's t prior
if any(tidx)
    lp = lp + sum(gammaln(0.5*(df(tidx)+1)) - gammaln(0.5*df(tidx)) - 0.5*log(pi*df(tidx)) ...
        - log(sigma(tidx)) - 0.5*(df(tidx)+1).*log1p(z2(tidx)./df(tidx)));
    if compute_grad
        dlp(tidx) = -(df(tidx)+1)./df(tidx)./(1+z2(tidx)./df(tidx)).*(hyp(tidx) - mu(tidx))./sigma(tidx).^2;
    end
end