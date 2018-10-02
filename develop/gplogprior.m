function [P,dP] = gplogprior(hyp,X,y,noisesize)
%GPPRIOR Log prior over GP hyperparameters

if nargin < 4 || isempty(noisesize); noisesize = [1e-2 1]; end

[N,D] = size(X);        % Number of training points and dimension
Nhyp = size(hyp,2);     % Multiple hyperparameter samples?

if size(hyp,1) ~= D + 3
    error('gplogprior:dimmismatch','Number of hyperparameters mismatched with dimension of training inputs.');
end

if Nhyp > 1
    error('gplogprior:nosampling','GP inference with log priors is available only for one-sample hyperparameter inputs.');
end

compute_grad = nargout > 1; % Compute gradient if required

% Extract GP hyperparameters from HYP
m = hyp(1);
ell = exp(hyp(2:D+1));
sf2 = exp(2*hyp(D+2));

% Gaussian prior on noise (we believe it is small)
lnsn = hyp(D+3);
if numel(noisesize) > 1 && isfinite(noisesize(2))
    lnsn_var = noisesize(2)^2;
else
    lnsn_var = 1;
end

m_scaling = 0;

P = -m*m_scaling - 0.5*log(2*pi*lnsn_var) - 0.5*(lnsn-log(noisesize(1)))^2/lnsn_var;

if compute_grad
    dP = zeros(size(hyp));    
    dP(1) = -1*m_scaling;
    dP(D+3) = -(lnsn-log(noisesize(1)))/lnsn_var;
end



end