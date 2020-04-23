function [fess,X] = fess_vbmc(vp,gp,X)
%FESS_VBMC Compute fractional effective sample size through importance sampling

if nargin < 3 || isempty(X); X = 100; end

% If a single number is passed, take it as the number of samples
if numel(X) == 1
    N = X;
    X = vbmc_rnd(vp,N,0);
else
    N = size(X,1);
end

% Can directly pass the estimated GP means instead of the full GP
if isstruct(gp)
    [~,~,fbar] = gplite_pred(gp,X,[],[],0,0);    
else
    fbar = mean(gp,2);
end

if size(fbar,1) ~= size(X,1)
    error('Mismatch between number of samples from VP and GP.');
end
                
% Compute effective sample size (ESS) with importance sampling
vlnpdf = max(vbmc_pdf(vp,X,0,1),log(realmin));
logw = fbar - vlnpdf;
w = exp(logw - max(logw));
w = w/sum(w);
fess = 1/sum(w.^2) / N; % fractional ESS

end