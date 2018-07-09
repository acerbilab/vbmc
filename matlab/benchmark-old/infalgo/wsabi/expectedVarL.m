function [NEV] = expectedVarL(xs, lambda, VV, lHatD, xxx, invKxx, noise, bb, BB)
% Active sampling loss function for wsabi_L

xs = xs';
xxxScaled   = xxx .* repmat(sqrt(1./VV),length(xxx(:,1)),1);

distxsxx    = ...
pdist2_squared_fast(xs.*repmat(sqrt(1./VV),length(xs(:,1)),1),xxxScaled);

Kxsx        = lambda^2 * (1/(prod(2*pi*VV).^0.5)) * exp(-0.5*distxsxx);
          
Kxsxs = lambda^2 * (1/(prod(2*pi*VV).^0.5))*(1+noise)*ones(length(xs(:,1)),1);

varPred = (Kxsxs - diag(Kxsx*(invKxx * Kxsx')));

l_0 = (Kxsx*(invKxx*lHatD));
if any(varPred <= 0) || any(isnan(xs(:))) || any(~isreal(varPred))
    %keyboard;
    varPred = 0;
end

priorWeighting = mvnpdf(xs,bb,BB);

NEV = -(l_0.^2 .* varPred) .* priorWeighting.^2; %Negative expected variance.


end
