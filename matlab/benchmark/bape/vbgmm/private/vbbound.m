function L = vbbound(X,vbmodel)
%VBBOUND Compute variational lower bound.

% Partially based on code written by Mo Chen:
% http://www.mathworks.com/matlabcentral/fileexchange/35362-variational-bayesian-inference-for-gaussian-mixture-model

prior = vbmodel.prior;
alpha0 = prior.alpha;
beta0 = prior.beta;
nu0 = prior.nu;
logW0 = prior.logW;
alpha = vbmodel.alpha; 
beta = vbmodel.beta; 
nu = vbmodel.nu;
logW = vbmodel.logW;
R = vbmodel.R;
logR = vbmodel.logR;
[d,n] = size(X);
k = size(R,2);

Epz = 0;
Eqz = dot(R(:),logR(:));
logCalpha0 = gammaln(k*alpha0)-k*gammaln(alpha0);
Eppi = logCalpha0;
logCalpha = gammaln(sum(alpha))-sum(gammaln(alpha));
Eqpi = logCalpha;
Epmu = 0.5*d*k*log(beta0);
Eqmu = 0.5*d*sum(log(beta));
logB0 = -0.5*nu0*(logW0+d*log(2))-logmvgamma(0.5*nu0,d);
EpLambda = k*logB0;
logB =  -0.5*nu.*(logW+d*log(2))-logmvgamma(0.5*nu,d);
EqLambda = sum(logB);
EpX = -0.5*d*n*log(2*pi);
EpK = sum(log(1:k));    % Penalty for number of components

% Compute Jacobian for transformed constrained variables
LB = prior.LB;
UB = prior.UB;
if any(isfinite(LB)) || any(isfinite(UB))
    lJac = -sum(sum(vbtransform(X,LB,UB,'lgrad'),2),1);
else
    lJac = 0;
end
L = Epz-Eqz+Eppi-Eqpi+Epmu-Eqmu+EpLambda-EqLambda+EpX+EpK+lJac;

end