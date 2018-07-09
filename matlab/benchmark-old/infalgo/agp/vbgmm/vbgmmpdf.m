function y = vbgmmpdf(vbmodel,X,flag)
%VBGMMPDF Posterior predictive pdf for variational Gaussian mixture model.
%   Y = VBGMMPDF(VBMODEL,X) computes posterior predictive probability
%   density (pdf) of the variational Gaussian mixture model VBMODEL, 
%   evaluated at the values in X. X is a D-by-N data matrix, where D is the
%   dimensionality of the data and N the number of data points.
%
%   Y = VBGMMPDF(VBMODEL,X,1) does *not* apply the Jacobian correction to
%   the pdf for bounded variables.
%
%   See also VBGMMFIT, VBGMMPRED, VBGMMRND.

% Author:   Luigi Acerbi
% Email:    luigi.acerbi@gmail.com

if nargin < 3 || isempty(flag); flag = 0; end

alpha = vbmodel.alpha; % Dirichlet
beta = vbmodel.beta;   % Gaussian
m = vbmodel.m;         % Gaussian
nu = vbmodel.nu;       % Wishart
U = vbmodel.U;         % Wishart 
n = size(X,2);
[d,k] = size(m);

% Compute predictions only for variables within bounds
LB = vbmodel.prior.LB;
UB = vbmodel.prior.UB;
idx = all(bsxfun(@gt, X, LB) & bsxfun(@lt, X, UB),1);

% Bounded variables?
boundedflag = any(~isfinite(LB)) || any(~isfinite(UB));

if boundedflag
    X = vbtransform(X(:,idx),LB,UB,'dir'); % Change of variables
end
ntilde = size(X,2);

% Compute posterior predictive density
X = X';
df = nu - d + 1;
ytemp = zeros(ntilde,1);
p = alpha/sum(alpha);
for i = 1:k
    if df(i) == Inf     % Multivariate normal
        S = U(:,:,i)'*U(:,:,i);
        ytemp = ytemp + p(i)*mvnpdf(X, m(:,i)', S);
    elseif df(i) <= 0   % Multivariate uniform ellipsoid
        S = U(:,:,i)'*U(:,:,i);
        ytemp = ytemp + p(i)*mvuepdf(X, m(:,i)', S);
    else
        C = (U(:,:,i)'*U(:,:,i))*((beta(i)+1)/(beta(i)*df(i)));
        s = sqrt(diag(C));
        Xres = bsxfun(@rdivide, bsxfun(@minus, X, m(:,i)'), s');
        ytemp = ytemp + p(i)*mvtpdf(Xres, C, df(i))/prod(s);
    end
end
if boundedflag && ~flag 
    ytemp = ytemp./exp(sum(vbtransform(X',LB,UB,'lgrad'),1))';
end

y = NaN(1,n);
y(idx) = ytemp;