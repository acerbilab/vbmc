function [labels,R] = vbgmmpred(vbmodel,X)
%VBGMMPRED Predict labels and responsibility for variational Gaussian mixture model.
%   LABELS = VBGMMPRED(VBMODEL,X) predicts the cluster label for each data 
%   point in X given the variational Gaussian mixture model VBMODEL.
%   X is a D-by-N data matrix, where D is the dimensionality of the data
%   and N the number of data points. LABELS is a 1-by-N array, whose n-th 
%   element is the number of the cluster most responsible for the n-th data
%   point (from 1 to K, where K is the total number of clusters).
%
%   [LABELS,R] = VBGMMPRED(VBMODEL,X) also returns the responsibilities for 
%   each data point and cluster. R is a N-by-K matrix, where the n-th row 
%   is a vector of responsibilities for the n-th data point, and the k-th 
%   column is the responsibility of the k-th mixture component (for K mixture
%   components). Each row sums to 1.
%
%   See also VBGMMFIT, VBGMMPDF, VBGMMRND.

% Author:   Luigi Acerbi
% Email:    luigi.acerbi@gmail.com
%
% Partially based on code written by Mo Chen:
% http://www.mathworks.com/matlabcentral/fileexchange/35362-variational-bayesian-inference-for-gaussian-mixture-model

alpha = vbmodel.alpha; % Dirichlet
beta = vbmodel.beta;   % Gaussian
m = vbmodel.m;         % Gaussian
nu = vbmodel.nu;       % Wishart
U = vbmodel.U;         % Wishart 
logW = vbmodel.logW;
n = size(X,2);
[d,k] = size(m);

% Compute predictions only for variables within bounds
LB = vbmodel.prior.LB;
UB = vbmodel.prior.UB;
idx = all(bsxfun(@gt, X, LB) & bsxfun(@lt, X, UB),1);

X = vbtransform(X(:,idx),LB,UB,'dir'); % Change of variables
ntilde = size(X,2);

% Compute posterior predictive responsibility (might need to transform?)
EQ = zeros(ntilde,k);
for i = 1:k
    Q = (U(:,:,i)'\bsxfun(@minus,X,m(:,i)));
    EQ(:,i) = d/beta(i)+nu(i)*dot(Q,Q,1);    % 10.64
end
ElogLambda = sum(psi(0,0.5*bsxfun(@minus,nu+1,(1:d)')),1)+d*log(2)+logW; % 10.65
Elogpi = psi(0,alpha)-psi(0,sum(alpha)); % 10.66
logRho = -0.5*bsxfun(@minus,EQ,ElogLambda-d*log(2*pi)); % 10.46
logRho = bsxfun(@plus,logRho,Elogpi);   % 10.46
logR = bsxfun(@minus,logRho,logsumexp(logRho,2)); % 10.49

z = zeros(1,ntilde);
[~,z(:)] = max(logR,[],2);
labels = NaN(1,n);
labels(idx) = z;

if nargout > 1
    R = NaN(n,k);
    R(idx,:) = exp(logR);
end

%if nargout > 2
%    logR = NaN(n,k);
%    logR(idx,:) = logR;
%end