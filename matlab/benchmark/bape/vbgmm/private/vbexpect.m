function vbmodel = vbexpect(X,vbmodel)
%VBEXPECT Variational expectation step.

% Partially based on code written by Mo Chen:
% http://www.mathworks.com/matlabcentral/fileexchange/35362-variational-bayesian-inference-for-gaussian-mixture-model

alpha = vbmodel.alpha;    % Dirichlet
beta = vbmodel.beta;      % Gaussian
m = vbmodel.m;            % Gasusian
nu = vbmodel.nu;          % Wishart
U = vbmodel.U;            % Wishart 
logW = vbmodel.logW;
n = size(X,2);
[d,k] = size(m);

EQ = zeros(n,k);
for i = 1:k
    Q = (U(:,:,i)'\bsxfun(@minus,X,m(:,i)));
    EQ(:,i) = d/beta(i)+nu(i)*dot(Q,Q,1);    % 10.64
end

ElogLambda = sum(psi(0,0.5*bsxfun(@minus,nu+1,(1:d)')),1)+d*log(2)+logW; % 10.65
Elogpi = psi(0,alpha)-psi(0,sum(alpha)); % 10.66
logRho = -0.5*bsxfun(@minus,EQ,ElogLambda-d*log(2*pi)); % 10.46
logRho = bsxfun(@plus,logRho,Elogpi);   % 10.46
logR = bsxfun(@minus,logRho,logsumexp(logRho,2)); % 10.49
R = exp(logR);

vbmodel.logR = logR;
vbmodel.R = R;
    
end