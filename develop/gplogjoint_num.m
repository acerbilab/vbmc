function [F,dF,gp] = gplogjoint_num(vp,gp)
%GPLOGJOINT_NUM Expected variational log joint probability via GP approximation 
% (numerical integration, no derivatives)

% VP is a struct with the variational posterior
% HYP is the vector of GP hyperparameters: [ell,sf2,sn2,m]
% Note that hyperparameters are already transformed
% X is a N-by-D matrix of training inputs
% Y is a N-by-1 vector of function values at X

DEBUG = 0;

D = vp.D;           % Number of dimensions
K = vp.K;           % Number of components
MaxSD = 6;                  % Maximum SDs for integration
Nx = ceil(1e5^(1/D));       % Grid size

mu(:,:) = vp.mu;
sigma(1,:) = vp.sigma;
lambda(:,1) = vp.lambda(:);

I_k = zeros(1,K);

if DEBUG
    [sigma(:)',lambda(:)']
end

% Loop over mixture components
for k = 1:K
    for d = 1:D
        xvec{d} = linspace(mu(d,k)-lambda(d)*sigma(k)*MaxSD,mu(d,k)+lambda(d)*sigma(k)*MaxSD,Nx);
        dx(d) = xvec{d}(2)-xvec{d}(1);
    end
    
    Xstar = combvec(xvec{:})';
    fstar = gplite_pred(gp,Xstar);
    
    % Single mixture component
    vp1 = vp;
    vp1.K = 1;
    vp1.w = 1;
    vp1.mu = vp1.mu(:,k);
    vp1.sigma = vp1.sigma(k);
    q1 = vbmc_pdf(vp1,Xstar,0);
   
    if DEBUG
        sum(q1(:))*prod(dx)
    end
    
    % Integral via Riemannian sum (coarse)
    I_k(k) = sum(fstar .* q1) * prod(dx);        
end

F = sum(vp.w.*I_k);
dF = [];                    % Gradient computation not supported