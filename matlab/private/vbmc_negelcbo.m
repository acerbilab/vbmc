function [F,dF,G,H,varF,dH,varGss] = vbmc_negelcbo(theta,beta,vp,gp,Ns,compute_grad,compute_var,altent_flag,thetabnd)
%VBMC_NEGELCBO Negative evidence lower confidence bound objective
%
% Note that THETA is a vector of *transformed* variational parameters:
% [MU_1,...,MU_K,log(SIGMA)] or 
% [MU_1,...,MU_K,log(SIGMA),log(LAMBDA)]

if nargin < 5 || isempty(Ns); Ns = 0; end
if nargin < 6 || isempty(compute_grad); compute_grad = nargout > 1; end
if nargin < 7; compute_var = []; end
if nargin < 8 || isempty(altent_flag); altent_flag = false; end
if nargin < 9; thetabnd = []; end
if isempty(beta) || ~isfinite(beta); beta = 0; end
if isempty(compute_var); compute_var = beta ~=0 || nargout > 4; end

if compute_grad && beta ~= 0 && compute_var ~= 2
    error('vbmc_negelcbo:vargrad', ...
        'Computation of the gradient of ELBO with full variance not supported.');
end

D = vp.D;
K = vp.K;

avg_flag = 1;       % Average over multiple GP hyperparameters if provided
jacobian_flag = 1;  % Variational parameters are transformed

% Reformat variational parameters from THETA
vp.mu(:,:) = reshape(theta(1:D*K),[D,K]);
vp.sigma(1,:) = exp(theta(D*K+(1:K)));
if vp.optimize_lambda; vp.lambda(:,1) = exp(theta(D*K+K+(1:D))); end

% Which gradients should be computed, if any?
grad_flags = compute_grad*[1,1,vp.optimize_lambda];

if compute_var
    if compute_grad
        [G,dG,varG,dvarG,varGss] = gplogjoint(vp,gp,grad_flags,avg_flag,jacobian_flag,compute_var);
    elseif numel(gp) > 1
        [G,varG] = gplogjoint_multi(vp,gp,avg_flag,compute_var);
        varGss = [];
    else
        [G,~,varG,~,varGss] = gplogjoint(vp,gp,grad_flags,avg_flag,jacobian_flag,compute_var);        
    end
else
    [G,dG] = gplogjoint(vp,gp,grad_flags,avg_flag,jacobian_flag,0);
end

% Entropy term
if Ns > 0   % Use Monte Carlo approximation
    if altent_flag  % Alternative entropy approximation
        [H,dH] = vbmc_entmcalt(vp,Ns,grad_flags,jacobian_flag);        
    else
        [H,dH] = vbmc_entmc(vp,Ns,grad_flags,jacobian_flag);
    end
else
    [H,dH] = vbmc_ent(vp,grad_flags,jacobian_flag);
end
%H_check = gmment_num(theta,lambda);
%[H - H_check, (H - H_check)/H_check ]

% Negative ELBO and its gradient
F = -G - H;
if compute_grad; dF = -dG - dH; else; dF = []; end

if compute_var
    varH = 0;   % For the moment use zero variance for entropy
    varF = varG + varH;
else
    varF = 0;
end

% Negative ELCBO (add confidence bound)
if beta ~= 0; F = F + beta*sqrt(varF); end
if beta ~= 0 && compute_grad
    dF = dF + 0.5*beta*dvarG/sqrt(varF);
end

% Additional loss for variational parameter bound violation (soft bounds)
% Only done when optimizing the variational parameters, but not when 
% computing the EL(C)BO at each iteration
if ~isempty(thetabnd)    
    if compute_grad
        [L,dL] = vpbndloss(theta,vp,thetabnd,thetabnd.TolCon);
        dF = dF + dL;
    else
        L = vpbndloss(theta,vp,thetabnd,thetabnd.TolCon);
    end
    F = F + L;    
end

end