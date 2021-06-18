function [F,dF,G,H,varF,dH,varGss,varG,varH,I_sk,J_sjk] = negelcbo_vbmc(theta,beta,vp,gp,Ns,compute_grad,compute_var,altent_flag,thetabnd,entropy_alpha)
%NEGELCBO_VBMC Negative evidence lower confidence bound objective
%
% Note that THETA is a vector of *transformed* variational parameters:
% [MU_1,...,MU_K,log(SIGMA)] or 
% [MU_1,...,MU_K,log(SIGMA),log(LAMBDA)] or
% [MU_1,...,MU_K,log(SIGMA),log(LAMBDA),log(W)]

if nargin < 5 || isempty(Ns); Ns = 0; end
if nargin < 6 || isempty(compute_grad); compute_grad = nargout > 1; end
if nargin < 7; compute_var = []; end
if nargin < 8 || isempty(altent_flag); altent_flag = false; end
if nargin < 9; thetabnd = []; end
if nargin < 10 || isempty(entropy_alpha); entropy_alpha = 0; end
if isempty(beta) || ~isfinite(beta); beta = 0; end
if isempty(compute_var); compute_var = beta ~=0 || nargout > 4; end
separate_K = nargout > 9; % Return expected log joint per component

% altent_flag and entropy_alpha are unused (kept here for retrocompatibility)

if compute_grad && beta ~= 0 && compute_var ~= 2
    error('negelcbo_vbmc:vargrad', ...
        'Computation of the gradient of ELBO with full variance not supported.');
end

D = vp.D;
K = vp.K;

avg_flag = 1;       % Average over multiple GP hyperparameters if provided
jacobian_flag = 1;  % Variational parameters are transformed

% Reformat variational parameters from THETA
if vp.optimize_mu
    vp.mu(:,:) = reshape(theta(1:D*K),[D,K]);
    idx_start = D*K;
else
    idx_start = 0;
end
if vp.optimize_sigma
    vp.sigma(1,:) = exp(theta(idx_start+(1:K)));
    idx_start = idx_start + K;
end
if vp.optimize_lambda; vp.lambda(:,1) = exp(theta(idx_start+(1:D))); end
if vp.optimize_weights
    vp.eta(1,:) = theta(end-K+1:end);
    vp.w(1,:) = exp(vp.eta);
    vp.w = vp.w/sum(vp.w);
end

% Which gradients should be computed, if any?
grad_flags = compute_grad*[vp.optimize_mu,vp.optimize_sigma,vp.optimize_lambda,vp.optimize_weights];

% Only weight optimization?
onlyweights_flag = vp.optimize_weights && ~vp.optimize_mu && ~vp.optimize_sigma && ~vp.optimize_lambda;

if separate_K
    if compute_grad
        error('Computing the gradient of variational parameters and requesting per-component results at the same time.'); 
    end
    
    if onlyweights_flag
        if compute_var
            [G,~,varG,~,~,I_sk,J_sjk] = gplogjoint_weights(vp,0,avg_flag,jacobian_flag,compute_var);        
        else
            [G,dG,~,~,~,I_sk] = gplogjoint_weights(vp,compute_grad,avg_flag,jacobian_flag,0);
            J_sjk = [];
        end
        varGss = NaN;
    else
        if compute_var
            [G,~,varG,~,varGss,I_sk,J_sjk] = gplogjoint(vp,gp,grad_flags,avg_flag,jacobian_flag,compute_var);        
        else
            [G,dG,~,~,~,I_sk] = gplogjoint(vp,gp,grad_flags,avg_flag,jacobian_flag,0);
            varGss = 0; varG = 0; J_sjk = [];
        end
    end
else
    if onlyweights_flag
        if compute_var
            if compute_grad
                [G,dG,varG,dvarG] = gplogjoint_weights(vp,1,avg_flag,jacobian_flag,compute_var);
            else
                [G,~,varG] = gplogjoint_weights(vp,0,avg_flag,jacobian_flag,compute_var);        
            end
        else
            [G,dG] = gplogjoint_weights(vp,compute_grad,avg_flag,jacobian_flag,0);                
        end
        varGss = NaN;
    else
        if compute_var
            if compute_grad
                [G,dG,varG,dvarG,varGss] = gplogjoint(vp,gp,grad_flags,avg_flag,jacobian_flag,compute_var);
            else
                [G,~,varG,~,varGss] = gplogjoint(vp,gp,grad_flags,avg_flag,jacobian_flag,compute_var);        
            end
        else
            [G,dG] = gplogjoint(vp,gp,grad_flags,avg_flag,jacobian_flag,0);
            varGss = 0; varG = 0;
        end
    end
end

% Entropy term
if Ns > 0   
    % Monte Carlo approximation
    [H,dH] = entmc_vbmc(vp,Ns,grad_flags,jacobian_flag);
else
    % Deterministic approximation via lower bound on the entropy
    [H,dH] = entlb_vbmc(vp,grad_flags,jacobian_flag);    
end

%H_check = gmment_num(theta,lambda);
%[H - H_check, (H - H_check)/H_check ]

% Negative ELBO and its gradient
F = -G - H;
if compute_grad; dF = -dG - dH; else; dF = []; end

varH = 0;   % For the moment use zero variance for entropy
if compute_var
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
% and for weight size (if optimizing mixture weights)
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
    
    % Penalty to reduce weight size
    if vp.optimize_weights
        Thresh = thetabnd.WeightThreshold;
        %L = sum(vp.w)*thetabnd.WeightPenalty;
        % L = sum(sqrt(vp.w))*thetabnd.WeightPenalty;
        L = sum(vp.w.*(vp.w<Thresh) + Thresh*(vp.w>=Thresh))*thetabnd.WeightPenalty;
        F = F + L;
        if compute_grad
            %w_grad = thetabnd.WeightPenalty*ones(K,1);
            % w_grad = 0.5./sqrt(vp.w(:))*thetabnd.WeightPenalty;
            w_grad = thetabnd.WeightPenalty.*(vp.w(:)<Thresh);
            eta_sum = sum(exp(vp.eta));
            J_w = bsxfun(@times,-exp(vp.eta)',exp(vp.eta)/eta_sum^2) + diag(exp(vp.eta)/eta_sum);
            w_grad = J_w*w_grad;
            dL = zeros(size(dF));
            dL(end-K+1:end) = w_grad;
            dF = dF + dL;
        end
    end
end

end