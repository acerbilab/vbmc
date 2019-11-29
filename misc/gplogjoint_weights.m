function [F,dF,varF,dvarF,varss] = gplogjoint_weights(vp,grad_flag,avg_flag,jacobian_flag,compute_var)
%GPLOGJOINT_WEIGHTS Expected variational log joint probability via GP approximation

% VP is a struct with the variational posterior
% HYP is the vector of GP hyperparameters: [ell,sf2,sn2,m]
% Note that hyperparameters are already transformed
% X is a N-by-D matrix of training inputs
% Y is a N-by-1 vector of function values at X

if nargin < 3; grad_flag = []; end
if nargin < 4 || isempty(avg_flag); avg_flag = true; end
if nargin < 5 || isempty(jacobian_flag); jacobian_flag = true; end
if nargin < 6; compute_var = []; end
if isempty(compute_var); compute_var = nargout > 2; end

% Check if gradient computation is required
if nargout < 2                              % No 2nd output, no gradients
    grad_flag = false;
elseif isempty(grad_flag)                  % By default compute all gradients
    grad_flag = true;
end

compute_vargrad = nargout > 3 && compute_var && grad_flag;

if compute_vargrad && compute_var ~= 2
    error('gplogjoint:FullVarianceGradient', ...
        'Computation of gradient of log joint variance is currently available only for diagonal approximation of the variance.');
end

K = vp.K;           % Number of components
w(1,:) = vp.w;
I_sk = vp.I_sk;
J_sjk = vp.J_sjk;

Ns = size(I_sk,1);            % Hyperparameter samples

F = zeros(1,Ns);
if grad_flag; w_grad = zeros(K,Ns); else, w_grad = []; end
if compute_var; varF = zeros(1,Ns); end
if compute_vargrad      % Compute gradient of variance?
    if grad_flag; w_vargrad = zeros(K,Ns); else, w_vargrad = []; end    
end

% Loop over hyperparameter samples
for s = 1:Ns    
    F(s) = sum(w.*I_sk(s,:));
    if grad_flag; w_grad(:,s) = I_sk(s,:)'; end
    
    if compute_var == 2
        J_diag = diag(squeeze(J_sjk(s,:,:)))';
        varF(s) = sum(w.^2.*max(eps,J_diag));
        if compute_vargrad
            w_vargrad(:,s) = 2*w.*max(eps,J_diag);
        end
    elseif compute_var
        J_jk = squeeze(J_sjk(s,:,:));
        varF(s) = sum(sum(J_jk.*(w'*w),1));        
    end
end

% Correct for numerical error
if compute_var; varF = max(varF,eps); end

if grad_flag
    if jacobian_flag
        eta_sum = sum(exp(vp.eta));
        J_w = bsxfun(@times,-exp(vp.eta)',exp(vp.eta)/eta_sum^2) + diag(exp(vp.eta)/eta_sum);
        w_grad = J_w*w_grad;
    end
    dF = w_grad;
else
    dF = [];
end

if compute_vargrad
    % Correct for standard softmax reparameterization of W
    if jacobian_flag && grad_flag
        w_vargrad = J_w*w_vargrad;
    end    
    dvarF = w_vargrad;
else
    dvarF = [];
end

% [varF; varF_diag]

% Average multiple hyperparameter samples
varss = 0;
if Ns > 1 && avg_flag
    Fbar = sum(F,2)/Ns;
    if compute_var
        varFss = sum((F - Fbar).^2,2)/(Ns-1);     % Estimated variance of the samples
        varss = varFss + std(varF); % Variability due to sampling
        varF = sum(varF,2)/Ns + varFss;
    end
    if compute_vargrad
        dvv = 2*sum(F.*dF,2)/(Ns-1) - 2*Fbar.*sum(dF,2)/(Ns-1);
        dvarF = sum(dvarF,2)/Ns + dvv;
    end
    F = Fbar;
    if grad_flag; dF = sum(dF,2)/Ns; end
end

end