function [F,dF,varF,dvarF,varss] = gplogjoint(vp,gp,grad_flags,avg_flag,jacobian_flag,compute_var)
%GPLOGJOINT Expected variational log joint probability via GP approximation

% VP is a struct with the variational posterior
% HYP is the vector of GP hyperparameters: [ell,sf2,sn2,m]
% Note that hyperparameters are already transformed
% X is a N-by-D matrix of training inputs
% Y is a N-by-1 vector of function values at X

if nargin < 3; grad_flags = []; end
if nargin < 4 || isempty(avg_flag); avg_flag = true; end
if nargin < 5 || isempty(jacobian_flag); jacobian_flag = true; end
if nargin < 6; compute_var = []; end
if isempty(compute_var); compute_var = nargout > 2; end

% Check if gradient computation is required
if nargout < 2                              % No 2nd output, no gradients
    grad_flags = false;
elseif isempty(grad_flags)                  % By default compute all gradients
    grad_flags = true;
end
if isscalar(grad_flags); grad_flags = ones(1,4)*grad_flags; end

compute_vargrad = nargout > 3 && compute_var && any(grad_flags);

if compute_vargrad && compute_var ~= 2
    error('gplogjoint:FullVarianceGradient', ...
        'Computation of gradient of log joint variance is currently available only for diagonal approximation of the variance.');
end

D = vp.D;           % Number of dimensions
K = vp.K;           % Number of components
N = size(gp.X,1);
mu(:,:) = vp.mu;
sigma(1,:) = vp.sigma;
lambda(:,1) = vp.lambda(:);
w(1,:) = vp.w;

Ns = numel(gp.post);            % Hyperparameter samples

if all(gp.meanfun ~= [0 1 4 6 8])
    error('gplogjoint:UnsupportedMeanFun', ...
        'Log joint computation currently only supports zero, constant, negative quadratic, or squared exponential mean functions.');
end

% Which mean function is being used?
quadratic_meanfun = gp.meanfun == 4;
sqexp_meanfun = gp.meanfun == 6;
quadsqexp_meanfun = gp.meanfun == 8;

F = zeros(1,Ns);
% Check which gradients are computed
if grad_flags(1); mu_grad = zeros(D,K,Ns); else, mu_grad = []; end
if grad_flags(2); sigma_grad = zeros(K,Ns); else, sigma_grad = []; end
if grad_flags(3); lambda_grad = zeros(D,Ns); else, lambda_grad = []; end
if grad_flags(4); w_grad = zeros(K,Ns); else, w_grad = []; end
if compute_var; varF = zeros(1,Ns); end
if compute_vargrad      % Compute gradient of variance?
    if grad_flags(1); mu_vargrad = zeros(D,K,Ns); else, mu_vargrad = []; end
    if grad_flags(2); sigma_vargrad = zeros(K,Ns); else, sigma_vargrad = []; end
    if grad_flags(3); lambda_vargrad = zeros(D,Ns); else, lambda_vargrad = []; end    
    if grad_flags(4); w_vargrad = zeros(K,Ns); else, w_vargrad = []; end    
end

% varF_diag = zeros(1,Nhyp);

% Loop over hyperparameter samples
for s = 1:Ns
    hyp = gp.post(s).hyp;
    
    % Extract GP hyperparameters from HYP
    ell = exp(hyp(1:D));
    ln_sf2 = 2*hyp(D+1);
    sn2 = exp(2*hyp(D+2));
    sum_lnell = sum(hyp(1:D));    
    
    % GP mean function hyperparameters
    if gp.meanfun > 0; m0 = hyp(D+3); else; m0 = 0; end
    if quadratic_meanfun || sqexp_meanfun || quadsqexp_meanfun
        xm = hyp(D+3+(1:D));
        omega = exp(hyp(2*D+3+(1:D)));
        if sqexp_meanfun
            h = exp(hyp(3*D+4));
        end
    end
    if quadsqexp_meanfun
        xm_se = hyp(3*D+3+(1:D));
        omega_se = exp(hyp(4*D+3+(1:D)));
        h_se = hyp(5*D+4);        
    end
    
    alpha = gp.post(s).alpha;
    L = gp.post(s).L;
    Lchol = gp.post(s).Lchol;
    sn2_eff = sn2*gp.post(s).sn2_mult;

    for k = 1:K

        tau_k = sqrt(sigma(k)^2*lambda.^2 + ell.^2);
        lnnf_k = ln_sf2 + sum_lnell - sum(log(tau_k));  % Covariance normalization factor
        delta_k = bsxfun(@rdivide,bsxfun(@minus, mu(:,k), gp.X'), tau_k);
        z_k = exp(lnnf_k -0.5 * sum(delta_k.^2,1));
        I_k = z_k*alpha + m0;

        if quadratic_meanfun || quadsqexp_meanfun
            nu_k = -0.5*sum(1./omega.^2 .* ...
                (mu(:,k).^2 + sigma(k)^2*lambda.^2 - 2*mu(:,k).*xm + xm.^2),1);            
            I_k = I_k + nu_k;        
        elseif sqexp_meanfun
            tau2_mfun = sigma(k)^2*lambda.^2 + omega.^2;
            s2 = ((mu(:,k) - xm).^2)./tau2_mfun;
            nu_k_se = h*prod(omega./sqrt(tau2_mfun))*exp(-0.5*sum(s2,1));            
            I_k = I_k + nu_k_se;
        end
        if quadsqexp_meanfun
            tau2_mfun = sigma(k)^2*lambda.^2 + omega_se.^2;
            s2 = ((mu(:,k) - xm_se).^2)./tau2_mfun;
            nu_k_se = h_se*prod(omega_se./sqrt(tau2_mfun))*exp(-0.5*sum(s2,1));
            I_k = I_k + nu_k_se;        
        end
        
        F(s) = F(s) + w(k)*I_k;

        if grad_flags(1)
            dz_dmu = bsxfun(@times, -bsxfun(@rdivide, delta_k, tau_k), z_k);
            mu_grad(:,k,s) = w(k)*dz_dmu*alpha;            
            if quadratic_meanfun || quadsqexp_meanfun
                mu_grad(:,k,s) = mu_grad(:,k,s) - w(k)./omega.^2.*(mu(:,k) - xm);
            elseif sqexp_meanfun
                mu_grad(:,k,s) = mu_grad(:,k,s) - w(k)*nu_k_se./tau2_mfun.*(mu(:,k) - xm);                
            end
            if quadsqexp_meanfun
                mu_grad(:,k,s) = mu_grad(:,k,s) - w(k)*nu_k_se./tau2_mfun.*(mu(:,k) - xm_se);
            end
            
        end
        
        if grad_flags(2)
            dz_dsigma = bsxfun(@times, sum(bsxfun(@times,(lambda./tau_k).^2, delta_k.^2 - 1),1), sigma(k)*z_k);
            sigma_grad(k,s) = w(k)*dz_dsigma*alpha;
            if quadratic_meanfun || quadsqexp_meanfun
                sigma_grad(k,s) = sigma_grad(k,s) - w(k)*sigma(k)*sum(1./omega.^2.*lambda.^2,1);
            elseif sqexp_meanfun
                sigma_grad(k,s) = sigma_grad(k,s) - w(k)*sigma(k)*sum(lambda.^2./tau2_mfun,1)*nu_k_se ...
                    + w(k)*sigma(k)*sum((mu(:,k) - xm).^2.*lambda.^2./tau2_mfun.^2,1)*nu_k_se;
            end
            if quadsqexp_meanfun
                sigma_grad(k,s) = sigma_grad(k,s) - w(k)*sigma(k)*sum(lambda.^2./tau2_mfun,1)*nu_k_se ...
                    + w(k)*sigma(k)*sum((mu(:,k) - xm_se).^2.*lambda.^2./tau2_mfun.^2,1)*nu_k_se;
            end                
        end

        if grad_flags(3)
            dz_dlambda = bsxfun(@times, bsxfun(@times, (sigma(k)./tau_k).^2, delta_k.^2 - 1), bsxfun(@times,lambda,z_k));
            lambda_grad(:,s) = lambda_grad(:,s) + w(k)*(dz_dlambda*alpha);
            if quadratic_meanfun || quadsqexp_meanfun
                lambda_grad(:,s) = lambda_grad(:,s) - w(k)*sigma(k)^2./omega.^2.*lambda;
            elseif sqexp_meanfun
                lambda_grad(:,s) = lambda_grad(:,s) - w(k)*sigma(k)^2*lambda./tau2_mfun*nu_k_se ...
                    + w(k)*sigma(k)^2*(mu(:,k) - xm).^2.*lambda./tau2_mfun.^2*nu_k_se;
            end
            if quadsqexp_meanfun
                lambda_grad(:,s) = lambda_grad(:,s) - w(k)*sigma(k)^2*lambda./tau2_mfun*nu_k_se ...
                    + w(k)*sigma(k)^2*(mu(:,k) - xm_se).^2.*lambda./tau2_mfun.^2*nu_k_se;                
            end
        end
        
        if grad_flags(4)
            w_grad(k,s) = I_k;
        end
        
        if compute_var == 2 % Compute only self-variance
            tau_kk = sqrt(2*sigma(k)^2*lambda.^2 + ell.^2);                
            nf_kk = exp(ln_sf2 + sum_lnell - sum(log(tau_kk)));
            if Lchol
                invKzk = (L\(L'\z_k'))/sn2_eff;
            else
                invKzk = -L*z_k';                
            end                
            J_kk = nf_kk - z_k*invKzk;
            varF(s) = varF(s) + w(k)^2*max(eps,J_kk);    % Correct for numerical error
            
            if compute_vargrad

                if grad_flags(1)
                    mu_vargrad(:,k,s) = -w(k)^2*(2*dz_dmu*invKzk);
                end

                if grad_flags(2)
                    sigma_vargrad(k,s) = -2*w(k)^2*(sigma(k)*nf_kk*sum(lambda.^2./tau_kk.^2) + dz_dsigma*invKzk);
                end

                if grad_flags(3)
                    lambda_vargrad(:,s) = lambda_vargrad(:,s) - 2*w(k)^2*(sigma(k)^2*nf_kk.*lambda./tau_kk.^2  + dz_dlambda*invKzk);
                end
                
                if grad_flags(4)
                    w_vargrad(k,s) = 2*w(k)*max(eps,J_kk);
                end
                
            end
                        
        elseif compute_var
            for j = 1:k
                tau_j = sqrt(sigma(j)^2*lambda.^2 + ell.^2);
                lnnf_j = ln_sf2 + sum_lnell - sum(log(tau_j));
                delta_j = bsxfun(@rdivide,bsxfun(@minus, mu(:,j), gp.X'), tau_j);
                z_j = exp(lnnf_j -0.5 * sum(delta_j.^2,1));                    
                
                tau_jk = sqrt((sigma(j)^2 + sigma(k)^2)*lambda.^2 + ell.^2);                
                lnnf_jk = ln_sf2 + sum_lnell - sum(log(tau_jk));
                delta_jk = (mu(:,j)-mu(:,k))./tau_jk;
                
                if Lchol
                    J_jk = exp(lnnf_jk -0.5*sum(delta_jk.^2,1)) ...
                     - z_k*(L\(L'\z_j'))/sn2_eff;
                else
                    J_jk = exp(lnnf_jk -0.5*sum(delta_jk.^2,1)) ...
                     + z_k*(L*z_j');                    
                end
                
%                 J(j,k) = w(j)*w(k)*J_jk;
                
                % Off-diagonal elements are symmetric (count twice)
                if j == k
                    varF(s) = varF(s) + w(k)^2*max(eps,J_jk);                                        
                else
                    varF(s) = varF(s) + 2*w(j)*w(k)*J_jk;                    
                end
            end
            
        end
        
    end
    
%     if compute_var
%         pause
%     end
    
end

% Correct for numerical error
if compute_var; varF = max(varF,eps); end

if any(grad_flags)
    if grad_flags(1)
        mu_grad = reshape(mu_grad,[D*K,Ns]);
    end
    % Correct for standard log reparameterization of SIGMA
    if jacobian_flag && grad_flags(2)
        sigma_grad = bsxfun(@times,sigma_grad, sigma(:));        
    end
    % Correct for standard log reparameterization of LAMBDA
    if jacobian_flag && grad_flags(3)
        lambda_grad = bsxfun(@times,lambda_grad, lambda(:));        
    end
    % Correct for standard softmax reparameterization of W
    if jacobian_flag && grad_flags(4)
        eta_sum = sum(exp(vp.eta));
        J_w = bsxfun(@times,-exp(vp.eta)',exp(vp.eta)/eta_sum^2) + diag(exp(vp.eta)/eta_sum);
        w_grad = J_w*w_grad;
    end
    dF = [mu_grad;sigma_grad;lambda_grad;w_grad];
else
    dF = [];
end

if compute_vargrad
    if grad_flags(1)
        mu_vargrad = reshape(mu_vargrad,[D*K,Ns]);
    end
    % Correct for standard log reparameterization of SIGMA
    if jacobian_flag && grad_flags(2)
        sigma_vargrad = bsxfun(@times,sigma_vargrad, sigma(:));        
    end
    % Correct for standard log reparameterization of LAMBDA
    if jacobian_flag && grad_flags(3)
        lambda_vargrad = bsxfun(@times,lambda_vargrad, lambda(:));        
    end
    % Correct for standard softmax reparameterization of W
    if jacobian_flag && grad_flags(4)
        w_vargrad = J_w*w_vargrad;
    end    
    dvarF = [mu_vargrad;sigma_vargrad;lambda_vargrad;w_vargrad];
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
    if any(grad_flags); dF = sum(dF,2)/Ns; end
end


end