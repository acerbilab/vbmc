function [F,dF,varF,dvarF,varss] = gplogjoint(vp,gp,grad_flags,avg_flag,jacobian_flag,compute_var,separate_K)
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
if nargin < 7 || isempty(separate_K); separate_K = false; end
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

% Number of GP hyperparameters
Ncov = gp.Ncov;
Nnoise = gp.Nnoise;
Nmean = gp.Nmean;

Ns = numel(gp.post);            % Hyperparameter samples

if all(gp.meanfun ~= [0,1,4,6,8,10,12,14,16,18,20,22])
    error('gplogjoint:UnsupportedMeanFun', ...
        'Log joint computation currently only supports zero, constant, negative quadratic, negative quadratic (fixed/isotropic), negative quadratic-only, or squared exponential mean functions.');
end

% Which mean function is being used?
quadratic_meanfun = gp.meanfun == 4 || gp.meanfun == 10 || gp.meanfun == 12;
fixediso_meanfun = gp.meanfun == 10;
fixed_meanfun = gp.meanfun == 12 || gp.meanfun == 14;
sqexp_meanfun = gp.meanfun == 6;
quadsqexp_meanfun = gp.meanfun == 8 || gp.meanfun == 14;
quadsqexpconstrained_meanfun = gp.meanfun == 14;
quadraticonly_meanfun = gp.meanfun == 16;
quadraticfixedonly_meanfun = gp.meanfun == 18;
quadraticlinonly_meanfun = gp.meanfun == 20;
quadraticmix_meanfun = gp.meanfun == 22;

% Integrated mean function being used?
integrated_meanfun = isfield(gp,'intmeanfun') && gp.intmeanfun > 0;

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

% Store contribution to the log joint separately for each component?
if separate_K
    I_sk = zeros(Ns,K);
    if compute_var; J_sjk = zeros(Ns,K,K); end    
end

% varF_diag = zeros(1,Nhyp);

if isfield(vp,'delta') && ~isempty(vp.delta)
    delta = vp.delta;
else
    delta = 0;
end

Xt = zeros(D,N,K);
for k = 1:K
    Xt(:,:,k) = bsxfun(@minus, mu(:,k), gp.X');
end

% Precompute expensive integrated mean function vectors
if integrated_meanfun && gp.intmeanfun == 4
    tril_mat = tril(true(D),-1); 
    tril_vec = tril_mat(:);
    mumu_mat = zeros(K,D*(D-1)/2);
    for k = 1:K
        mumu_tril = mu(:,k)*mu(:,k)';
        mumu_vec = mumu_tril(:);
        mumu_mat(k,:) = mumu_vec(tril_vec)';
    end
    if grad_flags(1)
        for k = 1:K
            dmumu{k} = zeros(D,D*(D-1)/2);
            idx = 0;
            for d = 1:D-1
                dmumu{k}(:,idx+(1:D-d)) = [zeros(d-1,D-d); mu(d+1:D,k)'; mu(d,k)*eye(D-d)];
                idx = idx + D-d;
            end
        end
    end
end

% Loop over hyperparameter samples
for s = 1:Ns
    hyp = gp.post(s).hyp;
    
    % Extract GP hyperparameters from HYP
    ell = exp(hyp(1:D));
    ln_sf2 = 2*hyp(D+1);
    sum_lnell = sum(hyp(1:D));
        
    % GP mean function hyperparameters
    if gp.meanfun > 0 && ~quadraticonly_meanfun && ~quadraticfixedonly_meanfun && ~quadraticlinonly_meanfun
        m0 = hyp(Ncov+Nnoise+1);
    else
        m0 = 0;
    end
    if quadratic_meanfun || sqexp_meanfun || quadsqexp_meanfun || quadsqexpconstrained_meanfun || quadraticmix_meanfun
        if fixediso_meanfun
            xm(:,1) = gp.meanfun_extras(1:D)';
            omega = exp(hyp(Ncov+Nnoise+2));
        elseif fixed_meanfun
            xm(:,1) = gp.meanfun_extras(1:D)';
            omega = exp(hyp(Ncov+Nnoise+1+(1:D)));
        else
            xm = hyp(Ncov+Nnoise+1+(1:D));
            omega = exp(hyp(Ncov+Nnoise+D+1+(1:D)));            
        end
        if sqexp_meanfun
            h = exp(hyp(Ncov+Nnoise+2*D+2));
        end
        if quadraticmix_meanfun
            hm = hyp(Ncov+Nnoise+2*D+2);          
            rho2 = exp(2*hyp(Ncov+Nnoise+2*D+3));
            beta2 = exp(2*hyp(Ncov+Nnoise+2*D+4));
            m0 = m0 + hm;
            Zm = (2*pi*rho2)^(D/2)*prod(omega);
        end
    end
    if quadsqexpconstrained_meanfun
        xm_se = xm;
        omega_se = omega*exp(hyp(Ncov+Nnoise+D+2));
        h_se = exp(hyp(Ncov+Nnoise+D+3));
        m0 = m0 - h_se;
    elseif quadsqexp_meanfun
        xm_se = hyp(Ncov+Nnoise+2*D+1+(1:D));
        omega_se = exp(hyp(Ncov+Nnoise+3*D+1+(1:D)));
        h_se = hyp(Ncov+Nnoise+4*D+2);        
    end
    if quadraticonly_meanfun
        omega = exp(hyp(Ncov+Nnoise+(1:D)));
    end
    if quadraticfixedonly_meanfun
        xm(:,1) = gp.meanfun_extras(1:D)';
        omega = exp(hyp(Ncov+Nnoise+(1:D)));        
    end
    if quadraticlinonly_meanfun
        xm = hyp(Ncov+Nnoise+(1:D));
        omega = exp(hyp(Ncov+Nnoise+D+(1:D)));            
    end
    
    % GP integrated mean function parameters
    if integrated_meanfun
        betabar = gp.post(s).intmean.betabar';
        KinvHtbetabar = gp.post(s).intmean.HKinv'*betabar;
        if compute_var
            plus_idx = gp.intmeanfun_var > 0;
            HKinv = gp.post(s).intmean.HKinv(plus_idx,:);
            Tplusinv = gp.post(s).intmean.Tplusinv;
        end
    end
    
    alpha = gp.post(s).alpha;
    L = gp.post(s).L;
    Lchol = gp.post(s).Lchol;
    
    sn2_eff = 1/gp.post(s).sW(1)^2;

    for k = 1:K

        tau_k = sqrt(sigma(k)^2*lambda.^2 + ell.^2 + delta.^2);
        lnnf_k = ln_sf2 + sum_lnell - sum(log(tau_k));  % Covariance normalization factor
%        delta_k = bsxfun(@rdivide,bsxfun(@minus, mu(:,k), gp.X'), tau_k);
        delta_k = bsxfun(@rdivide, Xt(:,:,k), tau_k);
        z_k = exp(lnnf_k -0.5 * sum(delta_k.^2,1));
        I_k = z_k*alpha + m0;

        if quadratic_meanfun || quadsqexp_meanfun || quadraticfixedonly_meanfun || quadraticlinonly_meanfun
            nu_k = -0.5*sum(1./omega.^2 .* ...
                (mu(:,k).^2 + sigma(k)^2*lambda.^2 - 2*mu(:,k).*xm + xm.^2 + delta.^2),1);
            I_k = I_k + nu_k;
        elseif sqexp_meanfun
            tau2_mfun = sigma(k)^2*lambda.^2 + omega.^2 + delta.^2; % delta might be wrong here
            s2 = ((mu(:,k) - xm).^2)./tau2_mfun;
            nu_k_se = h*prod(omega./sqrt(tau2_mfun))*exp(-0.5*sum(s2,1));            
            I_k = I_k + nu_k_se;
        elseif quadraticmix_meanfun
            nu1_k = -0.5/beta2*sum(1./omega.^2 .* ...
                (mu(:,k).^2 + sigma(k)^2*lambda.^2 - 2*mu(:,k).*xm + xm.^2),1);
            tautilde2_m = sigma(k)^2*lambda.^2 + rho2*omega.^2;
            s2 = ((xm - mu(:,k)).^2)./tautilde2_m;
            alphatilde_m = Zm/(2*pi)^(D/2)/sqrt(prod(tautilde2_m))*exp(-0.5*sum(s2,1));
            nu2_k = -hm*alphatilde_m;
            mutilde_m = (xm.*sigma(k)^2.*lambda.^2 + mu(:,k).*rho2.*omega.^2)./tautilde2_m;
            sigmatilde2_m = sigma(k)^2*lambda.^2.*rho2.*omega.^2./tautilde2_m;
            nu3_k = -0.5*alphatilde_m*(1-1/beta2)*sum(1./omega.^2.*(mutilde_m.^2 + sigmatilde2_m - 2*mutilde_m.*xm + xm.^2),1);
            I_k = I_k + nu1_k + nu2_k + nu3_k;
        end
        if quadsqexp_meanfun
            tau2_mfun = sigma(k)^2*lambda.^2 + omega_se.^2 + delta.^2; % delta might be wrong here
            s2 = ((mu(:,k) - xm_se).^2)./tau2_mfun;
            nu_k_se = h_se*prod(omega_se./sqrt(tau2_mfun))*exp(-0.5*sum(s2,1));
            I_k = I_k + nu_k_se;        
        end
        if quadraticonly_meanfun
            nu_k = -0.5*sum(1./omega.^2 .* (mu(:,k).^2 + sigma(k)^2*lambda.^2 + delta.^2),1);
            I_k = I_k + nu_k;            
        end
        if integrated_meanfun
            switch gp.intmeanfun
                case 1; u_k = 1;
                case 2; u_k = [1,mu(:,k)'];                  
                case 3; u_k = [1,mu(:,k)',(mu(:,k).^2 + sigma(k)^2*lambda.^2)'];
                case 4; u_k = [1,mu(:,k)',(mu(:,k).^2 + sigma(k)^2*lambda.^2)',mumu_mat(k,:)];                    
            end
            I_k = I_k + u_k*betabar - z_k*KinvHtbetabar;
        end
                
        F(s) = F(s) + w(k)*I_k;
        if separate_K; I_sk(s,k) = I_k; end

        if grad_flags(1)
            dz_dmu = bsxfun(@times, -bsxfun(@rdivide, delta_k, tau_k), z_k);
            mu_grad(:,k,s) = w(k)*dz_dmu*alpha;            
            if quadratic_meanfun || quadsqexp_meanfun || quadraticfixedonly_meanfun || quadraticlinonly_meanfun
                mu_grad(:,k,s) = mu_grad(:,k,s) - w(k)./omega.^2.*(mu(:,k) - xm);
            elseif sqexp_meanfun
                mu_grad(:,k,s) = mu_grad(:,k,s) - w(k)*nu_k_se./tau2_mfun.*(mu(:,k) - xm);                
            elseif quadraticmix_meanfun
                mu_grad(:,k,s) = mu_grad(:,k,s) ...
                    - w(k)/beta2./omega.^2.*(mu(:,k) - xm) ...
                    + w(k)*(xm - mu(:,k))./tautilde2_m.*nu2_k ...
                    + w(k)*((xm - mu(:,k))./tautilde2_m.*(nu3_k + alphatilde_m*rho2*(1-1/beta2)));
            end
            if quadsqexp_meanfun
                mu_grad(:,k,s) = mu_grad(:,k,s) - w(k)*nu_k_se./tau2_mfun.*(mu(:,k) - xm_se);
            end
            if quadraticonly_meanfun
                mu_grad(:,k,s) = mu_grad(:,k,s) - w(k)./omega.^2.*mu(:,k);                
            end
            if integrated_meanfun
                switch gp.intmeanfun
                    case 1; du_dmu = zeros(D,1);
                    case 2; du_dmu = [zeros(D,1),eye(D)];
                    case 3; du_dmu = [zeros(D,1),eye(D),diag(2*mu(:,k))];
                    case 4; du_dmu = [zeros(D,1),eye(D),diag(2*mu(:,k)),dmumu{k}];
                end
                mu_grad(:,k,s) = mu_grad(:,k,s) + w(k)*(du_dmu*betabar - dz_dmu*KinvHtbetabar);
            end
        end
        
        if grad_flags(2)
            dz_dsigma = bsxfun(@times, sum(bsxfun(@times,(lambda./tau_k).^2, delta_k.^2 - 1),1), sigma(k)*z_k);
            sigma_grad(k,s) = w(k)*dz_dsigma*alpha;
            if quadratic_meanfun || quadsqexp_meanfun || quadraticonly_meanfun || quadraticfixedonly_meanfun || quadraticlinonly_meanfun
                sigma_grad(k,s) = sigma_grad(k,s) - w(k)*sigma(k)*sum(1./omega.^2.*lambda.^2,1);
            elseif sqexp_meanfun
                sigma_grad(k,s) = sigma_grad(k,s) - w(k)*sigma(k)*sum(lambda.^2./tau2_mfun,1)*nu_k_se ...
                    + w(k)*sigma(k)*sum((mu(:,k) - xm).^2.*lambda.^2./tau2_mfun.^2,1)*nu_k_se;
            elseif quadraticmix_meanfun
                sigma_grad(k,s) = sigma_grad(k,s) ...
                    - w(k)/beta2*sigma(k)*sum(1./omega.^2.*lambda.^2,1) ...
                    - w(k)*nu2_k*sum(sigma(k)*lambda.^2./tautilde2_m.*(1 - (xm - mu(:,k)).^2./tautilde2_m),1) ...
                    - w(k)*nu3_k*sum(sigma(k)*lambda.^2./tautilde2_m.*(1 - (xm - mu(:,k)).^2./tautilde2_m),1) ...
                    - w(k)*(1-1/beta2)*alphatilde_m*sum(2*sqrt(sigmatilde2_m).*sigma(k).*lambda.^2.*(rho2*sqrt(tautilde2_m)-2*sigma(k)^2*lambda.^2*rho2)./tautilde2_m.^(3/2),1);
            end
            if quadsqexp_meanfun
                sigma_grad(k,s) = sigma_grad(k,s) - w(k)*sigma(k)*sum(lambda.^2./tau2_mfun,1)*nu_k_se ...
                    + w(k)*sigma(k)*sum((mu(:,k) - xm_se).^2.*lambda.^2./tau2_mfun.^2,1)*nu_k_se;
            end
            if integrated_meanfun
                switch gp.intmeanfun
                    case 1; du_dsigma = 0;
                    case 2; du_dsigma = zeros(1,1+D);
                    case 3; du_dsigma = [zeros(1,1+D),(2*sigma(k)*lambda.^2)'];
                    case 4; du_dsigma = [zeros(1,1+D),(2*sigma(k)*lambda.^2)',zeros(1,D*(D-1)/2)];
                end
                sigma_grad(k,s) = sigma_grad(k,s) + w(k)*(du_dsigma*betabar - dz_dsigma*KinvHtbetabar);
            end
        end

        if grad_flags(3)
            dz_dlambda = bsxfun(@times, bsxfun(@times, (sigma(k)./tau_k).^2, delta_k.^2 - 1), bsxfun(@times,lambda,z_k));
            lambda_grad(:,s) = lambda_grad(:,s) + w(k)*(dz_dlambda*alpha);
            if quadratic_meanfun || quadsqexp_meanfun || quadraticonly_meanfun || quadraticfixedonly_meanfun || quadraticlinonly_meanfun
                lambda_grad(:,s) = lambda_grad(:,s) - w(k)*sigma(k)^2./omega.^2.*lambda;
            elseif sqexp_meanfun
                lambda_grad(:,s) = lambda_grad(:,s) - w(k)*sigma(k)^2*lambda./tau2_mfun*nu_k_se ...
                    + w(k)*sigma(k)^2*(mu(:,k) - xm).^2.*lambda./tau2_mfun.^2*nu_k_se;
            elseif quadraticmix_meanfun
                lambda_grad(:,s) = lambda_grad(:,s) ...
                    - w(k)/beta2*sigma(k)^2./omega.^2.*lambda ...
                    - w(k)*sigma(k)^2*lambda./tautilde2_m.*(1 - (xm - mu(:,k)).^2./tautilde2_m)*nu2_k ...
                    - w(k)*nu3_k*sigma(k)^2*lambda./tautilde2_m.*(1 - (xm - mu(:,k)).^2./tautilde2_m) ...
                    - w(k)*(1-1/beta2)*alphatilde_m*2*sqrt(sigmatilde2_m).*sigma(k)^2.*lambda.*(rho2*sqrt(tautilde2_m)-2*sigma(k)^2*lambda.^2*rho2)./tautilde2_m.^(3/2);                    
            end
            if quadsqexp_meanfun
                lambda_grad(:,s) = lambda_grad(:,s) - w(k)*sigma(k)^2*lambda./tau2_mfun*nu_k_se ...
                    + w(k)*sigma(k)^2*(mu(:,k) - xm_se).^2.*lambda./tau2_mfun.^2*nu_k_se;                
            end
            if integrated_meanfun
                switch gp.intmeanfun
                    case 1; du_dlambda = zeros(D,1);
                    case 2; du_dlambda = zeros(D,1+D);
                    case 3; du_dlambda = [zeros(D,1+D),diag(2*sigma(k)^2*lambda)];
                    case 4; du_dlambda = [zeros(D,1+D),diag(2*sigma(k)^2*lambda),zeros(D,D*(D-1)/2)];
                end
                lambda_grad(:,s) = lambda_grad(:,s) + w(k)*(du_dlambda*betabar - dz_dlambda*KinvHtbetabar);
            end
        end
        
        if grad_flags(4)
            w_grad(k,s) = I_k;
        end
        
        if compute_var == 2 % Compute only self-variance
            tau_kk = sqrt(2*sigma(k)^2*lambda.^2 + ell.^2 + 2*delta.^2);                
            nf_kk = exp(ln_sf2 + sum_lnell - sum(log(tau_kk)));
            if Lchol
                invKzk = (L\(L'\z_k'))/sn2_eff;
            else
                invKzk = -L*z_k';                
            end
            J_kk = nf_kk - z_k*invKzk;
            
            if integrated_meanfun; error('Integrated basis function unsupported with diagonal covariance only.'); end
            
            varF(s) = varF(s) + w(k)^2*max(eps,J_kk);    % Correct for numerical error
            if separate_K; J_sjk(s,k,k) = J_kk; end            
            
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
                tau_j = sqrt(sigma(j)^2*lambda.^2 + ell.^2 + delta.^2);
                lnnf_j = ln_sf2 + sum_lnell - sum(log(tau_j));
                delta_j = bsxfun(@rdivide,bsxfun(@minus, mu(:,j), gp.X'), tau_j);
                z_j = exp(lnnf_j -0.5 * sum(delta_j.^2,1));                    
                
                tau_jk = sqrt((sigma(j)^2 + sigma(k)^2)*lambda.^2 + ell.^2 + 2*delta.^2);                
                lnnf_jk = ln_sf2 + sum_lnell - sum(log(tau_jk));
                delta_jk = (mu(:,j)-mu(:,k))./tau_jk;
                                
                if Lchol
                    J_jk = exp(lnnf_jk -0.5*sum(delta_jk.^2,1)) ...
                     - z_k*(L\(L'\z_j'))/sn2_eff;
                else
                    J_jk = exp(lnnf_jk -0.5*sum(delta_jk.^2,1)) ...
                     + z_k*(L*z_j');
                end
                
                % Contribution to the variance of integrated mean function
                if integrated_meanfun
                    switch gp.intmeanfun
                        case 1; u_j = 1;
                        case 2; u_j = [1,mu(:,j)'];
                        case 3; u_j = [1,mu(:,j)',(mu(:,j).^2 + sigma(j)^2*lambda.^2)'];
                        case 4; u_j = [1,mu(:,j)',(mu(:,j).^2 + sigma(j)^2*lambda.^2)',mumu_mat(j,:)];
                    end
                    u_j = u_j(plus_idx);
                    J_jk = J_jk + u_k(plus_idx)*(Tplusinv*u_j') ...
                        + (z_k*HKinv')*(Tplusinv*(HKinv*z_j')) ...
                        - u_k(plus_idx)*(Tplusinv*(HKinv*z_j')) ...
                        - (z_k*HKinv')*(Tplusinv*u_j');
                end
                
%                 J(j,k) = w(j)*w(k)*J_jk;
                
                % Off-diagonal elements are symmetric (count twice)
                if j == k
                    varF(s) = varF(s) + w(k)^2*max(eps,J_jk);                                        
                    if separate_K; J_sjk(k,k) = J_jk; end            
                else
                    varF(s) = varF(s) + 2*w(j)*w(k)*J_jk;
                    if separate_K; J_sjk(s,j,k) = J_jk; J_sjk(s,k,j) = J_jk; end            
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

% Return log joint contributions separately if requested
if separate_K
    F = I_sk;
    if compute_var; varF = J_sjk; end
end

end