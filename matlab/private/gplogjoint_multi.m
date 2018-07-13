function [F,varF] = gplogjoint_multi(vp,gp,avg_flag,compute_var)
%GPLOGJOINT_MULTI Expected log joint for multiple GPs with different ALPHAs

% VP is a struct with the variational posterior

if nargin < 3 || isempty(avg_flag); avg_flag = true; end
if nargin < 4; compute_var = []; end
if isempty(compute_var); compute_var = 2*(nargout > 2); end

D = vp.D;           % Number of dimensions
K = vp.K;           % Number of components
Ngp = numel(gp);
mu(:,:) = vp.mu;
sigma(1,:) = vp.sigma;
lambda(:,1) = vp.lambda(:);

Ns = numel(gp(1).post);            % Hyperparameter samples

if all(gp(1).meanfun ~= [0 1 4])
    error('gplogjoint_multi:UnsupportedMeanFun', ...
        'Log joint computation currently only supports zero, constant, or negative quadratic mean functions.');
end

% Using negative quadratic mean?
quadratic_meanfun = gp(1).meanfun == 4;

F = zeros(Ngp,Ns);
if compute_var; varF = zeros(1,Ns); end

nf = 1 / (2*pi)^(D/2);  % Normalization constant

% Loop over hyperparameter samples
for s = 1:Ns
    hyp = gp(1).post(s).hyp;
    
    % Extract GP hyperparameters from HYP
    ell = exp(hyp(1:D));
    ln_sf2 = 2*hyp(D+1);
    sn2 = exp(2*hyp(D+2));
    sum_lnell = sum(hyp(1:D));
    
    if gp(1).meanfun > 0; m0 = hyp(D+3); else; m0 = 0; end
    if quadratic_meanfun
        xm = hyp(D+3+(1:D));
        omega = exp(hyp(2*D+3+(1:D)));        
    end
    
    L = gp(1).post(s).L;
    Lchol = gp(1).post(s).Lchol;
    sn2_eff = sn2*gp(1).post(s).sn2_mult;

    alpha = zeros(size(gp(1).post(s).alpha,1),Ngp);
    for iGP = 1:Ngp; alpha(:,iGP) = gp(iGP).post(s).alpha; end
    
    for k = 1:K

        tau_k = sqrt(sigma(k)^2*lambda.^2 + ell.^2);
        nf_k = nf * exp(ln_sf2 + sum_lnell - sum(log(tau_k)));  % Covariance normalization factor
        delta_k = bsxfun(@rdivide,bsxfun(@minus, mu(:,k), gp(1).X'), tau_k);
        z_k = nf_k * exp(-0.5 * sum(delta_k.^2,1));    

        for iGP = 1:Ngp        
            F(iGP,s) = F(iGP,s) + (z_k*alpha(:,iGP) + m0)/K;
            if quadratic_meanfun
                nu_k = -0.5*sum(1./omega.^2 .* ...
                    (mu(:,k).^2 + sigma(k)^2*lambda.^2 - 2*mu(:,k).*xm + xm.^2),1);            
                F(iGP,s) = F(iGP,s) + nu_k/K;
            end
        end
        
        if compute_var == 2 % Compute only self-variance
            tau_kk = sqrt(2*sigma(k)^2*lambda.^2 + ell.^2);                
            nf_kk = nf * exp(ln_sf2 + sum_lnell - sum(log(tau_kk)));
            if Lchol
                invKzk = (L\(L'\z_k'))/sn2_eff;
            else
                invKzk = -L*z_k';                
            end                
            J_kk = nf_kk - z_k*invKzk;
            varF(s) = varF(s) + J_kk/K^2;
                                    
        elseif compute_var
            for j = 1:K
                tau_j = sqrt(sigma(j)^2*lambda.^2 + ell.^2);
                nf_j = nf * exp(ln_sf2 + sum_lnell - sum(log(tau_j)));
                delta_j = bsxfun(@rdivide,bsxfun(@minus, mu(:,j), gp(1).X'), tau_j);
                z_j = nf_j * exp(-0.5 * sum(delta_j.^2,1));                    
                
                tau_jk = sqrt((sigma(j)^2 + sigma(k)^2)*lambda.^2 + ell.^2);                
                nf_jk = nf * exp(ln_sf2 + sum_lnell - sum(log(tau_jk)));
                delta_jk = (mu(:,j)-mu(:,k))./tau_jk;
                
                if Lchol
                    J_jk = nf_jk*exp(-0.5*sum(delta_jk.^2,1)) ...
                     - z_k*(L\(L'\z_j'))/sn2_eff;
                else
                    J_jk = nf_jk*exp(-0.5*sum(delta_jk.^2,1)) ...
                     + z_k*(L*z_j');                    
                end

                varF(s) = varF(s) + J_jk/K^2;            
            end
            
        end        
        
    end
    

end

if compute_var
    varF = repmat(varF,[Ngp,1]);
end

% Average multiple hyperparameter samples
if Ns > 1 && avg_flag
    Fbar = sum(F,2)/Ns;
    if compute_var
        vv = sum((F - Fbar).^2,2)/(Ns-1);     % Estimated variance of the samples
        % vv = (sum(F.^2,2) - Ns*Fbar.^2)/(Ns-1);
        varF = sum(varF,2)/Ns + vv;
    end
    F = Fbar;
end

end