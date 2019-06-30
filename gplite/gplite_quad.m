function [F,varF] = gplite_quad(gp,mu,sigma,ssflag)
%GPLITE_QUAD Bayesian quadrature for given Gaussian process.

if nargin < 4 || isempty(ssflag); ssflag = false; end

compute_var = nargout > 1;     % Compute variance of the integral?

[N,D] = size(gp.X);            % Number of training points and dimension
Ns = numel(gp.post);           % Hyperparameter samples

% Number of GP hyperparameters
Ncov = gp.Ncov;
Nnoise = gp.Nnoise;
Nmean = gp.Nmean;

if all(gp.meanfun ~= [0 1 4 6 8])
    error('gplite_quad:UnsupportedMeanFun', ...
        'Bayesian quadrature currently only supports zero, constant, negative quadratic, or squared exponential mean functions.');
end

if gp.covfun ~= 1
    error('gplite_quad:UnsupportedCovFun', ...
        'Bayesian quadrature only supports the squared exponential kernel.');
end

Nstar = size(mu,1);
if size(sigma,1) == 1; sigma = repmat(sigma,[Nstar,1]); end

% Which mean function is being used?
quadratic_meanfun = gp.meanfun == 4;
sqexp_meanfun = gp.meanfun == 6;
quadsqexp_meanfun = gp.meanfun == 8;

F = zeros(Nstar,Ns);
if compute_var; varF = zeros(Nstar,Ns); end

% Loop over hyperparameter samples
for s = 1:Ns
    hyp = gp.post(s).hyp;
    
    % Extract GP hyperparameters from HYP
    ell(1,:) = exp(hyp(1:D));
    ln_sf2 = 2*hyp(D+1);
    sum_lnell = sum(hyp(1:D));
    
    % GP mean function hyperparameters
    if gp.meanfun > 0; m0 = hyp(Ncov+Nnoise+1); else; m0 = 0; end
    if quadratic_meanfun || sqexp_meanfun || quadsqexp_meanfun
        xm(1,:) = hyp(Ncov+Nnoise+1+(1:D));
        omega(1,:) = exp(hyp(Ncov+Nnoise+D+1+(1:D)));
        if sqexp_meanfun
            h = exp(hyp(Ncov+Nnoise+2*D+2));
        end
    end
    if quadsqexp_meanfun
        xm_se(1,:) = hyp(Ncov+Nnoise+2*D+1+(1:D));
        omega_se(1,:) = exp(hyp(Ncov+Nnoise+3*D+1+(1:D)));
        h_se = hyp(Ncov+Nnoise+4*D+2);        
    end
    
    % GP posterior parameters
    alpha = gp.post(s).alpha;
    L = gp.post(s).L;
    Lchol = gp.post(s).Lchol;
    
    sn2 = exp(2*hyp(Ncov+1));
    sn2_eff = sn2*gp.post(s).sn2_mult;

    % Compute posterior mean of the integral
    tau = sqrt(bsxfun(@plus,sigma.^2,ell.^2));
    lnnf = ln_sf2 + sum_lnell - sum(log(tau),2);  % Covariance normalization factor
    sumdelta2 = zeros(Nstar,N);
    for i = 1:D
        sumdelta2 = sumdelta2 + bsxfun(@rdivide,bsxfun(@minus, mu(:,i), gp.X(:,i)'),tau(:,i)).^2;
    end
    z = exp(bsxfun(@minus,lnnf,0.5*sumdelta2));
    F(:,s) = z*alpha + m0;

    if quadratic_meanfun || quadsqexp_meanfun
        nu_k = -0.5*sum(1./omega.^2 .* ...
            bsxfun(@plus,mu.^2 + sigma.^2 - bsxfun(@times,2*mu,xm), xm.^2),2);            
        F(:,s) = F(:,s) + nu_k;        
    elseif sqexp_meanfun
        tau2_mfun = bsxfun(@plus,sigma.^2,omega.^2);
        s2 = (bsxfun(@minus,mu,xm).^2)./tau2_mfun;
        nu_se = h*prod(bsxfun(@rdivide,omega,sqrt(tau2_mfun)),2).*exp(-0.5*sum(s2,2));            
        F(:,s) = F(:,s) + nu_se;
    end
    if quadsqexp_meanfun
        tau2_mfun = bsxfun(@plus,sigma.^2,omega_se.^2);
        s2 = (bsxfun(@minus,mu,xm_se).^2)./tau2_mfun;
        nu_se = h_se*prod(bsxfun(@rdivide,omega_se,sqrt(tau2_mfun)),2).*exp(-0.5*sum(s2,2));
        F(:,s) = F(:,s) + nu_se;
    end

    % Compute posterior variance of the integral
    if compute_var
        tau_kk = sqrt(bsxfun(@plus,2*sigma.^2,ell.^2));                
        nf_kk = exp(ln_sf2 + sum_lnell - sum(log(tau_kk),2));
        if Lchol
            invKzk = (L\(L'\z'))/sn2_eff;
        else
            invKzk = -L*z';                
        end                
        J_kk = nf_kk - sum(z.*invKzk',2);
        varF(:,s) = max(eps,J_kk);    % Correct for numerical error
    end
    
end

% Unless predictions for samples are requested separately, average over samples
if Ns > 1 && ~ssflag
    Fbar = sum(F,2)/Ns;
    if compute_var
        varFss = sum((F - Fbar).^2,2)/(Ns-1);     % Estimated variance of the samples
        varF = sum(varF,2)/Ns + varFss;
    end
    F = Fbar;
end
