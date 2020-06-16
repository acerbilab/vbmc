function F = intkernel(X,vp,gp,avg_flag)
%INTKERNEL Expected GP kernel in scalar correlation

if nargin < 4 || isempty(avg_flag); avg_flag = false; end

K = vp.K;           % Number of components
[N,D] = size(X);
mu(:,:) = vp.mu;
sigma(1,:) = vp.sigma;
lambda(:,1) = vp.lambda(:);
w(1,:) = vp.w;

Ns = numel(gp.post);            % Hyperparameter samples

F = zeros(N,Ns);

if isfield(vp,'delta') && ~isempty(vp.delta)
    delta = vp.delta;
else
    delta = 0;
end

% Integrated mean function being used?
integrated_meanfun = isfield(gp,'intmeanfun') && gp.intmeanfun > 0;

if integrated_meanfun
    % Evaluate basis functions
    Hs = gplite_intmeanfun(X,gp.intmeanfun);
end
    
% Loop over hyperparameter samples
for s = 1:Ns
    hyp = gp.post(s).hyp;
    
    % Extract GP hyperparameters from HYP
    ell = exp(hyp(1:D));
    ln_sf2 = 2*hyp(D+1);
    sum_lnell = sum(hyp(1:D));
    
    if integrated_meanfun
        %betabar = gp.post(s).intmean.betabar';
        %KinvHtbetabar = gp.post(s).intmean.HKinv'*betabar;
        plus_idx = gp.intmeanfun_var > 0;
        HKinv = gp.post(s).intmean.HKinv(plus_idx,:);
        Tplusinv = gp.post(s).intmean.Tplusinv;
    end
            
    L = gp.post(s).L;
    Lchol = gp.post(s).Lchol;
    
    sn2_eff = 1/gp.post(s).sW(1)^2;
    
    ddl = sq_dist(bsxfun(@rdivide,X',ell),bsxfun(@rdivide,gp.X',ell));
    ll = exp(ln_sf2 -0.5*ddl);
    
    if Lchol
        zz = (L\(L'\ll'))/sn2_eff;
    else
        zz = -L*ll';
    end

    for k = 1:K
        tau_k = sqrt(sigma(k)^2*lambda.^2 + ell.^2 + delta.^2);
        lnnf_k = ln_sf2 + sum_lnell - sum(log(tau_k));  % Covariance normalization factor
        delta_k = bsxfun(@rdivide,bsxfun(@minus, mu(:,k), gp.X'), tau_k);
        z_k = exp(lnnf_k -0.5 * sum(delta_k.^2,1));

        dd_k = bsxfun(@rdivide,bsxfun(@minus, mu(:,k), X'), tau_k);
        zz_k = exp(lnnf_k -0.5 * sum(dd_k.^2,1));
        
        F(:,s) = F(:,s) + w(k)*(zz_k - z_k*zz)';
        
        % Contribution of integrated mean function
        if integrated_meanfun
            switch gp.intmeanfun
                case 1; u_k = 1;
                case 2; u_k = [1,mu(:,k)'];
                case 3; u_k = [1,mu(:,k)',(mu(:,k).^2 + sigma(k)^2*lambda.^2)'];
                case 4; u_k = [1,mu(:,k)',(mu(:,k).^2 + sigma(k)^2*lambda.^2)',mumu_mat(k,:)];
            end
            
            F(:,s) = F(:,s) + w(k)*((u_k(plus_idx)*(Tplusinv*Hs)) ...
                + ((z_k*HKinv')*(Tplusinv*(HKinv*ll'))) ...
                - (u_k(plus_idx)*(Tplusinv*(HKinv*ll'))) ...
                - ((z_k*HKinv')*(Tplusinv*Hs)))';
        end
        
        
    end
end

% Average multiple hyperparameter samples
if Ns > 1 && avg_flag
    F = mean(F,2);
end

end


