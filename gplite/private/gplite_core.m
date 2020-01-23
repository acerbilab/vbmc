function [nlZ,dnlZ,post,K_mat,Q] = gplite_core(hyp,gp,compute_nlZ,compute_nlZ_grad)
%GPLITE_CORE Core kernel computations for lite GP regression.

%% Initialize GP hyperparameters
    
[N,D] = size(gp.X);         % Number of training points and dimension

Ncov = gp.Ncov;
Nnoise = gp.Nnoise;
Nmean = gp.Nmean;

% Output warping
outwarp_flag = isfield(gp,'outwarpfun') && ~isempty(gp.outwarpfun);
if outwarp_flag
    Noutwarp = gp.Noutwarp;
    hyp_outwarp = hyp(Ncov+Nnoise+Nmean+1:Ncov+Nnoise+Nmean+Noutwarp);
    if compute_nlZ_grad
        [y,dwarp_dt,dwarp_dtheta,d2warp_dthetadt] = gp.outwarpfun(hyp_outwarp,gp.y);
    else
        [y,dwarp_dt] = gp.outwarpfun(hyp_outwarp,gp.y);
    end
    if ~isempty(gp.s2)
        s2 = gp.s2 .* dwarp_dt.^2;  % Warped noise
    else
        s2 = [];
    end
else
    y = gp.y;
    s2 = gp.s2;
end

% Evaluate observation noise on training inputs
hyp_noise = hyp(Ncov+1:Ncov+Nnoise); % Get noise hyperparameters
if compute_nlZ_grad
    [sn2,dsn2] = gplite_noisefun(hyp_noise,gp.X,gp.noisefun,gp.y,s2);
else
    sn2 = gplite_noisefun(hyp_noise,gp.X,gp.noisefun,gp.y,s2);
end
sn2_mult = 1;  % Effective noise variance multiplier

% Evaluate mean function on training inputs
hyp_mean = hyp(Ncov+Nnoise+1:Ncov+Nnoise+Nmean); % Get mean function hyperparameters
if compute_nlZ_grad
    [m,dm] = gplite_meanfun(hyp_mean,gp.X,gp.meanfun,[],gp.meanfun_extras);
else
    m = gplite_meanfun(hyp_mean,gp.X,gp.meanfun,[],gp.meanfun_extras);
end

%% Observed covariance matrix inversion

% Compute kernel matrix K_mat
if gp.covfun(1) == 1
    ell = exp(hyp(1:D));
    sf2 = exp(2*hyp(D+1));
    K_mat = sq_dist(bsxfun(@rdivide,gp.X',ell));
    K_mat = sf2 * exp(-K_mat/2);
else
    hyp_cov = hyp(1:Ncov); % Get covariance function hyperparameters
    if compute_nlZ_grad
        [K_mat,dK_mat] = gplite_covfun(hyp_cov,gp.X,gp.covfun,[]);
    else
        K_mat = gplite_covfun(hyp_cov,gp.X,gp.covfun,[]);        
    end
end

% Use Cholesky representation of posterior for non-small noise
Lchol = min(sn2) >= 1e-6;

if Lchol
    if isscalar(sn2)
        sn2div = sn2;
        sn2_mat = eye(N);
    else
        sn2div = min(sn2);
        sn2_mat = diag(sn2/sn2div);
    end
    
    for iter = 1:10
        [L,p] = chol(K_mat/(sn2div*sn2_mult)+sn2_mat);
        if p > 0; sn2_mult = sn2_mult*10; else; break; end
    end
    sl = sn2div*sn2_mult;
    if nargout > 2
        pL = L;             % L = chol(eye(n)+sW*sW'.*K)
    end
else
    if isscalar(sn2)
        sn2_mat = sn2*eye(N);
    else
        sn2_mat = diag(sn2);
    end
    for iter = 1:10     % Cholesky decomposition until it works
        [L,p] = chol(K_mat+sn2_mult*sn2_mat);
        if p > 0; sn2_mult = sn2_mult*10; else; break; end
    end
    sl = 1;
    if nargout > 2
        pL = -L\(L'\eye(N));    % L = -inv(K+inv(sW^2))
    end
end

alpha = L\(L'\(y-m)) / sl;     % alpha = inv(K_mat + diag(sn2)) * (y - m)  I

%% Integrated basis functions

if gp.intmeanfun > 0    
    bb = gp.intmeanfun_mean(:);
    BB = gp.intmeanfun_var(:);
    
    H = gplite_intmeanfun(gp.X,gp.intmeanfun);
    plus_idx = (BB > 0);    % Non-delta parameters
    betabar = zeros(1,size(H,1));
    if any(~plus_idx)
        T_plus = diag(1./BB(plus_idx)) + H(plus_idx,:)*(L\(L'\H(plus_idx,:)')/sl);
        T_chol = chol(T_plus);
        betabar(plus_idx) = T_chol \ (T_chol' \ (bb(plus_idx)./BB(plus_idx) + H(plus_idx,:)*alpha));
        betabar(~plus_idx) = bb(~plus_idx);
    else
        T_plus = diag(1./BB) + H*(L\(L'\H')/sl);
        T_chol = chol(T_plus);
%        betabar(:) = T_plus \ (bb./BB + H*alpha);
        betabar(:) = T_chol \ (T_chol' \ (bb./BB + H*alpha));
    end
end


%% Negative log marginal likelihood computation
nlZ = []; dnlZ = []; Q = [];

if compute_nlZ
    Nhyp = size(hyp,1);    
    
    if gp.intmeanfun > 0
        % Negative log marginal likelihood with integrated basis functions
        prec_idx = BB > 0 & isfinite(BB);
        inf_idx = isinf(BB);
                
        vagueall_flag = all(inf_idx);  % Vague prior on *all* basis functions?
        vagueany_flag = any(inf_idx);  % Some vague priors?
        precany_flag = any(prec_idx);  % Some precise priors?
        
        % Compute first quadratic term
        nu = y-m - H'*bb;
        if vagueall_flag
            nlZ_1 = nu'*alpha/2;
        else
            if precany_flag
                HBH_prec = H(prec_idx,:)'*bsxfun(@times,BB(prec_idx),H(prec_idx,:));
                N_mat = L'*L*sl + HBH_prec;
                N_chol = chol(N_mat);                
                % Ninv = N_mat\eye(N);
                Ninv = N_chol\(N_chol'\eye(N));
                nlZ_1 = nu'*(Ninv*nu)/2;
            else
                nlZ_1 = nu'*(L\(L'\nu))/sl/2;
            end
        end
        
        % Compute second quadratic term (vague prior contribution)
        if vagueany_flag
            if ~precany_flag
                W = chol(H*(L\(L'\H'))/sl);                
                % A = H'*((H*Kinv*H')\H);
                A_mat = H'*(W\(W'\H));
                nlZ_v = -nu'*(L\(L'\(A_mat*(L\(L'\nu)))))/sl^2/2;
            else
                Hinf = H(inf_idx,:);
                W = chol(Hinf*Ninv*Hinf');
                % A = Hinf'*((Hinf*Ninv*Hinf')\Hinf);
                A_mat = Hinf'*(W\(W'\Hinf));
                C = Ninv*A_mat*Ninv;                
                nlZ_v = -nu'*C*nu/2;
            end
        else
            nlZ_v = 0;
        end
        
        % Compute determinants
        nldet = sum(log(diag(L)));                  % First component
        if precany_flag                             % Precise priors
            nldet = nldet + sum(log(BB(prec_idx)))/2;
            Tprec_idx = isfinite(BB(BB > 0));
            nldet = nldet + log(det(T_plus(Tprec_idx,Tprec_idx)))/2;
        end
        if vagueany_flag
            nldet = nldet + sum(log(diag(W)));
        end
        
        nlZ = nlZ_1 + nlZ_v + nldet + N*log(2*pi*sl)/2 - sum(inf_idx)*log(2*pi)/2;        
        
    else
        % Compute negative log marginal likelihood
        nlZ = (y-m)'*alpha/2 + sum(log(diag(L))) + N*log(2*pi*sl)/2;
    end
    
    if outwarp_flag     % Jacobian correction for output warping
        nlZ = nlZ - sum(log(abs(dwarp_dt)));
    end

    if compute_nlZ_grad
        % Compute gradient of negative log marginal likelihood

        dnlZ = zeros(Nhyp,1);    % allocate space for derivatives
        
        if gp.intmeanfun > 0
            % Gradient with integrated basis functions
            Kinv = L\(L'\eye(N))/sl;
            if ~precany_flag; Ninv = Kinv; end
            chi = Ninv*nu;
            Q = Kinv - chi*chi';
            if vagueany_flag
                phi = Ninv*A_mat*chi;
                Q = Q - phi*phi' + 2*chi*phi';
                if ~precany_flag
                    Q = Q - Kinv*A_mat*Kinv;                
                else
                    Q = Q - Ninv*A_mat*Ninv;
                end
            else
                phi = 0;                
            end
            if precany_flag
                Q = Q - Kinv*H(prec_idx,:)'*(T_plus(Tprec_idx,Tprec_idx)\(H(prec_idx,:)*Kinv));
            end            
        else        
            Q = L\(L'\eye(N))/sl - alpha*alpha';
        end
        
        if gp.covfun(1) == 1
            for i = 1:D                             % Grad of cov length scales
                K_temp = K_mat .* sq_dist(gp.X(:,i)'/ell(i));
                dnlZ(i) = sum(sum(Q.*K_temp))/2;
            end
            dnlZ(D+1) = sum(sum(Q.*(2*K_mat)))/2;   % Grad of cov output scale
        else            
            for i = 1:Ncov                          % Grad of cov hyperparameters
                dnlZ(i) = sum(sum(Q.*dK_mat(:,:,i)))/2;
            end
        end

        % Gradient of GP likelihood
        if isscalar(sn2)
            trQ = trace(Q);
            for i = 1:Nnoise; dnlZ(Ncov+i) = 0.5*sn2_mult*dsn2(i)*trQ; end
        else
            dgQ = diag(Q);
            for i = 1:Nnoise; dnlZ(Ncov+i) = 0.5*sn2_mult*sum(dsn2(:,i).*dgQ); end
            if outwarp_flag
                error('Input-dependent noise not supported with output warping yet.');
            end
        end

        % Gradient of mean function
        if Nmean > 0
            if gp.intmeanfun > 0
                % Mean function gradient with integrated basis functions
                dnlZ(Ncov+Nnoise+(1:Nmean)) = -dm'*(chi - phi);                
            else
                dnlZ(Ncov+Nnoise+(1:Nmean)) = -dm'*alpha;
            end
        end
        
        % Gradient of output warping function
        if outwarp_flag && Noutwarp > 0
            if gp.intmeanfun > 0
                error('Integrated basis functions are not supported with output warping yet.');
            end
            for i = 1:Noutwarp
                dnlZ(Ncov+Nnoise+Nmean+i) = dwarp_dtheta(:,i)'*alpha ...
                    - sum(d2warp_dthetadt(:,i)./dwarp_dt);
            end
        end

    end
end

%% Output posterior struct if requested
if nargout > 2
    post.hyp = hyp;
    post.alpha = alpha;
    post.sW = ones(N,1)./sqrt(min(sn2)*sn2_mult);   % sqrt of noise precision vector
    post.L = pL;
    post.sn2_mult = sn2_mult;
    post.Lchol = Lchol;
    if gp.intmeanfun > 0
        post.intmean.HKinv = H*(L\(L'\eye(N))/sl);
         % Inverse reduced T (only positive variances)
        post.intmean.Tplusinv = T_chol \ (T_chol' \eye(size(T_chol)));
        post.intmean.betabar = betabar;
    end
end


end