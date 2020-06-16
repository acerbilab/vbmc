function [trinfo,optimState,warp_action] = warp_input_vbmc(vp,optimState,gp,options)
%WARP_INPUT_VBMC Perform input warping of variables.

GPsample_flag = false;

trinfo = vp.trinfo;
trinfo_old = trinfo;
D = vp.D;

if ~isempty(trinfo.R_mat); R_mat_old = trinfo.R_mat; else; R_mat_old = eye(D); end
if ~isempty(trinfo.scale); scale_old = trinfo.scale; else; scale_old = ones(1,D); end

if GPsample_flag
    % Sample from GP
    Ns = 1e3*D;
    X_old = gpsample_vbmc(vp,gp,Ns,0);
elseif options.WarpNonlinear
    X_old = vbmc_rnd(vp,2e4,0,0);
end

%% Compute nonlinear warping
if options.WarpNonlinear    
    X_rev = warpvars_vbmc(X_old,'inv',trinfo_old);
    Nq = 1e3;
    qq = linspace(0.5/Nq,1-0.5/Nq,Nq);
    X_q = quantile(X_rev,qq);
    trinfo = warp_reparam_vbmc(X_q,trinfo,optimState.N,100);
end

%% Compute rotation and scaling in transformed space
if options.WarpRotoScaling
    if options.WarpNonlinear
        X_rev = warpvars_vbmc(X_rev,'d',trinfo);        
        vp_Sigma = cov(X_rev);          % Compute covariance matrix
    elseif GPsample_flag
        % Reverse rotation and scaling
        X_rev = bsxfun(@times,X_old,scale_old)*R_mat_old';
        vp_Sigma = cov(X_rev);          % Compute covariance matrix
    else
        % Get covariance matrix analytically
        [~,VV] = vbmc_moments(vp,0);
        vp_Sigma = R_mat_old*(diag(scale_old)*VV*diag(scale_old))*R_mat_old';
    end

    % Remove low-correlation entries
    if options.WarpRotoCorrThresh > 0
        vp_corr = vp_Sigma ./ sqrt(bsxfun(@times,diag(vp_Sigma),diag(vp_Sigma)'));
        mask_idx = abs(vp_corr) > options.WarpRotoCorrThresh;
        vp_Sigma(~mask_idx) = 0;
    end

    % Regularization of covariance matrix towards diagonal
    if isnumeric(options.WarpCovReg)
        w_reg = options.WarpCovReg;
    else
        w_reg = options.WarpCovReg(optimState.N);
    end
    w_reg = max(0,min(1,w_reg));
    vp_Sigma = (1-w_reg)*vp_Sigma + w_reg*diag(diag(vp_Sigma));

    % Compute whitening transform (rotoscaling)
    [U,S] = svd(vp_Sigma);
    if det(U) < 0; U(:,1) = -U(:,1); end
    %scale = fliplr(sqrt(diag(S+eps))');        
    scale = sqrt(diag(S+eps))';        
    trinfo.R_mat = U;
    trinfo.scale = scale;
end

optimState.trinfo = trinfo;

%% Apply warping

% Adjust stored points after warping
idx_n = 1:optimState.Xn;
X_orig = optimState.X_orig(idx_n,:);
y_orig = optimState.y_orig(idx_n);
X = warpvars_vbmc(X_orig,'dir',trinfo);
dy = warpvars_vbmc(X,'logp',trinfo);
y = y_orig + dy;
optimState.X(idx_n,:) = X;
optimState.y(idx_n) = y;

% Update plausible bounds
warpfun = @(x) warpvars_vbmc(x,'d',trinfo);
if 0
    xx = warpfun(X_old);
    optimState.PLB = quantile(xx,0.05);
    optimState.PUB = quantile(xx,0.95);
elseif 0
    mu = 0.5*(optimState.PLB_orig + optimState.PUB_orig);
    sigma = 0.5*(optimState.PUB_orig - optimState.PLB_orig)/sqrt(D);    
    [~,~,pw(:,:)] = unscent_warp(warpfun,mu,sigma);
    optimState.PLB = min(pw);
    optimState.PUB = max(pw);
else
    Nrnd = 1e3;    
    xx = bsxfun(@plus,bsxfun(@times,rand(Nrnd,D),optimState.PUB_orig-optimState.PLB_orig),optimState.PLB_orig);
    yy = warpfun(xx);
    delta = max(yy) - min(yy);
    optimState.PLB = min(yy) - delta/Nrnd;
    optimState.PUB = max(yy) + delta/Nrnd;    
end

% Update search bounds
warpfun = @(x) warpvars_vbmc(warpvars_vbmc(x,'i',trinfo_old),'d',trinfo);
if 0
    mu = 0.5*(optimState.LB_search + optimState.UB_search);
    sigma = 0.5*(optimState.UB_search - optimState.LB_search);
    [~,~,pw2(:,:)] = unscent_warp(warpfun,mu,sigma);
    optimState.LB_search = min(pw2(:,:));
    optimState.UB_search = max(pw2(:,:));
    optimState.LB_search(~isfinite(mu) | ~isfinite(sigma)) = -Inf;
    optimState.UB_search(~isfinite(mu) | ~isfinite(sigma)) = Inf;
else
    Nrnd = 1e3;    
    xx = bsxfun(@plus,bsxfun(@times,rand(Nrnd,D),optimState.UB_search-optimState.LB_search),optimState.LB_search);
    yy = warpfun(xx);    
    delta = max(yy) - min(yy);
    optimState.LB_search = min(yy) - delta/Nrnd;
    optimState.UB_search = max(yy) + delta/Nrnd;
end

% Update search cache
if ~isempty(optimState.SearchCache)
    optimState.SearchCache = warpfun(optimState.SearchCache);
end

% Increase warping counter
optimState.WarpingCount = optimState.WarpingCount + 1;

% Major change, fully recompute variational posterior and skip active sampling
optimState.RecomputeVarPost = true;
optimState.SkipActiveSampling = true;

% Last warping iteration
optimState.LastWarping = optimState.iter;

% Reset GP hyperparameters
optimState.RunMean = [];
optimState.RunCov = [];        
optimState.LastRunAvg = NaN;

% trinfo

%% Warp action for output display
if options.WarpNonlinear    
    warp_action = 'warp';
else
    warp_action = 'rotoscale';
end

end
