function [gp,hyp] = gplite_train(hyp0,Ns,X,y,meanfun,hprior,warp,options)
%GPLITE_TRAIN Train lite Gaussian Process hyperparameters.

% Fix functions

if nargin < 5; meanfun = []; end
if nargin < 6; hprior = []; end
if nargin < 7; warp = []; end
if nargin < 8; options = []; end

% Default mean function is constant
if isempty(meanfun); meanfun = 'const'; end

Nopts = [];
if isfield(options,'Nopts'); Nopts = options.Nopts; end
if isempty(Nopts); Nopts = 3; end   % Number of hyperparameter optimization runs

Ninit = [];
if isfield(options,'Ninit'); Ninit = options.Ninit; end
if isempty(Ninit); Ninit = 2^10; end   % Initial design size for hyperparameter optimization

Thin = [];
if isfield(options,'Thin'); Thin = options.Thin; end
if isempty(Thin); Thin = 5; end   % Thinning for hyperparameter sampling

Burnin = [];
if isfield(options,'Burnin'); Burnin = options.Burnin; end
if isempty(Burnin); Burnin = Thin*Ns; end   % Initial design size for hyperparameter optimization

DfBase = [];
if isfield(options,'DfBase'); DfBase = options.DfBase; end
if isempty(DfBase); DfBase = 7; end   % Default degrees of freedom for Student's t prior

[N,D] = size(X);            % Number of training points and dimension

ToL = 1e-6;

% Set up warped GP
if ~isempty(warp)
    if Ns > 0
        error('gplite_train:NoInputWarpingSampling',...
            'Sampling hyperparameters is currently not supported with input warping.');
    end
    warp.trinfo = warpvars(D,warp.LB,warp.UB);    
    if warp.Nwarp > 0
        warp.trinfo.type = 9*ones(1,D); % Kumaraswamy-logistic transform
        unbnd = (warp.LB == -Inf & warp.UB == Inf);
        warp.trinfo.type(unbnd) = 10;   % Logit-Kumaraswamy-logistic transform
        warp.trinfo.alpha = ones(1,D);
        warp.trinfo.beta = ones(1,D);
    end
    if isfield(warp,'mu') && ~isempty(warp.mu); warp.trinfo.mu = warp.mu; end
    if isfield(warp,'delta') && ~isempty(warp.delta); warp.trinfo.delta = warp.delta; end
    if ~isfield(warp,'logpdf_flag') || isempty(warp.logpdf_flag)
        warp.logpdf_flag = false;
    end
    if isfield(warp,'R_mat'); warp.trinfo.R_mat = warp.R_mat; end
    if isfield(warp,'scale'); warp.trinfo.scale = warp.scale; end
    input_warping = 1;
else
    input_warping = 0;
end

if input_warping
    X_prior = warpvars(X,'dir',warp.trinfo);
    y_prior = y + warpvars(X_prior,'logpdf',warp.trinfo);
else
    X_prior = X;
    y_prior = y;
end

Ncov = D+1;     % Number of covariance function hyperparameters

% Get mean function hyperparameter info
[Nmean,meaninfo] = gplite_meanfun([],X_prior,meanfun,y_prior);

if isempty(hyp0); hyp0 = zeros(Ncov+Nmean+1,1); end
[Nhyp,N0] = size(hyp0);      % Hyperparameters

LB = [];
UB = [];
if isfield(hprior,'LB'); LB = hprior.LB; end
if isfield(hprior,'UB'); UB = hprior.UB; end
if isempty(LB); LB = NaN(1,Nhyp); end
if isempty(UB); UB = NaN(1,Nhyp); end
LB = LB(:)'; UB = UB(:)';

if ~isfield(hprior,'mu') || isempty(hprior.mu)
    hprior.mu = NaN(Nhyp,1);
end
if ~isfield(hprior,'sigma') || isempty(hprior.sigma)
    hprior.sigma = NaN(Nhyp,1);
end
if ~isfield(hprior,'df') || isempty(hprior.df)
    hprior.df = DfBase*ones(Nhyp,1);
end
if numel(hprior.mu) < Nhyp; hprior.mu = [hprior.mu(:); NaN(Nhyp-numel(hprior.mu),1)]; end
if numel(hprior.sigma) < Nhyp; hprior.sigma = [hprior.sigma(:); NaN(Nhyp-numel(hprior.sigma),1)]; end
if numel(hprior.df) < Nhyp; hprior.df = [hprior.df(:); NaN(Nhyp-numel(hprior.df),1)]; end


% Default hyperparameter lower and upper bounds, if not specified
width = max(X_prior) - min(X_prior);
height = max(y_prior)-min(y_prior);


% Read hyperparameter bounds, if specified; otherwise set defaults
LB_ell = LB(1:D);   
idx = isnan(LB_ell);                 LB_ell(idx) = log(width(idx))+log(ToL);
LB_sf = LB(D+1);        if isnan(LB_sf); LB_sf = log(height)+log(ToL); end
LB_sn = LB(Ncov+1);     if isnan(LB_sn); LB_sn = log(ToL); end

% Set mean function hyperparameters lower bounds
LB_mean = LB(Ncov+2:D+2+Nmean);
idx = isnan(LB_mean);
LB_mean(idx) = meaninfo.LB(idx);

LB_alpha = -5*ones(1,D);
LB_beta = -5*ones(1,D);
if ~input_warping; LB_alpha = []; LB_beta = []; end
LB = [LB_ell,LB_sf,LB_sn,LB_mean,LB_alpha,LB_beta];

UB_ell = UB(1:D);   
idx = isnan(UB_ell);    UB_ell(idx) = log(width(idx)*10);
UB_sf = UB(D+1);        if isnan(UB_sf); UB_sf = log(height*10); end
UB_sn = UB(Ncov+1);     if isnan(UB_sn); UB_sn = log(height); end

% Set mean function hyperparameters upper bounds
UB_mean = UB(Ncov+2:D+2+Nmean);
idx = isnan(UB_mean);
UB_mean(idx) = meaninfo.UB(idx);

UB_alpha = 5*ones(1,D);
UB_beta = 5*ones(1,D);
if ~input_warping; UB_alpha = []; UB_beta = []; end
UB = [UB_ell,UB_sf,UB_sn,UB_mean,UB_alpha,UB_beta];

UB = max(LB,UB);

% Plausible bounds for generation of starting points
PLB_ell = log(width)+0.5*log(ToL);
PUB_ell = log(width);

PLB_sf = log(height)+0.5*log(ToL);
PUB_sf = log(height);

PLB_sn = 0.5*log(ToL);
PUB_sn = log(std(y_prior));

PLB_mean = meaninfo.PLB;
PUB_mean = meaninfo.PUB;

PLB_alpha = -2*ones(1,D);
PUB_alpha = 2*ones(1,D);

PLB_beta = -2*ones(1,D);
PUB_beta = 2*ones(1,D);

if ~input_warping; PLB_alpha = []; PUB_alpha = []; PLB_beta = []; PUB_beta = []; end

PLB = [PLB_ell,PLB_sf,PLB_sn,PLB_mean,PLB_alpha,PLB_beta];
PUB = [PUB_ell,PUB_sf,PUB_sn,PUB_mean,PUB_alpha,PUB_beta];

PLB = min(max(PLB,LB),UB);
PUB = max(min(PUB,UB),LB);

if input_warping; gradobj = 'on'; else; gradobj = 'on'; end
gptrain_options = optimoptions('fmincon','GradObj',gradobj,'Display','off');    

%% Hyperparameter optimization
if Ns > 0
    gptrain_options.OptimalityTolerance = 0.1;  % Limited optimization
else
    gptrain_options.OptimalityTolerance = 1e-6;        
end

hyp = zeros(Nhyp,Nopts);
nll = Inf(1,Nopts);

% Initialize GP
if input_warping
    gp.X = X;   % X and y must be in pre-warping coordinate space
    gp.y = y;
    gp.post = [];
    gp.hyp = hyp0(:,1);
    gp.warp = warp;
    gp.meanfun = meanfun;
    gp = gp_objfun(hyp0(:,1),gp,[],1,0);
    gp.X = X; gp.y = y;
else
    gp = gplite_post(hyp0(:,1),X,y,meanfun);
end

% Define objective functions for optimization and sampling
gpoptimize_fun = @(hyp_) gp_objfun(hyp_(:),gp,hprior,0,0);
gpsample_fun = @(hyp_) gp_objfun(hyp_(:),gp,hprior,0,1);

% First evaluate GP log posterior on an informed space-filling design
if Ninit > 0
    optfill.FunEvals = Ninit;
    [~,~,~,output_fill] = fminfill(gpoptimize_fun,hyp0',LB,UB,PLB,PUB,hprior,optfill);
    hyp(:,:) = output_fill.X(1:Nopts,:)';
    widths = std(output_fill.X,[],1);
else
    nll = Inf(1,size(hyp0,2));
    for i = 1:size(hyp0,2); nll(i) = gpoptimize_fun(hyp0(:,i)); end
    [nll,ord] = sort(nll,'ascend');
    hyp = hyp0(:,ord);
    widths = PUB - PLB;
end

% if input_warping
%     derivcheck(gpoptimize_fun,hyp0(:,1),1);
% end

%tic
% Perform optimization from most promising NOPTS hyperparameter vectors
for iTrain = 1:Nopts
    try
        hyp(:,iTrain) = min(UB'-eps(UB'),max(LB'+eps(LB'),hyp(:,iTrain)));
        [hyp(:,iTrain),nll(iTrain)] = ...
            fmincon(gpoptimize_fun,hyp(:,iTrain),[],[],[],[],LB,UB,[],gptrain_options);
    catch
        % Could not optimize, keep starting point
    end
end
%toc

[~,idx] = min(nll); % Take best hyperparameter vector

%% Sample from best hyperparameter vector using slice sampling
if Ns > 0
    sampleopts.Thin = Thin;
    sampleopts.Burnin = Burnin;
    sampleopts.Display = 'off';
    sampleopts.Diagnostics = false;
    [samples,fvals,exitflag,output] = ...
        slicesamplebnd(gpsample_fun,hyp(:,idx)',Ns,widths,LB,UB,sampleopts);
    hyp = samples';
    
%     nuts_opt.M = Ns*sampleopts.Thin;
%     nuts_opt.Madapt= 10*Ns;
%     [hyp_nuts,fvals_nuts,diagn_nuts] = hmc_nuts(@gpsample_fun,hyp(:,idx)',nuts_opt);
else
    hyp = hyp(:,idx);
end

% Recompute GP with finalized hyperparameters
gp = gp_objfun(hyp,gp,[],1);

% Check GP posteriors
% for s = 1:numel(gp.post)
%     if ~all(isfinite(gp.post(s).L(:)))
%         pause
%     end
% end


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nlZ,dnlZ] = gp_objfun(hyp,gp,hprior,gpflag,swapsign)
%GPLITE_OBJFUN Objective function for hyperparameter training.

if nargin < 5 || isempty(swapsign); swapsign = 0; end

compute_grad = nargout > 1 && ~gpflag;
[N,D] = size(gp.X);

if isfield(gp,'warp') && ~isempty(gp.warp)
    Nwarp = gp.warp.Nwarp;
    hyp_warp = hyp(end-Nwarp+1:end);          % Warping parameters

    if Nwarp > 0
        alpha = exp(hyp_warp(1:D));
        beta = exp(hyp_warp(D+(1:D)));
        gp.warp.trinfo.alpha = alpha(:)';
        gp.warp.trinfo.beta = beta(:)';
    end

    gp.X = warpvars(gp.X,'dir',gp.warp.trinfo);
    if gp.warp.logpdf_flag
        % The function is a log pdf, so apply log Jacobian correction
        gp.y = gp.y + warpvars(gp.X,'logp',gp.warp.trinfo);
    end
    warp = gp.warp;
else
    Nwarp = 0;
    warp = [];
end

if gpflag
    gp = gplite_post(hyp(1:end-Nwarp,:),gp.X,gp.y,gp.meanfun);
    if ~isempty(warp); gp.warp = warp; end
    nlZ = gp;
else

    % Compute negative log marginal likelihood (without prior)
    if compute_grad && Nwarp > 0
        if ~isempty(warp.trinfo.R_mat); R_mat = warp.trinfo.R_mat; else; R_mat = eye(D); end
        if ~isempty(warp.trinfo.scale); scale = warp.trinfo.scale; else; scale = ones(1,D); end        
        
        [nlZ,dnlZ,post,K_mat,Q] = gplite_nlZ(hyp(1:end-Nwarp,:),gp,[]);

        % Get gradient of warpings (ignores roto-scaling)
        dg_inv = warpvars(gp.X,'g',gp.warp.trinfo);
        
        % Get derivatives wrt warping parameters for warping and its derivative
        dgdtheta = warpvars(gp.X,'f',gp.warp.trinfo);
        dgprimedtheta = warpvars(gp.X,'m',gp.warp.trinfo);
                
        % Compute gradient of warping parameters
        ell2 = exp(2*hyp(1:D));
        
        dnlZ_warp = zeros(Nwarp,1);
        
        XX = zeros(N,N,D);  % Precompute matrix of cross-differences
        for j = 1:D
            XX(:,:,j) = bsxfun(@minus,gp.X(:,j)/ell2(j),gp.X(:,j)'/ell2(j));
        end
        
        for k = 1:D
            dxi_dalpha = bsxfun(@times,(1./scale .* R_mat(k,:)), dgdtheta(:,k));
            dxi_dbeta = bsxfun(@times,(1./scale .* R_mat(k,:)), dgdtheta(:,k+D));
            
            % Gradients due to warping of the covariance matrix
            K_temp = zeros(size(K_mat));
            for j = 1:D
                K_temp = K_temp + XX(:,:,j).*bsxfun(@minus,dxi_dalpha(:,j),dxi_dalpha(:,j)');
            end
            K_temp = -K_temp .* K_mat;
            dnlZ_warp(k) = dnlZ_warp(k) + sum(sum(Q.*K_temp))/2;
            
            K_temp = zeros(size(K_mat));
            for j = 1:D
                K_temp = K_temp + XX(:,:,j).*bsxfun(@minus,dxi_dbeta(:,j),dxi_dbeta(:,j)');
            end
            K_temp = -K_temp .* K_mat;
            dnlZ_warp(k+D) = dnlZ_warp(k+D) + sum(sum(Q.*K_temp))/2;
            
            % Gradients due to warping of the mean function
            switch gp.meanfun
                case {0,1}

                case 4
                    hyp_mean = hyp(gp.Ncov+2:gp.Ncov+1+gp.Nmean); % Get mean function hyperparameters
                    xm = hyp_mean(1+(1:D))';
                    omega = exp(hyp_mean(D+1+(1:D)))';
                    dm = -bsxfun(@rdivide,bsxfun(@minus,gp.X,xm), omega.^2);

                    dnlZ_warp(k) = dnlZ_warp(k) - sum(dm.*dxi_dalpha,2)'*post.alpha;
                    dnlZ_warp(k+D) = dnlZ_warp(k+D) - sum(dm.*dxi_dbeta,2)'*post.alpha;

                otherwise
                    error('gplite_train:WarpedMeanFun', ...
                        'Unsupported mean function for input warping.');
            end
        end
                    
        % Gradients due to warping of the function value
        if gp.warp.logpdf_flag
            dnlZ_warp(:) = dnlZ_warp(:) - (dgprimedtheta(:,:).*repmat(dg_inv(:,:),[1 2]))'*post.alpha;
        end
        
        % Jacobian correction for log representation of warping hyperparameters
        dnlZ_warp = dnlZ_warp .* exp(hyp_warp(:));
        
        % Add warping parameters gradients
        dnlZ = [dnlZ; dnlZ_warp];
        
        
    elseif compute_grad
        [nlZ,dnlZ] = gplite_nlZ(hyp(1:end-Nwarp,:),gp,[]);
    else
        nlZ = gplite_nlZ(hyp(1:end-Nwarp,:),gp,[]);
    end
    

    % Add log prior if present, with all parameters
    if ~isempty(hprior)
        if compute_grad
            [P,dP] = gplite_hypprior(hyp,hprior);
            nlZ = nlZ - P;
            dnlZ = dnlZ - dP;
        else
            P = gplite_hypprior(hyp,hprior);
            nlZ = nlZ - P;
        end
    end
    
    % Swap sign of negative log marginal likelihood (e.g., for sampling)
    if swapsign
        nlZ = -nlZ;
        if compute_grad; dnlZ = -dnlZ; end
    end
    
end

end