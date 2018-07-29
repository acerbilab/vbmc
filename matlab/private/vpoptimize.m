function [vp,elbo,elbo_sd,varss] = vpoptimize(Nfastopts,Nslowopts,useEntropyApprox,vp,gp,K,Xstar,ystar,optimState,stats,options,cmaes_opts,prnt)
%VPOPTIMIZE Optimize variational posterior.

%% Set up optimization variables and options

% Number of variational parameters
Ntheta = vp.D*K + K;
if vp.optimize_lambda; Ntheta = Ntheta + vp.D; end

if isempty(Nfastopts); Nfastopts = options.NSelbo * K; end  % Number of initial starting points
if isempty(Nslowopts); Nslowopts = 1; end
nelcbo_fill = zeros(Nfastopts,1);

% Set up empty stats structs for optimization
elbostats = eval_fullelcbo(Nslowopts*2,Ntheta);

% Number of samples per component for MC approximation of the entropy
if isa(options.NSent,'function_handle')
    NSentK = ceil(options.NSent(K)/K);
else
    NSentK = ceil(options.NSent/K);
end

% Number of samples per component for preliminary MC approximation of the entropy
if isa(options.NSentFast,'function_handle')
    NSentKFast = ceil(options.NSentFast(K)/K);
else
    NSentKFast = ceil(options.NSentFast/K);
end

% If the entropy switch is on, it means we are still using the deterministic entropy
if optimState.EntropySwitch
    NSentK = 0;
    NSentKFast = 0;
end

% Confidence weight
elcbo_beta = options.ELCBOWeight; % * sqrt(vp.D) / sqrt(optimState.N);
   
if elcbo_beta ~= 0
    compute_var = 2;    % Use diagonal-only approximation
else
    compute_var = 0;    % No beta, skip variance
end

% Set basic options for deterministic optimizer (FMINUNC)
vbtrain_options = optimoptions('fminunc','GradObj','on','Display','off');

% Compute soft bounds for variational parameters optimization

% Soft-bound loss is computed on MU and SCALE (which is SIGMA times LAMBDA)
if ~isfield(vp,'bounds') || isempty(vp.bounds)
    vp.bounds.mu_lb = Inf(1,vp.D);
    vp.bounds.mu_ub = -Inf(1,vp.D);
    vp.bounds.lnscale_lb = Inf(1,vp.D);
    vp.bounds.lnscale_ub = -Inf(1,vp.D);
    % vp.bounds
end

% Compute HPD region
[~,ord] = sort(gp.y,'descend');
N_hpd = round(options.HPDFrac*size(gp.y,1));
X_hpd = gp.X(ord(1:N_hpd),:);
y_hpd = gp.y(ord(1:N_hpd));

% Set bounds for mean parameters of variational components
vp.bounds.mu_lb = min(min(X_hpd),vp.bounds.mu_lb);
vp.bounds.mu_ub = max(max(X_hpd),vp.bounds.mu_ub);
%vp.bounds.mu_lb = min(min(gp.X),vp.bounds.mu_lb);
%vp.bounds.mu_ub = max(max(gp.X),vp.bounds.mu_ub);

% Set bounds for scale paramters of variational components
lnrange = log(max(gp.X) - min(gp.X));
vp.bounds.lnscale_lb = min(vp.bounds.lnscale_lb,lnrange + log(options.TolLength));
vp.bounds.lnscale_ub = max(vp.bounds.lnscale_ub,lnrange);

thetabnd.lb = [repmat(vp.bounds.mu_lb,[1,K]),repmat(vp.bounds.lnscale_lb,[1,K])];
thetabnd.ub = [repmat(vp.bounds.mu_ub,[1,K]),repmat(vp.bounds.lnscale_ub,[1,K])];
thetabnd.TolCon = options.TolConLoss;

%% Perform quick shotgun evaluation of many candidate parameters

% Generate a bunch of random candidate variational parameters
switch Nslowopts
    case 1
        [vp0_vec,vp0_type] = vbinit(1,Nfastopts,vp,K,Xstar,ystar);
    otherwise
        [vp0_vec1,vp0_type1] = vbinit(1,ceil(Nfastopts/3),vp,K,Xstar,ystar);
        [vp0_vec2,vp0_type2] = vbinit(2,ceil(Nfastopts/3),vp,K,Xstar,ystar);
        [vp0_vec3,vp0_type3] = vbinit(3,Nfastopts-2*ceil(Nfastopts/3),vp,K,Xstar,ystar);
        vp0_vec = [vp0_vec1,vp0_vec2,vp0_vec3];
        vp0_type = [vp0_type1;vp0_type2;vp0_type3];
end

% Quickly estimate ELCBO at each candidate variational posterior
for iOpt = 1:Nfastopts
    [theta0,vp0_vec(iOpt)] = get_theta(vp0_vec(iOpt),vp.optimize_lambda);        
    [nelbo_tmp,~,~,~,varF_tmp] = vbmc_negelcbo(theta0,0,vp0_vec(iOpt),gp,NSentKFast,0,compute_var,options.AltMCEntropy,thetabnd);
    nelcbo_fill(iOpt) = nelbo_tmp + elcbo_beta*sqrt(varF_tmp);
end

% Sort by negative ELCBO
[~,vp0_ord] = sort(nelcbo_fill,'ascend');
vp0_vec = vp0_vec(vp0_ord);
vp0_type = vp0_type(vp0_ord);

%% Perform optimization starting from one or few selected points

for iOpt = 1:Nslowopts
    iOpt_mid = iOpt*2-1;
    iOpt_end = iOpt*2;

    % Select points from best ones depending on subset
    switch Nslowopts
        case 1; idx = 1;
        case 2; if iOpt == 1; idx = find(vp0_type == 1,1); else; idx = find(vp0_type == 2 | vp0_type == 3,1); end
        otherwise; idx = find(vp0_type == (mod(iOpt-1,3)+1),1);
    end

    % idx

    vp0 = rescale_params(vp0_vec(idx));
    vp0_type(idx) = []; vp0_vec(idx) = [];

    theta0 = [vp0.mu(:); log(vp0.sigma(:))];
    if vp.optimize_lambda; theta0 = [theta0; log(vp0.lambda(:))]; end
    % theta0 = min(vp.UB_theta',max(vp.LB_theta', theta0));

    vbtrainmc_fun = @(theta_) vbmc_negelcbo(theta_,elcbo_beta,vp0,gp,NSentK,1,compute_var,options.AltMCEntropy,thetabnd);

    % First, fast optimization via deterministic entropy approximation
    if useEntropyApprox || NSentK == 0
        if NSentK == 0; TolOpt = options.DetEntTolOpt; else; TolOpt = sqrt(options.DetEntTolOpt); end
        vbtrain_options.OptimalityTolerance = TolOpt;
        vbtrain_fun = @(theta_) vbmc_negelcbo(theta_,elcbo_beta,vp0,gp,0,1,compute_var,0,thetabnd);
        try
            [thetaopt,~,~,output] = fminunc(vbtrain_fun,theta0(:)',vbtrain_options);
        catch
            % FMINUNC failed, try with CMA-ES
            if prnt > 0
                fprintf('Cannot optimize variational parameters with FMINUNC. Trying with CMA-ES (slower).\n');
            end
            insigma_mu = repmat(vp.bounds.mu_ub(:) - vp.bounds.mu_lb(:),[vp.K,1]);
            insigma_sigma = ones(K,1);
            if vp.optimize_lambda; insigma_lambda = ones(vp.D,1); else; insigma_lambda = []; end
            insigma = [insigma_mu(:); insigma_sigma(:); insigma_lambda];
            cmaes_opts.EvalParallel = 'off';
            cmaes_opts.TolX = '1e-8*max(insigma)';
            cmaes_opts.TolFun = 1e-6;
            cmaes_opts.TolHistFun = 1e-7;           
            thetaopt = cmaes_modded('vbmc_negelcbo',theta0(:),insigma,cmaes_opts, ...
                elcbo_beta,vp0,gp,0,1,compute_var,0,thetabnd); 
            thetaopt = thetaopt(:)';
        end
        % output, % pause
    else
        thetaopt = theta0(:)';
    end
        
    % Second, refine with unbiased stochastic entropy approximation
    if NSentK > 0
        switch lower(options.StochasticOptimizer)
            case 'adam'                
                [thetaopt,~,theta_lst,fval_lst] = ...
                    fminadam(vbtrainmc_fun,thetaopt,[],[],options.TolFunStochastic);

                if options.ELCBOmidpoint
                    % Recompute ELCBO at best midpoint with full variance and more precision
                    [~,idx_mid] = min(fval_lst);
                    elbostats = eval_fullelcbo(iOpt_mid,theta_lst(idx_mid,:),vp0,gp,elbostats,elcbo_beta,options);
                    % [idx_mid,numel(fval_lst)]
                end
%             case 'fmincon'
%                 vbtrain_options.OptimalityTolerance = options.TolFunStochastic;
%                 [thetaopt,~,~,output] = ...
%                     fmincon(vbtrainmc_fun,thetaopt,[],[],[],[],vp.LB_theta,vp.UB_theta,[],vbtrain_options);
                
            otherwise
                error('vbmc:VPoptimize','Unknown stochastic optimizer.');
        end
    end
        
	% Recompute ELCBO at endpoint with full variance and more precision
    elbostats = eval_fullelcbo(iOpt_end,thetaopt,vp0,gp,elbostats,elcbo_beta,options);
    % toc
    
    vp0_fine(iOpt_mid) = vp0;
    vp0_fine(iOpt_end) = vp0;   % Parameters get assigned later

    % [nelbo,nelcbo,sqrt(varF),G,H]
end

%% Finalize optimization by taking variational parameters with best ELCBO

[~,idx] = min(elbostats.nelcbo);
elbo = -elbostats.nelbo(idx);
elbo_sd = sqrt(elbostats.varF(idx));
varss = elbostats.varss(idx);
vp = vp0_fine(idx);
vp = rescale_params(vp,elbostats.theta(idx,:));

% L = vpbndloss(elbostats.theta(idx,:),vp,thetabnd,thetabnd.TolCon)
% if L > 0
%     lnscale = bsxfun(@plus,log(vp.sigma),log(vp.lambda));    
%     thetaext = [vp.mu(:)',lnscale(:)'];
%     outflag = thetaext < thetabnd.lb(:)' | thetaext > thetabnd.ub(:)';
%     [thetaext;thetabnd.lb(:)'; thetabnd.ub(:)';outflag]    
% end


% idx
% elbostats

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function elbostats = eval_fullelcbo(idx,theta,vp,gp,elbostats,beta,options)
%EVAL_FULLELCBO Evaluate full expected lower confidence bound.

if nargin == 2
    D = theta;
    elbostats.nelbo = Inf(1,idx);
    elbostats.G = NaN(1,idx);
    elbostats.H = NaN(1,idx);
    elbostats.varF = NaN(1,idx);
    elbostats.varss = NaN(1,idx);
    elbostats.nelcbo = Inf(1,idx);
    elbostats.theta = NaN(idx,D);
else
    
    % Number of samples per component for MC approximation of the entropy
    K = vp.K;
    if isa(options.NSentFine,'function_handle')
        NSentFineK = ceil(options.NSentFine(K)/K);
    else
        NSentFineK = ceil(options.NSentFine/K);
    end
        
    theta = theta(:)';
    [nelbo,~,G,H,varF,~,varss] = ...
        vbmc_negelcbo(theta,0,vp,gp,NSentFineK,0,1,options.AltMCEntropy,[]);
    nelcbo = nelbo + beta*sqrt(varF);

    elbostats.nelbo(idx) = nelbo;
    elbostats.G(idx) = G;
    elbostats.H(idx) = H;
    elbostats.varF(idx) = varF;
    elbostats.varss(idx) = varss;
    elbostats.nelcbo(idx) = nelcbo;
    elbostats.theta(idx,1:numel(theta)) = theta;
end

end

function [theta,vp] = get_theta(vp,optimize_lambda)
%GET_THETA Get vector of variational parameters from variational posterior.

vp = rescale_params(vp);
theta = [vp.mu(:); log(vp.sigma(:))];
if optimize_lambda; theta = [theta; log(vp.lambda(:))]; end

end