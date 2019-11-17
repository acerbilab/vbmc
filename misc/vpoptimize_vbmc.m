function [vp,elbo,elbo_sd,G,H,varG,varH,varss,pruned] = vpoptimize_vbmc(Nfastopts,Nslowopts,vp,gp,K,optimState,options,prnt)
%VPOPTIMIZE Optimize variational posterior.

if nargin < 8 || isempty(prnt); prnt = 0; end

% Assign default values to OPTIMSTATE
if ~isfield(optimState,'delta'); optimState.delta = 0; end
if ~isfield(optimState,'EntropySwitch'); optimState.EntropySwitch = false; end
if ~isfield(optimState,'Warmup'); optimState.Warmup = ~vp.optimize_weights; end
if ~isfield(optimState,'temperature'); optimState.temperature = 1; end

%% Set up optimization variables and options

vp.delta = optimState.delta(:);

% Number of variational parameters
Ntheta = K;
if vp.optimize_mu; Ntheta = Ntheta + vp.D*K; end
if vp.optimize_lambda; Ntheta = Ntheta + vp.D; end
if vp.optimize_weights; Ntheta = Ntheta + vp.K; end

if isempty(Nfastopts) % Number of initial starting points
    if isa(options.NSelbo,'function_handle')
        Nfastopts = ceil(options.NSelbo(K));
    else
        Nfastopts = ceil(options.NSelbo);
    end
end
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

% If only one component, use analytical expression for the entropy
if K == 1
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
[vp,thetabnd] = vpbounds(vp,gp,options,K);

%% Perform quick shotgun evaluation of many candidate parameters

% Get high-posterior density points
[Xstar,ystar] = gethpd_vbmc(gp.X,gp.y,options.HPDFrac);

% Generate a bunch of random candidate variational parameters
switch Nslowopts
    case 1
        [vp0_vec,vp0_type] = vbinit_vbmc(1,Nfastopts,vp,K,Xstar,ystar);
    otherwise
        [vp0_vec1,vp0_type1] = vbinit_vbmc(1,ceil(Nfastopts/3),vp,K,Xstar,ystar);
        [vp0_vec2,vp0_type2] = vbinit_vbmc(2,ceil(Nfastopts/3),vp,K,Xstar,ystar);
        [vp0_vec3,vp0_type3] = vbinit_vbmc(3,Nfastopts-2*ceil(Nfastopts/3),vp,K,Xstar,ystar);
        vp0_vec = [vp0_vec1,vp0_vec2,vp0_vec3];
        vp0_type = [vp0_type1;vp0_type2;vp0_type3];
end

% Quickly estimate ELCBO at each candidate variational posterior
for iOpt = 1:Nfastopts
    [theta0,vp0_vec(iOpt)] = get_vptheta(vp0_vec(iOpt),vp.optimize_mu,vp.optimize_lambda,vp.optimize_weights);        
    [nelbo_tmp,~,~,~,varF_tmp] = negelcbo_vbmc(theta0,0,vp0_vec(iOpt),gp,NSentKFast,0,compute_var,options.AltMCEntropy,thetabnd);
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

    if vp.optimize_mu; theta0 = vp0.mu(:); else theta0 = []; end
    theta0 = [theta0; log(vp0.sigma(:))];
    if vp.optimize_lambda; theta0 = [theta0; log(vp0.lambda(:))]; end
    if vp.optimize_weights; theta0 = [theta0; log(vp0.w(:))]; end
    % theta0 = min(vp.UB_theta',max(vp.LB_theta', theta0));

    vbtrainmc_fun = @(theta_) negelcbo_vbmc(theta_,elcbo_beta,vp0,gp,NSentK,1,compute_var,options.AltMCEntropy,thetabnd);

    if NSentK == 0
        % Fast optimization via deterministic entropy approximation
        TolOpt = options.DetEntTolOpt;
        vbtrain_options.TolFun = TolOpt;
        vbtrain_fun = @(theta_) negelcbo_vbmc(theta_,elcbo_beta,vp0,gp,0,1,compute_var,0,thetabnd);
        try
            [thetaopt,~,~,output] = fminunc(vbtrain_fun,theta0(:)',vbtrain_options);
        catch
            % FMINUNC failed, try with CMA-ES
            if prnt > 0
                fprintf('Cannot optimize variational parameters with FMINUNC. Trying with CMA-ES (slower).\n');
            end
            if vp.optimize_mu; insigma_mu = repmat(vp.bounds.mu_ub(:) - vp.bounds.mu_lb(:),[vp.K,1]); else; insigma_mu = []; end
            insigma_sigma = ones(K,1);
            if vp.optimize_lambda; insigma_lambda = ones(vp.D,1); else; insigma_lambda = []; end
            if vp.optimize_weights; insigma_eta = ones(vp.K,1); else; insigma_eta = []; end
            insigma = [insigma_mu(:); insigma_sigma(:); insigma_lambda; insigma_eta];
            cmaes_opts = options.CMAESopts;
            cmaes_opts.EvalParallel = 'off';
            cmaes_opts.TolX = '1e-8*max(insigma)';
            cmaes_opts.TolFun = 1e-6;
            cmaes_opts.TolHistFun = 1e-7;           
            thetaopt = cmaes_modded('negelcbo_vbmc',theta0(:),insigma,cmaes_opts, ...
                elcbo_beta,vp0,gp,0,1,compute_var,0,thetabnd); 
            thetaopt = thetaopt(:)';
        end
        % output, % pause
    else
        % Optimization via unbiased stochastic entropy approximation
        thetaopt = theta0(:)';
        
        switch lower(options.StochasticOptimizer)
            case 'adam'
                
                master_stepsize.min = min(options.SGDStepSize,0.001);
                if optimState.Warmup || ~vp.optimize_weights
                    scaling_factor = min(0.1,options.SGDStepSize*10);
                else
                    scaling_factor = min(0.1,options.SGDStepSize);
                end                    
                
                if options.GPStochasticStepsize
                    % Set Adam master stepsizes from GP hyperparameters
                    ll_ker = zeros(vp.D,numel(gp.post)); % GP kernel length scale
                    ll_mnf = zeros(vp.D,numel(gp.post)); % GP mean fcn length scale
                    
                    % Compute mean length scales from samples
                    for iSample = 1:numel(gp.post)
                        ll_ker(:,iSample) = exp(gp.post(iSample).hyp(1:vp.D));
                        switch gp.meanfun
                            case 1; ll_mnf(:,iSample) = Inf(vp.D,1);
                            case 4; ll_mnf(:,iSample) = exp(gp.post(iSample).hyp(end-vp.D+1:end));
                            case 6; ll_mnf(:,iSample) = exp(gp.post(iSample).hyp(end-vp.D:end-1));
                            case 8
                                omega = exp(gp.post(iSample).hyp(end-3*vp.D:end-2*vp.D-1));
                                omega_se = exp(gp.post(iSample).hyp(end-vp.D:end-1));
                                ll_mnf(:,iSample) = min(omega,omega_se);
                        end
                    end                    
                    ll_ker = mean(ll_ker,2);   
                    ll_mnf = mean(ll_mnf,2);
                    % [ll_ker'; ll_mnf']
                    
                    % For each dim, take minimum length scale (bounded)
                    ll = max(min(min(ll_ker,ll_mnf),0.1),0.001);
                    
                    % Compute stepsize for variational optimization
                    ssize = [];
                    if vp.optimize_mu; ssize = repmat(ll,[K,1]); end
                    % ssize = [ssize; scaling_factor*ones(K,1)];
                    ssize = [ssize; min(min(ll),scaling_factor)*ones(K,1)];
                    if vp.optimize_lambda; ssize = [ssize; ll]; end
                    % if vp.optimize_weights; ssize = [ssize; min(min(ll),scaling_factor)*ones(K,1)]; end
                    if vp.optimize_weights; ssize = [ssize; scaling_factor*ones(K,1)]; end
                    master_stepsize.max = ssize;
                    master_stepsize.min = min(master_stepsize.max,master_stepsize.min);                    
                else
                    % Fixed master stepsize
                    master_stepsize.max = scaling_factor;         
                end
                
                master_stepsize.max = max(master_stepsize.min,master_stepsize.max);
                master_stepsize.decay = 200;
                [thetaopt,~,theta_lst,fval_lst] = ...
                    fminadam(vbtrainmc_fun,thetaopt,[],[],options.TolFunStochastic,[],master_stepsize);

                if options.ELCBOmidpoint
                    % Recompute ELCBO at best midpoint with full variance and more precision
                    [~,idx_mid] = min(fval_lst);
                    elbostats = eval_fullelcbo(iOpt_mid,theta_lst(idx_mid,:),vp0,gp,elbostats,elcbo_beta,options);
                    % [idx_mid,numel(fval_lst)]
                end
%             case 'fmincon'
%                 vbtrain_options.TolFun = options.TolFunStochastic;
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
G = elbostats.G(idx);
H = elbostats.H(idx);
varss = elbostats.varss(idx);
varG = elbostats.varG(idx);
varH = elbostats.varH(idx);
vp = vp0_fine(idx);
vp = rescale_params(vp,elbostats.theta(idx,:));
vp.temperature = optimState.temperature;

%% Potentially prune mixture components

pruned = 0;
if vp.optimize_weights
    
    alreadychecked = false(1,vp.K);
    
    while any(vp.w < options.TolWeight & ~alreadychecked)
        vp_pruned = vp;
        
        % Choose a random component below threshold
        idx = find(vp_pruned.w < options.TolWeight & ~alreadychecked);
        idx = idx(randi(numel(idx)));
        vp_pruned.w(idx) = [];
        if isfield(vp_pruned,'eta'); vp_pruned.eta(idx) = []; end
        vp_pruned.sigma(idx) = [];
        vp_pruned.mu(:,idx) = [];
        vp_pruned.K = vp_pruned.K - 1;
        [theta_pruned,vp_pruned] = get_vptheta(vp_pruned,vp_pruned.optimize_mu,vp_pruned.optimize_lambda,vp_pruned.optimize_weights);
        
        % Recompute ELCBO
        elbostats = eval_fullelcbo(1,theta_pruned,vp_pruned,gp,elbostats,elcbo_beta,options);        
        elbo_pruned = -elbostats.nelbo(1);
        elbo_pruned_sd = sqrt(elbostats.varF(1));
        
        % Difference in ELCBO (before and after pruning)
        delta_elcbo = abs((elbo_pruned - options.ELCBOImproWeight*elbo_pruned_sd) ...
            - (elbo - options.ELCBOImproWeight*elbo_sd));
        
        % Prune component if it has negligible influence on ELCBO
        if isa(options.PruningThresholdMultiplier,'function_handle')
            PruningThreshold = options.TolImprovement*options.PruningThresholdMultiplier(vp.K);
        else
            PruningThreshold = options.TolImprovement*options.PruningThresholdMultiplier;
        end
        
        if delta_elcbo < PruningThreshold
            vp = vp_pruned;
            elbo = elbo_pruned;
            elbo_sd = elbo_pruned_sd;
            varss = elbostats.varss(1);
            pruned = pruned + 1;
            alreadychecked(idx) = [];
        else
            alreadychecked(idx) = true;
        end
    end
end

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
    elbostats.varG = NaN(1,idx);
    elbostats.varH = NaN(1,idx);
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
    [nelbo,~,G,H,varF,~,varss,varG,varH] = ...
        negelcbo_vbmc(theta,0,vp,gp,NSentFineK,0,1,options.AltMCEntropy,[]);
    nelcbo = nelbo + beta*sqrt(varF);

    elbostats.nelbo(idx) = nelbo;
    elbostats.G(idx) = G;
    elbostats.H(idx) = H;
    elbostats.varF(idx) = varF;
    elbostats.varG(idx) = varG;
    elbostats.varH(idx) = varH;
    elbostats.varss(idx) = varss;
    elbostats.nelcbo(idx) = nelcbo;
    elbostats.theta(idx,1:numel(theta)) = theta;
end

end