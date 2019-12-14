function [vp,varss,pruned] = vpoptimize_vbmc(Nfastopts,Nslowopts,vp,gp,K,optimState,options,prnt)
%VPOPTIMIZE Optimize variational posterior.

if nargin < 5 || isempty(K); K = vp.K; end
if nargin < 6; optimState = []; end
if nargin < 7; options = []; end
if nargin < 8 || isempty(prnt); prnt = 0; end

% Assign default values to OPTIONS
if isempty(options)
    options = vbmc('all');
    options = setupoptions_vbmc(vp.D,options,options);
end

% Assign default values to OPTIMSTATE
if ~isfield(optimState,'delta'); optimState.delta = 0; end
if ~isfield(optimState,'EntropySwitch'); optimState.EntropySwitch = false; end
if ~isfield(optimState,'Warmup'); optimState.Warmup = ~vp.optimize_weights; end
if ~isfield(optimState,'temperature'); optimState.temperature = 1; end
if ~isfield(optimState,'entropy_alpha'); optimState.entropy_alpha = 0; end

% Quick sieve optimization to determine starting point(s)
[vp0_vec,vp0_type,elcbo_beta,compute_var,NSentK] = ...
    vpsieve_vbmc(Nfastopts,Nslowopts,vp,gp,optimState,options,K);
    
% Compute soft bounds for variational parameters optimization
[vp,thetabnd] = vpbounds(vp,gp,options,K);

%% Perform optimization starting from one or few selected points

% Set up empty stats structs for optimization
Ntheta = numel(get_vptheta(vp0_vec(1)));
elbostats = eval_fullelcbo(Nslowopts*2,Ntheta);

% For the moment no gradient available for variance
gradient_available = (compute_var == 0);

if gradient_available
    % Set basic options for deterministic optimizer (FMINUNC)
    vbtrain_options = optimoptions('fminunc','GradObj','on','Display','off');
    compute_grad = 1;
else
    options.StochasticOptimizer = 'cmaes';
    vbtrain_options = optimoptions('fminunc','GradObj','off','Display','off');
    compute_grad = 0;
end    

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
    if vp.optimize_sigma; theta0 = [theta0; log(vp0.sigma(:))]; end
    if vp.optimize_lambda; theta0 = [theta0; log(vp0.lambda(:))]; end
    if vp.optimize_weights; theta0 = [theta0; log(vp0.w(:))]; end
    % theta0 = min(vp.UB_theta',max(vp.LB_theta', theta0));

    vbtrainmc_fun = @(theta_) negelcbo_vbmc(theta_,elcbo_beta,vp0,gp,NSentK,1,compute_var,options.AltMCEntropy,thetabnd,optimState.entropy_alpha);

    if NSentK == 0
        % Fast optimization via deterministic entropy approximation
        TolOpt = options.DetEntTolOpt;
        vbtrain_options.TolFun = TolOpt;
        vbtrain_options.MaxFunEvals = 50*(vp.D+2);
        vbtrain_fun = @(theta_) negelcbo_vbmc(theta_,elcbo_beta,vp0,gp,0,compute_grad,compute_var,0,thetabnd,optimState.entropy_alpha);
        try
            [thetaopt,~,~,output] = fminunc(vbtrain_fun,theta0(:)',vbtrain_options);
            % output.funcCount
        catch
            % FMINUNC failed, try with CMA-ES
            if prnt >= 0
                fprintf('Cannot optimize variational parameters with FMINUNC. Trying with CMA-ES (slower).\n');
            end
            if vp.optimize_mu; insigma_mu = repmat(vp.bounds.mu_ub(:) - vp.bounds.mu_lb(:),[K,1]); else; insigma_mu = []; end
            if vp.optimize_sigma; insigma_sigma = ones(K,1); else; insigma_sigma = []; end
            if vp.optimize_lambda; insigma_lambda = ones(vp.D,1); else; insigma_lambda = []; end
            if vp.optimize_weights; insigma_eta = ones(K,1); else; insigma_eta = []; end
            insigma = [insigma_mu(:); insigma_sigma(:); insigma_lambda; insigma_eta];
            cmaes_opts = options.CMAESopts;
            cmaes_opts.EvalParallel = 'off';
            cmaes_opts.TolX = '1e-8*max(insigma)';
            cmaes_opts.TolFun = 1e-6;
            cmaes_opts.TolHistFun = 1e-7;
            cmaes_opts.MaxFunEvals = 200*(vp.D+2);            
            thetaopt = cmaes_modded('negelcbo_vbmc',theta0(:),insigma,cmaes_opts, ...
                elcbo_beta,vp0,gp,0,0,compute_var,0,thetabnd,optimState.entropy_alpha); 
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
                    % if vp.optimize_sigma; ssize = [ssize; scaling_factor*ones(K,1)]; end
                    if vp.optimize_sigma; ssize = [ssize; min(min(ll),scaling_factor)*ones(K,1)]; end
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
                MaxIter = min(options.MaxIterStochastic,1e4);
                [thetaopt,~,theta_lst,fval_lst,niters] = ...
                    fminadam(vbtrainmc_fun,thetaopt,[],[],options.TolFunStochastic,MaxIter,master_stepsize);
                % niters
                
                if options.ELCBOmidpoint
                    % Recompute ELCBO at best midpoint with full variance and more precision
                    [~,idx_mid] = min(fval_lst);
                    elbostats = eval_fullelcbo(iOpt_mid,theta_lst(idx_mid,:),vp0,gp,elbostats,elcbo_beta,options,optimState.entropy_alpha);
                    % [idx_mid,numel(fval_lst)]
                end
%             case 'fmincon'
%                 vbtrain_options.TolFun = options.TolFunStochastic;
%                 [thetaopt,~,~,output] = ...
%                     fmincon(vbtrainmc_fun,thetaopt,[],[],[],[],vp.LB_theta,vp.UB_theta,[],vbtrain_options);

            case 'cmaes'
                
                if vp.optimize_mu; insigma_mu = repmat(vp.bounds.mu_ub(:) - vp.bounds.mu_lb(:),[K,1]); else; insigma_mu = []; end
                if vp.optimize_sigma; insigma_sigma = ones(K,1); else; insigma_sigma = []; end
                if vp.optimize_lambda; insigma_lambda = ones(vp.D,1); else; insigma_lambda = []; end
                if vp.optimize_weights; insigma_eta = ones(K,1); else; insigma_eta = []; end
                insigma = [insigma_mu(:); insigma_sigma; insigma_lambda; insigma_eta];
                cmaes_opts = options.CMAESopts;
                cmaes_opts.EvalParallel = 'off';
                cmaes_opts.TolX = '1e-6*max(insigma)';
                cmaes_opts.TolFun = 1e-4;
                cmaes_opts.TolHistFun = 1e-5;
                cmaes_opts.Noise.on = 1;    % Noisy evaluations
                try
                    thetaopt = cmaes_modded('negelcbo_vbmc',theta0(:),insigma,cmaes_opts, ...
                        elcbo_beta,vp0,gp,NSentK,0,compute_var,options.AltMCEntropy,thetabnd,optimState.entropy_alpha); 
                catch
                    pause
                end
                thetaopt = thetaopt(:)';
                
            otherwise
                error('vbmc:VPoptimize','Unknown stochastic optimizer.');
        end
    end
        
	% Recompute ELCBO at endpoint with full variance and more precision
    elbostats = eval_fullelcbo(iOpt_end,thetaopt,vp0,gp,elbostats,elcbo_beta,options,optimState.entropy_alpha);
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
        [theta_pruned,vp_pruned] = get_vptheta(vp_pruned,vp_pruned.optimize_mu,vp_pruned.optimize_sigma,vp_pruned.optimize_lambda,vp_pruned.optimize_weights);
        
        % Recompute ELCBO
        elbostats = eval_fullelcbo(1,theta_pruned,vp_pruned,gp,elbostats,elcbo_beta,options,optimState.entropy_alpha);        
        elbo_pruned = -elbostats.nelbo(1);
        elbo_pruned_sd = sqrt(elbostats.varF(1));
        
        % Difference in ELCBO (before and after pruning)
        delta_elcbo = abs((elbo_pruned - options.ELCBOImproWeight*elbo_pruned_sd) ...
            - (elbo - options.ELCBOImproWeight*elbo_sd));
        
        % Prune component if it has negligible influence on ELCBO
        PruningThreshold = options.TolImprovement * ...
            evaloption_vbmc(options.PruningThresholdMultiplier,K);
                
        if delta_elcbo < PruningThreshold
            vp = vp_pruned;
            elbo = elbo_pruned;
            elbo_sd = elbo_pruned_sd;
            G = elbostats.G(1);
            H = elbostats.H(1);
            varss = elbostats.varss(1);
            varG = elbostats.varG(1);
            varH = elbostats.varH(1);
            pruned = pruned + 1;
            alreadychecked(idx) = [];
        else
            alreadychecked(idx) = true;
        end
    end
end

vp.stats.elbo = elbo;               % ELBO
vp.stats.elbo_sd = elbo_sd;         % Error on the ELBO
vp.stats.elogjoint = G;             % Expected log joint
vp.stats.elogjoint_sd = sqrt(varG); % Error on expected log joint
vp.stats.entropy = H;               % Entropy
vp.stats.entropy_sd = sqrt(varH);   % Error on the entropy
vp.stats.stable = false;            % Unstable until proven otherwise


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
function elbostats = eval_fullelcbo(idx,theta,vp,gp,elbostats,beta,options,entropy_alpha)
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
    NSentFineK = ceil(evaloption_vbmc(options.NSentFine,K)/K);
        
    if isfield(options,'SkipELBOVariance') && options.SkipELBOVariance
        computevar_flag = false;
    else
        computevar_flag = true;
    end
    
    theta = theta(:)';
    [nelbo,~,G,H,varF,~,varss,varG,varH] = ...
        negelcbo_vbmc(theta,0,vp,gp,NSentFineK,0,computevar_flag,options.AltMCEntropy,[],entropy_alpha);
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