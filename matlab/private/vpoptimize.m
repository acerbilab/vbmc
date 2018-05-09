function [vp,elbo,elbo_sd,varss] = vpoptimize(Nslowopts,useEntropyApprox,vp,gp,K,Xstar,ystar,optimState,stats,options)
%VPOPTIMIZE Optimize variational posterior.

% Get bounds for variational parameters optimization    
[vp.LB_theta,vp.UB_theta] = vbmc_vpbnd(vp,Xstar,K,options);

Nfastopts = options.NSelbo * K; % Number of initial starting points    
nelcbo_fill = zeros(Nfastopts,1);

% Check variational posteriors from previous iterations
MaxBack = min(optimState.iter-1,options.VarParamsBack);

elbostats = eval_fullelcbo(Nslowopts*3+MaxBack,numel(vp.LB_theta));

% Generate random initial starting point for variational parameters
[vp0_vec,vp0_type] = vbinitrnd(Nfastopts,vp,K,Xstar,ystar);

compute_var = 2;    % Use diagonal-only approximation
% Confidence weight
elcbo_beta = options.ELCBOWeight; % * sqrt(vp.D) / sqrt(optimState.N);

% Quickly estimate ELCBO at each candidate variational posterior
for iOpt = 1:Nfastopts
    [theta0,vp0_vec(iOpt)] = ...
        get_theta(vp0_vec(iOpt),vp.LB_theta,vp.UB_theta,vp.optimize_lambda);        
    [nelbo_tmp,~,~,~,varF_tmp] = vbmc_negelcbo(theta0,0,vp0_vec(iOpt),gp,options.NSent,0,compute_var);
    nelcbo_fill(iOpt) = nelbo_tmp + elcbo_beta*sqrt(varF_tmp);
end

% Sort by negative ELCBO
[~,vp0_ord] = sort(nelcbo_fill,'ascend');
vp0_vec = vp0_vec(vp0_ord);
vp0_type = vp0_type(vp0_ord);

for iOpt = 1:Nslowopts
    iOpt_start = iOpt*3-2;
    iOpt_mid = iOpt*3-1;
    iOpt_end = iOpt*3;

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
    theta0 = min(vp.UB_theta',max(vp.LB_theta', theta0));

    vbtrainmc_fun = @(theta_) vbmc_negelcbo(theta_,elcbo_beta,vp0,gp,options.NSent,1,compute_var);

    % First, fast optimization via entropy approximation
    if useEntropyApprox
        vbtrain_options = optimoptions('fmincon','GradObj','on','Display','off','OptimalityTolerance',1e-3);
        vbtrain_fun = @(theta_) vbmc_negelcbo(theta_,elcbo_beta,vp0,gp,0,1,compute_var);
        [thetaopt,~,~,output] = ...
            fmincon(vbtrain_fun,theta0(:)',[],[],[],[],vp.LB_theta,vp.UB_theta,[],vbtrain_options);
        % output, % pause
    else
        thetaopt = theta0(:)';
    end

    % Second, refine with unbiased stochastic entropy approximation
    [thetaopt,~,theta_lst,fval_lst] = ...
        fminadam(vbtrainmc_fun,thetaopt,vp.LB_theta,vp.UB_theta,options.TolFunAdam);

    [~,idx_mid] = min(fval_lst);
    % [idx_mid,numel(fval_lst)]

    % Recompute ELCBO at start, mid and endpoint with full variance and more precision
    elbostats = eval_fullelcbo(iOpt_start,theta0,vp0,gp,elbostats,elcbo_beta,options);        
    elbostats = eval_fullelcbo(iOpt_mid,theta_lst(idx_mid,:),vp0,gp,elbostats,elcbo_beta,options);
    elbostats = eval_fullelcbo(iOpt_end,thetaopt,vp0,gp,elbostats,elcbo_beta,options);

    vp0_fine(iOpt_start) = vp0;
    vp0_fine(iOpt_mid) = vp0;
    vp0_fine(iOpt_end) = vp0;   % Parameters get assigned later

    % [nelbo,nelcbo,sqrt(varF),G,H]
end

% Finally, add variational parameters from previous iterations
for iBack = 1:MaxBack
    idx_prev = Nslowopts*3+iBack;
    vp_back = stats.vp(iter-iBack);
    vp0_fine(idx_prev) = vp_back;
    % Note that TRINFO might have changed, recompute it if needed       
    vp0_fine(idx_prev).trinfo = vp.trinfo;
    vp0_fine(idx_prev) = recompute_vp_and_hyp( ...
        vp0_fine(idx_prev),vp_back,optimState,[],options,0,[],[],...
        optimState.X(optimState.X_flag,:),optimState.y(optimState.X_flag),[Inf,Inf]);
    [theta_prev,vp0_fine(idx_prev)] = get_theta(vp0_fine(idx_prev),[],[],vp.optimize_lambda);
    elbostats = eval_fullelcbo(idx_prev,theta_prev,vp0_fine(idx_prev),gp,elbostats,elcbo_beta,options);
end

% Take variational parameters with best ELCBO
[~,idx] = min(elbostats.nelcbo);
elbo = -elbostats.nelbo(idx);
elbo_sd = sqrt(elbostats.varF(idx));
varss = elbostats.varss(idx);
vp = vp0_fine(idx);
vp = rescale_params(vp,elbostats.theta(idx,:));

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
    theta = theta(:)';    
    [nelbo,~,G,H,varF,~,varss] = ...
        vbmc_negelcbo(theta,0,vp,gp,options.NSentFine,0,1);
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