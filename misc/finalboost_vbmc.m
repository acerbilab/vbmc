function [vp,elbo,elbo_sd,changedflag] = finalboost_vbmc(vp,idx_best,optimState,stats,options)
%FINALBOOST_VBMC Final boost of variational components.

changedflag = false;

Knew = max(options.MinFinalComponents,vp.K);

% Current entropy samples during variational optimization
NSent = evaloption_vbmc(options.NSent,Knew);
NSentFast = evaloption_vbmc(options.NSentFast,Knew);
NSentFine = evaloption_vbmc(options.NSentFine,Knew);

% Entropy samples for final boost
NSentBoost = NSent;
NSentFastBoost = NSentFast;
NSentFineBoost = NSentFine;
if isfield(options,'NSentBoost') && ~isempty(options.NSentBoost)
    NSentBoost = evaloption_vbmc(options.NSentBoost,Knew);
end
if isfield(options,'NSentFastBoost') && ~isempty(options.NSentFastBoost)
    NSentFastBoost = evaloption_vbmc(options.NSentFastBoost,Knew);
end
if isfield(options,'NSentFineBoost') && ~isempty(options.NSentFineBoost)
    NSentFineBoost = evaloption_vbmc(options.NSentFineBoost,Knew);
end

% Perform final boost?
do_boost = vp.K < options.MinFinalComponents || ...
    (NSent ~= NSentBoost) || (NSentFine ~= NSentFineBoost);

if do_boost
    % Last variational optimization with large number of components
    Nfastopts = ceil(evaloption_vbmc(options.NSelbo,Knew));
    Nfastopts = ceil(Nfastopts * options.NSelboIncr);    
    Nslowopts = 1;
    gp_idx = gplite_post(stats.gp(idx_best));
    options.TolWeight = 0; % No pruning of components
    
    % End warmup
    optimState.Warmup = false;
    vp.optimize_mu = logical(options.VariableMeans);
    vp.optimize_weights = logical(options.VariableWeights);
    
    options.NSent = NSentBoost;
    options.NSentFast = NSentFastBoost;
    options.NSentFine = NSentFineBoost;
    options.MaxIterStochastic = Inf;
    optimState.entropy_alpha = 0;
        
    stable_flag = vp.stats.stable;
    vp = vpoptimize_vbmc(Nfastopts,Nslowopts,vp,gp_idx,Knew,optimState,options);
    vp.stats.stable = stable_flag;
    changedflag = true;
end

elbo = vp.stats.elbo;
elbo_sd = vp.stats.elbo_sd;

end