function [vp,elbo,elbo_sd,changedflag] = finalboost_vbmc(vp,idx_best,optimState,stats,options)
%FINALBOOST_VBMC Final boost of variational components.

changedflag = false;

if vp.K < options.MinFinalComponents
    % Last variational optimization with large number of components
    Knew = options.MinFinalComponents;
    if isa(options.NSelbo,'function_handle')
        Nfastopts = ceil(options.NSelbo(Knew));
    else
        Nfastopts = ceil(options.NSelbo);
    end
    Nfastopts = ceil(Nfastopts * options.NSelboIncr);    
    Nslowopts = 1;
    gp_idx = gplite_post(stats.gp(idx_best));
    options.TolWeight = 0; % No pruning of components
    
    % End warmup
    optimState.Warmup = false;
    vp.optimize_mu = logical(options.VariableMeans);
    vp.optimize_weights = logical(options.VariableWeights);
    
    if isfield(options,'NSentBoost') && ~isempty(options.NSentBoost)
        options.NSent = options.NSentBoost;
    end
    if isfield(options,'NSentFastBoost') && ~isempty(options.NSentFastBoost)
        options.NSentFast = options.NSentFastBoost;
    end
    if isfield(options,'NSentFineBoost') && ~isempty(options.NSentFineBoost)
        options.NSentFine = options.NSentFineBoost;
    end
    
    stable_flag = vp.stats.stable;
    vp = vpoptimize_vbmc(Nfastopts,Nslowopts,vp,gp_idx,Knew,optimState,options);
    vp.stable = stable_flag;
    changedflag = true; 
end

elbo = vp.stats.elbo;
elbo_sd = vp.stats.elbo_sd;

end