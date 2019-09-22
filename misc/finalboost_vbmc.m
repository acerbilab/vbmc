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
    options_temp = options;
    gp_idx = gplite_post(stats.gp(idx_best));
    options_temp.TolWeight = 0; % No pruning of components
    vp = vpoptimize_vbmc(Nfastopts,Nslowopts,vp,gp_idx,Knew,optimState,options_temp);
    changedflag = true; 
end

elbo = vp.stats.elbo;
elbo_sd = vp.stats.elbo_sd;

end