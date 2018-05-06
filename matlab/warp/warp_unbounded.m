function [vp,optimState,hyp] = warp_unbounded(vp,optimState,hyp,gp,cmaes_opts,options)
%WARP_UNBOUNDED Compute stretch warping for unbounded variables.

vp_old = vp;

mu_old = vp.trinfo.mu;
delta_old = vp.trinfo.delta;

X_orig = optimState.X_orig(1:optimState.Xmax,:);
mu_new = median(X_orig);

width_bnd = optimState.PUB - optimState.PLB; % Width of plausible bounds
width_bnd(~isfinite(width_bnd)) = 0;
width_orig = quantile(X_orig,0.9) - quantile(X_orig,0.1);
delta_new = max(width_bnd,width_orig);
 
if any(mu_old ~= mu_new) || any(delta_old ~= delta_new)

    vp.trinfo.mu = mu_new;
    vp.trinfo.delta = delta_new;
    
    % Readjust variational posterior and GP hyperparameters after stretching
    if isempty(gp)
        [vp,optimState] = recompute_vp_and_hyp(vp,vp_old,optimState,[],options,0);            
    else
        [vp,optimState,hyp_warped] = recompute_vp_and_hyp(vp,vp_old,optimState,cmaes_opts,options,0,hyp,gp);
        hyp = [hyp,hyp_warped];
    end        
    optimState.trinfo = vp.trinfo;

    % Major change, fully recompute variational posterior
    optimState.RecomputeVarPost = true;
end

end