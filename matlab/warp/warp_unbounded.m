function [vp,optimState,hyp] = warp_unbounded(vp,optimState,hyp,gp,cmaes_opts,options)
%WARP_UNBOUNDED Compute stretch warping for unbounded variables.

vp_old = vp;

X_orig = optimState.X_orig(1:optimState.Xmax,:);
vp.trinfo.mu = median(X_orig);
vp.trinfo.delta = quantile(X_orig,0.9) - quantile(X_orig,0.1);

% Readjust variational posterior and GP hyperparameters after stretching
if isempty(gp)
    [vp,optimState] = recompute_vp_and_hyp(vp,vp_old,optimState,[],options,0);            
else
    [vp,optimState,hyp_warped] = recompute_vp_and_hyp(vp,vp_old,optimState,cmaes_opts,options,0,hyp,gp);
    hyp = [hyp,hyp_warped];
end        
optimState.trinfo = vp.trinfo;

end