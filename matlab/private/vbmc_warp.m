function [optimState,vp,hyp,hyp_warp,action] = vbmc_warp(optimState,vp,gp,hyp,hyp_warp,action,options,cmaes_opts)
%VBMC_WARP Compute input space warpings (linear and nonlinear).

% Nonlinear warping iteration? (only after burn-in)
isNonlinearWarping = (optimState.N - optimState.LastNonlinearWarping) >= options.WarpNonlinearEpoch ...
    && (options.MaxFunEvals - optimState.N) >= options.WarpNonlinearEpoch ...
    && optimState.N >= options.WarpNonlinearMinFun ...
    && options.WarpNonlinear ...
    && ~optimState.Warmup;
if isNonlinearWarping
    optimState.LastNonlinearWarping = optimState.N;
    optimState.WarpingNonlinearCount = optimState.WarpingNonlinearCount + 1;
    if isempty(action); action = 'warp'; else; action = [action ', warp']; end
end

% Rotoscaling iteration? (only after burn-in)
isRotoscaling = (optimState.N - optimState.LastWarping) >= options.WarpEpoch ...
    && (optimState.N - optimState.LastNonlinearWarping) >= options.WarpEpoch ...
    && (options.MaxFunEvals - optimState.N) >= options.WarpEpoch ...
    && optimState.N >= options.WarpMinFun ...
    && options.WarpRotoScaling ...
    && ~optimState.Warmup;
if isRotoscaling
    optimState.LastWarping = optimState.N;
    optimState.WarpingCount = optimState.WarpingCount + 1;
    if isempty(action); action = 'rotoscale'; else; action = [action ', rotoscale']; end
end

%% Update stretching of unbounded variables
if any(isinf(LB) & isinf(UB)) && ...
        options.WarpNonlinear && (isNonlinearWarping || optimState.iter == 1)
    [vp,optimState,hyp] = ...
        warp_unbounded(vp,optimState,hyp,gp,cmaes_opts,options);
    optimState = ResetRunAvg(optimState);
end

%%  Rotate and rescale variables
if options.WarpRotoScaling && isRotoscaling
    [vp,optimState,hyp] = ...
        warp_rotoscaling(vp,optimState,hyp,gp,cmaes_opts,options);
    optimState = ResetRunAvg(optimState);
end

%% Learn nonlinear warping via GP
if options.WarpNonlinear && isNonlinearWarping
    [vp,optimState,hyp,hyp_warp] = ...
        warp_nonlinear(vp,optimState,hyp,hyp_warp,cmaes_opts,options);
    optimState.redoRotoscaling = true;
    optimState = ResetRunAvg(optimState);
end

%     if optimState.DoRotoscaling && isWarping
%         [~,X_hpd,y_hpd] = ...
%             vbmc_gphyp(optimState,optimState.gpMeanfun,0,options);            
%         vp = vpoptimize(Nfastopts,1,0,vp,gp,vp.K,X_hpd,y_hpd,optimState,stats,options,cmaes_opts,prnt);
%         [vp,optimState,hyp] = ...
%             warp_rotoscaling(vp,optimState,hyp,gp,cmaes_opts,options);
%         optimState.DoRotoscaling = false;
%     end


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function optimState = ResetRunAvg(optimState)
%RESETRUNAVG Reset running averages of moments after warping.

optimState.RunMean = [];
optimState.RunCov = [];        
optimState.LastRunAvg = NaN;

end