function [optimState,vp,hyp] = vbmc_rewarp(optimState,vp,gp,hyp,options,cmaes_opts)
%VBMC_REWARP Recompute input space warpings (linear and nonlinear).

%  Redo rotoscaling at the end
if options.WarpRotoScaling && optimState.redoRotoscaling
    [vp,optimState,hyp] = ...
        warp_rotoscaling(vp,optimState,hyp,gp,cmaes_opts,options);

    % Get priors and starting hyperparameters
    [hypprior,X_hpd,y_hpd,~,hyp0,optimState.gpMeanfun] = ...
        vbmc_gphyp(optimState,optimState.gpMeanfun,0,options);
    gptrain_options.Nopts = 1;

    [gp,hyp] = gplite_train(hyp,Ns_gp, ...
        optimState.X(optimState.X_flag,:),optimState.y(optimState.X_flag), ...
        optimState.gpMeanfun,hypprior,[],gptrain_options);

     vp = vpoptimize(Nfastopts,1,vp,gp,vp.K,X_hpd,y_hpd,optimState,stats,options,cmaes_opts,prnt);
end

end