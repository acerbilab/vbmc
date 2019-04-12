function [optimState,action] = vbmc_warmup(optimState,stats,action,elbo,elbo_sd,options)
%VBMC_WARMUP Check when warmup stage ends

iter = optimState.iter;

elbo_old = stats.elbo(iter-1);
elboSD_old = stats.elboSD(iter-1);

increaseUCB = (elbo - options.ELCBOImproWeight*elbo_sd) - ...
    (elbo_old - options.ELCBOImproWeight*elboSD_old);

if increaseUCB < options.StopWarmupThresh
    optimState.WarmupStableIter = optimState.WarmupStableIter + 1;
else
    optimState.WarmupStableIter = 0;
end        

if optimState.WarmupStableIter >= options.TolStableWarmup            
    optimState.Warmup = false;
    if isempty(action); action = 'end warm-up'; else; action = [action ', end warm-up']; end

    % Remove warm-up points from training set unless close to max
    ymax = max(optimState.y_orig(1:optimState.Xmax));
    D = numel(optimState.LB);
    NkeepMin = D+1; 
    idx_keep = (ymax - optimState.y_orig) < options.WarmupKeepThreshold;
    sum(idx_keep)
    if sum(idx_keep) < NkeepMin
        y_temp = optimState.y_orig;
        y_temp(~isfinite(y_temp)) = -Inf;
        [~,ord] = sort(y_temp,'descend');
        idx_keep(ord(1:min(NkeepMin,optimState.Xmax))) = true;
    end
    optimState.X_flag = idx_keep & optimState.X_flag;

    % Start warping
    optimState.LastWarping = optimState.N;
    optimState.LastNonlinearWarping = optimState.N;

    % Skip adaptive sampling for next iteration
    optimState.SkipActiveSampling = options.SkipActiveSamplingAfterWarmup;

    % Fully recompute variational posterior
    optimState.RecomputeVarPost = true;

    % Reset GP hyperparameter covariance
    optimState.RunHypCov = [];
end