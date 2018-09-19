function [optimState,action] = vbmc_warmup(optimState,stats,action,elbo,elbo_sd,options)
%VBMC_WARMUP Check when warmup stage ends

iter = optimState.iter;

elbo_old = stats.elbo(iter-1);
elboSD_old = stats.elboSD(iter-1);

% CI_95 = 1.6449; % Normal inverse cdf for 0.95
% increaseUCB = elbo - elbo_old + CI_95*sqrt(elbo_sd^2 + elboSD_old^2);
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
    NkeepMin = 4*D; 
    idx_keep = (ymax - optimState.y_orig) < options.WarmupKeepThreshold;
    if sum(idx_keep) < NkeepMin
        [~,ord] = sort(optimState.y_orig,'descend');
        idx_keep(ord(1:min(NkeepMin,optimState.Xmax))) = true;
    end
    optimState.X_flag = idx_keep & optimState.X_flag;

    % Start warping
    optimState.LastWarping = optimState.N;
    optimState.LastNonlinearWarping = optimState.N;

    % Skip adaptive sampling for next iteration
    optimState.SkipAdaptiveSampling = true;

    % Fully recompute variational posterior
    optimState.RecomputeVarPost = true;

    % Reset GP hyperparameter covariance
    optimState.RunHypCov = [];
end