function [optimState,action] = vbmc_warmup(optimState,stats,action,elbo,elbo_sd,options)
%VBMC_WARMUP Check when warmup stage ends

iter = optimState.iter;

elbo_old = stats.elbo(iter-1);
elboSD_old = stats.elbo_sd(iter-1);

% First requirement for stopping, no constant improvement of metric
StableCountFlag = false;
StopWarmupThresh = options.StopWarmupThresh*options.FunEvalsPerIter;
TolStableWarmupIters = ceil(options.TolStableWarmup/options.FunEvalsPerIter);

if 0
    if options.BOWarmup
        % Bayesian optimization like warmup - criterion is improvement over max
        y = optimState.y(optimState.X_flag);
        idx_last = false(size(y));
        idx_last(max(2,numel(y)-options.FunEvalsPerIter+1):numel(y)) = true;
        improCriterion = max(y(idx_last)) - max(y(~idx_last));
    else
        % Variational posterior like warmup - criterion is ELCBO
        improCriterion = (elbo - options.ELCBOImproWeight*elbo_sd) - ...
            (elbo_old - options.ELCBOImproWeight*elboSD_old);
    end

    if improCriterion < StopWarmupThresh
        % Increase warmup stability counter
        optimState.WarmupStableCount = optimState.WarmupStableCount + options.FunEvalsPerIter;
    else
        optimState.WarmupStableCount = 0;
    end
    StableCountFlag = optimState.WarmupStableCount >= options.TolStableWarmup;
elseif iter > TolStableWarmupIters + 1
    % Vector of ELCBO (ignore first two iterations, ELCBO is unreliable)
    elcbo_vec = [stats.elbo,elbo] - options.ELCBOImproWeight*[stats.elbo_sd,elbo_sd];    
    max_now = max(elcbo_vec(max(4,end-TolStableWarmupIters+1):end));
    max_before = max(elcbo_vec(3:max(3,end-TolStableWarmupIters)));
    
    StableCountFlag = (max_now - max_before) < StopWarmupThresh;
end

% Vector of maximum lower confidence bounds (LCB) of fcn values
lcbmax_vec = [stats.lcbmax(1:iter-1),optimState.lcbmax];

% Second requirement, also no substantial improvement of max fcn value 
% in recent iters (unless already performing BO-like warmup)
if ~options.BOWarmup && options.WarmupCheckMax
    if 0
        y = optimState.y(optimState.X_flag);
        idx_last = false(size(y));
        idx_last(max(2,numel(y)-options.TolStableWarmup+1):numel(y)) = true;
        improFcn = max(0,max(y(idx_last)) - max(y(~idx_last)));
    else
        idx_last = false(size(lcbmax_vec));
        RecentPast = iter-ceil(options.TolStableWarmup/options.FunEvalsPerIter)+1;
        idx_last(max(2,RecentPast):end) = true;
        improFcn = max(0,max(lcbmax_vec(idx_last)) - max(lcbmax_vec(~idx_last)));
    end
else
    improFcn = 0;
end

% Alternative criterion for stopping -- no improvement over max fcn value
if 0
    [y_max,pos] = max(optimState.y);
    currentpos = optimState.Xn;
else    
    max_thresh = max(lcbmax_vec) - options.TolImprovement;
    idx_1st = find(lcbmax_vec > max_thresh,1);
    yy = [stats.funccount(1:iter-1),optimState.funccount];
    pos = yy(idx_1st);
    currentpos = optimState.funccount;
end

stopWarmup = (StableCountFlag && improFcn < StopWarmupThresh) || ...
    (currentpos - pos) > options.WarmupNoImproThreshold;
    
if stopWarmup
    optimState.Warmup = false;
    if isempty(action); action = 'end warm-up'; else; action = [action ', end warm-up']; end

    % Remove warm-up points from training set unless close to max
    ymax = max(optimState.y_orig(1:optimState.Xn));
    D = numel(optimState.LB);
    NkeepMin = D+1; 
    idx_keep = (ymax - optimState.y_orig) < options.WarmupKeepThreshold;
    if sum(idx_keep) < NkeepMin
        y_temp = optimState.y_orig;
        y_temp(~isfinite(y_temp)) = -Inf;
        [~,ord] = sort(y_temp,'descend');
        idx_keep(ord(1:min(NkeepMin,optimState.Xn))) = true;
    end
    optimState.X_flag = idx_keep & optimState.X_flag;

    % Start warping
    optimState.LastWarping = optimState.N;
    optimState.LastNonlinearWarping = optimState.N;

    % Skip adaptive sampling for next iteration
    optimState.SkipActiveSampling = options.SkipActiveSamplingAfterWarmup;

    % Fully recompute variational posterior
    optimState.RecomputeVarPost = true;
end