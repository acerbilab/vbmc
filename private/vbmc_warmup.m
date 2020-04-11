function [optimState,action,trim_flag] = vbmc_warmup(optimState,stats,action,options)
%VBMC_WARMUP Check when warmup stage ends

iter = optimState.iter;
trim_flag = false;  % Report if training data are trimmed

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
    %elcbo_vec = [stats.elbo,elbo] - options.ELCBOImproWeight*[stats.elbo_sd,elbo_sd];    
    elcbo_vec = stats.elbo - options.ELCBOImproWeight*stats.elbo_sd;    
    max_now = max(elcbo_vec(max(4,end-TolStableWarmupIters+1):end));
    max_before = max(elcbo_vec(3:max(3,end-TolStableWarmupIters)));
    
    StableCountFlag = (max_now - max_before) < StopWarmupThresh;
end

% Vector of maximum lower confidence bounds (LCB) of fcn values
if isfield(optimState,'lcbmax_vec') && ~isempty(optimState.lcbmax_vec)
    lcbmax_vec = optimState.lcbmax_vec(1:iter);
else
    lcbmax_vec = stats.lcbmax(1:iter);
end

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
    yy = stats.funccount(1:iter);
    pos = yy(idx_1st);
    currentpos = optimState.funccount;
end

if ~isempty(optimState.DataTrimList)
    lastDataTrim = optimState.DataTrimList(end);
else
    lastDataTrim = -Inf;
end
stopWarmup = (StableCountFlag && improFcn < StopWarmupThresh) || ...
    (currentpos - pos) > options.WarmupNoImproThreshold;
stopWarmup = stopWarmup & (optimState.N - lastDataTrim) >= 10;

if stopWarmup

    if stats.rindex(iter) < options.StopWarmupReliability || numel(optimState.DataTrimList) >= 1
        optimState.Warmup = false;
        if isempty(action); action = 'end warm-up'; else; action = [action ', end warm-up']; end
        threshold = options.WarmupKeepThreshold * (numel(optimState.DataTrimList)+1);
        optimState.LastWarmup = optimState.iter;
        
        % Start warping
        optimState.LastWarping = optimState.iter;
        optimState.LastNonlinearWarping = optimState.iter;
    else
        % This may be a false alarm; prune and continue
        if isempty(options.WarmupKeepThresholdFalseAlarm)
            WarmupKeepThresholdFalseAlarm = options.WarmupKeepThreshold;
        else
            WarmupKeepThresholdFalseAlarm = options.WarmupKeepThresholdFalseAlarm;
        end            
        threshold = WarmupKeepThresholdFalseAlarm * (numel(optimState.DataTrimList)+1);
        optimState.DataTrimList = [optimState.DataTrimList, optimState.N]; 
        if isempty(action); action = 'trim data'; else; action = [action ', trim data']; end        
    end
    
    % Remove warm-up points from training set unless close to max
    ymax = max(optimState.y_orig(1:optimState.Xn));
    D = numel(optimState.LB);
    NkeepMin = D+1;
    idx_keep = (ymax - optimState.y_orig) < threshold;
    if sum(idx_keep) < NkeepMin
        y_temp = optimState.y_orig;
        y_temp(~isfinite(y_temp)) = -Inf;
        [~,ord] = sort(y_temp,'descend');
        idx_keep(ord(1:min(NkeepMin,optimState.Xn))) = true;
    end
    optimState.X_flag = idx_keep & optimState.X_flag;
    trim_flag = true;

    % Skip adaptive sampling for next iteration
    optimState.SkipActiveSampling = options.SkipActiveSamplingAfterWarmup;

    % Fully recompute variational posterior
    optimState.RecomputeVarPost = true;
    
end