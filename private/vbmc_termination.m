function [optimState,stats,isFinished_flag,exitflag,action,msg] = vbmc_termination(optimState,action,stats,options)
%VBMC_TERMINATION Compute stability index and check termination conditions.

iter = optimState.iter;
exitflag = 0;
isFinished_flag = false;
msg = [];

% Maximum number of new function evaluations
if optimState.funccount >= options.MaxFunEvals
    isFinished_flag = true;
    msg = 'Inference terminated: reached maximum number of function evaluations OPTIONS.MaxFunEvals.';
end

% Maximum number of iterations
if iter >= options.MaxIter
    isFinished_flag = true;
    msg = 'Inference terminated: reached maximum number of iterations OPTIONS.MaxIter.';
end

% Quicker stability check for entropy switching 
if optimState.EntropySwitch
    TolStableIters = options.TolStableEntropyIters;
else
    TolStableIters = ceil(options.TolStableCount/options.FunEvalsPerIter);
end

% Reached stable variational posterior with stable ELBO and low uncertainty
[idx_stable,dN,dN_last,w] = getStableIter(stats,optimState,options);
if ~isempty(idx_stable)
    sKL_list = stats.sKL;
    elbo_list = stats.elbo;
    
    sn = sqrt(optimState.sn2hpd);
    TolSN = sqrt(sn/options.TolSD)*options.TolSD;
    TolSD = min(max(options.TolSD,TolSN),options.TolSD*10);

    rindex_vec(1) = abs(elbo_list(iter) - elbo_list(iter-1)) / TolSD;
    rindex_vec(2) = stats.elbo_sd(iter) / TolSD;
    rindex_vec(3) = sKL_list(iter) / options.TolsKL;    % This should be fixed

    % Stop sampling after sample variance has stabilized below ToL
    if ~isempty(idx_stable) && optimState.StopSampling == 0 && ~optimState.Warmup
        varss_list = stats.gpSampleVar;
        if sum(w.*varss_list(idx_stable:iter)) < options.TolGPVarMCMC
            optimState.StopSampling = optimState.N;
        end
    end

    % Compute average ELCBO improvement per fcn eval in the past few iters
    idx0 = max(1,iter-ceil(0.5*TolStableIters)+1);
    xx = stats.funccount(idx0:iter);
    yy = stats.elbo(idx0:iter) - options.ELCBOImproWeight*stats.elbo_sd(idx0:iter);
    p = polyfit(xx,yy,1);
    ELCBOimpro = p(1);

else
    rindex_vec = Inf(1,3);
    ELCBOimpro = NaN;
end

% Store reliability index
rindex = mean(rindex_vec);
stats.rindex(iter) = rindex;
stats.elcbo_impro(iter) = ELCBOimpro;
optimState.R = rindex;

% Check stability termination condition
stableflag = false;
if iter >= TolStableIters && ... 
        rindex < 1 && ...
        ELCBOimpro < options.TolImprovement
    
    % Count how many good iters in the recent past (excluding current)
    stablecount = sum(stats.rindex(iter-TolStableIters+1:iter-1) < 1);
    
    % Iteration is stable if almost all recent iterations are stable
    if stablecount >= TolStableIters - floor(TolStableIters*options.TolStableExcptFrac) - 1
        % If stable but entropy switch is ON, turn it off and continue
        if optimState.EntropySwitch && isfinite(options.EntropyForceSwitch)
            optimState.EntropySwitch = false;
            if isempty(action); action = 'entropy switch'; else; action = [action ', entropy switch']; end 
        else
            % Allow termination only if distant from last warping
            if (iter - optimState.LastWarping) >= TolStableIters/3            
                isFinished_flag = true;
                exitflag = 1;
                msg = 'Inference terminated: variational solution stable for OPTIONS.TolStableCount fcn evaluations.';
            end
            stableflag = true;
            if isempty(action); action = 'stable'; else; action = [action ', stable']; end     
        end
    end
end
stats.stable(iter) = stableflag;        % Store stability flag    

% Prevent early termination
if optimState.funccount < options.MinFunEvals || ...
        optimState.iter < options.MinIter
    isFinished_flag = false;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [idx_stable,dN,dN_last,w] = getStableIter(stats,optimState,options)
%GETSTABLEITER Find index of starting stable iteration.

iter = optimState.iter;
idx_stable = [];
dN = [];    dN_last = [];   w = [];

if optimState.iter < 3; return; end

if ~isempty(stats)
    N_list = stats.N;
    idx_stable = 1;
    if ~isempty(idx_stable)
        dN = optimState.N - N_list(idx_stable);
        dN_last = N_list(end) - N_list(end-1);
    end
    
    % Compute weighting function
    Nw = numel(idx_stable:iter);    
    w1 = zeros(1,Nw);
    w1(end) = 1;
    w2 = exp(-(stats.N(end) - stats.N(end-Nw+1:end))/10);
    w2 = w2 / sum(w2);
    w = 0.5*w1 + 0.5*w2;
    
end

end

