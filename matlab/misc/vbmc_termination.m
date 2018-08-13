function [optimState,stats,isFinished_flag,exitflag,action] = vbmc_termination(optimState,action,stats,options)
%VBMC_TERMINATION Compute stability index and check termination conditions.

iter = optimState.iter;
exitflag = 0;
isFinished_flag = false;

% Maximum number of new function evaluations
if optimState.funccount >= options.MaxFunEvals
    isFinished_flag = true;
    exitflag = 1;
    % msg = 'Optimization terminated: reached maximum number of function evaluations OPTIONS.MaxFunEvals.';
end

% Maximum number of iterations
if iter >= options.MaxIter
    isFinished_flag = true;
    exitflag = 1;
    % msg = 'Optimization terminated: reached maximum number of iterations OPTIONS.MaxIter.';
end

% Reached stable variational posterior with stable ELBO and low uncertainty
[idx_stable,dN,dN_last,w] = getStableIter(stats,optimState,options);
if ~isempty(idx_stable)
    sKL_list = stats.sKL;
    elbo_list = stats.elbo;

    qindex_vec(1) = abs(elbo_list(iter) - elbo_list(iter-1))/options.TolSD;
    qindex_vec(2) = stats.elboSD(iter) / options.TolSD;
    qindex_vec(3) = sKL_list(iter) / options.TolsKL;    % This should be fixed

    % Stop sampling after sample variance has stabilized below ToL
    if ~isempty(idx_stable) && optimState.StopSampling == 0 && ~optimState.Warmup
        varss_list = stats.gpSampleVar;
        if sum(w.*varss_list(idx_stable:iter)) < options.TolGPVar
            optimState.StopSampling = optimState.N;
        end
    end

    % Compute average ELCBO improvement in the past few iterations
    idx0 = max(1,iter-options.TolStableIters+1);
    xx = stats.N(idx0:iter);
    yy = stats.elbo(idx0:iter) - options.ELCBOImproWeight*stats.elboSD(idx0:iter);
    p = polyfit(xx,yy,1);
    ELCBOimpro = p(1);

else
    qindex_vec = Inf(1,3);
    ELCBOimpro = NaN;
end

% Store reliability index
qindex = mean(qindex_vec);
stats.qindex(iter) = qindex;
stats.elcbo_impro(iter) = ELCBOimpro;
optimState.R = qindex;

% Check stability termination condition
stableflag = false;
if iter >= options.TolStableIters && ... 
        all(qindex_vec < 1) && ...
        all(stats.qindex(iter-options.TolStableIters+1:iter) < 1) && ...
        ELCBOimpro < options.TolImprovement
        % msg = 'Optimization terminated: reached maximum number of iterations OPTIONS.MaxIter.';

    % If stable but entropy switch is ON, turn it off and continue
    if optimState.EntropySwitch
        optimState.EntropySwitch = false;
        if isempty(action); action = 'entropy switch'; else; action = [action ', entropy switch']; end 
    else
        isFinished_flag = true;
        exitflag = 0;
        stableflag = true;
        if isempty(action); action = 'stable'; else; action = [action ', stable']; end     
    end
end
stats.stable(iter) = stableflag;        % Store stability flag    

% Prevent early termination
if optimState.N < options.MinFunEvals || ...
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
    %iter_list = stats.iter;
    %idx_stable = find(N_list <= optimState.N - options.TolStableFunEvals & ...
    %    iter_list <= iter - options.TolStableIters,1,'last');
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

