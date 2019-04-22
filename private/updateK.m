function Knew = updateK(optimState,stats,options)
%UPDATEK Update number of variational mixture components.

% Compute maximum number of components
Kfun_max = options.KfunMax;
Neff = optimState.Neff;
if isnumeric(Kfun_max)
    Kmax = ceil(Kfun_max);
elseif isa(Kfun_max,'function_handle')
    Kmax = ceil(Kfun_max(Neff));
end

% Evaluate bonus for stable solution
if isa(options.AdaptiveK,'function_handle')
    Kbonus = round(options.AdaptiveK(optimState.vpK));
else
    Kbonus = round(double(options.AdaptiveK));
end

Knew = optimState.vpK;

% If not warming up, check if number of components gets to be increased
if ~optimState.Warmup && optimState.iter > 1
    
    RecentIters = ceil(options.TolStableIters/2);
    
    % Check if ELCBO has improved wrt recent iterations
    elbos = stats.elbo(max(1,end-RecentIters+1):end);
    elboSDs = stats.elbo_sd(max(1,end-RecentIters+1):end);
    elcbos = elbos - options.ELCBOImproWeight*elboSDs;
    warmups = stats.warmup(max(1,end-RecentIters+1):end);
    elcbo_max = max(elcbos(~warmups));
    if elcbos(end) >= elcbo_max; improving_flag = true; else; improving_flag = false; end

    % Add one component if ELCBO is improving and no pruning in last iteration
    if stats.pruned(end) == 0 && improving_flag
        Knew = Knew + 1;
    end
    
    % Bonus components for stable solution (speed up exploration)
    if stats.rindex(end) < 1 && ~optimState.RecomputeVarPost && improving_flag
        % No bonus if any component was recently pruned
        if all(stats.pruned(max(1,end-RecentIters+1):end) == 0)
            Knew = Knew + Kbonus;
        end
    end
    Knew = max(optimState.vpK,min(Knew,Kmax));
end
    
end