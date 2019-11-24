function Knew = updateK(optimState,stats,options)
%UPDATEK Update number of variational mixture components.

Knew = optimState.vpK;

% Compute maximum number of components
Kmax = ceil(evaloption_vbmc(options.KfunMax,optimState.Neff));

% Evaluate bonus for stable solution
Kbonus = round(double(evaloption_vbmc(options.AdaptiveK,Knew)));


% If not warming up, check if number of components gets to be increased
if ~optimState.Warmup && optimState.iter > 1
    
    RecentIters = ceil(0.5*options.TolStableCount/options.FunEvalsPerIter);
    
    % Check if ELCBO has improved wrt recent iterations
    elbos = stats.elbo(max(1,end-RecentIters+1):end);
    elboSDs = stats.elbo_sd(max(1,end-RecentIters+1):end);
    elcbos = elbos - options.ELCBOImproWeight*elboSDs;
    warmups = stats.warmup(max(1,end-RecentIters+1):end);
    elcbos_after = elcbos(~warmups);
    elcbos_after(1:min(2,end)) = -Inf; % Ignore two iterations right after warmup
    elcbo_max = max(elcbos_after);
    improving_flag = elcbos_after(end) >= elcbo_max && isfinite(elcbos_after(end));

    % Add one component if ELCBO is improving and no pruning in last iteration
    if stats.pruned(end) == 0 && improving_flag
        Knew = Knew + 1;
    end
    
    % Bonus components for stable solution (speed up exploration)
    if stats.rindex(end) < 1 && ~optimState.RecomputeVarPost && improving_flag
        % No bonus if any component was very recently pruned
        if all(stats.pruned(max(1,end-ceil(0.5*RecentIters)+1):end) == 0)
            Knew = Knew + Kbonus;
        end
    end
    Knew = max(optimState.vpK,min(Knew,Kmax));
end
    
end