function Knew = updateK(optimState,stats,options)
%UPDATEK Update number of variational mixture components.

Kfun_max = options.KfunMax;
Neff = optimState.Neff;
if isnumeric(Kfun_max)
    Kmax = ceil(Kfun_max);
elseif isa(Kfun_max,'function_handle')
    Kmax = ceil(Kfun_max(Neff));
end

if isa(options.AdaptiveK,'function_handle')
    Kbonus = round(options.AdaptiveK(optimState.vpK));
else
    Kbonus = round(double(options.AdaptiveK));
end

if 0
    % Adaptive increase of number of components (this should be improved)
    [Kmin,Kmax] = getK(optimState,options);
    Knew = optimState.vpK;
    Knew = max(Knew,Kmin);
    % Bonus component for stable solution (speed up exploration)
    if optimState.iter > 1 && stats.sKL(end) < options.TolsKL*options.FunEvalsPerIter
        % No bonus if any component was recently pruned
        RecentPrunedIters = ceil(options.TolStableIters/2);
        if all(stats.pruned(max(1,end-RecentPrunedIters+1):end) == 0)
            Knew = optimState.vpK + Kbonus;
        end
    end
    Knew = min(Knew,Kmax);
else    
    Knew = optimState.vpK;
    if ~optimState.Warmup && optimState.iter > 1
        idx_range = ceil(options.TolStableIters/2);
        elbos = stats.elbo(max(1,end-idx_range+1):end);
        elboSDs = stats.elboSD(max(1,end-idx_range+1):end);
        elcbos = elbos - options.ELCBOImproWeight*elboSDs;
        warmups = stats.warmup(max(1,end-idx_range+1):end);
        elcbo_max = max(elcbos(~warmups));
        if elcbos(end) >= elcbo_max; improving_flag = true; else; improving_flag = false; end        
        
        if stats.pruned(end) == 0 && improving_flag
            Knew = Knew + 1;
        end
        % Bonus component for stable solution (speed up exploration)
        if stats.qindex(end) < 1 && ~optimState.RecomputeVarPost && improving_flag
            % No bonus if any component was recently pruned
            RecentPrunedIters = ceil(options.TolStableIters/2);
            if all(stats.pruned(max(1,end-RecentPrunedIters+1):end) == 0)
                Knew = Knew + Kbonus;
            end
        end
        Knew = min(Knew,Kmax);
    end
end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Kmin,Kmax] = getK(optimState,options)
%GETK Get number of variational components.

Neff = optimState.Neff;
Kfun = options.Kfun;
Kfun_max = options.KfunMax;

if optimState.Warmup
    Kmin = options.Kwarmup;
    Kmax = options.Kwarmup;
else
    if isnumeric(Kfun)
        Kmin = Kfun;
    elseif isa(Kfun,'function_handle')
        Kmin = Kfun(Neff);
    end
    if isnumeric(Kfun_max)
        Kmax = Kfun_max;
    elseif isa(Kfun_max,'function_handle')
        Kmax = Kfun_max(Neff);
    end
    
    Kmin = min(Neff,max(1,round(Kmin)));
    Kmax = max(Kmin,min(Neff,max(1,round(Kmax))));
end

end