function [optimState,t_func] = initdesign_vbmc(optimState,Ns,funwrapper,t_func,options)
%INITDESIGN_VBMC Initial sample design (provided or random box).

x0 = optimState.Cache.X_orig;
[N0,D] = size(x0);

if N0 <= Ns
    Xs = x0;
    ys = optimState.Cache.y_orig;
    if N0 < Ns
        switch lower(options.InitDesign)
            case 'plausible'
                % Uniform random samples in the plausible box (in transformed space)
                Xrnd = bsxfun(@plus,bsxfun(@times,rand(Ns-N0,D),optimState.PUB-optimState.PLB),optimState.PLB);
            case 'narrow'
                xstart = warpvars_vbmc(x0(1,:),'dir',optimState.trinfo);
                Xrnd = bsxfun(@plus,bsxfun(@times,rand(Ns-N0,D)-0.5,0.1*(optimState.PUB-optimState.PLB)),xstart);
                Xrnd = bsxfun(@min,bsxfun(@max,Xrnd,optimState.PLB),optimState.PUB);
            otherwise
                error('Unknown initial design for VBMC.');
        end
        Xrnd = warpvars_vbmc(Xrnd,'inv',optimState.trinfo);  % Convert back to original space
        Xs = [Xs; Xrnd];
        ys = [ys; NaN(Ns-N0,1)];
    end
    idx_remove = true(N0,1);

elseif N0 > Ns
    % Cluster starting points
    kmeans_options = struct('Display','off','Method',2,'Preprocessing','whiten');
    idx = fastkmeans(x0,Ns,kmeans_options);

    % From each cluster, take points with higher density in original space
    Xs = NaN(Ns,D); ys = NaN(Ns,1); idx_remove = false(N0,1);
    for iK = 1:Ns
        idxK = find(idx == iK);
        xx = optimState.Cache.X_orig(idxK,:);
        yy = optimState.Cache.y_orig(idxK);
        [~,idx_y] = max(yy);
        Xs(iK,:) = xx(idx_y,:);
        ys(iK) = yy(idx_y);      
        idx_remove(idxK(idx_y)) = true;
    end
end
% Remove points from starting cache
optimState.Cache.X_orig(idx_remove,:) = [];
optimState.Cache.y_orig(idx_remove) = [];

Xs = warpvars_vbmc(Xs,'d',optimState.trinfo);

for is = 1:Ns
    timer_func = tic;
    if isnan(ys(is))    % Function value is not available
        [~,optimState] = funlogger_vbmc(funwrapper,Xs(is,:),optimState,'iter');
    else
        [~,optimState] = funlogger_vbmc(funwrapper,Xs(is,:),optimState,'add',ys(is));
    end
    t_func = t_func + toc(timer_func);
end

end
