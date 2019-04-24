function [optimState,t_active,t_func] = ...
    activesample_vbmc(optimState,Ns,funwrapper,vp,vp_old,gp,options,cmaes_opts)
%ACTIVESAMPLE_VBMC Actively sample points iteratively based on acquisition function.

NSsearch = options.NSsearch;    % Number of points for acquisition fcn
t_func = 0;

timer_active = tic;

if isempty(gp)
    
    % No GP yet, just use provided points or sample from plausible box
    [optimState,t_func] = ...
        initdesign_vbmc(optimState,Ns,funwrapper,t_func,options);
    
else                    % Active uncertainty sampling
    
    SearchAcqFcn = options.SearchAcqFcn;
        
    for is = 1:Ns

        % Create search set from cache and randomly generated
        [Xsearch,idx_cache] = getSearchPoints(NSsearch,optimState,vp,options);
        
        % Evaluate acquisition function
        acq_fast = SearchAcqFcn{1}(Xsearch,vp,gp,optimState,0);

        if options.SearchCacheFrac > 0
            [~,ord] = sort(acq_fast,'ascend');
            optimState.SearchCache = Xsearch(ord,:);
            idx = ord(1);
        else
            [~,idx] = min(acq_fast);
        end
        % idx/numel(acq_fast)
        Xacq = Xsearch(idx,:);
        idx_cache_acq = idx_cache(idx);
        
        % Remove selected points from search set
        Xsearch(idx,:) = []; idx_cache(idx) = [];
        % [size(Xacq,1),size(Xacq2,1)]
        
        % Additional search with CMA-ES
        if options.SearchCMAES
            if options.SearchCMAESVPInit
                [~,Sigma] = vbmc_moments(vp,0);       
            else
                X_hpd = gethpd_vbmc(optimState,options);
                Sigma = cov(X_hpd,1);
            end
            insigma = sqrt(diag(Sigma));
            % cmaes_opts.PopSize = 16 + 3*vp.D;   % Large population size
            % cmaes_opts.DispModulo = Inf;
            fval_old = SearchAcqFcn{1}(Xacq(1,:),vp,gp,optimState,0);
            cmaes_opts.TolFun = max(1e-12,abs(fval_old*1e-3));
            [xsearch_cmaes,fval_cmaes] = cmaes_modded(func2str(SearchAcqFcn{1}),Xacq(1,:)',insigma,cmaes_opts,vp,gp,optimState,1);
            if fval_cmaes < fval_old            
                Xacq(1,:) = xsearch_cmaes';
                idx_cache_acq(1) = 0;
                % idx_cache = [idx_cache(:); 0];
                % Double check if the cache indexing is correct
            end
        end
        
        y_orig = [NaN; optimState.Cache.y_orig(:)]; % First position is NaN (not from cache)
        yacq = y_orig(idx_cache_acq+1);
        idx_nn = ~isnan(yacq);
        if any(idx_nn)
            yacq(idx_nn) = yacq(idx_nn) + warpvars(Xacq(idx_nn,:),'logp',optimState.trinfo);
        end
        
        xnew = Xacq(1,:);
        idxnew = 1;
        
        % See if chosen point comes from starting cache
        idx = idx_cache_acq(idxnew);
        if idx > 0; y_orig = optimState.Cache.y_orig(idx); else; y_orig = NaN; end
        timer_func = tic;
        if isnan(y_orig)    % Function value is not available, evaluate
            try
                [ynew,optimState] = funlogger_vbmc(funwrapper,xnew,optimState,'iter');
            catch func_error
                pause
            end
        else
            [ynew,optimState] = funlogger_vbmc(funwrapper,xnew,optimState,'add',y_orig);
            % Remove point from starting cache
            optimState.Cache.X_orig(idx,:) = [];
            optimState.Cache.y_orig(idx) = [];            
        end                
        t_func = t_func + toc(timer_func);
            
        gp = gplite_post(gp,xnew,ynew,[],1);   % Rank-1 update
    end
    
end

t_active = toc(timer_active) - t_func;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Xsearch,idx_cache] = getSearchPoints(NSsearch,optimState,vp,options)
%GETSEARCHPOINTS Get search points from starting cache or randomly generated.

% Take some points from starting cache, if not empty
x0 = optimState.Cache.X_orig;

if ~isempty(x0)
    cacheFrac = options.CacheFrac;  % Fraction of points from cache (if nonempty)
    Ncache = ceil(NSsearch*cacheFrac);            
    idx_cache = randperm(size(x0,1),min(Ncache,size(x0,1)));
    Xsearch = warpvars(x0(idx_cache,:),'d',optimState.trinfo);        
else
    Xsearch = []; idx_cache = [];
end

% Randomly sample remaining points        
if size(Xsearch,1) < NSsearch
    Nrnd = NSsearch-size(Xsearch,1);
    
    Xrnd = [];
    Nsearchcache = round(options.SearchCacheFrac*Nrnd);
    if Nsearchcache > 0 % Take points from search cache
        Xrnd = [Xrnd; optimState.SearchCache(1:min(end,Nsearchcache),:)];
    end
    Nheavy = round(options.HeavyTailSearchFrac*Nrnd);
    if Nheavy > 0
        Xrnd = [Xrnd; vbmc_rnd(vp,Nheavy,0,1,3)];
    end
    Nmvn = round(options.MVNSearchFrac*Nrnd);
    if Nmvn > 0
        [mubar,Sigmabar] = vbmc_moments(vp,0);
        Xrnd = [Xrnd; mvnrnd(mubar,Sigmabar,Nmvn)];
    end
    Nhpd = round(options.HPDSearchFrac*Nrnd);
    if Nhpd > 0
        X_hpd = gethpd_vbmc(optimState,options);
        mubar = mean(X_hpd,1);
        Sigmabar = cov(X_hpd,1);
        Xrnd = [Xrnd; mvnrnd(mubar,Sigmabar,Nhpd)];
    end
    Nvp = max(0,Nrnd-Nsearchcache-Nheavy-Nmvn-Nhpd);
    if Nvp > 0
        Xrnd = [Xrnd; vbmc_rnd(vp,Nvp,0,1)];
    end
    
    Xsearch = [Xsearch; Xrnd];
    idx_cache = [idx_cache(:); zeros(Nrnd,1)];
end

end
