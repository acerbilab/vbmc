function [optimState,t_adapt,t_func] = ...
    adaptive_sampling(optimState,Ns,funwrapper,vp,vp_old,gp,options,cmaes_opts)
%ADAPTIVE_SAMPLING Choose sampled points iteratively based on acquisition function.

NSsearch = options.NSsearch;    % Number of points for fast acquisition fcn
Nacq = options.Nacq;            % Number of evals of slow acquisition fcn
acqfun = options.AcqFcn;        % Slow acquisition fcn
cacheFrac = options.CacheFrac;  % Fraction of points from cache (if nonempty)
t_func = 0;
cmaes_search = 0;
refine_acq = 0;

timer_adapt = tic;

if isempty(gp)     % No GP yet, just use provided points or sample from plausible box

    x0 = optimState.Cache.X_orig;
    [N0,D] = size(x0);
    
    if N0 <= Ns
        Xs = x0;
        ys = optimState.Cache.y_orig;
        if N0 < Ns
            width = (optimState.PUB - optimState.PLB)*0.05;
            PLB = min(min(x0,[],1),optimState.PLB);
            PUB = max(max(x0,[],1),optimState.PUB);
            
            Xrnd = bsxfun(@plus,bsxfun(@times,rand(Ns-N0,D)-0.5,width),x0(1,:));
            Xrnd = bsxfun(@min,bsxfun(@max,Xrnd,PLB),PUB);
            % Xrnd = bsxfun(@plus,bsxfun(@times,rand(Ns-N0,D),optimState.PUB-optimState.PLB),optimState.PLB);
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
    
    Xs = warpvars(Xs,'d',optimState.trinfo);
    
    for is = 1:Ns
        timer_func = tic;
        if isnan(ys(is))    % Function value is not available
            [~,optimState] = vbmc_funlogger(funwrapper,Xs(is,:),optimState,'iter');
        else
            [~,optimState] = vbmc_funlogger(funwrapper,Xs(is,:),optimState,'add',ys(is));
        end
        t_func = t_func + toc(timer_func);
    end
    
else                    % Adaptive uncertainty sampling
        
    if ~isfield(optimState,'acqScore') || isempty(optimState.acqScore)
        optimState.acqScore = zeros(1,Nacq); % Measure metrics effectiveness
    end
        
    for is = 1:Ns

        % Compute expected log joint and its variance (only diagonal terms)
        [G,~,vardiagG] = gplogjoint(vp,gp,[0 0 0],1,[],2);

        % Create search set from cache and randomly generated
        [Xsearch,idx_cache] = getSearchPoints(NSsearch,Ns,optimState,gp,vp,options);
        
        % Evaluate fast search acquisition function(s)
        SearchAcqFcn = options.SearchAcqFcn;        
        acq_fast = [];
        for iAcqFast = 1:numel(SearchAcqFcn)
            acq_fast = [acq_fast,SearchAcqFcn{iAcqFast}(Xsearch,vp,gp,optimState,Nacq,0)];
        end
        % acq_fast = vbmc_fastacq(Xsearch,vp,vp_old,gp,G,vardiagG,acqfast_flags,0);
        [~,idx] = min(acq_fast);
        Xacq = Xsearch(idx,:);
        [Xacq,idx_unique] = unique(Xacq,'rows');   % Remove duplicates
        idx_cache_acq = idx_cache(idx(idx_unique));
        
        % Remove selected points from search set
        Xsearch(idx,:) = []; idx_cache(idx) = [];
        % [size(Xacq,1),size(Xacq2,1)]
        
        % Additional search with CMA-ES
        if options.SearchCMAES
            insigma = max(vp.sigma)*vp.lambda;
            %xsearch_cmaes = cmaes_modded('vbmc_acqGEV',Xacq(1,:)',insigma,cmaes_opts,vp,gp,optimState,1,1);
            [xsearch_cmaes,fval_cmaes] = cmaes_modded('vbmc_acqprop',Xacq(1,:)',insigma,cmaes_opts,vp,gp,optimState,1,1);
            fval_old = vbmc_acqprop(Xacq(1,:),vp,gp,optimState,1);
            if fval_cmaes < fval_old            
                Xacq(1,:) = xsearch_cmaes';
                idx_cache_acq(1) = 0;
                % idx_cache = [idx_cache(:); 0];
                % Double check if the cache indexing is correct
            end
        end

        if Nacq > size(Xacq,1)
            % Fill in with random search
            Nrnd = Nacq-size(Xacq,1);
            idx = randperm(size(Xsearch,1),Nrnd);
            Xacq = [Xacq; Xsearch(idx,:)];
            idx_cache_acq = [idx_cache_acq(:); idx_cache(idx)];
            % Remove selected points from search set            
            Xsearch(idx,:) = []; idx_cache(idx) = [];
        elseif Nacq < size(Xacq,1)
            % Choose randomly NACQ starting vectors
            idx = randperm(size(Xacq,1),Nacq);
            Xacq = Xacq(idx,:);
            idx_cache_acq = idx_cache_acq(idx);
        end
        
        y_orig = [NaN; optimState.Cache.y_orig(:)]; % First position is NaN (not from ca
        yacq = y_orig(idx_cache_acq+1);
        idx_nn = ~isnan(yacq);
        if any(idx_nn)
            yacq(idx_nn) = yacq(idx_nn) + warpvars(Xacq(idx_nn,:),'logp',optimState.trinfo);
        end
        
        % Evaluate expensive acquisition function on chosen batch
        if Nacq > 1
            [xnew,idxnew,acq] = eval_acq(acqfun,Xacq,yacq,vp,gp,G,vardiagG,options);
            
            [~,ord] = sort(acq,'ascend');
            optimState.acqScore(ord(1)) = optimState.acqScore(ord(1)) + 1;
            
            if refine_acq   % Refine search
                % Pick closest variational component
                d2 = sq_dist(vp.mu,xnew');
                [~,idx] = min(d2);
                scalingfactor = 0.25;
                xrnd = bsxfun(@plus,xnew,bsxfun(@times,scalingfactor*vp.sigma(idx)*vp.lambda(:)',randn(ceil(Nacq/2),vp.D)));
                [xnew_ref,acq_ref] = eval_acq(acqfun,xrnd,NaN(size(xrnd,1),1),vp,gp,G,vardiagG,options);
                if min(acq_ref) < min(acq); xnew = xnew_ref; end
            end
            
        else
            xnew = Xacq(1,:); idxnew = 1;
        end
        
        % See if chosen point comes from starting cache
        idx = idx_cache_acq(idxnew);
        if idx > 0; y_orig = optimState.Cache.y_orig(idx); else; y_orig = NaN; end
        timer_func = tic;
        if isnan(y_orig)    % Function value is not available, evaluate
            try
                [ynew,optimState] = vbmc_funlogger(funwrapper,xnew,optimState,'iter');
            catch
                pause
            end
        else
            [ynew,optimState] = vbmc_funlogger(funwrapper,xnew,optimState,'add',y_orig);
            % Remove point from starting cache
            optimState.Cache.X_orig(idx,:) = [];
            optimState.Cache.y_orig(idx) = [];            
        end                
        t_func = t_func + toc(timer_func);
        
        gp = gplite_post(gp,xnew,ynew,[],1);   % Rank-1 update
    end
    
    % optimState.acqScore
end

t_adapt = toc(timer_adapt) - t_func;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Xsearch,idx_cache] = getSearchPoints(NSsearch,Ns,optimState,gp,vp,options)
%GETSEARCHPOINTS Get search points from starting cache or randomly generated.

% Take some points from starting cache, if not empty
x0 = optimState.Cache.X_orig;
if ~isempty(x0)
    Ncache = ceil(NSsearch*cacheFrac);            
    idx_cache = randperm(size(x0,1),min(Ncache,size(x0,1)));
    Xsearch = warpvars(x0(idx_cache,:),'d',optimState.trinfo);        
else
    Xsearch = []; idx_cache = [];
end

% Randomly sample remaining points        
if size(Xsearch,1) < NSsearch
    Nrnd = NSsearch-size(Xsearch,1);
    if options.SearchSampleGP && ~isempty(gp)
        Thin = 1;
        tic
        Xrnd = vbmc_gpsample(gp,round(Nrnd/Ns),vp,optimState,0);
        toc
        Xrnd = Xrnd(Thin:Thin:end,:);
    else
        Xrnd = [];
        Nheavy = round(options.HeavyTailSearchFrac*Nrnd);
        Xrnd = [Xrnd; vbmc_rnd(Nheavy,vp,0,1,3)];
        [mubar,Sigmabar] = vbmc_moments(vp,0);
        Nmvn = round(options.MVNSearchFrac*Nrnd);
        Xrnd = [Xrnd; mvnrnd(mubar,Sigmabar,Nmvn)];
        Nvp = max(0,Nrnd-Nheavy-Nmvn);
        Xrnd = [Xrnd; vbmc_rnd(Nvp,vp,0,1)];
    end
    Xsearch = [Xsearch; Xrnd];
    idx_cache = [idx_cache(:); zeros(Nrnd,1)];
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xbest,idxbest,acq] = eval_acq(acqfun,Xacq,yacq,vp,gp,G_old,vardiagG_old,options)
%EVAL_ACQ Evaluate expensive acquisition function on a set of points

Nacq = size(Xacq,1);

% Get Gauss-Hermite quadrature abscissas and weights
Ngauss = 5;
[xx_gauss,ww_gauss] = hermquad(Ngauss);
ww_gauss = ww_gauss(:)'/sqrt(pi);

Gs = NaN(Nacq,Ngauss);
vardiagGs = NaN(Nacq,Ngauss);
for iAcq = 1:Nacq
    xacq = Xacq(iAcq,:);
    if isnan(yacq(iAcq))
        [ymacq,ys2acq] = gplite_pred(gp,xacq);
        % Compute y values according to Gauss-Hermite quadrature
        ybars = ymacq + sqrt(2*ys2acq)*xx_gauss(:)';
    else
        ybars = yacq(iAcq);
    end
    gpacqs = gplite_post(gp,xacq,ybars,[],1);
    
%     if 0
%         Nopts = 100;
%         try
%             theta0 = get_theta(vp,vp.LB_theta,vp.UB_theta,vp.optimize_lambda)';
%         catch
%             pause
%         end
%         theta0 = [theta0; theta0 + 0.1*randn(Nopts-1,size(theta0,2))];
%         for iOpt = 1:Nopts
%             [E_tmp,~,~,~,varE_tmp] = vbmc_negelcbo(theta0(iOpt,:),0,vp,gp,options.NSent,0,2);
%             [E_tmp,varE_tmp] = combine_quad(E_tmp,varE_tmp,ww_gauss);    
%             nelcbo_fill(iOpt) = E_tmp + options.ELCBOWeight*sqrt(varE_tmp);
%         end
%         [~,idx] = min(nelcbo_fill);
%         vptmp = rescale_params(vp,theta0(idx,:)');
%     else
%         vptmp = vp;
%     end
    
    [Gs(iAcq,:),vardiagGs(iAcq,:)] = gplogjoint_multi(vp,gpacqs,1,2);
end

% Compute Gauss-Hermite quadrature
[Gs,vardiagGs] = combine_quad(Gs,vardiagGs,ww_gauss);    

acq = acqfun(Gs,vardiagGs,G_old,vardiagG_old);

% Take point that optimizes acquisition function
[~,idxbest] = min(acq);
xbest = Xacq(idxbest,:);

%idx
%Gtemp2(:)'
%varGtemp2(:)' 
%acq(:)'
% [varGtemp(1) varGtemp(idx_test)]
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [g,v] = combine_quad(gs,vs,ww_gauss)
   
if numel(ww_gauss) > 1
    g = sum(bsxfun(@times,ww_gauss,gs),2);            
    gs_var = sum(bsxfun(@times, ww_gauss,bsxfun(@minus,gs,g).^2),2);
    v = sum(bsxfun(@times,ww_gauss,vs),2) + gs_var;
else
    g = gs;
    v = vs;
end

end