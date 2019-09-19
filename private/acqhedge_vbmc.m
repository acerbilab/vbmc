function hedge = acqhedge_vbmc(action,hedge,stats,options)
%ACQPORTFOLIO Evaluate and update portfolio of acquisition functions.

switch lower(action(1:3))
    case 'acq'
        % Choose acquisition function based on hedge strategy
                
        if isempty(hedge)
            % Initialize hedge struct
            hedge.g = zeros(1,numel(options.SearchAcqFcn));
            hedge.n = numel(options.SearchAcqFcn);
            hedge.count = 0;
            hedge.lambda = 0.2;     % Lapse rate - random choice
            hedge.beta = 1;
            hedge.decay = options.AcqHedgeDecay^(options.FunEvalsPerIter);
        end

        hedge.count = hedge.count + 1;
        hedge.p = exp(hedge.beta*(hedge.g - max(hedge.g)))./sum(exp(hedge.beta*(hedge.g - max(hedge.g))));
        hedge.p = hedge.p*(1-hedge.lambda) + hedge.lambda/hedge.n;
        
        hedge.chosen = find(rand() < cumsum(hedge.p),1);
        hedge.phat = Inf(size(hedge.p));
        hedge.phat(hedge.chosen) = hedge.p(hedge.chosen);
                
    case 'upd'
        % Update value of hedge portfolio based on uncertainty reduction
        
        HedgeCutoff = 5;
        
        if ~isempty(hedge)
            iter = stats.iter(end);        
            min_iter = max(1,iter-options.AcqHedgeIterWindow);
            
            min_sd = min(stats.elbo_sd(min_iter:iter-1));        
            er_sd = max(0, log(min_sd / stats.elbo_sd(iter)));

            elcbo = stats.elbo - options.ELCBOImproWeight*stats.elbo_sd;
            max_elcbo = max(elcbo(min_iter:iter-1));
            er_elcbo = max(0,elcbo(iter) - max_elcbo)/options.TolImprovement;
            if er_elcbo > 1; er_elcbo = 1 + log(er_elcbo); end

            min_r = min(stats.rindex(min_iter:iter-1));
            er_r = max(0, log(min_r / stats.rindex(iter)));
                        
            % er = 0.5*er_sd + 0.5*er_elcbo;  % Reward
            er = er_r;
            
            for iHedge = 1:hedge.n
                hedge.g(iHedge) = hedge.decay*hedge.g(iHedge) + er/hedge.phat(iHedge);
            end
            
            % Apply cutoff value on hedge
            hedge.g = min(hedge.g,HedgeCutoff);            
            hedge.g
        end
         
end