function [vp,elbo,elbo_sd,idx_best] = best_vbmc(stats,idx,SafeSD,FracBack,RankCriterion)
%VBMC_BEST Return best variational posterior from stats structure.

% Check up to this iteration (default, last)
if nargin < 2 || isempty(idx); idx = stats.iter(end); end

% Penalization for uncertainty (default, 5 SD)
if nargin < 3 || isempty(SafeSD); SafeSD = 5; end

% If no past stable iteration, go back up to this fraction of iterations
if nargin < 4 || isempty(FracBack); FracBack = 0.25; end

% Use new ranking criterion method to pick best solution
if nargin < 5 || isempty(RankCriterion); RankCriterion = false; end

if stats.stable(idx)
    % If the current iteration is stable, return it
    idx_best = idx;
    
else
    % Otherwise, find best solution according do various criteria
    
    if RankCriterion
        % Find solution that combines ELCBO, stability, and recency
        
        % Rank by position
        rank(:,1) = fliplr(1:idx)';

        % Rank by ELCBO
        lnZ_iter = stats.elbo(1:idx);
        lnZsd_iter = stats.elbo_sd(1:idx);        
        elcbo = lnZ_iter - SafeSD*lnZsd_iter;        
        [~,ord] = sort(elcbo,'descend');
        rank(ord,2) = 1:idx;

        % Rank by reliability index
        [~,ord] = sort(stats.rindex(1:idx),'ascend');
        rank(ord,3) = 1:idx;

        % Rank penalty to all non-stable iterations
        rank(:,4) = idx;
        rank(stats.stable(1:idx),4) = 1;
        
%         % Add rank penalty to warmup (and iteration immediately after)
%         last_warmup = find(stats.warmup(1:idx),1,'last');        
%         rank(:,5) = 1;
%         rank(1:min(last_warmup+2,end),5) = idx;
                
        [~,idx_best] = min(sum(rank,2));
        
    else
        % Find recent solution with best ELCBO
        laststable = find(stats.stable(1:idx),1,'last');
        if isempty(laststable)
            BackIter = ceil(idx*FracBack);  % Go up to this iterations back if no previous stable iteration
            idx_start = max(1,idx-BackIter);
        else
            idx_start = laststable;
        end
        lnZ_iter = stats.elbo(idx_start:idx);
        lnZsd_iter = stats.elbo_sd(idx_start:idx);
        elcbo = lnZ_iter - SafeSD*lnZsd_iter;
        [~,idx_best] = max(elcbo);
        idx_best = idx_start + idx_best - 1;
    end
end

% Return best variational posterior, its ELBO and SD
vp = vptrain2real(stats.vp(idx_best),1);
elbo = stats.elbo(idx_best);
elbo_sd = stats.elbo_sd(idx_best);
vp.stats.stable = stats.stable(idx_best);

end