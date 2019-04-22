function [vp,elbo,elbo_sd,idx_best] = vbmc_best(stats,idx,SafeSD,FracBack)
%VBMC_BEST Return best variational posterior from stats structure.

% Check up to this iteration (default, last)
if nargin < 2 || isempty(idx); idx = stats.iter(end); end

% Penalization for uncertainty (default, 5 SD)
if nargin < 3 || isempty(SafeSD); SafeSD = 5; end

% If no past stable iteration, go back up to this fraction of iterations
if nargin < 4 || isempty(FracBack); FracBack = 0.25; end

if stats.stable(idx)
    % If the current iteration is stable, return it
    idx_best = idx;
    
else
    % Otherwise, find recent solution with best expected lower confidence 
    % variational bound (ELCBO)
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

% Return best variational posterior, its ELBO and SD
vp = stats.vp(idx_best);
elbo = stats.elbo(idx_best);
elbo_sd = stats.elbo_sd(idx_best);
vp.stats.stable = stats.stable(idx_best);

end