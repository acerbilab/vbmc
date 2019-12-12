function [exitflag,best,idx_best,stats] = vbmc_diagnostics(vp_array,beta_lcb,elbo_thresh,sKL_thresh,maxmtv_thresh)
%VBMC_DIAGNOSTICS Convergence diagnostics between multiple VBMC runs.
%   EXITFLAG = VBMC_DIAGNOSTICS(VP_ARRAY) runs a series of diagnostic tests 
%   on an array of variational posteriors. VP_ARRAY is a cell array or 
%   struct array of variational posteriors obtained by separate runs of 
%   VBMC on the same problem. EXITFLAG describes the result of the analysis.
%   Possible values of EXITFLAG and the corresponding test results are
%
%    1  PASSED: All diagnostics tests passed.
%    0  FAILED: Only one solution converged, cannot perform useful diagnostics.
%   -1  FAILED: Not enough solutions agree with the best posterior (in terms 
%       of symmetrized KL-divergence or maximum marginal total variation 
%       distance).
%   -2  FAILED: Not enough solutions agree with the best ELBO.
%   -3  FAILED: No solution has converged. No "best" solution.
%
%   A minimum of 2 separate runs of VBMC are required to perform diagnostic
%   checks, and it is recommended to perform at least 3 or 4 runs.
%
%   [EXITFLAG,BEST] = VBMC_DIAGNOSTICS(...) returns a struct BEST that 
%   contains the "best" solution, that is the solution with highest ELCBO 
%   (lower confidence bound on the ELBO) among the solutions that have 
%   converged. The fields of BEST are 'vp', which contains the variational 
%   posterior, its associated ELBO ('elbo') and estimated error ('elbo_sd'). 
%   You should be wary of using a solution which has not fully passed the 
%   test diagnostics. BEST is returned empty if no solution has converged.
%
%   [EXITFLAG,BEST,IDX_BEST] = VBMC_DIAGNOSTICS(...) also returns the index
%   within the array VP_ARRAY of the returned "best" solution. IDX_BEST
%   is returned empty if no solution has converged.
%
%   [EXITFLAG,BEST,IDX_BEST,STATS] = VBMC_DIAGNOSTICS(...) returns a struct
%   STATS with summary statistics of the diagnostic tests.
%
%   [...] = VBMC_DIAGNOSTICS(VP_ARRAY,BETA_LCB) uses lower confidence bound
%   factor BETA_LCB to judge the best solution in terms of ELCBO (default
%   BETA_LCB = 3).
%
%   [...] = VBMC_DIAGNOSTICS(VP_ARRAY,BETA_LCB,ELBO_THRESH) specifies the
%   threshold on the ELBO difference to judge two variational solutions as 
%   "close" (default ELBO_THRESH = 1).
%
%   [...] = VBMC_DIAGNOSTICS(VP_ARRAY,BETA_LCB,ELBO_THRESH,SKL_THRESH) 
%   specifies the threshold on the symmetrized Kullback-Leibler divergence
%   to judge two variational posteriors as "close" (default SKL_THRESH = 1).
%
%   [...] = VBMC_DIAGNOSTICS(VP_ARRAY,BETA_LCB,ELBO_THRESH,SKL_THRESH,MAXMTV_THRESH) 
%   specifies the threshold on the maximum marginal total variation distance
%   to judge two variational posteriors as "close" (default MAXMTV_THRESH = 0.2).
%
%   See also VBMC, VBMC_EXAMPLES, VBMC_KLDIV, VBMC_MTV.

if nargin < 3 || isempty(beta_lcb); beta_lcb = 3; end
if nargin < 4 || isempty(elbo_thresh); elbo_thresh = 1; end
if nargin < 5 || isempty(sKL_thresh); sKL_thresh = 1; end
if nargin < 6 || isempty(maxmtv_thresh); maxmtv_thresh = 0.2; end

Nruns = numel(vp_array);
exitflag = Inf;

% At least one third of solutions need to be close to the best
TolClose = 1/3;

if Nruns == 1
    warning('vbmc_diagnostics:SingleInput', ...
            'VP_ARRAY needs to be a cell or struct array of variational posteriors resulting from *multiple* VBMC runs.');
end
    
% Convert struct arrays to cell arrays
if isstruct(vp_array)
    for iRun = 1:numel(vp_array); temp{iRun} = vp_array(iRun); end
    vp_array = temp;
    clear temp;
end

D = vp_array{1}.D;
rec_runs = max(2,ceil(log2(D)));

if Nruns < rec_runs
    % We give a warning but it does not affect diagnostic results
    warning(['For a problem in D=' num2str(D) ' dimensions, it is recommended to perform at least ' num2str(rec_runs) ' VBMC runs.']);
end

% Get stats for each run
elbo = NaN(1,Nruns);    elbo_sd = NaN(1,Nruns);     stable_flag = false(1,Nruns);
for iFit = 1:Nruns
    elbo(iFit) = vp_array{iFit}.stats.elbo;
    elbo_sd(iFit) = vp_array{iFit}.stats.elbo_sd;
    stable_flag(iFit) = vp_array{iFit}.stats.stable;
end

% Check which runs have converged
idx_ok = stable_flag;
idx_active = idx_ok;

fprintf('%d out of %d variational optimization runs have converged (%.1f%%).\n',sum(idx_ok),Nruns,sum(idx_ok)/Nruns*100);

if sum(idx_ok) == 0
    warning('No variational optimization run has converged, using potentially unstable solution.');
    idx_active = true(size(idx_ok));
    exitflag = -3;
    
elseif sum(idx_ok) == 1
    warning('Only one variational optimization run has converged. You should perform more runs.');
    exitflag = 0;
end

% Compute ELCBO, that is lower confidence bound on ELBO
elcbo = elbo - beta_lcb*elbo_sd;

% Pick best variational solution based on ELCBO
elcbo_eff = elcbo;
elcbo_eff(~idx_active) = -Inf;
[~,idx_best] = max(elcbo_eff);

% Compute distances (KL-divergence and MaxMTV) across all pairs of solutions
kl_mat = zeros(Nruns,Nruns);
maxmtv_mat = zeros(Nruns,Nruns);
for iRun = 1:Nruns
    for jRun = iRun+1:Nruns        
        [kl,xx1,xx2] = vbmc_kldiv(vp_array{iRun},vp_array{jRun});
        kl_mat(iRun,jRun) = kl(1);
        kl_mat(jRun,iRun) = kl(2);
        maxmtv_mat(iRun,jRun) = max(vbmc_mtv(xx1,xx2));
        maxmtv_mat(jRun,iRun) = maxmtv_mat(iRun,jRun);
    end
end

% Compute symmetrized KL-divergence between best solution and the others
sKL_best = NaN(1,Nruns);
for iRun = 1:Nruns
    sKL_best(iRun) = 0.5*(kl_mat(iRun,idx_best)+kl_mat(idx_best,iRun));
end

% Max marginal total variation between best solution and the others
maxmtv_best = maxmtv_mat(idx_best,:);

fprintf('\n Run #    Mean[ELBO]   Std[ELBO]      ELCBO    Converged    sKL[best]  Max-MTV[best]\n');
for iRun = 1:Nruns    
    if stable_flag(iRun) == 1; ctext = 'yes'; else ctext = 'no'; end
    if idx_best == iRun; btext = 'best'; else btext = ''; end
    fprintf('%4d  %12.2f %11.2f %12.2f %10s %11.2f   %12.2f %8s\n',iRun,elbo(iRun),elbo_sd(iRun),elcbo(iRun),ctext,sKL_best(iRun),maxmtv_best(iRun),btext);
end

fprintf('\n');

if Nruns > 1
    % Check closeness of solutions in terms of ELBO
    elbo_ok = abs(elbo(idx_best) - elbo) < elbo_thresh;
    if sum(elbo_ok) > 1
        fprintf('%d out of %d runs (%.1f%%) agree with the best ELBO (difference < %.2f).\n',...
            sum(elbo_ok),Nruns,sum(elbo_ok)/Nruns*100,elbo_thresh);        
    else
        fprintf('Among %d runs, no agreement with the best ELBO (difference > %.2f).\n',...
            Nruns,elbo_thresh);                
    end
    if sum(elbo_ok) < max(Nruns*TolClose,2)
       warning('Not enough solutions agree with the best ELBO.');
       exitflag = min(exitflag,-2);
    end

    % Check closeness of solutions in terms of symmetrized KL-divergence
    sKL_ok = sKL_best < sKL_thresh;
    if sum(sKL_ok) > 1
        fprintf('%d out of %d runs (%.1f%%) agree with the best posterior (symmetrized KL-divergence < %.2f).\n',...
            sum(sKL_ok),Nruns,sum(sKL_ok)/Nruns*100,sKL_thresh);
    else
        fprintf('Among %d runs, no agreement with the best posterior (symmetrized KL-divergence > %.2f).\n',...
            Nruns,elbo_thresh);
    end
    if sum(sKL_ok) < max(Nruns*TolClose,2)
       warning('Not enough solutions agree with the best posterior (symmetrized KL-divergence).');
       exitflag = min(exitflag,-1);
    end
    
    % Check closeness of solutions in terms of max MTV
    maxmtv_ok = maxmtv_best < maxmtv_thresh;
    if sum(maxmtv_ok) > 1
        fprintf('%d out of %d runs (%.1f%%) agree with the best posterior (max marginal total variation distance < %.2f).\n',...
            sum(maxmtv_ok),Nruns,sum(maxmtv_ok)/Nruns*100,maxmtv_thresh);
    else
        fprintf('Among %d runs, no agreement with the best posterior (max marginal total variation distance > %.2f).\n',...
            Nruns,maxmtv_thresh);
    end
    if sum(maxmtv_ok) < max(Nruns*TolClose,2)
       warning('Not enough solutions agree with the best posterior (max marginal total variation distance).');
       exitflag = min(exitflag,-1);
    end
    
    fprintf('\n');    
end

fprintf('Full KL-divergence matrix:');
kl_mat

fprintf('Full Max-marginal total variation distance matrix:');
maxmtv_mat

% Nothing bad found, diagnostic test passed
if isinf(exitflag); exitflag = 1; end

switch exitflag
    case 1; msg = 'Diagnostic test PASSED.';
    case 0; msg = 'Diagnostic test FAILED. Only one solution converged; cannot perform useful diagnostics.';
    case -1; msg = 'Diagnostic test FAILED. Not enough solutions agree with the best posterior (in terms of symmetrized KL-divergence or maximum marginal total variation distance).';
    case -2; msg = 'Diagnostic test FAILED. Not enough solutions agree with the best ELBO.';
    case -3; msg = 'Diagnostic test FAILED. No solution has converged.';    
end

fprintf('\n%s\n',msg);

% Return best solution, only if it has converged
if nargout > 1
    if stable_flag(idx_best)
        best.vp = vp_array{idx_best};    
        best.elbo = elbo(idx_best);
        best.elbo_sd = elbo_sd(idx_best);
    else
        best = [];
        idx_best = [];
    end
end

% Create diagnostics STATS struct
if nargout > 3
    stats.beta_lcb = beta_lcb;
    stats.elbo_thresh = elbo_thresh;
    stats.sKL_thresh = sKL_thresh;
    stats.maxmtv_thresh = maxmtv_thresh;
    stats.elbo = elbo;
    stats.elbo_sd = elbo_sd;
    stats.elcbo = elcbo;
    stats.idx_best = idx_best;
    stats.sKL_best = sKL_best;
    stats.maxmtv_best = maxmtv_best;
    stats.kl_mat = kl_mat;
    stats.maxmtv_mat = maxmtv_mat;
    stats.exitflag = exitflag;
    stats.msg = msg;
end


end