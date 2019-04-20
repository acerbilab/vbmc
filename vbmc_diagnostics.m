function [exitflag,output,best] = vbmc_diagnostics(vps,outputs,beta_lcb,thresh)
%VBMC_DIAGNOSTICS Convergence diagnostics between multiple VBMC runs.
%   (Documentation to be written; work in progress.)
%
%   See also VBMC, VBMC_EXAMPLES, VBMC_KLDIV.

if nargin < 3 || isempty(beta_lcb); beta_lcb = 3; end
if nargin < 4 || isempty(thresh); thresh = [1,1]; end

% Tolerance threshold on ELBO and SKL differences
elbo_thresh = thresh(1);
if numel(thresh) > 1; sKL_thresh = thresh(2); else; sKL_thresh = thresh(1); end

Nruns = numel(vps);
exitflag = Inf;

if Nruns == 1 || numel(outputs) == 1
    error('vbmc_diagnostics:SingleInput', ...
            'In order to perform diagnostics, VPS and OUTPUTS need to be cell arrays resulting from multiple VBMC runs.');
end

if numel(outputs) ~= Nruns
    error('vbmc_diagnostics:InputMismatch', ...
            'VPS and OUTPUTS should have the same number of runs.');    
end
    
% Convert struct arrays to cell arrays
if isstruct(vps)
    for iRun = 1:numel(vps); temp{iRun} = vps(iRun); end
    vps = temp;
    clear temp;
end
if isstruct(outputs)
    for iRun = 1:numel(outputs); temp{iRun} = outputs(iRun); end
    outputs = temp;
    clear temp;    
end

% Get stats for each run
elbo = NaN(1,Nruns);    elbo_sd = NaN(1,Nruns);     exitflags = NaN(1,Nruns);
for iFit = 1:Nruns
    elbo(iFit) = outputs{iFit}.elbo;
    elbo_sd(iFit) = outputs{iFit}.elbosd;
    exitflags(iFit) = strcmpi(outputs{iFit}.convergencestatus,'probable');
end

% Check which runs have converged
idx_ok = (exitflags == 1);
idx_active = idx_ok;

fprintf('%d out of %d variational optimization runs have converged (%.1f%%).\n',sum(idx_ok),Nruns,sum(idx_ok)/Nruns*100);

if sum(idx_ok) == 0
    warning('No variational optimization run has converged, using potentially unstable solution.');
    idx_active = true(size(idx_ok));
    exitflag = -2;
    
elseif sum(idx_ok) == 1
    warning('Only one variational optimization run has converged. You should perform more runs.');
    exitflag = 0;
end

% Compute ELCBO, that is lower confidence bound on ELBO
elcbo = elbo - beta_lcb*elbo_sd;

% Pick best variational solution based on ELCBO
elcbo_eff = elcbo;
elcbo_eff(~idx_ok) = -Inf;
[~,idx_best] = max(elcbo_eff);

% Compute KL-divergence across all pairs of solutions
kl_mat = zeros(Nruns,Nruns);
for iRun = 1:Nruns
    for jRun = iRun+1:Nruns        
        kl = vbmc_kldiv(vps{iRun},vps{jRun});
        kl_mat(iRun,jRun) = kl(1);
        kl_mat(jRun,iRun) = kl(2);        
    end
end

% Compute symmetrized KL-divergence between best solution and the others
sKL_best = NaN(1,Nruns);
for iRun = 1:Nruns
    sKL_best(iRun) = 0.5*(kl_mat(iRun,idx_best)+kl_mat(idx_best,iRun));
end

fprintf('\n Run #     Mean[ELBO]    Std[ELBO]      ELCBO      Converged     sKL[best]\n');
for iRun = 1:Nruns    
    if exitflags(iRun) == 1; ctext = 'yes'; else ctext = 'no'; end
    if idx_best == iRun; btext = 'best'; else btext = ''; end
    fprintf('%4d   %12.2f %12.2f %12.2f %12s %12.2f %8s\n',iRun,elbo(iRun),elbo_sd(iRun),elcbo(iRun),ctext,sKL_best(iRun),btext);
end

fprintf('\n');
if sum(idx_active) > 1
   elbo_ok = abs(elbo(idx_best) - elbo) < elbo_thresh;
   elbo_ok(idx_best) = false;
   if sum(elbo_ok) < (sum(idx_active)-1)/3
       warning('Less than 33% of variational solutions are close to the best solution in terms of ELBO.');
       exitflag = min(exitflag,-1);
   end
   sKL_ok = sKL_best < sKL_thresh;
   sKL_ok(idx_best) = false;
   if sum(sKL_ok) < (sum(idx_active)-1)/3
       warning('Less than 33% of variational solutions are close to the best solution in terms of symmetrized KL-divergence.');
       exitflag = min(exitflag,-1);
   end
end

fprintf('Full KL-divergence matrix:');
kl_mat

% Nothing bad found, diagnostic test passed
if isinf(exitflag); exitflag = 1; end

switch exitflag
    case 1; msg = 'Diagnostic test PASSED.';
    case 0; msg = 'Diagnostic test FAILED. Only one solution converged; cannot perform useful diagnostics.';
    case -1; msg = 'Diagnostic test FAILED. Less than 33% of valid solutions are close to the best solution.';
    case -2; msg = 'Diagnostic test FAILED. No solution has converged.';    
end

fprintf('\n%s\n',msg);

% Create diagnostics OUTPUT struct
if nargout > 1
    output.beta_lcb = beta_lcb;
    output.elbo_thresh = elbo_thresh;
    output.sKL_thresh = sKL_thresh;
    output.elbo = elbo;
    output.elbo_sd = elbo_sd;
    output.elcbo = elcbo;
    output.idx_best = idx_best;
    output.sKL_best = sKL_best;
    output.kl_mat = kl_mat;
    output.exitflag = exitflag;
    output.msg = msg;
end

% Return best solution
if nargout > 2
    best.vp = vps{idx_best};    
    best.elbo = elbo(idx_best);
    best.elbo_sd = elbo_sd(idx_best);
    best.exitflag = exitflags(idx_best);
    best.output = outputs{idx_best};
end

end