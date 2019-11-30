function acq = acqfreglog_vbmc(Xs,vp,gp,optimState,transpose_flag)
%ACQFREGLOG_VBMC Acquisition fcn. for prospective uncertainty search (regularized, log-valued).

% Xs is in *transformed* coordinates

if isempty(Xs)
    % Return acquisition function info struct
    acq.compute_varlogjoint = false;
    acq.log_flag = true;
    return;
end

if nargin < 5 || isempty(transpose_flag); transpose_flag = false; end

% Transposed input (useful for CMAES)
if transpose_flag; Xs = Xs'; end

Xs = real2int_vbmc(Xs,vp.trinfo,optimState.integervars);
[N,D] = size(Xs);    % Number of points and dimension

% Threshold on GP variance, try not to go below this
TolVar = optimState.TolGPVar;

% Probability density of variational posterior at test points
p = max(vbmc_pdf(vp,Xs,0),realmin);

if isfield(vp,'delta') && ~isempty(vp.delta) && any(vp.delta > 0)
    % Quadrature mean and variance for each hyperparameter sample
    [fmu,fs2] = gplite_quad(gp,Xs,vp.delta',1);    
else
    % GP mean and variance for each hyperparameter sample
    [~,~,fmu,fs2] = gplite_pred(gp,Xs,[],[],1,0);
end

Ns = size(fmu,2);
fbar = sum(fmu,2)/Ns;   % Mean across samples
vbar = sum(fs2,2)/Ns;   % Average variance across samples
if Ns > 1
    vf = sum(bsxfun(@minus,fmu,fbar).^2,2)/(Ns-1);
    % vf(fbar < optimState.ymax - optimState.OutwarpDelta) = 0;
else
    vf = 0; 
end  % Sample variance
vtot = vf + vbar;       % Total variance

z = optimState.ymax;

acq = -(log(vtot) + fbar-z + log(p));

% Regularization: penalize points where GP uncertainty is below threshold
if optimState.VarianceRegularizedAcqFcn
    idx = vtot < TolVar;
    if any(idx)
        acq(idx) = acq(idx) + (TolVar./vtot(idx)-1);
    end
end
acq(~isfinite(acq)) = log(realmax);

% Transposed output
if transpose_flag; acq = acq'; end

end