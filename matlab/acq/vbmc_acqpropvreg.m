function acq = vbmc_acqpropvreg(Xs,vp,gp,optimState,Nacq,transpose_flag)
%VBMC_ACQPROPVREG Acquisition function via weighted proposal uncertainty search.

% Xs is in *transformed* coordinates

if nargin < 6 || isempty(transpose_flag); transpose_flag = false; end

% Transposed input (useful for CMAES)
if transpose_flag; Xs = Xs'; end

[N,D] = size(Xs);    % Number of points and dimension

% Threshold on GP variance, try not to go below this
TolVar = optimState.TolGPVar;

% Search proposal function
Xs_orig = warpvars(Xs,'inv',vp.trinfo);
yp = optimState.ProposalFcn(Xs_orig) .* warpvars(Xs,'pdf',vp.trinfo);
yp = max(yp,realmin);

% GP mean and variance for each hyperparameter sample
[~,~,fmu,fs2] = gplite_pred(gp,Xs,[],1);

Ns = size(fmu,2);
fbar = sum(fmu,2)/Ns;   % Mean across samples
vbar = sum(fs2,2)/Ns;   % Average variance across samples
if Ns > 1; vf = sum((fmu - fbar).^2,2)/(Ns-1); else; vf = 0; end  % Sample variance
vtot = vf + vbar;       % Total variance

z = optimState.ymax;
acq = -vtot .* exp(fbar+0.5*vtot-z) .* yp;    

% Regularization: penalize points where GP uncertainty is below threshold
idx = vtot < TolVar;
if any(idx)
    acq(idx) = acq(idx) .* exp(-(TolVar./vtot(idx)-1));
end

acq = max(acq,-realmax);

% Transposed output
if transpose_flag; acq = acq'; end


end