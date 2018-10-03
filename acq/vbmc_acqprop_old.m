function acq = vbmc_acqprop(Xs,vp,gp,optimState,Nacq,transpose_flag)
%VBMC_ACQPROP Acquisition function via weighted proposal uncertainty search.

% Xs is in *transformed* coordinates

if nargin < 6 || isempty(transpose_flag); transpose_flag = false; end

% Transposed input (useful for CMAES)
if transpose_flag; Xs = Xs'; end

[N,D] = size(Xs);    % Number of points and dimension

% Probability density of variational posterior at test points
p = max(vbmc_pdf(vp,Xs,0),realmin);

% Search proposal function
Xs_orig = warpvars(Xs,'inv',vp.trinfo);
yp = optimState.ProposalFcn(Xs_orig) .* warpvars(Xs,'pdf',vp.trinfo);
yp = max(yp,realmin);

% Compute proposal vs. variational posterior weight
z = log(optimState.R)/log(20);
w = max(0, 2./(1 + exp(-z)) - 1);

if ~optimState.Warmup
    w = 0;
end
% w = min(max(log(optimState.R)/log(1e3),0),0.999);



% GP mean and variance for each hyperparameter sample
[~,~,fmu,fs2] = gplite_pred(gp,Xs,[],1);

Ns = size(fmu,2);
fbar = sum(fmu,2)/Ns;   % Mean across samples
vbar = sum(fs2,2)/Ns;   % Average variance across samples
if Ns > 1; vf = sum((fmu - fbar).^2,2)/(Ns-1); else; vf = 0; end  % Sample variance
vtot = vf + vbar;       % Total variance

acq = -vtot .* exp(2*((1-w)*log(p) + w*log(yp)));

% Transposed output
if transpose_flag; acq = acq'; end


end