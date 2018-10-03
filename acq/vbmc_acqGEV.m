function acq = vbmc_acqGEV(Xs,vp,gp,optimState,Nacq,transpose_flag,alphas)
%VBMC_ACQGEV Acquisition function via negative generalized expected variance.

% Xs is in *transformed* coordinates

if nargin < 6 || isempty(transpose_flag); transpose_flag = false; end
if nargin < 7 || isempty(alphas); alphas = []; end

% Exponents
if isempty(alphas)
    alphas0 = [1,2,0,exp(linspace(-log(Nacq),log(3),Nacq-3))];
    alphas = unique(alphas0(1:Nacq));
end

% Transposed input (useful for CMAES)
if transpose_flag; Xs = Xs'; end

[N,D] = size(Xs);    % Number of points and dimension

% Probability density of variational posterior at test points
p = max(vbmc_pdf(vp,Xs,0),realmin);

% GP mean and variance for each hyperparameter sample
[~,~,fmu,fs2] = gplite_pred(gp,Xs,[],1);

Ns = size(fmu,2);
fbar = sum(fmu,2)/Ns;   % Mean across samples
vbar = sum(fs2,2)/Ns;   % Average variance across samples
if Ns > 1; vf = sum((fmu - fbar).^2,2)/(Ns-1); else; vf = 0; end  % Sample variance
vtot = vf + vbar;       % Total variance

acq = zeros(N,numel(alphas));  % Prepare acquisition function

for iAcq = 1:numel(alphas)
    acq(:,iAcq) = -vtot .* p.^alphas(iAcq);
end

% Transposed output
if transpose_flag; acq = acq'; end


end