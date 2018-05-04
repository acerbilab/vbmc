function acq = vbmc_negGEV(Xs,vp,gp,alphas,transpose_flag)
%VBMC_NEGGEV Negative generalized expected variance.

% Xs is in *transformed* coordinates

if nargin < 4 || isempty(alphas); alphas = [0,0.5,1,2,3]; end
if nargin < 5 || isempty(transpose_flag); transpose_flag = false; end

% Transposed input (useful for CMAES)
if transpose_flag; Xs = Xs'; end

[N,D] = size(Xs);    % Number of points and dimension

% Probability density of variational posterior at test points
p = max(vbmc_pdf(Xs,vp,0),realmin);

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