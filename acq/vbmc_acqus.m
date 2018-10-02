function acq = vbmc_acqus(Xs,vp,gp,optimState,Nacq,transpose_flag)
%VBMC_ACQUS Acquisition function via uncertainty sampling.

% Xs is in *transformed* coordinates

if nargin < 6 || isempty(transpose_flag); transpose_flag = false; end

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

acq = -vtot .* p.^2;
acq = max(acq,-realmax);

% Transposed output
if transpose_flag; acq = acq'; end


end