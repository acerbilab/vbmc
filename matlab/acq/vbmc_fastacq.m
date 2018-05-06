function acq = vbmc_fastacq(Xs,vp,vp_old,gp,G,vardiagG,metrics_flags,transpose_flag)
%VBMC_FASTACQ Compute uncertainty-based fast acquisition functions.

% Xs is in *transformed* coordinates

if nargin < 7 || isempty(metrics_flags); metrics_flags = true(1,10); end
if nargin < 8 || isempty(transpose_flag); transpose_flag = false; end

% Transposed input (useful for CMAES)
if transpose_flag; Xs = Xs'; end

[N,D] = size(Xs);    % Number of points and dimension

% Probability density of variational posterior at test points
p = vbmc_pdf(Xs,vp,0);
p2 = p.^2;  % Squared density

% GP mean and variance for each hyperparameter sample
[~,~,fmu,fs2] = gplite_pred(gp,Xs,[],1);

Ns = size(fmu,2);
fbar = sum(fmu,2)/Ns;   % Mean across samples
vbar = sum(fs2,2)/Ns;   % Average variance across samples
if Ns > 1; vf = sum((fmu - fbar).^2,2)/(Ns-1); else; vf = 0; end  % Sample variance
vtot = vf + vbar;       % Total variance

acq = Inf(N,numel(metrics_flags));  % Prepare acquisition function

% Negative expected integrand variance at point
if metrics_flags(1)
    acq(:,1) = -vtot .* p2;
end

% Generalized negative expected variance at point, alpha = 1
if metrics_flags(2)
    acq(:,2) = -vtot .* p;
end

% Generalized negative expected variance at point, alpha = 1/2
if metrics_flags(3)
    acq(:,3) = -vtot .* sqrt(p);
end

% Upper/lower confidence bounds
if metrics_flags(4) || metrics_flags(5)    
    % By default use beta_t from Theorem 1 in Srinivas et al. (2010)
    delta = 0.1;
    nu = 0.2;           % Empirical correction
    t = size(gp.X,1);   % Number of training points
    sqrtbetat = sqrt(nu*2*log(D*t^2*pi^2/(6*delta)));

    if metrics_flags(4)
        % Upper confidence bound
        acq(:,4) = (-fbar - sqrtbetat * sqrt(vtot)) .* p;
    end

    if metrics_flags(5)
        % Lower confidence bound
        acq(:,5) = (fbar - sqrtbetat * sqrt(vtot)) .* p;
    end
end

% Deviations from integral mean
if metrics_flags(6)
    acq(:,6) = -((fbar - G).^2 .* vtot) .* p2;
end

% Variance-weighted negative K-L divergence old-to-new variational posterior
if metrics_flags(7)
    pback = max(vbmc_pdf(Xs,vp,1,0,1),realmin);
    qback = max(vbmc_pdf(Xs,vp_old,1,0,1),realmin);
    acq(:,7) = pback .* log(qback./pback) .* vtot;
end

% Only keep requested metrics
acq = acq(:,logical(metrics_flags));

% Transposed output
if transpose_flag; acq = acq'; end


end