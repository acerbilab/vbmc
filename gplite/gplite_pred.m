function [ymu,ys2,fmu,fs2,lp] = gplite_pred(gp,Xstar,ystar,s2star,ssflag,nowarpflag)
%GPLITE_PRED Prediction for lite Gaussian Processes regression.
% Should make GPLITE_QPRED for quantile prediction.

% HYP is a column vector. Multiple columns correspond to multiple samples.
if nargin < 3; ystar = []; end
if nargin < 4; s2star = []; end
if nargin < 5 || isempty(ssflag); ssflag = false; end
if nargin < 6 || isempty(nowarpflag); nowarpflag = false; end

[N,D] = size(gp.X);            % Number of training points and dimension
Ns = numel(gp.post);           % Hyperparameter samples
Nstar = size(Xstar,1);         % Number of test inputs

% Perform dimensionality checks
if ~isempty(ystar) && size(ystar,1) ~= Nstar
    error('gplite_pred:ydimmismatch', ...
        'YSTAR should be empty or a column vector of NSTAR observations.');
end
if ~isempty(s2star) && size(s2star,1) ~= Nstar
    error('gplite_pred:s2dimmismatch', ...
        'S2STAR should be empty or a column vector of NSTAR estimated variances.');
end

% Preallocate space
fmu = zeros(Nstar,Ns);
ymu = zeros(Nstar,Ns);
if nargout > 1
    fs2 = zeros(Nstar,Ns);
    ys2 = zeros(Nstar,Ns);    
end
if ~isempty(ystar) && nargout > 4
    lp = zeros(Nstar,Ns);
else
    lp = [];
end

Ncov = gp.Ncov;
Nnoise = gp.Nnoise;
Nmean = gp.Nmean;

% Output warping function
outwarp_flag = isfield(gp,'outwarpfun') && ~isempty(gp.outwarpfun) && ~nowarpflag;
if outwarp_flag
    Noutwarp = gp.Noutwarp;
    fmu_prewarp = zeros(Nstar,Ns);
else
    Noutwarp = 0;
end

% Loop over hyperparameter samples
for s = 1:Ns
    hyp = gp.post(s).hyp;

    alpha = gp.post(s).alpha;
    L = gp.post(s).L;
    Lchol = gp.post(s).Lchol;
    sW = gp.post(s).sW;
    sn2_mult = gp.post(s).sn2_mult;    
    
    % Get observation noise hyperpameters and evaluate noise at test points
    hyp_noise = hyp(Ncov+1:Ncov+Nnoise);
    sn2_star = gplite_noisefun(hyp_noise,Xstar,gp.noisefun,ystar,s2star);
    
    % Get mean function hyperpameters and evaluate GP mean at test points
    hyp_mean = hyp(Ncov+Nnoise+1:Ncov+Nnoise+Nmean);
    mstar = gplite_meanfun(hyp_mean,Xstar,gp.meanfun);
    
    % Compute cross-kernel matrix Ks_mat
    if gp.covfun(1) == 1    % Hard-coded SE-ard for speed
        ell = exp(hyp(1:D));
        sf2 = exp(2*hyp(D+1));
        Ks_mat = sq_dist(diag(1./ell)*gp.X',diag(1./ell)*Xstar');
        Ks_mat = sf2 * exp(-Ks_mat/2);
        kss = sf2 * ones(Nstar,1);        % Self-covariance vector
    else
        hyp_cov = hyp(1:Ncov);
        Ks_mat = gplite_covfun(hyp_cov,gp.X,gp.covfun,Xstar);            
        kss = gplite_covfun(hyp_cov,Xstar,gp.covfun,'diag');
    end
    
    if N > 0
        fmu(:,s) = mstar + Ks_mat'*alpha;            % Conditional mean
    else
        fmu(:,s) = mstar;
    end
    ymu(:,s) = fmu(:,s);                     % observed function mean
    if nargout > 1
        if N > 0
            if Lchol
                V = L'\(repmat(sW,[1,Nstar]).*Ks_mat);
                fs2(:,s) = kss - sum(V.*V,1)';       % predictive variances
            else
                LKs = L*Ks_mat;
                fs2(:,s) = kss + sum(Ks_mat.*LKs,1)';
            end
            fs2(:,s) = max(fs2(:,s),0);          % remove numerical noise i.e. negative variances
        else
            fs2(:,s) = kss;
        end
        ys2(:,s) = fs2(:,s) + sn2_star*sn2_mult;           % observed variance

        % Compute log probability of test inputs
        if ~isempty(ystar) && nargout > 4
            if outwarp_flag; error('output warping unsupported'); end
            lp(:,s) = -(ystar-ymu(:,s)).^2./(sn2_star*sn2_mult)/2-log(2*pi*sn2_star*sn2_mult)/2;
        end
    end
    
    % Adjust predictions for output-warped GP
    if outwarp_flag
        hyp_outwarp = hyp(Ncov+Nnoise+Nmean+1:Ncov+Nnoise+Nmean+Noutwarp);
        
        fmu_prewarp(:,s) = fmu(:,s);        
        fmu(:,s) = gp.outwarpfun(hyp_outwarp,fmu_prewarp(:,s),'inv');
        ymu(:,s) = fmu_prewarp(:,s);
        if nargout > 1
            
            [~,dwarp_dt] = gp.outwarpfun(hyp_outwarp,fmu(:,s));            
            fs2(:,s) = fs2(:,s)./dwarp_dt.^2;
            ys2(:,s) = ys2(:,s)./dwarp_dt.^2;
            % The problem is that the sample variance explodes for multiple
            % samples (because the predictive means can be very far apart)
                
            if nargout > 4
                lp(:,s) = lp(:,s) + log(abs(dwarp_dt));                
            end
        end
    end

end

% Unless predictions for samples are requested separately, average over samples
if Ns > 1 && ~ssflag
    fbar = sum(fmu,2)/Ns;
    ybar = sum(ymu,2)/Ns;
    if nargout > 1        
        vf = sum((fmu - fbar).^2,2)/(Ns-1);         % Sample variance
        fs2 = sum(fs2,2)/Ns + vf;
        vy = sum((ymu - ybar).^2,2)/(Ns-1);         % Sample variance
        ys2 = sum(ys2,2)/Ns + vy;
    end
    fmu = fbar;
    ymu = ybar;
end
