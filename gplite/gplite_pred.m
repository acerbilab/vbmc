function [ymu,ys2,fmu,fs2,lp] = gplite_pred(gp,Xstar,ystar,ssflag)
%GPLITE_PRED Prediction for lite Gaussian Processes regression.

% HYP is a column vector. Multiple columns correspond to multiple samples.
if nargin < 3; ystar = []; end
if nargin < 4 || isempty(ssflag); ssflag = false; end

[N,D] = size(gp.X);            % Number of training points and dimension
Ns = numel(gp.post);           % Hyperparameter samples
Nstar = size(Xstar,1);         % Number of test inputs

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
Nmean = gp.Nmean;

% Stripped-down representation
if ~isfield(gp.post(1),'alpha')
    
end

% Loop over hyperparameter samples
for s = 1:Ns
    hyp = gp.post(s).hyp;

    alpha = gp.post(s).alpha;
    L = gp.post(s).L;
    Lchol = gp.post(s).Lchol;
    sW = gp.post(s).sW;
    sn2_mult = gp.post(s).sn2_mult;    

    % Extract GP hyperparameters from HYP
    ell = exp(hyp(1:D));
    sf2 = exp(2*hyp(D+1));
    sn2 = exp(2*hyp(D+2));
    
    hyp_mean = hyp(Ncov+2:Ncov+1+Nmean);                % Get mean function hyperparameters
    mstar = gplite_meanfun(hyp_mean,Xstar,gp.meanfun);  % GP mean evaluated at test points

    % Compute cross-kernel matrix Ks_mat
    Ks_mat = sq_dist(diag(1./ell)*gp.X',diag(1./ell)*Xstar');
    Ks_mat = sf2 * exp(-Ks_mat/2);
    kss = sf2 * ones(Nstar,1);        % Self-covariance vector

    fmu(:,s) = mstar + Ks_mat'*alpha;            % Conditional mean
    ymu(:,s) = fmu(:,s);                     % observed function mean
    if nargout > 1
        if Lchol
            V = L'\(repmat(sW,[1,Nstar]).*Ks_mat);
            fs2(:,s) = kss - sum(V.*V,1)';       % predictive variances
        else
            LKs = L*Ks_mat;
            fs2(:,s) = kss + sum(Ks_mat.*LKs,1)';
        end            
        fs2(:,s) = max(fs2(:,s),0);          % remove numerical noise i.e. negative variances
        ys2(:,s) = fs2(:,s) + sn2*sn2_mult;           % observed variance

        % Compute log probability of test inputs
        if ~isempty(ystar) && nargout > 4
            lp(:,s) = -(ystar-ymu).^2./(sn2*sn2_mult)/2-log(2*pi*sn2*sn2_mult)/2;
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
