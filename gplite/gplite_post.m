function gp = gplite_post(hyp,X,y,covfun,meanfun,noisefun,s2,update1,outwarpfun)
%GPLITE_POST Compute posterior GP for a given training set.
%   GP = GPLITE_POST(HYP,X,Y,S2,MEANFUN) computes the posterior GP for a vector
%   of hyperparameters HYP and a given training set. HYP is a column vector 
%   of hyperparameters (see below). X is a N-by-D matrix of training inputs 
%   and Y a N-by-1 vector of training targets (function values at X).
%   MEANFUN is the GP mean function (see GPLITE_MEANFUN.M for a list).
%
%   GP = GPLITE_POST(HYP,X,Y,S2,MEANFUN) computes the posterior GP for a vector
%   of hyperparameters HYP and a given training set. HYP is a column vector 
%   of hyperparameters (see below). X is a N-by-D matrix of training inputs 
%   and Y a N-by-1 vector of training targets (function values at X).
%   MEANFUN is the GP mean function (see GPLITE_MEANFUN.M for a list).
%
%   GP = GPLITE_POST(GP,XSTAR,YSTAR,[],1) performs a fast rank-1 update for 
%   a GPLITE structure, given a single new observation at XSTAR with observed
%   value YSTAR.
%
%   Note that the returned GP contains auxiliary structs for faster
%   computations. To save memory, call GPLITE_CLEAN.
%
%   See also GPLITE_CLEAN, GPLITE_MEANFUN.

if nargin < 4; covfun = []; end
if nargin < 5; meanfun = []; end
if nargin < 6; noisefun = []; end
if nargin < 7; s2 = []; end
if nargin < 8 || isempty(update1); update1 = false; end
if nargin < 9; outwarpfun = []; end

gp = [];
if isstruct(hyp)
    gp = hyp;
    if nargin < 2; X = gp.X; end
    if nargin < 3; y = gp.y; end
    if nargin < 4; covfun = gp.covfun; end
    if nargin < 5; meanfun = gp.meanfun; end
    if nargin < 6; noisefun = gp.noisefun; end
    if nargin < 7; s2 = gp.s2; end
    if nargin < 9
        if isfield(gp,'outwarpfun'); outwarpfun = gp.outwarpfun; else; outwarpfun = []; end
    end
end

if update1
    if size(X,1) > 1
        error('gplite_post:NotRankOne', ...
          'GPLITE_POST with this input format only supports rank-one updates.');
    end
    if isempty(gp)
        error('gplite_post:NoGP', ...
          'GPLITE_POST can perform rank-one update only with an existing GP struct.');        
    end
    if ~isempty(covfun)
        warning('gplite_post:RankOneCovFunction', ...
            'No need to specify a GP covariance function when performing a rank-one update.');
    end
    if ~isempty(meanfun)
        warning('gplite_post:RankOneMeanFunction', ...
            'No need to specify a GP mean function when performing a rank-one update.');
    end
    if ~isempty(noisefun)
        warning('gplite_post:RankOneNoiseFunction', ...
            'No need to specify a GP noise function when performing a rank-one update.');
    end
end

% Create GP struct
if isempty(gp)
    gp.X = X;
    gp.y = y;
    gp.s2 = s2;
    [N,D] = size(X);            % Number of training points and dimension
    [Nhyp,Ns] = size(hyp);      % Number of hyperparameters and samples
    if isempty(covfun); covfun = 1; end
    if isempty(meanfun); meanfun = 1; end
    if isempty(noisefun)
        if isempty(s2); noisefun = [1 0 0]; else; noisefun = [1 1 0]; end
    end
    % Get number and field of covariance / noise / mean function
    [gp.Ncov,info] = gplite_covfun('info',X,covfun);
    gp.covfun = info.covfun;
    [gp.Nnoise,info] = gplite_noisefun('info',X,noisefun);
    gp.noisefun = info.noisefun;
    [gp.Nmean,info] = gplite_meanfun('info',X,meanfun);
    gp.meanfun = info.meanfun;
    
    % Output warping function (optional)
    if ~isempty(outwarpfun)
        [Noutwarp,info] = outwarpfun('info',y);
        gp.Noutwarp = Noutwarp;
        gp.outwarpfun = info.outwarpfun;
    else
        Noutwarp = 0;
    end
        
    % Create posterior structure
    postfields = {'hyp','alpha','sW','L','sn2_mult','Lchol'};
    for i = 1:numel(postfields); post.(postfields{i}) = []; end    
    for s = 1:size(hyp,2)
        gp.post(s) = post;
        gp.post(s).hyp = hyp(:,s);
    end
        
    if isempty(hyp) || isempty(y); return; end
    if Nhyp ~= gp.Ncov+gp.Nnoise+gp.Nmean+Noutwarp
        error('gplite_post:dimmismatch', ...
            'Number of hyperparameters mismatched with GP model specification.');
    end
else
    [N,D] = size(gp.X);         % Number of training points and dimension
    Ns = numel(gp.post);        % Hyperparameter samples    
end

if ~update1 || 1
    % Loop over hyperparameter samples
    for s = 1:Ns
        hyp = gp.post(s).hyp;
        [~,~,gp.post(s)] = gplite_core(hyp,gp,0,0);        
    end    
else
    % Perform rank-1 update of the GP posterior        
    Ncov = gp.Ncov;
    Nnoise = gp.Nnoise;
    
    % Added training input
    xstar = X;
    ystar = y;
    s2star = s2;
    
    % Rank-1 update for the same XSTAR but potentially different ystars
    Nstar = numel(ystar);

    if Nstar > 1; gp(2:Nstar) = gp(1); end
    
    % Compute prediction for all samples
    [mstar,vstar] = gplite_pred(gp(1),xstar,ystar,s2star,1);
        
    % Loop over hyperparameter samples
    for s = 1:Ns
        
        hyp = gp(1).post(s).hyp;
        
        hyp_noise = hyp(Ncov+1:Ncov+Nnoise); % Get noise hyperparameters
        sn2 = gplite_noisefun(hyp_noise,xstar,gp.noisefun,ystar,s2star);
        sn2_eff = sn2*gp(1).post(s).sn2_mult;            
        
        % Compute covariance and cross-covariance
        if gp.covfun(1) == 1    % Hard-coded SE-ard for speed
            ell = exp(hyp(1:D));
            sf2 = exp(2*hyp(D+1));        
            K = sf2;
            Ks_mat = sq_dist(diag(1./ell)*gp(1).X',diag(1./ell)*xstar');
            Ks_mat = sf2 * exp(-Ks_mat/2);
        else
            hyp_cov = hyp(1:Ncov);
            K = gplite_covfun(hyp_cov,xstar,gp.covfun,'diag');
            Ks_mat = gplite_covfun(hyp_cov,gp.X,gp.covfun,xstar);            
        end            
        
        L = gp(1).post(s).L;
        Lchol = gp(1).post(s).Lchol;
        
        if Lchol        % high-noise parameterization
            alpha_update = (L\(L'\Ks_mat)) / sn2_eff;
            new_L_column = linsolve(L, Ks_mat, ...
                                struct('UT', true, 'TRANSA', true)) / sn2_eff;
            gp(1).post(s).L = [L, new_L_column; ...
                               zeros(1, size(L, 1)), ...
                               sqrt(1 + K / sn2_eff - new_L_column' * new_L_column)];
        else            % low-noise parameterization
            alpha_update = -L * Ks_mat;
            v = -alpha_update / vstar(:,s);
            gp(1).post(s).L = [L + v * alpha_update', -v; -v', -1 / vstar(:,s)];
        end
        
        gp(1).post(s).sW = [gp(1).post(s).sW; 1/sqrt(sn2_eff)];

        for iStar = 2:Nstar
            gp(iStar).post(s) = gp(1).post(s);
        end
        
        % alpha_update now contains (K + \sigma^2 I) \ k*
        for iStar = 1:Nstar
            gp(iStar).post(s).alpha = ...
                [gp(iStar).post(s).alpha; 0] + ...
                (mstar(:,s) - ystar(iStar)) / vstar(:,s) * [alpha_update; -1];
        end
    end
    
    % Add single input to training set
    for iStar = 1:Nstar
        gp(iStar).X = [gp(iStar).X; xstar];
        gp(iStar).y = [gp(iStar).y; ystar(iStar)];
        if ~isempty(s2star)
            gp(iStar).s2 = [gp(iStar).s2; s2star(iStar)];
        end
    end
end