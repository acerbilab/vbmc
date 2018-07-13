function gp = gplite_post(hyp,X,y,meanfun,update1)
%GPLITE_POST Compute posterior GP for a given training set.
%   GP = GPLITE_POST(HYP,X,Y,MEANFUN) computes the posterior GP for a vector
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

if nargin < 5 || isempty(update1); update1 = false; end

gp = [];
if isstruct(hyp)
    gp = hyp;
    if nargin < 2; X = gp.X; end
    if nargin < 3; y = gp.y; end
    if nargin < 4; meanfun = gp.meanfun; end
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
    if ~isempty(meanfun)
        warning('gplite_post:RankOneMeanFunction', ...
            'No need to specify a GP mean function when performin a rank-one update.');
    end
end

% Create GP struct
if isempty(gp)
    gp.X = X;
    gp.y = y;
    gp.post = [];
    [N,D] = size(X);            % Number of training points and dimension
    [Nhyp,Ns] = size(hyp);      % Number of hyperparameters and samples
    gp.meanfun = meanfun;    
    for s = 1:size(hyp,2); gp.post(s).hyp = hyp(:,s); end
    gp.Ncov = D+1;                 % Number of covariance function hyperparameters
    gp.Nmean = gplite_meanfun([],X,meanfun);
    if Nhyp ~= gp.Ncov+1+gp.Nmean
        error('gplite_post:dimmismatch', ...
            'Number of hyperparameters mismatched with dimension of training inputs.');
    end
else
    [N,D] = size(gp.X);         % Number of training points and dimension
    Ns = numel(gp.post);        % Hyperparameter samples
end

Ncov = gp.Ncov;
Nmean = gp.Nmean;


if ~update1
        
    % Loop over hyperparameter samples
    for s = 1:Ns
        hyp = gp.post(s).hyp;

        % Extract GP hyperparameters from HYP
        ln_ell = hyp(1:D);
        ell = exp(ln_ell);
        ln_sf = hyp(D+1);
        % sf2 = exp(2*ln_sf);
        sn2 = exp(2*hyp(Ncov+1));
        sn2_mult = 1;  % Effective noise variance multiplier

        nf = 1 / (2*pi)^(D/2) * exp(2*ln_sf - sum(ln_ell));  % Kernel normalization factor
        
        % Evaluate mean function on training inputs
        hyp_mean = hyp(Ncov+2:Ncov+1+Nmean); % Get mean function hyperparameters        
        m = gplite_meanfun(hyp_mean,X,gp.meanfun);
        
        % Compute kernel matrix K_mat
        K_mat = sq_dist(diag(1./ell)*X');
        K_mat = nf * exp(-K_mat/2);

        if sn2 < 1e-6   % Different representation depending on noise size
            for iter = 1:10     % Cholesky decomposition until it works
                [L,p] = chol(K_mat+sn2*sn2_mult*eye(N));
                if p > 0; sn2_mult = sn2_mult*10; else; break; end
            end
            sl = 1;
            pL = -L\(L'\eye(N));    % L = -inv(K+inv(sW^2))
            Lchol = 0;         % Tiny noise representation
        else
            
            for iter = 1:10
                [L,p] = chol(K_mat/(sn2*sn2_mult)+eye(N));
                if p > 0; sn2_mult = sn2_mult*10; else; break; end
            end
            sl = sn2*sn2_mult;
            pL = L;                 % L = chol(eye(n)+sW*sW'.*K)
            Lchol = 1;
        end
        
        alpha = L\(L'\(y-m)) ./ sl;     % alpha = inv(K_mat + sn2.*eye(N)) * (y - m)
        
        % GP posterior parameters
        gp.post(s).alpha = alpha;
        gp.post(s).sW = ones(N,1)/sqrt(sn2*sn2_mult);   % sqrt of noise precision vector
        gp.post(s).L = pL;
        gp.post(s).sn2_mult = sn2_mult;
        gp.post(s).Lchol = Lchol;
    end
    
else
    % Added training input
    xstar = X;
    ystar = y;
    
    % Rank-1 update for the same XSTAR but different ystars
    Nstar = numel(ystar);

    if Nstar > 1; gp(2:Nstar) = gp(1); end
    
    % Compute prediction for all samples
    [mstar,vstar] = gplite_pred(gp(1),xstar,[],1);
        
    % Loop over hyperparameter samples
    for s = 1:Ns
        
        hyp = gp(1).post(s).hyp;

        % Extract GP hyperparameters from HYP
        ln_ell = hyp(1:D);
        ell = exp(ln_ell);
        ln_sf = hyp(D+1);
        % sf2 = exp(2*ln_sf);
        sn2 = exp(2*hyp(Ncov+1));
        sn2_eff = sn2*gp(1).post(s).sn2_mult;            

        nf = 1 / (2*pi)^(D/2) * exp(2*ln_sf - sum(ln_ell));  % Kernel normalization factor
        
        % Compute covariance and cross-covariance
        K = nf;
        Ks_mat = sq_dist(diag(1./ell)*gp(1).X',diag(1./ell)*xstar');
        Ks_mat = nf * exp(-Ks_mat/2);    
        
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
        
        gp(1).post(s).sW = [gp(1).post(s).sW; gp(1).post(s).sW(1)];

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
    end
end