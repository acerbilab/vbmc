function [hypprior,X_hpd,y_hpd,Nhyp,hyp0,meanfun] = vbmc_gphyp(optimState,meanfun,warpflag,options)
%VBMC_GPHYP Define bounds and priors over GP hyperparameters.

X_orig = optimState.X_orig(optimState.X_flag,:);
y_orig = optimState.y_orig(optimState.X_flag);

[N,D] = size(X_orig);
X = warpvars(X_orig,'d',optimState.trinfo);
y = y_orig + warpvars(X,'logp',optimState.trinfo);

% Subsample high posterior density dataset
[~,ord] = sort(y,'descend');
N_hpd = round(options.HPDFrac*N);
X_hpd = X(ord(1:N_hpd),:);
y_hpd = y(ord(1:N_hpd));

% Get number of hyperparameters
Ncov = D+1;
[Nmean,meaninfo] = gplite_meanfun([],X_hpd,meanfun,y_hpd);
meanfun = meaninfo.meanfun;     % Switch to number
Nhyp = Ncov+1+Nmean;

% Initial GP hyperparameters
hyp0 = zeros(Nhyp,1);
hyp0(1:D) = log(std(X_hpd))';
hyp0(D+1) = log(std(y_hpd));
hyp0(Ncov+1) = log(1e-2);
hyp0(Ncov+2:Ncov+1+Nmean) = meaninfo.x0;

% Change some of the default bounds over hyperparameters
LB_gp = NaN(1,Nhyp);
UB_gp = NaN(1,Nhyp);
LB_gp(Ncov+1) = log(1e-3);     % Increase minimum noise
if ~options.QuadraticMean
    UB_gp(Ncov+2) = min(y_hpd);    % Lower maximum constant mean
end

% Set priors over hyperparameters
hypprior = [];
hypprior.mu = NaN(1,Nhyp);
hypprior.sigma = NaN(1,Nhyp);
hypprior.df = 3*ones(1,Nhyp);    % Broad Student's t prior
noisesize = 0.01;
hypprior.mu(1:D) = log(std(X_hpd));
hypprior.sigma(1:D) = max(2,log(max(X_hpd)-min(X_hpd)) - log(std(X_hpd)));
%hypprior.mu(D+1) = log(std(y_hpd));
%hypprior.sigma(D+1) = 2;
hypprior.mu(Ncov+1) = log(noisesize);
hypprior.sigma(Ncov+1) = 0.5;
if options.QuadraticMean
    hypprior.mu(Ncov+2) = max(y_hpd);
    hypprior.sigma(Ncov+2) = max(y_hpd)-min(y_hpd);
end

hypprior.LB = LB_gp;
hypprior.UB = UB_gp;

if warpflag
    hyp0 = [hyp0;zeros(2*D,1)];    
    hypprior.mu = [hypprior.mu, zeros(1,2*D)];
    hypprior.sigma = [hypprior.sigma, 0.01*ones(1,2*D)]; % Prior for no or little warping
    hypprior.df = [hypprior.df, 3*ones(1,2*D)];    % Heavy tails - prior can be overridden
    LB_warp = -5*ones(1,2*D);
    UB_warp = 5*ones(1,2*D);
    hypprior.LB = [hypprior.LB, LB_warp];
    hypprior.UB = [hypprior.UB, UB_warp];
end