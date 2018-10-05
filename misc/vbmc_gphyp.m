function [hypprior,X_hpd,y_hpd,Nhyp,hyp0,meanfun,Ns_gp] = vbmc_gphyp(optimState,meanfun,warpflag,options)
%VBMC_GPHYP Define bounds, priors and samples for GP hyperparameters.

%% High-posterior density dataset

% Compute transformed training dataset
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
hpd_range = max(X_hpd)-min(X_hpd);

neff = optimState.Neff;

%% GP observation noise

MinNoise = 1e-3;
if options.UncertaintyHandling
    if isempty(options.NoiseSize)
        noisesize = 1;
        noisestd = 1;
    else
        noisesize = options.NoiseSize;
        noisestd = 0.5;
    end
else
    if isempty(options.NoiseSize)
        noisesize = MinNoise; % This was 0.01
    else
        noisesize = options.NoiseSize;
    end
    noisestd = 0.5;
end
noisesize = max(noisesize,MinNoise);

%% Set GP hyperparameters defaults for VBMC

% Get number of hyperparameters
Ncov = D+1;
[Nmean,meaninfo] = gplite_meanfun([],X_hpd,meanfun,y_hpd);
meanfun = meaninfo.meanfun;     % Switch to number
Nhyp = Ncov+1+Nmean;

% Initial GP hyperparameters
hyp0 = zeros(Nhyp,1);
hyp0(1:D) = log(std(X_hpd))';
hyp0(D+1) = log(std(y_hpd));
hyp0(Ncov+1) = log(noisesize);
hyp0(Ncov+2:Ncov+1+Nmean) = meaninfo.x0;

% Change some of the default bounds over hyperparameters
LB_gp = NaN(1,Nhyp);
UB_gp = NaN(1,Nhyp);
LB_gp(Ncov+1) = log(MinNoise);     % Increase minimum noise

switch meanfun
    case 1
        UB_gp(Ncov+2) = min(y_hpd);    % Lower maximum constant mean
end        

% Set priors over hyperparameters (might want to double-check this)
hypprior = [];
hypprior.mu = NaN(1,Nhyp);
hypprior.sigma = NaN(1,Nhyp);
hypprior.df = 3*ones(1,Nhyp);    % Broad Student's t prior
hypprior.mu(1:D) = log(std(X_hpd));
hypprior.sigma(1:D) = max(2,log(hpd_range) - log(std(X_hpd)));
%hypprior.mu(D+1) = log(std(y_hpd)) + 0.25*D*log(2*pi);
%hypprior.sigma(D+1) = log(10);
hypprior.mu(Ncov+1) = log(noisesize);
hypprior.sigma(Ncov+1) = noisestd;
switch meanfun
    case 1
        hypprior.mu(Ncov+2) = quantile1(y,0.25);
        hypprior.sigma(Ncov+2) = 0.5*(max(y)-min(y));
    case 4
        hypprior.mu(Ncov+2) = max(y_hpd);
        hypprior.sigma(Ncov+2) = max(y_hpd)-min(y_hpd);
        
        sigma_omega = options.AnnealedGPMean(neff,optimState.MaxFunEvals)
        if sigma_omega > 0 && isfinite(sigma_omega)
            hypprior.mu(Ncov+2+D+(1:D)) = log(hpd_range);
            hypprior.sigma(Ncov+2+D+(1:D)) = sigma_omega;
        end
    case 6
        hypprior.mu(Ncov+2) = median(y);
        hypprior.sigma(Ncov+2) = 0.5*(max(y)-min(y));
end

hypprior.LB = LB_gp;
hypprior.UB = UB_gp;

if warpflag
    warning('Warping priors need fixing; need to fill in mean function priors.');
    hyp0 = [hyp0;zeros(2*D,1)];    
    hypprior.mu = [hypprior.mu, zeros(1,2*D)];
    hypprior.sigma = [hypprior.sigma, 0.01*ones(1,2*D)]; % Prior for no or little warping
    hypprior.df = [hypprior.df, 3*ones(1,2*D)];    % Heavy tails - prior can be overridden
    LB_warp = -5*ones(1,2*D);
    UB_warp = 5*ones(1,2*D);
    hypprior.LB = [hypprior.LB, LB_warp];
    hypprior.UB = [hypprior.UB, UB_warp];
end

%% Number of GP hyperparameter samples

StopSampling = optimState.StopSampling;

% Check whether to perform hyperparameter sampling or optimization
if StopSampling == 0
    % Number of samples
    Ns_gp = round(options.NSgpMax/sqrt(optimState.N));

    % Maximum sample cutoff during warm-up
    if optimState.Warmup
        MaxWarmupGPSamples = ceil(options.NSgpMax/10);
        Ns_gp = min(Ns_gp,MaxWarmupGPSamples);
    end

    % Stop sampling after reaching max number of training points
    if optimState.N >= options.StableGPSampling
        StopSampling = optimState.N;
    end
end
if StopSampling > 0
    Ns_gp = options.StableGPSamples;
end
