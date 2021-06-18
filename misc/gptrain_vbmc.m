function [gp,hypstruct,Ns_gp,optimState] = gptrain_vbmc(hypstruct,optimState,stats,options)
%GPTRAIN_VBMC Train Gaussian process model.

% Initialize HYPSTRUCT if empty
hypfields = {'hyp','warp','logp','full','runcov'};
for f = hypfields
    if ~isfield(hypstruct,f{:}); hypstruct.(f{:}) = []; end
end

% Get training dataset
[X_train,y_train,s2_train,t_train] = get_traindata_vbmc(optimState,options);

% Heuristic fitness shaping (unused)
if options.FitnessShaping
    [y_train,s2_train] = outputwarp_vbmc(X_train,y_train,s2_train,optimState,options);
end

% Get priors, starting hyperparameters, number of samples, and mean function
if optimState.Warmup && options.BOWarmup
    [hypprior,hyp0,Ns_gp,meanfun] = ...
        vbmc_gphyp(optimState,optimState.gpMeanfun,X_train,y_train,0,options);
else
    [hypprior,hyp0,Ns_gp,meanfun] = ...
        vbmc_gphyp(optimState,optimState.gpMeanfun,X_train,y_train,0,options);
end

% Initial GP hyperparameters
if isempty(hypstruct.hyp); hypstruct.hyp = hyp0; end

% Get GP training options
gptrain_options = get_GPTrainOptions(Ns_gp,hypstruct,optimState,stats,options);    
% gptrain_options.LogP = hypstruct.logp;
if numel(gptrain_options.Widths) ~= numel(hyp0); gptrain_options.Widths = []; end

% Build starting points
hyp0 = [];
try
    if gptrain_options.Ninit > 0 && ~isempty(stats)
        for ii = ceil(numel(stats.gp)/2):numel(stats.gp)
            hyp0 = [hyp0, [stats.gp(ii).post.hyp]];
        end
        N0 = size(hyp0,2);
        if N0 > gptrain_options.Ninit/2
            hyp0 = hyp0(:,randperm(N0,floor(gptrain_options.Ninit/2)));
        end
    end
    hyp0 = unique([hyp0, hypstruct.hyp]','rows')';
catch
    hyp0 = hypstruct.hyp;
end

if size(hyp0,1) ~= size(hypprior.mu,2); hyp0 = []; end

if isfield(hypstruct,'hyp_vp') && ~isempty(hypstruct.hyp_vp) ...
    && strcmpi(gptrain_options.Sampler,'npv')
    hyp0 = hypstruct.hyp_vp;
end

% Fit GP to training set
[gp,hypstruct.hyp,gpoutput] = gplite_train(hyp0,Ns_gp,...
    X_train,y_train, ...
    optimState.gpCovfun,meanfun,optimState.gpNoisefun,...
    s2_train,hypprior,gptrain_options);
    
hypstruct.full = gpoutput.hyp_prethin; % Pre-thinning GP hyperparameters
hypstruct.logp = gpoutput.logp;
if isfield(gpoutput,'hyp_vp')
    hypstruct.hyp_vp = gpoutput.hyp_vp;
end

if isfield(gpoutput,'stepsize')
    optimState.gpmala_stepsize = gpoutput.stepsize;
    gpoutput.stepsize
end

gp.t = t_train;

%      if iter > 10
%          pause
%      end

% Update running average of GP hyperparameter covariance (coarse)
if size(hypstruct.full,2) > 1
    hypcov = cov(hypstruct.full');
    if isempty(hypstruct.runcov) || options.HypRunWeight == 0
        hypstruct.runcov = hypcov;
    else
        weight = options.HypRunWeight^options.FunEvalsPerIter;
        hypstruct.runcov = (1-weight)*hypcov + ...
            weight*hypstruct.runcov;
    end
else
    hypstruct.runcov = [];
end

% Sample from GP (for debug)
if ~isempty(gp) && 0
    Xgp = gpsample_vbmc(vp,gp,1e3,1);
    cornerplot(Xgp);
end

% Estimate of GP noise around the top high posterior density region
optimState.sn2hpd = estimate_GPnoise(gp);

end

%--------------------------------------------------------------------------

function [hypprior,hyp0,Ns_gp,meanfun] = vbmc_gphyp(optimState,meanfun,X_train,y_train,warpflag,options)
%VBMC_GPHYP Define bounds, priors and samples for GP hyperparameters.

% Get high-posterior density dataset
X = optimState.X(optimState.X_flag,:);
y = optimState.y(optimState.X_flag);
[X_hpd,y_hpd,hpd_range] = gethpd_vbmc(X,y,options.HPDFrac);
[N_hpd,D] = size(X_hpd);
s2 = [];
neff = optimState.Neff;

%% Set GP hyperparameters defaults for VBMC

% Get number and info for hyperparameters
[Ncov,covinfo] = gplite_covfun('info',X_hpd,optimState.gpCovfun,[],y_hpd);
[Nnoise,noiseinfo] = gplite_noisefun('info',X_hpd,optimState.gpNoisefun,y_hpd,s2);
[Nmean,meaninfo] = gplite_meanfun('info',X_hpd,meanfun,y_hpd);
if ~isempty(optimState.gpOutwarpfun)
    [Noutwarp,outwarpinfo] = optimState.gpOutwarpfun('info',y_hpd);
else
    Noutwarp = 0;
end

meanfun = meaninfo.meanfun;     % Switch to number
Nhyp = Ncov+Nnoise+Nmean+Noutwarp;

% Initial GP hyperparameters

% GP covariance hyperparameters
hyp0 = zeros(Nhyp,1);
hyp0(1:Ncov) = covinfo.x0;

% GP noise hyperparameters
hyp0(Ncov+(1:Nnoise)) = noiseinfo.x0;
MinNoise = options.TolGPNoise;
% MinNoise = max(1e-3,std(y_hpd)*1e-3)
noisemult = [];
switch optimState.UncertaintyHandlingLevel
    case 0
        noisesize = max(options.NoiseSize,MinNoise);
        if isempty(noisesize); noisesize = MinNoise; end
        noisestd = 0.5;
    case 1
        noisemult = max(options.NoiseSize,MinNoise);
        if isempty(noisemult)
            noisemult = 1;  noisemultstd = log(10);
        else
            noisemultstd = log(10)/2;
        end
        noisesize = MinNoise;
        noisestd = log(10);
    case 2
        noisesize = MinNoise;
        noisestd = 0.5;
end
hyp0(Ncov+1) = log(noisesize);
if ~isempty(noisemult); hyp0(Ncov+2) = log(noisemult); end

% GP mean function hyperparameters
hyp0(Ncov+Nnoise+(1:Nmean)) = meaninfo.x0;

% GP output warping function hyperparameters
if Noutwarp > 0; hyp0(Ncov+Nnoise+Nmean+(1:Noutwarp)) = outwarpinfo.x0; end

%% Change default bounds and set priors over hyperparameters
LB_gp = NaN(1,Nhyp);
UB_gp = NaN(1,Nhyp);

if options.UpperGPLengthFactor > 0
    UB_gp(1:D) = log(options.UpperGPLengthFactor*(optimState.PUB - optimState.PLB));  % Max GP input length scale
end
LB_gp(Ncov+1) = log(MinNoise);     % Increase minimum noise

switch meanfun
    case 1
        UB_gp(Ncov+Nnoise+1) = min(y_hpd);    % Lower maximum constant mean
    case {4,10,12,14}
        if options.gpQuadraticMeanBound
            deltay = max(options.TolSD,min(D,max(y_hpd)-min(y_hpd)));
            UB_gp(Ncov+Nnoise+1) = max(y_hpd)+deltay; 
        end
        if meanfun == 14
            UB_gp(Ncov+Nnoise+D+2) = log(1);        % Lower max scaling factor
            LB_gp(Ncov+Nnoise+D+2) = log(1e-3);    % Lower min scaling factor
        end        
    case 6
        hyp0(Ncov+Nnoise+1) = min(y);
        UB_gp(Ncov+Nnoise+1) = min(y_hpd);    % Lower maximum constant mean
    case 8
    case 22
        if options.gpQuadraticMeanBound
            deltay = max(options.TolSD,min(D,max(y_hpd)-min(y_hpd)));
            UB_gp(Ncov+Nnoise+1) = max(y_hpd)+deltay; 
        end        
end


% Set priors over hyperparameters (might want to double-check this)
hypprior = [];
hypprior.mu = NaN(1,Nhyp);
hypprior.sigma = NaN(1,Nhyp);
hypprior.df = 3*ones(1,Nhyp);    % Broad Student's t prior

% Hyperprior over observation noise
hypprior.mu(Ncov+1) = log(noisesize);
hypprior.sigma(Ncov+1) = noisestd;
if ~isempty(noisemult)
    hypprior.mu(Ncov+2) = log(noisemult);
    hypprior.sigma(Ncov+2) = noisemultstd;    
end

% Hyperpriors over mixture of quadratics mean function
if meanfun == 22
    deltay = max(y) - min(y);    
    hypprior.mu(Ncov+Nnoise+2*D+2) = 0;                 % hm
    hypprior.sigma(Ncov+Nnoise+2*D+2) = 0.5*deltay;
    hypprior.mu(Ncov+Nnoise+2*D+3) = 0;                 % rho
    hypprior.sigma(Ncov+Nnoise+2*D+3) = 1;
    hypprior.mu(Ncov+Nnoise+2*D+4) = 0;                 % beta
    hypprior.sigma(Ncov+Nnoise+2*D+4) = 1;    
end

% Change bounds and hyperprior over output-dependent noise modulation
if numel(optimState.gpNoisefun)>2 && optimState.gpNoisefun(3) == 1
    y_all = optimState.y(optimState.X_flag);
    
    UB_gp(Ncov+2) = max(y_all) - 10*D;
    LB_gp(Ncov+2) = min(min(y_all),max(y_all) - 20*D);
    
    %hypprior.mu(Ncov+2) = max(y_hpd) - 10*D;
    %hypprior.sigma(Ncov+2) = 1;
    
    hypprior.mu(Ncov+3) = log(0.01);
    hypprior.sigma(Ncov+3) = log(10);
end

% Priors and bounds for output warping hyperparameters (not used)
if Noutwarp > 0
    outwarp_delta = optimState.OutwarpDelta;
    
    y_all = optimState.y(optimState.X_flag);
    
    switch Noutwarp
        
        case 2
            UB_gp(Ncov+Nnoise+Nmean+1) = max(y_all) - outwarp_delta;
            LB_gp(Ncov+Nnoise+Nmean+1) = min(min(y_all),max(y_all) - 2*outwarp_delta);
            hypprior.mu(Ncov+Nnoise+Nmean+1) = max(y_all) - outwarp_delta;
            hypprior.sigma(Ncov+Nnoise+Nmean+1) = options.OutwarpThreshBase;
            hypprior.df(Ncov+Nnoise+Nmean+1) = 1;   % Half-Cauchy prior
            
            UB_gp(Ncov+Nnoise+Nmean+2) = log(2);
            hypprior.mu(Ncov+Nnoise+Nmean+2) = 0;
            hypprior.sigma(Ncov+Nnoise+Nmean+2) = log(2);
            
        case 3
    
            UB_gp(Ncov+Nnoise+Nmean+1) = max(y_all) - outwarp_delta;
            LB_gp(Ncov+Nnoise+Nmean+1) = min(min(y_all),max(y_all) - 2*outwarp_delta);
            hypprior.mu(Ncov+Nnoise+Nmean+1) = max(y_all) - outwarp_delta;
            hypprior.sigma(Ncov+Nnoise+Nmean+1) = options.OutwarpThreshBase;
            hypprior.df(Ncov+Nnoise+Nmean+1) = 1;   % Half-Cauchy prior

            hypprior.mu(Ncov+Nnoise+Nmean+2) = 0;
            hypprior.sigma(Ncov+Nnoise+Nmean+2) = log(2);
            
            UB_gp(Ncov+Nnoise+Nmean+3) = 0;
            hypprior.mu(Ncov+Nnoise+Nmean+3) = 0;
            hypprior.sigma(Ncov+Nnoise+Nmean+3) = log(2);
        
    end        
    
end

% VBMC used to have an empirical Bayes prior on some GP hyperparameters,
% such as input length scales, based on statistics of the GP training
% inputs. However, this approach could lead to instabilities. From the 2020 
% paper, we switched to a fixed prior based on the plausible bounds.

hypprior.mu(1:D) = log(options.GPLengthPriorMean*(optimState.PUB - optimState.PLB));
hypprior.sigma(1:D) = options.GPLengthPriorStd;

if meanfun == 14
    hypprior.mu(Ncov+Nnoise+D+2) = log(0.1);
    hypprior.sigma(Ncov+Nnoise+D+2) = log(10);
    hypprior.mu(Ncov+Nnoise+D+3) = log(0.1);
    hypprior.sigma(Ncov+Nnoise+D+3) = log(100);
end

hypprior.LB = LB_gp;
hypprior.UB = UB_gp;

if warpflag % Unused
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

    % Maximum sample cutoff
    if optimState.Warmup
        Ns_gp = min(Ns_gp,options.NSgpMaxWarmup);
    else
        Ns_gp = min(Ns_gp,options.NSgpMaxMain);        
    end
    
    % Stop sampling after reaching max number of training points
    if optimState.N >= options.StableGPSampling
        StopSampling = optimState.N;
    end
    
    % Stop sampling after reaching threshold number of variational components
    if optimState.vpK >= options.StableGPvpK
        StopSampling = optimState.N;        
    end
end
if StopSampling > 0
    Ns_gp = options.StableGPSamples;
end

end

%--------------------------------------------------------------------------
function sn2 =  estimate_GPnoise(gp)
%ESTIMATE_GPNOISE Estimate GP observation noise at high posterior density.

HPDTop = 0.2;

[N,D] = size(gp.X);

% Subsample high posterior density dataset
[~,ord] = sort(gp.y,'descend');
N_hpd = ceil(HPDTop*N);
X_hpd = gp.X(ord(1:N_hpd),:);
y_hpd = gp.y(ord(1:N_hpd));
if ~isempty(gp.s2)
    s2_hpd = gp.s2(ord(1:N_hpd));
else
    s2_hpd = [];
end

Ncov = gp.Ncov;
Nnoise = gp.Nnoise;
Ns = numel(gp.post);

sn2 = zeros(size(X_hpd,1),Ns);

for s = 1:Ns
    hyp_noise = gp.post(s).hyp(Ncov+(1:Nnoise));
    sn2(:,s) = gplite_noisefun(hyp_noise,X_hpd,gp.noisefun,y_hpd,s2_hpd);
end

sn2 = median(mean(sn2,2));

end