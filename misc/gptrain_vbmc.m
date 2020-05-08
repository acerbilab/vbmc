function [gp,hypstruct,Ns_gp,optimState] = gptrain_vbmc(hypstruct,optimState,stats,options)
%GPTRAIN_VBMC Train Gaussian process model.

if options.DoubleGP
    
    options.NSgpMaxWarmup = 0;
    options.NSgpMaxMain = 0;
    options.DoubleGP = false;
    
    [gp,hypstruct,Ns_gp,optimState] = gptrain_vbmc(hypstruct,optimState,stats,options);
    
    if isfield(optimState,'hypstruct2')
        hypstruct2 = optimState.hypstruct2;
    else
        hypstruct2 = [];        
    end
    
    optimState2 = optimState;
    optimState2.gpMeanfun = 12;  % Constant mean function
    [gp2,hypstruct2] = gptrain_vbmc(hypstruct2,optimState2,stats,options);
    
    optimState.hypstruct2 = hypstruct2;
    
    z1 = gplite_nlZ(gp.post.hyp,gp);
    z2 = gplite_nlZ(gp2.post.hyp,gp2);    
    H1 = hessian(@(x) gplite_nlZ(x(:),gp), gp.post.hyp'); 
    H2 = hessian(@(x) gplite_nlZ(x(:),gp2), gp2.post.hyp');
    N = size(gp.X,1); 
    dd = [numel(gp.post.hyp),numel(gp2.post.hyp)]; 
    % yy = -[z1,z2] - 0.5*2*dd; 
    yy = -[z1,z2] + 0.5*dd*log(2*pi) - 0.5*log(abs([det(H1),det(H2)]))
    
    D = size(gp.X,2);
    gp.post(2) = gp.post(1);
    switch gp2.meanfun
        case 1
            gp.post(2).hyp(1:gp.Ncov+gp.Nnoise+1) = gp2.post.hyp(1:gp.Ncov+gp.Nnoise+1);
            gp.post(2).hyp((gp.Ncov+gp.Nnoise+1)+(1:D)) = 0;
            gp.post(2).hyp((gp.Ncov+gp.Nnoise+1+D)+(1:D)) = Inf;
        case 12
            [~,idx] = max(gp.y);
            gp.post(2).hyp(1:gp.Ncov+gp.Nnoise+1) = gp2.post.hyp(1:gp.Ncov+gp.Nnoise+1);
            gp.post(2).hyp((gp.Ncov+gp.Nnoise+1)+(1:D)) = gp.X(idx,:);
            gp.post(2).hyp((gp.Ncov+gp.Nnoise+1+D)+(1:D)) = gp2.post.hyp((1:D)+(gp.Ncov+gp.Nnoise+1));
    end
    
    gp = gplite_post(gp);
    
    return;
end





% Initialize HYPSTRUCT if empty
hypfields = {'hyp','warp','logp','full','runcov'};
for f = hypfields
    if ~isfield(hypstruct,f{:}); hypstruct.(f{:}) = []; end
end

% Get training dataset
[X_train,y_train,s2_train,t_train] = get_traindata_vbmc(optimState,options);

% Heuristic fitness shaping
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

if gp.intmeanfun == 0 && options.IntegrateGPMean
    gp_intmean = trainintmeangp_vbmc(gp,optimState,stats,options);    
    errorflag = check_quadcoefficients_vbmc(gp_intmean);
    if ~errorflag; gp = gp_intmean; end
    
elseif gp.intmeanfun == 3 || gp.intmeanfun == 4    

    % Check for posite-definiteness of negative quadratic basis function
    errorflag = check_quadcoefficients_vbmc(gp);

    % If the coefficients are not negative, redo the fit
    if errorflag
        optimState_temp = optimState;
        optimState.gpMeanfun = 18;
        optimState.intMeanfun = 1;
        [gp,hypstruct,Ns_gp,optimState] = gptrain_vbmc([],optimState,stats,options);
        optimState.gpMeanfun = optimState_temp.gpMeanfun;
        optimState.intMeanfun = optimState_temp.intMeanfun;
        hypstruct.hyp = [];
        return;
    end
end
    
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

% Priors and bounds for output warping hyperparameters
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


% Empirical Bayes hyperprior on GP hyperparameters
if options.EmpiricalGPPrior
    hypprior.mu(1:D) = log(std(X_hpd));
    hypprior.sigma(1:D) = max(2,log(hpd_range) - log(std(X_hpd)));
    
    %hypprior.mu(D+1) = log(std(y_hpd)) + 0.25*D*log(2*pi);
    %hypprior.sigma(D+1) = log(10);
    switch meanfun
        case 1
            hypprior.mu(Ncov+Nnoise+1) = quantile(y_hpd,0.25);
            hypprior.sigma(Ncov+Nnoise+1) = 0.5*(max(y_hpd)-min(y_hpd));
        case 4
            hypprior.mu(Ncov+Nnoise+1) = max(y_hpd);
            hypprior.sigma(Ncov+Nnoise+1) = max(y_hpd)-min(y_hpd);

            sigma_omega = options.AnnealedGPMean(neff,optimState.MaxFunEvals);
            if sigma_omega > 0 && isfinite(sigma_omega)
                hypprior.mu(Ncov+Nnoise+1+D+(1:D)) = log(hpd_range);
                hypprior.sigma(Ncov+Nnoise+1+D+(1:D)) = sigma_omega;
            end

            if options.ConstrainedGPMean
                hypprior.mu(Ncov+Nnoise+1) = NaN;
                hypprior.sigma(Ncov+Nnoise+1) = NaN;

                hypprior.mu(Ncov+Nnoise+1+(1:D)) = 0.5*(optimState.PUB + optimState.PLB);
                hypprior.sigma(Ncov+Nnoise+1+(1:D)) = 0.5*(optimState.PUB - optimState.PLB);

                hypprior.mu(Ncov+Nnoise+1+D+(1:D)) = log(0.5*(optimState.PUB - optimState.PLB));
                hypprior.sigma(Ncov+Nnoise+1+D+(1:D)) = 0.01;            
            end                    

        case 6
            hypprior.mu(Ncov+Nnoise+1) = min(y) - std(y_hpd);
            hypprior.sigma(Ncov+Nnoise+1) = std(y_hpd);
            
        case 8
            
    end
    
else
    
    if numel(options.GPLengthPriorMean) == 1    
        hypprior.mu(1:D) = log(options.GPLengthPriorMean*(optimState.PUB - optimState.PLB));
    elseif numel(options.GPLengthPriorMean) == 2
        lpmu = log(options.GPLengthPriorMean);
        pmu = exp(rand()*(lpmu(2)-lpmu(1)) + lpmu(1)); 
        hypprior.mu(1:D) = log(pmu*(optimState.PUB - optimState.PLB));        
    end
    hypprior.sigma(1:D) = options.GPLengthPriorStd;
    
%      switch meanfun
%          case {4,6}
%              hypprior.mu(Ncov+Nnoise+1+D+(1:D)) = log(0.5*(optimState.PUB - optimState.PLB));
%              hypprior.sigma(Ncov+Nnoise+1+D+(1:D)) = log(100);            
%      end
    
    if meanfun == 14
        hypprior.mu(Ncov+Nnoise+D+2) = log(0.1);
        hypprior.sigma(Ncov+Nnoise+D+2) = log(10);
        hypprior.mu(Ncov+Nnoise+D+3) = log(0.1);
        hypprior.sigma(Ncov+Nnoise+D+3) = log(100);
    end

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


%% Integrated mean function

if optimState.intMeanfun > 0
    H = gplite_intmeanfun(zeros(1,D),optimState.intMeanfun);
    Nb = size(H,1);    
    bb = zeros(1,Nb);   % Prior mean over basis function coefficients
    BB = Inf(1,Nb);     % Prior variance over basis function coefficients
    if optimState.intMeanfun == 1
        bb(1) = max(y_train);
        BB(1) = 1e3;
    else
        bb(1) = max(y_train);
        BB(1) = 1e3;            
    end
    if optimState.intMeanfun > 1
        bb(1+(1:D)) = 0;
        BB(1+(1:D)) = 1e3;
    end
    if optimState.intMeanfun > 2
        bb(1+D+(1:D)) = 0;
        BB(1+D+(1:D)) = 1e3;            
    end
    if optimState.intMeanfun > 3
        bb(1+2*D+(1:D*(D-1)/2)) = 0;
        BB(1+2*D+(1:D*(D-1)/2)) = 1e4;
    end
    
    if isfield(optimState,'betabarmap')
        bb = optimState.betabarmap;
        BB = optimState.betabarvar;
        hyp0 = optimState.hypbeta;
    end
    
    % Fix variance for under-determined training set
    if size(X_train,1) <= Nb + Nhyp; BB(isinf(BB)) = 1e3; end
        
    meanfun = {meanfun,optimState.intMeanfun,bb,BB};
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
