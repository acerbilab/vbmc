function [vp,elbo,elbo_sd,exitflag,output,stats] = vbmc(fun,x0,LB,UB,PLB,PUB,options,varargin)
%VBMC Posterior and model inference via Variational Bayesian Monte Carlo (v0.7)
%   Documentation to be written -- work in progress.
% 

%--------------------------------------------------------------------------
% VBMC: Variational Bayesian Monte Carlo for posterior and model inference.
% To be used under the terms of the GNU General Public License 
% (http://www.gnu.org/copyleft/gpl.html).
%
%   Author (copyright): Luigi Acerbi, 2018
%   e-mail: luigi.acerbi@{gmail.com,nyu.edu,unige.ch}
%   URL: http://luigiacerbi.com
%   Version: 0.7 (alpha)
%   Release date: May 2, 2018
%   Code repository: https://github.com/lacerbi/vbmc
%--------------------------------------------------------------------------



%% Basic default options
defopts.Display                 = 'iter         % Level of display ("iter", "notify", "final", or "off")';
defopts.MaxIter                 = '20*nvars     % Max number of iterations';
defopts.MaxFunEvals             = '200*nvars    % Max number of objective fcn evaluations';
defopts.NonlinearScaling        = 'on           % Automatic nonlinear rescaling of variables';
defopts.OutputFcn               = '[]           % Output function'; 
defopts.UncertaintyHandling     = '[]           % Explicit noise handling (if empty, determine at runtime)';
defopts.NoiseSize               = '[]           % Base observation noise magnitude';
%defopts.NoiseFinalSamples       = '10           % Samples to estimate FVAL at the end (for noisy objectives)';
defopts.Fvals                   = '[]           % Evaluated fcn values at X0';
defopts.OptimToolbox            = '[]           % Use Optimization Toolbox (if empty, determine at runtime)';
defopts.Diagnostics             = 'on           % Run in diagnostics mode, get additional info';
defopts.ProposalFcn             = '[]           % Weighted proposal fcn for uncertainty search';

%% If called with no arguments or with 'defaults', return default options
if nargout <= 1 && (nargin == 0 || (nargin == 1 && ischar(fun) && strcmpi(fun,'defaults')))
    if nargin < 1
        fprintf('Basic default options returned (type "help vbmc" for help).\n');
    end
    vp = defopts;
    return;
end

%% Advanced options (do not modify unless you *know* what you are doing)
defopts.FunEvalStart       = 'max(D,10)         % Number of initial objective fcn evals';
defopts.FunEvalsPerIter    = '5                 % Number of objective fcn evals per iteration';
defopts.SearchAcqFcn       = '@vbmc_acqfreg     % Fast search acquisition fcn(s)';
defopts.NSsearch           = '2^13              % Samples for fast acquisition fcn eval per new point';
defopts.NSent              = '@(K) 100*K        % Total samples for Monte Carlo approx. of the entropy';
defopts.NSentFast          = '@(K) 100*K        % Total samples for preliminary Monte Carlo approx. of the entropy';
defopts.NSentFine          = '@(K) 2^15*K       % Total samples for refined Monte Carlo approx. of the entropy';
defopts.NSelbo             = '50                % Samples per component for fast approx. of ELBO';
defopts.NSelboIncr         = '0.1               % Multiplier to samples for fast approx. of ELBO for incremental iterations';
defopts.ElboStarts         = '2                 % Starting points to refine optimization of the ELBO';
defopts.NSgpMax            = '80                % Max GP hyperparameter samples (decreases with training points)';
defopts.StableGPSampling   = '200 + 10*nvars    % Force stable GP hyperparameter sampling (reduce samples or start optimizing)';
defopts.StableGPSamples    = '0                 % Number of GP samples when GP is stable (0 = optimize)';
defopts.GPSampleThin       = '5                 % Thinning for GP hyperparameter sampling';
defopts.TolGPVar           = '1e-4              % Threshold on GP variance, used to stabilize sampling and by some acquisition fcns';
defopts.gpMeanFun          = 'negquad           % GP mean function';
defopts.Kfun               = '@sqrt             % Variational components as a function of training points';
defopts.KfunMax            = '@(N) 2*sqrt(N)    % Max variational components as a function of training points';
defopts.Kwarmup            = '2                 % Variational components during warmup';
defopts.AdaptiveK          = '1                 % Added variational components for stable solution';
defopts.HPDFrac            = '0.8               % High Posterior Density region (fraction of training inputs)';
defopts.ELCBOImproWeight   = '3                 % Uncertainty weight on ELCBO for computing lower bound improvement';
defopts.TolLength          = '1e-6              % Minimum fractional length scale';
defopts.NoiseObj           = 'off               % Objective fcn returns noise estimate as 2nd argument (unsupported)';
defopts.CacheSize          = '1e4               % Size of cache for storing fcn evaluations';
defopts.CacheFrac          = '0.5               % Fraction of search points from starting cache (if nonempty)';
defopts.StochasticOptimizer = 'adam             % Stochastic optimizer for varational parameters';
defopts.TolFunStochastic   = '1e-3              % Stopping threshold for stochastic optimization';
defopts.TolSD              = '0.1               % Tolerance on ELBO uncertainty for stopping (iff variational posterior is stable)';
defopts.TolsKL             = '0.01*sqrt(nvars)  % Stopping threshold on change of variational posterior per training point';
defopts.TolStableIters     = '5                 % Number of stable iterations for checking stopping criteria';
defopts.TolStableFunEvals  = '5*nvars           % Number of stable fcn evals for checking stopping criteria';
defopts.TolStableWarmup    = '3                 % Number of stable iterations for stopping warmup';
defopts.TolImprovement     = '0.01              % Required ELCBO improvement per fcn eval before termination';
defopts.KLgauss            = 'yes               % Use Gaussian approximation for symmetrized KL-divergence b\w iters';
defopts.TrueMean           = '[]                % True mean of the target density (for debugging)';
defopts.TrueCov            = '[]                % True covariance of the target density (for debugging)';
defopts.MinFunEvals        = '2*nvars^2         % Min number of fcn evals';
defopts.MinIter            = 'nvars             % Min number of iterations';
defopts.HeavyTailSearchFrac = '0.25               % Fraction of search points from heavy-tailed variational posterior';
defopts.MVNSearchFrac      = '0.25              % Fraction of search points from multivariate normal';
defopts.AlwaysRefitVarPost = 'no                % Always fully refit variational posterior';
defopts.Plot               = 'off               % Show variational posterior triangle plots';
defopts.Warmup             = 'on                % Perform warm-up stage';
defopts.StopWarmupThresh   = '1                 % Stop warm-up when increase in ELBO is confidently below threshold';
defopts.WarmupKeepThreshold = '10*nvars         % Max log-likelihood difference for points kept after warmup';
defopts.SearchCMAES        = 'on                % Use CMA-ES for search';
defopts.MomentsRunWeight   = '0.9               % Weight of previous trials (per trial) for running avg of variational posterior moments';
defopts.GPRetrainThreshold = '1                 % Upper threshold on reliability index for full retraining of GP hyperparameters';
defopts.ELCBOmidpoint      = 'on                % Compute full ELCBO also at best midpoint';
defopts.GPSampleWidths     = '5                 % Multiplier to widths from previous posterior for GP sampling (Inf = do not use previous widths)';
defopts.HypRunWeight       = '0.9               % Weight of previous trials (per trial) for running avg of GP hyperparameter covariance';
defopts.WeightedHypCov     = 'on                % Use weighted hyperparameter posterior covariance';
defopts.TolCovWeight       = '0                 % Minimum weight for weighted hyperparameter posterior covariance';
defopts.GPHypSampler       = 'slicesample       % MCMC sampler for GP hyperparameters';
defopts.CovSampleThresh    = '10                % Switch to covariance sampling below this threshold of stability index';
defopts.DetEntTolOpt       = '1e-3              % Optimality tolerance for optimization of deterministic entropy';
defopts.EntropySwitch      = 'on                % Switch from deterministic entropy to stochastic entropy when reaching stability';
defopts.EntropyForceSwitch = '0.8               % Force switch to stochastic entropy at this fraction of total fcn evals';
defopts.DetEntropyMinD     = '5                 % Start with deterministic entropy only with this number of vars or more';
defopts.TolConLoss         = '0.01              % Fractional tolerance for constraint violation of variational parameters';

%% Advanced options for unsupported/untested features (do *not* modify)
defopts.AcqFcn             = '@vbmc_acqskl       % Expensive acquisition fcn';
defopts.Nacq               = '1                 % Expensive acquisition fcn evals per new point';
defopts.WarpRotoScaling    = 'off               % Rotate and scale input';
%defopts.WarpCovReg         = '@(N) 25/N         % Regularization weight towards diagonal covariance matrix for N training inputs';
defopts.WarpCovReg         = '0                 % Regularization weight towards diagonal covariance matrix for N training inputs';
defopts.WarpNonlinear      = 'off               % Nonlinear input warping';
defopts.WarpEpoch          = '100               % Recalculate warpings after this number of fcn evals';
defopts.WarpMinFun         = '10 + 2*D          % Minimum training points before starting warping';
defopts.WarpNonlinearEpoch = '100               % Recalculate nonlinear warpings after this number of fcn evals';
defopts.WarpNonlinearMinFun = '20 + 5*D         % Minimum training points before starting nonlinear warping';
defopts.ELCBOWeight        = '0                 % Uncertainty weight during ELCBO optimization';
defopts.SearchSampleGP     = 'false             % Generate search candidates sampling from GP surrogate';
defopts.VarParamsBack      = '0                 % Check variational posteriors back to these previous iterations';
defopts.AltMCEntropy       = 'no                % Use alternative Monte Carlo computation for the entropy';


%% If called with 'all', return all default options
if strcmpi(fun,'all')
    vp = defopts;
    return;
end

%% Check that all VBMC subfolders are on the MATLAB path
add2path();

%% Input arguments

if nargin < 3 || isempty(LB); LB = -Inf; end
if nargin < 4 || isempty(UB); UB = Inf; end
if nargin < 5; PLB = []; end
if nargin < 6; PUB = []; end
if nargin < 7; options = []; end

%% Initialize display printing options

if ~isfield(options,'Display') || isempty(options.Display)
    options.Display = defopts.Display;
end

switch lower(options.Display(1:3))
    case {'not','notify','notify-detailed'}
        prnt = 1;
    case {'non','none','off'}
        prnt = 0;
    case {'ite','all','iter','iter-detailed'}
        prnt = 3;
    case {'fin','final','final-detailed'}
        prnt = 2;
    otherwise
        prnt = 1;
end

%% Initialize variables and algorithm structures

if isempty(x0)
    if prnt > 2
        fprintf('X0 not specified. Taking the number of dimensions from PLB and PUB...');
    end
    if isempty(PLB) || isempty(PUB)
        error('vbmc:UnknownDims', ...
            'If no starting point is provided, PLB and PUB need to be specified.');
    end    
    x0 = NaN(size(PLB));
    if prnt > 2
        fprintf(' D = %d.\n', numel(x0));
    end
end

D = size(x0,2);     % Number of variables
optimState = [];

% Check/fix boundaries and starting points
[LB,UB,PLB,PUB] = boundscheck(x0,LB,UB,PLB,PUB,prnt);

% Convert from char to function handles
if ischar(fun); fun = str2func(fun); end

% Setup algorithm options
[options,cmaes_opts] = setupoptions(D,defopts,options);

% Setup and transform variables
K = getK(struct('Neff',options.FunEvalStart,'Warmup',options.Warmup),options);
[vp,optimState] = ...
    setupvars(x0,LB,UB,PLB,PUB,K,optimState,options,prnt);

% Store objective function
optimState.fun = fun;
if isempty(varargin)
    funwrapper = fun;   % No additional function arguments passed
else
    funwrapper = @(u_) fun(u_,varargin{:});
end

% Initialize function logger
[~,optimState] = vbmc_funlogger([],x0(1,:),optimState,'init',options.CacheSize,options.NoiseObj);

% GP struct and GP hyperparameters
gp = [];    hyp = [];   hyp_warp = [];
optimState.gpMeanfun = options.gpMeanFun;
switch optimState.gpMeanfun
    case {'zero','const','negquad','se'}
    otherwise
        error('vbmc:UnknownGPmean', ...
            'Unknown/unsupported GP mean function. Supported mean functions are ''zero'', ''const'', ''negquad'', and ''se''.');
end

if optimState.Cache.active
    displayFormat = ' %5.0f     %5.0f  /%5.0f   %12.2f  %12.2f  %12.2f     %4.0f %10.3g       %s\n';
else
    displayFormat = ' %5.0f       %5.0f    %12.2f  %12.2f  %12.2f     %4.0f %10.3g     %s\n';
end
if prnt > 2
    if optimState.Cache.active
        fprintf(' Iteration f-count/f-cache    Mean[ELBO]     Std[ELBO]     sKL-iter[q]   K[q]  Convergence    Action\n');
        % fprintf(displayFormat,0,0,0,NaN,NaN,NaN,NaN,Inf,'');        
    else
        fprintf(' Iteration   f-count     Mean[ELBO]     Std[ELBO]     sKL-iter[q]   K[q]  Convergence  Action\n');
        % fprintf(displayFormat,0,0,NaN,NaN,NaN,NaN,Inf,'');        
    end
end

%% Variational optimization loop
iter = 0;
isFinished_flag = false;
exitflag = 0;   output = [];    stats = [];     sKL = Inf;

while ~isFinished_flag    
    iter = iter + 1;
    optimState.iter = iter;
    vp_old = vp;
    action = '';
    optimState.redoRotoscaling = false;    
    
    if iter == 1 && optimState.Warmup; action = 'start warm-up'; end
    
    % Switch to stochastic entropy towards the end if still on deterministic
    if optimState.EntropySwitch && ...
            optimState.funccount >= options.EntropyForceSwitch*options.MaxFunEvals
        optimState.EntropySwitch = false;
        if isempty(action); action = 'entropy switch'; else; action = [action ', entropy switch']; end        
    end
    
    %% Actively sample new points into the training set
    t = tic;
    optimState.trinfo = vp.trinfo;
    if iter == 1; new_funevals = options.FunEvalStart; else; new_funevals = options.FunEvalsPerIter; end
    if optimState.Xmax > 0
        optimState.ymax = max(optimState.y(optimState.X_flag));
    end
    if optimState.SkipAdaptiveSampling
        optimState.SkipAdaptiveSampling = false;
    else
        [optimState,t_adapt(iter),t_func(iter)] = ...
            adaptive_sampling(optimState,new_funevals,funwrapper,vp,vp_old,gp,options,cmaes_opts);
    end
    optimState.N = optimState.Xmax;  % Number of training inputs
    optimState.Neff = sum(optimState.X_flag(1:optimState.Xmax));
    timer.activeSampling = toc(t);
    
    %% Input warping / reparameterization (unsupported!)
    if options.WarpNonlinear || options.WarpRotoScaling
        t = tic;
        [optimState,vp,hyp,hyp_warp,action] = ...
            vbmc_warp(optimState,vp,gp,hyp,hyp_warp,action,options,cmaes_opts);
        timer.warping = toc(t);        
    end
        
    %% Train GP
    t = tic;
        
    % Get priors, starting hyperparameters, and number of samples
    [hypprior,X_hpd,y_hpd,~,hyp0,optimState.gpMeanfun,Ns_gp] = ...
        vbmc_gphyp(optimState,optimState.gpMeanfun,0,options);
    if isempty(hyp); hyp = hyp0; end % Initial GP hyperparameters
    if Ns_gp == options.StableGPSamples && optimState.StopSampling == 0
        optimState.StopSampling = optimState.N; % Reached stable sampling
    end
    
    % Get GP training options
    gptrain_options = get_GPTrainOptions(Ns_gp,optimState,stats,options);    
    
    % Fit hyperparameters
    [gp,hyp,gpoutput] = gplite_train(hyp,Ns_gp, ...
        optimState.X(optimState.X_flag,:),optimState.y(optimState.X_flag), ...
        optimState.gpMeanfun,hypprior,[],gptrain_options);
    hyp_full = gpoutput.hyp_prethin; % Pre-thinning GP hyperparameters
    
    % Update running average of GP hyperparameter covariance (coarse)
    if size(hyp_full,2) > 1
        hypcov = cov(hyp_full');
        if isempty(optimState.RunHypCov) || options.HypRunWeight == 0
            optimState.RunHypCov = hypcov;
        else
            weight = options.HypRunWeight^options.FunEvalsPerIter;
            optimState.RunHypCov = (1-weight)*hypcov + ...
                weight*optimState.RunHypCov;
        end
        % optimState.RunHypCov
    else
        optimState.RunHypCov = [];
    end
    
    % Sample from GP (for debug)
    if ~isempty(gp) && 0
        Xgp = vbmc_gpsample(gp,1e3,optimState,1);
        cornerplot(Xgp);
    end
    
    timer.gpTrain = toc(t);
        
    %% Optimize variational parameters
    t = tic;
    
    % Adaptive increase of number of components
    if isa(options.AdaptiveK,'function_handle')
        Kbonus = round(options.AdaptiveK(optimState.vpK));
    else
        Kbonus = round(double(options.AdaptiveK));
    end     
    [Kmin,Kmax] = getK(optimState,options);
    Knew = optimState.vpK;
    Knew = max(Knew,Kmin);
    if sKL < options.TolsKL*options.FunEvalsPerIter
        Knew = optimState.vpK + Kbonus;
    end
    Knew = min(Knew,Kmax);

    % Decide number of fast/slow optimizations
    if optimState.RecomputeVarPost || options.AlwaysRefitVarPost
        Nfastopts = options.NSelbo * vp.K;
        Nslowopts = options.ElboStarts; % Full optimizations
        useEntropyApprox = true;
        optimState.RecomputeVarPost = false;
    else
        % Only incremental change from previous iteration
        Nfastopts = ceil(options.NSelbo * vp.K * options.NSelboIncr);
        Nslowopts = 1;
        useEntropyApprox = false;
    end
    
    % Run optimization of variational parameters
    [vp,elbo,elbo_sd,varss] = ...
        vpoptimize(Nfastopts,Nslowopts,useEntropyApprox,vp,gp,Knew,X_hpd,y_hpd,optimState,stats,options,cmaes_opts,prnt);
    optimState.vpK = vp.K;
            
    timer.variationalFit = toc(t);
    
    %% Recompute warpings at end iteration (unsupported)
    if options.WarpNonlinear || options.WarpRotoScaling    
        [optimState,vp,hyp] = ...
            vbmc_rewarp(optimState,vp,gp,hyp,options,cmaes_opts);
    end
    
    %% Plot current iteration (to be improved)
    if options.Plot
        
        if D == 1
            hold off;
            gplite_plot(gp);
            hold on;
            xlims = xlim;
            xx = linspace(xlims(1),xlims(2),1e3)';
            yy = vbmc_pdf(xx,vp,false,true);
            hold on;
            plot(xx,yy+elbo,':');
            drawnow;
            
        else
            Xrnd = vbmc_rnd(1e5,vp,1,1);
            X_train = gp.X;
            if ~isempty(vp.trinfo); X_train = warpvars(X_train,'inv',vp.trinfo); end
            try
                for i = 1:D; names{i} = ['x_{' num2str(i) '}']; end
                [~,ax] = cornerplot(Xrnd,names);
                for i = 1:D-1
                    for j = i+1:D
                        axes(ax(j,i));  hold on;
                        scatter(X_train(:,i),X_train(:,j),'ok');
                    end
                end
                drawnow;            
            catch
                % pause
            end            
        end
    end    
    
    %mubar
    %Sigma
    
    %----------------------------------------------------------------------
    %% Finalize iteration
    t = tic;
    
    % Compute symmetrized KL-divergence between old and new posteriors
    Nkl = 1e5;
    sKL = max(0,0.5*sum(vbmc_kldiv(vp,vp_old,Nkl,options.KLgauss,1)));
    
    % Compare variational posterior's moments with ground truth
    if ~isempty(options.TrueMean) && ~isempty(options.TrueCov) ...
        && all(isfinite(options.TrueMean(:))) ...
        && all(isfinite(options.TrueCov(:)))
    
        [mubar_orig,Sigma_orig] = vbmc_moments(vp,1,1e6);
        [kl(1),kl(2)] = mvnkl(mubar_orig,Sigma_orig,options.TrueMean,options.TrueCov);
        sKL_true = 0.5*sum(kl)
    else
        sKL_true = [];
    end
    
    % Record moments in transformed space
    [mubar,Sigma] = vbmc_moments(vp,0);
    if isempty(optimState.RunMean) || isempty(optimState.RunCov)
        optimState.RunMean = mubar;
        optimState.RunCov = Sigma;        
        optimState.LastRunAvg = optimState.N;
        % optimState.RunCorrection = 1;
    else
        Nnew = optimState.N - optimState.LastRunAvg;
        wRun = options.MomentsRunWeight^Nnew;
        optimState.RunMean = wRun*optimState.RunMean + (1-wRun)*mubar;        
        optimState.RunCov = wRun*optimState.RunCov + (1-wRun)*Sigma;
        optimState.LastRunAvg = optimState.N;
        % optimState.RunT = optimState.RunT + 1;
    end
        
    % Check if we are still warming-up
    if optimState.Warmup && iter > 1    
        [optimState,action] = vbmc_warmup(optimState,stats,action,elbo,elbo_sd,options);
    end    

    % t_fits(iter) = toc(timer_fits);    
    % dt = (t_adapt(iter)+t_fits(iter))/new_funevals;
    
    timer.finalize = toc(t);
    
    % timer
    
    % Record all useful stats
    stats = savestats(stats, ...
        optimState,vp,elbo,elbo_sd,varss,sKL,sKL_true,gp,hyp_full,Ns_gp,timer,options.Diagnostics);
    
    %----------------------------------------------------------------------
    %% Check termination conditions    

    [optimState,stats,isFinished_flag,exitflag,action] = ...
        vbmc_termination(optimState,action,stats,options);
    
    %% Write output
    
    % Stopped GP sampling this iteration?
    if Ns_gp == options.StableGPSamples && ...
            stats.gpNsamples(max(1,iter-1)) > options.StableGPSamples
        if isempty(action); action = 'stable GP sampling'; else; action = [action ', stable GP sampling']; end
    end    
    
    % Write iteration
    if optimState.Cache.active
        fprintf(displayFormat,iter,optimState.funccount,optimState.cachecount,elbo,elbo_sd,sKL,vp.K,optimState.R,action);
    else
        fprintf(displayFormat,iter,optimState.funccount,elbo,elbo_sd,sKL,vp.K,optimState.R,action);
    end    
    
%     if optimState.iter > 10 && stats.elboSD(optimState.iter-1) < 0.1 && stats.elboSD(optimState.iter) > 10
%         fprintf('\nmmmh\n');        
%     end
    
end

if nargout > 3
    output = optimState;    
end

if nargout > 4
    % Remove full GP hyperparameter samples from stats unless diagnostic run
    if ~options.Diagnostics
        stats = rmfield(stats,'gpHypFull');
    end
end


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function stats = savestats(stats,optimState,vp,elbo,elbo_sd,varss,sKL,sKL_true,gp,hyp_full,Ns_gp,timer,debugflag)

iter = optimState.iter;
stats.iter(iter) = iter;
stats.N(iter) = optimState.N;
stats.Neff(iter) = optimState.Neff;
stats.funccount(iter) = optimState.funccount;
stats.cachecount(iter) = optimState.cachecount;
stats.vpK(iter) = vp.K;
stats.elbo(iter) = elbo;
stats.elboSD(iter) = elbo_sd;
stats.sKL(iter) = sKL;
if ~isempty(sKL_true)
    stats.sKL_true = sKL_true;
end
stats.gpSampleVar(iter) = varss;
stats.gpNsamples(iter) = Ns_gp;
stats.gpHypFull{iter} = hyp_full;
stats.timer(iter) = timer;

if debugflag
    stats.vp(iter) = vp;
    stats.gp(iter) = gplite_clean(gp);
end

end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Kmin,Kmax] = getK(optimState,options)
%GETK Get number of variational components.

Neff = optimState.Neff;
Kfun = options.Kfun;
Kfun_max = options.KfunMax;

if optimState.Warmup
    Kmin = options.Kwarmup;
    Kmax = options.Kwarmup;
else
    if isnumeric(Kfun)
        Kmin = Kfun;
    elseif isa(Kfun,'function_handle')
        Kmin = Kfun(Neff);
    end
    if isnumeric(Kfun_max)
        Kmax = Kfun_max;
    elseif isa(Kfun_max,'function_handle')
        Kmax = Kfun_max(Neff);
    end
    
    Kmin = min(Neff,max(1,round(Kmin)));
    Kmax = max(Kmin,min(Neff,max(1,round(Kmax))));
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function add2path()
%ADD2PATH Adds VBMC subfolders to MATLAB path.

subfolders = {'acq','gplite','misc','utils','warp'};
pathCell = regexp(path, pathsep, 'split');
baseFolder = fileparts(mfilename('fullpath'));

onPath = true;
for iFolder = 1:numel(subfolders)
    folder = [baseFolder,filesep,subfolders{iFolder}];    
    if ispc  % Windows is not case-sensitive
      onPath = onPath & any(strcmpi(folder, pathCell));
    else
      onPath = onPath & any(strcmp(folder, pathCell));
    end
end

% ADDPATH is slow, call it only if folders are not on path
if ~onPath
    addpath(genpath(fileparts(mfilename('fullpath'))));
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TO-DO list:
% - Write a private quantile function to avoid calls to Stats Toolbox.
% - Fix call to fmincon if Optimization Toolbox is not available.
% - Check that I am not using other ToolBoxes by mistake.
