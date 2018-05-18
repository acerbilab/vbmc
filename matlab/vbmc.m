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

% TO-DO list:
% - Write a private quantile function to avoid calls to Stats Toolbox.
% - Fix call to fmincon if Optimization Toolbox is not available.
% - Check that I am not using other ToolBoxes by mistake.


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
defopts.AcqFcn             = '@vbmc_acqskl       % Expensive acquisition fcn';
defopts.SearchAcqFcn       = '@vbmc_acqpropf    % Fast search acquisition fcn(s)';
defopts.Nacq               = '1                 % Expensive acquisition fcn evals per new point';
defopts.NSsearch           = '2^13              % Samples for fast acquisition fcn eval per new point';
defopts.NSent              = '100               % Samples per component for fast Monte Carlo approx. of the entropy';
defopts.NSentFine          = '2^15              % Samples per component for refined Monte Carlo approx. of the entropy';
defopts.NSelbo             = '50                % Samples per component for fast approx. of ELBO';
defopts.ElboStarts         = '2                 % Starting points to refine optimization of the ELBO';
defopts.NSgpMax            = '80                % Max GP hyperparameter samples (decreases with training points)';
defopts.StopGPSampling     = '200 + 10*nvars    % Stop GP hyperparameter sampling (start optimizing)';
defopts.TolGPVar           = '1e-4              % Stop GP hyperparameter sampling if sample variance is below this threshold per fcn';
defopts.QuadraticMean      = 'yes               % Use GP with quadratic mean function (otherwise constant)';
defopts.Kfun               = '@sqrt             % Variational components as a function of training points';
defopts.KfunMax            = '@(N) 2*sqrt(N)    % Max variational components as a function of training points';
defopts.Kwarmup            = '2                 % Variational components during warmup';
defopts.AdaptiveK          = 'no                % Adaptive number of variational components';
defopts.HPDFrac            = '0.5               % High Posterior Density region (fraction of training inputs)';
defopts.WarpRotoScaling    = 'off               % Rotate and scale input';
%defopts.WarpCovReg         = '@(N) 25/N         % Regularization weight towards diagonal covariance matrix for N training inputs';
defopts.WarpCovReg         = '0                 % Regularization weight towards diagonal covariance matrix for N training inputs';
defopts.WarpNonlinear      = 'off               % Nonlinear input warping';
defopts.WarpEpoch          = '20 + 10*D         % Recalculate warpings after this number of fcn evals';
defopts.WarpMinFun         = '10 + 2*D          % Minimum training points before starting warping';
defopts.WarpNonlinearEpoch = '100               % Recalculate nonlinear warpings after this number of fcn evals';
defopts.WarpNonlinearMinFun = '20 + 5*D         % Minimum training points before starting nonlinear warping';
defopts.ELCBOWeight        = '0                 % Uncertainty weight during ELCBO optimization';
defopts.TolLength          = '1e-6              % Minimum fractional length scale';
defopts.NoiseObj           = 'off               % Objective fcn returns noise estimate as 2nd argument (unsupported)';
defopts.CacheSize          = '1e4               % Size of cache for storing fcn evaluations';
defopts.CacheFrac          = '0.5               % Fraction of search points from starting cache (if nonempty)';
defopts.TolFunAdam         = '0.001             % Stopping threshold for Adam optimizer';
defopts.TolSD              = '0.1               % Tolerance on ELBO uncertainty for stopping (iff variational posterior is stable)';
defopts.TolsKL             = '0.01*sqrt(nvars)  % Stopping threshold on change of variational posterior per training point';
defopts.TolStableIters     = '5                 % Number of stable iterations for checking stopping criteria';
defopts.TolStableFunEvals  = '5*nvars           % Number of stable fcn evals for checking stopping criteria';
defopts.TolStableWarmup    = '3                 % Number of stable iterations for stopping warmup';
defopts.KLgauss            = 'yes               % Use Gaussian approximation for symmetrized KL-divergence b\w iters';
defopts.TrueMean           = '[]                % True mean of the target density (for debugging)';
defopts.TrueCov            = '[]                % True covariance of the target density (for debugging)';
defopts.MinFunEvals        = '2*nvars^2         % Min number of fcn evals';
defopts.MinIter            = 'nvars             % Min number of iterations';
defopts.HeavyTailSearchFrac = '0.25               % Fraction of search points from heavy-tailed variational posterior';
defopts.MVNSearchFrac      = '0.25              % Fraction of search points from multivariate normal';
defopts.SearchSampleGP     = 'false             % Generate search candidates sampling from GP surrogate';
defopts.AlwaysRefitVarPost = 'no                % Always fully refit variational posterior';
defopts.VarParamsBack      = '0                 % Check variational posteriors back to these previous iterations';
defopts.Plot               = 'off               % Show variational posterior triangle plots';
defopts.Warmup             = 'on                % Perform warm-up stage';
defopts.StopWarmupThresh   = '1                 % Stop warm-up when increase in ELBO is confidently below threshold';
defopts.WarmupKeepThreshold = '10*nvars         % Max log-likelihood difference for points kept after warmup';
defopts.SearchCMAES        = 'on                % Use CMA-ES for search';
defopts.MomentsRunWeight   = '0.9               % Weight of previous trials (per trial) for running avg of variational posterior moments';

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

% Check correctness of boundaries and starting points
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

% GP hyperparameters
hyp = [];   hyp_warp = [];  gp = [];
if options.QuadraticMean
    optimState.gpMeanfun = 'negquad';
else
    optimState.gpMeanfun = 'const';
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
            
    % Nonlinear warping iteration? (only after burn-in)
    isNonlinearWarping = (optimState.N - optimState.LastNonlinearWarping) >= options.WarpNonlinearEpoch ...
        && (options.MaxFunEvals - optimState.N) >= options.WarpNonlinearEpoch ...
        && optimState.N >= options.WarpNonlinearMinFun ...
        && options.WarpNonlinear ...
        && ~optimState.Warmup;
    if isNonlinearWarping
        optimState.LastNonlinearWarping = optimState.N;
        optimState.WarpingNonlinearCount = optimState.WarpingNonlinearCount + 1;
        if isempty(action); action = 'warp'; else; action = [action ', warp']; end
    end
    
    % Rotoscaling iteration? (only after burn-in)
    t = tic;
    isRotoscaling = (optimState.N - optimState.LastWarping) >= options.WarpEpoch ...
        && (optimState.N - optimState.LastNonlinearWarping) >= options.WarpEpoch ...
        && (options.MaxFunEvals - optimState.N) >= options.WarpEpoch ...
        && optimState.N >= options.WarpMinFun ...
        && options.WarpRotoScaling ...
        && ~optimState.Warmup;
    if isRotoscaling
        optimState.LastWarping = optimState.N;
        optimState.WarpingCount = optimState.WarpingCount + 1;
        if isempty(action); action = 'rotoscale'; else; action = [action ', rotoscale']; end
    end
    
    %% Update stretching of unbounded variables
    if any(isinf(LB) & isinf(UB)) && ...
            options.WarpNonlinear && (isNonlinearWarping || iter == 1)
        [vp,optimState,hyp] = ...
            warp_unbounded(vp,optimState,hyp,gp,cmaes_opts,options);
        optimState = ResetRunAvg(optimState);
    end

    %%  Rotate and rescale variables
    if options.WarpRotoScaling && isRotoscaling
        [vp,optimState,hyp] = ...
            warp_rotoscaling(vp,optimState,hyp,gp,cmaes_opts,options);
        optimState = ResetRunAvg(optimState);
    end
    
    %% Learn nonlinear warping via GP
    if options.WarpNonlinear && isNonlinearWarping
        [vp,optimState,hyp,hyp_warp] = ...
            warp_nonlinear(vp,optimState,hyp,hyp_warp,cmaes_opts,options);
        optimState.redoRotoscaling = true;
        optimState = ResetRunAvg(optimState);
    end
    
%     if optimState.DoRotoscaling && isWarping
%         [~,X_hpd,y_hpd] = ...
%             vbmc_gphyp(optimState,optimState.gpMeanfun,0,options);            
%         vp = vpoptimize(Nfastopts,1,0,vp,gp,vp.K,X_hpd,y_hpd,optimState,stats,options);
%         [vp,optimState,hyp] = ...
%             warp_rotoscaling(vp,optimState,hyp,gp,cmaes_opts,options);
%         optimState.DoRotoscaling = false;
%     end
    
    timer.warping = toc(t);
    
    %% Train GP
    t = tic;
    
    % Check whether to perform hyperparameter sampling or optimization
    if optimState.StopSampling == 0
        % Number of samples
        Ns_gp = round(options.NSgpMax/sqrt(optimState.N));
        
        % Maximum sample cutoff during warm-up
        if optimState.Warmup
            MaxWarmupGPSamples = ceil(options.NSgpMax/10);
            Ns_gp = min(Ns_gp,MaxWarmupGPSamples);
        end
        
        % Stop sampling after reaching max number of training points
        if optimState.N >= options.StopGPSampling
            optimState.StopSampling = optimState.N;
        end
                
        if optimState.StopSampling > 0
            if isempty(action); action = 'stop GP sampling'; else; action = [action ', stop GP sampling']; end
        end
    end
    if optimState.StopSampling > 0
        Ns_gp = 0;
    end
    
    % Get priors and starting hyperparameters
    [hypprior,X_hpd,y_hpd,~,hyp0,optimState.gpMeanfun] = ...
        vbmc_gphyp(optimState,optimState.gpMeanfun,0,options);
    if isempty(hyp); hyp = hyp0; end % Initial GP hyperparameters
    gptrain_options.Thin = 5;    
    if optimState.RecomputeVarPost
        gptrain_options.Burnin = gptrain_options.Thin*Ns_gp;
    else
        gptrain_options.Burnin = gptrain_options.Thin*3;
    end
    if Ns_gp > 0; gptrain_options.Nopts = 1; else; gptrain_options.Nopts = 2; end
    
    [gp,hyp] = gplite_train(hyp,Ns_gp, ...
        optimState.X(optimState.X_flag,:),optimState.y(optimState.X_flag), ...
        optimState.gpMeanfun,hypprior,[],gptrain_options);
        
    % Sample from GP
    if ~isempty(gp) && 0
        Xgp = vbmc_gpsample(gp,1e3,optimState,1);
        cornerplot(Xgp);
    end
    
    timer.gpTrain = toc(t);
        
    %% Optimize variational parameters
    t = tic;
    
    % Adaptive increase of number of components
    if options.AdaptiveK
        [Kmin,Kmax] = getK(optimState,options);        
        Knew = optimState.vpK;
        Knew = max(Knew,Kmin);
        if sKL < options.TolsKL*options.FunEvalsPerIter
            Knew = optimState.vpK + 1;
        end
        Knew = min(Knew,Kmax);
    else
        Knew = getK(optimState,options);
    end

    if optimState.RecomputeVarPost || options.AlwaysRefitVarPost
        Nfastopts = options.NSelbo * vp.K;
        Nslowopts = options.ElboStarts; % Full optimizations
        useEntropyApprox = true;
        optimState.RecomputeVarPost = false;
    else
        % Only incremental change
        Nfastopts = ceil(options.NSelbo * vp.K / 10);
        Nslowopts = 1;
        useEntropyApprox = false;
    end
    
    [vp,elbo,elbo_sd,varss] = ...
        vpoptimize(Nfastopts,Nslowopts,useEntropyApprox,vp,gp,Knew,X_hpd,y_hpd,optimState,stats,options);
    optimState.vpK = vp.K;
    
    %%  Redo rotoscaling at the end
    if options.WarpRotoScaling && optimState.redoRotoscaling
        [vp,optimState,hyp] = ...
            warp_rotoscaling(vp,optimState,hyp,gp,cmaes_opts,options);
        
        % Get priors and starting hyperparameters
        [hypprior,X_hpd,y_hpd,~,hyp0,optimState.gpMeanfun] = ...
            vbmc_gphyp(optimState,optimState.gpMeanfun,0,options);
        gptrain_options.Nopts = 1;

        [gp,hyp] = gplite_train(hyp,Ns_gp, ...
            optimState.X(optimState.X_flag,:),optimState.y(optimState.X_flag), ...
            optimState.gpMeanfun,hypprior,[],gptrain_options);
        
         vp = vpoptimize(Nfastopts,1,0,vp,gp,vp.K,X_hpd,y_hpd,optimState,stats,options);
    end
    
    if options.Plot
        xx = vbmc_rnd(1e5,vp,1,1);
        try
            cornerplot(xx,[],[],[]);
        catch
            % pause
        end        
    end    
    
    %mubar
    %Sigma
        
    timer.variationalFit = toc(t);
    
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
        
    % Check if we are still warming-up (95% confidence)
    if optimState.Warmup && iter > 1
        elbo_old = stats.elbo(iter-1);
        elboSD_old = stats.elboSD(iter-1);
        increaseUCB = elbo - elbo_old + 1.6449*sqrt(elbo_sd^2 + elboSD_old^2);
        if increaseUCB < options.StopWarmupThresh
            optimState.WarmupStableIter = optimState.WarmupStableIter + 1;
        else
            optimState.WarmupStableIter = 0;
        end        
        if optimState.WarmupStableIter >= options.TolStableWarmup            
            optimState.Warmup = false;
            if isempty(action); action = 'end warm-up'; else; action = [action ', end warm-up']; end
            
            % Remove warm-up points from training set unless close to max
            ymax = max(optimState.y_orig(1:optimState.Xmax));
            NkeepMin = 4*D; 
            idx_keep = (ymax - optimState.y_orig) < options.WarmupKeepThreshold;
            if sum(idx_keep) < NkeepMin
                [~,ord] = sort(optimState.y_orig,'descend');
                idx_keep(ord(1:min(NkeepMin,optimState.Xmax))) = true;
            end
            optimState.X_flag = idx_keep & optimState.X_flag;
            
            % Start warping
            optimState.LastWarping = optimState.N;
            optimState.LastNonlinearWarping = optimState.N;
            
            % Skip adaptive sampling for next iteration
            optimState.SkipAdaptiveSampling = true;
            
            % Fully recompute variational posterior
            optimState.RecomputeVarPost = true;
        end
    end
    

    % t_fits(iter) = toc(timer_fits);    
    % dt = (t_adapt(iter)+t_fits(iter))/new_funevals;
    
    timer.finalize = toc(t);
    
    % timer
    
    % Record all useful stats
    stats = savestats(stats,optimState,vp,elbo,elbo_sd,varss,sKL,sKL_true,gp,Ns_gp,timer,options.Diagnostics);
    
    %----------------------------------------------------------------------
    %% Check termination conditions    
        
    % Maximum number of new function evaluations
    if optimState.funccount >= options.MaxFunEvals
        isFinished_flag = true;
        exitflag = 1;
        % msg = 'Optimization terminated: reached maximum number of function evaluations OPTIONS.MaxFunEvals.';
    end

    % Maximum number of iterations
    if iter >= options.MaxIter
        isFinished_flag = true;
        exitflag = 1;
        % msg = 'Optimization terminated: reached maximum number of iterations OPTIONS.MaxIter.';
    end

    % Reached stable variational posterior with stable ELBO and low uncertainty
    [idx_stable,dN,dN_last,w] = getStableIter(stats,optimState,options);
    if ~isempty(idx_stable)
        sKL_list = stats.sKL;
        elbo_list = stats.elbo;
        
        err2 = sum((elbo_list(idx_stable:iter) - mean(elbo_list(idx_stable:iter))).^2);

        wmean = sum(w.*elbo_list(idx_stable:iter));
        wvar = sum(w.*(elbo_list(idx_stable:iter) - wmean).^2) / (1 - sum(w.^2));
        
        qindex(1) = sqrt(wvar / (options.TolSD^2));
        qindex(2) = sum(w .* stats.elboSD(idx_stable:iter).^2) / options.TolSD^2;
        qindex(3) = sum(w .* sKL_list(idx_stable:iter)) / options.TolsKL;
        
%        qindex
        
%         qindex(1) = sqrt(err2 / (options.TolSD^2*dN));
%         qindex(2) = stats.elboSD(iter) / options.TolSD;
%         qindex(3) = sum(sKL_list(idx_stable:iter)) / (options.TolsKL*dN);
%         qindex(4) = sKL_list(iter) / (options.TolsKL*dN_last);        
        if all(qindex < 1)
            isFinished_flag = true;
            exitflag = 0;
                % msg = 'Optimization terminated: reached maximum number of iterations OPTIONS.MaxIter.';
        end
        qindex = mean(qindex);
        stats.qindex(iter) = qindex;
        
        % Stop sampling after sample variance has stabilized below ToL
        if ~isempty(idx_stable) && optimState.StopSampling == 0 && ~optimState.Warmup
            varss_list = stats.gpSampleVar;
            if sum(w.*varss_list(idx_stable:iter)) < options.TolGPVar
                optimState.StopSampling = optimState.N;
                if isempty(action); action = 'stop GP sampling'; else; action = [action ', stop GP sampling']; end
            end
        end
        
        
    else
        qindex = Inf;
    end
    
    optimState.R = qindex;
    
    % Prevent early termination
    if optimState.N < options.MinFunEvals || optimState.iter < options.MinIter
        isFinished_flag = false;
    end
    
    % Write iteration
    if optimState.Cache.active
        fprintf(displayFormat,iter,optimState.funccount,optimState.cachecount,elbo,elbo_sd,sKL,vp.K,qindex,action);
    else
        fprintf(displayFormat,iter,optimState.funccount,elbo,elbo_sd,sKL,vp.K,qindex,action);
    end    
    
end

if nargout > 3
    output = optimState;
end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function stats = savestats(stats,optimState,vp,elbo,elbo_sd,varss,sKL,sKL_true,gp,Ns_gp,timer,debugflag)

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

% subfolders = {'acq','gpdef','gpml_fast','init','poll','search','utils','warp','gpml-matlab-v3.6-2015-07-07'};
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
function [idx_stable,dN,dN_last,w] = getStableIter(stats,optimState,options)
%GETSTABLEITER Find index of starting stable iteration.

iter = optimState.iter;
idx_stable = [];
dN = [];    dN_last = [];   w = [];

if optimState.iter < 3; return; end

if ~isempty(stats)
    iter_list = stats.iter;
    N_list = stats.N;
    idx_stable = find(N_list <= optimState.N - options.TolStableFunEvals & ...
        iter_list <= iter - options.TolStableIters,1,'last');
    idx_stable = 1;
    if ~isempty(idx_stable)
        dN = optimState.N - N_list(idx_stable);
        dN_last = N_list(end) - N_list(end-1);
    end
    
    % Compute weighting function
    Nw = numel(idx_stable:iter);    
    w1 = zeros(1,Nw);
    w1(end) = 1;
    w2 = exp(-(stats.N(end) - stats.N(end-Nw+1:end))/10);
    w2 = w2 / sum(w2);
    w = 0.5*w1 + 0.5*w2;
    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function optimState = ResetRunAvg(optimState)
%RESETRUNAVG Reset running averages of moments after warping.

optimState.RunMean = [];
optimState.RunCov = [];        
optimState.LastRunAvg = NaN;

end