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
defopts.SearchAcqFcn       = '{@vbmc_acqGEV,@vbmc_acqexpent,@vbmc_acqgvar}  % Fast search acquisition fcn(s)';
defopts.Nacq               = '20                % Expensive acquisition fcn evals per new point';
defopts.NSsearch           = '2^15              % Samples for fast acquisition fcn eval per new point';
defopts.NSent              = '100               % Samples per component for fast Monte Carlo approx. of the entropy';
defopts.NSentFine          = '2^15              % Samples per component for refined Monte Carlo approx. of the entropy';
defopts.NSelbo             = '50                % Samples per component for fast approx. of ELBO';
defopts.ElboStarts         = '2                 % Starting points to refine optimization of the ELBO';
defopts.NSgpMax            = '80                % Max GP hyperparameter samples (decreases with training points)';
defopts.StopGPSampling     = '200 + 10*nvars    % Stop GP hyperparameter sampling (start optimizing)';
defopts.TolGPVar           = '1e-4              % Stop GP hyperparameter sampling if sample variance is below this threshold per fcn';
defopts.QuadraticMean      = 'yes               % Use GP with quadratic mean function (otherwise constant)';
defopts.Kfun               = '@sqrt             % Variational components as a function of training points';
defopts.HPDFrac            = '0.5               % High Posterior Density region (fraction of training inputs)';
defopts.WarpRotoScaling    = 'on                % Rotate and scale input';
%defopts.WarpCovReg         = '@(N) 25/N         % Regularization weight towards diagonal covariance matrix for N training inputs';
defopts.WarpCovReg         = '0                 % Regularization weight towards diagonal covariance matrix for N training inputs';
defopts.WarpNonlinear      = 'on                % Nonlinear input warping';
defopts.WarpEpoch          = '20                % Recalculate warpings after this number of fcn evals';
defopts.WarpMinFun         = '10 + 2*D          % Minimum training points before starting warping';
defopts.WarpNonlinearEpoch = '100               % Recalculate nonlinear warpings after this number of fcn evals';
defopts.WarpNonlinearMinFun = '20 + 5*D         % Minimum training points before starting nonlinear warping';
defopts.ELCBOWeight        = '1                 % Uncertainty weight during ELCBO optimization';
defopts.TolLength          = '1e-6              % Minimum fractional length scale';
defopts.NoiseObj           = 'off               % Objective fcn returns noise estimate as 2nd argument (unsupported)';
defopts.CacheSize          = '1e4               % Size of cache for storing fcn evaluations';
defopts.CacheFrac          = '0.5               % Fraction of search points from starting cache (if nonempty)';
defopts.TolFunAdam         = '0.001             % Stopping threshold for Adam optimizer';
defopts.TolSD              = '0.1               % Tolerance on ELBO uncertainty for stopping (iff variational posterior is stable)';
defopts.TolsKL             = '0.01*sqrt(nvars)  % Stopping threshold on change of variational posterior per training point';
defopts.TolStableIters     = '5                 % Number of stable iterations for checking stopping criteria';
defopts.TolStableFunEvals  = '5*nvars           % Number of stable fcn evals for checking stopping criteria';
defopts.KLgauss            = 'yes               % Use Gaussian approximation for symmetrized KL-divergence b\w iters';
defopts.TrueMean           = '[]                % True mean of the target density (for debugging)';
defopts.TrueCov            = '[]                % True covariance of the target density (for debugging)';
defopts.MinFunEvals        = '2*nvars^2         % Min number of fcn evals';
defopts.MinIter            = 'nvars             % Min number of iterations';
defopts.HeavyTailSearchFrac = '0.25               % Fraction of search points from heavy-tailed variational posterior';
defopts.MVNSearchFrac      = '0.25              % Fraction of search points from multivariate normal';
defopts.SearchSampleGP     = 'false             % Generate search candidates sampling from GP surrogate';
defopts.AlwaysRefitVarPost = 'true              % Always fully refit variational posterior';
defopts.VarParamsBack      = '0                 % Check variational posteriors back to these previous iterations';
defopts.Plot               = 'off               % Show variational posterior triangle plots';
defopts.Warmup             = 'on                % Perform warm-up stage';
defopts.StopWarmupThresh   = '1                 % Stop warm-up when increase in ELBO is confidently below threshold';
defopts.WarmupKeepThreshold = '10*nvars         % Max log-likelihood difference for points kept after warmup';
defopts.SearchCMAES        = 'no                % Use CMA-ES for search';

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
    displayFormat = ' %5.0f     %5.0f  /%5.0f   %12.3g  %12.3g  %12.3g     %4.0f %10.3g       %s\n';
else
    displayFormat = ' %5.0f       %5.0f    %12.3g  %12.3g  %12.3g     %4.0f %10.3g     %s\n';
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
exitflag = 0;   output = [];    stats = [];

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
    [optimState,t_adapt(iter),t_func(iter)] = ...
        adaptive_sampling(optimState,new_funevals,funwrapper,vp,vp_old,gp,options,cmaes_opts);
    optimState.N = optimState.Xmax;  % Number of training inputs
    optimState.Neff = sum(optimState.X_flag(1:optimState.Xmax));
    timer.activeSampling = toc(t);
        
    % Rotoscaling iteration?
    t = tic;
    isRotoscaling = (optimState.N - optimState.LastWarping) >= options.WarpEpoch ...
        && (options.MaxFunEvals - optimState.N) >= options.WarpEpoch ...
        && optimState.N >= options.WarpMinFun;
    if isRotoscaling
        optimState.LastWarping = optimState.N;
        optimState.WarpingCount = optimState.WarpingCount + 1;
        if isempty(action); action = 'rotoscale'; else; action = [action ', rotoscale']; end
    end
    
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
    
    %% Update stretching of unbounded variables
    if any(isinf(LB) & isinf(UB)) && (isNonlinearWarping || iter == 1)
        [vp,optimState,hyp] = ...
            warp_unbounded(vp,optimState,hyp,gp,cmaes_opts,options);
    end

    %%  Rotate and rescale variables
    if options.WarpRotoScaling && isRotoscaling
        [vp,optimState,hyp] = ...
            warp_rotoscaling(vp,optimState,hyp,gp,cmaes_opts,options);
    end
    
    %% Learn nonlinear warping via GP
    if options.WarpNonlinear && isNonlinearWarping
        [vp,optimState,hyp,hyp_warp] = ...
            warp_nonlinear(vp,optimState,hyp,hyp_warp,cmaes_opts,options);
        optimState.redoRotoscaling = true;
    end
    
%     if optimState.DoRotoscaling && isWarping
%         [~,X_hpd,y_hpd] = ...
%             vbmc_gphyp(optimState,optimState.gpMeanfun,0,options);            
%         vp = vpoptimize(1,0,vp,gp,vp.K,X_hpd,y_hpd,optimState,stats,options);
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
        
        % Stop sampling after sample variance stays below Tol for a while
        [idx_stable,dN,dN_last] = getStableIter(stats,optimState,options);
        if ~isempty(idx_stable) && idx_stable > 1
            varss_list = stats.gpSampleVar;
            if sum(varss_list(idx_stable-1:iter-1)) < options.TolGPVar*dN && ...
                    varss_list(end) < options.TolGPVar*dN_last
                optimState.StopSampling = optimState.N;
            end
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
    gptrain_options.Nopts = 3;
    
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
    Knew = getK(optimState,options);

    if optimState.RecomputeVarPost || options.AlwaysRefitVarPost
        Nslowopts = options.ElboStarts; % Full optimizations
        useEntropyApprox = true;
        optimState.RecomputeVarPost = false;
    else
        % Only incremental change
        Nslowopts = 1;
        useEntropyApprox = true;
    end
    
    [vp,elbo,elbo_sd,varss] = ...
        vpoptimize(Nslowopts,useEntropyApprox,vp,gp,Knew,X_hpd,y_hpd,optimState,stats,options);
        
    
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
        
         vp = vpoptimize(1,0,vp,gp,vp.K,X_hpd,y_hpd,optimState,stats,options);
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
    if ~isempty(options.TrueMean) && ~isempty(options.TrueCov)
        [mubar,Sigma] = vbmc_moments(vp,1,1e6);
        [kl(1),kl(2)] = mvnkl(mubar,Sigma,options.TrueMean,options.TrueCov);
        sKL_true = 0.5*sum(kl);
    else
        sKL_true = [];
    end
    
    % Check if we are still warming-up (95% confidence)
    if optimState.Warmup && iter > 1
        elbo_old = stats.elbo(iter-1);
        elboSD_old = stats.elboSD(iter-1);
        increaseUCB = elbo - elbo_old + 1.6449*sqrt(elbo_sd^2 + elboSD_old^2);
        if increaseUCB < options.StopWarmupThresh
            optimState.Warmup = false;
            if isempty(action); action = 'end warm-up'; else; action = [action ', end warm-up']; end
            
            % Remove warm-up points from training set unless close to max
            ymax = max(optimState.y_orig(1:optimState.Xmax));
            idx_keep = (ymax - optimState.y_orig) < options.WarmupKeepThreshold;
            optimState.X_flag = idx_keep & optimState.X_flag;
            
            % Start nonlinear warping
            optimState.LastNonlinearWarping = optimState.N;
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
    [idx_stable,dN,dN_last] = getStableIter(stats,optimState,options);
    if ~isempty(idx_stable) && ~optimState.Warmup
        sKL_list = stats.sKL;
        elbo_list = stats.elbo;
        err2 = sum((elbo_list(idx_stable:iter) - mean(elbo_list(idx_stable:iter))).^2);
        qindex(1) = sqrt(err2 / (options.TolSD^2*dN));
        qindex(2) = stats.elboSD(iter) / options.TolSD;
        qindex(3) = sum(sKL_list(idx_stable:iter)) / (options.TolsKL*dN);
        qindex(4) = sKL_list(iter) / (options.TolsKL*dN_last);        
        if all(qindex < 1)
            isFinished_flag = true;
            exitflag = 0;
                % msg = 'Optimization terminated: reached maximum number of iterations OPTIONS.MaxIter.';
        end
        qindex = mean(qindex);
        stats.qindex(iter) = qindex;
    else
        qindex = Inf;
    end
    
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
function K = getK(optimState,options)
%GETK Get number of variational components.

Neff = optimState.Neff;
Kfun = options.Kfun;

if optimState.Warmup
    K = 2;
else
    if isnumeric(Kfun)
        K = Kfun;
    elseif isa(Kfun,'function_handle')
        K = Kfun(Neff);
    end
    K = min(Neff,max(1,round(K)));
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
function [idx_stable,dN,dN_last] = getStableIter(stats,optimState,options)
%GETSTABLEITER Find index of starting stable iteration.

iter = optimState.iter;
idx_stable = [];
dN = [];    dN_last = [];

if ~isempty(stats)
    iter_list = stats.iter;
    N_list = stats.N;
    idx_stable = find(N_list <= optimState.N - options.TolStableFunEvals & ...
        iter_list <= iter - options.TolStableIters,1,'last');
    if ~isempty(idx_stable)
        dN = optimState.N - N_list(idx_stable);
        dN_last = N_list(end) - N_list(end-1);
    end
end

end