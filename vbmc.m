function [vp,elbo,elbo_sd,exitflag,output,optimState,stats] = vbmc(fun,x0,LB,UB,PLB,PUB,options,varargin)
%VBMC Posterior and model inference via Variational Bayesian Monte Carlo (v0.94)
%   VBMC computes a variational approximation of the full posterior and a 
%   lower bound on the normalization constant (marginal likelhood or model
%   evidence) for a provided unnormalized log posterior.
%
%   VP = VBMC(FUN,X0,LB,UB) initializes the variational posterior in the
%   proximity of X0 (ideally, a posterior mode) and iteratively computes
%   a variational approximation for a given target log posterior FUN.
%   FUN accepts input X and returns the value of the target (unnormalized) 
%   log posterior density at X. LB and UB define a set of strict lower and 
%   upper bounds coordinate vector, X, so that the posterior has support on 
%   LB < X < UB. LB and UB can be scalars or vectors. If scalars, the bound 
%   is replicated in each dimension. Use empty matrices for LB and UB if no 
%   bounds exist. Set LB(i) = -Inf and UB(i) = Inf if the i-th coordinate
%   is unbounded (while other coordinates may be bounded). Note that if LB 
%   and UB contain unbounded variables, the respective values of PLB and PUB
%   need to be specified (see below). VBMC returns a variational posterior
%   solution VP, which can then be manipulated via other functions in the
%   VBMC toolbox (see examples below).
%
%   VP = VBMC(FUN,X0,LB,UB,PLB,PUB) specifies a set of plausible lower and
%   upper bounds such that LB < PLB < PUB < UB. Both PLB and PUB
%   need to be finite. PLB and PUB represent a "plausible" range, which
%   should denote a region of high posterior probability mass. Among other 
%   things, the plausible box is used to draw initial samples and to set 
%   priors over hyperparameters of the algorithm. When in doubt, we found 
%   that setting PLB and PUB using the topmost ~68% percentile range of the 
%   prior (e.g, mean +/- 1 SD for a Gaussian prior) works well in many 
%   cases (but note that additional information might afford a better guess).
%  
%   VP = VBMC(FUN,X0,LB,UB,PLB,PUB,OPTIONS) performs variational inference
%   with the default parameters replaced by values in the structure OPTIONS.
%   VBMC('defaults') returns the default OPTIONS struct.
%  
%   VP = VBMC(FUN,X0,LB,UB,PLB,PUB,OPTIONS,...) passes additional arguments
%   to FUN.
%  
%   VP = VBMC(FUN,VP0,...) uses variational posterior VP0 (from a previous
%   run of VBMC) to initialize the current run. You can leave PLB and PUB
%   empty, in which case they will be set using VP0 (recommended).
%
%   [VP,ELBO] = VBMC(...) returns an estimate of the ELBO, the variational
%   expected lower bound on the log marginal likelihood (log model evidence).
%   This estimate is computed via Bayesian quadrature.
%
%   [VP,ELBO,ELBO_SD] = VBMC(...) returns the standard deviation of the
%   estimate of the ELBO, as computed via Bayesian quadrature. Note that
%   this standard deviation is *not* representative of the error between the 
%   ELBO and the true log marginal likelihood.
%
%   [VP,ELBO,ELBO_SD,EXITFLAG] = VBMC(...) returns an EXITFLAG that describes
%   the exit condition. Possible values of EXITFLAG and the corresponding
%   exit conditions are
%
%    1  Change in the variational posterior, in the ELBO and its uncertainty 
%       have reached a satisfactory level of stability across recent
%       iterations, suggesting convergence of the variational solution.
%    0  Maximum number of function evaluations or iterations reached. Note
%       that the returned solution has *not* stabilized.
%
%   [VP,ELBO,ELBO_SD,EXITFLAG,OUTPUT] = VBMC(...) returns a structure OUTPUT 
%   with the following information:
%          function: <Target probability density function name>
%        iterations: <Total iterations>
%         funccount: <Total function evaluations>
%          bestiter: <Iteration of returned solution>
%      trainsetsize: <Size of training set for returned solution>
%        components: <Number of mixture components of returned solution>
%            rindex: <Reliability index (< 1 is good)>
% convergencestatus: <"probable" or "no" convergence>
%          overhead: <Fractional overhead (total runtime / total fcn time - 1)>
%          rngstate: <Status of random number generator>
%         algorithm: <Variational Bayesian Monte Carlo>
%           message: <VBMC termination message>
%              elbo: <Estimated ELBO for returned solution>
%           elbo_sd: <Estimated standard deviation of ELBO at returned solution>
%           retried: <"yes", "no", or "failed" if a retry run was performed>
%
%   OPTIONS = VBMC('defaults') returns a basic default OPTIONS structure.
%
%   EXITFLAG = VBMC('test') runs a battery of tests. Here EXITFLAG is 0 if
%   everything works correctly.
%
%   Examples:
%     FUN can be a function handle (using @)
%       vp = vbmc(@rosenbrock_test, ...)
%     In this case, F = rosenbrock_test(X) returns the scalar log pdf F of 
%     the target pdf evaluated at X.
%
%     An example with no hard bounds, only plausible bounds
%       plb = [-5 -5]; pub = [5 5]; options.Plot = 'on';
%       [vp,elbo,elbo_sd] = vbmc(@rosenbrock_test,[0 0],[],[],plb,pub,options);
%
%     FUN can also be an anonymous function:
%        lb = [0 0]; ub = [pi 5]; plb = [0.1 0.1]; pub = [3 4]; options.Plot = 'on';
%        vp = vbmc(@(x) 3*sin(x(1))*exp(-x(2)),[1 1],lb,ub,plb,pub,options)
%
%   See VBMC_EXAMPLES for an extended tutorial with more examples. 
%   The most recent version of the algorithm and additional documentation 
%   can be found here: https://github.com/lacerbi/vbmc
%   Also, check out the FAQ: https://github.com/lacerbi/vbmc/wiki
%
%   Reference: Acerbi, L. (2018). "Variational Bayesian Monte Carlo". 
%   In Advances in Neural Information Processing Systems 31 (NeurIPS 2018), 
%   pp. 8213-8223.
%
%   See also VBMC_EXAMPLES, VBMC_KLDIV, VBMC_MODE, VBMC_MOMENTS, VBMC_PDF, 
%   VBMC_RND, VBMC_DIAGNOSTICS, @.

%--------------------------------------------------------------------------
% VBMC: Variational Bayesian Monte Carlo for posterior and model inference.
% To be used under the terms of the GNU General Public License 
% (http://www.gnu.org/copyleft/gpl.html).
%
%   Author (copyright): Luigi Acerbi, 2018
%   e-mail: luigi.acerbi@{gmail.com,nyu.edu,unige.ch}
%   URL: http://luigiacerbi.com
%   Version: 0.94 (beta)
%   Release date: May 2, 2019
%   Code repository: https://github.com/lacerbi/vbmc
%--------------------------------------------------------------------------

% The VBMC interface (such as details of input and output arguments) may 
% undergo minor changes before reaching the stable release (1.0).


%% Start timer

t0 = tic;

%% Basic default options
defopts.Display                 = 'iter         % Level of display ("iter", "notify", "final", or "off")';
defopts.Plot                    = 'off          % Plot marginals of variational posterior at each iteration';
defopts.MaxIter                 = '50*nvars     % Max number of iterations';
defopts.MaxFunEvals             = '50*(2+nvars) % Max number of target fcn evals';
defopts.TolStableIters          = '10           % Required stable iterations for termination';
defopts.RetryMaxFunEvals        = '0            % Max number of target fcn evals on retry (0 = no retry)';

%% If called with no arguments or with 'defaults', return default options
if nargout <= 1 && (nargin == 0 || (nargin == 1 && ischar(fun) && strcmpi(fun,'defaults')))
    if nargin < 1
        fprintf('Basic default options returned (type "help vbmc" for help).\n');
    end
    vp = defopts;
    return;
end

%% If called with one argument which is 'test', run test
if nargout <= 1 && nargin == 1 && ischar(fun) && strcmpi(fun,'test')
    vp = runtest();
    return;
end

%% Advanced options (do not modify unless you *know* what you are doing)

defopts.UncertaintyHandling     = 'no           % Explicit noise handling (only partially supported)';
defopts.NoiseSize               = '[]           % Base observation noise magnitude';
defopts.FunEvalStart            = 'max(D,10)    % Number of initial target fcn evals';
defopts.FunEvalsPerIter         = '5            % Number of target fcn evals per iteration';
defopts.SGDStepSize             = '0.005        % Base step size for stochastic gradient descent';
defopts.SkipActiveSamplingAfterWarmup   = 'yes  % Skip active sampling the first iteration after warmup';
defopts.RankCriterion           = 'yes          % Use ranking criterion to pick best non-converged solution';
defopts.TolStableEntropyIters   = '6            % Required stable iterations to switch entropy approximation';
defopts.VariableMeans           = 'yes          % Use variable component means for variational posterior';
defopts.VariableWeights         = 'yes          % Use variable mixture weight for variational posterior';
defopts.WeightPenalty           = '0.1          % Penalty multiplier for small mixture weights';
defopts.Diagnostics             = 'off          % Run in diagnostics mode, get additional info';
defopts.OutputFcn               = '[]           % Output function';
defopts.TolStableExceptions     = '2            % Allowed exceptions when computing iteration stability';
defopts.Fvals                   = '[]           % Evaluated fcn values at X0';
defopts.OptimToolbox            = '[]           % Use Optimization Toolbox (if empty, determine at runtime)';
defopts.ProposalFcn             = '[]           % Weighted proposal fcn for uncertainty search';
defopts.UncertaintyHandling     = '[]           % Explicit noise handling (if empty, determine at runtime)';
defopts.NonlinearScaling   = 'on                % Automatic nonlinear rescaling of variables';
defopts.SearchAcqFcn       = '@acqfreg_vbmc     % Fast search acquisition fcn(s)';
defopts.NSsearch           = '2^13              % Samples for fast acquisition fcn eval per new point';
defopts.NSent              = '@(K) 100*K        % Total samples for Monte Carlo approx. of the entropy';
defopts.NSentFast          = '@(K) 100*K        % Total samples for preliminary Monte Carlo approx. of the entropy';
defopts.NSentFine          = '@(K) 2^15*K       % Total samples for refined Monte Carlo approx. of the entropy';
defopts.NSelbo             = '@(K) 50*K         % Samples for fast approximation of the ELBO';
defopts.NSelboIncr         = '0.1               % Multiplier to samples for fast approx. of ELBO for incremental iterations';
defopts.ElboStarts         = '2                 % Starting points to refine optimization of the ELBO';
defopts.NSgpMax            = '80                % Max GP hyperparameter samples (decreases with training points)';
defopts.NSgpMaxWarmup      = '8                 % Max GP hyperparameter samples during warmup';
defopts.NSgpMaxMain        = 'Inf               % Max GP hyperparameter samples during main algorithm';
defopts.WarmupNoImproThreshold = '20 + 5*nvars  % Fcn evals without improvement before stopping warmup';
defopts.WarmupCheckMax     = 'yes               % Also check for max fcn value improvement before stopping warmup';
defopts.StableGPSampling   = '200 + 10*nvars    % Force stable GP hyperparameter sampling (reduce samples or start optimizing)';
defopts.StableGPSamples    = '0                 % Number of GP samples when GP is stable (0 = optimize)';
defopts.GPSampleThin       = '5                 % Thinning for GP hyperparameter sampling';
defopts.TolGPVar           = '1e-4              % Threshold on GP variance, used to stabilize sampling and by some acquisition fcns';
defopts.gpMeanFun          = 'negquad           % GP mean function';
defopts.KfunMax            = '@(N) N.^(2/3)     % Max variational components as a function of training points';
defopts.Kwarmup            = '2                 % Variational components during warmup';
defopts.AdaptiveK          = '2                 % Added variational components for stable solution';
defopts.HPDFrac            = '0.8               % High Posterior Density region (fraction of training inputs)';
defopts.ELCBOImproWeight   = '3                 % Uncertainty weight on ELCBO for computing lower bound improvement';
defopts.TolLength          = '1e-6              % Minimum fractional length scale';
defopts.NoiseObj           = 'off               % Objective fcn returns noise estimate as 2nd argument (unsupported)';
defopts.CacheSize          = '1e4               % Size of cache for storing fcn evaluations';
defopts.CacheFrac          = '0.5               % Fraction of search points from starting cache (if nonempty)';
defopts.StochasticOptimizer = 'adam             % Stochastic optimizer for varational parameters';
defopts.TolFunStochastic   = '1e-3              % Stopping threshold for stochastic optimization';
defopts.GPStochasticStepsize = 'off               % Set stochastic optimization stepsize via GP hyperparameters';
defopts.TolSD              = '0.1               % Tolerance on ELBO uncertainty for stopping (iff variational posterior is stable)';
defopts.TolsKL             = '0.01*sqrt(nvars)  % Stopping threshold on change of variational posterior per training point';
defopts.TolStableWarmup    = '3                 % Number of stable iterations for stopping warmup';
defopts.TolImprovement     = '0.01              % Required ELCBO improvement per fcn eval before termination';
defopts.KLgauss            = 'yes               % Use Gaussian approximation for symmetrized KL-divergence b\w iters';
defopts.TrueMean           = '[]                % True mean of the target density (for debugging)';
defopts.TrueCov            = '[]                % True covariance of the target density (for debugging)';
defopts.MinFunEvals        = '5*nvars           % Min number of fcn evals';
defopts.MinIter            = 'nvars             % Min number of iterations';
defopts.HeavyTailSearchFrac = '0.25               % Fraction of search points from heavy-tailed variational posterior';
defopts.MVNSearchFrac      = '0.25              % Fraction of search points from multivariate normal';
defopts.HPDSearchFrac      = '0                 % Fraction of search points from multivariate normal fitted to HPD points';
defopts.SearchCacheFrac    = '0                 % Fraction of search points from previous iterations';
defopts.AlwaysRefitVarPost = 'no                % Always fully refit variational posterior';
defopts.Warmup             = 'on                % Perform warm-up stage';
defopts.WarmupOptions      = '[]                % Special OPTIONS struct for warmup stage';
defopts.StopWarmupThresh   = '1                 % Stop warm-up when increase in ELBO is confidently below threshold';
defopts.WarmupKeepThreshold = '10*nvars         % Max log-likelihood difference for points kept after warmup';
defopts.SearchCMAES        = 'on                % Use CMA-ES for search';
defopts.SearchCMAESVPInit  = 'yes               % Initialize CMA-ES search SIGMA from variational posterior';
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
defopts.EntropySwitch      = 'off               % Switch from deterministic entropy to stochastic entropy when reaching stability';
defopts.EntropyForceSwitch = '0.8               % Force switch to stochastic entropy at this fraction of total fcn evals';
defopts.DetEntropyMinD     = '5                 % Start with deterministic entropy only with this number of vars or more';
defopts.TolConLoss         = '0.01              % Fractional tolerance for constraint violation of variational parameters';
defopts.BestSafeSD         = '5                 % SD multiplier of ELCBO for computing best variational solution';
defopts.BestFracBack       = '0.25              % When computing best solution, lacking stability go back up to this fraction of iterations';
defopts.TolWeight          = '1e-2              % Threshold mixture component weight for pruning';
defopts.AnnealedGPMean     = '@(N,NMAX) 0       % Annealing for hyperprior width of GP negative quadratic mean';
defopts.ConstrainedGPMean  = 'no                % Strict hyperprior for GP negative quadratic mean';
defopts.EmpiricalGPPrior   = 'yes               % Empirical Bayes prior over some GP hyperparameters';
defopts.InitDesign         = 'plausible         % Initial samples ("plausible" is uniform in the plausible box)';
defopts.BOWarmup           = 'no                % Bayesian-optimization-like warmup stage';


%% Advanced options for unsupported/untested features (do *not* modify)
defopts.WarpRotoScaling    = 'off               % Rotate and scale input';
%defopts.WarpCovReg         = '@(N) 25/N         % Regularization weight towards diagonal covariance matrix for N training inputs';
defopts.WarpCovReg         = '0                 % Regularization weight towards diagonal covariance matrix for N training inputs';
defopts.WarpNonlinear      = 'off               % Nonlinear input warping';
defopts.WarpEpoch          = '100               % Recalculate warpings after this number of fcn evals';
defopts.WarpMinFun         = '10 + 2*D          % Minimum training points before starting warping';
defopts.WarpNonlinearEpoch = '100               % Recalculate nonlinear warpings after this number of fcn evals';
defopts.WarpNonlinearMinFun = '20 + 5*D         % Minimum training points before starting nonlinear warping';
defopts.ELCBOWeight        = '0                 % Uncertainty weight during ELCBO optimization';
defopts.VarParamsBack      = '0                 % Check variational posteriors back to these previous iterations';
defopts.AltMCEntropy       = 'no                % Use alternative Monte Carlo computation for the entropy';
defopts.VarActiveSample    = 'no                % Variational active sampling';
defopts.FeatureTest        = 'no                % Test a new experimental feature';


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

switch lower(options.Display(1:min(end,3)))
    case {'not'}                        % notify
        prnt = 1;
    case {'no','non','off'}             % none
        prnt = 0;
    case {'ite','all','on','yes'}       % iter
        prnt = 3;
    case {'fin','end'}                  % final
        prnt = 2;
    otherwise
        prnt = 3;
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

% Initialize from variational posterior
if vbmc_isavp(x0)
    init_from_vp_flag = true;
    vp0 = x0;
    [x0,LB,UB,PLB,PUB,Xvp] = initFromVP(vp0,LB,UB,PLB,PUB,prnt);
else
    init_from_vp_flag = false;    
end
    
D = size(x0,2);     % Number of variables
optimState = [];

% Setup algorithm options
[options,cmaes_opts] = setupoptions(D,defopts,options);
if options.Warmup
    options_main = options;
    % Use special options during Warmup
    if isfield(options,'WarmupOptions')
        options = setupoptions(D,options,options.WarmupOptions);
    end
end

if init_from_vp_flag    % Finish initialization from variational posterior
    x0 = [x0; robustSampleFromVP(vp0,options.FunEvalStart-1,Xvp)];
    clear Xvp vp0;
end

% Check/fix boundaries and starting points
[x0,LB,UB,PLB,PUB] = boundscheck(x0,LB,UB,PLB,PUB,prnt);

% Convert from char to function handles
if ischar(fun); fun = str2func(fun); end

% Setup and transform variables
K = options.Kwarmup;
[vp,optimState] = ...
    setupvars(x0,LB,UB,PLB,PUB,K,optimState,options,prnt);

% Store target density function
optimState.fun = fun;
if isempty(varargin)
    funwrapper = fun;   % No additional function arguments passed
else
    funwrapper = @(u_) fun(u_,varargin{:});
end

% Initialize function logger
[~,optimState] = funlogger_vbmc([],x0(1,:),optimState,'init',options.CacheSize,options.NoiseObj);

% GP struct and GP hyperparameters
gp = [];    hyp = [];   hyp_warp = [];  hyp_logp = [];
optimState.gpMeanfun = options.gpMeanFun;
switch optimState.gpMeanfun
    case {'zero','const','negquad','se','negquadse'}
    otherwise
        error('vbmc:UnknownGPmean', ...
            'Unknown/unsupported GP mean function. Supported mean functions are ''zero'', ''const'', ''negquad'', and ''se''.');
end

if optimState.Cache.active
    displayFormat = ' %5.0f     %5.0f  /%5.0f   %12.2f  %12.2f  %12.2f     %4.0f %10.3g       %s\n';
    displayFormat_warmup = ' %5.0f     %5.0f  /%5.0f   %s\n';
else
    displayFormat = ' %5.0f       %5.0f    %12.2f  %12.2f  %12.2f     %4.0f %10.3g     %s\n';
    displayFormat_warmup = ' %5.0f       %5.0f    %12.2f  %s\n';
end
if prnt > 2
    if optimState.Cache.active
        fprintf(' Iteration f-count/f-cache    Mean[ELBO]     Std[ELBO]     sKL-iter[q]   K[q]  Convergence    Action\n');
    else
        if options.BOWarmup
            fprintf(' Iteration   f-count     Max[f]     Action\n');
        else
            fprintf(' Iteration   f-count     Mean[ELBO]     Std[ELBO]     sKL-iter[q]   K[q]  Convergence  Action\n');            
        end
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
    if optimState.SkipActiveSampling
        optimState.SkipActiveSampling = false;
    else
        if options.VarActiveSample
            [optimState,vp,t_active(iter),t_func(iter)] = ...
                variationalactivesample_vbmc(optimState,new_funevals,funwrapper,vp,vp_old,gp,options,cmaes_opts);            
        else
            [optimState,t_active(iter),t_func(iter)] = ...
                activesample_vbmc(optimState,new_funevals,funwrapper,vp,vp_old,gp,options,cmaes_opts);
        end
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
    if optimState.Warmup && options.BOWarmup
        [hypprior,X_hpd,y_hpd,~,hyp0,optimState.gpMeanfun,Ns_gp] = ...
            vbmc_gphyp(optimState,'const',0,options);
    else
        [hypprior,X_hpd,y_hpd,~,hyp0,optimState.gpMeanfun,Ns_gp] = ...
            vbmc_gphyp(optimState,optimState.gpMeanfun,0,options);
    end
    if isempty(hyp); hyp = hyp0; end % Initial GP hyperparameters
    if Ns_gp == options.StableGPSamples && optimState.StopSampling == 0
        optimState.StopSampling = optimState.N; % Reached stable sampling
    end
    
    % Get GP training options
    gptrain_options = get_GPTrainOptions(Ns_gp,optimState,stats,options);    
    gptrain_options.LogP = hyp_logp;
    if numel(gptrain_options.Widths) ~= numel(hyp0); gptrain_options.Widths = []; end
    
    % Get training dataset
    [X_train,y_train] = get_traindata(optimState,options);
    
    % Fit GP to training set
    [gp,hyp,gpoutput] = gplite_train(hyp,Ns_gp,X_train,y_train, ...
        optimState.gpMeanfun,hypprior,gptrain_options);
    hyp_full = gpoutput.hyp_prethin; % Pre-thinning GP hyperparameters
    hyp_logp = gpoutput.logp;
    
%      if iter > 10
%          pause
%      end
    
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
    
    if ~vp.optimize_mu  % Variational components fixed to training inputs
        vp.mu = gp.X';
        Knew = size(vp.mu,2);
    else
        % Update number of variational mixture components
        Knew = updateK(optimState,stats,options);
    end
    
    % Decide number of fast/slow optimizations
    if isa(options.NSelbo,'function_handle')
        Nfastopts = ceil(options.NSelbo(K));
    else
        Nfastopts = ceil(options.NSelbo);
    end
    if optimState.RecomputeVarPost || options.AlwaysRefitVarPost
        Nslowopts = options.ElboStarts; % Full optimizations
        optimState.RecomputeVarPost = false;
    else
        % Only incremental change from previous iteration
        Nfastopts = ceil(Nfastopts * options.NSelboIncr);
        Nslowopts = 1;
    end
    
    % Run optimization of variational parameters
    if optimState.Warmup && options.BOWarmup
        elbo = NaN;     elbo_sd = NaN;      varss = NaN;
        pruned = 0;     G = NaN;    H = NaN;    varG = NaN; VarH = NaN;
    else
        [vp,elbo,elbo_sd,G,H,varG,varH,varss,pruned] =  ...
            vpoptimize(Nfastopts,Nslowopts,vp,gp,Knew,X_hpd,y_hpd,optimState,stats,options,cmaes_opts,prnt);
    end
    % Save variational solution stats
    vp.stats.elbo = elbo;               % ELBO
    vp.stats.elbo_sd = elbo_sd;         % Error on the ELBO
    vp.stats.elogjoint = G;             % Expected log joint
    vp.stats.elogjoint_sd = sqrt(varG); % Error on expected log joint
    vp.stats.entropy = H;               % Entropy
    vp.stats.entropy_sd = sqrt(varH);   % Error on the entropy
    vp.stats.stable = false;            % Unstable until proven otherwise
    
    optimState.vpK = vp.K;
    optimState.H = H;   % Save current entropy
    
    timer.variationalFit = toc(t);
    
    %% Recompute warpings at end iteration (unsupported)
    if options.WarpNonlinear || options.WarpRotoScaling    
        [optimState,vp,hyp] = ...
            vbmc_rewarp(optimState,vp,gp,hyp,options,cmaes_opts);
    end
    
    %% Plot current iteration (to be improved)
    if options.Plot
        vbmc_iterplot(vp,gp,optimState,stats,elbo);
    end
    
    %mubar
    %Sigma
    
    %----------------------------------------------------------------------
    %% Finalize iteration
    t = tic;
    
    % Compute symmetrized KL-divergence between old and new posteriors
    Nkl = 1e5;
    sKL = max(0,0.5*sum(vbmc_kldiv(vp,vp_old,Nkl,options.KLgauss)));
    
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
        optimState.RunMean = mubar(:);
        optimState.RunCov = Sigma;        
        optimState.LastRunAvg = optimState.N;
        % optimState.RunCorrection = 1;
    else
        Nnew = optimState.N - optimState.LastRunAvg;
        wRun = options.MomentsRunWeight^Nnew;
        optimState.RunMean = wRun*optimState.RunMean + (1-wRun)*mubar(:);
        optimState.RunCov = wRun*optimState.RunCov + (1-wRun)*Sigma;
        optimState.LastRunAvg = optimState.N;
        % optimState.RunT = optimState.RunT + 1;
    end
            
    % Check if we are still warming-up
    if optimState.Warmup && iter > 1    
        [optimState,action] = vbmc_warmup(optimState,stats,action,elbo,elbo_sd,options);
        if ~optimState.Warmup
            vp.optimize_mu = logical(options.VariableMeans);
            vp.optimize_weights = logical(options.VariableWeights);
            if options.BOWarmup
                optimState.gpMeanfun = options.gpMeanFun;
                hyp = [];
            end
            % Switch to main algorithm options
            options = options_main;
        end
    end

%     if optimState.Warmup && iter >= floor(D/2)+3
%         % Remove warm-up points from training set unless close to max
%         ymax = max(optimState.y_orig(1:optimState.Xmax));
%         D = numel(optimState.LB);
%         NkeepMin = 2*D;
%         idx_keep = (ymax - optimState.y_orig) < options.WarmupKeepThreshold;
%         if sum(idx_keep) < NkeepMin
%             y_temp = optimState.y_orig;
%             y_temp(~isfinite(y_temp)) = -Inf;
%             [~,ord] = sort(y_temp,'descend');
%             idx_keep(ord(1:min(NkeepMin,optimState.Xmax))) = true;
%         end
%         optimState.X_flag = idx_keep & optimState.X_flag;
%         if isempty(action); action = 'trim'; else; action = [action ', trim']; end
%     end
    
    % t_fits(iter) = toc(timer_fits);    
    % dt = (t_active(iter)+t_fits(iter))/new_funevals;
    
    timer.finalize = toc(t);
    
    % timer
    
    % Record all useful stats
    stats = savestats(stats, ...
        optimState,vp,elbo,elbo_sd,varss,sKL,sKL_true,gp,hyp_full,Ns_gp,pruned,timer,options.Diagnostics);
    
    %----------------------------------------------------------------------
    %% Check termination conditions    

    [optimState,stats,isFinished_flag,exitflag,action,msg] = ...
        vbmc_termination(optimState,action,stats,options);
    
    %% Write iteration output
    
    % vp.w
    
    % Stopped GP sampling this iteration?
    if Ns_gp == options.StableGPSamples && ...
            stats.gpNsamples(max(1,iter-1)) > options.StableGPSamples
        if Ns_gp == 0
            if isempty(action); action = 'switch to GP opt'; else; action = [action ', switch to GP opt']; end
        else
            if isempty(action); action = 'stable GP sampling'; else; action = [action ', stable GP sampling']; end
        end
    end    
    
    if prnt > 2
        if options.BOWarmup && optimState.Warmup
            fprintf(displayFormat_warmup,iter,optimState.funccount,max(optimState.y_orig),action);            
        else
            if optimState.Cache.active
                fprintf(displayFormat,iter,optimState.funccount,optimState.cachecount,elbo,elbo_sd,sKL,vp.K,optimState.R,action);
            else
                fprintf(displayFormat,iter,optimState.funccount,elbo,elbo_sd,sKL,vp.K,optimState.R,action);
            end
        end
    end
        
end

% Pick "best" variational solution to return
[vp,elbo,elbo_sd,idx_best] = ...
    best_vbmc(stats,iter,options.BestSafeSD,options.BestFracBack,options.RankCriterion);

if ~stats.stable(idx_best); exitflag = 0; end

% Print final message
if prnt > 1
    fprintf('\n%s\n', msg);    
    fprintf('Estimated ELBO: %.3f +/- %.3f.\n', elbo, elbo_sd);
    if exitflag < 1
        fprintf('Caution: Returned variational solution may have not converged.\n');
    end
    fprintf('\n');
end

if nargout > 4
    output = vbmc_output(elbo,elbo_sd,optimState,msg,stats,idx_best);
    
    % Compute total running time and fractional overhead
    optimState.totaltime = toc(t0);    
    output.overhead = optimState.totaltime / optimState.totalfunevaltime - 1;    
end

if nargout > 6
    % Remove GP from stats struct unless diagnostic run
    if ~options.Diagnostics
        stats = rmfield(stats,'gp');
        stats = rmfield(stats,'gpHypFull');
    end
end

if exitflag < 1 && options.RetryMaxFunEvals > 0
    % Rerun VBMC with better initialization if first try did not work    
    if prnt > 0
        fprintf('First attempt did not converge. Trying to rerun variational optimization.\n');
    end    
    
    % Get better VBMC parameters and initialization from current run
    [x0,LB,UB,PLB,PUB,Xvp] = initFromVP(vp,LB,UB,PLB,PUB,0);
    Ninit = max(options.FunEvalStart,ceil(options.RetryMaxFunEvals/10));
    x0 = [x0; robustSampleFromVP(vp,Ninit-1,Xvp)];
    
    options.FunEvalStart = Ninit;
    options.MaxFunEvals = options.RetryMaxFunEvals;
    options.RetryMaxFunEvals = 0;                   % Avoid infinite loop
    options.SGDStepSize = 0.2*options.SGDStepSize;  % Increase stability
    
    try
        [vp,elbo,elbo_sd,exitflag,output2,optimState2,stats] = vbmc(fun,x0,LB,UB,PLB,PUB,options,varargin{:});
        
        if nargout > 4
            optimState2.totaltime = toc(t0);
            output2.overhead = optimState.totaltime / (optimState.totalfunevaltime + optimState2.totalfunevaltime) - 1;
            output2.iterations = output2.iterations + output.iterations;
            output2.funccount = output2.funccount + output.funccount;
            output2.retried = 'yes';
            output = output2;
            optimState = optimState2;
        end        
    catch retryException
        msgText = getReport(retryException);
        warning(msgText);
        if prnt > 0
            fprintf('Attempt of rerunning variational optimization FAILED. Keeping original results.\n');
        end
        if nargout > 4
            output.retried = 'error';
        end
    end
    
else
    if nargout > 4; output.retried = 'no'; end
end



end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function stats = savestats(stats,optimState,vp,elbo,elbo_sd,varss,sKL,sKL_true,gp,hyp_full,Ns_gp,pruned,timer,debugflag)

iter = optimState.iter;
stats.iter(iter) = iter;
stats.N(iter) = optimState.N;
stats.Neff(iter) = optimState.Neff;
stats.funccount(iter) = optimState.funccount;
stats.cachecount(iter) = optimState.cachecount;
stats.vpK(iter) = vp.K;
stats.warmup(iter) = optimState.Warmup;
stats.pruned(iter) = pruned;
stats.elbo(iter) = elbo;
stats.elbo_sd(iter) = elbo_sd;
stats.sKL(iter) = sKL;
if ~isempty(sKL_true)
    stats.sKL_true = sKL_true;
end
stats.gpSampleVar(iter) = varss;
stats.gpNsamples(iter) = Ns_gp;
stats.gpHypFull{iter} = hyp_full;
stats.timer(iter) = timer;
stats.vp(iter) = vp;
stats.gp(iter) = gplite_clean(gp);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function add2path()
%ADD2PATH Adds VBMC subfolders to MATLAB path.

subfolders = {'acq','ent','gplite','misc','utils'};
% subfolders = {'acq','ent','gplite','misc','utils','warp'};
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
function [x0,LB,UB,PLB,PUB,Xvp] = initFromVP(vp,LB,UB,PLB,PUB,prnt)

if prnt > 2
    fprintf('Initializing VBMC from variational posterior (D = %d).\n', vp.D);
    if ~isempty(PLB) && ~isempty(PUB)
        fprintf('Using provided plausible bounds. Note that it might be better to leave them empty,\nand allow VBMC to set them using the provided variational posterior.\n');
    end
end

% Find mode in transformed space
x0t = vbmc_mode(vp,0);
x0 = warpvars(x0t,'inv',vp.trinfo);

% Sample from variational posterior and set plausible bounds accordingly
if isempty(PLB) && isempty(PUB)
    Xvp = vbmc_rnd(vp,1e6,[],1);
    PLB = quantile(Xvp,0.05);
    PUB = quantile(Xvp,0.95);
else
    Xvp = [];
end    
if isempty(LB); LB = vp.trinfo.lb_orig; end
if isempty(UB); UB = vp.trinfo.ub_orig; end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Xrnd = robustSampleFromVP(vp,Ns,Xrnd,quantile_thresh)
%ROBUSTSAMPLEFROMVP Robust sample from variational posterior.

if nargin < 3; Xrnd = []; end
if nargin < 4 || isempty(quantile_thresh); quantile_thresh = 0.01; end

Ns_big = 1e4;
Xrnd = [Xrnd; vbmc_rnd(vp,max(0,Ns_big-size(Xrnd,1)),[],1)];
Xrnd = Xrnd(1:Ns_big,:);

y = vbmc_pdf(vp,Xrnd);
y_thresh = quantile(y,quantile_thresh);

Xrnd = Xrnd(y > y_thresh,:);
Xrnd = Xrnd(1:Ns,:);

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TO-DO list:
% - Initialization with multiple (e.g., cell array of) variational posteriors.
% - Combine multiple variational solutions?
% - GP sampling at the very end?
% - Quasi-random sampling from variational posterior (e.g., for initialization).
% - Write a private quantile function to avoid calls to Stats Toolbox.
% - Fix call to fmincon if Optimization Toolbox is not available.
% - Check that I am not using other ToolBoxes by mistake.
