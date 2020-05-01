function [vp,optimState] = setupvars_vbmc(x0,LB,UB,PLB,PUB,K,optimState,options,prnt)
%INITVARS Initialize variational posterior, transforms and variables for VBMC.

nvars = size(LB,2);

% Starting point
if any(~isfinite(x0))   % Invalid/not provided starting point
    if prnt > 0
        fprintf('Initial starting point is invalid or not provided. Starting from center of plausible region.\n');
    end
    x0 = 0.5*(PLB + PUB);       % Midpoint
end

% Integer variables
optimState.integervars = false(1, nvars);
if ~isempty(options.IntegerVars)
    optimState.integervars(options.IntegerVars) = true;    
    for d = find(optimState.integervars)
        if (~isfinite(LB(d)) && floor(LB(d)) ~= 0.5) || ...
                (~isfinite(UB(d)) && floor(UB(d)) ~= 0.5)
                error('Hard bounds of integer variables need to be set at +/- 0.5 points from their boundary values (e.g., -0.5 and 10.5 for a variable that takes values from 0 to 10).');
        end
    end
end

optimState.LB_orig = LB;
optimState.UB_orig = UB;
optimState.PLB_orig = PLB;
optimState.PUB_orig = PUB;
optimState.LBeps_orig = LB + (UB - LB)*options.TolBoundX;
optimState.UBeps_orig = UB - (UB - LB)*options.TolBoundX;

% Transform variables
trinfo = warpvars_vbmc(nvars,LB,UB,PLB,PUB);
switch lower(options.BoundedTransform)
    case 'logit'
    case 'norminv'
        trinfo.type(trinfo.type == 3) = 12;
    case 'student4'
        trinfo.type(trinfo.type == 3) = 13;        
    otherwise
        error('vbmc:UnknwonBoundedTransform','Unknown bounded transform.');
end

trinfo.x0_orig = x0;
if ~isfield(trinfo,'R_mat'); trinfo.R_mat = []; end
if ~isfield(trinfo,'scale'); trinfo.scale = []; end

optimState.LB = warpvars_vbmc(LB,'dir',trinfo);
optimState.UB = warpvars_vbmc(UB,'dir',trinfo);
optimState.PLB = warpvars_vbmc(PLB,'dir',trinfo);
optimState.PUB = warpvars_vbmc(PUB,'dir',trinfo);

% Record starting points (original coordinates)
optimState.Cache.X_orig = x0;
optimState.Cache.y_orig = options.Fvals(:);
if isempty(optimState.Cache.y_orig)
    optimState.Cache.y_orig = NaN(size(optimState.Cache.X_orig,1),1);
end
if size(optimState.Cache.X_orig,1) ~= size(optimState.Cache.y_orig,1)
    error('vbmc:MismatchedStartingInputs',...
        'The number of points in X0 and of their function values as specified in OPTIONS.Fvals are not the same.');
end

x0 = warpvars_vbmc(x0,'dir',trinfo);

% Report variable transformation
if any(optimState.integervars) && prnt > 0
    if sum(optimState.integervars) == 1
        fprintf('Index of variable restricted to integer values: %s.\n',mat2str(find(optimState.integervars)));        
    else
        fprintf('Indices of variables restricted to integer values: %s.\n',mat2str(find(optimState.integervars)));
    end
end

%% Initialize variational posterior

vp.D = nvars;
vp.K = K;
x0start = repmat(x0,[ceil(K/size(x0,1)),1]);
vp.w = ones(1,K)/K;
vp.mu = bsxfun(@plus,x0start(1:K,:)',1e-6*randn(vp.D,K));
vp.sigma = 1e-3*ones(1,K);
vp.lambda = ones(vp.D,1);
vp.trinfo = trinfo;
optimState.trinfo = vp.trinfo;
vp.optimize_sigma = true;
vp.optimize_lambda = true;
if options.Warmup
    vp.optimize_mu = true;
    vp.optimize_weights = false;
else
    vp.optimize_mu = logical(options.VariableMeans);
    vp.optimize_weights = logical(options.VariableWeights);
end
vp.temperature = NaN;
vp.delta = [];
vp.bounds = [];
vp.stats = [];

% Import prior function evaluations
% if ~isempty(options.FunValues)
%     if ~isfield(options.FunValues,'X') || ~isfield(options.FunValues,'Y')
%         error('bads:funValues', ...
%             'The ''FunValues'' field in OPTIONS needs to have a X and a Y field (respectively, inputs and their function values).');
%     end
%         
%     X = options.FunValues.X;
%     Y = options.FunValues.Y;
%     if size(X,1) ~= size(Y,1)
%         error('X and Y arrays in the OPTIONS.FunValues need to have the same number of rows (each row is a tested point).');        
%     end
%     
%     if ~all(isfinite(X(:))) || ~all(isfinite(Y(:))) || ~isreal(X) || ~isreal(Y)
%         error('X and Y arrays need to be finite and real-valued.');                
%     end    
%     if ~isempty(X) && size(X,2) ~= nvars
%         error('X should be a matrix of tested points with the same dimensionality as X0 (one input point per row).');
%     end
%     if ~isempty(Y) && size(Y,2) ~= 1
%         error('Y should be a vertical array of function values (one function value per row).');
%     end
%     
%     optimState.X = X;
%     optimState.Y = Y;    
%     
%     % Heteroskedastic noise
%     if isfield(options.FunValues,'S')
%         S = options.FunValues.S;
%         if size(S,1) ~= size(Y,1)
%             error('X, Y, and S arrays in the OPTIONS.FunValues need to have the same number of rows (each row is a tested point).');        
%         end    
%         if ~all(isfinite(S)) || ~isreal(S) || ~all(S > 0)
%             error('S array needs to be finite, real-valued, and positive.');
%         end
%         if ~isempty(S) && size(S,2) ~= 1
%             error('S should be a vertical array of estimated function SD values (one SD per row).');
%         end
%         optimState.S = S;        
%     end    
%     
% end

%% Initialize OPTIMSTATE variables

% Before first iteration
optimState.iter = 0;

% Estimate of GP observation noise around the high posterior density region
optimState.sn2hpd = Inf;

% Does the starting cache contain function values?
optimState.Cache.active = any(isfinite(optimState.Cache.y_orig));

% When was the last warping action performed (number of training inputs)
optimState.LastWarping = -Inf;

% Number of warpings performed
optimState.WarpingCount = 0;

% When GP hyperparameter sampling is switched with optimization
if options.NSgpMax > 0
    optimState.StopSampling = 0;
else
    optimState.StopSampling = Inf;    
end

% Fully recompute variational posterior
optimState.RecomputeVarPost = true;

% Start with warm-up?
optimState.Warmup = options.Warmup;
if optimState.Warmup
    optimState.LastWarmup = Inf;
else
    optimState.LastWarmup = 0;
end

% Number of stable function evaluations during warmup with small increment
optimState.WarmupStableCount = 0;

% Proposal function for search
if isempty(options.ProposalFcn)
    optimState.ProposalFcn = @(x) proposal_vbmc(x,optimState.PLB,optimState.PUB);
else
    optimState.ProposalFcn = options.ProposalFcn;
end

% Quality of the variational posterior
optimState.R = Inf;

% Start with adaptive sampling
optimState.SkipActiveSampling = false;

% Running mean and covariance of variational posterior in transformed space
optimState.RunMean = [];
optimState.RunCov = [];
optimState.LastRunAvg = NaN; % Last time running average was updated

% Current number of components for variational posterior
optimState.vpK = K;

% Number of variational components pruned in last iteration
optimState.pruned = 0;

% Need to switch from deterministic entropy to stochastic entropy
optimState.EntropySwitch = options.EntropySwitch;
% Only use deterministic entropy if NVARS larger than a fixed number
if nvars < options.DetEntropyMinD
    optimState.EntropySwitch = false;
end

% Tolerance threshold on GP variance (used by some acquisition fcns)
optimState.TolGPVar = options.TolGPVar;

% Copy maximum number of fcn. evaluations, used by some acquisition fcns.
optimState.MaxFunEvals = options.MaxFunEvals;

% By default, apply variance-based regularization to acquisition functions
optimState.VarianceRegularizedAcqFcn = true;

% Setup search cache
optimState.SearchCache = [];

% Set uncertainty handling level
% (0: none; 1: unknwon noise level; 2: user-provided noise)
if options.SpecifyTargetNoise
    optimState.UncertaintyHandlingLevel = 2;  % Provided noise
elseif options.UncertaintyHandling
    optimState.UncertaintyHandlingLevel = 1;  % Infer noise
else
    optimState.UncertaintyHandlingLevel = 0;  % No noise
end

% Empty hedge struct for acquisition functions
if options.AcqHedge; optimState.hedge = []; end

% List of points at the end of each iteration
optimState.iterList.u = [];
optimState.iterList.fval = [];
optimState.iterList.fsd = [];
optimState.iterList.fhyp = [];

optimState.delta = options.Bandwidth*(optimState.PUB-optimState.PLB);

% Posterior tempering temperature
if ~isempty(options.Temperature); T = options.Temperature; else; T = 1; end
if round(T) ~= T || T > 4 || T < 1
    error('vbmc:PosterioTemperature',...
        'OPTIONS.Temperature should be a small positive integer (allowed T = 1,2,3,4).');
end
optimState.temperature = T;

% Deterministic entropy approximation lower/upper factor
optimState.entropy_alpha = options.DetEntropyAlpha;

% Repository of variational solutions
optimState.vp_repo = [];

% Repeated measurement streak
optimState.RepeatedObservationsStreak = 0;

% List of data trimming events
optimState.DataTrimList = [];

% Expanding search bounds
prange = optimState.PUB - optimState.PLB;
optimState.LB_search = max(optimState.PLB - prange*options.ActiveSearchBound,optimState.LB);
optimState.UB_search = min(optimState.PUB + prange*options.ActiveSearchBound,optimState.UB);

%% Initialize Gaussian process settings

optimState.gpCovfun = 1;    % Squared exponential kernel with separate length scales
switch optimState.UncertaintyHandlingLevel
    case 0; optimState.gpNoisefun = [1 0];  % Observation noise for stability
    case 1; optimState.gpNoisefun = [1 2];  % Infer noise
    case 2; optimState.gpNoisefun = [1 1];  % Provided heteroskedastic noise
end
if options.NoiseShaping && optimState.gpNoisefun(2) == 0
    optimState.gpNoisefun(2) = 1;
end
optimState.gpMeanfun = options.gpMeanFun;
switch optimState.gpMeanfun
    case {'zero','const','negquad','se','negquadse','negquadfixiso','negquadfix','negquadsefix','negquadonly','negquadfixonly','negquadlinonly','negquadmix'}
    otherwise
        error('vbmc:UnknownGPmean', ...
            'Unknown/unsupported GP mean function. Supported mean functions are ''zero'', ''const'', ''negquad'', and ''se''.');
end
optimState.intMeanfun = options.gpIntMeanFun;
optimState.gpOutwarpfun = options.gpOutwarpFun;
if ischar(optimState.gpOutwarpfun)
    switch lower(optimState.gpOutwarpfun)
        case {'[]','off','no','none','0'}; optimState.gpOutwarpfun = [];
        otherwise
            optimState.gpOutwarpfun = str2func(optimState.gpOutwarpfun);
    end
end

% Starting threshold on y for output warping
if options.FitnessShaping || ~isempty(optimState.gpOutwarpfun)
    optimState.OutwarpDelta = options.OutwarpThreshBase;
else
    optimState.OutwarpDelta = [];    
end

%% Get warnings state

optimState.DefaultWarnings.singularMatrix = warning('query','MATLAB:singularMatrix');
warning('off',optimState.DefaultWarnings.singularMatrix.identifier);