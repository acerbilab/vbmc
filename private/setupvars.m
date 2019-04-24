function [vp,optimState] = setupvars(x0,LB,UB,PLB,PUB,K,optimState,options,prnt)
%INITVARS Initialize variational posterior, transforms and variables for VBMC.

nvars = size(LB,2);

% Starting point
if any(~isfinite(x0))   % Invalid/not provided starting point
    if prnt > 0
        fprintf('Initial starting point is invalid or not provided. Starting from center of plausible region.\n');
    end
    x0 = 0.5*(PLB + PUB);       % Midpoint
end

optimState.LB_orig = LB;
optimState.UB_orig = UB;
optimState.PLB_orig = PLB;
optimState.PUB_orig = PUB;

% Transform variables
trinfo = warpvars(nvars,LB,UB,PLB,PUB);
trinfo.x0_orig = x0;
if ~isfield(trinfo,'R_mat'); trinfo.R_mat = []; end
if ~isfield(trinfo,'scale'); trinfo.scale = []; end

optimState.LB = warpvars(LB,'dir',trinfo);
optimState.UB = warpvars(UB,'dir',trinfo);
optimState.PLB = warpvars(PLB,'dir',trinfo);
optimState.PUB = warpvars(PUB,'dir',trinfo);

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

x0 = warpvars(x0,'dir',trinfo);

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
vp.optimize_lambda = true;
if options.Warmup
    vp.optimize_mu = true;
    vp.optimize_weights = false;
else
    vp.optimize_mu = logical(options.VariableMeans);
    vp.optimize_weights = logical(options.VariableWeights);
end
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

% Maximum value
optimState.ymax = -Inf;

% Does the starting cache contain function values?
optimState.Cache.active = any(isfinite(optimState.Cache.y_orig));

% When was the last warping action performed (number of training inputs)
optimState.LastWarping = 0;
optimState.LastNonlinearWarping = 0;

% Number of warpings performed
optimState.WarpingCount = 0;
optimState.WarpingNonlinearCount = 0;

% Perform rotoscaling at the end of iteration
optimState.redoRotoscaling = false;

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

% Number of stable iteration of small increment
optimState.WarmupStableIter = 0;

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

% Running covariance of GP hyperparameter posterior
optimState.RunHypCov = [];

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

% Setup search cache
optimState.SearchCache = [];

% List of points at the end of each iteration
optimState.iterList.u = [];
optimState.iterList.fval = [];
optimState.iterList.fsd = [];
optimState.iterList.fhyp = [];

%% Get warnings state

optimState.DefaultWarnings.singularMatrix = warning('query','MATLAB:singularMatrix');
warning('off',optimState.DefaultWarnings.singularMatrix.identifier);