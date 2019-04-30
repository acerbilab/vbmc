function [options,cmaes_opts] = setupoptions(nvars,defopts,options)
%SETUPOPTIONS Initialize OPTIONS struct.

D = nvars;  % Both D and NVARS are accepted as number of dimensions

% Assign default values to OPTIONS struct
for f = fieldnames(defopts)'
    if ~isfield(options,f{:}) || isempty(options.(f{:}))
        options.(f{:}) = defopts.(f{:});
    end
end

% Remove comments and trailing empty spaces from options fields
for f = fieldnames(options)'
    if ischar(options.(f{:}))
        idx = find(options.(f{:}) == '%',1);
        if ~isempty(idx); options.(f{:})(idx:end) = []; end        
        idx = find(options.(f{:}) ~= ' ',1,'last');
        if ~isempty(idx); options.(f{:})(idx+1:end) = []; end                
    end
end

% OPTIONS fields that need to be evaluated
evalfields = {'MaxFunEvals','MaxIter','FunEvalStart','FunEvalsPerIter','SGDStepSize','RetryMaxFunEvals','SearchAcqFcn','ProposalFcn',...
    'NSsearch','NSent','NSentFast','NSentFine','NSelbo','ElboStarts','VariableMeans','VariableWeights', ...
    'NSgpMax','NSgpMaxWarmup','NSgpMaxMain','WarmupNoImproThreshold','WarmupCheckMax',...
    'StableGPSampling','StableGPSamples','TolGPVar','KfunMax','Kwarmup','AdaptiveK',...
    'HPDFrac', 'WarpRotoScaling', 'WarpNonlinear', 'WarpEpoch', 'WarpCovReg', 'WarpMinFun', 'WarpNonlinearEpoch', 'WarpNonlinearMinFun', ...
    'ELCBOWeight','TolLength','Fvals','OptimToolbox','RankCriterion', ...
    'NoiseObj','CacheSize','CacheFrac','TolFunStochastic','GPStochasticStepsize', ...
    'TolSD','TolsKL','TolStableIters','TolStableEntropyIters','TolStableExceptions',...
    'KLgauss','TrueMean','TrueCov',...
    'MinFunEvals','MinIter','HeavyTailSearchFrac','MVNSearchFrac','HPDSearchFrac',...
    'AlwaysRefitVarPost','VarParamsBack','Plot' ...
    'Warmup','WarmupOptions','StopWarmupThresh','WarmupKeepThreshold','SearchCMAES','SearchCMAESVPInit','TolStableWarmup','MomentsRunWeight', ...
    'ELCBOmidpoint','GPRetrainThreshold','NSelboIncr','TolImprovement','ELCBOImproWeight', ...
    'GPSampleThin','GPSampleWidths','HypRunWeight','WeightedHypCov','TolCovWeight', ...
    'CovSampleThresh','AltMCEntropy','DetEntTolOpt','EntropySwitch','EntropyForceSwitch','DetEntropyMinD', ...
    'TolConLoss','BestSafeSD','BestFracBack',...
    'UncertaintyHandling','NoiseSize','TolWeight','WeightPenalty',...
    'SkipActiveSamplingAfterWarmup','AnnealedGPMean','ConstrainedGPMean','EmpiricalGPPrior','FeatureTest','SearchCacheFrac',...
    'VarActiveSample','BOWarmup'
    };

% Evaluate string options
for f = evalfields
    if ischar(options.(f{:}))
        try
            options.(f{:}) = eval(options.(f{:}));
        catch
            try
                options.(f{:}) = evalbool(options.(f{:}));
            catch
                error('vbmc:BadOptions', ...
                    'Cannot evaluate OPTIONS field "%s".', f{:});
            end
        end
    end
end

% Make cell arrays
cellfields = {'SearchAcqFcn'};
for f = cellfields
    if ischar(options.(f{:})) || isa(options.(f{:}), 'function_handle')
        options.(f{:}) = {options.(f{:})};
    end
end

% Check if MATLAB's Optimization Toolbox (TM) is available
if isempty(options.OptimToolbox)
    if exist('fmincon.m','file') && exist('fminunc.m','file') && exist('optimoptions.m','file') ...
            && license('test', 'optimization_toolbox')
        options.OptimToolbox = 1;
    else
        options.OptimToolbox = 0;
        warning('vbmc:noOptimToolbox', 'Could not find the Optimization Toolbox�. Using alternative optimization functions. This will slightly degrade performance. If you do not wish this message to appear, set OPTIONS.OptimToolbox = 0.');
    end
end

% % Check options
if round(options.MaxFunEvals) ~= options.MaxFunEvals || options.MaxFunEvals <= 0
     error('vbmc:OptionsError','OPTIONS.MaxFunEvals needs to be a positive integer.');
end
if round(options.MaxIter) ~= options.MaxIter || options.MaxIter <= 0
     error('vbmc:OptionsError','OPTIONS.MaxIter needs to be a positive integer.');
end
if options.MaxIter < options.MinIter
    warning('vbmc:MinIter', ...
        ['OPTIONS.MaxIter cannot be smaller than OPTIONS.MinIter. Changing the value of OPTIONS.MaxIter to ' num2str(options.MinIter) '.']);
    options.MaxIter = options.MinIter;
end
if options.MaxFunEvals < options.MinFunEvals
    warning('vbmc:MinFunEvals', ...
        ['OPTIONS.MaxFunEvals cannot be smaller than OPTIONS.MinFunEvals. Changing the value of OPTIONS.MaxFunEvals to ' num2str(options.MinFunEvals) '.']);
    options.MinFunEvals = options.MinFunEvals;
end

if ~isempty(options.NoiseSize) && options.NoiseSize(1) <= 0
     error('vbmc:OptionsError','OPTIONS.NoiseSize, if specified, needs to be a positive scalar for numerical stability.');
end
   
% Setup options for CMA-ES optimization
cmaes_opts = cmaes_modded('defaults');
cmaes_opts.TolX = '1e-11*max(insigma)';
cmaes_opts.TolFun = 1e-12;
cmaes_opts.TolHistFun = 1e-13;
cmaes_opts.EvalParallel = 'yes';
cmaes_opts.DispFinal = 'off';
cmaes_opts.SaveVariables = 'off';
cmaes_opts.DispModulo = Inf;
cmaes_opts.LogModulo = 0;
cmaes_opts.CMA.active = 1;      % Use Active CMA (generally better)

