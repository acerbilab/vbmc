function [options,updated] = setupoptions_vbmc(nvars,defopts,options)
%SETUPOPTIONS_VBMC Initialize OPTIONS struct for VBMC.

D = nvars;  % Both D and NVARS are accepted as number of dimensions

% Assign default values to OPTIONS struct, keep track of the ones updated
updated = [];
for f = fieldnames(defopts)'
    if ~isfield(options,f{:}) || isempty(options.(f{:}))
        options.(f{:}) = defopts.(f{:});
    else
        updated{end+1} = f{:};
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
    'NSsearch','NSent','NSentFast','NSentFine','NSentBoost','NSentFastBoost','NSentFineBoost','NSentActive','NSentFastActive','NSentFineActive',...
    'NSelbo','ElboStarts','VariableMeans','VariableWeights','gpIntMeanFun', ...
    'NSgpMax','NSgpMaxWarmup','NSgpMaxMain','WarmupNoImproThreshold','WarmupCheckMax',...
    'StableGPSampling','StableGPvpK','StableGPSamples','TolGPVar','TolGPVarMCMC','KfunMax','Kwarmup','AdaptiveK',...
    'HPDFrac', 'WarpRotoScaling', 'WarpNonlinear', 'WarpCovReg', ...
    'ELCBOWeight','TolLength','Fvals','OptimToolbox','RankCriterion', ...
    'CacheSize','CacheFrac','TolFunStochastic','GPStochasticStepsize', ...
    'TolSD','TolsKL','TolStableCount','TolStableEntropyIters','TolStableExcptFrac',...
    'KLgauss','TrueMean','TrueCov',...
    'MinFunEvals','MinIter','HeavyTailSearchFrac','MVNSearchFrac','HPDSearchFrac','BoxSearchFrac',...
    'AlwaysRefitVarPost','VarParamsBack','Plot' ...
    'Warmup','WarmupOptions','StopWarmupThresh','WarmupKeepThreshold','WarmupKeepThresholdFalseAlarm','SearchCMAESVPInit','SearchCMAESbest','SearchMaxFunEvals','TolStableWarmup','MomentsRunWeight', ...
    'ELCBOmidpoint','GPRetrainThreshold','NSelboIncr','TolImprovement','ELCBOImproWeight', ...
    'GPSampleThin','GPSampleWidths','HypRunWeight','WeightedHypCov','TolCovWeight','GPTolOpt','GPTolOptMCMC','GPTolOptActive','GPTolOptMCMCActive', ...
    'CovSampleThresh','AltMCEntropy','DetEntTolOpt','EntropySwitch','EntropyForceSwitch','DetEntropyMinD', ...
    'TolConLoss','BestSafeSD','BestFracBack',...
    'UncertaintyHandling','NoiseSize','TolWeight','PruningThresholdMultiplier','WeightPenalty',...
    'SkipActiveSamplingAfterWarmup','AnnealedGPMean','ConstrainedGPMean','EmpiricalGPPrior','FeatureTest','SearchCacheFrac',...
    'VarActiveSample','BOWarmup','gpQuadraticMeanBound','Bandwidth',...
    'OutwarpThreshBase','OutwarpThreshMult','OutwarpThreshTol','IntegerVars','Temperature', ...
    'SpecifyTargetNoise','SeparateSearchGP','MaxRepeatedObservations','RepeatedAcqDiscount',...
    'NoiseShaping','NoiseShapingThreshold','NoiseShapingFactor',...
    'AcqHedge','AcqHedgeIterWindow','AcqHedgeDecay','MinFinalComponents',...
    'ActiveVariationalSamples','ScaleLowerBound','GPTrainNinit','GPTrainNinitFinal', ...
    'DetEntropyAlpha','UpdateRandomAlpha','VariationalInitRepo','AdaptiveEntropyAlpha','MaxIterStochastic', ...
    'StopWarmupReliability','UpperGPLengthFactor','SampleExtraVPMeans','OptimisticVariationalBound' ...
    'ActiveImportanceSamplingVPSamples','ActiveImportanceSamplingBoxSamples','ActiveImportanceSamplingMCMCSamples','ActiveImportanceSamplingMCMCThin', ...
    'ActiveSearchBound','IntegrateGPMean','GPLengthPriorMean','GPLengthPriorStd','TolGPNoise', ...
    'WarpEveryIters','IncrementalWarpDelay','WarpTolReliability','WarpMinK', ...
    'TolBoundX','WarpRotoCorrThresh','FitnessShaping','ActiveSampleFullUpdatePastWarmup', ...
    'ActiveSampleFullUpdateThreshold','ActiveSampleVPUpdate','ActiveSampleGPUpdate', ...
    'RecomputeLCBmax','ActiveSamplefESSThresh','ActiveImportanceSamplingfESSThresh', ...
    'DoubleGP' ...
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

% Use SPECIFYTARGETNOISE to set UNCERTAINTYHANDLING
if isempty(options.UncertaintyHandling)
    options.UncertaintyHandling = options.SpecifyTargetNoise;
end

if ~isempty(options.NoiseSize) && options.NoiseSize(1) <= 0
     error('vbmc:OptionsError','OPTIONS.NoiseSize, if specified, needs to be a positive scalar for numerical stability.');
end

if ~options.UncertaintyHandling && options.SpecifyTargetNoise
    error('vbmc:SpecifiedNoise','If OPTIONS.SpecifyTargetNoise is active, OPTIONS.UncertaintyHandling should be activated too.');
end

if options.SpecifyTargetNoise && ~isempty(options.NoiseSize)
    warning('vbmc:SpecifiedNoiseAndSize','If OPTIONS.SpecifyTargetNoise is active, OPTIONS.NoiseSize is ignored (leave OPTIONS.NoiseSize empty to suppress this warning).');
end

% Change default OPTIONS for UncertaintyHandling, if not specified
if options.UncertaintyHandling    
    if ~any(strcmp('MaxFunEvals',updated))
        options.MaxFunEvals = ceil(options.MaxFunEvals*1.5);
    end
    if ~any(strcmp('TolStableCount',updated))
        options.TolStableCount = ceil(options.TolStableCount*1.5);
    end
    if ~any(strcmp('ActiveSampleGPUpdate',updated))
        options.ActiveSampleGPUpdate = true;
    end
    if ~any(strcmp('ActiveSampleVPUpdate',updated))
        options.ActiveSampleVPUpdate = true;
    end
    if ~any(strcmp('SearchAcqFcn',updated))
        options.SearchAcqFcn = {@acqviqr_vbmc};
    end
%     if ~any(strcmp('TolStableWarmup',updated))
%         options.TolStableWarmup = options.TolStableWarmup*2;
%     end    
end
   
% Setup options for CMA-ES optimization
cmaes_opts = cmaes_modded('defaults');
% cmaes_opts.MaxFunEvals = 1000*D;
cmaes_opts.TolX = '1e-11*max(insigma)';
cmaes_opts.TolFun = 1e-12;
cmaes_opts.TolHistFun = 1e-13;
cmaes_opts.EvalParallel = 'yes';
cmaes_opts.DispFinal = 'off';
cmaes_opts.SaveVariables = 'off';
cmaes_opts.DispModulo = Inf;
cmaes_opts.LogModulo = 0;
cmaes_opts.CMA.active = 1;      % Use Active CMA (generally better)

options.CMAESopts = cmaes_opts;

