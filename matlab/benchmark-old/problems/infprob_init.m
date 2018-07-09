function probstruct = infprob_init(probset,prob,subprob,noise,id,options)
%INFPROB_INIT Initialize inference problem structure.

% Initialize current problem
problemfun = str2func(['infprob_' probset]);
probstruct = problemfun(prob,subprob,noise,id,options);

% Import fields from ProbInfo
fields = {'D','LB','UB','PLB','PUB','Mean','Cov','Mode','lnZ'};
for iField = 1:numel(fields)
    if isfield(probstruct.ProbInfo,fields{iField})
        probstruct.(fields{iField}) = probstruct.ProbInfo.(fields{iField});
    else
        probstruct.(fields{iField}) = [];
    end
end

% Assign default values to problem struct
defprob = infbench_defaults('problem',probstruct,options);
for f = fieldnames(defprob)'
    if ~isfield(probstruct,f{:}) || isempty(probstruct.(f{:}))
        probstruct.(f{:}) = defprob.(f{:});
    end
end

% Assign default values to OPTIONS struct (useful if called externally)
defopts = infbench_defaults('options');
for f = fieldnames(defopts)'
    if ~isfield(options,f{:}) || isempty(options.(f{:}))
        options.(f{:}) = defopts.(f{:});
    end
end

% Check if in debug mode
probstruct.Debug = options.Debug;

% Simulated noise
probstruct.NoiseIncrement = 0;  % Default noise is homoskedastic
if isempty(probstruct.NoiseSigma)
    if isempty(probstruct.Noise)
        probstruct.NoiseSigma = 0; % No noise
    else
        switch(probstruct.Noise)
            case 'lo'; probstruct.NoiseSigma = 0.25; % Low noise
            case 'me'; probstruct.NoiseSigma = 1; % Medium noise
            case 'hi'; probstruct.NoiseSigma = 4; % High noise
            case 'helo'     % Low heteroskedastic noise
                probstruct.NoiseSigma = 0.25;
                probstruct.NoiseIncrement = 0.05;
            case 'heme'     % Medium heteroskedastic noise
                probstruct.NoiseSigma = 1;
                probstruct.NoiseIncrement = 0.1;
            case 'hehi'     % High heteroskedastic noise
                probstruct.NoiseSigma = 4;
                probstruct.NoiseIncrement = 0.2;
        end
        if isempty(probstruct.NoiseEstimate)
            probstruct.NoiseEstimate = [probstruct.NoiseSigma, 0.2];
        else
            probstruct.NoiseEstimate(1) = sqrt(probstruct.NoiseEstimate(1)^2 + probstruct.NoiseSigma^2);
        end
    end
end

% Maximum function evaluations
probstruct.MaxFunEvals = probstruct.MaxFunEvals*options.MaxFunEvalMultiplier;
probstruct.TotalMaxFunEvals = probstruct.MaxFunEvals;
probstruct.Verbose = evalbool(options.Display);

if isempty(probstruct.SaveTicks)
     probstruct.SaveTicks = [5:5:500, 510:10:1000, 1020:20:probstruct.TotalMaxFunEvals];
     probstruct.SaveTicks(probstruct.SaveTicks > probstruct.TotalMaxFunEvals) = [];
end

% Load minimum from file
% filename = ['mindata_' probstruct.ProbSet '_' probstruct.Prob '.mat'];
% try    
%     temp = load(filename);
%     f = temp.mindata.(['f_' probstruct.SubProb]);
%     if ~isfield(probstruct,'TrueMinFval') || isempty(probstruct.TrueMinFval) || ~isfinite(probstruct.TrueMinFval)
%         probstruct.TrueMinFval = f.MinFval;
%     end
%     if ~isfield(probstruct,'TrueMinX') || isempty(probstruct.TrueMinX) || any(~isfinite(probstruct.TrueMinX))
%         probstruct.TrueMinX = f.BestX;
%     end    
% catch
%     warning('Could not load optimum location/value from file.');
% end

% Center and rescale variables (potentially transform to log space)
if evalbool(options.ScaleVariables)
    probstruct.trinfo = warpvars(probstruct.D,probstruct.LB,probstruct.UB,probstruct.PLB,probstruct.PUB);
    if any(probstruct.trinfo.type > 0); error('Nonlinear transforms unsupported yet.'); end
    probstruct.LB = warpvars(probstruct.LB,'d',probstruct.trinfo);
    probstruct.UB = warpvars(probstruct.UB,'d',probstruct.trinfo);
    probstruct.PLB = warpvars(probstruct.PLB,'d',probstruct.trinfo);
    probstruct.PUB = warpvars(probstruct.PUB,'d',probstruct.trinfo);
    if all(isfinite(probstruct.Mode))
        probstruct.Mode = warpvars(probstruct.Mode,'d',probstruct.trinfo);
    end
    if all(isfinite(probstruct.Mean))
        probstruct.Mean = warpvars(probstruct.Mean,'d',probstruct.trinfo);
    end
    if all(isfinite(probstruct.Cov(:)))
        probstruct.Cov = diag(1./probstruct.trinfo.delta)*probstruct.Cov*diag(1./probstruct.trinfo.delta);        
    end
    if isfield(probstruct.ProbInfo,'Post')
        probstruct.Post = probstruct.ProbInfo.Post;
        probstruct.Post.Mean = warpvars(probstruct.ProbInfo.Post.Mean,'d',probstruct.trinfo);        
        probstruct.Post.Mode = warpvars(probstruct.ProbInfo.Post.Mode,'d',probstruct.trinfo);        
        probstruct.Post.Cov = diag(1./probstruct.trinfo.delta)*probstruct.ProbInfo.Post.Cov*diag(1./probstruct.trinfo.delta);
    end
end

% Store Gaussian prior mean and variance
probstruct.PriorMean = 0.5*(probstruct.PLB+probstruct.PUB);
probstruct.PriorVar = (0.5*(probstruct.PUB-probstruct.PLB)).^2;

% if isfield(probstruct,'TrueMinFval') && isfinite(probstruct.TrueMinFval)
%     display(['Known minimum function value: ' num2str(probstruct.TrueMinFval,'%.3f')]);
% end

% Compute initial optimization point
probstruct.InitPoint = [];
probstruct.StartFromMode = options.StartFromMode;
if probstruct.StartFromMode
    if isfield(probstruct,'Post') ...
            && isfield(probstruct.Post,'Mode') ...
            && ~isempty(probstruct.Post.Mode) ...
            && all(isfinite(probstruct.Post.Mode))
        Mode = probstruct.Post.Mode;
    else
        Mode = probstruct.Mode;
    end
    if ~all(isfinite(Mode)) || isempty(Mode)
        warning('Mode not provided or invalid. Starting from prior or from random starting point.');
    else
        probstruct.InitPoint = Mode;
    end
end
if isempty(probstruct.InitPoint)
    if ~isempty(probstruct.PriorMean) && all(isfinite(probstruct.PriorMean))
        probstruct.InitPoint = probstruct.PriorMean;
    else
        probstruct.InitPoint = rand(1,probstruct.D).*(probstruct.PUB-probstruct.PLB) + probstruct.PLB;
    end
end

% Compute evaluation time and function noise
tic; f1 = infbench_func(probstruct.InitPoint,probstruct,1); toc
tic; f2 = infbench_func(probstruct.InitPoint,probstruct,1); toc
% [f1 f2]

% Assess whether function is intrinsically noisy
if ~isfield(probstruct,'IntrinsicNoisy') || isempty(probstruct.IntrinsicNoisy)
    probstruct.IntrinsicNoisy = (f1 ~= f2);
end

%--------------------------------------------------------------------------
function tf = evalbool(s)
%EVALBOOL Evaluate argument to a bool

if ~ischar(s) % S may not and cannot be empty
        tf = s;
        
else % Evaluation of string S
    if strncmpi(s, 'yes', 3) || strncmpi(s, 'on', 2) ...
        || strncmpi(s, 'true', 4) || strncmp(s, '1 ', 2)
            tf = 1;
    elseif strncmpi(s, 'no', 2) || strncmpi(s, 'off', 3) ...
        || strncmpi(s, 'false', 5) || strncmp(s, '0 ', 2)
            tf = 0;
    else
        try tf = evalin('caller', s); catch
            error(['String value "' s '" cannot be evaluated']);
        end
        try tf ~= 0; catch
            error(['String value "' s '" cannot be evaluated reasonably']);
        end
    end

end