function varargout = infbench_func(x,probstruct,debug,iter,toffset)
%INFBENCH_FUNC Wrapper for inference benchmark target log pdf.

persistent history;     % Log history of function calls

if nargin < 1   % No arguments return history log
    historyOut = history;
    if ~isempty(historyOut)     % Remove temporary fields
        historyOut = rmfield(historyOut,'FuncHandle');
        historyOut = rmfield(historyOut,'FuncTimeTemp');
    end

    % Allocate vectors for the inference algorithms to store their output
    Nticks = numel(historyOut.SaveTicks);
    historyOut.Output.N = NaN(1,Nticks);
    historyOut.Output.lnZs = NaN(1,Nticks);
    historyOut.Output.lnZs_var = NaN(1,Nticks);
    historyOut.Output.gsKL = NaN(1,Nticks);
    historyOut.Output.Mean = NaN(0,history.D);
    historyOut.Output.Cov = NaN(0,history.D,history.D);
    historyOut.Output.Mode = NaN(Nticks,history.D);
    
    varargout = {historyOut};
    return;
end

if isstruct(x)  % Swapped variable order
    temp = probstruct;
    probstruct = x;
    x = temp;
end

if nargin < 3 || isempty(debug); debug = 0; end
if nargin < 4; iter = []; end
if nargin < 5 || isempty(toffset); toffset = 0; end

if exist('istable','file') == 2 && istable(x)
    xnew = zeros(size(x));
    for iVar = 1:numel(x)
        xnew(iVar) = x.(['x' num2str(iVar)]);
    end
    x = xnew;
end

% Update current run number
if debug && ~isempty(iter)
    if iter > 1
        history.CurrentIter = iter;
        history.StartIterFunCalls = history.FunCalls;
        history.TimeOffset = history.TimeOffset + toffset;
        if size(history.ThresholdsHitPerIter,1) < iter
            history.ThresholdsHitPerIter = [history.ThresholdsHitPerIter; ...
                Inf(1, numel(history.Thresholds))];        
        end
    end
    return;
end

x = x(:)';  % Row vector in

if isempty(history)     % First function call, initialize log
    
    % Problem information
    history.ProbSet = probstruct.ProbSet;
    history.Prob = probstruct.Prob;
    history.SubProb = probstruct.SubProb;
    history.Id = probstruct.Id;
    history.D = probstruct.D;
    history.Noise = probstruct.Noise;
    if isfield(probstruct,'NoiseSigma') && ~isempty(probstruct.NoiseSigma)
        history.NoiseSigma = probstruct.NoiseSigma;
    else
        history.NoiseSigma = 0;        
    end
    if isfield(probstruct,'NoiseIncrement') && ~isempty(probstruct.NoiseIncrement)
        history.NoiseIncrement = probstruct.NoiseIncrement;
    else
        history.NoiseIncrement = 0;        
    end
    history.Func = probstruct.func;
    history.FuncHandle = str2func(probstruct.func);     % Removed later
    history.Mode = probstruct.Mode;
    if isfield(probstruct,'TotalMaxFunEvals')
        history.TotalMaxFunEvals = probstruct.TotalMaxFunEvals;
    else
        history.TotalMaxFunEvals = [];        
    end
    
    % Optimization record
    if ~isfield(probstruct,'SaveTicks'); probstruct.SaveTicks = []; end
    Nticks = numel(probstruct.SaveTicks);
    history.ElapsedTime = NaN(1,Nticks);
    history.FuncTime = NaN(1,Nticks);
    history.FuncTimeTemp = 0;   % Temporary variable to store function time
    history.MaxScores = NaN(1,Nticks);
    history.BestX = NaN(1,history.D);
    if isfield(probstruct,'Thresholds')
        history.Thresholds = probstruct.Thresholds;
        history.ThresholdsHit = Inf(1, numel(history.Thresholds));
        history.ThresholdsHitPerIter = Inf(1, numel(history.Thresholds));
    end
    history.MaxScore = -Inf;
    history.FunCalls = 0;
    history.StartIterFunCalls = 0;
    history.SaveTicks = probstruct.SaveTicks;
    if isfield(probstruct,'trinfo'); history.trinfo = probstruct.trinfo; end
    history.Clock = tic;
    history.TimeOffset = 0; % Time to be subtracted from clock    
    history.CurrentIter = 1;
end

% Check that x is within the hard bounds
isWithinBounds = (x >= probstruct.LB) & (x <= probstruct.UB);
if ~all(isWithinBounds); x = min(max(x,probstruct.LB),probstruct.UB); end

% Call function
if isfield(history,'FuncHandle')
    func = history.FuncHandle;
else
    func = str2func(history.Func);
end

% If requested, compute log prior in benchmark space
if isfield(probstruct,'AddLogPrior') && probstruct.AddLogPrior
    lnp = infbench_lnprior(x,probstruct);
else
    lnp = [];
end

if isfield(probstruct,'PriorMean') && ~isempty(probstruct.PriorMean)
    addJacobian = 0;    % Jacobian is already included in the rescaled prior
else
    addJacobian = 1;
end

if isfield(probstruct,'trinfo')
    dy = warpvars(x,'logpdf',probstruct.trinfo);
    x = warpvars(x,'inv',probstruct.trinfo);
end

% Computational precision (minimum 0; 1 default precision)
if ~isfield(probstruct,'Precision') || isempty(probstruct.Precision)
    probstruct.Precision = 1;
end

% Changing precision
% probstruct.Precision = min(4/3*history.FunCalls/probstruct.TotalMaxFunEvals, 1);

% Check if need to pass probstruct
try
    if strfind(probstruct.func,'probstruct_')
        tfun = tic; fval_orig = func(x,probstruct); t = toc(tfun);
    else
        tfun = tic; fval_orig = func(x); t = toc(tfun);
    end
    if isfield(probstruct,'trinfo') && addJacobian
        fval = fval_orig + dy;
    else
        fval = fval_orig;
    end
    if ~isempty(lnp)    % Add log prior if needed
        fval = fval + lnp;
    end
catch except
    warning(['Error in benchmark function ''' history.Func '''.' ...
        ' Message: ''' except.message '''']);
    rethrow(except);
    x
    fval_orig = NaN; fval = NaN;
    t = 0;
    except.stack.file
    except.stack.line
end

% Value for NaN result
if ~isfinite(fval) && ...
        ~isempty(probstruct.NonAdmissibleFuncValue) && ...
        ~isnan(probstruct.NonAdmissibleFuncValue)
    fval = probstruct.NonAdmissibleFuncValue;
end

if ~debug
    % Update records (not in debug mode)
    history.FunCalls = history.FunCalls + 1;
    history.FuncTimeTemp = history.FuncTimeTemp + t;
    if fval_orig > history.MaxScore
        history.MaxScore = fval_orig;
        history.BestX = x;
    end
    
    % Update history log every SavePeriod function calls
    idx = find(history.FunCalls == history.SaveTicks,1);
    if ~isempty(idx)
        history.FuncTime(idx) = history.FuncTimeTemp;
        history.FuncTimeTemp = 0;
        history.ElapsedTime(idx) = toc(history.Clock) - history.TimeOffset;
        history.MaxScores(idx) = history.MaxScore;
    end

    % Add artificial noise (not in debug mode)
    sigma = history.NoiseSigma + history.NoiseIncrement*abs(fval);
    if sigma > 0
        fval = fval + randn()*sigma;
    end
end

% x

varargout = {fval};    

end