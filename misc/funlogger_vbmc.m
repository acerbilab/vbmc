function [fval,optimState,idx] = funlogger_vbmc(fun,x,optimState,state,varargin)
%FUNLOGGER_VBMC Call objective function and do some bookkeeping.
%   [~,OPTIMSTATE] = FUNLOGGER_VBMC(FUN,NVARS,OPTIMSTATE,'init') starts logging
%   function FUN with starting point X and optimization struct OPTIMSTATE.
%
%   [~,OPTIMSTATE] = FUNLOGGER_VBMC(FUN,NVARS,OPTIMSTATE,'init',NMAX) stores up 
%   to NMAX function values (default NMAX=1e4).
%
%   [FVAL,OPTIMSTATE] = FUNLOGGER_VBMC(FUN,X,OPTIMSTATE,'iter') evaluates 
%   function FUN at X with optimization struct OPTIMSTATE, returns function 
%   value FVAL. FUN must take a vector input and return a scalar value and, 
%   optionally, the (estimated) SD of the returned value (if heteroskedastic 
%   noise handling is on).
%
%   [FVAL,OPTIMSTATE] = FUNLOGGER_VBMC(FUN,X,OPTIMSTATE,'single') as 'iter' 
%   but does not store function values in the cache.
%
%   [~,OPTIMSTATE] = FUNLOGGER_VBMC(FUN,X,OPTIMSTATE,'done') finalizes 
%   stored function values.

%   Luigi Acerbi 2019

fval = [];

% Switch warnings on
for ff = fields(optimState.DefaultWarnings)'
    warning(optimState.DefaultWarnings.(ff{:}).state,optimState.DefaultWarnings.(ff{:}).identifier);
end

switch lower(state)
    case 'init' % Start new function logging session
    
        nvars = x;

        % Number of stored function values
        if nargin > 4; nmax = varargin{1}; else; nmax = []; end
        if isempty(nmax); nmax = 500; end
        
        % Read noise level
        noise_flag = optimState.UncertaintyHandlingLevel > 0;

        optimState.funccount = 0;
        optimState.cachecount = 0;

        % Passing empty or new OPTIMSTATE
        if ~isfield(optimState,'X_orig') || isempty(optimState.X_orig)    
            optimState.X_orig = NaN(nmax,nvars);
            optimState.y_orig = NaN(nmax,1);
            optimState.X = NaN(nmax,nvars);
            optimState.y = NaN(nmax,1);
            if noise_flag; optimState.S = NaN(nmax,1); end
            optimState.nevals = zeros(nmax,1);
            optimState.Xn = 0;                    % Last filled entry
            optimState.Xmax = nmax;               % Maximum entry index
            optimState.X_flag = false(nmax,1);
            optimState.ymax = -Inf;
            
        else % Receiving previous evaluations (e.g. from previous run)
            
            error('Previous function evaluations not supported yet.');
            
            % Find last evaluated point
            optimState.Xmax = find(~isnan(optimState.Y),1,'last');
            if ~isfield(optimState,'Xn') || isempty(optimState.Xn)
                optimState.Xn = optimState.Xmax;
            end
            
            % Current cache is smaller than previous one, keep most recent
            % evaluated points
            if optimState.Xmax > nmax
                optimState.X_orig = circshift(optimState.X_orig,-optimState.Xn);
                optimState.X_orig = optimState.X_orig(end-nmax+1:end,:);
                optimState.y_orig = circshift(optimState.y_orig,-optimState.Xn);
                optimState.y_orig = optimState.y_orig(end-nmax+1:nmax,:);
                if noise_flag
                    optimState.S = circshift(optimState.S,-optimState.Xn);
                    optimState.S = optimState.S(end-nmax+1:nmax,:);
                end
                optimState.Xn = 0;
                optimState.Xmax = nmax;
            else
                offset = nmax - size(optimState.X_orig,1);
                optimState.X_orig = [optimState.X_orig; NaN(offset,nvars)];
                optimState.y_orig = [optimState.y_orig; NaN(offset,1)];
                if noise_flag
                    optimState.S = [optimState.S; NaN(offset,1)];
                end
            end
            optimState.X = warpvars_vbmc(optimState.X_orig,'d',optimState.trinfo);
        end
        optimState.funevaltime = NaN(nmax,1);
        optimState.totalfunevaltime = 0;
    
    case {'iter','single'} % Evaluate function (and store output for 'iter')

        x_orig = warpvars_vbmc(x,'inv',optimState.trinfo);    % Convert back to original space
        % Heteroscedastic noise?
        noise_flag = optimState.UncertaintyHandlingLevel > 0;
        
        try
            funtime = tic;
            if noise_flag && optimState.UncertaintyHandlingLevel == 2
                [fval_orig,fsd] = fun(x_orig);
            else
                fval_orig = fun(x_orig);
                if noise_flag
                    fsd = 1;
                else
                    fsd = [];
                end
            end
            t = toc(funtime);
            
            % Check returned function value
            %if isscalar(fval_orig) && isreal(fval_orig) && fval_orig == -Inf
            %    warning(['funlogger_vbmc:InfiniteFuncValue',...
            %        'The function returned -Inf as function value, which should not be allowed. Trying to continue, but results might be affected.'])
            %    fval_orig = log(realmin);
            if ~isscalar(fval_orig) || ~isfinite(fval_orig) || ~isreal(fval_orig)
                error(['funlogger_vbmc:InvalidFuncValue',...
                    'The returned function value must be a finite real-valued scalar (returned value: ' mat2str(fval_orig) ').']);
            end
            
            % Check returned function SD
            if noise_flag && ...
                    (~isscalar(fsd) || ~isfinite(fsd) || ~isreal(fsd) || fsd <= 0.0)
                error(['funlogger_vbmc:InvalidNoiseValue',...
                    'The returned estimated SD (second function output) must be a finite, positive real-valued scalar (returned SD: ' mat2str(fsd) ').']);
            end
            
            % Tempered posterior
            if isfield(optimState,'temperature') && ~isempty(optimState.temperature)
                fval_orig = fval_orig / optimState.temperature;
                fsd = fsd / optimState.temperature;
            end            
            
        catch fun_error
            warning(['funlogger_vbmc:FuncError',...
                'Error in executing the logged function ''' func2str(fun) ''' with input: ' mat2str(x)]);
            rethrow(fun_error);
        end
        
        % Update function records
        optimState.funccount = optimState.funccount + 1;
        if strcmpi(state, 'iter')
            [optimState,fval,idx] = record(optimState,x_orig,x,fval_orig,t,fsd);
        end
        optimState.totalfunevaltime = optimState.totalfunevaltime + t;

        
    case {'add'} % Add previously evaluated function

        fval_orig = varargin{1};
        x_orig = warpvars_vbmc(x,'inv',optimState.trinfo);    % Convert back to original space
        % Heteroscedastic noise?
        noise_flag = isfield(optimState,'S');
        if noise_flag
            fsd = varargin{2};
            if isempty(fsd); fsd = 1; end
        else
            fsd = [];
        end
        
        % Check function value
        if ~isscalar(fval_orig) || ~isfinite(fval_orig) || ~isreal(fval_orig)
            error(['funlogger_vbmc:InvalidFuncValue',...
                'The provided function value must be a finite real-valued scalar (provided value: ' mat2str(fval_orig) ').']);
        end

        % Check returned function SD
        if noise_flag && (~isscalar(fsd) || ~isfinite(fsd) || ~isreal(fsd) || fsd <= 0.0)
            error(['funlogger_vbmc:InvalidNoiseValue',...
                'The provided estimated SD (second function output) must be a finite, positive real-valued scalar (provided SD: ' mat2str(fsd) ').']);
        end
        
        % Tempered posterior
        if isfield(optimState,'temperature') && ~isempty(optimState.temperature)
            fval_orig = fval_orig / optimState.temperature;
            fsd = fsd / optimState.temperature;
        end
            
        % Update function records
        optimState.cachecount = optimState.cachecount + 1;
        [optimState,fval,idx] = record(optimState,x_orig,x,fval_orig,0,fsd);
        
    case 'done' % Finalize stored table
        
        noise_flag = isfield(optimState,'S');
        
        optimState.X_orig = optimState.X_orig(1:optimState.Xn,:);
        optimState.y_orig = optimState.y_orig(1:optimState.Xn);
        optimState.X_flag = optimState.X_flag(1:optimState.Xn);
        if noise_flag; optimState.S = optimState.S(1:optimState.Xn); end
        optimState.funevaltime = optimState.funevaltime(1:optimState.Xn);
        optimState.nevals = optimState.nevals(1:optimState.Xn);
            
        optimState = rmfield(optimState,'X');
        optimState = rmfield(optimState,'y');
        
    otherwise        
        error('funlogger_vbmc:UnknownAction','Unknown FUNLOGGER action.');
end

% Switch warnings off again
for ff = fields(optimState.DefaultWarnings)'
    warning('off',optimState.DefaultWarnings.(ff{:}).identifier);
end

end

%--------------------------------------------------------------------------
function [optimState,fval,idx] = record(optimState,x_orig,x,fval_orig,t,fsd)
%RECORD Record function evaluation.

if nargin < 6; fsd = []; end

duplicate_flag = all(bsxfun(@eq,x,optimState.X),2);

if any(duplicate_flag)    
    if sum(duplicate_flag) > 1; error('More than one match.'); end    
    idx = find(duplicate_flag);    
    N = optimState.nevals(idx);
    
    if ~isempty(fsd)
        tau_n = 1/optimState.S(idx)^2;
        tau_1 = 1/fsd^2;
        optimState.y_orig(idx) = (tau_n*optimState.y_orig(idx) + tau_1*fval_orig)/(tau_n + tau_1);
        optimState.S(idx) = 1/sqrt(tau_n + tau_1);
    else
        optimState.y_orig(idx) = (N*optimState.y_orig(idx) + fval_orig)/(N+1);
    end
    fval = optimState.y_orig(idx) + warpvars_vbmc(x,'logp',optimState.trinfo);
    optimState.y(optimState.Xn) = fval;
    % if ~isempty(fsd); optimState.S(idx) = sqrt((N^2*optimState.S(idx)^2 + fsd^2)/(N+1)^2); end
    optimState.funevaltime(idx) = (N*optimState.funevaltime(idx) + t)/(N+1);
    optimState.nevals(idx) = optimState.nevals(idx) + 1;

else
    optimState.Xn = optimState.Xn+1;
    idx = optimState.Xn;

    % Expand storage by 50% at a time
    if optimState.Xn > optimState.Xmax
        optimState.Xmax = ceil(optimState.Xmax*1.5);
        optimState.X_orig = expand(optimState.X_orig,optimState.Xmax);
        optimState.X = expand(optimState.X,optimState.Xmax);
        optimState.y_orig = expand(optimState.y_orig,optimState.Xmax);
        optimState.y = expand(optimState.y,optimState.Xmax);
        if ~isempty(fsd); optimState.S = expand(optimState.S,optimState.Xmax); end
        optimState.X_flag = expand(optimState.X_flag,optimState.Xmax,false);
        optimState.funevaltime = expand(optimState.funevaltime,optimState.Xmax);
        optimState.nevals = expand(optimState.nevals,optimState.Xmax);
    end

    optimState.X_orig(optimState.Xn,:) = x_orig;
    optimState.X(optimState.Xn,:) = x;
    optimState.y_orig(optimState.Xn) = fval_orig;
    fval = fval_orig + warpvars_vbmc(x,'logp',optimState.trinfo);
    optimState.y(optimState.Xn) = fval;
    if ~isempty(fsd); optimState.S(optimState.Xn) = fsd; end
    optimState.X_flag(optimState.Xn) = true;
    optimState.funevaltime(optimState.Xn) = t;
    optimState.nevals(optimState.Xn) = max(1,optimState.nevals(optimState.Xn) + 1);
end

optimState.ymax = max(optimState.y(optimState.X_flag));
optimState.N = optimState.Xn;  % Number of training inputs
optimState.Neff = sum(optimState.nevals(optimState.X_flag));

end

%--------------------------------------------------------------------------
function x = expand(x,nmax,fill)
%EXPAND Expand storage for given matrix
if nargin < 3 || isempty(fill); fill = NaN; end

if isnan(fill)
    x = [x; NaN(nmax-size(x,1),size(x,2))];
elseif islogical(fill) && fill
    x = [x; true(nmax-size(x,1),size(x,2))];
elseif islogical(fill) && ~fill
    x = [x; false(nmax-size(x,1),size(x,2))];    
else
    x = [x; fill*ones(nmax-size(x,1),size(x,2))];    
end

end