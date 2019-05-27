function [fval,optimState] = funlogger_vbmc(fun,x,optimState,state,varargin)
%FUNLOGGER_VBMC Call objective function and do some bookkeeping.
%   [~,OPTIMSTATE] = FUNLOGGER_VBMC(FUN,X,OPTIMSTATE,'init') starts logging
%   function FUN with starting point X and optimization struct OPTIMSTATE.
%
%   [~,OPTIMSTATE] = FUNLOGGER_VBMC(FUN,X,OPTIMSTATE,'init',NMAX) stores up 
%   to NMAX function values (default NMAX=1e4).
%
%   [~,OPTIMSTATE] = FUNLOGGER_VBMC(FUN,X,OPTIMSTATE,'init',NMAX,1) also stores
%   heteroskedastic noise (second output argument) from the logged function.
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

%   Luigi Acerbi 2018

fval = [];

% Switch warnings on
for ff = fields(optimState.DefaultWarnings)'
    warning(optimState.DefaultWarnings.(ff{:}).state,optimState.DefaultWarnings.(ff{:}).identifier);
end

switch lower(state)
    case 'init' % Start new function logging session
    
        nvars = numel(x);

        % Number of stored function values
        if nargin > 4; nmax = varargin{1}; else; nmax = []; end
        if isempty(nmax); nmax = 1e4; end
        
        % Heteroscedastic noise
        if nargin > 5; hescnoise = varargin{2}; else; hescnoise = []; end
        if isempty(hescnoise); hescnoise = false; end

        optimState.funccount = 0;
        optimState.cachecount = 0;

        % Passing empty or new OPTIMSTATE
        if ~isfield(optimState,'X_orig') || isempty(optimState.X_orig)    
            optimState.X_orig = NaN(nmax,nvars);
            optimState.y_orig = NaN(nmax,1);
            optimState.X = NaN(nmax,nvars);
            optimState.y = NaN(nmax,1);
            if hescnoise; optimState.S = NaN(nmax,1); end
            optimState.Xn = 0;                    % Last filled entry
            optimState.Xmax = 0;                  % Maximum entry index
            optimState.X_flag = false(nmax,1);

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
                if hescnoise
                    optimState.S = circshift(optimState.S,-optimState.Xn);
                    optimState.S = optimState.S(end-nmax+1:nmax,:);
                end
                optimState.Xn = 0;
                optimState.Xmax = nmax;
            else
                offset = nmax - size(optimState.X_orig,1);
                optimState.X_orig = [optimState.X_orig; NaN(offset,nvars)];
                optimState.y_orig = [optimState.y_orig; NaN(offset,1)];
                if hescnoise
                    optimState.S = [optimState.S; NaN(offset,1)];
                end
            end
            optimState.X = warpvars(optimState.X_orig,'d',optimState.trinfo);
        end
        optimState.funevaltime = NaN(nmax,1);
        optimState.totalfunevaltime = 0;
    
    case {'iter','single'} % Evaluate function (and store output for 'iter')

        x_orig = warpvars(x,'inv',optimState.trinfo);    % Convert back to original space
        % Heteroscedastic noise?
        if isfield(optimState,'S'); hescnoise = 1; else; hescnoise = 0; end
        
        try
            funtime = tic;
            if hescnoise
                [fval_orig,fsd] = fun(x_orig);
            else
                fval_orig = fun(x_orig);
                fsd = [];
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
            if hescnoise && (~isscalar(fsd) || ~isfinite(fsd) || ~isreal(fsd) || fsd <= 0.0)
                error(['funlogger_vbmc:InvalidNoiseValue',...
                    'The returned estimated SD (second function output) must be a finite, positive real-valued scalar (returned SD: ' mat2str(fsd) ').']);
            end            
            
        catch fun_error
            warning(['funlogger_vbmc:FuncError',...
                'Error in executing the logged function ''' func2str(fun) ''' with input: ' mat2str(x)]);
            rethrow(fun_error);
        end
        
        % Update function records
        optimState.funccount = optimState.funccount + 1;
        if strcmpi(state, 'iter')
            [optimState,fval] = record(optimState,x_orig,x,fval_orig,t,fsd);
        end
        optimState.totalfunevaltime = optimState.totalfunevaltime + t;

        
    case {'add'} % Add previously evaluated function

        fval_orig = varargin{1};
        x_orig = warpvars(x,'inv',optimState.trinfo);    % Convert back to original space
        % Heteroscedastic noise?
        if isfield(optimState,'S'); hescnoise = 1; else; hescnoise = 0; end
        if hescnoise; fsd = varargin{2}; else; fsd = []; end
        
        % Check function value
        if ~isscalar(fval_orig) || ~isfinite(fval_orig) || ~isreal(fval_orig)
            error(['funlogger_vbmc:InvalidFuncValue',...
                'The provided function value must be a finite real-valued scalar (provided value: ' mat2str(fval_orig) ').']);
        end

        % Check returned function SD
        if hescnoise && (~isscalar(fsd) || ~isfinite(fsd) || ~isreal(fsd) || fsd <= 0.0)
            error(['funlogger_vbmc:InvalidNoiseValue',...
                'The provided estimated SD (second function output) must be a finite, positive real-valued scalar (provided SD: ' mat2str(fsd) ').']);
        end
                    
        % Update function records
        optimState.cachecount = optimState.cachecount + 1;
        [optimState,fval] = record(optimState,x_orig,x,fval_orig,0,fsd);
        
    case 'done' % Finalize stored table
        
        if isfield(optimState,'S'); hescnoise = 1; else; hescnoise = 0; end
        
        if optimState.Xmax < size(optimState.X,1)
            optimState.X_orig = optimState.X_orig(1:optimState.Xmax,:);
            optimState.y_orig = optimState.y_orig(1:optimState.Xmax);
            optimState.X_flag = optimState.X_flag(1:optimState.Xmax);
            if hescnoise; optimState.S = optimState.S(1:optimState.Xmax); end
            optimState.funevaltime = optimState.funevaltime(1:optimState.Xmax);
        else
            optimState.X_orig = circshift(optimState.X_orig,-optimState.Xn);
            optimState.y_orig = circshift(optimState.y_orig,-optimState.Xn);
            optimState.X_flag = circshift(optimState.X_flag,-optimState.Xn);
            if hescnoise; optimState.S = circshift(optimState.S,-optimState.Xn); end
            optimState.funevaltime = circshift(optimState.funevaltime,-optimState.Xn);        
        end
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
function [optimState,fval] = record(optimState,x_orig,x,fval_orig,t,fsd)
%RECORD Record function evaluation.

if nargin < 6; fsd = []; end

optimState.Xn = max(1,mod(optimState.Xn+1, size(optimState.X_orig,1)));
optimState.Xmax = min(optimState.Xmax+1, size(optimState.X_orig,1));
optimState.X_orig(optimState.Xn,:) = x_orig;
optimState.X(optimState.Xn,:) = x;
optimState.y_orig(optimState.Xn) = fval_orig;
fval = fval_orig + warpvars(x,'logp',optimState.trinfo);
optimState.y(optimState.Xn) = fval;
if ~isempty(fsd); optimState.S(optimState.Xn) = fsd; end
optimState.X_flag(optimState.Xn) = true;
optimState.funevaltime(optimState.Xn) = t;

end