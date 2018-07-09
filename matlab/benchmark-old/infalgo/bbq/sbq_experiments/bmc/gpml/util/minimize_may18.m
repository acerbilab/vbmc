% minimize.m - minimize a smooth differentiable multivariate function using
% LBFGS (Limited memory LBFGS) or CG (Conjugate Gradients)
% Usage: [X, fX, i] = minimize(X, F, p, other, ... )
% where
%   X    is an initial guess (any type: vector, matrix, cell array, struct)
%   F    is the objective function (function pointer or name)
%   p    parameters - if p is a number, it corresponds to p.length below
%     p.length     allowed 1) # linesearches or 2) if -ve minus # func evals 
%     p.method     minimization method, 'LBFGS' (default) or 'CG'
%     p.verbosity  0 quiet, 1 line, 2 line + warnings (default), 3 graphical
%     p.mem        number of directions used in LBFGS (default 4)
%   other, ...     other parameters, passed to the function F
%   X     returned minimizer
%   fX    vector of function values showing minimization progress
%   i     final number of linesearches or function evaluations
% The function F must take the following syntax [f, df] = F(X, other, ...)
% where f is the function value and df its partial derivatives. The types of X
% and df must be identical (vector, matrix, cell array, struct, etc).
function [X, fX, i] = minimize(X, F, p, varargin)

if isnumeric(p), p = struct('length', p); end             % convert p to struct
if p.length > 0, p.S = 'linesearch #'; else p.S = 'function evaluation #'; end;
if ~isfield(p,'method'), p.method = @LBFGS; end            % set default method
if ~isfield(p,'verbosity'), p.verbosity = 2; end   % default 1 line text output
if ~isfield(p,'MFEPLS'), p.MFEPLS = 20; end    % Max Func Evals Per Line Search
if ~isfield(p,'MSR'), p.MSR = 100; end                % Max Slope Ratio default

x = unwrap(X);                                % convert initial guess to vector
f(F, X, varargin{:});                                   % set up the function f
[fx, dfx] = f(x);                      % initial function value and derivatives 
if p.verbosity, printf('Initial Function Value %4.6e\r', fx); end
if p.verbosity > 2,
  clf; subplot(211); hold on; xlabel(p.S); ylabel('function value');
  plot(p.length < 0, fx, '+'); drawnow;
end
[x, fX, i] = feval(p.method, x, fx, dfx, p);  % minimize using direction method 
X = rewrap(X, x);                   % convert answer to original representation
if p.verbosity, printf('\n'); end

function [x, fx, i] = CG(x0, fx0, dfx0, p);
i = p.length < 0;                                 % initialize resource counter
r = -dfx0; s = -r'*r; b = -1/s; bs = -1; ok = 0; fx = fx0;   % steepest descent
while i < abs(p.length)
  [x, b, fx0, dfx, i] = lineSearch(x0, fx0, dfx0, r, s, b*bs/min(b*s,bs/p.MSR), i, p);
  if i < 0                                              % if line search failed
    i = -i; if ok, ok = 0; r = -dfx; else break; end     % try steepest or stop
  else
    ok = 1; bs = b*s;        % record step times slope (for slope ratio method)
    r = (dfx'*(dfx-dfx0))/(dfx0'*dfx0)*r - dfx;             % Polack-Ribiere CG
  end  
  s = r'*dfx; if s >= 0, r = -dfx; s = r'*dfx; ok = 0; end  % slope must be -ve 
  x0 = x; dfx0 = dfx; fx = [fx; fx0];        % replace old values with new ones
end

function [x, fx, i] = LBFGS(x0, fx0, dfx0, p);
n = length(x0); k = 0; ok = 0; fx = fx0; bs = -dfx0'*dfx0/p.MSR;
if isfield(p, 'mem'), m = p.mem; else m = min(4, n); end
t = zeros(n, m); y = zeros(n, m);                             % allocate memory
i = p.length < 0;                                 % initialize resource counter
while i < abs(p.length)
  q = dfx0;
  for j = rem(k-1:-1:max(0,k-m),m)+1
    a(j) = t(:,j)'*q/rho(j); q = q-a(j)*y(:,j);
  end
  if k == 0, r = -q/(q'*q); else r = -t(:,j)'*y(:,j)/(y(:,j)'*y(:,j))*q; end
  for j = rem(max(0,k-m):k-1,m)+1
    r = r-t(:,j)*(a(j)+y(:,j)'*r/rho(j));
  end
  s = r'*dfx0; if s >= 0, r = -dfx0; s = r'*dfx0; k = 0; ok = 0; end
  [x, b, fx0, dfx, i] = lineSearch(x0, fx0, dfx0, r, s, bs/min(bs,s/p.MSR), i, p); 
  if i < 0                                              % if line search failed
    i = -i; if ok, ok = 0; k = 0; else break; end        % try steepest or stop
  else
    j = rem(k,m)+1; t(:,j) = x-x0; y(:,j) = dfx-dfx0; rho(j) = t(:,j)'*y(:,j);
    ok = 1; k = k+1; bs = b*s;
  end
  x0 = x; dfx0 = dfx; fx = [fx; fx0];                  % replace and add values
end

function [x, a, fx, df, i] = lineSearch(x0, f0, df0, d, s, a, i, p)

INT = 0.1; EXT = 5.0;                    % interpolate and extrapolation limits
SIG = 0.1; RHO = SIG/2;         % Wolfe-Powell condition constants, 0<RHO<SIG<1
if p.length < 0, LIMIT = min(p.MFEPLS, -i-p.length); else LIMIT = p.MFEPLS; end

j = 0; p0.x = 0.0; p0.f = f0; p0.df = df0; p0.s = s; p1 = p0; maxS = -SIG*s; p3.x = a;
if p.verbosity > 2
  A = [-a a]/5; nd = norm(d);
  subplot(212); hold off; plot(0, f0, 'k+'); hold on; plot(nd*A, f0+s*A, 'k-');
  xlabel('distance in line search direction'); ylabel('function value');
end
while 1                               % keep extrapolating as long as necessary
  ok = 0; while ~ok & j < LIMIT
    try           % try, catch and bisect to safeguard extrapolation evaluation
      [p3.f p3.df] = f(x0+p3.x*d); j = j+1; p3.s = p3.df'*d; ok = 1; 
      if isnan(p3.f+p3.s) | isinf(p3.f+p3.s)
        error('Objective function returned Inf or NaN','');
      end;
    catch
      if p.verbosity > 1, printf('\n'); warning(lasterr); end % warn or silence
      p3.x = (p1.x+p3.x)/2; ok = 0; p3.f = NaN;             % bisect, and retry
    end
  end
  if p.verbosity > 2
    plot(nd*p3.x, p3.f, 'b+'); plot(nd*(p3.x+A), p3.f+p3.s*A, 'b-'); drawnow
  end
  if p3.s > -maxS | p3.f > f0+p3.x*RHO*s | j >= LIMIT, break; end       % done?
  p0 = p1; p1 = p3;                                 % move points back one unit
  a = p1.x-p0.x; b = minCubic(a, p1.f-p0.f, p0.s, p1.s);  % cubic extrapolation
  if ~isreal(b) || isnan(b) || isinf(b) || b < a || b > a*EXT     % if b is bad
    b = a*EXT;                                % then extrapolate maximum amount
  end
  p3.x = p0.x+max(b, p1.x+INT*a);             % move away from current, off-set 
end
while 1                               % keep interpolating as long as necessary
  if p1.f > p3.f, p2 = p3; else p2 = p1; end          % make p2 the best so far
  if abs(p2.s) < maxS & p2.f < f0+a*RHO*s | j >= LIMIT, break; end      % done?
  a = p3.x-p1.x; b = minCubic(a, p3.f-p1.f, p1.s, p3.s);  % cubic interpolation
  if ~isreal(b) || isnan(b) || isinf(b) || b < 0 || b > a         % if b is bad
    b = a/2;                                                      % then bisect
  end
  p2.x = p1.x+min(max(b, INT*a), (1-INT)*a);  % move away from current, off-set 
  [p2.f p2.df] = f(x0+p2.x*d); j = j+1; p2.s = p2.df'*d;
  if p.verbosity > 2
    plot(nd*p2.x, p2.f, 'r+'); plot(nd*(p2.x+A), p2.f+p2.s*A, 'r'); drawnow
  end
  if p2.s > 0 | p2.f > f0+p2.x*RHO*s, p3 = p2; else p1 = p2; end  % new bracket
end
x = x0+p2.x*d; fx = p2.f; df = p2.df; a = p2.x;        % return the value found
if p.length < 0, i = i+j; else i = i+1; end % count func evals or line searches
if p.verbosity, printf('%s %6i;  value %4.6e\r', p.S, i, fx); end 
if abs(p2.s) > maxS || p2.f > f0+a*RHO*s, i = -i; end        % indicate faliure 
if p.verbosity > 2
  if i>0, plot(norm(d)*p2.x, fx, 'go'); end
  subplot(211); plot(abs(i), fx, '+'); drawnow;
end

function x = minCubic(dx, df, s0, s1)
A = -6*df+3*(s0+s1)*dx; B = 3*df-(2*s0+s1)*dx;
x = -s0*dx*dx/(B+sqrt(B*B-A*s0*dx));                  % num. error possible, ok
function [fx, dfx] = f(varargin);
persistent F p;
if nargout == 0
  p = varargin; if ischar(p{1}), F = str2func(p{1}); else F = p{1}; end
else
  [fx, dfx] = F(rewrap(p{2}, varargin{1}), p{3:end}); dfx = unwrap(dfx);
end

% Extract the numerical values from "s" into the column vector "v". The
% variable "s" can be of any type, including struct and cell array.
% Non-numerical elements are ignored. See also the reverse rewrap.m. 

function v = unwrap(s)
v = [];   
if isnumeric(s)
  v = s(:);                        % numeric values are recast to column vector
elseif isstruct(s)
  v = unwrap(struct2cell(orderfields(s))); % alphabetize, conv to cell, recurse
elseif iscell(s)                                      % cell array elements are
  for i = 1:numel(s), v = [v; unwrap(s{i})]; end         % handled sequentially
end                                                   % other types are ignored


% Map the numerical elements in the vector "v" onto the variables "s" which can
% be of any type. The number of numerical elements must match; on exit "v"
% should be empty. Non-numerical entries are just copied. See also unwrap.m.

function [s v] = rewrap(s, v)
if isnumeric(s)
  if numel(v) < numel(s)
    error('The vector for conversion contains too few elements')
  end
  s = reshape(v(1:numel(s)), size(s));            % numeric values are reshaped
  v = v(numel(s)+1:end);                        % remaining arguments passed on
elseif isstruct(s) 
  [s p] = orderfields(s); p(p) = 1:numel(p);      % alphabetize, store ordering
  [t v] = rewrap(struct2cell(s), v);                 % convert to cell, recurse
  s = orderfields(cell2struct(t,fieldnames(s),1),p);  % conv to struct, reorder
elseif iscell(s)
  for i = 1:numel(s)             % cell array elements are handled sequentially
    [s{i} v] = rewrap(s{i}, v);
  end
end                                             % other types are not processed

function printf(varargin);
fprintf(varargin{:}); if exist('fflush','builtin') fflush(stdout); end

