function [der,errest,finaldelta] = derivest(fun,x0,varargin)
% DERIVEST: estimate the n'th derivative of fun at x0, provide an error estimate
% usage: [der,errest] = DERIVEST(fun,x0)  % first derivative
% usage: [der,errest] = DERIVEST(fun,x0,prop1,val1,prop2,val2,...)
%
% Derivest will perform numerical differentiation of an
% analytical function provided in fun. It will not
% differentiate a function provided as data. Use gradient
% for that purpose, or differentiate a spline model.
%
% The methods used by DERIVEST are finite difference
% approximations of various orders, coupled with a generalized
% (multiple term) Romberg extrapolation. This also yields
% the error estimate provided. DERIVEST uses a semi-adaptive
% scheme to provide the best estimate that it can by its
% automatic choice of a differencing interval.
%
% Finally, While I have not written this function for the
% absolute maximum speed, speed was a major consideration
% in the algorithmic design. Maximum accuracy was my main goal.
%
%
% Arguments (input)
%  fun - function to differentiate. May be an inline function,
%        anonymous, or an m-file. fun will be sampled at a set
%        of distinct points for each element of x0. If there are
%        additional parameters to be passed into fun, then use of
%        an anonymous function is recommended.
%
%        fun should be vectorized to allow evaluation at multiple
%        locations at once. This will provide the best possible
%        speed. IF fun is not so vectorized, then you MUST set
%        'vectorized' property to 'no', so that derivest will
%        then call your function sequentially instead.
%
%        Fun is assumed to return a result of the same
%        shape as its input x0.
%
%  x0  - scalar, vector, or array of points at which to
%        differentiate fun.
%
% Additional inputs must be in the form of property/value pairs.
%  Properties are character strings. They may be shortened
%  to the extent that they are unambiguous. Properties are
%  not case sensitive. Valid property names are:
%
%  'DerivativeOrder', 'MethodOrder', 'Style', 'RombergTerms'
%  'FixedStep', 'MaxStep'
%
%  All properties have default values, chosen as intelligently
%  as I could manage. Values that are character strings may
%  also be unambiguously shortened. The legal values for each
%  property are:
%
%  'DerivativeOrder' - specifies the derivative order estimated.
%        Must be a positive integer from the set [1,2,3,4].
%
%        DEFAULT: 1 (first derivative of fun)
%
%  'MethodOrder' - specifies the order of the basic method
%        used for the estimation.
%
%        For 'central' methods, must be a positive integer
%        from the set [2,4].
%
%        For 'forward' or 'backward' difference methods,
%        must be a positive integer from the set [1,2,3,4].
%
%        DEFAULT: 4 (a second order method)
%
%        Note: higher order methods will generally be more
%        accurate, but may also suffere more from numerical
%        problems.
%
%        Note: First order methods would usually not be
%        recommended.
%
%  'Style' - specifies the style of the basic method
%        used for the estimation. 'central', 'forward',
%        or 'backwards' difference methods are used.
%
%        Must be one of 'Central', 'forward', 'backward'.
%
%        DEFAULT: 'Central'
%
%        Note: Central difference methods are usually the
%        most accurate, but sometiems one must not allow
%        evaluation in one direction or the other.
%
%  'RombergTerms' - Allows the user to specify the generalized
%        Romberg extrapolation method used, or turn it off
%        completely.
%
%        Must be a positive integer from the set [0,1,2,3].
%
%        DEFAULT: 2 (Two Romberg terms)
%
%        Note: 0 disables the Romberg step completely.
%
%  'FixedStep' - Allows the specification of a fixed step
%        size, preventing the adaptive logic from working.
%        This will be considerably faster, but not necessarily
%        as accurate as allowing the adaptive logic to run.
%
%        DEFAULT: []
%
%        Note: If specified, 'FixedStep' will define the
%        maximum excursion from x0 that will be used.
%
%  'Vectorized' - Derivest will normally assume that your
%        function can be safely evaluated at multiple locations
%        in a single call. This would minimize the overhead of
%        a loop and additional function call overhead. Some
%        functions are not easily vectorizable, but you may
%        (if your matlab release is new enough) be able to use
%        arrayfun to accomplish the vectorization.
%
%        When all else fails, set the 'vectorized' property
%        to 'no'. This will cause derivest to loop over the
%        successive function calls.
%
%        DEFAULT: 'yes'
%
%
%  'MaxStep' - Specifies the maximum excursion from x0 that
%        will be allowed, as a multiple of x0.
%
%        DEFAULT: 100
%
%  'StepRatio' - Derivest uses a proportionally cascaded
%        series of function evaluations, moving away from your
%        point of evaluation. The StepRatio is the ratio used
%        between sequential steps.
%
%        DEFAULT: 2.0000001
%
%        Note: use of a non-integer stepratio is intentional,
%        to avoid integer multiples of the period of a periodic
%        function under some circumstances.
%
%
% See the document DERIVEST.pdf for more explanation of the
% algorithms behind the parameters of DERIVEST. In most cases,
% I have chosen good values for these parameters, so the user
% should never need to specify anything other than possibly
% the DerivativeOrder. I've also tried to make my code robust
% enough that it will not need much. But complete flexibility
% is in there for your use.
%
%
% Arguments: (output)
%  der - derivative estimate for each element of x0
%        der will have the same shape as x0.
%
%  errest - 95% uncertainty estimate of the derivative, such that
%
%        abs(der(j) - f'(x0(j))) < erest(j)
%
%  finaldelta - The final overall stepsize chosen by DERIVEST
%
%
% Example usage:
%  First derivative of exp(x), at x == 1
%   [d,e]=derivest(@(x) exp(x),1)
%   d =
%       2.71828182845904
%
%   e =
%       1.02015503167879e-14
%
%  True derivative
%   exp(1)
%   ans =
%       2.71828182845905
%
% Example usage:
%  Third derivative of x.^3+x.^4, at x = [0,1]
%   derivest(@(x) x.^3 + x.^4,[0 1],'deriv',3)
%   ans =
%       6       30
%
%  True derivatives: [6,30]
%
%
% See also: gradient
%
%
% Author: John D'Errico
% e-mail: woodchips@rochester.rr.com
% Release: 1.0
% Release date: 12/27/2006

par.DerivativeOrder = 1;
par.MethodOrder = 4;
par.Style = 'central';
par.RombergTerms = 2;
par.FixedStep = [];
par.MaxStep = 100;
% setting a default stepratio as a non-integer prevents
% integer multiples of the initial point from being used.
% In turn that avoids some problems for periodic functions.
par.StepRatio = 2.0000001;
par.NominalStep = [];
par.Vectorized = 'yes';

na = length(varargin);
if (rem(na,2)==1)
  error 'Property/value pairs must come as PAIRS of arguments.'
elseif na>0
  par = parse_pv_pairs(par,varargin);
end
par = check_params(par);

% Was fun a string, or an inline/anonymous function?
if (nargin<1)
  help derivest
  return
elseif isempty(fun)
  error 'fun was not supplied.'
elseif ischar(fun)
  % a character function name
  fun = str2func(fun);
end

% no default for x0
if (nargin<2) || isempty(x0)
  error 'x0 was not supplied'
end
par.NominalStep = max(x0,0.02);

% was a single point supplied?
nx0 = size(x0);
n = prod(nx0);

% Set the steps to use.
if isempty(par.FixedStep)
  % Basic sequence of steps, relative to a stepsize of 1.
  delta = par.MaxStep*par.StepRatio .^(0:-1:-25)';
  ndel = length(delta);
else
  % Fixed, user supplied absolute sequence of steps.
  ndel = 3 + ceil(par.DerivativeOrder/2) + ...
     par.MethodOrder + par.RombergTerms;
  if par.Style(1) == 'c'
    ndel = ndel - 2;
  end
  delta = par.FixedStep*par.StepRatio .^(-(0:(ndel-1)))';
end

% generate finite differencing rule in advance.
% The rule is for a nominal unit step size, and will
% be scaled later to reflect the local step size.
fdarule = 1;
switch par.Style
  case 'central'
    % for central rules, we will reduce the load by an
    % even or odd transformation as appropriate.
    if par.MethodOrder==2
      switch par.DerivativeOrder
        case 1
          % the odd transformation did all the work
          fdarule = 1;
        case 2
          % the even transformation did all the work
          fdarule = 2;
        case 3
          % the odd transformation did most of the work, but
          % we need to kill off the linear term
          fdarule = [0 1]/fdamat(par.StepRatio,1,2);
        case 4
          % the even transformation did most of the work, but
          % we need to kill off the quadratic term
          fdarule = [0 1]/fdamat(par.StepRatio,2,2);
      end
    else
      % a 4th order method. We've already ruled out the 1st
      % order methods since these are central rules.
      switch par.DerivativeOrder
        case 1
          % the odd transformation did most of the work, but
          % we need to kill off the cubic term
          fdarule = [1 0]/fdamat(par.StepRatio,1,2);
        case 2
          % the even transformation did most of the work, but
          % we need to kill off the quartic term
          fdarule = [1 0]/fdamat(par.StepRatio,2,2);
        case 3
          % the odd transformation did much of the work, but
          % we need to kill off the linear & quintic terms
          fdarule = [0 1 0]/fdamat(par.StepRatio,1,3);
        case 4
          % the even transformation did much of the work, but
          % we need to kill off the quadratic and 6th order terms
          fdarule = [0 1 0]/fdamat(par.StepRatio,2,3);
      end
    end
  case {'forward' 'backward'}
    % These two cases are identical, except at the very end,
    % where a sign will be introduced.

    % No odd/even trans, but we already dropped
    % off the constant term
    if par.MethodOrder==1
      if par.DerivativeOrder==1
        % an easy one
        fdarule = 1;
      else
        % 2:4
        v = zeros(1,par.DerivativeOrder);
        v(par.DerivativeOrder) = 1;
        fdarule = v/fdamat(par.StepRatio,0,par.DerivativeOrder);
      end
    else
      % par.MethodOrder methods drop off the lower order terms,
      % plus terms directly above DerivativeOrder
      v = zeros(1,par.DerivativeOrder + par.MethodOrder - 1);
      v(par.DerivativeOrder) = 1;
      fdarule = v/fdamat(par.StepRatio,0,par.DerivativeOrder+par.MethodOrder-1);
    end
    
    % correct sign for the 'backward' rule
    if par.Style(1) == 'b'
      fdarule = -fdarule;
    end
    
end % switch on par.style (generating fdarule)
nfda = length(fdarule);

% will we need fun(x0)?
if (rem(par.DerivativeOrder,2) == 0) || ~strncmpi(par.Style,'central',7)
  if strcmpi(par.Vectorized,'yes')
    f_x0 = fun(x0);
  else
    % not vectorized, so loop
    f_x0 = zeros(size(x0));
    for j = 1:numel(x0)
      f_x0(j) = fun(x0(j));
    end
  end
else
  f_x0 = [];
end

% Loop over the elements of x0, reducing it to
% a scalar problem. Sorry, vectorization is not
% complete here, but this IS only a single loop.
der = zeros(nx0);
errest = der;
finaldelta = der;
for i = 1:n
  x0i = x0(i);
  h = par.NominalStep(i);

  % a central, forward or backwards differencing rule?
  % f_del is the set of all the function evaluations we
  % will generate. For a central rule, it will have the
  % even or odd transformation built in.
  if par.Style(1) == 'c'
    % A central rule, so we will need to evaluate
    % symmetrically around x0i.
    if strcmpi(par.Vectorized,'yes')
      f_plusdel = fun(x0i+h*delta);
      f_minusdel = fun(x0i-h*delta);
    else
      % not vectorized, so loop
      f_minusdel = zeros(size(delta));
      f_plusdel = zeros(size(delta));
      for j = 1:numel(delta)
        f_plusdel(j) = fun(x0i+h*delta(j));
        f_minusdel(j) = fun(x0i-h*delta(j));
      end
    end
    
    if ismember(par.DerivativeOrder,[1 3])
      % odd transformation
      f_del = (f_plusdel - f_minusdel)/2;
    else
      f_del = (f_plusdel + f_minusdel)/2 - f_x0(i);
    end
  elseif par.Style(1) == 'f'
    % forward rule
    % drop off the constant only
    if strcmpi(par.Vectorized,'yes')
      f_del = fun(x0i+h*delta) - f_x0(i);
    else
      % not vectorized, so loop
      f_del = zeros(size(delta));
      for j = 1:numel(delta)
        f_del(j) = fun(x0i+h*delta(j)) - f_x0(i);
      end
    end
  else
    % backward rule
    % drop off the constant only
    if strcmpi(par.Vectorized,'yes')
      f_del = fun(x0i-h*delta) - f_x0(i);
    else
      % not vectorized, so loop
      f_del = zeros(size(delta));
      for j = 1:numel(delta)
        f_del(j) = fun(x0i-h*delta(j)) - f_x0(i);
      end
    end
  end
  
  % check the size of f_del to ensure it was properly vectorized.
  f_del = f_del(:);
  if length(f_del)~=ndel
    error 'fun did not return the correct size result (fun must be vectorized)'
  end

  % Apply the finite difference rule at each delta, scaling
  % as appropriate for delta and the requested DerivativeOrder.
  % First, decide how many of these estimates we will end up with.
  ne = ndel + 1 - nfda - par.RombergTerms;

  % Form the initial derivative estimates from the chosen
  % finite difference method.
  der_init = vec2mat(f_del,ne,nfda)*fdarule.';

  % scale to reflect the local delta
  der_init = der_init(:)./(h*delta(1:ne)).^par.DerivativeOrder;
  
  % Each approximation that results is an approximation
  % of order par.DerivativeOrder to the desired derivative.
  % Additional (higher order, even or odd) terms in the
  % Taylor series also remain. Use a generalized (multi-term)
  % Romberg extrapolation to improve these estimates.
  switch par.Style
    case 'central'
      rombexpon = 2*(1:par.RombergTerms) + par.MethodOrder - 2;
    otherwise
      rombexpon = (1:par.RombergTerms) + par.MethodOrder - 1;
  end
  [der_romb,errors] = rombextrap(par.StepRatio,der_init,rombexpon);
  
  % Choose which result to return
  
  % first, trim off the 
  if isempty(par.FixedStep)
    % trim off the estimates at each end of the scale
    nest = length(der_romb);
    switch par.DerivativeOrder
      case {1 2}
        trim = [1 2 nest-1 nest];
      case 3
        trim = [1:4 nest+(-3:0)];
      case 4
        trim = [1:6 nest+(-5:0)];
    end
    
    [der_romb,tags] = sort(der_romb);
    
    der_romb(trim) = [];
    tags(trim) = [];
    errors = errors(tags);
    trimdelta = delta(tags);
    
    [errest(i),ind] = min(errors);
    
    finaldelta(i) = h*trimdelta(ind);
    der(i) = der_romb(ind);
  else
    [errest(i),ind] = min(errors);
    finaldelta(i) = h*delta(ind);
    der(i) = der_romb(ind);
  end
end

end % mainline end

% ============================================
% subfunction - romberg extrapolation
% ============================================
function [der_romb,errest] = rombextrap(StepRatio,der_init,rombexpon)
% do romberg extrapolation for each estimate
%
%  StepRatio - Ratio decrease in step
%  der_init - initial derivative estimates
%  rombexpon - higher order terms to cancel using the romberg step
%
%  der_romb - derivative estimates returned
%  errest - error estimates
%  amp - noise amplification factor due to the romberg step

srinv = 1/StepRatio;

% do nothing if no romberg terms
nexpon = length(rombexpon);
rmat = ones(nexpon+2,nexpon+1);
switch nexpon
  case 0
    % rmat is simple: ones(2,1)
  case 1
    % only one romberg term
    rmat(2,2) = srinv^rombexpon;
    rmat(3,2) = srinv^(2*rombexpon);
  case 2
    % two romberg terms
    rmat(2,2:3) = srinv.^rombexpon;
    rmat(3,2:3) = srinv.^(2*rombexpon);
    rmat(4,2:3) = srinv.^(3*rombexpon);
  case 3
    % three romberg terms
    rmat(2,2:4) = srinv.^rombexpon;
    rmat(3,2:4) = srinv.^(2*rombexpon);
    rmat(4,2:4) = srinv.^(3*rombexpon);
    rmat(5,2:4) = srinv.^(4*rombexpon);
end

% qr factorization used for the extrapolation as well
% as the uncertainty estimates
[qromb,rromb] = qr(rmat,0);

% the noise amplification is further amplified by the Romberg step.
% amp = cond(rromb);

% this does the extrapolation to a zero step size.
ne = length(der_init);
rhs = vec2mat(der_init,nexpon+2,max(1,ne - (nexpon+2)));
rombcoefs = rromb\(qromb.'*rhs); 
der_romb = rombcoefs(1,:).';

% uncertainty estimate of derivative prediction
s = sqrt(sum((rhs - rmat*rombcoefs).^2,1));
rinv = rromb\eye(nexpon+1);
cov1 = sum(rinv.^2,2); % 1 spare dof
errest = s.'*12.7062047361747*sqrt(cov1(1));

end % rombextrap


% ============================================
% subfunction - vec2mat
% ============================================
function mat = vec2mat(vec,n,m)
% forms the matrix M, such that M(i,j) = vec(i+j-1)
[i,j] = ndgrid(1:n,0:m-1);
ind = i+j;
mat = vec(ind);
if n==1
  mat = mat.';
end

end % vec2mat


% ============================================
% subfunction - fdamat
% ============================================
function mat = fdamat(sr,parity,nterms)
% Compute matrix for fda derivation.
% parity can be
%   0 (one sided, all terms included but zeroth order)
%   1 (only odd terms included)
%   2 (only even terms included)
% nterms - number of terms

% sr is the ratio between successive steps
srinv = 1./sr;

switch parity
  case 0
    % single sided rule
    [i,j] = ndgrid(1:nterms);
    c = 1./factorial(1:nterms);
    mat = c(j).*srinv.^((i-1).*j);
  case 1
    % odd order derivative
    [i,j] = ndgrid(1:nterms);
    c = 1./factorial(1:2:(2*nterms));
    mat = c(j).*srinv.^((i-1).*(2*j-1));
  case 2
    % even order derivative
    [i,j] = ndgrid(1:nterms);
    c = 1./factorial(2:2:(2*nterms));
    mat = c(j).*srinv.^((i-1).*(2*j));
end

end % fdamat



% ============================================
% subfunction - check_params
% ============================================
function par = check_params(par)
% check the parameters for acceptability
%
% Defaults
% par.DerivativeOrder = 1;
% par.MethodOrder = 2;
% par.Style = 'central';
% par.RombergTerms = 2;
% par.FixedStep = [];

% DerivativeOrder == 1 by default
if isempty(par.DerivativeOrder)
  par.DerivativeOrder = 1;
else
  if (length(par.DerivativeOrder)>1) || ~ismember(par.DerivativeOrder,1:4)
    error 'DerivativeOrder must be scalar, one of [1 2 3 4].'
  end
end

% MethodOrder == 2 by default
if isempty(par.MethodOrder)
  par.MethodOrder = 2;
else
  if (length(par.MethodOrder)>1) || ~ismember(par.MethodOrder,[1 2 3 4])
    error 'MethodOrder must be scalar, one of [1 2 3 4].'
  elseif ismember(par.MethodOrder,[1 3]) && (par.Style(1)=='c')
    error 'MethodOrder==1 or 3 is not possible with central difference methods'
  end
end

% style is char
valid = {'central', 'forward', 'backward'};
if isempty(par.Style)
  par.Style = 'central';
elseif ~ischar(par.Style)
  error 'Invalid Style: Must be character'
end
ind = find(strncmpi(par.Style,valid,length(par.Style)));
if (length(ind)==1)
  par.Style = valid{ind};
else
  error(['Invalid Style: ',par.Style])
end

% vectorized is char
valid = {'yes', 'no'};
if isempty(par.Vectorized)
  par.Vectorized = 'yes';
elseif ~ischar(par.Vectorized)
  error 'Invalid Vectorized: Must be character'
end
ind = find(strncmpi(par.Vectorized,valid,length(par.Vectorized)));
if (length(ind)==1)
  par.Vectorized = valid{ind};
else
  error(['Invalid Vectorized: ',par.Vectorized])
end

% RombergTerms == 2 by default
if isempty(par.RombergTerms)
  par.RombergTerms = 2;
else
  if (length(par.RombergTerms)>1) || ~ismember(par.RombergTerms,0:3)
    error 'RombergTerms must be scalar, one of [0 1 2 3].'
  end
end

% FixedStep == [] by default
if (length(par.FixedStep)>1) || (~isempty(par.FixedStep) && (par.FixedStep<=0))
  error 'FixedStep must be empty or a scalar, >0.'
end

% MaxStep == 10 by default
if isempty(par.MaxStep)
  par.MaxStep = 10;
elseif (length(par.MaxStep)>1) || (par.MaxStep<=0)
  error 'MaxStep must be empty or a scalar, >0.'
end

end % check_params


% ============================================
% Included subfunction - parse_pv_pairs
% ============================================
function params=parse_pv_pairs(params,pv_pairs)
% parse_pv_pairs: parses sets of property value pairs, allows defaults
% usage: params=parse_pv_pairs(default_params,pv_pairs)
%
% arguments: (input)
%  default_params - structure, with one field for every potential
%             property/value pair. Each field will contain the default
%             value for that property. If no default is supplied for a
%             given property, then that field must be empty.
%
%  pv_array - cell array of property/value pairs.
%             Case is ignored when comparing properties to the list
%             of field names. Also, any unambiguous shortening of a
%             field/property name is allowed.
%
% arguments: (output)
%  params   - parameter struct that reflects any updated property/value
%             pairs in the pv_array.
%
% Example usage:
% First, set default values for the parameters. Assume we
% have four parameters that we wish to use optionally in
% the function examplefun.
%
%  - 'viscosity', which will have a default value of 1
%  - 'volume', which will default to 1
%  - 'pie' - which will have default value 3.141592653589793
%  - 'description' - a text field, left empty by default
%
% The first argument to examplefun is one which will always be
% supplied.
%
%   function examplefun(dummyarg1,varargin)
%   params.Viscosity = 1;
%   params.Volume = 1;
%   params.Pie = 3.141592653589793
%
%   params.Description = '';
%   params=parse_pv_pairs(params,varargin);
%   params
%
% Use examplefun, overriding the defaults for 'pie', 'viscosity'
% and 'description'. The 'volume' parameter is left at its default.
%
%   examplefun(rand(10),'vis',10,'pie',3,'Description','Hello world')
%
% params = 
%     Viscosity: 10
%        Volume: 1
%           Pie: 3
%   Description: 'Hello world'
%
% Note that capitalization was ignored, and the property 'viscosity'
% was truncated as supplied. Also note that the order the pairs were
% supplied was arbitrary.

npv = length(pv_pairs);
n = npv/2;

if n~=floor(n)
  error 'Property/value pairs must come in PAIRS.'
end
if n<=0
  % just return the defaults
  return
end

if ~isstruct(params)
  error 'No structure for defaults was supplied'
end

% there was at least one pv pair. process any supplied
propnames = fieldnames(params);
lpropnames = lower(propnames);
for i=1:n
  p_i = lower(pv_pairs{2*i-1});
  v_i = pv_pairs{2*i};
  
  ind = strmatch(p_i,lpropnames,'exact');
  if isempty(ind)
    ind = find(strncmp(p_i,lpropnames,length(p_i)));
    if isempty(ind)
      error(['No matching property found for: ',pv_pairs{2*i-1}])
    elseif length(ind)>1
      error(['Ambiguous property name: ',pv_pairs{2*i-1}])
    end
  end
  p_i = propnames{ind};
  
  % override the corresponding default in params
  params = setfield(params,p_i,v_i); %#ok
  
end

end % parse_pv_pairs






