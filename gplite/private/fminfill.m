function [x,fval,exitflag,output] = fminfill(fun,x0,LB,UB,PLB,PUB,tprior,options)
%FMINFILL

if nargin < 2; x0 = []; end
if nargin < 3; LB = []; end
if nargin < 4; UB = []; end
if nargin < 5; PLB = []; end
if nargin < 6; PUB = []; end
if nargin < 7; tprior = []; end
if nargin < 8; options = []; end

N0 = size(x0,1);
N = options.FunEvals;

if isfield(options,'Method'); Method = options.Method; else; Method = []; end
if isempty(Method); Method = 'sobol'; end

nvars = max([size(x0,2),numel(LB),numel(UB),numel(PLB),numel(PUB)]);

if isempty(tprior)
    tprior.mu = [];     tprior.sigma = [];      tprior.df = [];
end
if ~isfield(tprior,'mu') || isempty(tprior.mu)
    tprior.mu = NaN(1,nvars);
end
if ~isfield(tprior,'sigma') || isempty(tprior.sigma)
    tprior.sigma = NaN(1,nvars);
end
if ~isfield(tprior,'df') || isempty(tprior.df)
    tprior.df = NaN(1,nvars);
end

if nvars == 0
    error('');
end

if isempty(LB); LB = -Inf(1,nvars); end
if isempty(UB); UB = Inf(1,nvars); end
if isempty(PLB); PLB = LB; end
if isempty(PUB); PUB = UB; end

% Force provided points to be inside bounds
if ~isempty(x0)
   x0 = bsxfun(@max,bsxfun(@min,x0,UB),LB);
end

if N > N0
    % First test hyperparameters on a space-filling initial design
    S = [];
    if strcmpi(Method,'sobol')
        if nvars <= 40   % Sobol generator handles up to 40 variables
            MaxSeed = 997;
            seed = randi(MaxSeed);                  % Random seed
            try
                S = (sobol_generate(nvars,N-N0,seed))';
                S = S(:,randperm(nvars));   % Randomly permute columns
            catch
                % Failed to generate Sobol sequence
            end
        end
    end
    if isempty(S)       % Just use random sampling
        S = rand(N-N0,nvars);
    end

    Xs = zeros(N-N0,nvars);

    % If a prior is specified, use that
    for iVar = 1:nvars
        mu = tprior.mu(iVar);
        sigma = tprior.sigma(iVar);

        if ~isfinite(mu) || ~isfinite(sigma)    % Uniform distribution?

            if isfinite(LB(iVar)) && isfinite(UB(iVar))
                % Mixture of uniforms (full bounds and plausible bounds)
                w = 0.5^(1/nvars); % Half of all starting points from inside the plausible box
                Xs(:,iVar) = uuinv(S(:,iVar),[LB(iVar),PLB(iVar),PUB(iVar),UB(iVar)],w);
            else
                % All starting points from inside the plausible box
                Xs(:,iVar) = S(:,iVar)*(PUB(iVar)-PLB(iVar)) + PLB(iVar);                
            end

        else                                    % Student's t prior
            df = tprior.df(iVar);
            if ~isfinite(df); df = 3; end       % Force fat tails
            df = min(df,3);
            if df == 0; df = Inf; end

            tcdf_lb = tcdf((LB(iVar)-mu)/sigma,df);
            tcdf_ub = tcdf((UB(iVar)-mu)/sigma,df);
            Sscaled = tcdf_lb + (tcdf_ub-tcdf_lb)*S(:,iVar);
            Xs(:,iVar) = tinv(Sscaled,df)*sigma + mu;
        end

    end
end

X = [x0;Xs];

y = Inf(1,N);
for iEval = 1:N
    try
        y(iEval) = fun(X(iEval,:));
    catch
        % Something went wrong, try to continue
    end
end

% Choose best starting points
[y,ord] = sort(y,'ascend');
X = X(ord,:);

x = X(1,:);
fval = y(1);

if nargout > 2
    exitflag = 0;
end

if nargout > 3
    output.X = X;
    output.fvals = y;
    output.funccount = N;
end

end

%--------------------------------------------------------------------------
function x = uuinv(p,B,w)
%UUINV Inverse of mixture of uniforms cumulative distribution function (cdf).

x = zeros(size(p));
L1 = B(4) - B(1);
L2 = B(3) - B(2);

% First step
idx1 = p <= (1-w) * (B(2) - B(1)) / L1;
x(idx1) = B(1) + p(idx1) * L1 / (1 - w);

% Second step
idx2 = (p <= (1-w) * (B(3) - B(1)) / L1 + w) & ~idx1;
x(idx2) = (p(idx2) * L1 * L2 + B(1)*(1-w)*L2 + w*B(2)*L1) / (L1*w + L2*(1-w));

% Third step
idx3 = p > (1-w) * (B(3) - B(1)) / L1 + w;
x(idx3) = (p(idx3) - w + B(1)*(1-w)/L1) * L1/(1-w);

x(p < 0 | p > 1) = NaN;

end

%--------------------------------------------------------------------------
function p = tcdf(x,v)
%TCDF   Student's T cumulative distribution function (cdf).
%   P = TCDF(X,V) computes the cdf for Student's T distribution
%   with V degrees of freedom, at the values in X.

normcutoff = 1e7;

% Initialize P.
p = NaN(size(x));

nans = (isnan(x) | ~(0<v)); %  v==NaN ==> (0<v)==false
cauchy = (v == 1);
normal = (v > normcutoff);

% General case: first compute F(-|x|) < .5, the lower tail.
general = ~(cauchy | normal | nans);
xsq = x.^2;
% For small v, form v/(v+x^2) to maintain precision
t = (v < xsq) & general;
if any(t(:))
    p(t) = betainc(v(t) ./ (v(t) + xsq(t)), v(t)/2, 0.5, 'lower') / 2;
end

% For large v, form x^2/(v+x^2) to maintain precision
t = (v >= xsq) & general;
if any(t(:))
    p(t) = betainc(xsq(t) ./ (v(t) + xsq(t)), 0.5, v(t)/2, 'upper') / 2;
end

% For x > 0, F(x) = 1 - F(-|x|).
xpos = (x > 0);
p(xpos) = 1 - p(xpos); % p < .5, cancellation not a problem

% Special case for Cauchy distribution
p(cauchy)  = xpos(cauchy) + acot(-x(cauchy))/pi; 

% Normal Approximation for very large nu.
p(normal) = normcdf(x(normal));

% Make the result exact for the median.
p(x == 0 & ~nans) = 0.5;
end

%--------------------------------------------------------------------------
function x = tinv(p,v)
%TINV   Inverse of Student's T cumulative distribution function (cdf).
%   X=TINV(P,V) returns the inverse of Student's T cdf with V degrees 
%   of freedom, at the values in P.

% Initialize Y to zero, or NaN for invalid d.f.
x = NaN(size(p));

if isscalar(v); v = v*ones(size(p)); end

% The inverse cdf of 0 is -Inf, and the inverse cdf of 1 is Inf.
x(p==0 & v > 0) = -Inf;
x(p==1 & v > 0) = Inf;

k0 = (0<p & p<1) & (v > 0);

% Invert the Cauchy distribution explicitly
k = find(k0 & (v == 1));
if any(k)
  x(k) = tan(pi * (p(k) - 0.5));
end

% For small d.f., call betaincinv which uses Newton's method
k = find(k0 & (v < 1000) & (v~=1));
if any(k)
    q = p(k) - .5;
    df = v(k);
    t = (abs(q) < .25);
    z = zeros(size(q), 'like', x);
    oneminusz = zeros(size(q), 'like', x);
    if any(t)
        % for z close to 1, compute 1-z directly to avoid roundoff
        oneminusz(t) = betaincinv(2.*abs(q(t)),0.5,df(t)/2,'lower');
        z(t) = 1 - oneminusz(t);
    end
    t = ~t; % (abs(q) >= .25);
    if any(t)
        z(t) = betaincinv(2.*abs(q(t)),df(t)/2,0.5,'upper');
        oneminusz(t) = 1 - z(t);
    end
    x(k) = sign(q) .* sqrt(df .* (oneminusz./z));
end

% For large d.f., use Abramowitz & Stegun formula 26.7.5
k = find(k0 & (v >= 1000));
if any(k)
   xn = norminv(p(k));
   df = v(k);
   x(k) = xn + (xn.^3+xn)./(4*df) + ...
           (5*xn.^5+16.*xn.^3+3*xn)./(96*df.^2) + ...
           (3*xn.^7+19*xn.^5+17*xn.^3-15*xn)./(384*df.^3) +...
           (79*xn.^9+776*xn.^7+1482*xn.^5-1920*xn.^3-945*xn)./(92160*df.^4);
end
end

%--------------------------------------------------------------------------
function r = sobol_generate(D,N,skip)
%SOBOL_GENERATE generates a Sobol dataset.
%
%   R = SOBOL_GENERATE(M,N) generates a quasirandom Sobol sequence of
%   N points in D dimensions. R is a D-by-N matrix of points in the
%   quasirandom sequence.
%
%   R = SOBOL_GENERATE(M,N,SKIP) skips the first SKIP initial points 
%   (default is zero).
%
%   Modified from John Burkardt's implementation.

if nargin < 3 || isempty(skip); skip = 0; end

r = zeros(D,N);
for j = 1:N
    seed = skip + j - 1;
    [r(:,j),seed] = sobol(D,seed);
end

end


function [quasi,seed] = sobol(D,seed)

%*****************************************************************************80
%
%% I4_SOBOL generates a new quasirandom Sobol vector with each call.
%
%  Discussion:
%
%    The routine adapts the ideas of Antonov and Saleev.
%
%    Thanks to Francis Dalaudier for pointing out that the range of allowed
%    values of DIM_NUM should start at 1, not 2!  17 February 2009.
%
%    This function was modified to use PERSISTENT variables rather than
%    GLOBAL variables, 13 December 2009.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    26 March 2012
%
%  Author:
%
%    Original FORTRAN77 version by Bennett Fox.
%    MATLAB version by John Burkardt.
%
%  Reference:
%
%    Antonov, Saleev,
%    USSR Computational Mathematics and Mathematical Physics,
%    Volume 19, 1980, pages 252 - 256.
%
%    Paul Bratley, Bennett Fox,
%    Algorithm 659:
%    Implementing Sobol's Quasirandom Sequence Generator,
%    ACM Transactions on Mathematical Software,
%    Volume 14, Number 1, pages 88-100, 1988.
%
%    Bennett Fox,
%    Algorithm 647:
%    Implementation and Relative Efficiency of Quasirandom 
%    Sequence Generators,
%    ACM Transactions on Mathematical Software,
%    Volume 12, Number 4, pages 362-376, 1986.
%
%    Ilya Sobol,
%    USSR Computational Mathematics and Mathematical Physics,
%    Volume 16, pages 236-242, 1977.
%
%    Ilya Sobol, Levitan, 
%    The Production of Points Uniformly Distributed in a Multidimensional 
%    Cube (in Russian),
%    Preprint IPM Akad. Nauk SSSR, 
%    Number 40, Moscow 1976.
%
%  Parameters:
%
%    Input, integer DIM_NUM, the number of spatial dimensions.
%    DIM_NUM must satisfy 1 <= DIM_NUM <= 40.
%
%    Input/output, integer SEED, the "seed" for the sequence.
%    This is essentially the index in the sequence of the quasirandom
%    value to be generated.  On output, SEED has been set to the
%    appropriate next value, usually simply SEED+1.
%    If SEED is less than 0 on input, it is treated as though it were 0.
%    An input value of 0 requests the first (0-th) element of the sequence.
%
%    Output, real QUASI(DIM_NUM), the next quasirandom vector.
%
  persistent atmost;
  persistent dim_max;
  persistent dim_num_save;
  persistent initialized;
  persistent lastq;
  persistent log_max;
  persistent maxcol;
  persistent poly;
  persistent recipd;
  persistent seed_save;
  persistent v;

  if isempty(initialized)
    initialized = 0;
    dim_num_save = -1;
  end


  if ( ~initialized || D ~= dim_num_save )

    initialized = 1;

    dim_max = 40;
    dim_num_save = -1;
    log_max = 30;
    seed_save = -1;
%
%  Initialize (part of) V.
%
    v(1:dim_max,1:log_max) = zeros(dim_max,log_max);

    v(1:40,1) = [ ...
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]';

    v(3:40,2) = [ ...
            1, 3, 1, 3, 1, 3, 3, 1, ...
      3, 1, 3, 1, 3, 1, 1, 3, 1, 3, ...
      1, 3, 1, 3, 3, 1, 3, 1, 3, 1, ...
      3, 1, 1, 3, 1, 3, 1, 3, 1, 3 ]';

    v(4:40,3) = [ ...
               7, 5, 1, 3, 3, 7, 5, ...
      5, 7, 7, 1, 3, 3, 7, 5, 1, 1, ...
      5, 3, 3, 1, 7, 5, 1, 3, 3, 7, ...
      5, 1, 1, 5, 7, 7, 5, 1, 3, 3 ]';

    v(6:40,4) = [ ...
                     1, 7, 9,13,11, ...
      1, 3, 7, 9, 5,13,13,11, 3,15, ...
      5, 3,15, 7, 9,13, 9, 1,11, 7, ...
      5,15, 1,15,11, 5, 3, 1, 7, 9 ]';
  
    v(8:40,5) = [ ...
                           9, 3,27, ...
     15,29,21,23,19,11,25, 7,13,17, ...
      1,25,29, 3,31,11, 5,23,27,19, ...
     21, 5, 1,17,13, 7,15, 9,31, 9 ]';

    v(14:40,6) = [ ...
              37,33, 7, 5,11,39,63, ...
     27,17,15,23,29, 3,21,13,31,25, ...
      9,49,33,19,29,11,19,27,15,25 ]';

    v(20:40,7) = [ ...
                                         13, ...
     33,115, 41, 79, 17, 29,119, 75, 73,105, ...
      7, 59, 65, 21,  3,113, 61, 89, 45,107 ]';

    v(38:40,8) = [ ...
                                7, 23, 39 ]';
%
%  Set POLY.
%
    poly(1:40)= [ ...
        1,   3,   7,  11,  13,  19,  25,  37,  59,  47, ...
       61,  55,  41,  67,  97,  91, 109, 103, 115, 131, ...
      193, 137, 145, 143, 241, 157, 185, 167, 229, 171, ...
      213, 191, 253, 203, 211, 239, 247, 285, 369, 299 ];

    atmost = 2^log_max - 1;
%
%  Find the number of bits in ATMOST.
%
    maxcol = bit_hi1(atmost);
%
%  Initialize row 1 of V.end

%
    v(1,1:maxcol) = 1;

  end
%
%  Things to do only if the dimension changed.
%
  if D ~= dim_num_save
%
%  Check parameters.
%
    if D < 1 || D > dim_max 
        error('sobol:WrongDims', ...
            ['The requested dimension D should be between 1 and ' num2str(dim_max) '.']);
    end

    dim_num_save = D;
%
%  Initialize the remaining rows of V.
%
    for i = 2:D
%
%  The bits of the integer POLY(I) gives the form of polynomial I.
%
%  Find the degree of polynomial I from binary encoding.
%
      j = poly(i);
      m = 0;

      while ( 1 )

        j = floor ( j / 2 );

        if ( j <= 0 )
          break;
        end

        m = m + 1;

      end
%
%  Expand this bit pattern to separate components of the logical array INCLUD.
%
      j = poly(i);
      for k = m : -1 : 1
        j2 = floor ( j / 2 );
        includ(k) = ( j ~= 2 * j2 );
        j = j2;
      end
%
%  Calculate the remaining elements of row I as explained
%  in Bratley and Fox, section 2.
%
      for j = m + 1 : maxcol 
        newv = v(i,j-m);
        l = 1;
        for k = 1 : m
          l = 2 * l;
          if ( includ(k) )
            newv = bitxor ( newv, l * v(i,j-k) );
          end
        end
        v(i,j) = newv;
      end
    end
%
%  Multiply columns of V by appropriate power of 2.
%
    l = 1;
    for j = maxcol-1 : -1 : 1
      l = 2 * l;
      v(1:D,j) = v(1:D,j) * l;
    end
%
%  RECIPD is 1/(common denominator of the elements in V).
%
    recipd = 1.0 / ( 2 * l );

    lastq(1:D) = 0;

  end

  seed = floor ( seed );

  if ( seed < 0 )
    seed = 0;
  end

  if ( seed == 0 )

    l = 1;
    lastq(1:D) = 0;

  elseif ( seed == seed_save + 1 )
%
%  Find the position of the right-hand zero in SEED.
%
    l = bit_lo0(seed);

  elseif ( seed <= seed_save )

    seed_save = 0;
    l = 1;
    lastq(1:D) = 0;

    for seed_temp = seed_save : seed - 1
      l = i4_bit_lo0 ( seed_temp );
      for i = 1 : D
        lastq(i) = bitxor ( lastq(i), v(i,l) );
      end
    end

    l = bit_lo0(seed);

  elseif ( seed_save + 1 < seed )

    for seed_temp = seed_save + 1 : seed - 1
      l = bit_lo0(seed_temp);
      for i = 1 : D
        lastq(i) = bitxor ( lastq(i), v(i,l) );
      end
    end

    l = bit_lo0(seed);

  end
%
%  Check that the user is not calling too many times!
%
  if ( maxcol < l )
      error('sobol:TooManyCalls', ...
          ['Too many calls to SOBOL. MAXCOL = ' num2str(maxcol) '; L = ' num2str(l) '.'])
  end
%
%  Calculate the new components of QUASI.
%
  for i = 1:D
    quasi(i) = lastq(i) * recipd;
    lastq(i) = bitxor ( lastq(i), v(i,l) );
  end

  seed_save = seed;
  seed = seed + 1;
  

end

function bit = bit_lo0(n)
% BIT_LO0 returns the position of the low 0 bit base 2 in an integer.

s = ['0' dec2bin(n)];
idx = find(s == '0',1,'last');
bit = numel(s)-idx+1;  

end

function bit = bit_hi1(n)
%BIT_HI1 returns the position of the high 1 bit base 2 in an integer.

s = dec2bin(n);
idx = find(s == '1',1,'first');
if isempty(idx) 
    bit = 0;
else
    bit = numel(s)-idx+1;
end
    
end


