function [p, err, funevals] = mvtcdfqmc(a, b, Rho, nu, tol, maxfunevals, verbose)
% Quasi-Monte-Carlo computation of cumulative probability for the multivariate
% Student's t distribution with correlation parameter matrix RHO and degrees of
% freedom NU, evaluated over the hyper-rectangle with "lower left" corner at A
% and "upper right" corner at B.  Returns the estimated probability, P, an
% estimate of the error, ERR, and the number of function evaluations required,
% FUNEVALS.  Use TOL, MAXFUNVALS, and VERBOSE to control the algorithm.

%   References:
%      [1] Genz, A. and F. Bretz (1999) "Numerical Computation of Multivariate
%          t Probabilities with Application to Power Calculation of Multiple
%          Contrasts", J.Statist.Comput.Simul., 63:361-378.
%      [2] Genz, A. and F. Bretz (2002) "Comparison of Methods for the
%          Computation of Multivariate t Probabilities", J.Comp.Graph.Stat.,
%          11(4):950-971.

%   Copyright 2005-2006 The MathWorks, Inc.
%   $Revision: 1.1.6.4 $  $Date: 2007/09/18 02:34:45 $

if nargin < 4
    error('stats:mvtcdfqmc:NumberOfInputs', 'Requires at least four input arguments.');
end
if nargin < 5, tol = 1e-4; end % this is an absolute error
if nargin < 6, maxfunevals = 1e7; end
if nargin < 7, verbose = 0; end

outclass = superiorfloat(a,b,Rho,nu);

P = [31 47 73 113 173 263 397 593 907 1361 2053 3079 4621 6947 10427 15641 23473 ...
     35221 52837 79259 118891 178349 267523 401287 601942 902933 1354471 2031713];

if ~all(a < b)
    if any(a > b)
        error('stats:mvtcdfqmc:BadLimits', ...
              'Lower integration limits A must be less than or equal to upper limits B.');
    elseif any(isnan(a) | isnan(b))
        p = NaN(outclass);
        err = NaN(outclass);
    else % any(a == b),this includes +/- Inf
        p = zeros(outclass);
        err = zeros(outclass);
    end
    funevals = 0;
    return;
end

% Dimensions with limits of (-Inf,Inf) can be ignored.
dblInfLims = (a == -Inf) & (b == Inf);
if any(dblInfLims)
    if all(dblInfLims)
        p = 1; err = 0; funevals = 0;
        return
    end
    a(dblInfLims) = [];
    b(dblInfLims) = [];
    Rho(:,dblInfLims) = []; Rho(dblInfLims,:) = [];
end
m = size(Rho,1);

% Sort the order of integration according to increasing length of interval,
% with half-infinite intervals last.
[dum,ord] = sort(b - a);
a = a(ord);
b = b(ord);
Rho = Rho(ord,ord);

if any(any(abs(tril(Rho,-1)) > .999))
    warning('stats:mvtcdfqmc:HighCorr', ...
            'Variables are highly correlated.  Results may be inaccurate.');
end

% Factor the correlation matrix and scale the integration limits and the
% Cholesky factor, to save having to divide everything by the diagonal
% elements later on.
C = chol(Rho);
c = diag(C);
a = a(:) ./ c;
b = b(:) ./ c;
C = C ./ repmat(c',m,1);

MCreps = 25; % use enough to get a decent error estimate
MCdims = m - isinf(nu);

p = zeros(outclass);
sigsq = Inf(outclass);
funevals = 0;
if verbose > 1
    disp(' estimate     error estimate    function evaluations');
    disp(' ---------------------------------------------------');
end
for i = 5:length(P)
    if (funevals + 2*MCreps*P(i)) > maxfunevals
        break
    end

    % Compute the Niederreiter point set generator
    q = 2.^((1:MCdims)/(MCdims+1));

    % Compute randomized quasi-MC estimate with P points
    [THat,sigsqTHat] = mvtqmcsub(MCreps, P(i), q, C, nu, a, b, outclass);
    funevals = funevals + 2*MCreps*P(i);

    % Recursively update the estimate and the error estimate
    p = p + (THat - p)./(1+sigsqTHat./sigsq);
    sigsq = sigsqTHat./(1+sigsqTHat./sigsq);

    % Compute a conservative estimate of error: 3.5 times the MC se
    err = 3.5*sqrt(sigsq);

    if verbose > 1
        disp(sprintf('  %.5g      %.5e           %d',p,err,funevals));
    end
    if err < tol
        if verbose > 0
            disp(sprintf(['Successfully satisfied error tolerance of %g in %d ' ...
                          'function evaluations.'],tol,funevals));
        end
        return
    end
end
warning('stats:mvtcdfqmc:ErrorTol', ...
       ['Unable to achieve error tolerance of %g in %d function evaluations.\n' ...
        'Increase the maximum number of function evaluations, or the error tolerance.'], ...
        tol, maxfunevals);


function [THat, sigsqTHat] = mvtqmcsub(MCreps, P, q, C, nu, a, b, outclass)
% Randomized Quasi-Monte-Carlo estimate of the integral

qq = [1:P]'*q;

THat = zeros(MCreps,1,outclass);
for rep = 1:MCreps
    % Generate a new random lattice of P points.  For MVT, this is in the
    % m-dimensional unit hypercube, for MVN, in the (m-1)-dimensional unit
    % hypercube.
    w = abs(2*mod(qq + repmat(rand(size(q)),P,1), 1) - 1);

    % Compute the mean of the integrand over all P of the points, and all P
    % of the antithetic points.
    THat(rep) = (F_qrsvn(a, b, C, nu, w) + F_qrsvn(a, b, C, nu, 1-w)) ./ 2;
end

% Return the MC mean and se^2
sigsqTHat = var(THat)./MCreps;
THat = mean(THat);


function TBar = F_qrsvn(a, b, C, nu, w)
% Integrand for computation of MVT probabilities
%
%   Given box bounds a and b (might be infinite), Cholesky factor C of the
%   correlation matrix, and degrees of freedom nu, compute the transformed MVT
%   integrand at each row of the quasi-random lattice of points w in the
%   m-dimensional unit hypercube.  If nu is Inf, compute the transformed MVN
%   integrand with points w in the (m-1)-dimensional unit hypercube.  Return
%   the mean of the P integrand values.

N = size(w,1); % number of quasirandom points
m = length(a); % number of dimensions
if isinf(nu)
    snu = 1;
else
    snu = chiq(w(:,m),nu)./sqrt(nu);
end
d = normp(snu.*a(1));       % a is already scaled by diag(C)
emd = normp(snu.*b(1)) - d; % b is already scaled by diag(C)
T = emd;
y = zeros(N,m,class(T));
for i = 2:m
    % limit how far out in the tail y can be to a finite value to prevent
    % infinite ysum and NaN ahat=a(i)-ysum or bhat=b(i)-ysum.
    z = min(max(d + emd.*w(:,i-1), eps/2), 1-eps/2);
    y(:,i-1) = normq(z);
    ysum = y*C(:,i);
    d = normp(snu.*a(i) - ysum);       % a is already scaled by diag(C)
    emd = normp(snu.*b(i) - ysum) - d; % b is already scaled by diag(C)
    T = T .* emd;
end
TBar = sum(T,1)./length(T);


function p = normp(z)
% Normal cumulative distribution function
p = 0.5 * erfcore(-z ./ sqrt(2),1);


function z = normq(p)
% Inverse of normal cumulative distribution function
z = -sqrt(2).*erfcore(2*p,4);


function x = chiq(p,nu)
% Inverse of chi cumulative distribution function
x = sqrt(gamicore(p,nu./2,gammaln(nu./2),3).*2);
