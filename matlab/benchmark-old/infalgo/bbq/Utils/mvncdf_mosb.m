function [y,err] = mvncdf_mosb(varargin)
%MVNCDF Multivariate normal cumulative distribution function (cdf).
%   Y = MVNCDF(X) returns the cumulative probability of the multivariate
%   normal distribution with zero mean and identity covariance matrix,
%   evaluated at each row of X.  Rows of the N-by-D matrix X correspond to
%   observations or points, and columns correspond to variables or
%   coordinates.  Y is an N-by-1 vector.
%
%   Y = MVNCDF(X,MU,SIGMA) returns the cumulative probability of the
%   multivariate normal distribution with mean MU and (cholesky decomposition of the) covariance SIGMA,
%   evaluated at each row of X.  MU is a 1-by-D vector, and SIGMA is a D-by-D
%   symmetric, positive definite matrix.  MU can also be a scalar value, which
%   MVNCDF replicates to match the size of X.  Pass in the empty matrix for MU
%   to use its default value when you want to only specify SIGMA.
%
%   The multivariate normal cumulative probability at X is defined as the
%   probability that a random vector V, distributed as multivariate normal,
%   will fall within the semi-infinite rectangle with upper limits defined by
%   X, i.e., Pr{V(1)<=X(1), V(2)<=X(2), ... V(D)<=X(D)}.
%
%   Y = MVNCDF(XL,XU,MU,SIGMA) returns the multivariate normal cumulative
%   probability evaluated over the rectangle with lower and upper limits
%   defined by XL and XU, respectively.
%
%   [Y,ERR] = MVNCDF(...) returns an estimate of the error in Y.  For
%   bivariate and trivariate distributions, MVNCDF uses adaptive quadrature on
%   a transformation of the t density, based on methods developed by Drezner
%   and Wesolowsky, and by Genz, as described in the references.  The default
%   absolute error tolerance for these cases is 1e-8.  For four or more
%   dimensions, MVNCDF uses a quasi-Monte Carlo integration algorithm based on
%   methods developed by Genz and Bretz, as described in the references.  The
%   default absolute error tolerance for these cases is 1e-4.
%
%   [...] = MVNCDF(...,OPTIONS) specifies control parameters for the numerical
%   integration used to compute Y.  This argument can be created by a call to
%   STATSET.  Choices of STATSET parameters are:
%
%         'TolFun'      - Maximum absolute error tolerance.  Default is 1e-8
%                         when D < 4, or 1e-4 when D >= 4.
%         'MaxFunEvals' - Maximum number of integrand evaluations allowed when
%                         D >= 4.  Default is 1e7.  Ignored when D < 4.
%         'Display'     - Level of display output.  Choices are 'off' (the
%                         default), 'iter', and 'final'.  Ignored when D < 4.
%
%   Example:
%
%      mu = [1 -1]; Sigma = [.9 .4; .4 .3];
%      [X1,X2] = meshgrid(linspace(-1,3,25)', linspace(-3,1,25)');
%      X = [X1(:) X2(:)];
%      p = mvncdf(X, mu, Sigma);
%      surf(X1,X2,reshape(p,25,25));
%
%   See also MVTCDF, MVNPDF, MVNRND, NORMCDF.

%   References:
%      [1] Drezner, Z. and G.O. Wesolowsky (1989) "On the Computation of the
%          Bivariate Normal Integral", J.Statist.Comput.Simul., 35:101-107.
%      [2] Drezner, Z. (1994) "Computation of the Trivariate Normal Integral",
%          Mathematics of Computation, 63:289-294.
%      [3] Genz, A. (2004) "Numerical Computation of Rectangular Bivariate
%          and Trivariate Normal and t Probabilities", Statistics and
%          Computing, 14(3):251-260.
%      [4] Genz, A. and F. Bretz (1999) "Numerical Computation of Multivariate
%          t Probabilities with Application to Power Calculation of Multiple
%          Contrasts", J.Statist.Comput.Simul., 63:361-378.
%      [5] Genz, A. and F. Bretz (2002) "Comparison of Methods for the
%          Computation of Multivariate t Probabilities", J.Comp.Graph.Stat.,
%          11(4):950-971.

%   Copyright 2005-2007 The MathWorks, Inc.
%   $Revision: 1.1.6.7 $  $Date: 2007/12/10 23:05:51 $

% Strip off an options structure if it's there.
if isstruct(varargin{end})
    opts = statset(statset('mvncdf'), varargin{end});
    nin = nargin - 1;
else
    opts = statset('mvncdf');
    nin = nargin;
end

if nin < 1
    error('stats:mvncdf:TooFewInputs','Requires at least one input.');

elseif nin < 4 % MVNCDF(XU,MU,SIGMA)
    upperLimitOnly = true;
    XU = varargin{1};
    if ndims(XU)~=2
        error('stats:mvncdf:InvalidData','X must be a matrix.');
    end
    XL = -Inf(size(XU),class(XU));
    if nin > 1, mu = varargin{2}; else mu = []; end
    if nin > 2, Sigma = varargin{3}; else Sigma = []; end

else % MVNCDF(XL,XU,MU,SIGMA)
    upperLimitOnly = false;
    XL = varargin{1};
    XU = varargin{2};
    mu = varargin{3};
    Sigma = varargin{4};
    if ndims(XU)~=2 || ~isequal(size(XL),size(XU))
        error('stats:mvncdf:InvalidData','XL and XU must be matrices and have the same size.');
    elseif any(any(XL > XU))
        error('stats:mvncdf:InvalidData','XL must be less than or equal to XU.');
    end
end

% Get size of data.  Column vectors provisionally interpreted as multiple scalar data.
[n,d] = size(XU);
if d<1
    error('stats:mvncdf:TooFewDimensions','X must have at least one column.');
end

% Assume zero mean, data are already centered
if isempty(mu)
    XL0 = XL;
    XU0 = XU;

% Get scalar mean, and use it to center data
elseif isscalar(mu)
    XL0 = XL - mu;
    XU0 = XU - mu;

% Get vector mean, and use it to center data
elseif isvector(mu)
    [n2,d2] = size(mu);
    if d2 ~= d % has to have same number of coords as X
        error('stats:mvncdf:InputSizeMismatch',...
              'MU must be a row vector with the same number of columns as X.');
    elseif n2 == 1 % mean is a single row, rep it out to match data
        XL0 = XL - repmat(mu,n,1);
        XU0 = XU - repmat(mu,n,1);
    elseif n2 == n
        % if X and mu are column vectors and lengths match, provisionally
        % interpret this as multivariate data
        XL0 = XL - mu;
        XU0 = XU - mu;
    else % sizes don't match
        error('stats:mvncdf:InputSizeMismatch',...
              'MU must be a row vector with the same number of columns as X.');
    end
else
    error('stats:mvncdf:InvalidMu', 'MU must be a row vector.');
end

% Assume identity covariance, data are already standardized
if isempty(Sigma)
    % Special case: if Sigma isn't supplied, then interpret X
    % and MU as row vectors if they were both column vectors
    if d == 1
        XL0 = XL0';
        XU0 = XU0';
        [n,d] = size(XU0);
    end
    Sigma = eye(d);

else
    % Special case: if Sigma is supplied, then use it to try to interpret
    % X and MU as row vectors if they were both column vectors.
    if (d == 1)
        if size(Sigma,1) == n
            XL0 = XL0';
            XU0 = XU0';
            [n,d] = size(XU0);
        elseif ~isscalar(mu)
            error('stats:mvncdf:InputSizeMismatch',...
                  'MU must be a row vector with the same number of columns as X.');
        end
    end

    % Make sure Sigma is a valid covariance matrix
    sz = size(Sigma);
    if sz(1) ~= sz(2)
        error('stats:mvncdf:BadCovariance',...
              'SIGMA must be a square matrix.');
    elseif ~isequal(sz, [d d])
        error('stats:mvncdf:InputSizeMismatch',...
              'SIGMA must be a square matrix with size equal to the number of columns in X.');
    else
        T=Sigma;
        Sigma=T'*T;
        %[T,err] = cholcov(Sigma,0);
%         if err ~= 0
%             error('stats:mvncdf:BadCovariance',...
%                   'SIGMA must be symmetric and positive definite.');
%         end
    end
end

% Standardize Sigma and X to correlation if necessary
s = sqrt(diag(Sigma));
Rho = Sigma ./ (s*s');
XL0 = XL0 ./ repmat(s',n,1);
XU0 = XU0 ./ repmat(s',n,1);

% Call the appropriate integration routine for the umber of dimensions.
if d == 1
    y = normcdf(XU0,0,1) - normcdf(XL0,0,1);
    if nargout > 1
        err = NaN(size(y),class(y));
    end

elseif d <= 3
    tol = opts.TolFun; if isempty(tol), tol = 1e-8; end
    if d == 2, rho = Rho(2); else rho = Rho([2 3 6]); end
    if upperLimitOnly
        if d == 2
            y = bvncdf(XU0, rho, tol);
        else
            y = tvncdf(XU0, rho, tol);
        end
    else % lower and upper limits
        % Compute the probability over the rectangle as sums and differences
        % of integrals over semi-infinite half-rectangles.  For degenerate
        % rectangles, force an exact zero by making each piece exactly zero.
        equalLims = (XL0==XU0);
        XL0(equalLims) = -Inf;
        XU0(equalLims) = -Inf;
        y = zeros(n,1,superiorfloat(XL0,XU0,Rho));
        for i = 0:d
            k = nchoosek(1:d,i);
            for j = 1:size(k,1)
                X = XU0; X(:,k(j,:)) = XL0(:,k(j,:));
                if d == 2
                    y = y + (-1)^i * bvncdf(X, rho, tol/4);
                else
                    y = y + (-1)^i * tvncdf(X, rho, tol/8);
                end
            end
        end
    end
    if nargout > 1
        err = repmat(cast(tol,class(y)),size(y));
    end

elseif d <= 25
    tol = opts.TolFun; if isempty(tol), tol = 1e-4; end
    maxfunevals = opts.MaxFunEvals;
    verbose = find(strcmp(opts.Display,{'off' 'final' 'iter'})) - 1;
    y = zeros(n,1,superiorfloat(XL0,XU0,Rho));
    err = zeros(n,1,class(y));
    for i = 1:n
        [y(i),err(i)] = mvtcdfqmc(XL0(i,:),XU0(i,:),Rho,Inf,tol,maxfunevals,verbose);
    end

else
    error('stats:mvncdf:DimensionTooLarge',...
          'Number of dimensions must be less than or equal to 25.');
end

y(y<0) = 0; % repair roundoff problems; max would drop NaNs
y(y>1) = 1; 
end


%----------------------------------------------------------------------
function p = bvncdf(b,rho,tol)
% CDF for the bivariate normal.
%
% Implements the unnumbered equation between (3) and (4) in Section 2.2 of
% Genz (2004), integrating in terms of theta between asin(rho) and +/- pi/2,
% using adaptive quadrature.

n = size(b,1);
if rho == 0
    p = cast(prod(normcdf(b),2), superiorfloat(b,rho));
else
    if rho > 0
        p1 = normcdf(min(b,[],2));
        p1(any(isnan(b),2)) = NaN;
    else
        p1 = normcdf(b(:,1))-normcdf(-b(:,2));
        p1(p1<0) = 0; % max would drop NaNs
    end
    if abs(rho) < 1
        loLimit = asin(rho);
        hiLimit = sign(rho).*pi./2;
        p2 = zeros(size(p1),class(p1));
        for i = 1:n
            b1 = b(i,1); b2 = b(i,2);
            if isfinite(b1) && isfinite(b2)
                p2(i) = quadgk(@bvnIntegrand,loLimit,hiLimit,'AbsTol',tol,'RelTol',0);
            else
                % This piece is zero if either limit is +/- infinity.  If
                % either is NaN, p1 will already be NaN.
            end
        end
    else
        p2 = zeros(class(p1));
    end
    p = cast(p1 - p2./(2.*pi), superiorfloat(b,rho));
end

    function integrand = bvnIntegrand(theta)
        % Integrand is exp( -(b1.^2 + b2.^2 - 2*b1*b2*sin(theta))/(2*cos(theta).^2) )
        sintheta = sin(theta);
        cossqtheta = cos(theta).^2; % always positive
        integrand = exp(-((b1*sintheta - b2).^2 ./ cossqtheta + b1.^2)/2);
    end
end


%----------------------------------------------------------------------
function p = tvncdf(b,rho,tol)
% CDF for the trivariate normal
%
% Implements equation (14) in Section 3.2 of Genz (2004), integrating each
% term in (14) separately in terms of theta between 0 and asin(rho_j1), using
% adaptive quadrature.

n = size(b,1);

% Find a permutation that makes rho_32 == max(rho)
[dum,imax] = max(abs(rho));
if imax == 1 % swap 1 and 3
    rho_21 = rho(3); rho_31 = rho(2); rho_32 = rho(1);
    b = b(:,[3 2 1]);
elseif imax == 2 % swap 1 and 2
    rho_21 = rho(1); rho_31 = rho(3); rho_32 = rho(2);
    b = b(:,[2 1 3]);
else % imax == 3
    rho_21 = rho(1); rho_31 = rho(2); rho_32 = rho(3);
    % b already in correct order
end

% CDF for the normal distribution.
Phi = @(z) 0.5 * erfc(-z ./ sqrt(2));

p1 = Phi(b(:,1)).*bvncdf(b(:,2:3),rho_32,tol/3);

if abs(rho_21) > 0
    loLimit = 0;
    hiLimit = asin(rho_21);
    rho_j1 = rho_21;
    rho_k1 = rho_31;
    p2 = zeros(size(p1),class(p1));
    for i = 1:n
        b1 = b(i,1); bj = b(i,2); bk = b(i,3);
        if isfinite(b1) && isfinite(bj) && ~isnan(bk)
            p2(i) = quadgk(@tvnIntegrand,loLimit,hiLimit,'AbsTol',tol/3,'RelTol',0);
        else
            % This piece is zero if either limit is +/- infinity.  If
            % either is NaN, p1 will already be NaN.
        end
    end
else
    p2 = zeros(class(p1));
end

if abs(rho_31) > 0
    loLimit = 0;
    hiLimit = asin(rho_31);
    rho_j1 = rho_31;
    rho_k1 = rho_21;
    p3 = zeros(size(p1),class(p1));
    for i = 1:n
        b1 = b(i,1); bj = b(i,3); bk = b(i,2);
        if isfinite(b1) && isfinite(bj) && ~isnan(bk)
            p3(i) = quadgk(@tvnIntegrand,loLimit,hiLimit,'AbsTol',tol/3,'RelTol',0);
        else
            % This piece is zero if either limit is +/- infinity.  If
            % either is NaN, p1 will already be NaN.
        end
    end
else
    p3 = zeros(class(p1));
end

p = cast(p1 + (p2 + p3)./(2.*pi), superiorfloat(b,rho));

    function integrand = tvnIntegrand(theta)
        % Integrand is exp( -(b1.^2 + bj.^2 - 2*b1*bj*sin(theta))/(2*cos(theta).^2) )
        sintheta = sin(theta);
        cossqtheta = cos(theta).^2; % always positive
        expon = ((b1*sintheta - bj).^2 ./ cossqtheta + b1.^2)/2;

        sinphi = sintheta .* rho_k1 ./ rho_j1;
        numeru = bk.*cossqtheta - b1.*(sinphi - rho_32.*sintheta) ...
                                - bj.*(rho_32 - sintheta.*sinphi);
        denomu = sqrt(cossqtheta.*(cossqtheta -sinphi.*sinphi ...
                                   - rho_32.*(rho_32 - 2.*sintheta.*sinphi)));
        integrand = exp(-expon) .* Phi(numeru./denomu);
    end
end
