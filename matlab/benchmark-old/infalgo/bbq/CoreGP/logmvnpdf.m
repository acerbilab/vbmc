function y = logmvnpdf(X, Mu, Mat, flag)
%LOGMVNPDF Log of multivariate normal probability density function (pdf).
%   Y = LOGMVNPDF(X) returns the log of the probability density of the
%   multivariate normal distribution with zero mean and identity covariance
%   matrix, evaluated at each row of X.  Rows of the N-by-D matrix X
%   correspond to observations or points, and columns correspond to
%   variables or coordinates.  Y is an N-by-1 vector.
%
%   Y = LOGMVNPDF(X,MU) returns the log of the density of the multivariate
%   normal distribution with mean MU and identity covariance matrix,
%   evaluated at each row of X.  MU is a 1-by-D vector, or an N-by-D
%   matrix, in which case the density is evaluated for each row of X with
%   the corresponding row of MU.  MU can also be a scalar value, which
%   MVNPDF replicates to match the size of X.
%
%   Y = LOGMVNPDF(X,MU,MAT) returns the log of the density of the
%   multivariate normal distribution with mean MU and covariance MAT,
%   evaluated at each row of X.  MAT is a D-by-D matrix. Pass in the empty
%   matrix for MU to use its default value when you want to only specify
%   MAT.
%
%   Y = LOGMVNPDF(X,MU,MAT,FLAG) returns the log of the density of the
%   multivariate normal distribution with mean MU. Depending on FLAG, MAT
%   represents either the covariance matrix, precision (inverse covariance)
%   matrix or the cholesky decomposition of the covariance matrix. The
%   latter two options allow us to readily apply the same covariance matrix
%   to multiple points. 
%
%   If X is a 1-by-D vector, LOGMVNPDF replicates it to match the leading
%   dimension of MU or the trailing dimension of MAT.
%
%   Example:
%
%      mu = [1 -1]; Sigma = [.9 .4; .4 .3];
%      [X1,X2] = meshgrid(linspace(-1,3,25)', linspace(-3,1,25)');
%      X = [X1(:) X2(:)];
%      p = logmvnpdf(X, mu, Sigma);
%      surf(X1,X2,reshape(p,25,25));
%
%   See also MVNPDF, MVTPDF, MVNCDF, MVNRND, NORMPDF.

if nargin<1
    error('stats:mvnpdf:TooFewInputs','Requires at least one input.');
elseif ndims(X)~=2
    error('stats:mvnpdf:InvalidData','X must be a matrix.');
end

Sigma=[];
Prec=[];
R=[];

if nargin==3
    % If no flag is supplied, Mat is interpreted as a covariance matrix
    Sigma=Mat;
elseif nargin==4
    switch flag
        case 'covariance'
            % Mat is interpreted as a covariance matrix
            Sigma=Mat;
        case 'precision'
            % Mat is interpreted as the precision (inverse covariance)
            % matrix
            Prec=Mat;
        case 'cholesky'
            % Mat is interpreted as the cholesky decomposition of the
            % covariance
            R=Mat;
    end
end

% Get size of data.  Column vectors provisionally interpreted as multiple scalar data.
[n,d] = size(X);
if d<1
    error('stats:mvnpdf:TooFewDimensions','X must have at least one column.');
end

% Assume zero mean, data are already centered
if nargin < 2 || isempty(Mu)
    X0 = X;

% Get scalar mean, and use it to center data
elseif numel(Mu) == 1
    X0 = X - Mu;

% Get vector mean, and use it to center data
elseif ndims(Mu) == 2
    [n2,d2] = size(Mu);
    if d2 ~= d % has to have same number of coords as X
        error('stats:mvnpdf:InputSizeMismatch',...
              'X and MU must have the same number of columns.');
    elseif n2 == n % lengths match
        X0 = X - Mu;
    elseif n2 == 1 % mean is a single row, rep it out to match data
        X0 = X - repmat(Mu,n,1);
    elseif n == 1 % data is a single row, rep it out to match mean
        n = n2;
        X0 = repmat(X,n2,1) - Mu;
    else % sizes don't match
        error('stats:mvnpdf:InputSizeMismatch',...
              'X or MU must be a row vector, or X and MU must have the same number of rows.');
    end
    
else
    error('stats:mvnpdf:BadMu','MU must be a matrix.');
end

% Assume identity covariance, data are already standardized
if nargin < 3
    % Special case: if Sigma isn't supplied, then interpret X
    % and Mu as row vectors if they were both column vectors
    if (d == 1) && (numel(X) > 1)
        X0 = X0';
        [n,d] = size(X0);
    end
    xRinv = X0;
    logsqrtInvDetSigma = 0;
    
elseif ndims(Mat)<=2
    
    % Special case: if Mat is supplied, then use it to try to interpret
    % X and Mu as row vectors if they were both column vectors.
    
    if (d == 1) && (numel(X) > 1) && (size(Mat,1) == n)
            X0 = X0';
            [n,d] = size(X0);
        end

        % Make sure Sigma is the right size
        sz = size(Mat);
        if sz(1) ~= sz(2)
            error('stats:mvnpdf:BadCovariance',...
                  'The covariance must be a square matrix.');
        elseif ~isequal(sz, [d d])
            error('stats:mvnpdf:InputSizeMismatch',...
                  'The covariance must be a square matrix with size equal to the number of columns in X.');
        else
            if ~isempty(Prec)
                
                Rinv=chol(Prec);
                logsqrtInvDetSigma = sum(log(diag(Rinv)));
                % det of Precision matrix is the inverse of the det of the
                % covariance matrix
                
                quadform=X0*Prec*X0';
                
            else
                if ~isempty(Sigma)
                    % Make sure Sigma is a valid covariance matrix
                    [R,err] = chol(Sigma);
                    if err ~= 0
                        error('stats:mvnpdf:BadCovariance', ...
                              'The covariance must be symmetric and positive definite.');
                    end
                end
                % Create array of standardized data, vector of inverse det
                xRinv = X0 / R;
                logsqrtInvDetSigma = -sum(log(diag(R)));
                % As Sigma=R'*R, det(Sigma)=det(R)^2
                % The determinant of any matrix can be expressed as the
                % product of its eigenvalues. For a triangular matrix, its
                % eigenvalues are its diagonal.

                % Exponents in pdf are the inner products of the standardized
                % data
                quadform = sum(xRinv.^2, 2);
            end
        end

elseif ndims(R) > 2
    error('stats:mvnpdf:BadCovariance',...
          'The covariance must be a matrix');
end


y = -0.5*d*log(2*pi) + logsqrtInvDetSigma -0.5*quadform;
