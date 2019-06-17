function [K,dK] = gplite_covfun(hyp,X,covfun,Xstar,y)
%GPLITE_COVFUN Covariance function for lite Gaussian Process regression.
%   K = GPLITE_COVFUN(HYP,X,COVFUN,XSTAR) returns the covariance matrix 
%   between the training points X and test points XSTAR using the GP kernel 
%   function specified by COVFUN. HYP is a single column vector of 
%   covariance function hyperparameters. COVFUN can be a scalar or a 
%   character array specifying the covariance function, as follows:
%
%      COVFUN          COVARIANCE FUNCTION TYPE            HYPERPARAMETERS
%      1 or 'se'        squared exponential                 D+1
%      3 or 'matern'    Matérn (option: degree nu=1,3,5)    D+1
%      function_handle  custom                              NCOV
%
%   Unless specified otherwise, covariance functions use automatic relevance 
%   determination (ARD), that is a different length scale parameter for 
%   each input dimension. COVFUN can be a function handle to a custom 
%   covariance function. 
%
%   If a COVFUN supports additional options (e.g., the degree of a Matérn 
%   covariance), COVFUN can be a numerical or cell array whose 2nd and
%   further elements contain these additional options. For example, 
%   COVFUN = {'matern',3} and COVFUN = [3,3] equivalently indicate a Matérn 
%   covariance function of degree 3.
%
%   K = GPLITE_COVFUN(HYP,X,COVFUN) computes the self-covariance matrix
%   of the training points X.
%
%   K = GPLITE_COVFUN(HYP,X,COVFUN,'diag') computes only the diagonal of
%   the self-covariance matrix, returned as a column vector.
%   
%   [K,DK] = GPLITE_COVFUN(HYP,X,COVFUN) also computes the gradient DK of 
%   the self-covariance with respect to GP hyperparameters. DK is a 
%   N-by-N-by-NCOV matrix, where N is the number of training inputs and 
%   the i-th N-by-N matrix represents the gradient with respect to the i-th 
%   hyperparameter (out of NCOV).
%
%   NCOV = GPLITE_COVFUN([],X,COVFUN) returns the number of covariance 
%   function hyperparameters requested by covariance function COVFUN.
%
%   [NCOV,COVINFO] = GPLITE_COVFUN([],X,COVFUN,[],Y), where X is the matrix
%   of training inputs and Y the matrix of training targets, also returns a 
%   struct COVINFO with additional information about the covariance function
%   hyperparameters, with fields: LB (lower bounds); UB (upper bounds); PLB
%   (plausible lower bounds); PUB (plausible upper bounds); x0 (starting
%   point); covfun (COVFUN numerical identifier); covfun_name (COVFUN name).
%
%   See also GPLITE_MEANFUN, GPLITE_NOISEFUN.

if nargin < 4; Xstar = []; end
if nargin < 5; y = []; end

covfeat = [];
if isnumeric(covfun)
    covbase = covfun(1);
    covfeat = covfun(2:end);
elseif isa(covfun,'function_handle')
    covbase = covfun;
elseif iscell(covfun)
    covbase = covfun{1};
    if numel(covfun) > 1; covfeat = covfun{2}; end
elseif ischar(covfun)
    covbase = covfun;
end

if isa(covbase,'function_handle')
    if nargout > 1
        [K,dK] = covbase(hyp,X,covfeat);
    else
        K = covbase(hyp,X,covfeat);
    end
    return;
end

[N,D] = size(X);            % Number of training points and dimension

% Read number of cov function hyperparameters
switch covbase
    case {0,'0','seiso'}
        Ncov = 2;
        covbase = 0;
    case {1,'1','se','seard'}
        Ncov = D+1;
        covbase = 1;
    case {3,'3','matern','maternard'}
        Ncov = D+1;
        covbase = 3;
    otherwise
        if isnumeric(covfun); covfun = num2str(covfun); end
        error('gplite_covfun:UnknownCovFun',...
            ['Unknown covariance function identifier: [' covfun '].']);
end

% Return number of covariance function hyperparameters and additional info
if ischar(hyp)
    K = Ncov;
    if nargout > 1
        ToL = 1e-6;
        Big = exp(3);
        dK.LB = -Inf(1,Ncov);
        dK.UB = Inf(1,Ncov);
        dK.PLB = -Inf(1,Ncov);
        dK.PUB = Inf(1,Ncov);
        dK.x0 = NaN(1,Ncov);
        
        width = max(X) - min(X);
        if numel(y) <= 1; y = [0;1]; end
        height = max(y) - min(y);
        
        % Single length scale?
        isoflag = covbase == 0 || covbase == 2;
        
        if isoflag
            % single length scale
            dK.LB(1) = mean(log(width))+log(ToL);
            dK.UB(1) = mean(log(width*10));
            dK.PLB(1) = mean(log(width))+0.5*log(ToL);
            dK.PUB(1) = mean(log(width));
            dK.x0(1) = mean(log(std(X)));
            Nell = 1;
        else
            % length scales
            dK.LB(1:D) = log(width)+log(ToL);
            dK.UB(1:D) = log(width*10);
            dK.PLB(1:D) = log(width)+0.5*log(ToL);
            dK.PUB(1:D) = log(width);
            dK.x0(1:D) = log(std(X));
            Nell = D;
        end
            
        dK.LB(Nell+1) = log(height)+log(ToL);   % gp output scale
        dK.UB(Nell+1) = log(height*10);
        dK.PLB(Nell+1) = log(height)+0.5*log(ToL);
        dK.PUB(Nell+1) = log(height);
        dK.x0(Nell+1) = log(std(y));       
        
        % Plausible starting point
        idx_nan = isnan(dK.x0);
        dK.x0(idx_nan) = 0.5*(dK.PLB(idx_nan) + dK.PUB(idx_nan));
        
        dK.covfun = [covbase,covfeat];
        switch covbase
            case 0;  dK.covfun_name = 'seiso';
            case 1;  dK.covfun_name = 'se';
            case 2;  dK.covfun_name = 'materniso';
            case 3;  dK.covfun_name = 'matern';
        end
        
    end
    
    return;
end

[Nhyp,Ns] = size(hyp);      % Hyperparameters and samples

if Nhyp ~= Ncov
    error('gplite_covfun:WrongCovHyp', ...
        ['Expected ' num2str(Ncov) ' covariance function hyperparameters, ' num2str(Nhyp) ' passed instead.']);
end
if Ns > 1
    error('gplite_covfun:nosampling', ...
        'Covariance function output is available only for one-sample hyperparameter inputs.');
end

% Compute covariance function gradient wrt hyperparameters only if requested
compute_grad = nargout > 1;

if compute_grad     % Allocate space for gradient
    dK = zeros(N,N,Ncov);    
end

% Compute covariance function
ell = exp(hyp(1:D));
sf2 = exp(2*hyp(D+1));

switch covbase
    case 1  % SE ard
        if isempty(Xstar)        
            K = sq_dist(diag(1./ell)*X');
        elseif ischar(Xstar)
            K = zeros(size(X,1),1);
        else
            K = sq_dist(diag(1./ell)*X',diag(1./ell)*Xstar');
        end
        K = sf2 * exp(-K/2);
            
        if compute_grad
            for i = 1:D             % Grad of cov length scales
                dK(:,:,i) = K .* sq_dist(X(:,i)'/ell(i));
            end
            dK(:,:,D+1) = 2*K;        % Grad of cov output scale
        end
        
    case 3  % Matérn 1/3/5
        
        if isempty(covfeat); d = 5; else; d = covfeat(1); end
        
        switch d
            case 1, f = @(t) 1;               df = @(t) 1./t;     % df(t) = (f(t)-f'(t))/t
            case 3, f = @(t) 1 + t;           df = @(t) 1;
            case 5, f = @(t) 1 + t.*(1+t/3);  df = @(t) (1+t)/3;
          otherwise, error('Only 1, 3 and 5 allowed for d.');
        end

        % precompute distances
        if isempty(Xstar)
            tmp = sqrt(sq_dist(diag(sqrt(d)./ell)*X'));
        elseif ischar(Xstar)
            tmp = zeros(size(X,1),1);
        else
            tmp = sqrt(sq_dist(diag(sqrt(d)./ell)*X',diag(sqrt(d)./ell)*Xstar'));
        end
        
        K = sf2*f(tmp).*exp(-tmp);
        
        if compute_grad
            for i = 1:D
                Ki = sq_dist(sqrt(d)/ell(i)*X(:,i)');
                dK(:,:,i) = sf2*(df(tmp).*exp(-tmp)).*Ki;
                % dK(:,:,i) = dK(:,:,i).*(Ki > 1e-12);  % fix numerical errors
            end
            dK(:,:,D+1) = 2*K;
        end

end

end