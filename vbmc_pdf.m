function [y,dy] = vbmc_pdf(vp,X,origflag,logflag,transflag,df)
%VBMC_PDF Probability density function of VBMC posterior approximation.
%   Y = VBMC_PDF(VP,X) returns the probability density of the variational 
%   posterior VP evaluated at each row of X.  Rows of the N-by-D matrix X 
%   correspond to observations or points, and columns correspond to variables 
%   or coordinates. Y is an N-by-1 vector.
%
%   Y = VBMC_PDF(VP,X,ORIGFLAG) returns the value of the posterior density
%   evaluated in the original parameter space for ORIGFLAG=1 (default), or 
%   in the transformed VBMC space if ORIGFLAG=0.
%
%   Y = VBMC_PDF(VP,X,ORIGFLAG,LOGFLAG) returns the value of the log 
%   posterior density if LOGFLAG=1, otherwise the posterior density for
%   LOGFLAG=0 (default).
%
%   Y = VBMC_PDF(VP,X,ORIGFLAG,LOGFLAG,TRANSFLAG) for TRANSFLAG=1 assumes 
%   that X is already specified in tranformed VBMC space. Otherwise, X is 
%   specified in the original parameter space (default TRANSFLAG=0).
%
%   Y = VBMC_PDF(VP,X,ORIGFLAG,LOGFLAG,TRANSFLAG,DF) returns the probability 
%   density of an heavy-tailed version of the variational posterior, 
%   in which the multivariate normal components have been replaced by
%   multivariate t-distributions with DF degrees of freedom. The default is
%   DF=Inf, limit in which the t-distribution becomes a multivariate normal.
%
%   See also VBMC, VBMC_RND.

if nargin < 3 || isempty(origflag); origflag = true; end
if nargin < 4 || isempty(logflag); logflag = false; end
if nargin < 5 || isempty(transflag); transflag = false; end
if nargin < 6 || isempty(df); df = Inf; end

gradflag = nargout > 1;     % Compute gradient

% Convert points to transformed space
if origflag && ~isempty(vp.trinfo) && ~transflag
    % Xold = X;
    X = warpvars_vbmc(X,'dir',vp.trinfo);
end

[N,D] = size(X);
K = vp.K;                       % Number of components
w = vp.w;                       % Mixture weights
lambda = vp.lambda(:)';         % LAMBDA is a row vector

mu_t(:,:) = vp.mu';             % MU transposed
sigma(1,:) = vp.sigma;

y = zeros(N,1); % Allocate probability vector
if gradflag; dy = zeros(N,D); end
    
if ~isfinite(df) || df == 0
    % Compute pdf of variational posterior
    
    % Common normalization factor
    nf = 1/(2*pi)^(D/2)/prod(lambda);
    
    for k = 1:K
        d2 = sum(bsxfun(@rdivide,bsxfun(@minus,X,mu_t(k,:)),sigma(k)*lambda).^2,2);
        nn = nf*w(k)/sigma(k)^D*exp(-0.5*d2);
        y = y + nn;
        if gradflag
            dy = dy - bsxfun(@times,nn, ...
                bsxfun(@rdivide,bsxfun(@minus,X,mu_t(k,:)),lambda.^2.*sigma(k)^2));
        end
    end
else
    % Compute pdf of heavy-tailed variant of variational posterior
    
    if df > 0    
        % (This uses a multivariate t-distribution which is not the same thing 
        % as the product of D univariate t-distributions)

        % Common normalization factor
        nf = exp(gammaln((df+D)/2) - gammaln(df/2))/(df*pi)^(D/2)/prod(lambda);

        for k = 1:K
            d2 = sum(bsxfun(@rdivide,bsxfun(@minus, X, mu_t(k,:)),sigma(k)*lambda).^2,2);
            nn = nf*w(k)/sigma(k)^D*(1+d2/df).^(-(df+D)/2);
            y = y + nn;
            if gradflag
                error('Gradient of heavy-tailed pdf not supported yet.');
                dy = dy - bsxfun(@times,nn, ...
                    bsxfun(@rdivide,bsxfun(@minus,X,mu_t(k,:)),lambda.^2.*sigma(k)^2));
            end
        end  
    else
        % (This uses a product of D univariate t-distributions)
        
        df = abs(df);

        % Common normalization factor
        nf = (exp(gammaln((df+1)/2) - gammaln(df/2))/sqrt(df*pi))^D/prod(lambda);

        for k = 1:K            
            d2 = bsxfun(@rdivide,bsxfun(@minus, X, mu_t(k,:)),sigma(k)*lambda).^2;            
            nn = nf*w(k)/sigma(k)^D*prod((1+d2/df).^(-(df+1)/2),2);
            y = y + nn;
            if gradflag
                error('Gradient of heavy-tailed pdf not supported yet.');
            end
        end    
    end
        
end

if logflag
    if gradflag; dy = bsxfun(@rdivide,dy,y); end
    y = log(y);
end

% Apply Jacobian correction
if origflag && ~isempty(vp.trinfo)
    if logflag
        y = y - warpvars_vbmc(X,'logprob',vp.trinfo);
        if gradflag
            error('vbmc_pdf:NoOriginalGrad',...
                'Gradient computation in original space not supported yet.');
            dy = dy - warpvars_vbmc(X,'g',vp.trinfo);
        end
    else
        y = y ./ warpvars_vbmc(X,'prob',vp.trinfo);
    end
end

end