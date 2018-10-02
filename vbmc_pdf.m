function [y,dy] = vbmc_pdf(X,vp,origflag,logflag,transflag)
%VBMC_PDF Probability density function of VBMC posterior approximation.

if nargin < 3 || isempty(origflag); origflag = true; end
if nargin < 4 || isempty(logflag); logflag = false; end
if nargin < 5 || isempty(transflag); transflag = false; end

gradflag = nargout > 1;     % Compute gradient

% Convert points to transformed space
if origflag && ~isempty(vp.trinfo) && ~transflag
    % Xold = X;
    X = warpvars(X,'dir',vp.trinfo);
end

[N,D] = size(X);
K = vp.K;                       % Number of components
w = vp.w;                       % Mixture weights
lambda = vp.lambda(:)';         % LAMBDA is a row vector

mu_t(:,:) = vp.mu';             % MU transposed
sigma(1,:) = vp.sigma;

nf = 1/(2*pi)^(D/2)/prod(lambda);  % Common normalization factor

y = zeros(N,1); % Allocate probability vector
if gradflag; dy = zeros(N,D); end
    
% Compute pdf
for k = 1:K
    d2 = sum(bsxfun(@rdivide,bsxfun(@minus, X, mu_t(k,:)),sigma(k)*lambda).^2,2);
    nn = nf*w(k)/sigma(k)^D*exp(-0.5*d2);
    y = y + nn;
    if gradflag
        dy = dy - bsxfun(@times,nn, ...
            bsxfun(@rdivide,bsxfun(@minus,X,mu_t(k,:)),lambda.^2.*sigma(k)^2));
    end
end

if logflag
    if gradflag; dy = bsxfun(@rdivide,dy,y); end
    y = log(y);
end

% Apply Jacobian correction
if origflag && ~isempty(vp.trinfo)
    if logflag
        y = y - warpvars(X,'logprob',vp.trinfo);
        if gradflag
            error('vbmc_pdf:NoOriginalGrad',...
                'Gradient computation in original space not supported yet.');
            dy = dy - warpvars(X,'g',vp.trinfo);
        end
    else
        y = y ./ warpvars(X,'prob',vp.trinfo);
    end
end

end