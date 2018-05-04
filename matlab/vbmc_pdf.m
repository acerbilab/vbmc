function y = vbmc_pdf(X,vp,origflag,logflag,transflag)
%VBMC_PDF Probability density function of VBMC posterior approximation.

if nargin < 3 || isempty(origflag); origflag = true; end
if nargin < 4 || isempty(logflag); logflag = false; end
if nargin < 5 || isempty(transflag); transflag = false; end

% Convert points to transformed space
if origflag && ~isempty(vp.trinfo) && ~transflag
    % Xold = X;
    X = pdftrans(X,'dir',vp.trinfo);
end

[N,D] = size(X);
K = vp.K;                       % Number of components
lambda = vp.lambda(:)';         % LAMBDA is a row vector

mu_t(:,:) = vp.mu';             % MU transposed
sigma(1,:) = vp.sigma;

nf = 1/(2*pi)^(D/2)/prod(lambda)/K;  % Common normalization factor

y = zeros(N,1); % Allocate probability vector

% Compute pdf
for k = 1:K
    d2 = sum(bsxfun(@rdivide,bsxfun(@minus, X, mu_t(k,:)),sigma(k)*lambda).^2,2);
    y = y + nf/sigma(k)^D*exp(-0.5*d2);    
end

if logflag; y = log(y); end

% Apply Jacobian correction
if origflag && ~isempty(vp.trinfo)
    if logflag
        y = y + pdftrans(X,'logprob',vp.trinfo);
        
    else
        y = y .* pdftrans(X,'prob',vp.trinfo);
    end
end

end