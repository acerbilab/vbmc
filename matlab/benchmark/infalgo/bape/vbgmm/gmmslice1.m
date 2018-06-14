function [w1,mu1,v1] = gmmslice1(vec,x,w,Mu,Sigma,ischol)
%GMMSLICE1 Conditional Gaussian mixture model sliced in one dimension.

if nargin < 6 || isempty(ischol); ischol = 0; end

TolZero = 1e-12;

k = numel(w);   % Number of components
nvars = numel(x);   % Number of dimensions

if sum(abs(vec) > TolZero) > 1  % Non-coordinate wise slice
    % Rotate space to (1,0,...,0)
    vec1 = zeros(1,nvars);
    vec1(1) = 1;
    d = 1;
    notd = true(1,nvars); notd(1) = false;
    R = rotmatx2y(vec,vec1);
    Mu = bsxfun(@minus,Mu,x)*R';
    x = zeros(size(x));
    for i = 1:k
        if ischol
            S = R*Sigma(:,:,i);
            iU = R*(Sigma(:,:,i) \ eye(nvars));
            Lambda(:,:,i) = iU*iU';
            Sigma(:,:,i) = S*S';
        else
            Sigma(:,:,i) = R*Sigma(:,:,i)*R';
            Lambda(:,:,i) = inv(Sigma(:,:,i));
        end
    end
else    
    d = find(vec ~= 0,1);
    notd = vec == 0;
    Mu = bsxfun(@minus,Mu,x);    
    x = zeros(size(x));
    for i = 1:k
        if ischol
            iU = Sigma(:,:,i) \ eye(nvars);     
            Lambda(:,:,i) = iU*iU';
            Sigma(:,:,i) = Sigma(:,:,i)*Sigma(:,:,i)';
        else
            Lambda(:,:,i) = inv(Sigma(:,:,i));
        end
    end    
end

lw = log(w);
v1 = zeros(1,k);
mu1 = zeros(1,k);
lw1 = zeros(1,k);

for i = 1:k
    v1(i) = 1/Lambda(d,d,i);
    mu1(i) = Mu(i,d) - v1(i)*Lambda(d,notd,i)*(x(notd) - Mu(i,notd))';
    lw1(i) = lw(i) + umvnlogpdf(x(notd),Mu(i,notd),Sigma(notd,notd,i));
end

lw1 = exp(lw1 - max(lw1));
w1 = lw1./sum(lw1);
 
end

function y = umvnlogpdf(X,Mu,Sigma)
%UMVNLOGPDF Unnormalized multivariate normal log pdf.

X0 = bsxfun(@minus,X,Mu);
[R,err] = chol(Sigma);
if err ~= 0; error('Cannot slice along the given direction, matrix is not positive definite.'); end
xRinv = X0 / R;
logSqrtDetSigma = sum(log(diag(R)));
quadform = sum(xRinv.^2, 2);
y = -0.5*quadform - logSqrtDetSigma;

end

