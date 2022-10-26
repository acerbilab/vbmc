function r = msmoothboxrnd(a,b,sigma,n)
%MSMOOTHBOXRND Random arrays from the multivariate smooth-box distribution.
%   R = MSMOOTHBOXRND(A,B,SIGMA) returns an N-by-D matrix R of random 
%   vectors chosen from the multivariate smooth-box distribution 
%   with pivots A and B and scale SIGMA. A, B and SIGMA are N-by-D matrices, 
%   and MSMOOTHBOXRND generates each row of R using the corresponding row 
%   of A, B and SIGMA.
%
%   R = MSMOOTHBOXRND(A,B,SIGMA,N) returns a N-by-D matrix R of random 
%   vectors chosen from the multivariate smooth-box distribution 
%   with pivots A and B and scale SIGMA.
%
%   See also MSMOOTHBOXPDF.

% Luigi Acerbi 2022

[Na,Da] = size(a);
[Nb,Db] = size(b);
[Nsigma,Dsigma] = size(sigma);

if any(sigma(:) <= 0)
    error('msmoothboxrnd:NonPositiveSigma', ...
        'All elements of SIGMA should be positive.');    
end

if nargin < 4 || isempty(n)
    n = max([Na,Nb,Nsigma]);
else
    if (Na ~= 1 && Na ~= n) || (Nb ~= 1 && Nb ~= n) || ...
            (Nsigma ~= 1 && Nsigma ~= n)
        error('msmoothboxrnd:SizeError', ...
            'A, B, SIGMA should be 1-by-D or N-by-D arrays.');
    end    
end
if Na ~= Nb || Da ~= Db || Na ~= Nsigma || Da ~= Dsigma
    error('msmoothboxrnd:SizeError', ...
        'A, B, SIGMA should be arrays of the same size.');
end

D = Da;

if size(a,1) == 1; a = repmat(a,[n,1]); end
if size(b,1) == 1; b = repmat(b,[n,1]); end
if size(sigma,1) == 1; sigma = repmat(sigma,[n,1]); end

r = zeros(n,D);

nf = 1 + 1/sqrt(2*pi)./sigma.*(b - a);

% Sample one dimension at a time
for d = 1:D    
    % Draw component (left/right tails or plateau)
    u = nf(:,d) .* rand(n,1);
    
    % Left Gaussian tails
    idx = u < 0.5;
    if any(idx)
        z1 = abs(randn(sum(idx),1).*sigma(idx,d));    
        r(idx,d) = a(idx) - z1;
    end
    
    % Right Gaussian tails
    idx = (u >= 0.5 & u < 1);
    if any(idx)
        z1 = abs(randn(sum(idx),1).*sigma(idx,d));
        r(idx,d) = b(idx) + z1;
    end
    
    % Plateau
    idx = u >= 1;
    if any(idx)
        r(idx,d) = a(idx,d) + (b(idx,d) - a(idx,d)).*rand(sum(idx),1);
    end
end