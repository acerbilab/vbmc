function r = msplinetrapezrnd(a,u,v,b,n)
%MSPLINETRAPEZRND Random arrays from the multivariate spline-trapezoidal distribution.
%   R = MSPLINETRAPEZRND(A,U,V,B) returns an N-by-D matrix R of random 
%   vectors chosen from the multivariate spline-trapezoidal distribution 
%   with external bounds A and B and internal points U and V. A, U, V and B 
%   are N-by-D matrices, and MSPLINETRAPEZRND generates each row of R using 
%   the corresponding row of A, U, V and B.
%
%   R = MSPLINETRAPEZRND(A,U,V,B,N) returns a N-by-D matrix R of random 
%   vectors chosen from the multivariate spline-trapezoidal distribution 
%   with external bounds A and B and internal points U and V.
%
%   See also MSPLINETRAPEZPDF.

% Luigi Acerbi 2022

[Na,Da] = size(a);
[Nu,Du] = size(u);
[Nv,Dv] = size(v);
[Nb,Db] = size(b);

if nargin < 3 || isempty(n)
    n = max([Na,Nu,Nv,Nb]);
else
    if (Na ~= 1 && Na ~= n) || (Nb ~= 1 && Nb ~= n) || ...
            (Nu ~= 1 && Nu ~= n) || (Nv ~= 1 && Nv ~= n)
        error('msplinetrapezrnd:SizeError', ...
            'A, U, V, B should be 1-by-D or N-by-D arrays.');
    end    
end
if Na ~= Nb || Da ~= Db || Na ~= Nu || Da ~= Du || Na ~= Nv || Da ~= Dv
    error('msplinetrapezrnd:SizeError', ...
        'A, U, V, B should be arrays of the same size.');
end

D = Da;

if size(a,1) == 1; a = repmat(a,[n,1]); end
if size(u,1) == 1; u = repmat(u,[n,1]); end
if size(v,1) == 1; v = repmat(v,[n,1]); end
if size(b,1) == 1; b = repmat(b,[n,1]); end

r = zeros(n,D);

% Sample one dimension at a time
for d = 1:D
    % Compute maximum of one-dimensional pdf
    x0 = 0.5*(u(:,d) + v(:,d));
    y_max = msplinetrapezpdf(x0,a(:,d),u(:,d),v(:,d),b(:,d));
    
    idx = true(n,1);
    r1 = zeros(n,1);
    n1 = sum(idx);
    
    % Keep doing rejection sampling
    while n1 > 0        
        % Uniform sampling in the box
        r1(idx) = bsxfun(@plus, a(idx,d), bsxfun(@times, rand(n1,1), b(idx,d) - a(idx,d)));
        
        % Rejection sampling
        z1 = rand(n1,1) .* y_max(idx);
        y1 = msplinetrapezpdf(r1(idx),a(idx,d),u(idx,d),v(idx,d),b(idx,d));
        
        idx_new = false(n,1);        
        idx_new(idx) = z1 > y1; % Resample points outside
        
        idx = idx_new;
        n1 = sum(idx);
    end
    
    % Assign d-th dimension
    r(:,d) = r1;
end