function z = qtrapz(y,dim)
%QTRAPZ  Quick trapezoidal numerical integration.
%   Z = QTRAPZ(Y) computes an approximation of the integral of Y via
%   the trapezoidal method (with unit spacing).  To compute the integral
%   for spacing different from one, multiply Z by the spacing increment.
%
%   For vectors, QTRAPZ(Y) is the integral of Y. For matrices, QTRAPZ(Y)
%   is a row vector with the integral over each column. For N-D
%   arrays, QTRAPZ(Y) works across the first non-singleton dimension.
%
%   Z = QTRAPZ(Y,DIM) integrates across dimension DIM of Y. The length of X 
%   must be the same as size(Y,DIM).
%
%   QTRAPZ is up to 3-4 times faster than TRAPZ for large arrays.
%
%   See also TRAPZ.

% Luigi Acerbi <luigi.acerbi@nyu.edu>
% Version 1.0. Release date: Jul/20/2015.

% By default integrate along the first non-singleton dimension
if nargin < 2; dim = find(size(y)~=1,1); end    

% Behaves as sum on empty array
if isempty(y); z = sum(y,dim); return; end

% Compute dimensions of input matrix    
if isvector(y); n = 1; else n = ndims(y); end

switch n
    case {1,2}      % 1-D or 2-D array
        switch dim
            case 1
                z = sum(y,1) - 0.5*(y(1,:) + y(end,:));
            case 2
                z = sum(y,2) - 0.5*(y(:,1) + y(:,end));
            otherwise
                error('qtrapz:dimMismatch', 'DIM must specify one of the dimensions of Y.');
        end

    case 3      % 3-D array
        switch dim
            case 1
                z = sum(y,1) - 0.5*(y(1,:,:) + y(end,:,:));
            case 2
                z = sum(y,2) - 0.5*(y(:,1,:) + y(:,end,:));
            case 3
                z = sum(y,3) - 0.5*(y(:,:,1) + y(:,:,end));
            otherwise
                error('qtrapz:dimMismatch', 'DIM must specify one of the dimensions of Y.');
        end                

    case 4      % 4-D array
        switch dim
            case 1
                z = sum(y,1) - 0.5*(y(1,:,:,:) + y(end,:,:,:));
            case 2
                z = sum(y,2) - 0.5*(y(:,1,:,:) + y(:,end,:,:));
            case 3
                z = sum(y,3) - 0.5*(y(:,:,1,:) + y(:,:,end,:));
            case 4
                z = sum(y,4) - 0.5*(y(:,:,:,1) + y(:,:,:,end));
            otherwise
                error('qtrapz:dimMismatch', 'DIM must specify one of the dimensions of Y.');
        end                

    otherwise   % 5-D array or more
        for iDim = 1:n; index{iDim} = 1:size(y,iDim); end
        index1 = index;     index1{dim} = 1;
        indexend = index;   indexend{dim} = size(y,dim);
        try
            z = sum(y,dim) - 0.5*(y(index1{:}) + y(indexend{:}));
        catch
            error('qtrapz:dimMismatch', 'DIM must specify one of the dimensions of Y.');            
        end
end