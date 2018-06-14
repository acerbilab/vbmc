function m = cell2mat2d(c)
%CELL2MAT Convert the contents of a cell array into a single matrix.
%   M = CELL2MAT(C) converts a multidimensional cell array with contents of
%   the same data type into a single matrix. The contents of C must be able
%   to concatenate into a hyperrectangle. Moreover, for each pair of
%   neighboring cells, the dimensions of the cell's contents must match,
%   excluding the dimension in which the cells are neighbors. This constraint
%   must hold true for neighboring cells along all of the cell array's
%   dimensions.
%
%   The dimensionality of M, i.e. the number of dimensions of M, will match
%   the highest dimensionality contained in the cell array.
%
%   CELL2MAT is not supported for cell arrays containing cell arrays or
%   objects.
%
%	Example:
%	   C = {[1] [2 3 4]; [5; 9] [6 7 8; 10 11 12]};
%	   M = cell2mat(C)
%
%	See also MAT2CELL, NUM2CELL

% Copyright 1984-2006 The MathWorks, Inc.
% $Revision: 1.10.4.7 $  $Date: 2006/11/11 22:43:58 $

% If cell array is 2-D, execute 2-D code for speed efficiency
    rows = size(c,1);
    cols = size(c,2);   
    if (rows < cols)
        m = cell(rows,1);
        % Concatenate one dim first
        for n=1:rows
            m{n} = cat(2,c{n,:});
        end
        % Now concatenate the single column of cells into a matrix
        m = cat(1,m{:});
    else
        m = cell(1, cols);
        % Concatenate one dim first
        for n=1:cols
            m{n} = cat(1,c{:,n});
        end    
        % Now concatenate the single column of cells into a matrix
        m = cat(2,m{:});
    end
