function c = mat2cell2d(x,varargin)
%MAT2CELL Break matrix up into a cell array of matrices.
%   C = MAT2CELL(X,M,N) breaks up the 2-D array X into a cell array of  
%   adjacent submatrices of X. X is an array of size [ROW COL], M is the 
%   vector of row sizes (must sum to ROW) and N is the vector of column 
%   sizes (must sum to COL). The elements of M and N determine the size of
%   each cell in C by satisfying the following formula for I = 1:LENGTH(M)
%   and J = 1:LENGTH(N),
%
%       SIZE(C{I,J}) == [M(I) N(J)]
%
%   C = MAT2CELL(X,D1,D2,D3,...,DN) breaks up the multidimensional array X
%   and returns a multidimensional cell array of adjacent submatrices of X.
%   Each of the vector arguments, D1 through DN, should sum to the
%   respective dimension sizes of X, such that for P = 1:N,
%
%       SIZE(X,P) == SUM(DP) 
%
%   The elements of D1 through DN determine the size of each cell in C by
%   satisfying the formula for IP = 1:LENGTH(DP),
%
%       SIZE(C{I1,I2,I3,...,IN}) == [D1(I1) D2(I2) D3(I3) ... DN(IN)]
%
%   C = MAT2CELL(X,R) breaks up an array X by returning a single column
%   cell array, containing the rows of X. R must sum to the number of rows
%   of X. The elements of R determine the size of each cell in C, subject
%   to the following formula for I = 1:LENGTH(R),
%
%       SIZE(C{I},1) == R(I)
%
%   C = MAT2CELL(X,...,[],...) will return an empty cell array whose empty
%   size matches the lengths of the vector arguments, D1 through DN. Note
%   that the length of an empty vector is zero.
%
%   MAT2CELL supports all array types.
%
%	Example:
%	   X = [1 2 3 4; 5 6 7 8; 9 10 11 12];
%	   C = mat2cell(X,[1 2],[1 3])
%	    
%	See also CELL2MAT, NUM2CELL

% Copyright 1984-2006 The MathWorks, Inc.
% $Revision: 1.10.4.4 $  $Date: 2006/12/20 07:15:30 $

    % If matrix is 2-D, execute 2-D code for speed efficiency
        rowSizes = varargin{1};
        colSizes = varargin{2};
        rows = length(rowSizes);
        cols = length(colSizes);
        c = cell(rows,cols);
        % Construct each cell element by indexing into X with iterations of 
        %   matrix subscripts (i,j)
        rowStart = 0;
        for i=1:rows
            colStart = 0;
            for j=1:cols
                c{i,j} = x(rowStart+(1:rowSizes(i)),colStart+(1:colSizes(j)));
                colStart = colStart + colSizes(j);
            end
            rowStart = rowStart + rowSizes(i);
        end
 