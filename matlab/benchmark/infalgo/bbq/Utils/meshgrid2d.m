function [xx,yy] = meshgrid2d(x,y)
%MESHGRID   X and Y arrays for 3-D plots.
%   [X,Y] = MESHGRID(x,y) transforms the domain specified by vectors
%   x and y into arrays X and Y that can be used for the evaluation
%   of functions of two variables and 3-D surface plots.
%   The rows of the output array X are copies of the vector x and
%   the columns of the output array Y are copies of the vector y.
%
%   [X,Y] = MESHGRID(x) is an abbreviation for [X,Y] = MESHGRID(x,x).
%   [X,Y,Z] = MESHGRID(x,y,z) produces 3-D arrays that can be used to
%   evaluate functions of three variables and 3-D volumetric plots.
%
%   For example, to evaluate the function  x*exp(-x^2-y^2) over the 
%   range  -2 < x < 2,  -2 < y < 2,
%
%       [X,Y] = meshgrid(-2:.2:2, -2:.2:2);
%       Z = X .* exp(-X.^2 - Y.^2);
%       surf(X,Y,Z)
%
%   MESHGRID is like NDGRID except that the order of the first two input
%   and output arguments are switched (i.e., [X,Y,Z] = MESHGRID(x,y,z)
%   produces the same result as [Y,X,Z] = NDGRID(y,x,z)).  Because of
%   this, MESHGRID is better suited to problems in cartesian space,
%   while NDGRID is better suited to N-D problems that aren't spatially
%   based.  MESHGRID is also limited to 2-D or 3-D.
%
%   Class support for inputs X,Y,Z:
%      float: double, single
%
%   See also SURF, SLICE, NDGRID.

%   J.N. Little 1-30-92, CBM 2-11-92.
%   Copyright 1984-2004 The MathWorks, Inc. 
%   $Revision: 5.14.4.4 $  $Date: 2004/07/05 17:01:22 $

x = x(:)'; % Make sure x is a full row vector.
y = y(:);   % Make sure y is a full column vector.

%nx = length(xx); ny = length(yy);
xx = x(ones(length(y), 1),:);
yy = y(:,ones(1, length(x)));