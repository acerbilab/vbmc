function t = trace3(A)
%TRACE  Sum of diagonal elements.
%   TRACE(A) is the sum of the diagonal elements of A, which is
%   also the sum of the eigenvalues of A.
%
%   Class support for input A:
%      float: double, single

%   Copyright 1984-2007 The MathWorks, Inc. 
%   $Revision: 5.8.4.2 $  $Date: 2007/11/01 12:38:53 $

[m,m,n] = size(A);

stack_inds = (0:(n-1))'*m^2;
mat_inds = 1:m+1:m^2;

inds = bsxfun(@plus, stack_inds, mat_inds);
%inds = inds(:);

t = sum(A(inds),2);
t = reshape(t,[1 1 n]);
