function K = kron3d(A,B)
%KRON   Kronecker tensor product.
%   KRON(X,Y) is the Kronecker tensor product of X and Y.
%   The result is a large matrix formed by taking all possible
%   products between the elements of X and those of Y.   For
%   example, if X is 2 by 3, then KRON(X,Y) is
%
%      [ X(1,1)*Y  X(1,2)*Y  X(1,3)*Y
%        X(2,1)*Y  X(2,2)*Y  X(2,3)*Y ]
%
%   If either X or Y is sparse, only nonzero elements are multiplied
%   in the computation, and the result is sparse.
%
%   Class support for inputs X,Y:
%      float: double, single

%   Previous versions by Paul Fackler, North Carolina State,
%   and Jordan Rosenthal, Georgia Tech.
%   Copyright 1984-2004 The MathWorks, Inc. 
%   $Revision: 5.17.4.2 $ $Date: 2004/06/25 18:52:18 $

[ma,na,oa] = size(A);
[mb,nb,ob] = size(B);

[ia,ib] = meshgrid2d(1:ma,1:mb);
[ja,jb] = meshgrid2d(1:na,1:nb);
[ka,kb] = meshgrid2d(1:oa,1:ob);

K = A(ia,ja,ka).*B(ib,jb,kb);