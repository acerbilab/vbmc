function D = pdist2_squared_fast( A, B )

D = bsxfun(@plus,sum(A.^2,2),sum(B.^2,2)') - 2*(A*B');