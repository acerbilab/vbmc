function [sqd_dist, dist] = sqd_dist_stack(X1, X2)

[N1, D] = size(X1);
[N2, D] = size(X2);

dist = bsxfun(@minus,...
                reshape(X1,N1,1,D),...
                reshape(X2,1,N2,D));
sqd_dist = dist.^2;