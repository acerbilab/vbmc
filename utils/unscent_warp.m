function [xw,sigmaw,xu] = unscent_warp(fun,x,sigma)
%UNSCENT_WARP Unscented transform (coordinate-wise only).

[N1,D] = size(x);
[N2,D2] = size(sigma);

N = max(N1,N2);

if N1 ~= N && N1 ~= 1; error('Mismatch between rows of X and SIGMA.'); end
if N2 ~= N && N2 ~= 1; error('Mismatch between rows of X and SIGMA.'); end
if D ~= D2; error('Mismatch between columns of X and SIGMA.'); end

if N1 == 1 && N > 1; x = repmat(x,[N,1]); end
if N2 == 1 && N > 1; sigma = repmat(sigma,[N,1]); end

U = 2*D+1;  % # unscented points

x3(1,:,:) = x;
xx = repmat(x3,[U,1,1]);

for d = 1:D
    sigma3(1,:,1) = sqrt(D)*sigma(:,d);
    xx(2*d,:,d) = bsxfun(@plus,xx(2*d,:,d),sigma3);
    xx(2*d+1,:,d) = bsxfun(@minus,xx(2*d+1,:,d),sigma3);
end

xu = reshape(fun(reshape(xx,[N*U,D])),[U,N,D]);

if N > 1
    xw(:,:) = mean(xu,1);
    sigmaw(:,:) = std(xu,[],1);
else
    xw(1,:) = mean(xu,1);
    sigmaw(1,:) = std(xu,[],1);    
end

end