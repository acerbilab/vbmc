function s = logsumexp(X,dim)
%LOGSUMEXP Compute log(sum(exp(X))) while avoiding numerical underflow.
%   S = LOGSUMEXP(X) computes log(sum(exp(X)) along the first non-singleton
%   dimension.
%
%   S = LOGSUMEXP(X,DIM) sums along dimension DIM.
%
%   See also EXP, LOG, SUM.

% Based on code by Tom Minka and Mo Chen.

if nargin == 1, 
    % Determine which dimension sum will use
    dim = find(size(X)~=1,1);
    if isempty(dim), dim = 1; end
end

% subtract the largest in each dim
y = max(X,[],dim);
s = y+log(sum(exp(bsxfun(@minus,X,y)),dim));
idx = isinf(y);
if any(idx(:))
    s(idx) = y(idx);
end