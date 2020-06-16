function [y,s] = lpostfun(x,llike,lprior)
%LPOSTFUN Log (unnormalized) posterior function.
%   Y = LPOSTFUN(X,LLIKE,LPRIOR) returns the unnormalized log posterior
%   evaluated at X where LLIKE is a function handle to the log likelihood
%   function and LPRIOR a function handle to the log prior.
%
%   [Y,S] = LPOSTFUN(X,LLIKE,LPRIOR) also returns an estimate S of the 
%   standard deviation of a noisy log-likelihood evaluation at X (obtained 
%   as second output of LLIKE, assuming LLIKE has two outputs). Note that
%   the log prior is assumed to be noiseless.

if nargin < 3; lprior = []; end

if nargout > 1
    [y,s] = llike(x);
else
    y = llike(x);
end
    
if ~isempty(lprior)
    y = y + lprior(x);
end

end