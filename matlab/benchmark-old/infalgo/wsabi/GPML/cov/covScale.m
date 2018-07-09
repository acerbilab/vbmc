function K = covScale(cov, hyp, x, z, i)

% covScale - compose a covariance function as a scaled version of another one.
%
% k(x^p,x^q) = sf^2 * k_0(x^p,x^q)
%
% The hyperparameter is:
%
% hyp = [ log(sf)  ]
%
% This function doesn't actually compute very much on its own, it merely does
% some bookkeeping, and calls other covariance functions to do the actual work.
%
% Copyright (c) by Carl Edward Rasmussen & Hannes Nickisch 2010-09-10.
%
% See also COVFUNCTIONS.M.

if nargin<3, K = ['(1+',feval(cov{:}),')']; return, end % report number of param
if nargin<4, z = []; end                                   % make sure, z exists

sf2 = exp(2*hyp(1));                                           % signal variance

if nargin<5                                                        % covariances
  K = sf2*feval(cov{:},hyp(2:end),x,z);
else                                                               % derivatives
  if i==1
    K = 2*sf2*feval(cov{:},hyp(2:end),x,z);
  else
    K = sf2*feval(cov{:},hyp(2:end),x,z,i-1);
  end
end