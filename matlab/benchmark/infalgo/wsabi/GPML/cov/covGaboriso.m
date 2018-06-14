function K = covGaboriso(hyp, x, z, i)

% Gabor covariance function with length scale ell and period p. The 
% covariance function is parameterized as:
%
% k(x,z) = h( ||x-z|| ) with h(t) = exp(-t^2/(2*ell^2))*cos(2*pi*t/p).
%
% The hyperparameters are:
%
% hyp = [ log(ell)
%         log(p)   ]
%
% Note that covSM implements a weighted sum of Gabor covariance functions, but
% using an alternative (spectral) parameterization.
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Copyright (c) by Hannes Nickisch, 2013-10-22.
%
% See also COVFUNCTIONS.M, COVGABORARD.M, COVSM.M.

if nargin<2, K = '2'; return; end                          % report no of params
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

ell = exp(hyp(1));                                                % length scale
p = exp(hyp(2));                                                        % period

if dg                                              % compute squared distance d2
  d2 = zeros([size(x,1),1]);
else
  if xeqz                                                 % symmetric matrix Kxx
    d2 = sq_dist(x'/ell);
  else                                                   % cross covariances Kxz
    d2 = sq_dist(x'/ell,z'/ell);
  end
end
dp = 2*pi*sqrt(d2)*ell/p;
K = exp(-d2/2).*cos(dp);                                           % covariances
if nargin==4                                                       % derivatives
  if i==1                                                         % length scale
    K = d2 .* K;
  elseif i==2                                                           % period
    K = tan(dp).*dp .* K;
  else
    error('Unknown hyperparameter')
  end
end
