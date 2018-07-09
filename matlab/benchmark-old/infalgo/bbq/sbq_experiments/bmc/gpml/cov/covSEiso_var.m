function K = covSEiso_var(hyp, x, z, i)

% Squared Exponential covariance function with isotropic, fixed distance measure. The 
% covariance function is parameterized as:
%
% k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*(x^p - x^q)/2) 
%
% where the sf^2 is the signal variance. The hyperparameters are:
%
% hyp = [ log(sf)  ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
% Made by David Duvenaud, May 2011
%
% See also COVFUNCTIONS.M.

if nargin<2, K = '1'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

sf2 = exp(2*hyp(1));                                           % signal variance

% precompute squared distances
if dg                                                               % vector kxx
  K = zeros(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = sq_dist(x');
  else                                                   % cross covariances Kxz
    K = sq_dist(x',z');
  end
end

if nargin<4                                                        % covariances
  K = sf2*exp(-K/2);
else                                                               % derivatives
  if i==1
    K = 2*sf2*exp(-K/2);
  else
    error('Unknown hyperparameter')
  end
end