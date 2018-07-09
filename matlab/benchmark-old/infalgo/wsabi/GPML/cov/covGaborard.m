function K = covGaborard(hyp, x, z, i)

% Gabor covariance function with length scales ell and periods p. The covariance
% function is parameterized as:
%
% k(x,z) = h(x-z), h(t) = prod(exp(-t.^2./(2*ell.^2))*cos(2*pi*t./p)).
%
% The hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          ..
%         log(ell_D)
%         log(p_1)
%         log(p_2)
%          ..
%         log(p_D) ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Note that covSM implements a weighted sum of Gabor covariance functions, but
% using an alternative (spectral) parameterization.
%
% Copyright (c) by Hannes Nickisch, 2013-10-22.
%
% See also COVFUNCTIONS.M, COVGABORISO, COVSM.M.

if nargin<2, K = '2*D'; return; end                        % report no of params
if nargin<3, z = []; end                                   % make sure, z exists

D = size(x,2);                                                       % dimension

fac = cell(1,D); for d=1:D, fac{d} = {'covMask',{d,{@covGaboriso}}}; end
cov = {@covProd,fac};                    % product of univariate Gabor functions
hyp = reshape(reshape(hyp,D,2)',2*D,1); % bring hyperparameters in correct shape

if nargin<4                                       % evaluation of the covariance
  K = feval(cov{:},hyp,x,z);
else
  if i<=2*D                                                        % derivatives
    [i1,i2] = ind2sub([D,2],i);
    K = feval(cov{:},hyp,x,z,sub2ind([2,D],i2,i1));
  else
    error('Unknown hyperparameter')
  end
end