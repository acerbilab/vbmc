function [dd,err,finaldelta] = directionaldiff(fun,x0,vec)
% directionaldiff: estimate of the directional derivative of a function of n variables
% usage: [grad,err,finaldelta] = directionaldiff(fun,x0,vec)
%
% Uses derivest to provide both a directional derivative
% estimates plus an error estimates. fun needs not be vectorized.
% 
% arguments: (input)
%  fun - analytical function to differentiate. fun must
%        be a function of the vector or array x0. Fun needs
%        not be vectorized.
% 
%  x0  - vector location at which to differentiate fun
%        If x0 is an nxm array, then fun is assumed to be
%        a function of n*m variables. 
%
%  vec - vector defining the line along which to take the
%        derivative. Vec should be the same size as x0. It
%        need not be a vector of unit length.
%
% arguments: (output)
%  dd  - scalar estimate of the first derivative of fun
%        in the SPECIFIED direction.
%
%  err - error estimate of the directional derivative
%
%  finaldelta - vector of final step sizes chosen for
%        each partial derivative.
%
%
% Example:
%  At the global minimizer (1,1) of the Rosenbrock function,
%  compute the directional derivative in the direction [1 2]
%  It should be 0.
%
%  rosen = @(x) (1-x(1)).^2 + 105*(x(2)-x(1).^2).^2;
%  [dd,err] = directionaldiff(rosen,[1 1])
%
%  dd =
%       0
%  err =
%       0
%
%
% See also: derivest, gradest, gradient
%
%
% Author: John D'Errico
% e-mail: woodchips@rochester.rr.com
% Release: 1.0
% Release date: 3/5/2007

% get the size of x0 so we can make sure vec is
% the same shape.
sx = size(x0);
if numel(x0)~=numel(vec)
  error 'vec and x0 must be the same sizes'
end
vec = vec(:);
vec = vec/norm(vec);
vec = reshape(vec,sx);

[dd,err,finaldelta] = derivest(@(t) fun(x0+t*vec), ...
    0,'deriv',1,'vectorized','no');

end % mainline function end


