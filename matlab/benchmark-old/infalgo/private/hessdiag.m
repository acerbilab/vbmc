function [HD,err,finaldelta] = hessdiag(fun,x0)
% HESSDIAG: diagonal elements of the Hessian matrix (vector of second partials)
% usage: [HD,err,finaldelta] = hessdiag(fun,x0)
%
% When all that you want are the diagonal elements of the hessian
% matrix, it will be more efficient to call HESSDIAG than HESSIAN.
% HESSDIAG uses DERIVEST to provide both second derivative estimates
% and error estimates. fun needs not be vectorized.
% 
% arguments: (input)
%  fun - SCALAR analytical function to differentiate.
%        fun must be a function of the vector or array x0.
% 
%  x0  - vector location at which to differentiate fun
%        If x0 is an nxm array, then fun is assumed to be
%        a function of n*m variables. 
%
% arguments: (output)
%  HD  - vector of second partial derivatives of fun.
%        These are the diagonal elements of the Hessian
%        matrix, evaluated at x0.
%        HD will be a row vector of length numel(x0).
%
%  err - vector of error estimates corresponding to
%        each second partial derivative in HD.
%
%  finaldelta - vector of final step sizes chosen for
%        each second partial derivative.
%
%
% Example usage:
%  [HD,err] = hessdiag(@(x) x(1) + x(2)^2 + x(3)^3,[1 2 3])
%  HD =
%     0     2    18
%
%  err =
%     0     0     0
%
%
% See also: derivest, gradient, gradest
%
%
% Author: John D'Errico
% e-mail: woodchips@rochester.rr.com
% Release: 1.0
% Release date: 2/9/2007

% get the size of x0 so we can reshape
% later.
sx = size(x0);

% total number of derivatives we will need to take
nx = numel(x0);

HD = zeros(1,nx);
err = HD;
finaldelta = HD;
for ind = 1:nx
  [HD(ind),err(ind),finaldelta(ind)] = derivest( ...
    @(xi) fun(swapelement(x0,ind,xi)), ...
    x0(ind),'deriv',2,'vectorized','no');
end

end % mainline function end

% =======================================
%      sub-functions
% =======================================
function vec = swapelement(vec,ind,val)
% swaps val as element ind, into the vector vec
vec(ind) = val;

end % sub-function end



