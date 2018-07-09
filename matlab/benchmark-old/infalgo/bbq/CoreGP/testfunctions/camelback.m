% function out = camelback(x)
% camelback function
%
% constraints:
% x is two-dimensional
% -3 <= x1 <= 3, -2 <= x2 <= 2
% two global optima at (-0.0898, 0.7126), (0.0898, -0.7126) where
% camelback = 1.0316

function [out] = camelback(x)

y = x(2);
x = x(1);

out = (4-2.1*x.^2+x.^4/3).*x.^2+x.*y+4*(y.^2-1).*y.^2;

