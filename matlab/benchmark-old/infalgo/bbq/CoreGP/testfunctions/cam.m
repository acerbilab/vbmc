function [f, grad] = cam(x, y)
if nargin == 1
  x1 = x(1);
  x2 = x(2);
else
  x1 = x;
  x2 = y;
end
f=(4-2.1.*x1.^2+x1.^4./3).*x1.^2+x1.*x2+(-4+4.*x2.^2).*x2.^2;       

grad = ...
[8 * x1 + (8.4) * x1.^3 + 2 * x1.^5 + x2;
 x1 - 8 * x2 + 16 * x2.^3];

