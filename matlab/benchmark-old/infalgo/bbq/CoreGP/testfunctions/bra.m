function f = bra(x)
a=1;
b=5.1/(4*pi*pi);
c=5/pi;
d=6;
h=10;
ff=1/(8*pi);
if nargin == 1
  x1 = x(1);
  x2 = x(2);
else
  x1 = x;
  x2 = y;
end
f=a.*(x2-b.*x1.^2+c.*x1-d).^2+h.*(1-ff).*cos(x1)+h;
