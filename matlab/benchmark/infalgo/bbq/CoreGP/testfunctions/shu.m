function f = shu(x)
sum1 = 0; sum2 = 0;
if nargin == 1
 x1 = x(1);
 x2 = x(2);
else
 x1 = x;
 x2 = y;
end
for i = 1:5
 sum1 = sum1 + i.*cos((i+1).*x1+i);
 sum2 = sum2 + i.*cos((i+1).*x2+i);
end
f = sum1.*sum2;
