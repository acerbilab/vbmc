function f = sch(x)
f = 1 + 6*x(1)^2 - cos(12*x(1));
for i=2:50
  f = f + 590*(x(i) - x(i-1))^2;
end
