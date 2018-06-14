%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% sgr.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function f = sgr(x)
% simplified Griewank function of dimension dimsgr
function f = sgr(x)
dimsgr = length(x);
f = -1;
for i=1:dimsgr
  f = f*cos(x(i));
end
for i=1:dimsgr
  f = f + x(i)^2/dimsgr;
end
f = f + 1;
