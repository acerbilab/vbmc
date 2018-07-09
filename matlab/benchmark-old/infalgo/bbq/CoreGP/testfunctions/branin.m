function y = branin(x)
% function y = branin(x)
% 
% branin function 
% Matlab Code by A. Hedar (Sep. 29, 2005).
% The number of variables n = 2.
% 
% constraints:
% -5 <= x1 <= 10, 0 <= x2 <= 15
% three global optima:  (-pi, 12.275), (pi, 2.275), (9.42478, 2.475), where
% branin = 0.397887

y = (x(2)-(5.1/(4*pi^2))*x(1)^2+5*x(1)/pi-6)^2+10*(1-1/(8*pi))*cos(x(1))+10;
