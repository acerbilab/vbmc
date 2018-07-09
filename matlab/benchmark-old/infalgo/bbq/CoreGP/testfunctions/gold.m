function y = gold(x)
% function y = gold(x)
%  
% Goldstein and Price function 
% Matlab Code by A. Hedar (Sep. 29, 2005).
% The number of variables n = 2.
%
% constraints:
% -2 <= xi <= 2, i = 1,2
% global optimimum at (0, -1), where gold = -3
% 
a = 1+(x(1)+x(2)+1)^2*(19-14*x(1)+3*x(1)^2-14*x(2)+6*x(1)*x(2)+3*x(2)^2);
b = 30+(2*x(1)-3*x(2))^2*(18-32*x(1)+12*x(1)^2+48*x(2)-36*x(1)*x(2)+27*x(2)^2);
y = a*b;
