function y = mich(x)
% function y = mich(x)
% 
% Michalewicz function 
% Matlab Code by A. Hedar (Nov. 23, 2005).
%
% constraints:
% x can be any dimensional, common values are 2, 5, 10
% 0 <= xi <= pi for i = 1..n
% global optimal values:
% n=2 -> 1.8013 
% n=5 -> 4.687658
% n=10 -> 9.66015

m = 10;
s = 0;
for i = 1:size(x,2);
    s = s+sin(x(i))*(sin(i*x(i)^2/pi))^(2*m);
end
y = -s;
