% function out = ackley(x)
% ackley test function
% 
% constraints:
% x can be any dimension
% -32.768 <= x_i <= 32.768
% global optimum at x1 = x2 = ... = xn = 0, where ackley = 0
%

function out = ackley(x)

out = (20+exp(1)-20*exp(-0.2*sqrt(sum(x.^2)/numel(x)))-exp(sum(cos(2*pi*x)/numel(x))));

