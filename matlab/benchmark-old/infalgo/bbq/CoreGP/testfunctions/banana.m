% function output = banana(x)
% rosenbrock banana test function
%
% constraints:
% x is two-dimensional
% -5 <= x_i <= 10
% global optimum at x1 = x2 = 1, where banana = 0

function [output, grad] = banana(x)

output = (1-x(1)).^2 + 100*(x(2)-x(1).^2).^2;

grad = [-2*(1-x(1)) + 400*x(1)*(x(2)-x(1).^2);...
        -200*x(1)*(x(2)-x(1).^2)];

    
