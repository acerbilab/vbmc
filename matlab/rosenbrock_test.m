function y = rosenbrock_test(x)
%ROSENBROCKS_TEST Rosenbrock's 'banana' function in any dimension (easy).

y = -sum((x(:,1:end-1) .^2 - x(:,2:end)) .^ 2 + (x(:,1:end-1)-1).^2/100);