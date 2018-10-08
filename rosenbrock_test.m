function y = rosenbrock_test(x)
%ROSENBROCKS_TEST Rosenbrock's broad 'banana' function.

% Likelihood according to a broad Rosenbrock's function
y = -sum((x(:,1:end-1) .^2 - x(:,2:end)) .^ 2 + (x(:,1:end-1)-1).^2/100);

% Might want to add a prior, such as
% sigma2 = 9;                               % Prior variance
% y = y - 0.5*sum(x.^2,2)/sigma2 - 0.5*D*log(2*pi*sigma2);