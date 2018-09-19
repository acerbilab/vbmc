function y = rosenbrock_test(x)
%ROSENBROCKS_TEST Rosenbrock's 'banana' function with Gaussian prior.

D = size(x,2);

% Likelihood according to a broad Rosenbrock's function
y = -sum((x(:,1:end-1) .^2 - x(:,2:end)) .^ 2 + (x(:,1:end-1)-1).^2/100);

% Wide Gaussian prior centered in zero
sigma2 = 9;     % Prior variance
y = y - 0.5*sum(x.^2,2)/sigma2 - 0.5*D*log(2*pi*sigma2);