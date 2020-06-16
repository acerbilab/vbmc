function [y,s] = rosenbrock_test(x,sigma)
%ROSENBROCKS_TEST Rosenbrock's broad 'banana' function.

if nargin < 2 || isempty(sigma); sigma = 0; end

% Likelihood according to a broad Rosenbrock's function
y = -sum((x(:,1:end-1) .^2 - x(:,2:end)) .^ 2 + (x(:,1:end-1)-1).^2/100,2);

% Noisy test
if sigma > 0
    n = size(x,1);
    y = y + sigma*randn([n,1]);
    if nargout > 1
        s = sigma*ones(n,1);
    end
end

% Might want to add a prior, such as
% sigma2 = 9;                               % Prior variance
% y = y - 0.5*sum(x.^2,2)/sigma2 - 0.5*D*log(2*pi*sigma2);