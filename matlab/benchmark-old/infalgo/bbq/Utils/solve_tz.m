function [x] = solve_tz(uT,b)
% Assuming uT is Toeplitz, solve uT x = b for x, as per Golub & Van Loan Sec
% 4.7.3.

% normalise the diagonal of uT
u =  uT(1);
T = uT / u;

r = @(i) T(1,i+1);

n = length(T);

y = nan(n,1);
x = nan(n,1);

y(1) = -r(1);
x(1) = b(1);
beta = 1;
alpha = -r(1);

for k = 1:n-1
    beta = (1 - alpha^2)*beta;
    mu = (b(k+1) - r(1:k)*x(k:-1:1)) / beta;
    x(1:k+1) = [x(1:k) + mu*y(k:-1:1); mu];
    if k < n-1
        alpha = (-r(k+1) + r(1:k)*y(k:-1:1)) / beta;
        y(1:k+1) = [y(1:k) + alpha*y(k:-1:1);alpha];
    end
end

x = x / u;