function y = proposal_vbmc(X,PLB,PUB,LB,UB)
%PROPOSAL_VBMC Default proposal function.

[N,D] = size(X);
y = zeros(N,1);

% df = 3;   % Three degrees of freedom
mu = 0.5*(PLB + PUB);
sigma = 0.5*(PUB-PLB);

for d = 1:D
    % y(:,d) = ( 1 + ((X(:,d)-mu(d))./sigma(d)).^2/df ).^(-(df+1)/2);
    y(:,d) = 1./( 1 + (((X(:,d)-mu(d))./sigma(d)).^2)/3 ).^2;
end

y = prod(y,2);

end