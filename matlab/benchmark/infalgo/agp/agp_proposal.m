function y = agp_proposal(X,PLB,PUB,LB,UB)
%AGP_PROPOSAL Default proposal function for AGP.

[N,D] = size(X);
y = zeros(N,1);

mu = 0.5*(PLB + PUB);
sigma = 0.5*(PUB-PLB);

for d = 1:D
    y = y - 0.5*((X(:,d)-mu(d))./sigma(d)).^2;
end

y = exp(y);

end