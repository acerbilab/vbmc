function D = revisedatahalf(S,L,C,two)
% D = updatedatahalf(S,L,C,R,two)
% D is inv(S')*L, and C is inv(R')*L(setdiff(1:end,two))

if nargin<4
    two=length(S);
end

one=1:(min(two)-1);
three=max(two)+1:length(L);

lowr.UT=true;
lowr.TRANSA=true;

D=nan(size(L));

D(one,:) = C(one,:);
D(two,:) = linsolve(S(two,two), L(two,:) - S(one,two)' * C(one,:), lowr);
D(three,:) = linsolve(S(three,three), ...
    L(three,:) - S(one, three)' * C(one,:) - S(two,three)' * D(two,:), lowr);