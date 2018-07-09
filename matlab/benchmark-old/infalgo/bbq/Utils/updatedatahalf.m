function D = updatedatahalf(S,L,C,R,two)
% D = updatedatahalf(S,L,C,R,two)
% D is inv(S')*L, and C is inv(R')*L(setdiff(1:end,two)). Note that
% L(two,:) is the only part of L ever called by this function.

if nargin<5
    two=length(S);
end

one=1:(min(two)-1);
rthree=(length(one)+1):length(R);
sthree=(length(one)+length(two)+1):length(S);

lowr.UT=true;
lowr.TRANSA=true;

D=nan(size(L));

D(one,:) = C(one,:);
D(two,:) = linsolve(S(two,two), L(two,:) - S(one,two)' * C(one,:), lowr);
D(sthree,:) = linsolve(S(sthree,sthree), R(rthree, rthree)' * C(rthree,:) - S(two,sthree)' * D(two,:), lowr);