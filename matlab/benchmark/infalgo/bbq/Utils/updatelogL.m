function logLnew = updatelogL(logL,S,R,D,C,two)
% Update the log-likelihood given
% logL: the old log-likelihood
% S: the new cholesky factor
% R: the old cholesky factor
% D: the new datahalf
% C: the old datahalf

if nargin<6
    two=length(S);
end

one=1:(min(two)-1);
rthree=(length(one)+1):length(R);
sthree=(length(one)+length(two)+1):length(S);

logLnew = logL - 0.5 * length(two) * ...
             log(2 * pi) - sum(log(diag(S(two,two)))) - sum(log(diag(S(sthree,sthree)))) ...
             + sum(log(diag(R(rthree, rthree)))) - 0.5 * (D(two,:)' * D(two,:) ...
                                                        + D(sthree,:)' * D(sthree,:) ...
                                                        - C(rthree,:)' * C(rthree,:));