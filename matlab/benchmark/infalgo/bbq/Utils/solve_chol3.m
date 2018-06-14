function [ T ] = solve_chol3(R, S )
% S is a stack (that is, size(.,3)>1).
% T(:,:,i) = solve_chol(R, S(:,:,i)) all i.

szR = size(R);
szS = size(S);

if length(szS) >2
    S = reshape(S, szS(1), szS(2)*szS(3));
    T = solve_chol(R, S);
    T = reshape(T, szR(1), szS(2), szS(3));
else
    T = solve_chol(R, S);
end

end

