function S = linsolve3(A, B, flag)
% B is a stack (that is, size(.,3)>1).
% S(:,:,i) = linsolve(A, B(:,:,i), flag) all i.

szA = size(A);
szB = size(B);

if length(szB) >2
    B = reshape(B, szB(1), szB(2)*szB(3));
    if nargin>2
        S = linsolve(A, B, flag);
    else
        S = linsolve(A, B);
    end
    S = reshape(S, szA(1), szB(2), szB(3));
else
    if nargin>2
        S = linsolve(A, B, flag);
    else
        S = linsolve(A, B);
    end
end
