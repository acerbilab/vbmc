function S = prod3(A,B)
% either A or B is a stack (that is, size(.,3)>1). Assume A is, wlog.
% S(:,:,i) = A(:,:,i)*B all i.

szA = size(A);
szB = size(B);

if length(szA) == 3 && length(szB) == 2
    
    A = permute(A,[2 1 3]);
    A = reshape(A, szA(2), szA(1)*szA(3));
    S = B'*A;
    S = reshape(S, szB(2), szA(1), szA(3));
    S = permute(S,[2 1 3]);
    
elseif length(szA) == 2 && length(szB) == 3

    B = reshape(B, szB(1), szB(2)*szB(3));
    S = A*B;
    S = reshape(S, szA(1), szB(2), szB(3));
    
elseif length(szA) == 2 && length(szB) == 2
    
    S = A*B;
    
elseif length(szA) == 3 && length(szB) == 3
    
    S = nan(szA(1), szB(2), szA(3));
    for i = 1:szA(3)
        S(:,:,i) = A(:,:,i) * B(:,:,i);
    end

end
