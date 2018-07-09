function S = dprod3(A,B)
% either A or B is a stack (that is, size(.,3)>1). Assume A is, wlog.
% S(:,:,i) = diag(A(:,:,i))*B all i.

szA = size(A);
szB = size(B);

if length(szA) == 3 && length(szB) == 2 && szA(2)~=1
    A = permute(A,[2 1 3]);
elseif length(szA) == 2 && length(szB) == 3 && szB(1)~=1
    B = permute(B,[2 1 3]);
end

S = bsxfun(@times,A,B);