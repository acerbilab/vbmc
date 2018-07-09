function B=allcombs(A)

if isempty(A)
    B=[];
elseif ~iscell(A)
    B=A(:,1);
    [K,J]=size(A);
    for j=2:J
        L=size(B,1);
        B=[kron2d(B,ones(K,1)),kron2d(ones(L,1),A(:,j))];
    end
    
elseif iscell(A)
    I=length(A);
    
    B=allcombs(A{1});
    for i=2:I
        [K,J]=size(A{i});
        for j=1:J
            L=size(B,1);
            B=[kron2d(B,ones(K,1)),kron2d(ones(L,1),A{i}(:,j))];
        end
    end        
end