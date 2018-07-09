function R = perturbchol(R, J)
% S is the cholesky factor of (R'*R + diag(J))

N = length(J);

to_change = find(and(~isnan(J), J~=0));
n_change = length(to_change);

for i=1:n_change
    
    to_change_i = to_change(i);
    
    vec = zeros(N,1);
    vec(to_change_i) = sqrt(J(to_change_i));
    
    R = cholupdate(R, vec);
    
end