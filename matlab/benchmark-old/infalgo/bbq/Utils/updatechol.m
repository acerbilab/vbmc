function [S,p]=updatechol(V,R,two)
% S=updatechol(V,R,two)
% S is the cholesky decomposition of V. R is the cholesky decomposition of
% a sub-matrix of V, defined by the removal of rows/cols two from V.


% I don't want to have to do checks on this, but we have to have
% two as a contiguous block. Also obviously have to have V positive
% definite and R upper triangular.

if nargin<3
    two=length(V);
end

no_errors = nargout>1;
    

one=1:(min(two)-1);
rthree=(length(one)+1):length(R);
vthree=(length(one)+length(two)+1):length(V);

lowr.UT=true;
lowr.TRANSA=true;

%S=...
%    [S11,S12,S13;...
%    zeros(length(two),length(one)),S22,S23;...
%    zeros(length(vthree),length(one)+length(two)),S33];

S = zeros(size(V));
%S11=R(one,one);
S(one,one) = R(one,one);
%S12=linsolve(R(one,one),V(one, two),lowr);
S(one,two) = linsolve(R(one,one),V(one, two),lowr);
%S13=R(one,rthree);
S(one,vthree) = R(one,rthree);
%S22=chol(V(two,two)-S12'*S12);
if no_errors
    [S2,p] = chol(V(two,two)-S(one,two)'*S(one,two));
    if p==0
        S(two,two) = S2;
    else
        S = [];
        return
    end
else
    S(two,two) = chol(V(two,two)-S(one,two)'*S(one,two));
end
%S23=linsolve(S22,V(two,vthree)-S12'*S13,lowr);
S(two,vthree) = linsolve(S(two,two),V(two,vthree)-S(one,two)'*S(one,vthree),lowr);
%S33=chol(R(rthree,rthree)'*R(rthree,rthree)-S23'*S23);
R2 = R(rthree,rthree);
% MIKE: S(2,3)'*S(2,3) = sum(i=two) S(i,3)'*S(i,3), so:

if no_errors
    for i = two
        [R2,p] = cholupdate(R2, S(i,vthree)', '-');
        if p~=0
            S = [];
            return
        end
    end 
else
    for i = two
      R2 = cholupdate(R2, S(i,vthree)', '-');
    end
end

S(vthree,vthree) = R2;
