function [S,p]=revisechol(V,R,two)
% S=updatechol(V,R,two)
% S is the cholesky decomposition of V. R is the cholesky decomposition of
% W, where W = V except W(two,two) ~= V(two,two);

% I don't want to have to do checks on this, but we have to have
% two as a contiguous block. Also obviously have to have V positive
% definite and R upper triangular.

if nargin<3
    two=length(V);
end

no_errors = nargout>1;
    

one=1:(min(two)-1);
three=max(two)+1:length(V);

lowr.UT=true;
lowr.TRANSA=true;

%S=...
%    [S11,S12,S13;...
%    zeros(length(two),length(one)),S22,S23;...
%    zeros(length(three),length(one)+length(two)),S33];

S = zeros(size(V));
%S11=R(one,one);S12=R(one,two);S13=R(one,three);
S(one,:) = R(one,:);
if no_errors
    [S2,p] = chol(V(two,two)-R(one,two)'*R(one,two));
    if p==0
        S(two,two) = S2;
    else
        S = [];
        return
    end
else
    S(two,two) = chol(V(two,two)-R(one,two)'*R(one,two));
end
%S23=inv(S22')*(R22'*R23)
S(two,three) = linsolve(S(two,two),R(two,two)'*R(two,three),lowr);

S3 = R(three,three);
if no_errors
    for i = two
        [S3] = cholupdate(S3,R(i,three)', '+');
        [S3,p] = cholupdate(S3,S(i,three)', '-');
        if p~=0
            return
        end
    end 
else
    for i = two
        S3 = cholupdate(S3,R(i,three)', '+');
        S3 = cholupdate(S3,S(i,three)', '-');
    end
end

S(three,three) = real(S3);
