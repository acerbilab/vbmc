function V=tri2(v) 
% tri(v) returns the matrix R'*R where R is the square upper triangular
% matrix of dimension constructed from the vector v. v contains firstly the
% log of the scale for each dimension of V, followed by the remainder of
% the degrees of freedom (angles which are passed through a cos before
% being put into matrix)

GD=(-1+sqrt(1+8*length(v)))/2;
if floor(GD)~=GD
    error('v is not of appropriate length');
end

R=eye(GD);
l=1+GD;
for gd=2:GD
    vec=ones(gd,1);
    for p=1:gd-1
        vec=vec.*listerine(v(l),p,gd);
    end
    R(1:gd,gd)=vec;
end
R=R*diag(exp(v(1:GD)));
V=R'*R;

% R=zeros(GD);
% l=1+GD;
% for gd=2:GD
%     R=R+diag(v(l:l+GD-gd),gd-1);
%     l=l+GD-gd+1;
% end
% R=R*diag(exp(v(1:GD)));
% V=R'*R;

function ls=listerine(a,p,q)
%ns=[1 2 2 3 3 3 4 4 4 4 5 5 5 5 5 6 6 6 6 6 6];
%offs=[0 0 1 0 1 2 0 1 2 3 0 1 2 3 4 0 1 2 3 4 5];

n=floor(1/2+sqrt(2*p));
off=p-1-n*(n-1)/2;
ls=ones(2^n,1);
ls(1+off)=cos(a);
ls(2^(n-1)+1+off)=sin(a);
ls=repmat(ls,ceil(q*2^(-n)),1);
ls=ls(1:q);