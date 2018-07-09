function m=linspacey(a,b,n)
if n~=1
    m=linspace(a,b,n);
elseif n==1
    m=0.5*(a+b);
end