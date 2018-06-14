function fout=Tom(ins)

ins=10.^(ins);
A=ins(1);
Sx=ins(2);
Sy=ins(3);

%fout= @(A,Sx,Sy) -(sqrt(2)*A.^-1.*sqrt((-A.^4.*Sx.^2+A.^4.*Sy.^2 ...
fout= (sqrt(2)*sqrt((-A.^2.*Sx.^2+A.^2.*Sy.^2 ...
        +sqrt(A.^4.*Sx.^4+2*A.^4.*Sx.^2.*Sy.^2+A.^4.*Sy.^4+64*Sx.^4.*Sy.^4))...
        ./(A.^4+16*Sx.^2.*Sy.^2)));
    

%\frac{A_{focus}}{A}=\sqrt{2}\sqrt{\frac{-A^2 Sx^2+A^2 Sy^2+\sqrt{A^4Sx^4+2A^4Sx^2Sy^2+A^4Sy^4+64Sx^4Sy^4}}{A^4+16Sx^2Sy^2}}


%fout=sqrt(2)*A.^-1*sqrt(+(A.^4*Sx.^2)./(A.^4 + 16*Sx.^2.*Sy.^2)+(A.^4.*Sy.^2)./(A.^4 + 16.*Sx.^2.*Sy.^2)+(sqrt(A.^8*Sx.^4+2*A.^8.*Sx.^2.*Sy.^2+A.^8.*Sy.^4+64.*A.^4.*Sx.^4.*Sy.^4))./(A.^4+16*Sx.^2.*Sy.^2));

% Sx=1;[A,Sys]=meshgrid(0.1:0.01:1,0.1:0.01:1);
% plot3(A,Sys,f(A,Sx,Sys))
% A=1;[Sxs,Sys]=meshgrid(150:0.5:200,150:0.5:200);
% contour(Sxs,Sys,f(1,Sxs,Sys))