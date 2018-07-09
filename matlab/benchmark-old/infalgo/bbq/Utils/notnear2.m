function [c,ceq,GC,GCeq] = notnear2(x,XData1,L1,closeness_num1,XData2,L2,closeness_num2)
% These are the constraints that x is not 'too near' to any of the XData,
% one constraint per element of XData, where a measure of distance is given
% by BOTH the vectors of length scales L1 and L2

m1=size(XData1,1);
m2=size(XData2,1);
n=size(XData1,2);

RawDists1=repmat(x,m1,1)-XData1;
SqdRawDists1=RawDists1.^2;

RawDists2=repmat(x,m2,1)-XData2;
SqdRawDists2=RawDists2.^2;

invsqdL1 = L1'.^-2;
invsqdL2 = L2'.^-2;

% Nonlinear inequalities at x
d1 = (SqdRawDists1*invsqdL1 - closeness_num1)';
d2 = (SqdRawDists2*invsqdL2 - closeness_num2)';

c = -[d1,d2];
%c = -d1.*(d2>=0) - d2.*(d1>=0) + (d1.^2+d2.^2).*and(d1<0,d2<0);
    
% Nonlinear equalities at x
ceq = 0; 

if nargout > 2   % nonlcon called with 4 outputs
    % Gradients of the inequalities
    
    Gd1 = 2*RawDists1'.*repmat(invsqdL1,1,m);
    Gd2 = 2*RawDists2'.*repmat(invsqdL2,1,m);
    
    GC = -[Gd1,Gd2];   
    
%         rd1 = repmat(d1,n,1);
%     rd2 = repmat(d2,n,1);
    
    %GC = - Gd1.*(rd2>=0) - Gd2.*(rd1>=0) ...
    %     + (2*rd1.*Gd1+2*rd2.*Gd2).*and(rd1<0,rd2<0);  
    % Gradients of the equalities
    GCeq = zeros(n,1);
end