function [c,ceq,GC,GCeq] = notnear(x,XData,L,closeness_num)
% This is the constraint that x is not 'too near' to any of the XData,
% where a measure of distance is given by the vector of length scales L

m=size(XData,1);
n=size(XData,2);

% Nonlinear inequalities at x
c = (closeness_num-(repmat(x,m,1)-XData).^2*L'.^-2)';
    
% Nonlinear equalities at x
ceq = 0; 

if nargout > 2   % nonlcon called with 4 outputs
    % Gradients of the inequalities
    GC = -2*(repmat(x,m,1)-XData)'.*repmat(L'.^2,1,m);    
    % Gradients of the equalities
    GCeq = zeros(n,1);
end