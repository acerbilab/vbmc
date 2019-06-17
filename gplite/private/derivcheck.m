function [err_rel,err_abs] = derivcheck(f,x,flag)
%DERIVCHECK Check analytical vs numerical differentiation for a function

if nargin < 3 || isempty(flag); flag = false; end

tic
if flag
    dy_num = fgrad(f,x,'five-points');
else
    dy_num = gradest(f,x);
end
toc
tic
[y,dy_ana] = f(x);
toc

if size(dy_num,1) == size(dy_num,2)
    dy_num = sum(dy_num,1);
end

% Reshape to row vectors
dy_num = dy_num(:)';
dy_ana = dy_ana(:)';

fprintf('Relative errors:\n');
err_rel = (dy_num(:)' - dy_ana(:)')./dy_num(:)'

fprintf('Absolute errors:\n');
err_abs = dy_num(:)' - dy_ana(:)'

end