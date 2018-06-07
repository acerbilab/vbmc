function H = grad2hess(fun,x0,h)
%GRAD2HESS Numerically compute Hessian given function gradient.

if nargin < 3 || isempty(h); h = 1e-13; end

method = 3;

x0 = x0(:)';

nvars = numel(x0);
if isscalar(h); h = h*ones(1,nvars); end

H = zeros(nvars,nvars);

switch method
    case 1
        [~,g0] = fun(x0);
        g0 = g0(:)';
        for i = 1:nvars
            v = zeros(1,nvars);
            v(i) = 1;
            [~,g] = fun(x0 + h.*v);
            H(i,:) = (g(:)' - g0) ./ h;
        end
    case 2        
        for i = 1:nvars
            v = zeros(1,nvars);
            v(i) = 1;
            [~,g1] = fun(x0 + h.*v);
            [~,g2] = fun(x0 - h.*v);
            H(i,:) = (g1(:)' - g2(:)') ./ (2*h);
        end
    case 3  % Five-point stencil
        for i = 1:nvars
            v = zeros(1,nvars);
            v(i) = 1;
            [~,g1] = fun(x0 + 2*h.*v);
            [~,g2] = fun(x0 + h.*v);
            [~,g3] = fun(x0 - h.*v);
            [~,g4] = fun(x0 - 2*h.*v);
            H(i,:) = (-g1(:)' + 8*g2(:)' - 8*g3(:)' + g4(:)') ./ (12*h);
        end
end

H = 0.5*(H' + H);   % Enforce symmetry