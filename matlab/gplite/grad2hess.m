function H = grad2hess(fun,x0)
%GRAD2HESS Numerically compute Hessian given function gradient.

method = 2;

x0 = x0(:)';

nvars = numel(x0);
h = 1e-15*ones(1,nvars);

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
            [~,g1] = fun(x0 + 0.5*h.*v);
            [~,g2] = fun(x0 - 0.5*h.*v);
            H(i,:) = (g1(:)' - g2(:)') ./ h;
        end
        
        
        
end

H = 0.5*(H' + H);   % Enforce symmetry