function [y,dy] = softbndloss(x,slb,sub,TolCon)
%SOFTBNDLOSS Loss function for soft bounds for function minimization.

% Penalization relative scale
if nargin < 4 || isempty(TolCon); TolCon = 1e-3; end

compute_grad = nargout > 1;     % Compute gradient only if requested

ell = (sub - slb).*TolCon;

y = 0;
dy = zeros(size(x));

idx = x < slb;
if any(idx)
    y = y + 0.5*sum(((slb(idx) - x(idx))./ell(idx)).^2);
    if compute_grad
        dy(idx) = (x(idx) - slb(idx))./ell(idx).^2;
    end
end

idx = x > sub;
if any(idx)
    y = y + 0.5*sum(((x(idx) - sub(idx))./ell(idx)).^2);
    if compute_grad
        dy(idx) = (x(idx) - sub(idx))./ell(idx).^2;
    end
end

end