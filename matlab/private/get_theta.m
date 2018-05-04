function [theta,vp] = get_theta(vp,LB_theta,UB_theta,optimize_lambda)
%GET_THETA Get vector of variational parameters from variational posterior.

vp = rescale_params(vp);
theta = [vp.mu(:); log(vp.sigma(:))];
if optimize_lambda; theta = [theta; log(vp.lambda(:))]; end

if ~isempty(LB_theta) && ~isempty(UB_theta)
    theta = min(UB_theta(:),max(LB_theta(:), theta));
end

end