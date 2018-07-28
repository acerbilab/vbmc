function [theta,vp] = get_theta(vp,optimize_lambda)
%GET_THETA Get vector of variational parameters from variational posterior.

vp = rescale_params(vp);
theta = [vp.mu(:); log(vp.sigma(:))];
if optimize_lambda; theta = [theta; log(vp.lambda(:))]; end

end