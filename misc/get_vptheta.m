function [theta,vp] = get_vptheta(vp,optimize_mu,optimize_sigma,optimize_lambda,optimize_weights)
%GET_VPTHETA Get vector of variational parameters from variational posterior.

if nargin < 5 || isempty(optimize_weights)
    optimize_weights = vp.optimize_weights;
    if nargin < 4 || isempty(optimize_lambda)
        optimize_lambda = vp.optimize_lambda;
        if nargin < 3 || isempty(optimize_sigma)
            optimize_sigma = vp.optimize_sigma;
            if nargin < 2 || isempty(optimize_mu)
                optimize_mu = vp.optimize_mu;
            end
        end
    end
end

vp = rescale_params(vp);
if optimize_mu; theta = vp.mu(:); else; theta = []; end
if optimize_sigma; theta = [theta; log(vp.sigma(:))]; end
if optimize_lambda; theta = [theta; log(vp.lambda(:))]; end
if optimize_weights; theta = [theta; log(vp.w(:))]; end

end