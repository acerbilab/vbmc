function vp = rescale_params(vp,theta)
%RESCALE_PARAMS Assign THETA and rescale SIGMA and LAMBDA variational parameters.

D = vp.D;

if nargin > 1 && ~isempty(theta)
    K = vp.K;
    vp.mu = reshape(theta(1:D*K),[D,K]);
    vp.sigma = exp(theta(D*K+(1:K)));
    if vp.optimize_lambda
        vp.lambda = exp(theta(D*K+K+(1:D)))';
    end
end

nl = sqrt(sum(vp.lambda.^2)/D);
vp.lambda = vp.lambda(:)/nl;
vp.sigma = vp.sigma(:)'*nl;

end