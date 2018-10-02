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
    if vp.optimize_weights
        eta = theta(end-K+1:end);
        eta = eta - max(eta);
        vp.w = exp(eta(:)');
    end
end

nl = sqrt(sum(vp.lambda.^2)/D);
vp.lambda = vp.lambda(:)/nl;
vp.sigma = vp.sigma(:)'*nl;

% Ensure that weights are normalized
if vp.optimize_weights
    vp.w = vp.w(:)'/sum(vp.w);   
    % Remove ETA, used only for optimization
    if isfield(vp,'eta'); vp = rmfield(vp,'eta'); end
end


end