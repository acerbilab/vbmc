function vp = rescale_params(vp,theta)
%RESCALE_PARAMS Assign THETA and rescale SIGMA and LAMBDA variational parameters.

D = vp.D;

if nargin > 1 && ~isempty(theta)
    K = vp.K;
    if vp.optimize_mu
        vp.mu = reshape(theta(1:D*K),[D,K]);
        idx_start = D*K;
    else
        idx_start = 0;
    end
    vp.sigma = exp(theta(idx_start+(1:K)));
    if vp.optimize_lambda
        vp.lambda = exp(theta(idx_start+K+(1:D)))';
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

% The mode may have moved
if isfield(vp,'mode'); vp = rmfield(vp,'mode'); end

end