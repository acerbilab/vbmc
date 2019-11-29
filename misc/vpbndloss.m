function [L,dL] = vpbndloss(theta,vp,thetabnd,TolCon)
%VPLOSS Variational parameter loss function for soft optimization bounds.

compute_grad = nargout > 1;     % Compute gradient only if requested

K = vp.K;
D = vp.D;

if vp.optimize_mu
    mu = theta(1:K*D);
    idx_start = K*D;
else
    mu = vp.mu(:)';
    idx_start = 0;
end
if vp.optimize_sigma
    lnsigma = theta(idx_start+(1:K));
    idx_start = idx_start + K;
else
    lnsigma = log(vp.sigma(:));
end
if vp.optimize_lambda
    lnlambda = theta(idx_start+(1:D));
else
    lnlambda = log(vp.lambda(:));
end
if vp.optimize_weights
    eta = theta(end-K+1:end);   
else
    eta = [];
end

lnscale = bsxfun(@plus,lnsigma(:)',lnlambda(:));
theta_ext = [];
if vp.optimize_mu; theta_ext = [theta_ext; mu(:)]; end
if vp.optimize_sigma || vp.optimize_lambda; theta_ext = [theta_ext; lnscale(:)]; end
if vp.optimize_weights; theta_ext = [theta_ext; eta(:)]; end

if compute_grad
    [L,dL] = softbndloss(theta_ext,thetabnd.lb(:),thetabnd.ub(:),TolCon);
    if vp.optimize_mu
        dmu = dL(1:D*K);
        idx_start = D*K;
    else
        dmu = [];
        idx_start = 0;
    end
    if vp.optimize_sigma || vp.optimize_lambda
        dlnscale = reshape(dL((1:D*K)+idx_start),[D,K]);
        if vp.optimize_sigma
            dsigma = sum(dlnscale,1);
        else
            dsigma = [];
        end
        if vp.optimize_lambda
            dlambda = sum(dlnscale,2);
        else
            dlambda = [];
        end
    else
        dsigma = []; dlambda = [];
    end
    if vp.optimize_weights
        deta = dL(end-K+1:end);   
    else
        deta = [];
    end
    dL = [dmu(:); dsigma(:); dlambda(:); deta(:)];
else
    L = softbndloss(theta_ext,thetabnd.lb(:),thetabnd.ub(:),TolCon);
end

end