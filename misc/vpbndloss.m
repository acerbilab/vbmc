function [L,dL] = vpbndloss(theta,vp,thetabnd,TolCon)
%VPLOSS Variational parameter loss function for soft optimization bounds.

compute_grad = nargout > 1;     % Compute gradient only if requested

K = vp.K;
D = vp.D;

mu = theta(1:K*D);
lnsigma = theta(D*K+(1:K));
if vp.optimize_lambda
    lnlambda = theta(K+D*K+(1:D));
else
    lnlambda = log(vp.lambda(:));
end
if vp.optimize_weights
    eta = theta(end-K+1:end);   
else
    eta = [];
end

lnscale = bsxfun(@plus,lnsigma(:)',lnlambda(:));    
theta_ext = [mu(:); lnscale(:); eta(:)];

if compute_grad
    [L,dL] = softbndloss(theta_ext,thetabnd.lb(:),thetabnd.ub(:),TolCon);
    dmu = dL(1:D*K);
    dlnscale = reshape(dL((1:D*K)+D*K),[D,K]);        
    dsigma = sum(dlnscale,1);
    if vp.optimize_lambda
        dlambda = sum(dlnscale,2);
    else
        dlambda = [];
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