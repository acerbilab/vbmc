function [LB_theta,UB_theta] = vbmc_vpbnd(vp,X,K,options)
%VBMC_VPBND Bounds over variational posterior parameters.

width = max(X) - min(X);

LB_mu = repmat(min(X)-0.5*width,[1,K]);
UB_mu = repmat(max(X)+0.5*width,[1,K]);
LB_sigma = min(width./vp.lambda')*options.TolLength*ones(1,K);
UB_sigma = max(width./vp.lambda')*ones(1,K);    
if vp.optimize_lambda
    LB_lambda = width*sqrt(options.TolLength);
    UB_lambda = width;
else
    LB_lambda = [];
    UB_lambda = [];        
end

LB_theta = [LB_mu,log(LB_sigma),log(LB_lambda)];
UB_theta = [UB_mu,log(UB_sigma),log(UB_lambda)];

end