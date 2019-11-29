function [vp,thetabnd] = vpbounds(vp,gp,options,K)
%VPBOUNDS Compute soft bounds for variational posterior parameters.

if nargin < 4 || isempty(K); K = vp.K; end

% Soft-bound loss is computed on MU and SCALE (which is SIGMA times LAMBDA)

% Start with reversed bounds (see below)
if ~isfield(vp,'bounds') || isempty(vp.bounds)
    vp.bounds.mu_lb = Inf(1,vp.D);
    vp.bounds.mu_ub = -Inf(1,vp.D);
    vp.bounds.lnscale_lb = Inf(1,vp.D);
    vp.bounds.lnscale_ub = -Inf(1,vp.D);
    % vp.bounds
end

% Set bounds for mean parameters of variational components
vp.bounds.mu_lb = min(min(gp.X),vp.bounds.mu_lb);
vp.bounds.mu_ub = max(max(gp.X),vp.bounds.mu_ub);    
    
% Set bounds for log scale parameters of variational components
lnrange = log(max(gp.X) - min(gp.X));
vp.bounds.lnscale_lb = min(vp.bounds.lnscale_lb,lnrange + log(options.TolLength));
vp.bounds.lnscale_ub = max(vp.bounds.lnscale_ub,lnrange);

% Set bounds for log weight parameters of variational components
if vp.optimize_weights
    vp.bounds.eta_lb = log(0.5*options.TolWeight);
    vp.bounds.eta_ub = 0;
end

thetabnd.lb = [];
thetabnd.ub = [];
if vp.optimize_mu
    thetabnd.lb = [thetabnd.lb,repmat(vp.bounds.mu_lb,[1,K])];
    thetabnd.ub = [thetabnd.ub,repmat(vp.bounds.mu_ub,[1,K])];
end
if vp.optimize_sigma || vp.optimize_lambda
    thetabnd.lb = [thetabnd.lb,repmat(vp.bounds.lnscale_lb,[1,K])];
    thetabnd.ub = [thetabnd.ub,repmat(vp.bounds.lnscale_ub,[1,K])];
end
if vp.optimize_weights
    thetabnd.lb = [thetabnd.lb,repmat(vp.bounds.eta_lb,[1,K])];
    thetabnd.ub = [thetabnd.ub,repmat(vp.bounds.eta_ub,[1,K])];
end

thetabnd.TolCon = options.TolConLoss;

% Weights below a certain threshold are penalized
if vp.optimize_weights
    thetabnd.WeightThreshold = max(1/(4*K),options.TolWeight);
    thetabnd.WeightPenalty = options.WeightPenalty;
end

end