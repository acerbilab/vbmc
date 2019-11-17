function [vp,samples] = vpsample_vbmc(Ns,vp,gp,optimState,options,wide_flag)

if nargin < 5 || isempty(wide_flag); wide_flag = false; end

% Assign default values to OPTIMSTATE
if ~isfield(optimState,'delta'); optimState.delta = 0; end
if ~isfield(optimState,'EntropySwitch'); optimState.EntropySwitch = false; end
if ~isfield(optimState,'Warmup'); optimState.Warmup = ~vp.optimize_weights; end
if ~isfield(optimState,'temperature'); optimState.temperature = 1; end

%% Set up sampling variables and options

% It is assumed that VP was already optimized
K = vp.K;
D = vp.D;

% Number of samples per component for MC approximation of the entropy
if isa(options.NSent,'function_handle')
    NSentK = ceil(options.NSent(K)/K);
else
    NSentK = ceil(options.NSent/K);
end

% Deterministic entropy switch or only one component -- be deterministic
if optimState.EntropySwitch || K == 1
    NSentK = 0;
end

% Confidence weight
elcbo_beta = options.ELCBOWeight; % * sqrt(vp.D) / sqrt(optimState.N);   
if elcbo_beta ~= 0
    compute_var = 2;    % Use diagonal-only approximation
else
    compute_var = 0;    % No beta, skip variance
end

% Compute soft bounds for variational parameters optimization
[vp,thetabnd] = vpbounds(vp,gp,options);

% Move lower bound on scale - we want *wider* distributions
if wide_flag
    lnscale = bsxfun(@plus,log(vp.sigma(:))',log(vp.lambda(:)));
    if vp.optimize_mu; idx = D*K; else; idx = 0; end
    thetabnd.lb(idx+1:idx+K*D) = lnscale;
end

%% Sample variational posterior starting from current

theta0 = get_vptheta(vp)';
Ntheta = numel(theta0);

vpmcmc_fun = @(theta_) -negelcbo_vbmc(theta_,elcbo_beta,vp,gp,NSentK,0,compute_var,0,thetabnd);

% MCMC parameters
Widths = 0.5;
sampleopts.Thin = 1;
sampleopts.Burnin = 0;
sampleopts.Display = 'off';
sampleopts.Diagnostics = false;
LB = -Inf(1,Ntheta);
UB = Inf(1,Ntheta);

if optimState.Warmup
    idx_fixed = false(size(theta0));
else
    if vp.optimize_mu; idx = D*K; else; idx = 0; end
    idx_fixed = true(size(theta0));
    idx_fixed(idx+1:idx+K) = false;
end

LB(idx_fixed) = theta0(idx_fixed);
UB(idx_fixed) = theta0(idx_fixed);

% Perform sampling
try
    [samples,fvals,exitflag,output] = ...
        slicesample_vbmc(vpmcmc_fun,theta0,Ns,Widths,LB,UB,sampleopts);
catch
    samples = repmat(theta0,[Ns,1]);
end
vp = rescale_params(vp,samples(end,:));