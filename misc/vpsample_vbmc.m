function [vp,samples,output] = vpsample_vbmc(Ns,Ninit,vp,gp,optimState,options,wide_flag)

if nargin < 7 || isempty(wide_flag); wide_flag = false; end

% Assign default values to OPTIMSTATE
if ~isfield(optimState,'delta'); optimState.delta = 0; end
if ~isfield(optimState,'EntropySwitch'); optimState.EntropySwitch = false; end
if ~isfield(optimState,'Warmup'); optimState.Warmup = ~vp.optimize_weights; end
if ~isfield(optimState,'temperature'); optimState.temperature = 1; end

%% Set up sampling variables and options

% Perform quick sieve to determine good starting point
[vp,~,elcbo_beta,compute_var,NSentK] = ...
    vpsieve_vbmc(Ninit,1,vp,gp,optimState,options);

K = vp.K;
D = vp.D;

% Compute soft bounds for variational parameters optimization
[vp,thetabnd] = vpbounds(vp,gp,options,K);

% Move lower bound on scale - we want *wider* distributions
if wide_flag
    lnscale = bsxfun(@plus,log(vp.sigma(:))',log(vp.lambda(:)));
    if vp.optimize_mu; idx = D*K; else; idx = 0; end
    thetabnd.lb(idx+1:idx+K*D) = lnscale;
end

%% Sample variational posterior starting from current

theta0 = get_vptheta(vp)';
Ntheta = numel(theta0);

% MCMC parameters
Widths = 0.5;
sampleopts.Thin = 1;
sampleopts.Burnin = 0;
sampleopts.Display = 'off';
sampleopts.Diagnostics = false;
LB = -Inf(1,Ntheta);
UB = Inf(1,Ntheta);

idx_fixed = false(size(theta0));
if ~optimState.Warmup && 0
    if vp.optimize_mu; idx_fixed(1:D*K) = true; end
%    idx_fixed = true(size(theta0));
%    idx_fixed(idx+1:idx+K) = false;
end

LB(idx_fixed) = theta0(idx_fixed);
UB(idx_fixed) = theta0(idx_fixed);

% Perform sampling
try
    switch lower(options.VariationalSampler)
        case 'slicesample'
            vpmcmc_fun = @(theta_) -negelcbo_vbmc(theta_,elcbo_beta,vp,gp,NSentK,0,compute_var,0,thetabnd);
            [samples,fvals,exitflag,output] = ...
                slicesample_vbmc(vpmcmc_fun,theta0,Ns,Widths,LB,UB,sampleopts);
        case 'malasample'
            if isfield(optimState,'mcmc_stepsize')
                sampleopts.Stepsize = optimState.mcmc_stepsize; 
                output.stepsize = sampleopts.Stepsize;
            end
            vpmcmc_fun = @(theta_) vpmcmcgrad_fun(theta_,elcbo_beta,vp,gp,NSentK,compute_var,thetabnd);
            [samples,fvals,exitflag,output] = ...
                malasample_vbmc(vpmcmc_fun,theta0,Ns,Widths,LB,UB,sampleopts);
            % output.accept_rate
    end
catch
    samples = repmat(theta0,[Ns,1]);
end
vp = rescale_params(vp,samples(end,:));

end

function [logp,dlogp] = vpmcmcgrad_fun(theta,elcbo_beta,vp,gp,NSentK,compute_var,thetabnd)
    [nlogp,ndlogp] = negelcbo_vbmc(theta,elcbo_beta,vp,gp,NSentK,1,compute_var,0,thetabnd);
    logp = -nlogp;
    dlogp = -ndlogp;
end


