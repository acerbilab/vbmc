function [vp,varss] = vpoptimizeweights_vbmc(vp,gp,optimState,options,prnt)
%VPOPTIMIZEWEIGHTS Optimize weights of variational posterior.

if nargin < 5 || isempty(prnt); prnt = 0; end

% Assign default values to OPTIMSTATE
if ~isfield(optimState,'delta'); optimState.delta = 0; end
if ~isfield(optimState,'EntropySwitch'); optimState.EntropySwitch = false; end
if ~isfield(optimState,'Warmup'); optimState.Warmup = ~vp.optimize_weights; end
if ~isfield(optimState,'temperature'); optimState.temperature = 1; end
    
optimize_mu = vp.optimize_mu;
optimize_sigma = vp.optimize_sigma;
optimize_lambda = vp.optimize_lambda;

vp.optimize_mu = false;
vp.optimize_sigma = false;
vp.optimize_lambda = false;
vp.optimize_weights = true;

% Get variational optimization options
[vp,~,elcbo_beta,compute_var,NSentK] = ...
    vpsieve_vbmc(0,1,vp,gp,optimState,options,vp.K);

if compute_var == 1     % For the moment no gradient available for variance
    options.StochasticOptimizer = 'cmaes';
end

% Compute soft bounds for variational parameters optimization
[vp,thetabnd] = vpbounds(vp,gp,options,vp.K);

%% Perform optimization starting from one or few selected points

% Set up empty stats structs for optimization
Ntheta = numel(get_vptheta(vp));
elbostats = eval_fullelcbo(1,Ntheta);

% Set basic options for deterministic optimizer (FMINUNC)
vbtrain_options = optimoptions('fminunc','GradObj','on','Display','off');

% Compute separate contributions to the log joint
[vp.I_sk,~,vp.J_sjk] = gplogjoint(vp,gp,0,0,0,1,1);

vp0 = rescale_params(vp);
theta0 = log(vp0.w(:));

vbtrainmc_fun = @(theta_) negelcbo_vbmc(theta_,elcbo_beta,vp0,gp,NSentK,1,compute_var,options.AltMCEntropy,thetabnd);

if NSentK == 0
    % Fast optimization via deterministic entropy approximation
    TolOpt = options.DetEntTolOpt;
    vbtrain_options.TolFun = TolOpt;
    vbtrain_fun = @(theta_) negelcbo_vbmc(theta_,elcbo_beta,vp0,gp,0,1,compute_var,0,thetabnd);
    try
        [thetaopt,~,~,output] = fminunc(vbtrain_fun,theta0(:)',vbtrain_options);
    catch
        % FMINUNC failed, try with CMA-ES
        if prnt > 0
            fprintf('Cannot optimize variational parameters with FMINUNC. Trying with CMA-ES (slower).\n');
        end
        insigma = ones(vp.K,1);
        cmaes_opts = options.CMAESopts;
        cmaes_opts.EvalParallel = 'off';
        cmaes_opts.TolX = '1e-8*max(insigma)';
        cmaes_opts.TolFun = 1e-6;
        cmaes_opts.TolHistFun = 1e-7;
        cmaes_opts.MaxFunEvals = 200*vp.D;            
        thetaopt = cmaes_modded('negelcbo_vbmc',theta0(:),insigma,cmaes_opts, ...
            elcbo_beta,vp0,gp,0,1,compute_var,0,thetabnd); 
        thetaopt = thetaopt(:)';
    end
    % output, % pause
else
    % Optimization via unbiased stochastic entropy approximation
    thetaopt = theta0(:)';

    switch lower(options.StochasticOptimizer)
        case 'adam'

            master_stepsize.min = min(options.SGDStepSize,0.001);
            scaling_factor = min(0.1,options.SGDStepSize);
            master_stepsize.max = max(master_stepsize.min,scaling_factor);
            master_stepsize.decay = 200;
            [thetaopt,~,theta_lst,fval_lst] = ...
                fminadam(vbtrainmc_fun,thetaopt,[],[],options.TolFunStochastic,[],master_stepsize);

        case 'cmaes'

            insigma = ones(vp.K,1);
            cmaes_opts = options.CMAESopts;
            cmaes_opts.EvalParallel = 'off';
            cmaes_opts.TolX = '1e-6*max(insigma)';
            cmaes_opts.TolFun = 1e-4;
            cmaes_opts.TolHistFun = 1e-5;
            cmaes_opts.Noise.on = 1;    % Noisy evaluations
            try
                thetaopt = cmaes_modded('negelcbo_vbmc',theta0(:),insigma,cmaes_opts, ...
                    elcbo_beta,vp0,gp,NSentK,0,compute_var,options.AltMCEntropy,thetabnd); 
            catch
                pause
            end
            thetaopt = thetaopt(:)';

        otherwise
            error('vbmc:VPoptimize','Unknown stochastic optimizer.');
    end
end

vp0 = rmfield(vp0,{'I_sk','J_sjk'});
vp0 = rescale_params(vp0,thetaopt);
vp0.optimize_mu = optimize_mu;
vp0.optimize_sigma = optimize_sigma;
vp0.optimize_lambda = optimize_lambda;

thetaopt = get_vptheta(vp0,vp0.optimize_mu,vp0.optimize_sigma,vp0.optimize_lambda,vp0.optimize_weights);

% Recompute ELCBO at endpoint with full variance and more precision

elbostats = eval_fullelcbo(1,thetaopt,vp0,gp,elbostats,elcbo_beta,options);


%% Finalize optimization by taking variational parameters with best ELCBO

[~,idx] = min(elbostats.nelcbo);
elbo = -elbostats.nelbo(idx);
elbo_sd = sqrt(elbostats.varF(idx));
G = elbostats.G(idx);
H = elbostats.H(idx);
varss = elbostats.varss(idx);
varG = elbostats.varG(idx);
varH = elbostats.varH(idx);
vp = vp0;
vp.temperature = optimState.temperature;

vp.stats.elbo = elbo;               % ELBO
vp.stats.elbo_sd = elbo_sd;         % Error on the ELBO
vp.stats.elogjoint = G;             % Expected log joint
vp.stats.elogjoint_sd = sqrt(varG); % Error on expected log joint
vp.stats.entropy = H;               % Entropy
vp.stats.entropy_sd = sqrt(varH);   % Error on the entropy
vp.stats.stable = false;            % Unstable until proven otherwise



% idx
% elbostats

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function elbostats = eval_fullelcbo(idx,theta,vp,gp,elbostats,beta,options)
%EVAL_FULLELCBO Evaluate full expected lower confidence bound.

if nargin == 2
    D = theta;
    elbostats.nelbo = Inf(1,idx);
    elbostats.G = NaN(1,idx);
    elbostats.H = NaN(1,idx);
    elbostats.varF = NaN(1,idx);
    elbostats.varG = NaN(1,idx);
    elbostats.varH = NaN(1,idx);
    elbostats.varss = NaN(1,idx);
    elbostats.nelcbo = Inf(1,idx);
    elbostats.theta = NaN(idx,D);
else
    
    % Number of samples per component for MC approximation of the entropy
    K = vp.K;
    NSentFineK = ceil(evaloption_vbmc(options.NSentFine,K)/K);
        
    theta = theta(:)';
    [nelbo,~,G,H,varF,~,varss,varG,varH] = ...
        negelcbo_vbmc(theta,0,vp,gp,NSentFineK,0,1,options.AltMCEntropy,[]);
    nelcbo = nelbo + beta*sqrt(varF);

    elbostats.nelbo(idx) = nelbo;
    elbostats.G(idx) = G;
    elbostats.H(idx) = H;
    elbostats.varF(idx) = varF;
    elbostats.varG(idx) = varG;
    elbostats.varH(idx) = varH;
    elbostats.varss(idx) = varss;
    elbostats.nelcbo(idx) = nelcbo;
    elbostats.theta(idx,1:numel(theta)) = theta;
end

end