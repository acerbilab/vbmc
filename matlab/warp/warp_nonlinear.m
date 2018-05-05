function [vp,optimState,hyp,hyp_warp] = warp_nonlinear(vp,optimState,hyp,hyp_warp,cmaes_opts,options)
%WARP_NONLINEAR Compute nonlinear warping of variables via GP fit.

vp_old = vp;

[hypprior_warp,~,~,Nhyp,hyp0] = ...
    vbmc_gphyp(optimState,optimState.gpMeanfun,1,options);
if isempty(hyp); hyp = hyp0(1:Nhyp); end % Initial GP hyperparameters

warp.LB = vp.trinfo.lb_orig;
warp.UB = vp.trinfo.ub_orig;
warp.logpdf_flag = true;    % The GP represents a log pdf
warp.Nwarp = 2*vp.D;

% Copy rotation and rescaling info
if isfield(vp.trinfo,'R_mat'); warp.R_mat = vp.trinfo.R_mat; end
if isfield(vp.trinfo,'scale'); warp.scale = vp.trinfo.scale; end
if isfield(vp.trinfo,'mu'); warp.mu = vp.trinfo.mu; end
if isfield(vp.trinfo,'delta'); warp.delta = vp.trinfo.delta; end

% If input warping, perform hyperparameter optimization to get the warping        
if isempty(hyp_warp)
    hyp_warp = [hyp; repmat(hyp0(Nhyp+1:end,1),[1,size(hyp,2)])];
else
    % Include previous warp and no warping as starting points
    hyp_warp = [repmat(hyp,[1,2]); ...
        [repmat(hyp_warp(Nhyp+1:end,1),[1,size(hyp,2)]),zeros(warp.Nwarp,size(hyp,2))]];
end

% Might want to include a smarter initialization for the warping, 
% such as fitting a Kumaraswamy-logit-normal to each marginal 
% variational pdf and getting the Kumaraswamy parameters.

%         xx = vbmc_rnd(1e5,vp,1,1);
%         hyp_warpt = fmincon(@(hyp) warpgaussfit(hyp,xx,vp.trinfo),hyp_warp(Nhyp+1:end,1),[],[],[],[],LB_warp,UB_warp);

gptrain_options.Nopts = 1;  % Warping is expensive, only one restart
X_orig = optimState.X_orig(1:optimState.Xmax,:);
y_orig = optimState.y_orig(1:optimState.Xmax);
[gp,hyp_warp] = ...
    gplite_train(hyp_warp,0,X_orig,y_orig,optimState.gpMeanfun,hypprior_warp,warp,gptrain_options);
% warp = gp.warp;
vp.trinfo = gp.warp.trinfo;
if ~isfield(vp.trinfo,'R_mat'); vp.trinfo.R_mat = []; end
if ~isfield(vp.trinfo,'scale'); vp.trinfo.scale = []; end        

% Add input warped hyperparameters to candidate starting points
hyp = [hyp,hyp_warp(1:Nhyp,:)];

% Update variational posterior after warping
[vp,optimState] = recompute_vp_and_hyp(vp,vp_old,optimState,cmaes_opts,options);
optimState.trinfo = vp.trinfo;

% Major change, fully recompute variational posterior
optimState.RecomputeVarPost = true;

% [vp.trinfo.alpha;vp.trinfo.beta]

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function nll = warpgaussfit(hyp_warp,xx,trinfo)

[N,D] = size(xx);
trinfo.type(trinfo.type == 3) = 9;
trinfo.type(trinfo.type == 0) = 10;

alpha = exp(hyp_warp(1:D));
beta = exp(hyp_warp(D+(1:D)));
trinfo.alpha = alpha(:)';
trinfo.beta = beta(:)';        
xx_t = pdftrans(xx,'d',trinfo);
mu = mean(xx_t);
sigma = std(xx_t);
nll = 0.5*mean(sum(bsxfun(@rdivide,bsxfun(@minus, xx_t, mu),sigma).^2,2)) + 0.5*D*log(2*pi) + log(prod(sigma));
nll = nll + mean(pdftrans(xx_t,'l',trinfo));

% Student's t prior
mu_prior = 0;
sigma_prior = 0.25;
df_prior = 3;
z2 = ((hyp_warp - mu_prior)./sigma_prior).^2;
nll = nll - sum(gammaln(0.5*(df_prior+1)) - gammaln(0.5*df_prior) - 0.5*log(pi*df_prior) ...
    - log(sigma_prior) - 0.5*(df_prior+1).*log1p(z2./df_prior));



end