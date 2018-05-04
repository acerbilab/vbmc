function [vp,optimState,hyp_warped] = recompute_vp_and_hyp(vp,vp_old,optimState,cmaes_opts,options,linearflag,hyp,gp)
%RECOMPUTE_VP_AND_HYP Recompute variational parameters and GP hyperparameters after warping.

if nargin < 6 || isempty(linearflag); linearflag = 0; end
if nargin < 7; hyp = []; end
if nargin < 8; gp = []; end

hpd_frac = options.HPDFrac;
lnToL = log(options.TolLength);

D = vp.D;
trinfo = vp.trinfo;
trinfo_old = vp_old.trinfo;

if ~isempty(trinfo_old.R_mat); R_mat_old = trinfo_old.R_mat; else; R_mat_old = eye(D); end
if ~isempty(trinfo_old.scale); scale_old = trinfo_old.scale; else; scale_old = ones(1,D); end        

% Transform variables
X_orig = optimState.X_orig(1:optimState.Xmax,:);
y_orig = optimState.y_orig(1:optimState.Xmax);
X = pdftrans(X_orig,'dir',trinfo);
dy = pdftrans(X,'logp',trinfo);
y = y_orig + dy;
optimState.X(1:optimState.Xmax,:) = X;
optimState.y(1:optimState.Xmax) = y;

mu_old = vp_old.mu;
if linearflag
    if ~isempty(trinfo.R_mat); R_mat = trinfo.R_mat; else; R_mat = eye(D); end
    if ~isempty(trinfo.scale); scale = trinfo.scale; else; scale = ones(1,D); end
    vp.mu = bsxfun(@rdivide,(bsxfun(@times,mu_old',scale_old)*R_mat_old')*R_mat,scale)';
    ln_scaling = -sum(log(scale) - log(scale_old));    
else
    % Update variational posterior
    vp.mu = pdftrans(pdftrans(mu_old','inv',trinfo_old),'dir',vp.trinfo)';
    ln_scaling = -(pdftrans(vp.mu','logpdf',vp.trinfo)' - pdftrans(mu_old','logpdf',trinfo_old)');
end

vp.sigma = vp.sigma .* exp((ln_scaling-mean(ln_scaling))/D);
%vp.sigma = vp.sigma .* exp(ln_scaling/D);

Xstar = vbmc_rnd(1e3,vp_old,1,1);
% mu_orig = pdftrans(vp_old.mu','inv',vp_old.trinfo);

if vp.optimize_lambda
    vp = recompute_lambda(vp,vp_old,Xstar,X,y,cmaes_opts,hpd_frac,lnToL);
end

% Warp GP hyperparameters
if ~isempty(hyp) && ~isempty(gp) && nargout > 2
    mu_orig = pdftrans(vp_old.mu','inv',vp_old.trinfo); 
    hyp_warped = recompute_hyp(hyp,gp,trinfo,trinfo_old,X,X_orig,mu_orig,dy);
else
    hyp_warped = [];
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function vp = recompute_lambda(vp,vp_old,Xstar,X,y,cmaes_opts,hpd_frac,lnToL)
%WARP_LAMBDA Recompute LAMBDA of variational posterior after warping.

Nkl_fast = 1e2;
Nkl_fine = 1e5;

ToLKL = [0.25,0.5];    % Maximum tolerance on KL for acceptance at first and second step

vbwarp_options = cmaes_opts;
vbwarp_options.EvalParallel = 'no';
vbwarp_options.Noise.on = 1;   % Use noise-handling
vbwarp_options.MaxFunEvals = 100*vp.D;
%vbwarp_options.DispFinal = 'on';
%vbwarp_options.DispModulo = 50;

% Xstar = pdftrans(vp_old.mu','inv',vp_old.trinfo);
lambda_warped = warp_lengths(vp.lambda',Xstar,vp.trinfo,vp_old.trinfo,0);

[~,ord] = sort(y,'descend');
N_hpd = round(hpd_frac*size(X,1));
X_hpd = X(ord(1:N_hpd),:);
width_hpd = max(X_hpd) - min(X_hpd);
LB_lnlambda = log(width_hpd*exp(0.5*lnToL));
UB_lnlambda = log(width_hpd);
lnlambda_warped = min(max(log(lambda_warped(:)'),LB_lnlambda),UB_lnlambda);
lnlambda_old = min(max(log(vp.lambda(:)'),LB_lnlambda),UB_lnlambda);

PLB_lnlambda = max(lnlambda_warped - std(lnlambda_warped),LB_lnlambda);
PUB_lnlambda = min(lnlambda_warped + std(lnlambda_warped),UB_lnlambda);

lnlambda(:,1) = lnlambda_warped;

skl = Inf(1,4);
skl(1) = minimize_skl(lnlambda(:,1),vp,vp_old,Nkl_fine);    % Warped LAMBDA

if skl(1) > ToLKL(1)   % Make other computations only if needed

    % Perform smart grid optimization
    optfill.FunEvals = 200;
    [lnlambda(:,2),~,~,output] = fminfill(@(lnlambda_) minimize_skl(lnlambda_(:),vp,vp_old,Nkl_fast), ...
        lnlambda_old,LB_lnlambda,UB_lnlambda,PLB_lnlambda,PUB_lnlambda,[],optfill);
    skl(2) = minimize_skl(lnlambda(:,2),vp,vp_old,Nkl_fine);    % Best point from Sobol grid 

    if min(skl(1:2)) > ToLKL(2) && ~isempty(cmaes_opts)  % Try harder, use CMA-ES
        
        try
            insigma = (UB_lnlambda(:) - LB_lnlambda(:))/10;
            vbwarp_options.LBounds = LB_lnlambda(:);
            vbwarp_options.UBounds = UB_lnlambda(:);
            [lnlambda(:,3),~,~,~,~,bestever] = ...
                cmaes('minimize_skl',lnlambda(:,2),insigma,vbwarp_options,vp,vp_old,Nkl_fast);
            lnlambda(:,4) = bestever.x;
            
            skl(3) = minimize_skl(lnlambda(:,3),vp,vp_old,Nkl_fine);    % Output from CMA-ES
            if any(lnlambda(:,4) ~= lnlambda(:,3))
                skl(4) = minimize_skl(lnlambda(:,4),vp,vp_old,Nkl_fine);    % Best-ever from CMA-ES
            end
        catch
            % CMA-ES did not work for some reason
            warning('recompute_vp_and_hyp:CMAESFail',...
                'Failed to optimize LAMBDA variational parameters with CMA-ES.')
        end
    end
end

% lnlambda
% skl

[~,idx] = min(skl);            
vp.lambda = exp(lnlambda(:,idx));
vp = rescale_params(vp);

if 0
    xx = vbmc_rnd(1e5,vp_old,1,1);
    cornerplot(xx);
    xx = vbmc_rnd(1e5,vp,1,1);
    cornerplot(xx);
    skl
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hyp_warped = recompute_hyp(hyp,gp,trinfo,trinfo_old,X,X_orig,mu_orig,dy)
%RECOMPUTE_HYP Recompute GP hyperparameters after warping.

hyp_warped = hyp;
if isempty(hyp); return; end

D = size(X,2);
Ncov = gp.Ncov;

quadratic_mean = gp.meanfun == 4;

% Warp GP hyperparameters

% Length scales get warped
for iHyp = 1:size(hyp_warped,2)
    ell = exp(hyp_warped(1:D,iHyp))';
    hyp_warped(1:D,iHyp) = log(warp_lengths(ell,X_orig,trinfo,trinfo_old,0))';
end

% We assume the impact of function scaling is small

if quadratic_mean    
    % Warp center of quadratic mean
    xm = hyp_warped(Ncov+2+(1:D),:)';
    xm_warped = pdftrans(pdftrans(xm,'inv',trinfo_old),'dir',trinfo);                
    hyp_warped(Ncov+2+(1:D),:) = xm_warped';

    % Warp maximum of quadratic mean
    dy_old = pdftrans(xm,'logpdf',trinfo_old)';
    dy = pdftrans(xm_warped,'logpdf',trinfo)';
    hyp_warped(Ncov+2,:) = hyp_warped(Ncov+2,:) + dy - dy_old;
    
    % Warp length scale of quadratic mean
    omega = exp(hyp_warped(Ncov+2+D+(1:D),:))';
    for iHyp = 1:size(hyp_warped,2)
        hyp_warped(Ncov+2+D+(1:D),iHyp) = log(warp_lengths(omega(iHyp,:),mu_orig,trinfo,trinfo_old,0))';
    end
else
    % Warp constant mean
    dy_old = pdftrans(X,'logp',trinfo_old);
    hyp_warped(Ncov+2,:) = hyp_warped(Ncov+2,:) + mean(dy) - mean(dy_old);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ell_warped = warp_lengths(ell,X_orig,trinfo,trinfo_old,normflag)
%WARP_LENGTHS Modify a length scale vector after warping.

if nargin < 5 || isempty(normflag); normflag = false; end

D = numel(ell);

% Compute stretch factor for old warping
if ~isempty(trinfo_old.R_mat); R_mat_old = trinfo_old.R_mat; else; R_mat_old = eye(D); end
if ~isempty(trinfo_old.scale); scale_old = trinfo_old.scale; else; scale_old = ones(1,D); end
trinfo_old.R_mat = [];
trinfo_old.scale = [];
Xt_old = pdftrans(X_orig,'dir',trinfo_old);
stretch_factor_old = std(Xt_old)./std(X_orig);

% Compute stretch factor for new warping
if ~isempty(trinfo.R_mat); R_mat = trinfo.R_mat; else; R_mat = eye(D); end
if ~isempty(trinfo.scale); scale = trinfo.scale; else; scale = ones(1,D); end
trinfo.R_mat = [];
trinfo.scale = [];
Xt = pdftrans(X_orig,'dir',trinfo);
stretch_factor = std(Xt)./std(X_orig);

% Compute overall transformation of scale lengths
S = diag(ell.^2);
S = R_mat_old*(diag(scale_old)*S*diag(scale_old))*R_mat_old';
S = diag(stretch_factor./stretch_factor_old)*S*diag(stretch_factor./stretch_factor_old);
S = diag(1./scale)*(R_mat'*S*R_mat)*diag(1./scale);

% Normalize vector to dimension (as per LAMBDA) if required
ell2_warped = diag(S);
if normflag
    nl2 = sum(ell2_warped)/D;
else
    nl2 = 1;
end
ell_warped = sqrt(ell2_warped/nl2);

end