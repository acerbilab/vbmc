function [vp,optimState,hyp] = warp_rotoscaling(vp,optimState,hyp,gp,cmaes_opts,options)
%WARP_ROTOSCALING Compute rotation and scaling warping of variables.

vp_old = vp;
D = vp.D;

if ~isempty(vp.trinfo.R_mat); R_mat_old = vp.trinfo.R_mat; else; R_mat_old = eye(D); end
if ~isempty(vp.trinfo.scale); scale_old = vp.trinfo.scale; else; scale_old = ones(1,D); end

% Get covariance matrix in transformed space
[~,vp_Sigma_mom] = vbmc_moments(vp,0);




% Alternative way of getting covariance - maximize GP and compute Hessian
if ~isempty(gp)
%     Thin = 1;
%     Xrnd = vbmc_gpsample(gp,1e3,vp,optimState,0);
%     Xrnd = Xrnd(Thin:Thin:end,:);
%     vp_Sigma_smpl = cov(Xrnd);    
    vp_Sigma_smpl = [];
    
%     fminopts.Display = 'off';
%     [~,idx] = max(gp.y);        
%     xmax = fminunc(@(x) -gplite_pred(gp,x),gp.X(idx,:),fminopts);
%     hess = hessianest(@(x) gplite_pred(gp,x),xmax);
%     vp_Sigma = inv(-hess);
%     [vp_Sigma,vp_Sigma_smpl,vp_Sigma_mom]
end

vp_Sigma = vp_Sigma_mom;

% Reverse rotation and scaling
vp_Sigma = R_mat_old*(diag(scale_old)*vp_Sigma*diag(scale_old))*R_mat_old';

% Regularization of covariance matrix towards diagonal
if isnumeric(options.WarpCovReg)
    w_reg = options.WarpCovReg;
else
    w_reg = options.WarpCovReg(optimState.N);
end
w_reg = max(0,min(1,w_reg));
vp_Sigma = (1-w_reg)*vp_Sigma + w_reg*diag(diag(vp_Sigma));

% Compute whitening transform (rotoscaling)
[U,S] = svd(vp_Sigma);
if det(U) < 0; U(:,1) = -U(:,1); end
scale = sqrt(diag(S+eps))';        
vp.trinfo.R_mat = U';
vp.trinfo.scale = scale;

% Readjust variational posterior and GP hyperparameters after rotoscaling
[vp,optimState,hyp_warped] = recompute_vp_and_hyp(vp,vp_old,optimState,cmaes_opts,options,1,hyp,gp);
optimState.trinfo = vp.trinfo;
hyp = [hyp,hyp_warped];

end
