function X = gpsample_vbmc(vp,gp,Ns,origflag)
%GPSAMPLE_VBMC Sample from GP obtained through VBMC.

if nargin < 4 || isempty(origflag); origflag = true; end

D = size(gp.X,2);

if isfield(gp,'s2') && ~isempty(gp.s2)
    % Evaluate GP input length scale (use geometric mean)
    Ns_gp = numel(gp.post);
    ln_ell = zeros(D,Ns_gp);
    for s = 1:Ns_gp; ln_ell(:,s) = gp.post(s).hyp(1:D); end
    gplengthscale = exp(mean(ln_ell,2))';
    X_rescaled = bsxfun(@rdivide,gp.X,gplengthscale); % Rescaled GP training inputs

    % Evaluate GP observation noise on training inputs
    sn2new = zeros(size(gp.X,1),Ns_gp);
    for s = 1:Ns_gp
        hyp_noise = gp.post(s).hyp(gp.Ncov+1:gp.Ncov+gp.Nnoise); % Get noise hyperparameters 
        if isfield(gp,'s2')
            s2 = gp.s2;
        else
            s2 = [];
        end
        % s2 = noiseshaping_vbmc(s2,gp.y,options);
        sn2new(:,s) = gplite_noisefun(hyp_noise,gp.X,gp.noisefun,gp.y,s2);
    end
    sn2new = mean(sn2new,2);    
    
    % Estimate observation noise variance over variational posterior
    xx = vbmc_rnd(vp,2e4,0,0);
    [~,pos] = min(sq_dist(bsxfun(@rdivide,xx,gplengthscale),X_rescaled),[],2);
    sn2_avg = mean(sn2new(pos));    % Use nearest neighbor approximation
else
    sn2_avg = 0;    
end

VarThresh = max(1,sn2_avg);

W = 2*(D+1);
x0 = vbmc_rnd(vp,W,0,0);
X = gplite_sample(gp,Ns,x0,'parallel',[],[],VarThresh);
if origflag
    X = warpvars_vbmc(X,'inv',vp.trinfo);
end

end



%SQ_DIST Compute matrix of all pairwise squared distances between two sets 
% of vectors, stored in the columns of the two matrices, a (of size n-by-D) 
% and b (of size m-by-D).
function C = sq_dist(a,b)

n = size(a,1);
m = size(b,1);
mu = (m/(n+m))*mean(b,1) + (n/(n+m))*mean(a,1);
a = bsxfun(@minus,a,mu); b = bsxfun(@minus,b,mu);
C = bsxfun(@plus,sum(a.*a,2),bsxfun(@minus,sum(b.*b,2)',2*a*b'));
C = max(C,0);

end