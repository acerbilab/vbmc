function acq = acqfsn2_vbmc(Xs,vp,gp,optimState,fmu,fs2,fbar,vtot)
%ACQFSN2_VBMC Acquisition fcn. for noisy prospective uncertainty search.

% Xs is in *transformed* coordinates

% Probability density of variational posterior at test points
p = max(vbmc_pdf(vp,Xs,0),realmin);

% Estimate observation noise at test points from nearest neighbor
[~,pos] = min(sq_dist(bsxfun(@rdivide,Xs,optimState.gplengthscale),gp.X_rescaled),[],2);
sn2 = gp.sn2new(pos);

z = optimState.ymax;

% Prospective uncertainty search corrected for noisy observations
acq = -vtot.*(1 - sn2./(vtot+sn2)) .* exp(fbar-z) .* p;

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