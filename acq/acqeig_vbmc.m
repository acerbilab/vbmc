function acq = acqeig_vbmc(Xs,vp,gp,optimState,fmu,fs2,fbar,vtot)
%ACQEIG_VBMC Expected information gain (EIG) acquisition function.

if isempty(Xs)
    % Return acquisition function info struct
    acq.compute_varlogjoint = true;
    return;
end

% Xs is in *transformed* coordinates
Ns = numel(gp.post);

% Estimate observation noise at test points from nearest neighbor
[~,pos] = min(sq_dist(bsxfun(@rdivide,Xs,optimState.gplengthscale),gp.X_rescaled),[],2);
sn2 = gp.sn2new(pos);

intK = intkernel(Xs,vp,gp,0);
ys2 = fs2 + sn2;    % Predictive variance at test points

rho2 = bsxfun(@rdivide,intK.^2,optimState.varlogjoint_samples.*ys2);
acq = 0.5*sum(log(max(realmin,1 - min(1,rho2))),2)/Ns;

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