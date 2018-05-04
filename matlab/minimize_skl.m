function sKL = minimize_skl(lnlambda,vp,vp_old,Nkl)

if nargin < 4 || isempty(Nkl); Nkl = 1e3; end

vp.lambda = exp(lnlambda(:));
sKL = 0.5*sum(vbmc_kldiv(vp,vp_old,Nkl,0,1));

end