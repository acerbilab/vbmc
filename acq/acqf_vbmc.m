function acq = acqf_vbmc(Xs,vp,gp,optimState,fmu,fs2,fbar,vtot)
%ACQF_VBMC Acquisition fcn. for prospective uncertainty search.

% Xs is in *transformed* coordinates

% Probability density of variational posterior at test points
p = max(vbmc_pdf(vp,Xs,0),realmin);

% Prospective uncertainty search
z = optimState.ymax;
acq = -vtot .* exp(fbar-z) .* p;

end