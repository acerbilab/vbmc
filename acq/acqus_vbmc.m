function acq = acqus_vbmc(Xs,vp,gp,optimState,fmu,fs2,fbar,vtot)
%ACQUS_VBMC Acquisition fcn via vanilla uncertainty sampling.

% Xs is in *transformed* coordinates

% Probability density of variational posterior at test points
p = max(vbmc_pdf(vp,Xs,0),realmin);

% Uncertainty search
acq = -vtot .* p.^2;

end