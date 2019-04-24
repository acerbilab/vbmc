function [Xs,gp] = vbmc_gpsample(gp,Ns,vp,optimState,origflag)
%VBMC_GPSAMPLE Draw random samples from GP approximation.

if nargin < 5 || isempty(origflag); origflag = true; end

method = 'parallel';

D = size(gp.X,2);
W = 2*(D+1);

if isempty(vp)
    x0 = [];
else
    x0 = vbmc_rnd(vp,W,0,1);
end

[Xs,gp] = gplite_sample(gp,Ns,x0,method);

if origflag
    Xs = warpvars(Xs,'inv',optimState.trinfo);
end

end