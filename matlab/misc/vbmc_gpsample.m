function Xs = vbmc_gpsample(gp,Ns,vp,optimState,origflag)
%VBMC_GPSAMPLE Draw random samples from GP approximation.

if nargin < 5 || isempty(origflag); origflag = true; end

method = 'eissample';
method = 'slicesamplebnd';

MaxBnd = 10;
D = vp.D;

widths = std(gp.X,[],1);
diam = max(gp.X) - min(gp.X);
LB = min(gp.X) - MaxBnd*diam;
UB = max(gp.X) + MaxBnd*diam;

logpfun = @(x) gplite_pred(gp,x);

switch method
    case 'slicesamplebnd'
        sampleopts.Burnin = ceil(Ns/10);
        sampleopts.Thin = 1;
        sampleopts.Display = 'off';
        sampleopts.Diagnostics = false;

        [~,idx0] = max(gp.y);
        Xs = slicesamplebnd(logpfun, ...
            gp.X(idx0,:),Ns,widths,LB,UB,sampleopts);
        
    case 'eissample'
        
        sampleopts.Burnin = ceil(Ns/5);
        sampleopts.Thin = 1;
        sampleopts.Display = 'off';
        sampleopts.Diagnostics = false;
        sampleopts.VarTransform = false;
        sampleopts.InversionSample = false;
        sampleopts.FitGMM = false;
        % Transition operators
        sampleopts.TransitionOperators = @transSliceSampleRD;
        
        W = 2*(D+1);
        x0 = vbmc_rnd(W,vp,0,1);
        x0 = bsxfun(@min,bsxfun(@max,x0,LB),UB);
        Xs = eissample(logpfun,x0,Ns,W,widths,LB,UB,sampleopts);
        
        
        
end

if origflag
    Xs = warpvars(Xs,'inv',optimState.trinfo);
end

end