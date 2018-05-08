function [Xs,gp] = vbmc_gpsample(gp,Ns,vp,optimState,origflag)
%VBMC_GPSAMPLE Draw random samples from GP approximation.

if nargin < 5 || isempty(origflag); origflag = true; end

method = 'parallel';

MaxBnd = 10;
D = size(gp.X,2);

widths = std(gp.X,[],1);
diam = max(gp.X) - min(gp.X);
LB = min(gp.X) - MaxBnd*diam;
UB = max(gp.X) + MaxBnd*diam;

% First, train GP
if ~isfield(gp,'post') || isempty(gp.post)
    % How many samples for the GP?
    if isfield(gp,'Ns') && ~isempty(gp.Ns)
        Ns_gp = gp.Ns;
    else
        Ns_gp = 0;
    end
    options.Nopts = 1;  % Do only one optimization    
    gp = gplite_train(...
        [],Ns_gp,gp.X,gp.y,gp.meanfun,[],[],options);
end

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
        
    case 'parallel'
        
        sampleopts.Burnin = ceil(Ns/5);
        sampleopts.Thin = 1;
        sampleopts.Display = 'off';
        sampleopts.Diagnostics = false;
        sampleopts.VarTransform = false;
        sampleopts.InversionSample = false;
        sampleopts.FitGMM = false;
        % sampleopts.TransitionOperators = {'transSliceSampleRD'};
        
        W = 2*(D+1);
        if isempty(vp)
            % Take starting points from high posterior density region
            hpd_frac = 0.25;
            N = numel(gp.y);
            N_hpd = min(N,max(W,round(hpd_frac*N)));
            [~,ord] = sort(gp.y,'descend');
            X_hpd = gp.X(ord(1:N_hpd),:);
            x0 = X_hpd(randperm(N_hpd,min(W,N_hpd)),:);
        else
            x0 = vbmc_rnd(W,vp,0,1);
        end
        x0 = bsxfun(@min,bsxfun(@max,x0,LB),UB);
        Xs = eissample_lite(logpfun,x0,Ns,W,widths,LB,UB,sampleopts);
end

if origflag
    Xs = warpvars(Xs,'inv',optimState.trinfo);
end

end