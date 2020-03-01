function acq = acqflog_vbmc(Xs,vp,gp,optimState,fmu,fs2,fbar,vtot)
%ACQFLOG_VBMC Acquisition fcn. for prospective uncertainty search (log-valued).

% Xs is in *transformed* coordinates

if isempty(Xs)
    % Return acquisition function info struct
    acq.compute_varlogjoint = false;
    acq.log_flag = true;
    return;
end

% Probability density of variational posterior at test points
p = max(vbmc_pdf(vp,Xs,0),realmin);

% Log prospective uncertainty search
z = optimState.ymax;
acq = -(log(vtot) + fbar-z + log(p));

end