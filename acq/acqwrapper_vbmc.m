function acq = acqwrapper_vbmc(Xs,vp,gp,optimState,transpose_flag,acqFun,acqInfo)
%ACQWRAPPER_VBMC Wrapper for all acquisition functions.

% Transposed input (useful for CMAES)
if transpose_flag; Xs = Xs'; end

% Map integer inputs
Xs = real2int_vbmc(Xs,vp.trinfo,optimState.integervars);

%% Compute GP posterior predictive mean and variance

if isfield(vp,'delta') && ~isempty(vp.delta) && any(vp.delta > 0)
    % Quadrature mean and variance for each hyperparameter sample
    [fmu,fs2] = gplite_quad(gp,Xs,vp.delta',1);    
else
    % GP mean and variance for each hyperparameter sample
    [~,~,fmu,fs2] = gplite_pred(gp,Xs,[],[],1,0);
end

% Compute total variance
Ns = size(fmu,2);
fbar = sum(fmu,2)/Ns;   % Mean across samples
vbar = sum(fs2,2)/Ns;   % Average variance across samples
if Ns > 1
    vf = sum(bsxfun(@minus,fmu,fbar).^2,2)/(Ns-1);
else
    vf = 0; 
end  % Sample variance
vtot = vf + vbar;       % Total variance

%% Compute acquisition function
acq = acqFun(Xs,vp,gp,optimState,fmu,fs2,fbar,vtot);

%% Regularization: penalize points where GP uncertainty is below threshold
if optimState.VarianceRegularizedAcqFcn
    TolVar = optimState.TolGPVar; % Try not to go below this variance
    idx = vtot < TolVar;
    
    if any(idx)        
        if isfield(acqInfo,'log_flag') && acqInfo.log_flag
            acq(idx) = acq(idx) + TolVar./vtot(idx) - 1;
        else
            acq(idx) = acq(idx) .* exp(-(TolVar./vtot(idx)-1));
        end        
    end
end
acq = max(acq,-realmax);

%% Hard bound checking: discard points too close to bounds
X_orig = warpvars_vbmc(Xs,'i',vp.trinfo);
idx = any(bsxfun(@lt,X_orig,optimState.LBeps_orig),2) | any(bsxfun(@gt,X_orig,optimState.UBeps_orig),2);
acq(idx) = Inf;

% Transposed output
if transpose_flag; acq = acq'; end

end