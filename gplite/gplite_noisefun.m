function [sn2,dsn2] = gplite_noisefun(hyp,X,noisefun,y,s2)
%GPLITE_NOISEFUN Noise function for lite Gaussian Process regression.
%   SN2 = GPLITE_NOISEFUN(HYP,X,NOISEFUN) computes the GP noise function
%   NOISEFUN, that is the variance of observation noise evaluated at test 
%   points X. HYP is a single column vector of noise function 
%   hyperparameters. NOISEFUN is a numeric array whose elements specify 
%   features of the noise function, as follows:
%
%                       FEATURE DESCRIPTION           EXTRA HYPERPARAMETERS
%
%      NOISEFUN(1) represents base constant noise
%      0                no constant noise                          0
%      1                constant noise                             1
%
%      NOISEFUN(2) represents input-dependent provided noise
%      0                ignore provided noise                      0
%      1                use provided noise as is                   0
%      2                scale uncertainty in provided noise        1
%
%      NOISEFUN(3) represents approximate output-dependent noise
%      0                no output-dependent noise                  0
%      1                rectified linear output-dependent noise    2
%
%   The total noise variance is obtained by summing the independent 
%   contribution of each noise feature (if present).
%
%   SN2 = GPLITE_NOISEFUN(HYP,X,NOISEFUN,[],S2) also takes as input a N-by-1
%   array S2 of estimated noise variance associated with each training input 
%   vector in X (used only if NOISEFUN(2) is different than 0).
%  
%   [SN2,DSN2] = GPLITE_NOISEFUN(HYP,X,NOISEFUN) also computes the gradient 
%   DSN2 with respect to GP hyperparameters. If the noise is input or output
%   dependent, DSN2 is a N-by-NNOISE matrix, where each row represents the 
%   gradient with respect to noise hyperparameters for a given training
%   input. Otherwise, DSN2 is a 1-by-NNOISE matrix array.
%
%   NNOISE = GPLITE_NOISEFUN('info',X,NOISEFUN) returns the number of noise 
%   function hyperparameters requested by likelihood function LIKFUN.
%
%   [NNOISE,NOISEINFO] = GPLITE_NOISEFUN([],X,NOISEFUN,Y,S2), where X is 
%   the matrix of training inputs, Y the matrix of training targets, and S2
%   an optional matrix of estimated noise variance, also returns a struct 
%   NOISEINFO with additional information about the noise function
%   hyperparameters, with fields: LB (lower bounds); UB (upper bounds); PLB
%   (plausible lower bounds); PUB (plausible upper bounds); x0 (starting
%   point); noisefun (NOISEFUN numerical identifier).
%
%   See also GPLITE_COVFUN, GPLITE_MEANFUN.

if nargin < 4; y = []; end
if nargin < 5 || isempty(s2); s2 = 0; end

[N,D] = size(X);            % Number of training points and dimension

if numel(noisefun) < 3
    noisefun = [noisefun(:)', zeros(1,3-numel(noisefun))];
end

% Compute number of likelihood function hyperparameters
Nnoise = 0;
switch noisefun(1)
    case 0
    case 1; Nnoise = Nnoise + 1;
    otherwise; Nnoise = NaN;
end
switch noisefun(2)
    case 0
    case 1
    case 2; Nnoise = Nnoise + 1;
    otherwise; Nnoise = NaN;
end
switch noisefun(3)
    case 0
    case 1; Nnoise = Nnoise + 2;
    otherwise; Nnoise = NaN;
end    

if ~isfinite(Nnoise)
    error('gplite_likfun:UnknownLikFun',...
        ['Unknown likelihood function identifier: [' mat2str(noisefun) '].']);
end

% Return number of mean function hyperparameters and additional info
if ischar(hyp)
    sn2 = Nnoise;
    if nargout > 1
        
        ToL = 1e-6;
        Big = exp(3);
        dsn2.LB = -Inf(1,Nnoise);
        dsn2.UB = Inf(1,Nnoise);
        dsn2.PLB = -Inf(1,Nnoise);
        dsn2.PUB = Inf(1,Nnoise);
        dsn2.x0 = NaN(1,Nnoise);

        if numel(y) <= 1; y = [0;1]; end        
        height = max(y) - min(y);
        
        idx = 1;
        
        switch noisefun(1)    % Base constant noise
            case 1  % Constant noise (log standard deviation)
                dsn2.LB(idx) = log(ToL);
                dsn2.UB(idx) = log(height);
                dsn2.PLB(idx) = 0.5*log(ToL);
                dsn2.PUB(idx) = log(std(y));
                dsn2.x0(idx) = log(1e-3);
                idx = idx + 1;
        end
        
        switch noisefun(2)    % User-provided noise
            case 1
            case 2
                dsn2.LB(idx) = log(1e-3);
                dsn2.UB(idx) = log(1e3);
                dsn2.PLB(idx) = log(0.5);
                dsn2.PUB(idx) = log(2);
                dsn2.x0(idx) = log(1);
                idx = idx + 1;
            
        end
        
        switch noisefun(3)    % Output-dependent noise
            case 1
                dsn2.LB(idx) = log(1e-6*D);
                dsn2.UB(idx) = log(1e6*D);
                dsn2.PLB(idx) = log(2*D);
                dsn2.PUB(idx) = log(20*D);
                dsn2.x0(idx) = log(5*D);
                idx = idx + 1;

                dsn2.LB(idx) = log(1e-3);
                dsn2.UB(idx) = log(0.1);
                dsn2.PLB(idx) = log(0.01);
                dsn2.PUB(idx) = log(0.1);
                dsn2.x0(idx) = log(0.1);
                idx = idx + 1;
        end
                
        % Plausible starting point
        idx_nan = isnan(dsn2.x0);
        dsn2.x0(idx_nan) = 0.5*(dsn2.PLB(idx_nan) + dsn2.PUB(idx_nan));
        
        dsn2.noisefun = noisefun;
        
    end
    
    return;
end

[Nhyp,Ns] = size(hyp);      % Hyperparameters and samples

if Nhyp ~= Nnoise
    error('gplite_noisefun:WrongLikHyp', ...
        ['Expected ' num2str(Nnoise) ' noise function hyperparameters, ' num2str(Nhyp) ' passed instead.']);
end
if Ns > 1
    error('gplite_noisefun:nosampling', ...
        'Noise function output is available only for one-sample hyperparameter inputs.');
end

% Compute mean function gradient wrt hyperparameters only if requested
compute_grad = nargout > 1;

if compute_grad     % Allocate space for gradient
    if any(noisefun(2:end)>0)
        dsn2 = zeros(N,Nnoise);
    else
        dsn2 = zeros(1,Nnoise);        
    end
end

% Compute likelihood function as sum of independent noise sources
idx = 1;
switch noisefun(1)
    case 0
        sn2 = eps;
    case 1
        sn2 = exp(2*hyp(idx));
        if compute_grad; dsn2(:,idx) = 2*sn2; end
        idx = idx+1;
end
        
switch noisefun(2)
    case 0        
    case 1
        sn2 = sn2 + s2;        
    case 2
        sn2 = sn2 + exp(hyp(idx))*s2;
        if compute_grad; dsn2(:,idx) = exp(hyp(idx))*s2; end
        idx = idx + 1;
end
    
switch noisefun(3)
    case 1
        if ~isempty(y)
            ymax = max(y);
            deltay = exp(hyp(idx));
            w2 = exp(2*hyp(idx+1));
            zz = max(0, ymax - y - deltay);

            sn2 = sn2 + w2*zz.^2;
            if compute_grad
                dsn2(:,idx) = -deltay*2*w2*(ymax - y - deltay).*(zz>0);
                dsn2(:,idx+1) = 2*w2*zz.^2;
            end
        end
        idx = idx + 2;
end

end