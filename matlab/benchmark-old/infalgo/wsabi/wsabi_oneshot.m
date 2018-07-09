function [ log_mu, log_Var, clktime, xxIter, loglIter, hyp ] = wsabi_oneshot( ...
            method,         ... 1) 'L' or 'M' wsabi method
            priorMu,        ... 2) Gaussian prior mean, 1 x D.
            priorVar,       ... 3) Gaussian prior covariance, D x D.
            kernelVar,      ... 4) Initial input length scales, D x D.
            lambda,         ... 5) Initial output length scale.
            alpha,          ... 6) Alpha offset fraction, as in paper.
            X,              ... 7) N-by-D matrix of training inputs.
            logliks         ... 8) Vector of log-likelihood values at X0.
            )
        
% Output structures:
% log_mu:   log of the integral posterior mean.
% log_var:  log of the integral posterior variance.
% clktime:  vector of times per iteration, may want to cumulative sum.
% xxIter:   numSamples x D array of sample locations used to build model.
% loglIter: numSamples x 1 vector of log likelihood fcn evaluations.
% hyp:      integral hyperparameters.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

method = upper(method(1));
if method ~= 'L' && method ~= 'M'
    error('wsabi:UnknownMethod', ...
        'Allowed values for METHOD are (L)inearized and (M)oment-matching.');
end

% Relabel prior mean and covariance for brevity of code.
bb          = priorMu;
BB          = diag(priorVar)';

% Relabel input hyperparameters for brevity of code.
VV          = diag(kernelVar)';

jitterNoise = 1e-6;     % Jitter on the GP model over the log likelihood.
numEigs     = inf;      % If trying to use Nystrom ** NOT RECOMMENDED **

dim         = length(bb);   % Dimensionality of integral.

% Limit absolute range of likelihood model hyperparameters for stability.
hypLims     = 30*ones(1,dim+1); 

% Allocate Storage
loglHatD_0_tmp  = zeros(size(logliks));
hyp             = zeros(1,1+dim);

% Minimiser options (fmincon for hyperparameters)
options1                        = optimset('fmincon');
options1.Display                = 'none';
options1.GradObj                = 'off';
options1.Algorithm              = 'active-set';
options1.TolX                   = 1e-5;
options1.TolFun                 = 1e-5;
options1.MaxTime                = 0.5;
options1.MaxFunEvals            = 50;
%options1.UseParallel           = 'always';
options1.AlwaysHonorConstraints = 'true';

currNumSamples = size(X,1);


for t = 1:1
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Pre-process samples -- i.e. convert to log space etc.
    
    % Get batch of samples & variables from stack.
    tmpT            = cputime; 
    xxIter          = X; % Curr samples
    
    % Call loglik handle for latest sample.
    loglHatD_0_tmp = logliks;
    
    % Find the max in log space.                                       
    logscaling   = max(loglHatD_0_tmp);
    
    % Scale batch by max, and exponentiate.
    lHatD_0_tmp = exp(loglHatD_0_tmp - logscaling);
  
    % Evaluate the offset, alpha fraction of minimum value seen.
    aa      = alpha * min( lHatD_0_tmp );
    
    % Transform into sqrt space.
    lHatD = sqrt(abs(lHatD_0_tmp - aa)*2);
         
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ML-II On GP Likelihood model hyperparameters
    
    hyp(1)          = log( lambda );
    hyp(2:end)      = log(VV);
    
    if currNumSamples < numEigs + 1
        hypLogLik = @(x) logLikGPDim(xxIter, lHatD, x);
    else
        hypLogLik = @(x) logLikGPDimNystrom(xxIter, lHatD, x, numEigs);
    end
    [hyp] = fmincon(hypLogLik, ...
                     hyp,[],[],[],[],-hypLims,hypLims,[],options1);
    
    lambda          = exp(hyp(1));
    VV              = exp(hyp(2:end));
    
    % Scale samples by input length scales.
    xxIterScaled    = xxIter .* repmat(sqrt(1./VV),currNumSamples,1);
    
    % Squared distance matrix
    dist2           = pdist2_squared_fast(xxIterScaled, xxIterScaled);
    
    % Evaluate Gram matrix
    Kxx = lambda.^2 * (1/(prod(2*pi*VV).^0.5)) * exp(-0.5*dist2);
    Kxx = Kxx + ...
              lambda.^2*(1/(prod(2*pi*VV).^0.5))*jitterNoise*eye(size(Kxx));
    Kxx = Kxx/2 + Kxx'/2; % Make sure symmetric for stability.
    
    % Invert Gram matrix.
    if currNumSamples < numEigs + 1 
        invKxx = Kxx \ eye(size(Kxx));
    else % If using nystrom
        idx = randperm( length(xxIter(:,1)) );
        xxuScaled = xxIterScaled( idx(1:numEigs), : );
        xxsScaled = xxIterScaled;
        
        AA1 = pdist2_squared_fast(xxuScaled,xxuScaled);
        AA2 = pdist2_squared_fast(xxsScaled,xxuScaled);
        
        Kuu = lambda^2 * 1/sqrt(det(2*pi*VV)) * (exp( -0.5 * AA1 ) + ...
              jitterNoise*eye(size(AA1)));
          
        Ksu = lambda^2 * 1/sqrt(det(2*pi*VV)) * exp( -0.5 * AA2 );
        
        [eVec, eVal] = eig(Kuu);
        
        eVec    = Ksu * ...
                repmat(sqrt(numEigs)./diag(eVal)',length(eVal(:,1)),1) ...
                .* eVec;
            
        eVal    = eVal / numEigs;
        
        Z       = jitterNoise * diag(1./diag(eVal)) + eVec'*eVec;
        invKxx  = (1./jitterNoise)*(eye(currNumSamples)-eVec*(Z \ eVec'));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Expected value of integral:
    ww              = invKxx * lHatD;
    Yvar            = (VV.*VV + 2*VV.*BB)./VV; 
    postProd        = ww*ww';
    
    xx2sq           = xxIter .* repmat(sqrt(1./Yvar),currNumSamples,1);
    bbch            = bb .* sqrt(1./Yvar);
    
    xx2sqFin        = pdist2_squared_fast(xx2sq,bbch);
    xxIterScaled2   = xxIter .* repmat(sqrt(BB./(VV.*Yvar)),currNumSamples,1);
    
    dist4           = pdist2_squared_fast(xxIterScaled2,xxIterScaled2);
    
    if method == 'M'
        % Sigma^2 term:
        sig2t = - ... 
                lambda^4 * (1 / prod(4*pi^2*((VV.*VV + 2*VV.*BB)))^0.5) * ...
                 exp(-0.5 * (pdist2(xx2sqFin,-xx2sqFin) + dist4)) .* invKxx;
    else
        sig2t = 0;
    end
    
    YY              = lambda^4 * ... 
                    (1 / prod(4*pi^2*((VV.*VV + 2*VV.*BB)))^0.5) * ...
                    exp(-0.5 * (pdist2(xx2sqFin,-xx2sqFin) + dist4)) .* ...
                    postProd + sig2t;
    
    % Mean of the integral at iteration 't', before scaling back up:
    mu = aa + 0.5*sum(YY(:));
    if method == 'M'
        mu = mu + 0.5*lambda.^2 * (1/(prod(2*pi*VV)^0.5));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Variance of the integral, before scaling back up:
    
    %---------------- Tmp Vars to calculate first term -------------------%
        
    GG_coeff = lambda^6 * 1/prod(16*pi^4*(VV.*VV + 2*VV.*BB).*VV.*BB.* ...
               (((VV.*VV + 2*VV.*BB)./BB) + ...
               ((VV.*VV + 2*VV.*BB)./VV) + VV + BB))^0.5 * ...
               prod(2*pi*(VV.*VV + 2*VV.*BB))^0.5;
    
    tmpVar   = ((VV.*VV + 2*VV.*BB).*(Yvar + Yvar.*(VV./BB + VV + BB)));
    
    xx3sq    = xxIter .* ...
               repmat(sqrt((Yvar.*VV)./tmpVar),currNumSamples,1);
    bb3ch    = bb .* sqrt((Yvar.*VV)./tmpVar);
    
    xx2sqGGlh = xx2sqFin + pdist2_squared_fast(xx3sq,bb3ch);
    
    xx4sq    = xxIter .* ...
             repmat(sqrt((VV.*Yvar)./tmpVar),currNumSamples,1);
    bb4ch    = bb .* sqrt((VV.*Yvar)./tmpVar);
    
    xx5sq    = xxIter .* ...
             repmat(sqrt((VV.*Yvar.^2)./(tmpVar.*BB)),currNumSamples,1);
    bb5ch    = bb .* sqrt((VV.*Yvar.^2)./(tmpVar.*BB));
    
    xx2sqGGrh       = pdist2_squared_fast(xx4sq,bb4ch) + ...
                      pdist2_squared_fast(xx5sq,bb5ch);
                  
    xxIterScaled3   = xxIter .* ...
                      repmat(sqrt(Yvar.*BB./tmpVar),currNumSamples,1);
                  
    dist4           = pdist2_squared_fast(xxIterScaled3,xxIterScaled3);
    xxIterScaled4   = xxIter .* ...
                      repmat(sqrt(BB./(Yvar.*VV)),currNumSamples,1);
                  
    dist5           = pdist2_squared_fast(xxIterScaled4,xxIterScaled4);
    
    
    GG      = GG_coeff * postProd .* ...
            exp(-0.5*(pdist2(xx2sqGGlh, -xx2sqGGrh) + dist4));
     
    YY2     = lambda^4 * (1 / prod(4*pi^2*((VV.*VV + 2*VV.*BB)))^0.5) * ...
            exp(-0.5 * (pdist2(xx2sqFin,-xx2sqFin) + dist5)) .* ...
            repmat(ww',length(ww),1);
    
    Var  = (sum(sum(GG)) - sum(YY2,2)'*(invKxx * sum(YY2,2)));
    
    if method == 'M'
        %---------------- Tmp Vars to calculate second term ------------------%

        tmp_2_mainvar = (BB.*(VV.*VV + 2*VV.*BB) + (((VV.*VV)./BB) + 2*VV) +...
                        (((VV.*VV+2*VV.*BB).*(VV.*VV+2*VV.*BB))./(VV)) + ...
                        VV.*(VV.*VV + 2*VV.*BB));
        tmp_2_coeff   = lambda^8 * 1/prod(8*pi^3*(VV.*VV+2*VV.*BB))^0.5 * ...
                        prod((VV+2*BB).*(VV.*VV./BB+2*VV))^0.5 * ...
                        1/prod(tmp_2_mainvar)^0.5;

        ScaledVar2_0       = ((VV)./(VV.*VV+2*VV.*BB));
        xxIterScaledVar2_0 = xxIter .* repmat(sqrt(ScaledVar2_0),currNumSamples,1);
        bbScaledVar2_0     = bb .* sqrt(ScaledVar2_0);
        distVar2_0         = pdist2_squared_fast(xxIterScaledVar2_0, bbScaledVar2_0);     

        ScaledVar2_1       = ((VV.*VV + 2*VV.*BB)./(tmp_2_mainvar));
        xxIterScaledVar2_1 = xxIter .* repmat(sqrt(ScaledVar2_1),currNumSamples,1);
        bbScaledVar2_1     = bb .* sqrt(ScaledVar2_1);
        distVar2_1         = pdist2_squared_fast(xxIterScaledVar2_1, bbScaledVar2_1);

        ScaledVar2_2       = ((VV.*VV + 2*VV.*BB).*(VV.*VV + 2*VV.*BB))./(tmp_2_mainvar.*(VV.*BB));
        xxIterScaledVar2_2 = xxIter .* repmat(sqrt(ScaledVar2_2),currNumSamples,1);
        bbScaledVar2_2     = bb .* sqrt(ScaledVar2_2);
        distVar2_2         = pdist2_squared_fast(xxIterScaledVar2_2, bbScaledVar2_2);

        ScaledVar2_3       = ScaledVar2_1;
        xxIterScaledVar2_3 = xxIter .* repmat(sqrt(ScaledVar2_3),currNumSamples,1);
        bbScaledVar2_3     = bb .* sqrt(ScaledVar2_3);
        distVar2_3         = pdist2_squared_fast(xxIterScaledVar2_3, bbScaledVar2_3);

        ScaledVar2_4       = ((VV.*BB)./(tmp_2_mainvar));
        xxIterScaledVar2_4 = xxIter .* repmat(sqrt(ScaledVar2_4),currNumSamples,1);
        bbScaledVar2_4     = bb .* sqrt(ScaledVar2_4);
        distVar2_4         = pdist2_squared_fast(xxIterScaledVar2_4, bbScaledVar2_4);

        distVar2lh         = distVar2_1 + distVar2_2;
        distVar2rh         = distVar2_0 + distVar2_3 + distVar2_4;
        distVar2           = pdist2(distVar2lh,-distVar2rh);  

        %---------------- Tmp Vars to calculate third term -------------------%

        tmp_3_mainvar      = VV.*VV + 2*VV.*BB;
        tmp_3_coeff        = lambda^12 * 1/prod(4*pi^2*tmp_3_mainvar);

        ScaledVar3_0       = (VV./((VV.*VV+2*VV.*BB)));
        xxIterScaledVar3_0 = xxIter .* repmat(sqrt(ScaledVar3_0),currNumSamples,1);
        bbScaledVar3_0     = bb .* sqrt(ScaledVar3_0);
        distVar3_0         = pdist2_squared_fast(xxIterScaledVar3_0, bbScaledVar3_0);    

        ScaledVar3_1       = (VV./(VV.*VV+2*VV.*BB));
        xxIterScaledVar3_1 = xxIter .* repmat(sqrt(ScaledVar3_1),currNumSamples,1);
        bbScaledVar3_1     = bb .* sqrt(ScaledVar3_1);
        distVar3_1         = pdist2_squared_fast(xxIterScaledVar3_1, bbScaledVar3_1);

        ScaledVar3_2       = (BB./(VV.*VV+2*VV.*BB));
        xxIterScaledVar3_2 = xxIter .* repmat(sqrt(ScaledVar3_2),currNumSamples,1);
        distVar3_2         = pdist2_squared_fast(xxIterScaledVar3_2, xxIterScaledVar3_2);

        ScaledVar3_3       = (BB./(VV.*VV+2*VV.*BB));
        xxIterScaledVar3_3 = xxIter .* repmat(sqrt(ScaledVar3_3),currNumSamples,1);
        distVar3_3         = pdist2_squared_fast(xxIterScaledVar3_3, xxIterScaledVar3_3);

        ScaledVar3_4       = (VV./(VV.*VV+2*VV.*BB));
        xxIterScaledVar3_4 = xxIter .* repmat(sqrt(ScaledVar3_4),currNumSamples,1);
        bbScaledVar3_4     = bb .* sqrt(ScaledVar3_4);
        distVar3_4         = pdist2_squared_fast(xxIterScaledVar3_4, bbScaledVar3_4);

        ScaledVar3_5       = (VV./(VV.*VV+2*VV.*BB));
        xxIterScaledVar3_5 = xxIter .* repmat(sqrt(ScaledVar3_5),currNumSamples,1);
        bbScaledVar3_5     = bb .* sqrt(ScaledVar3_5);
        distVar3_5         = pdist2_squared_fast(xxIterScaledVar3_5, bbScaledVar3_5);

        %----------------- Combine terms to get total var --------------------%

        tmp_1 = lambda.^4/prod(8*pi^2*VV.*(0.5*(VV+2*BB)+BB))^0.5;

        tmp_2 = invKxx .* (tmp_2_coeff * exp(-0.5*distVar2));

        tmp_3 = tmp_3_coeff * exp(-0.5*distVar3_0)' * ...
                (invKxx.*exp(-0.5*distVar3_2))* exp(-0.5*distVar3_1) * ...
                exp(-0.5*distVar3_4)'*(invKxx.*exp(-0.5*distVar3_3)) * ...
                exp(-0.5*distVar3_5);

        Var = Var + 0.5*(tmp_1 - 2*sum(tmp_2(:)) + sum(tmp_3(:)));  
    end
        
    clktime = cputime - tmpT;
    
end

log_mu  = log(mu) + logscaling;
log_Var = log(Var) + 2*logscaling;
loglIter = loglHatD_0_tmp;

end
