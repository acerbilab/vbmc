function [ log_mu, log_Var, clktime, xxIter, hyp ] = wsabiL( ...
            range,          ... 1) 2 x D matrix, lower bnd top row.
            priorMu,        ... 2) Gaussian prior mean, D x 1.
            priorVar,       ... 3) Gaussian prior covariance, D x D.
            kernelVar,      ... 4) Initial input length scales, D x D.
            lambda,         ... 5) Initial output length scale.
            alpha,          ... 6) Alpha offset fraction, as in paper.
            numSamples,     ... 7) Number of BBQ samples to run.
            loglikhandle,   ... 8) Handle to log-likelihood function. 
            printing,       ... 9) If true, print intermediate output.
            x0)             ... 10) (optional) Use as starting point.
        
% Output structures:
% log_mu:   log of the integral posterior mean.
% log_var:  log of the integral posterior variance.
% clktime:  vector of times per iteration, may want to cumulative sum.
% xxIter:   numSamples x D array of sample locations used to build model.
% hyp:      integral hyperparameters.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 10; x0 = []; end

% Relabel prior mean and covariance for brevity of code.
bb          = priorMu;
BB          = diag(priorVar)';

% Relabel input hyperparameters for brevity of code.
VV          = diag(kernelVar)';

jitterNoise = 1e-6;     % Jitter on the GP model over the log likelihood.
numEigs     = inf;      % If trying to use Nystrom ** NOT RECOMMENDED **
hypOptEvery = 1;        % 1 => optimise hyperparameters every iteration.

dim         = length(bb);   % Dimensionality of integral.

% Limit absolute range of likelihood model hyperparameters for stability.
hypLims     = 30*ones(1,dim+1); 

% Allocate Storage
mu              = zeros(numSamples-1,1);
logscaling      = zeros(numSamples-1,1);
Var             = zeros(numSamples-1,1);
clktime         = zeros(numSamples-1,1);
lHatD_0_tmp     = zeros(numSamples,1);
loglHatD_0_tmp  = zeros(size(lHatD_0_tmp));
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

% Minimiser options (fmincon if desired for active sampling)
options2                        = optimset('fmincon');
options2.Display                = 'none';
options2.GradObj                = 'on';
%options2.DerivativeCheck       = 'on';
options2.TolX                   = 1e-5;
options2.TolFun                 = 1e-5;
%options2.MaxTime               = 0.5;
%options2.MaxFunEvals           = 75;
options2.UseParallel            = 'always';
options2.AlwaysHonorConstraints = 'true';

% Minimiser options (CMAES - advised for active sampling)
opts                            = cmaes_modded('defaults');
opts.LBounds                    = range(1,:)';
opts.UBounds                    = range(2,:)';
opts.DispModulo                 = Inf;
opts.DispFinal                  = 'off';
opts.SaveVariables              = 'off';
opts.LogModulo                  = 0;
opts.CMA.active                 = 1;      % Use Active CMA (generally better)
%opts.EvalParallel              = 'on';
%opts.PopSize                   = 100;
%opts.Restarts                  = 1;

% Initial Sample:
xx = zeros(numSamples,dim);
if isempty(x0)
    xx(end,:) = bb+1e-6;    % Prior mean
else
    xx(end,:) = x0+1e-6;
end
currNumSamples = 1;

if printing
    fprintf('Iter:   ');
end

for t = 1:numSamples - 1
    if printing
        if ~mod(t,10)
            prstr = sprintf('Log Current Mean Integral: %g', ...
                            log(mu(t-1)) + logscaling(t-1));
            fprintf(prstr);
            pause(1);
            fprintf(repmat('\b',1,length(prstr)))
        end
        if t > 1
            fprintf(repmat('\b',1,length(num2str(t-1))));
        end
        fprintf('%i',t);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Pre-process new samples -- i.e. convert to log space etc.
    
    % Get batch of samples & variables from stack.
    tmpT            = cputime; 
    xxIter          = xx(numSamples-currNumSamples+1:end,:); % Curr samples
    
    % Call loglik handle for latest sample.
    loglHatD_0_tmp(numSamples-currNumSamples+1)  = ...
                                               loglikhandle( xxIter(1,:) );
    % Find the max in log space.                                       
    logscaling(t)   = max(loglHatD_0_tmp(numSamples-currNumSamples+1:end));
    
    % Scale batch by max, and exponentiate.
    lHatD_0_tmp(numSamples-currNumSamples+1:end) = ... 
      exp(loglHatD_0_tmp(numSamples-currNumSamples+1:end) - logscaling(t));
  
    % Evaluate the offset, alpha fraction of minimum value seen.
    aa      = alpha * min( lHatD_0_tmp(numSamples-currNumSamples+1:end) );
    
    % Transform into sqrt space.
    lHatD = sqrt(abs(lHatD_0_tmp(numSamples-currNumSamples+1:end)- aa)*2);
         
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ML-II On GP Likelihood model hyperparameters
    
    hyp(1)          = log( lambda );
    hyp(2:end)      = log(VV);
    
    if currNumSamples > 3 &&  ~mod(currNumSamples,hypOptEvery)
        if currNumSamples < numEigs + 1
            hypLogLik = @(x) logLikGPDim(xxIter, lHatD, x);
        else
            hypLogLik = @(x) logLikGPDimNystrom(xxIter, lHatD, x, numEigs);
        end
        [hyp] = fmincon(hypLogLik, ...
                         hyp,[],[],[],[],-hypLims,hypLims,[],options1);
    end
    
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
    
    YY              = lambda^4 * ... 
                    (1 / prod(4*pi^2*((VV.*VV + 2*VV.*BB)))^0.5) * ...
                    exp(-0.5 * (pdist2(xx2sqFin,-xx2sqFin) + dist4)) .* ...
                    postProd;
    
    % Mean of the integral at iteration 't', before scaling back up:
    mu(t) = (aa + 0.5*sum(YY(:)));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Variance of the integral, before scaling back up:
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
    
    Var(t)  = (sum(sum(GG)) - sum(YY2,2)'*(invKxx * sum(YY2,2)));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Actively select next sample point:
 
    if rand < 1.1 % Sample starting location for search from prior.
        strtSamp = mvnrnd(bb,diag(BB),1);
    else
        strtSamp = 2*range(2,:).*rand(1,dim) - 50;
    end
    
    % If using local optimiser (fast):
    %EV = @(x) expectedVarL( transp(x), lambda, VV, ... 
    %      lHatD, xxIter, invKxx, jitterNoise, bb, BB ); %Utility function
    %newX = fmincon( EV,  strtSamp,[],[],[],[], ...
    %                range(1,:),range(2,:),[],options2 );
    
    % If using global optimiser (cmaes):
    newX = cmaes_modded( 'expectedVarL', strtSamp', [],opts, lambda, VV, ...
                  lHatD, xxIter, invKxx, jitterNoise, bb, BB);
    newX = newX';
    
    xx(numSamples-currNumSamples,:) = newX;
    
    clktime(t) = cputime - tmpT;
    
    currNumSamples = currNumSamples + 1;

end

fprintf('\n done. \n');
log_mu  = log(mu) + logscaling;
log_Var = log(Var) + 2*logscaling;


end

