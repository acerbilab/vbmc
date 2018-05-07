function [history,post,algoptions] = infalgo_wsabi(algo,algoset,probstruct)

algoptions.Method = 'L';    % Default is WSABI-L
algoptions.Alpha = 0.8;     % Fractional offset, as in paper.

algoptions.MaxFunEvals = probstruct.MaxFunEvals;

% Options from current problem
switch algoset
    case {0,'debug'}; algoset = 'debug'; algoptions.Debug = 1; algoptions.Plot = 'scatter';
    case {1,'base'}; algoset = 'base';           % Use defaults
    case {2,'mm'}; algoset = 'mm'; algoptions.Method = 'M';
        
    otherwise
        error(['Unknown algorithm setting ''' algoset ''' for algorithm ''' algo '''.']);
end

switch upper(algoptions.Method(1))
    case 'L'; wsabi_fun = @wsabiL;
    case 'M'; wsabi_fun = @wsabiM;
    otherwise; error('Unknown METHOD for WSABI inference algorithm.');
end

% % Increase base noise with noisy functions
% if ~isempty(probstruct.Noise) || probstruct.IntrinsicNoisy
%     algoptions.UncertaintyHandling = 'on';
%     NoiseEstimate = probstruct.NoiseEstimate;
%     if isempty(NoiseEstimate); NoiseEstimate = 1; end    
%     algoptions.NoiseSize = NoiseEstimate(1);
% else
%     algoptions.UncertaintyHandling = 'off';
% end

PLB = probstruct.PLB;
PUB = probstruct.PUB;
LB = probstruct.LB;
UB = probstruct.UB;
x0 = probstruct.InitPoint;
D = size(x0,2);

diam = probstruct.PUB - probstruct.PLB;

kernelCov = diag(diam/10);     % Input length scales for GP likelihood model
lambda = 1;                     % Ouput length scale for GP likelihood model

range = [PLB - 3*diam; PUB + 3*diam];

% Do not add log prior to function evaluation, already passed to WSABI 
probstruct.AddLogPrior = false;

printing = 1;

algo_timer = tic;
[mu, ln_var, tt, X, hyp] = ...
    wsabi_fun(range,probstruct.PriorMean,diag(probstruct.PriorVar), ...
        kernelCov,lambda,algoptions.Alpha,algoptions.MaxFunEvals+1,...
        @(x) infbench_func(x,probstruct),printing,x0);
TotalTime = toc(algo_timer);

history = infbench_func(); % Retrieve history
% history.scratch.output = output;
history.TotalTime = TotalTime;
history.Output.stats.X = X;
history.Output.stats.tt = tt;
history.Output.stats.hyp = hyp;

% Store computation results
post.lZ = mu(end);
post.lZ_var = exp(ln_var(end));
% [post.gsKL,post.Mean,post.Cov] = compute_gsKL(vp,probstruct);

% Return estimate, SD of the estimate, and gauss-sKL with true moments
Nticks = numel(history.SaveTicks);
Nmax = numel(mu);
for iIter = 1:Nticks
    idx = history.SaveTicks(iIter);
    if idx > Nmax; break; end
    
    history.Output.N(iIter) = history.SaveTicks(iIter);
    history.Output.lZs(iIter) = mu(idx);
    history.Output.lZs_var(iIter) = exp(ln_var(idx));
    % history.Output.gsKL(iIter) = compute_gsKL(stats.vp(idx),probstruct);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [gsKL,Mean,Cov] = compute_gsKL(vp,probstruct)
%COMPUTE_GSKL Compute Gaussianized symmetric KL divergence with ground truth.
    
Ns_moments = 1e6;
xx = vbmc_rnd(Ns_moments,vp,1,1);
Mean = mean(xx,1);
Cov = cov(xx);
[kl1,kl2] = mvnkl(Mean,Cov,probstruct.Mean,probstruct.Cov);
gsKL = 0.5*(kl1 + kl2);

end