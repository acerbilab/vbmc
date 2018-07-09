function [history,post,algoptions] = infalgo_bmc(algo,algoset,probstruct)

% Add algorithm to MATLAB path
BaseFolder = fileparts(mfilename('fullpath'));
AlgoFolder = 'wsabi';
addpath(genpath([BaseFolder filesep() AlgoFolder]));

algoptions.MaxFunEvals = probstruct.MaxFunEvals;

% Options from current problem
switch algoset
    case {0,'debug'}; algoset = 'debug'; algoptions.Debug = 1; algoptions.Plot = 'scatter';
    case {1,'base'}; algoset = 'base';           % Use defaults        
    otherwise
        error(['Unknown algorithm setting ''' algoset ''' for algorithm ''' algo '''.']);
end

PLB = probstruct.PLB;
PUB = probstruct.PUB;
LB = probstruct.LB;
UB = probstruct.UB;
x0 = probstruct.InitPoint;
D = size(x0,2);

diam = probstruct.PUB - probstruct.PLB;

kernelCov = diag(diam/10);     % Input length scales for GP likelihood model
lambda = 1;                     % Ouput length scale for GP likelihood model

% Do not add log prior to function evaluation, already passed to BMC 
probstruct.AddLogPrior = false;

algo_timer = tic;

% Initialize Bayesian Monte Carlo with samples from prior
SaveTicks = probstruct.SaveTicks(probstruct.SaveTicks <= algoptions.MaxFunEvals);

Nmax = max(SaveTicks);
Niter = numel(SaveTicks);

priorMean = probstruct.PriorMean;
priorVar = probstruct.PriorVar;

% Generate random samples from prior
X = [x0(1,:); ...
        bsxfun(@plus, ...
        bsxfun(@times,randn(Nmax-1,D),sqrt(priorVar)),...
        priorMean)];

% Compute log likelihoods at points
y = zeros(Nmax,1);
for i = 1:Nmax
    y(i) = infbench_func(X(i,:),probstruct);
end

mu = zeros(1,Niter);
ln_var = zeros(1,Niter);
for i = 1:Niter
    X_train = X(1:SaveTicks(i),:);
    [mu(i), ln_var(i), kernelCov, lambda] = bq([], ...
        priorMean, diag(priorVar), kernelCov, lambda, [], ...
        X_train, y(1:SaveTicks(i)));
end
TotalTime = toc(algo_timer);

vvar = max(real(exp(ln_var)),0);

[history,post] = ...
    StoreAlgoResults(probstruct,[],Niter,X,y,mu,vvar,[],[],TotalTime);

end