function [history,post,algoptions] = infalgo_smc(algo,algoset,probstruct)

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

% Do not add log prior to function evaluation, already passed to BMC 
probstruct.AddLogPrior = false;

algo_timer = tic;

% Simple Monte Carlo from prior
SaveTicks = probstruct.SaveTicks(probstruct.SaveTicks <= algoptions.MaxFunEvals);

Nmax = max(SaveTicks);
Niter = numel(SaveTicks);

priorMean = probstruct.PriorMean;
priorVar = probstruct.PriorVar;

% Generate random samples from prior
X = bsxfun(@plus, ...
    bsxfun(@times,randn(Nmax,D),sqrt(priorVar)),...
    priorMean);

% Compute log likelihoods at points
y = zeros(Nmax,1);
for i = 1:Nmax
    y(i) = infbench_func(X(i,:),probstruct);
end

mu = zeros(1,Niter);
ln_var = zeros(1,Niter);
for i = 1:Niter
    N_train = SaveTicks(i);
    y_train = y(1:N_train);
    scaling = max(y_train);
    y_train = exp(y_train - scaling);
    
    mu(i) = log(mean(y_train)) + scaling;
    ln_var(i) = log(var(y_train)/(N_train-1)) + 2*scaling;    
end
TotalTime = toc(algo_timer);

vvar = max(real(exp(ln_var)),0);

post = []; Xiter = []; yiter = [];
[history,post] = ...
    StoreAlgoResults(probstruct,post,Niter,X,y,mu,vvar,Xiter,yiter,TotalTime);

end