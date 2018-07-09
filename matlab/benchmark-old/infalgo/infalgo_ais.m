function [history,post,algoptions] = infalgo_ais(algo,algoset,probstruct)

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
priorMean = probstruct.PriorMean;
priorVar = probstruct.PriorVar;

diam = probstruct.PUB - probstruct.PLB;

% Do not add log prior to function evaluation, already part of AIS 
probstruct.AddLogPrior = false;

prior.mean = priorMean;
prior.covariance = diag(priorVar);

N = probstruct.SaveTicks;
Niter = numel(probstruct.SaveTicks);
tot_num_samples = probstruct.SaveTicks;

aistimes = zeros(1,Niter);
X = []; y = [];
mu = zeros(1,Niter);
vvar = zeros(1,Niter);
for i = 1:Niter
    if i == Niter; algo_timer = tic; end
    opt.num_samples = ceil(tot_num_samples(i)) - 1;
    [mean_lnZ_tmp, var_lnZ_tmp, ~, ~, aistimes_tmp, stats] = ...
        ais_mh(@(x_) infbench_func(x_,probstruct), prior, opt);
    mu(i) = mean_lnZ_tmp(end);
    vvar(i) = var_lnZ_tmp(end);
    aistimes(i) = sum(aistimes_tmp);
    X_tmp{i} = stats.all_samples.locations;
    y_tmp{i} = stats.all_samples.logliks;
    if ~probstruct.AddLogPrior      % y stores log posteriors, so add prior now
        lnp = infbench_lnprior(X_tmp{i},probstruct);
        y_tmp{i} = y_tmp{i} + lnp;
    end    
end

TotalTime = toc(algo_timer);

[history,post] = ...
    StoreAlgoResults(probstruct,[],Niter,X_tmp{end},y_tmp{end},mu,vvar,X_tmp,y_tmp,TotalTime);

history.Output.tt = aistimes;
history.Output.stats = stats;

% [X_train,idx] = unique(history.Output.X,'rows');
% y_train = history.Output.y(idx);

end