function [history,post,algoptions] = infalgo_ais(algo,algoset,probstruct)

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

history = infbench_func(); % Retrieve history
% history.scratch.output = output;
history.TotalTime = TotalTime;
history.Output.X = X_tmp{end};
history.Output.y = y_tmp{end};
history.Output.tt = aistimes;
history.Output.stats = stats;

% Store computation results
post.lnZ = mu(end);
post.lnZ_var = vvar(end);
[X_train,idx] = unique(history.Output.X,'rows');
y_train = history.Output.y(idx);
[post.gsKL,post.Mean,post.Cov,post.Mode] = computeStats(X_train,y_train,probstruct);

% Return estimate, SD of the estimate, and gauss-sKL with true moments
history.Output.N = N;
history.Output.lnZs = mu;
history.Output.lnZs_var = vvar;
for iIter = 1:Niter
    X_train = X_tmp{iIter};
    y_train = y_tmp{iIter};
    [gsKL,~,~,Mode] = computeStats(X_train,y_train,probstruct);
    history.Output.gsKL(iIter) = gsKL;
    history.Output.Mode(iIter,:) = Mode;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [gsKL,Mean,Cov,Mode] = computeStats(X,y,probstruct)
%COMPUTE_STATS Compute additional statistics.
    
% Compute Gaussianized symmetric KL-divergence with ground truth
gp.X = X;
gp.y = y;
gp.meanfun = 4; % Negative quadratic mean fcn

Ns_moments = 2e4;
xx = gplite_sample(gp,Ns_moments);
Mean = mean(xx,1);
Cov = cov(xx);
[kl1,kl2] = mvnkl(Mean,Cov,probstruct.Mean,probstruct.Cov);
gsKL = 0.5*(kl1 + kl2);

Mode = gplite_fmin(gp,[],1);    % Max flag - finds maximum

end