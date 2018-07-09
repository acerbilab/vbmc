function [history,post,algoptions] = infalgo_agp(algo,algoset,probstruct)

% Add algorithm to MATLAB path
BaseFolder = fileparts(mfilename('fullpath'));
AlgoFolder = 'agp';
addpath(genpath([BaseFolder filesep() AlgoFolder]));

algoptions.MaxFunEvals = probstruct.MaxFunEvals;
algoptions.MaxIter = Inf;
algoptions.Nsamples = 5e3;      % Number of samples per iteration
algoptions.GPsamples = 80;

if probstruct.Debug
    algoptions.TrueMean = probstruct.Post.Mean;
    algoptions.TrueCov = probstruct.Post.Cov;
end

% Use prior as proposal function
% algoptions.ProposalFcn = @(X_) exp(infbench_lnprior(X_,probstruct));

% Options from current problem
switch algoset
    case {0,'debug'}; algoset = 'debug'; algoptions.Debug = 1; algoptions.Plot = 'scatter';
    case {1,'base'}; algoset = 'base';           % Use defaults
    case {2,'long'}; algoset = 'long'; algoptions.Nsamples = 2e4;
    case {3,'prop'}; algoset = 'prop'; algoptions.AcqFun = @acqagpprop;
    case {4,'acqg'}; algoset = 'acqg'; algoptions.AcqFun = @acqagpg;
        
    otherwise
        error(['Unknown algorithm setting ''' algoset ''' for algorithm ''' algo '''.']);
end

PLB = probstruct.PLB;
PUB = probstruct.PUB;
LB = probstruct.LB;
UB = probstruct.UB;
x0 = probstruct.InitPoint;
D = size(x0,2);

% Add log prior to function evaluation 
% (AGP is agnostic of the prior)
probstruct.AddLogPrior = true;

algo_timer = tic;
[vbmodel,exitflag,output] = ...
    agp_lite(@(x) infbench_func(x,probstruct),x0,PLB,PUB,algoptions);
TotalTime = toc(algo_timer);

stats = output.stats;

% Remove training data from GPs, too bulky (can be reconstructed)
for i = 1:numel(stats)
     stats(i).gp.X = [];
     stats(i).gp.y = [];
end

history = infbench_func(); % Retrieve history
history.scratch.output = output;
history.TotalTime = TotalTime;
history.Output.stats = stats;

% Store computation results (ignore points discarded after warmup)
history.Output.X = output.X;
history.Output.y = output.y;
post.vbmodel = vbmodel;
post.lnZ = stats(end).lnZ;
post.lnZ_var = stats(end).lnZ_var;
[post.gsKL,post.Mean,post.Cov,post.Mode] = computeStats(stats(end),probstruct);

% Return estimate, SD of the estimate, and gauss-sKL with true moments
Nticks = numel(history.SaveTicks);
N = zeros(1,numel(stats));
for i = 1:numel(stats); N(i) = stats(i).N; end
for iIter = 1:Nticks
    idx = find(N == history.SaveTicks(iIter),1);
    if isempty(idx); continue; end
    
    history.Output.N(iIter) = history.SaveTicks(iIter);
    history.Output.lnZs(iIter) = stats(idx).lnZ;
    history.Output.lnZs_var(iIter) = stats(idx).lnZ_var;
    [gsKL,Mean,Cov,Mode] = computeStats(stats(idx),probstruct);
    history.Output.Mean(iIter,:) = Mean;
    history.Output.Cov(iIter,:,:) = Cov;
    history.Output.gsKL(iIter) = gsKL;
    history.Output.Mode(iIter,:) = Mode;    
end

% Remove vbmodel from stats
for i = 1:numel(history.Output.stats)
    history.Output.stats(i).vbmodel = [];
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [gsKL,Mean,Cov,Mode] = computeStats(stats,probstruct)
%COMPUTE_STATS Compute additional statistics.
    
vbmodel = stats.vbmodel;

% Compute Gaussianized symmetric KL-divergence with ground truth
Ns_moments = 1e6;   Nb = 20;
xx = [];
for i = 1:Nb; xx = [xx; vbgmmrnd(vbmodel,Ns_moments/Nb)']; end
Mean = mean(xx,1);
Cov = cov(xx);
[kl1,kl2] = mvnkl(Mean,Cov,probstruct.Post.Mean,probstruct.Post.Cov);
gsKL = 0.5*(kl1 + kl2);

% Compute mode
opts = optimoptions('fminunc','GradObj','off','Display','off');
Mode = fminunc(@(x) -vbgmmpdf(vbmodel,x')',stats.mode,opts);

end