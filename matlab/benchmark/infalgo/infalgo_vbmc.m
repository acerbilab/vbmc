function [history,post,algoptions] = infalgo_vbmc(algo,algoset,probstruct)

algoptions = vbmc('all');                   % Get default settings

% VBMC old defaults -- some of these may have changed
algoptions.FunEvalsPerIter = 5;
algoptions.AcqFcn = '@vbmc_acqskl';

algoptions.MinFunEvals = probstruct.MaxFunEvals;
algoptions.MaxFunEvals = probstruct.MaxFunEvals;
algoptions.MinIter = 0;     % No limits on iterations
algoptions.MaxIter = Inf;
algoptions.WarpNonlinear = 'off';   % No nonlinear warping for now

if probstruct.Debug
    algoptions.TrueMean = probstruct.Post.Mean;
    algoptions.TrueCov = probstruct.Post.Cov;
end

% Use prior as proposal function
algoptions.ProposalFcn = @(X_) exp(infbench_lnprior(X_,probstruct));

% Options from current problem
switch algoset
    case {0,'debug'}; algoset = 'debug'; algoptions.Debug = 1; algoptions.Plot = 'scatter';
    case {1,'base'}; algoset = 'base';           % Use defaults
    case {2,'acqkl'}; algoset = 'acqkl'; algoptions.AcqFcn = '@vbmc_acqkl';
    case {3,'acqvar'}; algoset = 'acqvar'; algoptions.AcqFcn = '@vbmc_acqvar';
    case {4,'acqgev'}; algoset = 'acqgev'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqGEV';
    case {5,'acqcmaes'}; algoset = 'acqcmaes'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqGEV'; algoptions.SearchCMAES = 1;
    case {6,'acqprop'}; algoset = 'acqprop'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqprop'; algoptions.SearchCMAES = 1;
    case {11,'betazero'}; algoset = 'betazero'; algoptions.ELCBOWeight = 0;
    case {12,'betatiny'}; algoset = 'betatiny'; algoptions.ELCBOWeight = 0.1;
        
    otherwise
        error(['Unknown algorithm setting ''' algoset ''' for algorithm ''' algo '''.']);
end

% Increase base noise with noisy functions
if ~isempty(probstruct.Noise) || probstruct.IntrinsicNoisy
    algoptions.UncertaintyHandling = 'on';
    NoiseEstimate = probstruct.NoiseEstimate;
    if isempty(NoiseEstimate); NoiseEstimate = 1; end    
    algoptions.NoiseSize = NoiseEstimate(1);
else
    algoptions.UncertaintyHandling = 'off';
end

PLB = probstruct.PLB;
PUB = probstruct.PUB;
LB = probstruct.LB;
UB = probstruct.UB;
x0 = probstruct.InitPoint;
D = size(x0,2);

% Add log prior to function evaluation 
% (the current version of VBMC is agnostic of the prior)
probstruct.AddLogPrior = true;

algo_timer = tic;
[vp,elbo,elbo_sd,exitflag,output,stats] = ...
    vbmc(@(x) infbench_func(x,probstruct),x0,LB,UB,PLB,PUB,algoptions);
TotalTime = toc(algo_timer);

% Remove training data from GPs, too bulky (can be reconstructed)
for i = 1:numel(stats.gp)
    stats.gp(i).X = [];
    stats.gp(i).y = [];
end

history = infbench_func(); % Retrieve history
history.scratch.output = output;
history.TotalTime = TotalTime;
history.Output.stats = stats;

% Store computation results (ignore points discarded after warmup)
history.Output.X = output.X_orig(output.X_flag,:);
history.Output.y = output.y_orig(output.X_flag);
post.lnZ = elbo;
post.lnZ_var = elbo_sd^2;
[post.gsKL,post.Mean,post.Cov,post.Mode] = computeStats(vp,probstruct);

% Return estimate, SD of the estimate, and gauss-sKL with true moments
Nticks = numel(history.SaveTicks);
for iIter = 1:Nticks
    idx = find(stats.N == history.SaveTicks(iIter),1);
    if isempty(idx); continue; end
    
    history.Output.N(iIter) = history.SaveTicks(iIter);
    history.Output.lnZs(iIter) = stats.elbo(idx);
    history.Output.lnZs_var(iIter) = stats.elboSD(idx)^2;
    [gsKL,~,~,Mode] = computeStats(stats.vp(idx),probstruct);
    history.Output.gsKL(iIter) = gsKL;
    history.Output.Mode(iIter,:) = Mode;    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [gsKL,Mean,Cov,Mode] = computeStats(vp,probstruct)
%COMPUTE_STATS Compute additional statistics.
    
% Compute Gaussianized symmetric KL-divergence with ground truth
Ns_moments = 1e6;
xx = vbmc_rnd(Ns_moments,vp,1,1);
Mean = mean(xx,1);
Cov = cov(xx);
[kl1,kl2] = mvnkl(Mean,Cov,probstruct.Post.Mean,probstruct.Post.Cov);
gsKL = 0.5*(kl1 + kl2);

% Compute mode
Mode = vbmc_mode(vp,1);

end