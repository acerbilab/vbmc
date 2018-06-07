function [history,post,algoptions] = infalgo_vbmc(algo,algoset,probstruct)

algoptions = vbmc('all');                   % Get default settings

ControlRunFlag = false;     % Do NOT run in control mode

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
    case {10001,'baseluigi'}; algoset = 'baseluigi';           % Use defaults
    case {2,'acqkl'}; algoset = 'acqkl'; algoptions.AcqFcn = '@vbmc_acqkl';
    case {3,'acqvar'}; algoset = 'acqvar'; algoptions.AcqFcn = '@vbmc_acqvar';
    case {4,'acqgev'}; algoset = 'acqgev'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqGEV';
    case {5,'acqcmaes'}; algoset = 'acqcmaes'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqGEV'; algoptions.SearchCMAES = 1;
    case {6,'acqprop'}; algoset = 'acqprop'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqprop'; algoptions.SearchCMAES = 1;
    case {7,'acqprop2'}; algoset = 'acqprop2'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqprop2'; algoptions.SearchCMAES = 1;
    case {8,'acqpropnorot'}; algoset = 'acqpropnorot'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqprop'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0;
    case {9,'acqpropfewrot'}; algoset = 'acqpropfewrot'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqprop'; algoptions.SearchCMAES = 1; algoptions.WarpEpoch = '20 + 10*nvars';
    case {10,'acqprop2norot'}; algoset = 'acqprop2norot'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqprop2'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0;
    case {11,'betazero'}; algoset = 'betazero'; algoptions.ELCBOWeight = 0;
    case {12,'betatiny'}; algoset = 'betatiny'; algoptions.ELCBOWeight = 0.1;
    case {13,'norot'}; algoset = 'norot'; algoptions.WarpRotoScaling = 0;
    case {14,'acqfnorot'}; algoset = 'acqfnorot'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqf'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0;
    case {15,'acqpropfnorot'}; algoset = 'acqpropfnorot'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqpropf'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0;
    case {16,'acqpropf'}; algoset = 'acqpropf'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqpropf'; algoptions.SearchCMAES = 1;
    case {17,'acqpropfnorotbz'}; algoset = 'acqpropfnorotbz'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqpropf'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0; algoptions.ELCBOWeight = 0; algoptions.AlwaysRefitVarPost = 0; algoptions.NSsearch = 2^13;
    case {18,'acqpropfbz'}; algoset = 'acqpropfbz'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqpropf'; algoptions.SearchCMAES = 1; algoptions.ELCBOWeight = 0; algoptions.AlwaysRefitVarPost = 0; algoptions.NSsearch = 2^13;
    case {19,'adaptive'}; algoset = 'adaptive'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqpropf'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0; algoptions.ELCBOWeight = 0; algoptions.AlwaysRefitVarPost = 0; algoptions.NSsearch = 2^13; algoptions.AdaptiveK = 1;
    case {20,'adaptiveless'}; algoset = 'adaptiveless'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqpropf'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0; algoptions.ELCBOWeight = 0; algoptions.AlwaysRefitVarPost = 0; algoptions.NSsearch = 2^13; algoptions.AdaptiveK = 1; algoptions.KfunMax = '@(K) 2*sqrt(K)';
    case {21,'adaptivelessnogp'}; algoset = 'adaptivelessnogp'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqpropf'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0; algoptions.ELCBOWeight = 0; algoptions.AlwaysRefitVarPost = 0; algoptions.NSsearch = 2^13; algoptions.AdaptiveK = 1; algoptions.KfunMax = '@(K) 2*sqrt(K)'; algoptions.NSgpMax = 0;
    case {22,'basenogp'}; algoset = 'basenogp'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqpropf'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0; algoptions.ELCBOWeight = 0; algoptions.AlwaysRefitVarPost = 0; algoptions.NSsearch = 2^13; algoptions.AdaptiveK = 0; algoptions.KfunMax = '@(K) 2*sqrt(K)'; algoptions.NSgpMax = 0;
    case {23,'basekone'}; algoset = 'basekone'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqpropf'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0; algoptions.ELCBOWeight = 0; algoptions.AlwaysRefitVarPost = 0; algoptions.NSsearch = 2^13; algoptions.AdaptiveK = 1; algoptions.KfunMax = '@(K) 2*sqrt(K)'; algoptions.Kwarmup = 1;
    case {24,'acqproponly'}; algoset = 'acqproponly'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqprop'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0; algoptions.ELCBOWeight = 0; algoptions.AlwaysRefitVarPost = 0; algoptions.NSsearch = 2^13; algoptions.AdaptiveK = 1; algoptions.KfunMax = '@(K) 2*sqrt(K)';
    case {25,'acqusonly'}; algoset = 'acqusonly'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqus'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0; algoptions.ELCBOWeight = 0; algoptions.AlwaysRefitVarPost = 0; algoptions.NSsearch = 2^13; algoptions.AdaptiveK = 1; algoptions.KfunMax = '@(K) 2*sqrt(K)';
    case {26,'acqvusonly'}; algoset = 'acqvusonly'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqvus'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0; algoptions.ELCBOWeight = 0; algoptions.AlwaysRefitVarPost = 0; algoptions.NSsearch = 2^13; algoptions.AdaptiveK = 1; algoptions.KfunMax = '@(K) 2*sqrt(K)';
    case {27,'acqpropnoprune'}; algoset = 'acqpropnoprune'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqprop'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0; algoptions.ELCBOWeight = 0; algoptions.AlwaysRefitVarPost = 0; algoptions.NSsearch = 2^13; algoptions.AdaptiveK = 1; algoptions.KfunMax = '@(K) 2*sqrt(K)'; algoptions.WarmupKeepThreshold = Inf;
    case {28,'acqusnoprune'}; algoset = 'acqusnoprune'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqus'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0; algoptions.ELCBOWeight = 0; algoptions.AlwaysRefitVarPost = 0; algoptions.NSsearch = 2^13; algoptions.AdaptiveK = 1; algoptions.KfunMax = '@(K) 2*sqrt(K)'; algoptions.WarmupKeepThreshold = Inf;
    case {100,'acqpropcontrol'}; algoset = 'acqpropcontrol'; algoptions.Nacq = 1; algoptions.SearchAcqFcn = '@vbmc_acqprop'; algoptions.SearchCMAES = 1; algoptions.WarpRotoScaling = 0; algoptions.ELCBOWeight = 0; algoptions.AlwaysRefitVarPost = 0; algoptions.NSsearch = 2^13; algoptions.AdaptiveK = 1; algoptions.KfunMax = '@(K) 2*sqrt(K)'; ControlRunFlag = true;
    case {201,'cheapgp'}; algoset = 'cheapgp'; algoptions.CheapGPRetrain = 1;
    case {202,'cheapgpentsqrtk'}; algoset = 'cheapgpentsqrtk'; algoptions.CheapGPRetrain = 1; algoptions.NSent = @(K) 100*sqrt(K);
    case {203,'cheapgpentsqrtk2'}; algoset = 'cheapgpentsqrtk2'; algoptions.CheapGPRetrain = 1; algoptions.NSent = @(K) 100*sqrt(K); algoptions.NSentFine = @(K) 2^15*sqrt(K);
    case {204,'cheapgpentsqrtknomid'}; algoset = 'cheapgpentsqrtknomid'; algoptions.CheapGPRetrain = 1; algoptions.NSent = @(K) 100*sqrt(K); algoptions.ELCBOmidpoint = 0;
    case {205,'gpthreshmid'}; algoset = 'gpthreshmid'; algoptions.GPRetrainThreshold = 1; algoptions.ELCBOmidpoint = 0;
    case {206,'gpthreshmidextrak'}; algoset = 'gpthreshmidextrak'; algoptions.GPRetrainThreshold = 3; algoptions.ELCBOmidpoint = 0; algoptions.AdaptiveK = @(K) ceil(K/10); algoptions.KfunMax = @(N) N;
    case {207,'gpthresh'}; algoset = 'gpthresh'; algoptions.GPRetrainThreshold = 1;
    case {208,'cheapgpmid'}; algoset = 'cheapgpmid'; algoptions.CheapGPRetrain = 1; algoptions.ELCBOmidpoint = 0; algoptions.GPRetrainThreshold = Inf;
    case {209,'morelbomid'}; algoset = 'morelbomid'; algoptions.ELCBOmidpoint = 0; algoptions.NSelboIncr = 0.5;
    case {210,'cheapgpmidlessprune'}; algoset = 'cheapgpmidlessprune'; algoptions.CheapGPRetrain = 1; algoptions.ELCBOmidpoint = 0; algoptions.GPRetrainThreshold = Inf; algoptions.WarmupKeepThreshold = '50*nvars';
    case {211,'gpthreshmidlessprune'}; algoset = 'gpthreshmid'; algoptions.GPRetrainThreshold = 10; algoptions.ELCBOmidpoint = 0; algoptions.WarmupKeepThreshold = '50*nvars';
    case {212,'gpthreshlessprune'}; algoset = 'gpthreshmid'; algoptions.GPRetrainThreshold = 10; algoptions.WarmupKeepThreshold = '50*nvars';
    case {213,'lessprune'}; algoset = 'lessprune'; algoptions.WarmupKeepThreshold = '25*nvars';
    case {214,'gpthreshfastadapt'}; algoset = 'gpthreshfastadapt'; algoptions.GPRetrainThreshold = 1; algoptions.AdaptiveK = @(K) ceil(K/10); algoptions.KfunMax = '@(N) N';
    case {215,'gpthreshwidths'}; algoset = 'gpthreshwidths'; algoptions.GPRetrainThreshold = 1; algoptions.GPSampleWidths = 5;
    case {216,'gpthreshruncov'}; algoset = 'gpthreshruncov'; algoptions.GPRetrainThreshold = 1; algoptions.GPSampleWidths = 5; algoptions.HypRunWeight = 0.8;
    case {217,'gpthreshruncovzero'}; algoset = 'gpthreshruncov'; algoptions.GPRetrainThreshold = 1; algoptions.GPSampleWidths = 5; algoptions.HypRunWeight = 0;
    case {218,'gpthreshwcov'}; algoset = 'gpthreshwcov'; algoptions.GPRetrainThreshold = 1; algoptions.GPSampleWidths = 5; algoptions.HypRunWeight = 0.9; algoptions.WeightedHypCov = 1;
    case {219,'gpthreshcovsample'}; algoset = 'gpthreshcovsample'; algoptions.GPRetrainThreshold = 1; algoptions.GPSampleWidths = 5; algoptions.HypRunWeight = 0.9; algoptions.WeightedHypCov = 1; algoptions.GPHypSampler = 'covsample'; algoptions.CovSampleThresh = 10;
    case {220,'gpthreshlaplace'}; algoset = 'gpthreshlaplace'; algoptions.GPRetrainThreshold = 1; algoptions.GPSampleWidths = 5; algoptions.HypRunWeight = 0.9; algoptions.WeightedHypCov = 1; algoptions.GPHypSampler = 'laplace';
    case {221,'gpthreshcovsamplezero'}; algoset = 'gpthreshcovsamplezero'; algoptions.GPRetrainThreshold = 1; algoptions.GPSampleWidths = 5; algoptions.HypRunWeight = 0; algoptions.WeightedHypCov = 1; algoptions.GPHypSampler = 'covsample'; algoptions.CovSampleThresh = 10;
                
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

if ~ControlRunFlag
    
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
        [gsKL,Mean,Cov,Mode] = computeStats(stats.vp(idx),probstruct);
        history.Output.Mean(iIter,:) = Mean;
        history.Output.Cov(iIter,:,:) = Cov;
        history.Output.gsKL(iIter) = gsKL;
        history.Output.Mode(iIter,:) = Mode;    
    end
else
    % Control condition -- run VBMC as normal but compute marginal likelihood
    % and posterior via other methods

    history = infbench_func(); % Retrieve history
    
    % Start warmup
    
    % Store all points (no warmup pruning)
    X = output.X_orig(1:output.Xmax,:);
    y = output.y_orig(1:output.Xmax);
    Niter = find(size(X,1) == history.SaveTicks,1);
    N = history.SaveTicks(1:Niter);
    
    % Find when warm-up ends
    idx = find(stats.vpK > 2,1);
    if isempty(idx); endWarmupN = Inf; else; endWarmupN = stats.N(idx); end
    
    mu = zeros(1,Niter);
    ln_var = zeros(1,Niter);

    % Parameters for WSABI
    diam = probstruct.PUB - probstruct.PLB;
    kernelCov = diag(diam/10);     % Input length scales for GP likelihood model
    lambda = 1;                     % Ouput length scale for GP likelihood model
    alpha = 0.8;
    
    for iIter = 1:Niter        
        X_train = X(1:N(iIter),:);
        y_train = y(1:N(iIter));
        % Prune trials after warmup
        if N(iIter) >= endWarmupN
            idx_keep = output.X_flag(1:N(iIter));
            X_train = X_train(idx_keep,:);
            y_train = y_train(idx_keep);
        end
        X_iter{iIter} = X_train;
        y_iter{iIter} = y_train;
        lnp = infbench_lnprior(X_train,probstruct);
        y_train = y_train - lnp;  % Remove log prior for WSABI
        [mu(iIter),ln_var(iIter)] = ...
            wsabi_oneshot('L',probstruct.PriorMean,diag(probstruct.PriorVar),kernelCov,lambda,alpha,X_train,y_train);
    end
    vvar = max(real(exp(ln_var)),0);
    
    [history,post] = ...
        StoreAlgoResults(probstruct,[],Niter,X_iter{Niter},y_iter{Niter},mu,vvar,X_iter,y_iter,TotalTime);
    history.scratch.output = output;
    history.Output.stats = stats;    
    
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
