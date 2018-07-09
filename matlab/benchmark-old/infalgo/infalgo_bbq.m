function [history,post,algoptions] = infalgo_bbq(algo,algoset,probstruct)

% Add algorithm to MATLAB path
BaseFolder = fileparts(mfilename('fullpath'));
AlgoFolder = 'bbq';
addpath(genpath([BaseFolder filesep() AlgoFolder]));

algoptions.ApproxMarginalize = false;    % Approximate marginalization of hyperparameters
algoptions.Alpha = 0.8;     % Fractional offset, as in paper.

algoptions.MaxFunEvals = probstruct.MaxFunEvals;

% Options from current problem
switch algoset
    case {0,'debug'}; algoset = 'debug'; algoptions.Debug = 1; algoptions.Plot = 'scatter';
    case {1,'base'}; algoset = 'base';           % Use defaults
    case {2,'marginal'}; algoset = 'marginal'; algoptions.ApproxMarginalize = true;
        
    otherwise
        error(['Unknown algorithm setting ''' algoset ''' for algorithm ''' algo '''.']);
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

% Assign values to OPT struct for BBQ (use defaults)
opt.num_samples = algoptions.MaxFunEvals;
opt.gamma = 0.01;
opt.num_retrains = 10;
opt.num_box_scales = 5;
opt.train_gp_time = 120;
opt.train_gp_num_samples = 5*D;
opt.train_gp_print = false;
opt.exp_loss_evals = 1000 * D;
opt.start_pt = x0;
opt.print = 2;
opt.plots = false;
opt.debug = true;
opt.parallel = false;
opt.marginalise_scales = algoptions.ApproxMarginalize;

if DEBUG
    opt.num_retrains = 5;
    opt.train_gp_time = 20;
end

% Do not add log prior to function evaluation, already passed
probstruct.AddLogPrior = false;

% Assign prior mean and covariance
prior.mean = probstruct.PriorMean;
prior.covariance = diag(probstruct.PriorVar);

algo_timer = tic;
[mu, vvar, samples, diagnostics] = ...
    sbq(@(x) infbench_func(x,probstruct), prior, opt);
TotalTime = toc(algo_timer);

X = samples.locations;
y = samples.log_l;

Niter = numel(probstruct.SaveTicks);
Nmax = numel(mu);
idx = probstruct.SaveTicks(probstruct.SaveTicks <= Nmax);
mu = mu(idx);
vvar = vvar(idx);

[history,post] = StoreAlgoResults(...
    probstruct,[],Niter,X,y,mu,vvar,[],[],TotalTime);

history.Output.stats = diagnostics;

end