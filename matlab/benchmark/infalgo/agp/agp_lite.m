function [vbmodel,exitflag,output] = agp_lite(fun,x0,PLB,PUB,options)
%AGP_LITE Light implementation of AGP method for Bayesian inference.

% This is a variant of the AGP algorithm proposed in:
% Wang, H., & Li, J. (2017). Adaptive Gaussian process approximation for 
% Bayesian inference with expensive likelihood functions. 
% arXiv preprint arXiv:1703.09930.

% Code by Luigi Acerbi (2018).

% Add variational Gaussian mixture model toolbox to path
mypath = fileparts(mfilename('fullpath'));
addpath([mypath filesep 'vbgmm']);

% Check existence of GPlite toolbox in path
if ~exist('gplite_train.m','file')
    error('agp_lite:NoGPliteToolbox','The GPlite toolbox needs to be in the MATLAB path to run AGP_LITE.');
end

exitflag = 0;   % To be used in the future
D = size(x0,2);

%% Algorithm options

% Number of samples per iteration
MaxFunEvals = D*100;
if isfield(options,'MaxFunEvals') 
    MaxFunEvals = options.MaxFunEvals;
end
% Number of samples per iteration
Ns = 2e4;
if isfield(options,'Nsamples') && ~isempty(options.Nsamples)
    Ns = options.Nsamples;
end
% Max GP hyperparameter samples (0 = optimize)
NsMax_gp = 0;
if isfield(options,'GPsamples') && ~isempty(options.GPsamples)
    NsMax_gp = options.GPsamples;
end
% Training set size for switching to GP hyperparameter optimization
StopGPSampling = 200 + 10*D;
if isfield(options,'StopGPSampling') && ~isempty(options.StopGPSampling)
    StopGPSampling = options.StopGPSampling;
end
% AGP standard acquisition function
acqfun = @acqagp;
if isfield(options,'AcqFun') && ~isempty(options.AcqFun)
    acqfun = options.AcqFun;
end

Ninit = 20;
Nstep = 10;
Nsearch = 2^13; % Starting search points for acquisition fcn

% Variational Bayesian Gaussian mixture options
vbopts.Display     = 'off';     % Display
vbopts.TolBound    = 1e-8;         % Minimum relative improvement on variational lower bound
vbopts.Niter       = 2000;         % Maximum number of iterations
vbopts.Nstarts     = 2;            % Number of runs
vbopts.TolResponsibility = 0.5;    % Remove components with less than this total responsibility
vbopts.ClusterInit = 'kmeans';     % Initialization of VB (methods are 'rand' and 'kmeans')


% GPLITE model options
gpopts.Nopts = 1;       % Number of hyperparameter optimization runs
gpopts.Ninit = 2^10;    % Initial design size for hyperparameter optimization
gpopts.Thin = 5;        % Thinning for hyperparameter sampling (if sampling)
gp_meanfun = 'zero';    % Constant-zero mean function

% Setup options for CMA-ES optimization
cmaes_opts = cmaes_modded('defaults');
cmaes_opts.EvalParallel = 'yes';
cmaes_opts.DispFinal = 'off';
cmaes_opts.SaveVariables = 'off';
cmaes_opts.DispModulo = Inf;
cmaes_opts.LogModulo = 0;
cmaes_opts.CMA.active = 1;      % Use Active CMA (generally better)



% Evaluate fcn on random starting grid
Nrnd = Ninit - size(x0,1);
Xrnd = bsxfun(@plus,PLB,bsxfun(@times,PUB-PLB,rand(Nrnd,D)));
X = [x0;Xrnd];
y = zeros(Ninit,1);
for i = 1:Ninit; y(i) = fun(X(i,:)); end

mu0 = 0.5*(PLB + PUB);
width = 0.5*(PUB - PLB);
sigma0 = width;
Xs = bsxfun(@plus,bsxfun(@times,sigma0,randn(Ns,D)),mu0);

vbmodel = vbgmmfit(Xs',1,[],vbopts);

% Initial hyperparameter vector
hyp = [log(width(:));log(std(y));log(1e-3)];

% Set proposal fcn based on PLB and PUB
if ~isfield(options,'ProposalFcn') || isempty(options.ProposalFcn)
    options.ProposalFcn = @(x) agp_proposal(x,PLB,PUB);
end

iter = 1;

while 1
    fprintf('Iter %d...', iter);
    N = size(X,1);
    
    % Build GP approximation
    fprintf(' Building GP approximation...');
    Ns_gp = min(round(NsMax_gp/10),round(NsMax_gp / sqrt(N)));
    if N >= StopGPSampling; Ns_gp = 0; end
    py = vbgmmpdf(vbmodel,X');   % Evaluate approximation at X    
    y_gp = y - log(py(:));         % Log difference
    
    % Set priors over GP hyperparameters
    hypprior = [];
    Nhyp = D+2;
    hypprior.mu = NaN(1,Nhyp);
    hypprior.sigma = NaN(1,Nhyp);
    hypprior.df = 3*ones(1,Nhyp);    % Broad Student's t prior
    hypprior.mu(1:D) = log(std(X));
    hypprior.sigma(1:D) = max(2,log(max(X)-min(X)) - log(std(X)));
    hypprior.mu(D+2) = log(1e-2);
    hypprior.sigma(D+2) = 0.5;
        
    % Train GP
    [gp,hyp] = gplite_train(hyp,Ns_gp,X,y_gp,gp_meanfun,hypprior,[],gpopts);
    
    % Sample from GP
    fprintf(' Sampling from GP...');
    lnpfun = @(x) log(vbgmmpdf(vbmodel,x'))';    
    Xs = gplite_sample(gp,Ns,[],'parallel',lnpfun);
    
    % cornerplot(Xs);
    
    % Refit vbGMM
    fprintf(' Refit vbGMM...\n');
    vbmodel = vbgmmfit(Xs',[],[],vbopts);

    %Xrnd = vbgmmrnd(vbmodel,1e5)';
    %Mean = mean(Xrnd,1);
    %Cov = cov(Xrnd);
    
    % Estimate normalization constant in HPD region
    [lnZ,lnZ_var] = estimate_lnZ(X,y,vbmodel);
        
    fprintf('Estimate of lnZ = %f +/- %f.\n',lnZ,sqrt(lnZ_var));
    
    % Record stats
    stats(iter).N = N;
    % stats(iter).Mean = Mean;
    % stats(iter).Cov = Cov;
    stats(iter).lnZ = lnZ;
    stats(iter).lnZ_var = lnZ_var;
    stats(iter).vbmodel = vbmodel;
    stats(iter).gp = gplite_clean(gp);
    
    % Find max of approximation among GP samples and record approximate mode
    ys = vbgmmpdf(vbmodel,Xs')';
    [~,idx] = max(ys);
    stats(iter).mode = Xs(idx,:);
    
    if N >= MaxFunEvals; break; end
    
    % Select new points
    fprintf(' Active sampling...');
    for i = 1:Nstep
        fprintf(' %d..',i);
        % Random uniform search
        [xnew,fval] = fminfill(@(x) acqfun(x,vbmodel,gp,options),[],[],[],PLB,PUB,[],struct('FunEvals',floor(Nsearch/2)));
        
        % Random search sample from vbGMM
        xrnd = vbgmmrnd(vbmodel,ceil(Nsearch/2))';
        frnd = acqfun(xrnd,vbmodel,gp,options);
        [frnd_min,idx] = min(frnd);        
        if frnd_min < fval; xnew = xrnd(idx,:); fval = frnd_min; end

        % Optimize from best point with CMA-ES
        insigma = width(:)/sqrt(3);
        [xnew_cmaes,fval_cmaes] = cmaes_modded(func2str(acqfun),xnew',insigma,cmaes_opts,vbmodel,gp,options,1);
        if fval_cmaes < fval; xnew = xnew_cmaes'; end
        
        % Add point
        ynew = fun(xnew);
        X = [X; xnew];
        y = [y; ynew];
        
        py = vbgmmpdf(vbmodel,xnew');   % Evaluate approximation at X    
        ynew_gp = ynew - log(py);       % Log difference
        gp = gplite_post(gp,xnew,ynew_gp,[],1);   % Rank-1 update        
    end
    fprintf('\n');
    
    iter = iter + 1;
end

output.X = X;
output.y = y;
output.stats = stats;

end

%--------------------------------------------------------------------------
function [lnZ,lnZ_var] = estimate_lnZ(X,y,vbmodel)
%ESTIMATE_LNZ Rough approximation of normalization constant

hpd_frac = 0.2;     % Top 20% HPD
N = size(X,1);

lp = log(vbgmmpdf(vbmodel,X')');

% Take HPD points according to both fcn samples and model
[~,ord] = sort(lp + y,'descend');

idx_hpd = ord(1:ceil(N*hpd_frac));
lp_hpd = lp(idx_hpd);
y_hpd = y(idx_hpd);

delta = -(lp_hpd - y_hpd);

lnZ = mean(delta);
lnZ_var = var(delta)/numel(delta);

end