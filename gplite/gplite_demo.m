%GPLITE_DEMO Demo script with example usage for the GPLITE toolbox.

% Create example data in 1D
N = 31;
X = linspace(-5,5,N)';
s2 = 0.00*0.1*exp(0.5*X);
y = sin(X) + sqrt(s2).*randn(size(X));
y(y<0) = -abs(3*y(y<0)).^2;
s2 = [];

%idx = N+1:N+3;
%X(idx) = linspace(6,7,numel(idx))';
%s2(idx) = 1e-4;
%y(idx(randperm(numel(idx)))) = -linspace(1000,1001,numel(idx))';

hyp0 = [];          % Starting hyperparameter vector for optimization
Ns = 10;             % Number of hyperparameter samples
covfun = [3 3];     % GP covariance function
meanfun = 4;        % GP mean function
noisefun = [1 0 0]; % Constant plus user-provided noise
hprior = [];        % Prior over hyperparameters
options = [];       % Additional options

% Output warping function
outwarpfun = @outwarp_negpow;
%outwarpfun = [];
options.OutwarpFun = outwarpfun;

% Set prior over noise hyperparameters
gp = gplite_post([],X,y,covfun,meanfun,noisefun,s2,[],outwarpfun);
hprior = gplite_hypprior(gp);

hprior.mu(gp.Ncov+1) = log(1e-3);
hprior.sigma(gp.Ncov+1) = 0.5;

if gp.Nnoise > 1
    hprior.LB(gp.Ncov+2) = log(5);
    hprior.mu(gp.Ncov+2) = log(10);
    hprior.sigma(gp.Ncov+2) = 0.01;

    hprior.mu(gp.Ncov+3) = log(0.3);
    hprior.sigma(gp.Ncov+3) = 0.01;
    hprior.df(gp.Ncov+3) = Inf;
end

if ~isempty(outwarpfun)
    hprior.mu(gp.Ncov+gp.Nnoise+gp.Nmean+2) = 0;
    hprior.sigma(gp.Ncov+gp.Nnoise+gp.Nmean+2) = 1;
    hprior.mu(gp.Ncov+gp.Nnoise+gp.Nmean+3) = 0;
    hprior.sigma(gp.Ncov+gp.Nnoise+gp.Nmean+3) = 1;
end

% Train GP on data
[gp,hyp,output] = gplite_train(hyp0,Ns,X,y,covfun,meanfun,noisefun,s2,hprior,options);

hyp             % Hyperparameter samples

xstar = linspace(-15,15,200)';   % Test points

% Compute GP posterior predictive mean and variance at test points
[ymu,ys2,fmu,fs2] = gplite_pred(gp,xstar);

% Plot data and GP prediction
close all;
figure(1); hold on;
gplite_plot(gp);