%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test sqrtBBQ M & L versions vs: SMC, annealed importance sampling &
% Bayesian Monte Carlo on a synthetic binary classification problem with
% GPML.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Setup variables
close all;
clear all;

randn('state',10);
rand('state',10);

addpath(genpath(pwd));
startup;

% Generate Random binary classification problem:
ntr = 500;                             % number of training and test points
xtr = 10*sort(rand(ntr,1));                                % sample dataset
p = @(x) 1./(1+exp(-5*sin(x)));             % "true" underlying probability
ytr = 2*(p(xtr)>rand(ntr,1))-1;                         % draw labels +1/-1
i = randperm(ntr); nout = 3;                                 % add outliers 

xx = xtr;
yy = ytr;

dim = size(xx,2)+2;      % Dimensionality of integral over hyperparameters.

range = [-10*ones(1,dim);10*ones(1,dim)];   % Bounding box for integration.

priorMu = 0*ones(1,dim);                     % Gaussian prior in log-space.
priorVar = 2*eye(dim);

kernelVar = exp(0)*eye(dim); % Input length scales for GP likelihood model.
lambda = exp(0);              % Ouput length scale for GP likelihood model.

alpha = 0.8;                              % Fractional offset, as in paper.
numSamples = 100;                        % Total number of sqrtBBQ samples.

printing = 1;                     % Output intermediate results to console.

recalcGndTruth = 0;         % Recalculate ground truth, or pull old result.
if ~recalcGndTruth          % NB!! old result assumes seed unchanged, and 
                            % only 500 datapoints sampled above!
    load ./Results/synthgndtruth.mat;
end
numgndtruth = 1000000;   % If recalculating, number of SMC samples to draw.

currtimesmc = 0;
currtimebq = 0;

%%%%%%%%%%%%% GPML STUFF %%%%%%%%%%%%%%%%%%%
% 1) set GP parameters
cov = {@covSum,{@covSEard,@covNoise}}; hyp.cov = [zeros(dim-2,1); 0; -Inf];  
lik =  {'likErf'};     hyp.lik  = [];                  % Likelihood
mn  = {'meanZero'};      hyp.mean = [];
inf = 'infLaplace';

% 2) LogLikFunction
likFunc = @(x) -1*gp(rewrap(hyp,transp(x)), inf, mn, cov, lik, xx, yy);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Storage for results
smctic = zeros(1000,1);
smcval = zeros(1000,1);
sqrtbqtic = zeros(1000,1);

%------------------------------ sqrtBBQ ----------------------------------%

[muBBQ_M, varBBQ_M, timeBBQ_M, xxIter_M, hyps_M] = wsabiM(range, ...
                                                          priorMu, ...
                                                          priorVar, ...
                                                          kernelVar, ...
                                                          lambda, ...
                                                          alpha, ...
                                                          numSamples, ...
                                                          likFunc, ...
                                                          printing);




[muBBQ_L, varBBQ_L, timeBBQ_L, xxIter_L, hyps_L] = wsabiL(range, ...
                                                          priorMu, ...
                                                          priorVar, ...
                                                          kernelVar, ...
                                                          lambda, ...
                                                          alpha, ...
                                                          numSamples, ...
                                                          likFunc, ...
                                                          printing);
totime = cumsum(timeBBQ_L);

fprintf('\n wsabiM Integral: %g \n',muBBQ_M(end));
fprintf('\n wsabiL Integral: %g \n',muBBQ_L(end));
fprintf('Total wsabiL time: %g \n',totime(end));
logmuBBQ_M = muBBQ_M;
logmuBBQ_L = muBBQ_L;

%---------------------------- SMC Sampling -------------------------------%

smcinterclk = 0;
smcSamples = zeros(1,dim);
smcOutput = likFunc(smcSamples);
smcinter = exp(smcOutput);
logsmcinter = smcOutput;
prstr = [];
fprintf('\n SMC Sampling ... \n');
while sum(smcinterclk) < totime(end)
    t = cputime;
    smcSamples(end+1,:) = mvnrnd(priorMu,priorVar);
    smcOutput(end+1) = likFunc(smcSamples(end,:));
    maxtmp = max(smcOutput);
    smcinter(end+1) = mean(exp(smcOutput-maxtmp));
    logsmcinter(end+1) = log(smcinter(end)) + maxtmp;
    smcinterclk(end+1) = cputime - t;
    prstr = sprintf('%2i%%',ceil((sum(smcinterclk)/totime(end)) * 100 + 0.00001));
    fprintf(repmat('\b',1,length(prstr)));
    fprintf(strcat(prstr,'%%'));
end
fprintf('\n Done \n');
logsmcinter(end);

%--------------------------- BMC Sampling --------------------------------%

numbmc = ceil(length(timeBBQ_M(:,1))/1);
mubq = zeros(numbmc,1);
varbq = zeros(numbmc,1);
timebq = zeros(numbmc,1);
fprintf('\n BMC Sampling ... \n');
for i = 1:length(timebq)
    tic; 
    [mubq(i), varbq(i), kernelVar, lambda] = bq(range, ...
                                                        priorMu, ...
                                                        priorVar, ...
                                                        kernelVar, ...
                                                        lambda, ...
                                                        alpha, ...
                                                        smcSamples(1:i,:), ...
                                                        likFunc);
    timebq(i) = toc;
    if sum(timebq(1:i)) > totime;
        break
    end
    
    if i > 1
        fprintf(repmat('\b',1,length(prstr)));
    end
    
    prstr = sprintf('Iter = %i\n',i);
    fprintf(prstr);
end

%--------------------------- AIS Sampling --------------------------------%

prior.mean = priorMu;
prior.covariance = priorVar;
iters_ais = 10;
tot_num_samples = logspace(0,log10(length(smcinter)),iters_ais);
mean_log_evidence = zeros(1,iters_ais);
aistimes = zeros(1,iters_ais);
for i = 1:iters_ais
    opt.num_samples = ceil(tot_num_samples(i));
    [mean_log_evidence_tmp, var_log_evidence, sample_locs, logliks, aistimes_tmp, stats] =  ais_mh(likFunc, prior, opt);
    mean_log_evidence(i) = mean_log_evidence_tmp(end);
    aistimes(i) = sum(aistimes_tmp);
end

%------------------------ GND Truth Sampling -----------------------------%

if recalcGndTruth
    %Ground Truth
    smcsample2 = zeros(numgndtruth,dim);
    smcout2 = zeros(numgndtruth,1);
    smcint = zeros(numgndtruth,1);
    logsmcint = zeros(numgndtruth,1);
    fprintf('GndTruth...\n');
    for i = 1:numgndtruth
        smcsample2(i,:) = mvnrnd(priorMu,priorVar);
        smcout2(i) = likFunc(smcsample2(i,:));
        maxtru = max(smcout2);
        smcint(i) = mean(exp(smcout2(1:i) - maxtru));
        logsmcint(i) = log(smcint(i)) + maxtru;
        
        if ~mod(i,numgndtruth/20)
            fprintf('.');
        end
        
    end
    fprintf('\n Done \n');
    
end

%-------------------------------------------------------------------------%

% Save results to file.
times = clock;
dat = date;
savestr = strcat(sprintf('./Results/SyntheticClassRun_%g_%g',times(4),times(5)),dat,sprintf('.mat'));
save(savestr);

% Append actively sampled points to ground truth.
for i = 1:length(xxIter_M)
    smcout2(end+1) = likFunc(xxIter_M(i,:));
end

maxtru = max(smcout2);
actTrueInt = log(mean(exp(smcout2 - maxtru))) + maxtru;


%----------------------------- Plotting ----------------------------------%
close all;

cc = colormap(cbrewer('qual','Set1',8));  

figure;
semilogy(cumsum(timeBBQ_M(1:end)),ones(size(timeBBQ_M))*actTrueInt,'color',cc(1,:),'Linewidth',3); hold on;
semilogy(cumsum(timeBBQ_L(1:end)),logmuBBQ_L(1:end),'color',cc(2,:),'LineWidth',1.5);
semilogy(cumsum(timeBBQ_M(1:end)),logmuBBQ_M(1:end),'color',cc(3,:),'LineWidth',1.5);
semilogy(cumsum(smcinterclk),logsmcinter,'color',cc(4,:),'Linewidth',1.5);
semilogy(aistimes,mean_log_evidence,'color',cc(5,:),'Linewidth',1.5);
semilogy(cumsum(timebq),mubq,'color',cc(6,:),'Linewidth',1.5);
legend('Ground Truth','oldSqrtBBQ','sqrtBBQ','SMC','AIS','BMC');
xlabel('Time in seconds','FontSize', 16);
ylabel('logZ','FontSize', 14,'FontWeight','bold');
set(gca,'FontName','Helvetica', 'FontSize', 14);


figure;
plot(cumsum(timeBBQ_M(1:end)),ones(size(timeBBQ_M))*actTrueInt,'color',cc(1,:),'Linewidth',3); hold on;
plot(cumsum(timeBBQ_L(1:end)),logmuBBQ_L(1:end),'color',cc(2,:),'LineWidth',1.5);
plot(cumsum(timeBBQ_M(1:end)),logmuBBQ_M(1:end),'color',cc(3,:),'LineWidth',1.5);
plot(cumsum(smcinterclk),logsmcinter,'color',cc(4,:),'Linewidth',1.5);
plot(aistimes,mean_log_evidence,'color',cc(5,:),'Linewidth',1.5);
plot(cumsum(timebq),mubq,'color',cc(6,:),'Linewidth',1.5);
legend('Ground Truth','\linearsqrt','\momentsqrt','SMC','AIS','BMC');
xlabel('Time in seconds','FontName','Helvetica','FontSize', 14);
ylabel('logZ','FontSize', 14,'FontName','Helvetica');
set(gca,'FontName','Helvetica', 'FontSize', 14);





