%function [XStars,YMean,YSD] = tracking(XsFull,YsFull)
% An example of how to do on-line tracking

% load whatever data is needed and prepare it for 'mosb' format. This
% requires XsFull to contain two columns, the first of which labels the
% dimension (e.g. sensor) associated with each observation, the second the
% time of observation. YsFull is a column of the observations associated
% with XsFull.
% y = load('fixedCorrData');
% [L,D] = size(y.yi);
% XFull=allcombs({(1:D)',(1:L)'});
% exampleData = [y.yi(:,1); y.yi(:,2)];
% Both=[XFull,exampleData(:)];
% Both2=Both((Both(:,3)~=-Inf),:);
% Both3=sortrows(Both2,2);
% XsFull=Both3(:,1:2);
% YsFull=Both3(:,3);

% Edit this
load('mg');
[L,D] = size(t_te);
XsFull=allcombs({(1:D)',(1:L)'});
YsFull=(t_te-mean(t_te))/std(t_te);
%

NDims=length(unique(XsFull(:,1))); % Dimensionality of data e.g. number of 1-D sensors, or twice the number of 2-D sensors

Indep=false; % dependent GPs or not

lookahead=0; % how many steps to lookahead

params.maxpts=3; % max number of data points to store
params.threshold=1.e-3; % thresh for desired accuracy [determines the # data points to store]

% specify a function, such as Matern or sq exp etc
covvy=struct('covfn',@(hp) sensor_cov_fn(NDims,'sqdexp',hp));

% setting priors for specified cov function - if want marginalisation then
% NSamples > 1
covvy.hyperparams(1)=struct('name','mean',...
    'priorMean',0, 'priorSD',1, 'NSamples',1, 'type','real');
covvy.hyperparams(2)=struct('name','logNoiseSD',...
    'priorMean',log(0.01), 'priorSD',log(0.2), 'NSamples',1, 'type','real');
% Edit
covvy.hyperparams(3)=struct('name','logTimeScale',...
    'priorMean',log(40), 'priorSD',log(0.5), 'NSamples',5, 'type','real');
%


% Specify priors for the length scales:
% logLengthScales is a vector of length scales (root mean square), one for each sensor
% (windfarm)
logLengthScales=struct(...
    'priorMean',log(1*ones(1,NDims)), 'priorSD',log(ones(1,NDims)), 'NSamples',1);

% Specify priors for the correlation angles, used by the spherical
% parameterisation to determine the correlation matrix over. There are to
% be 0.5*(NDims^2+NDims) such angles, all in [0,pi]. If the priorMean or
% priorSD are empty or nonexistent, the function corrpriors will assume
% 'reasonable' values. If Indep is true, NSamples is ignored. 
corrAngles=struct('priorMean',[],'priorSD',[],'NSamples',1);

% Assign correlation priors
covvy=corrpriors(covvy,logLengthScales,corrAngles,Indep);

% Now actually perform the tracking!
[XStars,YMean,YSD,covvy,closestInd]=track(XsFull,YsFull,covvy,lookahead,params);

% The most strongly weighted hyperparameter samples
closest_hyperparameters=covvy.hypersamples(closestInd).hyperparameters;

%plot
plot_ts(XsFull,YsFull,XStars,YMean,YSD)

