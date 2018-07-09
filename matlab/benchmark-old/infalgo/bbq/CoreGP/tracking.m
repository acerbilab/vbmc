function [XStars,YMean,YSD] = tracking(XsFull,YsFull);
% An example of how to do on-line tracking

% load whatever data is needed and prepare it for 'mosb' format. This
% requires XsFull to contain two columns, the first of which labels the
% dimension (e.g. sensor) associated with each observation, the second the
% time of observation. YsFull is a column of the observations associated
% with XsFull.
%y = load('../../fixedCorrData');
% [L,D] = size(y.yi);
% XFull=allcombs({(1:D)',(1:L)'});
% exampleData = [y.yi(:,1); y.yi(:,2)];
% Both=[XFull,exampleData(:)];
% Both2=Both((Both(:,3)~=-Inf),:);
% Both3=sortrows(Both2,2);
% XsFull=Both3(:,1:2);
% YsFull=Both3(:,3);

Indep=false; % dependent GPs or not

lookahead=1; % how many steps to lookahead

params.maxpts=3; % max number of data points to store
params.threshold=1.e-3; % thresh for desired accuracy [determines the # data points to store]

% specify a function, such as Matern or sq exp etc
covvy=struct('covfn',@sensor_cov_fn);

% setting priors for specified cov function - if want maginalisation then
% NSamples > 1
covvy.hyperparams(1)=struct('name','mean',...
    'priorMean',0, 'priorSD',1, 'NSamples',1, 'type','real');
covvy.hyperparams(2)=struct('name','logNoiseSD',...
    'priorMean',log(0.5), 'priorSD',log(0.2), 'NSamples',1, 'type','real');
covvy.hyperparams(3)=struct('name','logTimeScale',...
    'priorMean',log(10000), 'priorSD',log(2), 'NSamples',1, 'type','real');

NDims=length(unique(XsFull(:,1))); % Dimensionality of data e.g. number of 1-D sensors, or twice the number of 2-D sensors

% Specify priors for the length scales:
logLengthScales=struct(...
    'priorMean',log(1*ones(1,NDims)), 'priorSD',log(ones(1,NDims)), 'NSamples',1);

% Specify priors for the correlation angles, used by the spherical
% parameterisation to determine the correlation matrix over. There are to
% be 0.5*(NDims^2+NDims) such angles, all in [0,pi]. If the priorMean or
% priorSD are empty or nonexistent, the function corrpriors will assume
% 'reasonable' values. If Indep is true, NSamples is ignored. 
corrAngles=struct('priorMean',[],'priorSD',[],'NSamples',3);

% Assign correlation priors
covvy=corrpriors(covvy,logLengthScales,corrAngles,Indep);

% Now actually perform the tracking!
[XStars,YMean,YSD,covvy,closestInd]=track(XsFull,YsFull,covvy,lookahead,params);

% The most strongly weighted hyperparameter samples
closest_hyperparameters=covvy.hypersamples(closestInd).hyperparameters;

%plot
%plot_ts(XsFull,YsFull,XStars,YMean,YSD)

