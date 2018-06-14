function [XStars,YMean,YSD,covvy,closestInd,monitor]=track_HMC(XsFull,YsFull,covvy,lookahead,params,training_set_size)

if nargin<4
    lookahead=0; % how many steps to lookahead
end
if nargin<5 || ~isfield(params,'maxpts')
    params.maxpts=1000; % max number of data points to store
end
if ~isfield(params,'threshold')
    params.threshold=1.e-3; % thresh for desired accuracy [determines the # data points to store]
end
if ~isfield(params,'deldrop')
    params.deldrop=1; % thresh for desired accuracy [determines the # data points to store]
end
if ~isfield(params,'step')
    params.step=max(1,lookahead); % how fine-grained should predictions be made - predictions will be made every params.step up to lookahead
end
if ~isfield(params,'print')
    params.print=1;%not reassuring dots;
end
if nargin<6
    % the number of points from XsFull and YsFull that the GP starts with
    training_set_size=1;
end

if size(XsFull,2)==1
    XsFull = allcombs({1,XsFull});
end

% Initialises hypersamples
covvy=hyperparams(covvy);
NSamples = numel(covvy.hypersamples);
% All hypersamples need to have their gpparams overwritten
covvy.lastHyperSampleMoved=1:NSamples;

NDims=length(unique(XsFull(:,1)));
NData=size(XsFull,1);
times=unique(XsFull(:,2));
delt=min(times(2:end)-times(1:end-1));

lowr.LT=true;
uppr.UT=true;

XData=XsFull(1:training_set_size,:);
YData=YsFull(1:training_set_size,:);

covvy=gpparams(XData, YData, covvy, 'overwrite', []);

XStars=[];
YMean=[];
YSD=[];
dropped=0;

count=0;

hmc2('state', 0);

for ind=training_set_size+1:NData
    
    
    if params.print==1 && ind>0.01*(1+count)*NData
        count=count+1;
        ind
    elseif params.print==0 && ( rem(ind,100) == 0)
        fprintf('.');
    end
        
    X = XsFull(ind, :);
    Y = YsFull(ind, :);
    
    
    no_data = any(isnan([X,Y]));
    
    if ~no_data
        dropped = max([dropped, length(YData) - (params.maxpts - size(X,1))]);
        % The -size(X,1) is because we're just about to add additional pts on

        if (dropped > 0)
            covvy = gpparams(XData, YData, covvy, 'downdate', 1:dropped);
            XData(1:dropped, :) = [];
            YData(1:dropped ,:) = [];
        end

        XData=[XData;X];
        YData=[YData;Y]; 
        
        covvy=gpparams(XData, YData, covvy, 'update', size(XData,1), ...
            setdiff(1:NSamples,covvy.lastHyperSampleMoved));
    end
    
    
    likelihood_fn = @(covvy,index) track_likelihood_fullfn(XData,YData,covvy,index);
    covvy = likelihood_fn(covvy,covvy.lastHyperSampleMoved);
    
    % Use ML_ind from previous time-step -improve_bmc_conditioning could
    % potentially moved to after manage_hyper_samples
    
    % nexttime is the time at which we will next get a reading
    if ind==NData
        nexttime=XsFull(ind,2)+delt;
    else
        nexttime=XsFull(ind+1,2);
    end
    
  % XStar is the point at which we wish to make predictions. Note XStar
  % will be empty if we are about to receive more observations at the
  % same time.
    XStar=allcombs({(1:NDims)',(max(X(:,2))+lookahead*delt:params.step*delt:nexttime+(lookahead)*delt)'});
    
    XStars=[XStars;XStar];
    
    rho = ones(NSamples,1)/NSamples;
    [wm,wC] = weighted_gpmeancov(rho,XStar,XData,covvy);
    
    wsig=sqrt(wC); 
    
    YMean=[YMean;wm];
    YSD=[YSD;wsig];
    
    [log_ML,closestInd] = max([covvy.hypersamples.logL]);
    [K]=covvy.covfn(covvy.hypersamples(closestInd).hyperparameters);
    Kstst=K(XStar,XStar);
    KStarData=K(XStar,XData);
    test_sig=wsig;
    drop=params.deldrop;
    cholK=covvy.hypersamples(closestInd).cholK;
    while ~isempty(test_sig) && min(test_sig < params.threshold) && drop < size(XData, 1)
        % The -1 above is because we're just about to add an additional pt
        % on at the next time step
        cholK=downdatechol(cholK,1:params.deldrop);
        KStarData=KStarData(:,1+params.deldrop:end);
        Khalf=linsolve(cholK',KStarData',lowr)';
        test_sig = min(sqrt(diag(Kstst-Khalf*Khalf')));    
        drop=drop+params.deldrop;
    end
    dropped=drop-params.deldrop;
    
    if ind==training_set_size+1
        % Start our chain off near the peak of the prior
        active_hp_inds = covvy.active_hp_inds;
        last_sample = covvy.hypersamples(ceil(end/2)).hyperparameters(active_hp_inds);
        covvy = manage_hyper_samples_HMC(covvy,likelihood_fn,last_sample);
    else
        covvy = manage_hyper_samples_HMC(covvy,likelihood_fn);
    end
   
    samples=cat(1,covvy.hypersamples.hyperparameters);
    num_samples=size(samples,1);
    monitor.t(ind).rho=zeros(num_samples,1);
    monitor.t(ind).rho(closestInd)=1;
    monitor.t(ind).hypersamples=cat(1,covvy.hypersamples.hyperparameters);
    
end
