function [XStars,YMean,YSD,XData,YData,covvy,closestInd]=activedataseln_old(XsFull,YsFull,XInit,YInit,covvy,lookahead,params)

if nargin<6
    lookahead=0; % how many steps to lookahead
end
if nargin<7 || ~isfield(params,'maxpts')
    params.maxpts=1000; % max number of data points to store
end
if ~isfield(params,'maxdata')
    params.maxdata=5000; % max number of data points to take
end
if ~isfield(params,'threshold')
    params.threshold=1.e-3; % thresh for desired accuracy [determines the # data points to take]
end
if ~isfield(params,'drop_threshold')
    params.drop_threshold=0; % thresh for desired accuracy [determines the # data points to store]
end
if ~isfield(params,'step')
    params.step=0.001; 
    % step size of numerical searches and predictions in time
end
if ~isfield(params,'threshdt')
    params.threshdt=0.03; 
    % how long we guess it to take a SD to grow to threshold - this guess is used
    % only by numerical search algorithms as a first try
end
if ~isfield(params,'deldrop')
    params.deldrop=1; % thresh for desired accuracy [determines the # data points to store]
end
if ~isfield(params,'step')
    params.step=max(1,lookahead); % how fine-grained should predictions be made - predictions will be made every params.step up to lookahead
end
if ~isfield(params,'print')
    params.print=0;%reassuring dots;
end

% draws samples and assigns parameters via bayes MC
covvy=hyperparams(covvy);
covvy=bmcparams(covvy);

NSensors=length(unique(XsFull(:,1)));
NData=size(XsFull,1);
times=unique(XsFull(:,2));
delt=min(times(2:end)-times(1:end-1));

lowr.LT=true;
%lowr.TRANSA=true;
uppr.UT=true;

XData=[];
TDataFull=[];
XDataFull=[];
YDataFull=[];
YData=[];
XStars=[];
YMean=[];
YSD=[];
dropped=0;

count=0;

init=1; 
% % Initial data are from the following time and sensors: 
% percy=perms(1:GD);
% firstallGD=NTInd; %the first TInd corresponding to an observation from all sensors
% for p=1:length(percy)
%     numy=[(min(strfind(num2str(XsFull(:,1)'),num2str(percy(p,:))))-1)/3+1,firstallGD];
%     firstallGD=min(numy);
% end
% XInit=XsFull(firstallGD:firstallGD+GD-1,:);
% YInit=YsFull(firstallGD:firstallGD+GD-1);
mintime=min(XInit(:,2));
time=mintime;
maxtime=max(XsFull(:,2));
sens=[1]; %This is a dummy, it doesn't actually matter
tstar=[time:params.step:time+3*params.threshdt]';
% This is the set of times and sensors we are interested in making
% predictions about. 
XStar=allcombs({(1:NSensors)',tstar});
for n=1:NSensors
    XStarInd{n}=find(XStar(:,1)==n);
end

% Options for fmincon, used below
optimoptions = optimset('GradObj','on','Hessian','off','Display','off','TolFun',1e-4);

% Set up the covariance structure we're going to use for inference about
% how the SDs change with time
covvySDs=struct('covfn',@(varargin)versatile_cov_fn('matern',varargin{:}));
covvySDs.hyperparams(1)=struct('name','mean','priorMean',0,'priorSD',1,'NSamples',1,'type','real','samples',0);
covvySDs.hyperparams(2)=struct('name','logNoiseSD','priorMean',-inf,'priorSD',1,'NSamples',1,'type','real','samples',-inf);


if ~isfield(covvy,'logTimeScalePos')
    logTimeScalePos=cellfun(@(x) strcmp(x,'logInputScale3'),covvy.names);
    covvy.logTimeScalePos=logTimeScalePos;
else
    logTimeScalePos=covvy.logTimeScalePos;
end

if ~isfield(covvy,'logOutputScalePos')
    logOutputScalePos=cellfun(@(x) strcmp(x,'logOutputScale'),covvy.names);
    covvy.logOutputScalePos=logOutputScalePos;
else
    logOutputScalePos=covvy.logOutputScalePos;
end

TInd=0;
while time<maxtime && TInd<params.maxdata
    TInd=TInd+1;
    
    if params.print==1
        TInd
    elseif params.print==0 && ( rem(TInd,100) == 0)
        fprintf('.');
    end;
    
    dropped=max([dropped,length(YData)-(params.maxpts-1)]);
    % The -1 is because we're just about to add an additional pt on
        
    if dropped>0
        covvy=gpparams(XData,YData,covvy,'downdate',1:dropped);
        XData(1:dropped,:)=[];
        YData(1:dropped,:)=[];
    end
    
    X=allcombs({sens,time}); 
    
    if ~init
            
        F1=find(XsFull(:,2)-time>0, 10);
        XTimes=XsFull(F1,:); 
        if isempty(XTimes)
            % we've run out of data!
            %break
        end

        F2=find(XTimes(:,1)==sens,1);
        X=XTimes(F2,:);
        if isempty(X) 
            % if there is not a reading from the sensor we want that is
            % sufficiently close to the desired time, instead take a
            % reading from whichever sensor has a reading closest to the
            % desired time.
            X=XTimes(1,:);
            F2=1;
        end

        time=X(1,2);
        sens=X(1,1);

        X=allcombs({sens,time}); 
        Y=YsFull(F1(F2),:); % new dep variables
    elseif init
        init=0;

        X=XInit;
        Y=YInit;

    end
   
    
    XData=[XData;X];
    TDataFull=[TDataFull;max(X(:,2))]; %we never drop anything from TDataFull
    XDataFull=[XDataFull;X];
    YData=[YData;Y]; 
    YDataFull=[YDataFull;Y];
    
    covvy=gpparams(XData,YData,covvy,'update',length(YData));
    rho=weights(covvy);
    %hackalicious below
    rho(rho<0)=0;
    rho=rho/sum(rho);
    
    XStars=[XStars;XStar];
    
    [wm,wC]=weighted_gpmeancov(rho,XStar,XData,covvy);

    wsig=sqrt(wC); 
    
    YMean=[YMean;wm];
    YSD=[YSD;wsig];
    
    % rearrange to make plotting later easier
    for n=1:NSensors
        Tplot{n}(TInd,:)=XStar(XStarInd{n},2);
        YMeanplot{n}(TInd,:)=wm(XStarInd{n});
        YSDplot{n}(TInd,:)=wsig(XStarInd{n});
    end
    
    [m,closestInd]=max(rho);
    closest_hyperparams=covvy.hypersamples(closestInd).hyperparameters;
    
    covvySDs.hyperparams(3)=struct('name','logInputScale1','priorMean',closest_hyperparams(logTimeScalePos),'priorSD',1,'NSamples',1,'type','real','samples',closest_hyperparams(logTimeScalePos));
    covvySDs.hyperparams(4)=struct('name','logOutputScale','priorMean',closest_hyperparams(logOutputScalePos),'priorSD',1,'NSamples',1,'type','real','samples',closest_hyperparams(logOutputScalePos));

    covvySDs=hyperparams(covvySDs);
    K_SDs=covvySDs.covfn(covvySDs.hypersamples(1).hyperparameters);
    cholK_SDs=chol(K_SDs(tstar,tstar));
    covvySDs.hypersamples(1).cholK=cholK_SDs;
    
    tau=inf(1,NSensors);
    for n=1:NSensors
        % Beliefs about how we expect the standard deviations to vary in
        % time
        datahalf=linsolve(cholK_SDs',real(YSDplot{n}(TInd,:)'),lowr);
        covvySDs.hypersamples(1).datahalf=datahalf;
        covvySDs.hypersamples(1).datatwothirds=linsolve(cholK_SDs,datahalf,uppr);
        tau(n) = fmincon(@(t) gpmean_thresh(params.threshold,t,tstar,covvySDs),time+params.threshdt,[],[],[],[],time,[],[],optimoptions); 
        % don't want results from earlier than current time - might put an
        % upper limit in as well at some point
    end
    
    if min(tau)>time
        time=min(tau); % this is when we need to take a measurement
        % Otherwise, sample again at the current time!
    end
    
    % future times about which we will make predictions
    tstar=(time:params.step:time+3*params.threshdt)';
    
    % This is the set of times and sensors we are interested in making
    % predictions about. 

    XStar=allcombs({(1:NSensors)',tstar});
    % this choice allows us to use our predictions for numerical
    % optimisation
    
    for n=1:NSensors
        XStarInd{n}=find(XStar(:,1)==n);
    end    
    

    
    K=covvy.covfn(closest_hyperparams);
    dKstst=diag(K(XStar,XStar));
    KStarData=K(XStar,XData);
    
    % now we need to work out which sensor to take a data point from next
    period_of_grace=zeros(NSensors,1);
    for sensortrial=1:NSensors 
        XTrial=[sensortrial,time];
        %for SampleInd=1:NSamples 
        % actually, with no new data, we may continue to use closestInd so
        % long as we stick to my hacky approximation of standard deviation.
        % Really, we should use existing weights but loop over all possible
        % SampleInd's to produce SD.
        
        KDataTrial=K(XData,XTrial);
        
        V=[nan(size(XData,1)),KDataTrial;KDataTrial',K(XTrial,XTrial)];
        R=covvy.hypersamples(closestInd).cholK;
        try
        RTrial=updatechol(V,R,length(V));

        KStarDataTrial=[KStarData,K(XStar,XTrial)];
        % Kterm=KStarDataTrial*inv(RTrial);
        Kterm=linsolve(RTrial',KStarDataTrial',lowr)';
        
        SDsTrial=sqrt(dKstst-diag(Kterm*Kterm'));        
        
        for n=1:NSensors
            % Beliefs about how we expect the standard deviations to vary in
            % time
            datahalf=linsolve(cholK_SDs',SDsTrial(XStarInd{n}),lowr);
            covvySDs.hypersamples(1).datahalf=datahalf;
            covvySDs.hypersamples(1).datatwothirds=linsolve(cholK_SDs,datahalf,uppr);
            tau(n) = fmincon(@(t) gpmean_thresh(params.threshold,t,tstar,covvySDs),time+params.threshdt,[],[],[],[],time,[],[],optimoptions); 
            % don't want results from earlier than current time - might put an
            % upper limit in as well at some point
        end
        period_of_grace(sensortrial)=min(tau)-time;
        catch
            if XData(end,1)==sensortrial
                period_of_grace(sensortrial)=-inf;
            else
               err(TInd)=1;
            end
        end
    end  
    
    SSD=nan(NSensors,1);
    for n=1:NSensors;SSD(n)=YSDplot{n}(TInd,1);end;
    if any(SSD>params.threshold) || all(period_of_grace<10^-3*params.step)
        % In this case, rounding etc. seem to render us unable to distinguish
        % between.
        [a,sens]=max(SSD);
        disp 'SSD criterion'
    else
        [maxpog,sens]=max(period_of_grace);
    end
    
    
    
    
    test_sig=wsig;
    drop=params.deldrop;
    cholK=covvy.hypersamples(closestInd).cholK;
    while all(test_sig<params.drop_threshold) && drop<size(XData,1)
        cholK=downdatechol(cholK,1:params.deldrop);
        KStarData=KStarData(:,1+params.deldrop:end);
        Khalf=linsolve(cholK',KStarData',lowr)';
        test_sig = min(sqrt(dKstst-diag(Khalf*Khalf')));    
        drop=drop+params.deldrop;
    end
    dropped=drop-params.deldrop;
    
    save activedataseln
end

NTInd=TInd-1;

% Delete 'overlaps' ie. given information up to t0, we make predictions
% about t2, and then later get an observation at t1, where t0<t1<t2.
Tplot2=cell(1,NSensors);
YMeanplot2=cell(1,NSensors);
YSDplot2=cell(1,NSensors);
for t=1:NTInd-1
    TNext=TDataFull(t+1);
    
    for n=1:NSensors

        FF=find(Tplot{n}(t,:)<TNext);
        Tplot2{n}=[Tplot2{n};Tplot{n}(t,FF)'];    
        YMeanplot2{n}=[YMeanplot2{n};YMeanplot{n}(t,FF)'];    
        YSDplot2{n}=[YSDplot2{n};YSDplot{n}(t,FF)'];
    end
end
XStars=Tplot2;
YMean=YMeanplot2;
YSD=YSDplot2;
XData=XDataFull;
YData=YDataFull;

function [m,gm]=gpmean_thresh(thresh,XStar,XData,covvyf)

[m,C,gm] = gpmeancov(XStar,XData,covvyf,1,'no_cov');

eps=2*(m>thresh)-1;
m = abs(m-thresh);
if nargout > 1   % fun called with two output arguments
    gm = eps.*gm{1};  % Gradient of the function evaluated at XStar
end


