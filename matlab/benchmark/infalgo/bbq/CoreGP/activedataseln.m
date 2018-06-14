function [XStars,YMeans,YSDs,XData,YData,covvy,closestInd]=activedataseln(XsFull,YsFull,XInit,YInit,covvy,lookahead,sample_cost,params)

if nargin<6
    lookahead=0; % how many steps to lookahead
end
if nargin<8 || ~isfield(params,'maxpts')
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
if ~isfield(params,'threshdt')
    params.threshdt=0.03; 
    % our guess for the length of time between successive selected samples
end
if ~isfield(params,'deldrop')
    params.deldrop=1; % thresh for desired accuracy [determines the # data points to store]
end
if ~isfield(params,'min_weight')
    % the minimum hypersample weight which we will consider when generating
    % sample observations
    params.min_weight = 0.1;
end
if ~isfield(params,'step')
    params.step=1; % how fine-grained should predictions be made - predictions will be made every params.step up to lookahead
end
if ~isfield(params,'print')
    params.print=0;%reassuring dots;
end

% draws samples and assigns parameters via bayes MC
covvy=hyperparams(covvy);
covvy=bmcparams(covvy);

num_hypersamples = numel(covvy.hypersamples);
num_obs_samples = min(3,ceil(3000/num_hypersamples));
num_sensors=length(unique(XsFull(:,1)));
num_stars = size(XsFull,1);
times=unique(XsFull(:,2));
delt=mode(diff(times));

lowr.LT=true;
%lowr.TRANSA=true;
uppr.UT=true;



XStars=zeros(0,2);
YMeans=[];
YSDs=[];
dropped=0;

star_ind=0;
if isempty(YInit) || isempty(XInit)
    XInit = XsFull(1,:);
    YInit = YsFull(1,:);
    star_ind=1; % the first datum is actually taken into the initialisation data
end
XData = XInit;
XDataFull = XInit;
YDataFull=YInit;
YData=YInit;
% XDataFull and YDataFull never have data dropped from them

covvy=gpparams(XInit,YInit,covvy,'overwrite');
rho=weights(covvy);
%hackalicious below
rho(rho<0)=0;
rho=rho/sum(rho);

sample_ind=0;
while star_ind<num_stars && sample_ind<params.maxdata
    star_ind=star_ind+1
    
    if params.print==1
        sample_ind
    elseif params.print==0 && ( rem(sample_ind,100) == 0)
        fprintf('.');
    end
    
    X = XsFull(star_ind,:);
    maxrho = max(rho);
    minrho = min(rho);
    
    good_inds = find(((rho-minrho+eps)/(maxrho-minrho+eps))>params.min_weight); %eps in case all rhos are equal
    [unused_wm,unused_wC,covvy] = weighted_gpmeancov(rho,X,XData,covvy,'store',good_inds);

    good_means = [covvy.hypersamples(good_inds).mean_star];
    good_SDs = [covvy.hypersamples(good_inds).SD_star];  
    
    prior = struct('type','mixture','mean',good_means,'SD',good_SDs,'mixtureWeights',rho(good_inds));
            
    obs_samples = nan(num_obs_samples,1);
    
    cdfs = 1/(num_obs_samples+1):1/(num_obs_samples+1):num_obs_samples/(num_obs_samples+1);
    if length(good_inds)>1
       
        start_pt=min(good_means);
        for obs_ind = 1:num_obs_samples
            start_pt = fsolve(@(x) normcdf(x,good_means,good_SDs)*rho(good_inds)-cdfs(obs_ind),start_pt);
            obs_samples(obs_ind) = start_pt;
        end
    elseif length(good_inds)==1
        obs_samples = norminv(cdfs,good_means,good_SDs)';
    end
    
%     % check that these samples are not too poorly conditioned
%     gaps = diff(sort(obs_samples));
%     
%     probs = find(gaps<0.2*wSD);
    
    
    XStar=allcombs({(1:num_sensors)',max(X(:,2))+lookahead});
    
    [wm,wC] = weighted_gpmeancov(rho,XStar,XData,covvy);
    expected_loss_no_sampling = sum(sqrt(wC));
    
    sum_SDs = nan(num_obs_samples,1);
    for obs_ind = 1:num_obs_samples
        Y = obs_samples(obs_ind); % fake observation
        obs_covvy=gpparams([XData;X],[YData;Y],covvy,'update',length(YData)+1);
        obs_rho=weights(obs_covvy);
        
        [wm,wC] = weighted_gpmeancov(obs_rho,XStar,[XData;X],obs_covvy);
        sum_SDs(obs_ind) = sum(sqrt(wC));
    end
    
    % something funny in simple_bmc_integral
    expected_sum_SDs = simple_bmc_integral(obs_samples, sum_SDs, prior);
    expected_loss_sampling = expected_sum_SDs + sample_cost;

    take_this_sample =  expected_loss_sampling < expected_loss_no_sampling;
    
    if take_this_sample
        % we are actually going to take this observation.
        sample_ind=sample_ind+1;
        
        dropped=max([dropped,length(YData)-(params.maxpts-1)]);
        % The -1 is because we're just about to add an additional pt on

        if dropped>0
            XData(1:dropped,:)=[];
            YData(1:dropped,:)=[];
            covvy=gpparams(XData,YData,covvy,'downdate',1:dropped);
        end
        
        Y = YsFull(star_ind,:);

        XData=[XData;X];
        XDataFull=[XDataFull;X];
        YData=[YData;Y]; 
        YDataFull=[YDataFull;Y];

        covvy=gpparams(XData,YData,covvy,'update',length(YData));
        rho=weights(covvy);
        %hackalicious below
        rho(rho<0)=0;
        rho=rho/sum(rho);
    end
    
    nowtime = X(2);
    % nexttime is the time of the next point in XsFull, not necessarily the
    % next point at which we get a sample
    if star_ind==num_stars
        nexttime=nowtime+delt;
    else
        nexttime=XsFull(star_ind+1,2);
    end

    XStar=allcombs({(1:num_sensors)',(nowtime+lookahead*delt:params.step*delt:nexttime+(lookahead)*delt)'});
    XStars=[XStars;XStar];

    [wm,wC]=weighted_gpmeancov(rho,XStar,XData,covvy);

    wsig=sqrt(wC); 

    YMeans=[YMeans;wm];
    YSDs=[YSDs;wsig]; 
    
    
    if take_this_sample
        [m, closestInd] = max(rho);

        [K] = covvy.covfn(covvy.hypersamples(closestInd).hyperparameters);
        Kstst = K(XStar, XStar);
        KStarData = K(XStar, XData);
        test_sig = wsig;
        drop = params.deldrop;
        cholK = covvy.hypersamples(closestInd).cholK;
        while (~isempty(test_sig) && min(test_sig < params.threshold) && drop < size(XData, 1))
            cholK = downdatechol(cholK, 1:params.deldrop); 
            KStarData = KStarData(:, 1 + params.deldrop:end);
            Khalf = linsolve(cholK', KStarData', lowr)';
            test_sig = min(sqrt(diag(Kstst - Khalf * Khalf')));    
            drop = drop + params.deldrop;
        end
        dropped = drop - params.deldrop;
    else
        dropped = 0;
    end
end

XData=XDataFull;
YData=YDataFull;
