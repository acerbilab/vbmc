function [XStars,YMean,YSD,covvy,closestInd,monitor]=track_ahs(XsFull,YsFull,covvy,lookahead,params,training_set_size,candidate_combs_template)

num_cands=6;
covvy.manage_h2s_method = 'optimal';

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
if nargin<7
    try
        candidate_combs_template=load(candidate_combs);
    catch
        candidate_combs_template{num_cands}=find_likelihood_samples(zeros(num_cands,1),ones(num_cands,1),100,300,false);
    end
end

plot_some_inds = isnumeric(covvy.plots);

if ~isfield(covvy,'plots')
    covvy.plots=false;
elseif plot_some_inds
    plot_inds = covvy.plots;
end

if size(XsFull,2)==1
    XsFull = allcombs({1,XsFull});
end

covvy.candidate_combs_template=candidate_combs_template;
% Initialises hypersamples
covvy=hyperparams(covvy);
% All hypersamples need to have their gpparams overwritten
covvy.lastHyperSampleMoved=1:numel(covvy.hypersamples);
covvy.ignoreHyper2Samples=[];
covvy.explored=zeros(0,numel(covvy.hyperparams));%cat(1,covvy.hypersamples.hyperparameters);

% Initialises hyper2samples
covvy=hyper2params(covvy);

actual_data=~any(isnan(XsFull),2);
NDims=length(unique(XsFull(actual_data,1)));
NData=size(XsFull,1);
NSamples=numel(covvy.hypersamples);
times=unique(XsFull(actual_data,2));
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

for ind=training_set_size+1:NData
    
    if plot_some_inds
        if ismember(ind,plot_inds)
            covvy.plots = true;
        else
            covvy.plots = false;
        end
    end

    
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
            XData(1:dropped, :) = [];
            YData(1:dropped ,:) = [];
            covvy = gpparams(XData, YData, covvy, 'downdate', 1:dropped);
        end

        XData=[XData;X];
        YData=[YData;Y]; 
        
        covvy=gpparams(XData, YData, covvy, 'update', size(XData,1), ...
            setdiff(1:NSamples,covvy.lastHyperSampleMoved));
    end
    
    likelihood_fn = @(covvy,index) track_likelihood_fullfn(XData,YData,covvy,index);
    covvy = likelihood_fn(covvy,covvy.lastHyperSampleMoved);
    
    
    % nexttime is the time at which we will next get a reading
    currenttime=XData(end,2);
    if ind==NData || any(isnan(XsFull(ind+1,2)))
        nexttime=currenttime+delt;
    else
        nexttime=XsFull(ind+1,2);
    end  
    % XStar is the point at which we wish to make predictions. Note XStar
	% will be empty if we are about to receive more observations at the
	% same time.
    XStar=allcombs({(1:NDims)',(currenttime+lookahead*delt:params.step*delt:nexttime+lookahead*delt)'});    
    XStars=[XStars;XStar]; 
    

    % Use ML_ind from previous time-step -improve_bmc_conditioning could
    % potentially moved to after manage_hyper_samples

%     hypermanaged=false;
%      managed=false;
%     covvy.ss_closeness_num=2.5;
    
    [dropped,covvy] = improve_bmc_conditioning(covvy); 
    managed=~isempty(dropped);
    if managed
        covvy = likelihood_fn(covvy,dropped);
    end
    % if we've just dropped a sample, we have lost information about the
    % likelihood surface, so there's no call for rushing about changing the
    % hyperscales 
    hypermanaged=managed; 
    
    % generate predictions so we can determine the ML input scale for the
    % GP over predictions
    [qs,covvy] = track_q_fullfn(XData,XStar,covvy,1:NSamples);
    
    covvy = calculate_hyper2sample_likelihoods(covvy,qs);
    
    covvy = determine_candidates(covvy, num_cands);
    covvy = bmcparams_ahs(covvy);   
    covvy = sample_candidate_likelihoods(covvy);
    [rho,invRL_wderivs_wcandidates]=weights_ahs(covvy); 
    
    



    [hyperrho,covvy]=hyperweights(rho,qs,covvy);
    
    monitor.t(ind).hyperrho=hyperrho;
    monitor.t(ind).prerho=rho;    
    monitor.t(ind).rho=hyperrho;
    
    [wm,wC,covvy]=weighted_gpmeancov(hyperrho,XStar,XData,covvy,'load');

    wsig=sqrt(wC); 
    
    YMean=[YMean;wm];
    YSD=[YSD;wsig];
    
    [m,closestInd]=max(hyperrho);
    
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
    


    
    ML_ind=covvy.ML_hyper2sample_ind;
    monitor.t(ind).hyper2samples=cat(1,covvy.hyper2samples.hyper2parameters);
    monitor.t(ind).hyper2logL=cat(1,covvy.hyper2samples.logL);
    monitor.t(ind).hypersamples=cat(1,covvy.hypersamples.hyperparameters);
    [cat(1,covvy.hypersamples.logL),cat(1,covvy.hypersamples.hyperparameters)]
    monitor.t(ind).closest_hypersamples=covvy.hypersamples(closestInd).hyperparameters;
    
    monitor.t(ind).ML_hyper2sample=covvy.hyper2samples(ML_ind).hyper2parameters;
    
    [cat(1,covvy.hyper2samples.tilda_logL),cat(1,covvy.hyper2samples.hyper2parameters)]
    
    monitor.t(ind).ML_hyper2sample;
    
    monitor.t(ind).HyperSample_justmoved=covvy.lastHyperSampleMoved;
    monitor.t(ind).Hyper2Sample_justmoved=covvy.lastHyper2SamplesMoved;
    monitor.t(ind).cond=cond(covvy.hyper2samples(ML_ind).cholK_wderivs'*covvy.hyper2samples(ML_ind).cholK_wderivs);
    
    if ~hypermanaged 
        % if we manage hyper2samples before hypersamples, it ensures that
        % added hypersamples are not going to be poorly conditioned
        covvy2 = manage_hyper2_samples(covvy,'move');
    	%hyperman1=[cat(1,covvy.hyper2samples.logL),cat(1,covvy.hyper2samples.hyper2parameters)]
    end
    if ~managed % don't do this if improve_bmc_conditioning has already moved samples

        % If we have been asked to make predictions about more than one point, we
        % arbitrarily choose to minimise our uncertainty around the first.
%         pt=wm(1);
%         %pt=Lvec'*q_mean_star(:,1)/sum(Lvec);
%         qs=normpdf(repmat(pt,NSamples,1),q_mean_star(:,1),q_SD_star(:,1));
        
        [no_change,no_change_uncertainty,covvy] = manage_hyper_samples(covvy,covvy2,qs,invRL_wderivs_wcandidates);
        no_change_uncertainty
        monitor.t(ind).no_change_uncertainty=no_change_uncertainty;
        if no_change
            [smallified,covvy] = zoom_hyper_samples(covvy);
    		%zoom1=[cat(1,covvy.hyper2samples.logL),cat(1,covvy.hyper2samples.hyper2parameters)]
            
            if smallified
                % put a gradient on those really small hyper2samples and
                % then move them
                moved = covvy.lastHyperSampleMoved;
                covvy = manage_hyper2_samples(covvy,'clear');
                covvy = likelihood_fn(covvy,moved);
                [qs_moved,covvy] = track_q_fullfn(XData,XStar,covvy,moved);
                qs(moved) = qs_moved(moved);
                covvy = calculate_hyper2sample_likelihoods(covvy,qs);
                covvy = manage_hyper2_samples(covvy,'move');
            end
            
        end
    elseif ~hypermanaged % if hypermanaged, covvy2 dne
        covvy=covvy2;
    end
    if ~hypermanaged 
        covvy = manage_hyper2_samples(covvy,'clear');
    end

    

end


