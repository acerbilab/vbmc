function [YMean,YSD,covvy,closestInd,monitor]=predict_ahs(XStar,XData,YData,num_steps,covvy,params,candidate_combs_template)

num_cands=6;
covvy.manage_h2s_method = 'optimal';

no_monitor = nargout<5;
if nargin<6 || ~isfield(params,'print')
    params.print=1;%not reassuring dots;
end
if nargin<7
    try
        load('candidate_combs_cell','candidate_combs_cell');
        candidate_combs_template=candidate_combs_cell;
    catch
        candidate_combs_template{num_cands}=find_likelihood_samples(zeros(num_cands,1),ones(num_cands,1),100,300,false);
    end
end

if ~isfield(covvy,'plots')
    covvy.plots=false;
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

lowr.LT=true;
uppr.UT=true;

NSamples=numel(covvy.hypersamples);


count=0;

for ind=1:num_steps
    

    
    if params.print==1 && ind>0.01*(1+count)*num_steps
        count=count+1;
        ind
    elseif params.print==0 && ( rem(ind,100) == 0)
        fprintf('.');
    end
        
    
    likelihood_fn = @(covvy,index) track_likelihood_fullfn(XData,YData,covvy,index);
    
    covvy = likelihood_fn(covvy,covvy.lastHyperSampleMoved);
    
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
    
    if ~no_monitor
        monitor.t(ind).hyperrho=hyperrho;
        monitor.t(ind).prerho=rho;    
        monitor.t(ind).rho=hyperrho;

        [m,closestInd]=max(hyperrho);

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
    end
    
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

[m,closestInd]=max(hyperrho);

[YMean,wC,covvy]=weighted_gpmeancov(hyperrho,XStar,XData,covvy,'load');
YSD=sqrt((wC)); 


