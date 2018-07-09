function [integral,covvy,monitor]=integrate_ahs(covvy,q_fn,likelihood_fn,candidate_combs_template,NData)

num_cands=6;
covvy.manage_h2s_method = 'optimal';

if nargin<3
    try
        candidate_combs_template=load(candidate_combs);
    catch
        candidate_combs_template{num_cands}=find_likelihood_samples(zeros(num_cands,1),ones(num_cands,1),100,300,false);
    end
end

if ~isfield(covvy,'plots')
    covvy.plots=false;
end

covvy=hyperparams(covvy);
NSamples=numel(covvy.hypersamples);
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

for ind=1:NData
    ind
    
    qs = q_fn(covvy,1:NSamples);
    covvy = likelihood_fn(covvy,1:NSamples);

    % Use ML_ind from previous time-step -improve_bmc_conditioning could
    % potentially moved to after manage_hyper_samples

%     hypermanaged=false;
     managed=false;
%     covvy.ss_closeness_num=2.5;

    [dropped,covvy] = improve_bmc_conditioning(covvy); 
    managed=~isempty(dropped);
    if managed
        qs_dropped = q_fn(covvy,dropped);
        qs(dropped) = qs_dropped(dropped);
        covvy = likelihood_fn(covvy,dropped);
    end
    % if we've just dropped a sample, we have lost information about the
    % likelihood surface, so there's no call for rushing about changing the
    % hyperscales 
    hypermanaged=managed; 

    covvy = calculate_hyper2sample_likelihoods(covvy,qs);
    
    covvy = determine_candidates(covvy, num_cands);
    covvy = bmcparams_ahs(covvy);   
    covvy = sample_candidate_likelihoods(covvy);
    [rho,invRL_wderivs_wcandidates]=weights_ahs(covvy); 
    [hyperrho,covvy]=hyperweights(rho,qs,covvy);
    
    [cat(1,covvy.hypersamples.logL),cat(1,covvy.hypersamples.hyperparameters)]
    [cat(1,covvy.hyper2samples.tilda_logL),cat(1,covvy.hyper2samples.hyper2parameters)]
  
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
                qs_moved = q_fn(covvy,moved);
                qs(moved) = qs_moved(moved);
                if ind ~= NData
                    covvy = calculate_hyper2sample_likelihoods(covvy,qs);
                    covvy = manage_hyper2_samples(covvy,'move');
                end
            end
            
        end
    elseif ~hypermanaged % if hypermanaged, covvy2 dne
        covvy=covvy2;
    end
    if ~hypermanaged && ind ~= NData
        covvy = manage_hyper2_samples(covvy,'clear');
    end
    
    
    
    monitor.t(ind).hyperrho=hyperrho;
    monitor.t(ind).prerho=rho;    
    monitor.t(ind).rho=hyperrho;   
    ML_ind=covvy.ML_hyper2sample_ind;
    monitor.t(ind).hyper2samples=cat(1,covvy.hyper2samples.hyper2parameters);
    monitor.t(ind).hyper2logL=cat(1,covvy.hyper2samples.logL);
    monitor.t(ind).hypersamples=cat(1,covvy.hypersamples.hyperparameters);
    %monitor.t(ind).closest_hypersamples=covvy.hypersamples(closestInd).hyperparameters;
    monitor.t(ind).ML_hyper2sample=covvy.hyper2samples(ML_ind).hyper2parameters;
    monitor.t(ind).HyperSample_justmoved=covvy.lastHyperSampleMoved;
    monitor.t(ind).Hyper2Sample_justmoved=covvy.lastHyper2SamplesMoved;
    %monitor.t(ind).cond=cond(covvy.hyper2samples(ML_ind).cholK_wderivs'*covvy.hyper2samples(ML_ind).cholK_wderivs);

    

end


                    % we are not going to get a chance to drop that sample,
                    % so dangerous to move hyper2_samples
                    covvy = calculate_hyper2sample_likelihoods(covvy,qs);
                    covvy = determine_candidates(covvy, num_cands);
                    covvy = bmcparams_ahs(covvy);   
                    covvy = sample_candidate_likelihoods(covvy);
                    [rho]=weights_ahs(covvy); 
                    [hyperrho,covvy]=hyperweights(rho,qs,covvy);

                    integral = qs'*hyperrho;
