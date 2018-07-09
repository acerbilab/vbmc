function [no_change,no_change_uncertainty_unscaled,covvy]=manage_hyper_samples(covvy,covvy2,qs,invRL_wderivs_wcandidates_original)
% Drop one hypersample, add another

if ~isfield(covvy,'debug')
    covvy.debug = false;
end
debug = covvy.debug;

lowr.UT=true;
lowr.TRANSA=true;
uppr.UT=true;

active_hp_inds=covvy.active_hp_inds;

candidates=cat(1,covvy.candidates.hyperparameters); 
samples=cat(1,covvy.hypersamples.hyperparameters);
explored=covvy.explored;
samples_and_explored=[samples;explored];
% samples and candidates should be treated according to their different
% closeness nums


hps=active_hp_inds;

num_samples = size(samples,1);
num_hyperparams = length(hps);
num_wderivs=num_samples*(num_hyperparams+1);
num_candidates = size(candidates,1);


%if two samples are closer than this number of input scales we have
%conditioning problems
ss_closeness_num=covvy.ss_closeness_num;
qq_closeness_num=covvy.qq_closeness_num;
%if two samples are closer than this number of input scales we have
%conditioning problems
sc_closeness_num=covvy.sc_closeness_num;

h2s_ind=covvy.ML_hyper2sample_ind;
tilda_h2s_ind=covvy.ML_tilda_hyper2sample_ind;
Q_h2s_ind=covvy.ML_Q_hyper2sample_ind;
tildaQ_h2s_ind=covvy.ML_tildaQ_hyper2sample_ind;

hyperscales=exp(covvy.hyper2samples(h2s_ind).hyper2parameters);
active_hyperscales = hyperscales(active_hp_inds);
tilda_hyperscales=exp(covvy.hyper2samples(tilda_h2s_ind).hyper2parameters);
active_tilda_hyperscales = tilda_hyperscales(active_hp_inds);

% The active stuff here is for conditioning purposes: really I should be
% ensuring that samples are well separated according to all FOUR
% hyperscales yikes.

Q_hyperscales=exp(covvy.hyper2samples(Q_h2s_ind).hyper2parameters);
tildaQ_hyperscales=exp(covvy.hyper2samples(tildaQ_h2s_ind).hyper2parameters);

big_L_scales = max(exp([covvy2.hyper2samples(h2s_ind).hyper2parameters;covvy2.hyper2samples(tilda_h2s_ind).hyper2parameters]));
big_Q_scales = (qq_closeness_num/ss_closeness_num)*max(exp([covvy2.hyper2samples(Q_h2s_ind).hyper2parameters;covvy2.hyper2samples(tildaQ_h2s_ind).hyper2parameters]));
big_scales = max(big_L_scales,big_Q_scales);

scaling_factor = tilda_hyperscales;
active_scaling_factor = scaling_factor(active_hp_inds);
%active_tilda_scaled = active_tilda_hyperscales./active_scaling_factor;
%active_scaled = active_hyperscales./active_scaling_factor;
active_big_L_scaled = big_L_scales(active_hp_inds)./active_scaling_factor;
active_big_Q_scaled = big_Q_scales(active_hp_inds)./active_scaling_factor;
active_big_scaled = max(active_big_L_scaled,active_big_Q_scaled);

scaled_active_cands = candidates(:,active_hp_inds)./repmat(active_scaling_factor,num_candidates,1);
scaled_active_sande = samples_and_explored(:,active_hp_inds)./repmat(active_scaling_factor,size(samples_and_explored,1),1);

priorMeans=[covvy.hyperparams.priorMean];
priorSDs=[covvy.hyperparams.priorSD];

scaled_active_lower_bound = priorMeans - 3*priorSDs;
scaled_active_lower_bound = scaled_active_lower_bound(active_hp_inds)./active_scaling_factor;
scaled_active_upper_bound = priorMeans + 3*priorSDs;
scaled_active_upper_bound = scaled_active_upper_bound(active_hp_inds)./active_scaling_factor;

hyper3scale = covvy.hyper3scale;
hyper3jitter = covvy.hyper3jitter;

h_L=covvy.hyper2samples(h2s_ind).likelihood_scale;
h_tildaL=covvy.hyper2samples(tilda_h2s_ind).tilda_likelihood_scale;
% This h_L is only of relevance with respect to the scaling factor for the
% likelihoods maxlogLvec computed below
h_tildaQ=covvy.hyper2samples(tildaQ_h2s_ind).tilda_prediction_scale;

candidate_Cov=covvy.hyper2samples(tilda_h2s_ind).Cov_tildaL;
candidate_SDs=sqrt(diag(candidate_Cov));
candidate_Means=covvy.hyper2samples(tilda_h2s_ind).Mean_tildaL;
candidate_combs_template = covvy.candidate_combs_template;

% This code is copied from that in calculate_hyper2sample_likelihoods
[logLcell{1:num_samples}]=covvy.hypersamples.logL;
logLvec=cat(1,logLcell{:});
[maxlogLvec,Best_sample]=max(logLvec);
logLvec=(logLvec-maxlogLvec); 
mean_tildal=covvy.mean_tildal;
logLvec2=logLvec-mean_tildal;

Lvec=exp(logLvec);%,eps);
[glogLcell{1:num_samples}]=covvy.hypersamples(:).glogL; % actually glogl is a cell itself
glogLmat=fliplr(cell2mat(cat(2,glogLcell{:}))');
glogLmat=glogLmat(:,end+1-active_hp_inds);
gLmat=fliplr(repmat(Lvec,1,size(glogLmat,2))).*glogLmat; %fliplr because derivs are actually in reverse order. NB: multiplying by Lvec takes care of the scale factor

LData=[Lvec,gLmat]';
LData=LData(:);

%glogLmat_active=glogLmat(:,end+1-active_hp_inds);

candidate_tilda_likelihoods=covvy.candidate_tilda_likelihood_combs;
candidate_likelihoods = exp(candidate_tilda_likelihoods);%,eps);
num_candfns = size(candidate_tilda_likelihoods,2);

max_cand_tildaL = covvy.max_cand_tildaL;

  
% These are absent h_L^2
K_wderivs_wcandidates=covvy.hyper2samples(h2s_ind).K_wderivs_wcandidates; % from bmcparams_ahs
cholK=covvy.hyper2samples(Q_h2s_ind).cholK; % from calculate_hyper2sample_likelihoods
cholK_wderivs=covvy.hyper2samples(h2s_ind).cholK_wderivs; % from calculate_hyper2sample_likelihoods
cholK_wderivs_wcandidates=covvy.hyper2samples(h2s_ind).cholK_wderivs_wcandidates; % from bmcparams_ahs

tilda_cholK=covvy.hyper2samples(tildaQ_h2s_ind).cholK; % from calculate_hyper2sample_likelihoods
tilda_cholK_wderivs=covvy.hyper2samples(tilda_h2s_ind).cholK_wderivs; % from calculate_hyper2sample_likelihoods
tilda_K_wderivs_wcandidates=covvy.hyper2samples(tilda_h2s_ind).K_wderivs_wcandidates; % from bmcparams_ahs
tilda_cholK_wderivs_wcandidates=covvy.hyper2samples(tilda_h2s_ind).cholK_wderivs_wcandidates; % from bmcparams_ahs


rearrange=covvy.rearrange_withcands;

N=covvy.hyper2samples(Q_h2s_ind).N;
n=covvy.hyper2samples(h2s_ind).n;
invRL_wderivs=covvy.hyper2samples(h2s_ind).datahalf;
invKL_wderivs=covvy.hyper2samples(h2s_ind).datatwothirds;
invRN=covvy.hyper2samples(Q_h2s_ind).invRN;

tilda_invKL_wderivs=covvy.hyper2samples(tilda_h2s_ind).datatwothirds;

qinvR = linsolve(cholK,qs,lowr)'; % no input scale over Q used - multiplicative constant to exp_uncertainty
qs_scale=covvy.qs_scale;%max(qs);
tilda_qs=log(qs)-log(qs_scale);
mean_tildaq=min(min(tilda_qs),-2); % not an error - the mean of the GP is the min of the observations
tilda_qs2=tilda_qs-mean_tildaq;
tilda_invKq = solve_chol(tilda_cholK,tilda_qs2);





sc_distance_matrix=sqrt(squared_distance(samples, candidates, hyperscales));
tilda_sc_distance_matrix=sqrt(squared_distance(samples, candidates, tilda_hyperscales));
[worst_sc_distance,worst_candidate_ind]=max(max(sc_distance_matrix));
% Now remove this worst candidate for 'prospective' and 'no_change'
% calculations of the expected uncertainty. Hence 'prospective',
% 'no_change' and 'candidate' calculations will employ the same number of
% candidates.

num_candidates_minuscand = num_candidates-1;
all_but_worst=setdiff((1:num_candidates),worst_candidate_ind);
candidates_minuscand = candidates(all_but_worst,:);
candidate_Means_minuscand=candidate_Means(all_but_worst);
candidate_Cov_minuscand=candidate_Cov(all_but_worst,all_but_worst);
candidate_SDs_minuscand=sqrt(diag(candidate_Cov_minuscand));


if num_candidates_minuscand>length(candidate_combs_template)
    candidate_combs_template{num_candidates_minuscand}=find_likelihood_samples(zeros(num_candidates_minuscand,1),ones(num_candidates_minuscand,1),2^num_candidates_minuscand,300,false);
end
num_candfns_minuscand = size(candidate_combs_template{num_candidates_minuscand},2);

num_SDs = min((-min(candidate_Means_minuscand,0)+max_cand_tildaL)./(max(candidate_combs_template{num_candidates_minuscand},[],2).*candidate_SDs_minuscand));
num_SDs = min(num_SDs,hyper3scale);

candidate_tilda_likelihoods_minuscand = candidate_combs_template{num_candidates_minuscand}.*repmat(candidate_SDs_minuscand,1,num_candfns_minuscand)*num_SDs + repmat(min(candidate_Means_minuscand,0),1,num_candfns_minuscand);
candidate_likelihoods_minuscand = exp(candidate_tilda_likelihoods_minuscand);%,eps);

K_wderivs_wcandidates_minuscand = K_wderivs_wcandidates;
K_wderivs_wcandidates_minuscand(num_wderivs+worst_candidate_ind,:)=[];
K_wderivs_wcandidates_minuscand(:,num_wderivs+worst_candidate_ind)=[];
cholK_wderivs_wcandidates_minuscand = downdatechol(cholK_wderivs_wcandidates,num_wderivs+worst_candidate_ind);

likelihoods_wderivs_wcandidates_minuscand=nan(num_wderivs+num_candidates_minuscand,num_candfns_minuscand);
likelihoods_wderivs_wcandidates_minuscand(num_wderivs+(1:num_candidates_minuscand),:)=candidate_likelihoods_minuscand;
invRL_wderivs_wcandidates_minuscand = updatedatahalf(cholK_wderivs_wcandidates_minuscand,likelihoods_wderivs_wcandidates_minuscand,repmat(invRL_wderivs,1,num_candfns_minuscand),cholK_wderivs,num_wderivs+(1:num_candidates_minuscand));
N_minuscand=N;
N_minuscand(:,num_wderivs+worst_candidate_ind)=[];
invRN_minuscand=invRN;
invRN_minuscand(:,num_wderivs+worst_candidate_ind)=[];
n_minuscand=n;
n_minuscand(num_wderivs+worst_candidate_ind)=[];
rearrange_minuscand = rearrange(1:end-1);


% recalibrate hyper3jitter and hyper3scale for our reduced set of
% candidates
    
    invKL_wderivs_wcandidates_minuscand = linsolve(cholK_wderivs_wcandidates_minuscand,invRL_wderivs_wcandidates_minuscand,uppr);

    mus_minuscand = (qinvR*invRN_minuscand*invKL_wderivs_wcandidates_minuscand)./ ...
                        (n_minuscand'*invKL_wderivs_wcandidates_minuscand);

    scales=10.^(linspace(0,0.7,20))';

    num_scales=length(scales);
    logL=nan(num_scales,1);
    jitters=nan(num_scales,1);
    for ind=1:num_scales;

        scale=scales(ind);

        mat=ones(num_candfns_minuscand);
        for candidate_ind=1:num_candidates_minuscand

             mat_candidate=matrify(@(x,y) normpdf(x,y,scale*candidate_SDs_minuscand(candidate_ind)),...
                                 candidate_tilda_likelihoods_minuscand(candidate_ind,:)',...
                                 candidate_tilda_likelihoods_minuscand(candidate_ind,:)');

             mat=mat.*mat_candidate;
        end


        mat_base=mat;
        jitterstep=0.02*sqrt(mat_base(1));
        jitter=jitterstep;
        while cond(mat)>100
            mat=mat_base+eye(length(mat))*jitter^2;
            jitter=jitter+jitterstep;
        end
        jitters(ind)=jitter-jitterstep;


        cholK_otrials_otrials=chol(mat);

        datahalf = linsolve(cholK_otrials_otrials, mus_minuscand', lowr);
        datahalf_all = datahalf(:);
        NData = length(datahalf_all);

        % the ML solution for h_L can be computed analytically:
        h_mu = sqrt((datahalf_all'*datahalf_all)/NData);

        % Maybe better stability elsewhere would result from sticking in this
        % output scale earlier?
        cholK_otrials_otrials=h_mu*cholK_otrials_otrials; 
        %datahalf_all=(h_mu)^(-1)*datahalf_all;

        logsqrtInvDetSigma = -sum(log(diag(cholK_otrials_otrials)));
        quadform = NData;%sum(datahalf_all.^2, 1);
        logL(ind) = -0.5 * NData * log(2 * pi) + logsqrtInvDetSigma -0.5 * quadform; 
    end

    [max_logL,max_ind]=max(logL);
    hyper3scale=scales(max_ind); % ML
    hyper3jitter=jitters(max_ind);
    
    num_SDs = min((-min(candidate_Means_minuscand,0)+max_cand_tildaL)./(max(candidate_combs_template{num_candidates_minuscand},[],2).*candidate_SDs_minuscand));
    num_SDs = min(num_SDs,hyper3scale);
    
    candidate_tilda_likelihoods_minuscand = min(max_cand_tildaL,candidate_combs_template{num_candidates_minuscand}.*repmat(candidate_SDs_minuscand,1,num_candfns_minuscand)*num_SDs + repmat(min(candidate_Means_minuscand,0),1,num_candfns_minuscand));
    %candidate_likelihoods_minuscand = max(exp(candidate_tilda_likelihoods_minuscand),eps);
    candidate_likelihoods_minuscand = exp(candidate_tilda_likelihoods_minuscand);
    
    likelihoods_wderivs_wcandidates_minuscand=nan(num_wderivs+num_candidates_minuscand,num_candfns_minuscand);
    likelihoods_wderivs_wcandidates_minuscand(num_wderivs+(1:num_candidates_minuscand),:)=candidate_likelihoods_minuscand;
    
    invRL_wderivs_wcandidates_minuscand = ...
        updatedatahalf(cholK_wderivs_wcandidates_minuscand,likelihoods_wderivs_wcandidates_minuscand,repmat(invRL_wderivs,1,num_candfns_minuscand),cholK_wderivs,num_wderivs+(1:num_candidates_minuscand));
  
    % redo candidate_likelihoods to go with new hyper3scale
    
    num_SDs = min((-min(candidate_Means,0)+max_cand_tildaL)./(max(candidate_combs_template{num_candidates},[],2).*candidate_SDs));
    num_SDs = min(num_SDs,hyper3scale);
    
    candidate_tilda_likelihoods = candidate_combs_template{num_candidates}.*repmat(candidate_SDs,1,num_candfns)*num_SDs + repmat(min(candidate_Means,0),1,num_candfns);
    %candidate_likelihoods = max(exp(candidate_tilda_likelihoods),eps);
    candidate_likelihoods = exp(candidate_tilda_likelihoods);
    
    likelihoods_wderivs_wcandidates = nan(num_wderivs+num_candidates,num_candfns);
    likelihoods_wderivs_wcandidates(num_wderivs+(1:num_candidates),:) = candidate_likelihoods;
    
    invRL_wderivs_wcandidates = ...
        updatedatahalf(cholK_wderivs_wcandidates,likelihoods_wderivs_wcandidates,repmat(invRL_wderivs,1,num_candfns),cholK_wderivs,num_wderivs+(1:num_candidates));

    invKL_wderivs_wcandidates_minuscand = linsolve(cholK_wderivs_wcandidates_minuscand,invRL_wderivs_wcandidates_minuscand,uppr);
    mus_minuscand = (qinvR*invRN_minuscand*invKL_wderivs_wcandidates_minuscand)./ ...
                        (n_minuscand'*invKL_wderivs_wcandidates_minuscand);
    
    
mat=ones(num_candfns_minuscand);
for candidate_ind=1:num_candidates_minuscand

     mat_candidate=matrify(@(x,y) normpdf(x,y,hyper3scale*candidate_SDs_minuscand(candidate_ind)),...
                         candidate_tilda_likelihoods_minuscand(candidate_ind,:)',...
                         candidate_tilda_likelihoods_minuscand(candidate_ind,:)');

     mat=mat.*mat_candidate;
end
mat = mat+eye(length(mat))*hyper3jitter^2;

width_plus_conditional_covariance = diag((hyper3scale*candidate_SDs_minuscand).^2)+candidate_Cov_minuscand;
arm=candidate_tilda_likelihoods_minuscand-repmat(candidate_Means_minuscand,1,num_candfns_minuscand);
vect=(det(2*pi*width_plus_conditional_covariance))^(-0.5)*...
        exp(-0.5*sum(arm.*(width_plus_conditional_covariance\arm),1));

combs_weights_minuscand = vect/mat;

   negative=mus_minuscand<0;  
   
   
%     non_zero_mean=max(mean(mus_minuscand(~negative)),0);
%     mus_minuscand(negative)=non_zero_mean;

no_change_uncertainty = -(mus_minuscand*combs_weights_minuscand')^2;
no_change_uncertainty_unscaled=no_change_uncertainty;

uncertainty_scale=1;%abs(no_change_uncertainty);
no_change_uncertainty=uncertainty_scale^(-1)*no_change_uncertainty_unscaled;


if covvy.plots && length(active_hp_inds) == 1
    scaled_h_tildaL=h_tildaL/sqrt(sqrt(prod(2*pi*active_tilda_hyperscales.^2)));
    scaled_h_L=h_L/sqrt(sqrt(prod(2*pi*active_hyperscales.^2)));

    figure(3)
    clf
    hold on
    plot_GP_wderivs(samples(:,active_hp_inds),logLvec2,glogLmat,active_tilda_hyperscales,scaled_h_tildaL,candidates_minuscand(:,active_hp_inds),candidate_tilda_likelihoods_minuscand-mean_tildal);
    % Note both logLvec and candidate_tilda_likelihoods_minuscand are
    % -mean_tildal
    ylabel('log L','FontName','Times','FontSize',22,'Rotation',0);

    if isfield(covvy,'real_logLs')
        real_logLs=covvy.real_logLs;
        real_hps=covvy.real_hps;
        [a,b]=min(abs(real_hps(:,active_hp_inds)-samples(Best_sample,active_hp_inds)));
        real_logLs2=real_logLs-real_logLs(b)-mean_tildal;
        plot((real_hps(:,active_hp_inds)),real_logLs2,'k','LineWidth',1);
    end
    
        figure(4)
    clf
    hold on
    plot_GP_wderivs(samples(:,active_hp_inds),Lvec,gLmat,active_hyperscales,scaled_h_L,candidates_minuscand(:,active_hp_inds),candidate_likelihoods_minuscand);
    % Note both logLvec and candidate_tilda_likelihoods_minuscand are
    % -mean_tildal
    ylabel('L','FontName','Times','FontSize',22,'Rotation',0);
    
    if isfield(covvy,'real_logLs')
        [a,b]=min(abs(real_hps(:,active_hp_inds)-samples(Best_sample,active_hp_inds)));
        real_logLs2=real_logLs-real_logLs(b);
        plot((real_hps(:,active_hp_inds)),exp(real_logLs2),'k','LineWidth',1);
    end
    
end

if isnan(no_change_uncertainty_unscaled) || any(negative)
    if (debug); disp('no change uncertainty is nan'); end
    get_outta_here;
    no_change_uncertainty_unscaled=nan;
    return;
end


opts.MaxFunEvals=80; % Parameter ahoy
if (debug)
  opts.Display='on'; 
else
  opts.Display = 'off';
end
opts.LargeScale='off';
opts.Algorithm='active-set';

%[max_L,Best_sample] = max(covvy.hyper);
Current_sample = get_hyper_samples_to_move(covvy);

% Step size for gradient ascent
step_size=ss_closeness_num;

if num_candidates>1
    far_points = cat(1, covvy.candidates.hyperparameters);
else
    far_points=[];
end

num_far_points = size(far_points,1);


% bad_candidate_inds are fine as candidates, but can't be considered as
% trial sample locations, they're too close to existing samples
[sample_inds,bad_candidate_inds]=find(sc_distance_matrix<ss_closeness_num);
[sample_inds,bad_candidate_inds2]=find(tilda_sc_distance_matrix<ss_closeness_num);
bad_candidate_inds = [bad_candidate_inds;bad_candidate_inds2];

lower_bound = priorMeans - covvy.box_nSDs*priorSDs;
upper_bound = priorMeans + covvy.box_nSDs*priorSDs;
outside_box = (candidates < repmat(lower_bound,num_candidates,1)) +...
                (candidates > repmat(upper_bound,num_candidates,1));
bad_candidate_inds=[bad_candidate_inds;find(any(outside_box,2))];

far_point_inds = setdiff(1:num_far_points,bad_candidate_inds);

num_far_points = size(far_point_inds,1);
num_starting_points = num_far_points + 1 + (Best_sample~=Current_sample);

if length(active_hp_inds) == 1
    AHS_plots_new

    if isempty(unc) %|| any(isnan(unc))
        % no permissible points to visit!
        if (debug); disp('no eligible sample to add'); end
        get_outta_here;
        return;
    end
    [min_uncertainty,min_ind]=min(unc);
    period=periods(min_ind);

    if covvy.plots
    figure(1)
    addedplot=plot(period,min_uncertainty,'r+','MarkerSize',7,'LineWidth',2);
    end

    if isempty(period)
        % No eligible sample to add
        if (debug); disp('no eligible sample to add (2)'); end
        get_outta_here;
        return;
    end
    
    added_hypersample=samples(1,:);
    added_hypersample(active_hp_inds)=period;

    exploration=min_ind>num;
    if exploration; if (debug); disp('explore'); end; explore_candidate_ind=min_ind-num;end

else

    added_hypersamples=nan(num_starting_points,numel(covvy.hyperparams));
    added_uncertainty=nan(num_starting_points,1);
    start=0;
    
    
    zero_inds = sum(glogLmat'.^2)' == 0;
    % move towards the mean
    glogLmat(zero_inds,:)=repmat(priorMeans(active_hp_inds),sum(zero_inds),1)-samples(zero_inds,active_hp_inds);

    % Now we find the optimal sample to add to the current set.

    for climb_sample_ind=unique([Best_sample,Current_sample])

        % Try climbing the gradient as a starting point for a local minimiser
        % of the expected uncertainty

        current_gradient = glogLmat(climb_sample_ind,:); % glogLmat is just over active inds
        starting_pt_active = gradient_ascent(samples(climb_sample_ind,active_hp_inds), logLvec(climb_sample_ind), current_gradient, step_size, active_tilda_hyperscales);
        starting_pt = samples(climb_sample_ind,:);
        starting_pt(:,active_hp_inds) = starting_pt_active;

        start_close_to=notnear2(starting_pt,samples_and_explored,big_scales,ss_closeness_num,candidates,big_L_scales,sc_closeness_num)>0;
        step_size_loop=step_size;
        while any(any(start_close_to))
            new_ind=find(start_close_to,1);
            if new_ind<=num_samples
                current_gradient = glogLmat(new_ind,:); % we know what the gradient is in the vicinity of this point
            end
            if new_ind<=size(samples_and_explored,1);
                current_location = samples_and_explored(new_ind,active_hp_inds);
            else
                current_location = candidates(new_ind-size(samples_and_explored,1),active_hp_inds);
            end
            starting_pt_active = gradient_ascent(current_location, 0, current_gradient, step_size_loop, active_tilda_hyperscales);
            starting_pt(:,active_hp_inds) = starting_pt_active;
            start_close_to=notnear2(starting_pt,samples_and_explored,big_scales,ss_closeness_num,candidates,big_L_scales,sc_closeness_num)>0;
            step_size_loop=step_size_loop*1.01;
        end

        % fminunc sucks. Try putting in input scales to take better gradient
        % ascent steps, also divide by no_change_uncertainty to get a function more
        % on the order of one.


        start=start+1;
        added_hypersamples(start,:)=starting_pt; 
        scaled_active_starting_pt = starting_pt(active_hp_inds)./active_scaling_factor;
        [added_hypersamples(start,active_hp_inds),added_uncertainty(start)]=fmincon(@(trial_active) ...
            uncertainty_scale^(-1)*expected_uncertainty(active2hypersample(trial_active,active_hp_inds,starting_pt,scaling_factor),...
                samples,h_tildaL,h_tildaQ,qs_scale,cholK,tilda_cholK,cholK_wderivs,tilda_cholK_wderivs,cholK_wderivs_wcandidates_minuscand,invRL_wderivs,tilda_invKL_wderivs,hyperscales,tilda_hyperscales,Q_hyperscales,tildaQ_hyperscales,hps,rearrange_minuscand,n_minuscand,invRN_minuscand,qinvR,tilda_invKq,mean_tildaq,mean_tildal,invRL_wderivs_wcandidates_minuscand,candidates_minuscand,priorMeans,priorSDs,candidate_likelihoods_minuscand,K_wderivs_wcandidates_minuscand,combs_weights_minuscand,candidate_Means_minuscand,candidate_Cov_minuscand,candidate_combs_template,hyper3scale,hyper3jitter,max_cand_tildaL,...
                    @(x) notnear2(x(active_hp_inds) ./ active_scaling_factor,scaled_active_sande,active_big_scaled,ss_closeness_num,scaled_active_cands,active_big_L_scaled,sc_closeness_num)),...
                    scaled_active_starting_pt,[],[],[],[],...
                    scaled_active_lower_bound,...
                    scaled_active_upper_bound, ...
                        [],opts);
        added_hypersamples(start,active_hp_inds)=added_hypersamples(start,active_hp_inds).*active_scaling_factor;

        %opts.MaxFunEvals=20; % Parameter ahoy
    end

% start = 1;
% starting_pt = samples(Best_sample, :);
% 
% Prob.samples = samples;
% Prob.h_tildaL = h_tildaL;
% Prob.h_tildaQ = h_tildaQ;
% Prob.qs_scale = qs_scale;
% Prob.cholK = cholK;
% Prob.tilda_cholK = tilda_cholK;
% Prob.cholK_wderivs = cholK_wderivs;
% Prob.tilda_cholK_wderivs = tilda_cholK_wderivs;
% Prob.cholK_wderivs_wcandidates_minuscand = cholK_wderivs_wcandidates_minuscand;
% Prob.invRL_wderivs = invRL_wderivs;
% Prob.tilda_invKL_wderivs = tilda_invKL_wderivs;
% Prob.hyperscales = hyperscales;
% Prob.tilda_hyperscales = tilda_hyperscales;
% Prob.Q_hyperscales = Q_hyperscales;
% Prob.tildaQ_hyperscales = tildaQ_hyperscales;
% Prob.hps = hps;
% Prob.rearrange_minuscand = rearrange_minuscand;
% Prob.n_minuscand = n_minuscand;
% Prob.invRN_minuscand = invRN_minuscand;
% Prob.qinvR = qinvR;
% Prob.tilda_invKq = tilda_invKq;
% Prob.mean_tildaq = mean_tildaq;
% Prob.mean_tildal = mean_tildal;
% Prob.invRL_wderivs_wcandidates_minuscand = invRL_wderivs_wcandidates_minuscand;
% Prob.candidates_minuscand = candidates_minuscand;
% Prob.priorMeans = priorMeans;
% Prob.priorSDs = priorSDs;
% Prob.candidate_likelihoods_minuscand = candidate_likelihoods_minuscand;
% Prob.K_wderivs_wcandidates_minuscand = K_wderivs_wcandidates_minuscand;
% Prob.combs_weights_minuscand = combs_weights_minuscand;
% Prob.candidate_Means_minuscand = candidate_Means_minuscand;
% Prob.candidate_Cov_minuscand = candidate_Cov_minuscand;
% Prob.candidate_combs_template = candidate_combs_template;
% Prob.hyper3scale = hyper3scale;
% Prob.hyper3jitter = hyper3jitter;
% Prob.max_cand_tildaL = max_cand_tildaL;
% 
% Problem.f = @(x) uncertainty_scale^(-1) * ...
%   expected_uncertainty_wrapper(active2hypersample(x', active_hp_inds, starting_pt, scaling_factor), Prob) + ...
%   1000 * ...
%   min(0, notnear2(x, scaled_active_sande, active_big_scaled, ss_closeness_num, scaled_active_cands, active_big_L_scaled, sc_closeness_num));
% 
% opts.maxevals = 1000;
% opts.showits = 0;
% bounds = [lower_bound(active_hp_inds) ./active_scaling_factor ; upper_bound(active_hp_inds) ./active_scaling_factor]';
%   
% added_hypersamples(start, active_hp_inds) = added_hypersamples(start, active_hp_inds) .* active_scaling_factor;
% 
% [added_hypersamples(start, active_hp_inds) added_uncertainty(start)] = Direct(Problem, bounds, opts);


    for far_point_ind = far_point_inds
        start=start+1;
        added_hypersamples(start,:) = far_points(far_point_ind,:);
        added_uncertainty(start) = uncertainty_scale^(-1)*expected_uncertainty({far_point_ind},samples,h_tildaL,h_tildaQ,qs_scale,cholK,tilda_cholK,cholK_wderivs,tilda_cholK_wderivs,cholK_wderivs_wcandidates,invRL_wderivs,tilda_invKL_wderivs,hyperscales,tilda_hyperscales,Q_hyperscales,tildaQ_hyperscales,hps,rearrange,n,invRN,qinvR,tilda_invKq,mean_tildaq,mean_tildal,invRL_wderivs_wcandidates_original,candidates,priorMeans,priorSDs,candidate_likelihoods,K_wderivs_wcandidates,[],candidate_Means,candidate_Cov,candidate_combs_template,hyper3scale,hyper3jitter,max_cand_tildaL);
    %     [added_hypersamples(start,:),uncertainty(start)]=fminunc(@(trial_hypersample) ...
    %         expected_uncertainty(trial_hypersample,...
    %             cholK,N,invRN,toMove,invKL,hyperrho,hps,samples,...
    %             candidates,inputscales,priorMeans,priorSDs,rearrange),...
    %             starting_points(start,:),opts);
    end

    [min_uncertainty,add_ind]=min(added_uncertainty);
    added_hypersample=real(added_hypersamples(add_ind,:)); % occasionally get a really really really small imaginary part?


    exploration=add_ind>num_starting_points-num_far_points;
    if exploration && debug; disp('explore');end
    explore_candidate_ind=add_ind-(num_starting_points-num_far_points);

    % Given that added_hypersample, we determine which of the current samples
    % we should drop. 

    % If we just added an exploratory point (one of the
    % candidates) that candidate has to be retired to avoid conditioning
    % errors.
end

 

if isnan(min_uncertainty) || any(isnan(added_hypersample))
    if (debug); disp('the sample we wish to add (or its uncertainty) is nan'); end
    get_outta_here;
    return;
end


% Numbers AFTER downdating
num_samples_after_drop = num_samples-1;
if exploration
    num_wderivs_wcandidates_after_drop = num_samples_after_drop*(num_hyperparams+1)+num_candidates;
else % we drop a candidate again 
    num_wderivs_wcandidates_after_drop = num_samples_after_drop*(num_hyperparams+1)+num_candidates_minuscand;
    num_candidates=num_candidates_minuscand;
    K_wderivs_wcandidates = K_wderivs_wcandidates_minuscand;
    cholK_wderivs_wcandidates = cholK_wderivs_wcandidates_minuscand;
    num_candfns = num_candfns_minuscand;
    candidate_likelihoods = candidate_likelihoods_minuscand;
    N = N_minuscand;
    n = n_minuscand;
end

candidate_inds=num_wderivs_wcandidates_after_drop-num_candidates+1:num_wderivs_wcandidates_after_drop;
% Have to redo this to account for downdating, code copied from
% bmcparams_ahs
rearrange=bsxfun(@plus,(0:num_samples_after_drop:num_hyperparams*num_samples_after_drop)',(1:num_samples_after_drop));
rearrange=rearrange(:);
rearrange=[rearrange;max(rearrange)+(1:num_candidates)'];


dropped_uncertainty=nan(num_samples,1);
for toDrop=1:num_samples %num_samples is NOT updated to include the added one
    Drop_inds=(toDrop-1)*(num_hyperparams+1)+(1:num_hyperparams+1);
    
    samples_new=samples;
    samples_new(toDrop,:)=[];

    % downdate inv(chol(K_wderivs)')*L_wderivs
    % then update to get
    % inv(chol(K_wderivs_wcandidates)')*L_wderivs_wcandidates

    % downdate
    cholK_new = downdatechol(cholK,toDrop);
    cholK_wderivs_new = downdatechol(cholK_wderivs,Drop_inds);
    cholK_wderivs_wcandidates_new = downdatechol(cholK_wderivs_wcandidates,Drop_inds);
    
    tilda_cholK_new = downdatechol(tilda_cholK,toDrop);
    tilda_cholK_wderivs_new = downdatechol(tilda_cholK_wderivs,Drop_inds);

    logLvec_new=logLvec;
    logLvec_new(toDrop)=[];
    logLvec_new = logLvec_new - max(logLvec_new);
    mean_tildal_new = mean_tildal;
    %mean_tildal_new = min(logLvec_new);
    logLvec2_new = logLvec_new-mean_tildal;

    glogLmat_new=glogLmat;
    glogLmat_new(toDrop,:)=[];
    tildaL_wderivs_new=[logLvec2_new,glogLmat_new]';
    tildaL_wderivs_new=tildaL_wderivs_new(:);
    NLData = size(tildaL_wderivs_new, 1);
    
    Lvec_new = exp(logLvec_new);
    gLmat_new=fliplr(repmat(Lvec_new,1,size(glogLmat_new,2))).*glogLmat_new;
    
    L_wderivs_new=[Lvec_new,gLmat_new]';
    L_wderivs_new=L_wderivs_new(:);

    % These are absent h_L^2 and include the likelihood scaling factor
    % invRL_wderivs = inv(chol(K_wderivs)')*L_wderivs
    invRL_wderivs_new=linsolve(cholK_wderivs_new,L_wderivs_new,lowr);
    % invKL_wderivs = inv(K_wderivs)'*L_wderivs
    tilda_invKL_wderivs_new=solve_chol(tilda_cholK_wderivs_new,tildaL_wderivs_new);
    
    quad = (tilda_invKL_wderivs_new'*tilda_invKL_wderivs_new);
    % the ML solution for h_L can be computed analytically:
    SD=covvy.SD_factor_tildaL*sqrt(quad/NLData);
    b=(0.5*NLData*SD^2);
    h_tildaL_new = sqrt(-b+sqrt(b^2+SD^2*quad));
    

    % invRL_wderivs_wcandidates = inv(chol(K_wderivs_wcandidates)')*L_wderivs_wcandidates
    L_wderivs_wcandidates_new=nan(num_wderivs_wcandidates_after_drop,num_candfns);
    L_wderivs_wcandidates_new(candidate_inds,:)=candidate_likelihoods;
    invRL_wderivs_wcandidates_new=updatedatahalf(cholK_wderivs_wcandidates_new,...
        L_wderivs_wcandidates_new,repmat(invRL_wderivs_new,1,num_candfns),...
        cholK_wderivs_new,candidate_inds);

    % downdate
    N_new=N;
    N_new(toDrop,:) = [];
    N_new(:,Drop_inds) = [];
    n_new=n;
    n_new(Drop_inds) = [];
    K_wderivs_wcandidates_new=K_wderivs_wcandidates;
    K_wderivs_wcandidates_new(Drop_inds,:)=[];
    K_wderivs_wcandidates_new(:,Drop_inds)=[];

    % invRN = inv(chol(K)')*N
    invRN_new = linsolve(cholK_new,N_new,lowr);

     % If we have been asked to make predictions about more than one point, we
    % arbitrarily choose to minimise our uncertainty around the first.
    qs_new=qs;
    qs_new(toDrop)=[];
    
    tilda_qs_new=tilda_qs;
    tilda_qs_new(toDrop)=[];
    %mean_tildaq_new = min(tilda_qs_new);
    mean_tildaq_new = mean_tildaq;
    tilda_qs2_new = tilda_qs_new-mean_tildaq_new;
    NQData = size(tilda_qs,1);

    qinvR_new = linsolve(cholK_new,qs_new,lowr)'; % no input scale over Q used - multiplicative constant to exp_uncertainty
    tilda_invKq_new = solve_chol(tilda_cholK_new,tilda_qs2_new);
    

    quad = (tilda_invKq_new'*tilda_invKq_new);
    % the ML solution for h_L can be computed analytically:
    SD=covvy.SD_factor_tildaQ*sqrt(quad/NQData);
    b=(0.5*NQData*SD^2);
    h_tildaQ_new = sqrt(-b+sqrt(b^2+SD^2*quad));
    

    % We don't actually know stuff about the added_hypersample, obviously,
    % so we still have to do an expected_uncertainty call. This is a bit
    % wasteful as in each case we are adding the same sample - we could
    % potentially recycle the update steps - but I am lazy.
    
    if exploration % it is an explore point
        dropped_uncertainty(toDrop)=uncertainty_scale^(-1)*expected_uncertainty({explore_candidate_ind},samples_new,h_tildaL_new,h_tildaQ_new,qs_scale,cholK_new,tilda_cholK_new,cholK_wderivs_new,tilda_cholK_wderivs_new,cholK_wderivs_wcandidates_new,invRL_wderivs_new,tilda_invKL_wderivs_new,hyperscales,tilda_hyperscales,Q_hyperscales,tildaQ_hyperscales,hps,rearrange,n_new,invRN_new,qinvR_new,tilda_invKq_new,mean_tildaq_new,mean_tildal_new,invRL_wderivs_wcandidates_new,candidates,priorMeans,priorSDs,candidate_likelihoods,K_wderivs_wcandidates_new,[],candidate_Means,candidate_Cov,candidate_combs_template,hyper3scale,hyper3jitter,max_cand_tildaL);
    else
        dropped_uncertainty(toDrop)=uncertainty_scale^(-1)*expected_uncertainty(added_hypersample,samples_new,h_tildaL_new,h_tildaQ_new,qs_scale,cholK_new,tilda_cholK_new,cholK_wderivs_new,tilda_cholK_wderivs_new,cholK_wderivs_wcandidates_new,invRL_wderivs_new,tilda_invKL_wderivs_new,hyperscales,tilda_hyperscales,Q_hyperscales,tildaQ_hyperscales,hps,rearrange,n_new,invRN_new,qinvR_new,tilda_invKq_new,mean_tildaq_new,mean_tildal_new,invRL_wderivs_wcandidates_new,candidates_minuscand,priorMeans,priorSDs,candidate_likelihoods_minuscand,K_wderivs_wcandidates_new,combs_weights_minuscand,candidate_Means_minuscand,candidate_Cov_minuscand,candidate_combs_template,hyper3scale,hyper3jitter,max_cand_tildaL);
    end
    
end
   
if all(isnan(dropped_uncertainty))
    if (debug); disp('all uncertainties after dropping a sample are nan'); end
    get_outta_here;
    return;
end
[min_dropped_uncertainty,drop_ind]=min(dropped_uncertainty);

if covvy.plots
figure(1)
droppedplot=plot(samples(:,active_hp_inds),dropped_uncertainty,'ko','MarkerSize',7,'LineWidth',2);

[min_unc,min_ind] = min(dropped_uncertainty);

bestdroppedplot=plot(samples(min_ind,active_hp_inds),dropped_uncertainty(min_ind),'ro','MarkerSize',7,'LineWidth',2);

legend([unchangedplot,expvarplot,addedplot,droppedplot,bestdroppedplot],'No-change',...
    'Trial added',['Best trial',sprintf('\n'),'added'],...
    ['Best trial',sprintf('\n'),'added and',sprintf('\n'),'sample $$$$',sprintf('\n'),'dropped'],...
        ['Best trial',sprintf('\n'),'added and',sprintf('\n'),'worst sample',sprintf('\n'),'dropped'],...
    'Location','EastOutside');
set(gca,'FontSize',22)
cd /homes/51/mosb/Documents/mikesthesis/contents/SBQ
postlaprint(gcf,['integrate2_expunc_',num2str(covvy.plot_ind)],'thesis_tall')
end

if (min_dropped_uncertainty - min_uncertainty)/abs(min_uncertainty-no_change_uncertainty) < -0.1
    % I don't believe that the uncertainty after dropping a sample can be
    % lower than with that sample -- caused by the gp mean over
    % likelihoods/qs dipping below zero
    if (debug); disp('uncertainty after dropping a sample is lower than with that sample'); end
    get_outta_here;
    return;
end

no_change = min_uncertainty>no_change_uncertainty;

if no_change
    if (debug); disp('no change'); end
    covvy.lastHyperSampleMoved=[];
else
    if (debug); drop_ind
    end
     covvy.explored=[explored;covvy.hypersamples(drop_ind).hyperparameters];
     covvy.hypersamples(drop_ind).hyperparameters=added_hypersample;
     covvy.lastHyperSampleMoved=drop_ind;
end

covvy.current_sample=Current_sample;

toMove = covvy2.lastHyper2SamplesMoved;
covvy.lastHyper2SamplesMoved = toMove; 

for i = toMove  
    covvy.hyper2samples(i).hyper2parameters = covvy2.hyper2samples(i).hyper2parameters;
end



%%%%
function hypersample = active2hypersample(active,active_hp_inds,hypersample,inputscales)

hypersample(active_hp_inds)=active.*inputscales(active_hp_inds);

function [tilda_ltsamples,tilda_ltmean,tilda_ltSD,tildaQ_qtmean,tildaQ_qtSD]=sample_GP_t(trial_hypersample,samples,h_tildaL,h_tildaQ,tildaQ_cholK,tilda_cholK_wderivs,tilda_invKL_wderivs,mean_tildal,tildaQ_invKq,mean_tildaq,tilda_hyperscales,tildaQ_hyperscales,hyper3scale,hps,rearrange)
% Note that this function generates l samples using the (potentially) downdated quantities


num_samples=size(samples,1);
num_hyperparams=length(hps);
num_wderivs=num_samples*(num_hyperparams+1);

tilda_K_t_t=1/sqrt(prod(2*pi*tilda_hyperscales(hps).^2));

tilda_K_t_osamples=ones(1,num_samples);
tilda_K_t_osamplesderivs=ones(1,num_wderivs);
ind=0;
for hyperparam=hps
    ind=ind+1;

    width=tilda_hyperscales(hyperparam);
    samples_hp=samples(:,hyperparam);
    trial_hypersample_hp=trial_hypersample(hyperparam);
    
    tilda_K_hp = normpdf(samples_hp,trial_hypersample_hp,width)';
    %matrify(@(x,y) normpdf(x,y,width),trial_hypersample_hp,samples_hp);
    tilda_K_t_osamples=tilda_K_t_osamples.*tilda_K_hp;
    
    % NB: the variable you're taking the derivative wrt is negative -
    % so if y>x for DK_hp below, ie. for the upper right corner of the
    % matrix, we expect there to be negatives
    tilda_DK_hp = ((trial_hypersample_hp-samples_hp)/width^2.*normpdf(trial_hypersample_hp,samples_hp,width))';
        %matrify(@(x,y) (x-y)/width^2.*normpdf(x,y,width),...
        %trial_hypersample_hp,samples_hp); 

    tilda_K_wderivs_hp=repmat(tilda_K_hp,1,num_hyperparams+1);
    inds=(num_hyperparams+1-ind)*num_samples+(1:num_samples); % derivatives are still listed in reverse order for a bit of consistency
    tilda_K_wderivs_hp(:,inds)=-tilda_DK_hp;
    
    tilda_K_t_osamplesderivs=tilda_K_t_osamplesderivs.*tilda_K_wderivs_hp;
end
tilda_K_t_osamplesderivs=tilda_K_t_osamplesderivs(:,rearrange(1:num_wderivs));

% This mean is over scaled likelihood space. h_tildaL's cancel.
tilda_ltmean=mean_tildal+tilda_K_t_osamplesderivs*tilda_invKL_wderivs;

SDhalf=tilda_K_t_osamplesderivs/tilda_cholK_wderivs;
Cov=h_tildaL^2*(tilda_K_t_t-SDhalf*SDhalf');
Cov(Cov<0)=eps; % hack
tilda_ltSD=sqrt(Cov);
if isempty(tilda_ltSD); tilda_ltSD=zeros(0,1); end

num_SDs=1*hyper3scale;
num_s=min((-tilda_ltmean+1)/(num_SDs*tilda_ltSD),18); % exp(1) is probably OK

tilda_ltsamples=tilda_ltmean+(-1:num_s)'*num_SDs*tilda_ltSD;

%[tilda_ltmean-num_SDs*tilda_ltSD;tilda_ltmean;tilda_ltmean+num_SDs*tilda_ltSD];

% This mean is over scaled likelihood space. h_tildaL's cancel. Note that
% ystar=ydata, both observations and predictions are to be made at wm.

tildaQ_K_t_t=1/sqrt(prod(2*pi*tildaQ_hyperscales(hps).^2));

tildaQ_K_t_osamples=ones(1,num_samples);
ind=0;
for hyperparam=hps
    ind=ind+1;

    width=tildaQ_hyperscales(hyperparam);
    samples_hp=samples(:,hyperparam);
    trial_hypersample_hp=trial_hypersample(hyperparam);
    
    tildaQ_K_hp = normpdf(samples_hp,trial_hypersample_hp,width)';
    %matrify(@(x,y) normpdf(x,y,width),trial_hypersample_hp,samples_hp);
    tildaQ_K_t_osamples=tildaQ_K_t_osamples.*tildaQ_K_hp;
end


tildaQ_qtmean=mean_tildaq+tildaQ_K_t_osamples*tildaQ_invKq;

SDhalf=tildaQ_K_t_osamples/tildaQ_cholK;
Cov=h_tildaQ^2*(tildaQ_K_t_t-SDhalf*SDhalf');
Cov(Cov<0)=eps; % hack
tildaQ_qtSD=sqrt(Cov);
if isempty(tildaQ_qtSD); tildaQ_qtSD=zeros(0,1); end

%%%% 
function sigma=expected_uncertainty(trial_hypersample,samples,h_tildaL,h_tildaQ,qs_scale,cholK,tildaQ_cholK,cholK_wderivs,tilda_cholK_wderivs,cholK_wderivs_wcandidates,invRL_wderivs,tilda_invKL_wderivs,hyperscales,tilda_hyperscales,Q_hyperscales,tildaQ_hyperscales,hps,rearrange,n,invRN,qinvR,tildaQ_invKq,mean_tildaq,mean_tildal,invRL_wderivs_wcandidates,candidates,priorMeans,priorSDs,candidate_likelihoods,K_wderivs_wcandidates,combs_weights,candidate_Means,candidate_Cov,candidate_combs_template,hyper3scale,hyper3jitter,max_cand_tildaL,nonlcon,debug)
% The expected uncertainty of adding trial_hypersample to the sample set.
% NB: downdating has already been done prior to the application of this
% function. Trial is added at the end of candidates. We assume that
% we do not get derivative observations at the trial location (otherwise we
% would have to sample them).


if (~exist('debug','var'))
  debug = false;
end

if (exist('nonlcon') == 1 && any(nonlcon(trial_hypersample) > 0))
  sigma = nan;
  return
end

if iscell(trial_hypersample)
    trial_type='candidate';
        % We are considering adding a point that is already used as a
    % candidate. Note that when we do this we use one fewer candidate and
    % as such have a slightly poorer estimate for the expected uncertainty.
    trial_candidate_ind=trial_hypersample{1};
    trial_hypersample=candidates(trial_candidate_ind,:);
elseif any(isnan(trial_hypersample))
    sigma=nan;
    return
else
    trial_type='prospective';
end

lowr.UT=true;
lowr.TRANSA=true;
uppr.UT=true;

num_samples=size(samples,1);
num_hyperparams=length(hps);
num_wderivs=num_samples*(num_hyperparams+1);
num_candidates=size(candidates,1);
candidate_inds=num_wderivs+(1:num_candidates);
switch trial_type
    case 'candidate'
        num_candidates=num_candidates-1;
        all_but_trial=setdiff((1:num_candidates+1),trial_candidate_ind);
        candidate_inds=num_wderivs+all_but_trial;
        candidate_Means=candidate_Means(all_but_trial);
        candidate_Cov=candidate_Cov(all_but_trial,all_but_trial);
        candidate_SDs=sqrt(diag(candidate_Cov));
        
        if num_candidates>length(candidate_combs_template)
            candidate_combs_template{num_candidates}=find_likelihood_samples(zeros(num_candidates,1),ones(num_candidates,1),2^num_candidates,300,false);
        end
        num_candfns = size(candidate_combs_template{num_candidates},2);
        
 
        num_SDs = min((-min(candidate_Means,0)+max_cand_tildaL)./(max(candidate_combs_template{num_candidates},[],2).*candidate_SDs));
        num_SDs = min(num_SDs,hyper3scale);
        
        candidate_tilda_likelihoods = min(max_cand_tildaL,candidate_combs_template{num_candidates}.*repmat(candidate_SDs,1,num_candfns)*num_SDs + repmat(min(candidate_Means,0),1,num_candfns));
        candidate_likelihoods = exp(candidate_tilda_likelihoods);%,eps);
        
        mat=ones(num_candfns);
        for candidate_ind=1:num_candidates
             mat_candidate=matrify(@(x,y) normpdf(x,y,hyper3scale*candidate_SDs(candidate_ind)),...
                                 candidate_tilda_likelihoods(candidate_ind,:)',...
                                 candidate_tilda_likelihoods(candidate_ind,:)');

             mat=mat.*mat_candidate;
        end
        mat = mat+eye(length(mat))*hyper3jitter^2;
        
        width_plus_conditional_covariance = diag((hyper3scale*candidate_SDs).^2)+candidate_Cov;
        arm=candidate_tilda_likelihoods-repmat(candidate_Means,1,num_candfns);
        vect=(det(2*pi*width_plus_conditional_covariance))^(-0.5)*...
                exp(-0.5*sum(arm.*(width_plus_conditional_covariance\arm),1));
        
        combs_weights = vect/mat;
end

rearrange=[rearrange;max(rearrange)+1]; % we add on a sample
num_wderivs_wcandidates=num_wderivs+num_candidates;
num_candfns = size(candidate_likelihoods,2);



% Update K and N given this new trial hypersample location. None of this
% updating needs to take place for the tilda case.


switch trial_type
    case 'prospective'
    
        % [samples,t]^2; one new row, one new column -
        % all else can be nans
        K_wt=ones(num_samples+1);
        Q_K_wt=ones(num_samples+1);
        % [samples,derivs,candidates,t]^2; one new row, one new column -
        % all else can be nans
        K_wderivs_wcandidates_wt=ones(num_wderivs_wcandidates+1);
        % [samples,t]x[samples,derivs,candidates,t]; one new row, one new column -
        % all else can be nans
        N_wt=ones(num_samples+1,num_wderivs_wcandidates+1);
        % [samples,derivs,t]x1; one new entry at the end. n_wt is just the new
        % entry
        n_wt=1;

        K_ocandidates_t=ones(num_candidates,1);

        ind=0;
        for hyperparam=hps
            ind=ind+1;

            width_L=hyperscales(hyperparam);
            width_Q=Q_hyperscales(hyperparam);
            priorSD=priorSDs(hyperparam);
            priorMean=priorMeans(hyperparam);

            candidates_hp=candidates(:,hyperparam);
            samples_hp=samples(:,hyperparam);
            trial_hypersample_hp=trial_hypersample(hyperparam);

            inds=(num_hyperparams+1-ind)*num_samples+(1:num_samples);

            % in the following, the postscript _A refers to a term over samples and
            % samples, _B to a term over samples and gradient samples and _C to a
            % term over samples and candidates.

            K_ocandidates_t_hp = normpdf(candidates_hp,trial_hypersample_hp,width_L);
            % matrify(@(x,y) normpdf(x,y,width),candidates_hp,trial_hypersample_hp);
            K_ocandidates_t = K_ocandidates_t_hp.*K_ocandidates_t;

            
            Q_K_wt_hp=nan(num_samples+1);
            Q_K_wt_hp_A = normpdf(trial_hypersample_hp,samples_hp,width_Q)';
            %matrify(@(x,y) normpdf(x,y,width), trial_hypersample_hp,samples_hp);
            Q_K_wt_hp(end,1:end-1)=Q_K_wt_hp_A;
            Q_K_wt_hp(1:end-1,end)=Q_K_wt_hp_A';
            Q_K_wt_hp(end,end)=normpdf(0,0,width_Q);
            Q_K_wt=Q_K_wt.*Q_K_wt_hp;

            K_wt_hp=nan(num_samples+1);
            K_wt_hp_A = normpdf(trial_hypersample_hp,samples_hp,width_L)';
            %matrify(@(x,y) normpdf(x,y,width), trial_hypersample_hp,samples_hp);
            K_wt_hp(end,1:end-1)=K_wt_hp_A;
            K_wt_hp(1:end-1,end)=K_wt_hp_A';
            K_wt_hp(end,end)=normpdf(0,0,width_L);
            K_wt=K_wt.*K_wt_hp;

            K_wderivs_wcandidates_wt_hp=nan(num_wderivs_wcandidates+1);

            K_wderivs_wcandidates_wt_hp_A=K_wt_hp_A;   
            K_wderivs_wcandidates_wt_hp(end,1:num_wderivs)=...
                repmat(K_wderivs_wcandidates_wt_hp_A,1,num_hyperparams+1);
            K_wderivs_wcandidates_wt_hp(1:num_wderivs,end)=...
                K_wderivs_wcandidates_wt_hp(end,1:num_wderivs)';

            % NB: the variable you're taking the derivative wrt is negative
            K_wderivs_wcandidates_hp_B = ((trial_hypersample_hp-samples_hp)/width_L^2.*normpdf(trial_hypersample_hp,samples_hp,width_L))';
            % matrify(@(x,y) (x-y)/width^2.*normpdf(x,y,width),trial_hypersample_hp,samples_hp);    

            K_wderivs_wcandidates_wt_hp(end,inds)=K_wderivs_wcandidates_hp_B; 
            K_wderivs_wcandidates_wt_hp(inds,end)=K_wderivs_wcandidates_hp_B';

            K_wderivs_wcandidates_wt_hp(end,candidate_inds)=K_ocandidates_t_hp';
            K_wderivs_wcandidates_wt_hp(candidate_inds,end)=K_ocandidates_t_hp;
            K_wderivs_wcandidates_wt_hp(end,end)=normpdf(0,0,width_L);

            K_wderivs_wcandidates_wt=K_wderivs_wcandidates_wt.*K_wderivs_wcandidates_wt_hp;

            N_wt_hp=nan(num_samples+1,num_wderivs_wcandidates+1);
            
            determ=priorSD^2*(width_L^2+width_Q^2)+width_L^2*width_Q^2;
            PrecX_L=(priorSD^2+width_L^2)/determ;
            PrecX_Q=(priorSD^2+width_Q^2)/determ;
            PrecY=-priorSD^2/determ;
            % Nfn2=@(x,y) mvnpdf([x;y],[SamplesMean(d);SamplesMean(d)],...
            %         [SamplesSD(d)^2+width^2,SamplesSD(d)^2;SamplesSD(d)^2,SamplesSD(d)^2+width^2]);
            Nfn=@(x,y) (4*pi^2*determ)^(-0.5)*...
                exp(-0.5*PrecX_L*(x-priorMean).^2-0.5*PrecX_Q*(y-priorMean).^2-...
                PrecY.*(x-priorMean).*(y-priorMean));
            N_wt_hp_A=Nfn(trial_hypersample_hp,samples_hp)';
            %matrify(Nfn,trial_hypersample_hp,samples_hp);
            %diag(normpdf(IndepSamples,SamplesMean(d),SamplesSD(d)));
            N_wt_hp_C=Nfn(trial_hypersample_hp,[candidates_hp;trial_hypersample_hp])';
            %matrify(Nfn,trial_hypersample_hp,[candidates_hp;trial_hypersample_hp]);

            N_wt_hp_B=-width_L^-2*(samples_hp'-priorMean-...
                priorSD^2*(width_Q^2*(trial_hypersample_hp-priorMean)+width_L^2*(samples_hp'-priorMean))/determ);
            %matrify(@(x,y) (PrecX+PrecY)*priorSD^2*((x-priorMean)+(y-priorMean)),trial_hypersample_hp,samples_hp));
            % NB: (PrecX+PrecY)*priorSD^2 == priorSD^2/(width^2+2*priorSD^2)

            N_wt_hp(end,:)=[repmat(N_wt_hp_A,1,num_hyperparams+1),N_wt_hp_C];
            N_wt_hp(end,inds)=N_wt_hp_A.*N_wt_hp_B;
            N_wt_hp(1:end-1,end)=N_wt_hp_A';

            N_wt=N_wt.*N_wt_hp;


            n_wt_hp=normpdf(trial_hypersample_hp,priorMean,sqrt(width_L^2+priorSD^2));
            n_wt=n_wt.*n_wt_hp;

        end

        % [samples,t]^2; one new row, one new column -
        % all else can be nans
        % K_wt=K_wt;
        % [samples,derivs,candidates,t]^2; one new row, one new column -
        % all else can be nans
        K_wderivs_wcandidates_wt=K_wderivs_wcandidates_wt(rearrange,rearrange); % This could be made quicker by just rearranging the final row/col
        % [samples,t]x[samples,derivs,candidates,t]; one new row, one new column -
        % all else can be nans
        N_wt(end,:)=N_wt(end,rearrange);
        % [samples,derivs,t]x1; one new entry at the end
        n=[n;n_wt];
        n_wt=n;
        
        
        try
        cholK_wt = updatechol(Q_K_wt,cholK,num_samples+1);
        catch
            keyboard
        end
        cholK_wderivs_wcandidates_wt = updatechol(K_wderivs_wcandidates_wt,cholK_wderivs_wcandidates,num_wderivs_wcandidates+1);

        % if cond(cholK_wt'*cholK_wt)>10^5
        %     error('conditioning problems');
        % end
        %
        % N has also had a column added:
        invRN = [invRN,linsolve(cholK,N_wt(1:end-1,end),lowr)];
        invRN_wt = updatedatahalf(cholK_wt,N_wt,invRN,cholK,num_samples+1);
    case 'candidate'
         % [samples,t]^2; one new row, one new column
        Q_K_wt=ones(num_samples+1);
        % [samples,derivs,candidates,t]^2; one new row, one new column -
        put_trial_at_end=[1:num_wderivs,num_wderivs+[setdiff(1:num_candidates+1,trial_candidate_ind),trial_candidate_ind]];
        K_wderivs_wcandidates_wt=K_wderivs_wcandidates(put_trial_at_end,put_trial_at_end); 
        % [samples,t]x[samples,derivs,candidates,t]; one new row, one new column -
        % all else can be nans
        N_wt=ones(num_samples+1,num_wderivs_wcandidates+1);

        ind=0;
        for hyperparam=hps
            ind=ind+1;
            
            width_L=hyperscales(hyperparam);
            width_Q=Q_hyperscales(hyperparam);
            priorSD=priorSDs(hyperparam);
            priorMean=priorMeans(hyperparam);

            candidates_hp=candidates(:,hyperparam);
            samples_hp=samples(:,hyperparam);
            trial_hypersample_hp=trial_hypersample(hyperparam);

            inds=(num_hyperparams+1-ind)*num_samples+(1:num_samples);

            Q_K_wt_hp=nan(num_samples+1);
            Q_K_wt_hp_A = normpdf(trial_hypersample_hp,samples_hp,width_Q)';
            %matrify(@(x,y) normpdf(x,y,width), trial_hypersample_hp,samples_hp);
            Q_K_wt_hp(end,1:end-1)=Q_K_wt_hp_A;
            Q_K_wt_hp(1:end-1,end)=Q_K_wt_hp_A';
            Q_K_wt_hp(end,end)=normpdf(0,0,width_Q);
            Q_K_wt=Q_K_wt.*Q_K_wt_hp;
            
            
            N_wt_hp=nan(num_samples+1,num_wderivs_wcandidates+1);

            determ=priorSD^2*(width_L^2+width_Q^2)+width_L^2*width_Q^2;
            PrecX_L=(priorSD^2+width_L^2)/determ;
            PrecX_Q=(priorSD^2+width_Q^2)/determ;
            PrecY=-priorSD^2/determ;
            % Nfn2=@(x,y) mvnpdf([x;y],[SamplesMean(d);SamplesMean(d)],...
            %         [SamplesSD(d)^2+width^2,SamplesSD(d)^2;SamplesSD(d)^2,SamplesSD(d)^2+width^2]);
            Nfn=@(x,y) (4*pi^2*determ)^(-0.5)*...
                exp(-0.5*PrecX_L*(x-priorMean).^2-0.5*PrecX_Q*(y-priorMean).^2-...
                PrecY.*(x-priorMean).*(y-priorMean));
           
            N_wt_hp_A=Nfn(trial_hypersample_hp,samples_hp)';
            %matrify(Nfn,trial_hypersample_hp,samples_hp);
            %diag(normpdf(IndepSamples,SamplesMean(d),SamplesSD(d)));
            N_wt_hp_C=Nfn(trial_hypersample_hp,candidates_hp)';
            %matrify(Nfn,trial_hypersample_hp,[candidates_hp]);

            N_wt_hp_B=-width_L^-2*(samples_hp'-priorMean-...
                priorSD^2*(width_Q^2*(trial_hypersample_hp-priorMean)+width_L^2*(samples_hp'-priorMean))/determ);
            %    matrify(@(x,y) (PrecX+PrecY)*priorSD^2*((x-priorMean)+(y-priorMean)),trial_hypersample_hp,samples_hp));
            % NB: (PrecX+PrecY)*priorSD^2 == priorSD^2/(width^2+2*priorSD^2)

            N_wt_hp(end,:)=[repmat(N_wt_hp_A,1,num_hyperparams+1),N_wt_hp_C];
            N_wt_hp(end,inds)=N_wt_hp_A.*N_wt_hp_B;

            N_wt=N_wt.*N_wt_hp;
        end
        N_wt(end,:)=N_wt(end,rearrange(put_trial_at_end));
        % [samples,derivs,t]x1; one new entry at the end
        n_wt=n(put_trial_at_end);
        
        try
        cholK_wt = updatechol(Q_K_wt,cholK,num_samples+1);
        catch
            1
        end

        cholK_wderivs_wcandidates_wt = downdatechol(cholK_wderivs_wcandidates,num_wderivs+trial_candidate_ind);
        cholK_wderivs_wcandidates_wt = updatechol(K_wderivs_wcandidates_wt,cholK_wderivs_wcandidates_wt,num_wderivs_wcandidates+1);

        % if cond(cholK_wt'*cholK_wt)>10^5
        %     error('conditioning problems');
        % end
        %
        % N has NOT had a column added, but has had cols rearranged
        invRN=invRN(:,put_trial_at_end);
        invRN_wt = updatedatahalf(cholK_wt,N_wt,invRN,cholK,num_samples+1);
end

% These samples are generated according to terms without the trial,
% obviously
% if trial_is_candidate, we could recycle the samples already generated for
% lt, but not a major deal
[tilda_ltsamples,tilda_ltmean,tilda_ltSD,tilda_qtmean,tilda_qtSD] =...
    sample_GP_t(trial_hypersample,samples,h_tildaL,h_tildaQ,tildaQ_cholK,tilda_cholK_wderivs,tilda_invKL_wderivs,mean_tildal,tildaQ_invKq,mean_tildaq,tilda_hyperscales,tildaQ_hyperscales,hyper3scale,hps,rearrange);

max_lt = 10;
max_qt = 10;
tilda_ltmean = min(tilda_ltmean,log(max_lt));
tilda_qtmean = min(tilda_qtmean,log(max_qt));
tilda_ltSD = min(tilda_ltSD,sqrt(0.5*log(4*max_lt^2)-tilda_ltmean));
tilda_qtSD = min(tilda_qtSD,sqrt(0.5*log(4*max_lt^2)-tilda_qtmean));

num_ltsamples=length(tilda_ltsamples);

qs=nan(num_samples+1,1);
qt=qs_scale*exp(tilda_qtmean + 0.5*(tilda_qtSD)^2);
qs(num_samples+1)=qt;
qs0=qs;
qs0(num_samples+1)=0;

% These are identical for all ltinds, so we precompute it.
qinvR_wt=updatedatahalf(cholK_wt,qs,qinvR',cholK,num_samples+1)';
qinvR_wt0=updatedatahalf(cholK_wt,qs0,qinvR',cholK,num_samples+1)';

mus_wt=nan(num_candfns,num_ltsamples);
%mus_wt0=nan(num_candfns,num_ltsamples);
rho_t=nan(num_candfns,num_ltsamples);
likelihoods_wderivs_wt=nan(num_wderivs+1,1);
likelihoods_wderivs_wcandidates_wt=nan(num_wderivs_wcandidates+1,num_candfns);
likelihoods_wderivs_wcandidates_wt(num_wderivs+(1:num_candidates),:)=candidate_likelihoods;

for ltind=1:num_ltsamples

    tilda_lt=tilda_ltsamples(ltind);
    lt=exp(tilda_lt);
    %lt=max(exp(tilda_lt),eps); %mean_tildal taken care of inside sample_GP_t
    likelihoods_wderivs_wt(num_wderivs+1)=lt;
    likelihoods_wderivs_wcandidates_wt(num_wderivs_wcandidates+1,:)=lt;

    switch trial_type
        case 'prospective'
        invRL_wderivs_wcandidates_wt=updatedatahalf(cholK_wderivs_wcandidates_wt,likelihoods_wderivs_wcandidates_wt,invRL_wderivs_wcandidates,cholK_wderivs_wcandidates,num_wderivs_wcandidates+1); 
        case 'candidate'
        invRL_wderivs_wcandidates_wt=updatedatahalf(cholK_wderivs_wcandidates_wt,likelihoods_wderivs_wcandidates_wt,repmat(invRL_wderivs,1,num_candfns),cholK_wderivs,num_wderivs+(1:num_candidates+1));

    end

    invRL_wderivs_wcandidates_wt = invRL_wderivs_wcandidates_wt / max(invRL_wderivs_wcandidates_wt(:));
    invKL_wderivs_wcandidates_wt=linsolve(cholK_wderivs_wcandidates_wt, invRL_wderivs_wcandidates_wt, uppr);
    
    numerator = invRN_wt*invKL_wderivs_wcandidates_wt;
    denominator = n_wt'*invKL_wderivs_wcandidates_wt;
    mus_wt_col = qinvR_wt*numerator./denominator;
    %mus_wt0(:,ltind) = qinvR_wt0*numerator./denominator;
    rho_t_col = (qinvR_wt/qt-qinvR_wt0/qt)*numerator./denominator;
    
     mx_mus_wt_col=max(mus_wt_col);
%     mus_wt_col=mx_mus_wt_col*roundto(mus_wt_col/mx_mus_wt_col,3);
    
    negative=mus_wt_col/mx_mus_wt_col<-0.001;  
%     non_zero_mean=max(mean(mus_wt_col(~negative)),0);
%     mus_wt_col(negative)=non_zero_mean;
    mus_wt(:,ltind) = mus_wt_col;
    if any(negative)
        sigma=nan;
        if (debug); disp('mus_wt < 0'); end
        return;
    end
    
    mx_rho_t_col=max(rho_t_col);
    %rho_t_col=mx_rho_t_col*roundto(rho_t_col/mx_rho_t_col,3);
    
    negative=rho_t_col/mx_rho_t_col<-0.001;  
    if any(negative)
        sigma=nan;
        if (debug); disp('rho_t < 0'); end
        return;
    end
%     non_zero_mean=max(mean(rho_t_col(~negative)),0);
%     rho_t_col(negative)=non_zero_mean;
    rho_t(:,ltind) = rho_t_col;    
    
    % typically the value of l and q at the trial location does
    % not greatly influence the distribution at the candidate
    % locations and so p_of_combs is likely to be constant over
    % ind
end

%rho_t = (mus_wt-mus_wt0)/qt;
constant = qs_scale*(exp(2*tilda_qtmean+2*(tilda_qtSD)^2)-exp(2*tilda_qtmean+(tilda_qtSD)^2));
nus_at_peak = (mus_wt(1,:).^2 + rho_t(1,:).^2*constant)';

% (NhypersamplesxNtrial_funs)
%  =(NhypersamplesxNhypersamples_wderivs_wcandidates)*(Nhypersamples_wderivs_wcandidatesxNtrial_funs)

lt_fracs=10.^(linspace(-0.3,1,7))';

num_scales = size(lt_fracs,1);
logL = nan(num_scales,1);
jitters = nan(num_scales,1);
jitter_fracs = zeros(1,7);
switch length(tilda_ltsamples)
    case 1
        jitter_fracs=[0     0     0     0     0     0     0];
    case 2
        jitter_fracs=[0         0         0         0         0    0.0900    0.1300];
    case 3
        jitter_fracs=[0         0         0    0.1300    0.1700    0.1800    0.1800];
    case 4
        jitter_fracs=[0         0         0    0.1800    0.2000    0.2000    0.2000];
    case 5
        jitter_fracs=[0         0    0.1100    0.2000    0.2200    0.2200    0.2300];
    case 6
        jitter_fracs=[0         0    0.1400    0.2100    0.2300    0.2400    0.2500];
    case 7
        jitter_fracs=[0         0    0.1600    0.2100    0.2400    0.2600    0.2700];
    case 8
        jitter_fracs=[0         0    0.1700    0.2200    0.2500    0.2700    0.2800];
    case 9
        jitter_fracs=[0         0    0.1700    0.2200    0.2600    0.2800    0.3000];
    case 10
        jitter_fracs=[0         0    0.1800    0.2300    0.2600    0.2900    0.3100];
  case 11
        jitter_fracs=[0         0    0.1800    0.2300    0.2700    0.3000    0.3200];
    case 12
        jitter_fracs=[0         0    0.1800    0.2300    0.2700    0.3100    0.3300];
    case 13
        jitter_fracs=[0         0    0.1800    0.2300    0.2800    0.3200    0.3500];
    case 14
        jitter_fracs=[0         0    0.1800    0.2300    0.2800    0.3300    0.3600];
    case 15
        jitter_fracs=[0         0    0.1800    0.2300    0.2800    0.3300    0.3600];
    case 16
        jitter_fracs=[0         0    0.1800    0.2300    0.2900    0.3400    0.3700];
    case 17
        jitter_fracs=[0         0    0.1800    0.2400    0.2900    0.3400    0.3800];
    case 18
        jitter_fracs=[0         0    0.1800    0.2400    0.2900    0.3400    0.3900];
    case 19
        jitter_fracs=[0         0    0.1900    0.2400    0.2900    0.3500    0.3900];
    case 20
        jitter_fracs=[0         0    0.1900    0.2400    0.2900    0.3500    0.4000];
    otherwise
        sigma = nan;
        return
end
%jitter_fracs = [0,0.15,0.19,0.23,0.28,0.32,0.33];
% These magic numbers are from test_conditioning.m

sqd_dists = bsxfun(@minus,tilda_ltsamples',tilda_ltsamples).^2;

for i=1:num_scales;
    tilda_lt_width = lt_fracs(i)*tilda_ltSD*hyper3scale;
    %norm2SE_factor=sqrt(sqrt(2*pi*lt_width^2*qt_width^2));

    mat_base = (2*pi*tilda_lt_width^2)^(-0.5)*exp(-0.5*sqd_dists/tilda_lt_width^2);
    %matrify(@(x,y) normpdf(x,y,tilda_lt_width),tilda_ltsamples,tilda_ltsamples);
    jitterstep=0.01*sqrt(mat_base(1));

    jitter=jitterstep;
    mat=mat_base;
    
    if any(isnan(mat(:))) || any(isinf(mat(:)))
        continue
    end

%     while cond(mat)>100
%         mat=mat_base+eye(length(mat_base))*jitter^2;
%         jitter=jitter+jitterstep;
%     end
%    jitters(i)=jitter-jitterstep;

    jitter = jitter_fracs(i)*sqrt(mat_base(1));
    jitters(i) = jitter;
    mat = mat_base+eye(length(mat_base))*jitter^2;

    chol_mat=chol(mat);

    datahalf = linsolve(chol_mat, nus_at_peak.^2, lowr); % Mean assumed as zero
    % Take each column of nus_at_peak (each corresponding to a different
    % likelihood comb) as an independent dataset
    datahalf_all = datahalf(:); 
    NData = length(datahalf_all);

    % the ML solution for h_nu can be computed analytically:
    h_nu = sqrt((datahalf_all'*datahalf_all)/NData);

    % Maybe better stability elsewhere would result from sticking in this
    % output scale earlier?
    scaled_chol_mat=h_nu*chol_mat; 
    %datahalf_all=(h_mu)^(-1)*datahalf_all;

    logsqrtInvDetSigma = -sum(log(diag(scaled_chol_mat)));
    quadform = NData;%sum(datahalf_all.^2, 1);
    logL(i) = -0.5 * NData * log(2 * pi) + logsqrtInvDetSigma -0.5 * quadform; 

end

[max_logL,max_ind] = max(logL);

lt_frac = lt_fracs(max_ind);
tilda_lt_width = lt_frac*tilda_ltSD*hyper3scale;
jitter = jitters(max_ind);
% 
% if lt_frac==min(lt_fracs)
%     disp('min lt_frac')
% elseif lt_frac==max(lt_fracs)
%     disp('max lt_frac')
% end

vect = normpdf(tilda_ltsamples,tilda_ltmean,sqrt(tilda_ltSD^2+tilda_lt_width^2))';
mat = matrify(@(x,y) normpdf(x,y,tilda_lt_width),tilda_ltsamples,tilda_ltsamples);
mat = mat+eye(length(mat))*jitter^2; % Already computed the chol of this above, but too costly to store them all
% try 
%     condy=cond(mat);
%     if condy>500
%         keyboard
%     end
% catch
%     keyboard
% end
trial_weights = vect/mat;
trial_weights = max(0,trial_weights);
trial_weights = trial_weights/sum(trial_weights);

sigma = -trial_weights*((combs_weights*mus_wt).^2+(combs_weights*rho_t).^2*constant)';
if isnan(sigma)
    if (debug); disp('sigma = nan'); end
    %keyboard
end

function uncertainty = expected_uncertainty_wrapper (x, Prob)

uncertainty = expected_uncertainty(x, ...
  Prob.samples, ...
  Prob.h_tildaL, ...
  Prob.h_tildaQ, ...
  Prob.qs_scale, ...
  Prob.cholK, ...
  Prob.tilda_cholK, ...
  Prob.cholK_wderivs, ...
  Prob.tilda_cholK_wderivs, ...
  Prob.cholK_wderivs_wcandidates_minuscand, ...
  Prob.invRL_wderivs, ...
  Prob.tilda_invKL_wderivs, ...
  Prob.hyperscales, ...
  Prob.tilda_hyperscales, ...
  Prob.Q_hyperscales, ...
  Prob.tildaQ_hyperscales, ...
  Prob.hps, ...
  Prob.rearrange_minuscand, ...
  Prob.n_minuscand, ...
  Prob.invRN_minuscand, ...
  Prob.qinvR, ...
  Prob.tilda_invKq, ...
  Prob.mean_tildaq, ...
  Prob.mean_tildal, ...
  Prob.invRL_wderivs_wcandidates_minuscand, ...
  Prob.candidates_minuscand, ...
  Prob.priorMeans, ...
  Prob.priorSDs, ...
  Prob.candidate_likelihoods_minuscand, ...
  Prob.K_wderivs_wcandidates_minuscand, ...
  Prob.combs_weights_minuscand, ...
  Prob.candidate_Means_minuscand, ...
  Prob.candidate_Cov_minuscand, ...
  Prob.candidate_combs_template, ...
  Prob.hyper3scale, ...
  Prob.hyper3jitter, ...
  Prob.max_cand_tildaL);
