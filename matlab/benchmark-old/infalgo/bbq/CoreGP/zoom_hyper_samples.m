function [smallified,covvy] = zoom_hyper_samples(covvy)

if ~isfield(covvy,'debug')
    covvy.debug = false;
end
debug = covvy.debug;

ss_closeness_num=covvy.ss_closeness_num;
qq_closeness_num=covvy.qq_closeness_num;

% any lower than this will give us covariance functions that are
% numerically infinite at their peaks, leading to all kinds of mischief
min_logscale = covvy.min_logscale;

h2s_ind=covvy.ML_hyper2sample_ind;
tilda_h2s_ind=covvy.ML_tilda_hyper2sample_ind;
Q_h2s_ind=covvy.ML_Q_hyper2sample_ind;
tildaQ_h2s_ind=covvy.ML_tildaQ_hyper2sample_ind;

if isempty(tilda_h2s_ind) || isempty(h2s_ind)
    % Can't do anything if we don't have any idea about the hyperscales.
    return
end

% A very small expense of computational time here can save our arses if
% manage_hyper_samples has actually added on a point close to one of our
% existing candidates.
covvy=determine_candidates(covvy);
candidates=cat(1,covvy.candidates.hyperparameters);
Ncandidates=size(candidates,1);

active_hp_inds=covvy.active_hp_inds;

samples=cat(1,covvy.hypersamples.hyperparameters);
Nsamples=size(samples,1);
Nhps=size(samples,2);
Nactive_hyperparams = length(active_hp_inds);
Nhyper2samples=numel(covvy.hyper2samples);

hyperscales=exp(covvy.hyper2samples(h2s_ind).hyper2parameters);
tilda_hyperscales=exp(covvy.hyper2samples(tilda_h2s_ind).hyper2parameters);
big_Q_scales = (qq_closeness_num/ss_closeness_num)*max(exp([covvy.hyper2samples(Q_h2s_ind).hyper2parameters;covvy.hyper2samples(tildaQ_h2s_ind).hyper2parameters]));
big_scales=max([hyperscales;tilda_hyperscales;big_Q_scales]);

mean_tildal = covvy.mean_tildal;

[logLcell{1:Nsamples}]=covvy.hypersamples.logL;
logLvec=cat(1,logLcell{:});
[maxlogLvec,Best_sample]=max(logLvec);
logLvec=(logLvec-maxlogLvec); 
logLvec2=logLvec-mean_tildal;

Lvec=exp(logLvec);
[glogLcell{1:Nsamples}]=covvy.hypersamples(:).glogL; % actually glogl is a cell itself
glogLmat=cell2mat(cat(2,glogLcell{:}))';
glogLmat_active=fliplr(glogLmat(:,active_hp_inds));
gLmat_active=fliplr(repmat(Lvec,1,size(glogLmat_active,2))).*glogLmat_active; %fliplr because derivs are actually in reverse order. NB: multiplying by Lvec takes care of the scale factor

% h_L=covvy.hyper2samples(ML_ind).likelihood_scale;
% scaled_h_L=h_L/sqrt(sqrt(prod(2*pi*MLinputscales.^2)));

zoomed=false(Nsamples,1);
top_pts=nan(Nsamples,Nhps);
top_mus=nan(Nsamples,1);
top_tilda_pts=nan(Nsamples,Nhps);
top_tilda_mus=nan(Nsamples,1);

for sample_ind=1:Nsamples
    
    sample=samples(sample_ind,:);
    K_wderivs=ones(1+Nactive_hyperparams);  
    tilda_K_wderivs=ones(1+Nactive_hyperparams);  

    ind=0;
    for hyperparam=active_hp_inds;
        ind=ind+1;
        
        width=hyperscales(hyperparam);
        tilda_width=tilda_hyperscales(hyperparam);
        sample_hp=sample(:,hyperparam);
        
        K_hp=matrify(@(x,y) normpdf(x,y,width),...
                                sample_hp,sample_hp);
        tilda_K_hp=matrify(@(x,y) normpdf(x,y,tilda_width),...
                                sample_hp,sample_hp);
           
                            
        % NB: the variable you're taking the derivative wrt is negative -
        % so if y>x for DK_hp below, ie. for the upper right corner of the
        % matrix, we expect there to be negatives
        DK_hp=matrify(@(x,y) (x-y)/width^2.*normpdf(x,y,width),...
            sample_hp,sample_hp); 
        DKD_hp=matrify(@(x,y) 1/width^2*(1-((x-y)/width).^2).*normpdf(x,y,width),...
            sample_hp,sample_hp);
        
        tilda_DK_hp=matrify(@(x,y) (x-y)/tilda_width^2.*normpdf(x,y,tilda_width),...
            sample_hp,sample_hp); 
        tilda_DKD_hp=matrify(@(x,y) 1/tilda_width^2*(1-((x-y)/tilda_width).^2).*normpdf(x,y,tilda_width),...
            sample_hp,sample_hp);

        K_wderivs_hp=repmat(K_hp,Nactive_hyperparams+1,Nactive_hyperparams+1);
        inds=Nactive_hyperparams+2-ind; % derivatives are still listed in reverse order for a bit of consistency
        K_wderivs_hp(inds,:)=repmat(-DK_hp,1,Nactive_hyperparams+1);
        K_wderivs_hp(:,inds)=K_wderivs_hp(inds,:)';
        K_wderivs_hp(inds,inds)=DKD_hp;

        K_wderivs=K_wderivs.*K_wderivs_hp;
        
        tilda_K_wderivs_hp=repmat(tilda_K_hp,Nactive_hyperparams+1,Nactive_hyperparams+1);
        inds=Nactive_hyperparams+2-ind; % derivatives are still listed in reverse order for a bit of consistency
        tilda_K_wderivs_hp(inds,:)=repmat(-tilda_DK_hp,1,Nactive_hyperparams+1);
        tilda_K_wderivs_hp(:,inds)=tilda_K_wderivs_hp(inds,:)';
        tilda_K_wderivs_hp(inds,inds)=tilda_DKD_hp;

        tilda_K_wderivs=tilda_K_wderivs.*tilda_K_wderivs_hp;
    end
    
    cholK_wderivs=chol(K_wderivs);
    tilda_cholK_wderivs=chol(tilda_K_wderivs);
    
    gradient_active=glogLmat_active(sample_ind,:)';
    alphas=solve_chol(tilda_cholK_wderivs,[logLvec2(sample_ind);gradient_active]);
    
    const_sum=sum((fliplr(alphas(2:end)')./tilda_hyperscales(active_hp_inds)).^2);
    const_term=sqrt(alphas(1)^2+4*const_sum);
    
    const1=(-alphas(1)+const_term)/(2*const_sum);
    const2=(-alphas(1)-const_term)/(2*const_sum);
    
    tilda_pt1=sample;
    tilda_pt1(active_hp_inds)=const1*fliplr(alphas(2:end)')+sample(active_hp_inds);
    tilda_K_pt1_sample=repmat(mvnpdf(tilda_pt1(active_hp_inds),sample(active_hp_inds),diag(tilda_hyperscales(active_hp_inds)).^2),1,length(active_hp_inds)+1);
    tilda_K_pt1_sample(2:end)=tilda_K_pt1_sample(2:end).*fliplr((-sample(active_hp_inds)+tilda_pt1(active_hp_inds))./tilda_hyperscales(active_hp_inds).^2);
    tilda_mu1=mean_tildal+tilda_K_pt1_sample*alphas;
    %mu1=fliplr(gradient_active)*(pt1(active_hp_inds)-sample(active_hp_inds))'+Lvec(sample_ind);
    


    tilda_pt2=sample;
    tilda_pt2(active_hp_inds)=const2*fliplr(alphas(2:end)')+sample(active_hp_inds); 
    tilda_K_pt2_sample=repmat(mvnpdf(tilda_pt2(active_hp_inds),sample(active_hp_inds),diag(tilda_hyperscales(active_hp_inds)).^2),1,length(active_hp_inds)+1);
    tilda_K_pt2_sample(2:end)=tilda_K_pt2_sample(2:end).*fliplr((-sample(active_hp_inds)+tilda_pt2(active_hp_inds))./tilda_hyperscales(active_hp_inds).^2);
    tilda_mu2=mean_tildal+tilda_K_pt2_sample*alphas;
    %mu2=fliplr(gradient_active)*(pt2(active_hp_inds)-sample(active_hp_inds))'+Lvec(sample_ind);
    
    gradient_active=gLmat_active(sample_ind,:)';
    alphas=solve_chol(cholK_wderivs,[Lvec(sample_ind);gradient_active]);

    const_sum=sum((fliplr(alphas(2:end)')./hyperscales(active_hp_inds)).^2);
    const_term=sqrt(alphas(1)^2+4*const_sum);

%     alphas=solve_chol(cholK_wderivs,[1;10]);
%     
%     const_sum=sum((fliplr(alphas(2:end)')./MLinputscales(active_hp_inds)).^2);
%     const_term=sqrt(alphas(1)^2+4*const_sum);
%     
%     const1=(alphas(1)+const_term)/(2*const_sum);
%     const2=(alphas(1)-const_term)/(2*const_sum);
%     
%     pt1=sample;alphas(2)
%     pt1(active_hp_inds)=const1*fliplr(alphas(2:end)')
%     pt2=sample;
%     pt2(active_hp_inds)=const2*fliplr(alphas(2:end)')
    
    if tilda_mu1>tilda_mu2
        %top_pt=tilda_pt1;
        %top_mu=tilda_mu1;
        %bottom_pt=pt2;
        
        top_tilda_pt=tilda_pt1;
        top_tilda_mu=tilda_mu1;

        const1=(-alphas(1)+const_term)/(2*const_sum);

        pt1=sample;
        pt1(active_hp_inds)=const1*fliplr(alphas(2:end)')+sample(active_hp_inds);
        K_pt1_sample=repmat(mvnpdf(pt1(active_hp_inds),sample(active_hp_inds),diag(hyperscales(active_hp_inds)).^2),1,length(active_hp_inds)+1);
        K_pt1_sample(2:end)=K_pt1_sample(2:end).*fliplr((-sample(active_hp_inds)+pt1(active_hp_inds))./hyperscales(active_hp_inds).^2);
        top_mu=K_pt1_sample*alphas;
        

        top_pt=pt1;
        
    else
        %top_pt=tilda_pt2;
        %top_mu=tilda_mu2;
        %bottom_pt=pt1;
        
        top_tilda_pt=tilda_pt2;
        top_tilda_mu=tilda_mu2;
        
        
        const2=(-alphas(1)-const_term)/(2*const_sum);

        pt2=sample;
        pt2(active_hp_inds)=const2*fliplr(alphas(2:end)')+sample(active_hp_inds);
        K_pt2_sample=repmat(mvnpdf(pt2(active_hp_inds),sample(active_hp_inds),diag(hyperscales(active_hp_inds)).^2),1,length(active_hp_inds)+1);
        K_pt2_sample(2:end)=K_pt2_sample(2:end).*fliplr((-sample(active_hp_inds)+pt2(active_hp_inds))./hyperscales(active_hp_inds).^2);
        top_mu=K_pt2_sample*alphas;
        
        
        
        top_pt=pt2;
    end
    
    top_to_sample=norm((top_pt-sample)./big_scales);
    tilda_top_to_sample=norm((top_tilda_pt-sample)./big_scales);
    %bottom_to_sample=norm((bottom_pt-sample)./big_scales);
    
    if (top_to_sample<ss_closeness_num) 
        % || bottom_to_sample<ss_closeness_num) % if this alone is satisfied,
        % could be that top_pt is way out whoop-whoop!
        %&& ...
            %any(abs(gradient_active)>0.5);
            %top_mu>3*scaled_h_L;


        zoomed(sample_ind)=true;
        top_pts(sample_ind,:)=top_pt;
        top_mus(sample_ind)=top_mu;

    end
    if (tilda_top_to_sample<ss_closeness_num) 
        % || bottom_to_sample<ss_closeness_num) % if this alone is satisfied,
        % could be that top_pt is way out whoop-whoop!
        %&& ...
            %any(abs(gradient_active)>0.5);
            %top_mu>3*scaled_h_L;

        zoomed(sample_ind)=true;
        top_tilda_pts(sample_ind,:)=top_tilda_pt;
        top_tilda_mus(sample_ind)=top_tilda_mu;
   end
    
end



if ~isempty(zoomed);
    %only zoom one sample per time step. We zoom the sample that is going
    %to gain us the most likelihood, although we zoom it to the position
    %predicted by the log (tilda) likelihood
    [max_gain,zoom_ind]=max(top_mus-Lvec);
    [max_gain_tilda,tilda_zoom_ind]=max(exp(top_tilda_mus)-exp(logLvec)); 
    % tilda_mus have already had mean_tildal taken into account
    
    if max_gain>max_gain_tilda
        if (debug); disp('zoom L'); end
        top_pt=top_pts(zoom_ind,:);
        covvy.scale_tilda_likelihood = false;
    else
        if (debug); disp('zoom tildaL'); end
        top_pt=top_tilda_pts(tilda_zoom_ind,:);
        covvy.scale_tilda_likelihood = true;
        zoom_ind = tilda_zoom_ind;
   end
    

    as_small_as_possible=nan(Nhyper2samples,Nactive_hyperparams);
    
    % Note that inactive dimensions do not contribute to distance anyway
    ssd_ts=separated_squared_distance(top_pt(active_hp_inds),samples(:,active_hp_inds)); % could put explored in here too?
    bad_hyper2sample=true(Nhyper2samples,1);
    good_hyper2sample=false(Nhyper2samples,1);
    
    for h2sample_ind=1:Nhyper2samples
        inputscales=exp(covvy.hyper2samples(h2sample_ind).hyper2parameters);      
        
        active_inputscales=inputscales(active_hp_inds);

        % already small enough
        good_hyper2sample(h2sample_ind)=...
            all(~poorly_conditioned(sqrt(scaled_ssd(ssd_ts,active_inputscales)),ss_closeness_num,'all'));
		
        as_small_as_possible(h2sample_ind,:)=log(active_inputscales*ss_closeness_num^(-1)*...
            sqrt(sum(((top_pt(active_hp_inds)-samples(zoom_ind,active_hp_inds))./active_inputscales).^2)));
        
        % even when made as_small_as_possible, this h2sample leads to
        % conditioning errors between the zoomed pt and other samples OR 
        % as_small_as_possible is smaller than min_logscale
        bad_hyper2sample(h2sample_ind)=...
            any(poorly_conditioned(sqrt(scaled_ssd(ssd_ts,exp(as_small_as_possible(h2sample_ind,:)))),ss_closeness_num,'all'))...
            || any(as_small_as_possible(h2sample_ind,:)<min_logscale);
    end
    
	scale_hyper2sample = any(~bad_hyper2sample);
	existing_hyper2sample = any(good_hyper2sample);	

    if scale_hyper2sample %|| existing_hyper2sample
        
%        if scale_hyper2sample && ~existing_hyper2sample
            if ~isfield(covvy.hyper2samples,'logL') || length([covvy.hyper2samples.logL])<Nhyper2samples
                for h2sample_ind=1:Nhyper2samples
                    covvy.hyper2samples(h2sample_ind).logL=0;
                end
            end

            hyperLogLs=[covvy.hyper2samples.logL]+[covvy.hyper2samples.Q_logL]+...
                        [covvy.hyper2samples.tilda_logL]+[covvy.hyper2samples.tildaQ_logL];
            [min_LogLs,replaced_ind]=min(hyperLogLs); % replace the lowest likelihood hyper2sample
            hyperLogLs(bad_hyper2sample)=nan;
            [max_LogLs,replacing_ind]=max(hyperLogLs); % with the highest likelihood not bad hypersample
            covvy.hyper2samples(replaced_ind).hyper2parameters = covvy.hyper2samples(replacing_ind).hyper2parameters;
            covvy.hyper2samples(replaced_ind).hyper2parameters(active_hp_inds)=as_small_as_possible(replacing_ind,:); 
            
            small_h2s_ind = replaced_ind;

            % This will get checked in improve_bmc_conditioning anyway
            %covvy.ignoreHyper2Samples=setdiff(1:Nhyper2samples,replaced_ind);
            covvy.lastHyper2SamplesMoved=unique([covvy.lastHyper2SamplesMoved,replaced_ind]);
            covvy.ignoreHyper2Samples = setdiff(1:Nhyper2samples,small_h2s_ind);
%         else
%             good_hyper2samples = find(good_hyper2sample);
%             norm_small_h2s = nan(length(good_hyper2samples),1);
%             for h2s_ind = good_hyper2samples'
%                 norm_small_h2s(h2s_ind) = norm(exp(covvy.hyper2samples(h2s_ind).hyper2parameters));
%             end
%             [max_norm_small_h2s,small_h2s_ind] = max(norm_small_h2s);
%             covvy.ignoreHyper2Samples = setdiff(1:Nhyper2samples,good_hyper2samples);
%        end    

        covvy.ML_hyper2sample_ind = small_h2s_ind;
        covvy.ML_tilda_hyper2sample_ind = small_h2s_ind;
        covvy.ML_Q_hyper2sample_ind = small_h2s_ind;
        covvy.ML_tildaQ_hyper2sample_ind = small_h2s_ind;

   
        [min_logL,min_ind]=min(logLvec);
        if (debug); min_ind
        end

        covvy.hypersamples(min_ind).hyperparameters=top_pt;
        covvy.lastHyperSampleMoved=min_ind;
        % and leave the pre-zoomed sample as is
        

        smallified = true;
        
   else       
    %if ~any(poorly_conditioned(sqrt(scaled_ssd(ssd_ts,big_scales(active_hp_inds))),ss_closeness_num,'all'))
        covvy.hypersamples(zoom_ind).hyperparameters=top_pt;
        covvy.lastHyperSampleMoved=zoom_ind;
        
        smallified = false;
   end
    


end

function [xinds,yinds]=poorly_conditioned(scaled_distance_matrix,num,flag)
% closer than num scales is too close for comfort
if nargin<3
    flag='all';
end
if strcmp(flag,'upper')
    Nsamples=size(scaled_distance_matrix,1);
    scaled_distance_matrix(tril(true(Nsamples)))=inf;
end
[xinds,yinds]=find(scaled_distance_matrix<num);


function out=separated_squared_distance(A,B)

num_As=size(A,1);
num_Bs=size(B,1);

A_perm=permute(A,[1 3 2]);
A_rep=repmat(A_perm,1,num_Bs);
B_perm=permute(B,[3 1 2]);
B_rep=repmat(B_perm,num_As,1);

out=(A_rep-B_rep).^2;

function out=scaled_ssd(ssd,scales)

[a,b,c]=size(ssd);
tower_scales=repmat(permute(scales.^-2,[1,3,2]),a,b);
out=sum(ssd.*tower_scales,3);
