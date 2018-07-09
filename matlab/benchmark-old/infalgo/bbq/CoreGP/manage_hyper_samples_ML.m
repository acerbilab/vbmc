function covvy=manage_hyper_samples_ML(covvy,method)
% Drop one hypersample, add another

if nargin<2
    method = 'rotating';
end

toMove = get_hyper_samples_to_move(covvy,method);
%[maximum, toMove] = max(cat(1, covvy.hypersamples.logL));
step_size = 0.01; %0.05
active_hp_inds = covvy.active_hp_inds;

if isfield(covvy, 'hyper2samples')
    h2s_ind=covvy.ML_hyper2sample_ind;
    length_scales=exp(covvy.hyper2samples(h2s_ind).hyper2parameters);
else
    length_scales = cat(2, covvy.hyperparams.priorSD);
end

hypersample = cat(1, covvy.hypersamples(toMove).hyperparameters);

% ascend in log likelihood space
logL = covvy.hypersamples(toMove).logL;
gradient = cell2mat(cat(2, covvy.hypersamples(toMove).glogL))';

added_hypersample = hypersample;
[active_added_hypersample, new_logL] = gradient_ascent(hypersample(:,active_hp_inds), logL, gradient(:,active_hp_inds), step_size, length_scales(active_hp_inds));
%[a,added_ind]=max(new_logLs);
%added_hypersample=gradient_ascent_points(added_ind,:);

added_hypersample(:,active_hp_inds) = active_added_hypersample;

for i = 1:numel(toMove)
  covvy.hypersamples(toMove(i)).hyperparameters=added_hypersample(i,:);
end

covvy.lastHyperSampleMoved=toMove;
%covvy.current_sample=toMove;


% ML_ind=covvy.ML_hyper2sample_ind;
% 
% if isempty(ML_ind)
%     % Can't do anything if we don't have any idea about the hyperscales.
%     return
% end
% 
% % A very small expense of computational time here can save our arses if
% % manage_hyper_samples has actually added on a point close to one of our
% % existing candidates.
% covvy=determine_candidates(covvy);
% candidates=cat(1,covvy.candidates.hyperparameters);
% 
% 
% active_hp_inds=covvy.active_hp_inds;
% 
% samples=cat(1,covvy.hypersamples.hyperparameters);
% Nsamples=size(samples,1);
% 
% Nactive_hyperparams = length(active_hp_inds);
% 
% MLinputscales=exp(covvy.hyper2samples(ML_ind).hyper2parameters);
% prior_dist=norm([covvy.hyperparams(active_hp_inds).priorSD]./MLinputscales(active_hp_inds));
% 
% [logLcell{1:Nsamples}]=covvy.hypersamples.logL;
% logLvec=cat(1,logLcell{:});
% maxlogLvec=max(logLvec);
% logLvec=(logLvec-maxlogLvec); 
% 
% Lvec=exp(logLvec);
% [glogLcell{1:Nsamples}]=covvy.hypersamples(:).glogL; % actually glogl is a cell itself
% glogLmat=cell2mat(cat(2,glogLcell{:}))';
% glogLmat_active=glogLmat(:,active_hp_inds);
% gLmat_active=fliplr(repmat(Lvec,1,size(glogLmat_active,2)).*glogLmat_active); %fliplr because derivs are actually in reverse order. NB: multiplying by Lvec takes care of the scale factor
% 
% 
% 
% for sample_ind=toMove
%     
%     sample=samples(sample_ind,:);
%     K_wderivs=ones(1+Nactive_hyperparams);  
% 
%     ind=0;
%     for hyperparam=active_hp_inds;
%         ind=ind+1;
%         
%         width=MLinputscales(hyperparam);
%         sample_hp=sample(:,hyperparam);
% 
%         K_hp=matrify(@(x,y) normpdf(x,y,width),...
%                                 sample_hp,sample_hp);
%            
%                             
%         % NB: the variable you're taking the derivative wrt is negative -
%         % so if y>x for DK_hp below, ie. for the upper right corner of the
%         % matrix, we expect there to be negatives
%         DK_hp=matrify(@(x,y) (x-y)/width^2.*normpdf(x,y,width),...
%             sample_hp,sample_hp); 
%         DKD_hp=matrify(@(x,y) 1/width^2*(1-((x-y)/width).^2).*normpdf(x,y,width),...
%             sample_hp,sample_hp);
% 
%         K_wderivs_hp=repmat(K_hp,Nactive_hyperparams+1,Nactive_hyperparams+1);
%         inds=Nactive_hyperparams+2-ind; % derivatives are still listed in reverse order for a bit of consistency
%         K_wderivs_hp(inds,:)=repmat(-DK_hp,1,Nactive_hyperparams+1);
%         K_wderivs_hp(:,inds)=K_wderivs_hp(inds,:)';
%         K_wderivs_hp(inds,inds)=DKD_hp;
% 
%         K_wderivs=K_wderivs.*K_wderivs_hp;
%     end
%     
%     cholK_wderivs=chol(K_wderivs);
%     gradient_active=gLmat_active(sample_ind,:)';
%     alphas=solve_chol(cholK_wderivs,[Lvec(sample_ind);gradient_active]);
%     
%     const_sum=sum((fliplr(alphas(2:end))./MLinputscales(active_hp_inds)).^2);
%     const_term=sqrt(alphas(1)^2+4*const_sum);
%     
%     const1=(-alphas(1)+const_term)/(2*const_sum);
%     const2=(-alphas(1)-const_term)/(2*const_sum);
%     
%     pt1=sample;
%     pt1(active_hp_inds)=const1*fliplr(alphas(2:end))+sample(active_hp_inds);
%      % seems to be some bug in this
% %     K_pt1_sample=repmat(mvnpdf(pt1(active_hp_inds),sample(active_hp_inds),diag(MLinputscales(active_hp_inds))),1,length(active_hp_inds)+1);
% %     K_pt1_sample(2:end)=K_pt1_sample(2:end).*fliplr((sample(active_hp_inds)-pt1(active_hp_inds))./MLinputscales(active_hp_inds).^2);
% %     mu1=K_pt1_sample*alphas;
%     mu1=fliplr(gradient_active)*(pt1(active_hp_inds)-sample(active_hp_inds))'+Lvec(sample_ind);
% 
%     pt2=sample;
%     pt2(active_hp_inds)=const2*fliplr(alphas(2:end))+sample(active_hp_inds); 
% %     K_pt2_sample=repmat(mvnpdf(pt2(active_hp_inds),sample(active_hp_inds),diag(MLinputscales(active_hp_inds))),1,length(active_hp_inds)+1);
% %     K_pt2_sample(2:end)=K_pt2_sample(2:end).*fliplr((sample(active_hp_inds)-pt2(active_hp_inds))./MLinputscales(active_hp_inds).^2);
% %     mu2=K_pt2_sample*alphas;
%     mu2=fliplr(gradient_active)*(pt2(active_hp_inds)-sample(active_hp_inds))'+Lvec(sample_ind);
%     
% 
% %     alphas=solve_chol(cholK_wderivs,[1;10]);
% %     
% %     const_sum=sum((fliplr(alphas(2:end))./MLinputscales(active_hp_inds)).^2);
% %     const_term=sqrt(alphas(1)^2+4*const_sum);
% %     
% %     const1=(alphas(1)+const_term)/(2*const_sum);
% %     const2=(alphas(1)-const_term)/(2*const_sum);
% %     
% %     pt1=sample;alphas(2)
% %     pt1(active_hp_inds)=const1*fliplr(alphas(2:end))
% %     pt2=sample;
% %     pt2(active_hp_inds)=const2*fliplr(alphas(2:end))
%     
%     if mu1>mu2
%         top_pt=pt1
%     else
%         top_pt=pt2
%     end
%     
%     covvy.hypersamples(sample_ind).hyperparameters=top_pt;
% end
