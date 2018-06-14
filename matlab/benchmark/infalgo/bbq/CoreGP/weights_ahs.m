function [rho,invRL_wderivs_wcandidates]=weights_ahs(covvy)
% This function returns the weights rho associated with a
% set of hypersamples, including candidates. It essentially assumes the
% likelihood function is known exactly as the mean of a GP.

L_h2sample_ind=covvy.ML_hyper2sample_ind;
Q_h2sample_ind=covvy.ML_Q_hyper2sample_ind;

uppr.UT=true;

candidate_tilda_likelihoods = covvy.candidate_tilda_likelihood_combs;
candidate_likelihoods = exp(candidate_tilda_likelihoods);%,eps);

cholK_wderivs=covvy.hyper2samples(L_h2sample_ind).cholK_wderivs;
cholK_wderivs_wcandidates=covvy.hyper2samples(L_h2sample_ind).cholK_wderivs_wcandidates;

candidates=cat(1,covvy.candidates.hyperparameters);
num_candidates=size(candidates,1);
num_trialfns=size(candidate_likelihoods,2);
num_wderivs_wcandidates=length(cholK_wderivs)+num_candidates;
inds=num_wderivs_wcandidates-num_candidates+1:num_wderivs_wcandidates;






% from calculate_hyper2sample_likelihoods
datahalf=covvy.hyper2samples(L_h2sample_ind).datahalf;
likelihoods=nan(num_wderivs_wcandidates,num_trialfns);
likelihoods(inds,:)=candidate_likelihoods;
invRL_wderivs_wcandidates=updatedatahalf(cholK_wderivs_wcandidates,likelihoods,repmat(datahalf,1,num_trialfns),cholK_wderivs,inds); % I checked this by computing directly
% datatwothirds is of size (Nhypersamples+num_candidates)xnum_trialfns 
invKL_wderivs_wcandidates=linsolve(cholK_wderivs_wcandidates, invRL_wderivs_wcandidates, uppr); 
%cholK_wderivs_wcandidates\invRL_wderivs_wcandidates; 
% This would need to be downdated before being used in
% manage_hyper_samples, so no point in storing it

% active_hp_inds=covvy.active_hp_inds;
% [logLcell{1:Nsamples}]=covvy.hypersamples(:).logL;
% logLvec=cat(1,logLcell{:});
% [max_logLvec,max_logL_ind]=max(logLvec);
% logLvec=(logLvec-max_logLvec); 
% Lvec=exp(logLvec);
% 
% [glogLcell{1:Nsamples}]=covvy.hypersamples(:).glogL; % actually glogl is a cell itself
% glogLmat=fliplr(cell2mat(cat(2,glogLcell{:}))');
% glogLmat=glogLmat(:,end+1-active_hp_inds);
% gLmat=fliplr(repmat(Lvec,1,size(glogLmat,2))).*glogLmat; %fliplr because derivs are actually in reverse order
% 
% % recall that observations are ordered as [L_1,gL_1,L_2,gL_2,...]
% 
% LData=[Lvec,gLmat]';
% LData=LData(:);
% likelihoods(1:min(inds)-1,:)=repmat(LData,1,num_trialfns);
% 
% invKL_wderivs_wcandidates2=solve_chol(cholK_wderivs_wcandidates,likelihoods);

% from bmcparams_ahs
n=covvy.hyper2samples(L_h2sample_ind).n;
denom=n'*invKL_wderivs_wcandidates;

% from bmcparams_ahs
invKN=covvy.hyper2samples(Q_h2sample_ind).invKN;
Nsamples=size(invKN,1);

rho=(invKN*invKL_wderivs_wcandidates)./repmat(denom,Nsamples,1);
% if any(any(rho<-0.5))
%     keyboard
% end

% 
% [logLcell{1:7}]=covvy.hypersamples.logL;
% logLvec=cat(1,logLcell{:});
% [maxlogLvec]=max(logLvec);
% logLvec=(logLvec-maxlogLvec); 
% 
% Lvec=exp(logLvec);
% [glogLcell{1:7}]=covvy.hypersamples(:).glogL; % actually glogl is a cell itself
% glogLmat=cell2mat(cat(2,glogLcell{:}))';
% gLmat=fliplr(repmat(Lvec,1,size(glogLmat,2)).*glogLmat);
% Ls=[Lvec,gLmat]';
% 
% likelihoods(1:num_wderivs_wcandidates-num_candidates,:)=repmat(Ls(:),1,num_trialfns);
% take2=solve_chol(cholK_wderivs_wcandidates,likelihoods);
% rho2=(invKN*take2)./repmat(n'*take2,Nsamples,1);