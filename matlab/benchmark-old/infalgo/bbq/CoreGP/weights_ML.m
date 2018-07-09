function [rho]=weights_ML(covvy)
% This function returns the weights rho associated with a
% set of hypersamples, including candidates. It essentially assumes the
% likelihood function is known exactly as the mean of a GP.

L_h2sample_ind=covvy.ML_hyper2sample_ind;
Q_h2sample_ind=covvy.ML_Q_hyper2sample_ind;

uppr.UT=true;



%cholK_wderivs=covvy.hyper2samples(L_h2sample_ind).cholK_wderivs;

% from calculate_hyper2sample_likelihoods
invKL_wderivs=covvy.hyper2samples(L_h2sample_ind).datatwothirds;

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
% n=covvy.hyper2samples(L_h2sample_ind).n;
% denom=n'*invKL_wderivs;

% from bmcparams_ahs
invKN=covvy.hyper2samples(Q_h2sample_ind).invKN;

rho=(invKN*invKL_wderivs);
rho = rho./sum(rho);

%repmat(denom,Nsamples,1);
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