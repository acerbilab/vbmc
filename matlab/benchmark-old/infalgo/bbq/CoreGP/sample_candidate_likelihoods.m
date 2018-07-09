function covvy=sample_candidate_likelihoods(covvy)
% Want to resample candidate likelihoods at every time step as we expect
% the likelihood surface to be constantly changing. Note that the
% likelihoods used to compute datatwothirds have been multiplied by a
% positive constant so that largest likelihood observed is always one.

if ~isfield(covvy,'max_cand_tildaL')
    covvy.max_cand_tildaL=2;
end
max_cand_tildaL = covvy.max_cand_tildaL;

tilda_h2s_ind=covvy.ML_tilda_hyper2sample_ind;
%inputscales=exp(covvy.hyper2samples(h2sample_ind).hyper2parameters);
h_tildaL=covvy.hyper2samples(tilda_h2s_ind).tilda_likelihood_scale;

mean_tildal = covvy.mean_tildal;

% from calculate_hyper2sample_likelihoods
% This is absent h_tildaL^2
cholK_wderivs=covvy.hyper2samples(tilda_h2s_ind).cholK_wderivs;
% This is absent h_tildaL^2 and includes the likelihood scaling factor
tilda_datatwothirds=covvy.hyper2samples(tilda_h2s_ind).tilda_datatwothirds; 

samples=cat(1,covvy.hypersamples.hyperparameters);
Nsamples=size(samples,1);

Nhyperparams=length(covvy.active_hp_inds);

candidates=cat(1,covvy.candidates.hyperparameters); % or maybe just far-away ones
Ncandidates=size(candidates,1);  
candidate_inds=Nsamples*(Nhyperparams+1)+(1:Ncandidates);

% This is absent h_tildaL^2
K_wderivs_wcandidates=covvy.hyper2samples(tilda_h2s_ind).K_wderivs_wcandidates; % from bmcparams_ahs
K_ocandidates_osamplesderivs=K_wderivs_wcandidates(candidate_inds,1:(Nsamples*(Nhyperparams+1)));
K_ocandidates_ocandidates=K_wderivs_wcandidates(candidate_inds,candidate_inds);

% This mean is over scaled tilda likelihood space. h_tildaL's cancel.
Mean=mean_tildal+K_ocandidates_osamplesderivs*tilda_datatwothirds;

% We effectively want a sqd exp covariance with output scale 1 over scaled
% tilda likelihood space. 
SDhalf=K_ocandidates_osamplesderivs/cholK_wderivs;
Cov=K_ocandidates_ocandidates-SDhalf*SDhalf';
Cov=h_tildaL^2*Cov;
SD=sqrt(diag(Cov));
if isempty(SD); SD=zeros(0,1); end

covvy.hyper2samples(tilda_h2s_ind).Mean_tildaL=Mean;
covvy.hyper2samples(tilda_h2s_ind).Cov_tildaL=Cov;


candidate_combs_template = covvy.candidate_combs_template;
if Ncandidates>length(candidate_combs_template)
    candidate_combs_template{Ncandidates}=find_likelihood_samples(zeros(Ncandidates,1),ones(Ncandidates,1),2^Ncandidates,300,false);
    covvy.candidate_combs_template=candidate_combs_template;
elseif Ncandidates==0
    covvy.candidate_tilda_likelihood_combs=[];
    return
end
num_combs = size(candidate_combs_template{Ncandidates},2);

Mean = min(Mean,0);
num_SDs = min((-Mean+max_cand_tildaL)./(max(candidate_combs_template{Ncandidates},[],2).*SD));
if isfield(covvy,'hyper3scale')
    num_SDs=min(num_SDs,covvy.hyper3scale);
end

covvy.candidate_tilda_likelihood_combs = candidate_combs_template{Ncandidates}.*repmat(SD,1,num_combs)*num_SDs + repmat(Mean,1,num_combs);
