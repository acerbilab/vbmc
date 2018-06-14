function [hyperrho,covvy]=hyperweights(rho,qs,covvy)
% weights=N(trial_likelihoods;...
%     m(candidates|samples,derivatives),...
%     C(candidates|samples,derivatives)+Var(candidates))*...
% inv(N(trial_likelihoods;trial_likelihoods,Var(candidates)))

lowr.UT=true;
lowr.TRANSA=true;
uppr.UT=true;

% Trial likelihood functions are completely changing at every time step, so
% no storage here.

num_samples=numel(covvy.hypersamples);
h2sample_ind=covvy.ML_tilda_hyper2sample_ind;
%h_L=covvy.hyper2samples(h2sample_ind).likelihood_scale;

candidate_tilda_likelihoods = covvy.candidate_tilda_likelihood_combs;

% All from sample_candidate_likelihoods
candidate_Means=covvy.hyper2samples(h2sample_ind).Mean_tildaL; % m(candidates|samples,derivatives)
candidate_Cov=covvy.hyper2samples(h2sample_ind).Cov_tildaL;
candidate_SDs=sqrt(diag(candidate_Cov));
num_candidates=numel(covvy.candidates);
num_candfns=size(candidate_tilda_likelihoods,2);

% Vec=N(trial_likelihoods;...
%     m(candidates|samples,derivatives),...
%     C(candidates|samples,derivatives)+Var(candidates))

mus=qs'*rho;

% Mat=N(trial_likelihoods;trial_likelihoods,Var(candidates))

% parameters
scales=10.^(linspace(0,0.7,20))';

num_scales=length(scales);
logL=nan(num_scales,1);
jitters=nan(num_scales,1);
for ind=1:num_scales;

    scale=scales(ind);
    
    Mat=ones(num_candfns);
    for candidate_ind=1:num_candidates

         Mat_candidate=matrify(@(x,y) normpdf(x,y,scale*candidate_SDs(candidate_ind)),...
                             candidate_tilda_likelihoods(candidate_ind,:)',...
                             candidate_tilda_likelihoods(candidate_ind,:)');

         Mat=Mat.*Mat_candidate;
    end
    Mat_base=Mat;
    
    if any(isnan(Mat(:))) || any(isinf(Mat(:)))
        continue
    end
    
    jitterstep=0.02*sqrt(Mat_base(1));
    jitter=jitterstep;
    while cond(Mat)>100
        Mat=Mat_base+eye(length(Mat))*jitter^2;
        jitter=jitter+jitterstep;
    end
    jitters(ind)=jitter-jitterstep;


    cholK_otrials_otrials=chol(Mat);

    datahalf = linsolve(cholK_otrials_otrials, mus', lowr);
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
scale=scales(max_ind); % ML
jitter=jitters(max_ind);

covvy.hyper3scale = scale;
covvy.hyper3jitter = jitter;
 
Mat=ones(num_candfns);
if ~isempty(Mat)
for candidate_ind=1:num_candidates
    
     Mat_candidate=matrify(@(x,y) normpdf(x,y,scale*candidate_SDs(candidate_ind)),...
                         candidate_tilda_likelihoods(candidate_ind,:)',...
                         candidate_tilda_likelihoods(candidate_ind,:)');
                     
     Mat=Mat.*Mat_candidate;
end
end

cholMat=chol(Mat+eye(length(Mat))*jitter^2); 
% Shouldn't really be necessary to add in too much jitter to Mat, given that the
% candidate combs should be well separated by design!

% candidate_Cov already has h_L in it
S=(candidate_Cov+scale^2*diag(diag(candidate_Cov)));
% Vec=matrify(@(x,y) mvnpdf(x,y,S),...
%    candidate_Means',candidate_likelihoods_combs);
% Vec=matrify(@(varargin)
% mvnpdf(cat(1,varargin{1:end/2}),cat(1,varargin{end/2+1:end}),S),...
%    candidate_Means',candidate_likelihoods_combs);
% Unfortunately the defns for Vec above do not work due to complications
% with matrify/mvnpdf

arm=candidate_tilda_likelihoods-repmat(candidate_Means,1,num_candfns);
Vec=(det(2*pi*S))^(-0.5)*exp(-0.5*sum(arm.*(S\arm),1));


% cholMat=1;
% cholMat_candidate=nan(length(covvy.candidates(1).Lsamples),length(covvy.candidates(1).Lsamples),num_candidates);
% for candidate_ind=1:num_candidates
%     
%     Mat_candidate=matrify(@(x,y) normpdf(x,y,candidate_SDs(candidate_ind)),...
%                         covvy.candidates(candidate_ind).Lsamples,...
%                         covvy.candidates(candidate_ind).Lsamples);
%     
%     cholMat_candidate(:,:,candidate_ind)=chol(Mat_candidate);
%  
%     cholMat=kron2d(cholMat,cholMat_candidate(:,:,candidate_ind));
% end
% % Really this kron stuff doesn't buy me a hell of a lot (maybe shaving
% % ~0.5s from the run-time of this function) - better results would probably
% % be obtained using candidate_Cov in the place of its diagonal. Honestly,
% % but, the candidates are so far distant from each other this is a pretty
% % good approximation.

%covvy.hyper2samples(h2sample_ind).cholK_otrials_otrials = cholMat;
%covvy.hyper2samples(h2sample_ind).cholMat_candidate = cholMat_candidate;

weights=solve_chol(cholMat,Vec')';

% weights=max(weights,0); % this is here because sometimes two samples are not too bad according to the OLD ML hyperscales, but become problematically close under the new ones
% weights=weights./sum(weights);

% no_change_uncertainty = -(mus*weights')^2;

hyperrho=rho*weights';

hyperrho=max(hyperrho,0); % this is here because sometimes two samples are not too bad according to the OLD ML hyperscales, but become problematically close under the new ones
hyperrho=hyperrho./sum(hyperrho);
% if any(hyperrho>1)
%     keyboard
% end

covvy.rho=hyperrho;


% this serves to weight the trial functions. Clearly most of the weight
% should be associated with the trial closest to the mean, which is
% normally the first in candidate_tilda_likelihoods (the one in which the
% likelihood at all candidate positions is zero).

%linsolve(cholMat,linsolve(cholMat,Vec',lowr),uppr)';
%weights=(Vec/cholMat)/cholMat';