function ev = evidence_ahs(covvy)

lowr.UT=true;
lowr.TRANSA=true;
uppr.UT=true;

h2sample_ind=covvy.ML_hyper2sample_ind;

Nhyperparams=numel(covvy.hyperparams);
hps=1:Nhyperparams;

inputscales=exp(covvy.hyper2samples(h2sample_ind).hyper2parameters);

samples=cat(1,covvy.hypersamples(:).hyperparameters);
Nsamples=size(samples,1);

candidates=cat(1,covvy.candidates(:).hyperparameters); 
Ncandidates=size(candidates,1);  

N_wderivs_wcandidates=Nsamples*(Nhyperparams+1)+Ncandidates;

% loaded from calculate_hyper2sample_likelihoods
rearrange=covvy.rearrange;

rearrange=[rearrange;(Nsamples*(Nhyperparams+1)+(1:Ncandidates))'];

% used later by manage_hyper_samples
covvy.rearrange=rearrange;

candidate_inds=Nsamples*(Nhyperparams+1)+(1:Ncandidates);

K=ones(Nsamples);
K_wderivs_wcandidates=ones(N_wderivs_wcandidates);
n=ones(N_wderivs_wcandidates,1);
N=ones(Nsamples,N_wderivs_wcandidates);
%M=ones(N_wderivs_wcandidates);



ind=0;
for hyperparam=hps
    ind=ind+1;
    
    width=inputscales(hyperparam);
    samples_hp=samples(:,hyperparam);
    candidates_hp=candidates(:,hyperparam);
    priorMean=covvy.hyperparams(hyperparam).priorMean;
    priorSD=covvy.hyperparams(hyperparam).priorSD;
    
    
    
    
    K_hp=matrify(@(x,y) normpdf(x,y,width),...
                    samples_hp,samples_hp);
    K=K.*K_hp;
    
    
    
    
    K_wderivs_wcandidates_hp=nan(N_wderivs_wcandidates);
    
    K_wderivs_wcandidates_hp_A=...
        matrify(@(x,y) normpdf(x,y,width),...
                    candidates_hp,samples_hp);   
    K_wderivs_wcandidates_hp(candidate_inds,1:min(candidate_inds)-1)=...
        repmat(K_wderivs_wcandidates_hp_A,1,Nhyperparams+1);
    K_wderivs_wcandidates_hp(1:min(candidate_inds)-1,candidate_inds)=...
        K_wderivs_wcandidates_hp(candidate_inds,1:min(candidate_inds)-1)';
    
    % NB: the variable you're taking the derivative wrt is negative
    K_wderivs_wcandidates_hp_B=matrify(@(x,y) (x-y)/width^2.*normpdf(x,y,width),...
        candidates_hp,samples_hp);    
    inds=(Nhyperparams+1-ind)*Nsamples+(1:Nsamples);
    K_wderivs_wcandidates_hp(candidate_inds,inds)=K_wderivs_wcandidates_hp_B; 
    K_wderivs_wcandidates_hp(inds,candidate_inds)=K_wderivs_wcandidates_hp_B';

    K_wderivs_wcandidates_hp(candidate_inds,candidate_inds)=...
            matrify(@(x,y) normpdf(x,y,width),...
                    candidates_hp,candidates_hp);
    
    K_wderivs_wcandidates=K_wderivs_wcandidates.*K_wderivs_wcandidates_hp;
    
    
    
    
    % in the following, the postscript _A refers to a term over samples and
    % samples, _B to a term over samples and gradient samples and _C to a
    % term over samples and candidates.
    
    PrecX=(priorSD^2+width^2-priorSD^4/(priorSD^2+width^2))^(-1);
    PrecY=(priorSD^2-(priorSD^2+width^2)^2/(priorSD^2))^(-1);
    % Nfn2=@(x,y) mvnpdf([x;y],[SamplesMean(d);SamplesMean(d)],...
    %         [SamplesSD(d)^2+width^2,SamplesSD(d)^2;SamplesSD(d)^2,SamplesSD(d)^2+width^2]);
    Nfn=@(x,y) (4*pi^2*(priorSD^2+width^2)/PrecX)^(-0.5)*...
        exp(-0.5*PrecX*((x-priorMean).^2+(y-priorMean).^2)-...
        PrecY.*(x-priorMean).*(y-priorMean));
    N_hp_A=...%diag(normpdf(IndepSamples,SamplesMean(d),SamplesSD(d)));
    matrify(Nfn,samples_hp,samples_hp);
    N_hp_C=matrify(Nfn,samples_hp,candidates_hp);

    N_hp_B=-width^-2*(repmat(samples_hp',Nsamples,1)-priorMean-...
        matrify(@(x,y) (PrecX+PrecY)*priorSD^2*((x-priorMean)+(y-priorMean)),samples_hp,samples_hp));
    % NB: (PrecX+PrecY)*priorSD^2 == priorSD^2/(width^2+2*priorSD^2)

    N_hp=[repmat(N_hp_A,1,Nhyperparams+1),N_hp_C];
    N_hp(:,inds)=N_hp_A.*N_hp_B;
    
    N=N.*N_hp;
    
    
    
    
    n_hp_A=normpdf(samples_hp,priorMean,sqrt(width^2+priorSD^2));
    n_hp_C=normpdf(candidates_hp,priorMean,sqrt(width^2+priorSD^2));
    n_hp_B=-width^-2*(priorSD^2*(priorSD^2+width^2)^(-1)-1)*(samples_hp-priorMean);
    
    n_hp=[repmat(n_hp_A,Nhyperparams+1,1);n_hp_C];
    n_hp(inds,:)=n_hp_A.*n_hp_B;
    
    n=n.*n_hp;

         
    
    
    
    %     MsSD(d)=sqrt(SamplesSD(d)^(-2)*(SamplesSD(d)^2+width^2)...
    %         *(width^2+2*(SamplesSD(d)^2-SamplesSD(d)^2*(SamplesSD(d)^2+width^2)^(-1)*SamplesSD(d)^2))...
    %         *(SamplesSD(d)^2+width^2)*SamplesSD(d)^(-2));
    %     %tends to zero as width tends to zero
    
    
%     M_hp_A=(n_hp_A*n_hp_A').*...
%         normpdf(priorSD^2*(priorSD^2+width^2)^(-1)*matrify(@minus,samples_hp,samples_hp),...
%             zeros(Nsamples),...
%             sqrt(3*width^2-2*width^2*(priorSD^2+width^2)^(-1)*width^2)*ones(Nsamples));
% 
%         phii=repmat(samples_hp,1,Nsamples);
%         phij=repmat(samples_hp',Nsamples,1);
%         
%     M_hp_B=width^(-2)*(-repmat(samples_hp,1,Nsamples)+priorMean+...
%         ((width^2+priorSD^2)*(width^2+3*priorSD^2))^(-1)*...
%         ((width^2*priorSD^2+2*priorSD^4)*(phii-priorMean)...
%             +priorSD^4*(phij-priorMean)));
%     
%     M_hp_C=(width^4*(width^4+4*width^2*priorSD^2+3*priorSD^4)^2)^(-1)*...
%         (width^6*priorSD^6+width^6*(6*priorSD^2+width^2)*(priorMean-phii).*(priorMean-phij) ...
%         -priorSD^6*(3*width^2+priorSD^2)*(phii-phij).^2 ...
%         +width^4*priorSD^4*(2*priorSD^4+9*priorMean^2 ...
%             -phii.^2+11*phii.*phij-phij.^2-9*priorMean*(phii+phij)));
% 
%     M_hp=repmat(M_hp_A,Nhyperparams+1,Nhyperparams+1);
%     inds=(Nhyperparams+1-ind)*Nsamples+(1:Nsamples); % derivatives are still listed in reverse order for a bit of consistency
%     M_hp(inds,:)=repmat(M_hp_A.*M_hp_B,1,Nhyperparams+1); 
%     M_hp(:,inds)=M_hp(inds,:)';
%     M_hp(inds,inds)=M_hp_A.*M_hp_C;
% 
%     M=M.*M_hp;

        
end

K_wderivs_wcandidates=K_wderivs_wcandidates(rearrange,rearrange);
N=N(:,rearrange);
n=n(rearrange,:);
% M=M(rearrange,rearrange);

% Used by sample_candidate_likelihoods
covvy.hyper2samples(h2sample_ind).K_wderivs_wcandidates=K_wderivs_wcandidates;

cholK_wderivs=covvy.hyper2samples(h2sample_ind).cholK_wderivs;
% used by bmcparams_ahs for BMC over likelihood II surface
try
cholK_wderivs_wcandidates = ...
    updatechol(K_wderivs_wcandidates,cholK_wderivs,...
    N_wderivs_wcandidates-Ncandidates+1:N_wderivs_wcandidates);
catch
    1
end
    
covvy.hyper2samples(h2sample_ind).cholK_wderivs_wcandidates=cholK_wderivs_wcandidates;

covvy.hyper2samples(h2sample_ind).n=n;

% This will be updated when determining uncertainties for different trial
% added hypersamples
cholK=chol(K);
covvy.hyper2samples(h2sample_ind).cholK=cholK;

covvy.hyper2samples(h2sample_ind).N=N;

invRN=cholK'\N;
covvy.hyper2samples(h2sample_ind).invRN=invRN;

% invKN = inv(K)*N
invKN = cholK\invRN;
covvy.hyper2samples(h2sample_ind).invKN=invKN;


% Mterm = inv(cholK_wderivs_wcandidates)'*M*inv(cholK_wderivs_wcandidates)
% Mterm = (cholK_wderivs_wcandidates'\M)/cholK_wderivs_wcandidates;
% covvy.hyper2samples(h2sample_ind).Mterm=Mterm;











h2sample_ind=covvy.ML_hyper2sample_ind;

lowr.UT=true;
lowr.TRANSA=true;
uppr.UT=true;

cholK_wderivs=covvy.hyper2samples(h2sample_ind).cholK_wderivs;
cholK_wderivs_wcandidates=covvy.hyper2samples(h2sample_ind).cholK_wderivs_wcandidates;

candidates=cat(1,covvy.candidates.hyperparameters);
Ncandidates=size(candidates,1);
Ntrialfns=size(candidate_likelihoods,2);
N_wderivs_wcandidates=length(cholK_wderivs)+Ncandidates;
inds=N_wderivs_wcandidates-Ncandidates+1:N_wderivs_wcandidates;

datahalf=covvy.hyper2samples(h2sample_ind).datahalf;
likelihoods=nan(N_wderivs_wcandidates,Ntrialfns);
likelihoods(inds,:)=candidate_likelihoods;
datahalf=updatedatahalf(cholK_wderivs_wcandidates,likelihoods,repmat(datahalf,1,Ntrialfns),cholK_wderivs,inds);
% datatwothirds is of size (Nhypersamples+Ncandidates)xNtrialfns 
invKL=cholK_wderivs_wcandidates\datahalf;

% from bmcparams_ahs
n=covvy.hyper2samples(h2sample_ind).n;
denom=n'*invKL;

% from bmcparams_ahs
invKN=covvy.hyper2samples(h2sample_ind).invKN;
Nsamples=size(invKN,1);

rho=(invKN*invKL)./repmat(denom,Nsamples,1);

