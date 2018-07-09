function covvy=bmcparams_ahs(covvy)
% bmcparams_ahs should do everything UP TO receipt of candidate
% likelihoods. Terms created by bmcparams_ahs can not be updated from
% previous time step as it is almost certain that the hyperscales used to
% create them will have changed since then. Note that many of the terms
% needed have already been created in calculate_hyper2sample_likelihoods -
% here we only update K matrices to include candidate locations.


h2s_ind=covvy.ML_hyper2sample_ind;
tilda_h2s_ind=covvy.ML_tilda_hyper2sample_ind;
Q_h2s_ind=covvy.ML_Q_hyper2sample_ind;

hps=covvy.active_hp_inds;
Nhyperparams=length(hps);

samples=cat(1,covvy.hypersamples(:).hyperparameters);
Nsamples=size(samples,1);

candidates=cat(1,covvy.candidates(:).hyperparameters); 
Ncandidates=size(candidates,1);  

N_wderivs_wcandidates=Nsamples*(Nhyperparams+1)+Ncandidates;

% loaded from calculate_hyper2sample_likelihoods
rearrange=covvy.rearrange;

rearrange=[rearrange;(Nsamples*(Nhyperparams+1)+(1:Ncandidates))'];

% used later by manage_hyper_samples
covvy.rearrange_withcands=rearrange;

candidate_inds=Nsamples*(Nhyperparams+1)+(1:Ncandidates);


n=ones(N_wderivs_wcandidates,1);
N=ones(Nsamples,N_wderivs_wcandidates);
%M=ones(N_wderivs_wcandidates);


for current_h2s_ind = unique([Q_h2s_ind,h2s_ind,tilda_h2s_ind])
    
    inputscales = exp(covvy.hyper2samples(current_h2s_ind).hyper2parameters);
    
    K=ones(Nsamples);
    K_wderivs_wcandidates=ones(N_wderivs_wcandidates);

    ind=0;
    for hyperparam=hps
        ind=ind+1;
        inds=(Nhyperparams+1-ind)*Nsamples+(1:Nsamples);

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

        K_wderivs_wcandidates_hp(candidate_inds,inds)=K_wderivs_wcandidates_hp_B; 
        K_wderivs_wcandidates_hp(inds,candidate_inds)=K_wderivs_wcandidates_hp_B';

        K_wderivs_wcandidates_hp(candidate_inds,candidate_inds)=...
                matrify(@(x,y) normpdf(x,y,width),...
                        candidates_hp,candidates_hp);

        K_wderivs_wcandidates=K_wderivs_wcandidates.*K_wderivs_wcandidates_hp;


        if current_h2s_ind == h2s_ind

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


    end
    

    K_wderivs_wcandidates=K_wderivs_wcandidates(rearrange,rearrange);

    % Used by sample_candidate_likelihoods
    covvy.hyper2samples(current_h2s_ind).K_wderivs_wcandidates=K_wderivs_wcandidates;




    
    if current_h2s_ind == h2s_ind
        
        cholK_wderivs=covvy.hyper2samples(current_h2s_ind).cholK_wderivs;
        % used by weights_ahs for BMC over likelihood II surface
     
        try
          cholK_wderivs_wcandidates = ...
            updatechol(K_wderivs_wcandidates,cholK_wderivs,...
            N_wderivs_wcandidates-Ncandidates+1:N_wderivs_wcandidates);
        catch
          disp('oops');
        end

        covvy.hyper2samples(current_h2s_ind).cholK_wderivs_wcandidates=cholK_wderivs_wcandidates;
        
        n=n(rearrange,:);
        % M=M(rearrange,rearrange);

        covvy.hyper2samples(current_h2s_ind).n=n;




        % Mterm = inv(cholK_wderivs_wcandidates)'*M*inv(cholK_wderivs_wcandidates)
        % Mterm = (cholK_wderivs_wcandidates'\M)/cholK_wderivs_wcandidates;
        % covvy.hyper2samples(h2s_ind).Mterm=Mterm;
    end
end

inputscales_L = exp(covvy.hyper2samples(h2s_ind).hyper2parameters);
inputscales_Q = exp(covvy.hyper2samples(Q_h2s_ind).hyper2parameters);

ind=0;
for hyperparam=hps
    ind=ind+1;
    inds=(Nhyperparams+1-ind)*Nsamples+(1:Nsamples);

    width_L=inputscales_L(hyperparam);
    width_Q=inputscales_Q(hyperparam);
    samples_hp=samples(:,hyperparam);
    candidates_hp=candidates(:,hyperparam);
    priorMean=covvy.hyperparams(hyperparam).priorMean;
    priorSD=covvy.hyperparams(hyperparam).priorSD;

        % in the following, the postscript _A refers to a term over samples and
    % samples, _B to a term over samples and gradient samples and _C to a
    % term over samples and candidates.

    determ=priorSD^2*(width_L^2+width_Q^2)+width_L^2*width_Q^2;
    PrecX_L=(priorSD^2+width_L^2)/determ;
    PrecX_Q=(priorSD^2+width_Q^2)/determ;
    PrecY=-priorSD^2/determ;
    % Nfn2=@(x,y) mvnpdf([x;y],[SamplesMean(d);SamplesMean(d)],...
    %         [SamplesSD(d)^2+width^2,SamplesSD(d)^2;SamplesSD(d)^2,SamplesSD(d)^2+width^2]);
    Nfn=@(x,y) (4*pi^2*determ)^(-0.5)*...
        exp(-0.5*PrecX_L*(x-priorMean).^2-0.5*PrecX_Q*(y-priorMean).^2-...
        PrecY.*(x-priorMean).*(y-priorMean));
    N_hp_A=...%diag(normpdf(IndepSamples,SamplesMean(d),SamplesSD(d)));
    matrify(Nfn,samples_hp,samples_hp);
    N_hp_C=matrify(Nfn,samples_hp,candidates_hp);

    N_hp_B=width_L^-2*(-repmat(samples_hp',Nsamples,1)+priorMean+...
        priorSD^2*determ^(-1)*matrify(@(x,y) (width_Q^2*(x-priorMean)+width_L^2*(y-priorMean)),samples_hp,samples_hp));
    % NB: (PrecX+PrecY)*priorSD^2 == priorSD^2/(width^2+2*priorSD^2)

    N_hp=[repmat(N_hp_A,1,Nhyperparams+1),N_hp_C];
    N_hp(:,inds)=N_hp_A.*N_hp_B;

    N=N.*N_hp;
end


N=N(:,rearrange);
covvy.hyper2samples(Q_h2s_ind).N=N; % Is actually dependent on Q_h2s_ind and h2s_ind

cholK = covvy.hyper2samples(Q_h2s_ind).cholK;

invRN=cholK'\N;
covvy.hyper2samples(Q_h2s_ind).invRN=invRN;

% invKN = inv(K)*N
invKN = cholK\invRN;
covvy.hyper2samples(Q_h2s_ind).invKN=invKN;



