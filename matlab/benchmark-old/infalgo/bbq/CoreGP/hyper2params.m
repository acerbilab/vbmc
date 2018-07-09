function covvy=hyper2params(covvy)
% Initialises hyper2samples

if ~isfield(covvy,'hyper2samples')
    Nhps=numel(covvy.hyperparams);
    hps=1:Nhps;
    
    inactive = setdiff(hps,covvy.active_hp_inds);

    log_hyperscales=cell(1,Nhps);
    for hyperparam=hps

        % Take a number of hyperscales in each hyperparam equal to the number
        % of samples drawn from the prior over that hyperparam. We want more
        % hyperscales over hyperparams that we are more interested in.
        if ismember(hyperparam,inactive)
            log_hyperscales{hyperparam}=1;
        else
            Nhyperscales=5*ceil((covvy.hyperparams(hyperparam).NSamples));

            IndepSamples=covvy.hyperparams(hyperparam).samples;
            log_hyperscales{hyperparam}=log(linspacey(0.01,0.4,Nhyperscales)'*min(separation(IndepSamples),covvy.hyperparams(hyperparam).priorSD));
        end
            
    end

    log_hyper2parameters_mat=allcombs(log_hyperscales);
    Nhyper2samples=size(log_hyper2parameters_mat,1);

    for n = 1:Nhyper2samples
        covvy.hyper2samples(n).hyper2parameters=log_hyper2parameters_mat(n,:);
    end
else
    Nhyper2samples = numel(covvy.hyper2samples);
end

covvy.lastHyper2SamplesMoved=1:Nhyper2samples;
covvy.ML_hyper2sample_ind=[];
covvy.ML_tilda_hyper2sample_ind=[];
covvy.ML_Q_hyper2sample_ind=[];
covvy.ML_tildaQ_hyper2sample_ind=[];


function s = separation(ls) 
if length(ls)<=1
    s=1;
else
    s=(max(ls)-min(ls))/(length(ls)-1);
end
