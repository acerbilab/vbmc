function [integral,covvy]=integrate_HMC(covvy,q_fn,likelihood_fn,candidate_combs_template,NData)


if nargin<3
    try
        candidate_combs_template=load(candidate_combs);
    catch
        candidate_combs_template{num_cands}=find_likelihood_samples(zeros(num_cands,1),ones(num_cands,1),100,300,false);
    end
end

if ~isfield(covvy,'plots')
    covvy.plots=false;
end

hmc2('state', 0);

covvy=hyperparams(covvy);
NSamples=numel(covvy.hypersamples);
covvy.candidate_combs_template=candidate_combs_template;
% Initialises hypersamples
covvy=hyperparams(covvy);
% All hypersamples need to have their gpparams overwritten
covvy.lastHyperSampleMoved=1:numel(covvy.hypersamples);

lowr.LT=true;
uppr.UT=true;

for ind=1:NData
          covvy = likelihood_fn(covvy,1:NSamples);

    covvy = manage_hyper_samples_HMC(covvy,likelihood_fn);

end
qs = q_fn(covvy,1:NSamples);

hyperrho = ones(NSamples,1)/NSamples;
integral = qs'*hyperrho;
