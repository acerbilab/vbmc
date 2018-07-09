function [prior, samples] = extract_likelihood_samples( gp )
% A helper function which converts the likelihood observations of a
% GP struct into samples in order to do SBQ.
% This used to live in both log_evidence and expected_uncertainty_evidence.
   
    samples.sample_locations = vertcat(gp.hypersamples.hyperparameters);
    samples.log_values = vertcat(gp.hypersamples.logL);
       
    prior.means = vertcat(gp.hyperparams.priorMean);
    prior.sds = vertcat(gp.hyperparams.priorSD);
