function [best_hypersample_struct] = ...
    best_hyperparams(gp, best_hypersample)
% Unpacks the different values of the best hyperparameter sample into a nicely
% formatted structure. No exponentiation, unlike disp_hyperparams, which is
% designed more for display purposes.

if nargin<2
    if isfield(gp.hypersamples,'logL')
        [logL, best_ind] = max([gp.hypersamples.logL]);
        best_hypersample = gp.hypersamples(best_ind).hyperparameters;
    else
        best_hypersample = [gp.hyperparams.priorMean];
    end
end

% Compute the indices of all the relevant hyperparameters.
hps_struct = set_hps_struct(gp);

% Collect the relevant hyperparameters in a big struct, untransformed.
best_hypersample_struct.log_output_scale = ...
    (best_hypersample(hps_struct.logOutputScale));
best_hypersample_struct.log_input_scales = ...
    (best_hypersample(hps_struct.logInputScales));
best_hypersample_struct.log_noise_sd = ...
    (best_hypersample(hps_struct.logNoiseSD));
best_hypersample_struct.mean = ...
    best_hypersample(hps_struct.mean_inds);