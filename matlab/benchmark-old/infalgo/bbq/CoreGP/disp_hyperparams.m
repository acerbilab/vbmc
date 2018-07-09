function [best_hypersample, best_hypersample_struct] = ...
    disp_hyperparams(gp, best_hypersample)
% Unpacks the different values of the best hyperparameter sample into a nicely
% formatted structure.

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
best_hypersample_struct.output_scale = ...
    exp(best_hypersample(hps_struct.logOutputScale));
best_hypersample_struct.input_scales = ...
    exp(best_hypersample(hps_struct.logInputScales));
best_hypersample_struct.noise_sd = ...
    exp(best_hypersample(hps_struct.logNoiseSD));
best_hypersample_struct.mean = ...
    best_hypersample(hps_struct.mean_inds);

% Print all values of the hyperparameter struct.
if nargout == 0
fprintf('Maximum likelihood hyperparameters:\n');
cellfun(@(name, value) fprintf('\t%s\t%g\n', name, value), ...
    {gp.hyperparams.name}', ...
    mat2cell2d(best_hypersample',...
    ones(numel(gp.hyperparams),1),1));
    return
end
