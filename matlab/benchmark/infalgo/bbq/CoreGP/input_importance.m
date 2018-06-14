function input_importances = input_importance(covvy, XData, best_ind)
% returns a vector reflecting the importance of each input. The larger the ith entry, the more important the ith input is.

if nargin<2
    XData = covvy.X_data;
end

if nargin<3
    [dummy,best_ind] = max([covvy.hypersamples(:).logL]);
end
    
hps_struct = set_hps_struct(covvy);
logInputScale_inds  = hps_struct.logInputScales;

best_hypersample = covvy.hypersamples(best_ind).hyperparameters;

input_importances = (exp(best_hypersample(logInputScale_inds))./range(XData)).^(-1);

if nargout==0
    disp_importances(input_importances);
end