function input_importances = input_importance_wm(gp, XStar, XData, best_ind)
% returns a vector reflecting the importance of each input. The larger the
% ith entry, the more important the ith input is. Relative to
% input_importance, input_importance_wm considers the influence of the
% inputs on the mean.

if nargin<4
    [dummy,best_ind] = max([gp.hypersamples(:).logL]);
end
    
hps_struct = set_hps_struct(gp);
logInputScale_inds  = hps_struct.logInputScales;

best_hypersample = gp.hypersamples(best_ind).hyperparameters;

% find the derivative of the means with respect to inputs (not input
% scales)
[dummy_m,dummy_C,gm] = posterior_gp(XStar,XData,gp,best_ind,'no_cov');
input_importances = cellfun(@(X) mean(abs(X)),gm).*range(XStar)';

