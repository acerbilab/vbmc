function [starting_points,num_ascent_points] = determine_starting_points(covvy, ML_ind, ...
																									num_ascent_points, ...
																									step_size)
if (nargin < 4); step_size = 0.1; end

% step_size is in units of a length scale - move a whole length

length_scales = exp(covvy.hyper2samples(ML_ind).hyper2parameters);
hypersamples_with_inactive = cat(1, covvy.hypersamples.hyperparameters);
active_hp_inds=covvy.active_hp_inds;

% can't take more points out than we actually produce.


% ascend in log likelihood space
logL = cat(1, covvy.hypersamples.logL);
gradients = cell2mat(cat(2, covvy.hypersamples.glogL))';

hypersamples=hypersamples_with_inactive(:,active_hp_inds);
gradients=gradients(:,active_hp_inds);
length_scales=length_scales(active_hp_inds);

[gradient_ascent_points, new_logLs] = gradient_ascent(hypersamples, logL, gradients, step_size, length_scales);

unchanged=isnan(gradient_ascent_points);
gradient_ascent_points(unchanged)=hypersamples(unchanged);

gradient_ascent_points_with_inactive = hypersamples_with_inactive;
gradient_ascent_points_with_inactive(:,active_hp_inds) = gradient_ascent_points;

gradient_ascent_points_with_inactive(all(unchanged,2),:) = [];
new_logLs(all(unchanged,2)) = [];

far_points = cat(1, covvy.candidates.hyperparameters);

sorted_ascent_points = sortrows([new_logLs,gradient_ascent_points_with_inactive], 1);
num_ascent_points=min(num_ascent_points,size(sorted_ascent_points,1));

starting_points = [sorted_ascent_points(end - num_ascent_points + 1:end, 2:end); far_points];
