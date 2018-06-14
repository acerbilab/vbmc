function hypersample = adjust_w_max(hypersample, w_max, w0_inds)
% we now have to adjust w_max, and hence adjust the hyperparameters
% specifying input scales
w_max_old = hypersample.w_max;
hypersample_vec = cat(1,hypersample.hyperparameters);

w0s = logistic(hypersample_vec(w0_inds), w_max_old);

invlogisticw0s = ones(size(w0s)) * 3.5;

less_than_max = bsxfun(@lt, w0s, w_max - eps);
invlogisticw0s(less_than_max) = inv_logistic(w0s(less_than_max),...
    w_max(less_than_max));


hypersample.hyperparameters(w0_inds) = invlogisticw0s;

hypersample.w_max = w_max;