function [exp_loss_min, next_sample_point, flag] = ...
    min_in_box( objective_fn, prior, samples, tl_gp_hypers, evals )
%
% Searches within a prior box.

evals_per_start = 100;

fraction_exploit = 0.5;
fraction_continue = 0.3;
fraction_explore = 1 - fraction_exploit - fraction_continue;

num_exploits = round(fraction_exploit * evals / evals_per_start);
num_continues = round(fraction_continue * evals / evals_per_start);
num_explores = round(fraction_explore * evals);

optim_opts = ...
    optimset('GradObj','off',...
    'Display','off', ...
    'MaxFunEvals', evals_per_start,...
    'LargeScale', 'off',...
    'Algorithm','interior-point'...
    );

num_dims = length(prior.mean);

lower_bound = prior.mean - 3.*sqrt(diag(prior.covariance))';
upper_bound = prior.mean + 3.*sqrt(diag(prior.covariance))';

[max_log_l, max_ind] = max(samples.log_l);
max_sample = samples.locations(max_ind, :);
scales = exp(tl_gp_hypers.log_input_scales);
exploit_starts = find_candidates(max_sample, num_exploits, scales, 1);

continued_starts = ...
    find_candidates(samples.locations(end, :), num_continues, scales, 1);

starts = [exploit_starts;continued_starts];
starts = bound(starts, lower_bound+eps, upper_bound-eps);
num_starts = num_exploits + num_continues;

end_points = nan(num_starts,num_dims);
end_exp_loss = nan(num_starts,1);
parfor i = 1:num_starts
    cur_start_pt = starts(i, :);
    try
        [end_points(i,:), end_exp_loss(i)] = ...
        fmincon(objective_fn,cur_start_pt, ...
        [],[],[],[],...
        lower_bound,upper_bound,[],...
        optim_opts);
    catch
        end_points(i,:) = cur_start_pt;
        end_exp_loss(i) = inf;
    end
end

explores = mvnrnd(prior.mean, prior.covariance, num_explores);
explores = bound(explores, lower_bound, upper_bound);
explore_loss = nan(num_explores, 1);
parfor i = 1:num_explores
    explore_loss(i) = objective_fn(explores(i, :));
end
end_points = [end_points; explores];
end_exp_loss = [end_exp_loss; explore_loss];


[exp_loss_min, best_ind] = min(end_exp_loss);
next_sample_point = end_points(best_ind, :);

if best_ind <= num_exploits
        flag = 'exploit';
elseif best_ind <= num_starts
        flag = 'continue';
else
        flag = 'explore';
end

%min_in_box_plots;

end
