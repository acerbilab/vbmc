function plot_sample_path_3d( problem, sample_locations, plots_struct )

fprintf('Plotting %s...\n', problem.name );


% Todo: work out if the first or last sample should disappear.
sample_locations = sample_locations(1:end-1);


start_ix = 3;
end_ix = 20;%length(sample_locations);
spacing =3;
num_func_plots = round(end_ix - start_ix) / spacing;


left_bound = min(problem.prior.mean - 2*sqrt(problem.prior.covariance), min(sample_locations));
right_bound = max(problem.prior.mean + 2*sqrt(problem.prior.covariance), max(sample_locations));


xrange = linspace( left_bound, right_bound, 1000)';
n = length(xrange);

figure;

cmap = colormap;

% Plot the estimated functions.
for z = round(linspace(start_ix, end_ix, num_func_plots ))
    
    scale_factor = (plots_struct{z}.y_points(1) * 10000) / 2.4;
    
    cur_color = cmap(floor(size(cmap, 1)*( 1- z/end_ix)) + 1, :);
    
    % Plot the mean of the posterior.
    meanfunc = plots_struct{z}.f_vals ./ scale_factor;
    cur_xrange = plots_struct{z}.xrange;
    h_est = plot3(repmat(z,n,1), cur_xrange, meanfunc, ...
        'Color', cur_color, 'LineWidth', 2); hold on;
    
    % Plot the variance.
    old_color = [0.87 0.89 1];
    trans = 0.4;
    sd = plots_struct{z}.std_vals ./ scale_factor;
    SDh = fill3(repmat(z,2*n,1), ...
        [cur_xrange'; cur_xrange(end:-1:1)'], ...
        [meanfunc' + 2 .* sd'; meanfunc(end:-1:1)' - 2 .* sd(end:-1:1)'], ...
        cur_color, 'EdgeColor', 'none', 'FaceAlpha', trans);
    
    % Plot the observations.
    n_obs = length(plots_struct{z}.x_points);
    plot3(repmat(z,n_obs,1), plots_struct{z}.x_points, plots_struct{z}.y_points ./ scale_factor, 'kd' ); hold on;
end

% Plot the prior.
h_prior = plot3(repmat(z + 1,n,1), xrange,...
    mvnpdf(xrange, problem.prior.mean, problem.prior.covariance), 'k', 'LineWidth', 2); hold on;

% Plot the true function.
true_plot_depth = z + 1;
h_ll = plot3(repmat(true_plot_depth,n,1), xrange, exp(problem.log_likelihood_fn(xrange)), 'g', 'LineWidth', 2);

% Plot the sample locations.
h_samples = plot3( (start_ix:end_ix)', ...
                   sample_locations(start_ix:end_ix), ...
                   repmat( 0, end_ix - start_ix + 1, 1 ), 'b.' );

legend( [ h_prior h_ll h_est SDh h_samples ], {'Prior', 'True Likelihood Function', 'Posterior mean', 'Marginal uncetainty', 'Sample locations'}, 'Location', 'Best');
grid on;
zlim( [ 0 .5  ] );
xlim( [ 0 true_plot_depth ] );
title('Active learning of an integral');
xlabel('number of samples');
ylabel('x');
zlabel('f(x)');
view(-78, 32);


