function [min_loss, next_sample_point] = ...
    plot_1d_minimize(objective_fn, bounds, samples, log_var_ev)
% Optimizes a 1D function it by exhaustive evaluation,
% and plots the function as well.

    % Evaluate exhaustively between the bounds.
    N = 100;
    test_pts = linspace(bounds(1), bounds(2), N);
    losses = nan(1, N);
    m = nan(1, N);
    V = nan(1, N);
    for loss_i=1:length(test_pts)
        [losses(loss_i), m(loss_i), V(loss_i)] = objective_fn(test_pts(loss_i));
    end

    % Choose the best point.
    [min_loss,min_ind] = min(losses);
    next_sample_point = test_pts(min_ind);
    
    % make a plot of the gp
%     figure(666);clf;
%     tl = samples.tl;
%     gp_plot(test_pts, m, sqrt(V), samples.locations, tl);
    
    % Plot the function.
    figure(1234); clf;
    h_surface = plot(test_pts, losses, 'b'); hold on;
    
    % Plot existing neg-sqd-mean-ev
    nsme = exp(log_var_ev);
    h_exist = plot(bounds, [nsme nsme], 'k');
    
    % Also plot previously chosen points.
    h_points = plot(samples.locations, nsme + 0*samples.locations, ...
        'k.', 'MarkerSize', 10); hold on;
    h_best = plot(next_sample_point, min_loss, 'r.', 'MarkerSize', 10); hold on;
    xlabel('Sample location');
    ylabel('Expected variance after adding a new sample');
    legend( [h_surface, h_points, h_best, h_exist], {'Expected uncertainty', ...
        'Previously Sampled Points', 'Best new sample', 'existing variance'}, 'Location', 'Best');
    legend boxoff     
    set(gca, 'TickDir', 'out')
    set(gca, 'Box', 'off', 'FontSize', 10); 
    set(gcf, 'color', 'white'); 
    set(gca, 'YGrid', 'off');
    
    drawnow;
end
