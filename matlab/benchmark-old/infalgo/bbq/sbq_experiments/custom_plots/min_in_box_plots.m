
for i = 1:num_dims
    figure(i);
        clf

    lwr = lower_bound(i);
    uppr = upper_bound(i);
    xs = linspace(lwr, uppr, 1000);
    scale = normpdf(0,0, sqrt(prior.covariance(i,i)));
    plot(xs, normpdf(xs, prior.mean(i), sqrt(prior.covariance(i,i)))/scale,'k')
    
    hold on
    plot(end_points(:,i), (-end_exp_loss + median(end_exp_loss))/range(end_exp_loss), '.', 'Color', colorbrew(1))
    plot(samples.locations(:,i), (samples.scaled_l - min(samples.scaled_l))/range(samples.scaled_l), 'k.', 'MarkerSize', 20)
    
    plot(samples.locations(end,i), (samples.scaled_l(end) - min(samples.scaled_l))/range(samples.scaled_l), 'w.', 'MarkerSize', 16)
    
    plot(end_points(best_ind, i), ...
        (-exp_loss_min + max(end_exp_loss))/range(end_exp_loss), ...
        '.', 'Color', colorbrew(1), 'MarkerSize', 20);
    
    line([end_points(best_ind, i), end_points(best_ind, i)], [0 1], ...
        'Color', colorbrew(1));
    

        
    
    set(gca, 'TickDir', 'out')
    set(gca, 'Box', 'off', 'FontSize', 10); 
    set(gcf, 'color', 'w'); 
    set(gca, 'YGrid', 'off');
    xlab = ['dim_',num2str(i)];
    xlabel(xlab);
    xlim([lwr, uppr ]);
    ylim([0 1])
    set(gca, 'YTick',[])
    
    
    y_limits = ylim;
    y_scale = y_limits(2) - y_limits(1);
    yval1 = y_limits(1) + 0.45.* y_scale;
    x_loc = lwr + 0.02 * (uppr-lwr);
    line([x_loc, x_loc + scales(i)],[yval1,yval1], ...
        'Color', 'k', 'Linewidth', 1);
    line([x_loc, x_loc],[yval1 + 0.02*y_scale,yval1 - 0.02*y_scale], ...
        'Color', 'k', 'Linewidth', 1);
    line([x_loc + scales(i), x_loc + scales(i)], ...
        [yval1 + 0.02*y_scale,yval1 - 0.02*y_scale], ...
        'Color', 'k', 'Linewidth', 1);

    yval2 = y_limits(1) + 0.40.* y_scale;
    text( x_loc, yval2, 'log l input scale', 'Fontsize', 8 );
    
    set(gcf, 'units', 'centimeters');
    pos = get(gcf, 'position'); 
    set(gcf, 'position', [pos(1:2), 25, 15]); 
    
    filename = ['~/Dropbox/papers/sbq-paper/exp_loss_mov_',xlab];
    if length(samples.locations)<=1
        mov = [];
    else
        load(filename, 'mov');
    end
    mov = [mov, getframe];
    save(filename, 'mov');
    clear mov
    
end