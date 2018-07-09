function log_ev_plots(x_sc, mean_l_sc, mean_tl_sc, delta_tl_sc, ...
        l_gp_hypers, tl_gp_hypers, del_gp_hypers, samples, opt)
    
    [x_sc, sortorder] = sort(x_sc);

%     figure(999); clf;
% 
%     h_l = plot( x_sc, mean_l_sc(sortorder), 'b' ); hold on;
%     h_samps = plot( samples.locations, samples.scaled_l, 'k.', 'MarkerSize', 10); hold on;
%     draw_lengthscale( exp(l_gp_hypers.log_input_scales), 'l lengthscale', 1 );
%     draw_lengthscale( exp(tl_gp_hypers.log_input_scales), 'tl lengthscale', 2 );
%     draw_lengthscale( exp(del_gp_hypers.log_input_scales), 'del lengthscale', 3 );
%     set(gca, 'TickDir', 'out')
%     set(gca, 'Box', 'off', 'FontSize', 10); 
%     set(gcf, 'color', 'white'); 
%     set(gca, 'YGrid', 'off');
%     title('Untransformed space');
%     legend([ h_l, h_samps], {'L GP', 'data'});
%     legend boxoff

    figure(998); clf;
    
    h_tl = plot( x_sc, mean_tl_sc(sortorder), 'g' ); hold on;
    h_l = plot( x_sc, log_transform(mean_l_sc(sortorder), opt.gamma), 'b' ); hold on;
    h_del = plot( x_sc, delta_tl_sc(sortorder), 'r' ); hold on;
    h_samps = plot( samples.locations, samples.tl, 'k.', 'MarkerSize', 10); hold on;
    draw_lengthscale( exp(l_gp_hypers.log_input_scales), 'l lengthscale', 1 );
    draw_lengthscale( exp(tl_gp_hypers.log_input_scales), 'tl lengthscale', 2 );
    draw_lengthscale( exp(del_gp_hypers.log_input_scales), 'del lengthscale', 3 );
    
    set(gca, 'TickDir', 'out')
    set(gca, 'Box', 'off', 'FontSize', 10); 
    set(gcf, 'color', 'white'); 
    set(gca, 'YGrid', 'off');
    title('Transformed space');
    legend([ h_l, h_tl, h_del, h_samps], {'L GP', 'TL GP', 'Del vals', 'data'});
    legend boxoff
