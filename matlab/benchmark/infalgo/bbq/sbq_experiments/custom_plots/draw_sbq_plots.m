% Draw some plots
% ================================
close all;


if draw_plots


opacity = 0.1;
edgecolor = 'none';
    
% Print legend.
% =====================
figure(999); clf;
for m_ix = 1:num_methods
    z_handle(m_ix) = plot( 0, 0, '-', 'Color', colorbrew(m_ix), 'LineWidth', 1); hold on;
end
truth_handle = plot( 1, 1, 'k-', 'LineWidth', 1); hold on;
h_l = legend([z_handle, truth_handle], {method_names{:}, 'True value'},...
             'Location', 'East', 'Fontsize', 8);
legend boxoff
axis off;
set_fig_units_cm( 3, 4 )
filename = 'legend';
matlabfrag([plotdir filename]);
%fprintf(autocontent, '\\psfragfig{%s}\n', [plotdirshort legend]);    


label_fontsize = 10;

figure_string = '\\psfragfig{%s}';


% Plot sample paths
% ===============================================================

chosen_repetition = 1;


for p_ix = 1:num_problems
    
    fprintf(autocontent, '\\psfragfig{%s}\n', [plotdirshort 'legend']);  
    
    cur_problem = problems{p_ix};
    for dimen = 1:cur_problem.dimension
        
        figure; clf;
        for m_ix = 1:num_methods
            cur_samples = samples{m_ix, p_ix};
            if isfield(cur_samples, 'locations')
                cur_samples = cur_samples.locations;
            end
            if ~isempty(cur_samples)
%                z_handle(m_ix) = plot( cur_samples(:,1), '.', ...
 %                   'Color', colorbrew(m_ix), 'LineWidth', 1); hold on;
                % Plot the sample locations.
                start_ix = 1;
                end_ix = length(cur_samples(:,1));
                h_samples = plot3( (start_ix:end_ix)', ...
                   cur_samples(start_ix:end_ix,dimen), ...
                   zeros( end_ix - start_ix + 1, 1 ), '.', ...
                   'Color', colorbrew(m_ix));   hold on;      
            end
        end
        
        bounds = ylim;
        xrange = linspace( bounds(1), bounds(2), 100)';
        n = length(xrange);        
        true_plot_depth = sample_sizes(end);
                
        % Plot the prior.
        h_prior = plot3(repmat(true_plot_depth + 1,n,1), xrange,...
            mvnpdf(xrange, cur_problem.prior.mean(dimen), ...
            cur_problem.prior.covariance(dimen,dimen)), 'k', 'LineWidth', 2); hold on;

        % Plot the likelihood function.
        like_func_vals = nan(n, 1);
        for ii = 1:n
            ii_sample = cur_problem.prior.mean;
            ii_sample(dimen) = xrange(ii);
            like_func_vals(ii) = cur_problem.log_likelihood_fn(ii_sample);
        end
        like_func_vals = exp(like_func_vals - max(like_func_vals));
        % Rescale it to match the vertical scale of the prior.
        like_func_vals = like_func_vals ./ max(like_func_vals) ...
            .* mvnpdf(0, 0, cur_problem.prior.covariance(dimen,dimen));        
        h_ll = plot3(repmat(true_plot_depth,n,1), xrange, like_func_vals, 'Color',[0.5 0.5 0.5], 'LineWidth', 2);
        
        xlabel('Number of samples');
        ylabel('sample location');
        set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 8);
        set(get(gca,'YLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 8);        
        title([cur_problem.name,' -- dim ',num2str(dimen)]);
        xlim( [ 0 true_plot_depth ] );
        grid on;
        set(gca,'ydir','reverse')
        view(-72, 42);
        
        filename = sprintf('sampleplot_%s_dim_%g', strrep(cur_problem.name, ' ', '_'),dimen);
        set_fig_units_cm( 8, 6 );
        
        matlabfrag([plotdir filename]);
        fprintf(autocontent, figure_string, [plotdirshort filename]);  
        close
        if dimen/2==ceil(dimen/2)
            fprintf(autocontent, '\\\\');
        end
    end
    fprintf(autocontent, '\\newpage');
end



% Plot samples v likelihoods
% ===============================================================

for p_ix = 1:num_problems
    
    fprintf(autocontent, '\\psfragfig{%s}\n', [plotdirshort 'legend']);  
    
    cur_problem = problems{p_ix};
    for dimen = 1:cur_problem.dimension
        
          
        figure; clf;

        for m_ix = 1:num_methods
            cur_samples = samples{m_ix, p_ix};
            start_ix = 1;
            if ~isstruct(cur_samples)
                continue
            end
            end_ix = length(cur_samples.locations(:,1));
            
            xs = cur_samples.locations(start_ix:end_ix,dimen);
            jit_x = 0.005*(rand(end_ix,1)-0.5) * range(xs);
            
            ys = exp(cur_samples.log_l(start_ix:end_ix)+100);
            jit_y = 0.05*(rand(end_ix,1)-0.5) * range(ys);
            
            h_samples = plot(...
               xs+ jit_x, ys + jit_y, '.', 'MarkerSize', 4, ...
               'Color', colorbrew(m_ix));   hold on;      
        end
        
         % Rescale likelihoods to match the vertical scale of the prior.
        bounds = ylim;
         scaling = bounds(2) / mvnpdf(0, 0, cur_problem.prior.covariance(dimen,dimen));
        
     
        
        bounds = xlim;
        xrange = linspace( bounds(1), bounds(2), 100)';
        n = length(xrange);        
        true_plot_depth = sample_sizes(end);
                
        % Plot the prior.
        h_prior = plot(xrange,...
            scaling * mvnpdf(xrange, cur_problem.prior.mean(dimen), ...
            cur_problem.prior.covariance(dimen,dimen)), 'k', 'LineWidth', 1); hold on;

    
        ylabel('likelihoods and prior');
        xlabel('sample location');
        title([cur_problem.name,' -- dim ',num2str(dimen)]);
        set(gca, 'TickDir', 'out')
        set(gca, 'Box', 'off', 'FontSize', 10); 
        set(gcf, 'color', 'white'); 
        set(gca, 'YGrid', 'off');
        
        axis tight
        
        filename = sprintf('lik_plot_%s_dim_%g', strrep(cur_problem.name, ' ', '_'),dimen);
        set_fig_units_cm( 8, 6 );
        
        matlabfrag([plotdir filename]);
        fprintf(autocontent, figure_string, [plotdirshort filename]);  
        close
        if dimen/2==ceil(dimen/2)
            fprintf(autocontent, '\\\\');
        end
    end
    fprintf(autocontent, '\\newpage');
end


% make plots of lengthscales
% ========================================

% make legend.
% =====================
figure(999); clf;
z_handle = [];
for m_ix = 1:2
    z_handle(m_ix) = plot( 0, 0, '-', 'Color', colorbrew(m_ix), 'LineWidth', 1); hold on;
end
h_l = legend(z_handle, {...'$\ell$',
    '$\log\ell$','$\delta$'},...
             'Location', 'East', 'Fontsize', 8);
legend boxoff
axis off;
set_fig_units_cm( 3, 4 )
filename = 'lengthscales_legend';
matlabfrag([plotdir filename]);

for p_ix = 1:num_problems
    
    
    cur_problem = problems{p_ix};
    for m_ix = 1:num_methods

          
        try
            % Load one results file.
            % These are written in run_one_experiment.m.
            filename = run_one_experiment( problems{p_ix}, methods{m_ix}, sample_sizes(end), 1, results_dir, true );
            results = load( filename );

            l_logscales = nan(max_samples, cur_problem.dimension);
            tl_logscales = nan(max_samples, cur_problem.dimension);
            delta_logscales = nan(max_samples, cur_problem.dimension);
            for smpl = 1:max_samples
%                 l_logscales(smpl,:) = ...
%                     results.diagnostics(smpl).l_gp_hypers.log_input_scales;
                tl_logscales(smpl,:) = ...
                    results.diagnostics(smpl).tl_gp_hypers.log_input_scales;
                delta_logscales(smpl,:) = ...
                    results.diagnostics(smpl).del_gp_hypers.log_input_scales;
            
            end
            
            fprintf(autocontent, '\\section{%s -- %s}', methods{m_ix}.acronym, cur_problem.name); 
            fprintf(autocontent, '\\psfragfig{%s}\n', [plotdirshort 'lengthscales_legend']);  
            for dimen = 1:cur_problem.dimension
                
                
                figure; clf;
                hold on
%                 plot(l_logscales(:, dimen), 'Color', colorbrew(1))
                 plot(tl_logscales(:, dimen), 'Color', colorbrew(1))
                  plot(delta_logscales(:, dimen), 'Color', colorbrew(2))
            
                ylabel('log length scales');
                xlabel('sample number');
                title(['dim ',num2str(dimen)]);
                set(gca, 'TickDir', 'out')
                set(gca, 'Box', 'off', 'FontSize', 10); 
                set(gcf, 'color', 'white'); 
                set(gca, 'YGrid', 'off');

                axis tight

                filename = sprintf('lengthscales_plot_%s_dim_%g', strrep(cur_problem.name, ' ', '_'),dimen);
                set_fig_units_cm( 8, 6 );

                matlabfrag([plotdir filename]);
                fprintf(autocontent, figure_string, [plotdirshort filename]);  
                close
                if dimen/2==ceil(dimen/2)
                    fprintf(autocontent, '\\\\');
                end
                
            end
            fprintf(autocontent, '\\newpage');

        end

    end
end


%zlabel('Z','fontsize',12,'userdata','matlabfrag:$\mathcal Z$')

% Plot log likelihood of true answer, versus number of samples
% ===============================================================
plotted_sample_set = min_samples:max_samples;
%figure_string = '\n\\begin{figure}\n\\centering\\setlength\\fheight{14cm}\\setlength\\fwidth{12cm}\\input{%s}\n\\end{figure}\n';


    if 0
for p_ix = 1:num_problems
    cur_problem_name = problem_names{p_ix};
    figure; clf;

    try
        for m_ix = 1:num_methods
            for r = 1:num_repititions
                for s = plotted_sample_set
                    true_log_evidence = true_log_ev( p_ix );
                    log_mean_prediction = mean_log_ev_table( m_ix, p_ix, s, r ) - true_log_evidence;
                    log_var_prediction = var_log_ev_table( m_ix, p_ix, s, r ) - 2*true_log_evidence;
                    neg_log_liks(m_ix, s) = -real(logmvnpdf(1, exp(log_mean_prediction), ...
                                                               exp(log_var_prediction)));
                    neg_lok_likes_all_probs(p_ix, m_ix, s) = neg_log_liks(m_ix, s);                                                 
                end
                z_handle(m_ix) = plot( plotted_sample_set, ...
                    real(neg_log_liks(m_ix, plotted_sample_set)), '-', ...
                    'Color', colorbrew(m_ix), 'LineWidth', 1); hold on;
            end
        end
        
        xlabel('Number of samples', 'fontsize', label_fontsize);
        ylabel('$-\log(p(Z))$', 'fontsize', label_fontsize, 'Rotation',90,'Interpreter','latex');
        %title(cur_problem_name, 'fontsize', label_fontsize);
        xlim([min_samples, max_samples]);
        filename = sprintf('log_of_truth_plot_%s', strrep(cur_problem_name, ' ', '_'));

        set_fig_units_cm( 8, 6 );
        matlabfrag([plotdir filename]);
        fprintf(autocontent, figure_string, [plotdirshort filename]);    
    catch e
        %e
    end
end



% Plot log of squared distance to true answer, versus number of samples
% ======================================================================
for p_ix = 1:num_problems
    cur_problem_name = problem_names{p_ix};
    figure; clf;

    %try
        for m_ix = 1:num_methods
            for r = 1:num_repititions
                for s = min_samples:num_sample_sizes
                    true_log_evidence = true_log_ev( p_ix );
                    mean_prediction = mean_log_ev_table( m_ix, p_ix, s, r );
                    cur_squared_error(s) = (exp(mean_prediction - true_log_evidence) - 1)^2;
                    squared_error_all_probs(p_ix, m_ix, s) = cur_squared_error(s);
                end
                z_handle(m_ix) = semilogy( plotted_sample_set, ...
                    cur_squared_error(plotted_sample_set), '-',...
                    'Color', colorbrew(m_ix), 'LineWidth', 1); hold on;
            end
        end
        xlabel('Number of samples');
        ylabel('Squared Error');
        set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 8);
        set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', 8);        
        %title(cur_problem_name);
        xlim([min_samples, max_samples]);

        filename = sprintf('se_plot_%s', strrep(cur_problem_name, ' ', '_'));
        set_fig_units_cm( 8, 6 );
        matlabfrag([plotdir filename]);
        fprintf(autocontent, figure_string, [plotdirshort filename]);    
    %catch e
        %e
    %end
end

  
% Plot one example of mean and variance versus number of samples, for one
% repetition, all methods on one problem.
% ===============================================================

chosen_repetition = 1;
for p_ix = 1:num_problems
    cur_problem_name = problem_names{p_ix};
    figure; clf;
    try
        set(gco,'LineSmoothing','on') 
        true_log_evidence = true_log_ev( p_ix );
        for m_ix = 1:num_methods
            % Draw transparent part.
            mean_predictions(m_ix, :) = exp(squeeze(mean_log_ev_table( m_ix, p_ix, ...
                                                          :, chosen_repetition ))');% - true_log_evidence);
            %if m_ix ~= 2
            std_predictions = (sqrt(exp(real(squeeze(var_log_ev_table( m_ix, p_ix, ...
                                                          :, chosen_repetition ))'))));% - 2*true_log_evidence)));
            jbfill(plotted_sample_set, mean_predictions(m_ix, plotted_sample_set) + 2.*std_predictions(plotted_sample_set), ...
                                 mean_predictions(m_ix, plotted_sample_set) - 2.*std_predictions(plotted_sample_set), ...
                                 color(m_ix,:), edgecolor, 1, opacity); hold on;
            %end
            z_handle(m_ix) = plot( plotted_sample_set, ...
                mean_predictions(m_ix, plotted_sample_set), '-', ...
                'Color', colorbrew(m_ix), 'LineWidth', 1); hold on;
        end

        true_log_evidence = squeeze(true_log_ev_table( 1, p_ix, ...
                                                          :, chosen_repetition ))';
        truth_handle = plot( plotted_sample_set, ...
            exp(true_log_evidence(1)).*ones(size(plotted_sample_set)), 'k-', 'LineWidth', 1); hold on;
        xlabel('Number of samples');
        ylabel('$Z$');
        set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 8);
        set(get(gca,'YLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 8);        
%        title(cur_problem_name);
        ylim( [min(min((mean_predictions(:, plotted_sample_set)))), max(max((mean_predictions(:, plotted_sample_set))))] );
        %legend([z_handle, truth_handle], {method_names{:}, 'True value'} );
        xlim([min_samples, max_samples]);
        filename = sprintf('varplot_%s', strrep(cur_problem_name, ' ', '_'));
        set_fig_units_cm( 8, 6 );
        %myaa(2);
        
        matlabfrag([plotdir filename], 'renderer', 'opengl', 'dpi', 200);
        fprintf(autocontent, figure_string, [plotdirshort filename]);   
        close
    catch e
        e
    end
end

    end




end
% Print average over all problems.
figure; clf;
try
    for m_ix = 1:num_methods
        z_handle(m_ix) = plot( plotted_sample_set, ...
            squeeze(mean(neg_lok_likes_all_probs(:, m_ix, plotted_sample_set), 1)), '-', ...
            'Color', colorbrew(m_ix), 'LineWidth', 1); hold on;
    end

    xlabel('Number of samples', 'fontsize', label_fontsize);
    ylabel('Avg NLL of True Value', 'fontsize', label_fontsize);
    title('average over all problems', 'fontsize', label_fontsize);
    xlim([min_samples, sample_sizes(end)]);
    filename = sprintf('avg_log_of_truth_plot_%s', ...
        strrep(cur_problem_name, ' ', '_'));

    set_fig_units_cm( 8, 6 );
    matlabfrag([plotdir filename]);
    fprintf(autocontent, figure_string, [plotdirshort filename]);    
catch e
    e
end



% Print average over all problems.
figure; clf;
try
    for m_ix = 1:num_methods
        z_handle(m_ix) = plot( plotted_sample_set, ...
            squeeze(nanmean(squared_error_all_probs(:, m_ix, ...
            plotted_sample_set), 1)), '-', ...
            'Color', colorbrew(m_ix), 'LineWidth', 1); hold on;
    end

    xlabel('Number of samples', 'fontsize', label_fontsize);
    ylabel('Avg Squared Dist to Z', 'fontsize', label_fontsize);
    title('average over all problems', 'fontsize', label_fontsize);
    xlim([min_samples, sample_sizes(end)]);
    filename = sprintf('avg_se_plot_%s', strrep(cur_problem_name, ' ', '_'));

    set_fig_units_cm( 8, 6 );
    matlabfrag([plotdir filename]);
    fprintf(autocontent, figure_string, [plotdirshort filename]);    
catch e
    e
end

fprintf(autocontent, '\n\n\\end{document}');
fclose(autocontent);

close all;