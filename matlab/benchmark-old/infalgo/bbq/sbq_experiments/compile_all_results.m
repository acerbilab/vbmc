function compile_all_results( results_dir, paper_dir )
% Main script to produce all figures.
%
% outdir: The directory to look in for all the results.
% plotdir: The directory to put all the pplots.

draw_plots = true;

if nargin < 1; results_dir = '~/large_results/sbq_results_gamma_one_hundredth_true_dla_likelihood/'; end
%if nargin < 1; results_dir = '~/large_results/fear_sbq_results/'; end
%if nargin < 1; results_dir = '~/large_results/sbq_results/'; end
if nargin < 2; paper_dir = '~/Dropbox/papers/sbq-paper/'; end
plotdirshort = 'figures/plots/';  % Paths relative to paper_dir.
tabledirshort = 'tables/';
plotdir = [paper_dir plotdirshort];
tabledir = [paper_dir tabledirshort];

min_samples = 3; % The minimum number of examples before we start making plots.

fprintf('Compiling all results...\n');

% Write the header for the tex file that will list all figures.
autocontent_filename = [paper_dir 'autocontent.tex'];
fprintf('All content listed in %s\n', autocontent_filename);
autocontent = fopen(autocontent_filename, 'w');
fprintf(autocontent, ['\\documentclass{article}\n' ...
    '\\usepackage{preamble}\n' ...
    '\\usepackage{graphicx}\n' ...
    '\\usepackage[margin=0.1in]{geometry}\n' ...
'\\usepackage{booktabs}\n' ...
'\\newcommand{\\acro}[1]{\\textsc{#1}}\n' ...
'\\usepackage{amsmath,amssymb,amsfonts,textcomp}\n' ...
'\\usepackage{color,psfrag,pstool}\n' ...
    '\\begin{document}\n\n']);
addpath(genpath(pwd))
    %'\\usepackage{morefloats}\n' ...
    %'\\usepackage{pgfplots}\n' ...

% Get the experimental configuration from the definition scripts.


sample_sizes = define_sample_sizes();
max_samples = sample_sizes(end);
problems = define_integration_problems();
methods = define_integration_methods();

num_problems = length(problems);
num_methods = length(methods);
num_sample_sizes = length(sample_sizes);
num_repititions = 1;


timing_table = NaN( num_methods, num_problems, num_sample_sizes, num_repititions);
mean_log_ev_table = NaN( num_methods, num_problems, num_sample_sizes, num_repititions);
var_log_ev_table = NaN( num_methods, num_problems, num_sample_sizes, num_repititions);
true_log_ev_table = NaN( num_methods, num_problems, num_sample_sizes, num_repititions);
num_missing = 0;

for p_ix = 1:num_problems
    fprintf('\nCompiling results for %s...\n', problems{p_ix}.name );
    for m_ix = 1:num_methods
        fprintf( '%6s |', methods{m_ix}.acronym);

        for r = 1:num_repititions
            try
                % Load one results file.
                % These are written in run_one_experiment.m.
                filename = run_one_experiment( problems{p_ix}, methods{m_ix}, sample_sizes(end), r, results_dir, true );
                results = load( filename );

                % Now save all relevant results into tables.
                for s_ix = 1:num_sample_sizes
                    timing_table(m_ix, p_ix, s_ix, r) = results.total_time;
                    if imag(results.mean_log_evidences(s_ix)) > 0
                        fprintf('i');
                    end
                    mean_log_ev_table(m_ix, p_ix, s_ix, r) = real(results.mean_log_evidences(end));
                    var_log_ev_table(m_ix, p_ix, s_ix, r) = results.var_log_evidences(end);
                    true_log_ev_table(m_ix, p_ix, s_ix, r) = results.problem.true_log_evidence;
                end
                samples{m_ix, p_ix} = results.samples;
                if any(isnan(results.mean_log_evidences(min_samples:end))) ...
                        || any(isnan(results.var_log_evidences((min_samples:end))))
                    fprintf('N');
                else
                    fprintf('O');       % O for OK
                end
            catch
                %disp(lasterror);
                fprintf('X');       % Never even finished.
                num_missing = num_missing + 1;
            end
        end
        fprintf(' ');
        
        fprintf('\n');
    end
    fprintf('\n');
end

% Some sanity checking.
for p_ix = 1:num_problems
    % Check that the true value for every problem was recorded as being the
    % same for all repititions, timesteps and methods tried.
    if ~(all(all(all(true_log_ev_table(:, p_ix, :, :) == ...
                       true_log_ev_table(1, p_ix, 1, 1)))))
        warning('Not all log evidences were the same, or some were missing.');
    end
end

% Normalize everything.
%for p_ix = 1:num_problems
%    mean_log_ev_table(:, p_ix, :, :) = mean_log_ev_table(:, p_ix, :, :) - true_log_ev_table(:, p_ix, :, :);
    % I think variances should stay the same... need to think about this more.
%end


method_names = cellfun( @(method) method.acronym, methods, 'UniformOutput', false );
%method_domains = cellfun( @(method) method.domain, methods, 'UniformOutput', false );
problem_names = cellfun( @(problem) problem.name, problems, 'UniformOutput', false );
true_log_ev = cellfun( @(problem) problem.true_log_evidence, problems);

% Print tables.
print_table( 'time taken (s)', problem_names, method_names, ...
    squeeze(timing_table(:,:,end,1))' );
fprintf('\n\n');
print_table( 'mean_log_ev', problem_names, { method_names{:}, 'Truth' }, ...
    [ squeeze(mean_log_ev_table(:,:,end, 1))' true_log_ev' ] );
fprintf('\n\n');
print_table( 'var_log_ev', problem_names, method_names, ...
    squeeze(var_log_ev_table(:,:,end, 1))' );


% Save tables.
latex_table( [tabledir, 'times_taken.tex'], squeeze(timing_table(:,:,end,1))', ...
    problem_names, method_names, 'time taken (s)' );
fprintf(autocontent, '\\input{%s}\n', [tabledirshort, 'times_taken.tex']);

log_error = abs(bsxfun(@minus, mean_log_ev_table(:,:,end, 1)', true_log_ev'));
latex_table( [tabledir, 'se.tex'], log_error, problem_names, method_names, ...
    sprintf('log error at %i samples', sample_sizes(end)) );
fprintf(autocontent, '\\input{%s}\n', [tabledirshort, 'se.tex']);

for p_ix = 1:num_problems
    for m_ix = 1:num_methods
        r = 1;
        true_log_evidence = true_log_ev( p_ix );

        log_mean_prediction = ...
            mean_log_ev_table( m_ix, p_ix, end, r ) - true_log_evidence;            
        log_var_prediction = ...
            var_log_ev_table( m_ix, p_ix, end, r ) - 2*true_log_evidence;
        
        squared_error(m_ix, p_ix) = (exp(log_mean_prediction) - 1)^2;
        
        try
        log_liks(m_ix, p_ix) = logmvnpdf(1, exp(log_mean_prediction), ...
                                         exp(log_var_prediction));
        catch
            log_liks(m_ix, p_ix) = nan;
        end

        normalized_mean(m_ix, p_ix) = exp(log_mean_prediction);
        normalized_std(m_ix, p_ix) = sqrt(exp(log_var_prediction));
        correct(m_ix, p_ix) = 1 < normalized_mean(m_ix, p_ix) + 0.6745 * normalized_std(m_ix, p_ix) ...
                           && 1 > normalized_mean(m_ix, p_ix) - 0.6745 * normalized_std(m_ix, p_ix);
    end
end
latex_table( [tabledir, 'truth_prob.tex'], -log_liks', problem_names, ...
     method_names, sprintf('neg log density of truth at %i samples', ...
                           sample_sizes(end)) );
fprintf(autocontent, '\\input{%s}\n', [tabledirshort, 'truth_prob.tex']);

fprintf('\n\n');
print_table( 'log error', problem_names, method_names, log_error );
fprintf('\n\n');
print_table( 'neg log density', problem_names, method_names, -log_liks' );
fprintf('\n\n');
print_table( 'squared_error', problem_names, method_names, squared_error' );
fprintf('\n\n');
print_table( 'calibration', problem_names, method_names, correct' );
fprintf('\n\n');
print_table( 'normalized_mean', problem_names, method_names, normalized_mean' );
fprintf('\n\n');
print_table( 'normalized_std', problem_names, method_names, normalized_std' );

latex_table( [tabledir, 'norm_mean.tex'], normalized_mean', problem_names, ...
     method_names, 'normalized means' );
fprintf(autocontent, '\\input{%s}\n', [tabledirshort, 'norm_mean.tex']);


latex_table( [tabledir, 'norm_std.tex'], normalized_std', problem_names, ...
     method_names, 'normalized stddev');
fprintf(autocontent, '\\input{%s}\n', [tabledirshort, 'norm_std.tex']);



combined_nll = sum(-log_liks(:, :)');
% log_error's rows are problems
combined_ale = mean(log_error(:, :));
combined_calibration = mean(correct(:, :)');
combined_synth = [combined_nll; combined_ale; combined_calibration];


table_method_names = cellfun(@(x) ['\\acro{\\lowercase{',x,'}}'],method_names, 'UniformOutput', false);
headers = {'$-\log p(\mathbf{Z})$', '\acro{ale}', '$\mathcal{C}$'};
final_results_table( [tabledir, 'combined_synth.tex'], combined_synth', table_method_names, headers, ...
     'Combined Synth Results');
fprintf(autocontent, '\\input{%s}\n', [tabledirshort, 'combined_synth.tex']);


combined_nll = sum(-log_liks(:, end-2:end)');
combined_se = sqrt(mean(squared_error(:, end-2:end)'));
combined_calibration = mean(correct(:, end-2:end)');
combined_prawn = [combined_nll; combined_se; combined_calibration];

final_results_table( [tabledir, 'combined_prawn.tex'], combined_prawn', table_method_names, headers, ...
     'Combined Prawn Results');
fprintf(autocontent, '\\input{%s}\n', [tabledirshort, 'combined_prawn.tex']);

% Draw some plots
% ================================
close all;


if draw_plots


opacity = 0.1;
edgecolor = 'none';
    
% Print legend.
% =====================
figure; clf;
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
fprintf(autocontent, '\\psfragfig{%s}\n', [plotdirshort filename]);    


label_fontsize = 10;


% Plot sample paths
% ===============================================================

chosen_repetition = 1;
figure_string = '\\psfragfig{%s}';

for p_ix = 1:num_problems
    
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
    
    cur_problem = problems{p_ix};
    for dimen = 1:cur_problem.dimension
        
          
        figure; clf;

        for m_ix = 1:num_methods
            cur_samples = samples{m_ix, p_ix};
            start_ix = 1;
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

draw_sbq_plots;