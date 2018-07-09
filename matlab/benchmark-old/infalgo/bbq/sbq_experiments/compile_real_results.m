function compile_real_results( results_dir, paper_dir )
% Main script to produce all figures.
%
% outdir: The directory to look in for all the results.
% plotdir: The directory to put all the pplots.

draw_plots = true;

if nargin < 1; results_dir = '~/large_results/sbq_results_gamma_0.01/'; end
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

real_probs = true;
sample_sizes = define_sample_sizes();
max_samples = sample_sizes(end);
problems = define_integration_problems(real_probs);
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
final_results_table( [tabledir, 'combined_dla.tex'], combined_synth', table_method_names, headers, ...
     'Combined Real Results');
fprintf(autocontent, '\\input{%s}\n', [tabledirshort, 'combined_dla.tex']);


combined_nll = sum(-log_liks(:, end-2:end)');
combined_se = sqrt(mean(squared_error(:, end-2:end)'));
combined_calibration = mean(correct(:, end-2:end)');
combined_prawn = [combined_nll; combined_se; combined_calibration];

% final_results_table( [tabledir, 'combined_prawn.tex'], combined_prawn', table_method_names, headers, ...
%      'Combined Prawn Results');
% fprintf(autocontent, '\\input{%s}\n', [tabledirshort, 'combined_prawn.tex']);
% 

draw_sbq_plots;