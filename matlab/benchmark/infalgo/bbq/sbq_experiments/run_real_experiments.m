function run_real_experiments( gamma, outdir )

if nargin < 2
    outdir = ['~/large_results/sbq_results_gamma_',num2str(gamma),'/'];
end
mkdir( outdir );

fprintf('Running all experiments...\n');

addpath(genpath(pwd));

real_probs = true;
problems = define_integration_problems(real_probs);
methods = define_integration_methods(gamma);
sample_sizes = 500;%define_sample_sizes();

% Run every combination of experiment.
num_problems = length(problems)
num_methods = length(methods)
num_sample_sizes = length(sample_sizes)
num_repititions = 1;

for r = 1:num_repititions
    for s_ix = 1:num_sample_sizes
        for p_ix = 1:num_problems
            for m_ix = 1:num_methods
                run_one_experiment( problems{p_ix}, methods{m_ix}, ...
                                    sample_sizes(s_ix), r, outdir, false );
            end
        end
    end
end


% Might as well compile all results while we're at it.
compile_all_results( outdir );
