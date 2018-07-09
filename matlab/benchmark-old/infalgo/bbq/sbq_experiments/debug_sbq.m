function debug_sbq(method_number, problem_number, nsamples, outdir)

% Set defaults.
if nargin < 1; method_number = 3; end
if nargin < 2; problem_number = 1; end
if nargin < 3; nsamples = 5; end
if nargin < 4; outdir = 'results/'; end

repitition = 11;

problems = define_integration_problems();
methods = define_integration_methods();

sbq_debug_method = methods{method_number};
sbq_debug_method.opt.plots = true;
sbq_debug_method.opt.set_ls_var_method = 'laplace';


% Run experiments.
run_one_experiment( problems{problem_number}, sbq_debug_method, ...
                    nsamples, repitition, outdir, 0 );
