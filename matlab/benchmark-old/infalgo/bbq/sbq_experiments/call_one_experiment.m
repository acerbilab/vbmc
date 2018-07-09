function call_one_experiment(problem_number, method_number, ...
                             nsamples, repitition, outdir)
% This function is designed to let a shell script start one experiment.
% Thus everything is idexed by integers.
%
% David Duvenaud
% Feb 2012
% =================

fprintf('Running one experiment...\n');

% Set defaults.
if nargin < 1; problem_number = 1; end
if nargin < 2; method_number = 1; end
if nargin < 3; nsamples = 5; end
if nargin < 4; repitition = 1; end
if nargin < 5; outdir = 'results/'; end

% If calling from the shell, all inputs will be strings, so we need to
% convert them to numbers.
if isstr(problem_number); problem_number = str2double(problem_number); end
if isstr(method_number); method_number = str2double(method_number); end
if isstr(nsamples); nsamples = str2double(nsamples); end
if isstr(repitition); repitition = str2double(repitition); end


problems = define_integration_problems();
methods = define_integration_methods();
%sample_sizes = define_sample_sizes();

% Run experiments.
run_one_experiment( problems{problem_number}, methods{method_number}, ...
                    nsamples, repitition, outdir, 0 );
