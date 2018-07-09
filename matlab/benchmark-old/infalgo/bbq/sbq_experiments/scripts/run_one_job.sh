#!/bin/sh
#
# the next line is a "magic" comment that tells codine to use bash
#$ -S /bin/bash
#
# This script should be what is passed to qsub; its job is just to run one matlab job.

# export PYTHONPATH=/home/mlg/dkd23/local/lib/python2.5/site-packages:$PYTHONPATH

/usr/local/apps/matlab/matlabR2009a/bin/matlab -nodisplay -nojvm -logfile "/home/mlg/dkd23/large_results/fear_logs_sbq/matlab_log_$1_$2_$3_$4_fold.txt" -r "cd '/home/mlg/dkd23/Dropbox/code/gp-code-osborne/'; addpath(genpath(pwd)); ls; call_one_experiment($1, $2, $3, $4, '/home/mlg/dkd23/large_results/fear_sbq_results/'); exit" 
                                                                                                                                                                          #function call_one_experiment(problem_number, method_number, ...
#                             nsamples, repitition, outdir)

