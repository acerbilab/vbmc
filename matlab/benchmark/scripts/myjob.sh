#!/bin/sh

module purge
#. /etc/profile.d/modules.sh

# Use Intel compiler
if [ ${CLUSTER} = "Mercer" ]; then
	module load matlab gcc
	export LD_PRELOAD=$GCC_LIB/libstdc++.so
else
        module load matlab/2017a
	export MATLAB_PREFDIR=$(mktemp -d $SLURM_JOBTMP/matlab-XXXXXX)
fi
export MATLABPATH=${MATLABPATH}:/${HOME}/${PROJECT}/matlab:${HOME}/MATLAB
source ${HOME}/MATLAB/setpath.sh

PROBLEMDIR="${HOME}/neurobench-problems"

#Check if running as an array job
if [[ ! -z "$PBS_ARRAYID" ]]; then
        IID=${PBS_ARRAYID}
fi
#Check if running as an array job
if [[ ! -z "$SGE_TASK_ID" ]]; then
        IID=${SGE_TASK_ID}
fi
if [[ ! -z "$SLURM_ARRAY_TASK_ID" ]]; then
        IID=${SLURM_ARRAY_TASK_ID}
fi

# Run the program

PARAMS=$(awk "NR==${IID} {print;exit}" ${INPUTFILE})

echo ${PARAMS} ${VERBOSE} ${USEPRIOR}

cat<<EOF | matlab -nodisplay
%addpath(genpath('${HOME}/MATLAB'));
cd('${WORKDIR}');
options=struct('RootDirectory','${WORKDIR}','Display',${VERBOSE},'TolFun',${TOLFUN},'MaxFunEvalMultiplier',${MAXFUNMULT},'StopSuccessfulRuns',${STOPSUCCRUNS},'ProblemDirectory','${PROBLEMDIR}');
${PARAMS}
benchmark_run(${PARAMS},options);
EOF
