#!/bin/sh
echo "Usage: makejobs job# [file#]"

PROJECT="infbench"
#source ${HOME}/MATLAB/setroot.sh

module purge
#. /etc/profile.d/modules.sh

# Use Intel compiler
module load matlab/2017a
source ${HOME}/MATLAB/setpath.sh
export MATLABPATH=${MATLABPATH}
WORKDIR="${SCRATCH}/${PROJECT}"

if [ -z "${2}" ]; 
	then DIRID=${1}; 
	else DIRID=${2}; 
fi

FILEID=${1}
FILENAME="joblist-${FILEID}.txt"
echo "Input #: ${1}   Output file: ${FILENAME}"

VBMC18="{'lumpy','studentt','cigar'}"

# Default job list
PROBSET="'vbmc18'"
PROBS=${VBMC18}
DIMS="{'2D','4D','6D','8D'}"
NOISE="'[]'"
ALGOS="{'vbmc'}"
ALGOSET="'base'"
IDS="{'1:2','3:4','5:6','7:8','9:10','11:12','13:14','15:16','17:18','19:20'}"

case "${1}" in
	0)      PROBS="{'lumpy'}"
		ALGOS="{'vbmc'}"
		DIMS="{'2D','4D'}"
		IDS="{'1','2'}"
		;;
	1)	ALGOSET="{'base'}"
		;;
	2)      ALGOSET="{'acqkl'}"
		;;
        3)      ALGOSET="{'acqvar'}"
                ;;
        4)      ALGOSET="{'betazero'}"
                ;;
        5)      ALGOSET="{'betatiny'}"
                ;;
        6)      ALGOSET="{''}"
                ;;
        50)     ALGOS="{'wsabi'}" 
                ;;
        51)     ALGOS="{'wsabi'}"
		ALGOSET="{'mm'}"
                ;;
	60)     ALGOS="{'bmc'}"
                ;;
        70)     ALGOS="{'smc'}"
                ;;

        101)    PROBSET="{'ccn17'}"
                PROBS="{'visvest_joint'}"
		ALGOS=$BESTALGOS
                DIMS="{'S1','S2','S3','S15','S16','S17'}"
                IDS="{'1:5','6:10','11:15','16:20','21:25','26:30','31:35','36:40','41:45','46:50'}"
                ;;
esac

echo "Job items: ${PROBSET},${PROBS},${DIMS},${NOISE},${ALGOS},${ALGOSET},${IDS}"

cat<<EOF | matlab -nodisplay
addpath(genpath('${HOME}/${PROJECT}'));
currentDir=cd;
cd('${WORKDIR}');
infbench_joblist('${FILENAME}','run${DIRID}',${PROBSET},${PROBS},${DIMS},${NOISE},${ALGOS},${ALGOSET},${IDS});
cd(currentDir);
EOF

cat ${WORKDIR}/${FILENAME}
cat ${WORKDIR}/${FILENAME} | wc
