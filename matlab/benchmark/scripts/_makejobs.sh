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
DIMS="{'2D','4D','6D','8D','10D'}"
NOISE="'[]'"
ALGOS="{'vbmc'}"
ALGOSET="'base'"
IDS="{'1:2','3:4','5:6','7:8','9:10','11:12','13:14','15:16','17:18','19:20'}"
IDS_SINGLE="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}"

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
        6)      ALGOSET="{'acqgev'}"
                ;;
	7)	ALGOSET="{'acqcmaes'}"
		;;
        8)      ALGOSET="{'acqprop'}"
                ;;
        9)      ALGOSET="{'acqpropnorot'}"
		IDS="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}"
                ;;
        10)     ALGOSET="{'acqpropfewrot'}"
                IDS="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}"
                ;;
        11)     ALGOSET="{'acqfnorot'}"
                IDS="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}"
 		;;
        12)     ALGOSET="{'acqpropfnorot'}"
                IDS="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}"
                ;;
       13)     ALGOSET="{'acqpropf'}"
                IDS="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}"
                ;;
       14)     ALGOSET="{'acqpropfnorotbz'}"
                IDS="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}"
                ;;
       15)     ALGOSET="{'acqpropfbz'}"
                IDS="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}"
                ;;
       16)     ALGOSET="{'adaptive'}"
                IDS="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}"
                ;;
       17)     ALGOSET="{'adaptiveless'}"
                IDS="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}"
                ;;
       18)     ALGOSET="{'adaptivelessnogp'}"
                IDS="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}"
                ;;
       19)     ALGOSET="{'basenogp'}"
                IDS="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}"
                ;;
       20)     ALGOSET="{'basekone'}"
               IDS=$IDS_SINGLE
                ;;
       21)     	ALGOSET="{'acqproponly'}"
               	IDS=$IDS_SINGLE
		;;
       22)      ALGOSET="{'acqusonly'}"
                IDS=$IDS_SINGLE
                ;;
       23)      ALGOSET="{'acqvusonly'}"
                IDS=$IDS_SINGLE
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
        80)     ALGOS="{'ais'}"
                ;;
	90)	ALGOS="{'bape'}"
		IDS=$IDS_SINGLE
		;;

        101)    PROBS="{'goris2015'}"
		ALGOS="{'laplace'}"
                DIMS="{'S7','S8','S9','S10','S11','S12'}"
                IDS="{'1','2','3'}"
                ;;
        102)    PROBS="{'goris2015'}"
                ALGOS="{'wsabi','wsabi@mm','bmc','smc','ais'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_SINGLE
                ;;
        103)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@adaptive'}"
                DIMS="{'S7','S8'}"
                IDS="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}"
                ;;
        104)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@adaptiveless'}"
                DIMS="{'S7','S8'}"
                IDS="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'}"
                ;;
        105)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@acqpropfnorotbz'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_SINGLE
                ;;
        106)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@adaptivelessnogp'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_SINGLE
                ;;
        107)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@basenogp'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_SINGLE
                ;;
        108)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@basekone'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_SINGLE
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
