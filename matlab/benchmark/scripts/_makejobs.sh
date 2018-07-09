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
IDS_FIFTY="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50'}"
IDS_CENTO="{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100'}"

case "${1}" in
	0)      PROBS="{'lumpy'}"
		ALGOS="{'vbmc'}"
		DIMS="{'2D','4D'}"
		IDS="{'1','2'}"
		;;
	1)	ALGOSET="{'base'}"
		IDS=${IDS_SINGLE}
		;;
	2)      ALGOSET="{'cheapgp'}"
		IDS=${IDS_SINGLE}
		;;
        3)      ALGOSET="{'cheapgpentsqrtk'}"
		IDS=${IDS_SINGLE}
                ;;
        4)      ALGOSET="{'gpthresh'}"
		IDS=${IDS_SINGLE}
                ;;
        4b)     ALGOSET="{'gpthreshwidths2'}"
                IDS=${IDS_SINGLE}
                ;;
        4c)      ALGOSET="{'gpthreshwidths'}"
                IDS=${IDS_SINGLE}
                ;;
        4d)     ALGOSET="{'gpthreshruncov'}"
                IDS=${IDS_SINGLE}
                ;;
        4e)     ALGOSET="{'gpthreshruncovzero'}"
                IDS=${IDS_SINGLE}
                ;;
        4f)     ALGOSET="{'gpthreshwcov'}"
                IDS=${IDS_SINGLE}
                ;;
        4g)     ALGOSET="{'gpthreshcovsample'}"
                IDS=${IDS_SINGLE}
                ;;
        4h)     ALGOSET="{'gpthreshcovsample2'}"
                IDS=${IDS_SINGLE}
                ;;
        4i)     ALGOSET="{'gpthreshcovsample3'}"
                IDS=${IDS_SINGLE}
                ;;
        4j)     ALGOSET="{'gpthreshcovsample4'}"
                IDS=${IDS_SINGLE}
                ;;
        4k)     ALGOSET="{'detent'}"
                IDS=${IDS_SINGLE}
                ;;
        4l)     ALGOSET="{'detentfast'}"
                IDS=${IDS_SINGLE}
                ;;
        4m)     ALGOSET="{'altent'}"
                IDS=${IDS_SINGLE}
                ;;
        4n)     ALGOSET="{'altentquick'}"
                IDS=${IDS_SINGLE}
                ;;
        4o)     ALGOSET="{'detentopt4'}"
                IDS=${IDS_SINGLE}
                ;;
        5)      ALGOSET="{'cheapgpmid'}"
                IDS=${IDS_SINGLE}
		;;
        6)      ALGOSET="{'morelbomid'}"
                IDS=${IDS_SINGLE}
		;;
	7)	ALGOSET="{'morelbomid'}"
		IDS=${IDS_SINGLE}
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
       24)      ALGOSET="{'acqpropnoprune'}"
                IDS=$IDS_SINGLE
                ;;
       25)      ALGOSET="{'acqusnoprune'}"
                IDS=$IDS_SINGLE
                ;;
       26)      ALGOSET="{'acqpropcontrol'}"
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
	90)	ALGOS="{'agp'}"
		IDS=$IDS_SINGLE
		;;
       	91)     ALGOS="{'agp@long'}"
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
                IDS=$IDS_FIFTY
                ;;
        103)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@base'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        103L)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@baseluigi'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_FIFTY
                ;;
        104)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@gpthresh'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        104b)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@gpthreshwidths2'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        104c)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@gpthreshwidths'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        104d)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@gpthreshruncov'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        104e)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@gpthreshruncovzero'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        104f)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@gpthreshwcov'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        104g)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@gpthreshcovsample'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        104h)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@gpthreshcovsample2'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        104i)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@gpthreshcovsample3'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        104j)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@gpthreshcovsample4'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        104k)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@detent'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        104l)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@detentfast'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        104m)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@altent'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        104n)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@altentquick'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        104o)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@detentopt4'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        105)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@cheapgpentsqrtk'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_FIFTY
                ;;
        106)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@gpthreshmid'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_FIFTY
                ;;
        107)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@morelbomid'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_FIFTY
                ;;
        108)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@cheapgpmid'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_FIFTY
                ;;
        109)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@acqproponly'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_FIFTY
                ;;
        110)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@acqusonly'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_FIFTY
                ;;
        111)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@acqvusonly'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_FIFTY
                ;;
        112)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@acqpropnoprune'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_FIFTY
                ;;
        113)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@acqusnoprune'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_FIFTY
                ;;
        114)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@acqpropcontrol'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_FIFTY
                ;;
        115)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@cheapgpmidlessprune'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_FIFTY
                ;;
	116)    PROBS="{'goris2015'}"
                ALGOS="{'vbmc@gpthreshmidlessprune'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        116b)   PROBS="{'goris2015'}"
                ALGOS="{'vbmc@lessprune'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_CENTO
                ;;
        120)    PROBS="{'goris2015'}"
                ALGOS="{'agp'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_FIFTY
                ;;
        121)    PROBS="{'goris2015'}"
                ALGOS="{'agp@long'}"
                DIMS="{'S7','S8'}"
                IDS=$IDS_FIFTY
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
