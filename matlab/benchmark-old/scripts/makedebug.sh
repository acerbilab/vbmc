#!/bin/sh

PROJECT="neurobench"

module purge
#. /etc/profile.d/modules.sh

# Use Intel compiler
module load matlab
source ${HOME}/MATLAB/setpath.sh

export MATLABPATH=${MATLABPATH}
FILENAME="'${1}'"

# Run the program
PROBS="{1}"
DIMS="{2,3,5,10,20}"
NOISE="'[]'"
ALGOS="{'fmincon','patternsearch'}"
IDS="'1:10'"

cat<<EOF | matlab -nodisplay
parlist=combcell('cec14',${PROBS},${DIMS},${NOISE},${ALGOS},'base',${IDS});
fout=fopen(${FILENAME},'w+');
for i=1:length(parlist)
	fprintf(fout,'''%s'',''%d'',''%d'',''%s'',''%s'',''%s'',''%s''\n',parlist{i}{1},parlist{i}{2},parlist{i}{3},parlist{i}{4},parlist{i}{5},parlist{i}{6},parlist{i}{7});
end
fclose(fout);
EOF
