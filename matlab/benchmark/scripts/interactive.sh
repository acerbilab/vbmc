#!/bin/bash
qsub -I -q interactive -l nodes=1:ppn=2,mem=8GB,walltime=04:00:00 -M la67@nyu.edu
