#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=40gb

FIRST_PORT = 8000
NUM_PORT = 30

cd $PBS_O_WORKDIR
module load anaconda3
module load anaconda3/personal
source activate fenicsproject2
python3 $PBS_O_WORKDIR/single_runner.py
