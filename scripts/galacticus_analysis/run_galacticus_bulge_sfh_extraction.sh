#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
#PBS -j oe

# account to charge
#PBS -A galsampler

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=1:mpiprocs=1

#PBS -l walltime=2:00:00

# Load software
source ~/.bash_profile
cd /home/ahearin/work/random/0729

python extract_galacticus_sfh_data.py bulge