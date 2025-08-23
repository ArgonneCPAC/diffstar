#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
#PBS -j oe

# account to charge
#PBS -A galsampler

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=1:ncpus=20:mpiprocs=20

#PBS -l walltime=04:00:00

# Load software
source ~/.bash_profile
cd ~/source/diffmah/scripts/

mpirun -n 20 python history_fitting_script_TNG_mgash.py /lcrc/project/halotools/alarcon/results/tng_diffstar_tpeak/ -istart 4 -iend 8