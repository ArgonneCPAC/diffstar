#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
#PBS -j oe

# account to charge
#PBS -A galsampler

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=1:mpiprocs=20

#PBS -l walltime=04:00:00

# Load software
source ~/.bash_profile
cd ~/source/diffmah/scripts/

mpiexec -n 20 python diffstar_fitting_script_umachine.py /lcrc/project/halotools/random_data/0328 -sim_name DR1 -test
