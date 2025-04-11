#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
#PBS -j oe

# account to charge
#PBS -A galsampler

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=8:mpiprocs=30

#PBS -l walltime=2:00:00

# Load software
source ~/.bash_profile
cd /home/ahearin/work/random/0411

mpirun -n 240 python galacticus_diffstar_fitter_script.py in_plus_ex_situ