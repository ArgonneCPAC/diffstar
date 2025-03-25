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

mpirun -n 20 python history_fitting_script_SMDPL_tpeak.py /lcrc/project/halotools/UniverseMachine/SMDPL/sfh_binaries_dr1_bestfit/diffstar_tpeak_fits/ -indir /lcrc/project/halotools/UniverseMachine/SMDPL/sfh_binaries_dr1_bestfit/a_1.000000/  -indir_diffmah /lcrc/project/halotools/UniverseMachine/SMDPL/sfh_binaries_dr1_bestfit/diffmah_tpeak_fits/ -sim_name DR1  -istart 0 -iend 10

