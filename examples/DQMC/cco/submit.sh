#!/bin/bash
#SBATCH -A ucb-summit-sha
##SBATCH --partition sknl
##SBATCH --partition smem
##SBATCH --qos ucb-summit-sha
##SBATCH --qos blanca-sha
#SBATCH --job-name cbd
#SBATCH --nodes 2
#SBATCH --time=50:00:00
#SBATCH --exclusive
#SBATCH --export=NONE
##SBATCH --exclude=bnode0303

export I_MPI_SLURM_EXT=0
export OMP_NUM_THREADS=1
mpirun -np 48 /projects/anma2640/VMC/dqmc_uihf/VMC/bin/DQMC afqmc.json > afqmc.out
python /projects/anma2640/VMC/dqmc/VMC/scripts/blocking.py samples.dat 50 > blocking.out;
#./scnevpt.sh
