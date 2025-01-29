#!/bin/bash
#SBATCH --job-name=stardata
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/za9132/analysis/stardata_log/slurm-%j.out
#SBATCH --error=/home/za9132/analysis/stardata_log/slurm-%j.out

module purge
module load anaconda3/2023.3
module load openmpi/gcc/4.1.2
conda activate fast-mpi4py

# read command line arguments
round=$1
simname=$2
aexpmax=${3:-0.1}
bturb=${4:-1.0}
epssfloc=${5:-1.0}
ncpu=${6:-384}
outfile=${7:-data}

srun python stardata.py $round $simname $outfile -a $aexpmax -b $bturb -l $epssfloc -n $ncpu
