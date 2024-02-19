#!/bin/bash
#SBATCH --job-name=movie
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --output=/home/za9132/analysis/movie_log/slurm-%j.out
#SBATCH --error=/home/za9132/analysis/movie_log/slurm-%j.out

module purge
module load anaconda3/2023.3
module load openmpi/gcc/4.1.2
conda activate fast-mpi4py

# read command line arguments
round=$1
simname=$2
idxproj=$3
var=$4

srun python movie.py $round $simname $idxproj $var
