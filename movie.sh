#!/bin/bash
#SBATCH --job-name=movie
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=00:04:00
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

cd /home/za9132/scratch/romain/round$round/$simname/movie$idxproj
mkdir -p $var-proj$idxproj
srun python /home/za9132/analysis/movie.py $round $simname $idxproj $var
ffmpeg -loglevel quiet -nostdin -i $var-proj$idxproj/img-%05d.png -y -vcodec libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -pix_fmt yuv420p -r 24 -qp 5 ../$var-proj$idxproj.mp4
rm -r $var-proj$idxproj
