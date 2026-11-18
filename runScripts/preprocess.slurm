#!/usr/bin/bash -l

#SBATCH --job-name=preprocess
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

set -e
module load apptainer

cd $HOME/SwissCrop25/data
sha256sum -c SwissCrop25-ParcelTS.tar.sha256

cd ..
apptainer exec conda.sif python ./preprocess.py