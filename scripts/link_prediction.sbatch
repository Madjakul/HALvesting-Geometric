#!/bin/bash
#
#SBATCH --job-name=halvesting_link_prediction   # Job name
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=1                              # Run a single task	
#SBATCH --cpus-per-task=24                      # Number of CPU cores per task
#SBATCH --partition=gpu
#SBATCH --time=24:00:00                         # Time limit hrs:min:sec
#SBATCH --output=logs/%x_%j.log                 # Standard output and error log. %x denotes the job name, %j the jobid.

module purge
module load cmake cuda/11.4

source /home/$USER/.bashrc
conda activate halph

mkdir logs || true

./scripts/link_prediction.sh
