#!/bin/bash
#
#SBATCH --job-name=halvesting_build_metadata    # Job name
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=1                              # Run a single task	
#SBATCH --partition=cpu_homogen
#SBATCH --time=2-00:00:00                       # Time limit hrs:min:sec
#SBATCH --output=logs/%x_%j.log                 # Standard output and error log. %x denotes the job name, %j the jobid.
#SBATCH --exclusive                             # Request exclusive node

module purge

source /home/$USER/.bashrc
conda activate halph

mkdir logs || true

./scripts/build_metadata.sh