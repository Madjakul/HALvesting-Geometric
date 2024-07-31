#!/bin/bash

#SBATCH --job-name=halvesting_tune_link_prediction  # Job name
#SBATCH --ntasks=1                                  # Run a single task
#SBATCH --cpus-per-task=24                          # Number of CPU cores per task
#SBATCH --partition=gpu
#SBATCH --time=24:00:00                             # Time limit hrs:min:sec
#SBATCH --output=logs/%x_%j.log                     # Standard output and error log. %x denotes the job name, %j the jobid.
#SBATCH --gres=gpu:1                                # GPU nodes are only available in gpu partition

module purge
module load singularity/3.4.1


echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "JOB ID            = $SLURM_JOB_ID"
echo ""
echo "Hostname                       = $SLURM_NODELIST"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of CPUs on host         = $SLURM_CPUS_ON_NODE"
echo "GPUs                           = $GPU_DEVICE_ORDINAL"

set -x

echo "++++++++++++++++++++++++++++++++++++++++++"
echo "+    Running the Singularity Container   +"
echo "++++++++++++++++++++++++++++++++++++++++++"


mkdir logs || true


SINGULARITY_IMG=/home/fkulumba/scratch/singularity/torch2.3.1-torch_geometric-cuda12.1-cudnn8.sif
LOCAL_SCRATCH=/home/fkulumba/scratch/
TARGET_SCRATCH=/scratch/
WORK_DIR=/home/fkulumba/scratch/HALvesting-Geometric
 
TRAIN_CMD="""cd /workspace && \
ls -l && \
./scripts/tune_link_prediction.sh
"""

singularity exec \
-H "$WORK_DIR:/workspace/" \
--bind "$LOCAL_SCRATCH:$TARGET_SCRATCH" \
--nv \
$SINGULARITY_IMG \
bash -c "$TRAIN_CMD"
