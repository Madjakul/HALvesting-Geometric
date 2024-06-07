#!/bin/bash
#
#SBATCH --job-name=halph_link_prediction      # Job name
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=24            # Number of CPU cores per task
#SBATCH --partition=gpu
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=logs/%x_%j.log            # Standard output and error log. %x denotes the job name, %j the jobid.
#SBATCH --gres=gpu:1     # GPU nodes are only available in gpu partition

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

mkdir logs || true


echo "++++++++++++++++++++++++++++++++++++++++++"
echo "+    Running the Singularity container   +"
echo "++++++++++++++++++++++++++++++++++++++++++"

SINGULARITY_IMG=/scratch/fkulumba/HALphv2/halph-pt2.3:latest.sif
LOCAL_SCRATCH=/scratch/fkulumba/
TARGET_SCRATCH=/scratch/
WORK_DIR=/home/fkulumba/scratch/HALph




TRAIN_CMD="""nvidia-smi && nvidia-smi topo -m && printenv && \
pip freeze && \
cd /workspace && \
ls -l && \
./scripts/bbp_link_prediction.sh
"""

singularity exec \
-B "$WORK_DIR:/workspace/",\
"$LOCAL_SCRATCH:$TARGET_SCRATCH" \
--nv \
$SINGULARITY_IMG \
bash -c "$TRAIN_CMD"
