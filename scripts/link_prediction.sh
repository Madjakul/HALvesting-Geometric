#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify

# ************************** Customizable Arguments ************************************

MAX_RUNS=5
ROOT_DIR=$DATA_ROOT
CONFIG_FILE="$DATA_ROOT/train_sage_config.yaml"
WANDB=true

# --------------------------------------------------------------------------------------

LANG_="en"
# NUM_PROC=4
# ACCELERATOR="cpu"

# **************************************************************************************

for run in {1..$MAX_RUNS}
do
    cmd=( python3 "$PROJECT_ROOT/link_prediction.py" \
      --root_dir "$ROOT_DIR" \
      --config_file "$CONFIG_FILE" \
      --wandb "$WANDB" \
      --run "$run" )
    
    if [[ -v LANG_ ]]; then
      cmd+=( --lang_ "$LANG_" )
    fi
    if [[ -v NUM_PROC ]]; then
      cmd+=( --num_proc "$NUM_PROC" )
    fi
    if [[ -v ACCELERATOR ]]; then
      cmd+=( --accelerator "$ACCELERATOR" )
    fi
    "${cmd[@]}"
done
