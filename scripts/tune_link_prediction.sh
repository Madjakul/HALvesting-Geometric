#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify

# ************************** Customizable Arguments ************************************

STORAGE_PATH=$PROJECT_ROOT
ROOT_DIR=$DATA_ROOT

# --------------------------------------------------------------------------------------

LANG_="fr"
# NUM_PROC=4
# ACCELERATOR="cpu"

# **************************************************************************************


cmd=( python3 "$PROJECT_ROOT/experiments/tune_link_prediction.py" \
    --storage_path "$STORAGE_PATH" \
    --root_dir "$ROOT_DIR" )

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
