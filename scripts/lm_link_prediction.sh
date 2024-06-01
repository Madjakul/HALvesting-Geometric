#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify

# ************************** Customizable Arguments ************************************

GNN="sage"
NUM_PROC=4
BATCH_SIZE=4
EPOCHS=10
ROOT="$DATA_ROOT/mock"
MAX_LENGTH=512

# **************************************************************************************

# --run "$run" \
python3 "$PROJECT_ROOT/lm_link_prediction.py" \
      --gnn "$GNN" \
      --num_proc "$NUM_PROC" \
      --batch_size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --root "$ROOT" \
      --max_length "$MAX_LENGTH"
