#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify

# ************************** Customizable Arguments ************************************

GNN="sage"
NUM_PROC=4
BATCH_SIZE=4
EPOCHS=10

# **************************************************************************************

for run in {1..5}
do
    python3 "$PROJECT_ROOT/link_prediction.py" \
      --gnn "$GNN" \
      --run "$run" \
      --num_proc "$NUM_PROC" \
      --batch_size "$BATCH_SIZE" \
      --epochs "$EPOCHS"
done
