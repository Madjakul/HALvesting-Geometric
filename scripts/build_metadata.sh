#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify

# ************************** Customizable Arguments ************************************

DATASET_CHECKPOINT="Madjakul/HALvest-Geometric"
ROOT_DIR=$DATA_ROOT
JSON_DIR=$DATA_ROOT/responses
XML_DIR=$DATA_ROOT/output_tei_xml
RAW_DIR="$DATA_ROOT/raw-it"
COMPUTE_NODES=true
COMPUTE_EDGES=false

# --------------------------------------------------------------------------------------

# CACHE_DIR="/local"

# **************************************************************************************

cmd=( python3 "$PROJECT_ROOT/build_metadata.py" \
  --dataset_checkpoint "$DATASET_CHECKPOINT" \
  --root_dir "$ROOT_DIR" \
  --json_dir "$JSON_DIR" \
  --xml_dir "$XML_DIR" \
  --raw_dir "$RAW_DIR" \
  --compute_nodes "$COMPUTE_NODES" \
  --compute_edges "$COMPUTE_EDGES" )

if [[ -v CACHE_DIR ]]; then
  cmd+=( --cache_dir "$CACHE_DIR" )
fi
"${cmd[@]}"
