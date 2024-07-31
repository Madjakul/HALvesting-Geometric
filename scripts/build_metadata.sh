#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify

# ************************** Customizable Arguments ************************************

ROOT_DIR=$DATA_ROOT
JSON_DIR=$DATA_ROOT/responses
XML_DIR=$DATA_ROOT/output_tei_xml
RAW_DIR="$DATA_ROOT/raw-abl"
COMPUTE_NODES=false
COMPUTE_EDGES=true

# --------------------------------------------------------------------------------------

DATASET_CHECKPOINT="Madjakul/HALvest-Geometric"
ZIP_COMPRESS=true
# CACHE_DIR="/local"
# DATASET_CONFIG_PATH="$PROJECT_ROOT/configs/dataset_config.txt"

# **************************************************************************************

cmd=( python3 "$PROJECT_ROOT/build_metadata.py" \
  --root_dir "$ROOT_DIR" \
  --json_dir "$JSON_DIR" \
  --xml_dir "$XML_DIR" \
  --raw_dir "$RAW_DIR" \
  --compute_nodes "$COMPUTE_NODES" \
  --compute_edges "$COMPUTE_EDGES" )

if [[ -v DATASET_CHECKPOINT ]]; then
  cmd+=( --dataset_checkpoint "$DATASET_CHECKPOINT" )
fi

if [[ -v CACHE_DIR ]]; then
  cmd+=( --cache_dir "$CACHE_DIR" )
fi

if [[ -v DATASET_CONFIG_PATH ]]; then
  cmd+=( --dataset_config_path "$DATASET_CONFIG_PATH" )
fi

if [[ -v ZIP_COMPRESS ]]; then
  cmd+=( --zip_compress "$ZIP_COMPRESS" )
fi

"${cmd[@]}"
