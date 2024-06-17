#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify

# ************************** Customizable Arguments ************************************

ROOT_DIR=$DATA_ROOT
JSON_DIR=$DATA_ROOT/responses
XML_DIR=$DATA_ROOT/output_tei_xml
COMPUTE_NODES=true
COMPUTE_EDGES=true

# --------------------------------------------------------------------------------------

# CACHE_DIR="/local"

# **************************************************************************************

cmd=( python3 "$PROJECT_ROOT/build_metadata.py" \
  --root_dir "$ROOT_DIR" \
  --json_dir "$JSON_DIR" \
  --xml_dir "$XML_DIR" \
  --compute_nodes "$COMPUTE_NODES" \
  --compute_edges "$COMPUTE_EDGES" )

if [[ -v CACHE_DIR ]]; then
  cmd+=( --cache_dir "$CACHE_DIR" )
fi
"${cmd[@]}"
