#!/bin/bash
set -e

# Ensure required persistent data directories exist
mkdir -p "${YLLM_DATA_DIR:-/data}/database" \
         "${YLLM_DATA_DIR:-/data}/models/huggingface" \
         "${YLLM_DATA_DIR:-/data}/logs" \
         "${YLLM_DATA_DIR:-/data}/config"

# Execute CMD with signal forwarding
exec "$@"

