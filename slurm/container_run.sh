#!/usr/bin/env bash
# Wrapper script for running with Apptainer container

set -euo pipefail

# Path to the Singularity/Apptainer image
SIF=env/monte_carlo.sif
DEF=env/project.def

# Build container if it doesn't exist
if [[ ! -f "$SIF" ]]; then
    echo "Container not found. Building from $DEF..."
    apptainer build "$SIF" "$DEF"
    echo "Container built successfully: $SIF"
fi

# Run command inside container
# --bind: mounts current directory inside container
# --pwd: sets working directory inside container
echo "Running inside container: $SIF"
apptainer exec --bind $PWD:$PWD --pwd $PWD $SIF "$@"
