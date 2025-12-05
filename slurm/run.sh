#!/bin/bash
# Wrapper script for running experiments
# Usage: ./slurm/run.sh [local|strong|weak]

case "${1:-local}" in
    local)
        echo "Running local experiments..."
        bash run_local_experiment.sh
        ;;
    strong)
        echo "Submitting strong scaling job..."
        sbatch slurm/submit_strong_scaling.sbatch
        ;;
    weak)
        echo "Submitting weak scaling job..."
        sbatch slurm/submit_weak_scaling.sbatch
        ;;
    *)
        echo "Usage: $0 [local|strong|weak]"
        exit 1
        ;;
esac