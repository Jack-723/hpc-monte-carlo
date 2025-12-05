#!/bin/bash
#SBATCH --job-name=mc-weak
#SBATCH --account=def-sponsor00       # CHANGE THIS to your allocation
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=results/weak_%j.out
#SBATCH --error=results/weak_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Start: $(date)"

module purge
module load gcc/11.3.0 openmpi/4.1.4 python/3.10

# Weak scaling: samples grow with number of tasks
SAMPLES_PER_TASK=10000000
TOTAL_SAMPLES=$((SAMPLES_PER_TASK * SLURM_NTASKS))
SEED=42

echo "Samples per task: $SAMPLES_PER_TASK"
echo "Total samples: $TOTAL_SAMPLES"

srun --mpi=pmix python src/monte_carlo.py --samples $TOTAL_SAMPLES --seed $SEED

echo "End: $(date)"