#!/bin/bash
#SBATCH --job-name=mc-strong
#SBATCH --account=def-sponsor00       # CHANGE THIS to your allocation
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=results/strong_%j.out
#SBATCH --error=results/strong_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Start: $(date)"

module purge
module load gcc/11.3.0 openmpi/4.1.4 python/3.10

SAMPLES=100000000
SEED=42

srun --mpi=pmix python src/monte_carlo.py --samples $SAMPLES --seed $SEED
srun --mpi=pmix python src/options.py --samples $SAMPLES --seed $SEED

echo "End: $(date)"