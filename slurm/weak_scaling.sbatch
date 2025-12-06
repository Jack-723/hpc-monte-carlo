#!/bin/bash
#SBATCH -J monte-carlo-weak
#SBATCH --account=def-someprof     # CHANGE THIS to your allocation
#SBATCH --nodes=4                   # Request 4 nodes
#SBATCH --ntasks-per-node=4         # 4 MPI ranks per node = 16 total
#SBATCH --cpus-per-task=1           # 1 CPU per task
#SBATCH --mem-per-cpu=2G            # 2GB memory per CPU
#SBATCH --time=00:30:00             # 30 minute time limit
#SBATCH --output=results/weak_scaling_%j.out
#SBATCH --error=results/weak_scaling_%j.err

# Record system information
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "========================================="

# Load required modules
module purge
module load StdEnv/2023
module load gcc/12.3.0
module load openmpi/4.1.5
module load python/3.11.5

# Set environment variables
export OMP_NUM_THREADS=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none

# Ensure results directory exists
mkdir -p results

# Clear old data for this specific run
RESULTS_FILE="results/weak_scaling_${SLURM_JOB_ID}.csv"
echo "ranks,samples_per_rank,total_samples,time_sec,pi_estimate,error" > $RESULTS_FILE

# Constant work per rank for weak scaling
SAMPLES_PER_RANK=10000000  # 10 million per rank

echo ""
echo "========================================="
echo "WEAK SCALING EXPERIMENT: Pi Approximation"
echo "Work per rank: $SAMPLES_PER_RANK samples"
echo "Total work scales with number of ranks"
echo "========================================="
echo ""

# Run with increasing number of MPI ranks (problem size grows proportionally)
for RANKS in 1 2 4 8 16; do
    if [ $RANKS -le $SLURM_NTASKS ]; then
        TOTAL_SAMPLES=$((SAMPLES_PER_RANK * RANKS))
        echo "Running with $RANKS MPI ranks ($TOTAL_SAMPLES total samples)..."
        
        srun --ntasks=$RANKS python src/monte_carlo.py \
            --samples $TOTAL_SAMPLES --output $RESULTS_FILE --seed 42 --weak-scaling
        
        echo "  -> Completed $RANKS ranks"
        echo ""
    fi
done

echo "========================================="
echo "Weak scaling experiment completed!"
echo "Results saved to: $RESULTS_FILE"
echo "========================================="

# Generate plots
echo "Generating plots..."
python src/plot_results.py --input $RESULTS_FILE --output results/weak_scaling_${SLURM_JOB_ID} --weak

echo "Job finished at: $(date)"
