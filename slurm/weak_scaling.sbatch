#!/bin/bash
#SBATCH -J monte-carlo-weak
#SBATCH --account=def-sponsor00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4         # 4 MPI ranks per node = 8 total
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
module load gcc
module load openmpi
module load python
module load mpi4py

# Set environment variables
export OMP_NUM_THREADS=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none

# Ensure results directory exists
mkdir -p results

# Clear old data for this specific run
PI_RESULTS_FILE="results/weak_scaling_pi_${SLURM_JOB_ID}.csv"
OPTIONS_RESULTS_FILE="results/weak_scaling_options_${SLURM_JOB_ID}.csv"

echo "ranks,samples_per_rank,total_samples,time_sec,pi_estimate,error" > $PI_RESULTS_FILE
echo "ranks,samples_per_rank,total_samples,time_sec,option_price" > $OPTIONS_RESULTS_FILE

# Constant work per rank for weak scaling
SAMPLES_PER_RANK=10000000  # 10 million per rank

echo ""
echo "========================================="
echo "WEAK SCALING EXPERIMENT: Pi Approximation"
echo "Work per rank: $SAMPLES_PER_RANK samples"
echo "Total work scales with number of ranks"
echo "========================================="
echo ""

# Run Pi approximation with increasing number of MPI ranks
for RANKS in 1 2 4 8 16; do
    if [ $RANKS -le $SLURM_NTASKS ]; then
        TOTAL_SAMPLES=$((SAMPLES_PER_RANK * RANKS))
        echo "Running Pi with $RANKS MPI ranks ($TOTAL_SAMPLES total samples)..."
        
        srun --ntasks=$RANKS python src/monte_carlo.py \
            --samples $TOTAL_SAMPLES --output $PI_RESULTS_FILE --seed 42 --weak-scaling
        
        echo "  -> Completed Pi with $RANKS ranks"
        echo ""
    fi
done

echo "========================================="
echo "WEAK SCALING EXPERIMENT: Options Pricing"
echo "Work per rank: $SAMPLES_PER_RANK samples"
echo "Total work scales with number of ranks"
echo "========================================="
echo ""

# Run options pricing with increasing number of MPI ranks
for RANKS in 1 2 4 8 16; do
    if [ $RANKS -le $SLURM_NTASKS ]; then
        TOTAL_SAMPLES=$((SAMPLES_PER_RANK * RANKS))
        echo "Running Options with $RANKS MPI ranks ($TOTAL_SAMPLES total samples)..."
        
        srun --ntasks=$RANKS python src/options.py \
            --samples $TOTAL_SAMPLES --output $OPTIONS_RESULTS_FILE --seed 42
        
        echo "  -> Completed Options with $RANKS ranks"
        echo ""
    fi
done

echo "========================================="
echo "Weak scaling experiments completed!"
echo "Pi Results saved to: $PI_RESULTS_FILE"
echo "Options Results saved to: $OPTIONS_RESULTS_FILE"
echo "========================================="

echo "Job finished at: $(date)"
