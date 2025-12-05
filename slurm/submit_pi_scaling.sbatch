#!/bin/bash
#SBATCH -J monte-carlo-pi
#SBATCH --account=def-someprof     # CHANGE THIS to your allocation
#SBATCH --nodes=4                   # Request 4 nodes
#SBATCH --ntasks-per-node=4         # 4 MPI ranks per node = 16 total
#SBATCH --cpus-per-task=1           # 1 CPU per task
#SBATCH --mem-per-cpu=2G            # 2GB memory per CPU
#SBATCH --time=00:30:00             # 30 minute time limit
#SBATCH --output=results/pi_strong_%j.out
#SBATCH --error=results/pi_strong_%j.err

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

# Load required modules (adjust for your cluster)
module purge
module load StdEnv/2023
module load gcc/12.3.0
module load openmpi/4.1.5
module load python/3.11.5

# Alternative: Use Apptainer container (uncomment if using containerization)
# SIF=env/monte_carlo.sif
# if [[ ! -f "$SIF" ]]; then
#     echo "Building Apptainer container..."
#     apptainer build "$SIF" env/project.def
# fi

# Set environment variables
export OMP_NUM_THREADS=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none

# Ensure results directory exists
mkdir -p results

# Clear old data for this specific run
RESULTS_FILE="results/pi_scaling_${SLURM_JOB_ID}.csv"
echo "ranks,samples,time_sec,pi_estimate,error" > $RESULTS_FILE

# Fixed problem size for strong scaling
SAMPLES=100000000  # 100 million samples

echo ""
echo "========================================="
echo "STRONG SCALING EXPERIMENT: Pi Approximation"
echo "Fixed problem size: $SAMPLES samples"
echo "========================================="
echo ""

# Run with increasing number of MPI ranks
for RANKS in 1 2 4 8 16; do
    if [ $RANKS -le $SLURM_NTASKS ]; then
        echo "Running with $RANKS MPI ranks..."
        
        # Using Apptainer (if container is built):
        # srun --ntasks=$RANKS apptainer exec $SIF python src/monte_carlo.py \
        #     --samples $SAMPLES --output $RESULTS_FILE --seed 42
        
        # Direct Python execution:
        srun --ntasks=$RANKS python src/monte_carlo.py \
            --samples $SAMPLES --output $RESULTS_FILE --seed 42
        
        echo "  -> Completed $RANKS ranks"
        echo ""
    fi
done

echo "========================================="
echo "Strong scaling experiment completed!"
echo "Results saved to: $RESULTS_FILE"
echo "========================================="

# Generate plots
echo "Generating plots..."
python src/plot_results.py --input $RESULTS_FILE --output results/pi_scaling_${SLURM_JOB_ID}

echo "Job finished at: $(date)"
