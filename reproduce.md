# Reproducibility Guide

## Quick Start
```bash
# Clone repo
git clone https://github.com/Jack-723/hpc-monte-carlo.git
cd hpc-monte-carlo

# Install dependencies
pip install -r requirements.txt

# Run test
python src/monte_carlo.py --samples 1000 --seed 42
```

## Run Local Experiments
```bash
bash run_local_experiment.sh
```

## Run on Cluster (Slurm)
```bash
# Edit account name first
# vim slurm/submit_strong_scaling.sbatch

# Submit job
sbatch slurm/submit_strong_scaling.sbatch
```

## Generate Plots
```bash
python src/plot_results.py
```

## Expected Results

With `--seed 42` and 10,000,000 samples, Pi estimate should be approximately 3.1416.
Results are reproducible across runs with the same seed.