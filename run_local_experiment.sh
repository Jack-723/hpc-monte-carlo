#!/bin/bash

# 1. Clean up old results so we don't graph duplicates
rm -f results/scaling_data.csv
rm -f results/options_data.csv

# 2. Run Pi Approximation (1, 2, and 4 processors)
echo "Running Pi Experiment..."
mpiexec -n 1 python src/monte_carlo.py --samples 10000000
mpiexec -n 2 python src/monte_carlo.py --samples 10000000
mpiexec -n 4 python src/monte_carlo.py --samples 10000000

# 3. Run Options Pricing (1, 2, and 4 processors)
echo "Running Options Experiment..."
mpiexec -n 1 python src/options.py --samples 10000000
mpiexec -n 2 python src/options.py --samples 10000000
mpiexec -n 4 python src/options.py --samples 10000000

echo "Experiments finished. Data saved to results/."