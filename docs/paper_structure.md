# HPC Project Paper 

## 1. Introduction
- Problem: Calculating Pi and Pricing Options is computationally expensive.
- Solution: Parallel computing using Monte Carlo methods.

## 2. Methodology
- Hardware: Ran on Magic Castle Cluster and Local dev env.
- Software: Python + mpi4py.
- Algorithm:
    - Master node distributes work? No, we used independent seeds.
    - Workers generate N samples.
    - Reduce operation sums them up.

## 3. Performance Results
- Strong Scaling: (Insert Speedup Graph here).
- We observed near-linear scaling because there is very little communication.

## 4. Bottleneck Analysis
- The main bottleneck is the Random Number Generator (CPU bound).
- Communication is minimal (only happens once at the end).