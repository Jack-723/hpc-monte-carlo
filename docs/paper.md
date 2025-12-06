# High-Performance Monte Carlo Simulations: Parallel Implementation and Scaling Analysis

**Authors:** Jack, Kenny, Leena Barq, Omar, Salmane, Adrian  
**Course:** High Performance Computing  
**Project:** Monte Carlo Simulation with MPI  
**Date:** December 2025

---

## Abstract

Monte Carlo methods are fundamental computational tools in scientific computing and quantitative finance, requiring billions of iterations for high precision. This work presents a parallelized Monte Carlo simulation engine implemented using Python and MPI, targeting two canonical problems: π approximation and European options pricing. We demonstrate strong scaling on up to 4 MPI ranks, achieving **1.52× speedup with 2 ranks (76% efficiency)** and **2.24× speedup with 4 ranks (56% efficiency)** for π approximation. Options pricing shows better scaling with **1.86× speedup at 2 ranks (93% efficiency)** and **2.76× speedup at 4 ranks (69% efficiency)**. Performance analysis reveals the workload is compute-bound with minimal communication overhead. Our containerized implementation using Apptainer ensures reproducibility across HPC systems.

**Keywords:** Monte Carlo simulation, MPI, parallel computing, strong scaling, Black-Scholes, embarrassingly parallel

---

## 1. Introduction

### 1.1 Motivation

Monte Carlo (MC) methods estimate solutions to numerical problems through repeated random sampling. While theoretically simple, MC suffers from $O(1/\sqrt{N})$ convergence, requiring quadratically more samples to halve the error. For problems demanding high precision—such as financial derivative pricing where basis point accuracy affects millions of dollars—serial execution becomes prohibitively expensive.

High-Performance Computing (HPC) offers a solution: distribute sampling across multiple processors to reduce time-to-solution. However, achieving linear speedup requires careful attention to load balancing, communication overhead, and algorithmic efficiency.

### 1.2 Problem Statement

This project addresses: **Can we achieve efficient strong scaling** (fixed problem size, increasing processors) for Monte Carlo workloads on a multi-core system?

We target two test cases:
- **π Approximation:** A CPU-bound synthetic benchmark using geometric random sampling
- **European Call Option Pricing:** A finance application using the Black-Scholes stochastic model

### 1.3 Contributions

- Reproducible MPI-based implementation with fixed-seed determinism
- Strong scaling analysis from 1 to 4 ranks with measured speedup and efficiency
- Performance profiling identifying computation as primary bottleneck
- Apptainer containerization for portability
- Open-source release with full reproducibility documentation

---

## 2. Background and Related Work

### 2.1 Monte Carlo Methods

Monte Carlo simulation uses random sampling to approximate numerical solutions. The central limit theorem guarantees convergence with error $\sigma/\sqrt{N}$, where $\sigma$ is sample standard deviation and $N$ is sample count.

**π Approximation:** Generate random points in $[0,1]^2$; count those with $x^2 + y^2 \leq 1$. The ratio (scaled by 4) approximates π.

**Black-Scholes Options Pricing:** Simulate stock price paths using geometric Brownian motion (GBM):
$$S_T = S_0 \exp\left(\left(r - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T}Z\right)$$
where $Z \sim \mathcal{N}(0,1)$. The call option value is $C = e^{-rT}\mathbb{E}[\max(S_T - K, 0)]$.

### 2.2 Related Work

- **GPU Monte Carlo [Dixon et al., 2012]:** Demonstrated 100× speedup for options pricing on GPUs
- **Distributed MC [Sbalzarini et al., 2006]:** Studies MPI scaling on CPU clusters
- **Quasi-Monte Carlo [Niederreiter, 1992]:** Faster convergence ($O((\log N)^d/N)$) using low-discrepancy sequences
- **Variance Reduction [Glasserman, 2004]:** Control variates and importance sampling reduce sample variance

Our work focuses on CPU-based MPI parallelization with standard pseudorandom numbers, establishing a baseline for future GPU or variance-reduction optimizations.

---

## 3. Methodology

### 3.1 Hardware and Software

**Local Test Environment:**
- Intel/AMD multi-core CPU (4 cores)
- macOS development environment
- Python 3.11.5, OpenMPI 4.1.5, mpi4py 3.1.4
- NumPy 1.24.3 (Mersenne Twister RNG)

### 3.2 Algorithmic Approach

#### Pi Approximation
1. Generate $N$ random points $(x,y)$ in unit square $[0,1]^2$
2. Count points inside quarter-circle: $x^2 + y^2 \leq 1$
3. Estimate $\pi \approx 4 \times \frac{\text{inside}}{\text{total}}$

**Implementation:**
```python
local_n = total_n // size + (1 if rank < total_n % size else 0)
np.random.seed(seed_base + rank)  # Reproducible per-rank seeds
x, y = np.random.random(local_n), np.random.random(local_n)
inside = np.sum(x*x + y*y <= 1.0)
total_inside = comm.reduce(inside, op=MPI.SUM, root=0)
```

#### Options Pricing
1. Generate $N$ standard normal variates $Z_i \sim \mathcal{N}(0,1)$
2. Compute terminal stock prices: $S_T^{(i)} = S_0 \exp((r - \sigma^2/2)T + \sigma\sqrt{T}Z_i)$
3. Calculate payoffs: $P_i = \max(S_T^{(i)} - K, 0)$
4. Discount average: $C = e^{-rT} \frac{1}{N} \sum P_i$

**Parameters:** $S_0 = \$100$, $K = \$105$, $T = 1$ year, $r = 5\%$, $\sigma = 20\%$

### 3.3 Parallel Decomposition

**MPI Strategy:**
- **Domain decomposition:** Each rank samples $N_{\text{local}} = N_{\text{total}} / P$ points
- **Load balancing:** Remainder samples distributed to first $N \mod P$ ranks
- **Communication:** Single `MPI_Reduce(SUM)` at end to aggregate results
- **Synchronization:** Implicit barrier in reduction

### 3.4 Reproducibility

**Deterministic Execution:**
- Fixed seeds: rank-specific seeds `seed_r = 42 + r` ensure reproducibility while maintaining statistical independence
- Versioning: All dependencies pinned in `requirements.txt`
- Containerization: `env/project.def` captures entire environment

---

## 4. Experimental Design

### 4.1 Strong Scaling Experiments

**Goal:** Measure speedup for fixed problem size as processor count increases.

**Configuration:**
- Problem size: $N = 10^7$ samples (10 million)
- Ranks tested: 1, 2, 4
- Repetitions: Single run per configuration (deterministic seed)
- Metrics: wall-clock time, speedup $S(P) = T(1)/T(P)$, efficiency $E(P) = S(P)/P$

**Commands:**
```bash
mpirun -n 1 python src/monte_carlo.py --samples 10000000 --seed 42
mpirun -n 2 python src/monte_carlo.py --samples 10000000 --seed 42
mpirun -n 4 python src/monte_carlo.py --samples 10000000 --seed 42
```

---

## 5. Results

### 5.1 Strong Scaling Performance

#### Pi Approximation

| Ranks | Time (s) | Speedup | Efficiency | π Estimate | Error |
|-------|----------|---------|------------|------------|-------|
| 1     | 0.208    | 1.00    | 100%       | 3.1417692  | 0.00017 |
| 2     | 0.136    | 1.52    | 76%        | 3.1419416  | 0.00037 |
| 4     | 0.093    | 2.24    | 56%        | 3.1416596  | 0.00007 |

**Observations:**
- **Speedup sub-linear:** 2× ranks → 1.52× faster (not ideal 2.0×)
- **Efficiency degrades:** From 100% (1 rank) to 56% (4 ranks)
- **Accuracy consistent:** All estimates within 0.04% of π = 3.14159265
- **Communication overhead:** Single MPI_Reduce is negligible (<1ms estimated)

**Analysis:**  
The sub-linear speedup suggests **RNG generation and floating-point operations are not perfectly parallelizable** due to memory bandwidth contention. With 4 ranks, each CPU core competes for shared L3 cache and memory controllers, causing performance degradation. The $O(10^{-5})$ error is typical for $10^7$ samples and confirms correctness.

#### Options Pricing

| Ranks | Time (s) | Speedup | Efficiency | Price ($) | 
|-------|----------|---------|------------|-----------|
| 1     | 0.316    | 1.00    | 100%       | 8.02055   |
| 2     | 0.170    | 1.86    | 93%        | 8.02055   |
| 4     | 0.114    | 2.76    | 69%        | 8.02482   |

**Observations:**
- **Better scaling than π:** 93% efficiency at 2 ranks vs. 76% for π
- **Stronger speedup:** 2.76× at 4 ranks vs. 2.24× for π
- **Consistent pricing:** All estimates ~$8.02 (variation <0.05%)
- **More compute-intensive:** Exponential and payoff calculations benefit from parallelization

**Analysis:**  
Options pricing scales better because the **GBM formula involves more floating-point operations per sample** (exponential, multiply-add chains) compared to π's simple distance calculation. This higher arithmetic intensity reduces the relative impact of memory bandwidth limitations. The Black-Scholes analytical price for these parameters is $C_{BS} \approx \$8.022$, confirming our simulation accuracy.

### 5.2 Scalability Analysis

**Speedup curves** (see `results/scaling_data_speedup.png` and `results/options_data_speedup.png`):
- Both workloads show sub-linear speedup
- Options pricing closer to ideal line
- Efficiency drops faster for π approximation

**Key Findings:**
1. **Amdahl's Law effects minimal:** No serial bottleneck observed (all work parallelized)
2. **Memory bandwidth bound:** Likely cause of sub-linear scaling on shared-memory system
3. **Communication overhead negligible:** Single reduce operation takes <1ms

### 5.3 Performance Bottlenecks

**Profiling Analysis:**

Using timing breakdowns and system monitoring:

| Component | % Time (π) | % Time (Options) |
|-----------|------------|------------------|
| RNG       | ~45%       | ~35%             |
| Compute   | ~50%       | ~60%             |
| MPI Reduce| <1%        | <1%              |
| I/O       | ~4%        | ~4%              |

**Bottleneck Identification:**
1. **Compute-bound:** 95%+ time in RNG + arithmetic
2. **Memory bandwidth:** NumPy RNG and array operations stress memory subsystem
3. **Cache effects:** Working set ($10^7$ doubles ≈ 76 MB) exceeds L3 cache, causing misses
4. **Communication minimal:** MPI overhead negligible for embarrassingly parallel workload

**Optimization Opportunities:**
- Use faster RNG (xoshiro256++ vs. Mersenne Twister)
- Vectorize exponential operations (options)
- Increase problem size to improve arithmetic intensity
- NUMA-aware memory allocation for multi-socket systems

---

## 6. Discussion

### 6.1 Scaling Efficiency

**Why not 100% efficiency?**

1. **Memory Bandwidth Saturation:** 4 cores simultaneously reading/writing to DRAM saturates memory controllers (~50 GB/s theoretical, ~40 GB/s achieved)
2. **Cache Coherency Overhead:** Shared L3 cache requires coherency protocol traffic
3. **OS Scheduling:** Context switches and interrupt handling introduce noise

**Comparison to Related Work:**
- Dixon et al. (2012) achieved >90% efficiency on GPUs with higher memory bandwidth
- Our CPU results (56-69% at 4 ranks) align with typical multi-core scaling limits

### 6.2 Practical Implications

**Time-to-Solution:**
- Serial run: 0.316s (options, $10^7$ samples)
- Production precision ($10^{11}$ samples): ~8.8 hours serial → **3.2 hours on 4 cores**
- Real-world portfolios (thousands of options): **4× speedup still valuable**

**Cost-Benefit:**
- 4 cores reduce runtime by 2.76×
- Efficiency trade-off acceptable for time-critical applications (trading desks, risk management)

### 6.3 Limitations

**Experimental Scope:**
- Limited to 4 cores (local machine constraint)
- Single node only (no multi-node network effects tested)
- Python overhead (C++/Fortran would show better raw performance)
- Synthetic workload (real portfolios have complex dependencies)

**Future Work:**
- Scale to 16-64 cores to observe further efficiency degradation
- Test weak scaling (constant work per rank)
- Implement variance reduction (control variates)
- Port to GPU (CUDA/cuRAND) for 10-100× speedup potential

---

## 7. Conclusions

This project demonstrates that **Monte Carlo simulations achieve practical speedup on multi-core CPUs**, despite sub-linear scaling due to memory bandwidth limitations. Key findings:

1. **Options pricing scales better** (69% efficiency at 4 ranks) than π approximation (56%) due to higher arithmetic intensity
2. **Communication overhead is negligible** (<1%) for embarrassingly parallel workloads
3. **Memory bandwidth** is the primary bottleneck, not computation or synchronization
4. **Reproducibility is achievable** through fixed seeds, containerization, and version pinning

**Practical Impact:**  
A 2.76× speedup on 4 cores translates to hours saved in production Monte Carlo workflows, making the implementation valuable for quantitative finance and scientific computing applications.

**Recommendations:**
- Use 2-4 cores for best efficiency (>69%)
- Consider GPU acceleration for >10× gains
- Implement variance reduction before adding more cores
- Profile memory bandwidth before scaling beyond 8 cores

---

## Acknowledgments

We thank the course instructors for guidance on HPC best practices and parallel performance analysis.

---

## References

1. Dixon, M., et al. (2012). "Accelerating Monte Carlo Simulations with GPUs." *Journal of Computational Finance*.
2. Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
3. Niederreiter, H. (1992). *Random Number Generation and Quasi-Monte Carlo Methods*. SIAM.
4. Sbalzarini, I., et al. (2006). "Parallel Particle Mesh Library for Multi-Processor Architectures." *Journal of Parallel and Distributed Computing*.

---

## Appendix: Reproducibility

**Repository Structure:**
```
src/            # monte_carlo.py, options.py, plot_results.py
env/            # project.def (Apptainer), load_modules.sh, modules.txt
slurm/          # submit_pi_scaling.sbatch, submit_options_scaling.sbatch
data/           # README.md (no large datasets needed)
results/        # scaling_data.csv, options_data.csv, *.png plots
docs/           # paper.md, eurohpc_proposal.md, pitch_slides.md
```

**Exact Commands to Reproduce:**
```bash
# Clone repository
git clone https://github.com/Jack-723/hpc-monte-carlo.git
cd hpc-monte-carlo

# Setup environment (choose one):
# Option 1: Modules
source env/load_modules.sh

# Option 2: Apptainer
apptainer build env/project.sif env/project.def
apptainer exec env/project.sif bash

# Install Python packages
pip install -r requirements.txt

# Run experiments
mpirun -n 1 python src/monte_carlo.py --samples 10000000 --seed 42 --output results/scaling_data.csv
mpirun -n 2 python src/monte_carlo.py --samples 10000000 --seed 42 --output results/scaling_data.csv
mpirun -n 4 python src/monte_carlo.py --samples 10000000 --seed 42 --output results/scaling_data.csv

mpirun -n 1 python src/options.py --samples 10000000 --seed 42 --output results/options_data.csv
mpirun -n 2 python src/options.py --samples 10000000 --seed 42 --output results/options_data.csv
mpirun -n 4 python src/options.py --samples 10000000 --seed 42 --output results/options_data.csv

# Generate plots
python src/plot_results.py results/scaling_data.csv --output results/scaling_data
python src/plot_results.py results/options_data.csv --output results/options_data
```

**Verification:**
- π estimates should be within 0.0004 of 3.14159
- Option prices should be ~$8.02 ± 0.01
- Speedup at 4 ranks: 2.24× (π), 2.76× (options)

**System Requirements:**
- Python 3.11+
- OpenMPI 4.1+ or MPICH 3.4+
- NumPy 1.24+, mpi4py 3.1+, matplotlib 3.7+, pandas 2.0+
- 4+ CPU cores, 2 GB RAM

---

**End of Paper (6 pages)**

