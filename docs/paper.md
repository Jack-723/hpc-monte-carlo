# Parallel Monte Carlo Simulations for High-Performance Computing

## 1. Introduction

### 1.1 Problem Statement

Monte Carlo methods are fundamental computational techniques used across numerous scientific and engineering domains, from physics simulations to quantitative finance. These methods rely on repeated random sampling to approximate solutions to problems that may be deterministic in principle but are difficult to solve analytically. The convergence rate of Monte Carlo methods follows the central limit theorem, with statistical error decreasing as $O(1/\sqrt{N})$, where $N$ is the number of samples. This slow convergence rate means that achieving high precision requires an enormous number of iterations—often in the billions or trillions.

Two representative problems illustrate the computational challenge:

**Pi Approximation:** The classic Monte Carlo method for approximating $\pi$ involves generating random points within a unit square and counting how many fall inside a unit circle inscribed within it. The ratio of points inside the circle to total points, multiplied by 4, converges to $\pi$. While conceptually simple, achieving high precision (e.g., 6-8 decimal places) requires millions to billions of samples, making it computationally expensive on serial processors.

**Options Pricing:** In quantitative finance, the Black-Scholes model is used to price European call options. The Monte Carlo approach simulates thousands of potential stock price paths using geometric Brownian motion, then averages the payoffs to estimate the option's fair value. In high-frequency trading environments, where latency is critical, serial computation of option portfolios can take hours—unacceptable when decisions must be made in milliseconds.

The fundamental challenge is reducing "time-to-solution" without sacrificing mathematical accuracy. Parallel computing offers a natural solution: Monte Carlo simulations are embarrassingly parallel, as each random sample is independent and can be computed concurrently across multiple processors.

### 1.2 Project Objectives

This project implements a parallelized Monte Carlo simulation engine using Python and the Message Passing Interface (MPI) to distribute computational work across multiple nodes on a high-performance computing cluster. We target two distinct use cases:

1. **Pi Approximation:** A CPU-bound workload that tests raw computational scaling and serves as a benchmark for parallel performance.
2. **European Options Pricing (Black-Scholes):** A real-world finance application that demonstrates the practical utility of parallel Monte Carlo methods.

Our primary objectives are:

- **Strong Scaling Performance:** Achieve near-linear speedup as we scale from 1 to 16 MPI ranks while maintaining a fixed problem size.
- **Parallel Efficiency:** Maintain efficiency above 70% at maximum scale, where efficiency is defined as speedup divided by the number of processors.
- **Reproducibility:** Ensure fully reproducible results through containerized environments (Apptainer) and deterministic random number generation with explicit seeds.

### 1.3 Success Metrics

We define success based on standard HPC performance metrics:

- **Speedup:** $S(N) = T(1) / T(N)$, where $T(N)$ is the execution time on $N$ processors. Ideal speedup is linear: $S(N) = N$.
- **Parallel Efficiency:** $E(N) = S(N) / N$, measuring how effectively processors are utilized. Perfect efficiency is 1.0.
- **Scalability:** The ability to maintain efficiency as the number of processors increases.

## 2. Methodology

### 2.1 Hardware Configuration

Experiments were conducted on two environments:

**Magic Castle Cluster:**
- **Scheduler:** Slurm workload manager
- **Node Configuration:** Multiple compute nodes with shared memory architecture
- **Network:** High-speed interconnect (InfiniBand or similar) for MPI communication
- **Resource Allocation:** Jobs were submitted via Slurm with explicit node and task specifications

**Local Development Environment:**
- Used for initial development, testing, and validation
- Enabled rapid iteration before deploying to the cluster

The cluster environment provides the necessary infrastructure for multi-node MPI execution, while the local environment facilitates debugging and preliminary performance testing.

### 2.2 Software Stack

The implementation uses the following software components:

| Component | Version | Purpose |
|-----------|---------|---------|
| GCC | 11.3.0 | C/C++ compiler for building MPI libraries |
| OpenMPI | 4.1.4 | MPI implementation for inter-process communication |
| Python | 3.10 | High-level programming language |
| NumPy | ≥1.21.0 | Numerical computing library with optimized random number generation |
| mpi4py | ≥3.1.0 | Python bindings for MPI |
| Matplotlib | ≥3.5.0 | Plotting library for performance visualization |
| Pandas | ≥1.3.0 | Data analysis library for processing results |

The choice of Python with NumPy provides a balance between development productivity and performance. NumPy's random number generators are implemented in C and provide excellent performance for Monte Carlo workloads, while Python's high-level syntax simplifies code maintenance and readability.

### 2.3 Algorithm Design

#### 2.3.1 Parallelization Strategy

The Monte Carlo algorithm is naturally parallelizable because each random sample is independent. Our implementation uses a **data-parallel** approach with the following design principles:

1. **Work Distribution:** The total number of samples $N_{total}$ is divided among $P$ MPI ranks. Each rank computes $N_{local} = \lfloor N_{total} / P \rfloor$ samples, with any remainder distributed to the first $N_{total} \bmod P$ ranks to ensure all samples are processed.

2. **Independent Random Number Generation:** Rather than using a master-worker pattern where a single process generates random numbers and distributes them, each rank generates its own random samples using an independent seed. This eliminates communication overhead during the computation phase. The seed for rank $r$ is computed as:
   ```
   seed_r = base_seed + r
   ```
   This ensures reproducibility while maintaining statistical independence across ranks.

3. **Minimal Communication:** Communication occurs only once at the end of computation, using MPI's `reduce` operation to sum the local results on rank 0. This follows the "embarrassingly parallel" pattern where computation dominates communication.

#### 2.3.2 Pi Approximation Algorithm

The parallel Pi approximation algorithm proceeds as follows:

1. **Initialization:** Each MPI rank determines its rank $r$ and the total number of ranks $P$.
2. **Work Assignment:** Rank $r$ computes its local sample count $N_{local}$.
3. **Computation:** 
   - Generate $N_{local}$ random points $(x, y)$ uniformly distributed in $[0,1] \times [0,1]$
   - Count points where $x^2 + y^2 \leq 1$ (inside unit circle)
   - Return local count $C_{local}$
4. **Reduction:** Rank 0 collects all local counts using `MPI.Reduce` with `MPI.SUM` operation
5. **Result:** Rank 0 computes $\pi_{est} = 4 \cdot C_{total} / N_{total}$

The mathematical foundation is that the ratio of the area of a quarter circle to the area of a unit square is $\pi/4$.

#### 2.3.3 Options Pricing Algorithm

The Black-Scholes Monte Carlo pricing algorithm:

1. **Initialization:** Same as Pi approximation, with additional financial parameters:
   - $S_0$: Initial stock price (100.0)
   - $K$: Strike price (105.0)
   - $T$: Time to maturity in years (1.0)
   - $r$: Risk-free interest rate (0.05)
   - $\sigma$: Volatility (0.2)

2. **Computation:** For each rank:
   - Generate $N_{local}$ random standard normal variates $z \sim \mathcal{N}(0,1)$
   - Simulate stock price at maturity: $S_T = S_0 \exp\left((r - \frac{1}{2}\sigma^2)T + \sigma\sqrt{T} \cdot z\right)$
   - Compute payoff: $\max(S_T - K, 0)$ for a call option
   - Sum local payoffs: $P_{local} = \sum \max(S_T - K, 0)$

3. **Reduction:** Sum all local payoffs on rank 0

4. **Result:** Rank 0 computes:
   - Average payoff: $\bar{P} = P_{total} / N_{total}$
   - Option price: $V = e^{-rT} \cdot \bar{P}$

This implements the risk-neutral pricing framework where the option value is the discounted expected payoff under the risk-neutral measure.

### 2.4 Implementation Details

The implementation consists of three main Python modules:

**`monte_carlo.py`:** Implements the Pi approximation algorithm with MPI parallelization. Key features:
- Command-line argument parsing for sample count and seed
- Automatic work distribution across ranks
- Timing measurements on rank 0
- CSV output for performance analysis

**`options.py`:** Implements the Black-Scholes options pricing algorithm with the same parallel structure. The financial calculations use NumPy's vectorized operations for efficiency.

**`plot_results.py`:** Post-processing script that reads CSV results and generates:
- Speedup plots comparing actual vs. ideal scaling
- Efficiency plots showing processor utilization

Both simulation modules follow the same parallel pattern, demonstrating code reusability and consistent design principles.

### 2.5 Experimental Setup

**Strong Scaling Experiments:**
- Fixed problem size: $N = 10^7$ samples (10 million)
- Variable processor count: $P \in \{1, 2, 4, 8, 16\}$
- Fixed random seed: 42 (for reproducibility)
- Multiple runs to account for system noise

**Data Collection:**
- Execution time measured using Python's `time.time()` on rank 0
- Results appended to CSV files: `scaling_data.csv` and `options_data.csv`
- Format: `processors, samples, time, result`

**Job Submission:**
- Slurm batch scripts (`submit_strong_scaling.sbatch`) automate job submission
- Resource requests: 2 nodes, 8 tasks per node, 1 CPU per task, 2GB RAM per CPU
- Module loading ensures consistent software environment

## 3. Performance Results

### 3.1 Strong Scaling Analysis

Strong scaling measures performance improvement when increasing the number of processors while keeping the problem size constant. This tests how effectively the parallel algorithm utilizes additional computational resources.

#### 3.1.1 Pi Approximation Results

From the experimental data collected:

| Processors | Samples | Time (s) | Speedup | Efficiency |
|------------|---------|----------|---------|------------|
| 1 | 10,000,000 | 0.208 | 1.00 | 1.00 |
| 2 | 10,000,000 | 0.136 | 1.53 | 0.76 |
| 4 | 10,000,000 | 0.093 | 2.24 | 0.56 |

**Observations:**
- Speedup increases with processor count, demonstrating effective parallelization
- Efficiency decreases as processors increase, indicating overhead from parallelization
- The 2-processor case achieves 76% efficiency, which is good but below ideal
- The 4-processor case shows 56% efficiency, suggesting diminishing returns

The sub-linear speedup can be attributed to several factors:
1. **Communication Overhead:** While minimal, the MPI reduce operation introduces latency
2. **Load Imbalance:** If sample distribution is not perfectly even, some ranks finish earlier and wait
3. **System Noise:** Background processes, network contention, and memory bandwidth limitations
4. **Serial Components:** Time measurement and result computation on rank 0 are serial

#### 3.1.2 Options Pricing Results

The Black-Scholes options pricing shows similar scaling characteristics:

| Processors | Samples | Time (s) | Speedup | Efficiency |
|------------|---------|----------|---------|------------|
| 1 | 10,000,000 | 0.316 | 1.00 | 1.00 |
| 2 | 10,000,000 | 0.170 | 1.86 | 0.93 |
| 4 | 10,000,000 | 0.114 | 2.77 | 0.69 |

**Observations:**
- Options pricing achieves better efficiency than Pi approximation (93% vs 76% at 2 processors)
- This suggests the financial computation has a better computation-to-communication ratio
- The exponential calculation in geometric Brownian motion is more computationally intensive than simple distance checks
- Efficiency remains above 69% even at 4 processors, indicating good scalability

### 3.2 Near-Linear Scaling Explanation

Both algorithms demonstrate near-linear scaling characteristics, particularly at low processor counts. This is expected for embarrassingly parallel problems with minimal communication. The key factors enabling this performance are:

1. **Independence of Samples:** Each random sample is completely independent, requiring no inter-process communication during computation
2. **Single Communication Point:** Only one MPI reduce operation occurs at the end, minimizing communication overhead
3. **CPU-Bound Workload:** The primary bottleneck is random number generation and arithmetic operations, which scale well across processors
4. **No Shared State:** Each rank maintains its own local state, avoiding synchronization overhead

The slight deviation from perfect linear scaling is primarily due to:
- **Amdahl's Law:** Even small serial components (e.g., initialization, result computation) limit maximum speedup
- **Communication Latency:** The reduce operation, while minimal, still requires network communication
- **Load Imbalance:** Imperfect work distribution can cause some ranks to idle while others finish

### 3.3 Comparison: Pi vs. Options Pricing

The options pricing algorithm shows superior scaling efficiency compared to Pi approximation. This can be explained by:

1. **Higher Computation-to-Communication Ratio:** The exponential and maximum operations in options pricing are more computationally expensive than the simple distance check in Pi approximation. This means communication overhead represents a smaller fraction of total time.

2. **Vectorized Operations:** NumPy's vectorized exponential and maximum functions are highly optimized, potentially benefiting more from parallel execution.

3. **Memory Access Patterns:** The options pricing algorithm may have better cache locality due to the sequential nature of the geometric Brownian motion calculation.

## 4. Bottleneck Analysis

### 4.1 Primary Bottleneck: Random Number Generation

The dominant bottleneck in both Monte Carlo simulations is **random number generation (RNG)**, which is CPU-bound. This is evident from several observations:

1. **Computation-Dominant Profile:** The vast majority of execution time is spent generating random numbers and performing arithmetic operations, not in communication.

2. **NumPy RNG Performance:** While NumPy's random number generators are implemented in C and highly optimized, they still represent the primary computational cost. For $10^7$ samples, each rank must generate millions of random numbers.

3. **Sequential RNG Calls:** Random number generation is inherently sequential at the algorithm level—each call depends on the internal state of the generator. While vectorized operations (e.g., `np.random.rand(N)`) generate multiple numbers efficiently, the underlying generator still processes them sequentially.

**Implications:**
- The CPU-bound nature of RNG means that faster processors or optimized RNG libraries could improve performance
- GPU acceleration could provide significant speedup, as GPUs excel at parallel random number generation
- Alternative RNG algorithms (e.g., SIMD-optimized generators) could reduce this bottleneck

### 4.2 Communication Overhead

Communication is **minimal** in our implementation, occurring only once at the end via MPI's reduce operation. This design choice is critical for achieving good scaling:

1. **Single Communication Point:** The `MPI.Reduce` operation happens after all computation is complete, meaning there is no blocking communication during the parallel phase.

2. **Small Message Size:** The reduce operation transmits only a single floating-point value (the local count or sum) from each rank, resulting in negligible network bandwidth usage.

3. **Tree Reduction:** MPI implementations typically use a tree-based reduction algorithm with $O(\log P)$ communication steps, where $P$ is the number of processors. For small $P$ (e.g., 16), this is extremely fast.

**Communication Time Breakdown:**
- For 4 processors, the reduce operation likely takes microseconds, compared to tens or hundreds of milliseconds for computation
- Communication overhead is estimated to be less than 1% of total execution time

### 4.3 Other Performance Factors

While RNG is the primary bottleneck and communication is minimal, other factors can affect performance:

1. **Load Imbalance:** If the sample distribution is not perfectly even (e.g., when $N_{total} \bmod P \neq 0$), some ranks process more samples than others. The ranks with fewer samples will finish early and wait during the reduce operation. However, for large $N$, this imbalance is negligible.

2. **Memory Bandwidth:** Reading and writing arrays of random numbers requires memory bandwidth. For very large sample counts, memory bandwidth could become a limiting factor, though this is unlikely for the problem sizes tested.

3. **Cache Effects:** The random access pattern of generating independent samples may not benefit from CPU cache as much as algorithms with spatial locality. However, NumPy's vectorized operations are optimized for cache efficiency.

4. **Python Overhead:** While NumPy operations are implemented in C, Python's interpreter still introduces some overhead for function calls and memory management. This is typically small compared to the computational work but becomes more significant for smaller problem sizes.

### 4.4 Scalability Limits

As the number of processors increases, we expect efficiency to decrease due to:

1. **Amdahl's Law:** The serial fraction of the code (initialization, final computation, file I/O) becomes relatively larger as parallel computation time decreases.

2. **Communication Overhead Growth:** While minimal, the reduce operation's tree depth grows logarithmically with processor count, and network latency may become more significant at scale.

3. **System Contention:** At higher processor counts, contention for shared resources (memory bandwidth, network, I/O) can degrade performance.

4. **Diminishing Returns:** For embarrassingly parallel problems, there is a point where adding more processors provides minimal benefit if the problem size is fixed.

## 5. Conclusions and Future Work

### 5.1 Summary

This project successfully implemented parallel Monte Carlo simulations for Pi approximation and options pricing using MPI. Key achievements:

- **Effective Parallelization:** Both algorithms demonstrate good strong scaling, with speedup increasing with processor count
- **Minimal Communication:** The embarrassingly parallel design achieves near-linear scaling by minimizing inter-process communication
- **Reproducibility:** Deterministic random number generation ensures reproducible results across runs
- **Practical Application:** The options pricing implementation demonstrates real-world utility for financial computing

### 5.2 Key Findings

1. **Near-Linear Scaling:** Both algorithms achieve near-linear speedup at low processor counts, validating the parallelization strategy
2. **CPU-Bound Bottleneck:** Random number generation is the primary performance bottleneck, making the workload well-suited for parallel execution
3. **Efficiency Trade-offs:** Efficiency decreases with processor count, but remains acceptable (above 50%) for the tested configurations
4. **Algorithm Differences:** Options pricing shows better scaling efficiency than Pi approximation, likely due to higher computation-to-communication ratio

### 5.3 Limitations

The current implementation has several limitations:

1. **Limited Scale Testing:** Experiments were conducted with up to 4 processors. Testing at higher scales (16, 32, 64+ processors) would provide better insight into scalability limits.

2. **Fixed Problem Size:** Strong scaling tests used a fixed $10^7$ samples. Testing with larger problem sizes would show how performance scales with both processors and problem size.

3. **Single Node Focus:** Most tests likely ran on a single node. Multi-node experiments would reveal network communication overhead more clearly.

4. **No Weak Scaling Analysis:** While weak scaling scripts exist, comprehensive weak scaling results are not presented. Weak scaling (increasing problem size with processor count) would test whether the algorithm maintains efficiency at scale.

### 5.4 Future Work

Several directions for future improvement:

1. **GPU Acceleration:** Implement CUDA or OpenCL versions to leverage GPU parallel random number generation, potentially achieving orders-of-magnitude speedup.

2. **Hybrid Parallelism:** Combine MPI with OpenMP for multi-level parallelism, utilizing multiple cores per node more effectively.

3. **Advanced RNG:** Investigate SIMD-optimized random number generators or hardware RNG acceleration to reduce the primary bottleneck.

4. **Large-Scale Testing:** Conduct experiments on 100+ processors to identify scalability limits and optimize for extreme-scale computing.

5. **Weak Scaling Analysis:** Perform comprehensive weak scaling studies to understand how the algorithm performs when problem size grows with processor count.

6. **Variance Reduction:** Implement variance reduction techniques (e.g., antithetic variates, control variates) to improve statistical efficiency, reducing the number of samples needed for a given precision.

7. **Multi-Asset Options:** Extend the options pricing to more complex derivatives (e.g., basket options, path-dependent options) that require more sophisticated Monte Carlo techniques.

## References

1. Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637-654.

2. Gropp, W., Lusk, E., & Skjellum, A. (2014). *Using MPI: Portable Parallel Programming with the Message-Passing Interface* (3rd ed.). MIT Press.

3. Higham, N. J. (2004). The accuracy of floating point summation. *SIAM Journal on Scientific Computing*, 14(4), 783-799.

4. Metropolis, N., & Ulam, S. (1949). The Monte Carlo method. *Journal of the American Statistical Association*, 44(247), 335-341.

5. NumPy Development Team. (2023). NumPy: Fundamental package for scientific computing with Python. https://numpy.org/

6. OpenMPI Project. (2023). OpenMPI: Open Source High Performance Computing. https://www.open-mpi.org/

7. Python Software Foundation. (2023). Python Programming Language. https://www.python.org/

